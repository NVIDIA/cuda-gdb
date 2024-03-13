/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2023 NVIDIA Corporation
 * Written by CUDA-GDB team at NVIDIA <cudatools@nvidia.com>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#include "defs.h"
#include <stdbool.h>
#include <signal.h>
#include "gdbarch.h"
#include "gdbthread.h"
#include "inferior.h"
#include "frame.h"
#include "regcache.h"
#include "remote.h"
#include "command.h"
#ifdef __QNXTARGET__
# include "remote-nto.h"
#endif

#include "remote-cuda.h"
#include "cuda-exceptions.h"
#include "cuda-packet-manager.h"
#include "cuda-state.h"
#include "cuda-utils.h"
#include "cuda-events.h"
#include "cuda-options.h"
#include "cuda-notifications.h"
#include "cuda-convvars.h"
#include "cuda-kernel.h"
#include "libcudbgipc.h"

#include "top.h"

static bool sendAck = false;
#ifdef __QNXTARGET__
static bool symbols_are_set = false;
#endif

template <class parent>
class cuda_remote_target : public parent
{
public:
  cuda_remote_target ()
  { }

  /* Disable scheduler locking for cuda targets */
  thread_control_capabilities get_thread_control_capabilities () override
  { return tc_none; }

  virtual void kill () override;

  virtual void mourn_inferior () override;

  virtual void detach (inferior *arg0, int arg1) override;

  virtual void resume (ptid_t arg0,
		       int TARGET_DEBUG_PRINTER (target_debug_print_step) arg1,
		       enum gdb_signal arg2) override;

  void cuda_do_resume (ptid_t ptid, int sstep, int host_sstep, enum gdb_signal ts);


  virtual ptid_t wait (ptid_t arg0, struct target_waitstatus * arg1,
		       target_wait_flags arg2) override;

  virtual void fetch_registers (struct regcache *arg0, int arg1) override;

  virtual void store_registers (struct regcache *arg0, int arg1) override;

  virtual int insert_breakpoint (struct gdbarch *arg0,
				 struct bp_target_info *arg1) override;

  virtual int remove_breakpoint (struct gdbarch *arg0,
				 struct bp_target_info *arg1,
				 enum remove_bp_reason arg2) override;

  virtual enum target_xfer_status xfer_partial (enum target_object object,
						const char *annex,
						gdb_byte *readbuf,
						const gdb_byte *writebuf,
						ULONGEST offset, ULONGEST len,
						ULONGEST *xfered_len) override;

  virtual struct gdbarch *thread_architecture (ptid_t arg0) override;

  virtual void prepare_to_store (struct regcache *arg0) override;
};

static
remote_target *
cuda_new_remote_target (void)
{
#ifdef __QNXTARGET__
  return new cuda_remote_target<qnx_remote_target<remote_target>> ();
#else
  return new cuda_remote_target<remote_target> ();
#endif
}

#ifndef __QNXTARGET__
static
remote_target *
cuda_new_extended_remote_target (void)
{
#ifdef __QNXTARGET__
  return new cuda_remote_target<qnx_remote_target<extended_remote_target>> ();
#else
  return new cuda_remote_target<extended_remote_target> ();
#endif
}
#endif

template <class parent>
void
cuda_remote_target<parent>::cuda_do_resume (ptid_t ptid, int sstep, int host_sstep, enum gdb_signal ts)
{
  uint32_t dev;

  cuda_sstep_reset (sstep);

  // Is focus on host?
  if (!cuda_current_focus::isDevice ())
    {
      // If not sstep - resume devices
      if (!host_sstep)
        for (dev = 0; dev < cuda_state::get_num_devices (); ++dev)
	  cuda_state::device_resume (dev);

      // resume the host
      parent::resume (ptid, sstep, ts);
      return;
    }

   // sstep the device
  if (sstep)
    {
      if (cuda_sstep_execute (inferior_ptid))
	{
	  /* The following is needed because, even though we are dealing with
	     a remote target, device single-stepping doesn't call into
	     remote_wait.  Thus, it doesn't set the appropriate state for the
	     async handler.

	     Therefore we fake the fact that there is a remote event waiting
	     in the queue so the event handler can call the proper hooks that
	     ultimately will call target_wait (and thus cuda_remote_wait) so
	     we can report the device single-stepping event back.  */
#ifndef __QNXTARGET__
	  /* On QNX this workaround does not seem to be required */
	  remote_report_event (this, 1);
#endif
	  return;
	}
      /* If single stepping failed, plant a temporary breakpoint
         at the previous frame and resume the device */
      cuda_sstep_reset (false);
      cuda_insert_step_resume_breakpoint_at_caller (get_current_frame ());
      cuda_insert_breakpoints ();
    }

  // resume the device
  const auto& cur = cuda_current_focus::get ().physical ();
  cuda_state::device_resume (cur.dev ());

  // resume other devices
  if (!cuda_notification_pending ())
    for (dev = 0; dev < cuda_state::get_num_devices (); ++dev)
      if (dev != cur.dev ())
	cuda_state::device_resume (dev);

  // resume the host
  parent::resume (ptid, 0, ts);
}

static bool
cuda_initialize_remote_target (struct target_ops *ops)
{
  CUDBGResult get_debugger_api_res;
  CUDBGResult set_callback_api_res;
  CUDBGResult api_initialize_res;
  uint32_t num_sms = 0;
  uint32_t num_warps = 0;
  uint32_t num_lanes = 0;
  uint32_t num_registers = 0;
  uint32_t num_uregisters = 0;
  uint32_t dev_id = 0;
  bool driver_is_compatible;
  char *dev_type;
  char *sm_type;
  remote_target *remote = (remote_target *)ops;
#ifdef __QNXTARGET__
  struct cmd_list_element *alias = NULL;
  struct cmd_list_element *prefix_cmd = NULL;
  struct cmd_list_element *handle_cmd = NULL;
#endif

  if (cuda_initialized)
    return true;

#ifdef __QNXTARGET__
  /* If shared objects haven't been loaded yet, we won't find any symbol. */
  bool found = false;

  for (so_list *so : current_program_space->solibs())
    {
      while (so)
        {
          if (strstr (so->so_name, "libcuda.so") != NULL)
            {
              found = true;
              break;
            }
          so = so->next;
        }

      if (found)
        break;
    }

  /* libcuda isn't loaded yet, we won't find any symbol. */
  if (!found)
    return false;

  /* Ignore signals that we use for notification passing on QNX.
     See cuda-notifications.c for details. */
  if (!lookup_cmd_composition ("handle", &alias, &prefix_cmd, &handle_cmd))
    {
      error (_("Failed to lookup the `handle` command."));
    }
  cmd_func (handle_cmd, "SIGEMT nostop noprint nopass", 0);
  cmd_func (handle_cmd, "SIGILL nostop noprint nopass", 0);

  /* Send the target the symbols it needs */
  if (!symbols_are_set)
    {
      /* First assume cuda-gdbserver is backwards compatible */
      cuda_remote_set_symbols ((struct remote_target *)ops, true, &symbols_are_set);
      if (!symbols_are_set)
	{
	  warning (_("Cannot access libcuda symbols, please set sysroot or solib-search-path. CUDA debugging will not be available."));
	  /* If the above failed, assume it's incompatible and only send
	     the core symbols */
	  cuda_remote_set_symbols ((struct remote_target *)ops, false, &symbols_are_set);
	  if (!symbols_are_set)
	    {
	      return false;
	    }
	}
    }
#endif

  /* Ask cuda-gdbserver to initialize. */
  uint32_t debugapi_major = 0;
  uint32_t debugapi_minor = 0;
  uint32_t debugapi_revision = 0;
  cuda_remote_initialize (remote,
			  &get_debugger_api_res, &set_callback_api_res, &api_initialize_res,
                          &cuda_initialized, &cuda_debugging_enabled, &driver_is_compatible,
			  &debugapi_major, &debugapi_minor, &debugapi_revision);

  cuda_debugapi::print_get_api_error (get_debugger_api_res);
  cuda_debugapi::handle_initialization_error (api_initialize_res);
  cuda_debugapi::handle_set_callback_api_error (set_callback_api_res);
  cuda_debugapi::set_api_version (debugapi_major, debugapi_minor, debugapi_revision);

  if (!driver_is_compatible)
    {
      target_kill ();
      error (_("CUDA application cannot be debugged. The CUDA driver is not compatible."));
    }
  if (!cuda_initialized)
    return false;

  cudbgipcInitialize ();
  cuda_state::initialize ();
  for (dev_id = 0; dev_id < cuda_state::get_num_devices (); dev_id++)
    {
      cuda_remote_query_device_spec (remote,
				     dev_id, &num_sms, &num_warps, &num_lanes,
                                     &num_registers, &num_uregisters, &dev_type, &sm_type);
      cuda_state::set_device_spec (dev_id, num_sms, num_warps, num_lanes,
                                   num_registers, num_uregisters, dev_type, sm_type);
    }
  cuda_remote_set_option (remote);
  cuda_gdb_session_create ();
  cuda_update_report_driver_api_error_flags ();
  
  return true;
}

#ifdef __QNXTARGET__
void
cuda_finalize_remote_target (void)
{
  symbols_are_set = false;
}
#endif

template <class parent>
void
cuda_remote_target<parent>::kill ()
{
  cuda_debugapi::finalize ();
  cuda_cleanup ();
  cuda_gdb_session_destroy ();

  parent::kill ();
}

template <class parent>
void
cuda_remote_target<parent>::mourn_inferior (void)
{
  /* Mark breakpoints uninserted in case something tries to delete a
     breakpoint while we delete the inferior's threads (which would
     fail, since the inferior is long gone).  */
  mark_breakpoints_out ();

  cuda_cleanup ();
  cuda_gdb_session_destroy ();

  parent::mourn_inferior ();
}

extern int cuda_host_want_singlestep;

template <class parent>
void
cuda_remote_target<parent>::resume (ptid_t ptid, int sstep, enum gdb_signal ts)
{
  bool cuda_event_found = false;
  CUDBGEvent event;
  int host_want_sstep = cuda_host_want_singlestep;

  cuda_trace ("cuda_resume: sstep=%d", sstep);
  cuda_host_want_singlestep = 0;

  /* In cuda-gdb we have two types of device exceptions :
     Recoverable : CUDA_EXCEPTION_WARP_ASSERT
     Nonrecoverable : All others (e.g. CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS)

     The main difference is that a recoverable exception ensures that device
     state is consistent. Therefore, the user can request that the device
     continue execution. Currently, CUDA_EXCEPTION_WARP_ASSERT is the only
     recoverable exception.

     When a device side exception is hit, it sets cuda_exception in cuda_wait.
     In the case of a nonrecoverable exception, the cuda_resume call
     kills the host application and return early. The subsequent cuda_wait
     call cleans up the exception state.
     In the case of a recoverable exception, cuda-gdb must reset the exception
     state here and can then continue executing.
     In the case of CUDA_EXCEPTION_WARP_ASSERT, the handling of the
     exception (i.e. printing the assert message) is done as part of the
     cuda_wait call.
  */
  cuda_exception ex;
  if (ex.valid ())
    {
      if (ex.recoverable ())
	{
	  cuda_trace ("cuda_resume: recoverable exception found\n");
	}
      else
	{
	  cuda_trace ("cuda_resume: exception found");
	  cuda_cleanup ();
	  parent::resume (ptid, 0, GDB_SIGNAL_KILL);
	  return;
	}
    }

  cuda_notification_mark_consumed ();
  cuda_sigtrap_restore_settings ();

  if (cuda_notification_aliased_event ())
    {
      cuda_notification_reset_aliased_event ();
      cuda_debugapi::get_next_sync_event (&event);
      cuda_event_found = event.kind != CUDBG_EVENT_INVALID;

      if (cuda_event_found)
        {
          cuda_process_events (&event, CUDA_EVENT_SYNC);
          sendAck = true;
        }
    }

  if (sendAck)
    {
      cuda_debugapi::acknowledge_sync_events ();
      sendAck = false;
    }

  cuda_do_resume (ptid, sstep, host_want_sstep, ts);

  cuda_clock_increment ();

  cuda_trace ("cuda_resume: done");
}

extern void (*gdb_old_sighand_func) (int);
extern void remote_interrupt (int);

template <class parent>
ptid_t
cuda_remote_target<parent>::wait (ptid_t ptid, struct target_waitstatus *ws, target_wait_flags target_options)
{
  ptid_t r;
  uint32_t dev, dev_id;
  uint64_t grid_id;
  kernel_t kernel;
  bool cuda_event_found = false;
  CUDBGEvent event, asyncEvent;
  struct thread_info *tp;

  cuda_trace ("cuda_wait");

  if (cuda_sstep_is_active ())
    {
      /* Cook the ptid and wait_status if single-stepping a CUDA device.
       * This will avoid the call to host wait and we will be able to
       * continue our handling code below. */
      cuda_trace ("cuda_wait: single-stepping");
      r = cuda_sstep_ptid ();

      /* Check if the device encountered an exception while stepping. */
      cuda_exception ex;
      if (ex.valid ())
	{
	  /* We will handle the exception printing below */
	  cuda_trace ("cuda_wait: single-stepping encountered an exception");
	}
      else
	{
	  /* Check if C-c was sent to a remote application or if quit_flag is set.
	   * quit_flag is set by gdb handle_sigint() signal handler */
#ifdef __QNXTARGET__
	  if (cuda_remote_check_pending_sigint (this, r) || check_quit_flag())
#else /* __QNXTARGET__ */
	  if (cuda_remote_check_pending_sigint (this) || check_quit_flag())
#endif /* __QNXTARGET__ */
	    {
	      ws->set_stopped (GDB_SIGNAL_INT);
	      cuda_set_signo (GDB_SIGNAL_INT);
	    }
	  else
	    {
	      ws->set_stopped (GDB_SIGNAL_TRAP);
	      cuda_set_signo (GDB_SIGNAL_TRAP);

	      /* If we single stepped the last warp on the device, then the
	       * launch has completed.  However, we do not see the event for
	       * kernel termination until we resume the application.  We must
	       * explicitly handle this here by indicating the kernel has
	       * terminated and switching to the remaining host thread. */
	      if (cuda_sstep_kernel_has_terminated ())
	        {
		  cuda_trace ("cuda_wait: single stepped to kernel exit");
		  /* Only destroy the kernel that has been stepped to its exit */
		  dev_id  = cuda_sstep_dev_id ();
		  grid_id = cuda_sstep_grid_id ();
		  kernel = kernels_find_kernel_by_grid_id (dev_id, grid_id);
		  kernels_terminate_kernel (kernel);

		  // Invalidate current coordinates
		  cuda_current_focus::invalidate ();

		  /* Consume any asynchronous events, if necessary.  We need to do 
		   * this explicitly here, since we're taking the quick path out of 
		   * this routine (and bypassing the normal check for API events). */ 
		  cuda_debugapi::get_next_async_event (&asyncEvent); 
		  if (asyncEvent.kind != CUDBG_EVENT_INVALID) 
		    cuda_process_events (&asyncEvent, CUDA_EVENT_ASYNC);

		  /* Update device state/kernels */
		  kernels_update_terminated ();
		  cuda_update_convenience_variables ();

		  /* Update focus */
		  cuda_current_focus::update ();

		  tp = find_thread_ptid (this, r);
		  gdb_assert (tp);
		  tp->control.step_range_end = 1;
		  /* Inform that we need to switch cuda focus if we are
		   * still focused on the device. */
		  if (cuda_current_focus::isDevice ())
		    {
		      tp->need_cuda_context_switch = true;
		      tp->new_cuda_coords = cuda_current_focus::get ();
		    }
		  return r;
	       	}
              }
	}
    }
  else
    {
      cuda_current_focus::invalidate ();
      cuda_trace ("cuda_wait: host_wait\n");
      r = parent::wait (ptid, ws, target_options);
      cuda_trace ("cuda_wait: host_wait done");

      /* GDB reads events asynchronously without blocking. The remote may have
         taken too long to reply and GDB did not get any events back.  Check if
         this is the case and just return.  */
      if (ws->kind () == TARGET_WAITKIND_IGNORE
	  || ws->kind () == TARGET_WAITKIND_NO_RESUMED)
	{
	  cuda_trace ("cuda_wait: ignored wait");
	  return r;
	}
      
      if (ws->kind () == TARGET_WAITKIND_STOPPED &&
	  ws->sig () == GDB_SIGNAL_0)
	{
	  /* GDB is trying to stop this thread. Let it do it. */
	  cuda_trace ("cuda_wait: host is stopping thread");
	  return r;
	}
    }

  /* Immediately detect if the inferior is exiting.
     In these situations, do not investigate the device. */
  if (ws->kind () == TARGET_WAITKIND_EXITED)
    {
      cuda_trace ("cuda_wait: target is exiting, avoiding device inspection");
      return r;
    }

  if (r != minus_one_ptid && r != null_ptid && find_thread_ptid (this, r) != nullptr)
    {
      /* FIXME: We shouldn't be using switch_to_thread here.
       * It is no longer valid to rely on this for inferior_wait. 
       * See the comment in infrun.c in do_target_wait_1. */
      switch_to_thread (this,r);
    }

  /* Try to initialize CUDA if we haven't done so already. This may throw an exception,
   * in which case we want to ignore it. */
  if (!cuda_initialized)
    {
      try
	{
	  auto res = cuda_initialize_remote_target (current_inferior ()->top_target ());
	  if (!res)
	    {
	      cuda_trace ("cuda_wait: cuda_initialize_remote_target() failed, return pid %d", r.pid ());
	      return r;
	    }
	}
      catch (const gdb_exception_error &e)
	{
	  cuda_trace ("cuda_wait: ignoring exception during initialize.");
	  return r;
	}
    }

  /* Suspend all the CUDA devices. */
  cuda_trace ("cuda_wait: suspend devices");
  for (dev = 0; dev < cuda_state::get_num_devices (); ++dev)
    cuda_state::device_suspend (dev);

  cuda_remote_query_trace_message (this);
  /* Check for ansynchronous events.  These events do not require
     acknowledgement to the debug API, and may arrive at any time
     without an explicit notification. */
  cuda_debugapi::get_next_async_event (&asyncEvent);
  if (asyncEvent.kind != CUDBG_EVENT_INVALID)
    cuda_process_events (&asyncEvent, CUDA_EVENT_ASYNC);

  cuda_notification_analyze (r, ws, 0);

  if (cuda_notification_received ())
    {
      /* Check if there is any CUDA event to be processed */
      cuda_debugapi::get_next_sync_event (&event);
      cuda_event_found = event.kind != CUDBG_EVENT_INVALID;
     }

  /* Handle all the CUDA events immediately.  In particular, for
     GPU events that may happen without prior notification (GPU
     grid launches, for example), API events will be packed
     alongside of them, so we need to process the API event first. */
  if (cuda_event_found)
    {
      cuda_process_events (&event, CUDA_EVENT_SYNC);
      sendAck = true;

    }

  /* Update the info about the kernels */
  kernels_update_terminated ();

  /* Decide which thread/kernel to switch focus to. */
  cuda_coords c;
  cuda_exception exp;
  if (exp.valid ())
    {
      cuda_trace ("cuda_wait: stopped because of an exception");
      cuda_coords coord {exp.coords ()};
      cuda_current_focus::set (coord);
      exp.printMessage ();
      ws->set_stopped (exp.gdbSignal ());
      cuda_set_signo (exp.gdbSignal ());
      tp = find_thread_ptid (this, r);
      gdb_assert (tp);
      tp->need_cuda_context_switch = true;
      tp->new_cuda_coords = cuda_current_focus::get ();
    }
  else if (cuda_sstep_is_active ())
    {
      cuda_trace ("cuda_wait: stopped because we are single-stepping");
      cuda_current_focus::update ();
      tp = find_thread_ptid (this, r);
      gdb_assert (tp);
      tp->need_cuda_context_switch = true;
      tp->new_cuda_coords = cuda_current_focus::get ();
    }
  else if (cuda_breakpoint_hit_p (c))
    {
      cuda_trace ("cuda_wait: stopped because of a breakpoint");
      cuda_set_signo (GDB_SIGNAL_TRAP);
      ws->set_stopped (GDB_SIGNAL_TRAP);
      /* Cannot trust the returned coords prior to ampere. */
      //cuda_coords_set_current (&c);
      cuda_current_focus::update (true);
      tp = find_thread_ptid (this, r);
      gdb_assert (tp);
      tp->need_cuda_context_switch = true;
      tp->new_cuda_coords = cuda_current_focus::get ();
    }
  else if (cuda_state::broken (c))
    {
      cuda_trace ("cuda_wait: stopped because there are broken warps (induced trap?)");
      cuda_set_signo (GDB_SIGNAL_TRAP);
      ws->set_stopped (GDB_SIGNAL_TRAP);
      cuda_current_focus::set (c);
      tp = find_thread_ptid (this, r);
      gdb_assert (tp);
      tp->need_cuda_context_switch = true;
      tp->new_cuda_coords = cuda_current_focus::get ();
    }
  else if (cuda_debugapi::get_attach_state () == CUDA_ATTACH_STATE_APP_READY)
    {
      /* Finished attaching to the CUDA app.
         Preferably switch focus to a device if possible */
      struct inferior *inf = find_inferior_pid (this, r.pid ());
      cuda_trace ("cuda_wait: stopped because we attached to the CUDA app");
      cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_COMPLETE);
      inf->control.stop_soon = STOP_QUIETLY;
      cuda_current_focus::update ();
    }
  else if (cuda_debugapi::get_attach_state () == CUDA_ATTACH_STATE_DETACH_COMPLETE)
    {
      /* Finished detaching from the CUDA app. */
      struct inferior *inf = find_inferior_pid (this, r.pid ());
      cuda_trace ("cuda_wait: stopped because we detached from the CUDA app");
      inf->control.stop_soon = STOP_QUIETLY;
    }
  else if (cuda_event_found)
    {
      cuda_trace ("cuda_wait: stopped because of a CUDA event");
      cuda_sigtrap_set_silent ();
      cuda_current_focus::update ();
      tp = find_thread_ptid (this, r);
      gdb_assert (tp);
      tp->need_cuda_context_switch = true;
      tp->new_cuda_coords = cuda_current_focus::get ();
    }
  else if ((ws->kind () == TARGET_WAITKIND_STOPPED ||
	    ws->kind () == TARGET_WAITKIND_SIGNALLED) &&
	   ws->sig () == GDB_SIGNAL_INT)
    {
      /* CTRL-C was hit. Preferably switch focus to a device if possible */
      cuda_trace ("cuda_wait: stopped because a SIGINT was received.");
      cuda_set_signo (GDB_SIGNAL_INT);
      cuda_current_focus::update ();
    }
  else if (cuda_notification_received ())
    {
      /* No reason found when actual reason was consumed in a previous iteration (timeout,...) */
      cuda_trace ("cuda_wait: stopped for no visible CUDA reason.");
      cuda_set_signo (GDB_SIGNAL_TRAP); /* Dummy signal. We stopped after all. */
      cuda_current_focus::invalidate ();
    }
  else
    {
      cuda_trace ("cuda_wait: stopped for a non-CUDA reason.");
      cuda_set_signo (GDB_SIGNAL_TRAP);
      cuda_current_focus::invalidate ();
    }

  cuda_adjust_host_pc (r);

  /* CUDA - managed memory */
  /* Bugfix: try to find the thread first, if we can't find it, the inferior has exited. */
  tp = find_thread_ptid (this, r);
  if ((ws->kind () == TARGET_WAITKIND_STOPPED || ws->kind () == TARGET_WAITKIND_SIGNALLED) &&
      (ws->sig () == GDB_SIGNAL_BUS || ws->sig () == GDB_SIGNAL_SEGV) &&
      tp)
    {
      uint64_t addr = 0;
      struct gdbarch *arch = target_gdbarch ();
      int arch_ptr_size = gdbarch_ptr_bit (arch) / 8;
      LONGEST len = arch_ptr_size;
      LONGEST offset = arch_ptr_size == 8 ? 0x10 : 0x0c;
      LONGEST read = 0;
      gdb_byte *buf = (gdb_byte *)&addr;
      bool inf_exec;

      inf_exec = tp->executing ();

      /* Mark inferior_ptid as not executing while reading object signal info*/
      set_executing (this, r, false);
      read = target_read (this, TARGET_OBJECT_SIGNAL_INFO, NULL, buf, offset, len);
      set_executing (this, r, inf_exec);

      /* Check the results */
      if (read == len && cuda_managed_address_p (addr))
        {
	  if (ws->kind () == TARGET_WAITKIND_STOPPED)
	    ws->set_stopped (GDB_SIGNAL_CUDA_INVALID_MANAGED_MEMORY_ACCESS);
	  else
	    ws->set_signalled (GDB_SIGNAL_CUDA_INVALID_MANAGED_MEMORY_ACCESS);
          cuda_set_signo (GDB_SIGNAL_CUDA_INVALID_MANAGED_MEMORY_ACCESS);
        }
    }
  cuda_managed_memory_clean_regions();

  /* Switch focus and update related data */
  cuda_update_convenience_variables ();
  if (cuda_current_focus::isDevice ())
    /* Must be last, once focus and elf images have been updated */
    switch_to_cuda_thread (cuda_current_focus::get ());

  cuda_trace ("cuda_wait: done");
  return r;
}

template <class parent>
void
cuda_remote_target<parent>::fetch_registers (struct regcache *regcache, int regno)
{
  uint64_t val = 0;
  struct gdbarch *gdbarch = regcache->arch ();
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);
  enum register_status status;

  /* delegate to the host routines when not on the device */
  if (!cuda_current_focus::isDevice ())
    {
      parent::fetch_registers (regcache, regno);
      return;
    }

  const auto& c = cuda_current_focus::get ().physical ();

  /* if all the registers are wanted, then we need the host registers and the
     device PC */
  if (regno == -1)
    {
      parent::fetch_registers (regcache, regno);
      val = cuda_state::lane_get_virtual_pc (c.dev (), c.sm (), c.wp (), c.ln ());
      regcache->raw_supply (pc_regnum, &val);
      return;
    }

  /* get the PC */
  if (regno == pc_regnum )
    {
      val = cuda_state::lane_get_virtual_pc (c.dev (), c.sm (), c.wp (), c.ln ());
      regcache->raw_supply (pc_regnum, &val);
      return;
    }

  if (cuda_regular_register_p (gdbarch, regno))
    {
      /* raw register */
      val = cuda_state::lane_get_register (c.dev (), c.sm (), c.wp (), c.ln (), regno);
      regcache->raw_supply (regno, &val);
      return;
    }

  status = cuda_pseudo_register_read (gdbarch, regcache, regno, (gdb_byte *)&val);
  gdb_assert (status == REG_VALID);
  
  regcache->raw_supply (regno, &val);
}

template <class parent>
void
cuda_remote_target<parent>::store_registers (struct regcache *regcache, int regno)
{
  uint64_t val;
  struct gdbarch *gdbarch = regcache->arch ();
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);
  int num_regs = gdbarch_num_regs (gdbarch);

  gdb_assert (regno >= 0 && regno < num_regs);

  if (!cuda_current_focus::isDevice ())
    {
      parent::store_registers (regcache, regno);
      return;
    }

  if (regno == pc_regnum)
    error (_("The PC of CUDA thread is not writable"));

  regcache->raw_collect (regno, &val);
  cuda_register_write (gdbarch, regcache, regno, (gdb_byte *) &val);
}

template <class parent>
int
cuda_remote_target<parent>::insert_breakpoint (struct gdbarch *gdbarch,
					       struct bp_target_info *bp_tgt)
{
  uint32_t dev;
  bool inserted;

  gdb_assert (bp_tgt->owner != NULL ||
              gdbarch_bfd_arch_info (gdbarch)->arch == bfd_arch_arm ||
              gdbarch_bfd_arch_info (gdbarch)->arch == bfd_arch_aarch64);

  if (!bp_tgt->owner || !bp_tgt->owner->cuda_breakpoint)
    return parent::insert_breakpoint (gdbarch, bp_tgt);

  /* Insert the breakpoint on whatever device accepts it (valid address). */
  inserted = false;
  for (dev = 0; dev < cuda_state::get_num_devices (); ++dev)
    {
      inserted |= cuda_debugapi::set_breakpoint (dev, bp_tgt->reqstd_address);
    }

  /* Make sure we save the address where the actual breakpoint was placed.  */
  if (inserted)
    bp_tgt->placed_address = bp_tgt->reqstd_address;

  return !inserted;
}

template <class parent>
int
cuda_remote_target<parent>::remove_breakpoint (struct gdbarch *gdbarch,
					       struct bp_target_info *bp_tgt,
					       enum remove_bp_reason reason)
{
  uint32_t dev;
  bool removed;

  if (!bp_tgt->owner || !bp_tgt->owner->cuda_breakpoint)
    return parent::remove_breakpoint (gdbarch, bp_tgt, reason);

  /* Removed the breakpoint on whatever device accepts it (valid address). */
  removed = false;
  for (dev = 0; dev < cuda_state::get_num_devices (); ++dev)
    {
      /* We need to remove breakpoints even if no kernels remain on the device */
      removed |= cuda_debugapi::unset_breakpoint (dev, bp_tgt->placed_address);
    }
  return !removed;
}

/* The whole Linux siginfo structure is presented to the user, but, internally,
   only the si_signo matters. We do not save the siginfo object. Instead we
   save only the signo. Therefore any read/write to any other field of the
   siginfo object will have no effect or will return 0. */
static enum target_xfer_status
cuda_remote_xfer_siginfo (remote_target *remote, enum target_object object,
                          const char *annex, gdb_byte *readbuf,
                          const gdb_byte *writebuf, ULONGEST offset,
			  LONGEST len, ULONGEST *xfered_len)
{
  /* the size of siginfo is not consistent between ptrace and other parts of
     GDB. On 32-bit Linux machines, the layout might be 64 bits. It does not
     matter for CUDA because only signo is used and the rest is set to zero. We
     just allocate 8 extra bytes and bypass the issue. On 64-bit Mac, the
     difference is 24 bytes. Therefore take the max of the 2 values. */
  gdb_byte buf[sizeof (siginfo_t) + 24];
  siginfo_t *siginfo = (siginfo_t *) buf;

  gdb_assert (remote);
  gdb_assert (object == TARGET_OBJECT_SIGNAL_INFO);
  gdb_assert (readbuf || writebuf);

  if (!cuda_current_focus::isDevice ())
    return TARGET_XFER_E_IO ;

  if (offset >= sizeof (buf))
    return TARGET_XFER_E_IO;

  if (offset + len > sizeof (buf))
    len = sizeof (buf) - offset;

  memset (buf, 0 , sizeof buf);

  if (readbuf)
    {
      siginfo->si_signo = cuda_get_signo ();
      memcpy (readbuf, siginfo + offset, len);
    }
  else
    {
      memcpy (siginfo + offset, writebuf, len);
      cuda_set_signo (siginfo->si_signo);
    }

  *xfered_len = len;
  return TARGET_XFER_OK;
}

template <class parent>
enum target_xfer_status
cuda_remote_target<parent>::xfer_partial (enum target_object object, const char *annex,
					  gdb_byte *readbuf, const gdb_byte *writebuf,
					  ULONGEST offset, ULONGEST len, ULONGEST *xfered_len)
{
  enum target_xfer_status status = TARGET_XFER_OK;

  /* If focus set on device, call the host routines directly */
  if (!cuda_current_focus::isDevice ())
    {
      status = parent::xfer_partial (object, annex, readbuf,
				     writebuf, offset, len,
				     xfered_len);
      return status;
    }

  switch (object)
  {
    /* See if this address is in pinned system memory first.  This refers to
       system memory allocations made by the inferior through the CUDA API, and
       not those made by directly using mmap(). */
    case TARGET_OBJECT_MEMORY:

      if (readbuf)
	{
	  if (cuda_debugapi::read_pinned_memory (offset, readbuf, len))
	    *xfered_len = len;
	  else
	    status = TARGET_XFER_E_IO;
	}
      else if (writebuf)
	{
	  if (cuda_debugapi::write_pinned_memory (offset, writebuf, len))
	    *xfered_len = len;
	  else
	    status = TARGET_XFER_E_IO;
	}
      break;

    /* The stack lives in local memory for ABI compilations. */
    case TARGET_OBJECT_STACK_MEMORY:
    {
      if (!cuda_current_focus::isDevice ())
	return TARGET_XFER_E_IO;
      
      const auto& c = cuda_current_focus::get ().physical ();

      if (readbuf)
        {
          if (cuda_debugapi::read_local_memory (c.dev (), c.sm (), c.wp (), c.ln (), offset, readbuf, len))
            *xfered_len = len;
	  else
	    status = TARGET_XFER_E_IO;
        }
      else if (writebuf)
        {
          if (cuda_debugapi::write_local_memory (c.dev (), c.sm (), c.wp (), c.ln (), offset, writebuf, len))
            *xfered_len = len;
	  else
	    status = TARGET_XFER_E_IO;
        }
      break;
    }
    /* When stopping on the device, build a simple siginfo object */
    case TARGET_OBJECT_SIGNAL_INFO:

      status = cuda_remote_xfer_siginfo (this, object, annex, readbuf,
					 writebuf, offset, len,
					 xfered_len);
      break;
  }

  if (*xfered_len < len)
    {
      status = parent::xfer_partial (object, annex, readbuf,
				     writebuf, offset, len,
				     xfered_len);
    }
  return status;
}

template <class parent>
struct gdbarch *
cuda_remote_target<parent>::thread_architecture (ptid_t ptid)
{
  if (cuda_current_focus::isDevice ())
    return cuda_get_gdbarch ();
  else
    return target_gdbarch();
}

template <class parent>
void
cuda_remote_target<parent>::prepare_to_store (struct regcache *regcache)
{
  if (regcache->arch () == cuda_get_gdbarch ())
    return;
  parent::prepare_to_store (regcache);
}

void
set_cuda_remote_flag (bool connected)
{
  cuda_remote = connected;
}

void
cuda_remote_attach (void)
{
  struct cmd_list_element *alias = NULL;
  struct cmd_list_element *prefix_cmd = NULL;
  struct cmd_list_element *cmd = NULL;
  const char *cudbgApiAttach = "(void) cudbgApiAttach()";
  CORE_ADDR debugFlagAddr;
  CORE_ADDR sessionIdAddr;
  CORE_ADDR attachDataAvailableFlagAddr;
  const unsigned char one = 1;
  unsigned char attachDataAvailable;
  uint32_t sessionId = 0;
  unsigned int timeOut = 5000; // ms
  unsigned int timeElapsed = 0;
  const unsigned int sleepTime = 1; // ms
  bool need_retry = 0;
  uint64_t internal_error_code = 0;
  unsigned retry_count = 0;
  unsigned retry_delay = 100; // ms
  unsigned app_init_timeout = 5000; // ms
  unsigned char *sigs = NULL;
  struct cuda_signal_info_st sigstop_save = {0};

  if (cuda_debugapi::get_attach_state() != CUDA_ATTACH_STATE_NOT_STARTED && 
    cuda_debugapi::get_attach_state() != CUDA_ATTACH_STATE_DETACH_COMPLETE)
    return;
#ifndef __QNXTARGET__
  if (!cuda_get_current_remote_target ()->remote_query_attached (inferior_ptid.pid ()))
    return;
#endif

  auto res = cuda_initialize_remote_target (current_inferior ()->top_target ());
  if (!res)
    {
      /* FIXME: initialize is non-fatal. Investigate why this fails. */
      cuda_trace ("Failed to initialize remote target.");
    }

  debugFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_IPC_FLAG_NAME));
  sessionIdAddr = cuda_get_symbol_address (_STRING_(CUDBG_SESSION_ID));

  /* Return early if CUDA driver isn't available. Attaching to the host
     process has already been completed at this point. */
  if (!debugFlagAddr || !sessionIdAddr)
    return;
  
  target_read_memory (sessionIdAddr, (gdb_byte *)&sessionId, sizeof(sessionId));
  if (!sessionId)
    return;

  /* If the CUDA driver has been loaded but software preemption has been turned
     on, stop the attach process. */
  if (cuda_options_software_preemption ())
    error (_("Attaching to a running CUDA process with software preemption "
             "enabled in the debugger is not supported."));

  cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_IN_PROGRESS);

  if (!lookup_cmd_composition ("call", &alias, &prefix_cmd, &cmd))
    error (_("Failed to initiate attach."));

  attachDataAvailableFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_ATTACH_HANDLER_AVAILABLE));

  /* If this is not available, the CUDA driver doesn't support attaching.  */
  if (!attachDataAvailableFlagAddr)
    error (_("This CUDA driver does not support attaching to a running CUDA process."));

  cuda_signal_set_silent (SIGSTOP, &sigstop_save);

  remote_target *rt = cuda_get_current_remote_target();

  do
  {
    /* Try to init debugger's backend */
    sigs = cuda_gdb_bypass_signals ();
    cuda_gdb_bypass_signals_cleanup cleanup (sigs);
    cmd_func (cmd, cudbgApiAttach, 0);
    /* manually cleanup */
    cleanup.release ();
    cuda_nat_bypass_signals_cleanup(sigs);

    internal_error_code = cuda_get_last_driver_internal_error_code();

    /* CUDBG_ERROR_ATTACH_NOT_POSSIBLE can be returned in two scenarios:
     * 1. Attach is really not possible
     * 2. Critical section's mutex is taken, attaching would cause a deadlock */
    need_retry = (unsigned int)internal_error_code == CUDBG_ERROR_ATTACH_NOT_POSSIBLE;

    if (need_retry)
    {
      /* Resume the target */
      prepare_execution_command (current_inferior ()->top_target (), true);
      continue_1 (true);

      usleep(retry_delay * 1000);

      /* This is similar to remote_target::interrupt(), except this will be
       * handled by our attach-specific code server-side.
       */
      rt->remote_serial_write(SERIAL_REMOTE_STOP_CMD, 1);

      /* Our code server-side will forge an expected stop event, so that it will
       * get reported to the client. */
      cuda_wait_for_inferior ();

      set_running (current_inferior ()->process_target (), minus_one_ptid, 0);
      set_executing (current_inferior ()->process_target (), minus_one_ptid, 0);

      retry_count++;
    }
  } while (need_retry && (retry_count * retry_delay < app_init_timeout));

  /* We are re-using the ATTACH_NOT_POSSIBLE error code for delayed attach,
   * therefore a timeout will allow us to determine if the error code is
   * genuinely not possible. */
  if (need_retry)
    error (_("Attaching not possible. "
             "Please verify that software preemption is disabled "
             "and that nvidia-cuda-mps-server is not running."));

  cuda_signal_restore_settings (SIGSTOP, &sigstop_save);

  /* Wait till the backend has started up and is ready to service API calls */
  while (cuda_debugapi::initialize () != CUDBG_SUCCESS)
    {
      if (timeElapsed < timeOut)
        usleep(sleepTime * 1000);
      else
        error (_("Timed out waiting for the CUDA API to initialize."));

      timeElapsed += sleepTime;
    }

  /* Check if more data is available from the inferior */
  target_read_memory (attachDataAvailableFlagAddr, &attachDataAvailable, 1);

  if (attachDataAvailable)
    {
      int  cnt;
      /* Resume the inferior to collect more data. CUDA_ATTACH_STATE_COMPLETE and
	 CUDBG_IPC_FLAG_NAME will be set once this completes. */

      for (cnt=0;
	   cnt < 1000
	     && cuda_debugapi::get_attach_state () == CUDA_ATTACH_STATE_IN_PROGRESS;
           cnt++)
	{
	  prepare_execution_command (current_inferior ()->top_target (), true);
	  continue_1 (false);
	  /* force resumed state to false */
	  auto resumed_state = current_inferior ()->process_target ()->commit_resumed_state;
	  current_inferior ()->process_target ()->commit_resumed_state = false;
	  cuda_wait_for_inferior ();
	  current_inferior ()->process_target ()->commit_resumed_state = resumed_state;
	  normal_stop ();
	}

      /* All threads are stopped at this point.  */
      set_running (current_inferior ()->process_target (), minus_one_ptid, 0);
    }
  else
    {
      /* Enable debugger callbacks from the CUDA driver */
      target_write_memory (debugFlagAddr, &one, 1);

      /* No data to collect, attach complete. */
      cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_COMPLETE);
    }
}

template <class parent>
void
cuda_remote_target<parent>::detach (inferior *inf, int from_tty)
{
  /* If attach wasn't completed,
     treat the inferior as a host-only process */
  if (cuda_debugapi::get_attach_state() == CUDA_ATTACH_STATE_COMPLETE &&
      cuda_get_current_remote_target ()->remote_query_attached (inferior_ptid.pid ()))
    cuda_do_detach (inf, true);

  parent::detach (inf, from_tty);
}

void
#ifdef __QNXTARGET__
cuda_version_handshake (const char *version_string)
#else /* __QNXTARGET__ */
cuda_remote_version_handshake (remote_target *remote,
			       const struct protocol_feature *feature,
			       enum packet_support support,
                               const char *version_string)
#endif /* __QNXTARGET__ */
{
  uint32_t server_major, server_minor, server_rev;

#ifndef __QNXTARGET__
  gdb_assert (strcmp (feature->name, "CUDAVersion") == 0);
  if (support != PACKET_ENABLE)
    error (_("Server doesn't support CUDA.\n"));
#endif /* __QNXTARGET__ */

  gdb_assert (version_string);
  sscanf (version_string, "%d.%d.%d", &server_major, &server_minor, &server_rev);

  if (server_major == CUDBG_API_VERSION_MAJOR &&
      server_minor == CUDBG_API_VERSION_MINOR &&
      server_rev   == CUDBG_API_VERSION_REVISION)
    return;

  target_kill ();
  error (_("cuda-gdb version (%d.%d.%d) is not compatible with "
           "cuda-gdbserver version (%d.%d.%d).\n"
           "Please use the same version of cuda-gdb and cuda-gdbserver."),
           CUDBG_API_VERSION_MAJOR,
           CUDBG_API_VERSION_MINOR,
           CUDBG_API_VERSION_REVISION,
           server_major, server_minor, server_rev);
}
