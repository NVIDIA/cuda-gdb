/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2013 NVIDIA Corporation
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


#include <stdbool.h>
#include <signal.h>
#include "defs.h"
#include "gdbarch.h"
#include "gdbthread.h"
#include "inferior.h"
#include "frame.h"
#include "regcache.h"
#include "remote.h"
#include "command.h"

#include "remote-cuda.h"
#include "cuda-exceptions.h"
#include "cuda-packet-manager.h"
#include "cuda-state.h"
#include "cuda-utils.h"
#include "cuda-events.h"
#include "cuda-options.h"
#include "cuda-notifications.h"
#include "cuda-convvars.h"
#include "libcudbgipc.h"

static struct target_ops *host_target_ops = NULL;
static struct target_ops  host_remote_target_ops;
static struct target_ops  host_extended_remote_target_ops;
static bool sendAck = false;
static bool interrupt_flag = false;

static void
cuda_do_resume (struct target_ops *ops, ptid_t ptid,
                     int sstep, int host_sstep, enum gdb_signal ts)
{
  uint32_t dev;

  cuda_sstep_reset (sstep);

  // Is focus on host?
  if (!cuda_focus_is_device())
    {
      // If not sstep - resume devices
      if (!host_sstep)
        for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
            device_resume (dev);

      // resume the host
      host_target_ops->to_resume (ops, ptid, sstep, ts);
      return;
    }

   // sstep the device
  if (sstep)
    {
      cuda_sstep_execute (inferior_ptid);
      return;
    }

  // resume the device
  device_resume (cuda_current_device ());

  // resume other devices
  if (!cuda_notification_pending ())
    for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
      if (dev != cuda_current_device ())
        device_resume (dev);

  // resume the host
  host_target_ops->to_resume (ops, ptid, 0, ts);
}

int remote_query_attached (int pid);

static void
cuda_initialize_remote_target ()
{
  CUDBGResult get_debugger_api_res;
  CUDBGResult set_callback_api_res;
  CUDBGResult api_initialize_res;
  uint32_t num_sms = 0;
  uint32_t num_warps = 0;
  uint32_t num_lanes = 0;
  uint32_t num_registers = 0;
  uint32_t dev_id = 0;
  bool driver_is_compatible;
  char *dev_type;
  char *sm_type;

  if (cuda_initialized)
    return;


  /* Ask cuda-gdbserver to initialize. */
  cuda_remote_initialize (&get_debugger_api_res, &set_callback_api_res, &api_initialize_res,
                          &cuda_initialized, &cuda_debugging_enabled, &driver_is_compatible);

  cuda_api_handle_get_api_error (get_debugger_api_res);
  cuda_api_handle_initialization_error (api_initialize_res);
  cuda_api_handle_set_callback_api_error (set_callback_api_res);
  if (!driver_is_compatible)
    {
      target_kill ();
      error (_("CUDA application cannot be debugged. The CUDA driver is not compatible."));
    }
  if (!cuda_initialized)
    return;

  cudbgipcInitialize ();
  cuda_system_initialize ();
  for (dev_id = 0; dev_id < cuda_system_get_num_devices (); dev_id++)
    {
      cuda_remote_query_device_spec (dev_id, &num_sms, &num_warps, &num_lanes,
                                     &num_registers, &dev_type, &sm_type);
      cuda_system_set_device_spec (dev_id, num_sms, num_warps, num_lanes,
                                   num_registers, dev_type, sm_type);
    }
  cuda_remote_set_option ();
  cuda_gdb_session_create ();
  cuda_initialize_driver_api_error_report ();
  cuda_initialize_driver_internal_error_report ();
}

static void
cuda_remote_open (char *name, int from_tty)
{
  host_target_ops = &host_remote_target_ops;
  host_target_ops->to_open (name, from_tty);
  /* CUDA - remote attach only when all host threads are enumerated */
  cuda_remote_attach();
}

static void
cuda_extended_remote_open (char *name, int from_tty)
{
  host_target_ops = &host_extended_remote_target_ops;
  host_target_ops->to_open (name, from_tty);
  /* CUDA - remote attach only when all host threads are enumerated */
  cuda_remote_attach();
}

static void
cuda_remote_close (int quitting)
{
  gdb_assert (host_target_ops);

  host_target_ops->to_close (quitting);
}

static void
cuda_remote_kill (struct target_ops *ops)
{
  gdb_assert (host_target_ops);

  cuda_api_finalize ();
  cuda_cleanup ();
  cuda_gdb_session_destroy ();
  host_target_ops->to_kill (ops);
}

static void
cuda_remote_mourn_inferior (struct target_ops *ops)
{
  gdb_assert (host_target_ops);

  /* Mark breakpoints uninserted in case something tries to delete a
     breakpoint while we delete the inferior's threads (which would
     fail, since the inferior is long gone).  */
  mark_breakpoints_out ();

  if (!cuda_exception_is_valid (cuda_exception))
  {
    cuda_cleanup ();
    cuda_gdb_session_destroy ();
    host_target_ops->to_mourn_inferior (ops);
  }
}

extern int cuda_host_want_singlestep;

static void
cuda_remote_resume (struct target_ops *ops,
                    ptid_t ptid, int sstep, enum gdb_signal ts)
{
  uint32_t dev;
  cuda_coords_t c;
  bool cuda_event_found = false;
  CUDBGEvent event;
  int host_want_sstep = cuda_host_want_singlestep;

  gdb_assert (host_target_ops);
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
  if (cuda_exception_is_valid (cuda_exception) &&
      !cuda_exception_is_recoverable (cuda_exception))
    {
      target_kill ();
      cuda_trace ("cuda_resume: exception found");
      return;
    }

  if (cuda_exception_is_valid (cuda_exception) &&
      cuda_exception_is_recoverable (cuda_exception))
    {
      cuda_exception_reset (cuda_exception);
      cuda_trace ("cuda_resume: recoverable exception found\n");
    }

  cuda_notification_mark_consumed ();
  cuda_sigtrap_restore_settings ();

  if (cuda_notification_aliased_event ())
    {
      cuda_notification_reset_aliased_event ();
      cuda_api_get_next_sync_event (&event);
      cuda_event_found = event.kind != CUDBG_EVENT_INVALID;

      if (cuda_event_found)
        {
          cuda_process_events (&event, CUDA_EVENT_SYNC);
          sendAck = true;
        }
    }

  if (sendAck)
    {
      cuda_api_acknowledge_sync_events ();
      sendAck = false;
    }

  cuda_do_resume (ops, ptid, sstep, host_want_sstep, ts);

  cuda_clock_increment ();
  cuda_trace ("cuda_resume: done");
}

extern void (*gdb_old_sighand_func) (int);
extern void remote_interrupt (int);

static ptid_t
cuda_remote_wait (struct target_ops *ops,
                  ptid_t ptid, struct target_waitstatus *ws, int target_options)
{
  ptid_t r;
  uint32_t dev, dev_id;
  uint64_t grid_id;
  kernel_t kernel;
  bool cuda_event_found = false;
  CUDBGEvent event, asyncEvent;
  struct thread_info *tp;
  cuda_coords_t c;

  gdb_assert (host_target_ops);
  cuda_trace ("cuda_wait");

  if (cuda_exception_is_valid (cuda_exception))
    {
      ws->kind = TARGET_WAITKIND_SIGNALLED;
      ws->value.sig = cuda_exception_get_value (cuda_exception);
      cuda_exception_reset (cuda_exception);
      cuda_trace ("cuda_wait: exception found");
      return inferior_ptid;
    }
  else if (cuda_sstep_is_active ())
    {
      /* Cook the ptid and wait_status if single-stepping a CUDA device. */
      cuda_trace ("cuda_wait: single-stepping");
      r = cuda_sstep_ptid ();

      /* Check if C-c was sent to a remote application or if quit_flag is set.
         quit_flag is set by gdb handle_sigint() signal handler */
      if (cuda_remote_check_pending_sigint () || check_quit_flag())
        {
          clear_quit_flag();
          ws->kind = TARGET_WAITKIND_STOPPED;
          ws->value.sig = GDB_SIGNAL_INT;
          cuda_set_signo (GDB_SIGNAL_INT);
        }
      else
        {

          ws->kind = TARGET_WAITKIND_STOPPED;
          ws->value.sig = GDB_SIGNAL_TRAP;
          cuda_set_signo (GDB_SIGNAL_TRAP);

          /* If we single stepped the last warp on the device, then the
             launch has completed.  However, we do not see the event for
             kernel termination until we resume the application.  We must
             explicitly handle this here by indicating the kernel has
             terminated and switching to the remaining host thread. */

          if (cuda_sstep_kernel_has_terminated ())
            {
              /* Only destroy the kernel that has been stepped to its exit */
              dev_id  = cuda_sstep_dev_id ();
              grid_id = cuda_sstep_grid_id ();
              kernel = kernels_find_kernel_by_grid_id (dev_id, grid_id);
              kernels_terminate_kernel (kernel);

              /* Invalidate current coordinates and device state */
              cuda_coords_invalidate_current ();
              device_invalidate (dev_id);

              /* Consume any asynchronous events, if necessary.  We need to do
                 this explicitly here, since we're taking the quick path out of
                 this routine (and bypassing the normal check for API events). */
              cuda_api_get_next_async_event (&asyncEvent);
              if (asyncEvent.kind != CUDBG_EVENT_INVALID)
                cuda_process_events (&asyncEvent, CUDA_EVENT_ASYNC);

              /* Update device state/kernels */
              kernels_update_terminated ();
              cuda_update_convenience_variables ();

              switch_to_thread (r);
              tp = inferior_thread ();
              tp->control.step_range_end = 1;
              return r;
            }

        }
    }
  else {
    cuda_trace ("cuda_wait: host_wait\n");
    cuda_coords_invalidate_current ();
    r = host_target_ops->to_wait (ops, ptid, ws, target_options);
  }

  /* Immediately detect if the inferior is exiting.
     In these situations, do not investigate the device. */
  if (ws->kind == TARGET_WAITKIND_EXITED) {
    cuda_trace ("cuda_wait: target is exiting, avoiding device inspection");
    return r;
  }

  if (!cuda_initialized)
    cuda_initialize_remote_target ();

  /* Suspend all the CUDA devices. */
  cuda_trace ("cuda_wait: suspend devices");
  for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
    device_suspend (dev);

  cuda_remote_query_trace_message ();
  /* Check for ansynchronous events.  These events do not require
     acknowledgement to the debug API, and may arrive at any time
     without an explicit notification. */
  cuda_api_get_next_async_event (&asyncEvent);
  if (asyncEvent.kind != CUDBG_EVENT_INVALID)
    cuda_process_events (&asyncEvent, CUDA_EVENT_ASYNC);

  cuda_notification_analyze (r, ws, 0);

  if (cuda_notification_received ())
    {
      /* Check if there is any CUDA event to be processed */
      cuda_api_get_next_sync_event (&event);
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

  kernels_update_terminated ();

  /* Decide which thread/kernel to switch focus to. */
  if (cuda_exception_hit_p (cuda_exception))
    {
      cuda_trace ("cuda_wait: stopped because of an exception");
      c = cuda_exception_get_coords (cuda_exception);
      cuda_coords_set_current (&c);
      cuda_exception_print_message (cuda_exception);
      ws->kind = TARGET_WAITKIND_STOPPED;
      ws->value.sig = cuda_exception_get_value (cuda_exception);
      cuda_set_signo (cuda_exception_get_value (cuda_exception));
    }
  else if (cuda_sstep_is_active ())
    {
      cuda_trace ("cuda_wait: stopped because we are single-stepping");
      cuda_coords_update_current (false, false);
    }
  else if (cuda_breakpoint_hit_p (cuda_clock ()))
    {
      cuda_trace ("cuda_wait: stopped because of a breakpoint");
      cuda_set_signo (GDB_SIGNAL_TRAP);
      cuda_coords_update_current (true, false);
    }
  else if (cuda_system_is_broken (cuda_clock ()))
    {
      cuda_trace ("cuda_wait: stopped because there are broken warps (induced trap?)");
      cuda_coords_update_current (false, false);
    }
  else if (cuda_api_get_attach_state () == CUDA_ATTACH_STATE_APP_READY)
    {
      /* Finished attaching to the CUDA app.
         Preferably switch focus to a device if possible */
      cuda_trace ("cuda_wait: stopped because we attached to the CUDA app");
      cuda_api_set_attach_state (CUDA_ATTACH_STATE_COMPLETE);
      cuda_set_signo (GDB_SIGNAL_INT);
      cuda_coords_update_current (false, false);
    }
  else if (cuda_api_get_attach_state () == CUDA_ATTACH_STATE_DETACH_COMPLETE)
    {
      /* Finished detaching from the CUDA app. */
      cuda_trace ("cuda_wait: stopped because we detached from the CUDA app");
      cuda_set_signo (GDB_SIGNAL_INT);
    }
  else if (cuda_event_found)
    {
      cuda_trace ("cuda_wait: stopped because of a CUDA event");
      cuda_sigtrap_set_silent ();
      cuda_coords_update_current (false, false);
    }
  else if (ws->value.sig == GDB_SIGNAL_INT)
    {
      /* CTRL-C was hit. Preferably switch focus to a device if possible */
      cuda_trace ("cuda_wait: stopped because a SIGINT was received.");
      cuda_set_signo (GDB_SIGNAL_INT);
      cuda_coords_update_current (false, false);
    }
  else if (cuda_notification_received ())
    {
      /* No reason found when actual reason was consumed in a previous iteration (timeout,...) */
      cuda_trace ("cuda_wait: stopped for no visible CUDA reason.");
      cuda_set_signo (GDB_SIGNAL_TRAP); /* Dummy signal. We stopped after all. */
      cuda_coords_invalidate_current ();
    }
  else
    {
      cuda_trace ("cuda_wait: stopped for a non-CUDA reason.");
      cuda_set_signo (GDB_SIGNAL_TRAP);
      cuda_coords_invalidate_current ();
    }

  cuda_adjust_host_pc (r);

  /* Switch focus and update related data */
  cuda_update_convenience_variables ();
  if (cuda_focus_is_device ())
    /* Must be last, once focus and elf images have been updated */
    switch_to_cuda_thread (NULL);

  cuda_trace ("cuda_wait: done");
  return r;
}

static void
cuda_remote_fetch_registers (struct target_ops *ops,
                             struct regcache *regcache,
                             int regno)
{
  uint64_t val;
  cuda_coords_t c;
  struct gdbarch *gdbarch = get_regcache_arch (regcache);
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);
  int num_regs = gdbarch_num_regs (gdbarch);

  gdb_assert (host_target_ops);

  /* delegate to the host routines when not on the device */
  if (!cuda_focus_is_device ())
    {
      host_target_ops->to_fetch_registers (ops, regcache, regno);
      return;
    }

  cuda_coords_get_current (&c);

  /* if all the registers are wanted, then we need the host registers and the
     device PC */
  if (regno == -1)
    {
      host_target_ops->to_fetch_registers (ops, regcache, regno);
      val = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);
      regcache_raw_supply (regcache, pc_regnum, &val);
      return;
    }

  /* get the PC */
  if (regno == pc_regnum )
    {
      val = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);
      regcache_raw_supply (regcache, pc_regnum, &val);
      return;
    }

  /* raw register */
  val = lane_get_register (c.dev, c.sm, c.wp, c.ln, regno);
  regcache_raw_supply (regcache, regno, &val);
}

static void
cuda_remote_store_registers (struct target_ops *ops,
                             struct regcache *regcache,
                             int regno)
{
  uint64_t val;
  cuda_coords_t c;
  struct gdbarch *gdbarch = get_regcache_arch (regcache);
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);
  int num_regs = gdbarch_num_regs (gdbarch);

  gdb_assert (host_target_ops);
  gdb_assert (regno >= 0 && regno < num_regs);

  if (!cuda_focus_is_device ())
    {
      host_target_ops->to_store_registers (ops, regcache, regno);
      return;
    }

  if (regno == pc_regnum)
    error (_("The PC of CUDA thread is not writable"));

  cuda_coords_get_current (&c);
  regcache_raw_collect (regcache, regno, &val);
  lane_set_register (c.dev, c.sm, c.wp, c.ln, regno, val);
}

static int
cuda_remote_insert_breakpoint (struct gdbarch *gdbarch, struct bp_target_info *bp_tgt)
{
  uint32_t dev;
  bool is_cuda_addr;
  bool inserted;

  gdb_assert (host_target_ops);
  cuda_api_is_device_code_address (bp_tgt->placed_address, &is_cuda_addr);

  if (is_cuda_addr)
    {
      /* Insert the breakpoint on whatever device accepts it (valid address). */
      inserted = false;
      for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
        {
          if (!device_is_any_context_present (dev))
            continue;

          inserted |= cuda_api_set_breakpoint (dev, bp_tgt->placed_address);
        }
      return !inserted;
    }
  else
    return host_target_ops->to_insert_breakpoint (gdbarch, bp_tgt);
}

static int
cuda_remote_remove_breakpoint (struct gdbarch *gdbarch, struct bp_target_info *bp_tgt)
{
  uint32_t dev;
  CORE_ADDR cuda_addr;
  bool is_cuda_addr;
  bool removed;

  gdb_assert (host_target_ops);
  cuda_api_is_device_code_address (bp_tgt->placed_address, &is_cuda_addr);

  if (is_cuda_addr)
    {
      /* Removed the breakpoint on whatever device accepts it (valid address). */
      removed = false;
      for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
        {
          if (!device_is_any_context_present (dev))
            continue;

          /* We need to remove breakpoints even if no kernels remain on the device */
          removed |= cuda_api_unset_breakpoint (dev, bp_tgt->placed_address);
        }
      return !removed;
    }
  else
    return host_target_ops->to_remove_breakpoint (gdbarch, bp_tgt);
}

/* The whole Linux siginfo structure is presented to the user, but, internally,
   only the si_signo matters. We do not save the siginfo object. Instead we
   save only the signo. Therefore any read/write to any other field of the
   siginfo object will have no effect or will return 0. */
static LONGEST
cuda_remote_xfer_siginfo (struct target_ops *ops, enum target_object object,
                          const char *annex, gdb_byte *readbuf,
                          const gdb_byte *writebuf, ULONGEST offset, LONGEST len)
{
  /* the size of siginfo is not consistent between ptrace and other parts of
     GDB. On 32-bit Linux machines, the layout might be 64 bits. It does not
     matter for CUDA because only signo is used and the rest is set to zero. We
     just allocate 8 extra bytes and bypass the issue. On 64-bit Mac, the
     difference is 24 bytes. Therefore take the max of the 2 values. */
  gdb_byte buf[sizeof (siginfo_t) + 24];
  siginfo_t *siginfo = (siginfo_t *) buf;

  gdb_assert (host_target_ops);
  gdb_assert (object == TARGET_OBJECT_SIGNAL_INFO);
  gdb_assert (readbuf || writebuf);

  if (!cuda_focus_is_device ())
    return -1;

  if (offset >= sizeof (buf))
    return -1;

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

  return len;
}

static LONGEST
cuda_remote_xfer_partial (struct target_ops *ops,
                          enum target_object object, const char *annex,
                          gdb_byte *readbuf, const gdb_byte *writebuf,
                          ULONGEST offset, LONGEST len)
{
  LONGEST nbytes = 0;
  uint32_t dev, sm, wp, ln;

  gdb_assert (host_target_ops);
  /* If focus set on device, call the host routines directly */
  if (!cuda_focus_is_device ())
    {
      nbytes = host_target_ops->to_xfer_partial (ops, object, annex, readbuf,
                                                writebuf, offset, len);
      return nbytes;
    }

  switch (object)
  {
    /* See if this address is in pinned system memory first.  This refers to
       system memory allocations made by the inferior through the CUDA API, and
       not those made by directly using mmap(). */
    case TARGET_OBJECT_MEMORY:

      if ((readbuf  && cuda_api_read_pinned_memory  (offset, readbuf, len)) ||
          (writebuf && cuda_api_write_pinned_memory (offset, writebuf, len)))
        nbytes = len;

      break;

    /* The stack lives in local memory for ABI compilations. */
    case TARGET_OBJECT_STACK_MEMORY:

      cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);
      if (readbuf)
        {
          cuda_api_read_local_memory (dev, sm, wp, ln, offset, readbuf, len);
          nbytes = len;
        }
      else if (writebuf)
        {
          cuda_api_write_local_memory (dev, sm, wp, ln, offset, writebuf, len);
          nbytes = len;
        }
      break;

    /* When stopping on the device, build a simple siginfo object */
    case TARGET_OBJECT_SIGNAL_INFO:

      nbytes = cuda_remote_xfer_siginfo (ops, object, annex, readbuf, writebuf,
                                      offset, len);
      break;
  }

  if (nbytes < len)
    nbytes = host_target_ops->to_xfer_partial (ops, object, annex, readbuf,
                                              writebuf, offset, len);

  return nbytes;
}

static struct gdbarch *
cuda_remote_thread_architecture (struct target_ops *ops, ptid_t ptid)
{
  if (cuda_focus_is_device ())
    return cuda_get_gdbarch ();
  else
    return target_gdbarch();
}

static void
cuda_remote_prepare_to_store (struct regcache *regcache)
{
  gdb_assert (host_target_ops);
  if (get_regcache_arch (regcache) == cuda_get_gdbarch ())
    return;
  host_target_ops->to_prepare_to_store (regcache);
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
  char *cudbgApiInitForAttach = "cudbgApiInit(1)";
  CORE_ADDR debugFlagAddr;
  CORE_ADDR sessionIdAddr;
  CORE_ADDR attachDataAvailableFlagAddr;
  const unsigned char one = 1;
  unsigned char attachDataAvailable;
  uint32_t sessionId = 0;
  unsigned int timeOut = 5000000; /* 5 seconds */
  unsigned int timeElapsed = 0;
  const unsigned int sleepTime = 1000;

  if (cuda_api_get_attach_state() != CUDA_ATTACH_STATE_NOT_STARTED && 
    cuda_api_get_attach_state() != CUDA_ATTACH_STATE_DETACH_COMPLETE)
    return;
  if (!remote_query_attached (inferior_ptid.pid))
    return;

  cuda_initialize_remote_target();

  debugFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_IPC_FLAG_NAME));
  sessionIdAddr = cuda_get_symbol_address (_STRING_(CUDBG_SESSION_ID));

  /* Return early if CUDA driver isn't available. Attaching to the host
     process has already been completed at this point. */
  if (!debugFlagAddr || !sessionIdAddr)
    return;
  
  target_read_memory (sessionIdAddr, (char*)&sessionId, sizeof(sessionId));
  if (!sessionId)
    return;

  /* If the CUDA driver has been loaded but software preemption has been turned
     on, stop the attach process. */
  if (cuda_options_software_preemption ())
    error (_("Attaching to a running CUDA process with software preemption "
             "enabled in the debugger is not supported."));

  cuda_api_set_attach_state (CUDA_ATTACH_STATE_IN_PROGRESS);

  if (!lookup_cmd_composition ("call", &alias, &prefix_cmd, &cmd))
    error (_("Failed to initiate attach."));

  /* Fork off the CUDA debugger process from the inferior */
  cmd_func (cmd, cudbgApiInitForAttach, 0);

  attachDataAvailableFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_ATTACH_HANDLER_AVAILABLE));

  /* If this is not available, the CUDA driver doesn't support attaching.  */
  if (!attachDataAvailableFlagAddr)
    error (_("This CUDA driver does not support attaching to a running CUDA process."));

  /* Wait till the backend has started up and is ready to service API calls */
  while (cuda_api_initialize () != CUDBG_SUCCESS)
    {
      if (timeElapsed < timeOut)
        usleep(sleepTime);
      else
        error (_("Timed out waiting for the CUDA API to initialize."));

      timeElapsed += sleepTime;
    }


  /* Check if more data is available from the inferior */
  target_read_memory (attachDataAvailableFlagAddr, &attachDataAvailable, 1);

  if (attachDataAvailable)
    /* Resume the inferior to collect more data. CUDA_ATTACH_STATE_COMPLETE and
       CUDBG_IPC_FLAG_NAME will be set once this completes. */
    /*Was: continue_command (0, 0);*/
    continue_1 (false);
  else
    {
      /* Enable debugger callbacks from the CUDA driver */
      target_write_memory (debugFlagAddr, &one, 1);

      /* No data to collect, attach complete. */
      cuda_api_set_attach_state (CUDA_ATTACH_STATE_COMPLETE);
    }
}

static void
cuda_remote_detach(struct target_ops *ops, char *args, int from_tty)
{
  gdb_assert (host_target_ops);

  /* If attach wasn't completed,
     treat the inferior as a host-only process */
  if (cuda_api_get_attach_state() == CUDA_ATTACH_STATE_COMPLETE &&
     remote_query_attached (inferior_ptid.pid))
    cuda_do_detach (true);
  host_target_ops->to_detach (host_target_ops, args, from_tty);

}

void
cuda_remote_version_handshake (const struct protocol_feature *feature,
                               enum packet_support support,
                               const char *version_string)
{
  uint32_t server_major, server_minor, server_rev;

  gdb_assert (strcmp (feature->name, "CUDAVersion") == 0);
  if (support != PACKET_ENABLE)
    error (_("Server doesn't support CUDA.\n"));

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

void
cuda_remote_add_target (struct target_ops *t)
{
  /* Save the original set of target operations */
  if (strcmp (t->to_shortname, "remote") == 0)
    {
      host_remote_target_ops = *t;
      t->to_open = cuda_remote_open;
    }
  else if (strcmp (t->to_shortname, "extended-remote") == 0)
    {
      host_extended_remote_target_ops = *t;
      t->to_open = cuda_extended_remote_open;
    }
  else
    gdb_assert (0);

  alloc_cuda_packet_buffer ();
  make_final_cleanup (free_cuda_packet_buffer, NULL);

  /* Override what we need to */
  t->to_close			= cuda_remote_close;
  t->to_kill			= cuda_remote_kill;
  t->to_mourn_inferior		= cuda_remote_mourn_inferior;
  t->to_detach			= cuda_remote_detach;
  t->to_resume			= cuda_remote_resume;
  t->to_wait			= cuda_remote_wait;
  t->to_fetch_registers		= cuda_remote_fetch_registers;
  t->to_store_registers         = cuda_remote_store_registers;
  t->to_insert_breakpoint       = cuda_remote_insert_breakpoint;
  t->to_remove_breakpoint       = cuda_remote_remove_breakpoint;
  t->to_xfer_partial            = cuda_remote_xfer_partial;
  t->to_thread_architecture	= cuda_remote_thread_architecture;
  t->to_prepare_to_store        = cuda_remote_prepare_to_store;
}

