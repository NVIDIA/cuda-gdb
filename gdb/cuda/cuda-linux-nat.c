/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2024 NVIDIA Corporation
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

/*Warning: this isn't intended as a standalone compile module! */

#include "defs.h"

#include <objfiles.h>
#include <sys/ptrace.h>
#include <sys/signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>

#include "arch-utils.h"
#include "block.h"
#include "buildsym-legacy.h"
#include "command.h"
#include "cuda-commands.h"
#include "cuda-convvars.h"
#include "cuda-events.h"
#include "cuda-exceptions.h"
#include "cuda-notifications.h"
#include "cuda-options.h"
#include "cuda-packet-manager.h"
#include "cuda-parser.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"
#include "demangle.h"
#include "dictionary.h"
#include "gdbcmd.h"
#include "gdbthread.h"
#include "inferior.h"
#include "language.h"
#include "observable.h"
#include "regcache.h"
#include "valprint.h"
#if defined(__linux__) && defined(GDB_NM_FILE)
#include "linux-nat.h"
#endif
#include "cuda-linux-nat.h"
#include "event-top.h"
#include "inf-child.h"
#include "inf-loop.h"
#include "top.h"
#include "remote.h"
#include "remote-cuda.h"

bool cuda_debugging_enabled = false;

struct cuda_cudart_symbols_st cuda_cudart_symbols;
static struct cuda_signal_info_st cuda_sigtrap_info;

#if defined(__linux__) && defined(GDB_NM_FILE)
static void
cuda_clear_pending_sigint (pid_t pid)
{
  int status = 0, options = 0;
  ptrace (PTRACE_CONT, pid, 0,
	  0); /* Resume the host to consume the pending SIGINT */
  waitpid (pid, &status, options); /* Ensure we return for the right reason */
  gdb_assert (WIFSTOPPED (status) && WSTOPSIG (status) == SIGINT);
}
#endif

bool
cuda_check_pending_sigint (pid_t pid)
{
#if defined(__linux__) && defined(GDB_NM_FILE)
  sigset_t pending, blocked, ignored;

  linux_proc_pending_signals (pid, &pending, &blocked, &ignored);
  if (sigismember (&pending, SIGINT))
    {
      cuda_clear_pending_sigint (pid);
      return true;
    }
#endif

  /* No pending SIGINT */
  return false;
}

void
cuda_signal_set_silent (int sig, struct cuda_signal_info_st *save)
{
  enum gdb_signal gdb_sig = gdb_signal_from_host (sig);

  gdb_assert (save);
  gdb_assert (gdb_sig != GDB_SIGNAL_UNKNOWN);
  gdb_assert (cuda_options_stop_signal () != gdb_sig);

  save->stop = signal_stop_state (gdb_sig);
  save->print = signal_print_state (gdb_sig);
  save->saved = true;

  signal_stop_update (gdb_sig, 0);
  signal_print_update (gdb_sig, 0);
}

void
cuda_signal_restore_settings (int sig, struct cuda_signal_info_st *save)
{
  enum gdb_signal gdb_sig = gdb_signal_from_host (sig);

  gdb_assert (save);
  gdb_assert (gdb_sig != GDB_SIGNAL_UNKNOWN);
  gdb_assert (cuda_options_stop_signal () != gdb_sig);

  if (save->saved)
    {
      signal_stop_update (gdb_sig, save->stop);
      signal_print_update (gdb_sig, save->print);
      save->saved = false;
    }
}

void
cuda_sigtrap_set_silent (void)
{
  cuda_signal_set_silent (SIGTRAP, &cuda_sigtrap_info);
}

void
cuda_sigtrap_restore_settings (void)
{
  cuda_signal_restore_settings (SIGTRAP, &cuda_sigtrap_info);
}

/* If a host event is hit while there are valid threads
   on the GPU, the focus ends up being switched to the
   GPU, leaving the host PC not rewound.

   This function determines if the host is at a breakpoint,
   and if so it manually rewinds the host PC so that the
   breakpoint can be hit again after a resume.
   r here is the return value of host_wait().
*/
void
cuda_adjust_host_pc (ptid_t r)
{
  bool pc_rewound = false;
  struct regcache *regcache;
  CORE_ADDR pc;

  if (!cuda_current_focus::isDevice ())
    return;

  /* Rewind host PC and consume pending SIGTRAP
     Sometimes, one thread can hit both a host and a device
     breakpoint at the same time, in which case host SIGTRAP
     is triggered while SIGTRAP from back end is blocked (pending).
     When resuming, host PC is not rewound because focus is on the
     device.

     Before switching to CUDA thread, we check if that's the case.
     If so, manually rewind the host PC and consume the pending SIGTRAP.
     This allows the host breakpoint to be hit again after resuming. */

  /* Temporarily invalidate the current coords so that the focus
     is set on the host. */
  cuda_current_focus::invalidate ();

  regcache = get_thread_arch_regcache (current_inferior ()->process_target (),
				       r, target_gdbarch ());
  pc = regcache_read_pc (regcache)
       - gdbarch_decr_pc_after_break (target_gdbarch ());
  if (breakpoint_inserted_here_p (regcache->aspace (), pc))
    {
      /* Rewind the PC */
      regcache_write_pc (regcache, pc);
      pc_rewound = true;
    }

  /* Restore coords */
  cuda_current_focus::forceValid ();

  /* Remove the pending notification if we rewound the pc */
  if (pc_rewound)
    cuda_notification_consume_pending ();
}

static void
cuda_nat_attach (inferior *inf)
{
  struct cmd_list_element *alias = NULL;
  struct cmd_list_element *prefix_cmd = NULL;
  struct cmd_list_element *cmd = NULL;
  const char *cudbgApiAttach = "(void) cudbgApiAttach()";
  CORE_ADDR debugFlagAddr;
  CORE_ADDR resumeAppOnAttachFlagAddr;
  CORE_ADDR attachDataAvailableFlagAddr;
  unsigned char resumeAppOnAttach;
  unsigned int timeOut = 5000; // ms
  unsigned int timeElapsed = 0;
  unsigned dev = 0;
  bool need_retry = 0;
  unsigned retry_count = 0;
  unsigned retry_delay = 100;	    // ms
  unsigned app_init_timeout = 5000; // ms
  const unsigned int sleepTime = 1; // ms
  uint64_t internal_error_code;
  unsigned char *sigs = NULL;

  if (is_remote_target (inf->process_target ()))
    {
      /* Make sure the debug API is in an attachable state for remote */
      if (cuda_debugapi::get_attach_state () != CUDA_ATTACH_STATE_NOT_STARTED
	  && cuda_debugapi::get_attach_state ()
		 != CUDA_ATTACH_STATE_DETACH_COMPLETE)
	return;
      /* Try to init remote target */
      cuda_remote_initialize_target ();

      CORE_ADDR sessionIdAddr
	  = cuda_get_symbol_address (_STRING_ (CUDBG_SESSION_ID));

      /* Return early if CUDA driver isn't available. Attaching to the host
	 process has already been completed at this point. */
      if (!sessionIdAddr)
	return;

      /* TODO: This isn't actually used. Do we need to continue reading this value? */
      uint32_t sessionId = 0;
      target_read_memory (sessionIdAddr, (gdb_byte *)&sessionId,
			  sizeof (sessionId));
      if (!sessionId)
	return;

      attachDataAvailableFlagAddr = cuda_get_symbol_address (
	  _STRING_ (CUDBG_ATTACH_HANDLER_AVAILABLE));

      /* If this is not available, the CUDA driver doesn't support attaching.
       */
      if (!attachDataAvailableFlagAddr)
	error (
	    _ ("This CUDA driver does not support attaching to a running CUDA "
	       "process."));

      cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_IN_PROGRESS);
    }
  else
    {
      /* Return early if CUDA driver isn't available. Attaching to the host
	 process has already been completed at this point. */
      cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_IN_PROGRESS);
      if (!cuda_initialize_target ())
	{
	  cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_NOT_STARTED);
	  return;
	}
    }

  /* If the CUDA driver has been loaded but software preemption has been turned
     on, stop the attach process. */
  if (cuda_options_software_preemption ())
    {
      cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_NOT_STARTED);
      error (_ ("Attaching to a running CUDA process with software preemption "
		"enabled in the debugger is not supported."));
    }

  if (!lookup_cmd_composition ("call", &alias, &prefix_cmd, &cmd))
    error (_ ("Failed to initiate attach."));

  do
    {
      /* Try to init debugger's backend */
      sigs = cuda_gdb_bypass_signals ();
      cuda_gdb_bypass_signals_cleanup cleanup (sigs);
      cmd_func (cmd, cudbgApiAttach, 0);
      /* Manually cleanup */
      cleanup.release ();
      cuda_nat_bypass_signals_cleanup (sigs);

      internal_error_code = cuda_get_last_driver_internal_error_code ();

      /* CUDBG_ERROR_ATTACH_NOT_POSSIBLE can be returned in two scenarios:
       * 1. Attach is really not possible
       * 2. Critical section's mutex is taken, attaching would cause a deadlock
       */
      need_retry = (unsigned int)internal_error_code
		   == CUDBG_ERROR_ATTACH_NOT_POSSIBLE;

      if (need_retry)
	{
	  /* Resume the target */
	  prepare_execution_command (inf->top_target (), true);
	  continue_1 (true);

	  usleep (retry_delay * 1000);

	  /* Trigger the future wait() */
	  interrupt_target_1 (true);

	  /* Get control back */
	  cuda_wait_for_inferior ();
	  set_running (inf->process_target (), minus_one_ptid,
		       0);

	  retry_count++;
	}
    }
  while (need_retry && (retry_count * retry_delay < app_init_timeout));

  /* We are re-using the ATTACH_NOT_POSSIBLE error code for delayed attach,
   * therefore a timeout will allow us to determine if the error code is
   * genuinely not possible. */
  if (need_retry)
    error (_ ("Attaching not possible. "
	      "Please verify that software preemption is disabled "
	      "and that nvidia-cuda-mps-server is not running."));

  if ((unsigned int)internal_error_code
      == CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED)
    error (
	_ ("Attaching to process running on watchdogged GPU is not possible.\n"
	   "Please repeat the attempt in console mode or "
	   "restart the process with CUDA_VISIBLE_DEVICES environment "
	   "variable set."));
  if (internal_error_code)
    error (_ ("Attach failed due to the internal driver error 0x%llx\n"),
	   (unsigned long long)internal_error_code);

  debugFlagAddr = cuda_get_symbol_address (_STRING_ (CUDBG_IPC_FLAG_NAME));
  resumeAppOnAttachFlagAddr
      = cuda_get_symbol_address (_STRING_ (CUDBG_RESUME_FOR_ATTACH_DETACH));

  /* If this is not available, the CUDA driver doesn't support attaching.  */
  if (resumeAppOnAttachFlagAddr == 0 || debugFlagAddr == 0)
    error (_ ("This CUDA driver does not support attaching to a running CUDA "
	      "process."));

  /* Setup our desired capabilities for the debugger backend. It is alright
   * if the older driver doesn't understand some of these flags. We will deal
   * with those situations after initialization. */
  CORE_ADDR capability_addr
      = cuda_get_symbol_address (_STRING_ (CUDBG_DEBUGGER_CAPABILITIES));
  if (capability_addr)
    {
      uint32_t capabilities = CUDBG_DEBUGGER_CAPABILITY_NONE;

      cuda_trace_domain (CUDA_TRACE_GENERAL,
			 "requesting CUDA lazy function loading support\n");
      capabilities |= CUDBG_DEBUGGER_CAPABILITY_LAZY_FUNCTION_LOADING;

      cuda_trace_domain (
	  CUDA_TRACE_GENERAL,
	  "requesting tracking of exceptions in exited warps\n");
      capabilities
	  |= CUDBG_DEBUGGER_CAPABILITY_REPORT_EXCEPTIONS_IN_EXITED_WARPS;

      cuda_trace_domain (
	  CUDA_TRACE_GENERAL,
	  "requesting no context push / pop events be delivered\n");
      capabilities
	  |= CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS;

      target_write_memory (capability_addr, (const gdb_byte *)&capabilities,
			   sizeof (capabilities));
    }

  /* Ensure the remote target has been initialized at this point */
  if (is_remote_target (inf->process_target ()))
    {
      while (!cuda_remote_initialize_target ())
	{
	  if (timeElapsed < timeOut)
	    usleep (sleepTime * 1000);
	  else
	    error (_ ("Timed out waiting for the CUDA remote target to initialize."));

	  timeElapsed += sleepTime;
	}
      timeElapsed = 0;
    }

  /* Wait till the backend has started up and is ready to service API calls */
  while (cuda_debugapi::initialize () != CUDBG_SUCCESS)
    {
      internal_error_code = cuda_get_last_driver_internal_error_code ();

      if ((unsigned int)internal_error_code == CUDBG_ERROR_ATTACH_NOT_POSSIBLE)
	error (_ ("Attaching not possible. "
		  "Please verify that software preemption is disabled "
		  "and that nvidia-cuda-mps-server is not running."));
      if (internal_error_code)
	error (_ ("Attach failed due to the internal driver error 0x%llx\n"),
	       (unsigned long long)internal_error_code);

      if (timeElapsed < timeOut)
	usleep (sleepTime * 1000);
      else
	error (_ ("Timed out waiting for the CUDA API to initialize."));

      timeElapsed += sleepTime;
    }

  /* Check if the inferior needs to be resumed */
  if (is_remote_target (inf->process_target ()))
    target_read_memory (attachDataAvailableFlagAddr, &resumeAppOnAttach, 1);
  else
    target_read_memory (resumeAppOnAttachFlagAddr, &resumeAppOnAttach, 1);

  if (resumeAppOnAttach)
    {
      int cnt;
      sigs = cuda_gdb_bypass_signals ();
      cuda_gdb_bypass_signals_cleanup cleanup (sigs);
      /* Resume the inferior to collect more data. CUDA_ATTACH_STATE_COMPLETE
	 and CUDBG_IPC_FLAG_NAME will be set once this completes. */
      for (cnt = 0; cnt < 1000
		    && cuda_debugapi::get_attach_state ()
			   == CUDA_ATTACH_STATE_IN_PROGRESS;
	   cnt++)
	{
	  prepare_execution_command (inf->top_target (), true);
	  continue_1 (false);
	  /* force resumed state to false */
	  bool resumed_state;
	  if (is_remote_target (inf->process_target ()))
	    {
	      resumed_state = current_inferior ()->process_target ()->commit_resumed_state;
	      current_inferior ()->process_target ()->commit_resumed_state = false;
	    }
	  cuda_wait_for_inferior ();
	  if (is_remote_target (inf->process_target ()))
	    {
	      current_inferior ()->process_target ()->commit_resumed_state = resumed_state;
	    }
	  /* infrun's async_event_handler is in the "ready" state after running
	     `continue_1` above. Since we've waited for inferior above, we now
	     run the completions and reset the "ready" state by calling the
	     below function. Doing this will lead to CUDA's wait function not
	     being called after `cuda_nat_attach` completes. */
	  inferior_event_handler (INF_EXEC_COMPLETE);
	}

      /* No threads are running at this point.  */
      set_running (inf->process_target (), minus_one_ptid, 0);

      /* Manually cleanup */
      cleanup.release ();
      cuda_nat_bypass_signals_cleanup (sigs);
      if (cuda_debugapi::get_attach_state () != CUDA_ATTACH_STATE_APP_READY
	  && cuda_debugapi::get_attach_state () != CUDA_ATTACH_STATE_COMPLETE)
	error ("Unexpected CUDA attach state %d, further debugging session "
	       "might be unreliable",
	       cuda_debugapi::get_attach_state ());
    }
  else
    {
      cuda_force_stop_print_frame ();

      /* Enable debugger callbacks from the CUDA driver */
      cuda_write_bool (debugFlagAddr, true);

      /* No data to collect, attach complete. */
      cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_COMPLETE);

      /* Initialize CUDA and suspend the devices */
      cuda_initialize ();
      for (dev = 0; dev < cuda_state::get_num_devices (); ++dev)
	cuda_state::device_suspend (dev);
    }

  /* The inferior just got signaled, we're not expecting any other stop */
  inf->control.stop_soon = NO_STOP_QUIETLY;
}

static void
cuda_inferior_created (inferior *inf)
{
#ifndef __QNXTARGET__
  // If attaching we need to initialize the debug API manually
  // QNX: QNX will always set the attach_flag to true, so this mechanism
  // won't work there. We don't support attaching to a running process on
  // QNX today.
  if (inf->attach_flag)
    cuda_nat_attach (inf);
#endif
}

void
cuda_do_detach (inferior *inf)
{
  struct cmd_list_element *alias = NULL;
  struct cmd_list_element *prefix_cmd = NULL;
  struct cmd_list_element *cmd = NULL;
  const char *cudbgApiDetach = "(void) cudbgApiDetach()";
  CORE_ADDR debugFlagAddr;
  CORE_ADDR rpcFlagAddr;
  CORE_ADDR resumeAppOnDetachFlagAddr;
  unsigned char resumeAppOnDetach;
  unsigned char *sigs = NULL;

  debugFlagAddr = cuda_get_symbol_address (_STRING_ (CUDBG_IPC_FLAG_NAME));

  /* Bail out if the CUDA driver isn't available or the host process doesn't
   * have execution. */
  if (!debugFlagAddr)
    return;

  /* If the host process doesn't have execution, we cannot ask the host thread
   * to detach. Cleanup and return.
   */
  if (!inf->has_execution ())
    {
      cuda_cleanup ();
      return;
    }

  cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_DETACHING);

  /* Make sure the focus is set on the host */
  switch_to_thread (inf->process_target (), inferior_ptid);

  if (!lookup_cmd_composition ("call", &alias, &prefix_cmd, &cmd))
    error (_ ("Failed to initiate detach."));

  /* Figure out if we need to clean up driver state before detaching */
  resumeAppOnDetachFlagAddr
      = cuda_get_symbol_address (_STRING_ (CUDBG_RESUME_FOR_ATTACH_DETACH));

  if (!resumeAppOnDetachFlagAddr)
    error (_ ("Failed to detach cleanly from the inferior."));

  /* Make dynamic call for cleanup. */
  sigs = cuda_gdb_bypass_signals ();
  cuda_gdb_bypass_signals_cleanup cleanup (sigs);
  cmd_func (cmd, cudbgApiDetach, 0);
  /* Manually cleanup */
  cleanup.release ();
  cuda_nat_bypass_signals_cleanup (sigs);

  /* Read the updated value of the flag */
  target_read_memory (resumeAppOnDetachFlagAddr, &resumeAppOnDetach, 1);

  /* If this flag is set, the debugger backend needs to be notified to cleanup
   * on detach */
  if (resumeAppOnDetach)
    cuda_debugapi::request_cleanup_on_detach (resumeAppOnDetach);

  /* Clear requested capabilities for the next debugger attach which
     may not support all of the ones requested by this instance. */
  CORE_ADDR capability_addr
      = cuda_get_symbol_address (_STRING_ (CUDBG_DEBUGGER_CAPABILITIES));
  if (capability_addr)
    {
      uint32_t capabilities = CUDBG_DEBUGGER_CAPABILITY_NONE;
      target_write_memory (capability_addr, (const gdb_byte *)&capabilities,
			   sizeof (capabilities));
    }

  /* Make sure the debugger is reinitialized from scratch on reattaching
     to the inferior */
  rpcFlagAddr
      = cuda_get_symbol_address (_STRING_ (CUDBG_DEBUGGER_INITIALIZED));

  if (!rpcFlagAddr)
    error (_ ("Failed to detach cleanly from the inferior."));

  cuda_write_bool (rpcFlagAddr, false);

  /* If a cleanup is needed, resume the app to allow the cleanup to complete.
     The debugger backend will send a cleanup event to stop the app when the
     cleanup finishes. */
  if (resumeAppOnDetach)
    {
      int cnt;

      /* Clear all breakpoints */
      cuda_delete_command (NULL, 0);
      cuda_state::cleanup_breakpoints ();
      cuda_options_disable_break_on_launch ();

      /* Now resume the app and wait for CUDA_ATTACH_STATE_DETACH_COMPLETE
       * event. */
      for (cnt = 0; cnt < 100
		    && cuda_debugapi::get_attach_state ()
			   != CUDA_ATTACH_STATE_DETACH_COMPLETE;
	   cnt++)
	{
	  prepare_execution_command (inf->top_target (), true);
	  continue_1 (false);
	  /* force resumed state to false */
	  auto resumed_state = inf->process_target ()->commit_resumed_state;
	  inf->process_target ()->commit_resumed_state = false;
	  cuda_wait_for_inferior ();
	  /* Process may have exited at this point. */
	  if (!inf->process_target ())
	    break;
	  inf->process_target ()->commit_resumed_state = resumed_state;
	}

      /* No threads are running at this point.  */
      if (inf->process_target ())
	set_running (inf->process_target (), minus_one_ptid, 0);
    }
  else
    cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_DETACH_COMPLETE);

  if (inf->process_target ())
    {
      if (cuda_debugapi::get_attach_state ()
	  != CUDA_ATTACH_STATE_DETACH_COMPLETE)
	warning (_ ("Unexpected CUDA API attach state %d."),
		 cuda_debugapi::get_attach_state ());

      cuda_write_bool (debugFlagAddr, false);
    }

  cuda_cleanup ();
}

void
switch_to_cuda_thread (const cuda_coords &coords)
{
  uint64_t pc;

  cuda_current_focus::set (const_cast<cuda_coords &> (coords));

  thread_info *thr = find_thread_ptid (current_inferior ()->process_target (),
				       inferior_ptid);
  /* Only update if a host thread still exists. */
  if (thr)
    {
      switch_to_thread_keep_cuda_focus (thr);

      cuda_update_convenience_variables ();
      cuda_update_cudart_symbols ();

      if (const_cast<cuda_coords &> (coords).isValidOnDevice ())
	pc = cuda_state::lane_get_pc (
	    coords.physical ().dev (), coords.physical ().sm (),
	    coords.physical ().wp (), coords.physical ().ln ());
      else
	pc = (CORE_ADDR)~0;

      thr->set_stop_pc (pc);
    }
}

struct objfile *cuda_create_builtins_objfile (void);

void
cuda_update_cudart_symbols (void)
{
  /* If not done yet, create a CUDA runtime symbols file */
  if (!cuda_cudart_symbols.objfile)
    {
      cuda_cudart_symbols.objfile = cuda_create_builtins_objfile ();
    }
}

void
cuda_cleanup_cudart_symbols (void)
{
  /* Free the objfile if allocated */
  if (cuda_cudart_symbols.objfile)
    {
      cuda_cudart_symbols.objfile->unlink ();
      cuda_cudart_symbols.objfile = NULL;
    }
}

/*
 * CUDA builtins construction routines
 */

/* cuda_alloc_dim3_type helper routine: initializes one of the structure fields
 * with a given name, offset and type */
static void
cuda_init_field (struct field &fp, const char *name, const int offs,
		 struct type *type)
{
  fp.set_name (xstrdup (name));
  fp.set_type (type);
  fp.set_loc_bitpos (offs * 8);
  FIELD_BITSIZE (fp) = type->length () * 8;
}

/* Allocates dim3 type as structure of 3 packed unsigned int: x, y and z */
static struct type *
cuda_alloc_dim3_type (struct objfile *objfile)
{
  struct gdbarch *gdbarch = objfile->arch ();
  struct type *uint32_type = builtin_type (gdbarch)->builtin_unsigned_int;
  struct type *dim3 = NULL;

  dim3 = alloc_type (objfile);

  dim3->set_name ("dim3");
  dim3->set_length (12);
  dim3->set_code (TYPE_CODE_STRUCT);

  dim3->set_num_fields (3);
  dim3->set_fields (
      (struct field *)TYPE_ZALLOC (dim3, 3 * sizeof (struct field)));

  cuda_init_field (dim3->field (0), "x", 0, uint32_type);
  cuda_init_field (dim3->field (1), "y", 4, uint32_type);
  cuda_init_field (dim3->field (2), "z", 8, uint32_type);

  return dim3;
}

/* Create symbol of a given type inside the objfile */
static struct symbol *
cuda_create_symbol (struct objfile *objfile, struct blockvector *bv,
		    const char *name, CORE_ADDR addr, struct type *type)
{
  /* Allocate a new symbol in OBJFILE's obstack.  */
  struct symbol *sym = new (&objfile->objfile_obstack) symbol;

  sym->set_language (language_c, &objfile->per_bfd->storage_obstack);
  sym->compute_and_set_names (name, true, objfile->per_bfd);
  sym->set_type (type);
  sym->set_domain (VAR_DOMAIN);
  sym->set_aclass_index (LOC_STATIC);
  sym->set_value_address (addr);

  /* Register symbol as global symbol with symtab */
  sym->set_symtab (objfile->compunit_symtabs->primary_filetab ());
  add_symbol_to_list (sym, get_global_symbols ());
  mdict_add_symbol (bv->global_block ()->multidict (), sym);

  return sym;
}

/* Symtab initialization helper routine:
 * Allocates blockvector as well as global and static blocks inside symtab
 */
static struct blockvector *
cuda_alloc_blockvector (struct symtab *symtab, int nblocks)
{
  struct obstack *obstack
      = &(symtab->compunit ()->objfile ()->objfile_obstack);

  /* At least enough room for the global and static blocks */
  gdb_assert (nblocks >= 2);

  /* allocate and zero the blockvector */
  uint32_t len
      = sizeof (struct blockvector) + (nblocks - 1) * sizeof (struct block *);
  struct blockvector *bv = (struct blockvector *)obstack_alloc (obstack, len);
  memset ((void *)bv, 0, len);

  bv->set_num_blocks (nblocks);

  /* Allocate the GLOBAL block */
  struct block *global_block = allocate_global_block (obstack);
  global_block->set_multidict (
      mdict_create_hashed_expandable (symtab->language ()));
  bv->set_block (GLOBAL_BLOCK, global_block);

  /* Only allowed for the GLOBAL block */
  set_block_compunit_symtab (global_block, symtab->compunit ());

  /* Allocate the STATIC block*/
  struct block *static_block = allocate_block (obstack);
  static_block->set_multidict (
      mdict_create_hashed_expandable (symtab->language ()));
  bv->set_block (STATIC_BLOCK, static_block);

  /* superblock of the static block is the global block - see block.h */
  static_block->set_superblock (global_block);

  return bv;
}

/* Allocate virtual objfile and construct the following symbols inside it:
 * threadIdx of type dim3 located at CUDBG_THREADIDX_OFFSET
 * blockIdx of type dim3 located at CUDBG_BLOCKIDX_OFFSET
 * clusterIdx of type dim3 located at CUDBG_CLUSTERIDX_OFFSET
 * gridDim of type dim3 located at CUDBG_GRIDDIM_OFFSET
 * blockDim of type dim3 located at CUDBG_BLOCKDIM_OFFSET
 * clusterDim of type dim3 located at CUDBG_CLUSTERDIM_OFFSET
 * warpSize of type int located at CUDBG_WARPSIZE_OFFSET
 */
struct objfile *
cuda_create_builtins_objfile (void)
{
  struct objfile *objfile = NULL;
  struct type *int32_type = NULL;
  struct type *dim3_type = NULL;
  struct symtab *symtab = NULL;

  /* Set the cleanup chain so we get things properly set after we're done
     assembling the symbol table.  */
  scoped_free_pendings free_pending;

  /* This is not a real objfile.  Mark it as so by passing
     OBJF_NOT_FILENAME.  */
  objfile = objfile::make (NULL, NULL, OBJF_SHARED | OBJF_NOT_FILENAME);
  objfile->per_bfd->gdbarch = cuda_get_gdbarch ();

  /* Get/allocate types */
  int32_type = builtin_type ((objfile->arch ()))->builtin_int32;
  dim3_type = cuda_alloc_dim3_type (objfile);

  /* Now that the objfile structure has been allocated, we need to allocate all
     the required data structures for symbols.  */
  objfile->compunit_symtabs = start_compunit_symtab (
      objfile, "<cuda-builtins>", NULL, 0, language_c);

  symtab = allocate_symtab (objfile->compunit_symtabs, "<cuda-builtins>");
  symtab->set_language (language_c);
  objfile->compunit_symtabs->set_primary_filetab (symtab);

  struct blockvector *bv = cuda_alloc_blockvector (symtab, 2);
  objfile->compunit_symtabs->set_blockvector (bv);

  cuda_create_symbol (objfile, bv, "threadIdx", CUDBG_THREADIDX_OFFSET,
		      dim3_type);
  cuda_create_symbol (objfile, bv, "blockIdx", CUDBG_BLOCKIDX_OFFSET,
		      dim3_type);
  cuda_create_symbol (objfile, bv, "clusterIdx", CUDBG_CLUSTERIDX_OFFSET,
		      dim3_type);
  cuda_create_symbol (objfile, bv, "gridDim", CUDBG_GRIDDIM_OFFSET, dim3_type);
  cuda_create_symbol (objfile, bv, "blockDim", CUDBG_BLOCKDIM_OFFSET,
		      dim3_type);
  cuda_create_symbol (objfile, bv, "clusterDim", CUDBG_CLUSTERDIM_OFFSET,
		      dim3_type);
  cuda_create_symbol (objfile, bv, "warpSize", CUDBG_WARPSIZE_OFFSET,
		      int32_type);

  return objfile;
}

void _initialize_cuda_nat ();
void
_initialize_cuda_nat ()
{
  /* Initialize the cleanup routines */
  make_final_cleanup (cuda_final_cleanup, NULL);

  gdb::observers::inferior_created.attach (cuda_inferior_created, "CUDA");

  cuda_debugging_enabled = true;
}