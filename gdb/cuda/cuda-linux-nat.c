/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2026 NVIDIA Corporation
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
#include <string.h>
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
#include "command.h"
#include "gdbsupport/scope-exit.h"
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
#include "extension.h"
#include "inf-child.h"
#include "infrun.h"
#include "inf-loop.h"
#include "main.h"
#include "remote-cuda.h"
#include "remote.h"
#include "top.h"
#include "interps.h"
#include "target/target.h"
#include "ui.h"

bool cuda_debugging_enabled = false;

static struct objfile *cuda_cudart_symbols;
static struct cuda_signal_info_st cuda_sigtrap_info;

static struct objfile *cuda_create_builtins_objfile (void);

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
  gdb_assert (GDB_SIGNAL_URG != gdb_sig);

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
  gdb_assert (GDB_SIGNAL_URG != gdb_sig);

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

  regcache = get_thread_arch_regcache (current_inferior (),
				       r, current_inferior ()->arch ());
  pc = regcache_read_pc (regcache)
       - gdbarch_decr_pc_after_break (current_inferior ()->arch ());
  if (breakpoint_inserted_here_p (current_inferior ()->aspace.get (), pc))
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

#ifndef __QNXTARGET__
/* Attach isn't yet supported on QNX */

enum cuda_attach_protocol_support
{
  /* The new protocol is not supported at all */
  v0_only,
  /* The new protocol is supported and we can immediately proceed with it */
  v1_supported,
  /* The new protocol will be supported after we let the driver initialize
     and set the FD. */
  v1_supported_later,
};

/* Check if we have a way to trigger the FD */
static enum cuda_attach_protocol_support
cuda_get_attach_protocol_support (inferior *inf)
{
  CORE_ADDR symbol_address = 0;
  int32_t fd = 0;
  gdb_byte mem[sizeof (fd)];
  int status = 0;

  if (batch_flag)
    /* We can't attach with v1 in -batch mode as no event loop is available. */
    return cuda_attach_protocol_support::v0_only;

  if (is_remote_target (inf->process_target ()))
    /* For now, use the old protocol for remote targets.
       There is no reason why it can't be done, but we need to
       rework the FD signalling mechanism to work properly on remote targets.
       We currently signal the fd by using /proc on the host machine. */
    return cuda_attach_protocol_support::v0_only;

  symbol_address = cuda_get_symbol_address (
      _STRING_ (CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD));

  if (!symbol_address)
    return cuda_attach_protocol_support::v0_only;

  status = target_read_memory (symbol_address, mem, sizeof (fd));

  if (status != 0)
    error (_ ("target_read_memory failed"));

  memcpy (&fd, mem, sizeof (fd));

  /* The fd is available but not initialized yet.  This means we're attaching
    very early before the driver has fully initialized. */
  if (fd < 0)
    return cuda_attach_protocol_support::v1_supported_later;

  return cuda_attach_protocol_support::v1_supported;
}

static bool
cuda_is_debugger_initialized ()
{
  CORE_ADDR initialized_flag_address
      = cuda_get_symbol_address (_STRING_ (CUDBG_DEBUGGER_INITIALIZED));

  if (!initialized_flag_address)
    error (_ ("Failed to get initialized flag address."));

  uint32_t initialized_flag = 0;
  gdb_byte mem[sizeof (initialized_flag)];
  int status = target_read_memory (initialized_flag_address, mem,
				   sizeof (initialized_flag));

  if (status != 0)
    error (_ ("target_read_memory failed"));

  memcpy (&initialized_flag, mem, sizeof (initialized_flag));

  return initialized_flag != 0;
}

static void
cuda_request_safe_library_injection (inferior *inf)
{
  CORE_ADDR symbol_address = 0;
  int32_t fd = 0;
  gdb_byte mem[sizeof (fd)];
  int status = 0;

  symbol_address = cuda_get_symbol_address (
      _STRING_ (CUDBG_INITIATE_DEBUGGER_ATTACH_PROCEDURE_FD));

  if (!symbol_address)
    error (_ ("Failed to get debug library injection request event fd."));

  status = target_read_memory (symbol_address, mem, sizeof (fd));

  if (status != 0)
    error (_ ("target_read_memory failed"));

  memcpy (&fd, mem, sizeof (fd));

  if (fd < 0)
    error (_ ("fd to request library injection is unset"));

  char filename[256];
  snprintf (filename, sizeof (filename), "/proc/%d/fd/%d", inf->pid, fd);

  int pipe_fd = open (filename, O_WRONLY);
  if (pipe_fd < 0)
    error (_ ("Failed to open file to trigger safe library injection"));

  SCOPE_EXIT { close (pipe_fd); };

  uint8_t magic_byte = 0x0;
  ssize_t written = write (pipe_fd, &magic_byte, sizeof (magic_byte));

  if (written == -1)
    error (_ ("Failed to write to library injection request event pipe"));
}

/* Returns true if attach is complete (synchronous), false if async
   continuation was added (caller should continue the target).  */
static bool cuda_nat_attach_post_library_injection (inferior *inf);
static void cuda_nat_attach_finish (inferior *inf, bool notify_stop);

static void
cuda_inject_debug_library_new (inferior *inf)
{
  cuda_trace ("cuda_inject_debug_library_new: entering");

  if (cuda_is_debugger_initialized ())
    {
      /* Nothing to inject, let's continue. */
      cuda_trace ("cuda_inject_debug_library_new: already initialized, calling post_library_injection");
      cuda_nat_attach_post_library_injection (inf);
      return;
    }

  /* Tell the driver to safely inject the library and initialize it. */
  cuda_request_safe_library_injection (inf);

  /* At this point we need to continue the target to let it handle
    the library injection request.  After it does so, we will hit a breakpoint
    in cudbgReportAttachProcedureFinished. */

  if (inferior_thread ()->state == THREAD_RUNNING)
    return;

  /* Mark that we're in the library injection phase.  */
  inf->cuda_attach_state = inferior::cuda_attach_state::INJECTING;

  prepare_execution_command (inf->top_target (), true);
  continue_1 (true);

  /* Block user input until CUDA attach completes.  We set
     keep_prompt_blocked to prevent async_enable_stdin() in
     normal_stop() (called from attach_post_wait) from re-enabling
     input.  This flag is cleared when attach completes.  */
  current_ui->keep_prompt_blocked = true;
}

/* Clean up attach state when cancelled by user (Ctrl-C).  */
static void
cuda_attach_cleanup_on_cancel (inferior *inf)
{
  cuda_trace ("cuda_attach_cleanup_on_cancel: cleaning up");

  inf->cuda_attach_state = inferior::cuda_attach_state::NONE;

  if (inf->cuda_saved_sigs != nullptr)
    {
      cuda_nat_bypass_signals_cleanup (inf->cuda_saved_sigs);
      inf->cuda_saved_sigs = nullptr;
    }

  /* Reset attach state so a future attach can succeed.  */
  cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_NOT_STARTED);

  /* Re-enable user input.  */
  current_ui->keep_prompt_blocked = false;
  async_enable_stdin ();
}

/* Pre-wait continuation for old attach protocol retry logic.
   This runs before each target_wait, after the target has stopped
   from our interrupt.  */
static void
cuda_inject_debug_library_old_continuation (inferior *inf,
					    unsigned retry_count,
					    unsigned retry_delay,
					    unsigned app_init_timeout)
{
  struct cmd_list_element *alias = NULL;
  struct cmd_list_element *prefix_cmd = NULL;
  struct cmd_list_element *cmd = NULL;
  const char *cudbgApiAttach = "(void) cudbgApiAttach()";

  cuda_trace ("cuda_inject_debug_library_old_continuation: retry_count=%u",
	      retry_count);

  /* Give up if the process has exited.  */
  if (!inf->process_target ())
    {
      cuda_trace ("cuda_inject_debug_library_old_continuation: process exited, giving up");
      return;
    }

  /* Check if user cancelled with Ctrl-C.  We check both the thread's
     stop signal and the quit flag.  */
  {
    bool is_sigint = false;
    if (inferior_ptid != null_ptid)
      {
	thread_info *tp = inferior_thread ();
	if (tp != nullptr)
	  is_sigint = (tp->stop_signal () == GDB_SIGNAL_INT);
      }

    if (is_sigint || check_quit_flag ())
      {
	cuda_trace ("cuda_inject_debug_library_old_continuation: cancelled by user");
	cuda_attach_cleanup_on_cancel (inf);
	gdb_printf (_ ("CUDA attach cancelled.\n"));
	return;
      }
  }

  /* Mark target as stopped - it should have stopped from our interrupt.  */
  set_running (inf->process_target (), minus_one_ptid, 0);

  /* Check timeout.  */
  if (retry_count * retry_delay >= app_init_timeout)
    {
      /* Timeout - proceed anyway.  */
      cuda_nat_attach_post_library_injection (inf);
      return;
    }

  if (!lookup_cmd_composition ("call", &alias, &prefix_cmd, &cmd))
    error (_ ("Failed to initiate attach."));

  /* Try to init debugger's backend.  */
  unsigned char *sigs = cuda_gdb_bypass_signals ();
  cuda_gdb_bypass_signals_cleanup cleanup (sigs);
  cmd_func (cmd, cudbgApiAttach, 0);
  cleanup.release ();
  cuda_nat_bypass_signals_cleanup (sigs);

  uint64_t internal_error_code = cuda_get_last_driver_internal_error_code ();

  /* CUDBG_ERROR_ATTACH_NOT_POSSIBLE can be returned in two scenarios:
   * 1. Attach is really not possible
   * 2. Critical section's mutex is taken, attaching would cause a deadlock
   */
  bool need_retry = (unsigned int)internal_error_code
		    == CUDBG_ERROR_ATTACH_NOT_POSSIBLE;

  if (need_retry)
    {
      /* Add continuation for next retry.  */
      unsigned next_count = retry_count + 1;
      inf->add_pre_wait_continuation (
	[inf, next_count, retry_delay, app_init_timeout] () {
	  cuda_inject_debug_library_old_continuation (inf, next_count,
						      retry_delay,
						      app_init_timeout);
	});

      /* Resume the target.  */
      prepare_execution_command (inf->top_target (), true);
      continue_1 (true);

      usleep (retry_delay * 1000);

      /* Trigger the future wait().  */
      interrupt_target_1 (true);
      return;
    }

  /* Success - proceed with post-injection.  */
  cuda_nat_attach_post_library_injection (inf);
}

static void
cuda_inject_debug_library_old (inferior *inf)
{
  struct cmd_list_element *alias = NULL;
  struct cmd_list_element *prefix_cmd = NULL;
  struct cmd_list_element *cmd = NULL;
  const char *cudbgApiAttach = "(void) cudbgApiAttach()";
  unsigned retry_delay = 100;	    // ms
  unsigned app_init_timeout = 5000; // ms

  cuda_trace ("cuda_inject_debug_library_old: entering");

  if (!lookup_cmd_composition ("call", &alias, &prefix_cmd, &cmd))
    error (_ ("Failed to initiate attach."));

  /* First attempt to init debugger's backend.  */
  unsigned char *sigs = cuda_gdb_bypass_signals ();
  cuda_gdb_bypass_signals_cleanup cleanup (sigs);
  cmd_func (cmd, cudbgApiAttach, 0);
  cleanup.release ();
  cuda_nat_bypass_signals_cleanup (sigs);

  uint64_t internal_error_code = cuda_get_last_driver_internal_error_code ();

  /* CUDBG_ERROR_ATTACH_NOT_POSSIBLE can be returned in two scenarios:
   * 1. Attach is really not possible
   * 2. Critical section's mutex is taken, attaching would cause a deadlock
   */
  bool need_retry = (unsigned int)internal_error_code
		    == CUDBG_ERROR_ATTACH_NOT_POSSIBLE;

  if (need_retry)
    {
      /* Mark that we're in the library injection phase.  */
      inf->cuda_attach_state = inferior::cuda_attach_state::INJECTING;

      /* Add continuation for retry - will run before next target_wait.  */
      unsigned retry_count = 1;
      inf->add_pre_wait_continuation (
	[inf, retry_count, retry_delay, app_init_timeout] () {
	  cuda_inject_debug_library_old_continuation (inf, retry_count,
						      retry_delay,
						      app_init_timeout);
	});

      /* Resume the target.  */
      prepare_execution_command (inf->top_target (), true);
      continue_1 (true);

      /* Block user input until CUDA attach completes.  */
      current_ui->keep_prompt_blocked = true;

      usleep (retry_delay * 1000);

      /* Trigger the future wait().  */
      interrupt_target_1 (true);
      return;
    }

  /* First attempt succeeded.  */
  cuda_nat_attach_post_library_injection (inf);
}

static void
cuda_nat_attach (inferior *inf)
{
  CORE_ADDR attachDataAvailableFlagAddr = 0;

  /* Give up if the process has exited.  */
  if (!inf->process_target ())
    {
      cuda_trace ("cuda_nat_attach: process exited, giving up");
      return;
    }

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

      /* TODO: This isn't actually used. Do we need to continue reading this
       * value? */
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

  switch (cuda_get_attach_protocol_support (inf))
    {
    case cuda_attach_protocol_support::v0_only:
      cuda_inject_debug_library_old (inf);
      break;
    case cuda_attach_protocol_support::v1_supported:
      cuda_inject_debug_library_new (inf);
      break;
    case cuda_attach_protocol_support::v1_supported_later:
      /* The CUDA driver hasn't fully initialized yet.  We need to continue
	 the target so the driver can finish initializing.  The pre-wait
	 continuation (cuda_initialize_pre_wait_continuation) will check
	 the cuda_attach_state and call cuda_nat_attach when initialization
	 succeeds.  */
      gdb_printf (_ ("The CUDA driver has not initialized yet, the attach "
		     "procedure will finish later.\n"));
      gdb_printf (_ ("CUDA features will not be available until the driver "
		     "has initialized.\n"));

      /* Set state so pre-wait continuation knows to call cuda_nat_attach
	 after driver initialization.  */
      inf->cuda_attach_state = inferior::cuda_attach_state::WAITING_FOR_DRIVER;

      /* Continue the target so the driver can finish initializing.  */
      if (inferior_thread ()->state != THREAD_RUNNING)
	{
	  prepare_execution_command (inf->top_target (), true);
	  continue_1 (true);
	}

      /* Block user input until CUDA attach completes.  */
      current_ui->keep_prompt_blocked = true;
      async_disable_stdin ();
      break;
    }
}

/* Normal stop observer for CUDA attach.
   Handles Ctrl-C for any attach phase, and the resumeAppOnAttach loop.  */
static void
cuda_attach_normal_stop_observer (struct bpstat *bs, int print_frame)
{
  inferior *inf = current_inferior ();

  cuda_trace ("cuda_attach_normal_stop_observer: called, inf=%p, print_frame=%d",
	      inf, print_frame);

  if (inf == nullptr)
    {
      cuda_trace ("cuda_attach_normal_stop_observer: inf is nullptr, returning");
      return;
    }

  cuda_trace ("cuda_attach_normal_stop_observer: attach_state=%d",
	      static_cast<int> (inf->cuda_attach_state));

  /* Give up if the process has exited.  */
  if (!inf->process_target ())
    {
      cuda_trace ("cuda_attach_normal_stop_observer: process exited, returning");
      return;
    }

  /* Check if user cancelled with Ctrl-C during ANY attach phase.
     We check both the thread's stop signal (SIGINT) and the quit flag.
     The quit flag might already be cleared by other GDB code, so we
     primarily rely on the stop signal.  */
  if (inf->cuda_attach_state != inferior::cuda_attach_state::NONE)
    {
      bool is_sigint = false;
      thread_info *tp = (inferior_ptid != null_ptid) ? inferior_thread () : nullptr;
      if (tp != nullptr)
	is_sigint = (tp->stop_signal () == GDB_SIGNAL_INT);

      if (is_sigint || check_quit_flag ())
	{
	  cuda_trace ("cuda_attach_normal_stop_observer: cancelled by user (sigint=%d)",
		      is_sigint);
	  cuda_attach_cleanup_on_cancel (inf);
	  gdb_printf (_ ("CUDA attach cancelled.\n"));
	  return;
	}
    }

  /* Not in the resumeAppOnAttach loop - nothing more to do.  */
  if (inf->cuda_attach_state != inferior::cuda_attach_state::RESUMING)
    return;

  cuda_trace ("cuda_attach_normal_stop_observer: api_attach_state=%d",
	      cuda_debugapi::get_attach_state ());

  /* Check if attach is complete.  */
  if (cuda_debugapi::get_attach_state () == CUDA_ATTACH_STATE_APP_READY
      || cuda_debugapi::get_attach_state () == CUDA_ATTACH_STATE_COMPLETE)
    {
      cuda_trace ("cuda_attach_normal_stop_observer: attach complete, finishing");

      /* Cleanup signal bypass.  */
      cuda_nat_bypass_signals_cleanup (inf->cuda_saved_sigs);
      inf->cuda_saved_sigs = nullptr;

      /* Complete the attach.  */
      cuda_nat_attach_finish (inf, true);
      return;
    }

  /* Attach still in progress - resume target to receive more events.  */
  cuda_trace ("cuda_attach_normal_stop_observer: still in progress, resuming");

  /* Resume the target to continue receiving CUDA events.
     Use proceed() which is the standard way to resume after a stop.  */
  clear_proceed_status (0);
  proceed ((CORE_ADDR) -1, GDB_SIGNAL_0);
}

/* Final steps of attach after all data collection is complete.  */
static void
cuda_nat_attach_finish (inferior *inf, bool notify_stop)
{
  cuda_trace ("cuda_nat_attach_finish: entering, notify_stop=%d", notify_stop);

  /* Clear attach state.  */
  inf->cuda_attach_state = inferior::cuda_attach_state::NONE;

  /* The inferior just got signaled, we're not expecting any other stop */
  inf->control.stop_soon = NO_STOP_QUIETLY;

  /* After attach, force this to "unknown state".
     This is required because we need to call `mark_async_event_handler()'
     and will be set to true later anyways.
     It is set to true as part of normal GDB attach code. */
  infrun_async (-1);

  /* Ensure GDB owns the terminal.  During async attach, the terminal
     state may not be properly restored.  */
  target_terminal::ours ();

  /* Clear the prompt block that was set when attach started.
     We set keep_prompt_blocked = true in cuda_inject_debug_library_*
     to prevent normal_stop() from enabling input during the attach.
     Now that attach is complete, clear it and re-enable input.  */
  current_ui->keep_prompt_blocked = false;
  current_ui->prompt_state = PROMPT_BLOCKED;
  async_enable_stdin ();

  /* If we completed attach via the normal_stop observer path, we need
     to notify the stop now since the attach breakpoint is silent.  */
  if (notify_stop)
    interps_notify_normal_stop (nullptr, 1);
}

/* See cuda-linux-nat.h.  */

bool
cuda_complete_async_attach (struct inferior *inf)
{
  cuda_trace ("cuda_complete_async_attach: entering, attach_state=%d",
	      inf ? static_cast<int> (inf->cuda_attach_state) : -1);

  if (inf == nullptr)
    return false;

  /* Check if we're in the resumeAppOnAttach loop.  */
  if (inf->cuda_attach_state != inferior::cuda_attach_state::RESUMING)
    return false;

  cuda_trace ("cuda_complete_async_attach: completing async attach");

  /* Cleanup signal bypass.  */
  if (inf->cuda_saved_sigs != nullptr)
    {
      cuda_nat_bypass_signals_cleanup (inf->cuda_saved_sigs);
      inf->cuda_saved_sigs = nullptr;
    }

  /* Complete the attach.  This sets stop_soon = NO_STOP_QUIETLY, which
     allows fetch_inferior_event to call normal_stop() after cuda_wait
     returns.  We pass notify_stop=false because normal_stop() will
     handle the notification.  */
  cuda_nat_attach_finish (inf, false);

  /* Return true so the template code does NOT set STOP_QUIETLY.
     This allows normal_stop() to be called from fetch_inferior_event.  */
  return true;
}

static bool
cuda_nat_attach_post_library_injection (inferior *inf)
{
  CORE_ADDR debugFlagAddr = 0;
  CORE_ADDR resumeAppOnAttachFlagAddr = 0;
  CORE_ADDR attachDataAvailableFlagAddr = 0;
  unsigned char resumeAppOnAttach = 0;
  unsigned int timeOut = 5000; // ms
  unsigned int timeElapsed = 0;
  unsigned dev = 0;
  const unsigned int sleepTime = 1; // ms
  uint64_t internal_error_code;

  cuda_trace ("cuda_nat_attach_post_library_injection: entering");

  /* Give up if the process has exited.  */
  if (!inf->process_target ())
    {
      cuda_trace ("cuda_nat_attach_post_library_injection: process exited, giving up");
      return false;
    }

  debugFlagAddr = cuda_get_symbol_address (_STRING_ (CUDBG_IPC_FLAG_NAME));
  resumeAppOnAttachFlagAddr
      = cuda_get_symbol_address (_STRING_ (CUDBG_RESUME_FOR_ATTACH_DETACH));
  attachDataAvailableFlagAddr
      = cuda_get_symbol_address (_STRING_ (CUDBG_ATTACH_HANDLER_AVAILABLE));

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
      capabilities |= CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS;

      cuda_trace_domain (CUDA_TRACE_GENERAL,
			 "requesting CUDA suspend events\n");
      capabilities |= CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS;

      if (cuda_options_driver_logs_enabled ())
	{
	  cuda_trace_domain (CUDA_TRACE_GENERAL,
			     "requesting CUDA UMD logs collection\n");
	  capabilities |= CUDBG_DEBUGGER_CAPABILITY_ENABLE_CUDA_LOGS;
	}

      if (cuda_options_printf_flushing ())
	{
	  cuda_trace_domain (CUDA_TRACE_GENERAL,
			     "requesting CUDA printf flushing on suspend\n");
	  capabilities |= CUDBG_DEBUGGER_CAPABILITY_FLUSH_PRINTF_ON_SUSPEND;
	}

      if (cuda_options_kernel_launch_backtrace_enabled ())
	{
	  cuda_trace_domain (
	      CUDA_TRACE_GENERAL,
	      "requesting collection of CPU call stack for kernel launches\n");
	  capabilities |= CUDBG_DEBUGGER_CAPABILITY_COLLECT_CPU_CALL_STACK_FOR_KERNEL_LAUNCHES;
	}

#if CUDBG_API_VERSION_REVISION > 167
      /* Always request break-on-launch capability for CUDA 13.2+ support.
	 The driver will ignore this flag if it doesn't understand it. */
      cuda_trace_domain (CUDA_TRACE_GENERAL,
			 "requesting break-on-launch capability\n");
      capabilities |= CUDBG_DEBUGGER_CAPABILITY_BREAK_ON_LAUNCH;
#endif

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
	    error (_ ("Timed out waiting for the CUDA remote target to "
		      "initialize."));

	  timeElapsed += sleepTime;
	}
      timeElapsed = 0;
    }

  /* Wait till the backend has started up and is ready to service API calls */
  while (cuda_debugapi::initialize () != CUDBG_SUCCESS)
    {
      internal_error_code = cuda_get_last_driver_internal_error_code ();
      if ((unsigned int)internal_error_code == CUDBG_ERROR_ATTACH_NOT_POSSIBLE)
	error (_ ("Failed to attach. For more information, please see https://docs.nvidia.com/cuda/cuda-gdb/index.html#known-issues"));
      else if (internal_error_code)
	error (_ ("Attach failed due to an internal driver error: %llu"),
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

  cuda_trace ("cuda_nat_attach_post_library_injection: resumeAppOnAttach=%d",
	      resumeAppOnAttach);

  if (resumeAppOnAttach)
    {
      cuda_trace ("cuda_nat_attach_post_library_injection: resumeAppOnAttach=1, "
		  "setting up async attach loop");

      /* Setup signal bypass - saved in inferior for cleanup later.  */
      inf->cuda_saved_sigs = cuda_gdb_bypass_signals ();

      /* Set state for the normal_stop observer to handle the loop.
	 The observer will check attach state after each stop and either
	 resume or complete the attach.  */
      inf->cuda_attach_state = inferior::cuda_attach_state::RESUMING;

      /* Block user input until attach completes.  The prompt will be
	 re-enabled in cuda_nat_attach_finish when the attach loop
	 completes via async_enable_stdin() in normal_stop().  */
      async_disable_stdin ();

      /* Return false to indicate async - caller should continue target.
	 The normal_stop observer will handle subsequent stops.  */
      return false;
    }
  else
    {
      cuda_trace ("cuda_nat_attach_post_library_injection: sync path (no resume needed)");
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

  /* Synchronous path - attach is complete.  */
  cuda_nat_attach_finish (inf, false);
  return true;
}

#endif /* !__QNXTARGET__ */

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

#ifndef __QNXTARGET__
/* Called from bpstat_what when bp_cuda_attach_initiated is hit.
   Returns true if attach is complete (should stop), false if async
   continuation was added (should continue).
   Attach is not supported on QNX.  */
bool
cuda_handle_attach_initiated_breakpoint (void)
{
  cuda_trace ("cuda_handle_attach_initiated_breakpoint: entering");

  inferior *inf = current_inferior ();
  bool complete = cuda_nat_attach_post_library_injection (inf);

  cuda_trace ("cuda_handle_attach_initiated_breakpoint: complete=%d", complete);

  return complete;
}
#endif /* !__QNXTARGET__ */

/* Final cleanup after detach loop completes.  */
static void
cuda_do_detach_finish (inferior *inf, CORE_ADDR debugFlagAddr)
{
  cuda_trace ("cuda_do_detach_finish: entering, attach_state=%d",
	      cuda_debugapi::get_attach_state ());

  if (inf->process_target ())
    {
      if (cuda_debugapi::get_attach_state ()
	  != CUDA_ATTACH_STATE_DETACH_COMPLETE)
	warning (_ ("Unexpected CUDA API attach state %d."),
		 cuda_debugapi::get_attach_state ());

      cuda_write_bool (debugFlagAddr, false);
    }

  /* Re-enable user input now that detach is complete.  */
  current_ui->keep_prompt_blocked = false;
  async_enable_stdin ();

  cuda_cleanup ();
}

/* Pre-wait continuation for detach cleanup loop.
   This runs before each target_wait, handling the detach state machine.  */
static void
cuda_do_detach_continuation (inferior *inf, CORE_ADDR debugFlagAddr, int cnt)
{
  const int max_iterations = 100;

  cuda_trace ("cuda_do_detach_continuation: cnt=%d, attach_state=%d",
	      cnt, cuda_debugapi::get_attach_state ());

  /* Check if user cancelled with Ctrl-C.  */
  if (check_quit_flag ())
    {
      cuda_trace ("cuda_do_detach_continuation: cancelled by user");
      /* Re-enable user input.  */
      current_ui->keep_prompt_blocked = false;
      async_enable_stdin ();
      /* Still need to clean up.  */
      cuda_do_detach_finish (inf, debugFlagAddr);
      return;
    }

  /* Restore commit_resumed_state.  */
  if (inf->process_target ())
    inf->process_target ()->commit_resumed_state
	= inf->cuda_saved_commit_resumed_state;

  /* Process may have exited at this point.  */
  if (!inf->process_target ())
    {
      cuda_do_detach_finish (inf, debugFlagAddr);
      return;
    }

  /* Check if we should continue the loop.  */
  if (cnt < max_iterations
      && cuda_debugapi::get_attach_state () != CUDA_ATTACH_STATE_DETACH_COMPLETE)
    {
      /* Add continuation for next iteration.  */
      inf->add_pre_wait_continuation ([inf, debugFlagAddr, cnt] () {
	cuda_do_detach_continuation (inf, debugFlagAddr, cnt + 1);
      });

      prepare_execution_command (inf->top_target (), true);
      continue_1 (false);

      /* Force resumed state to false.  */
      inf->cuda_saved_commit_resumed_state
	  = inf->process_target ()->commit_resumed_state;
      inf->process_target ()->commit_resumed_state = false;

      /* Brief sleep to allow CUDA events to be generated.  */
      usleep (1000);

      /* Trigger the future wait() - this will run our continuation.  */
      interrupt_target_1 (true);
      return;
    }

  /* Loop finished - cleanup.  */

  /* No threads are running at this point.  */
  if (inf->process_target ())
    set_running (inf->process_target (), minus_one_ptid, 0);

  cuda_do_detach_finish (inf, debugFlagAddr);
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

  cuda_trace ("cuda_do_detach: entering");

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

  /* This is a bit of a hack. We are about to tear down the debug API so future
   * calls would fail. But if there are any breakpoints set, those usually
   * would be removed after detach completes. A bug was found with the debug
   * API where if breakpoints are not removed, they would not get cleaned up on
   * detach correctly and left set. We were masking this bug in previous
   * implementations of CUDA-GDB. To work around this, always try to delete
   * breakpoints belonging to the inferiors program space before we tear down
   * the debug API. */
  cuda_options_disable_break_on_launch ();
  breakpoint_program_space_exit (inf->pspace);

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

  cuda_trace ("cuda_do_detach: resumeAppOnDetach=%d", resumeAppOnDetach);

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
      cuda_trace ("cuda_do_detach: adding continuation and resuming");

      /* For remote targets, handle the detach cleanup synchronously.
	 The async continuation approach doesn't work for remote because:
	 (a) the event loop may not run (e.g. during quit), so the
	 continuation and deferred target_detach would never execute, and
	 (b) the caller (BaseTarget::detach) needs to send packets
	 immediately after cuda_do_detach returns.
	 Loop with target_wait until the driver signals detach-complete,
	 mirroring the native async continuation logic.  */
      if (is_remote_target (inf->process_target ()))
	{
	  const int max_iterations = 100;

	  cuda_trace ("cuda_do_detach: remote sync detach path");

	  int cnt;
	  for (cnt = 0; cnt < max_iterations; cnt++)
	    {
	      prepare_execution_command (inf->top_target (), true);
	      continue_1 (false);

	      usleep (1000);
	      interrupt_target_1 (true);

	      target_waitstatus ws;
	      ptid_t ret_ptid = target_wait (minus_one_ptid, &ws, 0);

	      if (ret_ptid == minus_one_ptid)
		{
		  cuda_trace ("cuda_do_detach: target_wait returned "
			      "no event, aborting detach loop");
		  break;
		}

	      bool done = false;
	      switch (ws.kind ())
		{
		case TARGET_WAITKIND_EXITED:
		  cuda_trace ("cuda_do_detach: process exited with "
			      "status %d during detach",
			      ws.exit_status ());
		  done = true;
		  break;
		case TARGET_WAITKIND_SIGNALLED:
		  cuda_trace ("cuda_do_detach: process killed by "
			      "signal %d during detach",
			      (int) ws.sig ());
		  done = true;
		  break;
		case TARGET_WAITKIND_NO_RESUMED:
		  cuda_trace ("cuda_do_detach: no resumed threads, "
			      "aborting detach loop");
		  done = true;
		  break;
		default:
		  break;
		}

	      if (done)
		break;

	      set_running (inf->process_target (), minus_one_ptid, 0);

	      if (check_quit_flag ())
		{
		  cuda_trace ("cuda_do_detach: remote detach cancelled "
			      "by user");
		  break;
		}

	      if (cuda_debugapi::get_attach_state ()
		  == CUDA_ATTACH_STATE_DETACH_COMPLETE)
		break;

	      cuda_trace ("cuda_do_detach: remote detach iteration %d, "
			  "attach_state=%d", cnt,
			  cuda_debugapi::get_attach_state ());
	    }

	  if (cnt == max_iterations)
	    warning (_ ("CUDA detach cleanup did not complete after %d "
			"iterations (attach_state=%d)."),
		     max_iterations,
		     cuda_debugapi::get_attach_state ());

	  cuda_do_detach_finish (inf, debugFlagAddr);
	  return;
	}

      /* Native targets: use async pre-wait continuation.  */

      /* Add continuation for the detach cleanup loop.  */
      inf->add_pre_wait_continuation ([inf, debugFlagAddr] () {
	cuda_do_detach_continuation (inf, debugFlagAddr, 0);
      });

      /* Resume the app and wait for CUDA_ATTACH_STATE_DETACH_COMPLETE event.  */
      prepare_execution_command (inf->top_target (), true);
      continue_1 (false);

      /* Force resumed state to false.  */
      inf->cuda_saved_commit_resumed_state
	  = inf->process_target ()->commit_resumed_state;
      inf->process_target ()->commit_resumed_state = false;

      /* Brief sleep to allow CUDA events to be generated.  */
      usleep (1000);

      /* Block user input until detach completes.  */
      current_ui->keep_prompt_blocked = true;
      async_disable_stdin ();

      cuda_trace ("cuda_do_detach: calling interrupt_target_1");

      /* Trigger the future wait() - this will run our continuation.  */
      interrupt_target_1 (true);
      return;
    }
  else
    {
      cuda_trace ("cuda_do_detach: sync path (no resume needed)");
      cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_DETACH_COMPLETE);
    }

  cuda_do_detach_finish (inf, debugFlagAddr);
}

void
switch_to_cuda_thread (const cuda_coords &coords)
{
  uint64_t pc;

  cuda_current_focus::set (coords);

  thread_info *thr
      = current_inferior ()->process_target ()->find_thread (inferior_ptid);
  /* Only update if a host thread still exists. */
  if (thr)
    {
      switch_to_thread_keep_cuda_focus (thr);

      if (coords.isValidOnDevice ())
	pc = cuda_state::lane_get_pc (
	    coords.physical ().dev (), coords.physical ().sm (),
	    coords.physical ().wp (), coords.physical ().ln ());
      else
	pc = (CORE_ADDR)~0;

      thr->set_stop_pc (pc);
    }
}

void
cuda_init_cudart_symbols (void)
{
  /* If not done yet, create a CUDA runtime symbols file */
  if (!cuda_cudart_symbols)
    {
      cuda_cudart_symbols = cuda_create_builtins_objfile ();
    }
}

void
cuda_cleanup_cudart_symbols (void)
{
  /* Free the objfile if allocated */
  if (cuda_cudart_symbols)
    {
      cuda_cudart_symbols->unlink ();
      cuda_cudart_symbols = NULL;
    }
#ifdef __QNXTARGET__
  /* Reset the RT symbols in qnx */
  cuda_reset_qnx_symbols ();
#endif
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
  fp.set_bitsize (type->length () * 8);
}

/* Allocates dim3 type as structure of 3 packed unsigned int: x, y and z */
static struct type *
cuda_alloc_dim3_type (struct objfile *objfile)
{
  struct gdbarch *gdbarch = objfile->arch ();
  struct type *uint32_type = builtin_type (gdbarch)->builtin_unsigned_int;
  struct type *dim3 = type_allocator (uint32_type).new_type ();

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

/* Add a built-in symbol to the CUDA builtins objfile */
static void
cuda_add_builtin_symbol (struct objfile *objfile, struct symtab *symtab,
                         struct global_block *global_block, const char *name,
                         CORE_ADDR addr, struct type *type)
{
  struct symbol *sym = new (&objfile->objfile_obstack) symbol;

  sym->set_language (language_c, &objfile->per_bfd->storage_obstack);
  sym->compute_and_set_names (name, true, objfile->per_bfd);
  sym->set_type (type);
  sym->set_domain (VAR_DOMAIN);
  sym->set_aclass_index (LOC_STATIC);
  sym->set_value_address (addr);
  sym->set_symtab (symtab);

  mdict_add_symbol (global_block->multidict (), sym);
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
static struct objfile *
cuda_create_builtins_objfile (void)
{
  struct objfile *objfile = nullptr;
  struct type *int32_type = nullptr;
  struct type *dim3_type = nullptr;

  /* This is not a real objfile.  Mark it as so by passing OBJF_NOT_FILENAME.  */
  objfile = objfile::make(nullptr, current_program_space, nullptr, OBJF_NOT_FILENAME);
  objfile->per_bfd->gdbarch = cuda_get_gdbarch ();
  objfile->cuda_objfile = true;

  /* Get/allocate types */
  int32_type = builtin_type ((objfile->arch ()))->builtin_int32;
  dim3_type = cuda_alloc_dim3_type (objfile);

  /* Create minimal symbol table structures */
  struct compunit_symtab *cust = allocate_compunit_symtab (objfile, 
                                                           "<cuda-builtins>");
  struct symtab *symtab = allocate_symtab (cust, "<cuda-builtins>");
  symtab->set_language (language_c);
  cust->set_primary_filetab (symtab);
  add_compunit_symtab_to_objfile (cust);

  /* Create a minimal blockvector with just global and static blocks */
  struct blockvector *bv = (struct blockvector *)
    obstack_alloc (&objfile->objfile_obstack,
                   sizeof (struct blockvector) + sizeof (struct block *));
  bv->set_num_blocks (2);

  /* Create and set up the global block */
  struct global_block *gb = new (&objfile->objfile_obstack) global_block;
  gb->set_multidict (mdict_create_hashed_expandable (language_c));
  gb->set_compunit (cust);
  bv->set_block (GLOBAL_BLOCK, gb);

  /* Create and set up the static block */
  struct block *sb = new (&objfile->objfile_obstack) struct block;
  sb->set_multidict (mdict_create_hashed_expandable (language_c));
  sb->set_superblock (gb);
  bv->set_block (STATIC_BLOCK, sb);

  cust->set_blockvector (bv);

  /* Create CUDA built-in symbols and add them to the global block */
  cuda_add_builtin_symbol (objfile, symtab, gb, "threadIdx", 
                           CUDBG_THREADIDX_OFFSET, dim3_type);
  cuda_add_builtin_symbol (objfile, symtab, gb, "blockIdx", 
                           CUDBG_BLOCKIDX_OFFSET, dim3_type);
  cuda_add_builtin_symbol (objfile, symtab, gb, "clusterIdx", 
                           CUDBG_CLUSTERIDX_OFFSET, dim3_type);
  cuda_add_builtin_symbol (objfile, symtab, gb, "gridDim", 
                           CUDBG_GRIDDIM_OFFSET, dim3_type);
  cuda_add_builtin_symbol (objfile, symtab, gb, "blockDim", 
                           CUDBG_BLOCKDIM_OFFSET, dim3_type);
  cuda_add_builtin_symbol (objfile, symtab, gb, "clusterDim", 
                           CUDBG_CLUSTERDIM_OFFSET, dim3_type);
  cuda_add_builtin_symbol (objfile, symtab, gb, "warpSize", 
                           CUDBG_WARPSIZE_OFFSET, int32_type);

  return objfile;
}

/* Check if the given library name matches libcuda.so.
   The library name can be:
   - libcuda.so
   - libcuda.so.1
   - libcuda.so.XXX.YY.ZZ (versioned)
   - /path/to/libcuda.so.XXX.YY.ZZ
*/
static bool
cuda_is_libcuda (const char *name)
{
  if (name == nullptr || *name == '\0')
    return false;

  /* Find the basename by looking for the last '/' */
  const char *basename = strrchr (name, '/');
  if (basename != nullptr)
    basename++;  /* Skip the '/' */
  else
    basename = name;

  /* Check if it starts with "libcuda.so" */
  return strncmp (basename, "libcuda.so", 10) == 0;
}

/* Forward declaration for the pre-wait continuation.  */
static void cuda_initialize_pre_wait_continuation (inferior *inf);

/* Pre-wait continuation that tries to initialize CUDA before each wait.
   If the debugger API is not ready yet, re-adds itself to try again
   on the next wait.  Once initialization succeeds, it does not re-add
   itself and the continuation is consumed.  */
static void
cuda_initialize_pre_wait_continuation (inferior *inf)
{
  /* Don't initialize if already done */
  if (cuda_initialized || inf->cuda_initialized)
    {
      cuda_trace ("cuda_initialize_pre_wait_continuation: already initialized");
      return;
    }

  /* Give up if the process has exited.  */
  if (!inf->process_target ())
    {
      cuda_trace ("cuda_initialize_pre_wait_continuation: process exited, giving up");
      return;
    }

  /* Give up if we initiated a detach. */
  if (cuda_debugapi::get_attach_state () == CUDA_ATTACH_STATE_DETACHING)
    {
      cuda_trace ("cuda_initialize_pre_wait_continuation: detach initiated, giving up!");
      return;
    }

  /* For remote targets, initialization is done directly in cuda_wait,
     not via continuations.  Skip if this is a remote target.  */
  if (is_remote_target (inf->process_target ()))
    {
      cuda_trace ("cuda_initialize_pre_wait_continuation: remote target, "
		  "initialization handled in cuda_wait");
      return;
    }

  /* Try to initialize the CUDA target (native targets only).  */
  cuda_trace ("cuda_initialize_pre_wait_continuation: attempting initialization");
  bool res = cuda_initialize_target ();

  if (!res)
    {
      cuda_trace ("cuda_initialize_pre_wait_continuation: symbols not ready");
      /* Re-add continuation to try again later */
      inf->add_pre_wait_continuation ([inf] () {
	cuda_initialize_pre_wait_continuation (inf);
      });
      return;
    }

  /* cuda_initialize_target succeeded (setup done), but the API might
     not be fully ready yet (cuda_initialized could still be false).
     Keep retrying until fully initialized.  */
  if (!cuda_initialized && !inf->cuda_initialized)
    {
      cuda_trace ("cuda_initialize_pre_wait_continuation: API not ready yet, will retry");
      /* Re-add continuation to try again on next wait */
      inf->add_pre_wait_continuation ([inf] () {
	cuda_initialize_pre_wait_continuation (inf);
      });
      return;
    }

  cuda_trace ("cuda_initialize_pre_wait_continuation: CUDA fully initialized");

#ifndef __QNXTARGET__
  /* If we were waiting for the driver to initialize before completing
     attach (v1_supported_later case), call cuda_nat_attach now.
     Attach is not supported on QNX.  */
  if (inf->cuda_attach_state == inferior::cuda_attach_state::WAITING_FOR_DRIVER)
    {
      cuda_trace ("cuda_initialize_pre_wait_continuation: completing deferred attach");
      cuda_nat_attach (inf);
    }
#endif
}

/* Observer callback for objfile (symbol file) loading.
   When libcuda.so symbols are loaded, we set up a pre-wait continuation
   to initialize the CUDA debugger.  The continuation will keep retrying
   until the debugger API is ready.  */
static void
cuda_new_objfile_observer (struct objfile *objfile)
{
  /* Only process if CUDA debugging is enabled and not already initialized */
  if (!cuda_debugging_enabled)
    return;

  if (cuda_initialized)
    return;

  inferior *inf = current_inferior ();
  if (inf == nullptr)
    return;

  if (inf->cuda_initialized)
    return;

  /* Skip null objfiles */
  if (objfile == nullptr)
    return;

  /* Check if this is libcuda.so */
  if (!cuda_is_libcuda (objfile->original_name))
    return;

  cuda_trace ("cuda_new_objfile_observer: detected libcuda.so symbols loaded (%s)",
	      objfile->original_name);

  /* For remote targets, initialization is done directly in cuda_wait.
     For native targets, add a pre-wait continuation that will try to
     initialize CUDA in cuda_wait.  The continuation will re-add itself
     if the API is not ready yet.  */
  if (is_remote_target (inf->process_target ()))
    {
      cuda_trace ("cuda_new_objfile_observer: remote target, "
		  "initialization will be done in cuda_wait");
      return;
    }

  cuda_trace ("cuda_new_objfile_observer: adding pre-wait initialization continuation");
  inf->add_pre_wait_continuation ([inf] () {
    cuda_initialize_pre_wait_continuation (inf);
  });
}

void _initialize_cuda_nat ();
void
_initialize_cuda_nat ()
{
  /* Initialize the cleanup routines */
  add_final_cleanup ([] () { cuda_final_cleanup (nullptr); });

  gdb::observers::inferior_created.attach (cuda_inferior_created, "CUDA");
  gdb::observers::new_objfile.attach (cuda_new_objfile_observer, "CUDA");
#ifndef __QNXTARGET__
  /* Attach observer is only used for attach, which is not supported on QNX.  */
  gdb::observers::normal_stop.attach (cuda_attach_normal_stop_observer, "CUDA");
#endif

  cuda_debugging_enabled = true;
}
