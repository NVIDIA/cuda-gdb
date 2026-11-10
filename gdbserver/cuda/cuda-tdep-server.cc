/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2013-2026 NVIDIA Corporation
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

#include "cuda-tdep-server.h"
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifndef __QNXHOST__
#include <sys/ptrace.h>
#include <sys/signal.h>
#include <sys/syscall.h>
#endif /* __QNXHOST__ */
#include <sys/wait.h>
#ifndef __QNXHOST__
#include <sys/syscall.h>
#endif /* __QNXHOST__ */
#include "cuda/cuda-notifications.h"
#include <unistd.h>

#ifdef __QNXHOST__
/* For DS_DATA_MAX_SIZE used by cuda_enqueue_trace_message.  */
#include "remote-nto.h"
#endif

#ifdef __QNXHOST__
/* Defined in cuda-nto-server.cc.  */
extern int cuda_write_inferior_memory (CORE_ADDR memaddr,
				       gdb::array_view<const gdb_byte> src);
#else
static int
cuda_write_inferior_memory (CORE_ADDR memaddr,
			    gdb::array_view<const gdb_byte> src)
{
  return target_write_memory (memaddr, src.data (), src.size ());
}
#endif /* __QNXHOST__ */

/*----------------------------------------- Globals
 * ---------------------------------------*/
CUDBGResult api_initialize_res;
CUDBGResult api_finalize_res;
CUDBGResult get_debugger_api_res;
CUDBGResult set_callback_api_res;

struct cuda_sym cuda_symbol_list[] = {
  /* Old fields are left to maintain the binary compatibility with legacy CUDA
   * GDB server binaries */
  CUDA_SYM (CUDBG_IPC_FLAG_NAME),
  CUDA_SYM (CUDBG_RPC_ENABLED),
  CUDA_SYM (CUDBG_APICLIENT_PID),
  CUDA_SYM (CUDBG_APICLIENT_REVISION),
  CUDA_SYM (CUDBG_SESSION_ID),
  CUDA_SYM (CUDBG_ATTACH_HANDLER_AVAILABLE),
  CUDA_SYM (CUDBG_DEBUGGER_INITIALIZED),
  CUDA_SYM (CUDBG_REPORTED_DRIVER_API_ERROR_CODE),
  CUDA_SYM (CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE),
  CUDA_SYM (CUDBG_ENABLE_LAUNCH_BLOCKING),
  /* This symbol is not exposed through cudadebugger.h yet */
  CUDA_SYM (cudbgInjectionPath),
  CUDA_SYM (CUDBG_DEBUGGER_CAPABILITIES),
};

CUDBGAPI cudbgAPI = NULL;
bool cuda_initialized = false;
bool cuda_syms_looked_up = false;

/* CUDA cleanup state tracking for deferred connection teardown */
bool cuda_cleanup_completed = false;
bool cuda_exit_requested = false;

static bool inferior_in_debug_mode = false;
static char cuda_gdb_session_dir[CUDA_GDB_TMP_BUF_SIZE] = { 0 };
static uint32_t cuda_gdb_session_id = 0;
static uint32_t debugger_capabilities = CUDBG_DEBUGGER_CAPABILITY_NONE;

bool cuda_launch_blocking;
bool cuda_debug_general;
bool cuda_debug_libcudbg;
bool cuda_debug_notifications;
bool cuda_notify_youngest;
bool cuda_logs = true;
bool cuda_printf_flushing = false;
bool cuda_lazy_function_finalization = false;
std::deque<std::string> cuda_trace_messages;

ptid_t cuda_last_ptid;
struct target_waitstatus cuda_last_ws;
bool cuda_debugging_enabled = false;

bool
cuda_platform_supports_tid (void)
{
#if defined(linux) && defined(SYS_gettid)
  return true;
#else
  return false;
#endif
}

int
cuda_gdb_get_tid_or_pid (ptid_t ptid)
{
  if (cuda_platform_supports_tid ())
    return ptid.lwp ();
  else
    return ptid.pid ();
}

/*---------------------------------------- sigpending
 * -------------------------------------*/
#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

/* Parse LINE as a signal set and add its set bits to SIGS.  */

#ifndef __QNXHOST__
static int
add_line_to_sigset (const char *line, sigset_t *sigs)
{
  int len = strlen (line) - 1;
  const char *p;
  int signum;

  if (line[len] != '\n')
    {
      fprintf (stderr, "Could not parse signal set: %s", line);
      return -EINVAL;
    }

  p = line;
  signum = len * 4;
  while (len-- > 0)
    {
      int digit;

      if (*p >= '0' && *p <= '9')
	digit = *p - '0';
      else if (*p >= 'a' && *p <= 'f')
	digit = *p - 'a' + 10;
      else
	{
	  fprintf (stderr, "Could not parse signal set: %s", line);
	  return -EINVAL;
	}

      signum -= 4;

      if (digit & 1)
	sigaddset (sigs, signum + 1);
      if (digit & 2)
	sigaddset (sigs, signum + 2);
      if (digit & 4)
	sigaddset (sigs, signum + 3);
      if (digit & 8)
	sigaddset (sigs, signum + 4);

      p++;
    }
  return 0;
}
#endif

/* Find process PID's pending signals from /proc/pid/status and set
   SIGS to match.  */
#ifndef __QNXHOST__
static int
linux_proc_pending_signals (long pid, sigset_t *pending, sigset_t *blocked,
			    sigset_t *ignored)
{
  FILE *procfile;
  int rc = 0;
  char buffer[MAXPATHLEN], fname[MAXPATHLEN];

  sigemptyset (pending);
  sigemptyset (blocked);
  sigemptyset (ignored);
  sprintf (fname, "/proc/%ld/status", pid);
  procfile = fopen (fname, "r");
  if (procfile == NULL)
    {
      fprintf (stderr, "Could not open %s: %s(%d)", fname, strerror (errno),
	       errno);
      return -errno;
    }

  while (fgets (buffer, MAXPATHLEN, procfile) != NULL)
    {
      /* Normal queued signals are on the SigPnd line in the status
	 file.  However, 2.6 kernels also have a "shared" pending
	 queue for delivering signals to a thread group, so check for
	 a ShdPnd line also.

	 Unfortunately some Red Hat kernels include the shared pending
	 queue but not the ShdPnd status field.  */

      if (strncmp (buffer, "SigPnd:\t", 8) == 0)
	rc = add_line_to_sigset (buffer + 8, pending);
      else if (strncmp (buffer, "ShdPnd:\t", 8) == 0)
	rc = add_line_to_sigset (buffer + 8, pending);
      else if (strncmp (buffer, "SigBlk:\t", 8) == 0)
	rc = add_line_to_sigset (buffer + 8, blocked);
      else if (strncmp (buffer, "SigIgn:\t", 8) == 0)
	rc = add_line_to_sigset (buffer + 8, ignored);
      if (rc)
	break;
    }

  fclose (procfile);
  return rc;
}
#endif

/* FIXME: support cuda_clear_pending_sigint on QNX */
#ifndef __QNXHOST__
static void
cuda_clear_pending_sigint (pid_t pid)
{
  if (ptrace (PTRACE_CONT, pid, 0, 0) != 0)
    {
      warning ("Failed to resume %d to clear pending SIGINT: %s", (int)pid,
	       strerror (errno));
      return;
    }

  int status = 0;
  pid_t waited;
  do
    waited = waitpid (pid, &status, 0);
  while (waited < 0 && errno == EINTR);
  if (waited < 0)
    {
      warning ("waitpid for %d failed: %s", (int)pid, strerror (errno));
      return;
    }

  gdb_assert (WIFSTOPPED (status) && WSTOPSIG (status) == SIGINT);
}
#endif

bool
cuda_check_pending_sigint (ptid_t ptid)
{
#ifndef __QNXHOST__
  int rc;
  pid_t pid = cuda_gdb_get_tid_or_pid (ptid);
  sigset_t pending, blocked, ignored;

  rc = linux_proc_pending_signals ((long)pid, &pending, &blocked, &ignored);
  if (rc)
    return false;

  if (sigismember (&pending, SIGINT))
    {
      cuda_clear_pending_sigint (pid);
      return true;
    }
#endif

  /* No pending SIGINT */
  return false;
}

/*---------------------------------------- Routines
 * ---------------------------------------*/

/* Debugger API statistics collection is disabled on cuda-gdbserver side */
bool
cuda_options_statistics_collection_enabled (void)
{
  return false;
}

void
cuda_enqueue_trace_message (const char *prefix, const char *fmt, va_list ap)
{
  static constexpr char truncated_marker[] = "[truncated]";
#ifdef __QNXHOST__
  constexpr size_t max_trace_msg_len
      = DS_DATA_MAX_SIZE - sizeof (truncated_marker);
#else
  constexpr size_t max_trace_msg_len = PBUFSIZ - sizeof (truncated_marker);
#endif

  std::string msg = prefix;
  msg += string_vprintf (fmt, ap);
  if (msg.size () > max_trace_msg_len)
    {
      msg.resize (max_trace_msg_len);
      msg += truncated_marker;
    }
  cuda_trace_messages.emplace_back (std::move (msg));
}

ATTRIBUTE_PRINTF (1, 2) void cuda_trace (const char *fmt, ...)
{
  if (!cuda_options_debug_general ())
    return;

  va_list ap;
  va_start (ap, fmt);
  cuda_enqueue_trace_message ("[CUDAGDB] ", fmt, ap);
  va_end (ap);
}

void
cuda_cleanup_trace_messages (void)
{
  cuda_trace_messages.clear ();
}

bool
cuda_options_launch_blocking (void)
{
  return cuda_launch_blocking;
}

bool
cuda_options_debug_general (void)
{
  return cuda_debug_general;
}

bool
cuda_options_debug_libcudbg (void)
{
  return cuda_debug_libcudbg;
}

bool
cuda_options_debug_notifications (void)
{
  return cuda_debug_notifications;
}

bool
cuda_options_notify_youngest (void)
{
  return cuda_notify_youngest;
}

bool
cuda_options_logs (void)
{
  return cuda_logs;
}

bool
cuda_options_printf_flushing (void)
{
  return cuda_printf_flushing;
}

bool
cuda_options_lazy_function_finalization_enabled (void)
{
  return cuda_lazy_function_finalization;
}

/* We use the gdb initializers for some of the CUDA sources we share between
 * gdb and gdbserver. See gdb/make-init-c for more info. There is no
 * equivalent concept for gdbserver today. We need to explicitly call the
 * intializers once per execution. */

void
cuda_gdb_setup (void)
{
  static bool cuda_called_initializers = false;
  extern void _initialize_cuda_notification ();
  extern void _initialize_cuda_utils ();
  if (!cuda_called_initializers)
    {
      cuda_called_initializers = true;
      _initialize_cuda_notification ();
      _initialize_cuda_utils ();
    }
}

void
cuda_cleanup (void)
{
  int i;
  cuda_trace ("cuda_cleanup");

  cuda_cleanup_completed = false;

  api_finalize_res = cudbgAPI->finalize ();
  /* Notification reset must be called after notification thread has
   * been terminated, which is done as part of cudbgAPI->finalize() call.
   */
  cuda_notification_reset ();
  inferior_in_debug_mode = false;
  cuda_initialized = false;
  for (i = 0; i < cuda_get_symbol_cache_size (); i++)
    cuda_symbol_list[i].addr = 0;
  cuda_syms_looked_up = false;

  cuda_cleanup_completed = true;
}

void
cuda_look_up_symbols (void)
{
  cuda_syms_looked_up = true;
  for (auto i = 0; i < cuda_get_symbol_cache_size (); i++)
    {
      if (look_up_one_symbol (cuda_symbol_list[i].name,
			      &(cuda_symbol_list[i].addr), 1)
	  == 0)
	cuda_syms_looked_up = false;
    }
}

int
cuda_get_symbol_cache_size (void)
{
  return sizeof (cuda_symbol_list) / sizeof (cuda_symbol_list[0]);
}

CORE_ADDR
cuda_get_symbol_address_from_cache (const char *name)
{
  int i;

  for (i = 0; i < cuda_get_symbol_cache_size (); i++)
    if (strcmp (cuda_symbol_list[i].name, name) == 0)
      return cuda_symbol_list[i].addr;

  return 0;
}

bool
cuda_inferior_in_debug_mode (void)
{
  return inferior_in_debug_mode;
}

int
cuda_get_debugger_api (void)
{
  gdb_assert (!cudbgAPI);

  get_debugger_api_res
      = cudbgGetAPI (CUDBG_API_VERSION_MAJOR, CUDBG_API_VERSION_MINOR,
		     CUDBG_API_VERSION_REVISION, &cudbgAPI);

  return (get_debugger_api_res != CUDBG_SUCCESS);
}

/* Initialize the CUDA debugger API and collect the static data about
   the devices. Once per application run. */
static void
cuda_initialize (void)
{
  if (cuda_initialized)
    return;

  set_callback_api_res
      = cudbgAPI->setNotifyNewEventCallback41 (cuda_notification_notify);
  api_initialize_res = cudbgAPI->initialize ();
  if (api_initialize_res != CUDBG_SUCCESS)
    {
      cuda_trace ("CUDA initialize failed with error result: %u\n",
		  (uint32_t)api_initialize_res);
      return;
    }
  /* Sucessfully initialized */
  cuda_cleanup_completed = false;
  cuda_exit_requested = false;
  cuda_initialized = true;
  /* Check to see if we are using UD */
  CORE_ADDR useExtDebuggerAddr = 0;
  uint32_t useExtDebugger = 0;
  useExtDebuggerAddr = cuda_get_symbol_address_from_cache (
      _STRING_ (CUDBG_USE_EXTERNAL_DEBUGGER));
  if (useExtDebuggerAddr)
    {
      target_read_memory (useExtDebuggerAddr, (gdb_byte *)&useExtDebugger,
			  sizeof (useExtDebugger));

      if (!useExtDebugger)
	printf ("Running on legacy stack.\n");
    }
}

/* Can be removed when exposed through cudadebugger.h */
#define CUDBG_INJECTION_PATH_SIZE 4096
static bool
cuda_initialize_injection (void)
{
  CORE_ADDR injectionPathAddr;
  char *injectionPathEnv;
  void *injectionLib;

  injectionPathEnv = getenv ("CUDBG_INJECTION_PATH");
  if (!injectionPathEnv)
    {
      /* No injection - cuda-gdb is the API client */
      return true;
    }

  if (strlen (injectionPathEnv) >= CUDBG_INJECTION_PATH_SIZE)
    {
      error (_ ("CUDBG_INJECTION_PATH must be no longer than %d: %s is %zd"),
	     CUDBG_INJECTION_PATH_SIZE - 1, injectionPathEnv,
	     strlen (injectionPathEnv));
      return false;
    }

  injectionLib = dlopen (injectionPathEnv, RTLD_LAZY);

  if (injectionLib == NULL)
    {
      error (_ ("Cannot open library %s pointed by CUDBG_INJECTION_PATH: %s"),
	     injectionPathEnv, dlerror ());
      return false;
    }

  dlclose (injectionLib);

  injectionPathAddr
      = cuda_get_symbol_address_from_cache (_STRING_ (cudbgInjectionPath));
  if (!injectionPathAddr)
    {
      error (_ ("No `cudbgInjectionPath` symbol in the CUDA driver"));
      return false;
    }

  /* This message should be removed once we finalize the way the alternative
   * API backend is injected */
  printf (
      "CUDBG_INJECTION_PATH is set, forwarding it to the target (value: %s)\n",
      injectionPathEnv);

  cuda_write_inferior_memory (
      injectionPathAddr,
      gdb::make_array_view (
	  reinterpret_cast<const gdb_byte *> (injectionPathEnv),
	  strlen (injectionPathEnv) + 1));

  return true;
}

/* Tell the target application that it is being
   CUDA-debugged. Inferior must have been launched first. */
bool
cuda_initialize_target (void)
{
  const gdb_byte zero = 0;
  const gdb_byte one = 1;
  CORE_ADDR debugFlagAddr;
  CORE_ADDR rpcFlagAddr;
  CORE_ADDR gdbPidAddr;
  CORE_ADDR apiClientRevAddr;
  CORE_ADDR sessionIdAddr;
  CORE_ADDR launchblockingAddr;

  uint32_t pid;
  uint32_t apiClientRev = CUDBG_API_VERSION_REVISION;
  uint32_t sessionId;

  cuda_initialize ();
  if (cuda_inferior_in_debug_mode ())
    return true;

  debugFlagAddr
      = cuda_get_symbol_address_from_cache (_STRING_ (CUDBG_IPC_FLAG_NAME));
  if (!debugFlagAddr)
    return true;

  /* cuda_initialize might modify sessionId */
  sessionId = cuda_gdb_session_get_id ();

#ifdef __QNXHOST__
  /* FIXME: current_process() is not available, assume the process has been
   * spawned */
  if (1)
#else
  if (!current_process ()->attached)
#endif
    {
      cuda_write_inferior_memory (debugFlagAddr,
				  gdb::make_array_view (&one, 1));
    }

  rpcFlagAddr
      = cuda_get_symbol_address_from_cache (_STRING_ (CUDBG_RPC_ENABLED));
  pid = getpid ();
  gdbPidAddr
      = cuda_get_symbol_address_from_cache (_STRING_ (CUDBG_APICLIENT_PID));
  apiClientRevAddr = cuda_get_symbol_address_from_cache (
      _STRING_ (CUDBG_APICLIENT_REVISION));
  sessionIdAddr
      = cuda_get_symbol_address_from_cache (_STRING_ (CUDBG_SESSION_ID));
  launchblockingAddr = cuda_get_symbol_address_from_cache (
      _STRING_ (CUDBG_ENABLE_LAUNCH_BLOCKING));

  if (!(rpcFlagAddr && gdbPidAddr && apiClientRevAddr && launchblockingAddr
	&& sessionIdAddr))
    return false;

  if (!cuda_initialize_injection ())
    {
      error (_ ("Failed to initialize the injection."));
    }

  cuda_write_inferior_memory (
      gdbPidAddr,
      gdb::make_array_view (reinterpret_cast<const gdb_byte *> (&pid),
			    sizeof (pid)));
  cuda_write_inferior_memory (rpcFlagAddr, gdb::make_array_view (&one, 1));
  cuda_write_inferior_memory (
      apiClientRevAddr,
      gdb::make_array_view (reinterpret_cast<const gdb_byte *> (&apiClientRev),
			    sizeof (apiClientRev)));
  cuda_write_inferior_memory (
      sessionIdAddr,
      gdb::make_array_view (reinterpret_cast<const gdb_byte *> (&sessionId),
			    sizeof (sessionId)));
  cuda_write_inferior_memory (
      launchblockingAddr,
      gdb::make_array_view (cuda_options_launch_blocking () ? &one : &zero,
			    1));

  /* Setup our desired debugger capabilities for the debugger backend. It is
   * alright if the older driver doesn't understand some of these flags. We
   * will deal with those situations after initialization. */
  CORE_ADDR capability_addr = cuda_get_symbol_address_from_cache (
      _STRING_ (CUDBG_DEBUGGER_CAPABILITIES));
  if (capability_addr)
    {
      cuda_trace ("requesting CUDA lazy function loading support\n");
      debugger_capabilities |= CUDBG_DEBUGGER_CAPABILITY_LAZY_FUNCTION_LOADING;

      if (cuda_options_lazy_function_finalization_enabled ())
	{
	  cuda_trace ("requesting CUDA lazy function finalization support");
	  debugger_capabilities
	      |= CUDBG_DEBUGGER_CAPABILITY_ENABLE_CUDA_LAZY_FUNCTION_FINALIZATION;
	}

      cuda_trace ("requesting tracking of exceptions in exited warps\n");
      debugger_capabilities
	  |= CUDBG_DEBUGGER_CAPABILITY_REPORT_EXCEPTIONS_IN_EXITED_WARPS;

      cuda_trace ("requesting CUDA suspend events\n");
      debugger_capabilities |= CUDBG_DEBUGGER_CAPABILITY_SUSPEND_EVENTS;

      cuda_trace ("requesting no context push / pop events be delivered\n");
      debugger_capabilities
	  |= CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS;

      if (cuda_options_logs ())
	{
	  cuda_trace ("requesting CUDA logs\n");
	  debugger_capabilities |= CUDBG_DEBUGGER_CAPABILITY_ENABLE_CUDA_LOGS;
	}

      if (cuda_options_printf_flushing ())
	{
	  cuda_trace ("requesting CUDA printf flushing on suspend\n");
	  debugger_capabilities
	      |= CUDBG_DEBUGGER_CAPABILITY_FLUSH_PRINTF_ON_SUSPEND;
	}

#if CUDBG_API_VERSION_REVISION > 167
      cuda_trace ("requesting break-on-launch capability\n");
      debugger_capabilities |= CUDBG_DEBUGGER_CAPABILITY_BREAK_ON_LAUNCH;
#endif

      cuda_write_inferior_memory (
	  capability_addr,
	  gdb::make_array_view (
	      reinterpret_cast<const gdb_byte *> (&debugger_capabilities),
	      sizeof (debugger_capabilities)));
    }

  inferior_in_debug_mode = true;
  return true;
}

/* Set the CUDA logs capability */
void
cuda_set_logs (bool enable)
{
  CORE_ADDR capability_addr = cuda_get_symbol_address_from_cache (
      _STRING_ (CUDBG_DEBUGGER_CAPABILITIES));
  if (capability_addr)
    {
#ifndef __QNXHOST__
      /* Read the current capabilities from the target process memory */
      /* target_read_memory is not supported on QNX.
       * TODO: Implement target_read_memory for QNX and then remove the global
       */
      target_read_memory (capability_addr, (gdb_byte *)&debugger_capabilities,
			  sizeof (debugger_capabilities));
#endif
      cuda_trace ("Current CUDA debugger capabilities: 0x%x\n",
		  debugger_capabilities);

      /* Set or clear the CUDA logs capability */
      cuda_trace ("%s CUDA logs\n", enable ? "enabling" : "disabling");

      if (enable)
	debugger_capabilities |= CUDBG_DEBUGGER_CAPABILITY_ENABLE_CUDA_LOGS;
      else
	debugger_capabilities &= ~CUDBG_DEBUGGER_CAPABILITY_ENABLE_CUDA_LOGS;
      /* Write the new capabilities back to the target */
      cuda_trace ("setting CUDA debugger capabilities to 0x%x\n",
		  debugger_capabilities);
      cuda_write_inferior_memory (
	  capability_addr,
	  gdb::make_array_view (
	      reinterpret_cast<const gdb_byte *> (&debugger_capabilities),
	      sizeof (debugger_capabilities)));
    }
}

/********* Session Management **********/

int
cuda_gdb_session_create (void)
{
  int ret = 0;
  bool override_umask = false;
  bool dir_exists = false;

  /* Check if the previous session was cleaned up */
  if (cuda_gdb_session_dir[0] != '\0')
    error ("The directory for the previous CUDA session was not cleaned up. "
	   "Try deleting %s and retrying.",
	   cuda_gdb_session_dir);

  cuda_gdb_session_id++;

  snprintf (cuda_gdb_session_dir, CUDA_GDB_TMP_BUF_SIZE, "%s/session%d",
	    cuda_gdb_tmpdir_getdir (), cuda_gdb_session_id);

  cuda_trace ("creating new session %d", cuda_gdb_session_id);

  ret = cuda_gdb_dir_create (cuda_gdb_session_dir, S_IRWXU | S_IRWXG,
			     override_umask, &dir_exists);

  if (!ret && dir_exists)
    error ("A stale CUDA session directory was found. "
	   "Try deleting %s and retrying.",
	   cuda_gdb_session_dir);
  else if (ret)
    error ("Failed to create session directory: %s (ret=%d).",
	   cuda_gdb_session_dir, ret);

  return ret;
}

void
cuda_gdb_session_destroy (void)
{

#ifndef __QNXHOST__
  cuda_gdb_dir_cleanup_files (cuda_gdb_session_dir);
  rmdir (cuda_gdb_session_dir);
#else
  cuda_gdb_tmpdir_cleanup_dir (cuda_gdb_session_dir);
#endif /* __QNXHOST__ */

  memset (cuda_gdb_session_dir, 0, CUDA_GDB_TMP_BUF_SIZE);
}

uint32_t
cuda_gdb_session_get_id (void)
{
  return cuda_gdb_session_id;
}

const char *
cuda_gdb_session_get_dir (void)
{
  return cuda_gdb_session_dir;
}
