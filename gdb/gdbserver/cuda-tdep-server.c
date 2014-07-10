/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2013 NVIDIA Corporation
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

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ptrace.h>
#ifndef __ANDROID__
#include <sys/signal.h>
#endif
#include <sys/wait.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "cuda-tdep-server.h"
#include "../cuda-notifications.h"

/*----------------------------------------- Globals ---------------------------------------*/
struct cuda_sym cuda_symbol_list[] =
{
  CUDA_SYM(CUDBG_IPC_FLAG_NAME),
  CUDA_SYM(CUDBG_RPC_ENABLED),
  CUDA_SYM(CUDBG_APICLIENT_PID),
  CUDA_SYM(CUDBG_APICLIENT_REVISION),
  CUDA_SYM(CUDBG_SESSION_ID),
  CUDA_SYM(CUDBG_ATTACH_HANDLER_AVAILABLE),
  CUDA_SYM(CUDBG_DEBUGGER_INITIALIZED),
  CUDA_SYM(CUDBG_REPORTED_DRIVER_API_ERROR_CODE),
  CUDA_SYM(CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE),
  CUDA_SYM(CUDBG_DETACH_SUSPENDED_DEVICES_MASK),
  CUDA_SYM(CUDBG_ENABLE_INTEGRATED_MEMCHECK),
  CUDA_SYM(CUDBG_ENABLE_LAUNCH_BLOCKING),
  CUDA_SYM(CUDBG_ENABLE_PREEMPTION_DEBUGGING),
};

CUDBGAPI cudbgAPI = NULL;
bool cuda_initialized = false;
static bool inferior_in_debug_mode = false;
static char cuda_gdb_session_dir[CUDA_GDB_TMP_BUF_SIZE] = {0};
static uint32_t cuda_gdb_session_id = 0;

bool cuda_launch_blocking;
bool cuda_memcheck;
bool cuda_software_preemption;
bool cuda_debug_general;
bool cuda_debug_libcudbg;
bool cuda_debug_notifications;
bool cuda_notify_youngest;
struct cuda_trace_msg *cuda_first_trace_msg = NULL;
struct cuda_trace_msg *cuda_last_trace_msg = NULL;


/* For Mac OS X */
bool cuda_platform_supports_tid (void)
{
#if defined(linux) && defined(SYS_gettid)
    return true;
#else
    return false;
#endif
}

int
cuda_gdb_get_tid (ptid_t ptid)
{
  if (cuda_platform_supports_tid ())
    return ptid_get_lwp (ptid);
  else
    return ptid_get_pid (ptid);
}

/*---------------------------------------- sigpending -------------------------------------*/
#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

/* Parse LINE as a signal set and add its set bits to SIGS.  */

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

/* Find process PID's pending signals from /proc/pid/status and set
   SIGS to match.  */

static int
linux_proc_pending_signals (long pid, sigset_t *pending, sigset_t *blocked, sigset_t *ignored)
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
      fprintf (stderr, "Could not open %s: %s(%d)", fname, strerror(errno), errno);
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
        rc =add_line_to_sigset (buffer + 8, pending);
      else if (strncmp (buffer, "SigBlk:\t", 8) == 0)
        rc = add_line_to_sigset (buffer + 8, blocked);
      else if (strncmp (buffer, "SigIgn:\t", 8) == 0)
        rc = add_line_to_sigset (buffer + 8, ignored);
      if (rc) break;
    }

  fclose (procfile);
  return rc;
}

static void
cuda_clear_pending_sigint (pid_t pid)
{
  int status = 0, options = 0;
  ptrace (PTRACE_CONT, pid, 0, 0); /* Resume the host to consume the pending SIGINT */
  waitpid (pid, &status, options); /* Ensure we return for the right reason */
  gdb_assert (WIFSTOPPED (status) && WSTOPSIG (status) == SIGINT);
}

bool
cuda_check_pending_sigint (ptid_t ptid)
{
  int rc;
  pid_t pid = cuda_gdb_get_tid (ptid);
  sigset_t pending, blocked, ignored;

  rc = linux_proc_pending_signals ((long)pid, &pending, &blocked, &ignored);
  if (rc)
    return false;

  if (sigismember (&pending, SIGINT))
    {
      cuda_clear_pending_sigint (pid);
      return true;
    }

  /* No pending SIGINT */
  return false;
}

/*---------------------------------------- Routines ---------------------------------------*/

ATTRIBUTE_PRINTF(1, 2) void
cuda_trace (char *fmt, ...)
{
  struct cuda_trace_msg *msg;
  va_list ap;

  if (!cuda_options_debug_general())
    return;

  va_start (ap, fmt);
  msg = xmalloc (sizeof (*msg));
  if (!cuda_first_trace_msg)
    cuda_first_trace_msg = msg;
  else
    cuda_last_trace_msg->next = msg;
  sprintf (msg->buf, "[CUDAGDB] ");
  vsnprintf (msg->buf + strlen (msg->buf), sizeof (msg->buf), fmt, ap);
  msg->next = NULL;
  cuda_last_trace_msg = msg;
}

void
cuda_cleanup_trace_messages ()
{
  struct cuda_trace_msg *msg;
  if (!cuda_first_trace_msg)
    return;

  while (cuda_first_trace_msg)
    {
       msg = cuda_first_trace_msg;
       cuda_first_trace_msg = cuda_first_trace_msg->next;
       xfree (msg);
    }
}

bool
cuda_options_memcheck ()
{
  return cuda_memcheck;
}

bool
cuda_options_launch_blocking ()
{
  return cuda_launch_blocking;
}

bool
cuda_options_software_preemption ()
{
  return cuda_software_preemption;
}

bool
cuda_options_debug_general ()
{
  return cuda_debug_general;
}

bool
cuda_options_debug_libcudbg ()
{
  return cuda_debug_libcudbg;
}

bool
cuda_options_debug_notifications ()
{
  return cuda_debug_notifications;
}

bool
cuda_options_notify_youngest ()
{
  return cuda_notify_youngest;
}

void
cuda_cleanup ()
{
  int i;
  cuda_trace ("cuda_cleanup");

  api_finalize_res = cudbgAPI->finalize ();
  /* Notification reset must be called after notification thread has
   * been terminated, which is done as part of cudbgAPI->finalize() call.
   */
  cuda_notification_reset ();
  inferior_in_debug_mode = false;
  cuda_initialized = false;
  all_cuda_symbols_looked_up = 0;
  for (i = 0; i < cuda_get_symbol_cache_size (); i++)
    cuda_symbol_list[i].addr = 0;
}

int
cuda_get_symbol_cache_size ()
{
  return sizeof (cuda_symbol_list) / sizeof (cuda_symbol_list[0]);
}

CORE_ADDR
cuda_get_symbol_address_from_cache (char* name)
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
cuda_get_debugger_api ()
{
  gdb_assert (!cudbgAPI);

  get_debugger_api_res = cudbgGetAPI (CUDBG_API_VERSION_MAJOR,
                                      CUDBG_API_VERSION_MINOR,
                                      CUDBG_API_VERSION_REVISION,
                                      &cudbgAPI);

  return (get_debugger_api_res != CUDBG_SUCCESS);
}

/* Initialize the CUDA debugger API and collect the static data about
   the devices. Once per application run. */
static void
cuda_initialize ()
{
  if (cuda_initialized)
    return;

  set_callback_api_res = cudbgAPI->setNotifyNewEventCallback (cuda_notification_notify);
  api_initialize_res = cudbgAPI->initialize ();
  if (api_initialize_res == CUDBG_SUCCESS ||
      api_initialize_res == CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED)
    cuda_initialized = true;
}


/* Tell the target application that it is being
   CUDA-debugged. Inferior must have been launched first. */
bool
cuda_initialize_target ()
{
  const unsigned char zero = 0;
  const unsigned char one = 1;
  CORE_ADDR debugFlagAddr;
  CORE_ADDR rpcFlagAddr;
  CORE_ADDR gdbPidAddr;
  CORE_ADDR apiClientRevAddr;
  CORE_ADDR sessionIdAddr;
  CORE_ADDR memcheckAddr;
  CORE_ADDR launchblockingAddr;
  CORE_ADDR preemptionAddr;


  uint32_t pid;
  uint32_t apiClientRev = CUDBG_API_VERSION_REVISION;
  uint32_t sessionId = cuda_gdb_session_get_id ();


  cuda_initialize ();
  if (cuda_inferior_in_debug_mode())
    return true;

  debugFlagAddr = cuda_get_symbol_address_from_cache (_STRING_(CUDBG_IPC_FLAG_NAME));
  if (!debugFlagAddr)
    return true;

  if (!current_process()->attached)
    write_inferior_memory (debugFlagAddr, &one, 1);
  rpcFlagAddr = cuda_get_symbol_address_from_cache (_STRING_(CUDBG_RPC_ENABLED));
  pid = getpid ();
  gdbPidAddr = cuda_get_symbol_address_from_cache (_STRING_(CUDBG_APICLIENT_PID));
  apiClientRevAddr = cuda_get_symbol_address_from_cache (_STRING_(CUDBG_APICLIENT_REVISION));
  sessionIdAddr = cuda_get_symbol_address_from_cache (_STRING_(CUDBG_SESSION_ID));
  memcheckAddr = cuda_get_symbol_address_from_cache (_STRING_(CUDBG_ENABLE_INTEGRATED_MEMCHECK));
  launchblockingAddr = cuda_get_symbol_address_from_cache (_STRING_(CUDBG_ENABLE_LAUNCH_BLOCKING));
  preemptionAddr = cuda_get_symbol_address_from_cache (_STRING_(CUDBG_ENABLE_PREEMPTION_DEBUGGING));

  if (!(rpcFlagAddr && gdbPidAddr && apiClientRevAddr &&
      memcheckAddr && launchblockingAddr && sessionIdAddr &&
      preemptionAddr))
    return false;

  write_inferior_memory (gdbPidAddr, (unsigned char*)&pid, sizeof (pid));
  write_inferior_memory (rpcFlagAddr, &one, 1);
  write_inferior_memory (apiClientRevAddr, (unsigned char*)&apiClientRev, sizeof(apiClientRev));
  write_inferior_memory (sessionIdAddr, (unsigned char*)&sessionId, sizeof(sessionId));
  write_inferior_memory (preemptionAddr, cuda_options_software_preemption () ? &one : &zero, 1);
  write_inferior_memory (memcheckAddr, cuda_options_memcheck () ? &one : &zero, 1);
  write_inferior_memory (launchblockingAddr, cuda_options_launch_blocking () ? &one : &zero, 1);
  inferior_in_debug_mode = true;
  return true;
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
             "Try deleting %s and retrying.", cuda_gdb_session_dir);

  cuda_gdb_session_id++;

  snprintf (cuda_gdb_session_dir, CUDA_GDB_TMP_BUF_SIZE,
            "%s/session%d", cuda_gdb_tmpdir_getdir (),
            cuda_gdb_session_id);

  cuda_trace ("creating new session %d", cuda_gdb_session_id);

  ret = cuda_gdb_dir_create (cuda_gdb_session_dir, S_IRWXU | S_IRWXG,
                             override_umask, &dir_exists);

  if (!ret && dir_exists)
    error ("A stale CUDA session directory was found. "
             "Try deleting %s and retrying.", cuda_gdb_session_dir);
  else if (ret)
    error ("Failed to create session directory: %s (ret=%d).", cuda_gdb_session_dir, ret);

  return ret;
}

void
cuda_gdb_session_destroy (void)
{
  cuda_gdb_dir_cleanup_files (cuda_gdb_session_dir);

  rmdir (cuda_gdb_session_dir);

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
