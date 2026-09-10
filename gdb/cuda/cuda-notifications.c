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

/*--------------------- Notifications ---------------------
 *
 * A notification is send by the CUDA debugger API (the producer or sender) and
 * handled by GDB (the consumer or recipient). Three booleans are used to mark
 * the current of CUDA notifications: pending_send, sent, and received.
 *
 * From the producer's point of view, the notification framework can be in 3
 * states only: ready, pending, and sent. When ready, there has been
 * notification. When pending, a notification was tentatively sent but got
 * postponed because the notification mechanism was 'blocked'. When sent, a
 * notification was sent as a stop signal. Those 3 producer states are
 * implemeted as:
 *
 *         ready   == !sent && !pending
 *         pending == !sent &&  pending
 *         sent    ==  sent && !pending
 *        (illegal ==  sent &&  pending)
 *
 * From the consumer's point of view, the notification framework can be in 3
 * states as well: none, received, and pending. When none, there is no
 * notification to process. When received, a notification is ready to be
 * processed associated with host thread GDB woke up upon and the stop signal
 * that was sent has been consumed. When pending, a notification has been sent
 * but not to the host thread GDB woke up upon, and the stop signal that was
 * sent has not been consumed yet. Those 3 consumer states are implemented as:
 *
 *          none     == !sent && !received
 *          received ==  sent &&  received
 *          pending  ==  sent && !received
 *         (illegal) == !sent &&  received)
 *
 * Two extra booleans are used: 'initialized' to remember when
 * cuda_notification_info has already been initialized, and 'blocked'. When
 * 'blocked', a notification cannot be sent, and will be marked as (producer)
 * pending if no notification has been sent yet. The notification will be then
 * sent later, when notifications become unblocked, and the notification will
 * go from (producer) pending state to (producer) sent state. Additionally, if
 * a notification is received before a previous event has been serviced, it is
 * marked as an aliased_event, and an attempt is made to service it before the
 * inferior is resumed. No new stop signal is sent for an aliased_event.
 */

#ifdef GDBSERVER
#include "cuda/cuda-tdep-server.h"
#include "server.h"
#else
#include "defs.h"

#include "cuda-api.h"
#include "cuda-options.h"
#include "cuda-packet-manager.h"
#include "cuda-tdep.h"
#include "gdbthread.h"
#include "inferior.h"
#include "remote.h"
#endif

#include "cuda-notifications.h"
#include "cuda-stats.h"

#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#ifndef __QNXHOST__
#include <sys/syscall.h>
#endif

cuda_statistic &
get_cuda_notification_statistics (void)
{
  static cuda_statistic notification_statistics;
  return notification_statistics;
}

static struct
{
  bool initialized; /* True if the mutex is initialized */
  bool blocked; /* When blocked, stop signal will be marked pending and handled
		   later. */
  bool pending_send;  /* True if a stop signal was received while blocked was
			 true. */
  bool aliased_event; /* True if a stop signal was received while a previous
			 event was being processed. */
  bool sent;	      /* If already sent, do not send duplicates. */
  bool received;      /* True if the stop signal has been received. */
  uint32_t tid; /* Host notification target id. This is a thread id on most
		   platforms and the inferior pid on QNX. */
  pthread_mutex_t mutex; /* Mutex for the cuda_notification_* functions */
  CUDBGEventCallbackData41 pending_send_data;
} cuda_notification_info;

/* Per-wait-epoch flag.  Only touched by the wait/resume thread, so no mutex.
   Lives outside cuda_notification_info because it isn't part of the SIGURG
   protocol -- it's a state-passing channel between cuda_wait's drain and
   cuda_resume's aliased-event discriminator.  */
static bool cuda_notification_suspend_drained_state = false;

#if __QNXHOST__
extern uint32_t inferior_pid;
static int cuda_notification_notify_specific_thread (uint32_t tid);

static uint32_t
cuda_notification_qnx_target_pid (void)
{
  return inferior_pid;
}

static uint32_t
cuda_notification_qnx_send_process_notification (CUDBGEventCallbackData41 *data)
{
  const uint32_t target_pid = cuda_notification_qnx_target_pid ();
  const uint32_t callback_tid = data ? data->tid : 0;
  const uint32_t timeout = data ? data->timeout : 0;

  cuda_trace ("qnx notify: process-pid send callback_tid=%u timeout=%u "
	      "target_pid=%u",
	      callback_tid, timeout, target_pid);

  const bool sent
      = (cuda_notification_notify_specific_thread (target_pid) == 0);
  const uint32_t sent_to = sent ? target_pid : 0;
  cuda_trace ("qnx notify: process-pid send result target_pid=%u sent_to=%u",
	      target_pid, sent_to);

  return sent_to;
}

static uint32_t
cuda_notification_qnx_stop_target_pid (ptid_t ptid)
{
  return ptid.pid ();
}

static int
cuda_notification_signal_for_trace (const struct target_waitstatus *ws)
{
  if (ws->kind () == TARGET_WAITKIND_STOPPED
      || ws->kind () == TARGET_WAITKIND_SIGNALLED)
    return ws->sig ();

  return GDB_SIGNAL_0;
}

static bool
cuda_notification_qnx_match_received (ptid_t ptid,
				      const struct target_waitstatus *ws)
{
  return (cuda_notification_info.sent
	  && cuda_notification_info.tid
		 == cuda_notification_qnx_stop_target_pid (ptid)
	  && ws->kind () == TARGET_WAITKIND_STOPPED
	  && (ws->sig () == GDB_SIGNAL_EMT
	      || ws->sig () == GDB_SIGNAL_ILL));
}

static bool
cuda_notification_qnx_signal_stop (const struct target_waitstatus *ws)
{
  return (ws->kind () == TARGET_WAITKIND_STOPPED
	  && (ws->sig () == GDB_SIGNAL_EMT
	      || ws->sig () == GDB_SIGNAL_ILL));
}

static void
cuda_notification_qnx_trace_analyze (ptid_t ptid,
				     const struct target_waitstatus *ws,
				     bool matched)
{
  cuda_trace ("qnx notify: analyze notify_pid=%u stop_pid=%d stop_lwp=%ld "
	      "ws_kind=%d ws_sig=%d sent=%d received=%d match=%d",
	      cuda_notification_info.tid, ptid.pid (), (long) ptid.lwp (),
	      (int) ws->kind (), cuda_notification_signal_for_trace (ws),
	      cuda_notification_info.sent, cuda_notification_info.received,
	      matched);
}
#endif

static void
cuda_notification_trace (const char *fmt, ...)
{
  va_list ap;

  if (!cuda_options_debug_notifications ())
    return;

  va_start (ap, fmt);
#ifdef GDBSERVER
  cuda_enqueue_trace_message ("[CUDAGDB] notifications -- ", fmt, ap);
#else
  fprintf (stderr, "[CUDAGDB] notifications -- ");
  vfprintf (stderr, fmt, ap);
  fprintf (stderr, "\n");
  fflush (stderr);
#endif
  va_end (ap);
}

void
cuda_notification_reset (void)
{
  gdb_assert (cuda_notification_info.initialized);
  cuda_notification_info.blocked = false;
  cuda_notification_info.pending_send = false;
  cuda_notification_info.sent = false;
  cuda_notification_info.received = false;
  cuda_notification_info.tid = false;
  cuda_notification_suspend_drained_state = false;
}

static void
cuda_notification_acquire_lock (void)
{
  gdb_assert (cuda_notification_info.initialized);
  pthread_mutex_lock (&cuda_notification_info.mutex);
}

static void
cuda_notification_release_lock (void)
{
  gdb_assert (cuda_notification_info.initialized);
  pthread_mutex_unlock (&cuda_notification_info.mutex);
}

static int
cuda_notification_notify_thread (int tid)
{
  /* Start the global timer in cuda_notification_notify_thread - it will be
   * stopped in cuda_process_event */
  get_cuda_notification_statistics ().start_timing ();

  unsigned signal;
#ifdef __QNXHOST__
  static unsigned sig = 0;
  static unsigned signals[2] = { GDB_SIGNAL_EMT, GDB_SIGNAL_ILL };

  /* On QNX sending the same signal twice in a row triggers the default handler
     (which is often program exit). Avoid this behavior by alternating between
     the two signals.

     For details see bug 1986383. */
  signal = signals[sig++ % 2];
#else
  signal = SIGURG;
#endif
#if defined(__linux__) && !defined(__aarch64__)
  {
    static int tkill_failed;

    if (!tkill_failed)
      {
	int ret;

	errno = 0;
	ret = syscall (__NR_tkill, tid, signal);
	if (errno != ENOSYS)
	  return ret;
	tkill_failed = 1;
      }
  }
#endif

  // On aarch64, tid is really the pid
  return kill (tid, signal);
}

static int
cuda_notification_notify_specific_thread (uint32_t tid)
{
  int err = 1;

  err = cuda_notification_notify_thread (tid);

  cuda_notification_trace (
      "sent specifically to the given host thread: tid %d -> %s", tid,
      err ? "FAILED" : "success");

  return err;
}

#if !__QNXHOST__

#ifdef GDBSERVER
static int
find_and_notify_first_valid_thread (process_info *tp)
#else
static int
find_and_notify_first_valid_thread (struct thread_info *tp, void *data)
#endif
{
  int err, tid;

#ifdef GDBSERVER
  tid = tp->pid;
#else
  tid = cuda_gdb_get_tid_or_pid (tp->ptid);
#endif

  err = cuda_notification_notify_thread (tid);

  return err == 0;
}

static int
cmp_thread_tid (const void *tid1, const void *tid2)
{
  return ((*(int *)tid1) > (*(int *)tid2));
}

#define MAX_YOUNG_THREADS 128
typedef struct
{
  int num;
  int tid[MAX_YOUNG_THREADS];
} youngest_threads_t;

#ifdef GDBSERVER
static int
build_threads (process_info *tp, void *data)
#else
static int
build_threads (struct thread_info *tp, void *data)
#endif
{
  int tid;
  youngest_threads_t *youngest_threads = (youngest_threads_t *)data;

#ifdef GDBSERVER
  tid = tp->pid;
#else
  tid = cuda_gdb_get_tid_or_pid (tp->ptid);
#endif

  if (youngest_threads->num >= MAX_YOUNG_THREADS)
    return 1;

  youngest_threads->tid[youngest_threads->num] = tid;
  youngest_threads->num++;

  return 0;
}

static uint32_t
cuda_notification_notify_youngest_thread (void)
{
  int err = 1, i = 0, tid = 0;
  youngest_threads_t youngest_threads;

  cuda_notification_trace ("sending to the youngest valid thread");

  youngest_threads.num = 0;

#ifdef GDBSERVER
  for_each_process ([&youngest_threads] (process_info *process) {
    build_threads (process, &youngest_threads);
  });
#else
  iterate_over_threads (build_threads, &youngest_threads);
#endif

  qsort (youngest_threads.tid, youngest_threads.num,
	 sizeof *youngest_threads.tid, cmp_thread_tid);

  for (i = 0; err && i < youngest_threads.num; ++i)
    {
      tid = youngest_threads.tid[i];
      err = cuda_notification_notify_specific_thread (youngest_threads.tid[i]);
    }

  return err ? 0 : tid;
}

static uint32_t
cuda_notification_notify_first_valid_thread (void)
{
  uint32_t tid;

#ifdef GDBSERVER
  process_info *tp = find_process ([] (process_info *process) {
    return find_and_notify_first_valid_thread (process);
  });
  tid = tp ? tp->pid : 0;
#else
  struct thread_info *tp;
  tp = iterate_over_threads (find_and_notify_first_valid_thread, NULL);
  tid = tp ? cuda_gdb_get_tid_or_pid (tp->ptid) : 0;
#endif

  cuda_notification_trace ("sent to the first valid thread: tid %ld -> %s",
			   (long)tid, tid ? "success" : "FAILED");

  return tid;
}
#endif /* !__QNX_HOST__ */

static void
cuda_notification_send (CUDBGEventCallbackData41 *data)
{
  uint32_t tid = 0;
#ifndef __QNXHOST__
  int err = 1;
#endif

#ifndef __QNXHOST__
  // use the host thread id if given to us
  if (!tid && cuda_platform_supports_tid () && data && data->tid)
    {
      err = cuda_notification_notify_specific_thread (data->tid);
      if (!err)
	tid = data->tid;
    }
#endif

#ifdef __QNXHOST__
  /* Keep QNX notification delivery process-pid based while tracing callback
     tids for later comparison against stop lwps. */
  if (!tid)
    tid = cuda_notification_qnx_send_process_notification (data);
#else
#ifndef GDBSERVER
  // use the saved ptid used to init the debug API
  int api_ptid = cuda_debugapi::get_api_ptid ();
  if (!tid && api_ptid)
    {
      err = cuda_notification_notify_specific_thread (api_ptid);
      if (!err)
	tid = api_ptid;
    }
#endif
  // use the youngest thread if possible
  if (!tid && cuda_options_notify_youngest ())
    tid = cuda_notification_notify_youngest_thread ();

  // otherwise, use any valid host thread to send the notification to.
  if (!tid)
    tid = cuda_notification_notify_first_valid_thread ();
#endif

  if (tid)
    {
      cuda_notification_info.tid = tid;
      cuda_notification_info.sent = true;
      return;
    }
}

void
cuda_notification_accept (void)
{
  cuda_notification_acquire_lock ();

  cuda_notification_info.blocked = false;

  if (cuda_notification_info.pending_send)
    {
      cuda_notification_trace ("accept: sending pending notification");
      cuda_notification_send (&cuda_notification_info.pending_send_data);
      cuda_notification_info.pending_send = false;
      memset (&cuda_notification_info.pending_send_data, 0,
	      sizeof cuda_notification_info.pending_send_data);
    }

  cuda_notification_release_lock ();
}

void
cuda_notification_block (void)
{
  cuda_notification_acquire_lock ();

  cuda_notification_info.blocked = true;

  cuda_notification_release_lock ();
}

void
cuda_notification_notify (CUDBGEventCallbackData41 *data)
{
  cuda_notification_acquire_lock ();

  if (data->timeout)
    {
      /* Was there a timeout waiting for a response? */
      if (cuda_notification_info.sent && !cuda_notification_info.received)
	{
	  cuda_notification_trace ("timeout: resending notification");
	  cuda_notification_send (data);
	}
    }
  else if (cuda_notification_info.sent)
    {
      cuda_notification_trace ("aliased event: will examine before resuming");
      cuda_notification_info.aliased_event = true;
    }
  else if (cuda_notification_info.pending_send)
    cuda_notification_trace (
	"ignoring: another notification is already pending");
  else if (cuda_notification_info.blocked)
    {
      cuda_notification_trace (
	  "blocked: marking notification as pending_send");
      cuda_notification_info.pending_send = true;
      cuda_notification_info.pending_send_data = *data;
    }
  else
    cuda_notification_send (data);

  cuda_notification_release_lock ();
}

bool
cuda_notification_aliased_event (void)
{
  bool aliased_event;

#ifndef GDBSERVER
  if (is_remote_target (current_inferior ()->process_target ()))
    cuda_remote_notification_aliased_event ();
#endif

  cuda_notification_acquire_lock ();

  aliased_event = cuda_notification_info.aliased_event;

  cuda_notification_release_lock ();

  return aliased_event;
}

void
cuda_notification_reset_aliased_event (void)
{
  cuda_notification_acquire_lock ();

  cuda_notification_info.aliased_event = false;

  cuda_notification_release_lock ();
}

bool
cuda_notification_pending (void)
{
  bool pending;

#ifndef GDBSERVER
  if (is_remote_target (current_inferior ()->process_target ()))
    return cuda_remote_notification_pending ();
#endif

  cuda_notification_acquire_lock ();

  pending = cuda_notification_info.sent && !cuda_notification_info.received;

  cuda_notification_release_lock ();

  return pending;
}

bool
cuda_notification_received (void)
{
  bool received;

#ifndef GDBSERVER
  if (is_remote_target (current_inferior ()->process_target ()))
    return cuda_remote_notification_received ();
#endif

  cuda_notification_acquire_lock ();

  received = cuda_notification_info.received;

  cuda_notification_release_lock ();

  return received;
}

void
cuda_notification_set_suspend_drained (void)
{
  cuda_notification_suspend_drained_state = true;
}

void
cuda_notification_clear_suspend_drained (void)
{
  cuda_notification_suspend_drained_state = false;
}

bool
cuda_notification_suspend_drained (void)
{
  return cuda_notification_suspend_drained_state;
}

void
cuda_notification_analyze (ptid_t ptid, struct target_waitstatus *ws)
{
#ifndef GDBSERVER
  if (is_remote_target (current_inferior ()->process_target ()))
    {
      cuda_remote_notification_analyze (ptid, ws);
      return;
    }
#endif

  cuda_notification_acquire_lock ();

  /* A notification is deemed received when its corresponding signal is the
     reason we stopped.  Only match the actual notification signal (SIGURG
     on Linux, alternating EMT/ILL on QNX).  SIGTRAP must NOT be matched
     here.  Matching SIGTRAP caused real breakpoint hits (e.g., bp_cuda_api_error) 
     to be misidentified as notifications and silently consumed.  */
#ifdef __QNXHOST__
  bool matched = cuda_notification_qnx_match_received (ptid, ws);

  if (cuda_notification_info.sent || cuda_notification_qnx_signal_stop (ws))
    cuda_notification_qnx_trace_analyze (ptid, ws, matched);

  if (matched)
    {
      cuda_notification_trace ("received notification to thread %d",
			       cuda_notification_info.tid);
      cuda_notification_info.received = true;
    }
#else
  if (cuda_notification_info.sent
      && cuda_notification_info.tid == cuda_gdb_get_tid_or_pid (ptid)
      && ws->kind () == TARGET_WAITKIND_STOPPED
      && (ws->sig () == GDB_SIGNAL_URG
#ifdef __QNXHOST__
	  || ws->sig () == GDB_SIGNAL_EMT || ws->sig () == GDB_SIGNAL_ILL
#endif
	  ))
    {
      cuda_notification_trace ("received notification to thread %d",
			       cuda_notification_info.tid);
      cuda_notification_info.received = true;
    }
#endif

  cuda_notification_release_lock ();
}

void
cuda_notification_mark_consumed (void)
{
#ifndef GDBSERVER
  if (is_remote_target (current_inferior ()->process_target ()))
    {
      cuda_remote_notification_mark_consumed ();
      return;
    }
#endif

  cuda_notification_acquire_lock ();

  if (cuda_notification_info.received)
    {
      cuda_notification_trace ("consuming notification to thread %d",
			       cuda_notification_info.tid);
      cuda_notification_info.sent = false;
      cuda_notification_info.received = false;
      cuda_notification_info.tid = 0;
    }

  cuda_notification_release_lock ();
}

void
cuda_notification_consume_pending (void)
{
#ifndef GDBSERVER
  if (is_remote_target (current_inferior ()->process_target ()))
    {
      cuda_remote_notification_consume_pending ();
      return;
    }
#endif

  cuda_notification_info.pending_send = false;
}

void
cuda_notification_resend (void)
{
  cuda_notification_acquire_lock ();

  if (!cuda_notification_info.pending_send && !cuda_notification_info.sent)
    {
      cuda_notification_trace ("resend: scheduling notification for next "
			       "wait cycle");
      cuda_notification_info.pending_send = true;
      memset (&cuda_notification_info.pending_send_data, 0,
	      sizeof cuda_notification_info.pending_send_data);
    }
  else
    {
      cuda_notification_trace ("resend: notification already pending or sent");
    }

  cuda_notification_release_lock ();
}

void _initialize_cuda_notification ();
void
_initialize_cuda_notification ()
{
  memset (&cuda_notification_info, 0, sizeof cuda_notification_info);
  pthread_mutex_init (&cuda_notification_info.mutex, NULL);
  cuda_notification_info.initialized = true;
}
