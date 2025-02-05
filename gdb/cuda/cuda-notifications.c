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


/*-------------------------------------- Notifications -------------------------------------
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
 * sent later, when notifications become unblocked, and the notification will go
 * from (producer) pending state to (producer) sent state.
 * Additionally, if a notification is received before a previous event has been
 * serviced, it is marked as an aliased_event, and an attempt is made to service
 * it before the inferior is resumed. No new stop signal is sent for an aliased_event.
 */

#ifdef GDBSERVER
#include "server.h"
#include "cuda-tdep-server.h"
#else
#include "defs.h"
#include "cuda-options.h"
#include "cuda-tdep.h"
#include "cuda-api.h"
#include "gdbthread.h"
#include "inferior.h"
#include "remote.h"
#include "cuda-packet-manager.h"
#endif

#include "cuda-notifications.h"
#include <unistd.h>
#include <ctype.h>
#include <pthread.h>
#include <signal.h>
#ifndef __QNXHOST__
# include <sys/syscall.h>
#endif

static struct {
  bool initialized;       /* True if the mutex is initialized */
  bool blocked;           /* When blocked, stop signal will be marked pending and handled later. */
  bool pending_send;      /* True if a stop signal was received while blocked was true. */
  bool aliased_event;     /* True if a stop signal was received while a previous event was being processed. */
  bool sent;              /* If already sent, do not send duplicates. */
  bool received;          /* True if the stop signal has been received. */
  uint32_t tid;           /* The thread id of the thread to which the stop signal was sent to. */
  pthread_mutex_t mutex;  /* Mutex for the cuda_notification_* functions */
  CUDBGEventCallbackData pending_send_data;
} cuda_notification_info;

#if __QNXHOST__
extern uint32_t inferior_pid;
#endif

static void
cuda_notification_trace (const char *fmt, ...)
{
#ifdef GDBSERVER
  struct cuda_trace_msg *msg;
#endif
  va_list ap;

  if (!cuda_options_debug_notifications())
    return;

  va_start (ap, fmt);
#ifdef GDBSERVER
  msg = (struct cuda_trace_msg *) xmalloc (sizeof (*msg));
  if (!cuda_first_trace_msg)
    cuda_first_trace_msg = msg;
  else
    cuda_last_trace_msg->next = msg;
  sprintf (msg->buf, "[CUDAGDB] notifications -- ");
  vsnprintf (msg->buf + strlen (msg->buf), sizeof (msg->buf), fmt, ap);
  msg->next = NULL;
  cuda_last_trace_msg = msg;
#else
  fprintf (stderr, "[CUDAGDB] notifications -- ");
  vfprintf (stderr, fmt, ap);
  fprintf (stderr, "\n");
  fflush (stderr);
#endif
}

void
cuda_notification_reset (void)
{
  gdb_assert (cuda_notification_info.initialized);
  cuda_notification_info.blocked      = false;
  cuda_notification_info.pending_send = false;
  cuda_notification_info.sent         = false;
  cuda_notification_info.received     = false;
  cuda_notification_info.tid          = false;
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
  signal = cuda_options_stop_signal () == GDB_SIGNAL_URG ? SIGURG : SIGTRAP;
#endif
#ifdef __linux__
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

  return kill (tid, signal);
}

static int
cuda_notification_notify_specific_thread (uint32_t tid)
{
  int err = 1;

  err = cuda_notification_notify_thread (tid);

  cuda_notification_trace ("sent specifically to the given host thread: tid %d -> %s",
                           tid, err ? "FAILED" : "success");

  return err;
}

#if __QNXHOST__
static uint32_t
qnx_cuda_notification_notify_thread (uint32_t tid)
{
  if (cuda_notification_notify_specific_thread (tid) == 0)
    {
      return tid;
    }
  else
    {
      return 0;
    }
}

#else /* !__QNXHOST__ */

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
  return ((*(int*)tid1) > (*(int*)tid2));
}

#define MAX_YOUNG_THREADS 128
typedef struct {
  int   num;
  int   tid[MAX_YOUNG_THREADS];
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
                           (long) tid, tid ? "success" : "FAILED");

  return tid;
}
#endif /* !__QNX_HOST__ */

static void
cuda_notification_send (CUDBGEventCallbackData *data)
{
  uint32_t tid = 0;
  int err = 1;

  // use the host thread id if given to us
  if (!tid && cuda_platform_supports_tid () && data && data->tid)
    {
      err = cuda_notification_notify_specific_thread (data->tid);
      if (!err)
        tid = data->tid;
    }

#ifdef __QNXHOST__
  /* FIXME: we are not tracking any threads apart from the main one */
  if (!tid)
    {
      tid = qnx_cuda_notification_notify_thread (inferior_pid);
    }
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
      memset (&cuda_notification_info.pending_send_data, 0, sizeof cuda_notification_info.pending_send_data);
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
cuda_notification_notify (CUDBGEventCallbackData *data)
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
    cuda_notification_trace ("ignoring: another notification is already pending");
  else if (cuda_notification_info.blocked)
    {
      cuda_notification_trace ("blocked: marking notification as pending_send");
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
cuda_notification_analyze (ptid_t ptid, struct target_waitstatus *ws, int trap_expected)
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
     reason we stopped. */
  if (cuda_notification_info.sent &&
      cuda_notification_info.tid == cuda_gdb_get_tid_or_pid (ptid) &&
      ws->kind () == TARGET_WAITKIND_STOPPED &&
      (ws->sig() == GDB_SIGNAL_URG ||
#ifdef __QNXHOST__
       ws->sig () == GDB_SIGNAL_EMT ||
       ws->sig () == GDB_SIGNAL_ILL ||
#endif
       ws->sig () == GDB_SIGNAL_TRAP) &&
      !trap_expected)
    {
      cuda_notification_trace ("received notification to thread %d", cuda_notification_info.tid);
      cuda_notification_info.received = true;
    }

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
      cuda_notification_trace ("consuming notification to thread %d", cuda_notification_info.tid);
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

void _initialize_cuda_notification ();
void
_initialize_cuda_notification ()
{
  memset (&cuda_notification_info, 0, sizeof cuda_notification_info);
  pthread_mutex_init (&cuda_notification_info.mutex, NULL);
  cuda_notification_info.initialized = true;
}

