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

#ifndef _CUDA_EVENTS_H
#define _CUDA_EVENTS_H 1

#include "cuda-stats.h"
#include "cudadebugger.h"

#include "gdbsupport/function-view.h"

typedef enum
{
  CUDA_EVENT_SYNC,
  CUDA_EVENT_ASYNC
} cuda_event_kind_t;

bool cuda_process_events (
    cuda_event_kind_t kind,
    gdb::function_view<bool (const CUDBGEvent &)> custom_handler = nullptr);

cuda_statistics_table &get_cuda_event_statistics (void);

/* Async and sync log events share one backend queue (consumeCudaLogs), but
   only the SYNC pass decides stops and it runs after the ASYNC pass in
   wait().  If async drained first it would consume (and only print) the very
   record SYNC needs to stop on.  While this RAII guard is live, async events
   only set a "drain pending" flag; cuda_drain_deferred_async_logs()
   flushes the remainder after the SYNC pass.  Host only.  */
struct cuda_scoped_defer_async_log_drain
{
  DISABLE_COPY_AND_ASSIGN (cuda_scoped_defer_async_log_drain);

  cuda_scoped_defer_async_log_drain ();
  ~cuda_scoped_defer_async_log_drain ();
};

/* Perform the drain deferred by cuda_scoped_defer_async_log_drain.  Uses the
   stop-aware log consumer so fallback async draining cannot discard rule
   matches; no-op if the SYNC pass already captured a stop.  */
void cuda_drain_deferred_async_logs ();

#endif
