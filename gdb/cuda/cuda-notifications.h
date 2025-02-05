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

#ifndef _CUDA_NOTIFICATIONS_H
#define _CUDA_NOTIFICATIONS_H 1

#include "cudadebugger.h"

/*------------------------------- Notifications --------------------------------
  
   The CUDA notifications routines handle the SIGTRAP signal sent from the
   debugger API to the target application to wake up GDB. It also provides
   routines to help GDB decide if CUDA events/breakpoints/errors are to be
   handled.

------------------------------------------------------------------------------ */

/* The reset () routine should be called once at the end of each app run. */

void cuda_notification_reset (void);

/* The notify () routine is the callback function called by the debugger API
   when something CUDA-related is happening. It will send a SIGTRAP signal to
   the target application, which will have for effect to suspend the
   application and wake up GDB. Every non-static routines is protected with a
   mutex to avoid race conditions with the notify () routine. */

void cuda_notification_notify (CUDBGEventCallbackData *data);

/* The accept/block routines are used by GDB to decide if it is a good time to
   send a SIGTRAP to the target application. It is used to avoid race conditions
   and to bring some determinism to when SIGTRAPs are actually sent. In
   particular, GDB could wake up because of a host breakpoint while a device
   breakpoint is being hit. */

void cuda_notification_accept (void);
void cuda_notification_block (void);

/* From GDB's point of view, there is a notification to be handled when it is
   received (). That notification is consumed by calling mark_consumed (). There
   is a pending () notification when a notification has been sent but not to the
   host thread GDB woke up upon. The pending/received information is computed by
   calling analyze () first. At any point in time, there can be no notification,
   one received notification, or one pending notification. */

void cuda_notification_analyze (ptid_t ptid, struct target_waitstatus *ws, int trap_expected);
void cuda_notification_mark_consumed (void);
bool cuda_notification_pending (void);
bool cuda_notification_aliased_event (void);
void cuda_notification_reset_aliased_event (void);
bool cuda_notification_received (void);

void cuda_notification_consume_pending (void);

#endif
