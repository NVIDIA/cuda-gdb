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

#ifndef _CUDA_PACKET_MANAGER_H
#define _CUDA_PACKET_MANAGER_H 1

#include "cudadebugger.h"
#include "gdbsupport/ptid.h"

#include <cstdint>

struct target_waitstatus;

typedef enum : uint32_t
{
  /* Notifications (server-side notification queue / signaling).  */
  NOTIFICATION_ANALYZE,
  NOTIFICATION_PENDING,
  NOTIFICATION_RECEIVED,
  NOTIFICATION_ALIASED_EVENT,
  NOTIFICATION_MARK_CONSUMED,
  NOTIFICATION_CONSUME_PENDING,

  /* Lifecycle and aggregated server-side state.  */
  INITIALIZE_TARGET,
  API_FINALIZE,
  QUERY_DEVICE_SPEC,
  QUERY_TRACE_MESSAGE,
  CHECK_PENDING_SIGINT,
  SET_OPTION,

#if defined(__QNXTARGET__) || defined(__QNXHOST__)
  /* QNX-only: symbol address upload.  */
  SET_SYMBOLS,
#endif /* defined(__QNXTARGET__) || defined(__QNXHOST__) */

  /* Build-hash compatibility probe (QNX only on the wire). Reserved
     above the previously used packet-id range so a revision-skewed peer
     hits `handle_cuda_packet`'s `default:` and aborts via `error()`. */
  CUDA_PROTOCOL_HASH_HANDSHAKE = 0xFFFFFFFEu,
} cuda_packet_type_t;

/* Device Properties */
void cuda_remote_query_device_spec (uint32_t dev_id, uint32_t *num_sms,
				    uint32_t *num_warps, uint32_t *num_lanes,
				    uint32_t *num_registers, char **dev_type,
				    char **sm_type);

/* Notifications */
bool cuda_remote_notification_pending ();
bool cuda_remote_notification_received ();
bool cuda_remote_notification_aliased_event ();
void cuda_remote_notification_analyze (ptid_t ptid,
				       struct target_waitstatus *ws);
void cuda_remote_notification_mark_consumed ();
void cuda_remote_notification_consume_pending ();

#ifdef __QNXTARGET__
void cuda_remote_set_symbols (bool set_extra_symbols, bool *symbols_are_set);
#endif /* __QNXTARGET__ */
void cuda_remote_initialize (CUDBGResult *get_debugger_api_res,
			     CUDBGResult *set_callback_api_res,
			     CUDBGResult *initialize_api_res,
			     bool *cuda_initialized,
			     bool *cuda_debugging_enabled,
			     bool *driver_is_compatiable, uint32_t *major,
			     uint32_t *minor, uint32_t *revision);
CUDBGResult cuda_remote_api_finalize ();

bool cuda_remote_check_pending_sigint (ptid_t ptid);

void cuda_remote_set_option ();
void cuda_remote_query_trace_message ();

#ifdef __QNXTARGET__
void cuda_qnx_protocol_hash_handshake ();
#endif /* __QNXTARGET__ */

#endif
