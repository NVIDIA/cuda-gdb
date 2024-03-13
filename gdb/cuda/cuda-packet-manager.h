/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2023 NVIDIA Corporation
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
#ifdef __QNXTARGET__
#include "target.h"
#endif /* __QNXTARGET */

typedef enum {
    /* api */
    RESUME_DEVICE,
    SUSPEND_DEVICE,
    SINGLE_STEP_WARP65,
    SET_BREAKPOINT,
    UNSET_BREAKPOINT,
    READ_GRID_ID,
    READ_BLOCK_IDX,
    READ_THREAD_IDX,
    READ_BROKEN_WARPS,
    READ_VALID_WARPS,
    READ_VALID_LANES,
    READ_ACTIVE_LANES,
    READ_CODE_MEMORY,
    READ_CONST_MEMORY,
    READ_GENERIC_MEMORY,
    READ_PINNED_MEMORY,
    READ_PARAM_MEMORY,
    READ_SHARED_MEMORY,
    READ_TEXTURE_MEMORY,
    READ_TEXTURE_MEMORY_BINDLESS,
    READ_LOCAL_MEMORY,
    READ_REGISTER,
    READ_PC,
    READ_VIRTUAL_PC,
    READ_LANE_EXCEPTION,
    READ_CALL_DEPTH,
    READ_SYSCALL_CALL_DEPTH,
    READ_VIRTUAL_RETURN_ADDRESS,
    READ_ERROR_PC,
    WRITE_GENERIC_MEMORY,
    WRITE_PINNED_MEMORY,
    WRITE_PARAM_MEMORY,
    WRITE_SHARED_MEMORY,
    WRITE_LOCAL_MEMORY,
    WRITE_REGISTER,
    IS_DEVICE_CODE_ADDRESS,
    DISASSEMBLE,
    MEMCHECK_READ_ERROR_ADDRESS,
    GET_NUM_DEVICES,
    GET_GRID_STATUS,
    GET_GRID_INFO,
    GET_ADJUSTED_CODE_ADDRESS,
    GET_HOST_ADDR_FROM_DEVICE_ADDR,
    GET_ERROR_STRING_EX,

    /* notification */
    NOTIFICATION_ANALYZE,
    NOTIFICATION_PENDING,
    NOTIFICATION_RECEIVED,
    NOTIFICATION_ALIASED_EVENT,
    NOTIFICATION_MARK_CONSUMED,
    NOTIFICATION_CONSUME_PENDING,

    /* event */
    QUERY_SYNC_EVENT,
    QUERY_ASYNC_EVENT,
    ACK_SYNC_EVENTS,

    /* other */
    UPDATE_GRID_ID_IN_SM,
    UPDATE_BLOCK_IDX_IN_SM,
    UPDATE_THREAD_IDX_IN_WARP,
    INITIALIZE_TARGET,
    QUERY_DEVICE_SPEC,
    QUERY_TRACE_MESSAGE,
    CHECK_PENDING_SIGINT,
    API_INITIALIZE,
    API_FINALIZE,
    CLEAR_ATTACH_STATE,
    REQUEST_CLEANUP_ON_DETACH,
    SET_OPTION,
    SET_ASYNC_LAUNCH_NOTIFICATIONS,
    READ_DEVICE_EXCEPTION_STATE,
#if defined(__QNXTARGET__) || defined(__QNXHOST__)
    SET_SYMBOLS,
    VERSION_HANDSHAKE,
#endif /* defined(__QNXTARGET__) || defined(__QNXHOST__) */
    READ_CLUSTER_IDX,
    UPDATE_CLUSTER_IDX_IN_SM,
    GET_DEVICE_INFO_SIZES,
    GET_DEVICE_INFO,
    SINGLE_STEP_WARP,
} cuda_packet_type_t;

class remote_target;

extern int hex2bin (const char *hex, gdb_byte *bin, int count);
extern int bin2hex (const gdb_byte *bin, char *hex, int count);

/* Device Properties */
void cuda_remote_query_device_spec (remote_target *ops,
				    uint32_t dev_id, uint32_t *num_sms, uint32_t *num_warps, uint32_t *num_lanes,
				    uint32_t *num_registers,
				    uint32_t *num_uregisters,
				    char **dev_type, char **sm_type);

/* Notifications */
bool cuda_remote_notification_pending (remote_target *ops);
bool cuda_remote_notification_received (remote_target *ops);
bool cuda_remote_notification_aliased_event (remote_target *ops);
#ifdef __QNXTARGET__
void cuda_remote_notification_analyze (remote_target *ops, ptid_t ptid, struct target_waitstatus *ws);
#else /* __QNXTARGET__ */
void cuda_remote_notification_analyze (remote_target *ops, ptid_t ptid);
#endif /* __QNXTARDET__ */
void cuda_remote_notification_mark_consumed (remote_target *ops);
void cuda_remote_notification_consume_pending (remote_target *ops);

/* Events */
bool cuda_remote_query_sync_events (void);
bool cuda_remote_query_async_events (void);

/* Others */
void cuda_remote_update_grid_id_in_sm (remote_target *ops, uint32_t dev, uint32_t sm);
void cuda_remote_update_cluster_idx_in_sm (remote_target *ops, uint32_t dev, uint32_t sm);
void cuda_remote_update_block_idx_in_sm (remote_target *ops, uint32_t dev, uint32_t sm);
void cuda_remote_update_thread_idx_in_warp (remote_target *ops, uint32_t dev, uint32_t sm, uint32_t wp);
#ifdef __QNXTARGET__
void cuda_remote_set_symbols (remote_target *ops, bool set_extra_symbols, bool *symbols_are_set);
#endif /* __QNXTARGET__ */
void cuda_remote_initialize (remote_target *remote,
			     CUDBGResult *get_debugger_api_res, CUDBGResult *set_callback_api_res, 
                             CUDBGResult *initialize_api_res, bool *cuda_initialized, 
                             bool *cuda_debugging_enabled, bool *driver_is_compatiable,
			     uint32_t *major, uint32_t *minor, uint32_t *revision);
CUDBGResult cuda_remote_api_finalize (remote_target *remote);
#ifdef __QNXTARGET__
bool cuda_remote_check_pending_sigint (remote_target *ops, ptid_t ptid);
#else /* __QNXTARGET__ */
bool cuda_remote_check_pending_sigint (remote_target *ops);
#endif /* __QNXTARGET__ */

void cuda_remote_set_option (remote_target *ops);
void cuda_remote_query_trace_message (remote_target *ops);

#ifdef __QNXTARGET__
void cuda_qnx_version_handshake (remote_target *ops);
#endif /* __QNXTARGET__ */

#endif
