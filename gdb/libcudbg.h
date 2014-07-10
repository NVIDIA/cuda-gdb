/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2014 NVIDIA Corporation
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


#ifndef LIBCUDB_H
#define LIBCUDB_H 1

typedef enum {
    /* Deprecated API Version Query */
    CUDBGAPIREQ_STUB_getAPI,
    CUDBGAPIREQ_STUB_getAPIVersion,
    /* Initialization */
    CUDBGAPIREQ_initialize,
    CUDBGAPIREQ_finalize,

    /* Device Execution Control */
    CUDBGAPIREQ_suspendDevice,
    CUDBGAPIREQ_resumeDevice,

    /* Breakpoints */

    /* Device State Inspection */
    CUDBGAPIREQ_readThreadIdx,
    CUDBGAPIREQ_readBrokenWarps,
    CUDBGAPIREQ_readValidWarps,
    CUDBGAPIREQ_readValidLanes,
    CUDBGAPIREQ_readActiveLanes,
    CUDBGAPIREQ_readCodeMemory,
    CUDBGAPIREQ_readConstMemory,
    CUDBGAPIREQ_readParamMemory,
    CUDBGAPIREQ_readSharedMemory,
    CUDBGAPIREQ_readLocalMemory,
    CUDBGAPIREQ_readRegister,
    CUDBGAPIREQ_readPC,
    CUDBGAPIREQ_readVirtualPC,
    CUDBGAPIREQ_readLaneStatus,

    /* Device State Alteration */
    CUDBGAPIREQ_writeParamMemory,
    CUDBGAPIREQ_writeSharedMemory,
    CUDBGAPIREQ_writeLocalMemory,
    CUDBGAPIREQ_writeRegister,

    /* Grid Properties */
    CUDBGAPIREQ_getBlockDim,
    CUDBGAPIREQ_getTID,

    /* Device Properties */
    CUDBGAPIREQ_getDeviceType,
    CUDBGAPIREQ_getSmType,
    CUDBGAPIREQ_getNumDevices,
    CUDBGAPIREQ_getNumSMs,
    CUDBGAPIREQ_getNumWarps,
    CUDBGAPIREQ_getNumLanes,
    CUDBGAPIREQ_getNumRegisters,

    /* DWARF-related routines */
    CUDBGAPIREQ_disassemble,
    CUDBGAPIREQ_lookupDeviceCodeSymbol,

    /* Events */

    /* 3.1 Extensions */
    CUDBGAPIREQ_getGridAttribute,
    CUDBGAPIREQ_getGridAttributes,
    CUDBGAPIREQ_readLaneException,

    /* 3.1 - ABI */

    /* 3.2 Extensions */
    CUDBGAPIREQ_readPinnedMemory,
    CUDBGAPIREQ_writePinnedMemory,
    CUDBGAPIREQ_setBreakpoint,
    CUDBGAPIREQ_unsetBreakpoint,

    /* 4.0 Extensions */
    CUDBGAPIREQ_readTextureMemory,
    CUDBGAPIREQ_readBlockIdx,
    CUDBGAPIREQ_getGridDim,
    CUDBGAPIREQ_readCallDepth,
    CUDBGAPIREQ_readReturnAddress,
    CUDBGAPIREQ_readVirtualReturnAddress,

    /* 4.1 Extensions */
    CUDBGAPIREQ_getHostAddrFromDeviceAddr,
    CUDBGAPIREQ_singleStepWarp,
    CUDBGAPIREQ_setNotifyNewEventCallback,
    CUDBGAPIREQ_readSyscallCallDepth,

    /* 4.2 Extensions */
    CUDBGAPIREQ_readTextureMemoryBindless,

    /* 5.0 Extensions */
    CUDBGAPIREQ_clearAttachState,
    CUDBGAPIREQ_memcheckReadErrorAddress,
    CUDBGAPIREQ_acknowledgeSyncEvents,
    CUDBGAPIREQ_initializeAttachStub,

    /* 5.5 Extensions */
    CUDBGAPIREQ_getGridInfo,
    CUDBGAPIREQ_readGridId,
    CUDBGAPIREQ_getGridStatus,
    CUDBGAPIREQ_setKernelLaunchNotificationMode,
    CUDBGAPIREQ_getDevicePCIBusInfo,
    CUDBGAPIREQ_readDeviceExceptionState,

    /* 6.0 Extensions */
    CUDBGAPIREQ_getAdjustedCodeAddress,
    CUDBGAPIREQ_readErrorPC,
    CUDBGAPIREQ_getNextEvent,
    CUDBGAPIREQ_getElfImageByHandle,
    CUDBGAPIREQ_resumeWarpsUntilPC,
    CUDBGAPIREQ_readWarpState,
    CUDBGAPIREQ_readRegisterRange,
    CUDBGAPIREQ_readGenericMemory,
    CUDBGAPIREQ_writeGenericMemory,
    CUDBGAPIREQ_readGlobalMemory,
    CUDBGAPIREQ_writeGlobalMemory,
    CUDBGAPIREQ_getManagedMemoryRegionInfo,
    CUDBGAPIREQ_isDeviceCodeAddress,
    CUDBGAPIREQ_requestCleanupOnDetach,

    /* 6.5 Extensions */
    CUDBGAPIREQ_readPredicates,
    CUDBGAPIREQ_writePredicates,
    CUDBGAPIREQ_getNumPredicates,
    CUDBGAPIREQ_readCCRegister,
    CUDBGAPIREQ_writeCCRegister,

    CUDBGAPIREQ_getDeviceName,
} CUDBGAPIREQ_t;

typedef enum {
    LIBCUDBG_PIPE_ENDPOINT_RPCD = 999,
    LIBCUDBG_PIPE_ENDPOINT_DEBUG_CLIENT,
    LIBCUDBG_PIPE_ENDPOINT_RPCD_CB,
    LIBCUDBG_PIPE_ENDPOINT_DEBUG_CLIENT_CB,
} libcudbg_pipe_endpoint_t;

#pragma pack(push,1)
typedef struct  {
    uint32_t tid;
    uint32_t terminate;
    uint32_t timeout;
} CUDBGCBMSG_t;
#pragma pack(pop)

#endif
