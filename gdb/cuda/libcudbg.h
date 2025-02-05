/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2024 NVIDIA Corporation.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef LIBCUDBGREQ_H
#define LIBCUDBGREQ_H 1

typedef enum
{
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
  CUDBGAPIREQ_STUB_readTextureMemory,
  CUDBGAPIREQ_readBlockIdx,
  CUDBGAPIREQ_getGridDim,
  CUDBGAPIREQ_readCallDepth,
  CUDBGAPIREQ_readReturnAddress,
  CUDBGAPIREQ_readVirtualReturnAddress,

  /* 4.1 Extensions */
  CUDBGAPIREQ_getHostAddrFromDeviceAddr,
  CUDBGAPIREQ_singleStepWarp41,
  CUDBGAPIREQ_setNotifyNewEventCallback,
  CUDBGAPIREQ_readSyscallCallDepth,

  /* 4.2 Extensions */
  CUDBGAPIREQ_STUB_readTextureMemoryBindless,

  /* 5.0 Extensions */
  CUDBGAPIREQ_clearAttachState,
  CUDBGAPIREQ_memcheckReadErrorAddress,
  CUDBGAPIREQ_acknowledgeSyncEvents,
  CUDBGAPIREQ_initializeAttachStub,

  /* 5.5 Extensions */
  CUDBGAPIREQ_getGridInfo55,
  CUDBGAPIREQ_readGridId,
  CUDBGAPIREQ_getGridStatus,
  CUDBGAPIREQ_setKernelLaunchNotificationMode,
  CUDBGAPIREQ_getDevicePCIBusInfo,
  CUDBGAPIREQ_readDeviceExceptionState80,

  /* 6.0 Extensions */
  CUDBGAPIREQ_getAdjustedCodeAddress,
  CUDBGAPIREQ_readErrorPC,
  CUDBGAPIREQ_getNextEvent,
  CUDBGAPIREQ_getElfImageByHandle,
  CUDBGAPIREQ_resumeWarpsUntilPC,
  CUDBGAPIREQ_readWarpState60,
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
  CUDBGAPIREQ_singleStepWarp65,

  /* 9.0 Extensions */
  CUDBGAPIREQ_readDeviceExceptionState,

  /* 10.0 Extensions */
  CUDBGAPIREQ_getNumUniformRegisters,
  CUDBGAPIREQ_readUniformRegisterRange,
  CUDBGAPIREQ_writeUniformRegister,
  CUDBGAPIREQ_getNumUniformPredicates,
  CUDBGAPIREQ_readUniformPredicates,
  CUDBGAPIREQ_writeUniformPredicates,

  /* 11.8 extensions */
  CUDBGAPIREQ_getLoadedFunctionInfo118,

  /* 12.0 extensions */
  CUDBGAPIREQ_getGridInfo120,
  CUDBGAPIREQ_getClusterDim120,
  CUDBGAPIREQ_readWarpState120,
  CUDBGAPIREQ_readClusterIdx,

  /* 12.2 extensions */
  CUDBGAPIREQ_getErrorStringEx,

  /* 12.3 Extensions */
  CUDBGAPIREQ_getLoadedFunctionInfo,
  CUDBGAPIREQ_generateCoredump,
  CUDBGAPIREQ_getConstBankAddress123,

  /* 12.4 Extensions */
  CUDBGAPIREQ_getDeviceInfoSizes,
  CUDBGAPIREQ_getDeviceInfo,
  CUDBGAPIREQ_getConstBankAddress,
  CUDBGAPIREQ_singleStepWarp,

  /* 12.5 Extensions */
  CUDBGAPIREQ_readAllVirtualReturnAddresses,
  CUDBGAPIREQ_getSupportedDebuggerCapabilities,
  CUDBGAPIREQ_readSmException,

  /* 12.6 Extensions */
  CUDBGAPIREQ_executeInternalCommand,

  /* 12.7 extensions */
  CUDBGAPIREQ_getGridInfo,
  CUDBGAPIREQ_getClusterDim,
  CUDBGAPIREQ_readWarpState,
  CUDBGAPIREQ_getClusterExceptionTargetBlock,

  /* 12.8 extensions */
  CUDBGAPIREQ_readWarpResources,

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
