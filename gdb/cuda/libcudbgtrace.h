/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2020-2024 NVIDIA Corporation
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

#ifndef LIBCUDBGIPCTRACE_H
#define LIBCUDBGIPCTRACE_H 1

#include <stdint.h>
#include <stdio.h>

#include "libcudbgipc.h"

#include <cudadebugger.h>

static FILE *s_traceFile;
static bool s_isFirstRequest = true;
static bool s_isFirstArgument = false;

void
cudbgOpenIpcTraceFile ()
{
  s_traceFile = fopen ("ipc.log", "w");
}

void
cudbgCloseIpcTraceFile ()
{
  if (s_traceFile)
    {
      fclose (s_traceFile);
    }
}

void
cudbgTraceIpcRequest (const char *cmd)
{
  if (!s_isFirstRequest)
    {
      fprintf (s_traceFile, "\n");
    }
  else
    {
      s_isFirstRequest = false;
    }
  fprintf (s_traceFile, "%s(", cmd);
  fflush (s_traceFile);
  s_isFirstArgument = true;
}

void
cudbgTraceIpcResponse ()
{
  fprintf (s_traceFile, "):\n");
  fflush (s_traceFile);
}

template < typename T > static void
cudbgTraceValue (T * value)
{
  fprintf (s_traceFile, "<missing value printer>");
  fflush (s_traceFile);
}

static void
cudbgTraceValue (bool * value)
{
  fprintf (s_traceFile, "%s", *value ? "true" : "false");
  fflush (s_traceFile);
}

static void
cudbgTraceValue (uint32_t * value)
{
  fprintf (s_traceFile, "0x%x", *value);
  fflush (s_traceFile);
}

static void
cudbgTraceValue (uint64_t * value)
{
  fprintf (s_traceFile, "0x%llx", (unsigned long long) *value);
  fflush (s_traceFile);
}

static void
cudbgTraceValue (CUDBGResult * value)
{
  fprintf (s_traceFile, "%s (0x%x)", cudbgGetErrorString (*value),
	   (uint32_t) * value);
  fflush (s_traceFile);
}

#define CASE_PRINT_ENUM_VALUE(v)                                \
case v: fprintf (s_traceFile, "%s (0x%x)", #v, ((uint32_t)(v)));

#define CASE_PRINT_ENUM_DEFAULT(v)                              \
default: fprintf (s_traceFile, "<unknown> (0x%x)", ((uint32_t)(v)));

static void
cudbgTraceValue (CUDBGAttribute * value)
{
  switch (*value)
    {
      CASE_PRINT_ENUM_VALUE (CUDBG_ATTR_GRID_LAUNCH_BLOCKING) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_ATTR_GRID_TID) break;
      CASE_PRINT_ENUM_DEFAULT (*value) break;
    }
  fflush (s_traceFile);
}

static void
cudbgTraceValue (CUDBGGridStatus * value)
{
  switch (*value)
    {
      CASE_PRINT_ENUM_VALUE (CUDBG_GRID_STATUS_INVALID) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_GRID_STATUS_PENDING) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_GRID_STATUS_ACTIVE) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_GRID_STATUS_SLEEPING) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_GRID_STATUS_TERMINATED) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_GRID_STATUS_UNDETERMINED) break;
      CASE_PRINT_ENUM_DEFAULT (*value) break;
    }
  fflush (s_traceFile);
}

static void
cudbgTraceValue (CUDBGKernelType * value)
{
  switch (*value)
    {
      CASE_PRINT_ENUM_VALUE (CUDBG_KNL_TYPE_UNKNOWN) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_KNL_TYPE_SYSTEM) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_KNL_TYPE_APPLICATION) break;
      CASE_PRINT_ENUM_DEFAULT (*value) break;
    }
  fflush (s_traceFile);
}

static void
cudbgTraceValue (CUDBGKernelOrigin * value)
{
  switch (*value)
    {
      CASE_PRINT_ENUM_VALUE (CUDBG_KNL_ORIGIN_CPU) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_KNL_ORIGIN_GPU) break;
      CASE_PRINT_ENUM_DEFAULT (*value) break;
    }
  fflush (s_traceFile);
}

static void
cudbgTraceValue (CUDBGKernelLaunchNotifyMode * value)
{
  switch (*value)
    {
      CASE_PRINT_ENUM_VALUE (CUDBG_KNL_LAUNCH_NOTIFY_EVENT) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_KNL_LAUNCH_NOTIFY_DEFER) break;
      CASE_PRINT_ENUM_DEFAULT (*value) break;
    }
  fflush (s_traceFile);
}

static void
cudbgTraceValue (CUDBGEventQueueType * value)
{
  switch (*value)
    {
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_QUEUE_TYPE_SYNC) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_QUEUE_TYPE_ASYNC) break;
      CASE_PRINT_ENUM_DEFAULT (*value) break;
    }
  fflush (s_traceFile);
}

static void
cudbgTraceValue (CUDBGElfImageType * value)
{
  switch (*value)
    {
      CASE_PRINT_ENUM_VALUE (CUDBG_ELF_IMAGE_TYPE_NONRELOCATED) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_ELF_IMAGE_TYPE_RELOCATED) break;
      CASE_PRINT_ENUM_DEFAULT (*value) break;
    }
  fflush (s_traceFile);
}

static void
cudbgTraceValue (CUDBGAdjAddrAction * value)
{
  switch (*value)
    {
      CASE_PRINT_ENUM_VALUE (CUDBG_ADJ_PREVIOUS_ADDRESS) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_ADJ_CURRENT_ADDRESS) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_ADJ_NEXT_ADDRESS) break;
      CASE_PRINT_ENUM_DEFAULT (*value) break;
    }
  fflush (s_traceFile);
}

static void
cudbgTraceValue (CUDBGException_t * value)
{
  switch (*value)
    {
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_UNKNOWN) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_NONE) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_LANE_USER_STACK_OVERFLOW) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_DEVICE_HARDWARE_STACK_OVERFLOW) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_WARP_INVALID_PC) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_LANE_MISALIGNED_ADDRESS) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_WARP_ASSERT) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_LANE_SYSCALL_ERROR) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_LANE_NONMIGRATABLE_ATOMSYS) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_LANE_INVALID_ATOMSYS) break;
#if (CUDBG_API_VERSION_REVISION >= 131)
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT) break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_CLUSTER_OUT_OF_RANGE_ADDRESS) break;
#endif
      CASE_PRINT_ENUM_VALUE (CUDBG_EXCEPTION_WARP_STACK_CANARY) break;
      CASE_PRINT_ENUM_DEFAULT (*value) break;
    }
  fflush (s_traceFile);
}

static void
cudbgTraceValue (ptxStorageKind * value)
{
  switch (*value)
    {
      CASE_PRINT_ENUM_VALUE (ptxUNSPECIFIEDStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxCodeStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxRegStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxSregStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxConstStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxGlobalStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxLocalStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxParamStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxSharedStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxSurfStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxTexSamplerStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxGenericStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxIParamStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxOParamStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxFrameStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxURegStorage) break;
      CASE_PRINT_ENUM_VALUE (ptxMAXStorage) break;
      CASE_PRINT_ENUM_DEFAULT (*value) break;
    }
  fflush (s_traceFile);
}

static void
cudbgTraceValue (CuDim3 * value)
{
  fprintf (s_traceFile, "(0x%x, 0x%x, 0x%x)", value->x, value->y, value->z);
  fflush (s_traceFile);
}

template < typename T > static void
cudbgTraceField (const char *name, T * value, uint32_t indentLevel = 2)
{
  fprintf (s_traceFile, "\n");
  for (uint32_t i = 0; i < indentLevel; i++)
    {
      fprintf (s_traceFile, "    ");
    }
  fprintf (s_traceFile, "%s: ", name);
  cudbgTraceValue (value);
  fflush (s_traceFile);
}

#define PRINT_ATTRIBUTE_VALUE_PAIR_FIELD(pAttributeValuePair, field)    \
do {                                                                    \
  cudbgTraceField (#field, &pAttributeValuePair->field);                \
} while (0)

static void
cudbgTraceValue (CUDBGAttributeValuePair * value)
{
  PRINT_ATTRIBUTE_VALUE_PAIR_FIELD (value, attribute);
  PRINT_ATTRIBUTE_VALUE_PAIR_FIELD (value, value);
}

#define PRINT_EVENT_FIELD(pEvent, case, field)          \
do {                                                    \
  cudbgTraceField (#field, &pEvent->cases.case.field);  \
} while (0)

static void
cudbgTraceValue (CUDBGEvent * value)
{
  switch (value->kind)
    {
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_INVALID)
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_ELF_IMAGE_LOADED)
	PRINT_EVENT_FIELD (value, elfImageLoaded, dev);
	PRINT_EVENT_FIELD (value, elfImageLoaded, context);
	PRINT_EVENT_FIELD (value, elfImageLoaded, module);
	PRINT_EVENT_FIELD (value, elfImageLoaded, size);
	PRINT_EVENT_FIELD (value, elfImageLoaded, handle);
	PRINT_EVENT_FIELD (value, elfImageLoaded, properties);
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_KERNEL_READY)
	PRINT_EVENT_FIELD (value, kernelReady, dev);
	PRINT_EVENT_FIELD (value, kernelReady, tid);
	PRINT_EVENT_FIELD (value, kernelReady, gridId);
	PRINT_EVENT_FIELD (value, kernelReady, context);
	PRINT_EVENT_FIELD (value, kernelReady, module);
	PRINT_EVENT_FIELD (value, kernelReady, function);
	PRINT_EVENT_FIELD (value, kernelReady, functionEntry);
	PRINT_EVENT_FIELD (value, kernelReady, gridDim);
	PRINT_EVENT_FIELD (value, kernelReady, blockDim);
	PRINT_EVENT_FIELD (value, kernelReady, type);
	PRINT_EVENT_FIELD (value, kernelReady, parentGridId);
	PRINT_EVENT_FIELD (value, kernelReady, origin);
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_KERNEL_FINISHED)
	PRINT_EVENT_FIELD (value, kernelFinished, dev);
	PRINT_EVENT_FIELD (value, kernelFinished, tid);
	PRINT_EVENT_FIELD (value, kernelFinished, context);
	PRINT_EVENT_FIELD (value, kernelFinished, module);
	PRINT_EVENT_FIELD (value, kernelFinished, function);
	PRINT_EVENT_FIELD (value, kernelFinished, functionEntry);
	PRINT_EVENT_FIELD (value, kernelFinished, gridId);
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_INTERNAL_ERROR)
	PRINT_EVENT_FIELD (value, internalError, errorType);
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_CTX_PUSH)
	PRINT_EVENT_FIELD (value, contextPush, dev);
	PRINT_EVENT_FIELD (value, contextPush, tid);
	PRINT_EVENT_FIELD (value, contextPush, context);
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_CTX_POP)
	PRINT_EVENT_FIELD (value, contextPop, dev);
	PRINT_EVENT_FIELD (value, contextPop, tid);
	PRINT_EVENT_FIELD (value, contextPop, context);
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_CTX_CREATE)
	PRINT_EVENT_FIELD (value, contextCreate, dev);
	PRINT_EVENT_FIELD (value, contextCreate, tid);
	PRINT_EVENT_FIELD (value, contextCreate, context);
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_CTX_DESTROY)
	PRINT_EVENT_FIELD (value, contextDestroy, dev);
	PRINT_EVENT_FIELD (value, contextDestroy, tid);
	PRINT_EVENT_FIELD (value, contextDestroy, context);
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_TIMEOUT)
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_ATTACH_COMPLETE)
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_DETACH_COMPLETE)
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_ELF_IMAGE_UNLOADED)
	PRINT_EVENT_FIELD (value, elfImageUnloaded, dev);
	PRINT_EVENT_FIELD (value, elfImageUnloaded, context);
	PRINT_EVENT_FIELD (value, elfImageUnloaded, module);
	PRINT_EVENT_FIELD (value, elfImageUnloaded, size);
	PRINT_EVENT_FIELD (value, elfImageUnloaded, handle);
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_FUNCTIONS_LOADED)
	PRINT_EVENT_FIELD (value, functionsLoaded, dev);
	PRINT_EVENT_FIELD (value, functionsLoaded, count);
	PRINT_EVENT_FIELD (value, functionsLoaded, context);
	PRINT_EVENT_FIELD (value, functionsLoaded, module);
	break;
      CASE_PRINT_ENUM_VALUE (CUDBG_EVENT_ALL_DEVICES_SUSPENDED)
	PRINT_EVENT_FIELD (value, allDevicesSuspended, brokenDevicesMask);
	PRINT_EVENT_FIELD (value, allDevicesSuspended, faultedDevicesMask);
	break;
      CASE_PRINT_ENUM_DEFAULT (value->kind) break;
    }
  fflush (s_traceFile);
}

#define PRINT_CUDBG_LANE_STATE_FIELD(pLaneState, field)         \
do {                                                            \
  cudbgTraceField (#field, &pLaneState->field, 3);              \
} while (0)

static void
cudbgTraceValue (CUDBGLaneState * value)
{
  PRINT_CUDBG_LANE_STATE_FIELD (value, virtualPC);
  PRINT_CUDBG_LANE_STATE_FIELD (value, threadIdx);
  PRINT_CUDBG_LANE_STATE_FIELD (value, exception);
}

#define PRINT_CUDBG_WARP_LANE_STATE_FIELD(pWarpState, field, i) \
do {                                                            \
  char name[256];                                               \
  sprintf(name, #field "[%d]", i);                              \
  cudbgTraceField (name, &pWarpState->field[i]);                \
} while (0)

#define PRINT_CUDBG_WARP_STATE_FIELD(pWarpState, field)         \
do {                                                            \
  cudbgTraceField (#field, &pWarpState->field);                 \
} while (0)

static void
cudbgTraceValue (CUDBGWarpState * value)
{
  PRINT_CUDBG_WARP_STATE_FIELD (value, gridId);
  PRINT_CUDBG_WARP_STATE_FIELD (value, errorPC);
  PRINT_CUDBG_WARP_STATE_FIELD (value, blockIdx);
  PRINT_CUDBG_WARP_STATE_FIELD (value, validLanes);
  PRINT_CUDBG_WARP_STATE_FIELD (value, activeLanes);
  PRINT_CUDBG_WARP_STATE_FIELD (value, errorPCValid);
  PRINT_CUDBG_WARP_STATE_FIELD (value, clusterIdx);
  for (uint32_t i = 0; i < 32; i++)
    {
      if (value->validLanes & (1 << i))
	{
	  PRINT_CUDBG_WARP_LANE_STATE_FIELD (value, lane, i);
	}
    }
}

#define PRINT_CUDBG_GRID_INFO_FIELD(pGridInfo, field)           \
do {                                                            \
  cudbgTraceField (#field, &pGridInfo->field);                  \
} while (0)

static void
cudbgTraceValue (CUDBGGridInfo * value)
{
  PRINT_CUDBG_GRID_INFO_FIELD (value, dev);
  PRINT_CUDBG_GRID_INFO_FIELD (value, gridId64);
  PRINT_CUDBG_GRID_INFO_FIELD (value, tid);
  PRINT_CUDBG_GRID_INFO_FIELD (value, context);
  PRINT_CUDBG_GRID_INFO_FIELD (value, module);
  PRINT_CUDBG_GRID_INFO_FIELD (value, function);
  PRINT_CUDBG_GRID_INFO_FIELD (value, functionEntry);
  PRINT_CUDBG_GRID_INFO_FIELD (value, gridDim);
  PRINT_CUDBG_GRID_INFO_FIELD (value, blockDim);
  PRINT_CUDBG_GRID_INFO_FIELD (value, type);
  PRINT_CUDBG_GRID_INFO_FIELD (value, parentGridId);
  PRINT_CUDBG_GRID_INFO_FIELD (value, origin);
  PRINT_CUDBG_GRID_INFO_FIELD (value, clusterDim);
}

#define PRINT_CUDBG_MEMORY_INFO_FIELD(pMemoryInfo, field)       \
do {                                                            \
  cudbgTraceField (#field, &pMemoryInfo->field);                \
} while (0)

static void
cudbgTraceValue (CUDBGMemoryInfo * value)
{
  PRINT_CUDBG_MEMORY_INFO_FIELD (value, startAddress);
  PRINT_CUDBG_MEMORY_INFO_FIELD (value, size);
}

#if CUDBG_API_VERSION_REVISION >= 132
#define PRINT_CUDBG_LOADED_FUNCTION_INFO_FIELD(pLoadedFunctionInfo, field)  \
do {                                                                        \
  cudbgTraceField (#field, &pLoadedFunctionInfo->field);                    \
} while (0)

static void
cudbgTraceValue (CUDBGLoadedFunctionInfo * value)
{
  PRINT_CUDBG_LOADED_FUNCTION_INFO_FIELD (value, sectionIndex);
  PRINT_CUDBG_LOADED_FUNCTION_INFO_FIELD (value, address);
}
#endif

static const char *
cudbgTrimAmpersand (const char *name)
{
  return *name == '&' ? &name[1] : name;
}

template < typename T > static void
cudbgTraceIpcArgument (const char *name, T * value)
{
  if (!s_isFirstArgument)
    {
      fprintf (s_traceFile, ", ");
    }
  else
    {
      s_isFirstArgument = false;
    }
  fprintf (s_traceFile, "%s=", cudbgTrimAmpersand (name));
  cudbgTraceValue (value);
}

template < typename T > static void
cudbgTraceIpcOutput (const char *name, T * value)
{
  fprintf (s_traceFile, "    %s: ", cudbgTrimAmpersand (name));
  cudbgTraceValue (value);
  fprintf (s_traceFile, "\n");
}

template < typename T > static void
cudbgTraceIpcOutput (const char *name, T * values, uint32_t size)
{
  fprintf (s_traceFile, "    size: %u\n", size);

  for (uint32_t i = 0; i < size; i++)
    {
      fprintf (s_traceFile, "    %s[%u]: ", cudbgTrimAmpersand (name), i);
      cudbgTraceValue (&values[i]);
      fprintf (s_traceFile, "\n");
    }
}

static void
cudbgTraceIpcOutput (const char *name, void *bytes, uint32_t size)
{
  fprintf (s_traceFile, "    %s:", cudbgTrimAmpersand (name));

  const uint32_t countToPrint = size > 32 ? 32 : size;
  for (uint32_t i = 0; i < countToPrint; i++)
    {
      fprintf (s_traceFile, " %02X", ((uint8_t *) bytes)[i]);
    }
  if (countToPrint < size)
    {
      fprintf (s_traceFile, " ...");
    }

  const uint32_t crc = xcrc32 ((const unsigned char*) bytes, size, 0xffffffff);
  fprintf (s_traceFile, " (0x%x bytes, CRC32=%08X)\n", size, crc);
}

#undef CUDBG_IPC_BEGIN
#define CUDBG_IPC_BEGIN(cmd)                            \
do {                                                    \
  cudbgTraceIpcRequest(#cmd);                           \
  CUDBG_IPC_BEGIN_(cmd);                                \
} while (0)

#undef CUDBG_IPC_APPEND
#define CUDBG_IPC_APPEND(d, s)                          \
do {                                                    \
  cudbgTraceIpcArgument(#d, d);                         \
  CUDBG_IPC_APPEND_(d, s);                              \
} while (0)

#undef CUDBG_IPC_REQUEST
#define CUDBG_IPC_REQUEST(d)                            \
do {                                                    \
  cudbgTraceIpcResponse();                              \
  CUDBG_IPC_REQUEST_(d);                                \
} while (0)

#undef CUDBG_IPC_RECEIVE
#define CUDBG_IPC_RECEIVE(value, ipc_buf)               \
do {                                                    \
  CUDBG_IPC_RECEIVE_(value, ipc_buf);                   \
  cudbgTraceIpcOutput(#value, value);                   \
} while (0)

#undef CUDBG_IPC_RECEIVE_ARRAY
#define CUDBG_IPC_RECEIVE_ARRAY(value, size, ipc_buf)   \
do {                                                    \
  CUDBG_IPC_RECEIVE_ARRAY_(value, size, ipc_buf);       \
  cudbgTraceIpcOutput(#value, value, size);             \
} while (0)

#endif
