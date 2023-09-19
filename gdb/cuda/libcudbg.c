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

/* Set this to true to dump a libcudbg trace (ipc.log) */
#define TRACE_LIBCUDBG 0

#ifdef GDBSERVER
#include "server.h"
#include "cuda-tdep-server.h"
#include <ansidecl.h>
#else
#include "defs.h"
#include "cuda/cuda-options.h"
#include "cuda/cuda-tdep.h"
#endif
#include "cuda/cuda-version.h"

#include <stdio.h>
#include <signal.h>
#include <pthread.h>

#include "cuda/libcudbg.h"
#include "cuda/libcudbgipc.h"
#if TRACE_LIBCUDBG
#include "libcudbgtrace.h"
#endif
#include <cudadebugger.h>

/*Forward declarations */
static void cudbg_trace(const char *fmt, ...) ATTRIBUTE_PRINTF(1, 2);

/* Globals */
extern CUDBGNotifyNewEventCallback cudbgDebugClientCallback;
extern cuda_api_version cuda_backend_api_version;

static CUDBGResult
cudbgInitialize(void)
{
    CUDBGResult initRes, result;
    uint32_t major = CUDBG_API_VERSION_MAJOR;
    uint32_t minor = CUDBG_API_VERSION_MINOR;
    uint32_t revision = CUDBG_API_VERSION_REVISION;
    char *ipc_buf;

    initRes = cudbgipcInitialize();
    if (initRes != CUDBG_SUCCESS) {
        cudbg_trace ("IPC initialization failed (res=%d)", initRes);
        return initRes;
    }

#if TRACE_LIBCUDBG
    cudbgOpenIpcTraceFile ();
#endif

    cudbg_trace ("pre initialization successful");
    CUDBG_IPC_BEGIN(CUDBGAPIREQ_initialize);
    CUDBG_IPC_APPEND(&major, sizeof(minor));
    CUDBG_IPC_APPEND(&minor, sizeof(minor));
    CUDBG_IPC_APPEND(&revision, sizeof(revision));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(&major, &ipc_buf);
    CUDBG_IPC_RECEIVE(&minor, &ipc_buf);
    CUDBG_IPC_RECEIVE(&revision, &ipc_buf);

    if (result == CUDBG_ERROR_INCOMPATIBLE_API) {
        /* Allow newer cuda-gdb to work with driver from 11.x release. For more
           details see -
           https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility 
        */
        if (revision < 133)
            /* DEBUG API REVISION for 12.0 was 133 */
            printf("Incompatible CUDA driver version."
                   " Expected %d.%d.%d or later and found %d.%d.%d instead.",
                   CUDBG_API_VERSION_MAJOR,
                   CUDBG_API_VERSION_MINOR,
                   CUDBG_API_VERSION_REVISION,
                   (int)major, (int)minor, (int)revision);
        else {
            /* Retry with the version supported by driver */
            CUDBG_IPC_BEGIN(CUDBGAPIREQ_initialize);
            CUDBG_IPC_APPEND(&major, sizeof(minor));
            CUDBG_IPC_APPEND(&minor, sizeof(minor));
            CUDBG_IPC_APPEND(&revision, sizeof(revision));

            CUDBG_IPC_REQUEST((void **)&ipc_buf);
            CUDBG_IPC_RECEIVE(&result, &ipc_buf);
            CUDBG_IPC_RECEIVE(&major, &ipc_buf);
            CUDBG_IPC_RECEIVE(&minor, &ipc_buf);
            CUDBG_IPC_RECEIVE(&revision, &ipc_buf);
        }
    }

    cuda_backend_api_version = cuda_api_version (major, minor, revision);

    return result;
}

static CUDBGResult
cudbgSetNotifyNewEventCallback(CUDBGNotifyNewEventCallback callback)
{
    cudbgDebugClientCallback = callback;

    return CUDBG_SUCCESS;
}

static CUDBGResult
cudbgFinalize (void)
{
    char *ipc_buf;
    CUDBGResult result, ipcres;

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_finalize);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

#if TRACE_LIBCUDBG
    cudbgCloseIpcTraceFile ();
#endif

    ipcres = cudbgipcFinalize();
    if (ipcres != CUDBG_SUCCESS)
      cudbg_trace ("IPC finalize failed (res=%d)", ipcres);

    return result == CUDBG_SUCCESS ? ipcres : result;
}

static CUDBGResult
cudbgSuspendDevice (uint32_t dev)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_suspendDevice);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_suspendDevice, "suspendDevice");

    return result;
}

static CUDBGResult
cudbgResumeDevice (uint32_t dev)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_resumeDevice);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_resumeDevice, "resumeDevice");

    return result;
}

static CUDBGResult
STUB_cudbgSingleStepWarp40 (uint32_t dev, uint32_t sm, uint32_t wp)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgSetBreakpoint31 (uint64_t addr)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgUnsetBreakpoint31 (uint64_t addr)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgReadGridId50 (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *gridId)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgReadBlockIdx32 (uint32_t dev, uint32_t sm, uint32_t wp, CuDim2 *blockIdx)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadThreadIdx (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readThreadIdx);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(threadIdx, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readThreadIdx, "readThreadIdx");

    return result;
}

static CUDBGResult
cudbgReadBrokenWarps (uint32_t dev, uint32_t sm, uint64_t *brokenWarpsMask)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readBrokenWarps);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(brokenWarpsMask, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readBrokenWarps, "readBrokenWarps");

    return result;
}

static CUDBGResult
cudbgReadValidWarps (uint32_t dev, uint32_t sm, uint64_t *validWarpsMask)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readValidWarps);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(validWarpsMask, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readValidWarps, "readValidWarps");

    return result;
}

static CUDBGResult
cudbgReadValidLanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *validLanesMask)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readValidLanes);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(validLanesMask, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readValidLanes, "readValidLanes");

    return result;
}

static CUDBGResult
cudbgReadActiveLanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *activeLanesMask)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readActiveLanes);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(activeLanesMask, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readActiveLanes, "readActiveLanes");

    return result;
}

static CUDBGResult
cudbgReadCodeMemory (uint32_t dev, uint64_t addr, void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readCodeMemory);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readCodeMemory, "readCodeMemory");

    return result;
}

static CUDBGResult
cudbgReadConstMemory (uint32_t dev, uint64_t addr, void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readConstMemory);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readConstMemory, "readConstMemory");

    return result;
}

static CUDBGResult
STUB_cudbgReadGlobalMemory31 (uint32_t dev, uint64_t addr, void *buf, uint32_t buf_size)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadParamMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readParamMemory);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readParamMemory, "readParamMemory");

    return result;
}

static CUDBGResult
cudbgReadSharedMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readSharedMemory);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readSharedMemory, "readSharedMemory");

    return result;
}

static CUDBGResult
cudbgReadLocalMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readLocalMemory);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readLocalMemory, "readLocalMemory");

    return result;
}

static CUDBGResult
cudbgReadRegister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t *val)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readRegister);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&regno,sizeof(regno));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(val, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readRegister, "readRegister");

    return result;
}

static CUDBGResult
cudbgReadPC (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readPC);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(pc, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readPC, "readPC");

    return result;
}

static CUDBGResult
cudbgReadVirtualPC (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readVirtualPC);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(pc, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readVirtualPC, "readVirtualPC");

    return result;
}

static CUDBGResult
cudbgReadLaneStatus (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, bool *error)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readLaneStatus);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(error, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readLaneStatus, "readLaneStatus");

    return result;
}

static CUDBGResult
STUB_cudbgWriteGlobalMemory31 (uint32_t dev, uint64_t addr, const void *buf, uint32_t buf_size)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgWriteParamMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_writeParamMemory);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));
    CUDBG_IPC_APPEND((char *)buf,buf_size);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_writeParamMemory, "writeParamMemory");

    return result;
}

static CUDBGResult
cudbgWriteSharedMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_writeSharedMemory);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));
    CUDBG_IPC_APPEND((char *)buf,buf_size);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_writeSharedMemory, "writeSharedMemory");

    return result;
}

static CUDBGResult
cudbgWriteLocalMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_writeLocalMemory);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));
    CUDBG_IPC_APPEND((char *)buf,buf_size);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_writeLocalMemory, "writeLocalMemory");

    return result;
}

static CUDBGResult
cudbgWriteUniformRegister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t val)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_writeUniformRegister);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&regno,sizeof(regno));
    CUDBG_IPC_APPEND(&val,sizeof(val));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_writeUniformRegister, "writeUniformRegister");

    return result;
}

static CUDBGResult
cudbgWriteRegister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_writeRegister);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&regno,sizeof(regno));
    CUDBG_IPC_APPEND(&val,sizeof(val));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_writeRegister, "writeRegister");

    return result;
}

static CUDBGResult
STUB_cudbgGetGridDim32 (uint32_t dev, uint32_t sm, uint32_t wp, CuDim2 *gridDim)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgGetBlockDim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockDim)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getBlockDim);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(blockDim, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getBlockDim, "getBlockDim");

    return result;
}

static CUDBGResult
cudbgGetTID (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getTID);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(tid, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getTID, "getTID");

    return result;
}

static CUDBGResult
STUB_cudbgGetElfImage32 (uint32_t dev, uint32_t sm, uint32_t wp, bool relocated, void **elfImage, uint32_t *elfImage_size)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgGetDeviceType (uint32_t dev, char *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getDeviceType);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getDeviceType, "getDeviceType");

    return result;
}

static CUDBGResult
cudbgGetSmType (uint32_t dev, char *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getSmType);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getSmType, "getSmType");

    return result;
}

static CUDBGResult
cudbgGetNumDevices (uint32_t *numDev)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getNumDevices);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(numDev, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getNumDevices, "getNumDevices");

    return result;
}

static CUDBGResult
cudbgGetNumSMs (uint32_t dev, uint32_t *numSMs)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getNumSMs);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(numSMs, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getNumSMs, "getNumSMs");

    return result;
}

static CUDBGResult
cudbgGetNumWarps (uint32_t dev, uint32_t *numWarps)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getNumWarps);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(numWarps, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getNumWarps, "getNumWarps");

    return result;
}

static CUDBGResult
cudbgGetNumLanes (uint32_t dev, uint32_t *numLanes)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getNumLanes);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(numLanes, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getNumLanes, "getNumLanes");

    return result;
}

static CUDBGResult
cudbgGetNumRegisters (uint32_t dev, uint32_t *numRegs)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getNumRegisters);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(numRegs, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getNumRegisters, "getNumRegisters");

    return result;
}

static CUDBGResult
cudbgGetNumUniformRegisters (uint32_t dev, uint32_t *numRegs)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getNumUniformRegisters);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(numRegs, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getNumUniformRegisters, "getNumUniformRegisters");

    return result;
}

static CUDBGResult
STUB_cudbgGetPhysicalRegister30 (uint64_t pc, char *reg, uint32_t *buf, uint32_t buf_size, uint32_t *numPhysRegs, CUDBGRegClass *regClass)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgDisassemble (uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_disassemble);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(instSize, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_disassemble, "disassemble");

    return result;
}

static CUDBGResult
STUB_cudbgIsDeviceCodeAddress55 (uintptr_t addr, bool *isDeviceAddress)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgLookupDeviceCodeSymbol (char *symName, bool *symFound, uintptr_t *symAddr)
{
    /* NOTE: This API method has been implemented incorrectly in RPCD, but as
     * it's left unused, we can't start using it in the future (or we'd break
     * the compatibility with the older drivers). If this method is ever needed
     * in the future, a separate new RPCD call must be added to support it.
     */
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgSetNotifyNewEventCallback31 (CUDBGNotifyNewEventCallback31 callback, void *data)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgGetNextEvent30 (CUDBGEvent30 *event)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgAcknowledgeEvent30 (CUDBGEvent30 *event)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgGetGridAttribute (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGAttribute attr, uint64_t *value)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getGridAttribute);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&attr,sizeof(attr));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(value, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getGridAttribute, "getGridAttribute");

    return result;
}

static CUDBGResult
STUB_cudbgGetGridAttributes (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGAttributeValuePair *pairs, uint32_t numPairs)
{
    /* NOTE: This API method has been implemented incorrectly in RPCD, but as
     * it's left unused, we can't start using it in the future (or we'd break
     * the compatibility with the older drivers). If this method is ever needed
     * in the future, a separate new RPCD call must be added to support it.
     */
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgGetPhysicalRegister40 (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t pc, char *reg, uint32_t *buf, uint32_t buf_size, uint32_t *numPhysRegs, CUDBGRegClass *regClass)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadLaneException (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readLaneException);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(exception, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readLaneException, "readLaneException");

    return result;
}

static CUDBGResult
STUB_cudbgGetNextEvent32 (CUDBGEvent32 *event)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgAcknowledgeEvents42 (void)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgReadCallDepth32 (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *depth)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgReadReturnAddress32 (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t level, uint64_t *ra)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgReadVirtualReturnAddress32 (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t level, uint64_t *ra)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgReadGlobalMemory55 (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t buf_size)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgWriteGlobalMemory55 (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t buf_size)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadPinnedMemory (uint64_t addr, void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readPinnedMemory);
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readPinnedMemory, "readPinnedMemory");

    return result;
}

static CUDBGResult
cudbgWritePinnedMemory (uint64_t addr, const void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_writePinnedMemory);
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));
    CUDBG_IPC_APPEND((char *)buf,buf_size);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_writePinnedMemory, "writePinnedMemory");

    return result;
}

static CUDBGResult
cudbgSetBreakpoint (uint32_t dev, uint64_t addr)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_setBreakpoint);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_setBreakpoint, "setBreakpoint");

    return result;
}

static CUDBGResult
cudbgUnsetBreakpoint (uint32_t dev, uint64_t addr)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_unsetBreakpoint);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_unsetBreakpoint, "unsetBreakpoint");

    return result;
}

static CUDBGResult
STUB_cudbgSetNotifyNewEventCallback40 (CUDBGNotifyNewEventCallback40 callback)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgGetNextEvent42 (CUDBGEvent42 *event)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgReadTextureMemory (uint32_t devId, uint32_t vsm, uint32_t wp, uint32_t id, uint32_t coords_size, uint32_t *coords, void *buf, uint32_t buf_size)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadBlockIdx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readBlockIdx);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(blockIdx, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readBlockIdx, "readBlockIdx");

    return result;
}

static CUDBGResult
cudbgGetGridDim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *gridDim)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getGridDim);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(gridDim, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getGridDim, "getGridDim");

    return result;
}

static CUDBGResult
cudbgReadCallDepth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *depth)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readCallDepth);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(depth, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readCallDepth, "readCallDepth");

    return result;
}

static CUDBGResult
cudbgReadReturnAddress (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t level, uint64_t *ra)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readReturnAddress);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&level,sizeof(level));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(ra, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readReturnAddress, "readReturnAddress");

    return result;
}

static CUDBGResult
cudbgReadVirtualReturnAddress (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t level, uint64_t *ra)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readVirtualReturnAddress);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&level,sizeof(level));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(ra, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readVirtualReturnAddress, "readVirtualReturnAddress");

    return result;
}

static CUDBGResult
STUB_cudbgGetElfImage (uint32_t dev, uint32_t sm, uint32_t wp, bool relocated, void **elfImage, uint64_t *elfImage_size)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgGetHostAddrFromDeviceAddr (uint32_t dev, uint64_t device_addr, uint64_t *host_addr)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getHostAddrFromDeviceAddr);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&device_addr,sizeof(device_addr));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(host_addr, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getHostAddrFromDeviceAddr, "getHostAddrFromDeviceAddr");

    return result;
}

static CUDBGResult
cudbgSingleStepWarp41 (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *warpMask)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_singleStepWarp41);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(warpMask, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_singleStepWarp41, "singleStepWarp41");

    return result;
}

static CUDBGResult
cudbgReadSyscallCallDepth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *depth)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readSyscallCallDepth);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(depth, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readSyscallCallDepth, "readSyscallCallDepth");

    return result;
}

static CUDBGResult
STUB_cudbgReadTextureMemoryBindless (uint32_t devId, uint32_t vsm, uint32_t wp, uint32_t texSymtabIndex, uint32_t coords_size, uint32_t *coords, void *buf, uint32_t buf_size)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgClearAttachState (void)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_clearAttachState);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_clearAttachState, "clearAttachState");

    return result;
}

static CUDBGResult
STUB_cudbgGetNextSyncEvent50 (CUDBGEvent50 *event)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgMemcheckReadErrorAddress (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *address, ptxStorageKind *storage)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_memcheckReadErrorAddress);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(address, &ipc_buf);
    CUDBG_IPC_RECEIVE(storage, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_memcheckReadErrorAddress, "memcheckReadErrorAddress");

    return result;
}

static CUDBGResult
cudbgAcknowledgeSyncEvents (void)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_acknowledgeSyncEvents);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_acknowledgeSyncEvents, "acknowledgeSyncEvents");

    return result;
}

static CUDBGResult
STUB_cudbgGetNextAsyncEvent50 (CUDBGEvent50 *event)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgRequestCleanupOnDetach55 (void)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgInitializeAttachStub (void)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_initializeAttachStub);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_initializeAttachStub, "initializeAttachStub");

    return result;
}

static CUDBGResult
STUB_cudbgGetGridStatus50 (uint32_t dev, uint32_t gridId, CUDBGGridStatus *status)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgGetNextSyncEvent55 (CUDBGEvent55 *event)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgGetNextAsyncEvent55 (CUDBGEvent55 *event)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgGetGridInfo55 (uint32_t dev, uint64_t gridId64, CUDBGGridInfo55 *gridInfo)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadGridId (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *gridId64)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readGridId);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(gridId64, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readGridId, "readGridId");

    return result;
}

static CUDBGResult
cudbgGetGridStatus (uint32_t dev, uint64_t gridId64, CUDBGGridStatus *status)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getGridStatus);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&gridId64,sizeof(gridId64));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(status, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getGridStatus, "getGridStatus");

    return result;
}

static CUDBGResult
cudbgSetKernelLaunchNotificationMode (CUDBGKernelLaunchNotifyMode mode)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_setKernelLaunchNotificationMode);
    CUDBG_IPC_APPEND(&mode,sizeof(mode));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_setKernelLaunchNotificationMode, "setKernelLaunchNotificationMode");

    return result;
}

static CUDBGResult
cudbgGetDevicePCIBusInfo (uint32_t devId, uint32_t *pciBusId, uint32_t *pciDevId)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getDevicePCIBusInfo);
    CUDBG_IPC_APPEND(&devId,sizeof(devId));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(pciBusId, &ipc_buf);
    CUDBG_IPC_RECEIVE(pciDevId, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getDevicePCIBusInfo, "getDevicePCIBusInfo");

    return result;
}

static CUDBGResult
cudbgReadDeviceExceptionState80 (uint32_t devId, uint64_t *exceptionSMMask)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadDeviceExceptionState (uint32_t devId, uint64_t *mask, uint32_t sz)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readDeviceExceptionState);
    CUDBG_IPC_APPEND(&devId,sizeof(devId));
    CUDBG_IPC_APPEND(&sz,sizeof(sz));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(mask, sz, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readDeviceExceptionState, "readDeviceExceptionState");

    return result;
}

static CUDBGResult
cudbgGetAdjustedCodeAddress (uint32_t devId, uint64_t address, uint64_t *adjustedAddress, CUDBGAdjAddrAction adjAction)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getAdjustedCodeAddress);
    CUDBG_IPC_APPEND(&devId,sizeof(devId));
    CUDBG_IPC_APPEND(&address,sizeof(address));
    CUDBG_IPC_APPEND(&adjAction,sizeof(adjAction));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(adjustedAddress, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getAdjustedCodeAddress, "getAdjustedCodeAddress");

    return result;
}

static CUDBGResult
cudbgReadErrorPC (uint32_t devId, uint32_t sm, uint32_t wp, uint64_t *errorPC, bool *errorPCValid)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readErrorPC);
    CUDBG_IPC_APPEND(&devId,sizeof(devId));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(errorPC, &ipc_buf);
    CUDBG_IPC_RECEIVE(errorPCValid, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readErrorPC, "readErrorPC");

    return result;
}

static CUDBGResult
cudbgGetNextEvent (CUDBGEventQueueType type, CUDBGEvent  *event)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getNextEvent);
    CUDBG_IPC_APPEND(&type,sizeof(type));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(event, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getNextEvent, "getNextEvent");

    return result;
}

static CUDBGResult
cudbgGetElfImageByHandle (uint32_t devId, uint64_t handle, CUDBGElfImageType type, void *elfImage, uint64_t elfImage_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getElfImageByHandle);
    CUDBG_IPC_APPEND(&devId,sizeof(devId));
    CUDBG_IPC_APPEND(&handle,sizeof(handle));
    CUDBG_IPC_APPEND(&type,sizeof(type));
    CUDBG_IPC_APPEND(&elfImage_size,sizeof(elfImage_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(elfImage, elfImage_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getElfImageByHandle, "getElfImageByHandle");

    return result;
}

static CUDBGResult
cudbgResumeWarpsUntilPC (uint32_t devId, uint32_t sm, uint64_t warpMask, uint64_t virtPC)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_resumeWarpsUntilPC);
    CUDBG_IPC_APPEND(&devId,sizeof(devId));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&warpMask,sizeof(warpMask));
    CUDBG_IPC_APPEND(&virtPC,sizeof(virtPC));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_resumeWarpsUntilPC, "resumeWarpsUntilPC");

    return result;
}

static CUDBGResult
STUB_cudbgReadWarpState60 (uint32_t devId, uint32_t sm, uint32_t wp, CUDBGWarpState60 *state)
{
    return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadRegisterRange (uint32_t devId, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t index, uint32_t registers_size, uint32_t *registers)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readRegisterRange);
    CUDBG_IPC_APPEND(&devId,sizeof(devId));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&index,sizeof(index));
    CUDBG_IPC_APPEND(&registers_size,sizeof(registers_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(registers, registers_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readRegisterRange, "readRegisterRange");

    return result;
}

static CUDBGResult
cudbgReadUniformRegisterRange (uint32_t devId, uint32_t sm, uint32_t wp, uint32_t index, uint32_t bufSz, uint32_t *buf)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readUniformRegisterRange);
    CUDBG_IPC_APPEND(&devId,sizeof(devId));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&index,sizeof(index));
    CUDBG_IPC_APPEND(&bufSz,sizeof(bufSz));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, bufSz, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readUniformRegisterRange, "readUniformRegisterRange");

    return result;
}

static CUDBGResult
cudbgReadGenericMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readGenericMemory);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readGenericMemory, "readGenericMemory");

    return result;
}

static CUDBGResult
cudbgWriteGenericMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_writeGenericMemory);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));
    CUDBG_IPC_APPEND((char *)buf,buf_size);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_writeGenericMemory, "writeGenericMemory");

    return result;
}

static CUDBGResult
cudbgReadGlobalMemory (uint64_t addr, void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readGlobalMemory);
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readGlobalMemory, "readGlobalMemory");

    return result;
}

static CUDBGResult
cudbgWriteGlobalMemory (uint64_t addr, const void *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_writeGlobalMemory);
    CUDBG_IPC_APPEND(&addr,sizeof(addr));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));
    CUDBG_IPC_APPEND((char *)buf,buf_size);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_writeGlobalMemory, "writeGlobalMemory");

    return result;
}

static CUDBGResult
cudbgGetManagedMemoryRegionInfo (uint64_t startAddress, CUDBGMemoryInfo *memoryInfo, uint32_t memoryInfo_size, uint32_t *numEntries)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getManagedMemoryRegionInfo);
    CUDBG_IPC_APPEND(&startAddress,sizeof(startAddress));
    CUDBG_IPC_APPEND(&memoryInfo_size,sizeof(memoryInfo_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(numEntries, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(memoryInfo, memoryInfo_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getManagedMemoryRegionInfo, "getManagedMemoryRegionInfo");

    return result;
}

static CUDBGResult
cudbgIsDeviceCodeAddress (uintptr_t addr, bool *isDeviceAddress)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_isDeviceCodeAddress);
    CUDBG_IPC_APPEND(&addr,sizeof(addr));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(isDeviceAddress, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_isDeviceCodeAddress, "isDeviceCodeAddress");

    return result;
}

static CUDBGResult
cudbgRequestCleanupOnDetach (uint32_t appResumeFlag)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_requestCleanupOnDetach);
    CUDBG_IPC_APPEND(&appResumeFlag,sizeof(appResumeFlag));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_requestCleanupOnDetach, "requestCleanupOnDetach");

    return result;
}

static CUDBGResult
cudbgReadPredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, uint32_t *predicates)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readPredicates);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&predicates_size,sizeof(predicates_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(predicates, predicates_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readPredicates, "readPredicates");

    return result;
}

static CUDBGResult
cudbgReadUniformPredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, uint32_t *predicates)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readUniformPredicates);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&predicates_size,sizeof(predicates_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(predicates, predicates_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readUniformPredicates, "readUniformPredicates");

    return result;
}

static CUDBGResult
cudbgWritePredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_writePredicates);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&predicates_size,sizeof(predicates_size));
    CUDBG_IPC_APPEND((char *)predicates,predicates_size*sizeof(const uint32_t));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_writePredicates, "writePredicates");

    return result;
}

static CUDBGResult
cudbgWriteUniformPredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, const uint32_t *predicates)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_writeUniformPredicates);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&predicates_size,sizeof(predicates_size));
    CUDBG_IPC_APPEND((char *)predicates,predicates_size*sizeof(const uint32_t));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_writeUniformPredicates, "writeUniformPredicates");

    return result;
}

static CUDBGResult
cudbgGetNumPredicates (uint32_t dev, uint32_t *numPredicates)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getNumPredicates);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(numPredicates, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getNumPredicates, "getNumPredicates");

    return result;
}

static CUDBGResult
cudbgGetNumUniformPredicates (uint32_t dev, uint32_t *numPredicates)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getNumUniformPredicates);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(numPredicates, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getNumUniformPredicates, "getNumUniformPredicates");

    return result;
}

static CUDBGResult
cudbgReadCCRegister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *val)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readCCRegister);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(val, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readCCRegister, "readCCRegister");

    return result;
}

static CUDBGResult
cudbgWriteCCRegister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_writeCCRegister);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&ln,sizeof(ln));
    CUDBG_IPC_APPEND(&val,sizeof(val));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_writeCCRegister, "writeCCRegister");

    return result;
}

static CUDBGResult
cudbgGetDeviceName (uint32_t dev, char *buf, uint32_t buf_size)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getDeviceName);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&buf_size,sizeof(buf_size));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getDeviceName, "getDeviceName");

    return result;
}

#if CUDBG_API_VERSION_REVISION >= 132
static CUDBGResult
cudbgGetLoadedFunctionInfo (uint32_t dev, uint64_t handle, CUDBGLoadedFunctionInfo *info, uint32_t numEntries)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getLoadedFunctionInfo);
    CUDBG_IPC_APPEND(&dev, sizeof(dev));
    CUDBG_IPC_APPEND(&handle, sizeof(handle));
    CUDBG_IPC_APPEND(&numEntries, sizeof(numEntries));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE_ARRAY(info, numEntries, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getLoadedFunctionInfo, "getLoadedFunctionInfo");

    return result;
}
#endif

static CUDBGResult
cudbgSingleStepWarp (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t nsteps, uint64_t *warpMask)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_singleStepWarp);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));
    CUDBG_IPC_APPEND(&nsteps,sizeof(nsteps));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(warpMask, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_singleStepWarp, "singleStepWarp");

    return result;
}

static CUDBGResult
cudbgGetGridInfo (uint32_t dev, uint64_t gridId64, CUDBGGridInfo *gridInfo)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getGridInfo);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&gridId64,sizeof(gridId64));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(gridInfo, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getGridInfo, "getGridInfo");

    return result;
}

static CUDBGResult
cudbgGetClusterDim (uint32_t dev, uint64_t gridId64, CuDim3 *clusterDim)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getClusterDim);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&gridId64,sizeof(gridId64));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(clusterDim, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getClusterDim, "getClusterDim");

    return result;
}

static CUDBGResult
cudbgReadWarpState (uint32_t devId, uint32_t sm, uint32_t wp, CUDBGWarpState *state)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readWarpState);
    CUDBG_IPC_APPEND(&devId,sizeof(devId));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(state, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readWarpState, "readWarpState");

    return result;
}

static CUDBGResult
cudbgReadClusterIdx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *clusterIdx)
{
    char *ipc_buf;
    CUDBGResult result;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_readClusterIdx);
    CUDBG_IPC_APPEND(&dev,sizeof(dev));
    CUDBG_IPC_APPEND(&sm,sizeof(sm));
    CUDBG_IPC_APPEND(&wp,sizeof(wp));

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    CUDBG_IPC_RECEIVE(clusterIdx, &ipc_buf);

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_readClusterIdx, "readClusterIdx");

    return result;
}

static CUDBGResult
cudbgGetErrorStringEx (char *buf, uint32_t buf_size, uint32_t *msg_size)
{
    char *ipc_buf;
    CUDBGResult result;
    uint32_t err_msg_size;

    CUDBG_IPC_PROFILE_START();

    CUDBG_IPC_BEGIN(CUDBGAPIREQ_getErrorStringEx);

    CUDBG_IPC_REQUEST((void **)&ipc_buf);
    CUDBG_IPC_RECEIVE(&result, &ipc_buf);
    /* The err_msg_size could be in range from 0 to a positive number greater than buf_size */
    CUDBG_IPC_RECEIVE(&err_msg_size, &ipc_buf);
    /* The array size in the packet equals to the err_msg_size */
    CUDBG_IPC_RECEIVE_ARRAY(buf, buf_size > err_msg_size ? err_msg_size : buf_size, &ipc_buf);

    if (result != CUDBG_SUCCESS && result != CUDBG_ERROR_BUFFER_TOO_SMALL)
        err_msg_size = 0;

    if (msg_size)
        *msg_size = err_msg_size;

    if (buf_size > err_msg_size) {
        buf[err_msg_size > 0 ? err_msg_size - 1 : 0] = 0;
    }
    else {
        buf[buf_size > 0 ? buf_size - 1 : 0] = 0;
        ipc_buf += (err_msg_size - buf_size);
    }

    CUDBG_IPC_PROFILE_END(CUDBGAPIREQ_getErrorStringEx, "getErrorStringEx");

    return result;
}

static const struct CUDBGAPI_st cudbgCurrentApi={
    /* Initialization */
    cudbgInitialize,
    cudbgFinalize,

    /* Device Execution Control */
    cudbgSuspendDevice,
    cudbgResumeDevice,
    STUB_cudbgSingleStepWarp40,

    /* Breakpoints */
    STUB_cudbgSetBreakpoint31,
    STUB_cudbgUnsetBreakpoint31,

    /* Device State Inspection */
    STUB_cudbgReadGridId50,
    STUB_cudbgReadBlockIdx32,
    cudbgReadThreadIdx,
    cudbgReadBrokenWarps,
    cudbgReadValidWarps,
    cudbgReadValidLanes,
    cudbgReadActiveLanes,
    cudbgReadCodeMemory,
    cudbgReadConstMemory,
    STUB_cudbgReadGlobalMemory31,
    cudbgReadParamMemory,
    cudbgReadSharedMemory,
    cudbgReadLocalMemory,
    cudbgReadRegister,
    cudbgReadPC,
    cudbgReadVirtualPC,
    cudbgReadLaneStatus,

    /* Device State Alteration */
    STUB_cudbgWriteGlobalMemory31,
    cudbgWriteParamMemory,
    cudbgWriteSharedMemory,
    cudbgWriteLocalMemory,
    cudbgWriteRegister,

    /* Grid Properties */
    STUB_cudbgGetGridDim32,
    cudbgGetBlockDim,
    cudbgGetTID,
    STUB_cudbgGetElfImage32,

    /* Device Properties */
    cudbgGetDeviceType,
    cudbgGetSmType,
    cudbgGetNumDevices,
    cudbgGetNumSMs,
    cudbgGetNumWarps,
    cudbgGetNumLanes,
    cudbgGetNumRegisters,

    /* DWARF-related routines */
    STUB_cudbgGetPhysicalRegister30,
    cudbgDisassemble,
    STUB_cudbgIsDeviceCodeAddress55,
    STUB_cudbgLookupDeviceCodeSymbol,

    /* Events */
    STUB_cudbgSetNotifyNewEventCallback31,
    STUB_cudbgGetNextEvent30,
    STUB_cudbgAcknowledgeEvent30,

    /* 3.1 Extensions */
    cudbgGetGridAttribute,
    STUB_cudbgGetGridAttributes,
    STUB_cudbgGetPhysicalRegister40,
    cudbgReadLaneException,
    STUB_cudbgGetNextEvent32,
    STUB_cudbgAcknowledgeEvents42,

    /* 3.1 - ABI */
    STUB_cudbgReadCallDepth32,
    STUB_cudbgReadReturnAddress32,
    STUB_cudbgReadVirtualReturnAddress32,

    /* 3.2 Extensions */
    STUB_cudbgReadGlobalMemory55,
    STUB_cudbgWriteGlobalMemory55,
    cudbgReadPinnedMemory,
    cudbgWritePinnedMemory,
    cudbgSetBreakpoint,
    cudbgUnsetBreakpoint,
    STUB_cudbgSetNotifyNewEventCallback40,

    /* 4.0 Extensions */
    STUB_cudbgGetNextEvent42,
    STUB_cudbgReadTextureMemory,
    cudbgReadBlockIdx,
    cudbgGetGridDim,
    cudbgReadCallDepth,
    cudbgReadReturnAddress,
    cudbgReadVirtualReturnAddress,
    STUB_cudbgGetElfImage,

    /* 4.1 Extensions */
    cudbgGetHostAddrFromDeviceAddr,
    cudbgSingleStepWarp41,
    cudbgSetNotifyNewEventCallback,
    cudbgReadSyscallCallDepth,

    /* 4.2 Extensions */
    STUB_cudbgReadTextureMemoryBindless,

    /* 5.0 Extensions */
    cudbgClearAttachState,
    STUB_cudbgGetNextSyncEvent50,
    cudbgMemcheckReadErrorAddress,
    cudbgAcknowledgeSyncEvents,
    STUB_cudbgGetNextAsyncEvent50,
    STUB_cudbgRequestCleanupOnDetach55,
    cudbgInitializeAttachStub,
    STUB_cudbgGetGridStatus50,

    /* 5.5 Extensions */
    STUB_cudbgGetNextSyncEvent55,
    STUB_cudbgGetNextAsyncEvent55,
    STUB_cudbgGetGridInfo55,
    cudbgReadGridId,
    cudbgGetGridStatus,
    cudbgSetKernelLaunchNotificationMode,
    cudbgGetDevicePCIBusInfo,
    cudbgReadDeviceExceptionState80,

    /* 6.0 Extensions */
    cudbgGetAdjustedCodeAddress,
    cudbgReadErrorPC,
    cudbgGetNextEvent,
    cudbgGetElfImageByHandle,
    cudbgResumeWarpsUntilPC,
    STUB_cudbgReadWarpState60,
    cudbgReadRegisterRange,
    cudbgReadGenericMemory,
    cudbgWriteGenericMemory,
    cudbgReadGlobalMemory,
    cudbgWriteGlobalMemory,
    cudbgGetManagedMemoryRegionInfo,
    cudbgIsDeviceCodeAddress,
    cudbgRequestCleanupOnDetach,

    /* 6.5 Extensions */
    cudbgReadPredicates,
    cudbgWritePredicates,
    cudbgGetNumPredicates,
    cudbgReadCCRegister,
    cudbgWriteCCRegister,

    cudbgGetDeviceName,
    cudbgSingleStepWarp,

    /* 9.0 Extensions */
    cudbgReadDeviceExceptionState,

    /* 10.0 Extensions */
    cudbgGetNumUniformRegisters,
    cudbgReadUniformRegisterRange,
    cudbgWriteUniformRegister,
    cudbgGetNumUniformPredicates,
    cudbgReadUniformPredicates,
    cudbgWriteUniformPredicates,

#if CUDBG_API_VERSION_REVISION >= 132
    /* 11.8 Extensions */
    cudbgGetLoadedFunctionInfo,
#endif

    cudbgGetGridInfo,
    cudbgGetClusterDim,
    cudbgReadWarpState,
    cudbgReadClusterIdx,

    /* 12.2 Extensions */
    cudbgGetErrorStringEx,
};

CUDBGResult
cudbgGetAPI(uint32_t major, uint32_t minor, uint32_t rev, CUDBGAPI *api)
{
    *api = &cudbgCurrentApi;
    return CUDBG_SUCCESS;
}

ATTRIBUTE_PRINTF(1, 2) static void
cudbg_trace(const char *fmt, ...)
{
#ifdef GDBSERVER
  struct cuda_trace_msg *msg;
#endif
  va_list ap;

  if (!cuda_options_debug_libcudbg())
    return;

  va_start (ap, fmt);
#ifdef GDBSERVER
  msg = (struct cuda_trace_msg *)xmalloc (sizeof (*msg));
  if (!cuda_first_trace_msg)
    cuda_first_trace_msg = msg;
  else
    cuda_last_trace_msg->next = msg;
  sprintf (msg->buf, "[CUDAGDB] libcudbg ");
  vsnprintf (msg->buf + strlen (msg->buf), sizeof (msg->buf), fmt, ap);
  msg->next = NULL;
  cuda_last_trace_msg = msg;
#else
  fprintf (stderr, "[CUDAGDB] libcudbg ");
  vfprintf (stderr, fmt, ap);
  fprintf (stderr, "\n");
  fflush (stderr);
#endif
}
