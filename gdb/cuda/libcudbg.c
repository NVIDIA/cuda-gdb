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

/* Set this to true to dump a libcudbg trace (ipc.log) */
#define TRACE_LIBCUDBG 0

#ifdef GDBSERVER
#include "cuda-tdep-server.h"
#include "server.h"
#include <ansidecl.h>
#else
#include "defs.h"

#include "cuda/cuda-api.h"
#include "cuda/cuda-options.h"
#include "cuda/cuda-tdep.h"
#endif
#include "cuda/cuda-version.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>

#include "cuda/libcudbg.h"
#include "cuda/libcudbgipc.h"
#if TRACE_LIBCUDBG
#include "libcudbgtrace.h"
#endif
#include <cudadebugger.h>

/*Forward declarations */
static void cudbg_trace (const char *fmt, ...) ATTRIBUTE_PRINTF (1, 2);

#ifdef GDBSERVER
extern void cuda_gdbserver_set_api_version (uint32_t major, uint32_t minor,
					    uint32_t revision);
#endif

/* Globals */
extern CUDBGNotifyNewEventCallback cudbgDebugClientCallback;

static CUDBGResult
cudbgInitialize (void)
{
  CUDBGResult initRes, result;
  uint32_t major = CUDBG_API_VERSION_MAJOR;
  uint32_t minor = CUDBG_API_VERSION_MINOR;
  uint32_t revision = CUDBG_API_VERSION_REVISION;
  char *ipc_buf;

  initRes = cudbgipcInitialize ();
  if (initRes != CUDBG_SUCCESS)
    {
      cudbg_trace ("IPC initialization failed (res=%d)", initRes);
      return initRes;
    }

#if TRACE_LIBCUDBG
  cudbgOpenIpcTraceFile ();
#endif

  cudbg_trace ("pre initialization successful");
  CUDBG_IPC_BEGIN (CUDBGAPIREQ_initialize);
  CUDBG_IPC_APPEND (&major, sizeof (minor));
  CUDBG_IPC_APPEND (&minor, sizeof (minor));
  CUDBG_IPC_APPEND (&revision, sizeof (revision));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (&major, &ipc_buf);
  CUDBG_IPC_RECEIVE (&minor, &ipc_buf);
  CUDBG_IPC_RECEIVE (&revision, &ipc_buf);

  if (result == CUDBG_ERROR_INCOMPATIBLE_API)
    {
      /* Allow newer cuda-gdb to work with driver from 11.x release. For more
	 details see -
	 https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility
      */
      if (revision < 133)
	/* DEBUG API REVISION for 12.0 was 133 */
	printf ("Incompatible CUDA driver version."
		" Expected %d.%d.%d or later and found %d.%d.%d instead.",
		CUDBG_API_VERSION_MAJOR, CUDBG_API_VERSION_MINOR,
		CUDBG_API_VERSION_REVISION, (int)major, (int)minor,
		(int)revision);
      else
	{
	  /* Retry with the version supported by driver */
	  CUDBG_IPC_BEGIN (CUDBGAPIREQ_initialize);
	  CUDBG_IPC_APPEND (&major, sizeof (minor));
	  CUDBG_IPC_APPEND (&minor, sizeof (minor));
	  CUDBG_IPC_APPEND (&revision, sizeof (revision));

	  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
	  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
	  CUDBG_IPC_RECEIVE (&major, &ipc_buf);
	  CUDBG_IPC_RECEIVE (&minor, &ipc_buf);
	  CUDBG_IPC_RECEIVE (&revision, &ipc_buf);
	}
    }

#ifdef GDBSERVER
  cuda_gdbserver_set_api_version (major, minor, revision);
#else
  cuda_debugapi::set_api_version (major, minor, revision);
#endif

  return result;
}

static CUDBGResult
cudbgSetNotifyNewEventCallback (CUDBGNotifyNewEventCallback callback)
{
  cudbgDebugClientCallback = callback;

  return CUDBG_SUCCESS;
}

static CUDBGResult
cudbgFinalize (void)
{
  char *ipc_buf;
  CUDBGResult result, ipcres;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_finalize);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

#if TRACE_LIBCUDBG
  cudbgCloseIpcTraceFile ();
#endif

  ipcres = cudbgipcFinalize ();
  if (ipcres != CUDBG_SUCCESS)
    cudbg_trace ("IPC finalize failed (res=%d)", ipcres);

  return result == CUDBG_SUCCESS ? ipcres : result;
}

static CUDBGResult
cudbgSuspendDevice (uint32_t dev)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_suspendDevice);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgResumeDevice (uint32_t dev)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_resumeDevice);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

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
STUB_cudbgReadGridId50 (uint32_t dev, uint32_t sm, uint32_t wp,
			uint32_t *gridId)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgReadBlockIdx32 (uint32_t dev, uint32_t sm, uint32_t wp,
			  CuDim2 *blockIdx)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadThreadIdx (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		    CuDim3 *threadIdx)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readThreadIdx);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (threadIdx, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadBrokenWarps (uint32_t dev, uint32_t sm, uint64_t *brokenWarpsMask)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readBrokenWarps);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (brokenWarpsMask, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadValidWarps (uint32_t dev, uint32_t sm, uint64_t *validWarpsMask)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readValidWarps);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (validWarpsMask, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadValidLanes (uint32_t dev, uint32_t sm, uint32_t wp,
		     uint32_t *validLanesMask)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readValidLanes);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (validLanesMask, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadActiveLanes (uint32_t dev, uint32_t sm, uint32_t wp,
		      uint32_t *activeLanesMask)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readActiveLanes);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (activeLanesMask, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadCodeMemory (uint32_t dev, uint64_t addr, void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readCodeMemory);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadConstMemory (uint32_t dev, uint64_t addr, void *buf,
		      uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readConstMemory);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgReadGlobalMemory31 (uint32_t dev, uint64_t addr, void *buf,
			      uint32_t buf_size)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadParamMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr,
		      void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readParamMemory);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadSharedMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr,
		       void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readSharedMemory);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadLocalMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		      uint64_t addr, void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readLocalMemory);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadRegister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		   uint32_t regno, uint32_t *val)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readRegister);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&regno, sizeof (regno));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (val, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadPC (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readPC);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (pc, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadVirtualPC (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		    uint64_t *pc)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readVirtualPC);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (pc, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadLaneStatus (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		     bool *error)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readLaneStatus);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (error, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgWriteGlobalMemory31 (uint32_t dev, uint64_t addr, const void *buf,
			       uint32_t buf_size)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgWriteParamMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr,
		       const void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_writeParamMemory);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));
  CUDBG_IPC_APPEND ((char *)buf, buf_size);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgWriteSharedMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr,
			const void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_writeSharedMemory);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));
  CUDBG_IPC_APPEND ((char *)buf, buf_size);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgWriteLocalMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		       uint64_t addr, const void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_writeLocalMemory);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));
  CUDBG_IPC_APPEND ((char *)buf, buf_size);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgWriteUniformRegister (uint32_t dev, uint32_t sm, uint32_t wp,
			   uint32_t regno, uint32_t val)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_writeUniformRegister);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&regno, sizeof (regno));
  CUDBG_IPC_APPEND (&val, sizeof (val));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgWriteRegister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		    uint32_t regno, uint32_t val)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_writeRegister);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&regno, sizeof (regno));
  CUDBG_IPC_APPEND (&val, sizeof (val));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgGetGridDim32 (uint32_t dev, uint32_t sm, uint32_t wp,
			CuDim2 *gridDim)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgGetBlockDim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockDim)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getBlockDim);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (blockDim, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetTID (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getTID);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (tid, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgGetElfImage32 (uint32_t dev, uint32_t sm, uint32_t wp,
			 bool relocated, void **elfImage,
			 uint32_t *elfImage_size)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgGetDeviceType (uint32_t dev, char *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getDeviceType);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetSmType (uint32_t dev, char *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getSmType);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetNumDevices (uint32_t *numDev)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getNumDevices);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (numDev, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetNumSMs (uint32_t dev, uint32_t *numSMs)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getNumSMs);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (numSMs, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetNumWarps (uint32_t dev, uint32_t *numWarps)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getNumWarps);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (numWarps, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetNumLanes (uint32_t dev, uint32_t *numLanes)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getNumLanes);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (numLanes, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetNumRegisters (uint32_t dev, uint32_t *numRegs)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getNumRegisters);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (numRegs, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetNumUniformRegisters (uint32_t dev, uint32_t *numRegs)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getNumUniformRegisters);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (numRegs, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgGetPhysicalRegister30 (uint64_t pc, char *reg, uint32_t *buf,
				 uint32_t buf_size, uint32_t *numPhysRegs,
				 CUDBGRegClass *regClass)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgDisassemble (uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf,
		  uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_disassemble);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (instSize, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgIsDeviceCodeAddress55 (uintptr_t addr, bool *isDeviceAddress)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgLookupDeviceCodeSymbol (char *symName, bool *symFound,
				  uintptr_t *symAddr)
{
  /* NOTE: This API method has been implemented incorrectly in RPCD, but as
   * it's left unused, we can't start using it in the future (or we'd break
   * the compatibility with the older drivers). If this method is ever needed
   * in the future, a separate new RPCD call must be added to support it.
   */
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgSetNotifyNewEventCallback31 (CUDBGNotifyNewEventCallback31 callback,
				       void *data)
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
cudbgGetGridAttribute (uint32_t dev, uint32_t sm, uint32_t wp,
		       CUDBGAttribute attr, uint64_t *value)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getGridAttribute);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&attr, sizeof (attr));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (value, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgGetGridAttributes (uint32_t dev, uint32_t sm, uint32_t wp,
			     CUDBGAttributeValuePair *pairs, uint32_t numPairs)
{
  /* NOTE: This API method has been implemented incorrectly in RPCD, but as
   * it's left unused, we can't start using it in the future (or we'd break
   * the compatibility with the older drivers). If this method is ever needed
   * in the future, a separate new RPCD call must be added to support it.
   */
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgGetPhysicalRegister40 (uint32_t dev, uint32_t sm, uint32_t wp,
				 uint64_t pc, char *reg, uint32_t *buf,
				 uint32_t buf_size, uint32_t *numPhysRegs,
				 CUDBGRegClass *regClass)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadLaneException (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
			CUDBGException_t *exception)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readLaneException);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (exception, &ipc_buf);

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
STUB_cudbgReadCallDepth32 (uint32_t dev, uint32_t sm, uint32_t wp,
			   uint32_t *depth)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgReadReturnAddress32 (uint32_t dev, uint32_t sm, uint32_t wp,
			       uint32_t level, uint64_t *ra)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgReadVirtualReturnAddress32 (uint32_t dev, uint32_t sm, uint32_t wp,
				      uint32_t level, uint64_t *ra)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgReadGlobalMemory55 (uint32_t dev, uint32_t sm, uint32_t wp,
			      uint32_t ln, uint64_t addr, void *buf,
			      uint32_t buf_size)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
STUB_cudbgWriteGlobalMemory55 (uint32_t dev, uint32_t sm, uint32_t wp,
			       uint32_t ln, uint64_t addr, const void *buf,
			       uint32_t buf_size)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadPinnedMemory (uint64_t addr, void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readPinnedMemory);
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgWritePinnedMemory (uint64_t addr, const void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_writePinnedMemory);
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));
  CUDBG_IPC_APPEND ((char *)buf, buf_size);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgSetBreakpoint (uint32_t dev, uint64_t addr)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_setBreakpoint);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgUnsetBreakpoint (uint32_t dev, uint64_t addr)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_unsetBreakpoint);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

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
STUB_cudbgReadTextureMemory (uint32_t devId, uint32_t vsm, uint32_t wp,
			     uint32_t id, uint32_t coords_size,
			     uint32_t *coords, void *buf, uint32_t buf_size)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadBlockIdx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readBlockIdx);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (blockIdx, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetGridDim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *gridDim)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getGridDim);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (gridDim, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadCallDepth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		    uint32_t *depth)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readCallDepth);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (depth, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadReturnAddress (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
			uint32_t level, uint64_t *ra)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readReturnAddress);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&level, sizeof (level));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (ra, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadVirtualReturnAddress (uint32_t dev, uint32_t sm, uint32_t wp,
			       uint32_t ln, uint32_t level, uint64_t *ra)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readVirtualReturnAddress);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&level, sizeof (level));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (ra, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgGetElfImage (uint32_t dev, uint32_t sm, uint32_t wp, bool relocated,
		       void **elfImage, uint64_t *elfImage_size)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgGetHostAddrFromDeviceAddr (uint32_t dev, uint64_t device_addr,
				uint64_t *host_addr)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getHostAddrFromDeviceAddr);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&device_addr, sizeof (device_addr));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (host_addr, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgSingleStepWarp41 (uint32_t dev, uint32_t sm, uint32_t wp,
		       uint64_t *warpMask)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_singleStepWarp41);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (warpMask, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadSyscallCallDepth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
			   uint32_t *depth)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readSyscallCallDepth);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (depth, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgReadTextureMemoryBindless (uint32_t devId, uint32_t vsm, uint32_t wp,
				     uint32_t texSymtabIndex,
				     uint32_t coords_size, uint32_t *coords,
				     void *buf, uint32_t buf_size)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgClearAttachState (void)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_clearAttachState);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgGetNextSyncEvent50 (CUDBGEvent50 *event)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgMemcheckReadErrorAddress (uint32_t dev, uint32_t sm, uint32_t wp,
			       uint32_t ln, uint64_t *address,
			       ptxStorageKind *storage)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_memcheckReadErrorAddress);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (address, &ipc_buf);
  CUDBG_IPC_RECEIVE (storage, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgAcknowledgeSyncEvents (void)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_acknowledgeSyncEvents);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

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

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_initializeAttachStub);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgGetGridStatus50 (uint32_t dev, uint32_t gridId,
			   CUDBGGridStatus *status)
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
STUB_cudbgGetGridInfo55 (uint32_t dev, uint64_t gridId64,
			 CUDBGGridInfo55 *gridInfo)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadGridId (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *gridId64)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readGridId);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (gridId64, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetGridStatus (uint32_t dev, uint64_t gridId64, CUDBGGridStatus *status)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getGridStatus);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&gridId64, sizeof (gridId64));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (status, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgSetKernelLaunchNotificationMode (CUDBGKernelLaunchNotifyMode mode)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_setKernelLaunchNotificationMode);
  CUDBG_IPC_APPEND (&mode, sizeof (mode));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetDevicePCIBusInfo (uint32_t devId, uint32_t *pciBusId,
			  uint32_t *pciDevId)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getDevicePCIBusInfo);
  CUDBG_IPC_APPEND (&devId, sizeof (devId));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (pciBusId, &ipc_buf);
  CUDBG_IPC_RECEIVE (pciDevId, &ipc_buf);

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

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readDeviceExceptionState);
  CUDBG_IPC_APPEND (&devId, sizeof (devId));
  CUDBG_IPC_APPEND (&sz, sizeof (sz));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (mask, sz, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetAdjustedCodeAddress (uint32_t devId, uint64_t address,
			     uint64_t *adjustedAddress,
			     CUDBGAdjAddrAction adjAction)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getAdjustedCodeAddress);
  CUDBG_IPC_APPEND (&devId, sizeof (devId));
  CUDBG_IPC_APPEND (&address, sizeof (address));
  CUDBG_IPC_APPEND (&adjAction, sizeof (adjAction));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (adjustedAddress, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadErrorPC (uint32_t devId, uint32_t sm, uint32_t wp, uint64_t *errorPC,
		  bool *errorPCValid)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readErrorPC);
  CUDBG_IPC_APPEND (&devId, sizeof (devId));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (errorPC, &ipc_buf);
  CUDBG_IPC_RECEIVE (errorPCValid, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetNextEvent (CUDBGEventQueueType type, CUDBGEvent *event)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getNextEvent);
  CUDBG_IPC_APPEND (&type, sizeof (type));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (event, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetElfImageByHandle (uint32_t devId, uint64_t handle,
			  CUDBGElfImageType type, void *elfImage,
			  uint64_t elfImage_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getElfImageByHandle);
  CUDBG_IPC_APPEND (&devId, sizeof (devId));
  CUDBG_IPC_APPEND (&handle, sizeof (handle));
  CUDBG_IPC_APPEND (&type, sizeof (type));
  CUDBG_IPC_APPEND (&elfImage_size, sizeof (elfImage_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (elfImage, elfImage_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgResumeWarpsUntilPC (uint32_t devId, uint32_t sm, uint64_t warpMask,
			 uint64_t virtPC)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_resumeWarpsUntilPC);
  CUDBG_IPC_APPEND (&devId, sizeof (devId));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&warpMask, sizeof (warpMask));
  CUDBG_IPC_APPEND (&virtPC, sizeof (virtPC));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
STUB_cudbgReadWarpState60 (uint32_t devId, uint32_t sm, uint32_t wp,
			   CUDBGWarpState60 *state)
{
  return CUDBG_ERROR_UNKNOWN;
}

static CUDBGResult
cudbgReadRegisterRange (uint32_t devId, uint32_t sm, uint32_t wp, uint32_t ln,
			uint32_t index, uint32_t registers_size,
			uint32_t *registers)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readRegisterRange);
  CUDBG_IPC_APPEND (&devId, sizeof (devId));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&index, sizeof (index));
  CUDBG_IPC_APPEND (&registers_size, sizeof (registers_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (registers, registers_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadUniformRegisterRange (uint32_t devId, uint32_t sm, uint32_t wp,
			       uint32_t index, uint32_t bufSz, uint32_t *buf)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readUniformRegisterRange);
  CUDBG_IPC_APPEND (&devId, sizeof (devId));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&index, sizeof (index));
  CUDBG_IPC_APPEND (&bufSz, sizeof (bufSz));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, bufSz, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadGenericMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
			uint64_t addr, void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readGenericMemory);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgWriteGenericMemory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
			 uint64_t addr, const void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_writeGenericMemory);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));
  CUDBG_IPC_APPEND ((char *)buf, buf_size);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadGlobalMemory (uint64_t addr, void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readGlobalMemory);
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgWriteGlobalMemory (uint64_t addr, const void *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_writeGlobalMemory);
  CUDBG_IPC_APPEND (&addr, sizeof (addr));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));
  CUDBG_IPC_APPEND ((char *)buf, buf_size);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetManagedMemoryRegionInfo (uint64_t startAddress,
				 CUDBGMemoryInfo *memoryInfo,
				 uint32_t memoryInfo_size,
				 uint32_t *numEntries)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getManagedMemoryRegionInfo);
  CUDBG_IPC_APPEND (&startAddress, sizeof (startAddress));
  CUDBG_IPC_APPEND (&memoryInfo_size, sizeof (memoryInfo_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (numEntries, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (memoryInfo, memoryInfo_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgIsDeviceCodeAddress (uintptr_t addr, bool *isDeviceAddress)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_isDeviceCodeAddress);
  CUDBG_IPC_APPEND (&addr, sizeof (addr));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (isDeviceAddress, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgRequestCleanupOnDetach (uint32_t appResumeFlag)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_requestCleanupOnDetach);
  CUDBG_IPC_APPEND (&appResumeFlag, sizeof (appResumeFlag));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadPredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		     uint32_t predicates_size, uint32_t *predicates)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readPredicates);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&predicates_size, sizeof (predicates_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (predicates, predicates_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadUniformPredicates (uint32_t dev, uint32_t sm, uint32_t wp,
			    uint32_t predicates_size, uint32_t *predicates)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readUniformPredicates);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&predicates_size, sizeof (predicates_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (predicates, predicates_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgWritePredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		      uint32_t predicates_size, const uint32_t *predicates)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_writePredicates);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&predicates_size, sizeof (predicates_size));
  CUDBG_IPC_APPEND ((char *)predicates,
		    predicates_size * sizeof (const uint32_t));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgWriteUniformPredicates (uint32_t dev, uint32_t sm, uint32_t wp,
			     uint32_t predicates_size,
			     const uint32_t *predicates)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_writeUniformPredicates);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&predicates_size, sizeof (predicates_size));
  CUDBG_IPC_APPEND ((char *)predicates,
		    predicates_size * sizeof (const uint32_t));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetNumPredicates (uint32_t dev, uint32_t *numPredicates)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getNumPredicates);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (numPredicates, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetNumUniformPredicates (uint32_t dev, uint32_t *numPredicates)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getNumUniformPredicates);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (numPredicates, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadCCRegister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		     uint32_t *val)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readCCRegister);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (val, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgWriteCCRegister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
		      uint32_t val)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_writeCCRegister);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));
  CUDBG_IPC_APPEND (&val, sizeof (val));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetDeviceName (uint32_t dev, char *buf, uint32_t buf_size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getDeviceName);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&buf_size, sizeof (buf_size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (buf, buf_size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetLoadedFunctionInfo118 (uint32_t dev, uint64_t handle,
			       CUDBGLoadedFunctionInfo *info,
			       uint32_t numEntries)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getLoadedFunctionInfo118);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&handle, sizeof (handle));
  CUDBG_IPC_APPEND (&numEntries, sizeof (numEntries));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (info, numEntries, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgSingleStepWarp65 (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t nsteps,
		       uint64_t *warpMask)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_singleStepWarp65);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&nsteps, sizeof (nsteps));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (warpMask, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetGridInfo120 (uint32_t dev, uint64_t gridId64,
		     CUDBGGridInfo120 *gridInfo)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getGridInfo120);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&gridId64, sizeof (gridId64));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (gridInfo, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetClusterDim120 (uint32_t dev, uint64_t gridId64, CuDim3 *clusterDim)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getClusterDim120);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&gridId64, sizeof (gridId64));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (clusterDim, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadWarpState120 (uint32_t devId, uint32_t sm, uint32_t wp,
		       CUDBGWarpState120 *state)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readWarpState120);
  CUDBG_IPC_APPEND (&devId, sizeof (devId));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (state, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadClusterIdx (uint32_t dev, uint32_t sm, uint32_t wp,
		     CuDim3 *clusterIdx)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readClusterIdx);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (clusterIdx, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetErrorStringEx (char *buf, uint32_t buf_size, uint32_t *msg_size)
{
  char *ipc_buf;
  CUDBGResult result;
  uint32_t err_msg_size;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getErrorStringEx);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  /* The err_msg_size could be in range from 0 to a positive number greater
   * than buf_size */
  CUDBG_IPC_RECEIVE (&err_msg_size, &ipc_buf);
  /* The array size in the packet equals to the err_msg_size */
  CUDBG_IPC_RECEIVE_ARRAY (
      buf, buf_size > err_msg_size ? err_msg_size : buf_size, &ipc_buf);

  if (result != CUDBG_SUCCESS && result != CUDBG_ERROR_BUFFER_TOO_SMALL)
    err_msg_size = 0;

  if (msg_size)
    *msg_size = err_msg_size;

  if (buf_size > err_msg_size)
    {
      buf[err_msg_size > 0 ? err_msg_size - 1 : 0] = 0;
    }
  else
    {
      buf[buf_size > 0 ? buf_size - 1 : 0] = 0;
      ipc_buf += (err_msg_size - buf_size);
    }

  return result;
}

static CUDBGResult
cudbgGetLoadedFunctionInfo (uint32_t dev, uint64_t handle,
			    CUDBGLoadedFunctionInfo *info, uint32_t startIndex,
			    uint32_t numEntries)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getLoadedFunctionInfo);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&handle, sizeof (handle));
  CUDBG_IPC_APPEND (&startIndex, sizeof (startIndex));
  CUDBG_IPC_APPEND (&numEntries, sizeof (numEntries));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE_ARRAY (info, numEntries, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGenerateCoredump (const char *filename,
		       CUDBGCoredumpGenerationFlags flags)
{
  uint32_t filename_size = filename ? strlen (filename) + 1 : 0;
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_generateCoredump);
  CUDBG_IPC_APPEND (&flags, sizeof (flags));
  CUDBG_IPC_APPEND (&filename_size, sizeof (filename_size));

  if (filename)
    CUDBG_IPC_APPEND (filename, filename_size);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetConstBankAddress123 (uint32_t dev, uint32_t sm, uint32_t wp,
			     uint32_t bank, uint32_t offset, uint64_t *address)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getConstBankAddress123);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&bank, sizeof (bank));
  CUDBG_IPC_APPEND (&offset, sizeof (offset));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (address, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetDeviceInfoSizes (uint32_t dev, CUDBGDeviceInfoSizes *sizes)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getDeviceInfoSizes);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  if (result == CUDBG_SUCCESS)
    CUDBG_IPC_RECEIVE (sizes, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetDeviceInfo (uint32_t dev, CUDBGDeviceInfoQueryType_t type,
		    void *buffer, uint32_t size, uint32_t *data_length)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getDeviceInfo);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&type, sizeof (type));
  CUDBG_IPC_APPEND (&size, sizeof (size));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  if (result == CUDBG_SUCCESS)
    {
      CUDBG_IPC_RECEIVE (data_length, &ipc_buf);
      CUDBG_IPC_RECEIVE_ARRAY ((uint8_t *)buffer, *data_length, &ipc_buf);
    }

  return result;
}

static CUDBGResult
cudbgGetConstBankAddress (uint32_t dev, uint64_t gridId64, uint32_t bank,
			  uint64_t *address, uint32_t *size)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getConstBankAddress);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&gridId64, sizeof (gridId64));
  CUDBG_IPC_APPEND (&bank, sizeof (bank));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (address, &ipc_buf);
  CUDBG_IPC_RECEIVE (size, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgSingleStepWarp (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t laneHint,
		     uint32_t nsteps, uint32_t flags, uint64_t *warpMask)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_singleStepWarp);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&laneHint, sizeof (laneHint));
  CUDBG_IPC_APPEND (&nsteps, sizeof (nsteps));
  CUDBG_IPC_APPEND (&flags, sizeof (flags));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (warpMask, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadAllVirtualReturnAddresses (uint32_t dev, uint32_t sm, uint32_t wp,
				    uint32_t ln, uint64_t *addrs,
				    uint32_t numAddrs, uint32_t *callDepth,
				    uint32_t *syscallCallDepth)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readAllVirtualReturnAddresses);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));
  CUDBG_IPC_APPEND (&ln, sizeof (ln));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (callDepth, &ipc_buf);
  CUDBG_IPC_RECEIVE (syscallCallDepth, &ipc_buf);
  if (numAddrs < *callDepth)
    {
      CUDBG_IPC_RECEIVE_ARRAY (addrs, numAddrs, &ipc_buf);
    }
  else
    {
      CUDBG_IPC_RECEIVE_ARRAY (addrs, *callDepth, &ipc_buf);
    }

  return result;
}

static CUDBGResult
cudbgGetSupportedDebuggerCapabilities (CUDBGCapabilityFlags *capabilities)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getSupportedDebuggerCapabilities);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (capabilities, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadSmException (uint32_t dev, uint32_t sm, CUDBGException_t *exception,
		      uint64_t *errorPC, bool *errorPCValid)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readSmException);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  if (result == CUDBG_SUCCESS)
    {
      CUDBG_IPC_RECEIVE (exception, &ipc_buf);
      CUDBG_IPC_RECEIVE (errorPC, &ipc_buf);
      CUDBG_IPC_RECEIVE (errorPCValid, &ipc_buf);
    }

  return result;
}

static CUDBGResult
cudbgExecuteInternalCommand (const char *command, char *resultBuffer,
			     uint32_t sizeInBytes)
{
  if (!command || !resultBuffer || !sizeInBytes)
    return CUDBG_ERROR_INVALID_ARGS;

  uint32_t command_size = strlen (command) + 1;
  uint32_t result_length;
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_executeInternalCommand);
  CUDBG_IPC_APPEND (&sizeInBytes, sizeof (sizeInBytes));
  CUDBG_IPC_APPEND (&command_size, sizeof (command_size));

  if (command)
    CUDBG_IPC_APPEND (command, command_size);

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);

  if (result == CUDBG_SUCCESS)
    {
      CUDBG_IPC_RECEIVE (&result_length, &ipc_buf);
      CUDBG_IPC_RECEIVE_ARRAY (resultBuffer, result_length, &ipc_buf);
    }

  return result;
}

static CUDBGResult
cudbgGetGridInfo (uint32_t dev, uint64_t gridId64, CUDBGGridInfo *gridInfo)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getGridInfo);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&gridId64, sizeof (gridId64));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (gridInfo, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetClusterDim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *clusterDim)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getClusterDim);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (clusterDim, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadWarpState (uint32_t devId, uint32_t sm, uint32_t wp,
		    CUDBGWarpState *state)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readWarpState);
  CUDBG_IPC_APPEND (&devId, sizeof (devId));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (state, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgGetClusterExceptionTargetBlock (uint32_t dev, uint32_t sm, uint32_t wp,
				     CuDim3 *blockIdx, bool *isBlockIdxValid)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_getClusterExceptionTargetBlock);
  CUDBG_IPC_APPEND (&dev, sizeof (dev));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (blockIdx, &ipc_buf);
  CUDBG_IPC_RECEIVE (isBlockIdxValid, &ipc_buf);

  return result;
}

static CUDBGResult
cudbgReadWarpResources (uint32_t devId, uint32_t sm, uint32_t wp,
			CUDBGWarpResources *resources)
{
  char *ipc_buf;
  CUDBGResult result;

  CUDBG_IPC_BEGIN (CUDBGAPIREQ_readWarpResources);
  CUDBG_IPC_APPEND (&devId, sizeof (devId));
  CUDBG_IPC_APPEND (&sm, sizeof (sm));
  CUDBG_IPC_APPEND (&wp, sizeof (wp));

  CUDBG_IPC_REQUEST ((void **)&ipc_buf);
  CUDBG_IPC_RECEIVE (&result, &ipc_buf);
  CUDBG_IPC_RECEIVE (resources, &ipc_buf);

  return result;
}

static const struct CUDBGAPI_st cudbgCurrentApi = {
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
  cudbgSingleStepWarp65,

  /* 9.0 Extensions */
  cudbgReadDeviceExceptionState,

  /* 10.0 Extensions */
  cudbgGetNumUniformRegisters,
  cudbgReadUniformRegisterRange,
  cudbgWriteUniformRegister,
  cudbgGetNumUniformPredicates,
  cudbgReadUniformPredicates,
  cudbgWriteUniformPredicates,

  /* 11.8 Extensions */
  cudbgGetLoadedFunctionInfo118,

  /* 12.0 Extensions */
  cudbgGetGridInfo120,
  cudbgGetClusterDim120,
  cudbgReadWarpState120,
  cudbgReadClusterIdx,

  /* 12.2 Extensions */
  cudbgGetErrorStringEx,

  /* 12.3 Extensions */
  cudbgGetLoadedFunctionInfo,
  cudbgGenerateCoredump,
  cudbgGetConstBankAddress123,

  /* 12.4 Extensions */
  cudbgGetDeviceInfoSizes,
  cudbgGetDeviceInfo,
  cudbgGetConstBankAddress,
  cudbgSingleStepWarp,

  /* 12.5 Extensions */
  cudbgReadAllVirtualReturnAddresses,
  cudbgGetSupportedDebuggerCapabilities,
  cudbgReadSmException,

  /* 12.6 Extensions */
  cudbgExecuteInternalCommand,

  /* 12.7 Extensions */
  cudbgGetGridInfo,
  cudbgGetClusterDim,
  cudbgReadWarpState,
  cudbgGetClusterExceptionTargetBlock,

  /* 12.8 Extensions */
  cudbgReadWarpResources,
};

CUDBGResult
cudbgGetAPI (uint32_t major, uint32_t minor, uint32_t rev, CUDBGAPI *api)
{
  *api = &cudbgCurrentApi;
  return CUDBG_SUCCESS;
}

ATTRIBUTE_PRINTF (1, 2) static void cudbg_trace (const char *fmt, ...)
{
#ifdef GDBSERVER
  struct cuda_trace_msg *msg;
#endif
  va_list ap;

  if (!cuda_options_debug_libcudbg ())
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
