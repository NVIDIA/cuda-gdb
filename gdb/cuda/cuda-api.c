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

#include "defs.h"
#include "inferior.h"
#include "gdbcore.h"
#include "remote.h"

#include "cuda/cuda-version.h"
#include "cuda-api.h"
#include "cuda-options.h"
#include "cuda-tdep.h"
#include "cuda-packet-manager.h"
#include "cuda-utils.h"

#include <chrono>
#include <signal.h>
#include <unistd.h>
#include <execinfo.h>

#define CUDA_API_TRACE(fmt, ...)			\
  cuda_api_trace ("%s(): " fmt, __FUNCTION__, ## __VA_ARGS__)

#define CUDA_API_TRACE_FUNCTION(function, fmt, ...)		\
  cuda_api_trace ("%s(): " fmt, function, ## __VA_ARGS__)

#define CUDA_API_TRACE_DEV(dev, fmt, ...)		\
  cuda_api_trace ("%s(%u): " fmt, __FUNCTION__, dev, ## __VA_ARGS__)

#define CUDA_API_TRACE_DEV_SM(dev, sm, fmt, ...)	\
  cuda_api_trace ("%s(%u, %u): " fmt, __FUNCTION__, dev, sm, ## __VA_ARGS__)

#define CUDA_API_TRACE_DEV_SM_WARP(dev, sm, wp, fmt, ...)	\
  cuda_api_trace ("%s(%u, %u, %u): " fmt, __FUNCTION__, dev, sm, wp, ## __VA_ARGS__)

#define CUDA_API_TRACE_DEV_SM_WARP_LANE(dev, sm, wp, ln, fmt, ...)	\
  cuda_api_trace ("%s(%u, %u, %u, %u): " fmt, __FUNCTION__, dev, sm, wp, ln, ## __VA_ARGS__)


#define CUDA_API_ERROR(res, fmt, ...)				\
  cuda_api_error (res, "%s(): " fmt, __FUNCTION__, ## __VA_ARGS__)

#define CUDA_API_ERROR_DEV(res, dev, fmt, ...)				\
  cuda_api_error (res, "%s(%u): " fmt, __FUNCTION__, dev, ## __VA_ARGS__)

#define CUDA_API_ERROR_DEV_SM(res, dev, sm, fmt, ...)			\
  cuda_api_error (res, "%s(%u, %u): " fmt, __FUNCTION__, dev, sm, ## __VA_ARGS__)

#define CUDA_API_ERROR_DEV_SM_WARP(res, dev, sm, wp, fmt, ...)		\
  cuda_api_error (res, "%s(%u, %u, %u): " fmt, __FUNCTION__, dev, sm, wp, ## __VA_ARGS__)

#define CUDA_API_ERROR_DEV_SM_WARP_LANE(res, dev, sm, wp, ln, fmt, ...)	\
  cuda_api_error (res, "%s(%u, %u, %u, %u): " fmt, __FUNCTION__, dev, sm, wp, ln, ## __VA_ARGS__)

#define CUDA_API_FUNC_OFFSET(func)                                            \
  (s_api_call_stats[static_cast<std::size_t> (                                \
      offsetof (std::remove_pointer<CUDBGAPI>::type, func)                    \
      / sizeof (uintptr_t))])

#define CUDA_API_PROFILE(func)                                                \
  cuda_api_profiling timer { __FUNCTION__, CUDA_API_FUNC_OFFSET (func) }

// Globals
cuda_debugapi cuda_debugapi::s_instance;
decltype (cuda_debugapi::s_api_call_stats) cuda_debugapi::s_api_call_stats;

class cuda_api_profiling
{
public:
  // Assumes the lifetime of function matches the lifetime of
  // cuda_api_profiling
  cuda_api_profiling (const char *function, cuda_api_stat &stat)
      : m_stat (stat), m_func (function)
  {
    if (cuda_options_statistics_collection_enabled ())
      {
	m_start_time = std::chrono::steady_clock::now ();
      }
  }
  ~cuda_api_profiling ()
  {
    if (cuda_options_statistics_collection_enabled ())
      {
	const auto end = std::chrono::steady_clock::now ();
	const std::chrono::microseconds elapsed
	    = std::chrono::duration_cast<std::chrono::microseconds> (
		end - m_start_time);
	if (m_stat.times_called == 0)
	  {
	    // First time recording this function
	    m_stat.name = std::string (m_func);
	    m_stat.times_called = 1;
	    m_stat.total_time = elapsed;
	    m_stat.min_time = elapsed;
	    m_stat.max_time = elapsed;
	  }
	else
	  {
	    m_stat.times_called++;
	    m_stat.total_time += elapsed;
	    m_stat.min_time = std::min (m_stat.min_time, elapsed);
	    m_stat.max_time = std::max (m_stat.max_time, elapsed);
	  }
      }
  }

private:
  cuda_api_stat &m_stat;
  const char *m_func;
  std::chrono::steady_clock::time_point m_start_time;
};

void
cuda_debugapi::for_each_api_stat (std::function<void (const cuda_api_stat &)> func)
{
  for (const auto &stat : s_api_call_stats)
    func (stat);
}

void
cuda_debugapi::reset_api_stat ()
{
  std::fill (std::begin (s_api_call_stats), std::end (s_api_call_stats), cuda_api_stat {});
}

cuda_debugapi::cuda_debugapi ()
  : m_cudbgAPI { nullptr },
    m_api_ptid { 0 },
    m_api_state { CUDA_API_STATE_UNINITIALIZED },
    m_attach_state { CUDA_ATTACH_STATE_NOT_STARTED }
{
}

void
cuda_debugapi::cuda_api_trace (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_API, fmt, ap);
  va_end (ap);
}

void
cuda_debugapi::cuda_api_error(CUDBGResult res, const char *fmt, ...)
{
  va_list args;
  char errStr[cuda_debugapi::ErrorStringMaxLength] = {0};
  char errStrEx[cuda_debugapi::ErrorStringExMaxLength] = {0};

  va_start (args, fmt);
  vsnprintf (errStr, sizeof(errStr), fmt, args);
  va_end (args);

  get_error_string_ex (errStrEx, sizeof(errStrEx), nullptr);
  throw_error (GENERIC_ERROR, "Error: %s, error=%s, error message=%s\n",
               errStr, cudbgGetErrorString(res), errStrEx);
}

void
cuda_debugapi::cuda_api_print_api_call_result (const char *function, CUDBGResult res)
{
  if (res != CUDBG_SUCCESS)
    {
      char errStrEx[cuda_debugapi::ErrorStringExMaxLength] = {0};

      get_error_string_ex (errStrEx, sizeof(errStrEx), nullptr);
      cuda_api_trace ("%s API call received result: %s, error message=%s",
		      function, cudbgGetErrorString((CUDBGResult) res), errStrEx);
    }
}

const char*
cuda_api_mask_string(const cuda_api_warpmask* m)
{
    static char buf[sizeof(m->mask)*2 + 3] = {0};
    sprintf(buf, "0x%.*llx", (int)sizeof(buf) - 3, (unsigned long long)(m->mask));
    return buf;
}

int
cuda_debugapi::initialize ()
{
  if (api_state_initialized ())
    return 0;

  CUDA_API_TRACE ("");

  CUDA_API_PROFILE (initialize);

  // Save the inferior_ptid that we are initializing
  s_instance.m_api_ptid = cuda_gdb_get_tid_or_pid (inferior_ptid);

  CUDBGResult res = s_instance.m_cudbgAPI->initialize ();
  cuda_api_print_api_call_result (__FUNCTION__, res);
  handle_initialization_error (res);

  if (res == CUDBG_SUCCESS)
    {
      // For coredumps, default to the API version info in the
      // cudadebugger.h we were compiled with
      if (!target_has_execution ())
	cuda_debugapi::set_api_version (CUDBG_API_VERSION_MAJOR,
					CUDBG_API_VERSION_MINOR,
					CUDBG_API_VERSION_REVISION);

      CUDA_API_TRACE ("Attached to debug agent version %u.%u.%u",
		      s_instance.m_api_version.m_major,
		      s_instance.m_api_version.m_minor,
		      s_instance.m_api_version.m_revision);
    }

  return (res != CUDBG_SUCCESS && res != CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED);
}

void
cuda_debugapi::finalize ()
{
  if (!api_state_initialized ())
    return;

  CUDA_API_TRACE ("");

  CUDA_API_PROFILE (finalize);

  clear_state ();

  CUDBGResult res;
  if (is_remote_target (current_inferior ()->process_target ()))
    res = cuda_remote_api_finalize ();
  else
    res = s_instance.m_cudbgAPI->finalize ();
  if (res != CUDBG_SUCCESS)
    handle_finalize_api_error (res);
}

CUDBGCapabilityFlags
cuda_debugapi::get_supported_capabilities ()
{
  if (!api_state_initialized ())
    return CUDBG_DEBUGGER_CAPABILITY_NONE;

  // If the API version is too old, we can't get the capabilities,
  // so return none, even if the debugger supports lazy function loading.
  // We'll still get LFL events if we've subscribed to them.
  if (api_version ().m_revision < 144)
    return CUDBG_DEBUGGER_CAPABILITY_NONE;

  CUDA_API_PROFILE (getSupportedDebuggerCapabilities);

  CUDBGCapabilityFlags flags;
  CUDBGResult res = s_instance.m_cudbgAPI->getSupportedDebuggerCapabilities (&flags);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    {
      CUDA_API_TRACE ("Failed to get the supported debugger capabilities, assuming none");
      return CUDBG_DEBUGGER_CAPABILITY_NONE;
    }

  return flags;
}

void
cuda_debugapi::print_get_api_error (CUDBGResult res)
{
  switch (res)
    {
      case CUDBG_SUCCESS:
        return;

      case CUDBG_ERROR_INITIALIZATION_FAILURE:
        gdb_printf (gdb_stderr,
                            "The CUDA driver failed initialization. "
                            "Likely because X is running on all devices.\n");
        break;

      default:
        gdb_printf (gdb_stderr,
                            "The CUDA Debugger API failed with error %s\n",
			    cudbgGetErrorString(res));
        break;
    }

  gdb_printf (gdb_stderr, "[CUDA Debugging is disabled]\n");
}

void
cuda_debugapi::cuda_api_fatal (const char *msg, CUDBGResult res)
{
  CUDA_API_TRACE ("%s: %s", cudbgGetErrorString(res), msg);

  /* Finalize API */
  cuda_debugapi::finalize ();
  target_kill ();

  cuda_managed_memory_clean_regions ();

  /* Report error */
  throw_quit (_("fatal: %s (%d): %s"), cudbgGetErrorString(res), res, msg);
}

void
cuda_debugapi::handle_initialization_error (CUDBGResult res)
{
  CUDA_API_TRACE ("%s", cudbgGetErrorString(res));
  switch (res)
    {
    case CUDBG_SUCCESS:
      s_instance.m_api_state = CUDA_API_STATE_INITIALIZED;
      break;

    case CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED:
      warning (_("One or more CUDA devices are made unavailable to the application "
                 "because they are used for display and cannot be used while debugging. "
                 "This may change the application behavior."));
      s_instance.m_api_state = CUDA_API_STATE_INITIALIZED;
      break;

    case CUDBG_ERROR_UNINITIALIZED:
      /* Not ready yet. Will try later. */
      s_instance.m_api_state = CUDA_API_STATE_INITIALIZING;
      break;

    case CUDBG_ERROR_ALL_DEVICES_WATCHDOGGED:
      cuda_api_fatal ("All CUDA devices are used for display and cannot "
                              "be used while debugging.", res);
      break;

    case CUDBG_ERROR_INCOMPATIBLE_API:
      cuda_api_fatal ("Incompatible CUDA driver version.", res);
      break;

    case CUDBG_ERROR_INVALID_DEVICE:
      cuda_api_fatal ("One or more CUDA devices cannot be used for debugging. "
                      "Please consult the list of supported CUDA devices for more details.",
                      res);
      break;

    case CUDBG_ERROR_NO_DEVICE_AVAILABLE:
      cuda_api_fatal ("No CUDA capable device was found.", res);
      break;

    default:
      cuda_api_fatal ("The CUDA driver initialization failed.", res);
      break;
    }
}

void
cuda_debugapi::clear_state ()
{
  CUDA_API_TRACE ("");

  /* Mark the API as not initialized as early as possible. If the finalize()
   * call fails, we won't try to do anything stupid afterwards. */
  s_instance.m_api_state = CUDA_API_STATE_UNINITIALIZED;
  cuda_set_uvm_used (false);

  set_attach_state (CUDA_ATTACH_STATE_NOT_STARTED);
  cuda_managed_memory_clean_regions ();
}

void
cuda_debugapi::handle_finalize_api_error (CUDBGResult res)
{
  if (!api_state_initialized ())
    return;

  cuda_api_print_api_call_result (__FUNCTION__, res);

  /* Only emit a warning in case of a failure, because cuda_api_finalize () can
     be called when an error occurs. That would create an infinite loop and/or
     undesired side effects. */
  if (res != CUDBG_SUCCESS)
    warning (_("Failed to finalize the CUDA debugger API (error=%u).\n"), res);
}

void
cuda_debugapi::initialize_attach_stub ()
{
  if (!api_state_initialized ())
    return;

  CUDA_API_TRACE ("");

  CUDA_API_PROFILE (initializeAttachStub);

  /* Mark the API as not initialized as early as possible. If the finalize()
   * call fails, we won't try to do anything stupid afterwards. */
  s_instance.m_api_state = CUDA_API_STATE_UNINITIALIZED;
  cuda_set_uvm_used (false);

  CUDBGResult res = s_instance.m_cudbgAPI->initializeAttachStub ();
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR (res, "failed to initialize attach stub");
}

bool
cuda_debugapi::get_host_addr_from_device_addr (uint32_t dev, uint64_t devaddr, uint64_t *hostaddr)
{
  gdb_assert (hostaddr);
  *hostaddr = 0;
  
  if (!api_state_initialized ())
    return true;

  CUDA_API_PROFILE (getHostAddrFromDeviceAddr);

  auto res = s_instance.m_cudbgAPI->getHostAddrFromDeviceAddr (dev, devaddr, hostaddr);
  cuda_api_print_api_call_result (__FUNCTION__, res);
  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "Failed to translate device VA 0x%lx to host VA", devaddr);

  CUDA_API_TRACE_DEV (dev, "devaddr 0x%lx hostaddr 0x%lx", devaddr, *hostaddr);
  return true;
}

void
cuda_debugapi::read_grid_id (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *grid_id)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readGridId);

  CUDBGResult res = s_instance.m_cudbgAPI->readGridId (dev, sm, wp, grid_id);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read the grid index");

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "grid %ld", (int64_t)*grid_id);
}

void
cuda_debugapi::read_block_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readBlockIdx);

  CUDBGResult res = s_instance.m_cudbgAPI->readBlockIdx (dev, sm, wp, blockIdx);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read the block index");

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "blockIdx = (%u, %u, %u)",
			      blockIdx->x, blockIdx->y, blockIdx->z);
}

void
cuda_debugapi::read_cluster_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *clusterIdx)
{
  memset(clusterIdx, 0, sizeof(*clusterIdx));

  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readClusterIdx);

  CUDBGResult res = s_instance.m_cudbgAPI->readClusterIdx (dev, sm, wp, clusterIdx);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_NOT_SUPPORTED)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read the cluster index");

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "clusterIdx = (%u, %u, %u)",
			      clusterIdx->x, clusterIdx->y, clusterIdx->z);
}

void
cuda_debugapi::read_active_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *active_lanes)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readActiveLanes);

  CUDBGResult res = s_instance.m_cudbgAPI->readActiveLanes (dev, sm, wp, active_lanes);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read the active lanes mask");

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "active lanes = 0x%08x", *active_lanes);
}

void
cuda_debugapi::read_code_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readCodeMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->readCodeMemory (dev, addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to read code memory 0x%lx", addr);

  CUDA_API_TRACE_DEV (dev, "0x%lx (%u)", addr, sz);
}

void
cuda_debugapi::read_const_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readConstMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->readConstMemory (dev, addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "Failed to read const memory at address 0x%lx", addr);

  CUDA_API_TRACE_DEV (dev, "0x%lx (%u)", addr, sz);
}

bool
cuda_debugapi::read_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
				    uint64_t addr, void *buf, uint32_t sz)
{
  gdb_assert (buf);
  
  if (!api_state_initialized ())
    return false;

  CUDA_API_PROFILE (readGenericMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->readGenericMemory (dev, sm, wp, ln, addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  // If the address is not in device memory, return false to indicate
  // to the higher layers that it should try to use the fallback
  // host-access path.
  if (res == CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM)
    {
      CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln,
				       "address 0x%lx len %u address not on device", addr, sz);
      return false;
    }
  
  if (!target_has_execution () && (res == CUDBG_ERROR_MISSING_DATA))
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln,
				     "Generic memory address 0x%lx is not available in this corefile",
				     addr);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln,
				     "Failed to read generic memory at 0x%lx", addr);

  // Read succeeded
  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "0x%lx (%u)", addr, sz);

  return true;
}

bool
cuda_debugapi::read_pinned_memory (uint64_t addr, void *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return false;

  CUDA_API_PROFILE (readPinnedMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->readPinnedMemory (addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res == CUDBG_ERROR_MEMORY_MAPPING_FAILED)
    {
      CUDA_API_TRACE ("memory mapping failed: address 0x%lx len %u", addr, sz);
      return false;
    }

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR (res, "failed to read pinned memory at address 0x%lx size %u", addr, sz);

  if ((sz == 4) || (sz == 8))
    CUDA_API_TRACE ("address 0x%lx len %u = 0x%lx", addr, sz,
		    (sz == 4) ? (uint64_t)*(uint32_t *)buf : (uint64_t)*(uint64_t *)buf);
  else
    CUDA_API_TRACE ("address 0x%lx len %u", addr, sz);

  return true;
}

void
cuda_debugapi::read_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readParamMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->readParamMemory (dev, sm, wp, addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp,
				"failed to read param memory at address 0x%lx size %u",
				addr, sz);

  if ((sz == 4) || (sz == 8))
    CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "address 0x%lx len %u = 0x%lx", addr, sz,
				(sz == 4) ? (uint64_t)*(uint32_t *)buf : (uint64_t)*(uint64_t *)buf);
  else
    CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "address 0x%lx len %u", addr, sz);
}

void
cuda_debugapi::read_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readSharedMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->readSharedMemory (dev, sm, wp, addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (!target_has_execution () && (res == CUDBG_ERROR_MISSING_DATA))
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp,
				"shared memory is not available in this corefile");

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp,
				"failed to read shared memory at address 0x%lx size %u",
				addr, sz);

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "0x%lx (%u)", addr, sz);
}

bool
cuda_debugapi::read_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return false;

  CUDA_API_PROFILE (readLocalMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->readLocalMemory (dev, sm, wp, ln, addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (!target_has_execution () && (res == CUDBG_ERROR_MISSING_DATA))
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln,
				     "local memory is not available in this corefile");

  if (res != CUDBG_SUCCESS)
    {
      CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln,
				       "failed to read local memory at address 0x%lx size %u",
				       addr, sz);
      return false;
    }

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "0x%lx (%u)", addr, sz);

  return true;
}

void
cuda_debugapi::read_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
			      uint32_t regno, uint32_t *val)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readRegister);

  CUDBGResult res = s_instance.m_cudbgAPI->readRegister (dev, sm, wp, ln, regno, val);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln, "failed to read register R%u", regno);

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "R%u = 0x%08x", regno, *val);
}

void
cuda_debugapi::read_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t *val)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readUniformRegisterRange);

  CUDBGResult res = s_instance.m_cudbgAPI->readUniformRegisterRange (dev, sm, wp, regno, 1, val);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  /* Not all devices support uniform registers */
  if (res == CUDBG_ERROR_INVALID_DEVICE || res == CUDBG_ERROR_MISSING_DATA)
    *val = 0;
  else if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read uniform register UR%u", regno);

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "UR%u = 0x%08x", regno, *val);
}

void
cuda_debugapi::read_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
				uint32_t predicates_size, uint32_t *predicates)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readPredicates);

  CUDBGResult res = s_instance.m_cudbgAPI->readPredicates (dev, sm, wp, ln, predicates_size, predicates);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln, "failed to read predicates");

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_API))
    {
      uint32_t preds = 0;
      for (uint32_t i = 0; i < predicates_size; i++)
	if (predicates[i])
	  preds |= 1 << i;
      CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "predicates 0x%08x",
				       preds);
    }
}

void
cuda_debugapi::read_upredicates (uint32_t dev, uint32_t sm, uint32_t wp,
				 uint32_t predicates_size, uint32_t *predicates)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readUniformPredicates);

  CUDBGResult res = s_instance.m_cudbgAPI->readUniformPredicates (dev, sm, wp, predicates_size, predicates);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  /* Not all devices support uniform registers */
  if (res == CUDBG_ERROR_INVALID_DEVICE || res == CUDBG_ERROR_MISSING_DATA)
    {
      CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "invalid device or missing data");
      memset (predicates, 0, predicates_size * sizeof (*predicates));
    }
  else if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "Failed to read uniform predicates");

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_API))
    {
      uint32_t preds = 0;
      for (uint32_t i = 0; i < predicates_size; i++)
	if (predicates[i])
	  preds |= 1 << i;
      CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "predicates 0x%08x", preds);
    }
}

void
cuda_debugapi::read_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *val)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readCCRegister);

  CUDBGResult res = s_instance.m_cudbgAPI->readCCRegister (dev, sm, wp, ln, val);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln, "failed to read CC register");

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "cc 0x%08x", *val);
}

void
cuda_debugapi::read_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readPC);

  CUDBGResult res = s_instance.m_cudbgAPI->readPC (dev, sm, wp, ln, pc);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to read the program counter");

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "pc 0x%lx", *pc);
}

void
cuda_debugapi::read_virtual_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readVirtualPC);

  CUDBGResult res = s_instance.m_cudbgAPI->readVirtualPC (dev, sm, wp, ln, pc);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to read the virtual PC");

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "pc 0x%lx", *pc);
}

void
cuda_debugapi::read_lane_exception (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readLaneException);

  CUDBGResult res = s_instance.m_cudbgAPI->readLaneException (dev, sm, wp, ln, exception);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln, "failed to read the lane exception");

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "exception %u", (uint32_t)*exception);
}

void
cuda_debugapi::read_device_exception_state (uint32_t dev, uint64_t *exceptionSMMask, uint32_t n)
{
  gdb_assert (exceptionSMMask);

  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readDeviceExceptionState);

  CUDBGResult res = s_instance.m_cudbgAPI->readDeviceExceptionState (dev, exceptionSMMask, n);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to read device exception state");

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_API))
    for (auto i = 0; i < n; ++i)
      CUDA_API_TRACE_DEV (dev, "exceptionSMMask[%u] = 0x%016lx", i, exceptionSMMask[i]);
}

void
cuda_debugapi::read_sm_exception (uint32_t dev, uint32_t sm, CUDBGException_t *exception, uint64_t *errorPC, bool *errorPCValid)
{
  gdb_assert (exception);
  gdb_assert (errorPC);
  gdb_assert (errorPCValid);

  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readSmException);

  CUDBGResult res = CUDBG_ERROR_NOT_SUPPORTED;

  if (api_version ().m_revision >= 145)
    res = s_instance.m_cudbgAPI->readSmException (dev, sm, exception, errorPC, errorPCValid);

  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to read sm exception");

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_API))
    CUDA_API_TRACE_DEV (dev, "exception %u errorPC 0x%lx valid %s",
			(uint32_t)*exception, *errorPC,
			errorPCValid ? "true" : "false");
}

bool
cuda_debugapi::write_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
				     uint64_t addr, const void *buf, uint32_t sz)
{
  gdb_assert (buf);
  
  if (!api_state_initialized ())
    return false;

  CUDA_API_PROFILE (writeGenericMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->writeGenericMemory (dev, sm, wp, ln, addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  // If the address is not in device memory, return false to indicate
  // to the higher layers that it should try to use the fallback
  // host-access path.
  if (res == CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM)
    {
      CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln,
				       "Generic address 0x%lx (%u bytes) not in device memory",
				       addr, sz);
      return false;
    }

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln,
				     "Failed to write generic memory address 0x%lx size %u",
				     addr, sz);

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "0x%lx (%u)", addr, sz);

  return true;
}

bool
cuda_debugapi::write_pinned_memory (uint64_t addr, const void *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return false;

  CUDA_API_PROFILE (writePinnedMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->writePinnedMemory (addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_MEMORY_MAPPING_FAILED)
    CUDA_API_ERROR (res, "failed to write pinned memory at address 0x%lx size %u", addr, sz);

  CUDA_API_TRACE ("0x%lx (%u)", addr, sz);

  return res == CUDBG_SUCCESS;
}

void
cuda_debugapi::write_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (writeParamMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->writeParamMemory (dev, sm, wp, addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp,
				"failed to write param memory at address 0x%lx size %u", addr, sz);

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "0x%lx (%u)", addr, sz);
}

void
cuda_debugapi::write_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (writeSharedMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->writeSharedMemory (dev, sm, wp, addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp,
				"failed to write shared memory at address 0x%lx size %u",
				addr, sz);

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "0x%lx (%u)", addr, sz);
}

bool
cuda_debugapi::write_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return false;

  CUDA_API_PROFILE (writeLocalMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->writeLocalMemory (dev, sm, wp, ln, addr, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    {
      CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln,
				       "failed to write local memory at address 0x%lx size %u",
				       addr, sz);
      return false;
    }

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "0x%lx (%u)", addr, sz);

  return true;
}

void
cuda_debugapi::write_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (writeRegister);

  CUDBGResult res = s_instance.m_cudbgAPI->writeRegister (dev, sm, wp, ln, regno, val);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln, "failed to write register R%u", regno);

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "R%d = 0x%08x", regno, val);
}

void
cuda_debugapi::write_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t val)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (writeUniformRegister);

  CUDBGResult res = s_instance.m_cudbgAPI->writeUniformRegister (dev, sm, wp, regno, val);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to write uniform register UR%u", regno);

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "UR%d = 0x%08x", regno, val);
}

void
cuda_debugapi::write_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates)
{
  if (!api_state_initialized ())
    return;

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_API))
    {
      uint32_t preds = 0;
      for (uint32_t i = 0; i < predicates_size; i++)
        if (predicates[i])
	  preds |= 1 << i;
      CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "predicates 0x%08x",
				       preds);
    }

  CUDA_API_PROFILE (writePredicates);

  CUDBGResult res = s_instance.m_cudbgAPI->writePredicates (dev, sm, wp, ln, predicates_size, predicates);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln, "failed to write predicates");
}

void
cuda_debugapi::write_upredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, const uint32_t *predicates)
{
  if (!api_state_initialized ())
    return;

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_API))
    {
      uint32_t preds = 0;
      for (uint32_t i = 0; i < predicates_size; i++)
        if (predicates[i])
	  preds |= 1 << i;
      CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "predicates 0x%08x", preds);
    }

  CUDA_API_PROFILE (writeUniformPredicates);

  CUDBGResult res = s_instance.m_cudbgAPI->writeUniformPredicates (dev, sm, wp, predicates_size, predicates);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to write uniform predicates");
}

void
cuda_debugapi::write_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (writeCCRegister);

  CUDBGResult res = s_instance.m_cudbgAPI->writeCCRegister (dev, sm, wp, ln, val);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln, "failed to write CC register");

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "cc 0x%08x", val);
}

void
cuda_debugapi::get_grid_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *grid_dim)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getGridDim);

  CUDBGResult res = s_instance.m_cudbgAPI->getGridDim (dev, sm, wp, grid_dim);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read the grid dimensions");
}

void
cuda_debugapi::get_cluster_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *cluster_dim)
{
  memset(cluster_dim, 0, sizeof(*cluster_dim));

  if (!api_state_initialized ())
    return;

  CUDBGResult res;
  if (api_version ().m_revision >= 148)
    {
      CUDA_API_PROFILE (getClusterDim);
      res = s_instance.m_cudbgAPI->getClusterDim (dev, sm, wp, cluster_dim);
      CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "cluster_dim = (%u, %u, %u)",
				  cluster_dim->x, cluster_dim->y, cluster_dim->z);
    }
  else
    {
      CUDA_API_PROFILE (getClusterDim120);
      uint64_t gridId64;
      read_grid_id (dev, sm, wp, &gridId64);
      res = s_instance.m_cudbgAPI->getClusterDim120 (dev, gridId64, cluster_dim);
      CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "gridId %ld cluster_dim = (%u, %u, %u)",
				  (int64_t)gridId64, cluster_dim->x, cluster_dim->y, cluster_dim->z);
    }
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_NOT_SUPPORTED)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read grid cluster dimensions");
}

void
cuda_debugapi::get_block_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *block_dim)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getBlockDim);

  CUDBGResult res = s_instance.m_cudbgAPI->getBlockDim (dev, sm, wp, block_dim);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read the block dimensions");

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "block_dim = (%u, %u, %u)",
			      block_dim->x, block_dim->y, block_dim->z);
}

void
cuda_debugapi::get_blocking (uint32_t dev, uint32_t sm, uint32_t wp, bool *blocking)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getGridAttribute);

  uint64_t blocking64;
  CUDBGResult res = s_instance.m_cudbgAPI->getGridAttribute (dev, sm, wp, CUDBG_ATTR_GRID_LAUNCH_BLOCKING, &blocking64);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read the grid blocking attribute");

  *blocking = !!blocking64;

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "blocking %d", *blocking);
}

void
cuda_debugapi::get_tid (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getTID);

  CUDBGResult res = s_instance.m_cudbgAPI->getTID (dev, sm, wp, tid);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to get thread id");

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "tid 0x%08x", *tid);
}

void
cuda_debugapi::get_elf_image (uint32_t dev,  uint64_t handle, bool relocated,
			      void *elfImage, uint64_t size)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getElfImageByHandle);

  CUDBGResult res = s_instance.m_cudbgAPI->getElfImageByHandle (dev, handle,
								relocated ? CUDBG_ELF_IMAGE_TYPE_RELOCATED : CUDBG_ELF_IMAGE_TYPE_NONRELOCATED,
								elfImage, size);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "Failed to read the ELF image handle %lu relocated %d", handle, relocated);

  CUDA_API_TRACE_DEV (dev, "handle 0x%lx relocated %d size %ld", handle, relocated, size);
}

void
cuda_debugapi::get_device_type (uint32_t dev, char *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getDeviceType);

  CUDBGResult res = s_instance.m_cudbgAPI->getDeviceType (dev, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to get the device type");

  CUDA_API_TRACE_DEV (dev, "%s", buf);
}

void
cuda_debugapi::get_sm_type (uint32_t dev, char *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getSmType);

  CUDBGResult res = s_instance.m_cudbgAPI->getSmType (dev, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to get the SM type");

  CUDA_API_TRACE_DEV (dev, "%s", buf);
}

void
cuda_debugapi::get_device_name (uint32_t dev, char *buf, uint32_t sz)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getDeviceName);

  CUDBGResult res = s_instance.m_cudbgAPI->getDeviceName (dev, buf, sz);
  cuda_api_print_api_call_result (__FUNCTION__, res);
 
 if (res != CUDBG_SUCCESS)
   CUDA_API_ERROR_DEV (res, dev, "failed to get the device name");

 CUDA_API_TRACE_DEV (dev, "%s", buf);
}

void
cuda_debugapi::get_num_devices (uint32_t *numDev)
{
  *numDev = 0;

  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getNumDevices);

  CUDBGResult res = s_instance.m_cudbgAPI->getNumDevices (numDev);
  cuda_api_print_api_call_result (__FUNCTION__, res);
 
 if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR (res, "failed to get the number of devices");

 CUDA_API_TRACE ("%u", *numDev);
}

void
cuda_debugapi::get_num_sms (uint32_t dev, uint32_t *numSMs)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getNumSMs);

  CUDBGResult res = s_instance.m_cudbgAPI->getNumSMs (dev, numSMs);
  cuda_api_print_api_call_result (__FUNCTION__, res);
 
 if (res != CUDBG_SUCCESS)
   CUDA_API_ERROR_DEV (res, dev, "failed to get the number of SMs");

 CUDA_API_TRACE_DEV (dev, "%u", *numSMs);
}

void
cuda_debugapi::get_num_warps (uint32_t dev, uint32_t *numWarps)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getNumWarps);

  CUDBGResult res = s_instance.m_cudbgAPI->getNumWarps (dev, numWarps);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
   CUDA_API_ERROR_DEV (res, dev, "failed to get the number of warps");

 CUDA_API_TRACE_DEV (dev, "%u", *numWarps);
}

void
cuda_debugapi::get_num_lanes (uint32_t dev, uint32_t *numLanes)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getNumLanes);

  CUDBGResult res = s_instance.m_cudbgAPI->getNumLanes (dev, numLanes);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to get the number of lanes");

 CUDA_API_TRACE_DEV (dev, "%u", *numLanes);
}

void
cuda_debugapi::get_num_registers (uint32_t dev, uint32_t *numRegs)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getNumRegisters);

  CUDBGResult res = s_instance.m_cudbgAPI->getNumRegisters (dev, numRegs);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to get the number of registers");

 CUDA_API_TRACE_DEV (dev, "%u", *numRegs);
}

void
cuda_debugapi::get_num_predicates (uint32_t dev, uint32_t *numPredicates)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getNumPredicates);

  CUDBGResult res = s_instance.m_cudbgAPI->getNumPredicates (dev, numPredicates);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to get the number of predicates");

 CUDA_API_TRACE_DEV (dev, "%u", *numPredicates);
}

void
cuda_debugapi::get_num_uregisters (uint32_t dev, uint32_t *numRegs)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getNumUniformRegisters);

  CUDBGResult res = s_instance.m_cudbgAPI->getNumUniformRegisters (dev, numRegs);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to get the number of uniform registers");

 CUDA_API_TRACE_DEV (dev, "%u", *numRegs);
}

void
cuda_debugapi::get_num_upredicates (uint32_t dev, uint32_t *numPredicates)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getNumUniformPredicates);

  CUDBGResult res = s_instance.m_cudbgAPI->getNumUniformPredicates (dev, numPredicates);
  cuda_api_print_api_call_result (__FUNCTION__, res);
 
 if (res != CUDBG_SUCCESS)
   CUDA_API_ERROR_DEV (res, dev, "failed to get the number of uniform predicates");

 CUDA_API_TRACE_DEV (dev, "%u", *numPredicates);
}

void
cuda_debugapi::is_device_code_address (uint64_t addr, bool *is_device_address)
{
  if (!api_state_initialized ())
    {
      *is_device_address = false;
      return;
    }

  CUDA_API_PROFILE (isDeviceCodeAddress);

  CUDBGResult res = s_instance.m_cudbgAPI->isDeviceCodeAddress (addr, is_device_address);
  cuda_api_print_api_call_result (__FUNCTION__, res);
 
 if (res != CUDBG_SUCCESS)
   CUDA_API_ERROR (res, "failed to determine if address 0x%lx corresponds "
		   "to the host or device", addr);

 CUDA_API_TRACE ("0x%lx is-code %d", addr, *is_device_address); 
}

void
cuda_debugapi::handle_set_callback_api_error (CUDBGResult res)
{
  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR (res, "Failed to set the new event callback");
}

void
cuda_debugapi::set_notify_new_event_callback (CUDBGNotifyNewEventCallback callback)
{
  /* Nothing should restrict the callback from being setup.
     In particular, it must be done prior to the API being
     fully initialized, which means there should not be a
     check here. */

  CUDA_API_PROFILE (setNotifyNewEventCallback);

  CUDBGResult res = s_instance.m_cudbgAPI->setNotifyNewEventCallback (callback);
  cuda_api_print_api_call_result (__FUNCTION__, res);
  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR (res, "failed to set the new event callback");
}

void
cuda_debugapi::get_next_sync_event (CUDBGEvent *event)
{
  event->kind = CUDBG_EVENT_INVALID;
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getNextEvent);

  CUDBGResult res = s_instance.m_cudbgAPI->getNextEvent (CUDBG_EVENT_QUEUE_TYPE_SYNC, event);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_NO_EVENT_AVAILABLE)
    CUDA_API_ERROR (res, "failed to get the next sync CUDA event");
}

void
cuda_debugapi::acknowledge_sync_events ()
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (acknowledgeSyncEvents);

  CUDBGResult res = s_instance.m_cudbgAPI->acknowledgeSyncEvents ();
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR (res, "failed to acknowledge a sync CUDA event");
}

void
cuda_debugapi::get_next_async_event (CUDBGEvent *event)
{
  event->kind = CUDBG_EVENT_INVALID;
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getNextEvent);

  CUDBGResult res = s_instance.m_cudbgAPI->getNextEvent (CUDBG_EVENT_QUEUE_TYPE_ASYNC, event);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_NO_EVENT_AVAILABLE)
    CUDA_API_ERROR (res, "failed to get the next async CUDA event");
}

void
cuda_debugapi::disassemble (uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t bufSize)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (disassemble);

  CUDBGResult res = s_instance.m_cudbgAPI->disassemble (dev, addr, instSize, buf, bufSize);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to disassemble address 0x%lx", addr);

  CUDA_API_TRACE_DEV (dev, "0x%016lx: %s", addr, buf);
}

void
cuda_debugapi::set_attach_state (cuda_attach_state_t state)
{
  CUDA_API_TRACE ("state %d", state);

  s_instance.m_attach_state = state;

  if (state != CUDA_ATTACH_STATE_DETACHING)
    return;

  CUDA_API_PROFILE (clearAttachState);

  CUDBGResult res = s_instance.m_cudbgAPI->clearAttachState ();

  if (res != CUDBG_SUCCESS)
    warning (_("Failed to set attach state to %u (%s (0x%x / %u)).\n"),
	     state, cudbgGetErrorString(res), res, res);
  else
    CUDA_API_TRACE ("state set %d", state);
}

void
cuda_debugapi::request_cleanup_on_detach (uint32_t resumeAppFlag)
{
  CUDA_API_PROFILE (requestCleanupOnDetach);

  CUDBGResult res = s_instance.m_cudbgAPI->requestCleanupOnDetach (resumeAppFlag);

  CUDA_API_TRACE ("resume %u", resumeAppFlag);

  if (res != CUDBG_SUCCESS)
    warning (_("Failed to clear attach state (error=%s(0x%x)).\n"), cudbgGetErrorString(res), res);
}

void
cuda_debugapi::get_grid_status (uint32_t dev, uint64_t grid_id, CUDBGGridStatus *status)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getGridStatus);

  CUDBGResult res = s_instance.m_cudbgAPI->getGridStatus (dev, grid_id, status);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to get grid %ld status", (int64_t)grid_id);

  CUDA_API_TRACE_DEV (dev, "grid %ld status %u", (int64_t) grid_id, *status);
}

void
cuda_debugapi::get_grid_info (uint32_t dev, uint64_t grid_id, CUDBGGridInfo *info)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getGridInfo);

  CUDBGResult res;
  if (api_version ().m_revision >= 148)
    res = s_instance.m_cudbgAPI->getGridInfo (dev, grid_id, info);
  else
    {
      res = s_instance.m_cudbgAPI->getGridInfo120 (dev, grid_id, reinterpret_cast<CUDBGGridInfo120 *>(info));
      memset (reinterpret_cast<char *>(info) + sizeof(CUDBGGridInfo120), 0, sizeof(*info) - sizeof(CUDBGGridInfo120));
    }
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to get grid %ld info", (int64_t)grid_id);

  CUDA_API_TRACE_DEV (dev, "id %ld", (int64_t)grid_id);
}

void
cuda_debugapi::get_adjusted_code_address (uint32_t dev, uint64_t addr, uint64_t *adjusted_addr, CUDBGAdjAddrAction adj_action)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getAdjustedCodeAddress);

  CUDBGResult res = s_instance.m_cudbgAPI->getAdjustedCodeAddress (dev, addr, adjusted_addr, adj_action);
  cuda_api_print_api_call_result (__FUNCTION__, res);
 
 if (res != CUDBG_SUCCESS)
   CUDA_API_ERROR_DEV (res, dev, "failed to get adjusted code address addr 0x%lx action %d",
		       addr, adj_action);

 CUDA_API_TRACE_DEV (dev, "addr 0x%lx action %d: adjusted_addr 0x%lx", addr, adj_action, *adjusted_addr);
}

void
cuda_debugapi::set_kernel_launch_notification_mode(CUDBGKernelLaunchNotifyMode mode)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (setKernelLaunchNotificationMode);

  s_instance.m_cudbgAPI->setKernelLaunchNotificationMode (mode);

  CUDA_API_TRACE ("%d", mode);
}

void
cuda_debugapi::get_device_pci_bus_info (uint32_t dev, uint32_t *pci_bus_id, uint32_t *pci_dev_id)
{
  *pci_bus_id = 0xffff;
  *pci_dev_id = 0xffff;

  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getDevicePCIBusInfo);

  CUDBGResult res = s_instance.m_cudbgAPI->getDevicePCIBusInfo (dev, pci_bus_id, pci_dev_id);
  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to get PCI bus information");

  CUDA_API_TRACE_DEV (dev, "bus 0x%x dev 0x%x", *pci_bus_id, *pci_dev_id);
}

void
cuda_debugapi::read_warp_state (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState *state)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readWarpState);

  CUDBGResult res;
  if (api_version ().m_revision >= 148)
    res = s_instance.m_cudbgAPI->readWarpState (dev, sm, wp, state);
  else
    {
      res = s_instance.m_cudbgAPI->readWarpState120 (dev, sm, wp, reinterpret_cast<CUDBGWarpState120*>(state));
      memset (reinterpret_cast<char *>(state) + sizeof(CUDBGWarpState120), 0, sizeof(*state) - sizeof(CUDBGWarpState120));
      /* Handle clusterDim updates via old interface */
      cuda_debugapi::get_cluster_dim (dev, sm, wp, &state->clusterDim);
    }
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read warp state");

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "grid %ld valid 0x%08x active 0x%08x",
  			      (int64_t)state->gridId, state->validLanes, state->activeLanes);
}

void
cuda_debugapi::read_register_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t idx, uint32_t count, uint32_t *regs)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "idx %u count %u", idx, count);

  CUDA_API_PROFILE (readRegisterRange);

  CUDBGResult res = s_instance.m_cudbgAPI->readRegisterRange (dev, sm, wp, ln, idx, count, regs);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp,  ln, "failed to read register range idx %u count %u", idx, count);

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_API))
    for (auto i = 0; i < count; ++i)
      if (regs[i] || ((idx + i) < 32))
	CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "R%u = %d (0x%x)",
					 idx + i, regs[i], regs[i]);
}

void
cuda_debugapi::read_uregister_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t idx, uint32_t count, uint32_t *regs)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "idx %u count %u", idx, count);

  CUDA_API_PROFILE (readUniformRegisterRange);

  CUDBGResult res = s_instance.m_cudbgAPI->readUniformRegisterRange (dev, sm, wp, idx, count, regs);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res == CUDBG_ERROR_INVALID_DEVICE || res == CUDBG_ERROR_MISSING_DATA)
    memset (regs, 0, (count - idx) * sizeof(uint32_t));
  else if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read uniform register range idx %u count %u", idx, count);

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_API))
    for (auto i = 0; i < count; ++i)
      if (regs[i])
	CUDA_API_TRACE ("UR%u = %d (0x%x)", idx + i, regs[i], regs[i]);
}

void
cuda_debugapi::read_global_memory (uint64_t addr, void *buf, uint32_t buf_size)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readGlobalMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->readGlobalMemory (addr, buf, buf_size);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (!target_has_execution () && (res == CUDBG_ERROR_MISSING_DATA))
    {
      CUDA_API_ERROR (res, "Global memory is not available in this corefile");
      return;
    }

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR (res, "failed to read %u bytes of global memory from 0x%lx", buf_size, addr);
  CUDA_API_TRACE ("0x%lx (%u)", addr, buf_size);
}

void
cuda_debugapi::write_global_memory (uint64_t addr, const void *buf, uint32_t buf_size)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (writeGlobalMemory);

  CUDBGResult res = s_instance.m_cudbgAPI->writeGlobalMemory (addr, (void *)buf, buf_size);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR (res, "failed to write %u bytes of global memory to 0x%lx", buf_size, addr);
  CUDA_API_TRACE ("0x%lx (%u)", addr, buf_size);
}

void
cuda_debugapi::get_managed_memory_region_info (uint64_t start_addr, CUDBGMemoryInfo *meminfo, uint32_t entries_count, uint32_t *entries_written)
{
  if (entries_written)
    *entries_written = 0;
  if (!api_state_initialized () || !cuda_is_uvm_used())
    return;

  CUDA_API_PROFILE (getManagedMemoryRegionInfo);

  CUDBGResult res = s_instance.m_cudbgAPI->getManagedMemoryRegionInfo (start_addr, meminfo, entries_count, entries_written);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR (res, "failed to read %u entries starting from addr 0x%lx", entries_count, start_addr);

  CUDA_API_TRACE ("0x%lx", start_addr);
}

void
cuda_debugapi::suspend_device (uint32_t dev)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (suspendDevice);

  CUDBGResult res = s_instance.m_cudbgAPI->suspendDevice (dev);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_SUSPENDED_DEVICE)
    CUDA_API_ERROR_DEV (res, dev, "failed to suspend device");

  CUDA_API_TRACE_DEV (dev, "suspended (%s)", cudbgGetErrorString (res));
}

void
cuda_debugapi::resume_device (uint32_t dev)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (resumeDevice);

  CUDBGResult res = s_instance.m_cudbgAPI->resumeDevice (dev);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_RUNNING_DEVICE)
    CUDA_API_ERROR_DEV (res, dev, "failed to resume device");

  CUDA_API_TRACE_DEV (dev, "resumed (%s)", cudbgGetErrorString (res));
}

bool
cuda_debugapi::set_breakpoint (uint32_t dev, uint64_t addr)
{
  if (!api_state_initialized ())
    return true;

  CUDA_API_PROFILE (setBreakpoint);

  CUDBGResult res = s_instance.m_cudbgAPI->setBreakpoint (dev, addr);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_INVALID_ADDRESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to set breakpoint at address 0x%lx", addr);

  CUDA_API_TRACE_DEV (dev, "0x%lx %s", addr, cudbgGetErrorString (res));

  return res != CUDBG_ERROR_INVALID_ADDRESS;
}

bool
cuda_debugapi::unset_breakpoint (uint32_t dev, uint64_t addr)
{
  if (!api_state_initialized ())
    return true;

  CUDA_API_PROFILE (unsetBreakpoint);

  CUDBGResult res = s_instance.m_cudbgAPI->unsetBreakpoint (dev, addr);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_INVALID_ADDRESS)
    CUDA_API_ERROR_DEV (res, dev, "failed to unset breakpoint at address 0x%lx", addr);

  CUDA_API_TRACE_DEV (dev, "0x%lx %s", addr, cudbgGetErrorString (res));

  return res != CUDBG_ERROR_INVALID_ADDRESS;
}

void
cuda_debugapi::read_thread_idx (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readThreadIdx);

  CUDBGResult res = s_instance.m_cudbgAPI->readThreadIdx (dev, sm, wp, ln, threadIdx);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln, "failed to read the thread index");

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "threadIdx = (%u, %u, %u)",
				   threadIdx->x, threadIdx->y, threadIdx->z);
}

void
cuda_debugapi::read_broken_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *brokenWarpsMask)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readBrokenWarps);

  CUDBGResult res = s_instance.m_cudbgAPI->readBrokenWarps (dev, sm, &brokenWarpsMask->mask);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM (res, dev, sm, "failed to read the broken warps mask");

  CUDA_API_TRACE_DEV_SM (dev, sm, "Read broken warps: %" WARP_MASK_FORMAT, cuda_api_mask_string(brokenWarpsMask));
}

void
cuda_debugapi::read_valid_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *valid_lanes)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readValidLanes);

  CUDBGResult res = s_instance.m_cudbgAPI->readValidLanes (dev, sm, wp, valid_lanes);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed read the valid lanes mask");

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "valid lanes 0x%08x", *valid_lanes);
}

bool
cuda_debugapi::single_step_warp (uint32_t dev, uint32_t sm, uint32_t wp,
				 uint32_t nsteps, cuda_api_warpmask *warp_mask)
{
  gdb_assert (warp_mask);
  *warp_mask = {0};

  if (!api_state_initialized ())
    return false;

  CUDA_API_PROFILE (singleStepWarp65);

  CUDBGResult res = s_instance.m_cudbgAPI->singleStepWarp65 (dev, sm, wp, nsteps, &warp_mask->mask);

  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res == CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE)
    return false;

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to single-step");

  CUDA_API_TRACE_DEV_SM (dev, sm, "warp %u warp mask %" WARP_MASK_FORMAT,
			 wp, cuda_api_mask_string (warp_mask));

  return true;
}

bool
cuda_debugapi::single_step_warp (uint32_t dev, uint32_t sm, uint32_t wp,
				 uint32_t laneHint, uint32_t nsteps, uint32_t flags, cuda_api_warpmask *warp_mask)
{
  gdb_assert (warp_mask);
  *warp_mask = {0};

  if (!api_state_initialized ())
    return false;

  CUDA_API_PROFILE (singleStepWarp);

  CUDBGResult res = s_instance.m_cudbgAPI->singleStepWarp (dev, sm, wp, laneHint, nsteps, flags, &warp_mask->mask);

  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res == CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE)
    return false;

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to single-step");

  CUDA_API_TRACE_DEV_SM (dev, sm, "warp %u warp mask %" WARP_MASK_FORMAT,
			 wp, cuda_api_mask_string (warp_mask));

  return true;
}

bool
cuda_debugapi::resume_warps_until_pc (uint32_t dev, uint32_t sm, cuda_api_warpmask *warp_mask, uint64_t virt_pc)
{
  if (!api_state_initialized ())
    return false;

  CUDA_API_TRACE_DEV_SM (dev, sm, "mask %s pc 0x%lx", cuda_api_mask_string (warp_mask), virt_pc);

  CUDA_API_PROFILE (resumeWarpsUntilPC);

  CUDBGResult res = s_instance.m_cudbgAPI->resumeWarpsUntilPC (dev, sm, warp_mask->mask, virt_pc);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res == CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE)
    {
      CUDA_API_TRACE_DEV_SM (dev, sm, "warp resume not possible");
      return false;
    }

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM (res, dev, sm, "failed to resume warps %s to pc 0x%lx",
			   cuda_api_mask_string (warp_mask), virt_pc);

  CUDA_API_TRACE_DEV_SM (dev, sm, "return mask %s", cuda_api_mask_string (warp_mask));

  return true;
}

void
cuda_debugapi::read_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *depth)
{
  *depth = 0;

  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readCallDepth);

  CUDBGResult res = s_instance.m_cudbgAPI->readCallDepth (dev, sm, wp, ln, depth);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln, "failed to read call depth");

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "depth %u", *depth);
}

void
cuda_debugapi::read_syscall_call_depth (uint32_t dev, uint32_t sm,
					uint32_t wp, uint32_t ln, uint32_t *depth)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readSyscallCallDepth);

  CUDBGResult res = s_instance.m_cudbgAPI->readSyscallCallDepth (dev, sm, wp, ln, depth);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln, "failed to read syscall call depth");

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "depth %u", *depth);
}

void
cuda_debugapi::read_valid_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *valid_warps)
{  
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readValidWarps);

  CUDBGResult res = s_instance.m_cudbgAPI->readValidWarps (dev, sm, &valid_warps->mask);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM (res, dev, sm, "failed to read the valid warps mask");

  CUDA_API_TRACE_DEV_SM (dev, sm, "%" WARP_MASK_FORMAT,
			 cuda_api_mask_string(valid_warps));
}

void
cuda_debugapi::read_virtual_return_address (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
					    int32_t level, uint64_t *ra)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readVirtualReturnAddress);

  CUDBGResult res = s_instance.m_cudbgAPI->readVirtualReturnAddress (dev, sm, wp, ln,
							   (uint32_t)level,
							   ra);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP_LANE (res, dev, sm, wp, ln,
				     "failed to read virtual return address level %d", level);

  CUDA_API_TRACE_DEV_SM_WARP_LANE (dev, sm, wp, ln, "level %d pc 0x%lx", level, *ra);
}

void
cuda_debugapi::read_error_pc (uint32_t dev, uint32_t sm, uint32_t wp,
			      uint64_t *pc, bool* valid)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readErrorPC);

  CUDBGResult res = s_instance.m_cudbgAPI->readErrorPC (dev, sm, wp, pc, valid);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read error PC");

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "errorpc 0x%lx %s",
			      *pc, *valid ? "valid" : "invalid");
}

void
cuda_debugapi::get_loaded_function_info (uint32_t dev, uint64_t handle,
                                         CUDBGLoadedFunctionInfo *info,
                                         uint32_t startIndex,
                                         uint32_t numEntries)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getLoadedFunctionInfo);

  CUDBGResult res = CUDBG_ERROR_NOT_SUPPORTED;

  // The newer version of the call is now the default
  if (api_version ().m_revision >= 135)
    res = s_instance.m_cudbgAPI->getLoadedFunctionInfo (dev, handle, info, startIndex, numEntries);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "handle 0x%lx startIndex %u numEntries %u",
			handle, startIndex, numEntries);

  CUDA_API_TRACE_DEV (dev, "handle 0x%lx startIndex %u numEntries %u",
		      handle, startIndex, numEntries);
}

void
cuda_debugapi::get_loaded_function_info (uint32_t dev, uint64_t handle,
                                         CUDBGLoadedFunctionInfo *info,
                                         uint32_t numEntries)
{
  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (getLoadedFunctionInfo118);

  CUDBGResult res = CUDBG_ERROR_NOT_SUPPORTED;
  if (api_version ().m_revision >= 132)
    res = s_instance.m_cudbgAPI->getLoadedFunctionInfo118 (dev, handle, info, numEntries);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR_DEV (res, dev, "handle 0x%lx numEntries %u", handle, numEntries);

  CUDA_API_TRACE_DEV (dev, "handle 0x%lx numEntries %u", handle, numEntries);
}

void
cuda_debugapi::generate_coredump (const char *name,
                                  CUDBGCoredumpGenerationFlags flags)
{
  if (!api_state_initialized ())
    error ("CUDA is not initialized");

  CUDA_API_PROFILE (generateCoredump);

  CUDBGResult res = s_instance.m_cudbgAPI->generateCoredump (name, flags);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res != CUDBG_SUCCESS)
    CUDA_API_ERROR (res, "Failed to generate CUDA coredump");
}

void
cuda_debugapi::get_error_string_ex (char *buf, uint32_t bufSz, uint32_t *msgSz)
{
  // Buffer must always be provided
  gdb_assert (buf);

  CUDBGResult res = CUDBG_ERROR_NOT_SUPPORTED;

  // This method first became available in revision 134
  if (api_state_initialized () && (api_version ().m_revision >= 134))
    {
      CUDA_API_PROFILE (getErrorStringEx);
      res = s_instance.m_cudbgAPI->getErrorStringEx (buf, bufSz, msgSz);
      // Don't trace empty error strings
      if ((res == CUDBG_SUCCESS) || (res == CUDBG_ERROR_BUFFER_TOO_SMALL))
	CUDA_API_TRACE ("%s", buf);
      else if (res != CUDBG_ERROR_NOT_SUPPORTED)
	throw_error (GENERIC_ERROR, "Error: getErrorStringEx, error=%s.\n", cudbgGetErrorString(res));
    }

  // Either cuda-gdb or the debugger backend were built with a pre-134 cudadebugger.h
  if (res == CUDBG_ERROR_NOT_SUPPORTED)
    {
      buf[0] = 0;
      if (msgSz)
	*msgSz = 0;
    }
}

void
cuda_debugapi::get_const_bank_address (uint32_t dev, uint32_t sm, uint32_t wp,
                                       uint32_t bank, uint32_t offset, uint64_t* address)
{
  gdb_assert (address);

  *address = 0;

  if (!api_state_initialized ())
    return;

  if (api_version ().m_revision >= 138)
    {
      CUDA_API_PROFILE (getConstBankAddress123);
      CUDBGResult res = s_instance.m_cudbgAPI->getConstBankAddress123 (dev, sm, wp, bank, offset, address);
      cuda_api_print_api_call_result (__FUNCTION__, res);

      if (res != CUDBG_SUCCESS)
	error ("The requested value c[0x%x][0x%x] is not valid.", bank, offset);
    }
  else
    warning (_("get_const_bank_address isn't supported with this API version."));
}

void
cuda_debugapi::get_const_bank_address (uint32_t dev, uint64_t gridId64, uint32_t bank,
                                       uint64_t* address, uint32_t* size)
{
  gdb_assert (address);
  gdb_assert (size);

  *address = 0;
  *size = 0;

  if (!api_state_initialized ())
    return;

  if (api_version ().m_revision >= 141)
    {
      CUDA_API_PROFILE (getConstBankAddress);
      CUDBGResult res = s_instance.m_cudbgAPI->getConstBankAddress (dev, gridId64, bank, address, size);
      cuda_api_print_api_call_result (__FUNCTION__, res);

      if (res != CUDBG_SUCCESS)
	error ("The requested constbank c[0x%x] is not valid.", bank);
    }
  else
    warning (_("get_const_bank_address isn't supported with this API version."));
}

bool
cuda_debugapi::get_device_info_sizes (uint32_t dev, CUDBGDeviceInfoSizes* sizes)
{
  gdb_assert (sizes);

  if (!api_state_initialized ())
    return false;

  CUDA_API_PROFILE (getDeviceInfoSizes);

  CUDBGResult res = s_instance.m_cudbgAPI->getDeviceInfoSizes (dev, sizes);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res == CUDBG_SUCCESS)
    CUDA_API_TRACE_DEV (dev, "required buffer size %u", sizes->requiredBufferSize);

  return res == CUDBG_SUCCESS;
}

bool
cuda_debugapi::get_device_info (uint32_t dev, CUDBGDeviceInfoQueryType_t type,
				void* buffer, uint32_t length, uint32_t *data_length)
{
  gdb_assert (buffer);
  gdb_assert (data_length);
  
  if (!api_state_initialized ())
    return false;

  CUDA_API_PROFILE (getDeviceInfo);

  CUDBGResult res = s_instance.m_cudbgAPI->getDeviceInfo (dev, type, buffer, length, data_length);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if (res == CUDBG_SUCCESS)
    CUDA_API_TRACE_DEV (dev, "requested size %u returned size %u", length, *data_length);

  return res == CUDBG_SUCCESS;
}

// Returns false on error
bool cuda_debugapi::execute_internal_command (const char *command,
        char *resultBuffer, uint32_t sizeInBytes)
{
  gdb_assert (command);
  gdb_assert (resultBuffer);

  if (!api_state_initialized())
    return false;

  if (api_version ().m_revision < 146)
    return false;

  CUDA_API_PROFILE (executeInternalCommand);

  CUDBGResult res = s_instance.m_cudbgAPI->executeInternalCommand (command, resultBuffer, sizeInBytes);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  switch (res)
  {
    case CUDBG_SUCCESS:
      CUDA_API_TRACE("requested size %u\nresponse body: %s", sizeInBytes, resultBuffer);
      break;
    case CUDBG_ERROR_NOT_SUPPORTED:
      return false;
    case CUDBG_ERROR_BUFFER_TOO_SMALL:
      return false;
    case CUDBG_ERROR_INVALID_ARGS:
      return false;
    default:
      return false;
  }

  return true;
}

bool
cuda_debugapi::get_cluster_exception_target_block (uint32_t dev, uint32_t sm,
						   uint32_t wp,
						   CuDim3 *blockIdx,
						   bool *blockIdxValid)
{
  gdb_assert (blockIdx);
  gdb_assert (blockIdxValid);

  if (!api_state_initialized ())
    return false;

  if (api_version ().m_revision < 149)
    return false;

  CUDA_API_PROFILE (getClusterExceptionTargetBlock);

  CUDBGResult res = s_instance.m_cudbgAPI->getClusterExceptionTargetBlock (
      dev, sm, wp, blockIdx, blockIdxValid);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  CUDA_API_TRACE_DEV_SM_WARP (
      dev, sm, wp, "blockIdx = (%u, %u, %u), valid = %s", blockIdx->x,
      blockIdx->y, blockIdx->z, *blockIdxValid ? "true" : "false");

  return res == CUDBG_SUCCESS;
}

void
cuda_debugapi::read_warp_resources (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpResources *resources)
{
  gdb_assert (resources);

  if (!api_state_initialized ())
    return;

  CUDA_API_PROFILE (readWarpResources);

  memset (resources, 0, sizeof (*resources));

  CUDBGResult res = CUDBG_ERROR_NOT_SUPPORTED;
  if (api_version ().m_revision >= 152)
    res = s_instance.m_cudbgAPI->readWarpResources (dev, sm, wp, resources);
  cuda_api_print_api_call_result (__FUNCTION__, res);

  if ((res != CUDBG_SUCCESS) && (res != CUDBG_ERROR_NOT_SUPPORTED))
    CUDA_API_ERROR_DEV_SM_WARP (res, dev, sm, wp, "failed to read warp resources");

  CUDA_API_TRACE_DEV_SM_WARP (dev, sm, wp, "numRegisters %u sharedMemSize %u",
			      resources->numRegisters, resources->sharedMemSize);
}

void
cuda_api_clear_mask(cuda_api_warpmask* mask)
{
  memset ((void *)mask, 0, sizeof(*mask));
}

void
cuda_api_set_bit(cuda_api_warpmask* m, int i, int v)
{
  if (v)
    m->mask |= (1ULL << (i%64));
  else
    m->mask &= ~(1ULL << (i%64));
}

int
cuda_api_get_bit(const cuda_api_warpmask* m, int i)
{
  return (m->mask & (1ULL << (i%64))) != 0;
}

int
cuda_api_has_bit(const cuda_api_warpmask* m)
{
  return !!(m->mask);
}

int
cuda_api_has_multiple_bits(const cuda_api_warpmask* m)
{
  return !!(m->mask & (m->mask - 1));
}

int
cuda_api_eq_mask(const cuda_api_warpmask* m1, const cuda_api_warpmask* m2)
{
  return memcmp(m1, m2, sizeof(*m1)) == 0;
}

void
cuda_api_cp_mask(cuda_api_warpmask* dst, const cuda_api_warpmask* src)
{
  memcpy(dst, src, sizeof(*dst));
}

cuda_api_warpmask*
cuda_api_or_mask(cuda_api_warpmask* dst, const cuda_api_warpmask* m1, const cuda_api_warpmask* m2)
{
  dst->mask = m1->mask | m2->mask;
  return dst;
}
cuda_api_warpmask*
cuda_api_and_mask(cuda_api_warpmask* dst, const cuda_api_warpmask* m1, const cuda_api_warpmask* m2)
{
  dst->mask = m1->mask & m2->mask;
  return dst;
}

cuda_api_warpmask*
cuda_api_not_mask(cuda_api_warpmask* dst, const cuda_api_warpmask* m1)
{
  dst->mask = ~m1->mask;
  return dst;
}
