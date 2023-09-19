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

#include <signal.h>
#include <unistd.h>
#ifndef __ANDROID__
#include <execinfo.h>
#endif

#include <algorithm>
#include <iterator>
#include <tuple>
#include <unordered_map>

/* Globals */
cuda_debugapi cuda_debugapi::m_debugapi {};
extern cuda_api_version cuda_backend_api_version;

std::size_t coordinates::key (uint32_t extra) const noexcept
{
  constexpr int dev_bits = 8;
  constexpr int sm_bits = 16;
  constexpr int wp_bits = 8;
  constexpr int ln_bits = 8;

  // Reserve "extra" bottom bits for derived classes like the frame_coordinates
  constexpr int extra_bits = (sizeof (std::size_t) * 8) - dev_bits - sm_bits - wp_bits - ln_bits;
  gdb_assert (extra_bits >= 16);

  // Check that the requested coordinates fit in 64-bits
  gdb_assert (m_dev < (1ULL << dev_bits));
  gdb_assert (m_sm < (1ULL << sm_bits));
  gdb_assert (m_wp < (1ULL << wp_bits));
  gdb_assert (m_ln < (1ULL << ln_bits));
  gdb_assert (extra < (1ULL << extra_bits));

  std::size_t keynum = std::size_t (m_dev & ((1ULL << dev_bits) - 1));
  keynum = (keynum << sm_bits) | std::size_t (m_sm & ((1ULL << sm_bits) - 1));
  keynum = (keynum << wp_bits)  | std::size_t (m_wp & ((1ULL << wp_bits) - 1));
  keynum = (keynum << ln_bits)  | std::size_t (m_ln & ((1ULL << ln_bits) - 1));
  keynum = (keynum << extra_bits) | std::size_t (extra & ((1ULL << extra_bits) - 1));;

  return keynum;
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

  cuda_api_get_error_string_ex(errStrEx, sizeof(errStrEx), nullptr);
  throw_error (GENERIC_ERROR, "Error: %s, error=%s(0x%x), error message=%s.\n",
               errStr, cudbgGetErrorString(res), res, errStrEx);
}


void
cuda_debugapi::cuda_dev_api_error(const char *msg, uint32_t dev, CUDBGResult res)
{
  cuda_api_error (res, "Failed to %s for CUDA device %u", msg, dev);
}


void
cuda_debugapi::cuda_devsmwp_api_error(const char *msg, uint32_t dev, uint32_t sm, uint32_t wp, CUDBGResult res)
{
  cuda_api_error (res, "Failed to %s (dev=%u, sm=%u, wp=%u)", msg, dev, sm, wp);
}


void
cuda_debugapi::cuda_devsmwpln_api_error(const char *msg, uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGResult res)
{
  cuda_api_error (res, "Failed to %s (dev=%u, sm=%u, wp=%u, ln=%u)", msg, dev, sm, wp, ln);
}


void
cuda_debugapi::cuda_api_print_api_call_result (int res)
{
  if (res != CUDBG_SUCCESS) {
    char errStrEx[cuda_debugapi::ErrorStringExMaxLength] = {0};
    get_error_string_ex(errStrEx, sizeof(errStrEx), nullptr);
    cuda_api_trace ("API call received result: %s(0x%x), error message=%s", cudbgGetErrorString((CUDBGResult) res), res, errStrEx);
  }
}


/*
  This function needs updating whenever a new cache map is added.

  While we could skip the .clear() calls if caching is disabled,
  do it anyways just to be safe and in case you're debugging this
  in cuda-gdb with gdb.
*/
void
cuda_debugapi::clear_caches ()
{
  cuda_api_trace (_(__FUNCTION__));

  m_cache.m_call_depth.clear ();
  m_cache.m_syscall_call_depth.clear ();
  m_cache.m_valid_warps.clear ();
  m_cache.m_virtual_return_address.clear ();
  m_cache.m_error_pc.clear ();
  m_cache.m_warp_state.clear ();
}

cuda_api_state_t
cuda_api_get_state (void)
{
  return cuda_debugapi::instance ().get_state ();
}

int
cuda_api_get_ptid (void)
{
  return cuda_debugapi::instance ().get_api_ptid ();
}

int
cuda_debugapi::initialize ()
{
  if (m_api_state == CUDA_API_STATE_INITIALIZED)
    return 0;

  clear_caches ();

  /* Save the inferior_ptid that we are initializing. */
  m_api_ptid = cuda_gdb_get_tid_or_pid (inferior_ptid);
  // TODO: get the minor, major, and revision
  CUDBGResult res = m_cudbgAPI->initialize ();
  cuda_api_print_api_call_result (res);
  cuda_api_handle_initialization_error (res);

  return (res != CUDBG_SUCCESS && res != CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED);
}

int
cuda_api_initialize (void)
{
  return cuda_debugapi::instance ().initialize ();
}


void
cuda_debugapi::finalize ()
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  clear_caches ();
  cuda_api_clear_state ();

  CUDBGResult res;
  if (cuda_remote)
    res = cuda_remote_api_finalize (cuda_get_current_remote_target ());
  else
    res = m_cudbgAPI->finalize ();
  cuda_api_handle_finalize_api_error (res);
}

void
cuda_api_finalize (void)
{
  cuda_debugapi::instance ().finalize ();
}


const char* cuda_api_mask_string(const cuda_api_warpmask* m)
{
    static char buf[sizeof(m->mask)*2 + 3] = {0};
    sprintf(buf, "0x%.*llx", (int)sizeof(buf) - 3, (unsigned long long)(m->mask));
    return buf;
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
cuda_api_and_mask(cuda_api_warpmask* dst, const cuda_api_warpmask* m1, const cuda_api_warpmask* m2) {
    dst->mask = m1->mask & m2->mask;
    return dst;
}

cuda_api_warpmask*
cuda_api_not_mask(cuda_api_warpmask* dst, const cuda_api_warpmask* m1) {
    dst->mask = ~m1->mask;
    return dst;
}

static void
cuda_api_trace (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_API, fmt, ap);
  va_end (ap);
}

static void
cuda_api_print_api_call_result (int res)
{
  char errStrEx[cuda_debugapi::ErrorStringExMaxLength] = {0};
  cuda_api_get_error_string_ex(errStrEx, sizeof(errStrEx), nullptr);
  if (res != CUDBG_SUCCESS)
    cuda_api_trace ("API call received result: %s(0x%x), error message=%s",
                    cudbgGetErrorString((CUDBGResult) res), res, errStrEx);
}

static void cuda_api_error(CUDBGResult, const char *, ...) ATTRIBUTE_PRINTF (2, 3);

static void
cuda_api_error(CUDBGResult res, const char *fmt, ...)
{
  va_list args;
  char errStr[cuda_debugapi::cuda_debugapi::ErrorStringMaxLength] = {0};
  char errStrEx[cuda_debugapi::ErrorStringExMaxLength] = {0};

  va_start (args, fmt);
  vsnprintf (errStr,sizeof(errStr), fmt, args);
  va_end (args);

  cuda_api_get_error_string_ex(errStrEx, sizeof(errStrEx), nullptr);
  throw_error (GENERIC_ERROR, "Error: %s, error=%s(0x%x), error message=%s.\n",
               errStr, cudbgGetErrorString(res), res, errStrEx);
}

cuda_debugapi::cuda_debugapi ()
  : m_caching_enabled (true)
{
  const char *str = getenv ("CUDA_GDB_DISABLE_DEBUGAPI_CACHING");

  if (str)
    m_caching_enabled = !strcmp(str, "0") ? true : false;
}

void
cuda_api_set_api (CUDBGAPI api)
{
  cuda_debugapi::instance ().set_api (api);
}

void
cuda_api_handle_get_api_error (CUDBGResult res)
{
  switch (res)
    {
      case CUDBG_SUCCESS:
        return;

      case CUDBG_ERROR_INITIALIZATION_FAILURE:
        fprintf_unfiltered (gdb_stderr,
                            "The CUDA driver failed initialization. "
                            "Likely cause is X running on all devices.\n");
        break;

      default:
        fprintf_unfiltered (gdb_stderr,
                            "The CUDA Debugger API failed with error %d.\n",
                            res);
        break;
    }

  fprintf_unfiltered (gdb_stderr, "[CUDA Debugging is disabled]\n");
}

static void
cuda_api_fatal (const char *msg, CUDBGResult res)
{
  if (cuda_remote)
    {
      cuda_remote_api_finalize (cuda_get_current_remote_target ());
      target_kill ();
    }
  else
    {
      /* Finalize API */
      cuda_debugapi::instance ().finalize ();

      /* Kill inferior */
      kill (cuda_gdb_get_tid_or_pid (inferior_ptid), SIGKILL);
    }

  cuda_managed_memory_clean_regions ();

  /* Report error */
  throw_quit (_("fatal:  %s (error code = %s(0x%x)"), msg, cudbgGetErrorString(res), res);
}

void
cuda_debugapi::handle_initialization_error (CUDBGResult res)
{
  switch (res)
    {
    case CUDBG_SUCCESS:
      m_api_state = CUDA_API_STATE_INITIALIZED;
      break;

    case CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED:
      warning (_("One or more CUDA devices are made unavailable to the application "
                 "because they are used for display and cannot be used while debugging. "
                 "This may change the application behavior."));
      m_api_state = CUDA_API_STATE_INITIALIZED;
      break;

    case CUDBG_ERROR_UNINITIALIZED:
      /* Not ready yet. Will try later. */
      m_api_state = CUDA_API_STATE_INITIALIZING;
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
cuda_api_handle_initialization_error (CUDBGResult res)
{
  cuda_debugapi::instance ().handle_initialization_error (res);
}


void
cuda_debugapi::clear_state ()
{
  /* Mark the API as not initialized as early as possible. If the finalize()
   * call fails, we won't try to do anything stupid afterwards. */
  m_api_state = CUDA_API_STATE_UNINITIALIZED;
  cuda_set_uvm_used (false);

  m_attach_state = CUDA_ATTACH_STATE_NOT_STARTED;
  cuda_managed_memory_clean_regions ();
}

void
cuda_api_clear_state (void)
{
  cuda_debugapi::instance ().clear_state ();
}

void
cuda_api_handle_finalize_api_error (CUDBGResult res)
{
  if (!cuda_debugapi::instance ().get_api_initialized ())
    return;

  cuda_api_print_api_call_result (res);

  /* Only emit a warning in case of a failure, because cuda_api_finalize () can
     be called when an error occurs. That would create an infinite loop and/or
     undesired side effects. */
  if (res != CUDBG_SUCCESS)
    warning (_("Failed to finalize the CUDA debugger API (error=%u).\n"), res);
}

void
cuda_debugapi::initialize_attach_stub ()
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  /* Mark the API as not initialized as early as possible. If the finalize()
   * call fails, we won't try to do anything stupid afterwards. */
  m_api_state = CUDA_API_STATE_UNINITIALIZED;
  cuda_set_uvm_used (false);

  CUDBGResult res = m_cudbgAPI->initializeAttachStub ();
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to initialize attach stub"));
}

void
cuda_api_initialize_attach_stub (void)
{
  cuda_debugapi::instance ().initialize_attach_stub ();
}


void
cuda_debugapi::read_grid_id (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *grid_id)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readGridId (dev, sm, wp, grid_id);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the grid index"), dev, sm, wp, res);
}

void
cuda_api_read_grid_id (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *grid_id)
{
  cuda_debugapi::instance ().read_grid_id (dev, sm, wp, grid_id);
}


void
cuda_debugapi::read_block_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readBlockIdx (dev, sm, wp, blockIdx);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the block index"), dev, sm, wp, res);
}

void
cuda_api_read_block_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx)
{
  cuda_debugapi::instance ().read_block_idx (dev, sm, wp, blockIdx);
}


void
cuda_debugapi::read_cluster_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *cluster_idx)
{
  memset(cluster_idx, 0, sizeof(*cluster_idx));

  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readClusterIdx (dev, sm, wp, cluster_idx);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_NOT_SUPPORTED)
    cuda_devsmwp_api_error (_("read the cluster index"), dev, sm, wp, res);
}

void
cuda_api_read_cluster_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *cluster_idx)
{
  cuda_debugapi::instance ().read_cluster_idx (dev, sm, wp, cluster_idx);
}


void
cuda_debugapi::read_active_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *active_lanes)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readActiveLanes (dev, sm, wp, active_lanes);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the active lanes mask"), dev, sm, wp, res);
}

void
cuda_api_read_active_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *active_lanes)
{
  cuda_debugapi::instance ().read_active_lanes (dev, sm, wp, active_lanes);
}


void
cuda_debugapi::read_code_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readCodeMemory (dev, addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read code memory at address 0x%llx on device %u"),
                    (unsigned long long)addr, dev);
}

void
cuda_api_read_code_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz)
{
  cuda_debugapi::instance ().read_code_memory (dev, addr, buf, sz);
}


void
cuda_debugapi::read_const_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readConstMemory (dev, addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read const memory at address 0x%llx on device %u"),
                    (unsigned long long)addr, dev);
}

void
cuda_api_read_const_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz)
{
  cuda_debugapi::instance ().read_const_memory (dev, addr, buf, sz);
}


void
cuda_debugapi::read_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz)
{
  uint64_t hostaddr;

  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readGenericMemory (dev, sm, wp, ln, addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res == CUDBG_ERROR_MISSING_DATA) {
    cuda_api_error (res, _("Generic memory is not available in this corefile"));
    return;
  }

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM)
    cuda_api_error (res, _("Failed to read generic memory at address 0x%llx"
                    " on device %u sm %u warp %u lane %u"),
                    (unsigned long long)addr, dev, sm, wp, ln);

  if (res == CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM)
    {
      res = m_cudbgAPI->getHostAddrFromDeviceAddr (dev, addr, &hostaddr);
      cuda_api_print_api_call_result (res);
      if (res != CUDBG_SUCCESS)
        cuda_api_error (res, _("Failed to translate device VA to host VA"));
      read_memory (hostaddr, (gdb_byte *) buf, sz);
    }
}

void
cuda_api_read_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz)
{
  cuda_debugapi::instance ().read_generic_memory (dev, sm, wp, ln, addr, buf, sz);
}


bool
cuda_debugapi::read_pinned_memory (uint64_t addr, void *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return false;

  CUDBGResult res = m_cudbgAPI->readPinnedMemory (addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_MEMORY_MAPPING_FAILED)
    cuda_api_error (res, _("Failed to read pinned memory at address 0x%llx"), (unsigned long long)addr);

  return res == CUDBG_SUCCESS;
}

bool
cuda_api_read_pinned_memory (uint64_t addr, void *buf, uint32_t sz)
{
  cuda_debugapi::instance ().read_pinned_memory (addr, buf, sz);
}


void
cuda_debugapi::read_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readParamMemory (dev, sm, wp, addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read param memory at address 0x%llx"
                    " on device %u sm %u warp %u"),
                    (unsigned long long)addr, dev, sm, wp);
}

void
cuda_api_read_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz)
{
  cuda_debugapi::instance ().read_param_memory (dev, sm, wp, addr, buf, sz);
}


void
cuda_debugapi::read_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readSharedMemory (dev, sm, wp, addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res == CUDBG_ERROR_MISSING_DATA) {
    cuda_api_error (res, _("Shared memory is not available in this corefile"));
    return;
  }

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read shared memory at address 0x%llx"
                    " on device %u sm %u warp %u"),
                    (unsigned long long)addr, dev, sm, wp);
}

void
cuda_api_read_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz)
{
  cuda_debugapi::instance ().read_shared_memory (dev, sm, wp, addr, buf, sz);
}

bool
cuda_debugapi::read_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return false;

  CUDBGResult res = m_cudbgAPI->readLocalMemory (dev, sm, wp, ln, addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res == CUDBG_ERROR_MISSING_DATA)
    cuda_api_error (res, _("Local memory is not available in this corefile"));

  if (res != CUDBG_SUCCESS)
    {
      cuda_api_error (res, _("Failed to read local memory at address 0x%llx"
		      " on device %u sm %u warp %u lane %u"),
		      (unsigned long long)addr, dev, sm, wp, ln);
      return false;
    }

  return true;
}

bool
cuda_api_read_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz)
{
  return cuda_debugapi::instance ().read_local_memory (dev, sm, wp, ln, addr, buf, sz);
}


void
cuda_debugapi::read_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
			      uint32_t regno, uint32_t *val)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readRegister (dev, sm, wp, ln, regno, val);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
      cuda_api_error (res, _("Failed to read register %d (dev=%u, sm=%u, wp=%u, ln=%u)"),
                      regno, dev, sm, wp, ln);
}

void
cuda_api_read_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
                        uint32_t regno, uint32_t *val)
{
  cuda_debugapi::instance ().read_register (dev, sm, wp, ln, regno, val);
}


void
cuda_debugapi::read_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t *val)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readUniformRegisterRange (dev, sm, wp, regno, 1, val);
  cuda_api_print_api_call_result (res);

  /* Not all devices support uniform registers */
  if (res == CUDBG_ERROR_INVALID_DEVICE || res == CUDBG_ERROR_MISSING_DATA)
    *val = 0;
  else if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read uniform register %d (dev=%u, sm=%u, wp=%u)"),
		    regno, dev, sm, wp);
}

void
cuda_api_read_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t *val)
{
  cuda_debugapi::instance ().read_uregister (dev, sm, wp, regno, val);
}


void
cuda_debugapi::read_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
				uint32_t predicates_size, uint32_t *predicates)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readPredicates (dev, sm, wp, ln, predicates_size, predicates);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
      cuda_devsmwpln_api_error (_("read predicates"), dev, sm, wp, ln, res);
}

void
cuda_api_read_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
                          uint32_t predicates_size, uint32_t *predicates)
{
  cuda_debugapi::instance ().read_predicates (dev, sm, wp, ln, predicates_size, predicates);
}


void
cuda_debugapi::read_upredicates (uint32_t dev, uint32_t sm, uint32_t wp,
				 uint32_t predicates_size, uint32_t *predicates)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readUniformPredicates (dev, sm, wp, predicates_size, predicates);
  cuda_api_print_api_call_result (res);

  /* Not all devices support uniform registers */
  if (res == CUDBG_ERROR_INVALID_DEVICE || res == CUDBG_ERROR_MISSING_DATA)
    memset (predicates, 0, predicates_size);
  else if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read uniform predicates (dev=%u, sm=%u, wp=%u)"),
		    dev, sm, wp);
}

void
cuda_api_read_upredicates (uint32_t dev, uint32_t sm, uint32_t wp,
			   uint32_t predicates_size, uint32_t *predicates)
{
  cuda_debugapi::instance ().read_upredicates (dev, sm, wp, predicates_size, predicates);
}


void
cuda_debugapi::read_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *val)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readCCRegister (dev, sm, wp, ln, val);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
      cuda_devsmwpln_api_error (_("read CC register"), dev, sm, wp, ln, res);
}

void
cuda_api_read_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *val)
{
  cuda_debugapi::instance ().read_cc_register (dev, sm, wp, ln, val);
}


void
cuda_debugapi::read_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readPC (dev, sm, wp, ln, pc);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("read the program counter"), dev, res);
}

void
cuda_api_read_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc)
{
  cuda_debugapi::instance ().read_pc (dev, sm, wp, ln, pc);
}


void
cuda_debugapi::read_virtual_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readVirtualPC (dev, sm, wp, ln, pc);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("read the virtual PC"), dev, res);
}

void
cuda_api_read_virtual_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc)
{
  cuda_debugapi::instance ().read_virtual_pc (dev, sm, wp, ln, pc);
}


void
cuda_debugapi::read_lane_exception (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readLaneException (dev, sm, wp, ln, exception);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
      cuda_devsmwpln_api_error (_("read the lane exception"), dev, sm, wp, ln, res);
}

void
cuda_api_read_lane_exception (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception)
{
  cuda_debugapi::instance ().read_lane_exception (dev, sm, wp, ln, exception);
}


void
cuda_debugapi::read_device_exception_state (uint32_t dev, uint64_t *exceptionSMMask, uint32_t n)
{
  gdb_assert (exceptionSMMask);

  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readDeviceExceptionState (dev, exceptionSMMask, n);
  cuda_api_print_api_call_result(res);

  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("read device exception state"),dev, res);
}

void
cuda_api_read_device_exception_state (uint32_t dev, uint64_t *exceptionSMMask, uint32_t n)
{
  cuda_debugapi::instance ().read_device_exception_state (dev, exceptionSMMask, n);
}


void
cuda_debugapi::write_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->writeGenericMemory (dev, sm, wp, ln, addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM)
    cuda_api_error (res, _("Failed to write generic memory at address 0x%llx"
                         " on device %u sm %u warp %u lane %u"),
                         (unsigned long long)addr, dev, sm, wp, ln);

  if (res == CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM)
    {
      uint64_t hostaddr;
      res = m_cudbgAPI->getHostAddrFromDeviceAddr (dev, addr, &hostaddr);
      cuda_api_print_api_call_result (res);

      if (res != CUDBG_SUCCESS)
        cuda_api_error (res, _("Failed to translate device VA to host VA"));
      write_memory (hostaddr, (const gdb_byte *) buf, sz);
    }
}

void
cuda_api_write_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz)
{
  cuda_debugapi::instance ().write_generic_memory (dev, sm, wp, ln, addr, buf, sz);
}


bool
cuda_debugapi::write_pinned_memory (uint64_t addr, const void *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return false;

  CUDBGResult res = m_cudbgAPI->writePinnedMemory (addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_MEMORY_MAPPING_FAILED)
    cuda_api_error (res, _("Failed to write pinned memory at address 0x%llx"), (unsigned long long)addr);
  return res == CUDBG_SUCCESS;
}

bool
cuda_api_write_pinned_memory (uint64_t addr, const void *buf, uint32_t sz)
{
  cuda_debugapi::instance ().write_pinned_memory (addr, buf, sz);
}


void
cuda_debugapi::write_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->writeParamMemory (dev, sm, wp, addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to write param memory at address 0x%llx"
                    " on device %u sm %u warp %u"),
                    (unsigned long long)addr, dev, sm, wp);
}

void
cuda_api_write_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz)
{
  cuda_debugapi::instance ().write_param_memory (dev, sm, wp, addr, buf, sz);
}


void
cuda_debugapi::write_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->writeSharedMemory (dev, sm, wp, addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to write shared memory at address 0x%llx"
                    " on device %u sm %u warp %u"),
                    (unsigned long long)addr, dev, sm, wp);
}

void
cuda_api_write_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz)
{
  cuda_debugapi::instance ().write_shared_memory (dev, sm, wp, addr, buf, sz);
}


bool
cuda_debugapi::write_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return false;

  CUDBGResult res = m_cudbgAPI->writeLocalMemory (dev, sm, wp, ln, addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    {
      cuda_api_error (res, _("Failed to write local memory at address 0x%llx"
		      " on device %u sm %u warp %u lane %u"),
		      (unsigned long long)addr, dev, sm, wp, ln);
      return false;
    }

  return true;
}

bool
cuda_api_write_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz)
{
  return cuda_debugapi::instance ().write_local_memory (dev, sm, wp, ln, addr, buf, sz);
}


void
cuda_debugapi::write_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->writeRegister (dev, sm, wp, ln, regno, val);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
      cuda_api_error (res, _("Failed to write register %d (dev=%u, sm=%u, wp=%u, ln=%u)"),
                      regno, dev, sm, wp, ln);
}

void
cuda_api_write_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val)
{
  cuda_debugapi::instance ().write_register (dev, sm, wp, ln, regno, val);
}


void
cuda_debugapi::write_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t val)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->writeUniformRegister (dev, sm, wp, regno, val);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
      cuda_api_error (res, _("Failed to write uniform register %d (dev=%u, sm=%u, wp=%u)"),
                      regno, dev, sm, wp);
}

void
cuda_api_write_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t val)
{
  cuda_debugapi::instance ().write_uregister (dev, sm, wp, regno, val);
}


void
cuda_debugapi::write_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->writePredicates (dev, sm, wp, ln, predicates_size, predicates);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
      cuda_devsmwpln_api_error (_("write predicates"), dev, sm, wp, ln, res);
}

void
cuda_api_write_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates)
{
  cuda_debugapi::instance ().write_predicates (dev, sm, wp, ln, predicates_size, predicates);
}


void
cuda_debugapi::write_upredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, const uint32_t *predicates)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->writeUniformPredicates (dev, sm, wp, predicates_size, predicates);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
      cuda_api_error (res, _("Failed to write uniform predicates (dev=%u, sm=%u, wp=%u)"),
                      dev, sm, wp);
}

void
cuda_api_write_upredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, const uint32_t *predicates)
{
  cuda_debugapi::instance ().write_upredicates (dev, sm, wp, predicates_size, predicates);
}


void
cuda_debugapi::write_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->writeCCRegister (dev, sm, wp, ln, val);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
      cuda_devsmwpln_api_error (_("write CC register"), dev, sm, wp, ln, res);
}

void
cuda_api_write_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val)
{
  cuda_debugapi::instance ().write_cc_register (dev, sm, wp, ln, val);
}


void
cuda_debugapi::get_grid_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *grid_dim)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getGridDim (dev, sm, wp, grid_dim);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the grid dimensions "), dev, sm, wp, res);
}

void
cuda_api_get_grid_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *grid_dim)
{
  cuda_debugapi::instance ().get_grid_dim (dev, sm, wp, grid_dim);
}


void
cuda_debugapi::get_cluster_dim (uint32_t dev, uint64_t gridId64, CuDim3 *cluster_dim)
{
  memset(cluster_dim, 0, sizeof(*cluster_dim));

  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getClusterDim (dev, gridId64, cluster_dim);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_NOT_SUPPORTED)
    cuda_dev_api_error (_("read the cluster dimensions"), dev, res);
}

void
cuda_api_get_cluster_dim (uint32_t dev, uint64_t gridId64, CuDim3 *cluster_dim)
{
  cuda_debugapi::instance ().get_cluster_dim (dev, gridId64, cluster_dim);
}


void
cuda_debugapi::get_block_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *block_dim)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getBlockDim (dev, sm, wp, block_dim);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the block dimensions"), dev, sm, wp, res);
}

void
cuda_api_get_block_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *block_dim)
{
  cuda_debugapi::instance ().get_block_dim (dev, sm, wp, block_dim);
}


void
cuda_debugapi::get_blocking (uint32_t dev, uint32_t sm, uint32_t wp, bool *blocking)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  uint64_t blocking64;
  CUDBGResult res = m_cudbgAPI->getGridAttribute (dev, sm, wp, CUDBG_ATTR_GRID_LAUNCH_BLOCKING, &blocking64);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the grid blocking attribute"), dev, sm, wp, res);

  *blocking = !!blocking64;
}

void
cuda_api_get_blocking (uint32_t dev, uint32_t sm, uint32_t wp, bool *blocking)
{
  cuda_debugapi::instance ().get_blocking (dev, sm, wp, blocking);
}


void
cuda_debugapi::get_tid (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getTID (dev, sm, wp, tid);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("get thread id"), dev, sm, wp, res);
}

void
cuda_api_get_tid (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid)
{
  cuda_debugapi::instance ().get_tid (dev, sm, wp, tid);
}


void
cuda_debugapi::get_elf_image (uint32_t dev,  uint64_t handle, bool relocated,
			      void *elfImage, uint64_t size)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getElfImageByHandle (dev, handle, relocated ? CUDBG_ELF_IMAGE_TYPE_RELOCATED : CUDBG_ELF_IMAGE_TYPE_NONRELOCATED, elfImage, size);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read the ELF image (dev=%u, handle=%llu, relocated=%d)"),
           dev, (unsigned long long)handle, relocated);
}

void
cuda_api_get_elf_image (uint32_t dev,  uint64_t handle, bool relocated,
                        void *elfImage, uint64_t size)
{
  cuda_debugapi::instance ().get_elf_image (dev, handle, relocated, elfImage, size);
}


void
cuda_debugapi::get_device_type (uint32_t dev, char *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getDeviceType (dev, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the device type"), dev, res);
}

void
cuda_api_get_device_type (uint32_t dev, char *buf, uint32_t sz)
{
  cuda_debugapi::instance ().get_device_type (dev, buf, sz);
}


void
cuda_debugapi::get_sm_type (uint32_t dev, char *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getSmType (dev, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the SM type"), dev, res);
}

void
cuda_api_get_sm_type (uint32_t dev, char *buf, uint32_t sz)
{
  cuda_debugapi::instance ().get_sm_type (dev, buf, sz);
}


void
cuda_debugapi::get_device_name (uint32_t dev, char *buf, uint32_t sz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getDeviceName (dev, buf, sz);
  cuda_api_print_api_call_result (res);
 
 if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the device name"), dev, res);
}

void
cuda_api_get_device_name (uint32_t dev, char *buf, uint32_t sz)
{
  cuda_debugapi::instance ().get_device_name (dev, buf, sz);
}


void
cuda_debugapi::get_num_devices (uint32_t *numDev)
{
  *numDev = 0;

  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getNumDevices (numDev);
  cuda_api_print_api_call_result (res);
 
 if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get the number of devices"));
}

void
cuda_api_get_num_devices (uint32_t *numDev)
{
  cuda_debugapi::instance ().get_num_devices (numDev);
}



void
cuda_debugapi::get_num_sms (uint32_t dev, uint32_t *numSMs)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getNumSMs (dev, numSMs);
  cuda_api_print_api_call_result (res);
 
 if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get the number of SMs (dev=%u)"), dev);
}

void
cuda_api_get_num_sms (uint32_t dev, uint32_t *numSMs)
{
  cuda_debugapi::instance ().get_num_sms (dev, numSMs);
}


void
cuda_debugapi::get_num_warps (uint32_t dev, uint32_t *numWarps)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getNumWarps (dev, numWarps);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get the number of warps (dev=%u)"), dev);
}

void
cuda_api_get_num_warps (uint32_t dev, uint32_t *numWarps)
{
  cuda_debugapi::instance ().get_num_warps (dev, numWarps);
}


void
cuda_debugapi::get_num_lanes (uint32_t dev, uint32_t *numLanes)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getNumLanes (dev, numLanes);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the number of lanes"), dev, res);
}

void
cuda_api_get_num_lanes (uint32_t dev, uint32_t *numLanes)
{
  cuda_debugapi::instance ().get_num_lanes (dev, numLanes);
}


void
cuda_debugapi::get_num_registers (uint32_t dev, uint32_t *numRegs)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getNumRegisters (dev, numRegs);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the number of registers"), dev, res);
}

void
cuda_api_get_num_registers (uint32_t dev, uint32_t *numRegs)
{
  cuda_debugapi::instance ().get_num_registers (dev, numRegs);
}


void
cuda_debugapi::get_num_predicates (uint32_t dev, uint32_t *numPredicates)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getNumPredicates (dev, numPredicates);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the number of predicates"), dev, res);
}

void
cuda_api_get_num_predicates (uint32_t dev, uint32_t *numPredicates)
{
  cuda_debugapi::instance ().get_num_predicates (dev, numPredicates);
}


void
cuda_debugapi::get_num_uregisters (uint32_t dev, uint32_t *numRegs)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getNumUniformRegisters (dev, numRegs);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the number of uniform registers"), dev, res);
}

void
cuda_api_get_num_uregisters (uint32_t dev, uint32_t *numRegs)
{
  cuda_debugapi::instance ().get_num_uregisters (dev, numRegs);
}


void
cuda_debugapi::get_num_upredicates (uint32_t dev, uint32_t *numPredicates)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getNumUniformPredicates (dev, numPredicates);
  cuda_api_print_api_call_result (res);
 
 if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the number of uniform predicates"), dev, res);
}

void
cuda_api_get_num_upredicates (uint32_t dev, uint32_t *numPredicates)
{
  cuda_debugapi::instance ().get_num_upredicates (dev, numPredicates);
}


void
cuda_debugapi::is_device_code_address (uint64_t addr, bool *is_device_address)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    {
      *is_device_address = false;
      return;
    }

  CUDBGResult res = m_cudbgAPI->isDeviceCodeAddress (addr, is_device_address);
  cuda_api_print_api_call_result (res);
 
 if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to determine if address 0x%llx corresponds "
             "to the host or device"), (unsigned long long)addr);
}

void
cuda_api_is_device_code_address (uint64_t addr, bool *is_device_address)
{
  cuda_debugapi::instance ().is_device_code_address (addr, is_device_address);
}


void
cuda_api_handle_set_callback_api_error (CUDBGResult res)
{
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to set the new event callback"));
}

void
cuda_debugapi::set_notify_new_event_callback (CUDBGNotifyNewEventCallback callback)
{
  /* Nothing should restrict the callback from being setup.
     In particular, it must be done prior to the API being
     fully initialized, which means there should not be a
     check here. */

  CUDBGResult res = m_cudbgAPI->setNotifyNewEventCallback (callback);
  cuda_api_print_api_call_result (res);
  cuda_api_handle_set_callback_api_error (res);
}

void
cuda_api_set_notify_new_event_callback (CUDBGNotifyNewEventCallback callback)
{
  cuda_debugapi::instance ().set_notify_new_event_callback (callback);
}


void
cuda_debugapi::get_next_sync_event (CUDBGEvent *event)
{
  event->kind = CUDBG_EVENT_INVALID;
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getNextEvent (CUDBG_EVENT_QUEUE_TYPE_SYNC, event);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_NO_EVENT_AVAILABLE)
    cuda_api_error (res, _("Failed to get the next sync CUDA event"));
}

void
cuda_api_get_next_sync_event (CUDBGEvent *event)
{
  cuda_debugapi::instance ().get_next_sync_event (event);
}


void
cuda_debugapi::acknowledge_sync_events ()
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->acknowledgeSyncEvents ();
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to acknowledge a sync CUDA event"));
}

void
cuda_api_acknowledge_sync_events (void)
{
  cuda_debugapi::instance ().acknowledge_sync_events ();
}


void
cuda_debugapi::get_next_async_event (CUDBGEvent *event)
{
  event->kind = CUDBG_EVENT_INVALID;
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getNextEvent (CUDBG_EVENT_QUEUE_TYPE_ASYNC, event);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_NO_EVENT_AVAILABLE)
    cuda_api_error (res, _("Failed to get the next async CUDA event"));
}

void
cuda_api_get_next_async_event (CUDBGEvent *event)
{
  cuda_debugapi::instance ().get_next_async_event (event);
}


void
cuda_debugapi::disassemble (uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t bufSize)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->disassemble (dev, addr, instSize, buf, bufSize);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to disassemble instruction at address 0x%llx on CUDA device %u"),
                    (unsigned long long)addr, dev);
}

void
cuda_api_disassemble (uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t bufSize)
{
  cuda_debugapi::instance ().disassemble (dev, addr, instSize, buf, bufSize);
}


void
cuda_debugapi::set_attach_state (cuda_attach_state_t state)
{
  m_attach_state = state;

  if (state != CUDA_ATTACH_STATE_DETACHING)
    return;

  CUDBGResult res = m_cudbgAPI->clearAttachState ();

  if (res != CUDBG_SUCCESS)
    warning (_("Failed to set attach state (error=%s(0x%x)).\n"), cudbgGetErrorString(res), res);
}

void
cuda_api_set_attach_state (cuda_attach_state_t state)
{
  cuda_debugapi::instance ().set_attach_state (state);
}


cuda_attach_state_t
cuda_api_get_attach_state ()
{
  return cuda_debugapi::instance ().get_attach_state ();
}


void
cuda_debugapi::request_cleanup_on_detach (uint32_t resumeAppFlag)
{
  CUDBGResult res = m_cudbgAPI->requestCleanupOnDetach (resumeAppFlag);

  if (res != CUDBG_SUCCESS)
    warning (_("Failed to clear attach state (error=%s(0x%x)).\n"), cudbgGetErrorString(res), res);
}

void
cuda_api_request_cleanup_on_detach (uint32_t resumeAppFlag)
{
  cuda_debugapi::instance ().request_cleanup_on_detach (resumeAppFlag);
}


void
cuda_debugapi::get_grid_status (uint32_t dev, uint64_t grid_id, CUDBGGridStatus *status)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getGridStatus (dev, grid_id, status);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get grid status (dev=%u, grid_id=%lld)"),
                    dev, (long long)grid_id);
}

void
cuda_api_get_grid_status (uint32_t dev, uint64_t grid_id, CUDBGGridStatus *status)
{
  cuda_debugapi::instance ().get_grid_status (dev, grid_id, status);
}


void
cuda_debugapi::get_grid_info (uint32_t dev, uint64_t grid_id, CUDBGGridInfo *info)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getGridInfo (dev, grid_id, info);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get grid info (dev=%u, grid_id=%lld"),
                    dev, (long long)grid_id);
}

void
cuda_api_get_grid_info (uint32_t dev, uint64_t grid_id, CUDBGGridInfo *info)
{
  cuda_debugapi::instance ().get_grid_info (dev, grid_id, info);
}


bool
cuda_api_attach_or_detach_in_progress (void)
{
  return cuda_debugapi::instance ().attach_or_detach_in_progress ();
}


void
cuda_debugapi::get_adjusted_code_address (uint32_t dev, uint64_t addr, uint64_t *adjusted_addr, CUDBGAdjAddrAction adj_action)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getAdjustedCodeAddress (dev, addr, adjusted_addr, adj_action);
  cuda_api_print_api_call_result (res);
 
 if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get adjusted code address (dev=%u, addr=0x%llx)"),
                    dev, (unsigned long long)addr);
}

void
cuda_api_get_adjusted_code_address (uint32_t dev, uint64_t addr, uint64_t *adjusted_addr, CUDBGAdjAddrAction adj_action)
{
  cuda_debugapi::instance ().get_adjusted_code_address (dev, addr, adjusted_addr, adj_action);
}


void
cuda_debugapi::set_kernel_launch_notification_mode(CUDBGKernelLaunchNotifyMode mode)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;
  m_cudbgAPI->setKernelLaunchNotificationMode (mode);
}

void
cuda_api_set_kernel_launch_notification_mode(CUDBGKernelLaunchNotifyMode mode)
{
  cuda_debugapi::instance ().set_kernel_launch_notification_mode (mode);
}


void
cuda_debugapi::get_device_pci_bus_info (uint32_t dev, uint32_t *pci_bus_id, uint32_t *pci_dev_id)
{
  *pci_bus_id = 0xffff;
  *pci_dev_id = 0xffff;

  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->getDevicePCIBusInfo (dev, pci_bus_id, pci_dev_id);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get PCI bus information"), dev, res);
}

void
cuda_api_get_device_pci_bus_info (uint32_t dev, uint32_t *pci_bus_id, uint32_t *pci_dev_id)
{
  cuda_debugapi::instance ().get_device_pci_bus_info (dev, pci_bus_id, pci_dev_id);
}


void
cuda_debugapi::read_warp_state (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState *state)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  bool cached = false;
  coordinates coords = coordinates (dev, sm, wp);

  if (m_caching_enabled)
    {
      const auto iter = m_cache.m_warp_state.find (coords);
      if (iter != m_cache.m_warp_state.end ())
	{
	  cached = true;
	  *state = iter->second;
	}
    }

  if (!cached)
    {
      CUDBGResult res = m_cudbgAPI->readWarpState (dev, sm, wp, state);
      cuda_api_print_api_call_result (res);

      if (res != CUDBG_SUCCESS)
	cuda_devsmwp_api_error (_("get warp state"), dev, sm, wp, res);

      if (m_caching_enabled)
	m_cache.m_warp_state[coords] = *state;
    }

  cuda_api_trace(_("%u/%u/%u/* %s%s: errorPC 0x%016lx"),
		 dev, sm, wp,
		 __FUNCTION__, cached ? " (cached)" : "",
		 state->errorPC);
}

void
cuda_api_read_warp_state (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState *state)
{
  cuda_debugapi::instance ().read_warp_state (dev, sm, wp, state);
}


void
cuda_debugapi::read_register_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t idx, uint32_t count, uint32_t *regs)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readRegisterRange (dev, sm, wp, ln, idx, count, regs);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read register range (dev=%u sm=%u wp=%u, ln=%u, idx=%u, count=%u)"),
           dev, sm, wp, ln, idx, count);
}

void
cuda_api_read_register_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t idx, uint32_t count, uint32_t *regs)
{
  cuda_debugapi::instance ().read_register_range (dev, sm, wp, ln,  idx, count, regs);
}


void
cuda_debugapi::read_uregister_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t idx, uint32_t count, uint32_t *regs)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readUniformRegisterRange (dev, sm, wp, idx, count, regs);
  cuda_api_print_api_call_result (res);

  if (res == CUDBG_ERROR_INVALID_DEVICE || res == CUDBG_ERROR_MISSING_DATA)
    memset (regs, 0, (count - idx) * sizeof(uint32_t));
  else if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read uniform register range (dev=%u sm=%u wp=%u, idx=%u, count=%u)"),
           dev, sm, wp, idx, count);
}

void
cuda_api_read_uregister_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t idx, uint32_t count, uint32_t *regs)
{
  cuda_debugapi::instance ().read_uregister_range (dev, sm, wp, idx, count, regs);
}


void
cuda_debugapi::read_global_memory (uint64_t addr, void *buf, uint32_t buf_size)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readGlobalMemory (addr, buf, buf_size);
  cuda_api_print_api_call_result (res);

  if (res == CUDBG_ERROR_MISSING_DATA) {
    cuda_api_error (res, _("Global memory is not available in this corefile"));
    return;
  }

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read %u bytes of global memory from 0x%llx\n"),
           buf_size, (unsigned long long)addr);
}

void
cuda_api_read_global_memory (uint64_t addr, void *buf, uint32_t buf_size)
{
  cuda_debugapi::instance ().read_global_memory (addr, buf, buf_size);
}


void
cuda_debugapi::write_global_memory (uint64_t addr, const void *buf, uint32_t buf_size)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->writeGlobalMemory (addr, (void *)buf, buf_size);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to write %u bytes of global memory to 0x%llx"),
                    buf_size, (unsigned long long)addr);
}

void
cuda_api_write_global_memory (uint64_t addr, const void *buf, uint32_t buf_size)
{
  cuda_debugapi::instance ().write_global_memory (addr, buf, buf_size);
}


void
cuda_debugapi::get_managed_memory_region_info (uint64_t start_addr, CUDBGMemoryInfo *meminfo, uint32_t entries_count, uint32_t *entries_written)
{
  if (entries_written)
    *entries_written = 0;
  if (get_state () != CUDA_API_STATE_INITIALIZED || !cuda_is_uvm_used())
    return;

  CUDBGResult res = m_cudbgAPI->getManagedMemoryRegionInfo (start_addr, meminfo, entries_count, entries_written);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read %u entries starting from addr 0x%llx"),
                    entries_count, (unsigned long long)start_addr);
}

void
cuda_api_get_managed_memory_region_info (uint64_t start_addr, CUDBGMemoryInfo *meminfo, uint32_t entries_count, uint32_t *entries_written)
{
  cuda_debugapi::instance ().get_managed_memory_region_info (start_addr, meminfo, entries_count, entries_written);
}

void
cuda_debugapi::suspend_device (uint32_t dev)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->suspendDevice (dev);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_SUSPENDED_DEVICE)
    cuda_dev_api_error (_("suspend device"), dev, res);
}

void
cuda_api_suspend_device (uint32_t dev)
{
  cuda_debugapi::instance ().suspend_device (dev);
}


void
cuda_debugapi::resume_device (uint32_t dev)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  clear_caches ();

  cuda_api_trace (_("%u %s"), dev, __FUNCTION__);

  CUDBGResult res = m_cudbgAPI->resumeDevice (dev);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_RUNNING_DEVICE)
    cuda_dev_api_error (_("resume device"), dev, res);
}

void
cuda_api_resume_device (uint32_t dev)
{
  cuda_debugapi::instance ().resume_device (dev);
}


bool
cuda_debugapi::set_breakpoint (uint32_t dev, uint64_t addr)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return true;

  cuda_api_trace (_("%u 0x%016lx %s"), dev, addr, __FUNCTION__);

  CUDBGResult res = m_cudbgAPI->setBreakpoint (dev, addr);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_INVALID_ADDRESS)
    cuda_api_error (res, _("Failed to set a breakpoint on device %u at address 0x%llx"),
                    dev, (unsigned long long)addr);
  return res != CUDBG_ERROR_INVALID_ADDRESS;
}

bool
cuda_api_set_breakpoint (uint32_t dev, uint64_t addr)
{
  return cuda_debugapi::instance ().set_breakpoint (dev, addr);
}


bool
cuda_debugapi::unset_breakpoint (uint32_t dev, uint64_t addr)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return true;

  cuda_api_trace (_("%u 0x%016lx %s"), dev, addr, __FUNCTION__);

  CUDBGResult res = m_cudbgAPI->unsetBreakpoint (dev, addr);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_INVALID_ADDRESS)
    cuda_api_error (res, _("Failed to unset a breakpoint on device %u at address 0x%llx"),
                    dev, (unsigned long long)addr);
  return res != CUDBG_ERROR_INVALID_ADDRESS;
}

bool
cuda_api_unset_breakpoint (uint32_t dev, uint64_t addr)
{
  return cuda_debugapi::instance ().unset_breakpoint (dev, addr);
}


void
cuda_debugapi::read_thread_idx (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readThreadIdx (dev, sm, wp, ln, threadIdx);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the thread index"), dev, sm, wp, res);
}

void
cuda_api_read_thread_idx (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx)
{
  cuda_debugapi::instance ().read_thread_idx (dev, sm, wp, ln, threadIdx);
}


void
cuda_debugapi::read_broken_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *brokenWarpsMask)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readBrokenWarps (dev, sm, &brokenWarpsMask->mask);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read the broken warps mask (dev=%u, sm=%u)"),
                    dev, sm);
  cuda_trace(_("Read broken warps: %" WARP_MASK_FORMAT), cuda_api_mask_string(brokenWarpsMask));
}

void
cuda_api_read_broken_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *brokenWarpsMask)
{
  cuda_debugapi::instance ().read_broken_warps (dev, sm, brokenWarpsMask);
}


void
cuda_debugapi::read_valid_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *valid_lanes)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  CUDBGResult res = m_cudbgAPI->readValidLanes (dev, sm, wp, valid_lanes);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the valid lanes mask"), dev, sm, wp, res);
}

void
cuda_api_read_valid_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *valid_lanes)
{
  cuda_debugapi::instance ().read_valid_lanes (dev, sm, wp, valid_lanes);
}


bool
cuda_debugapi::single_step_warp (uint32_t dev, uint32_t sm, uint32_t wp,
				 uint32_t nsteps, cuda_api_warpmask *warp_mask)
{
  gdb_assert (warp_mask);

  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return false;

  clear_caches ();

  cuda_api_trace (_("%u/%u/%u %s"), dev, sm, wp, __FUNCTION__);

  CUDBGResult res = m_cudbgAPI->singleStepWarp (dev, sm, wp, nsteps, &warp_mask->mask);
  cuda_api_print_api_call_result (res);

  if (res == CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE)
    return false;

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error ("single-step the warp", dev, sm, wp, res);

  return true;
}

bool
cuda_api_single_step_warp (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t nsteps, cuda_api_warpmask *warp_mask)
{
  return cuda_debugapi::instance ().single_step_warp (dev, sm, wp, nsteps, warp_mask);
}


bool
cuda_debugapi::resume_warps_until_pc (uint32_t dev, uint32_t sm, cuda_api_warpmask *warp_mask, uint64_t virt_pc)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return false;

  clear_caches ();
  cuda_api_trace (_("%u/%u %s"), dev, sm, __FUNCTION__);

  CUDBGResult res = m_cudbgAPI->resumeWarpsUntilPC (dev, sm, warp_mask->mask, virt_pc);
  cuda_api_print_api_call_result (res);

  if (res == CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE)
    return false;

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to resume warps (dev=%d, sm=%u, warp_mask=%s)"),
                    dev, sm, cuda_api_mask_string (warp_mask));
  return true;
}

/* Return true on success and false if resuming warps is not possible */
bool
cuda_api_resume_warps_until_pc (uint32_t dev, uint32_t sm, cuda_api_warpmask *warp_mask, uint64_t virt_pc)
{
  return cuda_debugapi::instance ().resume_warps_until_pc (dev, sm, warp_mask, virt_pc);
}


void
cuda_debugapi::read_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  bool cached = false;
  uint32_t api_call_depth;
  coordinates coords = coordinates (dev, sm, wp, ln);

  if (m_caching_enabled)
    {
      const auto iter = m_cache.m_call_depth.find (coords);
      if (iter != m_cache.m_call_depth.end ())
	{
	  cached = true;
	  api_call_depth = iter->second;
	}
    }

  if (!cached)
    {
      CUDBGResult res = m_cudbgAPI->readCallDepth (dev, sm, wp, ln, &api_call_depth);
      cuda_api_print_api_call_result (res);

      if (res != CUDBG_SUCCESS)
	cuda_devsmwpln_api_error (_(__FUNCTION__), dev, sm, wp, ln, res);

      if (m_caching_enabled)
	m_cache.m_call_depth[coords] = api_call_depth;
    }

  cuda_api_trace (_("%u/%u/%u/%u %s%s: %u"),
		  dev, sm, wp, ln,
		  __FUNCTION__,
		  cached ? " (cached)" : "",
		  api_call_depth);

  *depth = (int32_t) api_call_depth;
}

void
cuda_api_read_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth)
{
  cuda_debugapi::instance ().read_call_depth (dev, sm, wp, ln, depth);
}


void
cuda_debugapi::read_syscall_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  bool cached = false;
  uint32_t api_call_depth;
  coordinates coords = coordinates (dev, sm, wp, ln);

  if (m_caching_enabled)
    {
      auto iter = m_cache.m_syscall_call_depth.find (coords);
      if (iter != m_cache.m_syscall_call_depth.end ())
	{
	  cached = true;
	  api_call_depth = iter->second;
	}
    }
  
  if (!cached)
    {
      CUDBGResult res = m_cudbgAPI->readSyscallCallDepth (dev, sm, wp, ln, &api_call_depth);
      cuda_api_print_api_call_result (res);

      if (res != CUDBG_SUCCESS)
	cuda_devsmwpln_api_error (_(__FUNCTION__), dev, sm, wp, ln, res);

      if (m_caching_enabled)
	m_cache.m_syscall_call_depth[coords] = api_call_depth;
    }

  cuda_api_trace (_("%u/%u/%u/%u %s%s: %u"),
		  dev, sm, wp, ln,
		  __FUNCTION__,
		  cached ? " (cached)" : "",
		  api_call_depth);

  *depth = (int32_t) api_call_depth;
}

void
cuda_api_read_syscall_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth)
{
  cuda_debugapi::instance ().read_syscall_call_depth (dev, sm, wp, ln, depth);
}


void
cuda_debugapi::read_valid_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *valid_warps)
{  
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  bool cached = false;
  coordinates coords = coordinates (dev, sm, 0, 0);

  if (m_caching_enabled)
    {
      auto iter = m_cache.m_valid_warps.find (coords);
      if (iter != m_cache.m_valid_warps.end ())
	{
	  cached = true;
	  *valid_warps = iter->second;
	}
    }

  if (!cached)
    {
      CUDBGResult res = m_cudbgAPI->readValidWarps (dev, sm, &valid_warps->mask);
      cuda_api_print_api_call_result (res);

      if (res != CUDBG_SUCCESS)
	cuda_api_error (res, _("Failed to read the valid warps mask (dev=%u, sm=%u)"),
			dev, sm);

      if (m_caching_enabled)
	m_cache.m_valid_warps[coords] = *valid_warps;
  }

  cuda_api_trace(_("%u/%u %s%s: %" WARP_MASK_FORMAT),
		 dev, sm,
		 __FUNCTION__,
		 cached ? " (cached)" : "",
		 cuda_api_mask_string(valid_warps));
}

void
cuda_api_read_valid_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *valid_warps)
{
  cuda_debugapi::instance ().read_valid_warps (dev, sm, valid_warps);
}


void
cuda_debugapi::read_virtual_return_address (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
					    int32_t level, uint64_t *ra)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  bool cached = false;
  uint32_t api_call_level;

  gdb_assert (level >= 0);
  api_call_level = (uint32_t) level;

  frame_coordinates fc = frame_coordinates (dev, sm, wp, ln, level);
  if (m_caching_enabled)
    {
      auto iter = m_cache.m_virtual_return_address.find (fc);
      if (iter != m_cache.m_virtual_return_address.end ())
	{
	  cached = true;
	  *ra = iter->second;
	}
    }
	  
  if (!cached)
    {
      CUDBGResult res = m_cudbgAPI->readVirtualReturnAddress (dev, sm, wp, ln, api_call_level, ra);
      cuda_api_print_api_call_result (res);

      if (res != CUDBG_SUCCESS)
	{
	  if (res == CUDBG_ERROR_INVALID_CALL_LEVEL)
	    cuda_api_error (res, _("Debugger API returned invalid call level for level %u"), api_call_level);
	  else
	    cuda_api_error (res, _("Could not read virtual return address for level %u "
				   "(dev=%u, sm=%u, warp=%u, lane=%u)"),
			    api_call_level, dev, sm, wp, ln);
	}

      if (m_caching_enabled)
	m_cache.m_virtual_return_address[fc] = *ra;
  }

  cuda_api_trace(_("%u/%u/%u/%u %s%s: level %u 0x%016lx"),
		 dev, sm, wp, ln,
		 __FUNCTION__,
		 cached ? " (cached)" : "",
		 api_call_level,
		 *ra);
}


void
cuda_api_read_virtual_return_address (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
                                      int32_t level, uint64_t *ra)
{
  cuda_debugapi::instance ().read_virtual_return_address (dev, sm, wp, ln, level, ra);
}

void
cuda_debugapi::read_error_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *pc, bool* valid)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  // Clear output parameters
  *pc = 0;
  *valid = false;

  bool cached = false;
  std::tuple<uint64_t, bool> value;
  coordinates coords = coordinates (dev, sm, wp, 0);

  if (m_caching_enabled)
    {
      auto iter = m_cache.m_error_pc.find (coords);
      if (iter != m_cache.m_error_pc.end ())
	{
	  cached = true;
	  value = iter->second;
	}
    }

  if (!cached)
    {
      CUDBGResult res = m_cudbgAPI->readErrorPC (dev, sm, wp, pc, valid);
      cuda_api_print_api_call_result (res);

      if (res != CUDBG_SUCCESS)
	cuda_devsmwp_api_error (_("Could not read error PC "), dev, sm, wp, res);

      value = std::tuple<uint64_t, bool> (*pc, *valid);
      if (m_caching_enabled)
	m_cache.m_error_pc[coords] = value;
    }

  *pc = std::get<0> (value);
  *valid = std::get<1> (value);
  cuda_api_trace(_("%u/%u/%u/* %s%s: valid %u pc 0x%016lx "),
		 dev, sm, wp,
		 __FUNCTION__,
		 cached ? " (cached)" : "",
		 *valid, *pc);
}

void
cuda_api_read_error_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *pc, bool* valid)
{
  cuda_debugapi::instance ().read_error_pc (dev, sm, wp, pc, valid);
}

#if CUDBG_API_VERSION_REVISION >= 132
void
cuda_debugapi::get_loaded_function_info (uint32_t dev, uint64_t handle, CUDBGLoadedFunctionInfo *info, uint32_t numEntries)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

#if (CUDBG_API_VERSION_REVISION >= 131)
  CUDBGResult res = m_cudbgAPI->getLoadedFunctionInfo (dev, handle, info, numEntries);
#else
  CUDBGResult res = CUDBG_ERROR_NOT_SUPPORTED;
#endif
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("getLoadedFunctionInfo"), dev, res);
}

void cuda_api_get_loaded_function_info (uint32_t dev, uint64_t handle, CUDBGLoadedFunctionInfo *info, uint32_t numEntries)
{
  cuda_debugapi::instance ().get_loaded_function_info (dev, handle, info, numEntries);
}
#endif

void
cuda_debugapi::get_error_string_ex (char *buf, uint32_t bufSz, uint32_t *msgSz)
{
  if (get_state () != CUDA_API_STATE_INITIALIZED)
    return;

  if (cuda_backend_api_version.m_revision >= 134)
    {
      CUDBGResult res = m_cudbgAPI->getErrorStringEx (buf, bufSz, msgSz);
      if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_BUFFER_TOO_SMALL && res != CUDBG_ERROR_NOT_SUPPORTED)
        throw_error (GENERIC_ERROR, "Error: getErrorStringEx, error=%s.\n", cudbgGetErrorString(res));
    }
  else
    {
      buf[0] = 0;
      if (msgSz)
        *msgSz = 0;
    }

}

void
cuda_api_get_error_string_ex (char *buf, uint32_t bufSz, uint32_t *msgSz)
{
  cuda_debugapi::instance ().get_error_string_ex (buf, bufSz, msgSz);
}
