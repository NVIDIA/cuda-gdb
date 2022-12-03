/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2021 NVIDIA Corporation
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

#include "cuda-api.h"
#include "cuda-options.h"
#include "cuda-tdep.h"
#include "cuda-packet-manager.h"
#include "cuda-utils.h"

#include <signal.h>
#ifndef __ANDROID__
#include <execinfo.h>
#endif

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
  if (res != CUDBG_SUCCESS)
    cuda_api_trace ("API call received result: %s(0x%x)", cudbgGetErrorString((CUDBGResult) res), res);
}

static void cuda_api_error(CUDBGResult, const char *, ...) ATTRIBUTE_PRINTF (2, 3);

static void
cuda_api_error(CUDBGResult res, const char *fmt, ...)
{
  va_list args;
  char errStr[512];

  va_start (args, fmt);
  vsnprintf (errStr,sizeof(errStr), fmt, args);
  va_end (args);

  throw_error (GENERIC_ERROR, "Error: %s, error=%s(0x%x).\n",
               errStr, cudbgGetErrorString(res), res);
}

static void
cuda_devsmwp_api_error(const char *msg, uint32_t dev, uint32_t sm, uint32_t wp, CUDBGResult res)
{
  cuda_api_error (res, "Failed to %s (dev=%u, sm=%u, wp=%u)", msg, dev, sm, wp);
}

static void
cuda_devsmwpln_api_error(const char *msg, uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGResult res)
{
  cuda_api_error (res, "Failed to %s (dev=%u, sm=%u, wp=%u, ln=%u)", msg, dev, sm, wp, ln);
}

static void
cuda_dev_api_error(const char *msg, uint32_t dev, CUDBGResult res)
{
  cuda_api_error (res, "Failed to %s for CUDA device %u", msg, dev);
}

static CUDBGAPI cudbgAPI = NULL;

static bool api_initialized = false;

static cuda_attach_state_t attach_state = CUDA_ATTACH_STATE_NOT_STARTED;

void
cuda_api_set_api (CUDBGAPI api)
{
  cudbgAPI = api;
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
      cudbgAPI->finalize ();

      /* Kill inferior */
      kill (cuda_gdb_get_tid_or_pid (inferior_ptid), SIGKILL);
    }

  cuda_managed_memory_clean_regions ();

  /* Report error */
  throw_quit (_("fatal:  %s (error code = %s(0x%x)"), msg, cudbgGetErrorString(res), res);
}

int
cuda_api_initialize (void)
{
  CUDBGResult res;

  if (api_initialized)
    return 0;

  res = cudbgAPI->initialize ();
  cuda_api_print_api_call_result (res);
  cuda_api_handle_initialization_error (res);

  return (res != CUDBG_SUCCESS && res != CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED);
}

void
cuda_api_handle_initialization_error (CUDBGResult res)
{
  switch (res)
    {
    case CUDBG_SUCCESS:
      api_initialized = true;
      break;

    case CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED:
      warning (_("One or more CUDA devices are made unavailable to the application "
                 "because they are used for display and cannot be used while debugging. "
                 "This may change the application behavior."));
      api_initialized = true;
      break;

    case CUDBG_ERROR_UNINITIALIZED:
      /* Not ready yet. Will try later. */
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
cuda_api_finalize (void)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  cuda_api_clear_state ();
  if (cuda_remote)
    res = cuda_remote_api_finalize (cuda_get_current_remote_target ());
  else
    res = cudbgAPI->finalize ();
  cuda_api_handle_finalize_api_error (res);
}

void
cuda_api_clear_state (void)
{
  /* Mark the API as not initialized as early as possible. If the finalize()
   * call fails, we won't try to do anything stupid afterwards. */
  api_initialized = false;
  cuda_set_uvm_used (false);

  attach_state = CUDA_ATTACH_STATE_NOT_STARTED;
  cuda_managed_memory_clean_regions ();
}

void
cuda_api_handle_finalize_api_error (CUDBGResult res)
{
  if (!api_initialized)
    return;

  cuda_api_print_api_call_result (res);

  /* Only emit a warning in case of a failure, because cuda_api_finalize () can
     be called when an error occurs. That would create an infinite loop and/or
     undesired side effects. */
  if (res != CUDBG_SUCCESS)
    warning (_("Failed to finalize the CUDA debugger API (error=%u).\n"), res);
}

void
cuda_api_initialize_attach_stub (void)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  /* Mark the API as not initialized as early as possible. If the finalize()
   * call fails, we won't try to do anything stupid afterwards. */
  api_initialized = false;
  cuda_set_uvm_used (false);

  res = cudbgAPI->initializeAttachStub ();
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to initialize attach stub"));
}

void
cuda_api_resume_device (uint32_t dev)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->resumeDevice (dev);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_RUNNING_DEVICE)
    cuda_dev_api_error (_("resume device"), dev, res);
}

void
cuda_api_suspend_device (uint32_t dev)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->suspendDevice (dev);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_SUSPENDED_DEVICE)
    cuda_dev_api_error (_("suspend device"), dev, res);
}

/* Return true on success and false if resuming warps is not possible */
bool
cuda_api_resume_warps_until_pc (uint32_t dev, uint32_t sm, cuda_api_warpmask *warp_mask, uint64_t virt_pc)
{
  CUDBGResult res;

  if (!api_initialized)
    return false;

  res = cudbgAPI->resumeWarpsUntilPC (dev, sm, warp_mask->mask, virt_pc);
  cuda_api_print_api_call_result (res);

  if (res == CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE)
    return false;

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to resume warps (dev=%d, sm=%u, warp_mask=%s)"),
                    dev, sm, cuda_api_mask_string (warp_mask));
  return true;
}

bool
cuda_api_single_step_warp (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t nsteps, cuda_api_warpmask *warp_mask)
{
  CUDBGResult res;

  gdb_assert (warp_mask);

  if (!api_initialized)
    return false;

  res = cudbgAPI->singleStepWarp (dev, sm, wp, nsteps, &warp_mask->mask);
  cuda_api_print_api_call_result (res);

  if (res == CUDBG_ERROR_WARP_RESUME_NOT_POSSIBLE)
    return false;

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error ("single-step the warp", dev, sm, wp, res);

  return true;
}

bool
cuda_api_set_breakpoint (uint32_t dev, uint64_t addr)
{
  CUDBGResult res;

  if (!api_initialized)
    return true;

  res = cudbgAPI->setBreakpoint (dev, addr);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_INVALID_ADDRESS)
    cuda_api_error (res, _("Failed to set a breakpoint on device %u at address 0x%llx"),
                    dev, (unsigned long long)addr);
  return res != CUDBG_ERROR_INVALID_ADDRESS;
}

bool
cuda_api_unset_breakpoint (uint32_t dev, uint64_t addr)
{
  CUDBGResult res;

  if (!api_initialized)
    return true;

  res = cudbgAPI->unsetBreakpoint (dev, addr);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_INVALID_ADDRESS)
    cuda_api_error (res, _("Failed to unset a breakpoint on device %u at address 0x%llx"),
                    dev, (unsigned long long)addr);
  return res != CUDBG_ERROR_INVALID_ADDRESS;
}

void
cuda_api_read_grid_id (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *grid_id)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readGridId (dev, sm, wp, grid_id);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the grid index"), dev, sm, wp, res);
}

void
cuda_api_read_block_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readBlockIdx (dev, sm, wp, blockIdx);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the block index"), dev, sm, wp, res);
}

void
cuda_api_read_thread_idx (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readThreadIdx (dev, sm, wp, ln, threadIdx);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the thread index"), dev, sm, wp, res);
}

void
cuda_api_read_broken_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *brokenWarpsMask)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readBrokenWarps (dev, sm, &brokenWarpsMask->mask);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read the broken warps mask (dev=%u, sm=%u)"),
                    dev, sm);
  cuda_trace(_("Read broken warps: %" WARP_MASK_FORMAT), cuda_api_mask_string(brokenWarpsMask));
}

void
cuda_api_read_valid_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *valid_warps)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readValidWarps (dev, sm, &valid_warps->mask);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read the valid warps mask (dev=%u, sm=%u)"),
                    dev, sm);
  cuda_trace(_("device %d sm %d read valid warps: %" WARP_MASK_FORMAT),
	     dev, sm, cuda_api_mask_string(valid_warps));
}

void
cuda_api_read_valid_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *valid_lanes)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readValidLanes (dev, sm, wp, valid_lanes);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the valid lanes mask"), dev, sm, wp, res);
}

void
cuda_api_read_active_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *active_lanes)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readActiveLanes (dev, sm, wp, active_lanes);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the active lanes mask"), dev, sm, wp, res);
}

void
cuda_api_read_code_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readCodeMemory (dev, addr, buf, sz);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read code memory at address 0x%llx on device %u"),
                    (unsigned long long)addr, dev);
}

void
cuda_api_read_const_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readConstMemory (dev, addr, buf, sz);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read const memory at address 0x%llx on device %u"),
                    (unsigned long long)addr, dev);
}

void
cuda_api_read_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz)
{
  CUDBGResult res;
  uint64_t hostaddr;

  if (!api_initialized)
    return;

  res = cudbgAPI->readGenericMemory (dev, sm, wp, ln, addr, buf, sz);
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
      res = cudbgAPI->getHostAddrFromDeviceAddr (dev, addr, &hostaddr);
      cuda_api_print_api_call_result (res);
      if (res != CUDBG_SUCCESS)
        cuda_api_error (res, _("Failed to translate device VA to host VA"));
      read_memory (hostaddr, (gdb_byte *) buf, sz);
    }
}

bool
cuda_api_read_pinned_memory (uint64_t addr, void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return false;

  res = cudbgAPI->readPinnedMemory (addr, buf, sz);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_MEMORY_MAPPING_FAILED)
    cuda_api_error (res, _("Failed to read pinned memory at address 0x%llx"), (unsigned long long)addr);
  return res == CUDBG_SUCCESS;
}

void
cuda_api_read_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readParamMemory (dev, sm, wp, addr, buf, sz);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read param memory at address 0x%llx"
                    " on device %u sm %u warp %u"),
                    (unsigned long long)addr, dev, sm, wp);
}

void
cuda_api_read_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readSharedMemory (dev, sm, wp, addr, buf, sz);
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
cuda_api_read_texture_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t id, uint32_t dim, uint32_t *coords, void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readTextureMemory (dev, sm, wp, id, dim, coords, buf, sz);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read texture memory of texture %u dim %u coords %u"
                    " on device %u sm %u warp %u"), id, dim, *coords, dev, sm, wp);
}

void
cuda_api_read_texture_memory_bindless (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t tex_symtab_index, uint32_t dim, uint32_t *coords, void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readTextureMemoryBindless (dev, sm, wp, tex_symtab_index, dim, coords, buf, sz);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read texture memory of texture %u dim %u coords %u"
                    " on device %u sm %u warp %u"), tex_symtab_index, dim, *coords, dev, sm, wp);
}

bool
cuda_api_read_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return false;

  res = cudbgAPI->readLocalMemory (dev, sm, wp, ln, addr, buf, sz);
  cuda_api_print_api_call_result (res);

  if (res == CUDBG_ERROR_MISSING_DATA)
    {
      cuda_api_error (res, _("Local memory is not available in this corefile"));
    }

  if (res != CUDBG_SUCCESS)
    {
      cuda_api_error (res, _("Failed to read local memory at address 0x%llx"
		      " on device %u sm %u warp %u lane %u"),
		      (unsigned long long)addr, dev, sm, wp, ln);
      return false;
    }

  return true;
}

void
cuda_api_read_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
                        uint32_t regno, uint32_t *val)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readRegister (dev, sm, wp, ln, regno, val);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
      cuda_api_error (res, _("Failed to read register %d (dev=%u, sm=%u, wp=%u, ln=%u)"),
                      regno, dev, sm, wp, ln);
}

void
cuda_api_read_uregister (uint32_t dev, uint32_t sm, uint32_t wp,
			 uint32_t regno, uint32_t *val)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readUniformRegisterRange (dev, sm, wp, regno, sizeof(uint32_t), val);
  cuda_api_print_api_call_result (res);
  /* Not all devices support uniform registers */
  if (res == CUDBG_ERROR_INVALID_DEVICE || res == CUDBG_ERROR_MISSING_DATA)
    *val = 0;
  else if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read uniform register %d (dev=%u, sm=%u, wp=%u)"),
		    regno, dev, sm, wp);
}

void
cuda_api_read_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
                          uint32_t predicates_size, uint32_t *predicates)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readPredicates (dev, sm, wp, ln, predicates_size, predicates);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
      cuda_devsmwpln_api_error (_("read predicates"), dev, sm, wp, ln, res);
}

void
cuda_api_read_upredicates (uint32_t dev, uint32_t sm, uint32_t wp,
			   uint32_t predicates_size, uint32_t *predicates)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readUniformPredicates (dev, sm, wp, predicates_size, predicates);
  cuda_api_print_api_call_result (res);

  /* Not all devices support uniform registers */
  if (res == CUDBG_ERROR_INVALID_DEVICE || res == CUDBG_ERROR_MISSING_DATA)
    memset (predicates, 0, predicates_size);
  else if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read uniform predicates (dev=%u, sm=%u, wp=%u)"),
		    dev, sm, wp);
}

void
cuda_api_read_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
                           uint32_t *val)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readCCRegister (dev, sm, wp, ln, val);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
      cuda_devsmwpln_api_error (_("read CC register"), dev, sm, wp, ln, res);
}

void
cuda_api_read_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readPC (dev, sm, wp, ln, pc);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("read the program counter"), dev, res);
}

void
cuda_api_read_virtual_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readVirtualPC (dev, sm, wp, ln, pc);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("read the virtual PC"), dev, res);
}

void
cuda_api_read_lane_exception (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readLaneException (dev, sm, wp, ln, exception);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
      cuda_devsmwpln_api_error (_("read the lane exception"), dev, sm, wp, ln, res);
}


void
cuda_api_read_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth)
{
  CUDBGResult res;
  uint32_t api_call_depth;

  if (!api_initialized)
    return;

  res = cudbgAPI->readCallDepth (dev, sm, wp, ln, &api_call_depth);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwpln_api_error (_("read the call depth"), dev, sm, wp, ln, res);

  *depth = (int32_t) api_call_depth;
}

void
cuda_api_read_syscall_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth)
{
  CUDBGResult res;
  uint32_t api_syscall_call_depth;

  if (!api_initialized)
    return;

  res = cudbgAPI->readSyscallCallDepth (dev, sm, wp, ln, &api_syscall_call_depth);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
      cuda_devsmwpln_api_error (_("read the syscall call depth"), dev, sm, wp, ln, res);

  *depth = (int32_t) api_syscall_call_depth;
}

void
cuda_api_read_virtual_return_address (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
                                      int32_t level, uint64_t *ra)
{
  CUDBGResult res;
  uint32_t api_call_level;

  gdb_assert (level >= 0);

  api_call_level = (uint32_t) level;

  if (!api_initialized)
    return;

  res = cudbgAPI->readVirtualReturnAddress (dev, sm, wp, ln, api_call_level, ra);
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
}

void
cuda_api_read_error_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *pc, bool* valid)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readErrorPC (dev, sm, wp, pc, valid);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("Could not read error PC "), dev, sm, wp, res);
}

void
cuda_api_read_device_exception_state (uint32_t dev, uint64_t *exceptionSMMask, uint32_t n)
{
  CUDBGResult res;

  gdb_assert (exceptionSMMask);

  if (!api_initialized)
    return;

  res = cudbgAPI->readDeviceExceptionState (dev, exceptionSMMask, n);
  cuda_api_print_api_call_result(res);
  if (res != CUDBG_SUCCESS)
    {
      cuda_dev_api_error (_("read device exception state"),dev, res);
    }
}

void
cuda_api_write_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz)
{
  CUDBGResult res;
  uint64_t hostaddr;

  if (!api_initialized)
    return;

  res = cudbgAPI->writeGenericMemory (dev, sm, wp, ln, addr, buf, sz);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM)
    cuda_api_error (res, _("Failed to write generic memory at address 0x%llx"
                         " on device %u sm %u warp %u lane %u"),
                         (unsigned long long)addr, dev, sm, wp, ln);

  if (res == CUDBG_ERROR_ADDRESS_NOT_IN_DEVICE_MEM)
    {
      res = cudbgAPI->getHostAddrFromDeviceAddr (dev, addr, &hostaddr);
      cuda_api_print_api_call_result (res);
      if (res != CUDBG_SUCCESS)
        cuda_api_error (res, _("Failed to translate device VA to host VA"));
      write_memory (hostaddr, (const gdb_byte *) buf, sz);
    }
}

bool
cuda_api_write_pinned_memory (uint64_t addr, const void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return false;

  res = cudbgAPI->writePinnedMemory (addr, buf, sz);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_MEMORY_MAPPING_FAILED)
    cuda_api_error (res, _("Failed to write pinned memory at address 0x%llx"), (unsigned long long)addr);
  return res == CUDBG_SUCCESS;
}

void
cuda_api_write_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->writeParamMemory (dev, sm, wp, addr, buf, sz);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to write param memory at address 0x%llx"
                    " on device %u sm %u warp %u"),
                    (unsigned long long)addr, dev, sm, wp);
}

void
cuda_api_write_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->writeSharedMemory (dev, sm, wp, addr, buf, sz);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to write shared memory at address 0x%llx"
                    " on device %u sm %u warp %u"),
                    (unsigned long long)addr, dev, sm, wp);
}

bool
cuda_api_write_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return false;

  res = cudbgAPI->writeLocalMemory (dev, sm, wp, ln, addr, buf, sz);
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

void
cuda_api_write_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->writeRegister (dev, sm, wp, ln, regno, val);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
      cuda_api_error (res, _("Failed to write register %d (dev=%u, sm=%u, wp=%u, ln=%u)"),
                      regno, dev, sm, wp, ln);
}

void
cuda_api_write_uregister (uint32_t dev, uint32_t sm, uint32_t wp,
			  uint32_t regno, uint32_t val)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->writeUniformRegister (dev, sm, wp, regno, val);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
      cuda_api_error (res, _("Failed to write uniform register %d (dev=%u, sm=%u, wp=%u)"),
                      regno, dev, sm, wp);
}

void
cuda_api_write_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->writePredicates (dev, sm, wp, ln, predicates_size, predicates);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
      cuda_devsmwpln_api_error (_("write predicates"), dev, sm, wp, ln, res);
}

void
cuda_api_write_upredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, const uint32_t *predicates)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->writeUniformPredicates (dev, sm, wp, predicates_size, predicates);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
      cuda_api_error (res, _("Failed to write uniform predicates (dev=%u, sm=%u, wp=%u)"),
                      dev, sm, wp);
}

void
cuda_api_write_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->writeCCRegister (dev, sm, wp, ln, val);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
      cuda_devsmwpln_api_error (_("write CC register"), dev, sm, wp, ln, res);
}

void
cuda_api_get_grid_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *grid_dim)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getGridDim (dev, sm, wp, grid_dim);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the grid dimensions "), dev, sm, wp, res);
}

void
cuda_api_get_block_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *block_dim)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getBlockDim (dev, sm, wp, block_dim);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the block dimensions"), dev, sm, wp, res);
}

void
cuda_api_get_blocking (uint32_t dev, uint32_t sm, uint32_t wp, bool *blocking)
{
  CUDBGResult res;
  uint64_t blocking64;

  if (!api_initialized)
    return;

  res = cudbgAPI->getGridAttribute (dev, sm, wp, CUDBG_ATTR_GRID_LAUNCH_BLOCKING, &blocking64);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("read the grid blocking attribute"), dev, sm, wp, res);

  *blocking = !!blocking64;
}

void
cuda_api_get_tid (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getTID (dev, sm, wp, tid);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("get thread id"), dev, sm, wp, res);
}

void
cuda_api_get_elf_image (uint32_t dev,  uint64_t handle, bool relocated,
                        void *elfImage, uint64_t size)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getElfImageByHandle (dev, handle, relocated ? CUDBG_ELF_IMAGE_TYPE_RELOCATED : CUDBG_ELF_IMAGE_TYPE_NONRELOCATED, elfImage, size);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read the ELF image (dev=%u, handle=%llu, relocated=%d)"),
           dev, (unsigned long long)handle, relocated);
}

void
cuda_api_get_device_type (uint32_t dev, char *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getDeviceType (dev, buf, sz);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the device type"), dev, res);
}

void
cuda_api_get_sm_type (uint32_t dev, char *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getSmType (dev, buf, sz);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the SM type"), dev, res);
}

void
cuda_api_get_device_name (uint32_t dev, char *buf, uint32_t sz)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getDeviceName (dev, buf, sz);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the device name"), dev, res);
}

void
cuda_api_get_num_devices (uint32_t *numDev)
{
  CUDBGResult res;

  *numDev = 0;

  if (!api_initialized)
    return;

  res = cudbgAPI->getNumDevices (numDev);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get the number of devices"));
}

void
cuda_api_get_num_sms (uint32_t dev, uint32_t *numSMs)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getNumSMs (dev, numSMs);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get the number of SMs (dev=%u)"), dev);
}

void
cuda_api_get_num_warps (uint32_t dev, uint32_t *numWarps)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getNumWarps (dev, numWarps);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get the number of warps (dev=%u)"), dev);
}

void
cuda_api_get_num_lanes (uint32_t dev, uint32_t *numLanes)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getNumLanes (dev, numLanes);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the number of lanes"), dev, res);
}

void
cuda_api_get_num_registers (uint32_t dev, uint32_t *numRegs)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getNumRegisters (dev, numRegs);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the number of registers"), dev, res);
}

void
cuda_api_get_num_predicates (uint32_t dev, uint32_t *numPredicates)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getNumPredicates (dev, numPredicates);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the number of predicates"), dev, res);
}

void
cuda_api_get_num_uregisters (uint32_t dev, uint32_t *numRegs)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getNumUniformRegisters (dev, numRegs);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the number of uniform registers"), dev, res);
}

void
cuda_api_get_num_upredicates (uint32_t dev, uint32_t *numPredicates)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getNumUniformPredicates (dev, numPredicates);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get the number of uniform predicates"), dev, res);
}

void
cuda_api_is_device_code_address (uint64_t addr, bool *is_device_address)
{
  CUDBGResult res;

  if (!api_initialized)
    {
      *is_device_address = false;
      return;
    }

  res = cudbgAPI->isDeviceCodeAddress (addr, is_device_address);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to determine if address 0x%llx corresponds "
             "to the host or device"), (unsigned long long)addr);
}

void
cuda_api_set_notify_new_event_callback (CUDBGNotifyNewEventCallback callback)
{
  CUDBGResult res;

  /* Nothing should restrict the callback from being setup.
     In particular, it must be done prior to the API being
     fully initialized, which means there should not be a
     check here. */

  res = cudbgAPI->setNotifyNewEventCallback (callback);
  cuda_api_print_api_call_result (res);
  cuda_api_handle_set_callback_api_error (res);
}

void
cuda_api_handle_set_callback_api_error (CUDBGResult res)
{
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to set the new event callback"));
}

void
cuda_api_get_next_sync_event (CUDBGEvent *event)
{
  CUDBGResult res;

  event->kind = CUDBG_EVENT_INVALID;
  if (!api_initialized)
    return;

  res = cudbgAPI->getNextEvent (CUDBG_EVENT_QUEUE_TYPE_SYNC, event);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_NO_EVENT_AVAILABLE)
    cuda_api_error (res, _("Failed to get the next sync CUDA event"));
}

void
cuda_api_acknowledge_sync_events (void)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->acknowledgeSyncEvents ();

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to acknowledge a sync CUDA event"));
}

void
cuda_api_get_next_async_event (CUDBGEvent *event)
{
  CUDBGResult res;

  event->kind = CUDBG_EVENT_INVALID;
  if (!api_initialized)
    return;

  res = cudbgAPI->getNextEvent (CUDBG_EVENT_QUEUE_TYPE_ASYNC, event);
  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS && res != CUDBG_ERROR_NO_EVENT_AVAILABLE)
    cuda_api_error (res, _("Failed to get the next async CUDA event"));
}

void
cuda_api_disassemble (uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t bufSize)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->disassemble (dev, addr, instSize, buf, bufSize);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to disassemble instruction at address 0x%llx on CUDA device %u"),
                    (unsigned long long)addr, dev);
}

void
cuda_api_set_attach_state (cuda_attach_state_t state)
{
  CUDBGResult res;

  attach_state = state;

  if (state != CUDA_ATTACH_STATE_DETACHING)
    return;

  res = cudbgAPI->clearAttachState ();

  if (res != CUDBG_SUCCESS)
    warning (_("Failed to set attach state (error=%s(0x%x)).\n"), cudbgGetErrorString(res), res);
}

cuda_attach_state_t
cuda_api_get_attach_state (void)
{

  return attach_state;
}

void
cuda_api_request_cleanup_on_detach (uint32_t resumeAppFlag)
{
  CUDBGResult res = CUDBG_SUCCESS;

  res = cudbgAPI->requestCleanupOnDetach (resumeAppFlag);

  if (res != CUDBG_SUCCESS)
    warning (_("Failed to clear attach state (error=%s(0x%x)).\n"), cudbgGetErrorString(res), res);
}

void
cuda_api_memcheck_read_error_address (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
                                      uint64_t *address, ptxStorageKind *storage)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->memcheckReadErrorAddress (dev, sm, wp, ln, address, storage);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_devsmwpln_api_error (_("read error address"), dev, sm, wp, ln, res);
}

cuda_api_state_t
cuda_api_get_state (void)
{
    return api_initialized ? CUDA_API_STATE_INITIALIZED:
                             CUDA_API_STATE_UNINITIALIZED;
}

void
cuda_api_get_grid_status (uint32_t dev, uint64_t grid_id, CUDBGGridStatus *status)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getGridStatus (dev, grid_id, status);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get grid status (dev=%u, grid_id=%lld)"),
                    dev, (long long)grid_id);
}

void
cuda_api_get_grid_info (uint32_t dev, uint64_t grid_id, CUDBGGridInfo *info)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getGridInfo(dev, grid_id, info);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get grid info (dev=%u, grid_id=%lld"),
                    dev, (long long)grid_id);
}

bool
cuda_api_attach_or_detach_in_progress (void)
{
  return (attach_state == CUDA_ATTACH_STATE_DETACHING ||
          attach_state == CUDA_ATTACH_STATE_IN_PROGRESS);
}

void
cuda_api_get_adjusted_code_address (uint32_t dev, uint64_t addr, uint64_t *adjusted_addr, CUDBGAdjAddrAction adj_action)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->getAdjustedCodeAddress (dev, addr, adjusted_addr, adj_action);

  cuda_api_print_api_call_result (res);
  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to get adjusted code address (dev=%u, addr=0x%llx)"),
                    dev, (unsigned long long)addr);
}

void
cuda_api_set_kernel_launch_notification_mode(CUDBGKernelLaunchNotifyMode mode)
{
  if (!api_initialized)
    return;
  cudbgAPI->setKernelLaunchNotificationMode (mode);
}

void
cuda_api_get_device_pci_bus_info (uint32_t dev, uint32_t *pci_bus_id, uint32_t *pci_dev_id)
{
  CUDBGResult res;

  *pci_bus_id = 0xffff;
  *pci_dev_id = 0xffff;

  if (!api_initialized)
    return;

  res = cudbgAPI->getDevicePCIBusInfo (dev, pci_bus_id, pci_dev_id);
  if (res != CUDBG_SUCCESS)
    cuda_dev_api_error (_("get PCI bus information"), dev, res);
}

void
cuda_api_read_warp_state (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState *state)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readWarpState (dev, sm, wp, state);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_devsmwp_api_error (_("get warp state"), dev, sm, wp, res);
}

void
cuda_api_read_register_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t idx, uint32_t count, uint32_t *regs)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readRegisterRange (dev, sm, wp, ln, idx, count, regs);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read register range (dev=%u sm=%u wp=%u, ln=%u, idx=%u, count=%u)"),
           dev, sm, wp, ln, idx, count);
}

void
cuda_api_read_uregister_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t idx, uint32_t count, uint32_t *regs)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readUniformRegisterRange (dev, sm, wp, idx, count, regs);
  cuda_api_print_api_call_result (res);

  if (res == CUDBG_ERROR_INVALID_DEVICE || res == CUDBG_ERROR_MISSING_DATA)
    memset (regs, 0, (count - idx) * sizeof(uint32_t));
  else if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read uniform register range (dev=%u sm=%u wp=%u, idx=%u, count=%u)"),
           dev, sm, wp, idx, count);
}

void
cuda_api_read_global_memory (uint64_t addr, void *buf, uint32_t buf_size)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->readGlobalMemory (addr, buf, buf_size);
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
cuda_api_write_global_memory (uint64_t addr, const void *buf, uint32_t buf_size)
{
  CUDBGResult res;

  if (!api_initialized)
    return;

  res = cudbgAPI->writeGlobalMemory (addr, (void *)buf, buf_size);
  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to write %u bytes of global memory to 0x%llx"),
                    buf_size, (unsigned long long)addr);
}

void
cuda_api_get_managed_memory_region_info (uint64_t start_addr, CUDBGMemoryInfo *meminfo, uint32_t entries_count, uint32_t *entries_written)
{
  CUDBGResult res;

  if (entries_written)
    *entries_written = 0;
  if (!api_initialized || !cuda_is_uvm_used())
    return;

  res = cudbgAPI->getManagedMemoryRegionInfo (start_addr, meminfo, entries_count, entries_written);

  cuda_api_print_api_call_result (res);

  if (res != CUDBG_SUCCESS)
    cuda_api_error (res, _("Failed to read %u entries starting from addr 0x%llx"),
                    entries_count, (unsigned long long)start_addr);
}

