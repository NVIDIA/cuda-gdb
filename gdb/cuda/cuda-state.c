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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */
#include "defs.h"

#include "breakpoint.h"
#include "remote.h"

#include "cuda-asm.h"
#include "cuda-defs.h"
#include "cuda-iterator.h"
#include "cuda-state.h"

/* GPU register cache */
#define CUDBG_CACHED_REGISTERS_COUNT 256
#define CUDBG_CACHED_PREDICATES_COUNT 8

/* GPU uniform register cache */
#define CUDBG_CACHED_UREGISTERS_COUNT 64
#define CUDBG_CACHED_UPREDICATES_COUNT 8

typedef struct
{
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint32_t registers[CUDBG_CACHED_REGISTERS_COUNT];
  uint32_t register_valid_mask[CUDBG_CACHED_REGISTERS_COUNT >> 5];
  uint32_t predicates[CUDBG_CACHED_PREDICATES_COUNT];
  bool predicates_valid_p;
  uint32_t cc_register;
  bool cc_register_valid_p;
} cuda_reg_cache_element_t;

typedef struct
{
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t registers[CUDBG_CACHED_UREGISTERS_COUNT];
  uint32_t register_valid_mask[CUDBG_CACHED_UREGISTERS_COUNT >> 5];
  uint32_t predicates[CUDBG_CACHED_UPREDICATES_COUNT];
  bool predicates_valid_p;
} cuda_ureg_cache_element_t;

// Don't make these const, as we want to be able to change them through
// the debugger on-the-fly without breaking compiler assumptions.
const bool CACHED = true;
bool GLOBAL_CACHING_ENABLED = true;
bool DEVICE_CACHING_ENABLED = true;
bool SM_CACHING_ENABLED = true;
bool WARP_CACHING_ENABLED = true;
bool LANE_CACHING_ENABLED = true;

static std::vector<cuda_reg_cache_element_t> cuda_register_cache;
static std::vector<cuda_ureg_cache_element_t> cuda_uregister_cache;

static void
cuda_trace_state (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_STATE, fmt, ap);
  va_end (ap);
}

/******************************************************************************
 *
 *				    System
 *
 ******************************************************************************/

cuda_state cuda_state::m_instance;

cuda_state::cuda_state ()
    : m_num_devices_p (false), m_num_devices (0), m_suspended_devices_mask (0)
{
}

void
cuda_state::reset (void)
{
  m_instance.m_num_devices_p = false;
  m_instance.m_num_devices = 0;
  m_instance.m_suspended_devices_mask = 0;

  /* Automatically deletes the cuda_device objects */
  m_instance.m_devs.clear ();
}

void
cuda_state::initialize (void)
{
  cuda_trace_state ("system: initialize");
  gdb_assert (cuda_initialized);

  reset ();

  for (uint32_t dev_id = 0; dev_id < get_num_devices (); ++dev_id)
    {
      auto dev = new cuda_device (dev_id);
      m_instance.m_devs.emplace_back (dev);
    }

  cuda_options_force_set_launch_notification_update ();
}

void
cuda_state::finalize (void)
{
  cuda_trace_state ("system: finalize");
  gdb_assert (cuda_initialized);

  reset ();
}

uint32_t
cuda_state::get_num_devices (void)
{
  if (!cuda_initialized)
    return 0;

  if (m_instance.m_num_devices_p)
    return m_instance.m_num_devices;

  cuda_debugapi::get_num_devices (&m_instance.m_num_devices);
  gdb_assert (m_instance.m_num_devices <= CUDBG_MAX_DEVICES);
  m_instance.m_num_devices_p = GLOBAL_CACHING_ENABLED;

  return m_instance.m_num_devices;
}

void
cuda_state::invalidate_kernels ()
{
  cuda_trace_state (__FUNCTION__);

  for (auto kernel = kernels_get_first_kernel (); kernel;
       kernel = kernels_get_next_kernel (kernel))
    kernel_invalidate (kernel);
}

uint32_t
cuda_state::get_num_present_kernels (void)
{
  uint32_t num_present_kernel = 0;

  if (!cuda_initialized)
    return 0;

  for (auto kernel = kernels_get_first_kernel (); kernel;
       kernel = kernels_get_next_kernel (kernel))
    if (kernel_is_present (kernel))
      ++num_present_kernel;

  return num_present_kernel;
}

void
cuda_state::resolve_breakpoints (int bp_number_from)
{
  cuda_trace_state ("system: resolve breakpoints\n");

  elf_image_t elf_image;
  CUDA_ALL_LOADED_ELF_IMAGES (elf_image)
  cuda_resolve_breakpoints (bp_number_from, elf_image);
}

void
cuda_state::cleanup_breakpoints (void)
{
  cuda_trace_state ("system: clean up breakpoints");

  elf_image_t elf_image;
  CUDA_ALL_LOADED_ELF_IMAGES (elf_image)
  cuda_unresolve_breakpoints (elf_image);
}

void
cuda_state::cleanup_contexts (void)
{
  cuda_trace_state ("system: clean up contexts");

  for (uint32_t dev_id = 0; dev_id < get_num_devices (); ++dev_id)
    device (dev_id)->cleanup_contexts ();
}

context_t
cuda_state::find_context_by_addr (CORE_ADDR addr)
{
  for (uint32_t dev_id = 0; dev_id < get_num_devices (); ++dev_id)
    {
      context_t context = device (dev_id)->find_context_by_addr (addr);
      if (context)
	return context;
    }

  return nullptr;
}

context_t
cuda_state::find_context_by_id (uint64_t context_id)
{
  for (uint32_t dev_id = 0; dev_id < get_num_devices (); ++dev_id)
    {
      context_t context = device (dev_id)->find_context_by_id (context_id);
      if (context)
	return context;
    }

  return nullptr;
}

void
cuda_state::flush_disasm_caches ()
{
  cuda_trace_state ("flush device disassembly caches");

  /* This is less than ideal, we want to iternate on modules, not on kernels */
  for (auto kernel = kernels_get_first_kernel (); kernel;
       kernel = kernels_get_next_kernel (kernel))
    {
      module_t module = kernel_get_module (kernel);
      module->disassembler->flush_device_cache ();
    }
}

void
cuda_state::invalidate_caches ()
{
  cuda_trace_state ("cuda_state::%s", __FUNCTION__);

  for (auto &dev : m_instance.m_devs)
    dev->invalidate_caches ();
}

bool
cuda_state::is_broken (cuda_coords &coords)
{
  cuda_iterator<cuda_iterator_type::lanes, select_valid | select_trap
                                               | select_current_clock
                                               | select_sngl>
      lanes{ cuda_coords::wild () };
  if (!lanes.size ())
    return false;

  coords = *lanes.begin ();
  return true;
}

/******************************************************************************
 *
 *				    Device
 *
 ******************************************************************************/

cuda_device::cuda_device (uint32_t dev_id) : m_dev_id (dev_id)
{
  cuda_trace_state ("device %u: initialize", number ());

  // Clear the caches to begin with to make sure everything is in a
  // consistent state
  invalidate_caches ();

  uint32_t n_sms = get_num_sms ();

  m_sms.reserve (n_sms);
  for (uint32_t sm_idx = 0; sm_idx < n_sms; ++sm_idx)
    m_sms.emplace_back (new cuda_sm (this, sm_idx));

  m_contexts = contexts_new ();
}

cuda_device::~cuda_device () { cleanup_contexts (); }

uint32_t
cuda_device::get_num_sms ()
{
  if (m_num_sms_p)
    return m_num_sms;

  cuda_debugapi::get_num_sms (number (), &m_num_sms);
  gdb_assert (m_num_sms <= CUDBG_MAX_SMS);
  m_num_sms_p = DEVICE_CACHING_ENABLED;

  return m_num_sms;
}

uint32_t
cuda_device::get_num_warps ()
{
  if (m_num_warps_p)
    return m_num_warps;

  cuda_debugapi::get_num_warps (number (), &m_num_warps);
  gdb_assert (m_num_warps <= CUDBG_MAX_WARPS);
  m_num_warps_p = DEVICE_CACHING_ENABLED;

  return m_num_warps;
}

uint32_t
cuda_device::get_num_lanes ()
{
  if (m_num_lanes_p)
    return m_num_lanes;

  cuda_debugapi::get_num_lanes (number (), &m_num_lanes);
  gdb_assert (m_num_lanes <= CUDBG_MAX_LANES);
  m_num_lanes_p = DEVICE_CACHING_ENABLED;

  return m_num_lanes;
}

void
cuda_device::invalidate_caches (bool quietly)
{
  if (!quietly)
    cuda_trace_state ("cuda_device::%s(%d)", __FUNCTION__, number ());

  for (auto &sm : m_sms)
    sm->invalidate_caches (true, true);

  cuda_state::invalidate_kernels ();

  m_valid_p = false;

  m_insn_size_p = false;
  m_num_sms_p = false;
  m_num_warps_p = false;
  m_num_lanes_p = false;
  m_num_registers_p = false;
  m_num_predicates_p = false;
  m_num_uregisters_p = false;
  m_num_upredicates_p = false;
  m_pci_bus_info_p = false;
  m_dev_type_p = false;
  m_dev_name_p = false;
  m_sm_exception_mask_valid_p = false;
  m_sm_version_p = false;
  m_sm_type[0] = '\0';
}

void
cuda_device::cleanup_contexts ()
{
  cuda_trace_state ("device %u: clean up contexts", number ());

  if (m_contexts)
    {
      contexts_delete (m_contexts);
      m_contexts = nullptr;
    }
}

const char *
cuda_device::get_device_type ()
{
  if (m_dev_type_p)
    return m_dev_type;

  cuda_debugapi::get_device_type (number (), m_dev_type, sizeof (m_dev_type));
  m_dev_type_p = DEVICE_CACHING_ENABLED;

  return m_dev_type;
}

const char *
cuda_device::get_sm_type ()
{
  if (!strlen (m_sm_type))
    cuda_debugapi::get_sm_type (number (), m_sm_type, sizeof (m_sm_type));

  return m_sm_type;
}

uint32_t
cuda_device::get_sm_version ()
{
  if (m_sm_version_p)
    return m_sm_version;

  auto sm_type = get_sm_type ();
  if (strlen (sm_type) < 4 || strncmp (sm_type, "sm_", 3) != 0)
    error ("unknown sm_type %s", sm_type);

  m_sm_version = atoi (&sm_type[3]);
  m_sm_version_p = DEVICE_CACHING_ENABLED;

  return m_sm_version;
}

const char *
cuda_device::get_device_name ()
{
  if (m_dev_name_p)
    return m_dev_name;

  cuda_debugapi::get_device_name (number (), m_dev_name, sizeof (m_dev_name));
  m_dev_name_p = DEVICE_CACHING_ENABLED;

  return m_dev_name;
}

/* This assumes that the GPU architecture has a uniform instruction size,
 * which is true on all GPU architectures except FERMI. Since cuda-gdb no
 * longer supports FERMI as of 9.0 toolkit, this assumption is valid.
 */
uint32_t
cuda_device::get_insn_size ()
{
  if (m_insn_size_p)
    return m_insn_size;

  auto sm_version = get_sm_version ();
  m_insn_size = (sm_version < 70) ? 8 : 16;
  m_insn_size_p = DEVICE_CACHING_ENABLED;

  return m_insn_size;
}

uint32_t
cuda_device::get_pci_bus_id ()
{
  if (m_pci_bus_info_p)
    return m_pci_bus_id;

  cuda_debugapi::get_device_pci_bus_info (number (), &m_pci_bus_id,
					  &m_pci_dev_id);
  m_pci_bus_info_p = DEVICE_CACHING_ENABLED;

  return m_pci_bus_id;
}

uint32_t
cuda_device::get_pci_dev_id ()
{
  if (m_pci_bus_info_p)
    return m_pci_dev_id;

  cuda_debugapi::get_device_pci_bus_info (number (), &m_pci_bus_id,
					  &m_pci_dev_id);
  m_pci_bus_info_p = DEVICE_CACHING_ENABLED;

  return m_pci_dev_id;
}

uint32_t
cuda_device::get_num_registers ()
{
  if (m_num_registers_p)
    return m_num_registers;

  cuda_debugapi::get_num_registers (number (), &m_num_registers);
  m_num_registers_p = DEVICE_CACHING_ENABLED;

  return m_num_registers;
}

uint32_t
cuda_device::get_num_predicates ()
{
  if (m_num_predicates_p)
    return m_num_predicates;

  cuda_debugapi::get_num_predicates (number (), &m_num_predicates);
  gdb_assert (m_num_predicates <= CUDBG_CACHED_PREDICATES_COUNT);
  m_num_predicates_p = DEVICE_CACHING_ENABLED;

  return m_num_predicates;
}

uint32_t
cuda_device::get_num_uregisters ()
{
  if (m_num_uregisters_p)
    return m_num_uregisters;

  cuda_debugapi::get_num_uregisters (number (), &m_num_uregisters);
  m_num_uregisters_p = DEVICE_CACHING_ENABLED;

  return m_num_uregisters;
}

uint32_t
cuda_device::get_num_upredicates ()
{
  if (m_num_upredicates_p)
    return m_num_upredicates;

  cuda_debugapi::get_num_upredicates (number (), &m_num_upredicates);
  gdb_assert (m_num_upredicates <= CUDBG_CACHED_UPREDICATES_COUNT);
  m_num_upredicates_p = DEVICE_CACHING_ENABLED;

  return m_num_upredicates;
}

uint32_t
cuda_device::get_num_kernels ()
{
  kernel_t kernel;
  uint32_t num_kernels = 0;

  for (kernel = kernels_get_first_kernel (); kernel;
       kernel = kernels_get_next_kernel (kernel))
    if (kernel_get_dev_id (kernel) == number ())
      ++num_kernels;

  return num_kernels;
}

bool
cuda_device::is_any_context_present ()
{
  contexts_t contexts;

  contexts = get_contexts ();

  return contexts_is_any_context_present (contexts);
}

bool
cuda_device::is_valid ()
{
  if (!cuda_initialized)
    return false;

  if (m_valid_p)
    return m_valid;

  m_valid = false;

  if (!is_any_context_present ())
    return m_valid;

  for (auto sm_id = 0; sm_id < get_num_sms () && !m_valid; ++sm_id)
    for (auto wp_id = 0; wp_id < get_num_warps () && !m_valid; ++wp_id)
      if (sm (sm_id)->warp_is_valid (wp_id))
	m_valid = true;

  m_valid_p = DEVICE_CACHING_ENABLED;
  return m_valid;
}

bool
cuda_device::has_exception ()
{
  update_exception_state ();

  auto nelems = sizeof (m_sm_exception_mask) / sizeof (*m_sm_exception_mask);
  for (auto i = 0; i < nelems; ++i)
    if (m_sm_exception_mask[i] != 0)
      {
	cuda_trace_state ("cuda_device::has_exception(%u) true", number ());
	return true;
      }

  cuda_trace_state ("cuda_device::has_exception(%u) false", number ());
  return false;
}

uint64_t
cuda_device::sm_has_exception (uint32_t sm_id)
{
  update_exception_state ();

  bool exception = (m_sm_exception_mask[sm_id / 64] >> (sm_id % 64)) & 1ULL;
  cuda_trace_state ("cuda_device::sm_has_exception(%u) %s", sm_id,
		    exception ? "true" : "false");

  return exception;
}

void
cuda_device::get_active_sms_mask (std::bitset<CUDBG_MAX_SMS> &mask)
{
  mask.reset ();

  /* For every sm */
  for (auto sm_id = 0; sm_id < get_num_sms (); ++sm_id)
    {
      /* For every warp in the sm */
      for (auto wp_id = 0; wp_id < get_num_warps (); ++wp_id)
	{
	  /* Set to true if there is a valid warp in the sm */
	  if (sm (sm_id)->warp_is_valid (wp_id))
	    {
	      mask.set (sm_id);
	      break;
	    }
	}
    }
}

context_t
cuda_device::find_context_by_id (uint64_t context_id)
{
  auto contexts = get_contexts ();
  return contexts_find_context_by_id (contexts, context_id);
}

context_t
cuda_device::find_context_by_addr (CORE_ADDR addr)
{
  auto contexts = get_contexts ();
  return contexts_find_context_by_address (contexts, addr);
}

void
cuda_device::print ()
{
  cuda_trace_state ("device %u:", number ());

  contexts_print (get_contexts ());
}

void
cuda_device::resume ()
{
  cuda_trace_state ("cuda_device::resume(%d) (%s)", number (),
		    m_suspended ? "suspended" : "running");

  if (!m_suspended)
    return;

  invalidate_caches ();

  cuda_debugapi::resume_device (number ());

  m_suspended = false;

  cuda_state::clear_suspended_devices_mask (number ());
}

void
cuda_device::create_kernel (uint64_t grid_id)
{
  // Get the grid status and check it's validity before requesting
  // the grid info.
  CUDBGGridStatus grid_status = CUDBG_GRID_STATUS_INVALID;
  cuda_debugapi::get_grid_status (number (), grid_id, &grid_status);
  if (grid_status == CUDBG_GRID_STATUS_INVALID)
    warning ("Invalid grid status: gridId 0x%ld status %u", grid_id,
	     grid_status);
  gdb_assert (grid_status != CUDBG_GRID_STATUS_INVALID);

  CUDBGGridInfo gridInfo = { 0 };
  cuda_debugapi::get_grid_info (number (), grid_id, &gridInfo);

  kernels_start_kernel (
      number (), grid_id, gridInfo.functionEntry, gridInfo.context,
      gridInfo.module, gridInfo.gridDim, gridInfo.blockDim, gridInfo.type,
      gridInfo.parentGridId, gridInfo.origin, true, gridInfo.clusterDim);
}

void
cuda_device::suspend ()
{
  cuda_trace_state ("cuda_device::suspend(%d) (%s)", number (),
		    m_suspended ? "suspended" : "running");

  cuda_debugapi::suspend_device (number ());

  m_suspended = true;

  cuda_state::set_suspended_devices_mask (number ());
}

void
cuda_device::update_exception_state ()
{
  cuda_trace_state ("cuda_device::update_exception_state(%d) (mask valid %u)",
		    number (), m_sm_exception_mask_valid_p);

  if (m_sm_exception_mask_valid_p)
    return;

  memset (m_sm_exception_mask, 0, sizeof (m_sm_exception_mask));
  auto nsms = get_num_sms ();

  if (is_any_context_present ())
    cuda_debugapi::read_device_exception_state (number (), m_sm_exception_mask,
						(nsms + 63) / 64);
  else
    cuda_trace_state ("update_exception_state(%u) no context present",
		      number ());

  auto nelems = sizeof (m_sm_exception_mask) / sizeof (*m_sm_exception_mask);
  for (auto i = 0; i < nelems; ++i)
    cuda_trace_state ("cuda_device::has_exception(%u) 0x%016lx", number (),
		      m_sm_exception_mask[i]);

  for (auto sm_id = 0; sm_id < nsms; ++sm_id)
    if (!((m_sm_exception_mask[sm_id / 64] >> (sm_id % 64)) & 1))
      sm (sm_id)->set_exception_none ();

  m_sm_exception_mask_valid_p = DEVICE_CACHING_ENABLED;
}

void
cuda_device::set_device_spec (uint32_t num_sms, uint32_t num_warps,
			      uint32_t num_lanes, uint32_t num_registers,
			      uint32_t num_uregisters, const char *dev_type,
			      const char *sm_type)
{
  gdb_assert (cuda_remote);
  gdb_assert (num_sms <= CUDBG_MAX_SMS);
  gdb_assert (num_warps <= CUDBG_MAX_WARPS);
  gdb_assert (num_lanes <= CUDBG_MAX_LANES);

  m_num_sms = num_sms;
  m_num_warps = num_warps;
  m_num_lanes = num_lanes;
  m_num_registers = num_registers;
  m_num_uregisters = num_uregisters;

  strcpy (m_dev_type, dev_type);
  strcpy (m_sm_type, sm_type);

  if (strlen (m_sm_type) < 4 || strncmp (m_sm_type, "sm_", 3) != 0)
    error ("unknown sm_type %s", m_sm_type);
  m_sm_version = atoi (&m_sm_type[3]);
  m_insn_size = (m_sm_version < 70) ? 8 : 16;

  // If num_uregisters==0, it's probably not in the device spec
  // Query it directly here
  if (!m_num_uregisters)
    cuda_debugapi::get_num_uregisters (number (), &m_num_uregisters);

  m_num_sms_p = DEVICE_CACHING_ENABLED;
  m_num_warps_p = DEVICE_CACHING_ENABLED;
  m_num_lanes_p = DEVICE_CACHING_ENABLED;
  m_num_registers_p = DEVICE_CACHING_ENABLED;
  m_num_uregisters_p = DEVICE_CACHING_ENABLED;
  m_dev_type_p = DEVICE_CACHING_ENABLED;
  m_dev_name_p = DEVICE_CACHING_ENABLED;
  m_sm_version_p = DEVICE_CACHING_ENABLED;
  m_insn_size_p = DEVICE_CACHING_ENABLED;

  m_num_predicates_p = false;
  m_num_upredicates_p = false;
}

/******************************************************************************
 *
 *				      SM
 *
 ******************************************************************************/

cuda_sm::cuda_sm (cuda_device *dev, uint32_t sm_idx)
    : m_sm_idx (sm_idx), m_device (dev)
{
  m_warps.clear ();
  for (uint32_t idx = 0; idx < device ()->get_num_warps (); ++idx)
    m_warps.emplace_back (new cuda_warp (this, idx));
}

cuda_sm::~cuda_sm () {}

bool
cuda_sm::is_valid ()
{
  bool valid = cuda_api_has_bit (get_valid_warps_mask ());

  cuda_trace_state ("cuda_sm::%s(%u, %u): valid %d", __FUNCTION__,
		    device ()->number (), number (), valid);

  return valid;
}

void
cuda_sm::invalidate_caches (bool recurse, bool quietly)
{
  if (!quietly)
    cuda_trace_state ("cuda_sm::%s(%u, %u) recurse %u", __FUNCTION__,
		      device ()->number (), number (), recurse);

  if (recurse)
    for (auto &warp : m_warps)
      warp->invalidate_caches (true);

  device ()->clear_sm_exception_mask_valid_p ();

  invalidate_valid_warp_mask ();
  invalidate_broken_warp_mask ();

  if (!quietly)
    cuda_trace_state ("cuda_sm::%s(%u, %u) done", __FUNCTION__,
		      device ()->number (), number ());
}

cuda_api_warpmask *
cuda_sm::get_valid_warps_mask ()
{
  if (!m_valid_warps_mask_p)
    {
      cuda_debugapi::read_valid_warps (device ()->number (), number (),
				       &m_valid_warps_mask);
      m_valid_warps_mask_p = SM_CACHING_ENABLED;
    }

  cuda_trace_state ("cuda_sm::%s(%u, %u): 0x%016llx", __FUNCTION__,
		    device ()->number (), number (), m_valid_warps_mask.mask);

  return &m_valid_warps_mask;
}

bool
cuda_sm::warp_is_valid (uint32_t wp_id)
{
  gdb_assert (wp_id < device ()->get_num_warps ());

  return cuda_api_get_bit (get_valid_warps_mask (), wp_id) != 0;
}

cuda_api_warpmask *
cuda_sm::get_broken_warps_mask ()
{
  if (!m_broken_warps_mask_p)
    {
      cuda_debugapi::read_broken_warps (device ()->number (), number (),
					&m_broken_warps_mask);
      m_broken_warps_mask_p = SM_CACHING_ENABLED;
    }

  cuda_trace_state ("cuda_sm::%s(%u, %u): 0x%016llx", __FUNCTION__,
		    device ()->number (), number (), m_broken_warps_mask.mask);

  return &m_broken_warps_mask;
}

bool
cuda_sm::warp_is_broken (uint32_t wp_id)
{
  gdb_assert (wp_id < device ()->get_num_warps ());

  return cuda_api_get_bit (get_broken_warps_mask (), wp_id) != 0;
}

bool
cuda_sm::has_exception ()
{
  device ()->update_exception_state ();

  cuda_trace_state ("cuda_sm::%s(%u, %u): %d", __FUNCTION__,
		    device ()->number (), number (),
		    device ()->sm_has_exception (number ()));

  return device ()->sm_has_exception (number ());
}

void
cuda_sm::set_exception_none ()
{
  for (auto wp_id = 0; wp_id < device ()->get_num_warps (); ++wp_id)
    for (auto ln_id = 0; ln_id < device ()->get_num_lanes (); ++ln_id)
      warp (wp_id)->lane (ln_id)->set_exception_none ();
}

/******************************************************************************
 *
 *				     Warps
 *
 ******************************************************************************/

/* Warps register cache */
static std::vector<cuda_ureg_cache_element_t>::iterator
cuda_ureg_cache_find_element (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  for (auto it = cuda_uregister_cache.begin ();
       it != cuda_uregister_cache.end (); ++it)
    if (it->dev == dev_id && it->sm == sm_id && it->wp == wp_id)
      return it;

  /* Not found - create it! */
  cuda_ureg_cache_element_t new_elem = {};
  new_elem.dev = dev_id;
  new_elem.sm = sm_id;
  new_elem.wp = wp_id;
  cuda_uregister_cache.emplace_back (std::move (new_elem));

  /* Return the new element */
  return cuda_uregister_cache.end () - 1;
}

static void
cuda_ureg_cache_remove_element (uint32_t dev_id, uint32_t sm_id,
				uint32_t wp_id)
{
  auto it = cuda_uregister_cache.begin ();
  for (; it != cuda_uregister_cache.end (); ++it)
    if (it->dev == dev_id && it->sm == sm_id && it->wp == wp_id)
      break;

  if (it != cuda_uregister_cache.end ())
    cuda_uregister_cache.erase (it);
}

uint32_t
cuda_warp::get_uregister (uint32_t regno)
{
  auto dev_id = sm ()->device ()->number ();
  auto sm_id = sm ()->number ();
  auto wp_id = number ();

  auto elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);
  if ((elem->register_valid_mask[regno >> 5] & (1UL << (regno & 31))) != 0)
    return elem->registers[regno];

  cuda_debugapi::read_uregister_range (
      dev_id, sm_id, wp_id, 0, CUDBG_CACHED_UREGISTERS_COUNT, elem->registers);
  elem->register_valid_mask[0] = 0xffffffff;
  elem->register_valid_mask[1] = 0xffffffff;

  return elem->registers[regno];
}

void
cuda_warp::set_uregister (uint32_t regno, uint32_t value)
{
  auto dev_id = sm ()->device ()->number ();
  auto sm_id = sm ()->number ();
  auto wp_id = number ();

  cuda_debugapi::write_uregister (dev_id, sm_id, wp_id, regno, value);

  /* If register can not be cached - return */
  if (regno > CUDBG_CACHED_UREGISTERS_COUNT)
    return;

  auto elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);
  elem->registers[regno] = value;
  elem->register_valid_mask[regno >> 5] |= 1UL << (regno & 31);
}

bool
cuda_warp::get_upredicate (uint32_t predicate)
{
  auto dev_id = sm ()->device ()->number ();
  auto sm_id = sm ()->number ();
  auto wp_id = number ();

  auto elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);

  if (elem->predicates_valid_p)
    return elem->predicates[predicate] != 0;

  cuda_debugapi::read_upredicates (dev_id, sm_id, wp_id,
				   sm ()->device ()->get_num_upredicates (),
				   elem->predicates);
  elem->predicates_valid_p = WARP_CACHING_ENABLED;

  return elem->predicates[predicate] != 0;
}

void
cuda_warp::set_upredicate (uint32_t predicate, bool value)
{
  auto npredicates = sm ()->device ()->get_num_upredicates ();

  gdb_assert (predicate < npredicates);

  auto dev_id = sm ()->device ()->number ();
  auto sm_id = sm ()->number ();
  auto wp_id = number ();

  auto elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);

  if (!elem->predicates_valid_p)
    {
      cuda_debugapi::read_upredicates (dev_id, sm_id, wp_id, npredicates,
				       elem->predicates);
      elem->predicates_valid_p = WARP_CACHING_ENABLED;
    }

  elem->predicates[predicate] = value;

  cuda_debugapi::write_upredicates (dev_id, sm_id, wp_id, npredicates,
				    elem->predicates);
}

cuda_warp::cuda_warp (cuda_sm *parent_sm, uint32_t warp_idx)
    : m_warp_idx (warp_idx), m_sm (parent_sm), m_lanes ()
{
  const auto n_lanes = sm ()->device ()->get_num_lanes ();
  for (uint32_t idx = 0; idx < n_lanes; ++idx)
    lane (idx)->configure (this, idx);
}

cuda_warp::~cuda_warp () {}

void
cuda_warp::invalidate_caches (bool quietly)
{
  if (!quietly)
    cuda_trace_state ("cuda_warp::%s(%d, %d, %d)", __FUNCTION__,
		      sm ()->device ()->number (), sm ()->number (),
		      number ());

  cuda_ureg_cache_remove_element (sm ()->device ()->number (),
				  sm ()->number (), number ());

  for (auto &ln : m_lanes)
    ln.invalidate_caches (true);

  // Since the warp is invalidated, we have to invalidate the warp masks in
  // the corresponding SM
  sm ()->invalidate_valid_warp_mask ();
  sm ()->invalidate_broken_warp_mask ();

  m_state_p = false;
  m_valid_p = false;
  m_block_idx_p = false;
  m_cluster_idx_p = false;
  m_kernel_p = false;
  m_grid_id_p = false;
  m_valid_lanes_mask_p = false;
  m_active_lanes_mask_p = false;
  m_timestamp_p = false;
  m_error_pc_p = false;
}

bool
cuda_warp::has_error_pc ()
{
  if (!m_error_pc_p)
    {
      cuda_debugapi::read_error_pc (sm ()->device ()->number (),
				    sm ()->number (), number (), &m_error_pc,
				    &m_error_pc_available);
      m_error_pc_p = WARP_CACHING_ENABLED;
    }

  return m_error_pc_available;
}

void
cuda_warp::update_cached_info ()
{
  auto dev_id = sm ()->device ()->number ();
  auto sm_id = sm ()->number ();
  auto wp_id = number ();

  cuda_trace_state ("cuda_warp::%s(%d, %d, %d): m_state_p %d", __FUNCTION__,
		    dev_id, sm_id, wp_id, m_state_p);

  cuda_debugapi::read_warp_state (dev_id, sm_id, wp_id, &m_state);
  m_state_p = WARP_CACHING_ENABLED;

  m_error_pc = m_state.errorPC;
  m_error_pc_available = m_state.errorPCValid;
  m_error_pc_p = WARP_CACHING_ENABLED;

  m_block_idx = m_state.blockIdx;
  m_block_idx_p = WARP_CACHING_ENABLED;

  m_cluster_idx = m_state.clusterIdx;
  m_cluster_idx_p = WARP_CACHING_ENABLED;

  m_grid_id = m_state.gridId;
  m_grid_id_p = WARP_CACHING_ENABLED;

  m_active_lanes_mask = m_state.activeLanes;
  m_active_lanes_mask_p = WARP_CACHING_ENABLED;

  m_valid_lanes_mask = m_state.validLanes;
  m_valid_lanes_mask_p = WARP_CACHING_ENABLED;

  for (uint32_t ln_id = 0; ln_id < sm ()->device ()->get_num_lanes (); ln_id++)
    {
      if (!(m_state.validLanes & (1U << ln_id)))
	continue;

      auto ln = lane (ln_id);
      ln->set_thread_idx (&m_state.lane[ln_id].threadIdx);
      ln->set_virtual_pc (m_state.lane[ln_id].virtualPC);
      ln->set_exception (m_state.lane[ln_id].exception);

      if (m_state.lane[ln_id].exception != CUDBG_EXCEPTION_NONE)
	cuda_trace_state ("lane %d: exception %d", ln_id,
			  m_state.lane[ln_id].exception);

      if (!ln->is_timestamp_valid ())
	ln->set_timestamp (cuda_clock ());
    }

  if (!is_timestamp_valid ())
    set_timestamp (cuda_clock ());
}

bool
cuda_sm::resume_warps_until_pc (cuda_api_warpmask *mask, uint64_t pc)
{
  auto dev_id = device ()->number ();
  auto sm_id = number ();

  cuda_trace_state ("cuda_sm::resume_warps_until_pc(%d, %d, 0x%llx)", dev_id,
		    sm_id, pc);

  /* No point in resuming warps, if one of them is already there */
  for (auto wp_id = 0; wp_id < device ()->get_num_warps (); ++wp_id)
    if (cuda_api_get_bit (mask, wp_id))
      if (pc == warp (wp_id)->get_active_virtual_pc ())
	return false;

  /* If resume warps is not possible - abort */
  if (!cuda_debugapi::resume_warps_until_pc (dev_id, sm_id, mask, pc))
    return false;

  if (cuda_options_software_preemption ())
    {
      device ()->invalidate_caches ();
      return true;
    }

  /* invalidate the cache for the warps that have been single-stepped. */
  for (auto wp_id = 0; wp_id < device ()->get_num_warps (); ++wp_id)
    if (cuda_api_get_bit (mask, wp_id))
      warp (wp_id)->invalidate_caches ();

  /* must invalidate the SM since that's where the warp valid mask lives */
  invalidate_caches (false);

  return true;
}

bool
cuda_sm::single_step_warp (uint32_t wp_id, uint32_t nsteps,
			   cuda_api_warpmask *single_stepped_warp_mask)
{
  bool rc;
  cuda_api_warpmask tmp;

  auto sm_id = number ();
  auto dev_id = device ()->number ();

  // Save the warps that were valid before the step in
  // case they need to be stepped as well, and exit as a result.
  // This can happen when stepping over an EXIT instruction
  // Make a copy, as the pointer returned by cuda_sm::get_valid_warps_mask()
  // points at the m_valid_warps_mask member, which may be updated after
  // stepping/resume/etc.
  const auto before_step_valid_warp_mask = *get_valid_warps_mask ();

  cuda_trace_state ("cuda_sm::single_step_warp(%d, %d, %d) nsteps %u valid "
		    "%" WARP_MASK_FORMAT,
		    dev_id, sm_id, wp_id, nsteps,
		    cuda_api_mask_string (&before_step_valid_warp_mask));

  gdb_assert (wp_id < device ()->get_num_warps ());

  cuda_api_clear_mask (single_stepped_warp_mask);
  cuda_api_clear_mask (&tmp);
  cuda_api_set_bit (&tmp, wp_id, 1);
  cuda_api_not_mask (
      &tmp, &tmp); // Select all but the single-stepped warp in the mask

  rc = cuda_debugapi::single_step_warp (dev_id, sm_id, wp_id, nsteps,
					single_stepped_warp_mask);
  if (!rc)
    return rc;

  if (cuda_options_software_preemption ())
    {
      device ()->invalidate_caches ();
      return true;
    }

  cuda_api_and_mask (&tmp, &tmp, single_stepped_warp_mask);

  if (cuda_api_has_bit (&tmp))
    {
      warning ("Warp(s) other than the current warp had to be "
	       "single-stepped:%" WARP_MASK_FORMAT,
	       cuda_api_mask_string (single_stepped_warp_mask));

      // More warps may have stepped, invalidate all caches for this device
      device ()->invalidate_caches ();
    }
  else
    {
      // Must invalidate the SM since that's where the
      // warp valid and broken masks live.
      // No need to recurse, handled that above.
      invalidate_caches (true);
    }

  // New valid warps mask --- must be after invalidate_caches() calls above
  // Make a copy, as the pointer returned by cuda_sm::get_valid_warps_mask()
  // points at the m_valid_warps_mask member, which may be updated after
  // stepping/resume/etc.
  const auto after_step_valid_warp_mask = *get_valid_warps_mask ();

  // Warps to invalidate
  cuda_api_warpmask invalidate_warp_mask;

  // Invalidate all the warps that just stepped
  cuda_api_cp_mask (&invalidate_warp_mask, single_stepped_warp_mask);

  // Invalidate the warp we were trying to step (sometimes the
  // stepped_warp_mask comes back with the wp_id bit 0 if the thread exited.
  // We want to invalidate it anyways.
  cuda_api_set_bit (&invalidate_warp_mask, wp_id, 1);

  // Invalidate any warps that exited (set in before but not after)
  cuda_api_warpmask exited_warp_mask;
  cuda_api_and_mask (&exited_warp_mask, &before_step_valid_warp_mask,
		     cuda_api_not_mask (&tmp, &after_step_valid_warp_mask));

  cuda_api_or_mask (&invalidate_warp_mask, &invalidate_warp_mask,
		    &exited_warp_mask);

  // Invalidate the cache for the warps that have been single-stepped.
  cuda_trace_state (
      "cuda_sm::%s(%d, %d, %d) invalidate_warps_mask %" WARP_MASK_FORMAT,
      __FUNCTION__, dev_id, sm_id, wp_id,
      cuda_api_mask_string (&invalidate_warp_mask));

  for (auto i = 0; i < device ()->get_num_warps (); ++i)
    if (cuda_api_get_bit (&invalidate_warp_mask, i))
      warp (i)->invalidate_caches ();

  return true;
}

uint64_t
cuda_warp::get_grid_id ()
{
  auto sm_id = sm ()->number ();
  auto dev_id = sm ()->device ()->number ();

  if (cuda_remote && !m_grid_id_p)
    cuda_remote_update_grid_id_in_sm (cuda_get_current_remote_target (),
				      dev_id, sm_id);

  if (m_grid_id_p)
    return m_grid_id;

  update_cached_info ();

  return m_grid_id;
}

kernel_t
cuda_warp::get_kernel ()
{
  if (m_kernel_p)
    return m_kernel;

  auto dev_id = sm ()->device ()->number ();
  auto grid_id = get_grid_id ();
  auto kernel = kernels_find_kernel_by_grid_id (dev_id, grid_id);

  if (!kernel)
    {
      sm ()->device ()->create_kernel (grid_id);
      kernel = kernels_find_kernel_by_grid_id (dev_id, grid_id);
    }

  m_kernel = kernel;
  m_kernel_p = WARP_CACHING_ENABLED;

  return m_kernel;
}

CuDim3
cuda_warp::get_block_idx ()
{
  if (cuda_remote && !m_block_idx_p && sm ()->warp_is_valid (number ()))
    cuda_remote_update_block_idx_in_sm (cuda_get_current_remote_target (),
					sm ()->device ()->number (),
					sm ()->number ());

  if (m_block_idx_p)
    return m_block_idx;

  update_cached_info ();

  return m_block_idx;
}

CuDim3
cuda_warp::get_cluster_idx ()
{
  if (cuda_remote && !m_cluster_idx_p && sm ()->warp_is_valid (number ()))
    cuda_remote_update_cluster_idx_in_sm (cuda_get_current_remote_target (),
					  sm ()->device ()->number (),
					  sm ()->number ());
  if (m_cluster_idx_p)
    return m_cluster_idx;

  update_cached_info ();

  return m_cluster_idx;
}

uint32_t
cuda_warp::get_valid_lanes_mask ()
{
  if (m_valid_lanes_mask_p)
    return m_valid_lanes_mask;

  if (sm ()->warp_is_valid (number ()))
    {
      update_cached_info ();
      return m_valid_lanes_mask;
    }

  m_valid_lanes_mask = 0;
  m_valid_lanes_mask_p = WARP_CACHING_ENABLED;

  if (!is_timestamp_valid ())
    set_timestamp (cuda_clock ());

  return 0;
}

uint32_t
cuda_warp::get_active_lanes_mask ()
{
  if (m_active_lanes_mask_p)
    return m_active_lanes_mask;

  update_cached_info ();

  return m_active_lanes_mask;
}

uint32_t
cuda_warp::get_divergent_lanes_mask ()
{
  uint32_t valid_lanes_mask = get_valid_lanes_mask ();
  uint32_t active_lanes_mask = get_active_lanes_mask ();
  uint32_t divergent_lanes_mask = valid_lanes_mask & ~active_lanes_mask;

  return divergent_lanes_mask;
}

uint32_t
cuda_warp::get_lowest_active_lane ()
{
  uint32_t active_lanes_mask;
  uint32_t ln_id;

  active_lanes_mask = get_active_lanes_mask ();

  for (ln_id = 0; ln_id < sm ()->device ()->get_num_lanes (); ++ln_id)
    if ((active_lanes_mask >> ln_id) & 1)
      break;

  return ln_id;
}

uint64_t
cuda_warp::get_active_pc ()
{
  uint32_t ln_id = get_lowest_active_lane ();
  uint64_t pc = lane (ln_id)->get_pc ();

  return pc;
}

uint64_t
cuda_warp::get_active_virtual_pc ()
{
  auto ln_id = get_lowest_active_lane ();
  auto pc = lane (ln_id)->get_virtual_pc ();

  return pc;
}

uint64_t
cuda_warp::get_error_pc ()
{
  bool error_pc_available = false;
  uint64_t error_pc = 0ULL;

  /*if (wp->m_error_pc_p)
    {
      gdb_assert (wp->m_error_pc_available);
      return wp->m_error_pc;
    }
*/
  auto dev_id = sm ()->device ()->number ();
  auto sm_id = sm ()->number ();
  auto wp_id = number ();

  cuda_debugapi::read_error_pc (dev_id, sm_id, wp_id, &error_pc,
				&error_pc_available);

  m_error_pc = error_pc;
  m_error_pc_available = error_pc_available;
  m_error_pc_p = WARP_CACHING_ENABLED;

  gdb_assert (m_error_pc_available);
  return m_error_pc;
}

void
cuda_warp::set_grid_id (uint64_t grid_id)
{
  gdb_assert (cuda_remote);

  m_grid_id = grid_id;
  m_grid_id_p = true;
}

void
cuda_warp::set_cluster_idx (const CuDim3 *cluster_idx)
{
  gdb_assert (cuda_remote);

  m_cluster_idx = *cluster_idx;
  m_cluster_idx_p = true;
}

void
cuda_warp::set_block_idx (const CuDim3 *block_idx)
{
  gdb_assert (cuda_remote);

  m_block_idx = *block_idx;
  m_block_idx_p = true;
}

/* Lanes register cache */
static std::vector<cuda_reg_cache_element_t>::iterator
cuda_reg_cache_find_element (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,
			     uint32_t ln_id)
{
  for (auto it = cuda_register_cache.begin ();
       it != cuda_register_cache.end (); ++it)
    if (it->dev == dev_id && it->sm == sm_id && it->wp == wp_id
	&& it->ln == ln_id)
      return it;

  /* Not found - create it! */
  cuda_reg_cache_element_t new_elem = {};
  new_elem.dev = dev_id;
  new_elem.sm = sm_id;
  new_elem.wp = wp_id;
  new_elem.ln = ln_id;
  cuda_register_cache.emplace_back (std::move (new_elem));

  /* Return the new element */
  return cuda_register_cache.end () - 1;
}

static void
cuda_reg_cache_remove_element (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,
			       uint32_t ln_id)
{
  auto it = cuda_register_cache.begin ();
  for (; it != cuda_register_cache.end (); ++it)
    if (it->dev == dev_id && it->sm == sm_id && it->wp == wp_id
	&& it->ln == ln_id)
      break;

  if (it != cuda_register_cache.end ())
    cuda_register_cache.erase (it);
}

bool
cuda_warp::lane_is_valid (uint32_t ln_id)
{
  bool valid = (get_valid_lanes_mask () >> ln_id) & 1;

  auto ln = lane (ln_id);
  if (!ln->is_timestamp_valid ())
    ln->set_timestamp (cuda_clock ());

  return valid;
}

bool
cuda_warp::lane_is_active (uint32_t ln_id)
{
  auto mask = get_active_lanes_mask ();
  return (mask >> ln_id) & 1;
}

bool
cuda_warp::lane_is_divergent (uint32_t ln_id)
{
  auto mask = get_divergent_lanes_mask ();
  return (mask >> ln_id) & 1;
}

/******************************************************************************
 *
 *				     Lanes
 *
 ******************************************************************************/

cuda_lane::cuda_lane () {}

cuda_lane::cuda_lane (cuda_warp *warp, uint32_t lane_idx)
    : m_lane_idx (lane_idx), m_warp (warp)
{
}

cuda_lane::~cuda_lane () {}

void
cuda_lane::invalidate_caches (bool quietly)
{
  m_timestamp_p = false;
  m_pc_p = false;
  m_virtual_pc_p = false;
  m_thread_idx_p = false;
  m_call_depth_p = false;
  m_syscall_call_depth_p = false;
  m_exception_p = false;
  m_virtual_return_address.clear ();

  cuda_reg_cache_remove_element (warp ()->sm ()->device ()->number (),
				 warp ()->sm ()->number (), warp ()->number (),
				 m_lane_idx);
}

bool
cuda_lane::is_active ()
{
  return warp ()->lane_is_active (number ());
}

bool
cuda_lane::is_valid ()
{
  return warp ()->lane_is_valid (number ());
}

bool
cuda_lane::is_divergent ()
{
  return warp ()->lane_is_divergent (number ());
}

CuDim3
cuda_lane::get_thread_idx ()
{
  /* In a remote session, we fetch the threadIdx of all valid thread in the
   * warp using one rsp packet to reduce the amount of communication. */
  if (cuda_remote && !(m_thread_idx_p) && warp ()->lane_is_valid (number ()))
    cuda_remote_update_thread_idx_in_warp (
	cuda_get_current_remote_target (),
	warp ()->sm ()->device ()->number (), warp ()->sm ()->number (),
	warp ()->number ());

  if (m_thread_idx_p)
    return m_thread_idx;

  warp ()->update_cached_info ();

  return m_thread_idx;
}

uint64_t
cuda_lane::get_virtual_pc ()
{
  if (m_virtual_pc_p)
    return m_virtual_pc;

  warp ()->update_cached_info ();

  return m_virtual_pc;
}

uint64_t
cuda_lane::get_pc ()
{
  if (m_pc_p)
    return m_pc;

  auto dev_id = warp ()->sm ()->device ()->number ();
  auto sm_id = warp ()->sm ()->number ();
  auto wp_id = warp ()->number ();
  cuda_debugapi::read_pc (dev_id, sm_id, wp_id, number (), &m_pc);

  m_pc_p = LANE_CACHING_ENABLED;

  /* Optimization: all the active lanes share the same virtual PC */
  if (is_active ())
    {
      auto num_lanes = warp ()->sm ()->device ()->get_num_lanes ();
      for (auto other_ln_id = 0; other_ln_id < num_lanes; ++other_ln_id)
	{
	  auto lane = warp ()->lane (other_ln_id);
	  if (warp ()->lane_is_valid (other_ln_id)
	      && warp ()->lane_is_active (other_ln_id))
	    lane->set_pc (m_pc, LANE_CACHING_ENABLED);
	}
    }
  return m_pc;
}

CUDBGException_t
cuda_lane::get_exception ()
{
  gdb_assert (warp ()->lane_is_valid (number ()));

  if (m_exception_p)
    return m_exception;

  warp ()->update_cached_info ();

  return m_exception;
}

uint32_t
cuda_lane::get_register (uint32_t regno)
{
  gdb_assert (warp ()->lane_is_valid (number ()));

  /* If register can not be cached - read it directly */
  auto dev_id = warp ()->sm ()->device ()->number ();
  auto sm_id = warp ()->sm ()->number ();
  auto wp_id = warp ()->number ();
  auto ln_id = number ();

  if (regno > CUDBG_CACHED_REGISTERS_COUNT)
    {
      uint32_t value;
      cuda_debugapi::read_register (dev_id, sm_id, wp_id, ln_id, regno,
				    &value);
      return value;
    }

  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);
  if ((elem->register_valid_mask[regno >> 5] & (1UL << (regno & 31))) != 0)
    return elem->registers[regno];

  if (regno < 32)
    {
      cuda_debugapi::read_register_range (dev_id, sm_id, wp_id, ln_id,
					  regno & ~31, 32,
					  &elem->registers[regno & ~31]);
      elem->register_valid_mask[regno >> 5] |= 0xffffffff;
    }
  else
    {
      cuda_debugapi::read_register (dev_id, sm_id, wp_id, ln_id, regno,
				    &elem->registers[regno]);
      elem->register_valid_mask[regno >> 5] |= 1UL << (regno & 31);
    }

  return elem->registers[regno];
}

void
cuda_lane::set_register (uint32_t regno, uint32_t value)
{
  gdb_assert (warp ()->lane_is_valid (number ()));

  auto dev_id = warp ()->sm ()->device ()->number ();
  auto sm_id = warp ()->sm ()->number ();
  auto wp_id = warp ()->number ();
  auto ln_id = number ();

  cuda_debugapi::write_register (dev_id, sm_id, wp_id, ln_id, regno, value);
  /* If register can not be cached - read it directly */
  if (regno > CUDBG_CACHED_REGISTERS_COUNT)
    return;

  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);
  elem->registers[regno] = value;
  elem->register_valid_mask[regno >> 5] |= 1UL << (regno & 31);
}

bool
cuda_lane::get_predicate (uint32_t predicate)
{
  gdb_assert (predicate < warp ()->sm ()->device ()->get_num_predicates ());
  gdb_assert (warp ()->lane_is_valid (number ()));

  auto dev_id = warp ()->sm ()->device ()->number ();
  auto sm_id = warp ()->sm ()->number ();
  auto wp_id = warp ()->number ();
  auto ln_id = number ();
  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);

  if (elem->predicates_valid_p)
    return elem->predicates[predicate] != 0;

  cuda_debugapi::read_predicates (
      dev_id, sm_id, wp_id, ln_id,
      cuda_state::device (dev_id)->get_num_predicates (), elem->predicates);
  elem->predicates_valid_p = LANE_CACHING_ENABLED;

  return elem->predicates[predicate] != 0;
}

void
cuda_lane::set_predicate (uint32_t predicate, bool value)
{
  gdb_assert (predicate < warp ()->sm ()->device ()->get_num_predicates ());
  gdb_assert (warp ()->lane_is_valid (number ()));

  auto dev_id = warp ()->sm ()->device ()->number ();
  auto sm_id = warp ()->sm ()->number ();
  auto wp_id = warp ()->number ();
  auto ln_id = number ();

  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);
  if (!elem->predicates_valid_p)
    {
      cuda_debugapi::read_predicates (
	  dev_id, sm_id, wp_id, ln_id,
	  cuda_state::device_get_num_predicates (dev_id), elem->predicates);
      elem->predicates_valid_p = LANE_CACHING_ENABLED;
    }

  elem->predicates[predicate] = value;

  cuda_debugapi::write_predicates (
      dev_id, sm_id, wp_id, ln_id,
      cuda_state::device_get_num_predicates (dev_id), elem->predicates);
}

uint32_t
cuda_lane::get_cc_register ()
{
  gdb_assert (warp ()->lane_is_valid (number ()));

  auto dev_id = warp ()->sm ()->device ()->number ();
  auto sm_id = warp ()->sm ()->number ();
  auto wp_id = warp ()->number ();
  auto ln_id = number ();

  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);

  if (elem->cc_register_valid_p)
    return elem->cc_register;

  cuda_debugapi::read_cc_register (dev_id, sm_id, wp_id, ln_id,
				   &elem->cc_register);
  elem->cc_register_valid_p = LANE_CACHING_ENABLED;

  return elem->cc_register;
}

void
cuda_lane::set_cc_register (uint32_t value)
{
  gdb_assert (warp ()->lane_is_valid (number ()));

  auto dev_id = warp ()->sm ()->device ()->number ();
  auto sm_id = warp ()->sm ()->number ();
  auto wp_id = warp ()->number ();
  auto ln_id = number ();

  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);

  elem->cc_register = value;
  elem->cc_register_valid_p = LANE_CACHING_ENABLED;

  cuda_debugapi::write_cc_register (dev_id, sm_id, wp_id, ln_id,
				    elem->cc_register);
}

int32_t
cuda_lane::get_call_depth ()
{
  gdb_assert (warp ()->lane_is_valid (number ()));

  if (m_call_depth_p)
    return (int32_t)m_call_depth;

  auto dev_id = warp ()->sm ()->device ()->number ();
  auto sm_id = warp ()->sm ()->number ();
  auto wp_id = warp ()->number ();
  auto ln_id = number ();

  cuda_debugapi::read_call_depth (dev_id, sm_id, wp_id, ln_id, &m_call_depth);
  m_call_depth_p = LANE_CACHING_ENABLED;

  return (int32_t)m_call_depth;
}

int32_t
cuda_lane::get_syscall_call_depth ()
{
  gdb_assert (warp ()->lane_is_valid (number ()));

  if (m_syscall_call_depth_p)
    return m_syscall_call_depth;

  auto dev_id = warp ()->sm ()->device ()->number ();
  auto sm_id = warp ()->sm ()->number ();
  auto wp_id = warp ()->number ();
  auto ln_id = number ();

  cuda_debugapi::read_syscall_call_depth (dev_id, sm_id, wp_id, ln_id,
					  &m_syscall_call_depth);
  m_syscall_call_depth_p = LANE_CACHING_ENABLED;

  return (int32_t)m_syscall_call_depth;
}

uint64_t
cuda_lane::get_virtual_return_address (int32_t level)
{
  if (LANE_CACHING_ENABLED)
    {
      auto iter = m_virtual_return_address.find (level);
      if (iter != m_virtual_return_address.end ())
	return iter->second;
    }

  uint64_t ra = 0;
  auto dev_id = warp ()->sm ()->device ()->number ();
  auto sm_id = warp ()->sm ()->number ();
  auto wp_id = warp ()->number ();
  auto ln_id = number ();
  cuda_debugapi::read_virtual_return_address (dev_id, sm_id, wp_id, ln_id,
					      level, &ra);

  if (LANE_CACHING_ENABLED)
    m_virtual_return_address[level] = ra;

  return ra;
}
