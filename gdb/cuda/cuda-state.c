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
#include "cuda-coord-set.h"
#include "cuda-defs.h"
#include "cuda-modules.h"
#include "cuda-state.h"

// GPU register cache
#define CUDA_STATE_REGISTER_RZ 255U
#define CUDA_STATE_REGISTER_RANGE_SIZE 16

// FIXME: used by other modules, remove
const bool CACHED = true;

static uint32_t unknown_device_attribute_warning;
static uint32_t unknown_sm_attribute_warning;
static uint32_t unknown_warp_attribute_warning;
static uint32_t unknown_lane_attribute_warning;

// Debugging flags, mostly designed to be set by GDB when debugging
// cuda-gdb. These are 0 by default, and can be set non-zero for
// increased levels of logging.
static uint32_t debug_skips = 0;
static uint32_t debug_invalidate = 0;

// General tracing
//
// Use macros for these so that the arguments are not evaluated unless
// 'domain' tracing is enabled.
//
#define CUDA_STATE_TRACE(fmt, ...)                                            \
  do                                                                          \
    {                                                                         \
      if (cuda_options_trace_domain_enabled (CUDA_TRACE_STATE))               \
	cuda_trace_domain (CUDA_TRACE_STATE, "%s(): " fmt, __FUNCTION__,      \
			   ##__VA_ARGS__);                                    \
    }                                                                         \
  while (0)

// Per-device/sm/warp/lane logging with an explicit trace domain
// General per-device/sm/warp/lane logging

#define CUDA_STATE_TRACE_DOMAIN_DEV(domain, dev, fmt, ...)                    \
  do                                                                          \
    {                                                                         \
      if (cuda_options_trace_domain_enabled (domain))                         \
	cuda_trace_domain (domain, "%s(%u): " fmt, __FUNCTION__,              \
			   (dev)->dev_idx (), ##__VA_ARGS__);                 \
    }                                                                         \
  while (0)

#define CUDA_STATE_TRACE_DOMAIN_SM(domain, sm, fmt, ...)                      \
  do                                                                          \
    {                                                                         \
      if (cuda_options_trace_domain_enabled (domain))                         \
	cuda_trace_domain (domain, "%s(%u, %u): " fmt, __FUNCTION__,          \
			   (sm)->dev_idx (), (sm)->sm_idx (), ##__VA_ARGS__); \
    }                                                                         \
  while (0)

#define CUDA_STATE_TRACE_DOMAIN_WARP(domain, wp, fmt, ...)                    \
  do                                                                          \
    {                                                                         \
      if (cuda_options_trace_domain_enabled (domain))                         \
	cuda_trace_domain (domain, "%s(%u, %u, %u): " fmt, __FUNCTION__,      \
			   (wp)->dev_idx (), (wp)->sm_idx (),                 \
			   (wp)->warp_idx (), ##__VA_ARGS__);                 \
    }                                                                         \
  while (0)

#define CUDA_STATE_TRACE_DOMAIN_LANE(domain, ln, fmt, ...)                    \
  do                                                                          \
    {                                                                         \
      if (cuda_options_trace_domain_enabled (domain))                         \
	cuda_trace_domain (domain, "%s(%u, %u, %u, %u): " fmt, __FUNCTION__,  \
			   (ln)->dev_idx (), (ln)->sm_idx (),                 \
			   (ln)->warp_idx (), (ln)->lane_idx (),              \
			   ##__VA_ARGS__);                                    \
    }                                                                         \
  while (0)

// General per-device/sm/warp/lane logging

#define CUDA_STATE_TRACE_DEV(dev, fmt, ...)                                   \
  CUDA_STATE_TRACE_DOMAIN_DEV (CUDA_TRACE_STATE, dev, fmt, ##__VA_ARGS__)

#define CUDA_STATE_TRACE_SM(sm, fmt, ...)                                     \
  CUDA_STATE_TRACE_DOMAIN_SM (CUDA_TRACE_STATE, sm, fmt, ##__VA_ARGS__)

#define CUDA_STATE_TRACE_WARP(warp, fmt, ...)                                 \
  CUDA_STATE_TRACE_DOMAIN_WARP (CUDA_TRACE_STATE, warp, fmt, ##__VA_ARGS__)

#define CUDA_STATE_TRACE_LANE(lane, fmt, ...)                                 \
  CUDA_STATE_TRACE_DOMAIN_LANE (CUDA_TRACE_STATE, lane, fmt, ##__VA_ARGS__)

/******************************************************************************
 *
 *				    System
 *
 ******************************************************************************/

cuda_state cuda_state::m_instance;

cuda_state::cuda_state ()
{
  reset ();
}

void
cuda_state::reset (void)
{
  m_instance.m_num_devices = 0;
  m_instance.m_suspended_devices_mask.fill (false);

  // Clear the warning masks
  // We may reattach to a different process using a different version debug
  // client
  unknown_device_attribute_warning = 0;
  unknown_sm_attribute_warning = 0;
  unknown_warp_attribute_warning = 0;
  unknown_lane_attribute_warning = 0;

  // Delete the contexts and modules through the use of unique pointers
  m_instance.set_current_context (nullptr);
  m_instance.m_context_map.clear ();
  m_instance.m_module_map.clear ();

  // Automatically deletes all device/sm/warp/lane objects
  m_instance.m_devs.clear ();
}

void
cuda_state::initialize (void)
{
  CUDA_STATE_TRACE ("");
  gdb_assert (cuda_initialized);

  reset ();

  cuda_options_force_set_launch_notification_update ();

  // Collect information about this system that doesn't change
  // during the debug session
  cuda_debugapi::get_num_devices (&m_instance.m_num_devices);
  gdb_assert (m_instance.m_num_devices > 0);

  // Size and clear the suspended devices mask
  m_instance.m_suspended_devices_mask
      = cuda_bitset (get_num_devices (), false);

  // Create the devices
  for (uint32_t dev_id = 0; dev_id < get_num_devices (); ++dev_id)
    {
      auto dev = new cuda_device (dev_id);
      m_instance.m_devs.emplace_back (dev);

      // Corefiles will have their state updated later after
      // processing stored CUDA events
      cuda_state::set_suspended_devices_mask (dev_id);
      if (target_has_execution ())
	{
	  // Suspend the device and read in the device state
	  cuda_debugapi::suspend_device (dev_id);
	  dev->update (CUDBG_RESPONSE_TYPE_FULL);
	}
    }
}

void
cuda_state::finalize (void)
{
  CUDA_STATE_TRACE ("");

  reset ();
}

CUDBGCapabilityFlags
cuda_state::get_supported_capabilities ()
{
  CUDA_STATE_TRACE ("");
  gdb_assert (cuda_initialized);

  return cuda_debugapi::get_supported_capabilities ();
}

cuda_context*
cuda_state::create_context (uint32_t dev_id, uint64_t context_id, uint32_t thread_id)
{
  cuda_trace ("Context create device %u context 0x%llx", dev_id, context_id);

  m_instance.m_context_map[context_id]
    = std::make_unique<cuda_context> (dev_id, context_id);

  return m_instance.m_context_map[context_id].get ();
}

void
cuda_state::destroy_context (uint64_t context_id)
{
  auto context = find_context_by_id (context_id);
  cuda_trace ("Context destroy device %u context 0x%llx", context->dev_id (), context_id);

  if (context == current_context ())
    set_current_context (nullptr);

  m_instance.m_context_map.erase (context_id);
}

cuda_context*
cuda_state::find_context_by_id (uint64_t context_id)
{
  auto iter = m_instance.m_context_map.find (context_id);
  if (iter != m_instance.m_context_map.end ())
    return iter->second.get ();
  return nullptr;
}

cuda_module*
cuda_state::create_module (uint64_t module_id,
			   CUDBGElfImageProperties properties,
			   uint64_t context_id,
			   uint64_t elf_image_size)
{
  CUDA_STATE_TRACE ("Module create module_id 0x%llx context_id 0x%llx size %llu",
		    module_id, context_id, elf_image_size);

  auto context = find_context_by_id (context_id);
  gdb_assert (context);
  
  // Install it in the map before doing anything else
  m_instance.m_module_map[module_id]
    = std::make_unique<cuda_module> (module_id, properties,
				     context, elf_image_size);

  // Now get the cuda_module* and update the context map
  auto module = m_instance.m_module_map[module_id].get ();
  context->add_module (module);

  // Now that the module is in the map, the objfile can be loaded.
  module->load_objfile ();

  return module;
}

void
cuda_state::destroy_module (uint64_t module_id)
{
  CUDA_STATE_TRACE ("Module destroy 0x%llx", module_id);

  auto module = find_module_by_id (module_id);
  gdb_assert (module);
  
  module->context ()->remove_module (module);
  m_instance.m_module_map.erase (module->id ());
}

cuda_module*
cuda_state::find_module_by_id (uint64_t module_id)
{
  CUDA_STATE_TRACE ("find_module_by_id 0x%llx", module_id);

  auto iter = m_instance.m_module_map.find (module_id);
  if (iter != m_instance.m_module_map.end ())
    return iter->second.get ();

  gdb_assert (false);
  return nullptr;
}

cuda_module*
cuda_state::find_module_by_address (CORE_ADDR addr)
{
  for (auto& iter : m_instance.m_module_map)
    if (iter.second.get ()->contains_address (addr))
      {
	CUDA_STATE_TRACE ("find_module_by_address 0x%llx module 0x%llx",
			  addr, iter.second.get ()->id ());
	return iter.second.get ();
      }

  CUDA_STATE_TRACE ("find_module_by_address 0x%llx - not found", addr);
  return nullptr;
}

bool
cuda_state::is_any_context_present (uint32_t dev_id)
{
  for (const auto& iter : m_instance.m_context_map)
    if (iter.second.get ()->dev_id () == dev_id)
      return true;
  return false;
}

void
cuda_state::device_invalidate_kernels (uint32_t dev_id)
{
  CUDA_STATE_TRACE_DEV (device (dev_id), "");

  kernels_invalidate (dev_id);
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
  CUDA_STATE_TRACE ("");

  for (auto& iter : m_instance.m_module_map)
    cuda_resolve_breakpoints (bp_number_from, iter.second.get ());
}

void
cuda_state::cleanup_breakpoints (void)
{
  CUDA_STATE_TRACE ("");

  // Passing in nullptr unresolves all CUDA breakpoints
  // across all elf_image objects in a single call
  cuda_unresolve_breakpoints (nullptr);
}

bool
cuda_state::is_context_active (cuda_context* context)
{
  const auto& iter = m_instance.m_context_map.find (context->id ());
  if (iter == m_instance.m_context_map.end ())
    return false;
  return true;
}

void
cuda_state::flush_disasm_caches ()
{
  CUDA_STATE_TRACE ("");

  for (auto& iter : m_instance.m_context_map)
    iter.second->flush_disasm_caches ();
}

void
cuda_state::invalidate ()
{
  CUDA_STATE_TRACE ("");

  // Invalidate everything
  for (auto &dev : m_instance.m_devs)
    dev->invalidate (!debug_invalidate, true);
}

bool
cuda_state::broken (cuda_coords &coords)
{
  cuda_coord_set<cuda_coord_set_type::lanes, select_valid | select_trap
						 | select_current_clock
						 | select_sngl>
      lanes{ cuda_coords::wild () };
  if (!lanes.size ())
    {
      CUDA_STATE_TRACE ("Not broken");
      return false;
    }

  coords = *lanes.begin ();

  const auto &coords_string = coords.to_string ();
  CUDA_STATE_TRACE ("Broken coords = %s", coords_string.c_str ());

  return true;
}

/******************************************************************************
 *
 *				    Device
 *
 ******************************************************************************/

cuda_device::cuda_device (uint32_t idx) : m_dev_id (idx)
{
  CUDA_STATE_TRACE_DEV (this, "initialize");

  // Collect information about this device that doesn't change
  cuda_debugapi::get_sm_type (dev_idx (), m_sm_type, sizeof (m_sm_type));

  cuda_debugapi::get_device_name (dev_idx (), m_dev_name, sizeof (m_dev_name));
  cuda_debugapi::get_device_type (dev_idx (), m_dev_type, sizeof (m_dev_type));

  auto sm_version = get_sm_version ();
  m_insn_size = (sm_version < 70) ? 8 : 16;

  cuda_debugapi::get_num_sms (dev_idx (), &m_num_sms);

  cuda_debugapi::get_num_warps (dev_idx (), &m_num_warps);
  gdb_assert (m_num_warps <= CUDBG_MAX_WARPS);

  cuda_debugapi::get_num_lanes (dev_idx (), &m_num_lanes);
  gdb_assert (m_num_lanes <= CUDBG_MAX_LANES);

  cuda_debugapi::get_num_registers (dev_idx (), &m_num_registers);
  cuda_debugapi::get_num_predicates (dev_idx (), &m_num_predicates);

  cuda_debugapi::get_num_uregisters (dev_idx (), &m_num_uregisters);
  cuda_debugapi::get_num_upredicates (dev_idx (), &m_num_upredicates);

  // Clear to start with, including the attribute size arrays
  m_info_sizes = { 0 };

  m_incremental = false;

  const auto update_env = getenv ("CUDBG_DEVICE_UPDATE");
  if (update_env && !strcmp (update_env, "LAZY"))
    m_incremental = true;
  else if (cuda_debugapi::api_version ().m_revision >= 140)
    m_incremental
	= !cuda_debugapi::get_device_info_sizes (dev_idx (), &m_info_sizes);
  else
    m_incremental = true;

  CUDA_STATE_TRACE_DEV (this, "Device %s Update: debugAPI revision %u",
			incremental () ? "Incremental" : "Batch",
			cuda_debugapi::api_version ().m_revision);

  if (incremental ())
    {
      const auto words
	  = (m_num_sms + 8 * sizeof (uint64_t) - 1) / sizeof (uint64_t);
      m_sm_active_mask.resize (m_num_sms, words * sizeof (uint64_t));
      m_sm_exception_mask.resize (m_num_sms, words * sizeof (uint64_t));
    }
  else
    {
      m_info_buffer = gdb::unique_xmalloc_ptr<uint8_t> (
	  (uint8_t *)xmalloc (m_info_sizes.requiredBufferSize));

      m_sm_active_mask.resize (m_num_sms,
			       m_info_sizes.deviceInfoAttributeSizes
				   [CUDBG_DEVICE_ATTRIBUTE_SM_ACTIVE_MASK]);

      m_sm_exception_mask.resize (
	  m_num_sms, m_info_sizes.deviceInfoAttributeSizes
			 [CUDBG_DEVICE_ATTRIBUTE_SM_EXCEPTION_MASK]);
    }

  m_sm_active_mask_p = false;
  m_sm_active_mask.fill (false);

  m_sm_exception_mask_p = false;
  m_sm_exception_mask.fill (false);

  // Now create the SMs / Warps / Lanes
  m_sms.reserve (m_num_sms);
  for (uint32_t sm_idx = 0; sm_idx < m_num_sms; ++sm_idx)
    m_sms.emplace_back (new cuda_sm (this, sm_idx));
}

cuda_device::~cuda_device ()
{
}

bool
cuda_device::suspended ()
{
  return cuda_state::device_suspended (dev_idx ());
}

void
cuda_device::invalidate (bool quietly, bool recurse)
{
  if (!quietly)
    CUDA_STATE_TRACE_DEV (this, "recurse %u", recurse);

  // FIXME: may want to handle this cache more intelligently by
  // removing grids as they exit instead of clearing the entire cache.
  m_grid_info.clear ();

  // Clear the SM masks
  m_sm_active_mask.fill (false);

  // Clear the exception mask
  m_sm_exception_mask.fill (false);
  m_sm_exception_mask_p = false;

  // Invalidate any kernels on the device
  kernels_invalidate (dev_idx ());

  // Quietly invalidate the SMs/Warps/Lanes
  if (recurse)
    for (auto &sm : m_sms)
      sm->invalidate (true, true);

  set_timestamp (0);
}

void
cuda_device::update (CUDBGDeviceInfoQueryType_t type)
{
  CUDA_STATE_TRACE_DEV (this, "%s state update type %u clock %u",
			incremental () ? "Incremental" : "Batch", type,
			cuda_clock ());

  // Device must be suspended at this point
  gdb_assert (suspended ());

  try
    {
      if (incremental ())
	{
	  CUDA_STATE_TRACE_DEV (this, "Incremental update - invalidate device");
	  invalidate (!debug_invalidate, true);
	  CUDA_STATE_TRACE_DEV (this, "Incremental update - invalidate done");
	}
      else
	{
	  uint32_t backend_length = 0;
	  const auto buffer_length = m_info_sizes.requiredBufferSize;
	  if (!cuda_debugapi::get_device_info (dev_idx (), type,
					       m_info_buffer.get (),
					       buffer_length, &backend_length))
	    error ("Failed to read device state information for device %u",
		   dev_idx ());

	  gdb_assert (backend_length > 0);
	  gdb_assert (backend_length <= buffer_length);

	  CUDA_STATE_TRACE_DEV (this, "Batch info decoding size %u",
				backend_length);
	  decode (m_info_sizes, m_info_buffer.get (), backend_length);
	  CUDA_STATE_TRACE_DEV (this, "Batch info decoding done");
	}
    }
  catch (const gdb_exception_error &exception)
    {
      CUDA_STATE_TRACE_DEV (this, "Exception during update_state(): %s",
			    exception.what ());
      throw exception;
    }
}

void
cuda_device::update_exception_state ()
{
  if (m_sm_exception_mask_p)
    return;

  CUDA_STATE_TRACE_DEV (this, "mask valid %u", m_sm_exception_mask_p);

  m_sm_exception_mask.fill (false);

  // Read the SM exception mask, pad additional uint64_t to meet the
  // requirements of cuda_debugapi::read_device_exception_state()
  if (is_any_context_present ())
    {
      const auto words = (get_num_sms () + 8 * sizeof (uint64_t) - 1)
			 / (8 * sizeof (uint64_t));
      cuda_debugapi::read_device_exception_state (
	  dev_idx (), (uint64_t *)m_sm_exception_mask.data (), words);

      for (auto sm_id = 0; sm_id < get_num_sms (); ++sm_id)
	if (m_sm_exception_mask[sm_id])
	  CUDA_STATE_TRACE_SM (sm (sm_id), "SM has an exception");
    }
  else
    CUDA_STATE_TRACE_DEV (this, "no context present");

  for (auto sm_id = 0; sm_id < get_num_sms (); ++sm_id)
    if (!m_sm_exception_mask[sm_id])
      for (auto wp_id = 0; wp_id < get_num_warps (); ++wp_id)
	for (auto ln_id = 0; ln_id < get_num_lanes (); ++ln_id)
	  sm (sm_id)->warp (wp_id)->lane (ln_id)->set_exception_none ();

  m_sm_exception_mask_p = true;
}

bool
cuda_device::sm_has_exception (uint32_t sm_id)
{
  update_exception_state ();
  return m_sm_exception_mask[sm_id] ? true : false;
}

bool
cuda_device::has_exception ()
{
  update_exception_state ();
  CUDA_STATE_TRACE_DEV (this, "exception %s",
			m_sm_exception_mask.any () ? "true" : "false");
  return m_sm_exception_mask.any ();
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
  m_sm_version_p = true;

  return m_sm_version;
}

uint32_t
cuda_device::get_pci_bus_id ()
{
  if (m_pci_bus_info_p)
    return m_pci_bus_id;

  cuda_debugapi::get_device_pci_bus_info (dev_idx (), &m_pci_bus_id,
					  &m_pci_dev_id);
  m_pci_bus_info_p = true;

  return m_pci_bus_id;
}

uint32_t
cuda_device::get_pci_dev_id ()
{
  if (m_pci_bus_info_p)
    return m_pci_dev_id;

  cuda_debugapi::get_device_pci_bus_info (dev_idx (), &m_pci_bus_id,
					  &m_pci_dev_id);
  m_pci_bus_info_p = true;

  return m_pci_dev_id;
}

uint32_t
cuda_device::get_num_kernels ()
{
  gdb_assert (valid ());

  uint32_t num_kernels = 0;
  for (auto kernel = kernels_get_first_kernel (); kernel;
       kernel = kernels_get_next_kernel (kernel))
    if (kernel_get_dev_id (kernel) == dev_idx ())
      ++num_kernels;

  return num_kernels;
}

bool
cuda_device::is_any_context_present () const
{
  return cuda_state::is_any_context_present (dev_idx ());
}

const cuda_bitset &
cuda_device::get_active_sms_mask ()
{
  CUDA_STATE_TRACE_DEV (this, "");

  // Recalculate if needed
  // Always up-to-date in batch-update mode
  if (!m_sm_active_mask_p)
    {
      m_sm_active_mask.fill (false);

      /* For every sm */
      for (auto sm_id = 0; sm_id < get_num_sms (); ++sm_id)
	{
	  /* For every warp in the sm */
	  for (auto wp_id = 0; wp_id < get_num_warps (); ++wp_id)
	    {
	      /* Set to true if there is a valid warp in the sm */
	      if (sm (sm_id)->warp_valid (wp_id))
		{
		  m_sm_active_mask.set (sm_id, true);
		  break;
		}
	    }
	}
      m_sm_active_mask_p = true;
    }

  return m_sm_active_mask;
}

kernel_t
cuda_device::get_kernel (uint64_t grid_id)
{
  gdb_assert (grid_id != 0);

  auto kernel = kernels_find_kernel_by_grid_id (dev_idx (), grid_id);
  if (!kernel)
    {
      create_kernel (grid_id);
      kernel = kernels_find_kernel_by_grid_id (dev_idx (), grid_id);
      gdb_assert (kernel);
    }

  return kernel;
}

void
cuda_device::print ()
{
  CUDA_STATE_TRACE_DEV (this, "");
}

void
cuda_device::resume ()
{
  // There can be redundant calls to this, don't error out if so
  if (suspended ())
    {
      CUDA_STATE_TRACE_DEV (this, "Resuming device");

      cuda_debugapi::resume_device (dev_idx ());

      cuda_state::clear_suspended_devices_mask (dev_idx ());
    }
  else
    CUDA_STATE_TRACE_DEV (this, "Device already resumed");
}

const CUDBGGridInfo &
cuda_device::get_grid_info (uint64_t grid_id)
{
  // Helper function that may be called even when the device is not
  // valid.
  gdb_assert (grid_id != 0);

  const auto iter = m_grid_info.find (grid_id);
  if (iter != m_grid_info.end ())
    return iter->second;

  // Get the grid status and check it's validity before requesting
  // the grid info.
  CUDBGGridStatus grid_status = CUDBG_GRID_STATUS_INVALID;
  cuda_debugapi::get_grid_status (dev_idx (), grid_id, &grid_status);
  if (grid_status == CUDBG_GRID_STATUS_INVALID)
    error ("Invalid grid status: gridId %d status %u", (int)grid_id,
	   grid_status);

  CUDBGGridInfo grid_info = { 0 };
  cuda_debugapi::get_grid_info (dev_idx (), grid_id, &grid_info);

  CUDA_STATE_TRACE_DEV (
      this, "grid %ld gridDim (%u, %u, %u) blockDim (%u, %u, %u)", grid_id,
      grid_info.gridDim.x, grid_info.gridDim.y, grid_info.gridDim.z,
      grid_info.blockDim.x, grid_info.blockDim.y, grid_info.blockDim.z);

  // Return the reference to the CUDBGGridInfo in the map, not the
  // local variable "grid_info" which is on the stack.
  m_grid_info[grid_id] = grid_info;
  return m_grid_info[grid_id];
}

void
cuda_device::create_kernel (uint64_t grid_id)
{
  const auto &grid_info = get_grid_info (grid_id);

  kernels_start_kernel (
      dev_idx (), grid_id, grid_info.functionEntry, grid_info.context,
      grid_info.module, grid_info.gridDim, grid_info.blockDim, grid_info.type,
      grid_info.parentGridId, grid_info.origin, true, grid_info.clusterDim);
}

void
cuda_device::suspend ()
{
  // There can be redundant calls to this, don't error out if so
  if (!suspended ())
    {
      CUDA_STATE_TRACE_DEV (this, "Suspending device");

      cuda_debugapi::suspend_device (dev_idx ());

      cuda_state::set_suspended_devices_mask (dev_idx ());

      update (CUDBG_RESPONSE_TYPE_UPDATE);
    }
  else
    CUDA_STATE_TRACE_DEV (this, "Device already suspended");
}

void
cuda_device::set_device_spec (uint32_t num_sms, uint32_t num_warps,
			      uint32_t num_lanes, uint32_t num_registers,
			      const char *dev_type, const char *sm_type)
{
  gdb_assert (is_remote_target (current_inferior ()->process_target ()));

  // Number of SMs is variable, and may exceed the CUDBG_MAX_SMS cuda-gdb
  // was compiled with.

  // Number of warps/lanes needs to be checked as certain datastructures
  // depend on these being correct.
  gdb_assert (num_warps <= CUDBG_MAX_WARPS);
  gdb_assert (num_lanes <= CUDBG_MAX_LANES);

  m_num_sms = num_sms;
  m_num_warps = num_warps;
  m_num_lanes = num_lanes;
  m_num_registers = num_registers;
  strcpy (m_dev_type, dev_type);
  strcpy (m_sm_type, sm_type);

  if (strlen (m_sm_type) < 4 || strncmp (m_sm_type, "sm_", 3) != 0)
    error ("unknown sm_type %s", m_sm_type);

  m_sm_version = atoi (&m_sm_type[3]);
  m_sm_version_p = true;

  m_insn_size = (m_sm_version < 70) ? 8 : 16;
}

/******************************************************************************
 *
 *				      SM
 *
 ******************************************************************************/

cuda_sm::cuda_sm (cuda_device *parent_dev, uint32_t sm_idx)
    : m_sm_idx (sm_idx), m_device (parent_dev)
{
  m_warps.clear ();
  for (uint32_t idx = 0; idx < device ()->get_num_warps (); ++idx)
    m_warps.emplace_back (new cuda_warp (this, idx));
}

cuda_sm::~cuda_sm () {}

uint32_t
cuda_sm::dev_idx () const
{
  return m_device->dev_idx ();
}

void
cuda_sm::invalidate (bool quietly, bool recurse)
{
  if (!quietly)
    CUDA_STATE_TRACE_SM (this, "recurse %u", recurse);

  device ()->clear_sm_exception_mask_p ();

  // Reset optional SM exception state
  reset_sm_exception_info ();

  m_valid_warps_mask_p = false;
  cuda_api_clear_mask (&m_valid_warps_mask);

  m_broken_warps_mask_p = false;
  cuda_api_clear_mask (&m_broken_warps_mask);

  // Quietly invalidate the Warps/Lanes if requested
  if (recurse)
    for (auto &wp : m_warps)
      wp->invalidate (true, true);
}

bool
cuda_sm::has_exception ()
{
  return device ()->sm_has_exception (sm_idx ());
}

void
cuda_sm::reset_sm_exception_info ()
{
  m_sm_exception_info_p = false;
  m_exception = CUDBG_EXCEPTION_NONE;
  m_error_pc = 0;
  m_error_pc_available = false;
}

bool
cuda_sm::fetch_sm_exception_info ()
{
  // This works with invalid warps. It is the only way to
  // detect an exception on the SM but the triggering warp
  // has already exited.
  if (!m_sm_exception_info_p)
    {
      try
	{
	  cuda_debugapi::read_sm_exception (dev_idx (), sm_idx (),
					    &m_exception, &m_error_pc,
					    &m_error_pc_available);
	  m_sm_exception_info_p = true;
	}
      catch (const gdb_exception_error &exception)
	{
	  /* This can happen if we are running on the classic stack or
	   * when using an older corefile format. */
	  reset_sm_exception_info ();
	  return false;
	}
    }
  return true;
}

CUDBGException_t
cuda_sm::get_exception ()
{
  if (cuda_debugapi::api_version ().m_revision < 145)
    return CUDBG_EXCEPTION_NONE;

  if (!fetch_sm_exception_info ())
    return CUDBG_EXCEPTION_NONE;

  return m_exception;
}

bool
cuda_sm::has_error_pc ()
{
  if (!has_exception () || (cuda_debugapi::api_version ().m_revision < 145))
    return false;

  if (!fetch_sm_exception_info ())
    return false;

  return m_error_pc_available;
}

uint64_t
cuda_sm::get_error_pc ()
{
  if (has_error_pc ())
    return m_error_pc;
  gdb_assert (0);
}

const cuda_api_warpmask *
cuda_sm::get_valid_warps_mask ()
{
  if (!m_valid_warps_mask_p)
    {
      if (device ()->incremental ())
	cuda_debugapi::read_valid_warps (dev_idx (), sm_idx (),
					 &m_valid_warps_mask);
      else
	cuda_api_clear_mask (&m_valid_warps_mask);
      m_valid_warps_mask_p = true;
    }
  return &m_valid_warps_mask;
}

const cuda_api_warpmask *
cuda_sm::get_broken_warps_mask ()
{
  if (!m_broken_warps_mask_p)
    {
      if (device ()->incremental ())
	cuda_debugapi::read_broken_warps (dev_idx (), sm_idx (),
					  &m_broken_warps_mask);
      else
	cuda_api_clear_mask (&m_broken_warps_mask);
      m_broken_warps_mask_p = true;
    }
  return &m_broken_warps_mask;
}

bool
cuda_sm::resume_warps_until_pc (cuda_api_warpmask *mask, uint64_t pc)
{
  CUDA_STATE_TRACE_SM (this, "pc 0x%llx", pc);

  // No point in resuming warps, if one of them is already there
  for (auto wp_id = 0; wp_id < device ()->get_num_warps (); ++wp_id)
    if (cuda_api_get_bit (mask, wp_id)
	&& (pc == warp (wp_id)->get_active_pc ()))
      {
	CUDA_STATE_TRACE_SM (
	    this, "Skipping resume: warp %u already at pc 0x%llx", wp_id, pc);
	return false;
      }

  CUDA_STATE_TRACE_SM (
      this, "Resuming warp mask %" WARP_MASK_FORMAT " until pc 0x%llx",
      cuda_api_mask_string (mask), pc);

  // If resume warps is not possible - abort
  if (!cuda_debugapi::resume_warps_until_pc (dev_idx (), sm_idx (), mask, pc))
    return false;

  if (cuda_options_software_preemption ())
    {
      if (device ()->incremental ())
	device ()->invalidate (!debug_invalidate, true);
      else
	device ()->update (CUDBG_RESPONSE_TYPE_UPDATE);
      return true;
    }

  if (device ()->incremental ())
    {
      // Invalidate the cache for the warps that have been single-stepped.
      for (auto wp_id = 0; wp_id < device ()->get_num_warps (); ++wp_id)
	if (cuda_api_get_bit (mask, wp_id))
	  warp (wp_id)->invalidate (!debug_invalidate, true);

      // must invalidate the SM since that's where the warp valid mask lives
      invalidate (!debug_invalidate, false);
    }
  else
    device ()->update (CUDBG_RESPONSE_TYPE_UPDATE);

  return true;
}

bool
cuda_sm::single_step_warp (uint32_t wp_id, uint32_t lane_id_hint,
			   uint32_t nsteps, uint32_t flags,
			   cuda_api_warpmask *single_stepped_warp_mask)
{
  CUDA_STATE_TRACE_SM (
      this, "wp_id %u nsteps %u valid warp mask %" WARP_MASK_FORMAT, wp_id,
      nsteps, cuda_api_mask_string (get_valid_warps_mask ()));

  gdb_assert (wp_id < device ()->get_num_warps ());

  cuda_api_clear_mask (single_stepped_warp_mask);
  bool rc;
  if (cuda_debugapi::api_version ().m_revision >= 142)
    rc = cuda_debugapi::single_step_warp (dev_idx (), sm_idx (), wp_id,
					  lane_id_hint, nsteps, flags,
					  single_stepped_warp_mask);
  else
    rc = cuda_debugapi::single_step_warp (dev_idx (), sm_idx (), wp_id, nsteps,
					  single_stepped_warp_mask);
  if (!rc)
    {
      CUDA_STATE_TRACE_SM (this, "single_step_warp() failed");
      device ()->update (CUDBG_RESPONSE_TYPE_FULL);
      return rc;
    }

  if (cuda_options_software_preemption ())
    {
      device ()->update (CUDBG_RESPONSE_TYPE_FULL);
      return true;
    }

  if (!cuda_api_get_bit (single_stepped_warp_mask, wp_id))
    CUDA_STATE_TRACE_SM (this,
			 "Updated warp mask does not include the stepped warp "
			 "%" WARP_MASK_FORMAT,
			 cuda_api_mask_string (single_stepped_warp_mask));

  // Select all but the single-stepped warp in the mask
  cuda_api_warpmask other_warps;
  cuda_api_clear_mask (&other_warps);
  cuda_api_set_bit (&other_warps, wp_id, 1);
  cuda_api_not_mask (&other_warps, &other_warps);

  // Check if any additional warps were stepped, and warn if so
  cuda_api_warpmask all_stepped_warp_mask;
  cuda_api_and_mask (&all_stepped_warp_mask, &other_warps,
		     single_stepped_warp_mask);
  if (cuda_api_has_bit (&all_stepped_warp_mask))
    {
      warning ("Warp(s) other than the current warp had to be "
	       "single-stepped:%" WARP_MASK_FORMAT,
	       cuda_api_mask_string (&all_stepped_warp_mask));
      if (device ()->incremental ())
	device ()->invalidate (false, true);
      else
	device ()->update (CUDBG_RESPONSE_TYPE_UPDATE);
    }
  else
    {
      if (device ()->incremental ())
	invalidate (false, true);
      else
	device ()->update (CUDBG_RESPONSE_TYPE_UPDATE);
    }

  // Active mask is valid due to cuda_device::update_state() call above
  if (!warp (wp_id)->valid ())
    CUDA_STATE_TRACE_WARP (warp (wp_id), "Warp %u exited during single-step",
			   wp_id);
  else
    {
      const auto active_lanes_mask = warp (wp_id)->get_active_lanes_mask ();
      CUDA_STATE_TRACE_WARP (
	  warp (wp_id),
	  "Stepped warp mask %" WARP_MASK_FORMAT
	  " active lanes 0x%08x active pc 0x%llx",
	  cuda_api_mask_string (single_stepped_warp_mask), active_lanes_mask,
	  active_lanes_mask ? warp (wp_id)->get_active_pc () : 0);
    }

  // Return single-step success/failure
  return rc;
}

void
cuda_sm::update_state ()
{
  if (device ()->incremental () && (timestamp () < cuda_clock ()))
    {
      CUDA_STATE_TRACE_SM (this, "timestamp %lu clock %lu", timestamp (),
			   cuda_clock ());

      // Invalidate the SM state
      // Warp invalidation is handled by the loop below
      invalidate (!debug_invalidate, false);

      // Update all the valid warps
      for (auto &wp : m_warps)
	if (warp_valid (wp->warp_idx ()))
	  wp->update_state ();
	else
	  wp->invalidate (true, true);

      set_timestamp (cuda_clock ());

      CUDA_STATE_TRACE_SM (this, "done");
    }
}

/******************************************************************************
 *
 *				     Warps
 *
 ******************************************************************************/

cuda_warp::cuda_warp (cuda_sm *parent_sm, uint32_t wp)
    : m_warp_idx (wp), m_sm (parent_sm), m_lanes ()
{
  // Just size it.
  m_uregisters_p.resize (sm ()->device ()->get_num_uregisters ());

  const auto n_lanes = sm ()->device ()->get_num_lanes ();
  for (auto idx = 0; idx < n_lanes; ++idx)
    lane (idx)->configure (this, idx);
}

cuda_warp::~cuda_warp () {}

uint32_t
cuda_warp::dev_idx () const
{
  return m_sm->device ()->dev_idx ();
}

uint32_t
cuda_warp::sm_idx () const
{
  return m_sm->sm_idx ();
}

bool
cuda_warp::valid ()
{
  return sm ()->warp_valid (warp_idx ());
}

uint32_t
cuda_warp::get_uregister (uint32_t regno)
{
  gdb_assert (valid ());

  // If requesting uniform register values on a device w/o uniform
  // registers, simply return 0. This is to support commands like
  // "info registers"
  if (regno >= sm ()->device ()->get_num_uregisters ())
    return 0;

  // If uniform registers are accessed, just allocate and read
  // them all in.
  if (regno >= m_uregisters.size ())
    m_uregisters.resize (sm ()->device ()->get_num_uregisters ());
  gdb_assert (m_uregisters.size () == sm ()->device ()->get_num_uregisters ());

  // m_uregisters_p was sized in the constructor.
  if (!m_uregisters_p[regno])
    {
      cuda_debugapi::read_uregister_range (dev_idx (), sm_idx (), warp_idx (),
					   0, sm ()->device ()->get_num_uregisters (),
					   &m_uregisters[0]);

      // We read them all
      m_uregisters_p.fill (true);
    }

  return m_uregisters[regno];
}

void
cuda_warp::set_uregister (uint32_t regno, uint32_t value)
{
  gdb_assert (valid ());

  if (!sm ()->device ()->get_num_uregisters ())
    error ("Attempting to set register UR%u for a device without uniform "
	   "registers",
	   regno);

  if (regno >= sm ()->device ()->get_num_uregisters ())
    error ("Attempting to set register UR%u: register out of range for this "
	   "device",
	   regno);

  cuda_debugapi::write_uregister (dev_idx (), sm_idx (), warp_idx (), regno,
				  value);

  // If uniform registers are accessed, just allocate and read them
  // all in. Each warp gets the full set anyways unlike general
  // purpose registers.
  if (m_uregisters.size () == 0)
    m_uregisters.resize (sm ()->device ()->get_num_uregisters ());

  m_uregisters[regno] = value;

  // m_uregisters_p already sized in the constructor
  m_uregisters_p.set (regno, true);
}

bool
cuda_warp::get_upredicate (uint32_t pred)
{
  gdb_assert (valid ());

  // If there are no uniform predicate registers in this device,
  // simply return false.
  const auto num_upreds = sm ()->device ()->get_num_upredicates ();
  if (!num_upreds)
    return false;

  gdb_assert (pred < num_upreds);

  if (!(m_upredicates_p & (1 << pred)))
    {
      cuda_debugapi::read_upredicates (dev_idx (), sm_idx (), warp_idx (),
                                       num_upreds, m_upredicates);
      m_upredicates_p = (1 << num_upreds) - 1;
    }

  return m_upredicates[pred] != 0;
}

void
cuda_warp::set_upredicate (uint32_t pred, bool value)
{
  gdb_assert (valid ());

  if (!sm ()->device ()->get_num_upredicates ())
    error ("Attempting to set predicate UP%u for a device without uniform "
	   "registers",
	   pred);

  gdb_assert (pred < sm ()->device ()->get_num_upredicates ());

  // If we don't have them all, read them all in
  auto all_upredicate_mask
      = (1 << sm ()->device ()->get_num_upredicates ()) - 1;
  if (m_upredicates_p != all_upredicate_mask)
    {
      cuda_debugapi::read_upredicates (
	  dev_idx (), sm_idx (), warp_idx (),
	  sm ()->device ()->get_num_upredicates (), m_upredicates);
      m_upredicates_p = all_upredicate_mask;
    }

  m_upredicates[pred] = value;
  m_upredicates_p |= (1 << pred);

  cuda_debugapi::write_upredicates (dev_idx (), sm_idx (), warp_idx (),
				    sm ()->device ()->get_num_upredicates (),
				    m_upredicates);
}

void
cuda_warp::invalidate (bool quietly, bool recurse)
{
  if (!quietly)
    CUDA_STATE_TRACE_WARP (this, "recurse %d", recurse);

  // Grid IDs are never 0
  m_grid_id = 0;
  m_kernel = nullptr;

  m_base_thread_idx = { 0, 0, 0 };

  m_block_idx = { 0, 0, 0 };
  m_block_idx_p = false;

  m_valid_lanes_mask = 0;
  m_valid_lanes_mask_p = false;

  m_active_lanes_mask = 0;
  m_active_lanes_mask_p = false;

  // Even though we have valid bits, clear the vector in order
  // to save space after this warp exits (until it's reused)
  m_uregisters.clear ();
  m_uregisters_p.fill (false);

  // Default to no errorpc
  m_error_pc_p = false;
  m_error_pc = 0;
  m_error_pc_available = false;

  // Default to no cluster exception target block idx
  m_cluster_exception_target_block_idx_p = false;
  m_cluster_exception_target_block_idx = { 0 };
  m_cluster_exception_target_block_idx_available = false;

  m_cluster_idx = CuDim3{ 0, 0, 0 };
  m_cluster_idx_p = false;
  m_cluster_dim = CuDim3{ 0, 0, 0 };
  m_cluster_dim_p = false;

  // Reset the warp resources predicate and values
  m_warp_resources_p = false;
  // Number of registers currently allocated
  m_registers_allocated = 0;
  // Amount of shared memory currently allocated
  m_shared_mem_size = 0;
  
  // Quietly invalidate the lanes
  if (recurse)
    for (auto &ln : m_lanes)
      ln.invalidate (true);

  set_timestamp (0);
}

uint32_t
cuda_warp::get_lowest_active_lane ()
{
  gdb_assert (valid ());
  gdb_assert (get_active_lanes_mask () != 0);

  for (auto ln_id = 0; ln_id < sm ()->device ()->get_num_lanes (); ++ln_id)
    if (lane_active (ln_id))
      return ln_id;

  return ~0;
}

void
cuda_warp::set_grid_id (uint64_t grid_id)
{
  m_grid_id = grid_id;
  m_kernel = nullptr;
}

uint64_t
cuda_warp::get_grid_id ()
{
  if (is_remote_target (current_inferior ()->process_target ()) && !m_grid_id)
    cuda_remote_update_grid_id_in_sm (dev_idx (), sm_idx ());

  if (!m_grid_id)
    update_state ();

  gdb_assert (valid ());
  gdb_assert (m_grid_id != 0);

  return m_grid_id;
}

void
cuda_warp::set_block_idx (const CuDim3 &block_idx)
{
  gdb_assert (is_remote_target (current_inferior ()->process_target ()));

  m_block_idx = block_idx;
  m_block_idx_p = true;
}

const CuDim3 &
cuda_warp::get_block_idx ()
{
  if (is_remote_target (current_inferior ()->process_target ())
      && !m_block_idx_p && sm ()->warp_valid (warp_idx ()))
    cuda_remote_update_block_idx_in_sm (dev_idx (), sm_idx ());
  if (!m_block_idx_p)
    update_state ();

  return m_block_idx;
}

void
cuda_warp::set_cluster_idx (const CuDim3 &cluster_idx)
{
  gdb_assert (is_remote_target (current_inferior ()->process_target ()));

  m_cluster_idx = cluster_idx;
  m_cluster_idx_p = true;
}

const CuDim3 &
cuda_warp::get_cluster_idx ()
{
  if (is_remote_target (current_inferior ()->process_target ())
      && !m_cluster_idx_p && sm ()->warp_valid (warp_idx ()))
    cuda_remote_update_cluster_idx_in_sm (dev_idx (), sm_idx ());
  if (!m_cluster_idx_p)
    update_state ();

  return m_cluster_idx;
}

void
cuda_warp::set_cluster_dim (const CuDim3 &cluster_dim)
{
  gdb_assert (is_remote_target (current_inferior ()->process_target ()));

  m_cluster_dim = cluster_dim;
  m_cluster_dim_p = true;
}

const CuDim3 &
cuda_warp::get_cluster_dim ()
{
  if (is_remote_target (current_inferior ()->process_target ())
      && !m_cluster_dim_p && sm ()->warp_valid (warp_idx ()))
    cuda_remote_update_cluster_dim_in_sm (dev_idx (), sm_idx ());
  if (!m_cluster_dim_p)
    update_state ();

  return m_cluster_dim;
}

kernel_t
cuda_warp::get_kernel ()
{
  if (!m_kernel)
    m_kernel = sm ()->device ()->get_kernel (get_grid_id ());

  gdb_assert (m_kernel);
  return m_kernel;
}

uint32_t
cuda_warp::get_valid_lanes_mask ()
{
  if (m_valid_lanes_mask_p)
    return m_valid_lanes_mask;

  if (sm ()->warp_valid (warp_idx ()))
    {
      update_state ();
      return m_valid_lanes_mask;
    }

  m_valid_lanes_mask = 0;
  m_valid_lanes_mask_p = true;

  if (!timestamp_valid ())
    set_timestamp (cuda_clock ());

  return 0;
}

uint32_t
cuda_warp::get_active_lanes_mask ()
{
  if (m_active_lanes_mask_p)
    return m_active_lanes_mask;

  update_state ();

  return m_active_lanes_mask;
}

bool
cuda_warp::has_error_pc ()
{
  if (!m_error_pc_p)
    {
      cuda_debugapi::read_error_pc (dev_idx (), sm_idx (), warp_idx (),
				    &m_error_pc, &m_error_pc_available);
      m_error_pc_p = true;
    }

  return m_error_pc_available;
}

uint64_t
cuda_warp::get_error_pc ()
{
  if (has_error_pc ())
    return m_error_pc;
  gdb_assert (0);
}

bool
cuda_warp::has_cluster_exception_target_block_idx ()
{
  if (!m_cluster_exception_target_block_idx_p)
    {
      cuda_debugapi::get_cluster_exception_target_block (
	  dev_idx (), sm_idx (), warp_idx (),
	  &m_cluster_exception_target_block_idx,
	  &m_cluster_exception_target_block_idx_available);
      m_cluster_exception_target_block_idx_p = true;
    }

  return m_cluster_exception_target_block_idx_available;
}

CuDim3
cuda_warp::get_cluster_exception_target_block_idx ()
{
  if (has_cluster_exception_target_block_idx ())
    return m_cluster_exception_target_block_idx;
  gdb_assert (0);
}

void
cuda_warp::update_warp_resources ()
{
  if (!m_warp_resources_p)
    {
      // If reading warp resources isn't supported by the debugger backend,
      // the cuda_debugapi call will return with resources set to 0.
      CUDBGWarpResources resources;
      cuda_debugapi::read_warp_resources (dev_idx (), sm_idx (), warp_idx (),
    					  &resources);
      m_registers_allocated = resources.numRegisters;
      m_shared_mem_size = resources.sharedMemSize;
      m_warp_resources_p = true;
    }
}

uint32_t
cuda_warp::registers_allocated ()
{
  update_warp_resources ();
  return m_registers_allocated;
}

uint32_t
cuda_warp::shared_mem_size ()
{
  update_warp_resources ();
  return m_shared_mem_size;
}

void
cuda_warp::update_state ()
{
  CUDA_STATE_TRACE_WARP (this, "timestamp %lu clock %lu", timestamp (),
			 cuda_clock ());

  // Just invalidate the warp, and not the lanes
  // We'll handle the lanes below
  invalidate (!debug_invalidate, false);

  CUDBGWarpState warp_state{ 0 };
  cuda_debugapi::read_warp_state (dev_idx (), sm_idx (), warp_idx (),
				  &warp_state);

  CUDA_STATE_TRACE_WARP (
      this,
      "gridId %d block (%u, %u, %u) validLanes 0x%08x activeLanes 0x%08x",
      (int)warp_state.gridId, warp_state.blockIdx.x, warp_state.blockIdx.y,
      warp_state.blockIdx.z, warp_state.validLanes, warp_state.activeLanes);

  m_grid_id = warp_state.gridId;
  m_kernel = nullptr;

  m_valid_lanes_mask = warp_state.validLanes;
  m_valid_lanes_mask_p = true;

  m_active_lanes_mask = warp_state.activeLanes;
  m_active_lanes_mask_p = true;

  m_block_idx = warp_state.blockIdx;
  m_block_idx_p = true;

  m_cluster_idx = warp_state.clusterIdx;
  m_cluster_idx_p = true;
  m_cluster_dim = warp_state.clusterDim;
  m_cluster_dim_p = true;

  m_error_pc_p = true;
  m_error_pc = warp_state.errorPC;
  m_error_pc_available = warp_state.errorPCValid;

  m_cluster_exception_target_block_idx_p = true;
  m_cluster_exception_target_block_idx
      = warp_state.clusterExceptionTargetBlockIdx;
  m_cluster_exception_target_block_idx_available
      = warp_state.clusterExceptionTargetBlockIdxValid;

  const auto num_lanes = sm ()->device ()->get_num_lanes ();
  for (auto ln_id = 0; ln_id < num_lanes; ++ln_id)
    if (m_valid_lanes_mask & (1ULL << ln_id))
      lane (ln_id)->update (warp_state.lane[ln_id]);
    else
      lane (ln_id)->invalidate (!debug_invalidate);

  if (!timestamp_valid ())
    set_timestamp (cuda_clock ());

  CUDA_STATE_TRACE_WARP (this, "done");
}

/******************************************************************************
 *
 *				     Lanes
 *
 ******************************************************************************/

cuda_lane::cuda_lane () : m_lane_idx (~0), m_warp (nullptr) {}

cuda_lane::cuda_lane (cuda_warp *parent_warp, uint32_t idx)
    : m_lane_idx (idx), m_warp (parent_warp)
{
}

cuda_lane::~cuda_lane () {}

// This method handles most post-constructor initialization. Necessary to
// do it this way as cuda_lane objects are in an array instead of a
// vector of pointers, which means the empty constructor is used
// which can't intializae these fields.
void
cuda_lane::configure (cuda_warp *warp, uint32_t idx)
{
  m_warp = warp;
  m_lane_idx = idx;
  m_registers_p.resize (warp->sm ()->device ()->get_num_registers ());
}

uint32_t
cuda_lane::dev_idx () const
{
  return m_warp->sm ()->device ()->dev_idx ();
}

uint32_t
cuda_lane::sm_idx () const
{
  return m_warp->sm ()->sm_idx ();
}

uint32_t
cuda_lane::warp_idx () const
{
  return m_warp->warp_idx ();
}

const CuDim3 &
cuda_lane::get_thread_idx ()
{
  /* In a remote session, we fetch the threadIdx of all valid thread in the
   * warp using one rsp packet to reduce the amount of communication. */
  if (is_remote_target (current_inferior ()->process_target ())
      && !m_thread_idx_p && warp ()->lane_valid (lane_idx ()))
    cuda_remote_update_thread_idx_in_warp (dev_idx (), sm_idx (), warp_idx ());

  if (!m_thread_idx_p)
    warp ()->update_state ();

  return m_thread_idx;
}

CUDBGException_t
cuda_lane::get_exception ()
{
  gdb_assert (warp ()->lane_valid (lane_idx ()));

  if (!m_exception_p)
    warp ()->update_state ();

  return m_exception;
}

void
cuda_lane::invalidate (bool quietly)
{
  if (!quietly)
    CUDA_STATE_TRACE_LANE (this, "");

  m_pc = 0;
  m_pc_p = false;

  m_thread_idx = { 0, 0, 0 };
  m_thread_idx_p = false;

  m_exception = CUDBG_EXCEPTION_NONE;
  m_exception_p = false;

  // The following fields are read on demand, so they need predicate flags
  m_call_depth_p = false;
  m_call_depth = 0;

  m_syscall_call_depth_p = false;
  m_syscall_call_depth = 0;

  m_cc_register_p = false;
  m_cc_register = 0;

  // Even though we have valid bits, clear the vectors in order
  // to save space after this warp/lane exits (until it's reused)
  m_registers.clear ();
  m_registers_p.fill (false);

  m_return_address.clear ();

  // Just clear the valid predicate bits
  m_predicates_p = false;

  set_timestamp (0);
}

bool
cuda_lane::active ()
{
  return warp ()->lane_active (lane_idx ());
}

bool
cuda_lane::divergent ()
{
  return warp ()->lane_divergent (lane_idx ());
}

uint64_t
cuda_lane::get_pc ()
{
  if (!m_pc_p)
    warp ()->update_state ();

  return m_pc;
}

uint32_t
cuda_lane::get_register (uint32_t regno)
{
  if (regno == CUDA_STATE_REGISTER_RZ)
    return 0;

  // Round down
  auto reg_range_size = CUDA_STATE_REGISTER_RANGE_SIZE;
  auto start_regno = regno & ~(reg_range_size - 1);

  // end_regno is the number just past the last register we want to read.
  // Some of the UD backends and generated coredumps encode 255 instead of
  // 256 for the number of registers, but exclude the zero register from
  // the range
  auto end_regno
      = std::min (start_regno + reg_range_size, CUDA_REG_MAX_REGISTERS);

  gdb_assert (end_regno > start_regno);
  gdb_assert (regno < end_regno);

  // Expand the register vectors as needed
  if (end_regno > m_registers.size ())
    m_registers.resize (end_regno);

  // m_registers_p already sized in the constructor
  if (!m_registers_p[regno])
    {
      cuda_debugapi::read_register_range (
	  dev_idx (), sm_idx (), warp_idx (), lane_idx (), start_regno,
	  end_regno - start_regno, &m_registers[start_regno]);
      // Update the predicate bits
      for (auto i = start_regno; i < end_regno; ++i)
	m_registers_p.set (i, true);
    }

  gdb_assert (m_registers_p[regno]);

  return m_registers[regno];
}

void
cuda_lane::set_register (uint32_t regno, uint32_t value)
{
  // Validate the parameters
  gdb_assert (regno < CUDA_STATE_REGISTER_RZ);
  gdb_assert (regno < warp ()->sm ()->device ()->get_num_registers ());

  cuda_debugapi::write_register (dev_idx (), sm_idx (), warp_idx (),
				 lane_idx (), regno, value);

  if (regno > m_registers.size ())
    m_registers.resize (regno);

  m_registers[regno] = value;
  m_registers_p.set (regno, true);
}

bool
cuda_lane::get_predicate (uint32_t pred)
{
  auto num_predicates = warp ()->sm ()->device ()->get_num_predicates ();
  gdb_assert (pred < num_predicates);
  gdb_assert (warp ()->lane_valid (lane_idx ()));

  // If the requested predicate isn't valid, read them all in
  if (!(m_predicates_p & (1 << pred)))
    {
      cuda_debugapi::read_predicates (dev_idx (), sm_idx (), warp_idx (),
				      lane_idx (), num_predicates,
				      m_predicates);
      m_predicates_p = (1 << num_predicates) - 1;
    }

  return m_predicates[pred] != 0;
}

void
cuda_lane::set_predicate (uint32_t pred, bool value)
{
  auto num_predicates = warp ()->sm ()->device ()->get_num_predicates ();
  gdb_assert (pred < num_predicates);
  gdb_assert (warp ()->lane_valid (lane_idx ()));

  // If we don't have them all, read them all in
  if (m_predicates_p != (num_predicates - 1))
    {
      cuda_debugapi::read_predicates (dev_idx (), sm_idx (), warp_idx (),
				      lane_idx (), num_predicates,
				      m_predicates);
      m_predicates_p = (num_predicates - 1);
    }

  m_predicates[pred] = value;
  m_predicates_p |= (1 << pred);

  cuda_debugapi::write_predicates (dev_idx (), sm_idx (), warp_idx (),
				   lane_idx (), num_predicates, m_predicates);
}

uint32_t
cuda_lane::get_cc_register ()
{
  if (!m_cc_register_p)
    {
      cuda_debugapi::read_cc_register (dev_idx (), sm_idx (), warp_idx (),
				       lane_idx (), &m_cc_register);
      m_cc_register_p = true;
    }

  return m_cc_register;
}

void
cuda_lane::set_cc_register (uint32_t value)
{
  cuda_debugapi::write_cc_register (dev_idx (), sm_idx (), warp_idx (),
				    lane_idx (), value);

  m_cc_register = value;
  m_cc_register_p = true;
}

int32_t
cuda_lane::get_call_depth ()
{
  if (m_call_depth_p)
    return (int32_t)m_call_depth;

  cuda_debugapi::read_call_depth (dev_idx (), sm_idx (), warp_idx (),
				  lane_idx (), &m_call_depth);
  m_call_depth_p = true;

  return (int32_t)m_call_depth;
}

int32_t
cuda_lane::get_syscall_call_depth ()
{
  if (m_syscall_call_depth_p)
    return m_syscall_call_depth;

  cuda_debugapi::read_syscall_call_depth (dev_idx (), sm_idx (), warp_idx (),
					  lane_idx (), &m_syscall_call_depth);
  m_syscall_call_depth_p = true;

  return (int32_t)m_syscall_call_depth;
}

uint64_t
cuda_lane::get_return_address (int32_t level)
{
  auto iter = m_return_address.find (level);
  if (iter != m_return_address.end ())
    return iter->second;

  uint64_t ra = 0;
  cuda_debugapi::read_virtual_return_address (
      dev_idx (), sm_idx (), warp_idx (), lane_idx (), level, &ra);

  m_return_address[level] = ra;

  return ra;
}

void
cuda_lane::update (const CUDBGLaneState &state)
{
  CUDA_STATE_TRACE_LANE (
      this,
      "timestamp %lu clock %lu pc 0x%lx thread (%u, %u, %u) exception %u",
      timestamp (), cuda_clock (), state.virtualPC, state.threadIdx.x,
      state.threadIdx.y, state.threadIdx.z, state.exception);

  // Clear cached data
  invalidate (!debug_invalidate);

  m_pc_p = true;
  m_pc = state.virtualPC;

  m_thread_idx_p = true;
  m_thread_idx = state.threadIdx;

  m_exception_p = true;
  m_exception = state.exception;

  set_timestamp (cuda_clock ());
}

//
// Batched device info update buffer decoding logic
//

// Convert ThreadIdx to a FlatThreadId
static uint64_t
cuda_convert_idx_to_flat_idx (const CuDim3 &thread_idx,
			      const CuDim3 &block_dim)
{
  return ((block_dim.x * block_dim.y * thread_idx.z)
	  + (block_dim.x * thread_idx.y) + thread_idx.x);
}

static CuDim3
cuda_convert_flat_idx_to_idx (uint64_t thread_id, const CuDim3 &block_dim)
{
  if ((block_dim.x == 0) || (block_dim.y == 0) || (block_dim.z == 0))
    return CuDim3{ 0, 0, 0 };

  CuDim3 thread_idx = { 0, 0, 0 };
  uint64_t current_id = thread_id;

  thread_idx.x = current_id % block_dim.x;
  current_id /= block_dim.x;

  thread_idx.y = current_id % block_dim.y;
  current_id /= block_dim.y;

  thread_idx.z = current_id % block_dim.z;

  return thread_idx;
}

void
cuda_lane::decode (const CUDBGDeviceInfoSizes &info_sizes,
		   const CUDBGDeviceInfo *dev_info, const CUDBGSMInfo *sm_info,
		   const CUDBGWarpInfo *warp_info,
		   const CUDBGGridInfo &grid_info, CUDBGException_t exception,
		   uint8_t *&ptr)
{
  // For lanes we start by invalidating the caches, and then filling
  // in the encoded data.
  invalidate (!debug_invalidate);

  // All valid lanes share the same passed-in exception
  m_exception = exception;
  m_exception_p = true;

  auto start_offset = ptr - ((uint8_t *)dev_info);

  const auto lane_info = (CUDBGLaneInfo *)ptr;
  ptr += info_sizes.laneInfoSize;

  // warpAttrbute[CUDBG_WARP_ATTRIBUTE_LANE_ATTRIBUTES] is special and
  // indicates that the lane flags are present.
  if (warp_info->warpAttributeFlags
      & (1 << CUDBG_WARP_ATTRIBUTE_LANE_ATTRIBUTES))
    {
      // Read the lane flags
      uint32_t lane_attribute_flags = *(uint32_t *)ptr;
      ptr += sizeof (uint32_t);

      CUDA_STATE_TRACE_DOMAIN_LANE (CUDA_TRACE_STATE_DECODE, this,
				    "Decoding Lane attribute flags 0x%08x",
				    lane_attribute_flags);

      // Process all lane attributes (none of which are defined yet)
      // Move past all of them - even those that we don't know about
      for (auto attr = 0; attr < 8 * sizeof (lane_attribute_flags); ++attr)
	{
	  if (lane_attribute_flags & (1 << attr))
	    {
	      // Unknown lane attribute, warn if we've not seen it before
	      if (!(unknown_lane_attribute_warning & (1 << attr)))
		{
		  CUDA_STATE_TRACE_DOMAIN_LANE (
		      CUDA_TRACE_STATE_DECODE, this,
		      "Unknown lane attribute %u (0x%08x) length %u", attr,
		      1 << attr, info_sizes.laneInfoAttributeSizes[attr]);
		  unknown_lane_attribute_warning |= 1 << attr;
		}

	      // Move past the lane attribute, even if we don't know what it
	      // was
	      ptr += info_sizes.laneInfoAttributeSizes[attr];
	    }
	}
    }

  auto flat_idx = cuda_convert_idx_to_flat_idx (warp_info->baseThreadIdx,
						grid_info.blockDim);
  m_thread_idx = cuda_convert_flat_idx_to_idx (flat_idx + lane_idx (),
					       grid_info.blockDim);
  m_thread_idx_p = true;

  // These fields are always valid, so don't need m_*_p predicates
  m_pc = lane_info->virtualPC;
  m_pc_p = true;

  set_timestamp (cuda_clock ());

  CUDA_STATE_TRACE_DOMAIN_LANE (
      CUDA_TRACE_STATE_DECODE, this,
      "Decoded Lane at offset %lu: pc 0x%lx thread (%u, %u, %u)",
      start_offset, m_pc, m_thread_idx.x, m_thread_idx.y,
      m_thread_idx.z);
}

void
cuda_warp::decode (const CUDBGDeviceInfoSizes &info_sizes,
		   const CUDBGDeviceInfo *dev_info, const CUDBGSMInfo *sm_info,
		   uint8_t *&ptr)
{
  auto start_offset = ptr - ((uint8_t *)dev_info);

  const auto warp_info = (CUDBGWarpInfo *)ptr;
  ptr += info_sizes.warpInfoSize;

  CUDA_STATE_TRACE_DOMAIN_WARP (CUDA_TRACE_STATE_DECODE, this,
				"Decoding Warp at offset %lu grid %lld valid "
				"0x%08x active 0x%08x flags 0x%08x",
				start_offset, warp_info->gridId,
				warp_info->validLanes, warp_info->activeLanes,
				warp_info->warpAttributeFlags);

  // Can be positive or negative, but not 0
  gdb_assert (warp_info->gridId != 0);

  // Clear the register caches
  m_uregisters.clear ();
  m_uregisters_p.fill (false);
  m_upredicates_p = 0;

  // Grid ID and logical block/thread coordinates
  m_grid_id = warp_info->gridId;
  m_kernel = nullptr;

  m_block_idx = warp_info->blockIdx;
  m_block_idx_p = true;

  m_base_thread_idx = warp_info->baseThreadIdx;

  // Valid / Active lane masks
  m_valid_lanes_mask = warp_info->validLanes;
  m_valid_lanes_mask_p = true;

  m_active_lanes_mask = warp_info->activeLanes;
  m_active_lanes_mask_p = true;

  // Clear optional attributes (errorPC, clusterIdx, etc)
  m_error_pc_p = true;
  m_error_pc = 0;
  m_error_pc_available = false;

  m_cluster_idx = CuDim3{ 0, 0, 0 };
  m_cluster_idx_p = false;
  m_cluster_dim = CuDim3{ 0, 0, 0 };
  m_cluster_dim_p = false;

  // Default to no exception
  CUDBGException_t exception = CUDBG_EXCEPTION_NONE;

  // Default to all lanes updated
  uint32_t updated_lane_mask
      = (1ULL << sm ()->device ()->get_num_lanes ()) - 1;

  // Scan for warp attributes and unpack them
  // Use a copy of flags so we can exit early when no more are set
  auto flags = warp_info->warpAttributeFlags;
  for (uint32_t attr = 0; (attr < 8 * sizeof (uint32_t)) && flags; ++attr)
    {
      if (flags & (1 << attr))
	{
	  flags &= ~(1 << attr);
	  switch (attr)
	    {
	      // Indicates that the lane attribute flags are present
	      // Does not extend warp_info, so the size is 0
	    case CUDBG_WARP_ATTRIBUTE_LANE_ATTRIBUTES:
	      // Nothing to do here
	      break;

	    case CUDBG_WARP_ATTRIBUTE_EXCEPTION:
	      {
		exception = (CUDBGException_t) * (uint32_t *)ptr;
		CUDA_STATE_TRACE_DOMAIN_WARP (CUDA_TRACE_STATE_DECODE, this,
					      "Decoding Warp Exception %u",
					      exception);

		// CUDBG_EXCEPTION_NONE should never be encoded
		gdb_assert (exception != CUDBG_EXCEPTION_NONE);
	      }
	      break;

	    case CUDBG_WARP_ATTRIBUTE_ERRORPC:
	      m_error_pc = *(uint64_t *)ptr;
	      m_error_pc_available = true;
	      CUDA_STATE_TRACE_DOMAIN_WARP (CUDA_TRACE_STATE_DECODE, this,
					    "Decoding Warp ERRORPC 0x%lx",
					    m_error_pc);
	      break;

	    case CUDBG_WARP_ATTRIBUTE_CLUSTERIDX:
	      m_cluster_idx = *(CuDim3 *)ptr;
	      m_cluster_idx_p = true;
	      CUDA_STATE_TRACE_DOMAIN_WARP (
		  CUDA_TRACE_STATE_DECODE, this,
		  "Decoding Warp CLUSTERIDX (%u, %u, %u)", m_cluster_idx.x,
		  m_cluster_idx.y, m_cluster_idx.z);
	      break;

	    case CUDBG_WARP_ATTRIBUTE_CLUSTERDIM:
	      m_cluster_dim = *(CuDim3 *)ptr;
	      m_cluster_dim_p = true;
	      CUDA_STATE_TRACE_DOMAIN_WARP (
		  CUDA_TRACE_STATE_DECODE, this,
		  "Decoding Warp CLUSTERDIM (%u, %u, %u)", m_cluster_dim.x,
		  m_cluster_dim.y, m_cluster_dim.z);
	      break;

	    case CUDBG_WARP_ATTRIBUTE_LANE_UPDATE_MASK:
	      updated_lane_mask = *(uint32_t *)ptr;
	      CUDA_STATE_TRACE_DOMAIN_WARP (
		  CUDA_TRACE_STATE_DECODE, this,
		  "Decoding Warp updated Lane mask 0x%08x", updated_lane_mask);
	      break;

	    case CUDBG_WARP_ATTRIBUTE_CLUSTER_EXCEPTION_TARGET_BLOCK_IDX:
	      m_cluster_exception_target_block_idx = *(CuDim3 *)ptr;
	      m_cluster_exception_target_block_idx_p = true;
	      m_cluster_exception_target_block_idx_available = true;
	      CUDA_STATE_TRACE_DOMAIN_WARP (
		  CUDA_TRACE_STATE_DECODE, this,
		  "Decoding Warp CLUSTER_EXCEPTION_TARGET_BLOCK_IDX (%u, %u, "
		  "%u)",
		  m_cluster_exception_target_block_idx.x,
		  m_cluster_exception_target_block_idx.y,
		  m_cluster_exception_target_block_idx.z);
	      break;

	      // Ignore anything we don't understand
	    default:
	      // Unknown warp attribute, warn if we've not seen it before
	      if (!(unknown_warp_attribute_warning & (1 << attr)))
		{
		  CUDA_STATE_TRACE_DOMAIN_WARP (
		      CUDA_TRACE_STATE_DECODE, this,
		      "Decoding unknown Warp attribute %u (flag 0x%08x) "
		      "length %u",
		      attr, 1 << attr,
		      info_sizes.warpInfoAttributeSizes[attr]);
		  unknown_warp_attribute_warning |= 1 << attr;
		}
	      break;
	    }
	  // Move past all specified flags, even those we don't understand
	  ptr += info_sizes.warpInfoAttributeSizes[attr];
	}
    }

  // Get the cached grid_info for this warp so that we can later reconstruct
  // the lane thread_idx values in cuda_lane::decode().
  const auto &grid_info = sm ()->device ()->get_grid_info (warp_info->gridId);

  // Decode lanes that are both updated and valid
  uint32_t needs_decoding = updated_lane_mask & warp_info->validLanes;

  // Updated but not valid means that the lane has exited and needs
  // invalidation.
  uint32_t needs_invalidating = updated_lane_mask & ~warp_info->validLanes;

  // Otherwise, nothing has changed and the lane can be ignored/skipped.
  for (uint32_t ln_id = 0; ln_id < sm ()->device ()->get_num_lanes (); ++ln_id)
    {
      // If the lane was updated - decode it
      // Else if the lane isn't in the valid mask - invalidate it
      // Else it's still valid but just not updated - leave it alone
      if (needs_decoding & (1 << ln_id))
	lane (ln_id)->decode (info_sizes, dev_info, sm_info, warp_info,
			      grid_info, exception, ptr);
      else if (needs_invalidating & (1 << ln_id))
	lane (ln_id)->invalidate (!debug_invalidate);
      else
	{
	  if (debug_skips)
	    CUDA_STATE_TRACE_DOMAIN_WARP (CUDA_TRACE_STATE_DECODE, this,
					  "Skipping lane %u", ln_id);
	}
    }

  CUDA_STATE_TRACE_DOMAIN_WARP (
      CUDA_TRACE_STATE_DECODE, this,
      "Decoded Warp at offset %lu: updated 0x%08x valid 0x%08x active 0x%08x "
      "exception %u errorpc 0x%lx",
      start_offset, updated_lane_mask, warp_info->validLanes,
      warp_info->activeLanes, exception, m_error_pc);

  // We are now valid
  set_timestamp (cuda_clock ());
}

void
cuda_sm::decode (const CUDBGDeviceInfoSizes &info_sizes,
		 const CUDBGDeviceInfo *dev_info, uint8_t *&ptr)
{
  const auto sm_info = (CUDBGSMInfo *)ptr;
  ptr += info_sizes.smInfoSize;

  m_valid_warps_mask_p = true;
  m_valid_warps_mask = { sm_info->warpValidMask };

  m_broken_warps_mask_p = true;
  m_broken_warps_mask = { sm_info->warpBrokenMask };

  // Reset optional exception tracking
  reset_sm_exception_info ();

  // The ((1ULL << n) - 1) trick for generating the mask doesn't work on
  // devices with 64 warps, so special case it.
  const auto n_warps = device ()->get_num_warps ();
  cuda_api_warpmask_t updated_warp_mask
      = { (n_warps == 8 * sizeof (uint64_t)) ? ~0ULL
					     : ((1ULL << n_warps) - 1) };

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_STATE_DECODE))
    {
      const auto valid
	  = std::string (cuda_api_mask_string (&m_valid_warps_mask));
      const auto broken
	  = std::string (cuda_api_mask_string (&m_broken_warps_mask));
      CUDA_STATE_TRACE_DOMAIN_SM (CUDA_TRACE_STATE_DECODE, this,
				  "Decoding SM at offset %lu valid warps %s "
				  "broken warps %s flags 0x%08x",
				  ptr - ((uint8_t *)dev_info), valid.c_str (),
				  broken.c_str (), sm_info->smAttributeFlags);
    }

  // Handle optional SM attribute fields
  // Use a copy of flags so we can exit early when no more are set
  auto flags = sm_info->smAttributeFlags;
  for (auto attr = 0; (attr < 8 * sizeof (sm_info->smAttributeFlags)) && flags;
       ++attr)
    {
      if (flags & (1 << attr))
	{
	  flags &= ~(1 << attr);
	  switch (attr)
	    {
	    case CUDBG_SM_ATTRIBUTE_WARP_UPDATE_MASK:
	      updated_warp_mask = cuda_api_warpmask{ *(uint64_t *)ptr };
	      CUDA_STATE_TRACE_DOMAIN_SM (
		  CUDA_TRACE_STATE_DECODE, this,
		  "Decoding updated Warp mask %s",
		  cuda_api_mask_string (&updated_warp_mask));
	      break;

	      // None yet
	    default:
	      // Unknown SM attribute, warn if we've not seen it before
	      if (!(unknown_sm_attribute_warning & (1 << attr)))
		{
		  CUDA_STATE_TRACE_DOMAIN_SM (
		      CUDA_TRACE_STATE_DECODE, this,
		      "Unknown SM attribute %u (0x%08x) length %u", attr,
		      1 << attr, info_sizes.smInfoAttributeSizes[attr]);
		  unknown_sm_attribute_warning |= 1 << attr;
		}
	      break;
	    }
	  // Move past all specified flags, even those we don't understand
	  ptr += info_sizes.smInfoAttributeSizes[attr];
	}
    }

  // Need to decode warps that are (updated & valid)
  cuda_api_warpmask needs_decoding = { 0 };
  cuda_api_and_mask (&needs_decoding, &updated_warp_mask, &m_valid_warps_mask);

  // Need to invalidate warps that are (updated & ~valid)
  cuda_api_warpmask needs_invalidating = { 0 };
  cuda_api_not_mask (&needs_invalidating, &m_valid_warps_mask);
  cuda_api_and_mask (&needs_invalidating, &needs_invalidating,
		     &updated_warp_mask);

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_STATE_DECODE))
    {
      const auto valid
	  = std::string (cuda_api_mask_string (&m_valid_warps_mask));
      const auto updated
	  = std::string (cuda_api_mask_string (&updated_warp_mask));
      const auto broken
	  = std::string (cuda_api_mask_string (&m_broken_warps_mask));
      const auto decoding
	  = std::string (cuda_api_mask_string (&needs_decoding));
      const auto invalidating
	  = std::string (cuda_api_mask_string (&needs_invalidating));
      CUDA_STATE_TRACE_DOMAIN_SM (CUDA_TRACE_STATE_DECODE, this,
				  "Decoding SM warps: valid %s updated %s "
				  "broken %s decoding %s invalidate %s",
				  valid.c_str (), updated.c_str (),
				  broken.c_str (), decoding.c_str (),
				  invalidating.c_str ());
    }

  // All other warps can be left alone
  for (uint32_t wp_id = 0; wp_id < device ()->get_num_warps (); ++wp_id)
    {
      if (cuda_api_get_bit (&needs_decoding, wp_id))
	warp (wp_id)->decode (info_sizes, dev_info, sm_info, ptr);
      else if (cuda_api_get_bit (&needs_invalidating, wp_id))
	warp (wp_id)->invalidate (!debug_invalidate, true);
      else
	{
	  if (debug_skips)
	    CUDA_STATE_TRACE_DOMAIN_SM (CUDA_TRACE_STATE_DECODE, this,
					"Skipping warp %u", wp_id);
	}
    }

  // We are now up to date
  set_timestamp (cuda_clock ());
}

void
cuda_device::decode (const CUDBGDeviceInfoSizes &info_sizes, uint8_t *buffer,
		     uint32_t &length)
{
  CUDA_STATE_TRACE_DOMAIN_DEV (CUDA_TRACE_STATE_DECODE, this,
			       "Device info size %u", length);

  // Start at the beginning of the buffer
  auto ptr = buffer;

  const auto dev_info = (CUDBGDeviceInfo *)ptr;
  ptr += info_sizes.deviceInfoSize;

  CUDA_STATE_TRACE_DOMAIN_DEV (
      CUDA_TRACE_STATE_DECODE, this,
      "Decoding Device at offset 0 responseType %u flags 0x%08x",
      dev_info->responseType, dev_info->deviceAttributeFlags);

  // Invalidate top-level datastructures in cuda_device.
  // No need to recurse, we'll handle updating SM/Warp/Lane
  // information below
  invalidate (!debug_invalidate, false);

  // Invalidate any kernels on the device
  cuda_state::device_invalidate_kernels (dev_idx ());

  // Set the SM mask defaults
  m_sm_active_mask.fill (false);
  m_sm_active_mask_p = true;

  m_sm_exception_mask.fill (false);
  m_sm_exception_mask_p = true;

  cuda_bitset sm_update_mask (
      m_num_sms, true,
      m_info_sizes
	  .deviceInfoAttributeSizes[CUDBG_DEVICE_ATTRIBUTE_SM_UPDATE_MASK]);

  // Handle optional Device attribute fields
  // Use a copy of flags so we can exit early when no more are set
  auto flags = dev_info->deviceAttributeFlags;
  for (auto attr = 0;
       (attr < 8 * sizeof (dev_info->deviceAttributeFlags)) && flags; ++attr)
    {
      if (flags & (1 << attr))
	{
	  flags &= ~(1 << attr);
	  switch (attr)
	    {
	    case CUDBG_DEVICE_ATTRIBUTE_SM_ACTIVE_MASK:
	      m_sm_active_mask.read (
		  ptr, info_sizes.deviceInfoAttributeSizes[attr]);
	      break;

	    case CUDBG_DEVICE_ATTRIBUTE_SM_UPDATE_MASK:
	      sm_update_mask.read (ptr,
				   info_sizes.deviceInfoAttributeSizes[attr]);
	      break;

	    case CUDBG_DEVICE_ATTRIBUTE_SM_EXCEPTION_MASK:
	      m_sm_exception_mask.read (
		  ptr, info_sizes.deviceInfoAttributeSizes[attr]);
	      break;

	    default:
	      // Unknown Device attribute, warn if we've not seen it before
	      if (!(unknown_device_attribute_warning & (1 << attr)))
		{
		  CUDA_STATE_TRACE_DOMAIN_DEV (
		      CUDA_TRACE_STATE_DECODE, this,
		      "Unknown Device attribute %u (0x%08x) length %u", attr,
		      1 << attr, info_sizes.deviceInfoAttributeSizes[attr]);
		  unknown_device_attribute_warning |= 1 << attr;
		}
	      break;
	    }
	  // Move past all optional attributes, even those we don't know about
	  ptr += info_sizes.deviceInfoAttributeSizes[attr];
	}
    }

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_STATE_DECODE))
    {
      auto active_mask = m_sm_active_mask.to_hex_string ();
      CUDA_STATE_TRACE_DOMAIN_DEV (CUDA_TRACE_STATE_DECODE, this,
				   "SM active mask %s", active_mask.c_str ());
      auto update_mask = sm_update_mask.to_hex_string ();
      CUDA_STATE_TRACE_DOMAIN_DEV (CUDA_TRACE_STATE_DECODE, this,
				   "SM updated mask %s", update_mask.c_str ());
      auto exception_mask = m_sm_exception_mask.to_hex_string ();
      CUDA_STATE_TRACE_DOMAIN_DEV (CUDA_TRACE_STATE_DECODE, this,
				   "SM exception mask %s",
				   exception_mask.c_str ());
    }

  // Now decode/invalidate/skip the SMs as needed
  for (uint32_t sm_id = 0; sm_id < get_num_sms (); ++sm_id)
    {
      // If the SM has been updated - encode it
      // Else if it's not active - invalidate it
      // Otherwise it's not changed - leave it alone
      if (sm_update_mask.get (sm_id))
	{
	  if (m_sm_active_mask.get (sm_id))
	    sm (sm_id)->decode (info_sizes, dev_info, ptr);
	  else
	    sm (sm_id)->invalidate (!debug_invalidate, true);
	}
      else
	{
	  if (debug_skips)
	    CUDA_STATE_TRACE_DOMAIN_SM (
		CUDA_TRACE_STATE_DECODE, sm (sm_id), "Skipping %s SM",
		m_sm_active_mask.get (sm_id) ? "active" : "inactive");
	}
    }

  const auto size = (uint32_t)(ptr - buffer);
  CUDA_STATE_TRACE_DOMAIN_DEV (CUDA_TRACE_STATE_DECODE, this,
			       "Decoded size %u of returned %u buffer size %u",
			       size, length, info_sizes.requiredBufferSize);

  // Final check that we parsed everything we were given
  gdb_assert (size == length);
}
