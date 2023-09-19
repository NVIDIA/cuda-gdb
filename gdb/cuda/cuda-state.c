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

#include "cuda-defs.h"
#include "cuda-asm.h"
#include "cuda-state.h"

/* GPU register cache */
#define CUDBG_CACHED_REGISTERS_COUNT  256
#define CUDBG_CACHED_PREDICATES_COUNT 8

/* GPU uniform register cache */
#define CUDBG_CACHED_UREGISTERS_COUNT  64
#define CUDBG_CACHED_UPREDICATES_COUNT 8

typedef struct {
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint32_t registers[CUDBG_CACHED_REGISTERS_COUNT];
  uint32_t register_valid_mask[CUDBG_CACHED_REGISTERS_COUNT>>5];
  uint32_t predicates[CUDBG_CACHED_PREDICATES_COUNT];
  bool	   predicates_valid_p;
  uint32_t cc_register;
  bool	   cc_register_valid_p;
} cuda_reg_cache_element_t;

typedef struct {
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t registers[CUDBG_CACHED_UREGISTERS_COUNT];
  uint32_t register_valid_mask[CUDBG_CACHED_UREGISTERS_COUNT>>5];
  uint32_t predicates[CUDBG_CACHED_UPREDICATES_COUNT];
  bool	   predicates_valid_p;
} cuda_ureg_cache_element_t;

const bool CACHED = true;

static bool cuda_state_debug = false;

static void device_cleanup_contexts	  (uint32_t dev_id);
static void device_flush_disasm_cache	  (uint32_t dev_id);
static void device_update_exception_state (uint32_t dev_id);
static void sm_set_exception_none	  (uint32_t dev_id, uint32_t sm_id);
static void warp_invalidate		  (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
static void lane_set_exception_none	  (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);

static std::vector<cuda_reg_cache_element_t> cuda_register_cache;
static std::vector<cuda_ureg_cache_element_t> cuda_uregister_cache;


static inline sm_state *
sm_get (uint32_t dev_id, uint32_t sm_id)
{
  return cuda_state::instance ().device (dev_id)->sm (sm_id);
}

static inline warp_state *
warp_get (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  return sm_get (dev_id, sm_id)->warp (wp_id);
}

static inline lane_state *
lane_get (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  return warp_get(dev_id, sm_id, wp_id)->lane (ln_id);
}

/******************************************************************************
 *
 *				    System
 *
 ******************************************************************************/

static struct
{
  uint32_t device_ctor_count;
  uint32_t device_dtor_count;
  uint32_t sm_ctor_count;
  uint32_t sm_dtor_count;
  uint32_t warp_ctor_count;
  uint32_t warp_dtor_count;
  uint32_t lane_ctor_count;
  uint32_t lane_default_ctor_count;
  uint32_t lane_dtor_count;
} cuda_state_stats;

cuda_state cuda_state::m_instance {};

cuda_state::cuda_state ()
{
}

void
cuda_state::reset (void)
{
  m_num_devices_p = false;
  m_num_devices = 0;
  m_suspended_devices_mask = 0;

  /* Automatically deletes the device_state objects */
  m_devs.clear ();

  if (cuda_state_debug)
    {
      printf_unfiltered ("device_state ctor %u\n", cuda_state_stats.device_ctor_count);
      printf_unfiltered ("device_state dtor %u\n", cuda_state_stats.device_dtor_count);
      printf_unfiltered ("sm_state     ctor %u\n", cuda_state_stats.sm_ctor_count);
      printf_unfiltered ("sm_state     dtor %u\n", cuda_state_stats.sm_dtor_count);
      printf_unfiltered ("warp_state   ctor %u\n", cuda_state_stats.warp_ctor_count);
      printf_unfiltered ("warp_state   dtor %u\n", cuda_state_stats.warp_dtor_count);
      printf_unfiltered ("lane_state   ctor %u\n", cuda_state_stats.lane_ctor_count);
      printf_unfiltered ("lane_state   ctor %u (default)\n",
			 cuda_state_stats.lane_default_ctor_count);
      printf_unfiltered ("lane_state   dtor %u\n", cuda_state_stats.lane_dtor_count);
    }

  memset (&cuda_state_stats, 0, sizeof(cuda_state_stats));
}

void
cuda_state::initialize (void)
{
  cuda_trace ("system: initialize");
  gdb_assert (cuda_initialized);

  reset ();

  for (uint32_t dev_id = 0; dev_id < get_num_devices (); ++dev_id)
    m_devs.emplace_back (new device_state (dev_id));

  cuda_options_force_set_launch_notification_update ();
}

void
cuda_system_initialize (void)
{
  cuda_state::instance ().initialize ();
}

void
cuda_state::finalize (void)
{
  cuda_trace ("system: finalize");
  gdb_assert (cuda_initialized);

  reset ();
}

void
cuda_system_finalize (void)
{
  cuda_state::instance ().finalize ();
}

uint32_t
cuda_state::get_num_devices (void)
{
  if (!cuda_initialized)
    return 0;

  if (m_num_devices_p)
    return m_num_devices;

  cuda_api_get_num_devices (&m_num_devices);
  gdb_assert (m_num_devices <= CUDBG_MAX_DEVICES);
  m_num_devices_p = CACHED;

  return m_num_devices;
}

uint32_t
cuda_system_get_num_devices (void)
{
  return cuda_state::instance ().get_num_devices ();
}

void
cuda_state::invalidate_kernels ()
{
  cuda_trace ("invalidate kernels");

  for (auto kernel = kernels_get_first_kernel (); kernel; kernel = kernels_get_next_kernel (kernel))
    kernel_invalidate (kernel);
}

uint32_t
cuda_state::get_num_present_kernels (void)
{
  uint32_t num_present_kernel = 0;

  if (!cuda_initialized)
    return 0;

  for (auto kernel = kernels_get_first_kernel (); kernel; kernel = kernels_get_next_kernel (kernel))
    if (kernel_is_present (kernel))
      ++num_present_kernel;

  return num_present_kernel;
}

uint32_t
cuda_system_get_num_present_kernels (void)
{
  return cuda_state::instance ().get_num_present_kernels ();
}

void
cuda_state::resolve_breakpoints (int bp_number_from)
{
  cuda_trace ("system: resolve breakpoints\n");

  elf_image_t elf_image;
  CUDA_ALL_LOADED_ELF_IMAGES (elf_image)
    cuda_resolve_breakpoints (bp_number_from, elf_image);
}

void
cuda_system_resolve_breakpoints (int bp_number_from)
{
  cuda_state::instance ().resolve_breakpoints (bp_number_from);
}

void
cuda_state::cleanup_breakpoints (void)
{
  cuda_trace ("system: clean up breakpoints");

  elf_image_t elf_image;
  CUDA_ALL_LOADED_ELF_IMAGES (elf_image)
    cuda_unresolve_breakpoints (elf_image);
}

void
cuda_system_cleanup_breakpoints (void)
{
  cuda_state::instance ().cleanup_breakpoints ();
}

void
cuda_state::cleanup_contexts (void)
{
  cuda_trace ("system: clean up contexts");

  for (uint32_t dev_id = 0; dev_id < cuda_system_get_num_devices (); ++dev_id)
    device_cleanup_contexts (dev_id);
}

void
cuda_system_cleanup_contexts (void)
{
  cuda_state::instance ().cleanup_contexts ();
}

void
cuda_state::flush_disasm_cache (void)
{
  cuda_trace ("system: flush disassembly cache");

  for (uint32_t dev_id = 0; dev_id < cuda_system_get_num_devices (); ++dev_id)
    device_flush_disasm_cache (dev_id);
}

void
cuda_system_flush_disasm_cache (void)
{
  cuda_state::instance ().flush_disasm_cache ();
}

bool
cuda_state::is_broken (cuda_coords_t &coords)
{
  cuda_iterator itr;
  cuda_coords_t filter = CUDA_WILDCARD_COORDS;
  bool broken = false;

  itr = cuda_iterator_create (CUDA_ITERATOR_TYPE_LANES, &filter,
		(cuda_select_t)(CUDA_SELECT_VALID | CUDA_SELECT_TRAP |
				CUDA_SELECT_CURRENT_CLOCK | CUDA_SELECT_SNGL));

  cuda_iterator_start (itr);
  if (!cuda_iterator_end (itr))
    {
      coords = cuda_iterator_get_current (itr);
      broken = true;
    }

  cuda_iterator_destroy (itr);

  return broken;
}

bool
cuda_system_is_broken (cuda_coords_t &coords)
{
  return cuda_state::instance ().is_broken (coords);
}

uint32_t
cuda_state::get_suspended_devices_mask (void)
{
  return m_suspended_devices_mask;
}

uint32_t
cuda_system_get_suspended_devices_mask (void)
{
  return cuda_state::instance ().get_suspended_devices_mask ();
}

context_t
cuda_state::find_context_by_addr (CORE_ADDR addr)
{
  for (uint32_t dev_id = 0; dev_id < cuda_system_get_num_devices (); ++dev_id)
    {
      context_t context = device_find_context_by_addr (dev_id, addr);
      if (context)
	return context;
    }

  return nullptr;
}

context_t
cuda_system_find_context_by_addr (CORE_ADDR addr)
{
  return cuda_state::instance ().find_context_by_addr (addr);
}

/******************************************************************************
 *
 *				    Device
 *
 ******************************************************************************/

device_state::device_state (uint32_t dev_idx)
: m_dev_idx (dev_idx)
{
  ++cuda_state_stats.device_ctor_count;

  cuda_trace ("device %u: initialize", m_dev_idx);

  uint32_t n_sms = get_num_sms ();

  m_sms.reserve (n_sms);
  for (uint32_t sm_idx = 0; sm_idx < n_sms; ++sm_idx)
    m_sms.emplace_back (new sm_state (this, sm_idx));

  m_contexts = contexts_new ();
}

device_state::~device_state ()
{
  ++cuda_state_stats.device_dtor_count;
}

uint32_t device_state::get_num_sms ()
{
  if (m_num_sms_p)
    return m_num_sms;

  cuda_api_get_num_sms (m_dev_idx, &m_num_sms);
  gdb_assert (m_num_sms <= CUDBG_MAX_SMS);
  m_num_sms_p = CACHED;

  return m_num_sms;
}

uint32_t
device_get_num_sms (uint32_t dev_id)
{
  return cuda_state::instance ().device (dev_id)->get_num_sms ();
}

uint32_t
device_state::get_num_warps ()
{
  if (m_num_warps_p)
    return m_num_warps;

  cuda_api_get_num_warps (m_dev_idx, &m_num_warps);
  gdb_assert (m_num_warps <= CUDBG_MAX_WARPS);
  m_num_warps_p = CACHED;

  return m_num_warps;
}

uint32_t
device_get_num_warps (uint32_t dev_id)
{
  return cuda_state::instance ().device (dev_id)->get_num_warps ();
}

uint32_t
device_state::get_num_lanes ()
{
  if (m_num_lanes_p)
    return m_num_lanes;

  cuda_api_get_num_lanes (m_dev_idx, &m_num_lanes);
  gdb_assert (m_num_lanes <= CUDBG_MAX_LANES);
  m_num_lanes_p = CACHED;

  return m_num_lanes;
}

uint32_t
device_get_num_lanes (uint32_t dev_id)
{
  return cuda_state::instance ().device (dev_id)->get_num_lanes ();
}

void
device_state::invalidate ()
{
  cuda_trace ("device %u: invalidate", number ());

  for (auto& sm : m_sms)
    sm.get ()->invalidate (true);

  cuda_state::instance ().invalidate_kernels ();

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
device_invalidate (uint32_t dev_id)
{
  cuda_state::instance ().device (dev_id)->invalidate ();
}

static void
device_flush_disasm_cache (uint32_t dev_id)
{
  cuda_trace ("device %u: flush disassembly caches", dev_id);

  gdb_assert (dev_id < cuda_system_get_num_devices ());

  /* This is less than ideal, we want to iternate on modules, not on kernels */
  for (auto kernel = kernels_get_first_kernel (); kernel; kernel = kernels_get_next_kernel (kernel))
    {
      module_t module = kernel_get_module (kernel);
      module->disassembler->flush_device_cache ();
    }
}

static void
device_cleanup_contexts (uint32_t dev_id)
{
  cuda_trace ("device %u: clean up contexts", dev_id);

  contexts_delete (device_get_contexts (dev_id));

  auto dev = cuda_state::instance ().device (dev_id);
  dev->m_contexts = nullptr;
}

const char*
device_get_device_type (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_dev_type_p)
    return dev->m_dev_type;

  cuda_api_get_device_type (dev_id, dev->m_dev_type, sizeof(dev->m_dev_type));
  dev->m_dev_type_p = CACHED;

  return dev->m_dev_type;
}

const char*
device_get_sm_type (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  if (!strlen (dev->m_sm_type))
    cuda_api_get_sm_type (dev_id, dev->m_sm_type, sizeof (dev->m_sm_type));

  return dev->m_sm_type;
}

uint32_t
device_get_sm_version (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_sm_version_p)
    return dev->m_sm_version;

  auto sm_type = device_get_sm_type (dev_id);
  if (strlen (sm_type) < 4 || strncmp (sm_type, "sm_", 3) != 0)
    error ("unknown sm_type %s", sm_type);

  dev->m_sm_version = atoi (&sm_type[3]);

  dev->m_sm_version_p = CACHED;
  return dev->m_sm_version;
}

const char *
device_get_device_name (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_dev_name_p)
    return dev->m_dev_name;

  cuda_api_get_device_name (dev_id, dev->m_dev_name, sizeof (dev->m_dev_name));
  dev->m_dev_name_p = CACHED;
  return dev->m_dev_name;
}

/* This assumes that the GPU architecture has a uniform instruction size,
 * which is true on all GPU architectures except FERMI. Since cuda-gdb no
 * longer supports FERMI as of 9.0 toolkit, this assumption is valid.
 */
uint32_t
device_get_insn_size (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_insn_size_p)
    return dev->m_insn_size;

  auto sm_version = device_get_sm_version (dev_id);
  dev->m_insn_size = (sm_version < 70) ? 8 : 16;
  dev->m_insn_size_p = CACHED;

  return dev->m_insn_size;
}

uint32_t
device_get_pci_bus_id (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_pci_bus_info_p)
    return dev->m_pci_bus_id;

  cuda_api_get_device_pci_bus_info (dev_id, &dev->m_pci_bus_id, &dev->m_pci_dev_id);
  dev->m_pci_bus_info_p = CACHED;

  return dev->m_pci_bus_id;
}

uint32_t
device_get_pci_dev_id (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_pci_bus_info_p)
    return dev->m_pci_dev_id;

  cuda_api_get_device_pci_bus_info (dev_id, &dev->m_pci_bus_id, &dev->m_pci_dev_id);
  gdb_assert (dev->m_num_sms <= CUDBG_MAX_SMS);
  dev->m_pci_bus_info_p = CACHED;

  return dev->m_pci_dev_id;
}

uint32_t
device_get_num_registers (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_num_registers_p)
    return dev->m_num_registers;

  cuda_api_get_num_registers (dev_id, &dev->m_num_registers);
  dev->m_num_registers_p = CACHED;

  return dev->m_num_registers;
}

uint32_t
device_get_num_predicates (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_num_predicates_p)
    return dev->m_num_predicates;

  cuda_api_get_num_predicates (dev_id, &dev->m_num_predicates);
  dev->m_num_predicates_p = CACHED;
  gdb_assert (dev->m_num_predicates <= CUDBG_CACHED_PREDICATES_COUNT);

  return dev->m_num_predicates;
}

uint32_t
device_get_num_uregisters (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_num_uregisters_p)
    return dev->m_num_uregisters;

  cuda_api_get_num_uregisters (dev_id, &dev->m_num_uregisters);
  dev->m_num_uregisters_p = CACHED;

  return dev->m_num_uregisters;
}

uint32_t
device_get_num_upredicates (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_num_upredicates_p)
    return dev->m_num_upredicates;

  cuda_api_get_num_upredicates (dev_id, &dev->m_num_upredicates);
  dev->m_num_upredicates_p = CACHED;
  gdb_assert (dev->m_num_upredicates <= CUDBG_CACHED_UPREDICATES_COUNT);

  return dev->m_num_upredicates;
}

uint32_t
device_get_num_kernels (uint32_t dev_id)
{
  kernel_t kernel;
  uint32_t num_kernels = 0;

  gdb_assert (dev_id < cuda_system_get_num_devices ());

  for (kernel = kernels_get_first_kernel (); kernel; kernel = kernels_get_next_kernel (kernel))
    if (kernel_get_dev_id (kernel) == dev_id)
      ++num_kernels;

  return num_kernels;
}

bool
device_is_any_context_present (uint32_t dev_id)
{
  contexts_t contexts;

  gdb_assert (dev_id < cuda_system_get_num_devices ());

  contexts = device_get_contexts (dev_id);

  return contexts_is_any_context_present (contexts);
}

bool
device_is_active_context (uint32_t dev_id, context_t context)
{
  contexts_t contexts;

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  contexts = device_get_contexts (dev_id);

  return contexts_is_active_context (contexts, context);
}

bool
device_is_valid (uint32_t dev_id)
{
  uint32_t sm, wp;

  if (!cuda_initialized)
    return false;

  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_valid_p)
    return dev->m_valid;

  dev->m_valid = false;

  if (!device_is_any_context_present (dev_id))
    return dev->m_valid;

  for (sm = 0; sm < device_get_num_sms (dev_id) && !dev->m_valid; ++sm)
    for (wp = 0; wp < device_get_num_warps (dev_id) && !dev->m_valid; ++wp)
      if (warp_is_valid (dev_id, sm, wp))
	  dev->m_valid = true;

  dev->m_valid_p = CACHED;
  return dev->m_valid;
}

bool
device_has_exception (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  device_update_exception_state (dev_id);

  return dev->m_sm_exception_mask != 0;
}

void
device_get_active_sms_mask (uint32_t dev_id, std::bitset<CUDBG_MAX_SMS> &mask)
{
  mask.reset ();

  /* For every sm */
  for (uint32_t sm = 0; sm < device_get_num_sms (dev_id); ++sm)
    {
      /* For every warp in the sm */
      for (uint32_t wp = 0; wp < device_get_num_warps (dev_id); ++wp)
	{
	  /* Set to true if there is a valid warp in the sm */
	  if (warp_is_valid (dev_id, sm, wp))
	    {
	      mask.set (sm);
	      break;
	    }
	}
    }
}

contexts_t
device_get_contexts (uint32_t dev_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  gdb_assert (dev->m_contexts);

  return dev->m_contexts;
}

context_t
device_find_context_by_id (uint32_t dev_id, uint64_t context_id)
{
  contexts_t	  contexts = device_get_contexts (dev_id);

  return contexts_find_context_by_id (contexts, context_id);
}

context_t
device_find_context_by_addr (uint32_t dev_id, CORE_ADDR addr)
{
  contexts_t	  contexts = device_get_contexts (dev_id);

  return contexts_find_context_by_address (contexts, addr);
}

void
device_print (uint32_t dev_id)
{
  contexts_t	  contexts;

  cuda_trace ("device %u:", dev_id);

  contexts = device_get_contexts (dev_id);

  contexts_print (contexts);
}

void
device_state::resume ()
{
  cuda_trace ("device %u: resume", m_dev_idx);

  if (!m_suspended)
    return;
  
  invalidate ();

  cuda_api_resume_device (m_dev_idx);

  m_suspended = false;

  cuda_state::instance ().m_suspended_devices_mask &= ~(1 << m_dev_idx);
}

void
device_resume (uint32_t dev_id)
{
  cuda_state::instance ().device (dev_id)->resume ();
}

bool
device_suspended (uint32_t dev_id)
{
  return cuda_state::instance ().device (dev_id)->m_suspended;
}

static void
device_create_kernel(uint32_t dev_id, uint64_t grid_id)
{
  CUDBGGridInfo gridInfo = {0};

  cuda_api_get_grid_info(dev_id, grid_id, &gridInfo);
  kernels_start_kernel(dev_id, grid_id,
		       gridInfo.functionEntry,
		       gridInfo.context,
		       gridInfo.module,
		       gridInfo.gridDim,
		       gridInfo.blockDim,
		       gridInfo.type,
		       gridInfo.parentGridId,
		       gridInfo.origin,
		       true,
		       gridInfo.clusterDim);
}

void
device_state::suspend ()
{
  cuda_trace ("device %u: suspend", m_dev_idx);

  cuda_api_suspend_device (m_dev_idx);

  m_suspended = true;

  cuda_state::instance ().m_suspended_devices_mask |= (1 << m_dev_idx);
}

void
device_suspend (uint32_t dev_idx)
{
  cuda_state::instance ().device (dev_idx)->suspend ();
}

static void
device_update_exception_state (uint32_t dev_id)
{
  uint32_t sm_id;
  uint32_t nsms;

  auto dev = cuda_state::instance ().device (dev_id);

  if (dev->m_sm_exception_mask_valid_p)
    return;

  memset(&dev->m_sm_exception_mask, 0, sizeof(dev->m_sm_exception_mask));
  nsms = device_get_num_sms (dev_id);

  if (device_is_any_context_present (dev_id))
    cuda_api_read_device_exception_state (dev_id, dev->m_sm_exception_mask, (nsms+63) / 64);

  for (sm_id = 0; sm_id < nsms; ++sm_id)
    if (!((dev->m_sm_exception_mask[sm_id / 64] >> (sm_id % 64)) & 1))
      sm_set_exception_none (dev_id, sm_id);

  dev->m_sm_exception_mask_valid_p = true;
}

void
cuda_system_set_device_spec (uint32_t dev_id, uint32_t num_sms,
			     uint32_t num_warps, uint32_t num_lanes,
			     uint32_t num_registers,
			     const char *dev_type, const char *sm_type)
{
  auto dev = cuda_state::instance ().device (dev_id);

  gdb_assert (cuda_remote);
  gdb_assert (num_sms <= CUDBG_MAX_SMS);
  gdb_assert (num_warps <= CUDBG_MAX_WARPS);
  gdb_assert (num_lanes <= CUDBG_MAX_LANES);

  dev->m_num_sms	= num_sms;
  dev->m_num_warps	= num_warps;
  dev->m_num_lanes	= num_lanes;
  dev->m_num_registers	= num_registers;
  strcpy (dev->m_dev_type, dev_type);
  strcpy (dev->m_sm_type, sm_type);

  if (strlen (dev->m_sm_type) < 4 || strncmp (dev->m_sm_type, "sm_", 3) != 0)
    error ("unknown sm_type %s", dev->m_sm_type);
  dev->m_sm_version = atoi (&dev->m_sm_type[3]);
  dev->m_insn_size = (dev->m_sm_version < 70) ? 8 : 16;

  dev->m_num_sms_p	 = CACHED;
  dev->m_num_warps_p	 = CACHED;
  dev->m_num_lanes_p	 = CACHED;
  dev->m_num_registers_p = CACHED;
  dev->m_dev_type_p	 = CACHED;
  dev->m_dev_name_p	 = CACHED;
  dev->m_num_predicates_p = false;
  dev->m_sm_version_p	 = CACHED;
  dev->m_insn_size_p	 = CACHED;
}

/******************************************************************************
 *
 *				      SM
 *
 ******************************************************************************/

sm_state::sm_state (device_state *dev, uint32_t sm_idx)
: m_sm_idx (sm_idx), m_device (dev)
{
  ++cuda_state_stats.sm_ctor_count;
  
  m_warps.clear ();
  for (uint32_t idx = 0; idx < device ()->get_num_warps (); ++idx)
    m_warps.emplace_back (new warp_state (this, idx));
}

sm_state::~sm_state ()
{
  ++cuda_state_stats.sm_dtor_count;
}

void
sm_state::invalidate (bool recurse)
{
  if (recurse)
    for (auto& warp : m_warps)
      warp.get ()->invalidate ();
  
  m_device->m_sm_exception_mask_valid_p = false;

  m_valid_warps_mask_p	= false;
  m_broken_warps_mask_p = false;
}

bool
sm_is_valid (uint32_t dev_id, uint32_t sm_id)
{
  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));

  return sm_get_valid_warps_mask (dev_id, sm_id);
}

bool
sm_has_exception (uint32_t dev_id, uint32_t sm_id)
{
  auto dev = cuda_state::instance ().device (dev_id);

  gdb_assert (sm_id < device_get_num_sms (dev_id));

  device_update_exception_state (dev_id);

  return (dev->m_sm_exception_mask[sm_id / 64] >> (sm_id % 64)) & 1ULL;
}

cuda_api_warpmask*
sm_get_valid_warps_mask (uint32_t dev_id, uint32_t sm_id)
{
  auto sm = sm_get (dev_id, sm_id);

  if (!sm->m_valid_warps_mask_p) {
      cuda_api_read_valid_warps (dev_id, sm_id, &sm->m_valid_warps_mask);
      sm->m_valid_warps_mask_p = CACHED;
  }

  return &sm->m_valid_warps_mask;
}

cuda_api_warpmask*
sm_get_broken_warps_mask (uint32_t dev_id, uint32_t sm_id)
{
  auto sm = sm_get (dev_id, sm_id);

  if (!sm->m_broken_warps_mask_p) {
      cuda_api_read_broken_warps (dev_id, sm_id, &sm->m_broken_warps_mask);
      sm->m_broken_warps_mask_p = CACHED;
  }

  return &sm->m_broken_warps_mask;
}

static void
sm_set_exception_none (uint32_t dev_id, uint32_t sm_id)
{
  uint32_t wp_id;
  uint32_t ln_id;

  for (wp_id = 0; wp_id < device_get_num_warps (dev_id); ++wp_id)
    for (ln_id = 0; ln_id < device_get_num_lanes (dev_id); ++ln_id)
      lane_set_exception_none (dev_id, sm_id, wp_id, ln_id);
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
  for (auto it = cuda_uregister_cache.begin (); it != cuda_uregister_cache.end (); ++it)
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
cuda_ureg_cache_remove_element (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  auto it = cuda_uregister_cache.begin ();
  for (; it != cuda_uregister_cache.end (); ++it)
    if (it->dev == dev_id && it->sm == sm_id && it->wp == wp_id)
      break;

  if (it != cuda_uregister_cache.end ())
    cuda_uregister_cache.erase (it);
}

uint32_t
warp_get_uregister (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t regno)
{
  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));

  auto elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);
  if ( (elem->register_valid_mask[regno>>5]&(1UL<<(regno&31))) != 0)
    return elem->registers[regno];

  cuda_api_read_uregister_range (dev_id, sm_id, wp_id,
				 0, CUDBG_CACHED_UREGISTERS_COUNT, elem->registers);
  elem->register_valid_mask[0] = 0xffffffff;
  elem->register_valid_mask[1] = 0xffffffff;

  return elem->registers[regno];
}

void
warp_set_uregister (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t regno, uint32_t value)
{
  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));

  cuda_api_write_uregister (dev_id, sm_id, wp_id, regno, value);

  /* If register can not be cached - return */
  if (regno > CUDBG_CACHED_UREGISTERS_COUNT)
      return;

  auto elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);
  elem->registers[regno] = value;
  elem->register_valid_mask[regno>>5] |= 1UL << (regno & 31);
}

bool
warp_get_upredicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t predicate)
{
  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));
  auto elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);

  if (elem->predicates_valid_p)
    return elem->predicates[predicate] != 0;

  cuda_api_read_upredicates (dev_id, sm_id, wp_id,
			     device_get_num_upredicates (dev_id),
			     elem->predicates);
  elem->predicates_valid_p = CACHED;

  return elem->predicates[predicate] != 0;
}

void
warp_set_upredicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t predicate, bool value)
{
  gdb_assert (predicate < device_get_num_upredicates (dev_id));
  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));
  auto elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);

  if (!elem->predicates_valid_p)
    {
      cuda_api_read_upredicates (dev_id, sm_id, wp_id,
				device_get_num_upredicates (dev_id),
				elem->predicates);
      elem->predicates_valid_p = CACHED;
    }

  elem->predicates[predicate] = value;

  cuda_api_write_upredicates (dev_id, sm_id, wp_id,
			      device_get_num_upredicates (dev_id),
			      elem->predicates);
}

warp_state::warp_state (sm_state *parent_sm, uint32_t warp_idx)
  : m_warp_idx (warp_idx), m_sm (parent_sm), m_lanes()
{
  ++cuda_state_stats.warp_ctor_count;

  const auto n_lanes = sm ()->device ()->get_num_lanes ();
  for (uint32_t idx = 0; idx < n_lanes; ++idx)
    lane (idx)->configure (this, idx);
}

warp_state::~warp_state ()
{
  ++cuda_state_stats.warp_dtor_count;
}

void
warp_state::invalidate()
{
  cuda_ureg_cache_remove_element (sm ()->device ()->number (), sm ()->number (), number ());

  for (auto& ln : m_lanes)
    ln.invalidate ();
 
  // XXX decouple the masks from the SM state data structure to avoid this
  // little hack.
  /* If a warp is invalidated, we have to invalidate the warp masks in the
     corresponding SM-> */
  sm ()->invalidate (false);

  m_valid_p		= false;
  m_broken_p		= false;
  m_block_idx_p		= false;
  m_cluster_idx_p	= false;
  m_kernel_p		= false;
  m_grid_id_p		= false;
  m_valid_lanes_mask_p	= false;
  m_active_lanes_mask_p = false;
  m_timestamp_p		= false;
}

static void
warp_invalidate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  cuda_state::instance ().device (dev_id)->sm (sm_id)->warp (wp_id)->invalidate ();
}

bool
warps_resume_until (uint32_t dev_id, uint32_t sm_id, cuda_api_warpmask* mask, uint64_t pc)
{
  uint32_t i;

  /* No point in resuming warps, if one them is already there */
  for (i = 0; i < device_get_num_warps (dev_id); ++i)
    if (cuda_api_get_bit(mask, i))
      if (pc == warp_get_active_virtual_pc (dev_id, sm_id, i))
	return false;

  /* If resume warps is not possible - abort */
  if (!cuda_api_resume_warps_until_pc (dev_id, sm_id, mask, pc))
    return false;

  if (cuda_options_software_preemption ())
    {
      device_invalidate (dev_id);
      return true;
    }
  /* invalidate the cache for the warps that have been single-stepped. */
  for (i = 0; i < device_get_num_warps (dev_id); ++i)
    if (cuda_api_get_bit(mask, i))
       warp_invalidate (dev_id, sm_id, i);

  /* must invalidate the SM since that's where the warp valid mask lives */
  auto sm = cuda_state::instance ().device (dev_id)->sm (sm_id);
  sm->invalidate (false);

  return true;
}

bool
warp_single_step (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,
		  uint32_t nsteps, cuda_api_warpmask *single_stepped_warp_mask)
{
  uint32_t i;
  bool rc;
  cuda_api_warpmask tmp;

  cuda_trace ("device %u sm %u warp %u nsteps %u: single-step", dev_id, sm_id, wp_id, nsteps);

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));
  gdb_assert (wp_id < device_get_num_warps (dev_id));

  cuda_api_clear_mask(single_stepped_warp_mask);
  cuda_api_clear_mask(&tmp);
  cuda_api_set_bit(&tmp, wp_id, 1);
  cuda_api_not_mask(&tmp, &tmp);    // Select all but the single-stepped warp in the mask

  rc = cuda_api_single_step_warp (dev_id, sm_id, wp_id, nsteps, single_stepped_warp_mask);
  if (!rc)
    return rc;

  if (cuda_options_software_preemption ())
    {
      device_invalidate (dev_id);
      return true;
    }

  cuda_api_and_mask(&tmp, &tmp, single_stepped_warp_mask);

  if (cuda_api_has_bit(&tmp))
    {
      warning ("Warp(s) other than the current warp had to be single-stepped:%" WARP_MASK_FORMAT,
	  cuda_api_mask_string(single_stepped_warp_mask));
      device_invalidate (dev_id);
    }

  /* invalidate the cache for the warps that have been single-stepped. */
  for (i = 0; i < device_get_num_warps (dev_id); ++i)
    if (cuda_api_get_bit(single_stepped_warp_mask, i))
      warp_invalidate (dev_id, sm_id, i);

  /* must invalidate the SM since that's where the warp valid mask lives */
  auto sm = cuda_state::instance ().device (dev_id)->sm (sm_id);

  sm->invalidate (false);

  return true;
}

bool
warp_is_valid (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  gdb_assert (wp_id < device_get_num_warps (dev_id));
  return cuda_api_get_bit(sm_get_valid_warps_mask (dev_id, sm_id), wp_id) != 0;
}

bool
warp_is_broken (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  gdb_assert (wp_id < device_get_num_warps (dev_id));
  return cuda_api_get_bit(sm_get_broken_warps_mask (dev_id, sm_id), wp_id) != 0;
}

bool
warp_has_error_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);
  bool error_pc_available = false;
  uint64_t error_pc = 0ULL;

  if (wp->m_error_pc_p)
    return wp->m_error_pc_available;

  cuda_api_read_error_pc (dev_id, sm_id, wp_id, &error_pc, &error_pc_available);

  wp->m_error_pc = error_pc;
  wp->m_error_pc_available = error_pc_available;
  wp->m_error_pc_p = CACHED;

  return wp->m_error_pc_available;
}

static void
update_warp_cached_info (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  CUDBGWarpState state;

  cuda_api_read_warp_state (dev_id, sm_id, wp_id, &state);

  auto wp = warp_get (dev_id, sm_id, wp_id);

  wp->m_error_pc = state.errorPC;
  wp->m_error_pc_available = state.errorPCValid;
  wp->m_error_pc_p = CACHED;

  wp->m_block_idx = state.blockIdx;
  wp->m_block_idx_p = CACHED;

  wp->m_cluster_idx = state.clusterIdx;
  wp->m_cluster_idx_p = CACHED;

  wp->m_grid_id = state.gridId;
  wp->m_grid_id_p = CACHED;

  wp->m_active_lanes_mask   = state.activeLanes;
  wp->m_active_lanes_mask_p = CACHED;

  wp->m_valid_lanes_mask   = state.validLanes;
  wp->m_valid_lanes_mask_p = CACHED;

  for (uint32_t ln_id = 0; ln_id < device_get_num_lanes (dev_id); ln_id++) {
    if ( !(state.validLanes & (1U<<ln_id)) )
      continue;
    auto ln = wp->lane (ln_id);
    ln->m_thread_idx = state.lane[ln_id].threadIdx;
    ln->m_virtual_pc = state.lane[ln_id].virtualPC;
    ln->m_exception = state.lane[ln_id].exception;
    ln->m_exception_p = ln->m_thread_idx_p = ln->m_virtual_pc_p = CACHED;

    if (!ln->m_timestamp_p)
      {
	ln->m_timestamp_p = true;
	ln->m_timestamp = cuda_clock ();
      }
  }
  if (!wp->m_timestamp_p)
    {
      wp->m_timestamp_p = true;
      wp->m_timestamp = cuda_clock ();
    }
}

uint64_t
warp_get_grid_id (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);

  if (cuda_remote && !(wp->m_grid_id_p)
      && sm_is_valid (dev_id, sm_id))
    cuda_remote_update_grid_id_in_sm (cuda_get_current_remote_target (), dev_id, sm_id);

  if (wp->m_grid_id_p)
    return wp->m_grid_id;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return wp->m_grid_id;
}

kernel_t
warp_get_kernel (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);
  uint64_t	grid_id;
  kernel_t	kernel;

  if (wp->m_kernel_p)
    return wp->m_kernel;

  grid_id = warp_get_grid_id (dev_id, sm_id, wp_id);
  kernel  = kernels_find_kernel_by_grid_id (dev_id, grid_id);

  if (!kernel)
    {
      device_create_kernel (dev_id, grid_id);
      kernel = kernels_find_kernel_by_grid_id (dev_id, grid_id);
    }

  wp->m_kernel	 = kernel;
  wp->m_kernel_p = CACHED;

  return wp->m_kernel;
}

CuDim3
warp_get_block_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);

  if (cuda_remote && !(wp->m_block_idx_p)
      && sm_is_valid (dev_id, sm_id))
    cuda_remote_update_block_idx_in_sm (cuda_get_current_remote_target (), dev_id, sm_id);

  if (wp->m_block_idx_p)
    return wp->m_block_idx;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return wp->m_block_idx;
}

CuDim3
warp_get_cluster_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);

  if (cuda_remote && !(wp->m_cluster_idx_p)
      && sm_is_valid (dev_id, sm_id))
    cuda_remote_update_cluster_idx_in_sm (cuda_get_current_remote_target (), dev_id, sm_id);

  if (wp->m_cluster_idx_p)
    return wp->m_cluster_idx;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return wp->m_cluster_idx;
}

uint32_t
warp_get_valid_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);

  if (wp->m_valid_lanes_mask_p)
    return wp->m_valid_lanes_mask;

  if (warp_is_valid (dev_id, sm_id, wp_id))
    {
      update_warp_cached_info (dev_id, sm_id, wp_id);
      return wp->m_valid_lanes_mask;
    }

  wp->m_valid_lanes_mask   = 0;
  wp->m_valid_lanes_mask_p = CACHED;

  if (!wp->m_timestamp_p)
    {
      wp->m_timestamp_p = true;
      wp->m_timestamp = cuda_clock ();
    }

  return 0;
}

uint32_t
warp_get_active_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);

  if (wp->m_active_lanes_mask_p)
    return wp->m_active_lanes_mask;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return wp->m_active_lanes_mask;
}

uint32_t
warp_get_divergent_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  uint32_t valid_lanes_mask;
  uint32_t active_lanes_mask;
  uint32_t divergent_lanes_mask;

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));
  gdb_assert (wp_id < device_get_num_warps (dev_id));

  valid_lanes_mask     = warp_get_valid_lanes_mask  (dev_id, sm_id, wp_id);
  active_lanes_mask    = warp_get_active_lanes_mask (dev_id, sm_id, wp_id);
  divergent_lanes_mask = valid_lanes_mask & ~active_lanes_mask;

  return divergent_lanes_mask;
}

uint32_t
warp_get_lowest_active_lane (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  uint32_t active_lanes_mask;
  uint32_t ln_id;

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));
  gdb_assert (wp_id < device_get_num_warps (dev_id));

  active_lanes_mask = warp_get_active_lanes_mask (dev_id, sm_id, wp_id);

  for (ln_id = 0; ln_id < device_get_num_lanes (dev_id); ++ln_id)
    if ((active_lanes_mask >> ln_id) & 1)
      break;

  return ln_id;
}

uint64_t
warp_get_active_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  uint32_t ln_id;
  uint64_t pc;

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));
  gdb_assert (wp_id < device_get_num_warps (dev_id));

  ln_id = warp_get_lowest_active_lane (dev_id, sm_id, wp_id);
  pc = lane_get_pc (dev_id, sm_id, wp_id, ln_id);

  return pc;
}

uint64_t
warp_get_active_virtual_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  uint32_t ln_id;
  uint64_t pc;

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));
  gdb_assert (wp_id < device_get_num_warps (dev_id));

  ln_id = warp_get_lowest_active_lane (dev_id, sm_id, wp_id);
  pc = lane_get_virtual_pc (dev_id, sm_id, wp_id, ln_id);

  return pc;
}

uint64_t
warp_get_error_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);
  bool error_pc_available = false;
  uint64_t error_pc = 0ULL;

  /*if (wp->m_error_pc_p)
    {
      gdb_assert (wp->m_error_pc_available);
      return wp->m_error_pc;
    }
*/
  cuda_api_read_error_pc (dev_id, sm_id, wp_id, &error_pc, &error_pc_available);

  wp->m_error_pc = error_pc;
  wp->m_error_pc_available = error_pc_available;
  wp->m_error_pc_p = CACHED;

  gdb_assert (wp->m_error_pc_available);
  return wp->m_error_pc;
}

bool
warp_valid_timestamp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);

  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));

  return wp->m_timestamp_p;
}

cuda_clock_t
warp_get_timestamp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);

  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));

  gdb_assert (wp->m_timestamp_p);

  return wp->m_timestamp;
}

void
warp_set_grid_id (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint64_t grid_id)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);

  gdb_assert (cuda_remote);

  wp->m_grid_id = grid_id;
  wp->m_grid_id_p = true;
}

void
warp_set_cluster_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, const CuDim3 *cluster_idx)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);

  gdb_assert (cuda_remote);
  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));

  wp->m_cluster_idx = *cluster_idx;
  wp->m_cluster_idx_p = true;
}

void
warp_set_block_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, const CuDim3 *block_idx)
{
  auto wp = warp_get (dev_id, sm_id, wp_id);

  gdb_assert (cuda_remote);
  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));

  wp->m_block_idx = *block_idx;
  wp->m_block_idx_p = true;
}

/* Lanes register cache */
static std::vector<cuda_reg_cache_element_t>::iterator
cuda_reg_cache_find_element (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  for (auto it = cuda_register_cache.begin (); it != cuda_register_cache.end (); ++it)
    if (it->dev == dev_id && it->sm == sm_id && it->wp == wp_id && it->ln == ln_id)
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
cuda_reg_cache_remove_element (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  auto it = cuda_register_cache.begin ();
  for (; it != cuda_register_cache.end (); ++it)
    if (it->dev == dev_id && it->sm == sm_id &&
	it->wp == wp_id && it->ln == ln_id)
      break;

  if (it != cuda_register_cache.end ())
    cuda_register_cache.erase (it);
}

/******************************************************************************
 *
 *				     Lanes
 *
 ******************************************************************************/

lane_state::lane_state ()
{
  ++cuda_state_stats.lane_default_ctor_count;
}

lane_state::lane_state (warp_state *warp, uint32_t lane_idx)
: m_lane_idx (lane_idx), m_warp (warp)
{
  ++cuda_state_stats.lane_ctor_count;
}

lane_state::~lane_state ()
{
  ++cuda_state_stats.lane_dtor_count;
}

void
lane_state::invalidate ()
{
  m_pc_p	 = false;
  m_virtual_pc_p = false;
  m_thread_idx_p = false;
  m_exception_p	 = false;
  m_timestamp_p	 = false;

  cuda_reg_cache_remove_element (warp ()->sm ()->device ()->number (),
    warp ()->sm ()->number (),
    warp ()->number (), m_lane_idx);
}

bool
lane_is_valid (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  uint32_t valid_lanes_mask = warp_get_valid_lanes_mask (dev_id, sm_id, wp_id);
  bool valid = (valid_lanes_mask >> ln_id) & 1;

  auto ln = lane_get (dev_id, sm_id, wp_id, ln_id);
  if (!ln->m_timestamp_p)
    {
      ln->m_timestamp_p = true;
      ln->m_timestamp = cuda_clock ();
    }

  return valid;
}

bool
lane_is_active (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  uint32_t active_lanes_mask = warp_get_active_lanes_mask (dev_id, sm_id, wp_id);
  bool active = (active_lanes_mask >> ln_id) & 1;

  return active;
}

bool
lane_is_divergent (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  uint32_t divergent_lanes_mask = warp_get_divergent_lanes_mask (dev_id, sm_id, wp_id);
  bool divergent = (divergent_lanes_mask >> ln_id) & 1;

  return divergent;
}

CuDim3
lane_get_thread_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  auto ln = lane_get(dev_id, sm_id, wp_id, ln_id);

  /* In a remote session, we fetch the threadIdx of all valid thread in the warp using
   * one rsp packet to reduce the amount of communication. */
  if (cuda_remote && !(ln->m_thread_idx_p)
      && warp_is_valid (dev_id, sm_id, wp_id))
    cuda_remote_update_thread_idx_in_warp (cuda_get_current_remote_target (), dev_id, sm_id, wp_id);

  if (ln->m_thread_idx_p)
    return ln->m_thread_idx;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return ln->m_thread_idx;
}

uint64_t
lane_get_virtual_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  auto ln = lane_get (dev_id, sm_id, wp_id, ln_id);

  if (ln->m_virtual_pc_p)
    return ln->m_virtual_pc;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return ln->m_virtual_pc;
}

uint64_t
lane_get_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  auto ln = lane_get (dev_id, sm_id, wp_id, ln_id);
  auto wp = warp_get (dev_id, sm_id, wp_id);
  uint64_t	pc;
  uint32_t	other_ln_id;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  if (ln->m_pc_p)
    return ln->m_pc;

  cuda_api_read_pc (dev_id, sm_id, wp_id, ln_id, &pc);

  ln->m_pc_p = CACHED;
  ln->m_pc   = pc;

  /* Optimization: all the active lanes share the same virtual PC */
  if (lane_is_active (dev_id, sm_id, wp_id, ln_id))
    for (other_ln_id = 0; other_ln_id < device_get_num_lanes (dev_id); ++other_ln_id)
      if (lane_is_valid (dev_id, sm_id, wp_id, other_ln_id) &&
	  lane_is_active (dev_id, sm_id, wp_id, other_ln_id))
	{
	  wp->lane (other_ln_id)->m_pc_p = CACHED;
	  wp->lane (other_ln_id)->m_pc   = pc;
	}

  return ln->m_pc;
}

CUDBGException_t
lane_get_exception (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  auto ln = lane_get (dev_id, sm_id, wp_id, ln_id);

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  if (ln->m_exception_p)
    return ln->m_exception;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return ln->m_exception;
}

uint32_t
lane_get_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id,
		   uint32_t regno)
{
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  /* If register can not be cached - read it directly */
  if (regno > CUDBG_CACHED_REGISTERS_COUNT)
    {
      uint32_t value;
      cuda_api_read_register (dev_id, sm_id, wp_id, ln_id, regno, &value);
      return value;
    }

  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);
  if ((elem->register_valid_mask[regno >> 5] & (1UL << (regno & 31))) != 0)
    return elem->registers[regno];

  if (regno < 32)
    {
      cuda_api_read_register_range (dev_id, sm_id, wp_id, ln_id, regno & ~31, 32, &elem->registers[regno & ~31]);
      elem->register_valid_mask[regno >> 5] |= 0xffffffff;
    }
  else
    {
      cuda_api_read_register (dev_id, sm_id, wp_id, ln_id, regno, &elem->registers[regno]);
      elem->register_valid_mask[regno >> 5] |= 1UL << (regno & 31);
    }

  return elem->registers[regno];
}

void
lane_set_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id,
		   uint32_t regno, uint32_t value)
{
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  cuda_api_write_register (dev_id, sm_id, wp_id, ln_id, regno, value);
  /* If register can not be cached - read it directly */
  if (regno > CUDBG_CACHED_REGISTERS_COUNT)
      return;

  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);
  elem->registers[regno] = value;
  elem->register_valid_mask[regno>>5]|=1UL<<(regno&31);
}

bool
lane_get_predicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id,
		    uint32_t predicate)
{
  gdb_assert (predicate < device_get_num_predicates (dev_id));
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));
  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);

  if (elem->predicates_valid_p)
    return elem->predicates[predicate] != 0;

  cuda_api_read_predicates (dev_id, sm_id, wp_id, ln_id,
			    device_get_num_predicates (dev_id),
			    elem->predicates);
  elem->predicates_valid_p = CACHED;

  return elem->predicates[predicate] != 0;
}

void
lane_set_predicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id,
		    uint32_t predicate, bool value)
{
  gdb_assert (predicate < device_get_num_predicates (dev_id));
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));
  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);

  if (!elem->predicates_valid_p)
    {
      cuda_api_read_predicates (dev_id, sm_id, wp_id, ln_id,
				device_get_num_predicates (dev_id),
				elem->predicates);
      elem->predicates_valid_p = CACHED;
    }

  elem->predicates[predicate] = value;

  cuda_api_write_predicates (dev_id, sm_id, wp_id, ln_id,
			     device_get_num_predicates (dev_id),
			     elem->predicates);
}

uint32_t
lane_get_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));
  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);

  if (elem->cc_register_valid_p)
    return elem->cc_register;

  cuda_api_read_cc_register (dev_id, sm_id, wp_id, ln_id,
			    &elem->cc_register);
  elem->cc_register_valid_p = CACHED;

  return elem->cc_register;
}

void
lane_set_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id,
		      uint32_t value)
{
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));
  auto elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);

  elem->cc_register = value;
  elem->cc_register_valid_p = CACHED;

  cuda_api_write_cc_register (dev_id, sm_id, wp_id, ln_id, elem->cc_register);
}

int32_t
lane_get_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  int32_t call_depth;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  cuda_api_read_call_depth (dev_id, sm_id, wp_id, ln_id, &call_depth);

  return call_depth;
}

int32_t
lane_get_syscall_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  int32_t syscall_call_depth;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  cuda_api_read_syscall_call_depth (dev_id, sm_id, wp_id, ln_id, &syscall_call_depth);

  return syscall_call_depth;
}

uint64_t
lane_get_virtual_return_address (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,
				 uint32_t ln_id, int32_t level)
{
  uint64_t virtual_return_address;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  cuda_api_read_virtual_return_address (dev_id, sm_id, wp_id, ln_id, level,
					     &virtual_return_address);

  return virtual_return_address;
}

cuda_clock_t
lane_get_timestamp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,uint32_t ln_id)
{
  auto ln = lane_get (dev_id, sm_id, wp_id, ln_id);;

  gdb_assert (ln->m_timestamp_p);

  return ln->m_timestamp;
}

void
lane_set_thread_idx (uint32_t dev_id, uint32_t sm_id,
		     uint32_t wp_id, uint32_t ln_id, CuDim3 *thread_idx)
{
  auto ln = lane_get (dev_id, sm_id, wp_id, ln_id);

  gdb_assert (cuda_remote);
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  ln->m_thread_idx = *thread_idx;
  ln->m_thread_idx_p = true;
}

static void
lane_set_exception_none (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,
			 uint32_t ln_id)
{
  auto ln = lane_get (dev_id, sm_id, wp_id, ln_id);

  ln->m_exception = CUDBG_EXCEPTION_NONE;
  ln->m_exception_p = true;
}
