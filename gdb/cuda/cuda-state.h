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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _CUDA_STATE_H
#define _CUDA_STATE_H 1

#include "cuda-defs.h"
#include "cuda-bitset.h"
#include "cuda-tdep.h"
#include "cuda-context.h"
#include "cuda-modules.h"
#include "cuda-utils.h"
#include "cuda-packet-manager.h"
#include "cuda-options.h"

#include "cudadebugger.h"

#include <array>
#include <bitset>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

/* So we can have links to our parent objects */
class cuda_lane;
class cuda_warp;
class cuda_sm;
class cuda_device;


class cuda_lane final
{
public:
  /* Default constructor required to make std::array<> work */
  cuda_lane ();
  cuda_lane (cuda_warp *parent_warp, uint32_t idx);
  cuda_lane (const cuda_lane&) { gdb_assert (0); }
  cuda_lane& operator=(const cuda_lane&) { gdb_assert (0); }

  ~cuda_lane ();

  // This method handles most post-constructor initialization. Necessary to
  // do it this way as cuda_lane objects are in an array instead of a
  // vector of pointers, which means the empty constructor is used
  // which can't intializae these fields.
  void configure (cuda_warp *warp, uint32_t idx);

  uint32_t dev_idx () const;
  uint32_t sm_idx () const;
  uint32_t warp_idx () const;
  uint32_t lane_idx () const
  { return m_lane_idx; }

  // Return a pointer to the warp this lane belongs to
  cuda_warp *warp () const
  { return m_warp; }

  void invalidate (bool quietly);

  bool active ();
  bool divergent ();

  cuda_clock_t timestamp () const
  { return m_timestamp; }

  bool timestamp_valid ()
  { return m_timestamp != 0; }

  void set_timestamp (cuda_clock_t now)
  { m_timestamp = now; }

  uint64_t get_pc ();
  void set_pc (uint64_t pc)
  { m_pc_p = true; m_pc = pc; }

  const CuDim3& get_thread_idx ();
  void set_thread_idx (const CuDim3& dim)
  { m_thread_idx_p = true; m_thread_idx = dim; }

  CUDBGException_t get_exception ();
  void set_exception (CUDBGException_t exception)
  { m_exception_p = true; m_exception = exception; }

  void set_exception_none ()
  { set_exception (CUDBG_EXCEPTION_NONE); }

  // Attributes that are less commonly used, or that are
  // expensive to calculate (like the return address vector)
  uint64_t	get_return_address (int32_t level);

  uint32_t	get_register (uint32_t regno);
  uint32_t	get_cc_register ();
  bool		get_predicate (uint32_t predicate);
  void		set_register (uint32_t regno, uint32_t value);
  void		set_predicate (uint32_t predicate, bool value);
  void		set_cc_register (uint32_t value);

  int32_t	get_call_depth ();
  int32_t	get_syscall_call_depth ();

  // For internal use only - lazy device state handling
  void update (const CUDBGLaneState& state);

  // For internal use only - batched device state handling
  void decode (const CUDBGDeviceInfoSizes& info_sizes,
	       const CUDBGDeviceInfo *dev_info, const CUDBGSMInfo *sm_info,
	       const CUDBGWarpInfo *warp_info, const CUDBGGridInfo& grid_info,
	       CUDBGException_t exception, uint8_t*& ptr);

private:

  // parent and index
  uint32_t	   m_lane_idx = ~0;
  cuda_warp	   *m_warp = nullptr;

  // Time the warp was last updated
  cuda_clock_t	   m_timestamp = 0;

  uint64_t	   m_pc = 0;
  bool             m_pc_p = false;

  CuDim3	   m_thread_idx = { 0, 0, 0 };
  bool             m_thread_idx_p = false;

  CUDBGException_t m_exception = { CUDBG_EXCEPTION_NONE };
  bool             m_exception_p = false;

  // These fields are less used and typically more expensive to compute
  // so are only read on demand
  uint32_t	   m_call_depth = 0;
  bool		   m_call_depth_p = false;

  uint32_t	   m_syscall_call_depth = 0;
  bool		   m_syscall_call_depth_p = false;

  uint32_t	   m_cc_register;
  bool		   m_cc_register_p;

  // return addresses on a per-frame basis, indexed by level
  std::unordered_map<int32_t, uint64_t> m_return_address;

  // Predicates
  uint32_t	   m_predicates[CUDA_REG_MAX_PREDICATES];
  // m_predicates_p is a bitmask of which individual predicates are valid
  // (m_predicates_p == (num_predicates - 1)) indicates that all are valid
  uint32_t	   m_predicates_p;

  // Register caches
  cuda_bitset           m_registers_p;
  std::vector<uint32_t>	m_registers;
};

class cuda_warp final
{
public:
  cuda_warp (cuda_sm *parent_sm, uint32_t idx);
  cuda_warp (const cuda_warp&) = delete;
  cuda_warp& operator=(const cuda_warp&) = delete;

  ~cuda_warp ();

  uint32_t dev_idx () const;
  uint32_t sm_idx () const;
  uint32_t warp_idx () const
  { return m_warp_idx; }

  cuda_sm *sm () const
  { return m_sm; }

  cuda_lane *lane (uint32_t idx)
  { return &m_lanes.at (idx); }

  // So warps can iterate over their lanes
  std::array<cuda_lane, CUDBG_MAX_LANES>::iterator lanes ()
  { return m_lanes.begin (); }

  bool valid ();

  cuda_clock_t timestamp () const
  { return m_timestamp; }

  bool timestamp_valid ()
  { return m_timestamp != 0; }

  void set_timestamp (cuda_clock_t now)
  { m_timestamp = now; }

  void invalidate (bool quietly, bool recurse);

  uint64_t get_grid_id ();
  void     set_grid_id (uint64_t grid_id);

  bool has_error_pc ();
  uint64_t get_error_pc ();

  bool has_cluster_exception_target_block_idx ();
  CuDim3 get_cluster_exception_target_block_idx ();

  const CuDim3& get_block_idx ();
  void set_block_idx (const CuDim3& block_idx);

  const CuDim3& get_cluster_idx ();
  void set_cluster_idx (const CuDim3& cluster_idx);
  const CuDim3& get_cluster_dim ();
  void set_cluster_dim (const CuDim3& cluster_dim);

  uint32_t get_valid_lanes_mask ();
  uint32_t get_active_lanes_mask ();
  uint32_t get_lowest_active_lane ();

  uint32_t get_divergent_lanes_mask ()
  { return get_valid_lanes_mask () & ~get_active_lanes_mask (); }

  bool lane_valid (uint32_t ln_id)
  { return (get_valid_lanes_mask () & (1ULL << ln_id)) ? true : false; }

  bool lane_active (uint32_t ln_id)
  { return (get_active_lanes_mask () & (1ULL << ln_id)) ? true : false; }

  bool lane_divergent (uint32_t ln_id)
  { return (get_divergent_lanes_mask () & (1ULL << ln_id)) ? true : false; }

  uint64_t get_active_pc ()
  { return lane (get_lowest_active_lane ())->get_pc (); }

  uint32_t get_uregister (uint32_t regno);
  void	   set_uregister (uint32_t regno, uint32_t value);

  bool     get_upredicate (uint32_t predicate);
  void	   set_upredicate (uint32_t predicate, bool value);

  kernel_t get_kernel ();

  uint32_t registers_allocated ();
  uint32_t shared_mem_size ();

  // For internal use only - lazy device state handling
  void update_state ();

  // For internal use only - batched device state handling
  void decode (const CUDBGDeviceInfoSizes& info_sizes,
	       const CUDBGDeviceInfo *dev_info,
	       const CUDBGSMInfo *sm_info, uint8_t*& ptr);

private:
  void update_warp_resources ();

  /* parent and index */
  uint32_t	m_warp_idx = ~0;
  cuda_sm	*m_sm = nullptr;

  cuda_clock_t	m_timestamp = 0;

  uint64_t	m_grid_id = 0;
  kernel_t	m_kernel = nullptr;

  CuDim3	m_block_idx = { 0, 0, 0 };
  bool	        m_block_idx_p = false;

  uint32_t      m_valid_lanes_mask = 0;
  bool          m_valid_lanes_mask_p = false;

  uint32_t      m_active_lanes_mask = 0;
  bool          m_active_lanes_mask_p = false;

  bool	        m_error_pc_p = false;
  bool          m_error_pc_available = false;
  uint64_t      m_error_pc = 0;

  bool m_cluster_exception_target_block_idx_p = false;
  bool m_cluster_exception_target_block_idx_available = false;
  CuDim3 m_cluster_exception_target_block_idx = { 0 };

  // thread index for lane 0 in this warp
  CuDim3	m_base_thread_idx = { 0, 0, 0 };

  // Less frequently used values, read and cached on demand
  bool		m_cluster_idx_p = false;
  CuDim3	m_cluster_idx = { 0, 0, 0 };
  bool		m_cluster_dim_p = false;
  CuDim3	m_cluster_dim = { 0, 0, 0 };

  // Uniform register caches
  uint32_t	m_upredicates[CUDA_UREG_MAX_PREDICATES];
  uint32_t	m_upredicates_p;

  cuda_bitset	        m_uregisters_p;
  std::vector<uint32_t> m_uregisters;

  // Resources assigned to this warp
  bool m_warp_resources_p = false;
  // Number of registers currently allocated to this warp
  uint32_t m_registers_allocated = 0;
  // Amount of shared memory currently allocated to this warp
  uint32_t m_shared_mem_size = 0;

  /* Array of lanes belonging to this warp */
  std::array<cuda_lane, CUDBG_MAX_LANES> m_lanes;
};

class cuda_sm final
{
public:
  cuda_sm (cuda_device *parent_dev, uint32_t idx);
  cuda_sm (const cuda_sm&) = delete;
  cuda_sm& operator=(const cuda_sm&) = delete;

  ~cuda_sm ();

  uint32_t sm_idx () const
  { return m_sm_idx; }

  uint32_t dev_idx () const;

  cuda_device *device () const
  { return m_device; }

  cuda_warp *warp (uint32_t idx) const
  { return m_warps.at (idx).get (); }

  bool valid ()
  { return cuda_api_has_bit (get_valid_warps_mask ()); }

  cuda_clock_t timestamp () const
  { return m_timestamp; }

  void set_timestamp (cuda_clock_t now)
  { m_timestamp = now; }

  bool warp_valid (uint32_t wp_id)
  { return cuda_api_get_bit(get_valid_warps_mask (), wp_id) != 0; }

  bool warp_broken (uint32_t wp_id)
  { return cuda_api_get_bit(get_broken_warps_mask (), wp_id) != 0; }

  const cuda_api_warpmask* get_valid_warps_mask ();
  const cuda_api_warpmask* get_broken_warps_mask ();

  void invalidate (bool quietly, bool recurse);

  bool has_exception ();
  CUDBGException_t get_exception ();
  bool has_error_pc ();
  uint64_t get_error_pc ();

  bool single_step_warp (uint32_t wp_id, uint32_t lane_id_hint, uint32_t nsteps, uint32_t flags, cuda_api_warpmask *single_stepped_warp_mask);
  bool resume_warps_until_pc (cuda_api_warpmask* wp_mask, uint64_t pc);

  // For internal use only - lazy state updates
  void update_state ();

  // For internal use only - batch state updates
  void decode (const CUDBGDeviceInfoSizes& info_sizes,
	       const CUDBGDeviceInfo* dev_info, uint8_t*& ptr);

private:
  // Helpers for read_sm_exception calls
  bool fetch_sm_exception_info ();
  void reset_sm_exception_info ();

private:
  uint32_t          m_sm_idx = ~0;
  cuda_device       *m_device = nullptr;

  cuda_clock_t      m_timestamp = 0;

  bool              m_valid_warps_mask_p = false;
  cuda_api_warpmask m_valid_warps_mask = { 0 };

  bool              m_broken_warps_mask_p = false;
  cuda_api_warpmask m_broken_warps_mask = { 0 };

  bool             m_sm_exception_info_p = false;
  CUDBGException_t m_exception = { CUDBG_EXCEPTION_NONE };
  bool		   m_error_pc_available = false;
  uint64_t	   m_error_pc = 0;

  std::vector<std::unique_ptr<cuda_warp>> m_warps;
};

class cuda_device final
{
public:
  cuda_device (uint32_t idx);
  cuda_device (const cuda_device&) = delete;
  cuda_device& operator=(const cuda_device&) = delete;

  ~cuda_device ();

  cuda_sm *sm (uint32_t sm_id) const
  { return m_sms.at (sm_id).get (); }

  uint32_t dev_idx () const
  { return m_dev_id; }

  bool valid ()
  { return get_active_sms_mask ().any (); }

  uint32_t get_num_sms () const
  { return m_num_sms; }

  uint32_t get_num_warps () const
  { return m_num_warps; }

  uint32_t get_num_lanes () const
  { return m_num_lanes; }

  cuda_clock_t timestamp () const
  { return m_timestamp; }

  void set_timestamp (cuda_clock_t now)
  { m_timestamp = now; }

  bool has_exception ();
  bool sm_has_exception (uint32_t sm_id);

  void clear_sm_exception_mask_p ()
  { m_sm_exception_mask_p = false; }

  void create_kernel (uint64_t grid_id);

  void invalidate (bool quietly, bool recurse);

  bool is_any_context_present () const;

  bool is_active_context (cuda_context* context) const;

  bool suspended ();

  uint32_t get_num_registers () const
  { return m_num_registers; }

  uint32_t get_num_predicates () const
  { return m_num_predicates; }

  uint32_t get_num_uregisters () const
  { return m_num_uregisters; }

  uint32_t get_num_upredicates () const
  { return m_num_upredicates; }

  uint32_t	get_insn_size () const
  { return m_insn_size; }

  const char* get_sm_type () const
  { return m_sm_type; }

  uint32_t	get_sm_version ();

  uint32_t	get_num_kernels ();
  uint32_t	get_pci_bus_id ();
  uint32_t	get_pci_dev_id ();

  const char*	get_device_name () const
  { return m_dev_name; }

  const char*	get_device_type () const
  { return m_dev_type; }

  const cuda_bitset& get_active_sms_mask ();

  kernel_t get_kernel (uint64_t grid_id);
  const CUDBGGridInfo& get_grid_info (uint64_t grid_id);

  void print ();
  void suspend ();
  void resume ();

  void set_device_spec (uint32_t num_sms, uint32_t num_warps,
			uint32_t num_lanes, uint32_t num_registers,
			const char *dev_type, const char *sm_type);

  void update (CUDBGDeviceInfoQueryType_t type);

  const bool incremental () const
  { return m_incremental; }

private:
  // Internal use only - lazy device updates
  void update_exception_state ();

  // Internal use only - batch device updates
  void decode (const CUDBGDeviceInfoSizes& info_sizes, uint8_t *buffer, uint32_t& length);

  // Our device id
  uint32_t m_dev_id = ~0;

  cuda_clock_t m_timestamp = 0;

  char	m_dev_type[256] = { 0 };
  char	m_dev_name[256] = { 0 };
  char	m_sm_type[64] = { 0 };

  uint32_t m_insn_size = 0;

  uint32_t m_sm_version = 0;
  bool	   m_sm_version_p = false;

  uint32_t m_num_sms = 0;
  uint32_t m_num_warps = 0;
  uint32_t m_num_lanes = 0;
  uint32_t m_num_registers = 0;
  uint32_t m_num_predicates = 0;
  uint32_t m_num_uregisters = 0;
  uint32_t m_num_upredicates = 0;

  uint32_t m_pci_dev_id = 0;
  uint32_t m_pci_bus_id = 0;
  bool	   m_pci_bus_info_p = false;

  // Buffer used for batched device state update
  // Length is m_info_sizes.requiredBufferSize
  bool m_incremental = false;

  CUDBGDeviceInfoSizes             m_info_sizes = { 0 };
  gdb::unique_xmalloc_ptr<uint8_t> m_info_buffer;

  // SM bitmasks
  cuda_bitset m_sm_active_mask;
  bool m_sm_active_mask_p = false;

  cuda_bitset m_sm_exception_mask;
  bool m_sm_exception_mask_p = false;

  // Grid info cache
  std::unordered_map<uint64_t, CUDBGGridInfo> m_grid_info;

  // Vector of cuda_sm pointers
  std::vector<std::unique_ptr<cuda_sm>> m_sms;
};

class cuda_state final
{
public:
  cuda_state ();
  ~cuda_state () = default;

  static cuda_device *device (uint32_t dev_id)
  { return m_instance.m_devs.at (dev_id).get (); }

  static cuda_sm *sm (uint32_t dev_id, uint32_t sm_id)
  { return device (dev_id)->sm (sm_id); }

  static cuda_warp *warp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id); }

  static cuda_lane *lane (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->lane (ln_id); }

  static void reset ();
  static void invalidate_kernels ();
  static void invalidate ();

  /* System State */
  static void initialize ();
  static void finalize ();
  static CUDBGCapabilityFlags get_supported_capabilities ();

  static uint32_t get_num_devices ()
  { return m_instance.m_num_devices; }

  static uint32_t get_num_kernels ();
  static uint32_t get_num_present_kernels ();

  static void resolve_breakpoints (int bp_number_from);
  static void cleanup_breakpoints ();

  static cuda_context* create_context (uint32_t dev_id, uint64_t context_id, uint32_t thread_id);
  static void destroy_context (uint64_t context_id);
  static bool is_context_active (cuda_context* context);
  static bool is_any_context_present (uint32_t dev_id);
  static cuda_context* find_context_by_id (uint64_t context_id);
  
  static cuda_context* current_context ()
  { return m_instance.m_current_context; }

  static void set_current_context (cuda_context* context)
  { m_instance.m_current_context = context; }

  static std::unordered_map<uint64_t, std::unique_ptr<cuda_context>>& contexts ()
  { return m_instance.m_context_map; }

  static std::unordered_map<uint64_t, std::unique_ptr<cuda_module>>& modules ()
  { return m_instance.m_module_map; }

  static cuda_module* create_module (uint64_t module_id,
				     CUDBGElfImageProperties properties,
				     uint64_t context_id,
				     uint64_t elf_image_size);

  static void destroy_module (uint64_t module_id);

  static cuda_module* find_module_by_id (uint64_t module_id);
  static cuda_module* find_module_by_address (CORE_ADDR addr);

  static bool broken (cuda_coords &coords);

  static void update_all_state (CUDBGDeviceInfoQueryType_t type)
  {
    for (auto& dev : m_instance.m_devs)
      dev->update (type);
  }

  static const cuda_bitset& suspended_devices_mask ()
  { return m_instance.m_suspended_devices_mask; }

  static void clear_suspended_devices_mask (uint32_t dev_id)
  { m_instance.m_suspended_devices_mask.set (dev_id, false); }

  static void set_suspended_devices_mask (uint32_t dev_id)
  { m_instance.m_suspended_devices_mask.set (dev_id, true); }

  static void flush_disasm_caches ();

  static void set_device_spec (uint32_t dev_id,
			       uint32_t num_sms,
			       uint32_t num_warps,
			       uint32_t num_lanes,
			       uint32_t num_registers,
			       const char *dev_type, const char *sm_type)
  { device (dev_id)->set_device_spec (num_sms, num_warps, num_lanes, num_registers, dev_type, sm_type); }

  // System helper functions

  // Device helper functions
  static void device_invalidate_kernels (uint32_t dev_id);

  static bool device_suspended (uint32_t dev_id)
  { return m_instance.m_suspended_devices_mask[dev_id]; }

  static cuda_clock_t device_timestamp (uint32_t dev_id)
  { return device (dev_id)->timestamp (); }

  static const char* device_get_device_name (uint32_t dev_id)
  { return device (dev_id)->get_device_name (); }

  static const char* device_get_device_type (uint32_t dev_id)
  { return device (dev_id)->get_device_type (); }

  static const char* device_get_sm_type (uint32_t dev_id)
  { return device (dev_id)->get_sm_type (); }

  static uint32_t device_get_sm_version (uint32_t dev_id)
  { return device (dev_id)->get_sm_version (); }

  static uint32_t device_get_insn_size (uint32_t dev_id)
  { return device (dev_id)->get_insn_size (); }

  static uint32_t device_get_num_sms	(uint32_t dev_id)
  { return device (dev_id)->get_num_sms (); }

  static uint32_t device_get_num_warps (uint32_t dev_id)
  { return device (dev_id)->get_num_warps (); }

  static uint32_t device_get_num_lanes (uint32_t dev_id)
  { return device (dev_id)->get_num_lanes (); }

  static uint32_t device_get_num_registers (uint32_t dev_id)
  { return device (dev_id)->get_num_registers (); }

  static uint32_t device_get_num_predicates (uint32_t dev_id)
  { return device (dev_id)->get_num_predicates (); }

  static uint32_t device_get_num_uregisters (uint32_t dev_id)
  { return device (dev_id)->get_num_uregisters (); }

  static uint32_t device_get_num_upredicates (uint32_t dev_id)
  { return device (dev_id)->get_num_upredicates (); }

  static uint32_t device_get_num_kernels (uint32_t dev_id)
  { return device (dev_id)->get_num_kernels (); }

  static uint32_t device_get_pci_bus_id (uint32_t dev_id)
  { return device (dev_id)->get_pci_bus_id (); }

  static uint32_t device_get_pci_dev_id (uint32_t dev_id)
  { return device (dev_id)->get_pci_dev_id (); }

  static bool device_valid (uint32_t dev_id)
  { return device (dev_id)->valid (); }

  static bool device_has_exception (uint32_t dev_id)
  { return device (dev_id)->has_exception (); }

  static const cuda_bitset& device_get_active_sms_mask (uint32_t dev_id)
  { return device (dev_id)->get_active_sms_mask (); }

  static void device_print (uint32_t dev_id)
  { device (dev_id)->print (); }

  static void device_suspend (uint32_t dev_id)
  { device (dev_id)->suspend (); }

  static void device_resume (uint32_t dev_id)
  { device (dev_id)->resume (); }

  static kernel_t device_get_kernel (uint32_t dev_id, uint64_t grid_id)
  { return device (dev_id)->get_kernel (grid_id); }

  static bool single_step_warp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t lane_id_hint, uint32_t nsteps, uint32_t flags, cuda_api_warpmask *single_stepped_warp_mask)
  { return sm (dev_id, sm_id)->single_step_warp (wp_id, lane_id_hint, nsteps, flags, single_stepped_warp_mask); }

  static bool resume_warps_until_pc (uint32_t dev_id, uint32_t sm_id, cuda_api_warpmask* wp_mask, uint64_t pc)
  { return sm (dev_id, sm_id)->resume_warps_until_pc (wp_mask, pc); }

  // SM helper functions
  static cuda_clock_t sm_timestamp (uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->timestamp (); }

  static void sm_update_state(uint32_t dev_id, uint32_t sm_id)
  { sm (dev_id, sm_id)->update_state (); }

  static bool sm_valid (uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->valid (); }

  static bool sm_has_exception (uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->has_exception (); }

  static bool sm_has_error_pc (uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->has_error_pc (); }

  static const cuda_api_warpmask* sm_get_valid_warps_mask (uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->get_valid_warps_mask (); }

  static const cuda_api_warpmask* sm_get_broken_warps_mask (uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->get_broken_warps_mask (); }

  static CUDBGException_t sm_get_exception (uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->get_exception (); }

  static uint64_t sm_get_error_pc (uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->get_error_pc (); }

  static cuda_clock_t warp_timestamp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->timestamp (); }

  static bool warp_timestamp_valid (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->timestamp_valid (); }

  static bool warp_valid (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return sm (dev_id, sm_id)->warp_valid (wp_id); }

  static bool warp_broken (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return sm (dev_id, sm_id)->warp_broken (wp_id); }

  // Warp helper functions
  static bool warp_has_error_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->has_error_pc (); }

  static bool
  warp_has_cluster_exception_target_block_idx (uint32_t dev_id, uint32_t sm_id,
					       uint32_t wp_id)
  {
    return warp (dev_id, sm_id, wp_id)
	->has_cluster_exception_target_block_idx ();
  }

  static kernel_t warp_get_kernel (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_kernel (); }

  static const CuDim3& warp_get_block_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_block_idx (); }

  static void warp_set_block_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, const CuDim3& block_idx)
  { device (dev_id)->sm (sm_id)->warp (wp_id)->set_block_idx (block_idx); }

  static const CuDim3& warp_get_cluster_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_cluster_idx (); }

  static const CuDim3& warp_get_cluster_dim (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_cluster_dim (); }

  static uint64_t warp_get_grid_id (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_grid_id (); }

  static uint32_t warp_get_valid_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_valid_lanes_mask (); }

  static uint32_t warp_get_active_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_active_lanes_mask (); }

  static uint32_t warp_get_divergent_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_divergent_lanes_mask (); }

  static uint32_t warp_get_lowest_active_lane (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_lowest_active_lane (); }

  static uint64_t warp_get_active_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->get_active_pc (); }

  static uint64_t warp_get_error_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->get_error_pc (); }

  static CuDim3
  warp_get_cluster_exception_target_block_idx (uint32_t dev_id, uint32_t sm_id,
					       uint32_t wp_id)
  {
    return warp (dev_id, sm_id, wp_id)
	->get_cluster_exception_target_block_idx ();
  }

  static void warp_set_grid_id (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint64_t grid_id)
  { warp (dev_id, sm_id, wp_id)->set_grid_id (grid_id); }

  static void warp_set_cluster_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, const CuDim3& cluster_idx)
  { warp (dev_id, sm_id, wp_id)->set_cluster_idx (cluster_idx); }

  static void warp_set_cluster_dim (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, const CuDim3& cluster_dim)
  { warp (dev_id, sm_id, wp_id)->set_cluster_dim (cluster_dim); }

  static uint32_t warp_get_uregister (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t regno)
  { return warp (dev_id, sm_id, wp_id)->get_uregister (regno); }

  static bool warp_get_upredicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t pred)
  { return warp (dev_id, sm_id, wp_id)->get_upredicate (pred); }

  static void warp_set_uregister (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t regno, uint32_t value)
  { warp (dev_id, sm_id, wp_id)->set_uregister (regno, value); }

  static void warp_set_upredicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t pred, bool value)
  { warp (dev_id, sm_id, wp_id)->set_upredicate (pred, value); }

  static uint32_t warp_registers_allocated (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->registers_allocated (); }

  static uint32_t warp_shared_mem_size (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->shared_mem_size (); }

  // Lane helper functions
  static cuda_clock_t lane_timestamp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->timestamp (); }

  static bool lane_timestamp_valid (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->timestamp_valid (); }

  static bool lane_valid (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return warp (dev_id, sm_id, wp_id)->lane_valid (ln_id); }

  static bool lane_active (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return warp (dev_id, sm_id, wp_id)->lane_active (ln_id); }

  static bool lane_divergent (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return warp (dev_id, sm_id, wp_id)->lane_divergent (ln_id); }

  static uint64_t lane_get_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_pc (); }

  static CuDim3	lane_get_thread_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_thread_idx (); }

  static CUDBGException_t lane_get_exception (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_exception (); }

  static uint32_t lane_get_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t regno)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_register (regno); }

  static uint32_t lane_get_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_cc_register (); }

  static bool lane_get_predicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t pred)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_predicate (pred); }

  static void lane_set_thread_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, const CuDim3& dim)
  { lane (dev_id, sm_id, wp_id, ln_id)->set_thread_idx (dim); }

  static void lane_set_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t regno, uint32_t value)
  { lane (dev_id, sm_id, wp_id, ln_id)->set_register (regno, value); }

  static void lane_set_predicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t pred, bool value)
  { lane (dev_id, sm_id, wp_id, ln_id)->set_predicate (pred, value); }

  static void lane_set_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t value)
  { lane (dev_id, sm_id, wp_id, ln_id)->set_cc_register (value); }

  static int32_t lane_get_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_call_depth (); }

  static int32_t lane_get_syscall_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_syscall_call_depth (); }

  static uint64_t lane_get_return_address (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,
					   uint32_t ln_id, int32_t level)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_return_address (level); }

private:
  static cuda_state m_instance;

  uint32_t    m_num_devices = 0;

  cuda_bitset m_suspended_devices_mask;

  std::vector<std::unique_ptr<cuda_device>> m_devs;

  std::unordered_map<uint64_t, std::unique_ptr<cuda_context>> m_context_map;
  std::unordered_map<uint64_t, std::unique_ptr<cuda_module>> m_module_map;

  cuda_context* m_current_context;
};

#endif
