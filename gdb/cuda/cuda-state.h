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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _CUDA_STATE_H
#define _CUDA_STATE_H 1

#include "cuda-defs.h"
#include "cuda-tdep.h"
#include "cuda-context.h"
#include "cuda-utils.h"
#include "cuda-packet-manager.h"
#include "cuda-options.h"
#include "cuda-elf-image.h"

#include <array>
#include <bitset>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __ANDROID__
#undef CUDBG_MAX_DEVICES
#define CUDBG_MAX_DEVICES 4
#endif /*__ANDROID__*/

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
  cuda_lane (cuda_warp *warp, uint32_t lane_idx);
  cuda_lane (const cuda_lane&) { gdb_assert (0); }
  cuda_lane& operator=(const cuda_lane&) { gdb_assert (0); }

  ~cuda_lane ();

  uint32_t number () { return m_lane_idx; }
  cuda_warp *warp () { return m_warp; }

  void invalidate_caches (bool quietly = false);

  void configure (cuda_warp *warp, uint32_t idx)
  {
    m_warp = warp;
    m_lane_idx = idx;
  }

  bool is_timestamp_valid ()
  { return m_timestamp_p; }

  cuda_clock_t get_timestamp ()
  { gdb_assert (m_timestamp_p); return m_timestamp; }

  void set_timestamp (cuda_clock_t ts)
  { m_timestamp = ts; m_timestamp_p = true; }

  bool is_active ();
  bool is_divergent ();
  bool is_valid ();

  uint64_t get_virtual_return_address (int32_t level);

  uint64_t get_pc ();

  void set_pc (uint64_t pc, bool cached)
  { m_pc = pc; m_pc_p = cached; }

  uint64_t get_virtual_pc ();

  void set_virtual_pc (uint64_t pc)
  { m_virtual_pc = pc; m_virtual_pc_p = true; }

  CuDim3 get_thread_idx ();

  void set_thread_idx (const CuDim3* idx)
  { m_thread_idx = *idx; m_thread_idx_p = true; }

  CUDBGException_t get_exception ();

  void set_exception (CUDBGException_t exception)
  { m_exception = exception; m_exception_p = true; }

  void set_exception_none ()
  { set_exception (CUDBG_EXCEPTION_NONE); }

  uint32_t	 get_register   (uint32_t regno);
  uint32_t	 get_cc_register ();
  bool		 get_predicate  (uint32_t predicate);
  void		 set_register   (uint32_t regno, uint32_t value);
  void		 set_predicate  (uint32_t predicate, bool value);
  void		 set_cc_register (uint32_t value);

  int32_t	 get_call_depth ();

  int32_t	 get_syscall_call_depth ();

private:
  /* cached data */
  bool		   m_timestamp_p = false;
  cuda_clock_t	   m_timestamp = 0;

  bool		   m_pc_p = false;
  uint64_t	   m_pc = 0;

  bool		   m_virtual_pc_p = false;
  uint64_t	   m_virtual_pc = 0;

  bool		   m_thread_idx_p = false;
  CuDim3	   m_thread_idx = { 0, 0, 0 };

  bool             m_call_depth_p = false;
  uint32_t         m_call_depth = 0;

  bool             m_syscall_call_depth_p = false;
  uint32_t         m_syscall_call_depth = 0;

  bool		   m_exception_p = false;
  CUDBGException_t m_exception = { (CUDBGException_t) 0 };

  // virtual return addresses on a per-frame basis, indexed by level
  std::unordered_map<int32_t, uint64_t> m_virtual_return_address;

  // parent and index
  uint32_t	   m_lane_idx = 0;
  cuda_warp	   *m_warp = nullptr;
};

class cuda_warp final
{
public:
  cuda_warp (cuda_sm *sm, uint32_t warp_idx);
  cuda_warp (const cuda_warp&) = delete;
  cuda_warp& operator=(const cuda_warp&) = delete;

  ~cuda_warp ();

  uint32_t number () { return m_warp_idx; }
  cuda_sm *sm () { return m_sm; }
  cuda_lane *lane (uint32_t idx) { return &m_lanes.at (idx); }

  void invalidate_caches (bool quietly = false);

  void update_cached_info ();

  bool has_error_pc ();
  uint64_t get_error_pc ();

  CuDim3 get_block_idx ();
  void set_block_idx (const CuDim3* idx);

  CuDim3 get_cluster_idx ();
  void set_cluster_idx (const CuDim3* idx);

  uint32_t get_uregister (uint32_t regno);
  bool get_upredicate (uint32_t predicate);

  void set_uregister (uint32_t regno, uint32_t value);
  void set_upredicate (uint32_t predicate, bool value);

  uint32_t get_valid_lanes_mask ();
  uint32_t get_active_lanes_mask ();
  uint32_t get_divergent_lanes_mask ();
  uint32_t get_lowest_active_lane ();

  uint64_t get_grid_id ();
  void set_grid_id (uint64_t grid_id);

  bool lane_is_valid (uint32_t ln_id);
  bool lane_is_active (uint32_t ln_id);
  bool lane_is_divergent (uint32_t ln_id);

  uint64_t get_active_pc ();
  uint64_t get_active_virtual_pc ();

  bool is_timestamp_valid ()
  { return m_timestamp_p; }

  cuda_clock_t get_timestamp ()
  { gdb_assert (m_timestamp_p); return m_timestamp; }

  void set_timestamp (cuda_clock_t clock)
  { m_timestamp = clock; m_timestamp_p = true; }

  kernel_t get_kernel ();

private:
  /* cached data */
  bool           m_state_p = false;
  CUDBGWarpState m_state;

  bool	       m_valid_p = false;
  bool	       m_valid = false;

  bool	       m_block_idx_p = false;
  CuDim3       m_block_idx = { 0, 0, 0 };

  bool	       m_cluster_idx_p = false;
  CuDim3       m_cluster_idx = { 0, 0, 0 };

  bool	       m_kernel_p = false;
  kernel_t     m_kernel = nullptr;

  bool	       m_grid_id_p = false;
  uint64_t     m_grid_id = 0;

  bool	       m_valid_lanes_mask_p = false;
  uint32_t     m_valid_lanes_mask = 0;

  bool	       m_active_lanes_mask_p = false;
  uint32_t     m_active_lanes_mask = 0;

  bool	       m_timestamp_p = false;
  cuda_clock_t m_timestamp = { 0 };

  bool	       m_error_pc_p = false;
  bool	       m_error_pc_available = false;
  uint64_t     m_error_pc = 0;

  /* parent and index */
  uint32_t     m_warp_idx = 0;
  cuda_sm     *m_sm = nullptr;

  /* Array of lanes belonging to this warp */
  std::array<cuda_lane, CUDBG_MAX_LANES> m_lanes;
};

class cuda_sm final
{
public:
  cuda_sm (cuda_device *dev, uint32_t sm_idx);
  cuda_sm (const cuda_sm&) = delete;
  cuda_sm& operator=(const cuda_sm&) = delete;

  ~cuda_sm ();

  uint32_t number () { return m_sm_idx; }
  cuda_device *device () { return m_device; }
  cuda_warp *warp (uint32_t idx) { return m_warps.at (idx).get (); }

  void invalidate_valid_warp_mask ()
  { m_valid_warps_mask_p = false; }

  void invalidate_broken_warp_mask ()
  { m_broken_warps_mask_p = false; }

  void invalidate_caches (bool recurse, bool quietly = false);

  bool is_valid ();

  bool has_exception ();

  bool warp_is_valid (uint32_t wp_id);
  bool warp_is_broken (uint32_t wp_id);

  cuda_api_warpmask* get_valid_warps_mask ();
  cuda_api_warpmask* get_broken_warps_mask ();

  void set_exception_none ();

  bool single_step_warp (uint32_t wp_id, uint32_t nsteps, cuda_api_warpmask *single_stepped_warp_mask);
  bool resume_warps_until_pc (cuda_api_warpmask* wp_mask, uint64_t pc);

private:
  bool		    m_valid_warps_mask_p = false;
  cuda_api_warpmask m_valid_warps_mask = { 0 };

  bool		    m_broken_warps_mask_p = false;
  cuda_api_warpmask m_broken_warps_mask = { 0 };

  std::vector<std::unique_ptr<cuda_warp>> m_warps;

private:
  uint32_t     m_sm_idx = 0;
  cuda_device *m_device;
};

class cuda_device final
{
public:
  cuda_device (uint32_t dev_id);
  cuda_device (const cuda_device&) = delete;
  cuda_device& operator=(const cuda_device&) = delete;

  ~cuda_device ();

  uint32_t number () { return m_dev_id; }

  uint32_t get_num_sms ();
  uint32_t get_num_warps ();
  uint32_t get_num_lanes ();

  cuda_sm *sm (uint32_t idx)
  { return m_sms.at (idx).get (); }

  uint64_t sm_has_exception (uint32_t sm_id);

  void clear_sm_exception_mask_valid_p ()
  { m_sm_exception_mask_valid_p = false; }

  void create_kernel (uint64_t grid_id);

  void invalidate_caches (bool quietly = false);

  bool is_valid ();

  bool is_any_context_present ();

  bool is_active_context (context_t context)
  { return contexts_is_active_context (m_contexts, context); }

  bool is_suspended ()
  { return m_suspended; }

  contexts_t get_contexts ()
  { return m_contexts; }

  uint32_t get_num_registers ();
  uint32_t get_num_predicates ();
  uint32_t get_num_uregisters ();
  uint32_t get_num_upredicates ();

  uint32_t get_num_kernels ();
  uint32_t get_pci_bus_id ();
  uint32_t get_pci_dev_id ();

  const char* get_device_name ();
  
  const char* get_device_type ();

  const char* get_sm_type ();

  uint32_t    get_sm_version ();
    
  uint32_t    get_insn_size ();

  context_t find_context_by_id (uint64_t context_id);
  context_t find_context_by_addr (CORE_ADDR addr);
  
  void print ();
  void suspend ();
  void resume ();

  void set_device_spec (uint32_t num_sms,
			uint32_t num_warps,
			uint32_t num_lanes,
			uint32_t num_registers,
			uint32_t num_uregisters,
			const char *dev_type, const char *sm_type);

  bool	    has_exception ();
  void      update_exception_state ();

  void	    get_active_sms_mask (std::bitset<CUDBG_MAX_SMS> &mask);

  void	    cleanup_contexts ();

private:
  bool	   m_valid_p = false;
  bool	   m_insn_size_p = false;
  bool	   m_num_sms_p = false;
  bool	   m_num_warps_p = false;
  bool	   m_num_lanes_p = false;
  bool	   m_num_registers_p = false;
  bool	   m_num_predicates_p = false;
  bool	   m_num_uregisters_p = false;
  bool	   m_num_upredicates_p = false;
  bool	   m_pci_bus_info_p = false;
  bool	   m_dev_type_p = false;
  bool	   m_dev_name_p = false;
  bool	   m_sm_exception_mask_valid_p = false;
  bool	   m_sm_version_p = false;

  /* the above fields are invalidated on resume */

  bool	   m_valid = false;	// at least one active lane
  bool	   m_suspended = false;	// true if the device is suspended

  char	   m_dev_type[256] = { 0 };
  char	   m_dev_name[256] = { 0 };
  char	   m_sm_type[64] = { 0 };
  uint32_t m_sm_version = 0;
  uint32_t m_insn_size = 0;
  uint32_t m_num_sms = 0;
  uint32_t m_num_warps = 0;
  uint32_t m_num_lanes = 0;
  uint32_t m_num_registers = 0;
  uint32_t m_num_predicates = 0;
  uint32_t m_num_uregisters = 0;
  uint32_t m_num_upredicates = 0;
  uint32_t m_pci_dev_id = 0;
  uint32_t m_pci_bus_id = 0;

  // Mask needs to be large enough to hold all the SMs, rounded up
  uint64_t m_sm_exception_mask[(CUDBG_MAX_SMS + 63) / 64] = { 0 };

  // state for contexts associated with this device
  contexts_t m_contexts = nullptr;

  std::vector<std::unique_ptr<cuda_sm>> m_sms;

  uint32_t m_dev_id = 0;
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
  static void invalidate_caches ();

  /* System State */
  static void initialize ();
  static void finalize ();

  static uint32_t get_num_devices ();
  static uint32_t get_num_kernels ();
  static uint32_t get_num_present_kernels ();

  static void resolve_breakpoints (int bp_number_from);
  static void cleanup_breakpoints ();

  static void cleanup_contexts ();
  static context_t find_context_by_id (uint64_t context_id);
  static context_t find_context_by_addr (CORE_ADDR addr);

  static bool is_broken (cuda_coords &coords);

  static uint32_t get_suspended_devices_mask ()
  { return m_instance.m_suspended_devices_mask; }

  static void clear_suspended_devices_mask (uint32_t dev_id)
  { m_instance.m_suspended_devices_mask &= ~(1 << dev_id); }

  static void set_suspended_devices_mask (uint32_t dev_id)
  { m_instance.m_suspended_devices_mask |= (1 << dev_id); }

  static void flush_disasm_caches ();

  static void set_device_spec (uint32_t dev_id,
			       uint32_t num_sms,
			       uint32_t num_warps,
			       uint32_t num_lanes,
			       uint32_t num_registers,
			       uint32_t num_uregisters,
			       const char *dev_type, const char *sm_type)
  { device (dev_id)->set_device_spec (num_sms, num_warps, num_lanes, num_registers, num_uregisters, dev_type, sm_type); }

  // System helper functions

  // Device helper functions
  static const char* device_get_device_name	   (uint32_t dev_id)
  { return device (dev_id)->get_device_name (); }
  
  static const char* device_get_device_type	   (uint32_t dev_id)
  { return device (dev_id)->get_device_type (); }

  static const char* device_get_sm_type		   (uint32_t dev_id)
  { return device (dev_id)->get_sm_type (); }

  static uint32_t    device_get_sm_version          (uint32_t dev_id)
  { return device (dev_id)->get_sm_version (); }
    
  static uint32_t    device_get_insn_size	   (uint32_t dev_id)
  { return device (dev_id)->get_insn_size (); }

  static uint32_t    device_get_num_sms		   (uint32_t dev_id)
  { return device (dev_id)->get_num_sms (); }

  static uint32_t    device_get_num_warps	   (uint32_t dev_id)
  { return device (dev_id)->get_num_warps (); }

  static uint32_t    device_get_num_lanes	   (uint32_t dev_id)
  { return device (dev_id)->get_num_lanes (); }

  static uint32_t    device_get_num_registers	   (uint32_t dev_id)
  { return device (dev_id)->get_num_registers (); }

  static uint32_t    device_get_num_predicates	   (uint32_t dev_id)
  { return device (dev_id)->get_num_predicates (); }

  static uint32_t    device_get_num_uregisters	   (uint32_t dev_id)
  { return device (dev_id)->get_num_uregisters (); }

  static uint32_t    device_get_num_upredicates	   (uint32_t dev_id)
  { return device (dev_id)->get_num_upredicates (); }

  static uint32_t    device_get_num_kernels (uint32_t dev_id)
  { return device (dev_id)->get_num_kernels (); }

  static uint32_t    device_get_pci_bus_id (uint32_t dev_id)
  { return device (dev_id)->get_pci_bus_id (); }

  static uint32_t    device_get_pci_dev_id	   (uint32_t dev_id)
  { return device (dev_id)->get_pci_dev_id (); }

  static bool	    device_is_valid		   (uint32_t dev_id)
  { return device (dev_id)->is_valid (); }

  static bool	    device_is_any_context_present  (uint32_t dev_id)
  { return device (dev_id)->is_any_context_present (); }

  static bool	    device_is_active_context	   (uint32_t dev_id, context_t context)
  { return device (dev_id)->is_active_context (context); }

  static bool	    device_has_exception	   (uint32_t dev_id)
  { return device (dev_id)->has_exception (); }

  static void	    device_get_active_sms_mask	   (uint32_t dev_id, std::bitset<CUDBG_MAX_SMS> &mask)
  { device (dev_id)->get_active_sms_mask (mask); }

  static contexts_t  device_get_contexts (uint32_t dev_id)
  { return device (dev_id)->get_contexts (); }

  static context_t   device_find_context_by_id (uint32_t dev_id, uint64_t context_id)
  { return device (dev_id)->find_context_by_id (context_id); }
    
  static context_t   device_find_context_by_addr (uint32_t dev_id, CORE_ADDR addr)
  { return device (dev_id)->find_context_by_addr (addr); }

  static void device_invalidate_caches (uint32_t dev_id)
  { device (dev_id)->invalidate_caches (); }

  static bool device_is_suspended (uint32_t dev_id)
  { return device (dev_id)->is_suspended (); }
  
  static void device_print (uint32_t dev_id)
  { device (dev_id)->print (); }

  static void device_suspend (uint32_t dev_id)
  { device (dev_id)->suspend (); }

  static void device_resume (uint32_t dev_id)
  { device (dev_id)->resume (); }

  static bool single_step_warp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t nsteps, cuda_api_warpmask *single_stepped_warp_mask)
  { return sm (dev_id, sm_id)->single_step_warp (wp_id, nsteps, single_stepped_warp_mask); }
    
  static bool resume_warps_until_pc (uint32_t dev_id, uint32_t sm_id, cuda_api_warpmask* wp_mask, uint64_t pc)
  { return sm (dev_id, sm_id)->resume_warps_until_pc (wp_mask, pc); }

  // SM helper functions
  static bool sm_is_valid (uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->is_valid (); }

  static bool sm_has_exception	(uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->has_exception (); }

  static cuda_api_warpmask* sm_get_valid_warps_mask (uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->get_valid_warps_mask (); }
  
  static cuda_api_warpmask* sm_get_broken_warps_mask (uint32_t dev_id, uint32_t sm_id)
  { return sm (dev_id, sm_id)->get_broken_warps_mask (); }

  static bool warp_is_valid (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return sm (dev_id, sm_id)->warp_is_valid (wp_id); }
  
  static bool warp_is_broken (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return sm (dev_id, sm_id)->warp_is_broken (wp_id); }

  // Warp helper functions
  static bool warp_has_error_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->has_error_pc (); }

  static kernel_t warp_get_kernel	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_kernel (); }

  static CuDim3	 warp_get_block_idx	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_block_idx (); }

  static CuDim3	 warp_get_cluster_idx	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_cluster_idx (); }

  static uint64_t warp_get_grid_id	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_grid_id (); }

  static uint32_t warp_get_valid_lanes_mask     (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_valid_lanes_mask (); }

  static uint32_t warp_get_active_lanes_mask    (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_active_lanes_mask (); }

  static uint32_t warp_get_divergent_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_divergent_lanes_mask (); }

  static uint32_t warp_get_lowest_active_lane   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_lowest_active_lane (); }

  static uint64_t warp_get_active_pc	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return device (dev_id)->sm (sm_id)->warp (wp_id)->get_active_pc (); }

  static uint64_t warp_get_active_virtual_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->get_active_virtual_pc (); }

  static uint64_t warp_get_error_pc	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->get_error_pc (); }

  static bool	 warp_is_timestamp_valid       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->is_timestamp_valid (); }

  static cuda_clock_t warp_get_timestamp       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { return warp (dev_id, sm_id, wp_id)->get_timestamp (); }

  static void	 warp_set_grid_id	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint64_t grid_id)
  { warp (dev_id, sm_id, wp_id)->set_grid_id (grid_id); }

  static void	 warp_set_cluster_idx	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, const CuDim3 *cluster_idx)
  { warp (dev_id, sm_id, wp_id)->set_cluster_idx (cluster_idx); }

  static void	 warp_set_block_idx	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, const CuDim3 *block_idx)
  { warp (dev_id, sm_id, wp_id)->set_block_idx (block_idx); }
  
  static uint32_t warp_get_uregister	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t regno)
  { return warp (dev_id, sm_id, wp_id)->get_uregister (regno); }

  static bool	 warp_get_upredicate	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t predicate)
  { return warp (dev_id, sm_id, wp_id)->get_upredicate (predicate); }

  static void	 warp_set_uregister	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t regno, uint32_t value)
  { warp (dev_id, sm_id, wp_id)->set_uregister (regno, value); }

  static void	warp_set_upredicate	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t predicate, bool value)
  { warp (dev_id, sm_id, wp_id)->set_upredicate (predicate, value); }

  static void warp_update_cached_info (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
  { warp (dev_id, sm_id, wp_id)->update_cached_info (); }

  // Lane helper functions
  static bool lane_is_valid (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return warp (dev_id, sm_id, wp_id)->lane_is_valid (ln_id); }

  static bool lane_is_active	     (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return warp (dev_id, sm_id, wp_id)->lane_is_active (ln_id); }

  static bool lane_is_divergent   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return warp (dev_id, sm_id, wp_id)->lane_is_divergent (ln_id); }

  static uint64_t	 lane_get_pc	     (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_pc (); }

  static uint64_t	 lane_get_virtual_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_virtual_pc (); }

  static CuDim3		 lane_get_thread_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_thread_idx (); }

  static CUDBGException_t lane_get_exception  (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_exception (); }

  static uint32_t	 lane_get_register   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t regno)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_register (regno); }

  static uint32_t	 lane_get_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_cc_register (); }

  static bool		 lane_get_predicate  (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t predicate)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_predicate (predicate); }

  static void		 lane_set_register   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t regno, uint32_t value)
  { return lane (dev_id, sm_id, wp_id, ln_id)->set_register (regno, value); }

  static void		 lane_set_predicate  (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t predicate, bool value)
  { return lane (dev_id, sm_id, wp_id, ln_id)->set_predicate (predicate, value); }

  static void		 lane_set_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t value)
  { return lane (dev_id, sm_id, wp_id, ln_id)->set_cc_register (value); }

  static int32_t	 lane_get_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_call_depth (); }

  static int32_t	 lane_get_syscall_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_syscall_call_depth (); }

  static cuda_clock_t	 lane_get_timestamp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,uint32_t ln_id)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_timestamp (); }

  static void		 lane_set_thread_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, CuDim3 *thread_idx)
  { return lane (dev_id, sm_id, wp_id, ln_id)->set_thread_idx (thread_idx); }

  static uint64_t lane_get_virtual_return_address (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,
						   uint32_t ln_id, int32_t level)
  { return lane (dev_id, sm_id, wp_id, ln_id)->get_virtual_return_address (level); }

private:
  static cuda_state m_instance;

  bool m_num_devices_p;
  uint32_t m_num_devices;
  uint32_t m_suspended_devices_mask;

  std::vector<std::unique_ptr<cuda_device>> m_devs;
};

#endif
