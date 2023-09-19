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

#include "defs.h"
#include "cuda-defs.h"
#include "cuda-tdep.h"
#include "cuda-context.h"
#include "cuda-iterator.h"
#include "cuda-utils.h"
#include "cuda-packet-manager.h"
#include "cuda-options.h"
#include "cuda-elf-image.h"

#include <array>
#include <bitset>
#include <string>
#include <vector>

#ifdef __ANDROID__
#undef CUDBG_MAX_DEVICES
#define CUDBG_MAX_DEVICES 4
#endif /*__ANDROID__*/

/* So we can have links to our parent objects */
class lane_state;
class warp_state;
class sm_state;
class device_state;

class lane_state final
{
public:
  /* Default constructor required to make std::array<> work */
  lane_state ();
  lane_state (warp_state *warp, uint32_t lane_idx);
  lane_state (const lane_state&) { gdb_assert (0); }
  lane_state& operator=(const lane_state&) { gdb_assert (0); }

  ~lane_state ();

  uint32_t number () { return m_lane_idx; }
  warp_state *warp () { return m_warp; }

  void invalidate ();

  void configure (warp_state *warp, uint32_t idx)
  {
    m_warp = warp;
    m_lane_idx = idx;
  }

  bool		   m_thread_idx_p = false;
  bool		   m_pc_p = false;
  bool		   m_exception_p = false;
  bool		   m_virtual_pc_p = false;
  bool		   m_timestamp_p = false;
  CuDim3	   m_thread_idx = { 0 };
  uint64_t	   m_pc = 0;
  CUDBGException_t m_exception = { (CUDBGException_t) 0 };
  uint64_t	   m_virtual_pc = 0;
  cuda_clock_t	   m_timestamp = { 0 };

private:
  uint32_t	   m_lane_idx = 0;
  warp_state	   *m_warp;
};

class warp_state final
{
public:
  warp_state (sm_state *sm, uint32_t warp_idx);
  warp_state (const warp_state&) = delete;
  warp_state& operator=(const warp_state&) = delete;

  ~warp_state ();

  uint32_t number () { return m_warp_idx; }
  sm_state *sm () { return m_sm; }
  lane_state *lane (uint32_t idx) { return &m_lanes.at (idx); }

  void invalidate ();

  bool	       m_valid_p = false;
  bool	       m_broken_p = false;
  bool	       m_block_idx_p = false;
  bool	       m_cluster_idx_p = false;
  bool	       m_kernel_p = false;
  bool	       m_grid_id_p = false;
  bool	       m_valid_lanes_mask_p = false;
  bool	       m_active_lanes_mask_p = false;
  bool	       m_timestamp_p = false;
  bool	       m_error_pc_p = false;
  bool	       m_valid = false;
  bool	       m_broken = false;
  bool	       m_error_pc_available = false;
  CuDim3       m_block_idx = { 0 };
  CuDim3       m_cluster_idx = { 0 };
  kernel_t     m_kernel = nullptr;
  uint64_t     m_grid_id = 0;
  uint64_t     m_error_pc = 0;
  uint32_t     m_valid_lanes_mask = 0;
  uint32_t     m_active_lanes_mask = 0;
  cuda_clock_t m_timestamp = { 0 };

private:
  uint32_t     m_warp_idx = 0;
  sm_state     *m_sm;
  std::array<lane_state, CUDBG_MAX_LANES> m_lanes;
};

class sm_state final
{
public:
  sm_state (device_state *dev, uint32_t sm_idx);
  sm_state (const sm_state&) = delete;
  sm_state& operator=(const sm_state&) = delete;

  ~sm_state ();

  uint32_t number () { return m_sm_idx; }
  device_state *device () { return m_device; }
  warp_state *warp (uint32_t idx) { return m_warps.at (idx).get (); }

  void invalidate (bool recurse);

  bool		    m_valid_warps_mask_p = false;
  bool		    m_broken_warps_mask_p = false;
  cuda_api_warpmask m_valid_warps_mask = { 0 };
  cuda_api_warpmask m_broken_warps_mask = { 0 };
  std::vector<std::unique_ptr<warp_state>> m_warps;

private:
  uint32_t     m_sm_idx = 0;
  device_state *m_device;
};

class device_state final
{
public:
  device_state (uint32_t dev_idx);
  device_state (const device_state&) = delete;
  device_state& operator=(const device_state&) = delete;

  ~device_state ();

  uint32_t number () { return m_dev_idx; }

  uint32_t get_num_sms ();
  uint32_t get_num_warps ();
  uint32_t get_num_lanes ();

  sm_state *sm (uint32_t idx) { return m_sms.at (idx).get (); }

  void invalidate ();

  void suspend ();
  void resume ();

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
  uint64_t m_sm_exception_mask[(CUDBG_MAX_SMS + 63) / 64] = { 0 };    // Mask needs to be large enough to hold all the SMs, rounded up
  contexts_t m_contexts = nullptr;    // state for contexts associated with this device

  std::vector<std::unique_ptr<sm_state>> m_sms;

private:
  uint32_t m_dev_idx = 0;
};

class cuda_state final
{
public:
  cuda_state ();
  ~cuda_state () = default;

  static cuda_state& instance () { return m_instance; }

  device_state *device (uint32_t idx) { return m_devs.at (idx).get (); }

  void reset ();
  void invalidate_kernels ();

  /* System State */
  void initialize ();
  void finalize ();

  uint32_t get_num_devices ();
  uint32_t get_num_kernels ();
  uint32_t get_num_present_kernels ();

  void resolve_breakpoints (int bp_number_from);
  void cleanup_breakpoints ();
  void cleanup_contexts ();
  bool is_broken (cuda_coords_t &coords);
  uint32_t get_suspended_devices_mask ();
  void flush_disasm_cache ();

  void set_device_spec (uint32_t, uint32_t, uint32_t,
			uint32_t, uint32_t, const char *, const char *);

  context_t find_context_by_addr (CORE_ADDR addr);

private:
  static cuda_state m_instance;

  bool			    m_num_devices_p;
  uint32_t		    m_num_devices;
  std::vector<std::unique_ptr<device_state>> m_devs;

public:
  uint32_t	m_suspended_devices_mask;
};

/* System State */
void	 cuda_system_initialize			  (void);
void	 cuda_system_finalize			  (void);
uint32_t cuda_system_get_num_devices		  (void);
uint32_t cuda_system_get_num_kernels		  (void);
uint32_t cuda_system_get_num_present_kernels	  (void);
void	 cuda_system_resolve_breakpoints	  (int bp_number_from);
void	 cuda_system_cleanup_breakpoints	  (void);
void	 cuda_system_cleanup_contexts		  (void);
bool	 cuda_system_is_broken			  (cuda_coords_t &coords);
uint32_t cuda_system_get_suspended_devices_mask	  (void);
void	 cuda_system_flush_disasm_cache		  (void);

void	 cuda_system_set_device_spec	(uint32_t, uint32_t, uint32_t,
					 uint32_t, uint32_t, const char *, const char *);

context_t cuda_system_find_context_by_addr     (CORE_ADDR addr);

/* Device State */
const char* device_get_device_name	   (uint32_t dev_id);
const char* device_get_device_type	   (uint32_t dev_id);
const char* device_get_sm_type		   (uint32_t dev_id);
uint32_t    device_get_sm_version          (uint32_t dev_id);
uint32_t    device_get_insn_size	   (uint32_t dev_id);
uint32_t    device_get_num_sms		   (uint32_t dev_id);
uint32_t    device_get_num_warps	   (uint32_t dev_id);
uint32_t    device_get_num_lanes	   (uint32_t dev_id);
uint32_t    device_get_num_registers	   (uint32_t dev_id);
uint32_t    device_get_num_predicates	   (uint32_t dev_id);
uint32_t    device_get_num_uregisters	   (uint32_t dev_id);
uint32_t    device_get_num_upredicates	   (uint32_t dev_id);
uint32_t    device_get_num_kernels	   (uint32_t dev_id);
uint32_t    device_get_pci_bus_id	   (uint32_t dev_id);
uint32_t    device_get_pci_dev_id	   (uint32_t dev_id);

bool	    device_is_valid		   (uint32_t dev_id);
bool	    device_is_any_context_present  (uint32_t dev_id);
bool	    device_is_active_context	   (uint32_t dev_id, context_t context);
bool	    device_has_exception	   (uint32_t dev_id);
void	    device_get_active_sms_mask	   (uint32_t dev_id, std::bitset<CUDBG_MAX_SMS> &mask);
contexts_t  device_get_contexts		   (uint32_t dev_id);

context_t   device_find_context_by_id	   (uint32_t dev_id, uint64_t context_id);
context_t   device_find_context_by_addr	   (uint32_t dev_id, CORE_ADDR addr);

void	    device_print      (uint32_t dev_id);
void	    device_resume     (uint32_t dev_id);
void	    device_suspend    (uint32_t dev_id);
void	    device_invalidate (uint32_t dev_id);
bool        device_suspended  (uint32_t dev_id);

/* SM State */
bool	    sm_is_valid			   (uint32_t dev_id, uint32_t sm_id);
bool	    sm_has_exception		   (uint32_t dev_id, uint32_t sm_id);
cuda_api_warpmask*    sm_get_valid_warps_mask	     (uint32_t dev_id, uint32_t sm_id);
cuda_api_warpmask*    sm_get_broken_warps_mask	     (uint32_t dev_id, uint32_t sm_id);

/* Warp State */
bool	 warp_is_valid		       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
bool	 warp_is_broken		       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
bool	 warp_has_error_pc	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
kernel_t warp_get_kernel	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
CuDim3	 warp_get_block_idx	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
CuDim3	 warp_get_cluster_idx	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint64_t warp_get_grid_id	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint32_t warp_get_valid_lanes_mask     (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint32_t warp_get_active_lanes_mask    (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint32_t warp_get_divergent_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint32_t warp_get_lowest_active_lane   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint64_t warp_get_active_pc	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint64_t warp_get_active_virtual_pc    (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
uint64_t warp_get_error_pc	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
bool	 warp_valid_timestamp	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
cuda_clock_t warp_get_timestamp	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
void	 warp_set_grid_id	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint64_t grid_id);
void	 warp_set_cluster_idx	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, const CuDim3 *cluster_idx);
void	 warp_set_block_idx	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, const CuDim3 *block_idx);
uint32_t warp_get_uregister	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t regno);
bool	 warp_get_upredicate	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t predicate);
void	 warp_set_uregister	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t regno, uint32_t value);
void	warp_set_upredicate	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t predicate, bool value);

bool	 warp_single_step	       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t nsteps, cuda_api_warpmask *single_stepped_warp_mask);
bool	 warps_resume_until	       (uint32_t dev_id, uint32_t sm_id, cuda_api_warpmask* wp_mask, uint64_t pc);

/* Lane State */
bool		 lane_is_valid	     (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
bool		 lane_is_active	     (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
bool		 lane_is_divergent   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
uint64_t	 lane_get_pc	     (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
uint64_t	 lane_get_virtual_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
CuDim3		 lane_get_thread_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
CUDBGException_t lane_get_exception  (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
uint32_t	 lane_get_register   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t regno);
uint32_t	 lane_get_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
bool		 lane_get_predicate  (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t predicate);
void		 lane_set_register   (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t regno, uint32_t value);
void		 lane_set_predicate  (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t predicate, bool value);
void		 lane_set_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, uint32_t value);
int32_t		 lane_get_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
int32_t		 lane_get_syscall_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
uint64_t	 lane_get_virtual_return_address (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, int32_t level);
cuda_clock_t	 lane_get_timestamp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,uint32_t ln_id);
void		 lane_set_thread_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id, CuDim3 *thread_idx);
#endif
