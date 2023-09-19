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

#ifndef _CUDA_API_H
#define _CUDA_API_H 1

#include "defs.h"
#include "cudadebugger.h"

#include <tuple>
#include <unordered_map>

class coordinates
{
public:
  coordinates (uint32_t dev, uint32_t sm = 0, uint32_t wp = 0, uint32_t ln = 0)
    : m_dev (dev), m_sm (sm), m_wp (wp), m_ln (ln) { }

  std::size_t key (uint32_t extra) const noexcept;
  bool operator==(const coordinates& fs) const noexcept { return (fs.m_dev == m_dev) && (fs.m_sm == m_sm) && (fs.m_wp == m_wp) && (fs.m_ln == m_ln); }

  uint32_t m_dev;
  uint32_t m_sm;
  uint32_t m_wp;
  uint32_t m_ln;
};

class frame_coordinates: public coordinates
{
public:
  frame_coordinates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t level)
    : coordinates (dev, sm, wp, ln), m_level (level) { }

  bool operator==(const frame_coordinates& fc) const noexcept { return (((coordinates) fc) == ((coordinates) *this)) && (fc.m_level == m_level); }

  uint32_t m_level;
};


namespace std {

template <>
struct hash<coordinates>
{
  std::size_t operator()(coordinates const& coords) const noexcept
  {
    return std::size_t (coords.key (0));
  }  
};

template <>
struct hash<frame_coordinates>
{
  std::size_t operator()(frame_coordinates const& fcoords) const noexcept
  {
    return std::size_t (fcoords.key (fcoords.m_level));
  }  
};

}


typedef enum {
  CUDA_ATTACH_STATE_NOT_STARTED,
  CUDA_ATTACH_STATE_IN_PROGRESS,
  CUDA_ATTACH_STATE_APP_READY,
  CUDA_ATTACH_STATE_COMPLETE,
  CUDA_ATTACH_STATE_DETACHING,
  CUDA_ATTACH_STATE_DETACH_COMPLETE
} cuda_attach_state_t;

typedef enum {
  CUDA_API_STATE_UNINITIALIZED,
  CUDA_API_STATE_INITIALIZING,
  CUDA_API_STATE_INITIALIZED,
} cuda_api_state_t;

#define WARP_MASK_FORMAT "s"

typedef struct cuda_api_warpmask_t cuda_api_warpmask;

struct cuda_api_warpmask_t {
    uint64_t mask;
};

class cuda_debugapi final
{
public:
  static constexpr size_t ErrorStringMaxLength = 512;
  static constexpr size_t ErrorStringExMaxLength = 4096;

  cuda_debugapi ();
  ~cuda_debugapi () = default;


  static cuda_debugapi &instance () { return m_debugapi; }

  void set_api(CUDBGAPI api) { m_cudbgAPI = api; }

  int  get_api_ptid () { return m_api_ptid; }

  bool get_api_initialized () { return m_api_state == CUDA_API_STATE_INITIALIZED; }

  void set_attach_state (cuda_attach_state_t state);
  cuda_attach_state_t get_attach_state () { return m_attach_state; }

  bool attach_or_detach_in_progress ()
  {
    return (m_attach_state == CUDA_ATTACH_STATE_DETACHING) ||
      (m_attach_state == CUDA_ATTACH_STATE_IN_PROGRESS);
  }

  cuda_api_state_t get_state () { return m_api_state; }

  void clear_state ();

  void initialize_attach_stub ();
  void handle_initialization_error (CUDBGResult res);
  
  int initialize ();
  void finalize ();

  void suspend_device (uint32_t dev);
  void resume_device (uint32_t dev);

  bool single_step_warp (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t nsteps, cuda_api_warpmask *warp_mask);
  bool resume_warps_until_pc (uint32_t dev, uint32_t sm, cuda_api_warpmask *warp_mask, uint64_t virt_pc);
  
  bool set_breakpoint (uint32_t dev, uint64_t addr);
  bool unset_breakpoint (uint32_t dev, uint64_t addr);
  
  void read_thread_idx (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx);
  void read_broken_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *brokenWarpsMask);
  void read_valid_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *valid_lanes);
  void read_valid_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *valid_warps);
  void read_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth);

  void read_syscall_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth);
  void read_virtual_return_address (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
				    int32_t level, uint64_t *ra);

  void read_error_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *pc, bool* valid);
  void read_warp_state (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState *state);

  void read_grid_id (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *grid_id);
  void read_cluster_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *clusterIdx);
  void read_block_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx);
  void read_active_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *active_lanes);

  void read_code_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
  void read_const_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
  void read_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
  bool read_pinned_memory (uint64_t addr, void *buf, uint32_t sz);
  void read_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
  void read_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
  void write_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
  bool read_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
  void read_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates);
  void read_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, uint32_t *predicates);
  void read_upredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, uint32_t *predicates);

  void read_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t *val);
  void read_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t *val);
  void read_virtual_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
  void read_lane_exception (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception);
  void read_device_exception_state (uint32_t dev, uint64_t *exceptionSMMask, uint32_t n);

  void write_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
  bool write_pinned_memory (uint64_t addr, const void *buf, uint32_t sz);
  void write_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
  bool write_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
  void write_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val);
  void write_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t val);
  void write_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates);
  void write_upredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, const uint32_t *predicates);
  void read_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *val);
  void write_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val);
  void read_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);

  void get_grid_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *grid_dim);
  void get_cluster_dim (uint32_t dev, uint64_t gridId64, CuDim3 *cluster_dim);
  void get_block_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *block_dim);
  void get_blocking (uint32_t dev, uint32_t sm, uint32_t wp, bool *blocking);
  void get_tid (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid);
  void get_elf_image (uint32_t dev,  uint64_t handle, bool relocated, void *elfImage, uint64_t size);
  void get_device_type (uint32_t dev, char *buf, uint32_t sz);
  void get_sm_type (uint32_t dev, char *buf, uint32_t sz);
  void get_device_name (uint32_t dev, char *buf, uint32_t sz);
  void get_num_devices (uint32_t *numDev);
  void get_num_sms (uint32_t dev, uint32_t *numSMs);
  void get_num_warps (uint32_t dev, uint32_t *numWarps);
  void get_num_lanes (uint32_t dev, uint32_t *numLanes);
  void get_num_registers (uint32_t dev, uint32_t *numRegs);
  void get_num_predicates (uint32_t dev, uint32_t *numPredicates);
  void get_num_uregisters (uint32_t dev, uint32_t *numRegs);
  void get_num_upredicates (uint32_t dev, uint32_t *numPredicates);

  void is_device_code_address (uint64_t addr, bool *is_device_address);

  void set_notify_new_event_callback (CUDBGNotifyNewEventCallback callback);
  void get_next_sync_event (CUDBGEvent *event);
  void acknowledge_sync_events ();
  void get_next_async_event (CUDBGEvent *event);

  void disassemble (uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t bufSize);
  void request_cleanup_on_detach (uint32_t resumeAppFlag);
  void get_grid_status (uint32_t dev, uint64_t grid_id, CUDBGGridStatus *status);
  void get_grid_info (uint32_t dev, uint64_t grid_id, CUDBGGridInfo *info);

  void get_adjusted_code_address (uint32_t dev, uint64_t addr, uint64_t *adjusted_addr, CUDBGAdjAddrAction adj_action);
  void set_kernel_launch_notification_mode(CUDBGKernelLaunchNotifyMode mode);
  void get_device_pci_bus_info (uint32_t dev, uint32_t *pci_bus_id, uint32_t *pci_dev_id);
  void read_register_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t idx, uint32_t count, uint32_t *regs);
  void read_uregister_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t idx, uint32_t count, uint32_t *regs);
  void read_global_memory (uint64_t addr, void *buf, uint32_t buf_size);
  void write_global_memory (uint64_t addr, const void *buf, uint32_t buf_size);
  void get_managed_memory_region_info (uint64_t start_addr, CUDBGMemoryInfo *meminfo, uint32_t entries_count, uint32_t *entries_written);

#if CUDBG_API_VERSION_REVISION >= 132
  void get_loaded_function_info (uint32_t dev, uint64_t handle, CUDBGLoadedFunctionInfo *info, uint32_t numEntries);
#endif

  /* Error string */
  void get_error_string_ex (char *buf, uint32_t bufSz, uint32_t *msgSz);
  
private:
  // This function needs updating whenever a new cache map is added.
  void clear_caches ();

  void cuda_api_trace (const char *fmt, ...);

  static void cuda_api_error(CUDBGResult res, const char *fmt, ...)  ATTRIBUTE_PRINTF (2, 3);
  void cuda_dev_api_error(const char *msg, uint32_t dev, CUDBGResult res);

  void cuda_devsmwp_api_error(const char *msg, uint32_t dev, uint32_t sm, uint32_t wp, CUDBGResult res);
  void cuda_devsmwpln_api_error(const char *msg, uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGResult res);

  void cuda_api_print_api_call_result (int res);


  static cuda_debugapi m_debugapi;

  CUDBGAPI m_cudbgAPI = nullptr;
  int  m_api_ptid = 0;
  cuda_api_state_t m_api_state = CUDA_API_STATE_UNINITIALIZED;
  bool m_caching_enabled = true;
  cuda_attach_state_t m_attach_state = CUDA_ATTACH_STATE_NOT_STARTED;
  
  // Cache datastructures. Any additions here require updating clear_caches()
  struct
  {
    std::unordered_map<coordinates, uint32_t> m_call_depth;
    std::unordered_map<coordinates, uint32_t> m_syscall_call_depth;
    std::unordered_map<coordinates, cuda_api_warpmask> m_valid_warps;
    std::unordered_map<frame_coordinates, uint64_t> m_virtual_return_address;
    std::unordered_map<coordinates, std::tuple<uint64_t, bool>> m_error_pc;
    std::unordered_map<coordinates, CUDBGWarpState> m_warp_state;
  } m_cache;
};

const char* cuda_api_mask_string(const cuda_api_warpmask* mask);
void cuda_api_clear_mask(cuda_api_warpmask* mask);
void cuda_api_set_bit(cuda_api_warpmask* mask, int i, int v);
int cuda_api_get_bit(const cuda_api_warpmask* mask, int i);
int cuda_api_has_bit(const cuda_api_warpmask* mask);
int cuda_api_has_multiple_bits(const cuda_api_warpmask* mask);
int cuda_api_eq_mask(const cuda_api_warpmask* m1, const cuda_api_warpmask* m2);
void cuda_api_cp_mask(cuda_api_warpmask* dst, const cuda_api_warpmask* src);
cuda_api_warpmask* cuda_api_or_mask(cuda_api_warpmask* dst, const cuda_api_warpmask* m1, const cuda_api_warpmask* m2);
cuda_api_warpmask* cuda_api_and_mask(cuda_api_warpmask* dst, const cuda_api_warpmask* m1, const cuda_api_warpmask* m2);
cuda_api_warpmask* cuda_api_not_mask(cuda_api_warpmask* dst, const cuda_api_warpmask* m1);

/* Initialization */
void cuda_api_handle_initialization_error (CUDBGResult res);
void cuda_api_handle_get_api_error (CUDBGResult res);
void cuda_api_handle_finalize_api_error (CUDBGResult res);
void cuda_api_set_api (CUDBGAPI api);
int  cuda_api_initialize (void);
void cuda_api_initialize_attach_stub (void);
void cuda_api_finalize (void);
void cuda_api_clear_state (void);
cuda_api_state_t cuda_api_get_state (void);
int  cuda_api_get_ptid (void);

/* Attach support */
void cuda_api_set_attach_state (cuda_attach_state_t state);
bool cuda_api_attach_or_detach_in_progress (void);
cuda_attach_state_t cuda_api_get_attach_state (void);
void cuda_api_request_cleanup_on_detach (uint32_t resumeAppFlag);

/* Device Execution Control */
void cuda_api_suspend_device (uint32_t dev);
void cuda_api_resume_device (uint32_t dev);
bool cuda_api_resume_warps_until_pc (uint32_t dev, uint32_t sm, cuda_api_warpmask *warp_mask, uint64_t virt_pc);
bool cuda_api_single_step_warp (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t nsteps, cuda_api_warpmask *warp_mask);

/* Breakpoints */
bool cuda_api_set_breakpoint (uint32_t dev, uint64_t addr);
bool cuda_api_unset_breakpoint (uint32_t dev, uint64_t addr);
void cuda_api_get_adjusted_code_address (uint32_t dev, uint64_t addr, uint64_t *adjusted_addr, CUDBGAdjAddrAction adj_action);

/* Device State Inspection */
void cuda_api_read_grid_id (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *grid_id);
void cuda_api_read_cluster_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *clusterIdx);
void cuda_api_read_block_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx);
void cuda_api_read_thread_idx (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx);
void cuda_api_read_broken_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *brokenWarpsMask);
void cuda_api_read_valid_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *valid_warps);
void cuda_api_read_valid_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *valid_lanes);
void cuda_api_read_active_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *active_lanes);
void cuda_api_read_code_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_const_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
bool cuda_api_read_pinned_memory (uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
bool cuda_api_read_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
void cuda_api_read_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t *val);
void cuda_api_read_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, uint32_t *predicates);
void cuda_api_read_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t *val);
void cuda_api_read_upredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, uint32_t *predicates);
void cuda_api_read_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *val);
void cuda_api_read_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
void cuda_api_read_virtual_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
void cuda_api_read_lane_exception (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception);
void cuda_api_read_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth);
void cuda_api_read_syscall_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t *depth);
void cuda_api_read_virtual_return_address (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, int32_t level, uint64_t *ra);
void cuda_api_read_device_exception_state (uint32_t dev, uint64_t *exceptionSMMask, uint32_t n);
void cuda_api_read_error_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *pc, bool* valid);
void cuda_api_read_warp_state (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState *state);
void cuda_api_read_register_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t idx, uint32_t count, uint32_t *regs);
void cuda_api_read_uregister_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t idx, uint32_t count, uint32_t *regs);
void cuda_api_read_global_memory (uint64_t addr, void *buf, uint32_t buf_size);
void cuda_api_write_global_memory (uint64_t addr, const void *buf, uint32_t buf_size);
void cuda_api_get_managed_memory_region_info (uint64_t start_addr, CUDBGMemoryInfo *meminfo, uint32_t entries_count, uint32_t *entries_written);

/* Device State Alteration */
void cuda_api_write_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
bool cuda_api_write_pinned_memory (uint64_t addr, const void *buf, uint32_t sz);
void cuda_api_write_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
void cuda_api_write_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
bool cuda_api_write_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
void cuda_api_write_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val);
void cuda_api_write_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicate_size, const uint32_t *predicates);
void cuda_api_write_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t val);
void cuda_api_write_upredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicate_size, const uint32_t *predicates);
void cuda_api_write_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val);

/* Grid Properties */
void cuda_api_get_grid_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *grid_dim);
void cuda_api_get_cluster_dim (uint32_t dev, uint64_t gridId64, CuDim3 *cluster_dim);
void cuda_api_get_block_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *block_dim);
void cuda_api_get_tid (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid);
void cuda_api_get_elf_image (uint32_t dev, uint64_t handle, bool relocated, void *elfImage, uint64_t size);
void cuda_api_get_blocking (uint32_t dev, uint32_t sm, uint32_t wp, bool *blocking);
void cuda_api_get_grid_status (uint32_t dev, uint64_t grid_id, CUDBGGridStatus *status);
void cuda_api_get_grid_info (uint32_t dev, uint64_t grid_id, CUDBGGridInfo *info);

/* Device Properties */
void cuda_api_get_device_name (uint32_t dev, char *buf, uint32_t sz);
void cuda_api_get_device_type (uint32_t dev, char *buf, uint32_t sz);
void cuda_api_get_sm_type (uint32_t dev, char *buf, uint32_t sz);
void cuda_api_get_num_devices (uint32_t *numDev);
void cuda_api_get_num_sms (uint32_t dev, uint32_t *numSMs);
void cuda_api_get_num_warps (uint32_t dev, uint32_t *numWarps);
void cuda_api_get_num_lanes (uint32_t dev, uint32_t *numLanes);
void cuda_api_get_num_registers (uint32_t dev, uint32_t *numRegs);
void cuda_api_get_num_predicates (uint32_t dev, uint32_t *numPredicates);
void cuda_api_get_num_uregisters (uint32_t dev, uint32_t *numRegs);
void cuda_api_get_num_upredicates (uint32_t dev, uint32_t *numPredicates);
void cuda_api_get_device_pci_bus_info (uint32_t dev, uint32_t *pci_bus_id, uint32_t *pci_dev_id);

/* DWARF-related routines */
void cuda_api_disassemble (uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t bufSize);
void cuda_api_is_device_code_address (uint64_t addr, bool *is_device_address);

/* Events */
void cuda_api_handle_set_callback_api_error (CUDBGResult res);
void cuda_api_set_notify_new_event_callback (CUDBGNotifyNewEventCallback callback);
void cuda_api_acknowledge_sync_events (void);
void cuda_api_get_next_sync_event (CUDBGEvent *event);
void cuda_api_get_next_async_event (CUDBGEvent *event);
void cuda_api_set_kernel_launch_notification_mode (CUDBGKernelLaunchNotifyMode mode);

#if CUDBG_API_VERSION_REVISION >= 132
/* Lazy function loading */
void cuda_api_get_loaded_function_info (uint32_t dev, uint64_t handle, CUDBGLoadedFunctionInfo *info, uint32_t numEntries);
#endif

#if CUDBG_API_VERSION_REVISION >= 134
/* Error string */
void cuda_api_get_error_string_ex (char *buf, uint32_t bufSz, uint32_t *msgSz);
#endif

#endif
