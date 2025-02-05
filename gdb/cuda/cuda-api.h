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

#ifndef _CUDA_API_H
#define _CUDA_API_H 1

#ifndef GDBSERVER
#include "defs.h"
#endif

#include <array>
#include <chrono>
#include <type_traits>

#include "cudadebugger.h"

typedef enum {
  CUDA_ATTACH_STATE_NOT_STARTED,
  CUDA_ATTACH_STATE_IN_PROGRESS,
  CUDA_ATTACH_STATE_APP_READY,
  CUDA_ATTACH_STATE_COMPLETE,
  CUDA_ATTACH_STATE_DETACHING,
  CUDA_ATTACH_STATE_DETACH_COMPLETE
} cuda_attach_state_t;

#define WARP_MASK_FORMAT "s"

struct cuda_api_warpmask_t {
    uint64_t mask;
};

typedef struct cuda_api_warpmask_t cuda_api_warpmask;

class cuda_debugapi_version
{
public:
    explicit cuda_debugapi_version()
      : m_major {0}, m_minor {0}, m_revision {0} { }
    explicit cuda_debugapi_version(uint32_t major, uint32_t minor, uint32_t revision)
      : m_major {major}, m_minor {minor}, m_revision {revision} { }
        
    uint32_t m_major;
    uint32_t m_minor;
    uint32_t m_revision;
};

// Profiling interface
struct cuda_api_stat
{
  std::string name;
  uint32_t times_called;
  std::chrono::microseconds total_time;
  std::chrono::microseconds min_time;
  std::chrono::microseconds max_time;
};

class cuda_debugapi final
{
public:
  static constexpr size_t ErrorStringMaxLength = 512;
  static constexpr size_t ErrorStringExMaxLength = 4096;

  cuda_debugapi ();
  ~cuda_debugapi () = default;

  static void set_api(CUDBGAPI api)
  { s_instance.m_cudbgAPI = api; }

  static void set_api_version (const cuda_debugapi_version& api_version)
  { s_instance.m_api_version = api_version; }
    
  static void set_api_version (uint32_t major, uint32_t minor, uint32_t revision)
  { s_instance.m_api_version = cuda_debugapi_version (major, minor, revision); }
    
  static const cuda_debugapi_version& api_version ()
  { return s_instance.m_api_version; }

  static int  get_api_ptid ()
  { return s_instance.m_api_ptid; }

  static bool api_state_initializing ()
  { return s_instance.m_api_state == CUDA_API_STATE_INITIALIZING; }

  static bool api_state_initialized ()
  { return s_instance.m_api_state == CUDA_API_STATE_INITIALIZED; }

  static bool api_state_uninitialized ()
  { return s_instance.m_api_state == CUDA_API_STATE_UNINITIALIZED; }

  // Attach support
  static void set_attach_state (cuda_attach_state_t state);
  
  static cuda_attach_state_t get_attach_state ()
  { return s_instance.m_attach_state; }

  static bool attach_or_detach_in_progress ()
  {
    return (s_instance.m_attach_state == CUDA_ATTACH_STATE_DETACHING) ||
      (s_instance.m_attach_state == CUDA_ATTACH_STATE_IN_PROGRESS);
  }

  static void clear_state ();

  // Initialization
  static int initialize ();
  static void finalize ();
  static CUDBGCapabilityFlags get_supported_capabilities ();

  static void initialize_attach_stub ();
  static void handle_initialization_error (CUDBGResult res);
  static void handle_finalize_api_error (CUDBGResult res);
  static void handle_set_callback_api_error (CUDBGResult res);

  static void print_get_api_error (CUDBGResult res);

  // Device Execution Control
  static void suspend_device (uint32_t dev);
  static void resume_device (uint32_t dev);

  static bool single_step_warp (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t laneHint, uint32_t nsteps, uint32_t flags, cuda_api_warpmask *warp_mask);
  static bool single_step_warp (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t nsteps, cuda_api_warpmask *warp_mask);
  static bool resume_warps_until_pc (uint32_t dev, uint32_t sm, cuda_api_warpmask *warp_mask, uint64_t virt_pc);
  
  // Device Breakpoint Handling
  static bool set_breakpoint (uint32_t dev, uint64_t addr);
  static bool unset_breakpoint (uint32_t dev, uint64_t addr);
  
  // Device State Inspection
  static void read_thread_idx (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CuDim3 *threadIdx);
  static void read_broken_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *brokenWarpsMask);
  static void read_valid_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *valid_lanes);
  static void read_valid_warps (uint32_t dev, uint32_t sm, cuda_api_warpmask *valid_warps);
  static void read_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *depth);

  static void read_syscall_call_depth (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *depth);
  static void read_virtual_return_address (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
					   int32_t level, uint64_t *ra);

  static void read_error_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *pc, bool* valid);
  static void read_warp_state (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpState *state);
  static void read_warp_resources (uint32_t dev, uint32_t sm, uint32_t wp, CUDBGWarpResources *resources);
  
  static void read_grid_id (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *grid_id);
  static void read_cluster_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *clusterIdx);
  static void read_block_idx (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx);
  static void read_active_lanes (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *active_lanes);

  static void read_code_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
  static void read_const_memory (uint32_t dev, uint64_t addr, void *buf, uint32_t sz);
  static bool read_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
  static bool read_pinned_memory (uint64_t addr, void *buf, uint32_t sz);
  static void read_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
  static void read_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, void *buf, uint32_t sz);
  static bool read_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, void *buf, uint32_t sz);
  static void read_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates);
  static void read_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, uint32_t *predicates);
  static void read_upredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, uint32_t *predicates);

  static void read_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t *val);
  static void read_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t *val);
  static void read_virtual_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);
  static void read_lane_exception (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, CUDBGException_t *exception);
  static void read_device_exception_state (uint32_t dev, uint64_t *exceptionSMMask, uint32_t n);
  static void read_sm_exception (uint32_t dev, uint32_t sm, CUDBGException_t *exception, uint64_t *errorPC, bool *errorPCValid);

  // Device State Alteration
  static bool write_generic_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
  static bool write_pinned_memory (uint64_t addr, const void *buf, uint32_t sz);
  static void write_param_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
  static bool write_local_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t addr, const void *buf, uint32_t sz);
  static void write_shared_memory (uint32_t dev, uint32_t sm, uint32_t wp, uint64_t addr, const void *buf, uint32_t sz);
  static void write_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t regno, uint32_t val);
  static void write_uregister (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t regno, uint32_t val);
  static void write_predicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t predicates_size, const uint32_t *predicates);
  static void write_upredicates (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t predicates_size, const uint32_t *predicates);
  static void read_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t *val);
  static void write_cc_register (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t val);
  static void read_pc (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *pc);

  // Grid properties
  static void get_grid_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *grid_dim);
  static void get_cluster_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *cluster_dim);
  static void get_block_dim (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *block_dim);
  static void get_blocking (uint32_t dev, uint32_t sm, uint32_t wp, bool *blocking);

  static void get_tid (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid);
  static void get_elf_image (uint32_t dev,  uint64_t handle, bool relocated, void *elfImage, uint64_t size);

  // Device properties
  static void get_device_type (uint32_t dev, char *buf, uint32_t sz);
  static void get_sm_type (uint32_t dev, char *buf, uint32_t sz);
  static void get_device_name (uint32_t dev, char *buf, uint32_t sz);
  static void get_num_devices (uint32_t *numDev);
  static void get_num_sms (uint32_t dev, uint32_t *numSMs);
  static void get_num_warps (uint32_t dev, uint32_t *numWarps);
  static void get_num_lanes (uint32_t dev, uint32_t *numLanes);
  static void get_num_registers (uint32_t dev, uint32_t *numRegs);
  static void get_num_predicates (uint32_t dev, uint32_t *numPredicates);
  static void get_num_uregisters (uint32_t dev, uint32_t *numRegs);
  static void get_num_upredicates (uint32_t dev, uint32_t *numPredicates);

  static void is_device_code_address (uint64_t addr, bool *is_device_address);

  static void set_notify_new_event_callback (CUDBGNotifyNewEventCallback callback);
  static void get_next_sync_event (CUDBGEvent *event);
  static void acknowledge_sync_events ();
  static void get_next_async_event (CUDBGEvent *event);

  static void disassemble (uint32_t dev, uint64_t addr, uint32_t *instSize, char *buf, uint32_t bufSize);
  static void request_cleanup_on_detach (uint32_t resumeAppFlag);
  static void get_grid_status (uint32_t dev, uint64_t grid_id, CUDBGGridStatus *status);
  static void get_grid_info (uint32_t dev, uint64_t grid_id, CUDBGGridInfo *info);

  static void get_adjusted_code_address (uint32_t dev, uint64_t addr, uint64_t *adjusted_addr, CUDBGAdjAddrAction adj_action);
  static void set_kernel_launch_notification_mode(CUDBGKernelLaunchNotifyMode mode);
  static void get_device_pci_bus_info (uint32_t dev, uint32_t *pci_bus_id, uint32_t *pci_dev_id);
  static void read_register_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint32_t idx, uint32_t count, uint32_t *regs);
  static void read_uregister_range (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t idx, uint32_t count, uint32_t *regs);
  static void read_global_memory (uint64_t addr, void *buf, uint32_t buf_size);
  static void write_global_memory (uint64_t addr, const void *buf, uint32_t buf_size);
  static void get_managed_memory_region_info (uint64_t start_addr, CUDBGMemoryInfo *meminfo, uint32_t entries_count, uint32_t *entries_written);

  static void get_loaded_function_info (uint32_t dev, uint64_t handle, CUDBGLoadedFunctionInfo *info, uint32_t numEntries);
  static void get_loaded_function_info (uint32_t dev, uint64_t handle, CUDBGLoadedFunctionInfo *info, uint32_t startIndex, uint32_t numEntries);

  static void generate_coredump(const char *name, CUDBGCoredumpGenerationFlags flags);

  static bool get_host_addr_from_device_addr (uint32_t dev, uint64_t devaddr, uint64_t *hostaddr);

  // Error string
  static void get_error_string_ex (char *buf, uint32_t bufSz, uint32_t *msgSz);
  
  static void get_const_bank_address (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t bank, uint32_t offset, uint64_t* address);
  static void get_const_bank_address (uint32_t dev, uint64_t gridId64, uint32_t bank, uint64_t* address, uint32_t* size);

  // Used for batched device info updates
  static bool get_device_info_sizes (uint32_t dev, CUDBGDeviceInfoSizes* sizes);
  static bool get_device_info (uint32_t dev, CUDBGDeviceInfoQueryType_t type, void *buffer, uint32_t length, uint32_t *data_length);

  // Internal debugging commands
  static bool execute_internal_command (const char *command, char *resultBuffer, uint32_t sizeInBytes);

  static bool get_cluster_exception_target_block (uint32_t dev, uint32_t sm,
						  uint32_t wp,
						  CuDim3 *blockIdx,
						  bool *blockIdxValid);

  static void for_each_api_stat (std::function<void(const cuda_api_stat&)> func);

  static void reset_api_stat ();

private:
  // Everyone else can use the getters
  typedef enum
  {
    CUDA_API_STATE_UNINITIALIZED,
    CUDA_API_STATE_INITIALIZING,
    CUDA_API_STATE_INITIALIZED,
  } cuda_api_state_t;

  static void cuda_api_trace (const char *fmt, ...) ATTRIBUTE_PRINTF (1, 2);
  static void cuda_api_error (CUDBGResult res, const char *fmt, ...) ATTRIBUTE_PRINTF (2, 3);
  static void cuda_api_fatal (const char *msg, CUDBGResult res);
  static void cuda_api_print_api_call_result (const char *function, CUDBGResult res);

  // The singleton instance
  static cuda_debugapi s_instance;

  cuda_debugapi_version m_api_version;
  CUDBGAPI m_cudbgAPI;

  int  m_api_ptid;
  cuda_api_state_t m_api_state;
  cuda_attach_state_t m_attach_state;

  // Profiling interface
  // Ensure we can use offsetof on CUDBGAPI. This assumes that the struct
  // CUDBGAPI_st will only have function pointers and no other data members.
  gdb_static_assert (
      std::is_trivial<std::remove_pointer<CUDBGAPI>::type>::value == true);
  static std::array<cuda_api_stat, sizeof (*m_cudbgAPI) / sizeof (uintptr_t)>
      s_api_call_stats;
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

#endif
