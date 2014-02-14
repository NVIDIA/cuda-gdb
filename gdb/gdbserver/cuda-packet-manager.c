/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2013 NVIDIA Corporation
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

#include <sys/stat.h>
#include <unistd.h>
#include "server.h"
#include "cudadebugger.h"
#include "cuda-tdep-server.h"
#include "../cuda-notifications.h"
#include "../cuda-packet-manager.h"
#include "../cuda-utils.h"
#include "../cuda-events.h"
#include "../libcudbgipc.h"

#define TEXTURE_DIM_MAX 4

extern CUDBGAPI cudbgAPI;
extern bool cuda_initialized;
extern ptid_t last_ptid;
extern struct target_waitstatus last_ws;
char *buf_head = NULL;


char *
append_string (const char *src, char *dest, bool sep)
{
  char *p;

  if (dest + strlen (src) - buf_head >= PBUFSIZ)
    error ("Exceed the size of cuda packet.\n");

  sprintf (dest, "%s", src);
  p = strchr (dest, '\0');
  if (sep)
    {
      *p = ';';
      *(++p) = '\0';
    }
  return p;
}

char *
append_bin (const unsigned char *src, char *dest, int size, bool sep)
{
  char *p;

  if (dest + size * 2 - buf_head >= PBUFSIZ)
    error ("Exceed the size of cuda packet.\n");

  convert_int_to_ascii (src, dest, size);
  p = strchr (dest, '\0');
  if (sep)
    {
      *p = ';';
      *(++p) = '\0';
    }
  return p;
}

char *
extract_string (char *src)
{
  return strtok (src, ";");
}

char *
extract_bin (char *src, unsigned char *dest, int size)
{
  char *p;

  p = extract_string (src);
  if (!p)
    error ("The data in the cuda packet is not complete.\n");
  convert_ascii_to_int (p, dest, size);
  return p;
}


static uint64_t cuda_resumed_devices_mask = 0LL;

bool
cuda_has_resumed_devices (void)
{
  return cuda_resumed_devices_mask != 0;
}

void
cuda_process_suspend_device_packet (char *buf)
{
  CUDBGResult res;
  uint32_t dev;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  res = cudbgAPI->suspendDevice (dev);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);

  if (res == CUDBG_SUCCESS && dev < 64)
    cuda_resumed_devices_mask &= ~(1<<dev);
}


void
cuda_process_resume_device_packet (char *buf)
{
  CUDBGResult res;
  uint32_t dev;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  res = cudbgAPI->resumeDevice (dev);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);

  if (res == CUDBG_SUCCESS && dev < 64)
    cuda_resumed_devices_mask |= (1<<dev);
}

void
cuda_process_disassemble_packet (char *buf)
{
  CUDBGResult res;
  uint32_t dev;
  uint64_t addr;
  uint32_t inst_buf_size;
  uint32_t inst_size;
  char *inst_buf;
  char *p;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &inst_buf_size, sizeof (inst_buf_size));

  inst_buf = xmalloc (inst_buf_size);
  res = cudbgAPI->disassemble (dev, addr, &inst_size, inst_buf, inst_buf_size);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &inst_size, buf, sizeof (inst_size), true);
  p = append_bin ((unsigned char *) inst_buf, p, inst_buf_size, false);
  xfree (inst_buf);
}

void
cuda_process_set_breakpoint_packet (char *buf)
{
  CUDBGResult res;
  uint32_t dev;
  uint64_t addr;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  res = cudbgAPI->setBreakpoint (dev, addr);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

void
cuda_process_unset_breakpoint_packet (char *buf)
{
  CUDBGResult res;
  uint32_t dev;
  uint64_t addr;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));

  res = cudbgAPI->unsetBreakpoint (dev, addr);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

void
cuda_process_get_adjusted_code_address (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint64_t addr;
  uint64_t adjusted_addr;
  CUDBGAdjAddrAction adj_action;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &adj_action, sizeof (adj_action));

  res = cudbgAPI->getAdjustedCodeAddress (dev, addr, &adjusted_addr, adj_action);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &adjusted_addr, p, sizeof (adjusted_addr), false);
}

void
cuda_process_get_host_addr_from_device_addr_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint64_t addr;
  uint64_t hostaddr;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));

  res = cudbgAPI->getHostAddrFromDeviceAddr (dev, addr, &hostaddr);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &hostaddr, p, sizeof (hostaddr), false);
}

void
cuda_process_memcheck_read_error_address_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint64_t address;
  ptxStorageKind storage;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,  sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,  sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,  sizeof (ln));

  res = cudbgAPI->memcheckReadErrorAddress (dev, sm, wp, ln, &address, &storage);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &address, p, sizeof (address), true);
  p = append_bin ((unsigned char *) &storage, p, sizeof (storage), false);
}

void
cuda_process_update_grid_id_in_sm_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint64_t valid_warps_mask = 0;
  uint32_t num_warps;
  uint64_t grid_id;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,  sizeof (sm));
  extract_bin (NULL, (unsigned char *) &num_warps, sizeof (num_warps));

  res = cudbgAPI->readValidWarps (dev, sm, &valid_warps_mask);
  p = append_bin ((unsigned char *) &valid_warps_mask, buf, sizeof (valid_warps_mask), true);
  for (wp = 0; wp < num_warps; wp++)
    {
      if (valid_warps_mask & (1ULL << wp))
        {
          if (res == CUDBG_SUCCESS)
            res = cudbgAPI->readGridId (dev, sm, wp, &grid_id);
          p = append_bin ((unsigned char *) &grid_id, p, sizeof (grid_id), true);
        }
    }
  p = append_bin ((unsigned char *) &res, p, sizeof (res), false);
}

void
cuda_process_update_block_idx_in_sm_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint64_t valid_warps_mask = 0;
  uint32_t num_warps;
  CuDim3 block_idx;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm, sizeof (sm));
  extract_bin (NULL, (unsigned char *) &num_warps, sizeof (num_warps));

  res = cudbgAPI->readValidWarps (dev, sm, &valid_warps_mask);
  p = append_bin ((unsigned char *) &valid_warps_mask, buf, sizeof (valid_warps_mask), true);
  for (wp = 0; wp < num_warps; wp++)
    {
      if (valid_warps_mask & (1ULL << wp))
        {
          if (res == CUDBG_SUCCESS)
            res = cudbgAPI->readBlockIdx (dev, sm, wp, &block_idx);
          p = append_bin ((unsigned char *) &block_idx, p, sizeof (block_idx), true);
        }
    }
  p = append_bin ((unsigned char *) &res, p, sizeof (res), false);
}

void
cuda_process_update_thread_idx_in_warp_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint32_t valid_lanes_mask;
  uint32_t num_lanes;
  CuDim3 thread_idx;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,  sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,  sizeof (wp));
  extract_bin (NULL, (unsigned char *) &num_lanes, sizeof (num_lanes));

  res = cudbgAPI->readValidLanes (dev, sm, wp, &valid_lanes_mask);
  p = append_bin ((unsigned char *) &valid_lanes_mask, buf, sizeof (valid_lanes_mask), true);
  for (ln = 0; ln < num_lanes; ln++)
    {
      if (valid_lanes_mask & (1 << ln))
        {
          if (res == CUDBG_SUCCESS)
            res = cudbgAPI->readThreadIdx (dev, sm, wp, ln, &thread_idx);
          p = append_bin ((unsigned char *) &thread_idx, p, sizeof (thread_idx), true);
        }
    }
  p = append_bin ((unsigned char *) &res, p, sizeof (res), false);
}

void
cuda_process_notification_analyze_packet (char *buf)
{
  int trap_expected;

  extract_bin (NULL, (unsigned char *) &trap_expected, sizeof (trap_expected));
  cuda_notification_analyze (last_ptid, &last_ws, trap_expected);
  append_string ("OK", buf, false);
}

void
cuda_process_notification_received_packet (char *buf)
{
  bool received;

  received = cuda_notification_received ();
  append_bin ((unsigned char *) &received, buf, sizeof (received), false);
}

void
cuda_process_notification_pending_packet (char *buf)
{
  bool pending;

  pending = cuda_notification_pending ();
  append_bin ((unsigned char *) &pending, buf, sizeof (pending), false);
}

void
cuda_process_notification_mark_consumed_packet (char *buf)
{
  cuda_notification_mark_consumed ();
  append_string ("OK", buf, false);
}

void
cuda_process_notification_consume_pending_packet (char *buf)
{
  cuda_notification_consume_pending ();
  append_string ("OK", buf, false);
}

void
cuda_process_notification_aliased_event_packet (char *buf)
{
  bool aliased_event;

  aliased_event = cuda_notification_aliased_event ();
  if (aliased_event)
    cuda_notification_reset_aliased_event ();
  append_bin ((unsigned char *) &aliased_event, buf, sizeof (aliased_event), false);
}

void
cuda_process_single_step_warp_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint64_t warp_mask;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm, sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp, sizeof (wp));
  extract_bin (NULL, (unsigned char *) &warp_mask, sizeof (warp_mask));
  res = cudbgAPI->singleStepWarp (dev, sm, wp, &warp_mask);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &warp_mask, p, sizeof (warp_mask), false);
}

void
cuda_process_initialize_target_packet (char *buf)
{
  char *p;
  bool driver_is_compatible;

  extract_bin (NULL, (unsigned char *) &cuda_software_preemption, sizeof (cuda_software_preemption));
  extract_bin (NULL, (unsigned char *) &cuda_memcheck, sizeof (cuda_memcheck));
  extract_bin (NULL, (unsigned char *) &cuda_launch_blocking, sizeof (cuda_launch_blocking));

  driver_is_compatible = cuda_initialize_target ();

  p = append_bin ((unsigned char *) &get_debugger_api_res, buf, sizeof (get_debugger_api_res), true);
  p = append_bin ((unsigned char *) &set_callback_api_res, p, sizeof (set_callback_api_res), true);
  p = append_bin ((unsigned char *) &api_initialize_res, p, sizeof (api_initialize_res), true);
  p = append_bin ((unsigned char *) &cuda_initialized, p, sizeof (cuda_initialized), true);
  p = append_bin ((unsigned char *) &cuda_debugging_enabled, p, sizeof (cuda_debugging_enabled), true);
  p = append_bin ((unsigned char *) &driver_is_compatible, p, sizeof (driver_is_compatible), false);
}

void
cuda_process_get_num_devices_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t num_dev;

  res = cudbgAPI->getNumDevices (&num_dev);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &num_dev, p, sizeof (num_dev), false);
}

void
cuda_process_query_device_spec_packet (char *buf)
{
  char *p;
  CUDBGResult res;
  char device_type[256];
  char sm_type[16];
  uint32_t dev;
  uint32_t num_sms = 0;
  uint32_t num_warps = 0;
  uint32_t num_lanes = 0;
  uint32_t num_registers = 0;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));

  res = cudbgAPI->getNumSMs (dev, &num_sms);
  if (res == CUDBG_SUCCESS) 
    res = cudbgAPI->getNumWarps (dev, &num_warps);
  if (res == CUDBG_SUCCESS) 
    res = cudbgAPI->getNumLanes (dev, &num_lanes);
  if (res == CUDBG_SUCCESS) 
    res = cudbgAPI->getNumRegisters (dev, &num_registers);
  if (res == CUDBG_SUCCESS) 
    res = cudbgAPI->getDeviceType (dev, device_type, sizeof (device_type));
  if (res == CUDBG_SUCCESS) 
    res = cudbgAPI->getSmType (dev, sm_type, sizeof (sm_type));

  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &num_sms, p, sizeof (num_sms), true);
  p = append_bin ((unsigned char *) &num_warps, p, sizeof (num_warps), true);
  p = append_bin ((unsigned char *) &num_lanes, p, sizeof (num_lanes), true);
  p = append_bin ((unsigned char *) &num_registers, p, sizeof (num_registers), true);
  p = append_string (device_type, p, true);
  p = append_string (sm_type, p, false);
}

void
cuda_process_is_device_code_address_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint64_t addr = 0;
  bool is_device_address;

  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  res = cudbgAPI->isDeviceCodeAddress (addr, &is_device_address);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &is_device_address, p, sizeof (is_device_address), false);
}

void
cuda_process_get_grid_status_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint64_t grid_id;
  CUDBGGridStatus status;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &grid_id, sizeof (grid_id));

  res = cudbgAPI->getGridStatus (dev, grid_id, &status);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &status, p, sizeof (status), false);
}

void
cuda_process_get_grid_info_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint64_t grid_id;
  CUDBGGridInfo info;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &grid_id, sizeof (grid_id));

  res = cudbgAPI->getGridInfo (dev, grid_id, &info);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &info, p, sizeof (info), false);
}

void
cuda_process_read_grid_id_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint64_t grid_id;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm, sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp, sizeof (wp));

  res = cudbgAPI->readGridId (dev, sm, wp, &grid_id);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &grid_id, p, sizeof (grid_id), false);
}

void
cuda_process_read_block_idx_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  CuDim3 block_idx;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,  sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,  sizeof (wp));

  res = cudbgAPI->readBlockIdx (dev, sm, wp, &block_idx);  
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &block_idx, p, sizeof (block_idx), false);
}

void
cuda_process_read_thread_idx_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  CuDim3 thread_idx;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,  sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,  sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,  sizeof (ln));

  res = cudbgAPI->readThreadIdx (dev, sm, wp, ln, &thread_idx);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &thread_idx, p, sizeof (thread_idx), false);
}

void
cuda_process_read_broken_warps_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint64_t broken_warps_mask;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm, sizeof (sm));

  res = cudbgAPI->readBrokenWarps (dev, sm, &broken_warps_mask);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &broken_warps_mask, p, sizeof (broken_warps_mask), false);
}

void
cuda_process_read_valid_warps_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint64_t valid_warps_mask;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm, sizeof (sm));

  res = cudbgAPI->readValidWarps (dev, sm, &valid_warps_mask);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &valid_warps_mask, p, sizeof (valid_warps_mask), false);
}

void
cuda_process_read_valid_lanes_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t valid_lanes_mask;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm, sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp, sizeof (wp));

  res = cudbgAPI->readValidLanes (dev, sm, wp, &valid_lanes_mask);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &valid_lanes_mask, p, sizeof (valid_lanes_mask), false);
}

void
cuda_process_read_active_lanes_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t active_lanes_mask;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm, sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp, sizeof (wp));

  res = cudbgAPI->readActiveLanes (dev, sm, wp, &active_lanes_mask);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &active_lanes_mask, p, sizeof (active_lanes_mask), false);
}

void
cuda_process_read_virtual_pc_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint64_t value;
 
  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm, sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp, sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln, sizeof (ln));

  res = cudbgAPI->readVirtualPC (dev, sm, wp, ln, &value);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &value, p, sizeof (value), false);
}

void
cuda_process_read_pc_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint64_t value;
 
  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,  sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,  sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,  sizeof (ln));

  res = cudbgAPI->readPC (dev, sm, wp, ln, &value);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &value, p, sizeof (value), false);
}

void
cuda_process_read_register_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  int regno;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint32_t value;

  extract_bin (NULL, (unsigned char *) &dev,   sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,    sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,    sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,    sizeof (ln));
  extract_bin (NULL, (unsigned char *) &regno, sizeof (regno));

  res = cudbgAPI->readRegister (dev, sm, wp, ln, regno, &value);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &value, p, sizeof (value), false);
}


void
cuda_process_read_lane_exception_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  CUDBGException_t exception;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,  sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,  sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,  sizeof (ln));

  res = cudbgAPI->readLaneException (dev, sm, wp, ln, &exception);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &exception, p, sizeof (exception), false);
}

void
cuda_process_read_call_depth_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint32_t value;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,  sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,  sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,  sizeof (ln));

  res = cudbgAPI->readCallDepth (dev, sm, wp, ln, &value);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &value, p, sizeof (value), false);;
}

void
cuda_process_read_syscall_call_depth_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint32_t value;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,  sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,  sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,  sizeof (ln));

  res = cudbgAPI->readSyscallCallDepth (dev, sm, wp, ln, &value);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &value, p, sizeof (value), false);;
}

void
cuda_process_read_virtual_return_address_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint32_t level;
  uint64_t value;

  extract_bin (NULL, (unsigned char *) &dev,   sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,    sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,    sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,    sizeof (ln));
  extract_bin (NULL, (unsigned char *) &level, sizeof (level));

  res = cudbgAPI->readVirtualReturnAddress (dev, sm, wp, ln, level, &value);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &value, p, sizeof (value), false);
}

void
cuda_process_read_code_memory_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  res = cudbgAPI->readCodeMemory (dev, addr, value, sz);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) value, p, sz, false);
  xfree (value);
}

void
cuda_process_read_const_memory_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  res = cudbgAPI->readConstMemory (dev, addr, value, sz);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) value, p, sz, false);
  xfree (value);
}

void
cuda_process_read_generic_memory_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,   sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,   sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,   sizeof (ln));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  res = cudbgAPI->readGenericMemory (dev, sm, wp, ln, addr, value, sz);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) value, p, sz, false);
  xfree (value);
}

void
cuda_process_read_pinned_memory_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  res = cudbgAPI->readPinnedMemory (addr, value, sz);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) value, p, sz, false);
  xfree (value);
}

void
cuda_process_read_param_memory_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,   sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,   sizeof (wp));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  res = cudbgAPI->readParamMemory (dev, sm, wp, addr, value, sz);  
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) value, p, sz, false);
  xfree (value);
}

void
cuda_process_read_shared_memory_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,   sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,   sizeof (wp));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  res = cudbgAPI->readSharedMemory (dev, sm, wp, addr, value, sz);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) value, p, sz, false);
  xfree (value);
}

void
cuda_process_read_texture_memory_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t id;
  uint32_t dim;
  uint32_t coords[TEXTURE_DIM_MAX];
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,   sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,    sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,    sizeof (wp));
  extract_bin (NULL, (unsigned char *) &id,    sizeof (id));
  extract_bin (NULL, (unsigned char *) &dim,   sizeof (dim));
  extract_bin (NULL, (unsigned char *) coords, sizeof (coords[0]) * TEXTURE_DIM_MAX);
  extract_bin (NULL, (unsigned char *) &sz,    sizeof (sz));

  value = xmalloc (sz);
  res = cudbgAPI->readTextureMemory (dev, sm, wp, id, dim, coords, value, sz);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) value, p, sz, false);
  xfree (value);
}

void
cuda_process_read_texture_memory_bindless_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t tex_symtab_index;
  uint32_t dim;
  uint32_t coords[TEXTURE_DIM_MAX];
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,   sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,    sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,    sizeof (wp));
  extract_bin (NULL, (unsigned char *) &tex_symtab_index, sizeof (tex_symtab_index));
  extract_bin (NULL, (unsigned char *) &dim,   sizeof (dim));
  extract_bin (NULL, (unsigned char *) coords, sizeof (coords[0]) * TEXTURE_DIM_MAX);
  extract_bin (NULL, (unsigned char *) &sz,    sizeof (sz));

  value = xmalloc (sz);
  res = cudbgAPI->readTextureMemoryBindless (dev, sm, wp, tex_symtab_index, dim, coords, value, sz);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) value, p, sz, false);
  xfree (value);
}

void
cuda_process_read_local_memory_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,   sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,   sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,   sizeof (ln));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  res = cudbgAPI->readLocalMemory (dev, sm, wp, ln, addr, value, sz);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) value, p, sz, false);
  xfree (value);
}

void
cuda_process_write_generic_memory_packet (char *buf)
{
  CUDBGResult res;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,   sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,   sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,   sizeof (ln));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  extract_bin (NULL, (unsigned char *) value, sz);

  res = cudbgAPI->writeGenericMemory (dev, sm, wp, ln, addr, value, sz);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
  xfree (value);
}

void
cuda_process_write_pinned_memory_packet (char *buf)
{
  CUDBGResult res;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  extract_bin (NULL, (unsigned char *) value, sz);
  res = cudbgAPI->writePinnedMemory (addr, value, sz);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
  xfree (value);
}

void
cuda_process_write_param_memory_packet (char *buf)
{
  CUDBGResult res;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,   sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,   sizeof (wp));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  extract_bin (NULL, (unsigned char *) value, sz);

  res = cudbgAPI->writeParamMemory (dev, sm, wp, addr, value, sz);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
  xfree (value);
}

void
cuda_process_write_shared_memory_packet (char *buf)
{
  CUDBGResult res;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,   sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,   sizeof (wp));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  extract_bin (NULL, (unsigned char *) value, sz);

  res = cudbgAPI->writeSharedMemory (dev, sm, wp, addr, value, sz);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
  xfree (value);
}

void
cuda_process_write_local_memory_packet (char *buf)
{
  CUDBGResult res;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint64_t addr;
  uint32_t sz;
  void *value;

  extract_bin (NULL, (unsigned char *) &dev,  sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,   sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,   sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,   sizeof (ln));
  extract_bin (NULL, (unsigned char *) &addr, sizeof (addr));
  extract_bin (NULL, (unsigned char *) &sz,   sizeof (sz));

  value = xmalloc (sz);
  extract_bin (NULL, (unsigned char *) value, sz);

  res = cudbgAPI->writeLocalMemory (dev, sm, wp, ln, addr, value, sz);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
  xfree (value);
}

void
cuda_process_write_register_packet (char *buf)
{
  CUDBGResult res;
  int regno;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint32_t value;

  extract_bin (NULL, (unsigned char *) &dev,   sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,    sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,    sizeof (wp));
  extract_bin (NULL, (unsigned char *) &ln,    sizeof (ln));
  extract_bin (NULL, (unsigned char *) &regno, sizeof (regno));
  extract_bin (NULL, (unsigned char *) &value, sizeof (value));

  res = cudbgAPI->writeRegister (dev, sm, wp, ln, regno, value);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

void
cuda_process_check_pending_sigint_packet (char *buf)
{
  bool ret_val;

  ret_val = cuda_check_pending_sigint (last_ptid);
  append_bin ((unsigned char *) &ret_val, buf, sizeof (ret_val), false);
}

void
cuda_process_api_initialize_packet (char *buf)
{
  CUDBGResult res;

  res = cudbgAPI->initialize ();
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

void
cuda_process_api_finalize_packet (char *buf)
{
  CUDBGResult res;

  /* If finalize() has been called in cuda_cleanup(), then return the
     recorded cudbgAPI result. */
  if (cuda_initialized)
    res = cudbgAPI->finalize ();
  else
    res = api_finalize_res;
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

void
cuda_process_api_request_clear_attach_state (char *buf)
{
  CUDBGResult res;
  res = cudbgAPI->clearAttachState ();
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

void
cuda_process_api_request_cleanup_on_detach_packet (char *buf)
{
  CUDBGResult res;
  uint32_t resumeAppFlag;

  extract_bin (NULL, (unsigned char *) &resumeAppFlag, sizeof (resumeAppFlag));

  res = cudbgAPI->requestCleanupOnDetach (resumeAppFlag);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

void
cuda_process_set_option_packet (char *buf)
{
  extract_bin (NULL, (unsigned char *) &cuda_debug_general,       sizeof (cuda_debug_general));
  extract_bin (NULL, (unsigned char *) &cuda_debug_libcudbg,      sizeof (cuda_debug_libcudbg));
  extract_bin (NULL, (unsigned char *) &cuda_debug_notifications, sizeof (cuda_debug_notifications));
  extract_bin (NULL, (unsigned char *) &cuda_notify_youngest,     sizeof (cuda_notify_youngest));

  append_string ("OK", buf, false);
}

void
cuda_process_set_async_launch_notifications (char *buf)
{
  CUDBGResult res;
  uint32_t mode;
  extract_bin (NULL, (unsigned char *) &mode, sizeof (mode));

  res = cudbgAPI->setKernelLaunchNotificationMode (mode);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

void
cuda_process_api_read_device_exception_state (char *buf)
{
  CUDBGResult res;
  uint32_t dev;
  uint64_t smMask = 0;
  char *p;

  extract_bin (NULL, (unsigned char*) &dev, sizeof (dev));

  res = cudbgAPI->readDeviceExceptionState (dev, &smMask);

  p = append_bin ((unsigned char*) &res, buf, sizeof (res), true);
  append_bin ((unsigned char*) &smMask, p, sizeof (smMask), false);
}

void
cuda_process_query_trace_message (char *buf)
{
  struct cuda_trace_msg *msg;

  if (!cuda_first_trace_msg)
    {
      append_string ("NO_TRACE_MESSAGE", buf, false);
      return;
    }

  append_string (cuda_first_trace_msg->buf, buf, false);
  msg = cuda_first_trace_msg->next;
  xfree (cuda_first_trace_msg);
  cuda_first_trace_msg = msg;
  if (!cuda_first_trace_msg)
    cuda_last_trace_msg = NULL;
}

void
handle_cuda_packet (char *buf)
{
  cuda_packet_type_t packet_type;
  buf_head = buf;
  extract_bin (buf + strlen ("qnv."), (unsigned char *) &packet_type, sizeof (packet_type));

  switch (packet_type)
    {
    case RESUME_DEVICE:
      cuda_process_resume_device_packet (buf);
      break;
    case SUSPEND_DEVICE:
      cuda_process_suspend_device_packet (buf);
      break;
    case SINGLE_STEP_WARP:
      cuda_process_single_step_warp_packet (buf);
      break;
    case SET_BREAKPOINT:
      cuda_process_set_breakpoint_packet (buf);
      break;
    case UNSET_BREAKPOINT:
      cuda_process_unset_breakpoint_packet (buf);
      break;
    case READ_GRID_ID:
      cuda_process_read_grid_id_packet (buf);
      break;
    case READ_BLOCK_IDX:
      cuda_process_read_block_idx_packet (buf);
      break;
    case READ_THREAD_IDX:
      cuda_process_read_thread_idx_packet (buf);
      break;
    case READ_BROKEN_WARPS:
      cuda_process_read_broken_warps_packet (buf);
      break;
    case READ_VALID_WARPS:
      cuda_process_read_valid_warps_packet (buf);
      break;
    case READ_VALID_LANES:
      cuda_process_read_valid_lanes_packet (buf);
      break;
    case READ_ACTIVE_LANES:
      cuda_process_read_active_lanes_packet (buf);
      break;
    case READ_CODE_MEMORY:
      cuda_process_read_code_memory_packet (buf);
      break;
    case READ_CONST_MEMORY:
      cuda_process_read_const_memory_packet (buf);
      break;
    case READ_GENERIC_MEMORY:
      cuda_process_read_generic_memory_packet (buf);
      break;
    case READ_PINNED_MEMORY:
      cuda_process_read_pinned_memory_packet (buf);
      break;
    case READ_PARAM_MEMORY:
      cuda_process_read_param_memory_packet (buf);
      break;
    case READ_SHARED_MEMORY:
      cuda_process_read_shared_memory_packet (buf);
      break;
    case READ_TEXTURE_MEMORY:
      cuda_process_read_texture_memory_packet (buf);
      break;
    case READ_TEXTURE_MEMORY_BINDLESS:
      cuda_process_read_texture_memory_bindless_packet (buf);
      break;
    case READ_LOCAL_MEMORY:
      cuda_process_read_local_memory_packet (buf);
      break;
    case READ_REGISTER:
      cuda_process_read_register_packet (buf);
      break;
    case READ_PC:
      cuda_process_read_pc_packet (buf);
      break;
    case READ_VIRTUAL_PC:
      cuda_process_read_virtual_pc_packet (buf);
      break;
    case READ_LANE_EXCEPTION:
      cuda_process_read_lane_exception_packet (buf);
      break;
    case READ_CALL_DEPTH:
      cuda_process_read_call_depth_packet (buf);
      break;
    case READ_SYSCALL_CALL_DEPTH:
      cuda_process_read_syscall_call_depth_packet (buf);
      break;
    case READ_VIRTUAL_RETURN_ADDRESS:
      cuda_process_read_virtual_return_address_packet (buf);
      break;
    case WRITE_GENERIC_MEMORY:
      cuda_process_write_generic_memory_packet (buf);
      break;
    case WRITE_PINNED_MEMORY:
      cuda_process_write_pinned_memory_packet (buf);
      break;
    case WRITE_PARAM_MEMORY:
      cuda_process_write_param_memory_packet (buf);
      break;
    case WRITE_SHARED_MEMORY:
      cuda_process_write_shared_memory_packet (buf);
      break;
    case WRITE_LOCAL_MEMORY:
      cuda_process_write_local_memory_packet (buf);
      break;
    case WRITE_REGISTER:
      cuda_process_write_register_packet (buf);
      break;
    case IS_DEVICE_CODE_ADDRESS:
      cuda_process_is_device_code_address_packet (buf);
      break;
    case DISASSEMBLE:
      cuda_process_disassemble_packet (buf);
      break;
    case MEMCHECK_READ_ERROR_ADDRESS:
      cuda_process_memcheck_read_error_address_packet (buf);
      break;
    case GET_NUM_DEVICES:
      cuda_process_get_num_devices_packet (buf);
      break;
    case GET_GRID_STATUS:
      cuda_process_get_grid_status_packet (buf);
      break;
    case GET_GRID_INFO:
      cuda_process_get_grid_info_packet (buf);
      break;
    case GET_ADJUSTED_CODE_ADDRESS:
      cuda_process_get_adjusted_code_address (buf);
      break;
    case GET_HOST_ADDR_FROM_DEVICE_ADDR:
      cuda_process_get_host_addr_from_device_addr_packet (buf);
      break;
    case NOTIFICATION_ANALYZE:
      cuda_process_notification_analyze_packet (buf);
      break;
    case NOTIFICATION_PENDING:
      cuda_process_notification_pending_packet (buf);
      break;
    case NOTIFICATION_RECEIVED:
      cuda_process_notification_received_packet (buf);
      break;
    case NOTIFICATION_ALIASED_EVENT:
      cuda_process_notification_aliased_event_packet (buf);
      break;
    case NOTIFICATION_MARK_CONSUMED:
      cuda_process_notification_mark_consumed_packet (buf);
      break;
    case NOTIFICATION_CONSUME_PENDING:
      cuda_process_notification_consume_pending_packet (buf);
      break;
    case UPDATE_GRID_ID_IN_SM:
      cuda_process_update_grid_id_in_sm_packet (buf);
      break;
    case UPDATE_BLOCK_IDX_IN_SM:
      cuda_process_update_block_idx_in_sm_packet (buf);
      break;
    case UPDATE_THREAD_IDX_IN_WARP:
      cuda_process_update_thread_idx_in_warp_packet (buf);
      break;
    case INITIALIZE_TARGET:
      cuda_process_initialize_target_packet (buf);
      break;
    case QUERY_DEVICE_SPEC:
      cuda_process_query_device_spec_packet (buf);
      break;
    case QUERY_TRACE_MESSAGE:
      cuda_process_query_trace_message (buf);
      break;
    case CHECK_PENDING_SIGINT:
      cuda_process_check_pending_sigint_packet (buf);
      break;
    case API_INITIALIZE:
      cuda_process_api_initialize_packet (buf);
      break;
    case API_FINALIZE:
      cuda_process_api_finalize_packet (buf);
      break;
    case CLEAR_ATTACH_STATE:
      cuda_process_api_request_clear_attach_state (buf);
      break;
    case REQUEST_CLEANUP_ON_DETACH:
      cuda_process_api_request_cleanup_on_detach_packet (buf);
      break;
    case SET_OPTION:
      cuda_process_set_option_packet (buf);
      break;
    case SET_ASYNC_LAUNCH_NOTIFICATIONS:
      cuda_process_set_async_launch_notifications (buf);
      break;
    case READ_DEVICE_EXCEPTION_STATE:
      cuda_process_api_read_device_exception_state (buf);
      break;
    default:
      error ("unknown cuda packet.\n");
      break;
    }
}

void
cuda_append_api_finalize_res (char *buf)
{
  gdb_assert (buf);
  sprintf (buf, ";cuda_finalize:%x", api_finalize_res);
}

/*
 * Regardless of the packet_len, buf can contain up to PBUFSIZ bytes
 */
int handle_vCuda (char *buf, int packet_len, int *new_packet_len)
{
  CUDBGResult res;
  gdb_byte *lbuf;
  static void *data = NULL;
  static size_t size = 0;
  int out_len;
  size_t offset = 0;

  /* Handle multipacket vCUDARetr; command */
  if (strncmp (buf, "vCUDARetr;", 10) == 0) {
    offset = (size_t) atol (buf + strlen ("vCUDARetr;"));

    if (offset >= size) {
      sprintf (buf, "E%02d", EINVAL);
      *new_packet_len = 3;
      return 1;
    }

    memcpy (buf, "OK;", strlen("OK;"));
    lbuf = (gdb_byte *)buf + strlen("OK;");
    *new_packet_len  = strlen ("OK;");
    *new_packet_len += remote_escape_output ((const gdb_byte *)data+offset,
                                             size-offset, lbuf,
                                             &out_len, PBUFSIZ-strlen ("OK;"));
    if (out_len != size - offset)
      memcpy (buf, "MP", 2);

    return 1;
  }

  /* Handle vCUDA; command */
  if (strncmp (buf, "vCUDA;", 6) != 0) {
    sprintf (buf, "E%02d", EINVAL);
    *new_packet_len = 3;
    return 1;
  }

  lbuf = (gdb_byte *)buf + strlen("vCUDA;");
  packet_len -= strlen ("vCUDA;");

  data = xmalloc (packet_len);
  gdb_assert (data);

  packet_len = remote_unescape_input (lbuf, packet_len, data, packet_len);
  res = cudbgipcAppend (data, packet_len);
  if (res != CUDBG_SUCCESS) {
      sprintf (buf, "E%02d", res);
      *new_packet_len = 3;
      return 1;
  }
  xfree (data);

  res = cudbgipcRequest (&data, &size);
  if (res != CUDBG_SUCCESS) {
      sprintf (buf, "E%02d", res);
      *new_packet_len = 3;
      return 1;
  }

  memcpy (buf, "OK;", strlen("OK;"));
  lbuf = (gdb_byte *)buf + strlen("OK;");
  *new_packet_len  = strlen ("OK;");
  *new_packet_len += remote_escape_output (data, size, lbuf,
                                          &out_len, PBUFSIZ-strlen ("OK;"));
  if (out_len != size)
    memcpy (buf, "MP", 2);

  return 1;
}
