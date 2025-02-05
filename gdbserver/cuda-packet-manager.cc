/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2013-2024 NVIDIA Corporation
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

#ifndef __QNXHOST__
#include <sys/stat.h>
#endif
#include "server.h"
#include <unistd.h>
#include "cudadebugger.h"
#include "cuda-tdep-server.h"
#include "cuda/cuda-notifications.h"
#include "cuda/cuda-packet-manager.h"
#include "cuda/cuda-utils.h"
#include "cuda/cuda-events.h"
#include "cuda/libcudbgipc.h"
#ifdef __QNXHOST__
# include "remote-nto.h"
#endif /* __QNXHOST__ */
#include "gdbsupport/rsp-low.h"

#ifdef __QNXHOST__
/* On QNX the packet size is much smaller */
# undef PBUFSIZ
# define PBUFSIZ DS_DATA_MAX_SIZE
#endif

#ifndef __QNXHOST__
/* We don't have gdbserver-managed structures on QNX */
extern ptid_t cuda_last_ptid;
extern struct target_waitstatus cuda_last_ws;
#endif
char *buf_head = NULL;

static uint32_t cuda_debugapi_version_major;
static uint32_t cuda_debugapi_version_minor;
static uint32_t cuda_debugapi_version_revision;

extern void
cuda_gdbserver_set_api_version (uint32_t major, uint32_t minor, uint32_t revision);

void
cuda_gdbserver_set_api_version (uint32_t major, uint32_t minor, uint32_t revision)
{
  cuda_debugapi_version_major = major;
  cuda_debugapi_version_minor = minor;
  cuda_debugapi_version_revision = revision;
}

static char *
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

static char *
append_bin (const unsigned char *src, char *dest, int size, bool sep)
{
  char *p;

  if (dest + size * 2 - buf_head >= PBUFSIZ)
    error ("Exceed the size of cuda packet.\n");

  bin2hex (src, dest, size);
  p = strchr (dest, '\0');
  if (sep)
    {
      *p = ';';
      *(++p) = '\0';
    }
  return p;
}

static char *
extract_string (char *src)
{
  return strtok (src, ";");
}

static char *
extract_bin (char *src, unsigned char *dest, int size)
{
  char *p;

  p = extract_string (src);
  if (!p)
    error ("The data in the cuda packet is not complete (cuda-gdbserver).\n");
  hex2bin (p, dest, size);
  return p;
}

static void
cuda_process_suspend_device_packet (char *buf)
{
  CUDBGResult res;
  uint32_t dev;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  res = cudbgAPI->suspendDevice (dev);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}


static void
cuda_process_resume_device_packet (char *buf)
{
  CUDBGResult res;
  uint32_t dev;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  res = cudbgAPI->resumeDevice (dev);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

static void
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

  inst_buf = (char *) xmalloc (inst_buf_size);
  res = cudbgAPI->disassemble (dev, addr, &inst_size, inst_buf, inst_buf_size);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &inst_size, buf, sizeof (inst_size), true);
  p = append_bin ((unsigned char *) inst_buf, p, inst_buf_size, false);
  xfree (inst_buf);
}

static void
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

static void
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

static void
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

static void
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

static void
cuda_process_get_error_string_ex_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t err_str_len;
  const ptrdiff_t packet_hdr_sz = buf - buf_head;
  const uint32_t max_err_str_len = (uint32_t) (PBUFSIZ - ((size_t) packet_hdr_sz + sizeof (res) + sizeof (err_str_len)));
  void *error_str_buf = xmalloc (max_err_str_len);
  res = cudbgAPI->getErrorStringEx ((char *) error_str_buf, max_err_str_len, &err_str_len);
  if (res != CUDBG_SUCCESS)
    {
      if (res == CUDBG_ERROR_BUFFER_TOO_SMALL)
        {
          err_str_len = max_err_str_len;
        }
      else
        {
          err_str_len = 0;
        }
    }
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &err_str_len, p, sizeof (err_str_len), true);
  p = append_string ((char *) error_str_buf, p, false);
}

#ifndef __QNXHOST__
template <typename TValue>
using FnGetValueFromAPI = CUDBGResult (*)(uint32_t dev, uint32_t sm, uint32_t wp, TValue *value);

template <typename TValue>
static void
cuda_process_update_per_warp_info_in_sm_packet (char *buf, FnGetValueFromAPI<TValue>&& get_value)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint64_t valid_warps_mask = 0;
  uint32_t num_warps;
  TValue value;

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
            res = get_value (dev, sm, wp, &value);
          p = append_bin ((unsigned char *) &value, p, sizeof (value), true);
        }
    }
  p = append_bin ((unsigned char *) &res, p, sizeof (res), false);
}

static void
cuda_process_update_grid_id_in_sm_packet (char *buf)
{
  cuda_process_update_per_warp_info_in_sm_packet<uint64_t> (
    buf, [](uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *gridId64) {
      return cudbgAPI->readGridId (dev, sm, wp, gridId64);
    });
}

static void
cuda_process_update_cluster_idx_in_sm_packet (char *buf)
{
  cuda_process_update_per_warp_info_in_sm_packet<CuDim3> (
    buf, [](uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *cluster_idx) {
      return cudbgAPI->readClusterIdx (dev, sm, wp, cluster_idx);
    });
}

static void
cuda_process_update_cluster_dim_in_sm_packet (char *buf)
{
  cuda_process_update_per_warp_info_in_sm_packet<CuDim3> (
      buf, [] (uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *cluster_dim) {
	/* Need to be backwards compatible with older CUDA drivers. */
	if (cuda_debugapi_version_revision >= 148)
	  return cudbgAPI->getClusterDim (dev, sm, wp, cluster_dim);
	else
	  {
	    uint64_t gridId64;
	    cudbgAPI->readGridId (dev, sm, wp, &gridId64);
	    return cudbgAPI->getClusterDim120 (dev, gridId64, cluster_dim);
	  }
      });
}

static void
cuda_process_update_block_idx_in_sm_packet (char *buf)
{
  cuda_process_update_per_warp_info_in_sm_packet<CuDim3> (
    buf, [](uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *block_idx) {
      return cudbgAPI->readBlockIdx (dev, sm, wp, block_idx);
    });
}
#endif

static void
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

static void
cuda_process_notification_analyze_packet (char *buf)
{
  int trap_expected;

#ifdef __QNXHOST__
  /* On QNX, ptid and ws are passed in from host */
  ptid_t cuda_last_ptid;
  struct target_waitstatus cuda_last_ws;

  extract_bin (NULL, (unsigned char *) &cuda_last_ptid, sizeof (cuda_last_ptid));
  extract_bin (NULL, (unsigned char *) &cuda_last_ws, sizeof (cuda_last_ws));
#endif /* __QNXHOST__ */
  extract_bin (NULL, (unsigned char *) &trap_expected, sizeof (trap_expected));
  cuda_notification_analyze (cuda_last_ptid, &cuda_last_ws, trap_expected);
  append_string ("OK", buf, false);
}

static void
cuda_process_notification_received_packet (char *buf)
{
  bool received;

  received = cuda_notification_received ();
  append_bin ((unsigned char *) &received, buf, sizeof (received), false);
}

static void
cuda_process_notification_pending_packet (char *buf)
{
  bool pending;

  pending = cuda_notification_pending ();
  append_bin ((unsigned char *) &pending, buf, sizeof (pending), false);
}

static void
cuda_process_notification_mark_consumed_packet (char *buf)
{
  cuda_notification_mark_consumed ();
  append_string ("OK", buf, false);
}

static void
cuda_process_notification_consume_pending_packet (char *buf)
{
  cuda_notification_consume_pending ();
  append_string ("OK", buf, false);
}

static void
cuda_process_notification_aliased_event_packet (char *buf)
{
  bool aliased_event;

  aliased_event = cuda_notification_aliased_event ();
  if (aliased_event)
    cuda_notification_reset_aliased_event ();
  append_bin ((unsigned char *) &aliased_event, buf, sizeof (aliased_event), false);
}

static void
cuda_process_single_step_warp_packet65 (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t nsteps;
  uint64_t warp_mask;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm, sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp, sizeof (wp));
  extract_bin (NULL, (unsigned char *) &nsteps, sizeof (nsteps));
  extract_bin (NULL, (unsigned char *) &warp_mask, sizeof (warp_mask));
  res = cudbgAPI->singleStepWarp65 (dev, sm, wp, nsteps, &warp_mask);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &warp_mask, p, sizeof (warp_mask), false);
}

#ifdef __QNXHOST__
static void
cuda_process_set_symbols (char *buf)
{
  bool symbols_are_set = false;
  unsigned char symbols_count;
  CORE_ADDR address;
  int server_symbols_count;

  extract_bin (NULL, (unsigned char *) &symbols_count, sizeof (symbols_count));
  /* For compatibility with newer cuda-gdb binaries we handle packets that
     provide more symbols than we have statically built with. */
  server_symbols_count = cuda_get_symbol_cache_size ();
  if (symbols_count >= server_symbols_count)
    {
      symbols_are_set = true;
      for (int i = 0; i < server_symbols_count; i++)
        {
          extract_bin (NULL, (unsigned char *) &address, sizeof (CORE_ADDR));
          if (address == 0)
            {
              symbols_are_set = false;
              break;
            }
          cuda_symbol_list[i].addr = address;
        }
    }

  append_bin ((unsigned char *) &symbols_are_set, buf, sizeof (symbols_are_set), false);
}
#endif /* __QNXHOST__ */

static void
cuda_process_initialize_target_packet (char *buf)
{
  char *p;
  bool driver_is_compatible;
  bool cuda_memcheck;

  extract_bin (NULL, (unsigned char *) &cuda_software_preemption, sizeof (cuda_software_preemption));
  /* CUDA MEMCHECK support is removed from CUDA GDB: this field is left to maintain
   * the binary compatibility with legacy CUDA GDB server binaries */
  extract_bin (NULL, (unsigned char *) &cuda_memcheck, sizeof (cuda_memcheck));
  extract_bin (NULL, (unsigned char *) &cuda_launch_blocking, sizeof (cuda_launch_blocking));

  driver_is_compatible = cuda_initialize_target ();

  p = append_bin ((unsigned char *) &get_debugger_api_res, buf, sizeof (get_debugger_api_res), true);
  p = append_bin ((unsigned char *) &set_callback_api_res, p, sizeof (set_callback_api_res), true);
  p = append_bin ((unsigned char *) &api_initialize_res, p, sizeof (api_initialize_res), true);
  p = append_bin ((unsigned char *) &cuda_initialized, p, sizeof (cuda_initialized), true);
  p = append_bin ((unsigned char *) &cuda_debugging_enabled, p, sizeof (cuda_debugging_enabled), true);
  p = append_bin ((unsigned char *) &driver_is_compatible, p, sizeof (driver_is_compatible), true);

  p = append_bin ((unsigned char *) &cuda_debugapi_version_major, p,
		  sizeof (cuda_debugapi_version_major), true);

  p = append_bin ((unsigned char *) &cuda_debugapi_version_minor, p,
		  sizeof (cuda_debugapi_version_minor), true);

  p = append_bin ((unsigned char *) &cuda_debugapi_version_revision, p,
		  sizeof (cuda_debugapi_version_revision), false);
}

static void
cuda_process_get_num_devices_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t num_dev;

  res = cudbgAPI->getNumDevices (&num_dev);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &num_dev, p, sizeof (num_dev), false);
}

static void
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

static void
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

static void
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

static void
cuda_process_get_grid_info_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint64_t grid_id;
  CUDBGGridInfo120 info;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &grid_id, sizeof (grid_id));

  res = cudbgAPI->getGridInfo120 (dev, grid_id, &info);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &info, p, sizeof (info), false);
}

static void
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

static void
cuda_process_read_cluster_idx_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  CuDim3 cluster_idx;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm,  sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp,  sizeof (wp));

  res = cudbgAPI->readClusterIdx (dev, sm, wp, &cluster_idx);  
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &cluster_idx, p, sizeof (cluster_idx), false);
}

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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


static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
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

static void
cuda_process_check_pending_sigint_packet (char *buf)
{
  bool ret_val;
#ifdef __QNXHOST__
  /* On QNX, ptid is passed in from host */
  ptid_t cuda_last_ptid;

  extract_bin (NULL, (unsigned char *) &cuda_last_ptid, sizeof (cuda_last_ptid));
#endif
  ret_val = cuda_check_pending_sigint (cuda_last_ptid);
  append_bin ((unsigned char *) &ret_val, buf, sizeof (ret_val), false);
}

static void
cuda_process_api_initialize_packet (char *buf)
{
  CUDBGResult res;

  res = cudbgAPI->initialize ();
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

static void
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

static void
cuda_process_api_request_clear_attach_state (char *buf)
{
  CUDBGResult res;
  res = cudbgAPI->clearAttachState ();
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

static void
cuda_process_api_request_cleanup_on_detach_packet (char *buf)
{
  CUDBGResult res;
  uint32_t resumeAppFlag;

  extract_bin (NULL, (unsigned char *) &resumeAppFlag, sizeof (resumeAppFlag));

  res = cudbgAPI->requestCleanupOnDetach (resumeAppFlag);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

static void
cuda_process_set_option_packet (char *buf)
{
  const char *stop_signal_str = NULL;
  extract_bin (NULL, (unsigned char *) &cuda_debug_general,       sizeof (cuda_debug_general));
  extract_bin (NULL, (unsigned char *) &cuda_debug_libcudbg,      sizeof (cuda_debug_libcudbg));
  extract_bin (NULL, (unsigned char *) &cuda_debug_notifications, sizeof (cuda_debug_notifications));
  extract_bin (NULL, (unsigned char *) &cuda_notify_youngest,     sizeof (cuda_notify_youngest));

  stop_signal_str = extract_string (NULL);
  /* Be lenient towards older clients: if extra argument was not passed, use SIGTRAP */
  cuda_stop_signal = (stop_signal_str == NULL || strcmp (stop_signal_str, "SIGTRAP")==0) ?
                     GDB_SIGNAL_TRAP : GDB_SIGNAL_URG;

  append_string ("OK", buf, false);
}

static void
cuda_process_set_async_launch_notifications (char *buf)
{
  CUDBGResult res;
  uint32_t mode;
  extract_bin (NULL, (unsigned char *) &mode, sizeof (mode));

  res = (CUDBGResult) cudbgAPI->setKernelLaunchNotificationMode ((CUDBGKernelLaunchNotifyMode) mode);
  append_bin ((unsigned char *) &res, buf, sizeof (res), false);
}

static void
cuda_process_api_read_device_exception_state (char *buf)
{
  CUDBGResult res;
  uint32_t dev, sz;
  char *p;
  uint64_t *value;

  extract_bin (NULL, (unsigned char*) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char*) &sz, sizeof (sz));

  value = (uint64_t *) xmalloc (sz * sizeof (*value));
  res = cudbgAPI->readDeviceExceptionState (dev, value, sz);
  p = append_bin ((unsigned char*) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char*) value, p, sz * sizeof (*value), false);
  xfree (value);
}

static void
cuda_process_api_get_device_info_sizes (char *buf)
{
  CUDBGResult res;
  uint32_t dev;
  char *p;
  CUDBGDeviceInfoSizes sizes;

  extract_bin (NULL, (unsigned char*) &dev, sizeof (dev));

  res = cudbgAPI->getDeviceInfoSizes (dev, &sizes);
  p = append_bin ((unsigned char*) &res, buf, sizeof (res), true);
  if (res == CUDBG_SUCCESS)
    p = append_bin ((unsigned char*) &sizes, p, sizeof (CUDBGDeviceInfoSizes), false);
}

static void
cuda_process_api_get_device_info (char *buf)
{
  CUDBGResult res;
  char *p;
  void *buffer;
  uint32_t dev, length, data_length;
  CUDBGDeviceInfoQueryType_t type;

  extract_bin (NULL, (unsigned char*) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char*) &type, sizeof (type));
  extract_bin (NULL, (unsigned char*) &length, sizeof (length));

  buffer = xmalloc (length);
  res = cudbgAPI->getDeviceInfo (dev, type, buffer, length, &data_length);

  if (res == CUDBG_SUCCESS)
    {
      p = append_bin ((unsigned char*) &res, buf, sizeof (res), true);
      p = append_bin ((unsigned char*) &data_length, p, sizeof (data_length), true);
      p = append_bin ((unsigned char*) buffer, p, data_length, false);
    }
  else
    p = append_bin ((unsigned char*) &res, buf, sizeof (res), false);
  xfree (buffer);
}

static void
cuda_process_single_step_warp_packet (char *buf)
{
  CUDBGResult res;
  char *p;
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t laneHint;
  uint32_t nsteps;
  uint32_t flags;
  uint64_t warp_mask;

  extract_bin (NULL, (unsigned char *) &dev, sizeof (dev));
  extract_bin (NULL, (unsigned char *) &sm, sizeof (sm));
  extract_bin (NULL, (unsigned char *) &wp, sizeof (wp));
  extract_bin (NULL, (unsigned char *) &laneHint, sizeof (laneHint));
  extract_bin (NULL, (unsigned char *) &nsteps, sizeof (nsteps));
  extract_bin (NULL, (unsigned char *) &flags, sizeof (flags));
  extract_bin (NULL, (unsigned char *) &warp_mask, sizeof (warp_mask));
  res = cudbgAPI->singleStepWarp (dev, sm, wp, laneHint, nsteps, flags, &warp_mask);
  p = append_bin ((unsigned char *) &res, buf, sizeof (res), true);
  p = append_bin ((unsigned char *) &warp_mask, p, sizeof (warp_mask), false);
}

static void
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

#ifdef __QNXHOST__
static void
cuda_process_version_handshake (char *buf)
{
  char str[256];

  sprintf (str, "%d.%d.%d",
           CUDBG_API_VERSION_MAJOR,
           CUDBG_API_VERSION_MINOR,
           CUDBG_API_VERSION_REVISION);

  append_string (str, buf, false);
}
#endif /* __QNXHOST__ */

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
    case SINGLE_STEP_WARP65:
      cuda_process_single_step_warp_packet65 (buf);
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
    /* Texture support has been removed - let the other
     * side know we no longer support it. */
    case READ_TEXTURE_MEMORY:
    case READ_TEXTURE_MEMORY_BINDLESS:
      buf[0] = '\0';
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
    /* CUDA MEMCHECK support is removed from CUDA GDB: this field is left
     * to maintain the binary compatibility with legacy CUDA GDB server
     * binaries */
    case MEMCHECK_READ_ERROR_ADDRESS:
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
    case GET_ERROR_STRING_EX:
      cuda_process_get_error_string_ex_packet (buf);
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
#ifndef __QNXHOST__
    case UPDATE_GRID_ID_IN_SM:
      cuda_process_update_grid_id_in_sm_packet (buf);
      break;
    case UPDATE_BLOCK_IDX_IN_SM:
      cuda_process_update_block_idx_in_sm_packet (buf);
      break;
#endif
    case UPDATE_THREAD_IDX_IN_WARP:
      cuda_process_update_thread_idx_in_warp_packet (buf);
      break;
#ifdef __QNXHOST__
    case SET_SYMBOLS:
      cuda_process_set_symbols (buf);
      break;
#endif /* __QNXHOST__ */
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
#ifdef __QNXHOST__
    case VERSION_HANDSHAKE:
      cuda_process_version_handshake (buf);
      break;
#endif /* __QNXHOST__ */
    case READ_CLUSTER_IDX:
      cuda_process_read_cluster_idx_packet (buf);
      break;
#ifndef __QNXHOST__
    case UPDATE_CLUSTER_IDX_IN_SM:
      cuda_process_update_cluster_idx_in_sm_packet (buf);
      break;
    case UPDATE_CLUSTER_DIM_IN_SM:
      cuda_process_update_cluster_dim_in_sm_packet (buf);
      break;
#endif /* !__QNXHOST__ */
    case GET_DEVICE_INFO_SIZES:
      cuda_process_api_get_device_info_sizes (buf);
      break;
    case GET_DEVICE_INFO:
      cuda_process_api_get_device_info (buf);
      break;
    case SINGLE_STEP_WARP:
      cuda_process_single_step_warp_packet (buf);
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
                                             size-offset, 1, lbuf,
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

  packet_len = remote_unescape_input (lbuf, packet_len, (gdb_byte *) data, packet_len);
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
  *new_packet_len += remote_escape_output ((const gdb_byte *) data, size, 1, lbuf,
                                          &out_len, PBUFSIZ-strlen ("OK;"));
  if (out_len != size)
    memcpy (buf, "MP", 2);

  return 1;
}
