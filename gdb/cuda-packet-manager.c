/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2013 NVIDIA Corporation
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

#include <stdbool.h>
#include "defs.h"
#include "gdbthread.h"
#include "remote.h"
#include "remote-cuda.h"
#include "cuda-packet-manager.h"
#include "cuda-context.h"
#include "cuda-events.h"
#include "cuda-state.h"
#include "cuda-textures.h"
#include "cuda-utils.h"
#include "cuda-options.h"


#define PBUFSIZ 16384

struct {
  char *buf;
  long int buf_size;
} pktbuf;

void
alloc_cuda_packet_buffer (void)
{
  if (pktbuf.buf == NULL)
    {
      pktbuf.buf = xmalloc (PBUFSIZ);
      pktbuf.buf_size = PBUFSIZ;
    }
}

void
free_cuda_packet_buffer (void *unused)
{
  if (pktbuf.buf != NULL)
    {
      xfree (pktbuf.buf);
      pktbuf.buf = NULL;
      pktbuf.buf_size = 0;
    }
}

static char *
append_string (const char *src, char *dest, bool sep)
{
  char *p;

  if (dest + strlen (src) - pktbuf.buf >= pktbuf.buf_size)
    error (_("Exceed the size of cuda packet.\n"));

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
append_bin (const gdb_byte *src, char *dest, int size, bool sep)
{
  char *p;

  if (dest + size * 2 - pktbuf.buf >= pktbuf.buf_size)
    error (_("Exceed the size of cuda packet.\n"));

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
extract_bin (char *src, gdb_byte *dest, int size)
{
  char *p;

  p = extract_string (src);
  if (!p)
    error (_("The data in the cuda packet is not complete.\n")); 
  hex2bin (p, dest, size);
  return p;
}

bool
cuda_remote_notification_pending (void)
{
  char *p;
  bool ret_val;
  cuda_packet_type_t packet_type = NOTIFICATION_PENDING;

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);

  extract_bin (pktbuf.buf, (gdb_byte *) &ret_val, sizeof (ret_val));
  return ret_val;
}

bool
cuda_remote_notification_received (void)
{
  char *p;
  bool ret_val;
  cuda_packet_type_t packet_type = NOTIFICATION_RECEIVED;

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);

  extract_bin (pktbuf.buf, (gdb_byte *) &ret_val, sizeof (ret_val));
  return ret_val;
}

bool
cuda_remote_notification_aliased_event (void)
{
  char *p;
  bool ret_val;
  cuda_packet_type_t packet_type = NOTIFICATION_ALIASED_EVENT;

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);

  extract_bin (pktbuf.buf, (gdb_byte *) &ret_val, sizeof (ret_val));
  return ret_val;
}

void
cuda_remote_notification_analyze (void)
{
  char *p;
  cuda_packet_type_t packet_type = NOTIFICATION_ANALYZE;
  struct thread_info *tp = inferior_thread ();

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &(tp->control.trap_expected), p, sizeof (tp->control.trap_expected), false);
  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);
}

void
cuda_remote_notification_mark_consumed (void)
{
  char *p;
  cuda_packet_type_t packet_type = NOTIFICATION_MARK_CONSUMED;

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);
}

void
cuda_remote_notification_consume_pending (void)
{
  char *p;
  cuda_packet_type_t packet_type = NOTIFICATION_CONSUME_PENDING;

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);
}

void
cuda_remote_update_grid_id_in_sm (uint32_t dev, uint32_t sm)
{
  CUDBGResult res;
  char *p;
  uint32_t wp;
  uint64_t valid_warps_mask_c;
  uint64_t valid_warps_mask_s;
  uint32_t num_warps;
  uint64_t grid_id;
  cuda_packet_type_t packet_type = UPDATE_GRID_ID_IN_SM;

  valid_warps_mask_c = sm_get_valid_warps_mask (dev, sm);
  num_warps = device_get_num_warps (dev);
  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &dev, p, sizeof (dev), true);
  p = append_bin ((gdb_byte *) &sm,  p, sizeof (sm), true);
  p = append_bin ((gdb_byte *) &num_warps, p, sizeof (num_warps), false);

  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);

  extract_bin (pktbuf.buf, (gdb_byte *) &valid_warps_mask_s, sizeof (valid_warps_mask_s));
  gdb_assert (valid_warps_mask_s == valid_warps_mask_c);
  for (wp = 0; wp < num_warps; wp++)
    {
      if (warp_is_valid (dev, sm, wp))
        {
          extract_bin (NULL, (gdb_byte *) &grid_id, sizeof (grid_id));
          warp_set_grid_id (dev, sm, wp, grid_id);
        }
    }
  extract_bin (NULL, (gdb_byte *) &res, sizeof (res));
  if (res != CUDBG_SUCCESS)
    error (_("Error: Failed to read the grid index (error=%u).\n"), res);
}

void
cuda_remote_update_block_idx_in_sm (uint32_t dev, uint32_t sm)
{
  CUDBGResult res;
  char *p;
  uint32_t wp;
  uint64_t valid_warps_mask_c;
  uint64_t valid_warps_mask_s;
  uint32_t num_warps;
  CuDim3 block_idx;
  cuda_packet_type_t packet_type = UPDATE_BLOCK_IDX_IN_SM;

  valid_warps_mask_c = sm_get_valid_warps_mask (dev, sm);
  num_warps = device_get_num_warps (dev);
  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &dev, p, sizeof (dev), true);
  p = append_bin ((gdb_byte *) &sm,  p, sizeof (sm), true);
  p = append_bin ((gdb_byte *) &num_warps, p, sizeof (num_warps), false);

  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);

  extract_bin (pktbuf.buf, (gdb_byte *) &valid_warps_mask_s, sizeof (valid_warps_mask_s));
  gdb_assert (valid_warps_mask_s == valid_warps_mask_c);
  for (wp = 0; wp < num_warps; wp++)
    {
       if (warp_is_valid (dev, sm, wp))
         {
           extract_bin (NULL, (gdb_byte *) &block_idx, sizeof (block_idx));
           warp_set_block_idx (dev, sm, wp, &block_idx);
         }
    }
  extract_bin (NULL, (gdb_byte *) &res, sizeof (res));
  if (res != CUDBG_SUCCESS)
    error (_("Error: Failed to read the block index (error=%u).\n"), res);
}

void
cuda_remote_update_thread_idx_in_warp (uint32_t dev, uint32_t sm, uint32_t wp)
{
  CUDBGResult res;
  char *p;
  uint32_t ln;
  uint32_t valid_lanes_mask_c;
  uint32_t valid_lanes_mask_s;
  uint32_t num_lanes;
  CuDim3 thread_idx;
  cuda_packet_type_t packet_type = UPDATE_THREAD_IDX_IN_WARP;

  valid_lanes_mask_c = warp_get_valid_lanes_mask (dev, sm, wp);
  num_lanes = device_get_num_lanes (dev);
  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &dev, p, sizeof (dev), true);
  p = append_bin ((gdb_byte *) &sm,  p, sizeof (sm), true);
  p = append_bin ((gdb_byte *) &wp,  p, sizeof (wp), true);
  p = append_bin ((gdb_byte *) &num_lanes, p, sizeof (num_lanes), false);

  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);

  extract_bin (pktbuf.buf, (gdb_byte *) &valid_lanes_mask_s, sizeof (valid_lanes_mask_s));
  gdb_assert (valid_lanes_mask_s == valid_lanes_mask_c);
  for (ln = 0; ln < num_lanes; ln++)
    {
       if (lane_is_valid (dev, sm, wp, ln))
         {
           extract_bin (NULL, (gdb_byte *) &thread_idx, sizeof (thread_idx));
           lane_set_thread_idx (dev, sm, wp, ln, &thread_idx);
         }
    }
  extract_bin (NULL, (gdb_byte *) &res, sizeof (res));
  if (res != CUDBG_SUCCESS)
    error (_("Error: Failed to read the thread index (error=%u).\n"), res);
}

void
cuda_remote_initialize (CUDBGResult *get_debugger_api_res, CUDBGResult *set_callback_api_res,
                        CUDBGResult *initialize_api_res, bool *cuda_initialized,
                        bool *cuda_debugging_enabled, bool *driver_is_compatible)
{
  char *p;
  cuda_packet_type_t packet_type = INITIALIZE_TARGET;
  bool preemption          = cuda_options_software_preemption ();
  bool memcheck            = cuda_options_memcheck ();
  bool launch_blocking     = cuda_options_launch_blocking ();

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type,     p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &preemption,      p, sizeof (preemption), true);
  p = append_bin ((gdb_byte *) &memcheck,        p, sizeof (memcheck), true);
  p = append_bin ((gdb_byte *) &launch_blocking, p, sizeof (launch_blocking), false);

  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);

  extract_bin (pktbuf.buf, (gdb_byte *) get_debugger_api_res, sizeof (*get_debugger_api_res));
  extract_bin (NULL, (gdb_byte *) set_callback_api_res, sizeof (*set_callback_api_res));
  extract_bin (NULL, (gdb_byte *) initialize_api_res, sizeof (*initialize_api_res));
  extract_bin (NULL, (gdb_byte *) cuda_initialized, sizeof (*cuda_initialized));
  extract_bin (NULL, (gdb_byte *) cuda_debugging_enabled, sizeof (*cuda_debugging_enabled));
  extract_bin (NULL, (gdb_byte *) driver_is_compatible, sizeof (*driver_is_compatible));
}

void
cuda_remote_query_device_spec (uint32_t dev_id, uint32_t *num_sms, uint32_t *num_warps,
                               uint32_t *num_lanes, uint32_t *num_registers,
                               char **dev_type, char **sm_type)
{
  char *p;
  CUDBGResult res;
  cuda_packet_type_t packet_type = QUERY_DEVICE_SPEC;

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &dev_id, p, sizeof (uint32_t), false);

  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);

  extract_bin (pktbuf.buf, (gdb_byte *) &res, sizeof (res));
  if (res != CUDBG_SUCCESS)
    error (_("Error: Failed to read device specification (error=%u).\n"), res);  
  extract_bin (NULL, (gdb_byte *) num_sms, sizeof (num_sms));
  extract_bin (NULL, (gdb_byte *) num_warps, sizeof (num_warps));
  extract_bin (NULL, (gdb_byte *) num_lanes, sizeof (num_lanes));
  extract_bin (NULL, (gdb_byte *) num_registers, sizeof (num_registers));
  *dev_type = extract_string (NULL);
  *sm_type  = extract_string (NULL);
}

bool
cuda_remote_check_pending_sigint (void)
{
  char *p;
  bool ret_val;
  cuda_packet_type_t packet_type = CHECK_PENDING_SIGINT;

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);

  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);

  extract_bin (pktbuf.buf, (gdb_byte *) &ret_val, sizeof (ret_val));
  return ret_val;
}

CUDBGResult
cuda_remote_api_finalize (void)
{
  char *p;
  CUDBGResult res;
  cuda_packet_type_t packet_type = API_FINALIZE;

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);

  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);

  extract_bin (pktbuf.buf, (gdb_byte *) &res, sizeof (res));
  return res;
}

void
cuda_remote_set_option ()
{
  char *p;
  bool general_trace       = cuda_options_debug_general ();
  bool libcudbg_trace      = cuda_options_debug_libcudbg ();
  bool notifications_trace = cuda_options_debug_notifications ();
  bool notify_youngest     = cuda_options_notify_youngest ();

  cuda_packet_type_t packet_type = SET_OPTION;

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type,         p, sizeof (cuda_packet_type_t), true);
  p = append_bin ((gdb_byte *) &general_trace,       p, sizeof (general_trace), true);
  p = append_bin ((gdb_byte *) &libcudbg_trace,      p, sizeof (libcudbg_trace), true);
  p = append_bin ((gdb_byte *) &notifications_trace, p, sizeof (notifications_trace), true);
  p = append_bin ((gdb_byte *) &notify_youngest,     p, sizeof (notify_youngest), false);

  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);
}

void
cuda_remote_query_trace_message ()
{
  char *p;
  cuda_packet_type_t packet_type = QUERY_TRACE_MESSAGE;

  if (!cuda_options_debug_general () &&
      !cuda_options_debug_libcudbg () &&
      !cuda_options_debug_notifications ())
    return;

  p = append_string ("qnv.", pktbuf.buf, false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);

  putpkt (pktbuf.buf);
  getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);
  p = extract_string (pktbuf.buf);
  while (strcmp ("NO_TRACE_MESSAGE", p) != 0)
    {
      fprintf (stderr, "%s\n", p);

      p = append_string ("qnv.", pktbuf.buf, false);
      p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
      putpkt (pktbuf.buf);
      getpkt (&pktbuf.buf, &pktbuf.buf_size, 1);
      p = extract_string (pktbuf.buf);
    }
  fflush (stderr);
}
