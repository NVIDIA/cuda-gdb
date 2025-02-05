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

#include "defs.h"
#include <stdbool.h>
#include <string>
#include "gdbthread.h"
#include "inferior.h"
#include "remote.h"
#include "cuda-packet-manager.h"
#include "cuda-context.h"
#include "cuda-events.h"
#include "cuda-state.h"
#include "cuda-utils.h"
#include "cuda-options.h"

#include "gdbsupport/rsp-low.h"

#ifdef __QNXTARGET__
#include "remote-nto.h"
/* Maximum supported data size in QNX protocol is DS_DATA_MAX_SIZE (1024).
   cuda-gdbserver can be modified to handle 16384 instead, but in order to
   use bigger packets for CUDA, we would need first ensure that they can be
   packed/unpacked by the pdebug putpkt/getpkt functions.

   Until then, use pdebug max allowed size.
   Each DS_DATA_MAX_SIZE can be escaped (*2), 2 frame chars (+2) plus a checksum
   that can be escaped (+2). */
# define PBUFSIZE (DS_DATA_MAX_SIZE * 2 + 4)
#else
# define PBUFSIZE 16384
#endif

struct cuda_remote_callbacks : public send_remote_packet_callbacks
{
public:
  cuda_remote_callbacks ()
  {
    m_send_buf.reserve (PBUFSIZE);
    m_recv_buf.reserve (PBUFSIZE);
  }

  void
  append_string (std::string str)
  {
    if (m_send_buf.size () + str.size () >= m_send_buf.capacity ())
      error (_ ("Exceed the size of cuda packet.\n"));

    m_send_buf.insert (m_send_buf.end (), str.begin (), str.end ());
  }

  void
  append_bin (const gdb_byte *src, int size)
  {
    if (m_send_buf.size () + size >= m_send_buf.capacity ())
      error (_ ("Exceed the size of cuda packet.\n"));

    std::string hex_str = bin2hex (src, size);
    m_send_buf.insert (m_send_buf.end (), hex_str.begin (), hex_str.end ());
  }

  void
  append_separator ()
  {
    m_send_buf.push_back (';');
  }

  void
  send_request (void)
  {
    gdb::array_view<const char> view (m_send_buf.data (), m_send_buf.size ());
    send_remote_packet (view, this);
  }

  char *
  extract_string ()
  {
    /* Locate the first string in the buffer */
    auto pos = m_recv_buf.find (';', m_recv_pos);

    /* Allow for one final lookup to grab the ending data */
    if (pos == std::string::npos && m_recv_pos == std::string::npos)
      error (_ ("The data in the cuda packet is not complete (cuda-gdb).\n"));

    /* Null terminate substring and advance recv position
     * Handle the case where we are accessing the last piece of data. */
    if (pos != std::string::npos)
      m_recv_buf[pos] = '\0';
    else
      m_recv_buf.push_back ('\0');

    char *ret = (char *)m_recv_buf.data () + m_recv_pos;

    if (pos != std::string::npos)
      m_recv_pos = pos + 1;
    else
      m_recv_pos = std::string::npos;

    /* TODO: Maybe rethink this as we are exposing the underlying m_recv_buf to
     * the outside world */
    return ret;
  }

  void
  extract_bin (gdb_byte *dest, int size)
  {
    /* Locate the first string in the buffer */
    auto pos = m_recv_buf.find (';', m_recv_pos);

    /* Allow for one final lookup to grab the ending data */
    if (pos == std::string::npos && m_recv_pos == std::string::npos)
      error (_ ("The data in the cuda packet is not complete (cuda-gdb).\n"));
    if (((pos == std::string::npos ? m_recv_buf.size () : pos) - m_recv_pos) < size)
      error (_ ("The data in the cuda packet is not complete (cuda-gdb).\n"));

    /* Extract binary data */
    if (pos != std::string::npos)
      m_recv_buf[pos] = '\0';

    hex2bin (m_recv_buf.data () + m_recv_pos, dest, size);

    if (pos != std::string::npos)
      m_recv_pos = pos + 1;
    else
      m_recv_pos = std::string::npos;
  }

  void
  sending (gdb::array_view<const char> &buf) override
  {
  }

  void
  received (gdb::array_view<const char> &buf) override
  {
    /* Clear the recv buffer */
    m_recv_buf.clear ();
    /* Copy result to pktbuf */
    m_recv_buf.insert (m_recv_buf.begin (), buf.begin (), buf.end ());
    /* Reset position offset in recv buffer */
    m_recv_pos = 0;
    /* Clear send buffer */
    m_send_buf.clear ();
  }

private:
  std::string m_send_buf;
  std::string m_recv_buf;
  std::string::size_type m_recv_pos;
};
static cuda_remote_callbacks remote_callbacks;

static void
cuda_remote_send_packet (cuda_packet_type_t packet_type)
{
  remote_callbacks.append_string ("qnv.");
  remote_callbacks.append_bin ((gdb_byte *) &packet_type, sizeof (packet_type));

  remote_callbacks.send_request ();
}

static bool
cuda_remote_get_return_value ()
{
  bool ret_val;
  remote_callbacks.extract_bin ((gdb_byte *) &ret_val, sizeof (ret_val));

  return ret_val;
}

bool
cuda_remote_notification_pending ()
{
  cuda_remote_send_packet (NOTIFICATION_PENDING);
  return cuda_remote_get_return_value ();
}

bool
cuda_remote_notification_received ()
{
  cuda_remote_send_packet (NOTIFICATION_RECEIVED);
  return cuda_remote_get_return_value ();
}

bool
cuda_remote_notification_aliased_event ()
{
  cuda_remote_send_packet (NOTIFICATION_ALIASED_EVENT);
  return cuda_remote_get_return_value ();
}

void
cuda_remote_notification_analyze (ptid_t ptid, struct target_waitstatus *ws)
{
  int trap_expected = 0;

  /* Upon connecting to gdbserver, we may not have stablished an inferior_ptid,
     so it is still null_ptid.  In that case, use the event ptid that should be
     the thread that triggered this code path.  */
  if (inferior_ptid == null_ptid)
    {
      struct thread_info *tp = find_thread_ptid (current_inferior ()->process_target (), ptid);
      if (tp != nullptr)
	trap_expected = tp->control.trap_expected;
    }
  else
    {
      struct thread_info *tp = inferior_thread ();
      trap_expected = tp->control.trap_expected;
    }

  remote_callbacks.append_string ("qnv.");
  cuda_packet_type_t packet_type = NOTIFICATION_ANALYZE;
  remote_callbacks.append_bin ((gdb_byte *) &packet_type, sizeof (packet_type));
  remote_callbacks.append_separator ();
#ifdef __QNXTARGET__
  /* We only send the wait status for QNX as we don't have an easy way of
     getting that server side. */
  remote_callbacks.append_bin ((gdb_byte *)&ptid, sizeof (ptid));
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *)ws, sizeof (*ws));
  remote_callbacks.append_separator ();
#endif
  remote_callbacks.append_bin ((gdb_byte *)&trap_expected, sizeof (trap_expected));

  remote_callbacks.send_request ();
}

void
cuda_remote_notification_mark_consumed ()
{
  cuda_remote_send_packet (NOTIFICATION_MARK_CONSUMED);
}

void
cuda_remote_notification_consume_pending ()
{
  cuda_remote_send_packet (NOTIFICATION_CONSUME_PENDING);
}

template <typename TValue>
using FnPerWarpUpdateGet = void (*)(uint32_t dev, uint32_t sm, uint32_t wp, TValue *value);
template <typename TValue>
using FnPerWarpUpdateSet = void (*)(uint32_t dev, uint32_t sm, uint32_t wp, const TValue& value);

template <typename TValue>
void
cuda_remote_update_per_warp_info_in_sm (uint32_t dev, uint32_t sm,
  cuda_packet_type_t packet_type, const char* name,
  FnPerWarpUpdateGet<TValue>&& get_value, FnPerWarpUpdateSet<TValue>&& set_value)
{
#ifdef __QNXTARGET__
  uint32_t wp;
  const cuda_api_warpmask* valid_warps_mask_c;
  cuda_api_warpmask valid_warps_mask_s;
  uint32_t num_warps;
  TValue value;

  /* On QNX the packet size is limited and the response for such packets
     doesn't fit. Therefore gather all the data from the client side
     without server-side batching, using individual vCUDA packets via
     the CUDA API which are constant in size in this case. */

  valid_warps_mask_c = cuda_state::sm_get_valid_warps_mask (dev, sm);
  num_warps = cuda_state::device_get_num_warps (dev);

  cuda_debugapi::read_valid_warps (dev, sm, &valid_warps_mask_s);
  gdb_assert (cuda_api_eq_mask(&valid_warps_mask_s, valid_warps_mask_c));

  for (wp = 0; wp < num_warps; wp++)
    {
      if (cuda_state::warp_valid (dev, sm, wp))
        {
          /* Get the value from the API first, then store it in warp state. */
          get_value (dev, sm, wp, &value);
          set_value (dev, sm, wp, value);
        }
    }
#else
  const cuda_api_warpmask* valid_warps_mask_c = cuda_state::sm_get_valid_warps_mask (dev, sm);
  uint32_t num_warps = cuda_state::device_get_num_warps (dev);

  remote_callbacks.append_string ("qnv.");
  remote_callbacks.append_bin ((gdb_byte *) &packet_type, sizeof (packet_type));
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *) &dev, sizeof (dev));
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *) &sm, sizeof (sm));
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *) &num_warps, sizeof (num_warps));
  remote_callbacks.append_separator ();

  remote_callbacks.send_request ();

  cuda_api_warpmask valid_warps_mask_s;
  remote_callbacks.extract_bin ((gdb_byte *) &valid_warps_mask_s, sizeof (valid_warps_mask_s));
  gdb_assert (cuda_api_eq_mask(&valid_warps_mask_s, valid_warps_mask_c));

  for (uint32_t wp = 0; wp < num_warps; wp++)
    {
      if (cuda_state::warp_valid (dev, sm, wp))
        {
	  TValue value;
	  remote_callbacks.extract_bin ((gdb_byte *) &value, sizeof (value));
          set_value (dev, sm, wp, value);
        }
    }

  CUDBGResult res;
  remote_callbacks.extract_bin ((gdb_byte *) &res, sizeof (res));
  if (res != CUDBG_SUCCESS)
    error (_("Error: Failed to read %s (error=%u).\n"), name, res);
#endif
}

void
cuda_remote_update_grid_id_in_sm (uint32_t dev, uint32_t sm)
{
  cuda_remote_update_per_warp_info_in_sm<uint64_t> (
    dev, sm, UPDATE_GRID_ID_IN_SM, "grid ID", cuda_debugapi::read_grid_id,
    [](uint32_t _dev, uint32_t _sm, uint32_t _wp, const uint64_t& _grid_id) {
      cuda_state::warp_set_grid_id (_dev, _sm, _wp, _grid_id);
    });
}

void
cuda_remote_update_cluster_idx_in_sm (uint32_t dev, uint32_t sm)
{
  cuda_remote_update_per_warp_info_in_sm<CuDim3> (
    dev, sm, UPDATE_CLUSTER_IDX_IN_SM, "cluster index", cuda_debugapi::read_cluster_idx,
    [](uint32_t _dev, uint32_t _sm, uint32_t _wp, const CuDim3& _cluster_idx) {
      cuda_state::warp_set_cluster_idx (_dev, _sm, _wp, _cluster_idx);
    });
}

void
cuda_remote_update_cluster_dim_in_sm (uint32_t dev, uint32_t sm)
{
  cuda_remote_update_per_warp_info_in_sm<CuDim3> (
    dev, sm, UPDATE_CLUSTER_DIM_IN_SM, "cluster dimension", cuda_debugapi::get_cluster_dim,
    [](uint32_t _dev, uint32_t _sm, uint32_t _wp, const CuDim3& _cluster_dim) {
      cuda_state::warp_set_cluster_dim (_dev, _sm, _wp, _cluster_dim);
    });
}

void
cuda_remote_update_block_idx_in_sm (uint32_t dev, uint32_t sm)
{
  cuda_remote_update_per_warp_info_in_sm<CuDim3> (
    dev, sm, UPDATE_BLOCK_IDX_IN_SM, "block index", cuda_debugapi::read_block_idx,
    [](uint32_t _dev, uint32_t _sm, uint32_t _wp, const CuDim3& _block_idx) {
      cuda_state::warp_set_block_idx (_dev, _sm, _wp, _block_idx);
    });
}

void
cuda_remote_update_thread_idx_in_warp (uint32_t dev, uint32_t sm, uint32_t wp)
{
  uint32_t valid_lanes_mask_c = cuda_state::warp_get_valid_lanes_mask (dev, sm, wp);
  uint32_t num_lanes = cuda_state::device_get_num_lanes (dev);

  remote_callbacks.append_string ("qnv.");
  cuda_packet_type_t packet_type = UPDATE_THREAD_IDX_IN_WARP;
  remote_callbacks.append_bin ((gdb_byte *) &packet_type, sizeof (packet_type));
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *) &dev, sizeof (dev));
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *) &sm, sizeof (sm));
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *) &wp, sizeof (wp));
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *) &num_lanes, sizeof (num_lanes));

  remote_callbacks.send_request ();

  uint32_t valid_lanes_mask_s;
  remote_callbacks.extract_bin ((gdb_byte *) &valid_lanes_mask_s, sizeof (valid_lanes_mask_s));
  gdb_assert (valid_lanes_mask_s == valid_lanes_mask_c);

  for (uint32_t ln = 0; ln < num_lanes; ln++)
    {
       if (cuda_state::lane_valid (dev, sm, wp, ln))
         {
	   CuDim3 thread_idx;
	   remote_callbacks.extract_bin ((gdb_byte *) &thread_idx, sizeof (thread_idx));
           cuda_state::lane_set_thread_idx (dev, sm, wp, ln, thread_idx);
         }
    }

  CUDBGResult res;
  remote_callbacks.extract_bin ((gdb_byte *) &res, sizeof (res));
  if (res != CUDBG_SUCCESS)
    error (_("Error: Failed to read the thread index (error=%u).\n"), res);
}

#ifdef __QNXTARGET__
void
cuda_remote_set_symbols (bool set_extra_symbols, bool *symbols_are_set)
{
  constexpr unsigned char CORE_SYMBOLS_COUNT = 13;
  constexpr unsigned char EXTRA_SYMBOLS_COUNT = 2;
  unsigned char symbols_count = CORE_SYMBOLS_COUNT;

  *symbols_are_set = false;

  /* Old fields are left to maintain the binary compatibility with legacy CUDA
   * GDB server binaries */
  /* Remote side will also check for zeros, here we test only one symbol
     to avoid unnecessary back and forth with it.
     Sent symbols must be kept in sync with those in cuda_symbol_list[] */
  CORE_ADDR address = cuda_get_symbol_address (_STRING_(CUDBG_IPC_FLAG_NAME));
  if (address == 0)
    {
      return;
    }

  if (set_extra_symbols)
    {
      symbols_count += EXTRA_SYMBOLS_COUNT;
    }

  remote_callbacks.append_string ("qnv.");
  cuda_packet_type_t packet_type = SET_SYMBOLS;
  remote_callbacks.append_bin ((gdb_byte *) &packet_type, sizeof (packet_type));
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *) &symbols_count, sizeof (symbols_count));
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  address = cuda_get_symbol_address (_STRING_(CUDBG_RPC_ENABLED));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  address = cuda_get_symbol_address (_STRING_(CUDBG_APICLIENT_PID));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  address = cuda_get_symbol_address (_STRING_(CUDBG_APICLIENT_REVISION));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  address = cuda_get_symbol_address (_STRING_(CUDBG_SESSION_ID));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  address = cuda_get_symbol_address (_STRING_(CUDBG_ATTACH_HANDLER_AVAILABLE));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  address = cuda_get_symbol_address (_STRING_(CUDBG_DEBUGGER_INITIALIZED));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  address = cuda_get_symbol_address (_STRING_(CUDBG_REPORTED_DRIVER_API_ERROR_CODE));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  address = cuda_get_symbol_address (_STRING_(CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  /* CUDBG_DETACH_SUSPENDED_DEVICES_MASK is deprecated */
  address = cuda_get_symbol_address (_STRING_(CUDBG_DETACH_SUSPENDED_DEVICES_MASK));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  /* CUDA MEMCHECK support is removed from CUDA GDB */
  address = cuda_get_symbol_address (_STRING_(CUDBG_ENABLE_INTEGRATED_MEMCHECK));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  address = cuda_get_symbol_address (_STRING_(CUDBG_ENABLE_LAUNCH_BLOCKING));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  remote_callbacks.append_separator ();
  address = cuda_get_symbol_address (_STRING_(CUDBG_ENABLE_PREEMPTION_DEBUGGING));
  remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
  /* All new symbols should be placed under this condition to preserve
     compatibility between newer cuda-gdb and older cuda-gdbserver.
     Recent cuda-gdbserver binaries will gracefully handle more symbols
     than they need, but the old ones won't, so we'll need to only set
     the exact core symbols that they expect, those are defined above. */
  if (set_extra_symbols)
    {
      remote_callbacks.append_separator ();
      address = cuda_get_symbol_address (_STRING_(cudbgInjectionPath));
      remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
      remote_callbacks.append_separator ();
      address = cuda_get_symbol_address (_STRING_(CUDBG_DEBUGGER_CAPABILITIES));
      remote_callbacks.append_bin ((gdb_byte *) &address, sizeof (address));
      /* NOTE: When adding new symbols, add a call to append_separator and update `EXTRA_SYMBOLS_COUNT`. */
    }

  remote_callbacks.send_request ();

  remote_callbacks.extract_bin ((gdb_byte *) symbols_are_set, sizeof (*symbols_are_set));
}
#endif /* __QNXTARGET__ */

void
cuda_remote_initialize (CUDBGResult *get_debugger_api_res,
			CUDBGResult *set_callback_api_res,
			CUDBGResult *initialize_api_res,
			bool *cuda_initialized, bool *cuda_debugging_enabled,
			bool *driver_is_compatible, uint32_t *major,
			uint32_t *minor, uint32_t *revision)
{
  remote_callbacks.append_string ("qnv.");
  cuda_packet_type_t packet_type = INITIALIZE_TARGET;
  remote_callbacks.append_bin ((gdb_byte *) &packet_type, sizeof (packet_type));
  remote_callbacks.append_separator ();
  bool preemption = cuda_options_software_preemption ();
  remote_callbacks.append_bin ((gdb_byte *) &preemption, sizeof (preemption));
  remote_callbacks.append_separator ();
  /* CUDA MEMCHECK support is removed from CUDA GDB: this field is left to maintain
   * the binary compatibility with legacy CUDA GDB server binaries */
  bool memcheck = false;
  remote_callbacks.append_bin ((gdb_byte *) &memcheck, sizeof (memcheck));
  remote_callbacks.append_separator ();
  bool launch_blocking = cuda_options_launch_blocking ();
  remote_callbacks.append_bin ((gdb_byte *) &launch_blocking, sizeof (launch_blocking));

  remote_callbacks.send_request ();

  remote_callbacks.extract_bin ((gdb_byte *) get_debugger_api_res, sizeof (*get_debugger_api_res));
  remote_callbacks.extract_bin ((gdb_byte *) set_callback_api_res, sizeof (*set_callback_api_res));
  remote_callbacks.extract_bin ((gdb_byte *) initialize_api_res, sizeof (*initialize_api_res));
  remote_callbacks.extract_bin ((gdb_byte *) cuda_initialized, sizeof (*cuda_initialized));
  remote_callbacks.extract_bin ((gdb_byte *) cuda_debugging_enabled, sizeof (*cuda_debugging_enabled));
  remote_callbacks.extract_bin ((gdb_byte *) driver_is_compatible, sizeof (*driver_is_compatible));
  remote_callbacks.extract_bin ((gdb_byte *) major, sizeof (*major));
  remote_callbacks.extract_bin ((gdb_byte *) minor, sizeof (*minor));
  remote_callbacks.extract_bin ((gdb_byte *) revision, sizeof (*revision));
}

void
cuda_remote_query_device_spec (uint32_t dev_id, uint32_t *num_sms,
			       uint32_t *num_warps, uint32_t *num_lanes,
			       uint32_t *num_registers, char **dev_type,
			       char **sm_type)
{
  remote_callbacks.append_string ("qnv.");
  cuda_packet_type_t packet_type = QUERY_DEVICE_SPEC;
  remote_callbacks.append_bin ((gdb_byte *) &packet_type, sizeof (packet_type));
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *) &dev_id, sizeof (dev_id));

  remote_callbacks.send_request ();

  CUDBGResult res;
  remote_callbacks.extract_bin ((gdb_byte *) &res, sizeof (res));
  if (res != CUDBG_SUCCESS)
    error (_("Error: Failed to read device specification (error=%u).\n"), res);
  remote_callbacks.extract_bin ((gdb_byte *) num_sms, sizeof (num_sms));
  remote_callbacks.extract_bin ((gdb_byte *) num_warps, sizeof (num_warps));
  remote_callbacks.extract_bin ((gdb_byte *) num_lanes, sizeof (num_lanes));
  remote_callbacks.extract_bin ((gdb_byte *) num_registers, sizeof (num_registers));
  *dev_type = remote_callbacks.extract_string ();
  *sm_type  = remote_callbacks.extract_string ();
}

bool
cuda_remote_check_pending_sigint (ptid_t ptid)
{
  remote_callbacks.append_string ("qnv.");
  cuda_packet_type_t packet_type = CHECK_PENDING_SIGINT;
  remote_callbacks.append_bin ((gdb_byte *) &packet_type, sizeof (packet_type));
#ifdef __QNXTARGET__
  /* Only send ptid for QNX targets since the server has trouble grabbing that */
  remote_callbacks.append_separator ();
  remote_callbacks.append_bin ((gdb_byte *) &ptid, sizeof (ptid));
#endif

  remote_callbacks.send_request ();
  
  return cuda_remote_get_return_value ();
}

CUDBGResult
cuda_remote_api_finalize ()
{
  cuda_remote_send_packet (API_FINALIZE);

  CUDBGResult res;
  remote_callbacks.extract_bin ((gdb_byte *) &res, sizeof (res));
  return res;
}

void
cuda_remote_set_option ()
{
  remote_callbacks.append_string ("qnv.");
  cuda_packet_type_t packet_type = CHECK_PENDING_SIGINT;
  remote_callbacks.append_bin ((gdb_byte *) &packet_type, sizeof (packet_type));
  remote_callbacks.append_separator ();
  bool general_trace = cuda_options_debug_general ();
  remote_callbacks.append_bin ((gdb_byte *) &general_trace, sizeof (general_trace));
  remote_callbacks.append_separator ();
  bool libcudbg_trace = cuda_options_debug_libcudbg ();
  remote_callbacks.append_bin ((gdb_byte *) &libcudbg_trace, sizeof (libcudbg_trace));
  remote_callbacks.append_separator ();
  bool notifications_trace = cuda_options_debug_notifications ();
  remote_callbacks.append_bin ((gdb_byte *) &notifications_trace, sizeof (notifications_trace));
  remote_callbacks.append_separator ();
  bool notify_youngest = cuda_options_notify_youngest ();
  remote_callbacks.append_bin ((gdb_byte *) &notify_youngest, sizeof (notify_youngest));
  remote_callbacks.append_separator ();
  unsigned stop_signal = cuda_options_stop_signal ();
  if (stop_signal == GDB_SIGNAL_TRAP)
    remote_callbacks.append_string ("SIGTRAP");
  else
    remote_callbacks.append_string ("SIGURG");

  remote_callbacks.send_request ();
}

void
cuda_remote_query_trace_message ()
{
  if (!cuda_options_debug_general () &&
      !cuda_options_debug_libcudbg () &&
      !cuda_options_debug_notifications ())
    return;

  cuda_remote_send_packet (QUERY_TRACE_MESSAGE);

  const char *str = remote_callbacks.extract_string ();
  while (strcmp ("NO_TRACE_MESSAGE", str) != 0)
    {
      fprintf (stderr, "%s\n", str);

      cuda_remote_send_packet (QUERY_TRACE_MESSAGE);
      str = remote_callbacks.extract_string ();
    }
  fflush (stderr);
}

#ifdef __QNXTARGET__
/* On QNX targets version is queried explicitly */
void
cuda_qnx_version_handshake ()
{
  cuda_remote_send_packet (VERSION_HANDSHAKE);

  const char *version = remote_callbacks.extract_string ();
  cuda_qnx_version_handshake_check (version);
}
#endif /* __QNXTARGET__ */
