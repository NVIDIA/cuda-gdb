/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2026 NVIDIA Corporation
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

#include "cuda-options.h"
#include "cuda-packet-format.h"
#include "cuda-packet-manager.h"
#include "inferior.h"
#include "remote.h"
#include <cstdio>
#include <optional>
#include <string.h>
#include <stdbool.h>
#include <string>
#include <string_view>

#ifdef __QNXTARGET__
#include "cuda-tdep.h"
#include "cuda-utils.h"
#include "cuda/cuda-protocol-hash.h"
#include "remote-nto.h"
#include "target/waitstatus.h"
/* Maximum supported data size in QNX protocol is DS_DATA_MAX_SIZE (1024).
   cuda-gdbserver can be modified to handle 16384 instead, but in order to
   use bigger packets for CUDA, we would need first ensure that they can be
   packed/unpacked by the pdebug putpkt/getpkt functions.

   Until then, use pdebug max allowed size.
   Each DS_DATA_MAX_SIZE can be escaped (*2), 2 frame chars (+2) plus a
   checksum that can be escaped (+2). */
#define PBUFSIZE (DS_DATA_MAX_SIZE * 2 + 4)
#else
#define PBUFSIZE 16384
#endif

struct cuda_remote_callbacks : public send_remote_packet_callbacks
{
public:
  cuda_remote_callbacks ()
  {
    m_recv_buf.reserve (PBUFSIZE);
  }

  cuda_packet_encoder &
  encoder (cuda_packet_type_t packet_type)
  {
    m_encoder.emplace (PBUFSIZE, packet_type);
    return *m_encoder;
  }

  template<typename T>
  void
  get (T *value)
  {
    *value = m_decoder.get<T> ();
  }

  void
  send_request (void)
  {
    gdb_assert (m_encoder.has_value ());
    const std::string_view view = m_encoder->view ();
    gdb::array_view<const char> packet { view.data (), view.size () };
    send_remote_packet (packet, this);
  }

  /* Return a NUL-terminated C string into m_recv_buf for the next
     field.  Pointer is invalidated by the next received ().  */
  char *
  get_string ()
  {
    const std::string_view field = m_decoder.get<std::string_view> ();
    char *const recv_begin = m_recv_buf.data ();
    const size_t offset = field.data () - recv_begin;

    gdb_assert (field.data () >= recv_begin
		&& offset + field.size () < m_recv_buf.size ());

    m_recv_buf[offset + field.size ()] = '\0';
    return recv_begin + offset;
  }

  void
  sending (gdb::array_view<const char> &buf) override
  {
  }

  void
  received (gdb::array_view<const char> &buf) override
  {
    m_recv_buf.assign (buf.begin (), buf.end ());
    const size_t packet_size = m_recv_buf.size ();
    m_recv_buf.push_back ('\0');
    m_decoder.reset (std::string_view (m_recv_buf.data (), packet_size));
  }

private:
  std::optional<cuda_packet_encoder> m_encoder;
  std::string m_recv_buf;
  cuda_packet_decoder m_decoder;
};
static cuda_remote_callbacks remote_callbacks;

static void
cuda_remote_send_packet (cuda_packet_type_t packet_type)
{
  remote_callbacks.encoder (packet_type);

  remote_callbacks.send_request ();
}

static void
cuda_remote_check_ok_response (const char *operation)
{
  const std::string_view response = remote_callbacks.get_string ();
  if (response != "OK")
    error (_("Unexpected CUDA remote %s response: %.*s"), operation,
	   static_cast<int> (response.size ()), response.data ());
}

static bool
cuda_remote_get_return_value ()
{
  bool ret_val;
  remote_callbacks.get (&ret_val);

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
cuda_remote_notification_analyze ([[maybe_unused]] ptid_t ptid,
				  [[maybe_unused]] struct target_waitstatus *ws)
{
#ifdef __QNXTARGET__
  auto &encoder = remote_callbacks.encoder (NOTIFICATION_ANALYZE);
  gdb_assert (ws != nullptr);
  const uint32_t wait_kind = static_cast<uint32_t> (ws->kind ());
  const uint32_t wait_signal
      = (ws->kind () == TARGET_WAITKIND_STOPPED
	 || ws->kind () == TARGET_WAITKIND_SIGNALLED)
	    ? static_cast<uint32_t> (ws->sig ())
	    : static_cast<uint32_t> (GDB_SIGNAL_0);

  encoder.put (static_cast<int32_t> (ptid.pid ()));
  encoder.put (static_cast<int64_t> (ptid.lwp ()));
  encoder.put (wait_kind);
  encoder.put (wait_signal);
#else
  remote_callbacks.encoder (NOTIFICATION_ANALYZE);
#endif

  remote_callbacks.send_request ();
  cuda_remote_check_ok_response ("notification analyze");
}

void
cuda_remote_notification_mark_consumed ()
{
  cuda_remote_send_packet (NOTIFICATION_MARK_CONSUMED);
  cuda_remote_check_ok_response ("notification mark-consumed");
}

void
cuda_remote_notification_consume_pending ()
{
  cuda_remote_send_packet (NOTIFICATION_CONSUME_PENDING);
  cuda_remote_check_ok_response ("notification consume-pending");
}

#ifdef __QNXTARGET__
void
cuda_remote_set_symbols (bool set_extra_symbols, bool *symbols_are_set)
{
  constexpr unsigned char CORE_SYMBOLS_COUNT = 10;
  constexpr unsigned char EXTRA_SYMBOLS_COUNT = 2;
  unsigned char symbols_count = CORE_SYMBOLS_COUNT;

  *symbols_are_set = false;

  /* Old fields are left to maintain the binary compatibility with legacy CUDA
   * GDB server binaries */
  /* Remote side will also check for zeros, here we test only one symbol
     to avoid unnecessary back and forth with it.
     Sent symbols must be kept in sync with those in cuda_symbol_list[] */
  CORE_ADDR address = cuda_get_symbol_address (_STRING_ (CUDBG_IPC_FLAG_NAME));
  if (address == 0)
    {
      return;
    }

  if (set_extra_symbols)
    {
      symbols_count += EXTRA_SYMBOLS_COUNT;
    }

  auto &encoder = remote_callbacks.encoder (SET_SYMBOLS);
  encoder.put (symbols_count);
  encoder.put (address);
  address = cuda_get_symbol_address (_STRING_ (CUDBG_RPC_ENABLED));
  encoder.put (address);
  address = cuda_get_symbol_address (_STRING_ (CUDBG_APICLIENT_PID));
  encoder.put (address);
  address = cuda_get_symbol_address (_STRING_ (CUDBG_APICLIENT_REVISION));
  encoder.put (address);
  address = cuda_get_symbol_address (_STRING_ (CUDBG_SESSION_ID));
  encoder.put (address);
  address
      = cuda_get_symbol_address (_STRING_ (CUDBG_ATTACH_HANDLER_AVAILABLE));
  encoder.put (address);
  address = cuda_get_symbol_address (_STRING_ (CUDBG_DEBUGGER_INITIALIZED));
  encoder.put (address);
  address = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_CODE));
  encoder.put (address);
  address = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE));
  encoder.put (address);
  address = cuda_get_symbol_address (_STRING_ (CUDBG_ENABLE_LAUNCH_BLOCKING));
  encoder.put (address);

  /* All new symbols should be placed under this condition to preserve
     compatibility between newer cuda-gdb and older cuda-gdbserver.
     Recent cuda-gdbserver binaries will gracefully handle more symbols
     than they need, but the old ones won't, so we'll need to only set
     the exact core symbols that they expect, those are defined above. */
  if (set_extra_symbols)
    {
      address = cuda_get_symbol_address (_STRING_ (cudbgInjectionPath));
      encoder.put (address);
      address
	  = cuda_get_symbol_address (_STRING_ (CUDBG_DEBUGGER_CAPABILITIES));
      encoder.put (address);
    }

  remote_callbacks.send_request ();

  remote_callbacks.get (symbols_are_set);
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
  auto &encoder = remote_callbacks.encoder (INITIALIZE_TARGET);
  const bool launch_blocking = cuda_options_launch_blocking ();
  encoder.put (launch_blocking);

  remote_callbacks.send_request ();

  remote_callbacks.get (get_debugger_api_res);
  remote_callbacks.get (set_callback_api_res);
  remote_callbacks.get (initialize_api_res);
  remote_callbacks.get (cuda_initialized);
  remote_callbacks.get (cuda_debugging_enabled);
  remote_callbacks.get (driver_is_compatible);
  remote_callbacks.get (major);
  remote_callbacks.get (minor);
  remote_callbacks.get (revision);
}

void
cuda_remote_query_device_spec (uint32_t dev_id, uint32_t *num_sms,
			       uint32_t *num_warps, uint32_t *num_lanes,
			       uint32_t *num_registers, char **dev_type,
			       char **sm_type)
{
  auto &encoder = remote_callbacks.encoder (QUERY_DEVICE_SPEC);
  encoder.put (dev_id);

  remote_callbacks.send_request ();

  CUDBGResult res;
  remote_callbacks.get (&res);
  if (res != CUDBG_SUCCESS)
    error (_ ("Error: Failed to read device specification (error=%u).\n"),
	   res);
  remote_callbacks.get (num_sms);
  remote_callbacks.get (num_warps);
  remote_callbacks.get (num_lanes);
  remote_callbacks.get (num_registers);
  *dev_type = remote_callbacks.get_string ();
  *sm_type = remote_callbacks.get_string ();
}

bool
cuda_remote_check_pending_sigint ([[maybe_unused]] ptid_t ptid)
{
#ifdef __QNXTARGET__
  auto &encoder = remote_callbacks.encoder (CHECK_PENDING_SIGINT);
  encoder.put (static_cast<int32_t> (ptid.pid ()));
  encoder.put (static_cast<int64_t> (ptid.lwp ()));
#else
  remote_callbacks.encoder (CHECK_PENDING_SIGINT);
#endif

  remote_callbacks.send_request ();

  return cuda_remote_get_return_value ();
}

CUDBGResult
cuda_remote_api_finalize ()
{
  cuda_remote_send_packet (API_FINALIZE);

  CUDBGResult res;
  remote_callbacks.get (&res);
  return res;
}

void
cuda_remote_set_option ()
{
  auto &encoder = remote_callbacks.encoder (SET_OPTION);
  const bool general_trace = cuda_options_debug_general ();
  encoder.put (general_trace);
  const bool libcudbg_trace = cuda_options_debug_libcudbg ();
  encoder.put (libcudbg_trace);
  const bool notifications_trace = cuda_options_debug_notifications ();
  encoder.put (notifications_trace);
  const bool notify_youngest = cuda_options_notify_youngest ();
  encoder.put (notify_youngest);
  const bool driver_logs = cuda_options_driver_logs_enabled ();
  encoder.put (driver_logs);
  const bool printf_flushing = cuda_options_printf_flushing ();
  encoder.put (printf_flushing);

  remote_callbacks.send_request ();
}

void
cuda_remote_query_trace_message ()
{
  if (!cuda_options_debug_general () && !cuda_options_debug_libcudbg ()
      && !cuda_options_debug_notifications ())
    return;

  cuda_remote_send_packet (QUERY_TRACE_MESSAGE);

  const char *str = remote_callbacks.get_string ();
  while (std::string_view (str) != "NO_TRACE_MESSAGE")
    {
      fprintf (stderr, "%s\n", str);

      cuda_remote_send_packet (QUERY_TRACE_MESSAGE);
      str = remote_callbacks.get_string ();
    }
  fflush (stderr);
}

#ifdef __QNXTARGET__
void
cuda_qnx_protocol_hash_handshake ()
{
  auto &encoder = remote_callbacks.encoder (CUDA_PROTOCOL_HASH_HANDSHAKE);
  encoder.put (CUDA_PROTOCOL_HASH);

  remote_callbacks.send_request ();

  const char *const server_hash = remote_callbacks.get_string ();
  if (std::string_view (server_hash) != CUDA_PROTOCOL_HASH)
    error (_ ("CUDA GDB / cuda-gdbserver mismatch: protocol-hash mismatch "
	      "(cuda-gdb=%s, cuda-gdbserver=%s).  Please use cuda-gdb "
	      "and cuda-gdbserver built from the same sources."),
	   CUDA_PROTOCOL_HASH,
	   server_hash);
}
#endif /* __QNXTARGET__ */
