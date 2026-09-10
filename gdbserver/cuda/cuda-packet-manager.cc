/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2013-2026 NVIDIA Corporation
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

#include "cuda/cuda-packet-manager.h"
#include "cuda-tdep-server.h"
#include "cuda/cuda-notifications.h"
#include "cuda/libcudbgipc.h"
#include "cudadebugger.h"
#include "server.h"
#ifdef __QNXHOST__
#include "cuda/cuda-protocol-hash.h"
#include "remote-nto.h"
#endif /* __QNXHOST__ */
#include "cuda/cuda-packet-format.h"
#include "gdbsupport/array-view.h"
#include "gdbsupport/byte-vector.h"
#include "gdbsupport/rsp-low.h"

#include <array>
#include <cstdlib>
#ifdef __QNXHOST__
#include <string>
#endif
#include <string_view>

#ifndef __QNXHOST__
/* We don't have gdbserver-managed structures on QNX */
extern ptid_t cuda_last_ptid;
extern struct target_waitstatus cuda_last_ws;
#endif

static uint32_t cuda_debugapi_version_major;
static uint32_t cuda_debugapi_version_minor;
static uint32_t cuda_debugapi_version_revision;

extern void cuda_gdbserver_set_api_version (uint32_t major, uint32_t minor,
					    uint32_t revision);

void
cuda_gdbserver_set_api_version (uint32_t major, uint32_t minor,
				uint32_t revision)
{
  cuda_debugapi_version_major = major;
  cuda_debugapi_version_minor = minor;
  cuda_debugapi_version_revision = revision;
}

static void
cuda_process_notification_analyze_packet (cuda_packet_decoder &decoder,
					  cuda_packet_encoder &encoder)
{
#ifdef __QNXHOST__
  const int32_t pid = decoder.get<int32_t> ();
  const int64_t lwp = decoder.get<int64_t> ();
  const ptid_t cuda_last_ptid
      = ptid_t (static_cast<ptid_t::pid_type> (pid),
		static_cast<ptid_t::lwp_type> (lwp), 0);
  const uint32_t wait_kind = decoder.get<uint32_t> ();
  const uint32_t wait_signal = decoder.get<uint32_t> ();

  /* QNX analyzer only acts on STOPPED + EMT/ILL; everything else is
     IGNORE so the analyzer skips it.  */
  struct target_waitstatus cuda_last_ws;
  switch (static_cast<target_waitkind> (wait_kind))
    {
    case TARGET_WAITKIND_STOPPED:
      cuda_last_ws.set_stopped (static_cast<gdb_signal> (wait_signal));
      break;
    case TARGET_WAITKIND_SIGNALLED:
      cuda_last_ws.set_signalled (static_cast<gdb_signal> (wait_signal));
      break;
    default:
      cuda_last_ws.set_ignore ();
      break;
    }
#endif /* __QNXHOST__ */
  cuda_notification_analyze (cuda_last_ptid, &cuda_last_ws);
  encoder.put ("OK");
}

static void
cuda_process_notification_received_packet (cuda_packet_encoder &encoder)
{
  const bool received = cuda_notification_received ();
  encoder.put (received);
}

static void
cuda_process_notification_pending_packet (cuda_packet_encoder &encoder)
{
  const bool pending = cuda_notification_pending ();
  encoder.put (pending);
}

static void
cuda_process_notification_mark_consumed_packet (cuda_packet_encoder &encoder)
{
  cuda_notification_mark_consumed ();
  encoder.put ("OK");
}

static void
cuda_process_notification_consume_pending_packet (cuda_packet_encoder &encoder)
{
  cuda_notification_consume_pending ();
  encoder.put ("OK");
}

static void
cuda_process_notification_aliased_event_packet (cuda_packet_encoder &encoder)
{
  const bool aliased_event = cuda_notification_aliased_event ();
  if (aliased_event)
    cuda_notification_reset_aliased_event ();
  encoder.put (aliased_event);
}

#ifdef __QNXHOST__
static void
cuda_process_set_symbols (cuda_packet_decoder &decoder,
			  cuda_packet_encoder &encoder)
{
  bool symbols_are_set = false;

  const unsigned char symbols_count = decoder.get<unsigned char> ();
  /* For compatibility with newer cuda-gdb binaries we handle packets that
     provide more symbols than we have statically built with. */
  const int server_symbols_count = cuda_get_symbol_cache_size ();
  if (symbols_count >= server_symbols_count)
    {
      symbols_are_set = true;
      for (int i = 0; i < server_symbols_count; i++)
	{
	  const CORE_ADDR address = decoder.get<CORE_ADDR> ();
	  if (address == 0)
	    {
	      symbols_are_set = false;
	      break;
	    }
	  cuda_symbol_list[i].addr = address;
	}
    }

  encoder.put (symbols_are_set);
}
#endif /* __QNXHOST__ */

static void
cuda_process_initialize_target_packet (cuda_packet_decoder &decoder,
				       cuda_packet_encoder &encoder)
{
  /* Extract options that need to be set before initialization */
  cuda_launch_blocking = decoder.get<bool> ();

  const bool driver_is_compatible = cuda_initialize_target ();

  encoder.put (get_debugger_api_res);
  encoder.put (set_callback_api_res);
  encoder.put (api_initialize_res);
  encoder.put (cuda_initialized);
  encoder.put (cuda_debugging_enabled);
  encoder.put (driver_is_compatible);
  encoder.put (cuda_debugapi_version_major);
  encoder.put (cuda_debugapi_version_minor);
  encoder.put (cuda_debugapi_version_revision);
}

static void
cuda_process_query_device_spec_packet (cuda_packet_decoder &decoder,
				       cuda_packet_encoder &encoder)
{
  CUDBGResult res;
  std::array<char, 256> device_type{};
  std::array<char, 16> sm_type{};
  uint32_t num_sms = 0;
  uint32_t num_warps = 0;
  uint32_t num_lanes = 0;
  uint32_t num_registers = 0;

  const uint32_t dev = decoder.get<uint32_t> ();

  res = cudbgAPI->getNumSMs (dev, &num_sms);
  if (res == CUDBG_SUCCESS)
    res = cudbgAPI->getNumWarps (dev, &num_warps);
  if (res == CUDBG_SUCCESS)
    res = cudbgAPI->getNumLanes (dev, &num_lanes);
  if (res == CUDBG_SUCCESS)
    res = cudbgAPI->getNumRegisters (dev, &num_registers);
  if (res == CUDBG_SUCCESS)
    res = cudbgAPI->getDeviceType (dev, device_type.data (),
				   device_type.size ());
  if (res == CUDBG_SUCCESS)
    res = cudbgAPI->getSmType (dev, sm_type.data (), sm_type.size ());

  encoder.put (res);
  encoder.put (num_sms);
  encoder.put (num_warps);
  encoder.put (num_lanes);
  encoder.put (num_registers);
  encoder.put (device_type.data ());
  encoder.put (sm_type.data ());
}

static void
cuda_process_check_pending_sigint_packet (cuda_packet_decoder &decoder,
					  cuda_packet_encoder &encoder)
{
#ifdef __QNXHOST__
  const int32_t pid = decoder.get<int32_t> ();
  const int64_t lwp = decoder.get<int64_t> ();
  const ptid_t cuda_last_ptid
      = ptid_t (static_cast<ptid_t::pid_type> (pid),
		static_cast<ptid_t::lwp_type> (lwp), 0);
#endif
  const bool ret_val = cuda_check_pending_sigint (cuda_last_ptid);
  encoder.put (ret_val);
}

static void
cuda_process_api_finalize_packet (cuda_packet_encoder &encoder)
{
  /* If finalize() has been called in cuda_cleanup(), then return the
     recorded cudbgAPI result. */
  const CUDBGResult res
      = cuda_initialized ? cudbgAPI->finalize () : api_finalize_res;
  encoder.put (res);
}

static void
cuda_process_set_option_packet (cuda_packet_decoder &decoder,
				cuda_packet_encoder &encoder)
{
  cuda_debug_general = decoder.get<bool> ();
  cuda_debug_libcudbg = decoder.get<bool> ();
  cuda_debug_notifications = decoder.get<bool> ();
  cuda_notify_youngest = decoder.get<bool> ();
  cuda_driver_logs = decoder.get<bool> ();
  cuda_printf_flushing = decoder.get<bool> ();

  /* Apply the runtime option */
  cuda_set_driver_logging (cuda_driver_logs);

  encoder.put ("OK");
}

static void
cuda_process_query_trace_message (cuda_packet_encoder &encoder)
{
  if (cuda_trace_messages.empty ())
    {
      encoder.put ("NO_TRACE_MESSAGE");
      return;
    }

  encoder.put (cuda_trace_messages.front ());
  cuda_trace_messages.pop_front ();
}

#ifdef __QNXHOST__
static void
cuda_process_protocol_hash_handshake (cuda_packet_decoder &decoder,
				      cuda_packet_encoder &encoder)
{
  const std::string_view client_hash = decoder.get<std::string_view> ();
  if (client_hash != CUDA_PROTOCOL_HASH)
    {
      const std::string client_hash_str (client_hash);
      error ("CUDA GDB / cuda-gdbserver mismatch: protocol-hash mismatch "
	     "(cuda-gdb=%s, cuda-gdbserver=%s).  Please use cuda-gdb "
	     "and cuda-gdbserver built from the same sources.\n",
	     client_hash_str.c_str (), CUDA_PROTOCOL_HASH);
    }

  encoder.put (CUDA_PROTOCOL_HASH);
}
#endif /* __QNXHOST__ */

void
handle_cuda_packet (cuda_packet_decoder &decoder, cuda_packet_encoder &encoder)
{
  const cuda_packet_type_t packet_type
      = decoder.get_packet_type<cuda_packet_type_t> ();

  switch (packet_type)
    {
    case NOTIFICATION_ANALYZE:
      cuda_process_notification_analyze_packet (decoder, encoder);
      break;
    case NOTIFICATION_PENDING:
      cuda_process_notification_pending_packet (encoder);
      break;
    case NOTIFICATION_RECEIVED:
      cuda_process_notification_received_packet (encoder);
      break;
    case NOTIFICATION_ALIASED_EVENT:
      cuda_process_notification_aliased_event_packet (encoder);
      break;
    case NOTIFICATION_MARK_CONSUMED:
      cuda_process_notification_mark_consumed_packet (encoder);
      break;
    case NOTIFICATION_CONSUME_PENDING:
      cuda_process_notification_consume_pending_packet (encoder);
      break;
    case INITIALIZE_TARGET:
      cuda_process_initialize_target_packet (decoder, encoder);
      break;
    case API_FINALIZE:
      cuda_process_api_finalize_packet (encoder);
      break;
    case QUERY_DEVICE_SPEC:
      cuda_process_query_device_spec_packet (decoder, encoder);
      break;
    case QUERY_TRACE_MESSAGE:
      cuda_process_query_trace_message (encoder);
      break;
    case CHECK_PENDING_SIGINT:
      cuda_process_check_pending_sigint_packet (decoder, encoder);
      break;
    case SET_OPTION:
      cuda_process_set_option_packet (decoder, encoder);
      break;
#ifdef __QNXHOST__
    case SET_SYMBOLS:
      cuda_process_set_symbols (decoder, encoder);
      break;
    case CUDA_PROTOCOL_HASH_HANDSHAKE:
      cuda_process_protocol_hash_handshake (decoder, encoder);
      break;
#endif /* __QNXHOST__ */
    default:
      error ("unknown cuda packet type: %u\n", (uint32_t)packet_type);
    }
}

void
cuda_append_api_finalize_res (char *buf)
{
  gdb_assert (buf);
  xsnprintf (buf, 64, ";cuda_finalize:%x", api_finalize_res);
}

static void
write_error_response (gdb::array_view<char> response, int err,
		      int *new_packet_len)
{
  gdb_assert (response.size () >= 4);
  xsnprintf (response.data (), 4, "E%02d", err);
  *new_packet_len = 3;
}

static void
write_ok_response (gdb::array_view<char> buf,
		   gdb::array_view<const gdb_byte> payload, bool *truncated,
		   int *new_packet_len)
{
  static constexpr std::string_view kOK = "OK;";
  static constexpr std::string_view kMP = "MP";

  constexpr size_t prefix = kOK.size ();
  gdb_assert (buf.size () > prefix);
  kOK.copy (buf.data (), prefix);
  int out_len = 0;
  const int escaped = remote_escape_output (
      payload.data (), payload.size (), 1, (gdb_byte *)buf.data () + prefix,
      &out_len, buf.size () - prefix);
  *truncated = (size_t)out_len != payload.size ();
  if (*truncated)
    kMP.copy (buf.data (), kMP.size ());
  *new_packet_len = (int)prefix + escaped;
}

int
handle_vCuda (std::string_view request, gdb::array_view<char> response,
	      int *new_packet_len)
{
  static constexpr std::string_view kVCUDARetr = "vCUDARetr;";
  static constexpr std::string_view kVCUDA = "vCUDA;";

  static gdb::byte_vector last_vcuda_reply;

  if (startswith (request, kVCUDARetr))
    {
      /* The offset field is parsed by strtoul, which scans until a
	 non-numeric byte.  The request buffer is NUL-terminated past
	 the request, so this is safe.  */
      const size_t offset
	  = std::strtoul (request.data () + kVCUDARetr.size (), nullptr, 10);

      if (last_vcuda_reply.empty () || offset >= last_vcuda_reply.size ())
	{
	  write_error_response (response, EINVAL, new_packet_len);
	  return 1;
	}

      bool truncated;
      write_ok_response (
	  response,
	  gdb::make_array_view (last_vcuda_reply.data () + offset,
				last_vcuda_reply.size () - offset),
	  &truncated, new_packet_len);
      if (!truncated)
	last_vcuda_reply.clear ();
      return 1;
    }

  last_vcuda_reply.clear ();

  if (!startswith (request, kVCUDA))
    {
      write_error_response (response, EINVAL, new_packet_len);
      return 1;
    }

  const auto *escaped_payload
      = (const gdb_byte *)(request.data () + kVCUDA.size ());
  const int payload_len = (int)(request.size () - kVCUDA.size ());

  static gdb::byte_vector input;
  input.resize ((size_t)payload_len);

  const int unescaped_len = remote_unescape_input (
      escaped_payload, payload_len, input.data (), input.size ());

  CUDBGResult res = cudbgipcAppend (input.data (), unescaped_len);
  if (res != CUDBG_SUCCESS)
    {
      write_error_response (response, res, new_packet_len);
      return 1;
    }

  void *ipc_reply = nullptr;
  size_t ipc_reply_size = 0;
  res = cudbgipcRequest (&ipc_reply, &ipc_reply_size);
  if (res != CUDBG_SUCCESS)
    {
      write_error_response (response, res, new_packet_len);
      return 1;
    }

  const auto *reply_bytes = (const gdb_byte *)ipc_reply;
  bool truncated;
  write_ok_response (response,
		     gdb::make_array_view (reply_bytes, ipc_reply_size),
		     &truncated, new_packet_len);
  if (truncated)
    last_vcuda_reply.assign (reply_bytes, reply_bytes + ipc_reply_size);
  return 1;
}
