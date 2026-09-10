/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2026 NVIDIA Corporation
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

#ifndef GDB_CUDA_CUDA_PACKET_FORMAT_H
#define GDB_CUDA_CUDA_PACKET_FORMAT_H

#include "gdbsupport/array-view.h"
#include "gdbsupport/common-utils.h"
#include "gdbsupport/errors.h"
#include "gdbsupport/rsp-low.h"

#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

static constexpr std::string_view cuda_packet_prefix = "qnv.";

template <typename T>
using is_scalar_field
    = std::integral_constant<bool, std::is_integral<T>::value
				       || std::is_enum<T>::value>;

class cuda_packet_encoder
{
public:
  explicit cuda_packet_encoder (size_t max_packet_size)
      : m_max_packet_size (max_packet_size)
  {
    gdb_assert (max_packet_size > 0);
  }

  template <typename T>
  explicit cuda_packet_encoder (size_t max_packet_size, const T &packet_type)
      : cuda_packet_encoder (max_packet_size)
  {
    m_buffer.append (cuda_packet_prefix);
    put (packet_type);
  }

  std::string_view
  view () const noexcept
  {
    return m_buffer;
  }

  template <typename T>
  void
  put (const T &value)
  {
    static auto ensure_space
	= [] (size_t used, size_t requested, size_t max_packet_size) {
	    const size_t max_payload_size = max_packet_size - 1;
	    if (used > max_payload_size || requested > max_payload_size - used)
	      error ("cuda packet write exceeds capacity "
		     "(used=%zu, requested=%zu, max-size=%zu)",
		     used, requested, max_packet_size);
	  };

    if (std::exchange (m_has_fields, true))
      {
	ensure_space (m_buffer.size (), 1, m_max_packet_size);
	m_buffer.push_back (';');
      }

    using value_type = std::decay_t<T>;

    if constexpr (std::is_convertible<const T &, std::string_view>::value)
      {
	const std::string_view str (value);
	ensure_space (m_buffer.size (), str.size (), m_max_packet_size);
	m_buffer.append (str);
      }
    else if constexpr (is_scalar_field<value_type>::value)
      {
	const auto bytes = gdb::make_array_view (
	    reinterpret_cast<const gdb_byte *> (&value), sizeof (value));
	const size_t hex_encoded_size = bytes.size () * 2;
	ensure_space (m_buffer.size (), hex_encoded_size, m_max_packet_size);
	const size_t old_size = m_buffer.size ();
	m_buffer.resize (old_size + hex_encoded_size);
	char *out = m_buffer.data () + old_size;
	for (gdb_byte byte : bytes)
	  out = pack_hex_byte (out, byte);
      }
    else
      static_assert (std::is_same<value_type, void>::value,
		     "cuda packet template fields must be string views or "
		     "scalar wire fields");
  }

private:
  DISABLE_COPY_AND_ASSIGN (cuda_packet_encoder);

  std::string m_buffer;
  bool m_has_fields = false;
  const size_t m_max_packet_size;
};

class cuda_packet_decoder
{
public:
  cuda_packet_decoder () = default;
  explicit cuda_packet_decoder (const std::string_view &packet)
  {
    reset (packet);
  }

  void
  reset (const std::string_view &packet)
  {
    m_packet = packet;
  }

  template <typename T>
  T
  get ()
  {
    if (!m_packet)
      error ("cuda packet exhausted: no more fields to read");

    std::string_view field;
    const size_t pos = m_packet->find (';');
    if (pos == std::string_view::npos)
      {
	field = *m_packet;
	m_packet.reset ();
      }
    else
      {
	field = m_packet->substr (0, pos);
	m_packet->remove_prefix (pos + 1);
      }

    using value_type = std::decay_t<T>;

    if constexpr (std::is_same<value_type, std::string_view>::value)
      return field;
    else if constexpr (is_scalar_field<value_type>::value)
      {
	value_type value;
	auto bytes
	    = gdb::make_array_view (reinterpret_cast<gdb_byte *> (&value),
				    sizeof (value));
	const size_t expected_size = bytes.size () * 2;

	if (field.size () != expected_size)
	  error ("cuda packet field has wrong size "
		 "(expected %zu hex chars, got %zu)",
		 expected_size, field.size ());

	if (field == "OK" || field == "MP")
	  error ("unexpected CUDA packet envelope in scalar field: %.*s",
		 static_cast<int> (field.size ()), field.data ());

	if (hex2bin (field.data (), bytes.data (), bytes.size ())
	    != bytes.size ())
	  error ("cuda packet field contains invalid hex");
	return value;
      }
    else
      static_assert (std::is_same<value_type, void>::value,
		     "cuda packet template fields must be string views or "
		     "scalar wire fields");
  }

  template <typename T>
  T
  get_packet_type ()
  {
    if (!m_packet || !startswith (*m_packet, cuda_packet_prefix))
      error ("unknown cuda packet\n");

    m_packet->remove_prefix (cuda_packet_prefix.size ());
    return get<T> ();
  }

private:
  std::optional<std::string_view> m_packet;
};

#endif /* GDB_CUDA_CUDA_PACKET_FORMAT_H */
