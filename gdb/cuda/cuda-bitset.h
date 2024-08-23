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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _CUDA_BITSET_H
#define _CUDA_BITSET_H 1

#include "defs.h"

#include <string>
#include <vector>

// A simple, variable-length bitset class Mostly used for
// variable-length SM masks, so that we don't depend on CUDBG_MAX_SMS
// for compatibility with later debugger backends which may support
// more in the future.

class cuda_bitset {
public:
  cuda_bitset ()
    : m_bit_count (0), m_rounding (sizeof (uint8_t))
  {
  }

  cuda_bitset (uint32_t bits, bool default_value = false, uint32_t length = 0)
    : m_bit_count (bits)
  {
    // Can't specify a length shorter than the minimal required
    gdb_assert (!length || length*bits_per_byte >= m_bit_count);

    // If default length (0) is passed in, set the length to the
    // minimal number of bytes.
    if (length)
      m_data.resize (length);
    else
      m_data.resize ((m_bit_count + bits_per_byte - 1) / bits_per_byte);

    // std::vector.resize() fills the vector with 0s as it was zero
    // length to start with, so we only have to call fill() if
    // the default_value is true.
    if (default_value)
      fill (default_value);
  }

  cuda_bitset (const cuda_bitset &) = default;
  cuda_bitset (cuda_bitset &&) = default;
  cuda_bitset &operator= (const cuda_bitset &) = default;
  cuda_bitset &operator= (cuda_bitset &&) = default;

  ~cuda_bitset () = default;
  
  // Set the length of the bitset. Old bits are left in place. Follow
  // this call by fill() if the bitset should be reinitialed to all
  // 0's or all 1's
  void resize (uint32_t bit_count, uint32_t len = 0)
  {
    // Update this first so we can use length()
    m_bit_count = bit_count;

    // length() is the minimal number of bytes required to
    // represent m_bit_count bits.
    m_data.resize (len ? len : length ());

    clear_upper_bits ();
  }

  // How many bits in the bitset
  uint32_t size () const
  { return m_bit_count; }

  // Pointer to the underlying data
  // Used for cuda_debugapi calls with .vector_length()
  void *data ()
  { return m_data.data (); }

  // Copy the bits from the provided buffer
  void read (uint8_t *buffer, size_t len)
  {
    gdb_assert (len <= m_data.size ());
    memcpy (m_data.data (), buffer, len);
    clear_upper_bits ();
  }

  // Copy the bits into the provided buffer
  void write (uint8_t *buffer, size_t len)
  {
    gdb_assert (len <= m_data.size ());
    memcpy (buffer, m_data.data (), len);
  }

  // Minimal number of bytes to store m_bit_count w/o including rounding
  size_t length () const
  { return (m_bit_count + bits_per_byte - 1) / bits_per_byte; }

  // Length of the underlying vector in bytes (may be rounded up from
  // minimal number of bytes to represent m_bit_count bits due to m_rounding > 1).
  size_t vector_length () const
  { return m_data.size (); }

  // Return the specified bit
  uint32_t get (uint32_t bit) const
  {
    gdb_assert (bit < m_bit_count);
    gdb_assert ((bit / bits_per_byte) < m_data.size ());
    return (m_data[bit / bits_per_byte] & (1ULL << (bit % bits_per_byte))) ? 1 : 0;
  }

  // Alias for .get (bit)
  const uint32_t operator[] (uint32_t bit) const
  { return get (bit); }

  // Set the specified bit to 1 or 0
  void set (uint32_t bit, bool value = true)
  {
    gdb_assert (bit < m_bit_count);
    gdb_assert ((bit / bits_per_byte) < m_data.size ());

    if (value)
      m_data[bit / bits_per_byte] |= (1ULL << (bit % bits_per_byte));
    else
      m_data[bit / bits_per_byte] &= ~(1ULL << (bit % bits_per_byte));
  }

  // Set all bits to the specified value
  // If setting to 1's, make sure we don't set any bits beyond m_bit_count.
  // This makes bitset comparisons much easier
  void fill (bool value)
  {
    if (value)
      {
	std::fill (m_data.begin (), m_data.end (), 0xff);
	clear_upper_bits ();
      }
    else
      std::fill (m_data.begin (), m_data.end (), 0x00);
  }

  // Return true if any bits are set, false otherwise
  bool any () const
  {
    for (const auto& data : m_data)
      if (data)
	return true;
    return false;
  }

  // Return true if no bits are set, false otherwise
  bool none () const
  {
    for (const auto& data : m_data)
      if (data)
	return false;
    return true;
  }

  bool operator!=(const cuda_bitset& other)
  {
    if (m_bit_count != other.m_bit_count)
      return true;
    if (m_data != other.m_data)
      return true;
    return false;
  }

  bool operator==(const cuda_bitset& other)
  {
    if (m_bit_count != other.m_bit_count)
      return false;
    if (m_data != other.m_data)
      return false;
    return true;
  }

  // Return hex representation of the bitset
  std::string to_hex_string () const;

private:
  void clear_upper_bits ()
  {
    // Now 0-out any trailing bits
    for (auto bit = m_bit_count; bit < (m_data.size () * bits_per_byte); bit++)
      m_data[bit / bits_per_byte] &= ~(1ULL << (bit % bits_per_byte));
  }

  // Number of valid bits in the bitset
  uint32_t              m_bit_count;

  // Rounding factor (in bytes)
  uint32_t              m_rounding;

  // A vector of uint64_t holding the bits
  std::vector<uint8_t>	m_data;

  // Some helpful constants
  static constexpr uint32_t bits_per_byte = 8*sizeof (uint8_t);
};

#endif
