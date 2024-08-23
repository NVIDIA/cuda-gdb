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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */
#include "defs.h"

#include "cuda-bitset.h"

#include <iomanip>
#include <sstream>
#include <string>

static void
bits_to_hex (std::stringstream &ss, const cuda_bitset& bits, size_t &pos, const int nbits)
{
  gdb_assert (nbits <= 8);

  // The current byte we are constructing
  int cur_byte = 0;

  // This needs to handle the case where we are constructing less than a full
  // byte
  for (auto i = 0; i < nbits; ++i)
    {
      // Shift cur_byte left
      cur_byte <<= 1;

      // Assign this bit position
      if (pos < bits.size ())
	cur_byte |= bits[pos];
      pos--;
    }

  // Write the hex value to the output
  ss << std::setfill ('0') << std::setw (2) << std::hex << cur_byte;
}

std::string
cuda_bitset::to_hex_string () const
{
  // setup the result stream
  std::stringstream ss;

  ss << "0x";

  // Start construction from the high order bit based on the minimum
  // bitset size of 32 bits.
  auto mask_len = std::max (size (), (uint32_t)32UL);
  size_t pos = mask_len - 1;

  // Handle the case where we don't have a number of bits divisible by bytes
  auto rem = mask_len % 8;
  if (rem)
    bits_to_hex (ss, *this, pos, rem);

  // Ensure pos is on a byte boundary
  gdb_assert (((pos + 1) % 8) == 0);

  // Construct and output the remaining bytes a byte at a time
  for (auto num = 0; num < (mask_len / 8); ++num)
    bits_to_hex (ss, *this, pos, 8);

  return ss.str ();
}
