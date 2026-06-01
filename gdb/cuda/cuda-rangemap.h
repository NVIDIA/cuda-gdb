/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2024-2026 NVIDIA Corporation
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

#ifndef _CUDA_RANGEMAP_H
#define _CUDA_RANGEMAP_H 1

#include "defs.h"

#include <map>
#include <optional>

template <typename T> class cuda_rangemap
{
private:
  /* The std::pair can make understanding tricky here.
     pair first == end or end of the range
     pair second == value or the associated value for this range */
  std::map<CORE_ADDR, std::pair<CORE_ADDR, T>> m_ranges;

public:
  cuda_rangemap () = default;
  ~cuda_rangemap () = default;

  void
  add (CORE_ADDR start, size_t size, T value)
  {
    /* Calculate the end of the range */
    const auto end = start + size;
    /* Sanity check - guarantee disjoint ranges */
    /* Find the first range that begins greater than or equal to end.
       This range is guaranteed to start immediately after the range
       we are trying to insert. End is exclusive to a range, so the
       start of the next range can equal the previous ranges end. */
    auto it = m_ranges.lower_bound (end);
    /* Only check if we found a candidate. If the begining range
       starts after end exclusive, we know we have a valid range to insert. */
    if (it != m_ranges.begin ())
      {
	/* Backup the iterator */
	--it;
	/* Now ensure that the previous range end exclusive is
	   prior to start */
	const auto prev_end = it->second.first;
	gdb_assert (prev_end <= start);
      }

    /* Insert the range */
    m_ranges[start] = std::make_pair (end, value);
  }

  /* Remove the entire range containing addr */
  void
  remove_range (CORE_ADDR addr)
  {
    auto it = find (addr);
    if (it != m_ranges.cend ())
      m_ranges.erase (it);
  }

  std::optional<T>
  get (CORE_ADDR addr) const
  {
    const auto it = find (addr);
    if (it == m_ranges.cend ())
      return {};
    /* Return the stored value */
    const auto &value = it->second.second;
    return { value };
  }

private:
  /* Internal method used to get an iterator to a range containing addr */
  using const_iterator = typename decltype (m_ranges)::const_iterator;
  const_iterator
  find (CORE_ADDR addr) const
  {
    /* Find the first range that begins greater than addr. */
    auto it = m_ranges.upper_bound (addr);

    /* If the first range starts after addr, it is not contained */
    if (it == m_ranges.cbegin ())
      return m_ranges.cend ();

    /* Backup the iterator */
    --it;

    /* At this point we are guaranteed that this is the first range that
       begins prior to or at addr. We require ranges to be disjoint in this
       data structure. We only need to check to see if addr is contained by end
       exclusive. */
    const auto end = it->second.first;
    if (end < addr)
      return m_ranges.cend ();

    /* Range contained by addr found. */
    return it;
  }
};

#endif