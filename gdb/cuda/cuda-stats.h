/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2025 NVIDIA Corporation
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

#ifndef _CUDA_STATS_H
#define _CUDA_STATS_H 1

#ifndef GDBSERVER
#include "defs.h"
#endif

#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "gdbsupport/traits.h"

/*
 * Support for internal profiling of various CUDA GDB operations.
 */

/* Unified statistics class that combines data storage and timing functionality
 */
class cuda_statistic
{
public:
  cuda_statistic () = default;
  ~cuda_statistic ()
  {
    if (m_timing_active)
      stop_timing ();
  }

  /* Data access methods */
  uint64_t
  count () const
  {
    return m_count;
  }
  std::chrono::nanoseconds
  total_time () const
  {
    return m_total_time;
  }
  std::chrono::nanoseconds
  min_time () const
  {
    return m_min_time;
  }
  std::chrono::nanoseconds
  max_time () const
  {
    return m_max_time;
  }

  std::chrono::nanoseconds
  average_time () const
  {
    return m_count > 0
	       ? std::chrono::nanoseconds (m_total_time.count () / m_count)
	       : std::chrono::nanoseconds::zero ();
  }

  /* Manual timing interface */
  void
  start_timing ()
  {
    if (m_timing_active)
      return;

    m_start_time = std::chrono::steady_clock::now ();
    m_timing_active = true;
  }

  void
  stop_timing ()
  {
    if (!m_timing_active)
      return;

    auto elapsed = calculate_elapsed (m_start_time);
    record_measurement (elapsed);
    m_timing_active = false;
  }

  bool
  is_timing () const
  {
    return m_timing_active;
  }

  /* Direct measurement recording (for external timing) */
  void
  record_measurement (std::chrono::nanoseconds elapsed)
  {
    ++m_count;
    m_total_time += elapsed;
    if (m_count == 1 || elapsed < m_min_time)
      m_min_time = elapsed;
    if (elapsed > m_max_time)
      m_max_time = elapsed;
  }

  /* RAII timing helper */
  class scoped_timer
  {
  public:
    scoped_timer (cuda_statistic &stat)
	: m_stat (stat), m_start (std::chrono::steady_clock::now ())
    {
    }

    /* Delete copy operations */
    scoped_timer (const scoped_timer &) = delete;
    scoped_timer &operator= (const scoped_timer &) = delete;

    /* Enable move operations */
    scoped_timer (scoped_timer &&) = default;
    scoped_timer &operator= (scoped_timer &&) = default;

    ~scoped_timer ()
    {
      if (m_start != std::chrono::steady_clock::time_point{})
	{
	  auto elapsed = cuda_statistic::calculate_elapsed (m_start);
	  m_stat.record_measurement (elapsed);
	}
    }

  private:
    cuda_statistic &m_stat;
    std::chrono::steady_clock::time_point m_start;
  };

  /* Factory method for RAII timing */
  scoped_timer
  time_scope ()
  {
    return scoped_timer (*this);
  }

  void
  reset ()
  {
    m_count = 0;
    m_total_time = std::chrono::nanoseconds::zero ();
    m_min_time = std::chrono::nanoseconds::zero ();
    m_max_time = std::chrono::nanoseconds::zero ();
    m_timing_active = false;
  }

private:
  static std::chrono::nanoseconds
  calculate_elapsed (std::chrono::steady_clock::time_point start)
  {
    const auto end = std::chrono::steady_clock::now ();
    return std::chrono::duration_cast<std::chrono::nanoseconds> (end - start);
  }

  uint64_t m_count = 0;
  std::chrono::nanoseconds m_total_time = std::chrono::nanoseconds::zero ();
  std::chrono::nanoseconds m_min_time = std::chrono::nanoseconds::zero ();
  std::chrono::nanoseconds m_max_time = std::chrono::nanoseconds::zero ();

  /* For manual timing */
  std::chrono::steady_clock::time_point m_start_time;
  bool m_timing_active = false;
};

/* Statistics collection table */
class cuda_statistics_table
{
public:
  cuda_statistics_table (std::string name) : m_name (std::move (name)) {}

  DISABLE_COPY_AND_ASSIGN (cuda_statistics_table);

  ~cuda_statistics_table () = default;

  const std::string &
  name () const
  {
    return m_name;
  }

  cuda_statistic &
  get_statistic (const std::string &name)
  {
    return m_statistics_map[name];
  }

  void
  reset ()
  {
    m_statistics_map.clear ();
  }

  // Modern C++ iteration support
  template <typename Func>
  bool
  foreach_statistic (Func func) const
  {
    for (const auto &pair : m_statistics_map)
      {
	if (!func (pair.first, pair.second))
	  return false;
      }
    return true;
  }

private:
  std::string m_name;
  std::map<std::string, cuda_statistic> m_statistics_map;
};

#endif
