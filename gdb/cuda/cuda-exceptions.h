/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2015-2024 NVIDIA Corporation
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

#ifndef _CUDA_EXCEPTIONS_H
#define _CUDA_EXCEPTIONS_H 1

#include "gdb/signals.h"

#include "cuda-coords.h"
#include "cuda-defs.h"

class cuda_exception
{
private:
  /* Data */
  bool m_valid;
  bool m_recoverable;
  enum gdb_signal m_gdb_sig;
  cuda_coords m_coord;

  /* Methods */
  void print_exception_origin () const;
  void print_exception_device () const;
  void print_assert_message () const;
  void print_exception_name () const;
  void print_cluster_exception_origin () const;

public:
  /* CTOR/DTOR */
  cuda_exception ();
  ~cuda_exception () = default;
  cuda_exception (const cuda_exception &) = default;
  cuda_exception (cuda_exception &&) = default;

  /* Methods */
  bool
  valid () const
  {
    return m_valid;
  }

  bool
  recoverable () const
  {
    gdb_assert (m_valid);
    return m_recoverable;
  }

  enum gdb_signal
  gdbSignal () const
  {
    gdb_assert (m_valid);
    return m_gdb_sig;
  }

  const bool
  has_coords () const
  {
    return m_coord.valid ();
  }

  const cuda_coords &
  coords () const
  {
    gdb_assert (m_valid);
    gdb_assert (m_coord.valid ());
    return m_coord;
  }

  void printMessage () const;
  const char *name () const;

  static const char *type_to_name (CUDBGException_t type);
};

#endif
