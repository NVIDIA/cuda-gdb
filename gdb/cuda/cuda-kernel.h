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

#ifndef _CUDA_KERNEL_H
#define _CUDA_KERNEL_H 1

#include "cuda-context.h"
#include "cuda-defs.h"
#include "cuda-modules.h"

#include <string>
#include <unordered_map>
#include <vector>

class cuda_bitset;

class cuda_kernel
{
public:
  cuda_kernel (uint64_t kernel_id, uint32_t dev_id, uint64_t grid_id,
	       uint64_t virt_code_base, cuda_module *module,
	       const CuDim3 &grid_dim, const CuDim3 &block_dim,
	       const CuDim3 &cluster_dim_default, const CuDim3 &cluster_dim_preferred,
               CUDBGKernelType type, CUDBGKernelOrigin origin, uint64_t parent_grid_id);

  const uint32_t
  dev_id () const
  {
    return m_dev_id;
  }

  const uint64_t
  id () const
  {
    return m_id;
  }

  const std::string &
  name () const
  {
    return m_name;
  }

  const uint64_t
  grid_id () const
  {
    return m_grid_id;
  }

  const CUDBGKernelOrigin
  get_origin () const
  {
    return m_origin;
  }

  const uint64_t
  parent_grid_id () const
  {
    return m_parent_grid_id;
  }

  const uint64_t
  virt_code_base () const
  {
    return m_virt_code_base;
  }

  cuda_module *
  module () const
  {
    return m_module;
  }

  const CuDim3 &
  grid_dim () const
  {
    return m_grid_dim;
  }

  const CuDim3 &
  block_dim () const
  {
    return m_block_dim;
  }

  const CuDim3 &cluster_dim_default ();
  const CuDim3 &cluster_dim_preferred ();

  const std::string &
  dimensions () const
  {
    return m_dimensions;
  }

  const CUDBGKernelType
  type () const
  {
    return m_type;
  }

  const bool
  launched () const
  {
    return m_launched;
  }

  void
  launched (bool value)
  {
    m_launched = value;
  }

  cuda_context *
  context ()
  {
    gdb_assert (m_module);
    return m_module->context ();
  }

  uint32_t depth ();
  std::vector<cuda_kernel *> children ();

  CUDBGGridStatus grid_status ();

  bool present ();

  void populate_args ();

  const std::string args ();

  void invalidate ();
  void compute_sms_mask (cuda_bitset &mask);

  bool should_print_kernel_event ();

  void print ();

private:
  void get_grid_info ();

  uint64_t m_id;      // unique kernel id per GDB session
  uint32_t m_dev_id;  // device where the kernel was launched
  uint64_t m_grid_id; // unique kernel id per device

  cuda_module *m_module;     // cuda_module of the kernel
  uint64_t m_virt_code_base; // virtual address of the kernel entry point

  CuDim3 m_grid_dim;  // The grid dimensions of the kernel
  CuDim3 m_block_dim; // The block dimensions of the kernel.

  bool m_cluster_dim_default_p; // Is the default cluster dimension valid?
  CuDim3 m_cluster_dim_default; // The default cluster dimension

  bool m_cluster_dim_preferred_p; // Is the preferred cluster dimension valid?
  CuDim3 m_cluster_dim_preferred; // The preferred cluster dimension

  std::string m_name;		     // name of the kernel if available
  std::string m_dimensions;	     // A string repr. of the kernel dimensions
  gdb::optional<std::string> m_args; // kernel arguments in string format

  bool m_grid_status_p;
  CUDBGGridStatus m_grid_status; // current grid status of the kernel

  CUDBGKernelType m_type;     // The kernel type: system or application.
  CUDBGKernelOrigin m_origin; // The kernel origin: CPU or GPU
  uint64_t m_parent_grid_id;  // The kernel that launched this grid (for origin
			      // == GPU)

  bool m_depth_p;   // Is the kernel depth valid?
  uint32_t m_depth; // kernel nest level (0 - host launched kernel)

  bool m_children_p;			 // Are the children kernels valid?
  std::vector<cuda_kernel *> m_children; // children kernels

  bool m_launched; // Has the kernel been seen on the HW?
};

#endif
