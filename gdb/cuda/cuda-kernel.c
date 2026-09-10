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

#include <string>

#include "frame.h"
#include "ui-out.h"

#include "cuda-api.h"
#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-coord-set.h"
#include "cuda-kernel.h"
#include "cuda-modules.h"
#include "cuda-state.h"
#include "cuda-tdep.h"

//******************************************************************************
//
//                                   Kernel
//
//******************************************************************************

cuda_kernel::cuda_kernel (uint64_t kernel_id, uint32_t dev_id,
			  uint64_t grid_id, uint64_t virt_code_base,
			  cuda_module *module, const CuDim3 &grid_dim,
			  const CuDim3 &block_dim,
			  const CuDim3 &cluster_dim_default,
			  const CuDim3 &cluster_dim_preferred,
			  CUDBGKernelType type)
    : m_id (kernel_id), m_dev_id (dev_id), m_grid_id (grid_id),
      m_module (module), m_virt_code_base (virt_code_base),
      m_grid_dim (grid_dim), m_block_dim (block_dim),
      /* Kernel-ready event will not report cluster info and pass 0,
	 mark the field as invalid and read it later from the grid info. */
      m_cluster_dim_default_p (cluster_dim_default.x != 0),
      m_cluster_dim_default (cluster_dim_default),
      m_cluster_dim_preferred_p (m_cluster_dim_default_p),
      m_cluster_dim_preferred (cluster_dim_preferred), m_grid_status_p (false),
      m_grid_status (CUDBG_GRID_STATUS_INVALID), m_type (type),
      m_launched (false)
{
  // NOTE: Not having an entry function is a normal situation, this means
  // an internal kernel contained in a public module was launched.
  auto name = cuda_find_function_name_from_pc (virt_code_base, true);
  if (name.get () == nullptr)
    m_name = "<internal>";
  else
    m_name = name.get ();

  char dimensions[1024];
  snprintf (dimensions, sizeof (dimensions), "<<<(%u,%u,%u),(%u,%u,%u)>>>",
	    m_grid_dim.x, m_grid_dim.y, m_grid_dim.z, m_block_dim.x,
	    m_block_dim.y, m_block_dim.z);
  m_dimensions = dimensions;
}

void
cuda_kernel::compute_sms_mask (cuda_bitset &mask)
{
  cuda_coords filter{ CUDA_WILDCARD,
		      CUDA_WILDCARD,
		      CUDA_WILDCARD,
		      CUDA_WILDCARD,
		      m_id,
		      CUDA_WILDCARD,
		      CUDA_WILDCARD_DIM,
		      CUDA_WILDCARD_DIM,
		      CUDA_WILDCARD_DIM,
		      CUDA_WILDCARD_DIM };
  cuda_coord_set<cuda_coord_set_type::sms, select_valid,
		 cuda_coord_compare_type::physical>
      coords{ filter };

  // Reset the bitset passed in
  mask.resize (cuda_state::device_get_num_sms (dev_id ()));
  mask.fill (false);
  for (const auto &coord : coords)
    mask.set (coord.physical ().sm ());
}

void
cuda_kernel::invalidate ()
{
  cuda_trace ("kernel %lu: invalidate", m_id);

  m_grid_status_p = false;
  m_cluster_dim_default_p = false;
  m_cluster_dim_preferred_p = false;
}

void
cuda_kernel::populate_args ()
{
  if (m_args)
    {
      cuda_trace ("kernel %lu: populate_args (cached): %s", m_id,
		  m_args->c_str ());
      return;
    }

  if (!cuda_current_focus::isDevice ()
      || (cuda_current_focus::get ().logical ().kernelId () != m_id))
    {
      cuda_trace ("kernel %lu: populate_args - skipping due to lack of device "
		  "focus on kernel",
		  m_id);
      return;
    }

  cuda_trace ("kernel %lu: populate_args", m_id);
  try
    {
      // Find the outermost frame
      frame_info_ptr frame = get_current_frame ();
      frame_info_ptr prev_frame = get_prev_frame (frame);
      while (prev_frame)
	{
	  frame = prev_frame;
	  prev_frame = get_prev_frame (frame);
	}

      // Print the arguments and save the output
      string_file stream;
      current_uiout->redirect (&stream);
      print_args_frame (frame);
      m_args = stream.string ();
      cuda_trace ("kernel %lu: populate_args: %s", m_id, m_args->c_str ());
    }
  catch (const gdb_exception_error &e)
    {
    }

  // Restore environment, do this outside of the try/catch in
  // case an exception was thrown.
  current_uiout->redirect (NULL);
}

const std::string
cuda_kernel::args ()
{
  if (!m_args)
    populate_args ();
  if (m_args)
    return *m_args;
  return "";
}

void
cuda_kernel::get_grid_info ()
{
  CUDBGGridInfo grid_info;
  cuda_debugapi::get_grid_info (m_dev_id, m_grid_id, &grid_info);
  m_cluster_dim_default = grid_info.clusterDim;
  m_cluster_dim_preferred = grid_info.preferredClusterDim;
  m_cluster_dim_default_p = true;
  m_cluster_dim_preferred_p = true;
}

/* This will return the default cluster size, if it is all zeros
   that means no clusters are present and both cluster sizes are ignored. */
const CuDim3 &
cuda_kernel::cluster_dim_default ()
{
  if (!m_cluster_dim_default_p)
    get_grid_info ();

  return m_cluster_dim_default;
}

const CuDim3 &
cuda_kernel::cluster_dim_preferred ()
{
  if (!m_cluster_dim_preferred_p)
    get_grid_info ();

  return m_cluster_dim_preferred;
}

CUDBGGridStatus
cuda_kernel::grid_status ()
{
  if (!m_grid_status_p)
    {
      cuda_debugapi::get_grid_status (m_dev_id, m_grid_id, &m_grid_status);
      m_grid_status_p = true;
    }

  return m_grid_status;
}

bool
cuda_kernel::present ()
{
  const auto status = grid_status ();
  return (status == CUDBG_GRID_STATUS_ACTIVE
	  || status == CUDBG_GRID_STATUS_SLEEPING);
}

void
cuda_kernel::print ()
{
  gdb_printf ("    Kernel %lu:\n", m_id);
  gdb_printf ("        name        : %s\n", m_name.c_str ());
  gdb_printf ("        device id   : %u\n", m_dev_id);
  gdb_printf ("        grid id     : %ld\n", (int64_t)m_grid_id);
  gdb_printf ("        module id   : 0x%lx\n", m_module->id ());
  gdb_printf ("        entry point : 0x%lx\n", m_virt_code_base);
  gdb_printf ("        dimensions  : %s\n", m_dimensions.c_str ());
  gdb_printf ("        launched    : %s\n", m_launched ? "yes" : "no");
  gdb_printf ("        present     : %s\n", present () ? "yes" : "no");
}
