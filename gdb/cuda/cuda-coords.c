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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#include "defs.h"

#include "command.h"
#include "inferior.h"
#include "language.h"
#include "target.h"
#include "ui-out.h"

#include "cuda-coords.h"
#include "cuda-coord-set.h"
#include "cuda-options.h"
#include "cuda-state.h"

#include <sstream>
#include <string>

/* Static globals */
cuda_current_focus cuda_current_focus::m_current_focus{};

/* Methods */
cuda_coords_physical::cuda_coords_physical (uint32_t dev, uint32_t sm,
                                            uint32_t wp, uint32_t ln)
    : m_dev{ dev }, m_sm{ sm }, m_wp{ wp }, m_ln{ ln }
{
  // Replace CUDA_CURRENT with coords
  if (cuda_current_focus::isDevice ())
    {
      const auto &focus = cuda_current_focus::get ().physical ();
      cuda_evaluate_current (m_dev, focus.dev ());
      cuda_evaluate_current (m_sm, focus.sm ());
      cuda_evaluate_current (m_wp, focus.wp ());
      cuda_evaluate_current (m_ln, focus.ln ());
    }
}

bool
cuda_coords_physical::isValidOnDevice (
    const cuda_coords_logical &expected) const
{
  /* Ensure the physical coords are valid on the device */
  if (!cuda_state::sm_valid (m_dev, m_sm)
      || !cuda_state::warp_valid (m_dev, m_sm, m_wp)
      || !cuda_state::lane_valid (m_dev, m_sm, m_wp, m_ln))
    return false;

  /* Grab the kernel for the phyiscal coords */
  auto kernel = cuda_state::warp_get_kernel (m_dev, m_sm, m_wp);
  /* Ensure the kernelId matches exactly */
  if (kernel_get_id (kernel) != expected.kernelId ())
    return false;
  /* Ensure the gridId matches exactly */
  if (kernel_get_grid_id (kernel) != expected.gridId ())
    return false;
  /* Ensure the blockIdx matches exactly */
  auto block_idx = cuda_state::warp_get_block_idx (m_dev, m_sm, m_wp);
  if (block_idx != expected.blockIdx ())
    return false;
  /* Ensure the threadIdx matches exactly */
  auto thread_idx = cuda_state::lane_get_thread_idx (m_dev, m_sm, m_wp, m_ln);
  if (thread_idx != expected.threadIdx ())
    return false;
  /* Everything matches */
  return true;
}

cuda_coords_logical::cuda_coords_logical (uint64_t kernelId, uint64_t gridId,
                                          CuDim3 clusterIdx, CuDim3 blockIdx,
                                          CuDim3 threadIdx)
    : m_kernelId{ kernelId }, m_gridId{ gridId }, m_clusterIdx{ clusterIdx },
      m_blockIdx{ blockIdx }, m_threadIdx{ threadIdx }
{
  // Replace CUDA_CURRENT with coords
  if (cuda_current_focus::isDevice ())
    {
      const auto &focus = cuda_current_focus::get ().logical ();
      cuda_evaluate_current (m_kernelId, focus.kernelId ());
      cuda_evaluate_current (m_gridId, focus.gridId ());
      cuda_evaluate_current (m_clusterIdx, focus.clusterIdx ());
      cuda_evaluate_current (m_blockIdx, focus.blockIdx ());
      cuda_evaluate_current (m_threadIdx, focus.threadIdx ());
    }
}

kernel_t
cuda_coords_logical::kernel () const
{
  return kernels_find_kernel_by_kernel_id (m_kernelId);
}

void
cuda_coords::resetPhysical ()
{
  /* Reset valid */
  m_valid = false;

  /* Search for the matching coord given the logical coordinates.
   * Assume threads cannot migrate between devices. */
  cuda_coords filter{ m_physical.dev (),       CUDA_WILDCARD,
                      CUDA_WILDCARD,           CUDA_WILDCARD,
                      m_logical.kernelId (),   m_logical.gridId (),
                      m_logical.clusterIdx (), m_logical.blockIdx (),
                      m_logical.threadIdx () };
  cuda_coord_set<cuda_coord_set_type::threads, select_valid | select_sngl> coord{
    filter
  };

  /* Ensure we found a valid thread - valid is already false if not. */
  if (!coord.size ())
    return;

  /*
   * Reset physical and return.
   * Coord is valid at this point.
   */
  auto itr = coord.begin ();
  m_physical = itr->physical ();
  m_valid = true;
}

/* to string impl for coords */
template <typename T>
const std::string
cuda_coord_to_string (const T &coord)
{
  /* Do not expose CUDA_IGNORE - this denotes coords that should never be exposed to the user */
  gdb_assert (coord != CUDA_IGNORE);
  if (coord == CUDA_INVALID)
    return std::string{ "invalid" };
  else if (coord == CUDA_WILDCARD)
    return std::string{ "*" };
  else if (coord == CUDA_CURRENT)
    return std::string{ "current" };
  std::stringstream ss;
  ss << coord;
  return ss.str ();
}
template <>
const std::string
cuda_coord_to_string<CuDim3> (const CuDim3 &c)
{
  std::string ret{ "(" };
  ret += cuda_coord_to_string (c.x);
  ret += ",";
  ret += cuda_coord_to_string (c.y);
  ret += ",";
  ret += cuda_coord_to_string (c.z);
  ret += ")";
  return ret;
}

const std::string
cuda_coords::to_string () const
{
  std::string ret;
  bool first = true;

  auto sep = [&] () {
    if (first)
      first = false;
    else
      ret += ", ";
  };

  const auto &p = m_physical;
  const auto &l = m_logical;

  if (l.kernelId () != CUDA_INVALID)
    {
      sep ();
      ret += "kernel ";
      ret += cuda_coord_to_string (l.kernelId ());
    }
  if (l.gridId () != CUDA_INVALID)
    {
      sep ();
      ret += "grid ";
      ret += cuda_coord_to_string ((int64_t)l.gridId ());
    }
  if (!cuda_coord_is_ignored (l.clusterIdx ()))
    {
      sep ();
      ret += "cluster ";
      ret += cuda_coord_to_string (l.clusterIdx ());
    }
  if ((l.blockIdx ().x != CUDA_INVALID) || (l.blockIdx ().y != CUDA_INVALID)
      || (l.blockIdx ().z != CUDA_INVALID))
    {
      sep ();
      ret += "block ";
      ret += cuda_coord_to_string (l.blockIdx ());
    }
  if ((l.threadIdx ().x != CUDA_INVALID) || (l.threadIdx ().z != CUDA_INVALID)
      || (l.threadIdx ().z != CUDA_INVALID))
    {
      sep ();
      ret += "thread ";
      ret += cuda_coord_to_string (l.threadIdx ());
    }
  if (p.dev () != CUDA_INVALID)
    {
      sep ();
      ret += "device ";
      ret += cuda_coord_to_string (p.dev ());
    }
  if (p.sm () != CUDA_INVALID)
    {
      sep ();
      ret += "sm ";
      ret += cuda_coord_to_string (p.sm ());
    }
  if (p.wp () != CUDA_INVALID)
    {
      sep ();
      ret += "warp ";
      ret += cuda_coord_to_string (p.wp ());
    }
  if (p.ln () != CUDA_INVALID)
    {
      sep ();
      ret += "lane ";
      ret += cuda_coord_to_string (p.ln ());
    }

  /* Return the pretty string */
  return ret;
}

void
cuda_coords::emit_mi_output () const
{
  struct ui_out *uiout = current_uiout;
  const auto &p = m_physical;
  const auto &l = m_logical;

  ui_out_emit_type<ui_out_type_tuple> cuda_focus (uiout, "CudaFocus");

  uiout->field_fmt ("device", "%s", cuda_coord_to_string (p.dev ()).c_str ());
  uiout->field_fmt ("sm", "%s", cuda_coord_to_string (p.sm ()).c_str ());
  uiout->field_fmt ("warp", "%s", cuda_coord_to_string (p.wp ()).c_str ());
  uiout->field_fmt ("lane", "%s", cuda_coord_to_string (p.ln ()).c_str ());
  uiout->field_fmt ("kernel", "%s", cuda_coord_to_string (l.kernelId ()).c_str ());
  uiout->field_fmt ("grid", "%s", cuda_coord_to_string (l.gridId ()).c_str ());

  if (!cuda_coord_is_ignored (l.clusterIdx ()))
    uiout->field_fmt ("clusterIdx", "%s",
		      cuda_coord_to_string (l.clusterIdx ()).c_str ());

  uiout->field_fmt ("blockIdx", "%s", cuda_coord_to_string (l.blockIdx ()).c_str ());
  uiout->field_fmt ("threadIdx", "%s", cuda_coord_to_string (l.threadIdx ()).c_str ());
}

void
cuda_current_focus::set (cuda_coords &coords)
{
  const auto &focus = cuda_current_focus::get ();

  if (focus.valid ())
    cuda_trace ("focus changed from %u/%u/%u/%u to %u/%u/%u/%u",
                focus.physical ().dev (), focus.physical ().sm (),
                focus.physical ().wp (), focus.physical ().ln (),
                coords.physical ().dev (), coords.physical ().sm (),
                coords.physical ().wp (), coords.physical ().ln ());
  else
    cuda_trace ("focus changed from invalid to %u/%u/%u/%u",
                coords.physical ().dev (), coords.physical ().sm (),
                coords.physical ().wp (), coords.physical ().ln ());

  /* Assign the new coords
   * FIXME: Check to ensure coords are valid on device. The following check
   * breaks remote debugging. */
  // gdb_assert (coords.isValidOnDevice ());

  m_current_focus.m_coords = coords;

  cuda_trace (
      "focus set to dev %u sm %u wp %u ln %u "
      "kernel %llu grid %lld, block (%u,%u,%u), thread (%u,%u,%u)",
      coords.physical ().dev (), coords.physical ().sm (),
      coords.physical ().wp (), coords.physical ().ln (),
      (unsigned long long)coords.logical ().kernelId (),
      (long long)coords.logical ().gridId (), coords.logical ().blockIdx ().x,
      coords.logical ().blockIdx ().y, coords.logical ().blockIdx ().z,
      coords.logical ().threadIdx ().x, coords.logical ().threadIdx ().y,
      coords.logical ().threadIdx ().z);
}

/* indicate the current cuda focus is _not_ valid */
void
cuda_current_focus::invalidate ()
{
  const auto &focus = cuda_current_focus::get ();
  if (focus.valid ())
    cuda_trace ("focus changed from %u/%u/%u/%u to invalid",
                focus.physical ().dev (), focus.physical ().sm (),
                focus.physical ().wp (), focus.physical ().ln ());
  // Invalidate directly
  m_current_focus.m_coords.invalidate ();
}

/*Update the current coordinates.
 *
 * Thread Selection Policy:
 *  (1) Choose the thread that was previously current if it is still
 *      valid and active. (step_divergent_lanes disabled for active)
 *  (2) If the previously current thread is no longer valid/active,
 *      select the nearest neighbor thread. (step_divergent_lanes disabled for active)
 *  (3) If there was no thread, choose either the thread with lowest logical
 *      coordinates (blockIdx/threadIdx) or the thread with the lowest physical
 *      coordinates (dev/sm/wp/ln). The choice is left to the user
 *      a CUDA option. */
void
cuda_current_focus::update ()
{
  /* First try the previous set of current coordinates (fast). */
  if (m_current_focus.m_coords.valid ())
    {
      cuda_trace ("Previous focus is still cached valid.");
      return;
    }

  /* Try the previous set of current coordinates if they are still valid and
   * active. */
  if (m_current_focus.m_coords.isValidOnDevice ())
    {
      const auto &p = m_current_focus.m_coords.physical ();
      if (cuda_options_step_divergent_lanes_enabled ()
	  || cuda_state::lane_active (p.dev (), p.sm (), p.wp (), p.ln ()))
	{
	  cuda_trace (
	      "Previous focus is still valid and active on the device.");
	  return;
	}
    }

  /* We changed away from current, find a new coordinate to focus on. */
  cuda_trace ("Could not find exact valid coordinates. Searching for nearest "
	      "valid and active neighbor.");

  cuda_coords res;

  /* With software preepmtion the physical coords might have changed. */
  if (cuda_options_software_preemption ())
    {
      const auto &l = m_current_focus.m_coords.logical ();
      cuda_coords filter{ CUDA_WILDCARD,   CUDA_WILDCARD, CUDA_WILDCARD,
			  CUDA_WILDCARD,   l.kernelId (), l.gridId (),
			  l.clusterIdx (), l.blockIdx (), l.threadIdx () };
      cuda_coord_set<cuda_coord_set_type::threads, select_valid | select_sngl>
	  coord{ filter };
      if (coord.size ())
	{
	  res = *coord.begin ();
	  const auto &p = res.physical ();
	  /* If step_divergent_threads is on, we want to lock the focus to the
	   * current CUDA thread even if it is no longer active. Otherwise
	   * ensure that the thread is still active. */
	  if (cuda_options_step_divergent_lanes_enabled ()
	      || cuda_state::lane_active (p.dev (), p.sm (), p.wp (), p.ln ()))
	    {
	      cuda_current_focus::set (res);
	      cuda_trace ("Software preemption enabled. Found new physical "
			  "coordinates (%u, %u, %u, %u)",
			  p.dev (), p.sm (), p.wp (), p.ln ());
	      return;
	    }
	}
    }

  /* Try to sort the coords based on the nearest neighbor if the previous
   * coords are valid. We need to grab a copy before calling isValidOnDevice
   * as that will reset valid. */
  gdb::optional<cuda_coords> origin;
  if (m_current_focus.m_coords.valid ())
    origin = m_current_focus.m_coords;

  if (cuda_options_thread_selection_logical ())
    {
      /* Logical selection needs to iterate over the entire physical device,
       * and sort the coordinates logically. We cannot rely on select_sngl here
       * as it will iterate over the device in sequential physical coords.*/

      /* Also check if step_divergent_threads is on. If it is, we want to lock
       * the focus to the current CUDA thread even if it is no longer active.
       */
      if (cuda_options_step_divergent_lanes_enabled ())
	{
	  cuda_coord_set<cuda_coord_set_type::threads, select_valid,
			 cuda_coord_compare_type::logical>
	      coord{ cuda_coords::wild (), origin };
	  if (coord.size ())
	    res = *coord.begin ();
	}
      else
	{
	  cuda_coord_set<cuda_coord_set_type::threads,
			 select_valid | select_active,
			 cuda_coord_compare_type::logical>
	      coord{ cuda_coords::wild (), origin };
	  if (coord.size ())
	    res = *coord.begin ();
	}
    }
  else
    {
      /* Physical can use select_sngl as we will always encounter the lowest
       * active physical coordinate first. */

      /* Also check if step_divergent_threads is on. If it is, we want to lock
       * the focus to the current CUDA thread even if it is no longer active.
       */
      if (cuda_options_step_divergent_lanes_enabled ())
	{
	  cuda_coord_set<cuda_coord_set_type::threads,
			 select_valid | select_sngl,
			 cuda_coord_compare_type::physical>
	      coord{ cuda_coords::wild (), origin };
	  if (coord.size ())
	    res = *coord.begin ();
	}
      else
	{
	  cuda_coord_set<cuda_coord_set_type::threads,
			 select_valid | select_active | select_sngl,
			 cuda_coord_compare_type::physical>
	      coord{ cuda_coords::wild (), origin };
	  if (coord.size ())
	    res = *coord.begin ();
	}
    }

  // If we found valid coords, switch to them.
  if (res.valid ())
    {
      cuda_trace ("Found new valid and active coordinates.");
      cuda_current_focus::set (res);
      return;
    }

  // No coords to switch to!
  cuda_trace ("Failed to find valid and active coordinates. Switching back to "
	      "host focus.");
  cuda_current_focus::invalidate ();
}

void
cuda_current_focus::printFocus (bool switching)
{
  struct ui_out *uiout = current_uiout;

  gdb_assert (cuda_current_focus::isDevice ());

  if (uiout->is_mi_like_p ())
    {
      m_current_focus.m_coords.emit_mi_output ();
    }
  else
    {
      std::string str = m_current_focus.m_coords.to_string ();
      if (switching)
        gdb_printf (_ ("[Switching focus to CUDA %s]\n"), str.c_str ());
      else
        gdb_printf (_ ("[Current focus set to CUDA %s]\n"), str.c_str ());
    }

  gdb_flush (gdb_stdout);
}

cuda_focus_restore::cuda_focus_restore ()
    : m_restored{ false }, m_ptid{ inferior_ptid },
      m_coords{ cuda_current_focus::get () }
{
}

void
cuda_focus_restore::do_restore ()
{
  if (!m_restored)
    {
      m_restored = true;
      if (m_coords.valid ())
        switch_to_cuda_thread (m_coords);
      else
        {
          if (m_ptid == minus_one_ptid || m_ptid == null_ptid)
            return;
          if (find_thread_ptid (current_inferior ()->process_target (), m_ptid)
              == nullptr)
            return;
          switch_to_thread (current_inferior ()->process_target (), m_ptid);
        }
    }
}
