/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2023 NVIDIA Corporation
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

#include "cuda-options.h"
#include "cuda-coords.h"
#include "cuda-iterator.h"
#include "cuda-state.h"

static uint64_t cuda_coords_distance_logical (cuda_coords_t *c1, cuda_coords_t *c2, CuDim3 gridDim, CuDim3 blockDim);
static uint64_t cuda_coords_distance_physical (cuda_coords_t *c1, cuda_coords_t *c2);

// set of current coordinates to which to apply the debugger api commands
static cuda_coords_t current_coords = CUDA_INVALID_COORDS;

/* indicate the current coordinates are _not_ valid */
void
cuda_coords_invalidate_current (void)
{
  if (current_coords.valid)
    cuda_trace ("focus changed from %u/%u/%u/%u to invalid",
		current_coords.dev, current_coords.sm, current_coords.wp, current_coords.ln);
  else
    cuda_trace ("focus invalidated");
  current_coords.valid = false;
  current_coords.clusterIdx_p = false;
  current_coords.clusterIdx_valid = false;
}

/* Scrub the current coordinates */
void
cuda_coords_reset_current (void)
{
  current_coords = CUDA_INVALID_COORDS;
  cuda_trace ("focus reset");
}

static void
cuda_read_cluster_idx (cuda_coords_t *c)
{
  gdb_assert (c != nullptr);

  if (c->valid && !c->clusterIdx_p)
    {
      c->clusterIdx_valid = false;
      c->clusterIdx = { CUDA_INVALID, CUDA_INVALID, CUDA_INVALID };
      kernel_t kernel = warp_get_kernel (c->dev, c->sm, c->wp);
      if (!kernel)
	return;

      CuDim3 clusterDim = kernel_get_cluster_dim (kernel);
      if (clusterDim.x != 0 && clusterDim.y != 0 && clusterDim.z != 0)
	{
	  c->clusterIdx = warp_get_cluster_idx (c->dev, c->sm, c->wp); 
	  c->clusterIdx_valid = true;
	}
      c->clusterIdx_p = true;
    }
}

bool
cuda_focus_is_device (void)
{
  return current_coords.valid;
}

/*returns 1 if set current failed (because the coordinates are not
   valid), and 0 otherwise. */
int
cuda_coords_set_current (cuda_coords_t *c)
{
  kernel_t kernel;
  uint64_t kernelId;
  uint64_t gridId;
  CuDim3 blockIdx;
  CuDim3 threadIdx;

  gdb_assert (c != nullptr);
  
  if (!c->valid ||
      !device_is_any_context_present (c->dev) ||
      c->sm >= device_get_num_sms (c->dev) ||
      c->wp >= device_get_num_warps (c->dev) ||
      c->ln >= device_get_num_lanes (c->dev) ||
      !warp_is_valid (c->dev, c->sm, c->wp) ||
      !lane_is_valid (c->dev, c->sm, c->wp, c->ln))
    return 1;

  kernel = warp_get_kernel (c->dev, c->sm, c->wp);
  kernelId = kernel_get_id (kernel);
  gridId = kernel_get_grid_id (kernel);
  blockIdx = warp_get_block_idx (c->dev, c->sm, c->wp);
  threadIdx = lane_get_thread_idx (c->dev, c->sm, c->wp, c->ln);

  if (c->kernelId != kernelId ||
      c->gridId != gridId ||
      c->blockIdx.x != blockIdx.x ||
      c->blockIdx.y != blockIdx.y ||
      c->blockIdx.z != blockIdx.z ||
      c->threadIdx.x != threadIdx.x ||
      c->threadIdx.y != threadIdx.y ||
      c->threadIdx.z != threadIdx.z)
    return 1;

  if (current_coords.valid)
    cuda_trace ("focus changed from %u/%u/%u/%u to %u/%u/%u/%u",
		current_coords.dev, current_coords.sm, current_coords.wp, current_coords.ln,
		c->dev, c->sm, c->wp, c->ln);
  else
    cuda_trace ("focus changed from invalid to %u/%u/%u/%u", c->dev, c->sm, c->wp, c->ln);

  current_coords = *c;

  /* Do not re-read the cluster index on focus change. Focus changes
     happen a lot behind the scenes, and often in a context where
     we're not in a position to make a debugAPI call. So, just
     invalidate the fields, and we'll read them in later if/when
     needed. */
  current_coords.clusterIdx_p = false;
  current_coords.clusterIdx_valid = false;
  current_coords.clusterIdx = { CUDA_INVALID, CUDA_INVALID, CUDA_INVALID };

  cuda_trace ("focus set to dev %u sm %u wp %u ln %u "
              "kernel %llu grid %lld, block (%u,%u,%u), thread (%u,%u,%u)",
              c->dev, c->sm, c->wp, c->ln,
              (unsigned long long)c->kernelId, (long long)c->gridId,
              c->blockIdx.x, c->blockIdx.y, c->blockIdx.z,
              c->threadIdx.x, c->threadIdx.y, c->threadIdx.z);

  return 0;
}

/* Force the current coords to be set without verification. */
void
cuda_coords_set_current_no_verify (cuda_coords_t *c)
{
  current_coords = *c;
}

int
cuda_coords_set_current_logical (uint64_t kernelId, uint64_t gridId, CuDim3 blockIdx, CuDim3 threadIdx)
{
  cuda_coords_t c = CUDA_INVALID_COORDS;

  c.kernelId = kernelId;
  c.gridId = gridId;
  c.blockIdx = blockIdx;
  c.threadIdx = threadIdx;

  if (cuda_coords_complete_physical (&c))
    {
      c.valid = false;
      return 1;
    }

  c.valid = true;

  return cuda_coords_set_current (&c);
}

int
cuda_coords_set_current_physical (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln)
{
  cuda_coords_t c = CUDA_INVALID_COORDS;

  c.dev = dev;
  c.sm = sm;
  c.wp = wp;
  c.ln = ln;

  if (!device_is_any_context_present (c.dev) ||
      c.sm >= device_get_num_sms (c.dev) ||
      c.wp >= device_get_num_warps (c.dev) ||
      c.ln >= device_get_num_lanes (c.dev) ||
      !warp_is_valid (c.dev, c.sm, c.wp) ||
      !lane_is_valid (c.dev, c.sm, c.wp, c.ln))
    {
      c.valid = false;
      return 1;
    }

  cuda_coords_complete_logical (&c);
  c.valid = true;

  return cuda_coords_set_current (&c);
}

int
cuda_coords_get_current (cuda_coords_t *coords)
{
  *coords = current_coords;
  return !current_coords.valid;
}

int
cuda_coords_get_current_logical (uint64_t *kernelId, uint64_t *gridId, CuDim3 *blockIdx, CuDim3 *threadIdx)
{
  if (!current_coords.valid)
    return 1;

  if (kernelId)
    *kernelId = current_coords.kernelId;
  if (gridId)
    *gridId = current_coords.gridId;
  if (blockIdx)
    *blockIdx = current_coords.blockIdx;
  if (threadIdx)
    *threadIdx = current_coords.threadIdx;
  return 0;
}

int
cuda_coords_get_current_physical (uint32_t *dev, uint32_t *sm, uint32_t *wp, uint32_t *ln)
{
  if (!current_coords.valid)
    return 1;

  if (dev)
    *dev = current_coords.dev;
  if (sm)
    *sm = current_coords.sm;
  if (wp)
    *wp = current_coords.wp;
  if (ln)
    *ln = current_coords.ln;
  return 0;
}

bool
cuda_coords_is_current_logical (cuda_coords_t *c)
{
  return (c->valid &&
          current_coords.valid &&
          !cuda_coords_compare_logical (c, &current_coords));
}

bool
cuda_coords_is_current (cuda_coords_t *c)
{
  return (c->valid &&
          current_coords.valid &&
          cuda_coords_equal (c, &current_coords));
}

bool
cuda_coords_equal (cuda_coords_t *c1, cuda_coords_t *c2)
{
  return !cuda_coords_compare_logical (c1, c2) && !cuda_coords_compare_physical (c1, c2);
}

#define CUDA_COORD_COMPARE(x,y)                                         \
  if (!CUDA_COORD_IS_SPECIAL (x) && !CUDA_COORD_IS_SPECIAL (y))         \
    {                                                                   \
      if (x < y)                                                        \
        return -1;                                                      \
      if (x > y)                                                        \
        return 1;                                                       \
    }

int
cuda_coords_compare_logical (cuda_coords_t *c1, cuda_coords_t *c2)
{
  CUDA_COORD_COMPARE (c1->kernelId, c2->kernelId);
  CUDA_COORD_COMPARE (c1->gridId, c2->gridId);
  CUDA_COORD_COMPARE (c1->blockIdx.x, c2->blockIdx.x);
  CUDA_COORD_COMPARE (c1->blockIdx.y, c2->blockIdx.y);
  CUDA_COORD_COMPARE (c1->blockIdx.z, c2->blockIdx.z);
  CUDA_COORD_COMPARE (c1->threadIdx.x, c2->threadIdx.x);
  CUDA_COORD_COMPARE (c1->threadIdx.y, c2->threadIdx.y);
  CUDA_COORD_COMPARE (c1->threadIdx.z, c2->threadIdx.z);
  return 0;
}

int
cuda_coords_compare_physical (cuda_coords_t *c1, cuda_coords_t *c2)
{
  CUDA_COORD_COMPARE (c1->dev, c2->dev);
  CUDA_COORD_COMPARE (c1->sm, c2->sm);
  CUDA_COORD_COMPARE (c1->wp, c2->wp);
  CUDA_COORD_COMPARE (c1->ln, c2->ln);
  return 0;
}

#undef CUDA_COORD_COMPARE

void
cuda_coords_increment_block (cuda_coords_t *c, CuDim3 grid_dim)
{
  if (c->blockIdx.z < grid_dim.z - 1)
    {
      ++c->blockIdx.z;
    }
  else if (c->blockIdx.y < grid_dim.y - 1)
    {
      ++c->blockIdx.y;
      c->blockIdx.z = 0;
    }
  else if (c->blockIdx.x < grid_dim.x - 1)
    {
      ++c->blockIdx.x;
      c->blockIdx.y = 0;
      c->blockIdx.z = 0;
    }
  else
    {
      c->blockIdx.x = CUDA_INVALID;
      c->blockIdx.y = CUDA_INVALID;
      c->blockIdx.z = CUDA_INVALID;
    }
}

void
cuda_coords_increment_thread (cuda_coords_t *c, CuDim3 grid_dim, CuDim3 block_dim)
{
  if (c->threadIdx.z < block_dim.z - 1)
    {
      ++c->threadIdx.z;
    }
  else if (c->threadIdx.y < block_dim.y - 1)
    {
      ++c->threadIdx.y;
      c->threadIdx.z = 0;
    }
  else if (c->threadIdx.x < block_dim.x - 1)
    {
      ++c->threadIdx.x;
      c->threadIdx.y = 0;
      c->threadIdx.z = 0;
    }
  else if (c->blockIdx.z < grid_dim.z - 1)
    {
      ++c->blockIdx.z;
      c->threadIdx.x = 0;
      c->threadIdx.y = 0;
      c->threadIdx.z = 0;
    }
  else if (c->blockIdx.y < grid_dim.y - 1)
    {
      ++c->blockIdx.y;
      c->blockIdx.z  = 0;
      c->threadIdx.x = 0;
      c->threadIdx.y = 0;
      c->threadIdx.z = 0;
    }
  else if (c->blockIdx.x < grid_dim.x - 1)
    {
      ++c->blockIdx.x;
      c->blockIdx.y  = 0;
      c->blockIdx.z  = 0;
      c->threadIdx.x = 0;
      c->threadIdx.y = 0;
      c->threadIdx.z = 0;
    }
  else
    {
      c->blockIdx.x  = CUDA_INVALID;
      c->blockIdx.y  = CUDA_INVALID;
      c->blockIdx.z  = CUDA_INVALID;
      c->threadIdx.x = CUDA_INVALID;
      c->threadIdx.y = CUDA_INVALID;
      c->threadIdx.z = CUDA_INVALID;
    }
}

void
cuda_coords_to_fancy_string (cuda_coords_t *c, char *string, uint32_t size)
{
  bool first = true;
  char buffer[1000];
  char *s = buffer;

#define SPRINTF_COORD(s, c, x)                                          \
  switch (c->x) {                                                       \
  case CUDA_INVALID:  s += sprintf (s, "invalid"); break;               \
  case CUDA_WILDCARD: s += sprintf (s, "*");       break;               \
  case CUDA_CURRENT:  s += sprintf (s, "current"); break;               \
  default:            s += sprintf (s, "%lld", (long long) c->x);       \
  }

#define SPRINTF_SEPARATOR(s, first)                 \
  if (!first)                                       \
    s += sprintf (s, ", ");                         \
  else                                              \
    first = false;                                  \

  *s = 0;

  if (c->kernelId != CUDA_INVALID)
    {
      SPRINTF_SEPARATOR (s, first);
      s += sprintf (s, "kernel ");
      SPRINTF_COORD (s, c, kernelId);
    }
  if (c->gridId != CUDA_INVALID)
    {
      SPRINTF_SEPARATOR (s, first);
      s += sprintf (s, "grid ");
      SPRINTF_COORD (s, c, gridId);
    }
  if (c->clusterIdx_p && c->clusterIdx_valid)
    {
      SPRINTF_SEPARATOR (s, first);
      s += sprintf (s, "cluster (");
      SPRINTF_COORD (s, c, clusterIdx.x);
      s += sprintf (s, ",");
      SPRINTF_COORD (s, c, clusterIdx.y);
      s += sprintf (s, ",");
      SPRINTF_COORD (s, c, clusterIdx.z);
      s += sprintf (s, ")");
    }
  if (c->blockIdx.x != CUDA_INVALID || c->blockIdx.y != CUDA_INVALID || c->blockIdx.z != CUDA_INVALID)
    {
      SPRINTF_SEPARATOR (s, first);
      s += sprintf (s, "block (");
      SPRINTF_COORD (s, c, blockIdx.x);
      s += sprintf (s, ",");
      SPRINTF_COORD (s, c, blockIdx.y);
      s += sprintf (s, ",");
      SPRINTF_COORD (s, c, blockIdx.z);
      s += sprintf (s, ")");
    }
  if (c->threadIdx.x != CUDA_INVALID || c->threadIdx.z != CUDA_INVALID || c->threadIdx.z != CUDA_INVALID)
    {
      SPRINTF_SEPARATOR (s, first);
      s += sprintf (s, "thread (");
      SPRINTF_COORD (s, c, threadIdx.x);
      s += sprintf (s, ",");
      SPRINTF_COORD (s, c, threadIdx.y);
      s += sprintf (s, ",");
      SPRINTF_COORD (s, c, threadIdx.z);
      s += sprintf (s, ")");
    }
  if (c->dev != CUDA_INVALID)
    {
      SPRINTF_SEPARATOR (s, first);
      s += sprintf (s, "device ");
      SPRINTF_COORD (s, c, dev);
    }
  if (c->sm != CUDA_INVALID)
    {
      SPRINTF_SEPARATOR (s, first);
      s += sprintf (s, "sm ");
      SPRINTF_COORD (s, c, sm);
    }
  if (c->wp != CUDA_INVALID)
    {
      SPRINTF_SEPARATOR (s, first);
      s += sprintf (s, "warp ");
      SPRINTF_COORD (s, c, wp);
    }
  if (c->ln != CUDA_INVALID)
    {
      SPRINTF_SEPARATOR (s, first);
      s += sprintf (s, "lane ");
      SPRINTF_COORD (s, c, ln);
    }

  if (strlen (buffer) + 1 > size)
    error (_("Coordinates too long to print"));

  strcpy (string, buffer);

#undef SPRINTF_COORD
#undef SPRINTF_SEPARATOR
}

/*Given logical coordinates (valid or wild), fill up the physical
   coordinates that can be derived from the logical coordinates. Use
   the wildcard if not possible. */
int
cuda_coords_complete_physical (cuda_coords_t *c)
{
  cuda_iterator iter;
  cuda_coords_t current = CUDA_INVALID_COORDS;
  bool found = false;

  gdb_assert (c);

  c->dev = CUDA_WILDCARD;
  c->sm  = CUDA_WILDCARD;
  c->wp  = CUDA_WILDCARD;
  c->ln  = CUDA_WILDCARD;

  iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_THREADS, c, 
		  (cuda_select_t)(CUDA_SELECT_VALID | CUDA_SELECT_SNGL));

  cuda_iterator_start (iter);
  if (!cuda_iterator_end (iter))
    {
      current = cuda_iterator_get_current (iter);
      found = true;
    }

  cuda_iterator_destroy (iter);

  if (!found)
    return 1;

  if (c->kernelId != CUDA_WILDCARD)
    {
      c->dev = current.dev;
      if (c->blockIdx.x != CUDA_WILDCARD &&
          c->blockIdx.y != CUDA_WILDCARD &&
          c->blockIdx.z != CUDA_WILDCARD &&
          c->threadIdx.x != CUDA_WILDCARD &&
          c->threadIdx.y != CUDA_WILDCARD &&
          c->threadIdx.z != CUDA_WILDCARD)
        {
          c->sm  = current.sm;
          c->wp  = current.wp;
          c->ln  = current.ln;
        }
    }

  return 0;
}

/*Given physical coordinates (valid or wild), fill up the logical
   coordinates that can be derived from the physical coordinates. Use
   the wildcard if not possible. */
int
cuda_coords_complete_logical (cuda_coords_t *c)
{
  kernel_t kernel;

  gdb_assert (c);
  gdb_assert (c->dev == CUDA_WILDCARD || device_is_valid (c->dev));
  gdb_assert (c->sm  == CUDA_WILDCARD || sm_is_valid (c->dev, c->sm));
  gdb_assert (c->wp  == CUDA_WILDCARD || warp_is_valid (c->dev, c->sm, c->wp));
  gdb_assert (c->ln  == CUDA_WILDCARD || lane_is_valid (c->dev, c->sm, c->wp, c->ln));

  c->kernelId  = CUDA_WILDCARD;
  c->gridId    = CUDA_WILDCARD;
  c->blockIdx  = (CuDim3) { CUDA_WILDCARD, CUDA_WILDCARD, CUDA_WILDCARD };
  c->threadIdx = (CuDim3) { CUDA_WILDCARD, CUDA_WILDCARD, CUDA_WILDCARD };

  if (c->dev != CUDA_WILDCARD &&
      c->sm  != CUDA_WILDCARD &&
      c->wp  != CUDA_WILDCARD)
    {
      kernel = warp_get_kernel (c->dev, c->sm, c->wp);
      c->kernelId = kernel_get_id (kernel);
      c->gridId   = kernel_get_grid_id (kernel);
      c->blockIdx = warp_get_block_idx (c->dev, c->sm, c->wp);
      if (c->ln != CUDA_WILDCARD)
        c->threadIdx = lane_get_thread_idx (c->dev, c->sm, c->wp, c->ln);
    }

  return 0;
}

/*Take the coords object and replace any CUDA_CURRENT value with the
   corresponding value in the current focus.  If the current focus is not set to
   a CUDA device, replace CUDA_CURRENT with CUDA_WILDCARDS if use_wildcards is
   set.  Otherwise, return an error. */
void
cuda_coords_evaluate_current (cuda_coords_t *coords, bool use_wildcards)
{
  cuda_coords_t current = CUDA_INVALID_COORDS;

  if (cuda_focus_is_device ())
    cuda_coords_get_current (&current);

#define EVALUATE_CURRENT(x)                                             \
  if (coords->x == CUDA_CURRENT) {                                      \
    if (cuda_focus_is_device ())                                        \
      coords->x = current.x;                                            \
    else if (use_wildcards)                                             \
      coords->x = CUDA_WILDCARD;                                        \
    else                                                                \
      error (_("Focus not set on any active CUDA kernel."));            \
  }

  EVALUATE_CURRENT (dev);
  EVALUATE_CURRENT (sm);
  EVALUATE_CURRENT (wp);
  EVALUATE_CURRENT (ln);
  EVALUATE_CURRENT (kernelId);
  EVALUATE_CURRENT (gridId);
  EVALUATE_CURRENT (blockIdx.x);
  EVALUATE_CURRENT (blockIdx.y);
  EVALUATE_CURRENT (blockIdx.z);
  EVALUATE_CURRENT (threadIdx.x);
  EVALUATE_CURRENT (threadIdx.y);
  EVALUATE_CURRENT (threadIdx.z);

#undef EVALUATE_CURRENT
}

/*Assert if any of the coordinates in coords have special values (CUDA_INVALID,
   CUDA_CURRENT, CUDA_WILDCARD) not allowed by the 3 booleans. */
void
cuda_coords_check_fully_defined (cuda_coords_t *coords, bool accept_invalid,
                                 bool accept_current, bool accept_wildcards)
{
#define CHECK_DEFINED(x)                                         \
  gdb_assert (accept_invalid   || coords->x != CUDA_INVALID);    \
  gdb_assert (accept_current   || coords->x != CUDA_CURRENT);    \
  gdb_assert (accept_wildcards || coords->x != CUDA_WILDCARD);

  CHECK_DEFINED (dev);
  CHECK_DEFINED (sm);
  CHECK_DEFINED (wp);
  CHECK_DEFINED (ln);
  CHECK_DEFINED (kernelId);
  CHECK_DEFINED (gridId);
  CHECK_DEFINED (blockIdx.x);
  CHECK_DEFINED (blockIdx.y);
  CHECK_DEFINED (blockIdx.z);
  CHECK_DEFINED (threadIdx.x);
  CHECK_DEFINED (threadIdx.y);
  CHECK_DEFINED (threadIdx.z);

#undef CHECK_DEFINED
}

static void
cuda_coords_initialized_wished_coords (cuda_coords_t *wished)
{
  uint32_t  dev_id;
  kernel_t  kernel;

  /* (gridId, blockIdx, threadIdx) is not enough to identify a single thread.
     The complete representation is (kernelId, blockIdx, threadIdx). But there
     is a unique mapping between (dev, gridId) and (kernelId). Therefore:
       - use kernelId and not gridId in distance computation
       - if gridId is set in wished, but kernelId is not, derive kernelId from
         the current device. If no current device, use 0.
 */
  if (wished->kernelId == CUDA_WILDCARD && wished->gridId != CUDA_WILDCARD)
    {
      if (cuda_focus_is_device ())
        {
          dev_id = cuda_current_device ();
          kernel  = kernels_find_kernel_by_grid_id (dev_id, wished->gridId);
          wished->kernelId = kernel_get_id (kernel);
        }
      else
        wished->kernelId = 0;
    }
}

static void
cuda_coords_find_valid_exact (cuda_coords_t wished, cuda_coords_t *found, bool breakpoint_hit)
{
  uint32_t dev = wished.dev;
  uint32_t sm = wished.sm;
  uint32_t wp = wished.wp;
  uint32_t ln = wished.ln;
  uint64_t kernelId;
  uint64_t gridId;
  CuDim3 blockIdx;
  CuDim3 threadIdx;
  kernel_t kernel;

  gdb_assert (found);
  *found = CUDA_INVALID_COORDS;

  if (dev == CUDA_INVALID || dev == CUDA_CURRENT || dev == CUDA_WILDCARD ||
      sm == CUDA_INVALID  || sm == CUDA_CURRENT  || sm == CUDA_WILDCARD  ||
      wp == CUDA_INVALID  || wp == CUDA_CURRENT  || wp == CUDA_WILDCARD  ||
      ln == CUDA_INVALID  || ln == CUDA_CURRENT  || ln == CUDA_WILDCARD)
    return;

  if (!warp_is_valid (dev, sm, wp) ||
      !lane_is_valid (dev, sm, wp, ln))
    return;

  if (breakpoint_hit &&
      /* CUDA - no address space management */
      !breakpoint_here_p (NULL, lane_get_virtual_pc (dev, sm, wp, ln)))
    return;

  kernel    = warp_get_kernel (dev, sm, wp);
  kernelId  = kernel_get_id (kernel);
  gridId    = kernel_get_grid_id (kernel);
  blockIdx  = warp_get_block_idx (dev, sm, wp);
  threadIdx = lane_get_thread_idx (dev, sm, wp, ln);

  if (kernelId != wished.kernelId ||
      gridId != wished.gridId ||
      blockIdx.x != wished.blockIdx.x ||
      blockIdx.y != wished.blockIdx.y ||
      blockIdx.z != wished.blockIdx.z ||
      threadIdx.x != wished.threadIdx.x ||
      threadIdx.y != wished.threadIdx.y ||
      threadIdx.z != wished.threadIdx.z)
    return;

  *found = (cuda_coords_t) { true, false, false, kernelId, gridId, {CUDA_INVALID, CUDA_INVALID, CUDA_INVALID}, blockIdx, threadIdx, dev, sm, wp, ln };
}

void
cuda_coords_find_valid (cuda_coords_t wished, cuda_coords_t found[CK_MAX], cuda_select_t select_mask /*= CUDA_SELECT_VALID*/)
{
  int kind;
  cuda_coords_t origin = { true, false, false, 0, 0, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, 0, 0, 0, 0 };
  uint64_t relative_distance_logical, relative_distance_physical;
  uint64_t absolute_distance_logical, absolute_distance_physical;
  uint64_t best_relative_distance_logical = 0ULL, best_relative_distance_physical = 0ULL;
  uint64_t best_absolute_distance_logical = 0ULL, best_absolute_distance_physical = 0ULL;
  uint64_t best_next_distance_logical = 0ULL, best_next_distance_physical = 0ULL;
  CuDim3 grid_dim;
  CuDim3 block_dim;
  cuda_coords_t temp, filter = CUDA_WILDCARD_COORDS;
  cuda_iterator iter;
  kernel_t kernel;

  gdb_assert (found);

  for (kind = 0; kind < CK_MAX; ++kind)
    found[kind] = CUDA_INVALID_COORDS;

  cuda_coords_initialized_wished_coords (&wished);
  iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_THREADS, &filter,
			       select_mask);

  for (cuda_iterator_start (iter);
       !cuda_iterator_end (iter);
       cuda_iterator_next (iter))
    {
      temp = cuda_iterator_get_current (iter);
      kernel = warp_get_kernel (temp.dev, temp.sm, temp.wp);
      grid_dim = kernel_get_grid_dim (kernel);
      block_dim = kernel_get_block_dim (kernel);

      /* compute distances */
      absolute_distance_logical  = cuda_coords_distance_logical (&origin, &temp, grid_dim, block_dim);
      relative_distance_logical  = cuda_coords_distance_logical (&wished, &temp, grid_dim, block_dim);
      absolute_distance_physical = cuda_coords_distance_physical (&origin, &temp);
      relative_distance_physical = cuda_coords_distance_physical (&wished, &temp);

      /* update set of solutions */
      if (relative_distance_logical == 0)
        found[CK_EXACT_LOGICAL] = temp;

      if (relative_distance_physical == 0)
        found[CK_EXACT_PHYSICAL] = temp;

      if (!found[CK_CLOSEST_LOGICAL].valid || relative_distance_logical < best_relative_distance_logical)
        {
          found[CK_CLOSEST_LOGICAL] = temp;
          best_relative_distance_logical = relative_distance_logical;
        }

      if (!found[CK_CLOSEST_PHYSICAL].valid || relative_distance_physical < best_relative_distance_physical)
        {
          found[CK_CLOSEST_PHYSICAL] = temp;
          best_relative_distance_physical = relative_distance_physical;
        }

      if (!found[CK_LOWEST_LOGICAL].valid || absolute_distance_logical < best_absolute_distance_logical)
        {
          found[CK_LOWEST_LOGICAL] = temp;
          best_absolute_distance_logical = absolute_distance_logical;
        }

      if (!found[CK_LOWEST_PHYSICAL].valid || absolute_distance_physical < best_absolute_distance_physical)
        {
          found[CK_LOWEST_PHYSICAL] = temp;
          best_absolute_distance_physical = absolute_distance_physical;
        }

      if (relative_distance_logical >  0 &&
          (!found[CK_NEXT_LOGICAL].valid || relative_distance_logical < best_next_distance_logical) &&
          cuda_coords_compare_logical (&temp, &wished) > 0)
        {
          found[CK_NEXT_LOGICAL] = temp;
          best_next_distance_logical = relative_distance_logical;
        }

      if (relative_distance_physical > 0 &&
          (!found[CK_NEXT_PHYSICAL].valid || relative_distance_physical < best_next_distance_physical) &&
          cuda_coords_compare_physical (&temp, &wished) > 0)
        {
          found[CK_NEXT_PHYSICAL] = temp;
          best_next_distance_physical = relative_distance_physical;
        }

    }

  cuda_iterator_destroy (iter);
}

/*Update the current coordinates.
 *
 * Thread Selection Policy:
 *  (1) choose the thread that was previously current if it matches the
 *      selection criteria.
 *  (2) if not, choose either the thread with lowest logical coordinates
 *      (blockIdx/threadIdx) or the thread with the lowest physical
 *      coordinates (dev/sm/wp/ln). The choice is left to the user
 *      a CUDA option. */
void
cuda_coords_update_current (bool breakpoint_hit /*= false*/)
{
  int kind;
  cuda_coords_t coords[CK_MAX];
  cuda_coords_t result = CUDA_INVALID_COORDS;

  /* Initialize the coordinate array */
  for (kind = 0; kind < CK_MAX; ++kind)
    coords[kind] = CUDA_INVALID_COORDS;

  /* first try the previous set of current coordinates (fast). Note,
     current_coords may be invalid at this point. */
  cuda_coords_find_valid_exact (current_coords, &result, breakpoint_hit);

  /* if that does not work, brute-force it */
  if (!result.valid)
    {
      cuda_select_t mask = CUDA_SELECT_VALID;
      if (breakpoint_hit)
	mask = (cuda_select_t)(CUDA_SELECT_VALID | CUDA_SELECT_BKPT);
      cuda_trace ("could not find exact valid coordinates, trying brute force");
      cuda_coords_find_valid (current_coords, coords, mask);

      if (cuda_options_software_preemption () && coords[CK_EXACT_LOGICAL].valid)
        kind = CK_EXACT_LOGICAL;
      else if (cuda_options_thread_selection_logical () && coords[CK_LOWEST_LOGICAL].valid)
        kind = CK_LOWEST_LOGICAL;
      else if (cuda_options_thread_selection_physical () && coords[CK_LOWEST_PHYSICAL].valid)
        kind = CK_LOWEST_PHYSICAL;
      else
        kind = CK_MAX;

      if (kind != CK_MAX)
        result = coords[kind];
    }

  if (result.valid)
    {
      cuda_trace ("found exact valid coordinates");
      cuda_coords_set_current (&result);
    }
  else
    {
      cuda_coords_invalidate_current ();
    }
}

void
cuda_print_message_focus (bool switching)
{
  cuda_coords_t current;
  const uint32_t size = 1000;
  char *string;
  struct ui_out *uiout = current_uiout;

  gdb_assert (cuda_focus_is_device ());

  string = (char *) xmalloc (size);
  cuda_coords_get_current (&current);

  /* For valid coordinates this call will retrieve the clusterIdx info if it hasn't been
     collected already. */
  cuda_read_cluster_idx (&current);

  cuda_coords_to_fancy_string (&current, string, size);

  if (uiout->is_mi_like_p ())
    {
      ui_out_emit_type<ui_out_type_tuple> cuda_focus(uiout, "CudaFocus");

      uiout->field_signed ("device"    , current.dev);
      uiout->field_signed ("sm"        , current.sm);
      uiout->field_signed ("warp"      , current.wp);
      uiout->field_signed ("lane"      , current.ln);
      uiout->field_signed ("kernel"    , current.kernelId);
      uiout->field_signed ("grid"      , current.gridId);

      if (current.clusterIdx_p && current.clusterIdx_valid)
	uiout->field_fmt ("clusterIdx", "(%u,%u,%u)", current.clusterIdx.x, current.clusterIdx.y, current.clusterIdx.z);

      uiout->field_fmt ("blockIdx"  , "(%u,%u,%u)", current.blockIdx.x, current.blockIdx.y, current.blockIdx.z);
      uiout->field_fmt ("threadIdx" , "(%u,%u,%u)", current.threadIdx.x, current.threadIdx.y, current.threadIdx.z);
    }
  else
    {
      if (switching)
        printf_filtered (_("[Switching focus to CUDA %s]\n"), string);
      else
        printf_filtered (_("[Current focus set to CUDA %s]\n"), string);
    }

  gdb_flush (gdb_stdout);
  xfree (string);
}

#define dist(x,y) ((x) > (y) ? (x) - (y) : (y) - (x))

/* The distances are one-dimensional Euclidian distance of the kernel
   coordinates (physical or logical) projected onto a one-dimensional plan (to
   avoid having to compute the nth roots). To make sure that the distance
   between 2 points of kernel A and kernel B cannot be seen as closer as 2
   points from either kernel A or B, we introduced an artifically large
   distance between points of separate kernel (or device). */

static uint64_t
cuda_coords_flat_logical (cuda_coords_t *c, CuDim3 gridDim, CuDim3 blockDim)
{
  uint64_t d = 0;

  if (!CUDA_COORD_IS_SPECIAL (c->kernelId))
    // Force kernels to be far enough from each other
    d += c->kernelId * 0xffffffffULL;
  if (!CUDA_COORD_IS_SPECIAL (c->blockIdx.x))
    d += c->blockIdx.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
  if (!CUDA_COORD_IS_SPECIAL (c->blockIdx.y))
    d += c->blockIdx.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
  if (!CUDA_COORD_IS_SPECIAL (c->blockIdx.z))
    d += c->blockIdx.z * blockDim.x * blockDim.y * blockDim.z;
  if (!CUDA_COORD_IS_SPECIAL (c->threadIdx.x))
    d += c->threadIdx.x * blockDim.y * blockDim.z;
  if (!CUDA_COORD_IS_SPECIAL (c->threadIdx.y))
    d += c->threadIdx.y * blockDim.z;
  if (!CUDA_COORD_IS_SPECIAL (c->threadIdx.z))
    d += c->threadIdx.z;

  return d;
}

static uint64_t
cuda_coords_flat_physical (cuda_coords_t *c)
{
  uint64_t d = 0;

  if (!CUDA_COORD_IS_SPECIAL (c->dev))
    // Force devices to be far enough from each other
    d += c->dev * 0xffffffffULL;
  if (!CUDA_COORD_IS_SPECIAL (c->sm))
    d += c->sm * CUDBG_MAX_SMS * CUDBG_MAX_WARPS;
  if (!CUDA_COORD_IS_SPECIAL (c->wp))
    d += c->wp * CUDBG_MAX_WARPS;
  if (!CUDA_COORD_IS_SPECIAL (c->ln))
    d += c->ln;

  return d;
}

/* The distance is computed in a flat one-dimensional system. The
   7-dimensional coordinates array is therefore first flattened to a
   1-dimensional array. If there are any CUDA_WILDCARD coordinates in any
   point, that coordinate must be entirely ignored for the distance
   computation.  */

#define FILTER_OUT_WILDCARDED_COORDINATE(coord)                \
  if (coords1.coord == CUDA_WILDCARD || coords2.coord== CUDA_WILDCARD) \
    coords1.coord = coords2.coord = CUDA_WILDCARD;

static uint64_t
cuda_coords_distance_logical (cuda_coords_t *c1, cuda_coords_t *c2,
                              CuDim3 gridDim, CuDim3 blockDim)
{
  cuda_coords_t coords1 = *c1;
  cuda_coords_t coords2 = *c2;
  uint64_t dist1, dist2;

  FILTER_OUT_WILDCARDED_COORDINATE(kernelId);
  FILTER_OUT_WILDCARDED_COORDINATE(blockIdx.x);
  FILTER_OUT_WILDCARDED_COORDINATE(blockIdx.y);
  FILTER_OUT_WILDCARDED_COORDINATE(blockIdx.z);
  FILTER_OUT_WILDCARDED_COORDINATE(threadIdx.x);
  FILTER_OUT_WILDCARDED_COORDINATE(threadIdx.y);
  FILTER_OUT_WILDCARDED_COORDINATE(threadIdx.z);

  dist1 = cuda_coords_flat_logical (&coords1, gridDim, blockDim);
  dist2 = cuda_coords_flat_logical (&coords2, gridDim, blockDim);

  return dist (dist1, dist2);
}

static uint64_t
cuda_coords_distance_physical (cuda_coords_t *c1, cuda_coords_t *c2)
{
  cuda_coords_t coords1 = *c1;
  cuda_coords_t coords2 = *c2;
  uint64_t dist1, dist2;

  FILTER_OUT_WILDCARDED_COORDINATE(dev);
  FILTER_OUT_WILDCARDED_COORDINATE(sm);
  FILTER_OUT_WILDCARDED_COORDINATE(wp);
  FILTER_OUT_WILDCARDED_COORDINATE(ln);

  dist1 = cuda_coords_flat_physical (&coords1);
  dist2 = cuda_coords_flat_physical (&coords2);

  return dist (dist1, dist2);
}

#undef FILTER_OUT_WILDCARDED_COORDINATE


uint32_t
cuda_current_device (void)
{
  uint32_t dev;

  gdb_assert (cuda_focus_is_device ());
  cuda_coords_get_current_physical (&dev, NULL, NULL, NULL);
  return dev;
}

uint32_t
cuda_current_sm (void)
{
  uint32_t sm;

  gdb_assert (cuda_focus_is_device ());
  cuda_coords_get_current_physical (NULL, &sm, NULL, NULL);
  return sm;
}

uint32_t
cuda_current_warp (void)
{
  uint32_t wp;

  gdb_assert (cuda_focus_is_device ());
  cuda_coords_get_current_physical (NULL, NULL, &wp, NULL);
  return wp;
}

uint32_t
cuda_current_lane (void)
{
  uint32_t ln;

  gdb_assert (cuda_focus_is_device ());
  cuda_coords_get_current_physical (NULL, NULL, NULL, &ln);
  return ln;
}

kernel_t
cuda_current_kernel (void)
{
  uint32_t dev, sm, wp;
  kernel_t kernel;

  if (!cuda_focus_is_device ())
    return NULL;

  cuda_coords_get_current_physical (&dev, &sm, &wp, NULL);
  kernel = warp_get_kernel (dev, sm, wp);
  return kernel;
}
