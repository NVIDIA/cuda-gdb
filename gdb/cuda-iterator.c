/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2014 NVIDIA Corporation
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
#include "breakpoint.h"
#include "inferior.h"
#include "target.h"

#include "cuda-defs.h"
#include "cuda-iterator.h"
#include "cuda-state.h"

#define CUDA_ITERATOR_TYPE_MASK_PHYSICAL  0x0f
#define CUDA_ITERATOR_TYPE_MASK_LOGICAL   0xf0
static CuDim3 CUDA_WILDCARD_DIM = {CUDA_WILDCARD, CUDA_WILDCARD, CUDA_WILDCARD};
static CuDim3 CUDA_INVALID_DIM = {CUDA_INVALID, CUDA_INVALID, CUDA_INVALID};

struct cuda_iterator_t
{
  cuda_iterator_type type;
  cuda_coords_t filter;
  cuda_select_t mask;
  bool     completed;
  uint32_t num_elements;
  uint32_t num_unique_elements;
  uint32_t list_size;
  uint32_t index;
  cuda_coords_t current;
  cuda_coords_t *list;
};

/*
 * Returns true if 32-bit values match
 * i.e. either equals to each other or one of the arguments equal to wildcard
 */
static inline bool
cuda_val_matches (uint32_t a, uint32_t b)
{
  return a == CUDA_WILDCARD || b == CUDA_WILDCARD || a == b;
}
/*
 * Returns true if 64-bit values matches
 * i.e. either equals to each other or one of the arguments equal to wildcard
 */
static inline bool
cuda_val64_matches (uint64_t a, uint64_t b)
{
  return a == CUDA_WILDCARD || b == CUDA_WILDCARD || a == b;
}
/*
 * Returns true if individual coordinates of dim3 matches
 */
static bool
cuda_dim3_matches (CuDim3 *a, CuDim3 *b)
{
  gdb_assert (a);
  gdb_assert (b);

  return cuda_val_matches (a->x, b->x) &&
         cuda_val_matches (a->y, b->y) &&
         cuda_val_matches (a->z, b->z);
}

static inline uint64_t
warp_get_kernel_id (uint32_t dev, uint32_t sm, uint32_t wp)
{
  kernel_t kernel = warp_get_kernel (dev, sm, wp);
  return kernel ? kernel_get_id (kernel) : CUDA_INVALID;
}

/*
 * Returns fitlered cuda coords based on iterator
 */
static cuda_coords_t
cuda_iterator_filter_coords(const cuda_coords_t *in, cuda_iterator_type type)
{
  cuda_coords_t c;
  bool store_dev, store_sm, store_warp, store_lane, store_kernel, store_grid, store_block, store_thread;

  /* only store information that can be uniquely identified given an object of the iterator type */
  store_dev =
    (type & CUDA_ITERATOR_TYPE_MASK_PHYSICAL) >= CUDA_ITERATOR_TYPE_DEVICES ||
    (type & CUDA_ITERATOR_TYPE_MASK_LOGICAL) >= CUDA_ITERATOR_TYPE_KERNELS;
  store_sm =
    (type & CUDA_ITERATOR_TYPE_MASK_PHYSICAL) >= CUDA_ITERATOR_TYPE_SMS ||
    (type & CUDA_ITERATOR_TYPE_MASK_LOGICAL) >= CUDA_ITERATOR_TYPE_BLOCKS;
  store_warp =
    (type & CUDA_ITERATOR_TYPE_MASK_PHYSICAL) >= CUDA_ITERATOR_TYPE_WARPS ||
    (type & CUDA_ITERATOR_TYPE_MASK_LOGICAL) >= CUDA_ITERATOR_TYPE_THREADS;
  store_lane =
    (type & CUDA_ITERATOR_TYPE_MASK_PHYSICAL) >= CUDA_ITERATOR_TYPE_LANES ||
    (type & CUDA_ITERATOR_TYPE_MASK_LOGICAL) >= CUDA_ITERATOR_TYPE_THREADS;
  store_kernel =
    (type & CUDA_ITERATOR_TYPE_MASK_PHYSICAL) >= CUDA_ITERATOR_TYPE_SMS ||
    (type & CUDA_ITERATOR_TYPE_MASK_LOGICAL) >= CUDA_ITERATOR_TYPE_KERNELS;
  store_grid =
    (type & CUDA_ITERATOR_TYPE_MASK_PHYSICAL) >= CUDA_ITERATOR_TYPE_SMS ||
    (type & CUDA_ITERATOR_TYPE_MASK_LOGICAL) >= CUDA_ITERATOR_TYPE_KERNELS;
  store_block =
    (type & CUDA_ITERATOR_TYPE_MASK_PHYSICAL) >= CUDA_ITERATOR_TYPE_WARPS ||
    (type & CUDA_ITERATOR_TYPE_MASK_LOGICAL) >= CUDA_ITERATOR_TYPE_BLOCKS;
  store_thread =
    (type & CUDA_ITERATOR_TYPE_MASK_PHYSICAL) >= CUDA_ITERATOR_TYPE_LANES ||
    (type & CUDA_ITERATOR_TYPE_MASK_LOGICAL) >= CUDA_ITERATOR_TYPE_THREADS;

  c.valid = true;
  c.dev       = store_dev    ? in->dev       : CUDA_WILDCARD;
  c.sm        = store_sm     ? in->sm        : CUDA_WILDCARD;
  c.wp        = store_warp   ? in->wp        : CUDA_WILDCARD;
  c.ln        = store_lane   ? in->ln        : CUDA_WILDCARD;
  c.kernelId  = store_kernel ? in->kernelId  : CUDA_WILDCARD;
  c.gridId    = store_grid   ? in->gridId    : CUDA_WILDCARD;
  c.blockIdx  = store_block  ? in->blockIdx  : CUDA_WILDCARD_DIM;
  c.threadIdx = store_thread ? in->threadIdx : CUDA_WILDCARD_DIM;

  return c;
}

static bool
cuda_iterator_step (cuda_iterator itr)
{
  uint64_t validWarpsMask;
  uint64_t validLanesMask;
  bool validWarp;
  bool validLane;
  cuda_coords_t *c = &itr->current;
  cuda_coords_t *filter = itr->filter.valid ? &itr->filter: NULL;
  bool valid            = itr->mask & CUDA_SELECT_VALID;
  bool at_breakpoint    = itr->mask & CUDA_SELECT_BKPT;
  bool at_exception     = itr->mask & CUDA_SELECT_EXCPT;
  bool single           = itr->mask & CUDA_SELECT_SNGL;
  struct address_space *aspace = NULL;

  if (single && itr->num_elements)
    return false;

  if (!ptid_equal (inferior_ptid, null_ptid))
    aspace = target_thread_address_space (inferior_ptid);

  for (; c->dev < cuda_system_get_num_devices (); ++c->dev)
    {
      if (filter && !cuda_val_matches (filter->dev, c->dev))
        continue;

      if (at_exception)
        device_filter_exception_state (c->dev);

      for (; c->sm < device_get_num_sms (c->dev); ++c->sm)
        {
          if (filter && !cuda_val_matches (filter->sm, c->sm))
            continue;

          validWarpsMask = sm_get_valid_warps_mask (c->dev, c->sm);
          if (valid && validWarpsMask == 0)
             continue;

          for (; c->wp < device_get_num_warps (c->dev); ++c->wp)
            {
              validWarp = (validWarpsMask>>c->wp)&1;
              if (valid && !validWarp)
                continue;
              if (filter && !cuda_val_matches (filter->wp, c->wp))
                continue;

              c->kernelId = validWarp ? warp_get_kernel_id (c->dev, c->sm, c->wp) : CUDA_INVALID;
              c->gridId   = validWarp ? warp_get_grid_id (c->dev, c->sm, c->wp) : CUDA_INVALID;
              c->blockIdx = validWarp ? warp_get_block_idx (c->dev, c->sm, c->wp) : CUDA_INVALID_DIM;

              if (filter &&
                  (!cuda_val64_matches (filter->kernelId, c->kernelId) ||
                   !cuda_val64_matches (filter->gridId, c->gridId)     ||
                   !cuda_dim3_matches (&filter->blockIdx, &c->blockIdx)))
                continue;

              validLanesMask = warp_get_valid_lanes_mask (c->dev, c->sm, c->wp);
              for (; c->ln < device_get_num_lanes (c->dev); ++c->ln)
                {
                  validLane = (validLanesMask>>c->ln)&validWarp;
                  if (valid && !validLane)
                    continue;
                  if (filter && !cuda_val_matches (filter->ln, c->ln))
                    continue;

                  c->threadIdx = validLane ? lane_get_thread_idx (c->dev, c->sm, c->wp, c->ln) : CUDA_INVALID_DIM;

                  if (filter && !cuda_dim3_matches (&filter->threadIdx, &c->threadIdx))
                    continue;

                  /* if looking for breakpoints, skip non-broken kernels */
                  if (at_breakpoint &&
                      (!validLane ||
                       !lane_is_active (c->dev, c->sm, c->wp, c->ln) ||
                       !breakpoint_here_p (aspace, lane_get_virtual_pc (c->dev, c->sm, c->wp, c->ln))))
                    continue;

                  /* if looking for exceptions, skip healthy kernels */
                  if (at_exception &&
                      (!validLane ||
                       !lane_is_active (c->dev, c->sm, c->wp, c->ln) ||
                       !lane_get_exception (c->dev, c->sm, c->wp, c->ln)))
                    continue;

                  /* allocate more memory if needed */
                  if (itr->num_elements >= itr->list_size)
                    {
                      itr->list_size *= 2;
                      itr->list = xrealloc (itr->list, itr->list_size * sizeof (*itr->list));
                    }

                  itr->list[itr->num_elements++] = cuda_iterator_filter_coords (c, itr->type);
                  ++c->ln;
                  return true;
                }
              c->ln = 0;
            }
          c->wp = 0;
        }
      c->sm = 0;
    }


  return false;
}

/* Count the number of unique elements. The duplicates are not eliminated to
   save time. We can simply hop them when iterating. */
static uint32_t
cuda_iterator_count_unique_elements (cuda_iterator itr)
{
  uint32_t i, rc;

  if (itr->num_elements == 0)
    return 0;
  for (rc = i = 1; i < itr->num_elements; ++i)
    {
      if ((itr->type & CUDA_ITERATOR_TYPE_MASK_PHYSICAL) &&
          !cuda_coords_compare_physical (&itr->list[i], &itr->list[i-1]))
        continue;
      if ((itr->type & CUDA_ITERATOR_TYPE_MASK_LOGICAL) &&
          !cuda_coords_compare_logical (&itr->list[i], &itr->list[i-1]))
        continue;
      ++rc;
    }
  return rc;
}

/* Return a threads iterator sorted by coordinates. Entries must satisfy the
   filter and other arguments.  The iterator does not return any duplicate,
   although its internal implementation will have them. */
cuda_iterator
cuda_iterator_create (cuda_iterator_type type, cuda_coords_t *filter, cuda_select_t select_mask)
{
  uint32_t i;
  cuda_iterator itr;

  itr = (cuda_iterator) xmalloc (sizeof *itr);
  itr->type         = type;
  itr->filter       = filter ? *filter: CUDA_INVALID_COORDS;
  itr->mask         = select_mask;
  itr->num_elements = 0;
  itr->num_unique_elements = 0;
  itr->list_size    = 1024;
  itr->index        = 0;
  itr->completed    = false;
  itr->list         = (cuda_coords_t*) xmalloc (itr->list_size * sizeof (*itr->list));

  if (filter)
    itr->filter.valid = true;
  itr->current.dev = itr->current.sm = itr->current.wp = itr->current.ln = 0;

  /* Iterators by physical coordinates can be lazy */
  if ((type & CUDA_ITERATOR_TYPE_MASK_PHYSICAL) != 0)
    return itr;

  while (cuda_iterator_step (itr));
  itr->completed = true;

  /* sort the list by coordinates */
  qsort (itr->list, itr->num_elements, sizeof (*itr->list),
         (int(*)(const void*, const void*))cuda_coords_compare_logical);

  return itr;
}

void
cuda_iterator_destroy (cuda_iterator itr)
{
  xfree (itr->list);
  xfree (itr);
}

cuda_iterator
cuda_iterator_start (cuda_iterator itr)
{
  itr->index = 0;
  return itr;
}

bool
cuda_iterator_end (cuda_iterator itr)
{
  if (itr->index == itr->num_elements && !itr->completed)
    itr->completed = !cuda_iterator_step (itr);
  return itr->index >= itr->num_elements;
}

cuda_iterator
cuda_iterator_next (cuda_iterator itr)
{
  if (cuda_iterator_end(itr))
    return itr;

  /* hop over the duplicate elements */
  do ++itr->index;
  while ( !cuda_iterator_end (itr) &&
          ( ((itr->type & CUDA_ITERATOR_TYPE_MASK_LOGICAL) ?
               cuda_coords_compare_logical (&itr->list[itr->index],
                                            &itr->list[itr->index-1]) :
               cuda_coords_compare_physical (&itr->list[itr->index],
                                             &itr->list[itr->index-1])) == 0)
         );

  return itr;
}

cuda_coords_t
cuda_iterator_get_current (cuda_iterator itr)
{
  return !cuda_iterator_end (itr) ? itr->list[itr->index] : CUDA_INVALID_COORDS;
}

uint32_t
cuda_iterator_get_size (cuda_iterator itr)
{
  if (!itr->completed)
    {
      while (cuda_iterator_step(itr));
      itr->completed = true;
    }

  if (itr->num_unique_elements == 0 && itr->num_elements >0)
    itr->num_unique_elements = cuda_iterator_count_unique_elements (itr);

  return itr->num_unique_elements;
}
