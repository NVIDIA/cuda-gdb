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

#include <string>

#include "frame.h"
#include "ui-out.h"

#include "cuda-api.h"
#include "cuda-asm.h"
#include "cuda-coord-set.h"
#include "cuda-context.h"
#include "cuda-modules.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"

/* counter for the CUDA kernel ids */
static uint64_t next_kernel_id = 0;

uint64_t
cuda_latest_launched_kernel_id (void)
{
  return next_kernel_id - 1;
}

/* forward declaration */
static void kernels_add_parent_kernel (uint32_t dev_id, uint64_t grid_id,
                                       uint64_t *parent_grid_id);

/******************************************************************************
 *
 *                                   Kernel
 *
 *****************************************************************************/

struct kernel_st
{
  bool grid_status_p;
  bool cluster_dim_p;
  uint64_t id;                 /* unique kernel id per GDB session */
  uint32_t dev_id;             /* device where the kernel was launched */
  uint64_t grid_id;            /* unique kernel id per device */
  CUDBGGridStatus grid_status; /* current grid status of the kernel */
  kernel_t parent;             /* the kernel that launched this grid */
  kernel_t children;           /* list of children */
  kernel_t siblings; /* next sibling when traversing the list of children */
  gdb::unique_xmalloc_ptr<char> name; /* name of the kernel if available */
  gdb::unique_xmalloc_ptr<char> args; /* kernel arguments in string format */
  uint64_t virt_code_base;  /* virtual address of the kernel entry point */
  cuda_module* module;      /* cuda_module of the kernel */
  bool launched;            /* Has the kernel been seen on the hw? */
  CuDim3 grid_dim;          /* The grid dimensions of the kernel. */
  CuDim3 cluster_dim;       /* The cluster dimensions of the kernel. */
  CuDim3 block_dim;         /* The block dimensions of the kernel. */
  char dimensions[128];     /* A string repr. of the kernel dimensions. */
  CUDBGKernelType type;     /* The kernel type: system or application. */
  CUDBGKernelOrigin origin; /* The kernel origin: CPU or GPU */
  kernel_t next;            /* next kernel on the same device */
  unsigned int depth;       /* kernel nest level (0 - host launched kernel) */
};

static void
kernel_add_child (kernel_t parent, kernel_t child)
{
  gdb_assert (child);

  if (!parent)
    return;

  child->siblings = parent->children;
  parent->children = child;
}

static void
kernel_remove_child (kernel_t parent, kernel_t child)
{
  kernel_t cur, prev;

  gdb_assert (child);

  if (!parent)
    return;

  if (parent->children == child)
    {
      parent->children = child->siblings;
      return;
    }

  for (prev = parent->children, cur = parent->children->siblings; cur != NULL;
       prev = cur, cur = cur->siblings)
    if (cur == child)
      {
        prev->siblings = cur->siblings;
        break;
      }
}

static bool
should_print_kernel_event (kernel_t kernel)
{
  unsigned int depth_or_disabled = cuda_options_show_kernel_events_depth ();

  if (depth_or_disabled && kernel->depth > depth_or_disabled - 1)
    return false;

  return (kernel->type == CUDBG_KNL_TYPE_SYSTEM
          && cuda_options_show_kernel_events_system ())
         || (kernel->type == CUDBG_KNL_TYPE_APPLICATION
             && cuda_options_show_kernel_events_application ());
}

static kernel_t
kernel_new (uint32_t dev_id, uint64_t grid_id, uint64_t virt_code_base,
            gdb::unique_xmalloc_ptr<char> name, cuda_module* module,
            CuDim3 grid_dim, CuDim3 block_dim, CUDBGKernelType type,
            uint64_t parent_grid_id, CUDBGKernelOrigin origin,
            bool has_cluster_dim, CuDim3 cluster_dim)
{
  kernel_t kernel;
  kernel_t parent_kernel;

  parent_kernel = kernels_find_kernel_by_grid_id (dev_id, parent_grid_id);
  if (!parent_kernel && origin == CUDBG_KNL_ORIGIN_GPU)
    {
      kernels_add_parent_kernel (dev_id, grid_id, &parent_grid_id);
      parent_kernel = kernels_find_kernel_by_grid_id (dev_id, parent_grid_id);
    }

  kernel = new struct kernel_st;

  kernel->grid_status_p = false;
  kernel->cluster_dim_p = has_cluster_dim;
  kernel->id = next_kernel_id++;
  kernel->dev_id = dev_id;
  kernel->grid_id = grid_id;
  kernel->parent = parent_kernel;
  kernel->children = NULL;
  kernel->siblings = NULL;
  kernel->virt_code_base = virt_code_base;
  kernel->name = name ? std::move (name) : make_unique_xstrdup ("??");
  kernel->args = make_unique_xstrdup ("");
  kernel->module = module;
  kernel->grid_dim = grid_dim;
  kernel->cluster_dim = cluster_dim;
  kernel->block_dim = block_dim;
  kernel->type = type;
  kernel->origin = origin;
  kernel->next = NULL;
  kernel->depth = !parent_kernel ? 0 : parent_kernel->depth + 1;

  snprintf (kernel->dimensions, sizeof (kernel->dimensions),
            "<<<(%d,%d,%d),(%d,%d,%d)>>>", grid_dim.x, grid_dim.y, grid_dim.z,
            block_dim.x, block_dim.y, block_dim.z);

  kernel->launched = false;

  kernel_add_child (parent_kernel, kernel);

  if (should_print_kernel_event (kernel))
    printf_unfiltered (
        _ ("[Launch of CUDA Kernel %llu (%s%s) on Device %u, level %u]\n"),
        (unsigned long long)kernel->id, kernel->name.get (),
        kernel->dimensions, kernel->dev_id, kernel->depth);

  return kernel;
}

static void
kernel_delete (kernel_t kernel)
{
  gdb_assert (kernel);

  kernel_remove_child (kernel->parent, kernel);

  if (should_print_kernel_event (kernel))
    printf_unfiltered (_ ("[Termination of CUDA Kernel %llu (%s%s) on Device "
                          "%u, level %u]\n"),
                       (unsigned long long)kernel->id, kernel->name.get (),
                       kernel->dimensions, kernel->dev_id, kernel->depth);

  delete kernel;
}

void
kernel_invalidate (kernel_t kernel)
{
  cuda_trace ("kernel %llu: invalidate", (unsigned long long)kernel->id);

  kernel->grid_status_p = false;
  kernel->cluster_dim_p = false;
}

uint64_t
kernel_get_id (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->id;
}

const char *
kernel_get_name (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->name.get ();
}

static void
kernel_populate_args (kernel_t kernel)
{
  /* Find an active lane for the kernel */
  cuda_coords filter{ CUDA_WILDCARD,          CUDA_WILDCARD,
                      CUDA_WILDCARD,          CUDA_WILDCARD,
                      kernel_get_id (kernel), CUDA_WILDCARD,
                      CUDA_WILDCARD_DIM,      CUDA_WILDCARD_DIM,
                      CUDA_WILDCARD_DIM };
  cuda_coord_set<cuda_coord_set_type::lanes, select_valid | select_sngl> coord{
    filter
  };
  /* Cannot populate args if we didn't find an active lane */
  if (!coord.size ())
    return;

  /* Save environment */
  string_file stream;
  current_uiout->redirect (&stream);

  /* Make sure we switch back to the current focus when done. */
  cuda_focus_restore r;

  try
    {
      /* Switch focus to that lane/kernel, temporarily */
      switch_to_cuda_thread (*coord.cbegin ());

      /* Find the outermost frame */
      frame_info_ptr frame = get_current_frame ();
      frame_info_ptr prev_frame = get_prev_frame (frame);
      while (prev_frame)
        {
          frame = prev_frame;
          prev_frame = get_prev_frame (frame);
        }

      /* Print the arguments */
      print_args_frame (frame);
      kernel->args = make_unique_xstrdup (stream.string ().c_str ());
    }
  catch (const gdb_exception_error &e)
    {
      kernel->args = make_unique_xstrdup ("");
    }

  /* Restore environment */
  current_uiout->redirect (NULL);
}

const char *
kernel_get_args (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->args.get ();
}

uint64_t
kernel_get_grid_id (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->grid_id;
}

kernel_t
kernel_get_parent (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->parent;
}

kernel_t
kernel_get_children (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->children;
}

kernel_t
kernel_get_sibling (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->siblings;
}

uint64_t
kernel_get_virt_code_base (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->virt_code_base;
}

cuda_context*
kernel_get_context (kernel_t kernel)
{
  gdb_assert (kernel);
  gdb_assert (kernel->module);
  return kernel->module->context ();
}

cuda_module*
kernel_get_module (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->module;
}

uint32_t
kernel_get_dev_id (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->dev_id;
}

CuDim3
kernel_get_grid_dim (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->grid_dim;
}

/* This will return the normal cluster size only. If it is all zero,
   that means no clusters are present and the preferred cluster size
   is also ignored. This value may differ from the per warp cluster
   dim sizes. */
CuDim3
kernel_get_cluster_dim (kernel_t kernel)
{
  gdb_assert (kernel);
  if (!kernel->cluster_dim_p)
    {
      CUDBGGridInfo grid_info;
      cuda_debugapi::get_grid_info (kernel->dev_id, kernel->grid_id,
                                      &grid_info);
      kernel->cluster_dim = grid_info.clusterDim;
      kernel->cluster_dim_p = CACHED;
    }
  return kernel->cluster_dim;
}

CuDim3
kernel_get_block_dim (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->block_dim;
}

const char *
kernel_get_dimensions (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->dimensions;
}

CUDBGKernelType
kernel_get_type (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->type;
}

CUDBGGridStatus
kernel_get_status (kernel_t kernel)
{
  gdb_assert (kernel);

  if (!kernel->grid_status_p)
    {
      cuda_debugapi::get_grid_status (kernel->dev_id, kernel->grid_id,
                                      &kernel->grid_status);
      kernel->grid_status_p = CACHED;
    }

  return kernel->grid_status;
}

CUDBGKernelOrigin
kernel_get_origin (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->origin;
}

uint32_t
kernel_get_depth (kernel_t kernel)
{
  kernel_t k;
  uint32_t depth = -1;

  gdb_assert (kernel);

  for (k = kernel; k; k = kernel_get_parent (k))
    ++depth;

  return depth;
}

uint32_t
kernel_get_num_children (kernel_t kernel)
{
  kernel_t k;
  uint32_t num_children = 0;

  gdb_assert (kernel);

  for (k = kernel_get_children (kernel); k; k = kernel_get_sibling (k))
    ++num_children;

  return num_children;
}

bool
kernel_has_launched (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->launched;
}

bool
kernel_is_present (kernel_t kernel)
{
  CUDBGGridStatus status;
  bool present;

  gdb_assert (kernel);

  status = kernel_get_status (kernel);
  present = (status == CUDBG_GRID_STATUS_ACTIVE
             || status == CUDBG_GRID_STATUS_SLEEPING);

  return present;
}

void
kernel_compute_sms_mask (kernel_t kernel, cuda_bitset &sms_mask)
{
  gdb_assert (kernel);

  cuda_coords filter{
    kernel->dev_id,    CUDA_WILDCARD,     CUDA_WILDCARD,
    CUDA_WILDCARD,     kernel->id,        kernel->grid_id,
    CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM
  };
  cuda_coord_set<cuda_coord_set_type::sms, select_valid,
                cuda_coord_compare_type::physical>
      coords{ filter };

  // Reset the bitset passed in
  sms_mask.resize (cuda_state::device_get_num_sms (kernel->dev_id));

  for (const auto &coord : coords)
    sms_mask.set (coord.physical ().sm ());
}

void
kernel_print (kernel_t kernel)
{
  gdb_assert (kernel);

  fprintf (stderr, "    Kernel %llu:\n", (unsigned long long)kernel->id);
  fprintf (stderr, "        name        : %s\n", kernel->name.get ());
  fprintf (stderr, "        device id   : %u\n", kernel->dev_id);
  fprintf (stderr, "        grid id     : %lld\n", (long long)kernel->grid_id);
  fprintf (stderr, "        module id   : 0x%llx\n",
           (unsigned long long)kernel->module->id ());
  fprintf (stderr, "        entry point : 0x%llx\n",
           (unsigned long long)kernel->virt_code_base);
  fprintf (stderr, "        dimensions  : %s\n", kernel->dimensions);
  fprintf (stderr, "        launched    : %s\n",
           kernel->launched ? "yes" : "no");
  fprintf (stderr, "        present     : %s\n",
           kernel_is_present (kernel) ? "yes" : "no");
  fprintf (stderr, "        next        : 0x%llx\n",
           (unsigned long long)(uintptr_t)kernel->next);
  fflush (stderr);
}

/******************************************************************************
 *
 *                                   Kernels
 *
 *****************************************************************************/

/* head of the system list of kernels */
static kernel_t kernels = NULL;

void
kernels_print (void)
{
  kernel_t kernel;

  for (kernel = kernels; kernel; kernel = kernels_get_next_kernel (kernel))
    kernel_print (kernel);
}

void
kernels_start_kernel (uint32_t dev_id, uint64_t grid_id,
                      uint64_t virt_code_base, uint64_t context_id,
                      uint64_t module_id, CuDim3 grid_dim, CuDim3 block_dim,
                      CUDBGKernelType type, uint64_t parent_grid_id,
                      CUDBGKernelOrigin origin, bool has_cluster_dim,
                      CuDim3 cluster_dim)
{
  auto context = cuda_state::find_context_by_id (context_id);
  if (!context)
    {
      warning ("Could not find CUDA context for context_id 0x%llx",
	       (unsigned long long)context_id);
      return;
    }

  auto module = cuda_state::find_module_by_id (module_id);
  if (!module)
    {
      warning ("Could not find CUDA module for context_id 0x%llx module_id 0x%llx",
	       (unsigned long long)context_id, (unsigned long long)module_id);
      return;
    }

  cuda_state::set_current_context (context);

  gdb::unique_xmalloc_ptr<char> kernel_name
    = cuda_find_function_name_from_pc (virt_code_base, true);

  // NOTE: Not having an entry function is a normal situation, this means
  // an internal kernel contained in a public module was launched.
  if (kernel_name.get () == nullptr)
    kernel_name = make_unique_xstrdup ("<internal>");

  auto kernel
    = kernel_new (dev_id, grid_id, virt_code_base, std::move (kernel_name),
		  module, grid_dim, block_dim, type, parent_grid_id, origin,
		  has_cluster_dim, cluster_dim);

  kernel->next = kernels;
  kernels = kernel;
}

static void
kernels_add_parent_kernel (uint32_t dev_id, uint64_t grid_id,
                           uint64_t *parent_grid_id)
{
  CUDBGGridInfo grid_info;
  CUDBGGridInfo parent_grid_info;
  CUDBGGridStatus grid_status;

  cuda_debugapi::get_grid_status (dev_id, grid_id, &grid_status);
  if (grid_status == CUDBG_GRID_STATUS_INVALID)
    return;

  cuda_debugapi::get_grid_info (dev_id, grid_id, &grid_info);

  cuda_debugapi::get_grid_status (dev_id, grid_info.parentGridId,
                                  &grid_status);
  if (grid_status == CUDBG_GRID_STATUS_INVALID)
    return;

  cuda_debugapi::get_grid_info (dev_id, grid_info.parentGridId,
                                &parent_grid_info);
  *parent_grid_id = parent_grid_info.gridId64;
  kernels_start_kernel (parent_grid_info.dev, parent_grid_info.gridId64,
                        parent_grid_info.functionEntry,
                        parent_grid_info.context, parent_grid_info.module,
                        parent_grid_info.gridDim, parent_grid_info.blockDim,
                        parent_grid_info.type, parent_grid_info.parentGridId,
                        parent_grid_info.origin);
}

void
kernels_terminate_kernel (kernel_t kernel)
{
  kernel_t prev, ker;

  if (!kernel)
    return;

  // must keep kernel object until all the children have terminated
  if (kernel->children)
    return;

  for (ker = kernels, prev = NULL; ker && ker != kernel;
       prev = ker, ker = kernels_get_next_kernel (ker))
    ;
  gdb_assert (ker);

  if (prev)
    prev->next = kernels_get_next_kernel (kernel);
  else
    kernels = kernels_get_next_kernel (kernel);

  kernel_delete (kernel);
}

void
kernels_terminate_module (cuda_module* module)
{
  kernel_t kernel, next_kernel;

  gdb_assert (module);

  kernel = kernels_get_first_kernel ();
  while (kernel)
    {
      next_kernel = kernels_get_next_kernel (kernel);
      if (kernel_get_module (kernel) == module)
        kernels_terminate_kernel (kernel);
      kernel = next_kernel;
    }
}

kernel_t
kernels_get_first_kernel (void)
{
  return kernels;
}

kernel_t
kernels_get_next_kernel (kernel_t kernel)
{
  if (!kernel)
    return NULL;

  return kernel->next;
}

kernel_t
kernels_find_kernel_by_grid_id (uint32_t dev_id, uint64_t grid_id)
{
  kernel_t kernel;

  for (kernel = kernels; kernel; kernel = kernels_get_next_kernel (kernel))
    if (kernel->dev_id == dev_id && kernel->grid_id == grid_id)
      return kernel;

  return NULL;
}

kernel_t
kernels_find_kernel_by_kernel_id (uint64_t kernel_id)
{
  kernel_t kernel;

  for (kernel = kernels; kernel; kernel = kernels_get_next_kernel (kernel))
    if (kernel->id == kernel_id)
      return kernel;

  return NULL;
}

void
kernels_invalidate (uint32_t dev_id)
{

  for (auto kernel = kernels_get_first_kernel (); kernel; )
    {
      if (kernel->dev_id == dev_id)
	kernel_invalidate (kernel);
      kernel = kernels_get_next_kernel (kernel);
    }
}

void
kernels_update_args (void)
{
  kernel_t kernel;

  for (kernel = kernels_get_first_kernel (); kernel;
       kernel = kernels_get_next_kernel (kernel))
    if (!kernel->args && kernel_is_present (kernel))
      kernel_populate_args (kernel);
}

void
kernels_update_terminated (void)
{
  kernel_t kernel;
  kernel_t next_kernel;

  /* Make sure we have up-to-date information about running kernels.
   * TODO: This feels like we are relying on a side-effect by creating an
   * iterator. */
  cuda_coord_set<cuda_coord_set_type::kernels, select_valid> coord{
    cuda_coords::wild ()
  };

  /* rediscover the kernels currently running on the hardware */
  kernel = kernels_get_first_kernel ();
  while (kernel)
    {
      next_kernel = kernels_get_next_kernel (kernel);

      if (kernel_is_present (kernel))
        kernel->launched = true;

      /* terminate the kernels that we had seen running at some point
         but are not here on the hardware anymore. If there is any child kernel
         still present, keep the data available. */
      if (kernel->launched && !kernel_is_present (kernel))
        kernels_terminate_kernel (kernel);

      kernel = next_kernel;
    }
}
