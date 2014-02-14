/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2014 NVIDIA Corporation
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

#include <time.h>
#include <sys/stat.h>
#include <ctype.h>
#include <pthread.h>

#include "defs.h"
#include "gdbtypes.h"
#include "gdbarch.h"
#include "value.h"
#include "arch-utils.h"
#include "command.h"

#include "cuda-iterator.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"
#include "cuda-convvars.h"


/* To update the convenience variable error code for api_failures.
 * cuda_get_last_driver_api_error_code and cuda_get_last_driver_api_error_func_name
 * are not used for updating convenience variables as they throw false errors.
 * convenience variable update can be called even if symbol values are not properly
 * initialized and can lead to false errors. cv_get_last_driver_api_error_code and
 * cv_get_last_driver_api_error_func_name ignore any such errors while updating variables.
 * */
static uint64_t
cv_get_last_driver_api_error_code ()
{
  CORE_ADDR error_code_addr;
  uint64_t res;

  error_code_addr = cuda_get_symbol_address (_STRING_(CUDBG_REPORTED_DRIVER_API_ERROR_CODE));
  if (!error_code_addr)
    {
      return 0;
    }

  target_read_memory (error_code_addr, (char *)&res, sizeof (uint64_t));
  return res;
}

static void
cv_get_last_driver_api_error_func_name (CORE_ADDR *name)
{
  CORE_ADDR error_func_name_core_addr;
  uint64_t error_func_name_addr;

  error_func_name_core_addr = cuda_get_symbol_address (
    _STRING_(CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_ADDR));
  if (!error_func_name_core_addr)
    {
      *name = 0;
      return;
    }

  target_read_memory (error_func_name_core_addr, (char *)&error_func_name_addr, sizeof (uint64_t));
  *name = (CORE_ADDR)error_func_name_addr;
}

static struct value *
cuda_convenience_convert_to_kernel_id_array_value (uint64_t *kernels,
                                                   uint32_t num_kernels)
{
  struct type *type_uint32 = builtin_type (get_current_arch ())->builtin_uint32;
  struct value *kernel_id_array_value;
  struct value **kernel_id_values;
  uint32_t i;

  kernel_id_values = xmalloc (num_kernels * sizeof(*kernel_id_values));

  for (i = 0; i < num_kernels; i++)
    kernel_id_values[i] = value_from_longest (type_uint32, (LONGEST) kernels[i]);

  kernel_id_array_value = value_array (1, num_kernels, kernel_id_values);

  xfree (kernel_id_values);

  return kernel_id_array_value;
}

static struct value *
cuda_convenience_convert_to_block_idx_array_value (CuDim3 **blocks,
                                                   uint32_t num_kernels,
                                                   uint32_t max_blocks)
{
  struct type *type_uint32 = builtin_type (get_current_arch ())->builtin_uint32;
  struct value **kernel_block_idx_array_value;
  struct value *block_idx_array_value;
  struct value **block_idx_values;
  struct value *block_idx_value[3];
  CuDim3 blockIdx;
  uint32_t i, j;

  kernel_block_idx_array_value = xmalloc (num_kernels * sizeof(*kernel_block_idx_array_value));

  for (i = 0; i < num_kernels; i++)
    {
      block_idx_values = xmalloc (max_blocks * sizeof (*block_idx_values));

      for (j = 0; j < max_blocks; j++)
      {
        blockIdx = blocks[i][j];
        block_idx_value[0] = value_from_longest (type_uint32, (LONGEST) blockIdx.x);
        block_idx_value[1] = value_from_longest (type_uint32, (LONGEST) blockIdx.y);
        block_idx_value[2] = value_from_longest (type_uint32, (LONGEST) blockIdx.z);
        block_idx_values[j] = value_array (1, 3, block_idx_value);
      }

      kernel_block_idx_array_value[i] = value_array (1, max_blocks, block_idx_values);
      xfree (block_idx_values);
    }

  block_idx_array_value = value_array (1, num_kernels, kernel_block_idx_array_value);
  xfree (kernel_block_idx_array_value);

  return block_idx_array_value;
}

/* This function creates array values of present blocks and kernel IDs and
   returns a number of active kernels */
static uint32_t
cuda_convenience_get_present_blocks_kernels(struct value **kernel_id_array_value,
                                            struct value **block_idx_array_value)
{
  struct type *type_uint32 = builtin_type (get_current_arch ())->builtin_uint32;
  struct type *type_array = lookup_array_range_type (type_uint32, 1, 0);
  CuDim3 invalid_blockIdx = (CuDim3) { CUDA_INVALID, CUDA_INVALID, CUDA_INVALID };

  cuda_coords_t c, filter;
  cuda_iterator block_iter, kernel_iter;
  uint32_t num_blocks, max_blocks, num_kernels;
  uint64_t kernel_id, i, j;

  CuDim3 **blocks;
  uint64_t *kernels;

  gdb_assert (kernel_id_array_value);
  gdb_assert (block_idx_array_value);

  /* compute num kernels and max blocks across all the kernels */
  max_blocks = 0;
  filter = CUDA_WILDCARD_COORDS;
  kernel_iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_KERNELS, &filter, CUDA_SELECT_VALID);
  num_kernels = cuda_iterator_get_size (kernel_iter);
  for (cuda_iterator_start (kernel_iter); !cuda_iterator_end (kernel_iter); cuda_iterator_next (kernel_iter))
    {
      c  = cuda_iterator_get_current (kernel_iter);
      kernel_id = c.kernelId;

      filter = CUDA_WILDCARD_COORDS;
      filter.kernelId = kernel_id;
      block_iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_BLOCKS, &filter, CUDA_SELECT_VALID);
      num_blocks = cuda_iterator_get_size (block_iter);
      max_blocks = num_blocks > max_blocks ? num_blocks : max_blocks;
      cuda_iterator_destroy (block_iter);
    }
  cuda_iterator_destroy (kernel_iter);

  /* if no block or kernel, then return empty arrays */
  if (max_blocks == 0)
    {
      *kernel_id_array_value = allocate_value (type_array);
      *block_idx_array_value = allocate_value (type_array);
      return 0;
    }

  /* fill up kernels and blocks */
  kernels = xmalloc (num_kernels * sizeof(*kernels));
  blocks  = xmalloc (num_kernels * sizeof(*blocks));

  i = 0;
  filter = CUDA_WILDCARD_COORDS;
  kernel_iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_KERNELS, &filter, CUDA_SELECT_VALID);
  for (cuda_iterator_start (kernel_iter); !cuda_iterator_end (kernel_iter); cuda_iterator_next (kernel_iter))
    {
      c  = cuda_iterator_get_current (kernel_iter);
      kernel_id = c.kernelId;

      kernels[i] = kernel_id;
      blocks[i]  = xmalloc (max_blocks * sizeof(**blocks)); // max_blocks, yes.
      for (j = 0; j < max_blocks; ++j)
        blocks[i][j] = invalid_blockIdx;

      j = 0;
      filter = CUDA_WILDCARD_COORDS;
      filter.kernelId = kernel_id;
      block_iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_BLOCKS, &filter, CUDA_SELECT_VALID);
      for (cuda_iterator_start (block_iter); !cuda_iterator_end (block_iter); cuda_iterator_next (block_iter))
        {
          c  = cuda_iterator_get_current (kernel_iter);
          blocks[i][j] = c.blockIdx;
          ++j;
        }
      cuda_iterator_destroy (block_iter);

      ++i;
    }
    cuda_iterator_destroy (kernel_iter);

  /* conversion */
  *block_idx_array_value = cuda_convenience_convert_to_block_idx_array_value (blocks, num_kernels, max_blocks);
  *kernel_id_array_value = cuda_convenience_convert_to_kernel_id_array_value (kernels, num_kernels);

  /* cleanup */
  for (i = 0; i < num_kernels; i++)
    xfree (blocks[i]);
  xfree (blocks);
  xfree (kernels);

  return num_kernels;
}


static inline void
cv_set_uint32_var (char *name, uint32_t val)
{
  struct gdbarch *gdbarch = get_current_arch ();
  struct type *type = builtin_type (gdbarch)->builtin_uint32;
  struct internalvar *var = lookup_internalvar(name);

  set_internalvar (var, value_from_longest (type, (LONGEST)val));
}

static inline void
cv_set_uint64_var (char *name, uint64_t val)
{
  struct gdbarch *gdbarch = get_current_arch ();
  struct type *type = builtin_type (gdbarch)->builtin_uint64;
  struct internalvar *var = lookup_internalvar(name);

  set_internalvar (var, value_from_longest (type, (LONGEST)val));
}

static void
cv_update_hw_vars(void)
{
  cuda_coords_t current;
  bool valid;
  uint32_t num_dev;

  valid = !cuda_coords_get_current (&current);
  num_dev   = cuda_system_get_num_devices ();

  cv_set_uint32_var ("cuda_latest_launched_kernel_id", cuda_latest_launched_kernel_id ());

  if (!valid) {
    cv_set_uint32_var ("cuda_num_devices", CUDA_INVALID);
    cv_set_uint32_var ("cuda_num_sm", CUDA_INVALID);
    cv_set_uint32_var ("cuda_num_warps", CUDA_INVALID);
    cv_set_uint32_var ("cuda_num_lanes", CUDA_INVALID);
    cv_set_uint32_var ("cuda_num_registers", CUDA_INVALID);
    return;
  }

  cv_set_uint32_var ("cuda_num_devices", cuda_system_get_num_devices());
  cv_set_uint32_var ("cuda_num_sm", device_get_num_sms (current.dev));
  cv_set_uint32_var ("cuda_num_warps", device_get_num_warps (current.dev));
  cv_set_uint32_var ("cuda_num_lanes", device_get_num_lanes (current.dev));
  cv_set_uint32_var ("cuda_num_registers", device_get_num_registers (current.dev));
}

static void
cv_update_focus_vars(void)
{
  cuda_coords_t current;
  bool valid, active;
  kernel_t kernel;
  CuDim3 grid_dim;
  CuDim3 block_dim;

  valid = !cuda_coords_get_current (&current);

  if (!valid)
  {
      cv_set_uint32_var ("cuda_grid_dim_x", CUDA_INVALID);
      cv_set_uint32_var ("cuda_grid_dim_y", CUDA_INVALID);
      cv_set_uint32_var ("cuda_grid_dim_z", CUDA_INVALID);
      cv_set_uint32_var ("cuda_block_dim_x", CUDA_INVALID);
      cv_set_uint32_var ("cuda_block_dim_y", CUDA_INVALID);
      cv_set_uint32_var ("cuda_block_dim_z", CUDA_INVALID);
      cv_set_uint32_var ("cuda_focus_device", CUDA_INVALID);
      cv_set_uint32_var ("cuda_focus_sm", CUDA_INVALID);
      cv_set_uint32_var ("cuda_focus_warp", CUDA_INVALID);
      cv_set_uint32_var ("cuda_focus_lane", CUDA_INVALID);
      cv_set_uint64_var ("cuda_focus_grid", CUDA_INVALID);
      cv_set_uint64_var ("cuda_focus_kernel_id", CUDA_INVALID);
      cv_set_uint32_var ("cuda_focus_block_x", CUDA_INVALID);
      cv_set_uint32_var ("cuda_focus_block_y", CUDA_INVALID);
      cv_set_uint32_var ("cuda_focus_block_z", CUDA_INVALID);
      cv_set_uint32_var ("cuda_focus_thread_x", CUDA_INVALID);
      cv_set_uint32_var ("cuda_focus_thread_y", CUDA_INVALID);
      cv_set_uint32_var ("cuda_focus_thread_z", CUDA_INVALID);
      cv_set_uint32_var ("cuda_thread_active", false);
      return;
  }

  kernel = warp_get_kernel (current.dev, current.sm, current.wp);
  grid_dim  = kernel_get_grid_dim (kernel);
  block_dim = kernel_get_block_dim (kernel);
  active    = lane_is_valid (current.dev, current.sm, current.wp, current.ln) &&
              lane_is_active (current.dev, current.sm, current.wp, current.ln);

  cv_set_uint32_var ("cuda_grid_dim_x", grid_dim.x);
  cv_set_uint32_var ("cuda_grid_dim_y", grid_dim.y);
  cv_set_uint32_var ("cuda_grid_dim_z", grid_dim.z);
  cv_set_uint32_var ("cuda_block_dim_x", block_dim.x);
  cv_set_uint32_var ("cuda_block_dim_y", block_dim.y);
  cv_set_uint32_var ("cuda_block_dim_z", block_dim.z);

  cv_set_uint32_var ("cuda_focus_device", current.dev);
  cv_set_uint32_var ("cuda_focus_sm", current.sm);
  cv_set_uint32_var ("cuda_focus_warp", current.wp);
  cv_set_uint32_var ("cuda_focus_lane", current.ln);
  cv_set_uint64_var ("cuda_focus_grid", current.gridId);
  cv_set_uint64_var ("cuda_focus_kernel_id", current.kernelId);
  cv_set_uint32_var ("cuda_focus_block_x", current.blockIdx.x);
  cv_set_uint32_var ("cuda_focus_block_y", current.blockIdx.y);
  cv_set_uint32_var ("cuda_focus_block_z", current.blockIdx.z);
  cv_set_uint32_var ("cuda_focus_thread_x", current.threadIdx.x);
  cv_set_uint32_var ("cuda_focus_thread_y", current.threadIdx.y);
  cv_set_uint32_var ("cuda_focus_thread_z", current.threadIdx.z);
  cv_set_uint32_var ("cuda_thread_active", active);
}

static void
cv_update_memcheck_vars(void)
{
  cuda_coords_t cur;
  bool valid;
  uint64_t addr = 0;
  ptxStorageKind segm = ptxUNSPECIFIEDStorage;

  valid = !cuda_coords_get_current (&cur);
  if (valid)
    {
      addr = lane_get_memcheck_error_address (cur.dev, cur.sm, cur.wp, cur.ln);
      segm = lane_get_memcheck_error_address_segment (cur.dev, cur.sm, cur.wp, cur.ln);
    }

  cv_set_uint64_var ("cuda_memcheck_error_address", addr);
  cv_set_uint32_var ("cuda_memcheck_error_address_segment", segm);
}

static void
cv_update_lineno_var(void)
{
  cuda_coords_t cur;
  bool valid;
  uint64_t pc = 0ULL;
  uint32_t lineno = 0;

  valid = !cuda_coords_get_current (&cur);
  if (valid)
    {
      pc      = lane_get_virtual_pc (cur.dev, cur.sm, cur.wp, cur.ln);
      lineno  = find_pc_line (pc, 0).line;
    }

  cv_set_uint32_var ("cuda_thread_lineno", lineno);
}

static void
cv_update_call_depth_var(void)
{
  cuda_coords_t cur;
  bool valid;
  uint32_t  depth = CUDA_INVALID;

  valid = !cuda_coords_get_current (&cur);
  if (valid)
    {
      depth = lane_get_call_depth (cur.dev, cur.sm, cur.wp, cur.ln);
    }

  cv_set_uint32_var ("cuda_call_depth", depth);
}

static void
cv_update_syscall_depth_var(void)
{
  cuda_coords_t cur;
  bool valid;
  uint32_t  depth = CUDA_INVALID;

  valid = !cuda_coords_get_current (&cur);
  if (valid)
    {
      depth = lane_get_syscall_call_depth (cur.dev, cur.sm, cur.wp, cur.ln);
    }

  cv_set_uint32_var ("cuda_syscall_call_depth", depth);
}

static void
cv_update_api_failures_vars(void)
{
  struct gdbarch *gdbarch = get_current_arch ();
  struct type *type_data_ptr = builtin_type (gdbarch)->builtin_data_ptr;
  CORE_ADDR func_name = 0;
  uint64_t rc;

  rc = cv_get_last_driver_api_error_code ();
  cv_get_last_driver_api_error_func_name (&func_name);

  cv_set_uint64_var ("cuda_api_failure_return_code", rc);
  set_internalvar (lookup_internalvar ("cuda_api_failure_func_name"),
                   value_from_pointer (type_data_ptr, func_name));
}

static void
cv_update_present_kernels_vars(void)
{
  uint32_t num_kernels;
  struct value *kernel_array, *blocks_array;

  num_kernels = cuda_convenience_get_present_blocks_kernels (&kernel_array, &blocks_array);

  cv_set_uint32_var ("cuda_num_present_kernels", num_kernels);
  set_internalvar (lookup_internalvar ("cuda_present_kernel_ids"),
                   kernel_array);
  set_internalvar (lookup_internalvar ("cuda_present_block_idxs"),
                   blocks_array);
}

static void cv_update_total_kernels_var(void)
{
  cv_set_uint32_var ("cuda_num_total_kernels", cuda_system_get_num_present_kernels ());
}

struct cv_variable_group {
  char *group_name;
  bool enabled;
  void (*update_func)(void);
  char *group_desc;
};

static struct cv_variable_group cv_var_groups[] =
{
  {"hw_vars", false, cv_update_hw_vars,
    "HW specific variables, like number of sms, lanes, warps, etc"},
  {"focus", false, cv_update_focus_vars,
    "CUDA focus specific variables"},
  {"memcheck", false, cv_update_memcheck_vars,
    "Memcheck error address and segment"},
  {"lineno", false, cv_update_lineno_var,
    "Line number that matches $pc in focus"},
  {"call_depth", false, cv_update_call_depth_var,
    "Call depth"},
  {"syscall_depth", false, cv_update_syscall_depth_var,
    "Systemcall depth"},
  {"api_failures", false, cv_update_api_failures_vars,
    "API failure error number and function name"},
  {"present_kernels", false, cv_update_present_kernels_vars,
    "Kernel and their ids currently present on GPUs"},
  {"total_kernels", false, cv_update_total_kernels_var,
    "Total number of kernels on GPUs"},
  {NULL, false, NULL, NULL},
};


int
cuda_enable_convenience_variables_group (char *name, bool enable)
{
  struct cv_variable_group *grp;
  int rc = false;

  for (grp=cv_var_groups;grp->group_name;grp++)
    if (!name || strcasecmp(name,grp->group_name)==0)
      {
         grp->enabled = enable;
         rc = true;
      }
  return rc;
}

void
cuda_update_convenience_variables (void)
{
  struct value *mark = NULL;
  struct cv_variable_group *grp;

  mark = value_mark ();

  for (grp=cv_var_groups;grp->group_name;grp++)
    if (grp->enabled)
      grp->update_func();

  /* Free the temporary values */
  value_free_to_mark (mark);
}

/* Prepare help for [set|show] debug cuda convenience_vars */
void
cuda_build_covenience_variables_help_message (char *ptr, int size)
{
  int rc;
  struct cv_variable_group *grp;

  rc = snprintf (ptr, size,
    _("Specifies which convenience variables groups are available for debugging.\n"));
  ptr += rc; size -= rc;
  rc = snprintf (ptr, size,
    _("Groups names are: \"all\",\"none\" or comma separate list of the folowing:\n"));
  ptr += rc; size -= rc;
  for (grp=cv_var_groups;grp->group_name;grp++)
    {
      rc = snprintf (ptr, size, " %*s : %s\n",
                     20, grp->group_name, grp->group_desc);
      if (rc <= 0) break;
      ptr += rc;
      size -= rc;
    }
}
