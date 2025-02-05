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
#include "defs.h"
#include <ctype.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "arch-utils.h"
#include "command.h"
#include "gdbarch.h"
#include "gdbtypes.h"
#include "value.h"

#include "cuda-convvars.h"
#include "cuda-coord-set.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"

/* To update the convenience variable error code for api_failures.
 * cuda_get_last_driver_api_error_code and
 * cuda_get_last_driver_api_error_func_name are not used for updating
 * convenience variables as they throw false errors. convenience variable
 * update can be called even if symbol values are not properly initialized and
 * can lead to false errors. cv_get_last_driver_api_error_code and
 * cv_get_last_driver_api_error_func_name ignore any such errors while updating
 * variables.
 * */
static uint64_t
cv_get_last_driver_api_error_code (void)
{
  CORE_ADDR error_code_addr;
  uint64_t res;

  error_code_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_CODE));
  if (!error_code_addr)
    {
      return 0;
    }

  target_read_memory (error_code_addr, (gdb_byte *)&res, sizeof (uint64_t));
  return res;
}

static void
cv_get_last_driver_api_error_func_name (CORE_ADDR *name)
{
  CORE_ADDR error_func_name_core_addr;
  uint64_t error_func_name_addr;

  error_func_name_core_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_ADDR));
  if (!error_func_name_core_addr)
    {
      *name = 0;
      return;
    }

  target_read_memory (error_func_name_core_addr,
                      (gdb_byte *)&error_func_name_addr, sizeof (uint64_t));
  *name = (CORE_ADDR)error_func_name_addr;
}

/* This function creates array values of present blocks and kernel IDs and
   returns a number of active kernels */
static uint32_t
cuda_convenience_get_present_blocks_kernels (
    struct value **kernel_id_array_value, struct value **block_idx_array_value)
{
  gdb_assert (kernel_id_array_value);
  gdb_assert (block_idx_array_value);

  struct type *type_uint32
      = builtin_type (get_current_arch ())->builtin_uint32;
  struct type *type_array = lookup_array_range_type (type_uint32, 1, 0);

  /* This represents the array of active kernel Ids */
  std::vector<uint64_t> kernels;
  /*
   * gdb value arrays require elements to be the same size, so we ensure
   * each kernel has the same number of blocks and set the blocks outside
   * the range to CUDA_INVALID.
   */
  uint32_t max_blocks = 0;
  /* This represents the array of iterators for every block in every kernel */
  std::vector<cuda_coord_set<cuda_coord_set_type::blocks, select_valid> >
      blocks_itr;

  /* Build up kernel and block iterators */
  cuda_coord_set<cuda_coord_set_type::kernels, select_valid> kernels_itr{
    cuda_coords::wild ()
  };
  /* Iterate over each kernel to count the block sizes */
  for (const auto &kern : kernels_itr)
    {
      /* Save this kernelId */
      kernels.push_back (kern.logical ().kernelId ());
      /* Create an iterator over every block in the kernel */
      cuda_coords filter{ kern.physical ().dev (),
                          CUDA_WILDCARD,
                          CUDA_WILDCARD,
                          CUDA_WILDCARD,
                          kern.logical ().kernelId (),
                          CUDA_WILDCARD,
                          CUDA_WILDCARD_DIM,
                          CUDA_WILDCARD_DIM,
                          CUDA_WILDCARD_DIM };
      cuda_coord_set<cuda_coord_set_type::blocks, select_valid> blocks{ filter };
      /* Update max blocks */
      max_blocks = std::max ((uint32_t)blocks.size (), max_blocks);
      /* Save the blocks iterator */
      blocks_itr.push_back (std::move (blocks));
    }

  /* if no block or kernel, then return empty arrays */
  if (!kernels.size ())
    {
      *kernel_id_array_value = allocate_value (type_array);
      *block_idx_array_value = allocate_value (type_array);
      return 0;
    }

  gdb_assert (kernels.size () == blocks_itr.size ());

  /* Create the block value array */
  std::vector<struct value *> kernel_block_idx_val_arr;
  /* Iterate over every block iterator for every kernel*/
  for (auto &block_itr : blocks_itr)
    {
      std::vector<struct value *> block_idx_val_arr;
      // Iterate over every block in the block iterator
      for (const auto &block : block_itr)
        {
          std::vector<struct value *> block_dim_val_arr;
          block_dim_val_arr.push_back (value_from_longest (
              type_uint32, (LONGEST)block.logical ().blockIdx ().x));
          block_dim_val_arr.push_back (value_from_longest (
              type_uint32, (LONGEST)block.logical ().blockIdx ().y));
          block_dim_val_arr.push_back (value_from_longest (
              type_uint32, (LONGEST)block.logical ().blockIdx ().z));
          block_idx_val_arr.push_back (
              value_array (1, 3, block_dim_val_arr.data ()));
        }
      // Value arrays require each element to be the same size - push back any
      // uninitialized values up to max_blocks
      for (auto i = max_blocks - block_itr.size (); i > 0; i--)
        {
          std::vector<struct value *> block_dim_val_arr;
          block_dim_val_arr.push_back (
              value_from_longest (type_uint32, (LONGEST)CUDA_INVALID));
          block_dim_val_arr.push_back (
              value_from_longest (type_uint32, (LONGEST)CUDA_INVALID));
          block_dim_val_arr.push_back (
              value_from_longest (type_uint32, (LONGEST)CUDA_INVALID));
          block_idx_val_arr.push_back (
              value_array (1, 3, block_dim_val_arr.data ()));
        }
      // Push back the completed block arrays for this kernel
      kernel_block_idx_val_arr.push_back (
          value_array (1, max_blocks, block_idx_val_arr.data ()));
    }
  /* We have completed the block value array. */
  *block_idx_array_value
      = value_array (1, kernels.size (), kernel_block_idx_val_arr.data ());

  /* Create the kernelId value array */
  std::vector<struct value *> kernel_val_arr;
  /* Iterate over every kernelId */
  for (auto kernelId : kernels)
    kernel_val_arr.push_back (
        value_from_longest (type_uint32, (LONGEST)kernelId));
  /* We have completed the kernelId value array. */
  *kernel_id_array_value
      = value_array (1, kernels.size (), kernel_val_arr.data ());

  return kernels.size ();
}

static inline void
cv_set_uint32_var (const char *name, uint32_t val)
{
  struct gdbarch *gdbarch = get_current_arch ();
  struct type *type = builtin_type (gdbarch)->builtin_uint32;
  struct internalvar *var = lookup_internalvar (name);

  set_internalvar (var, value_from_longest (type, (LONGEST)val));
}

static inline void
cv_set_uint64_var (const char *name, uint64_t val)
{
  struct gdbarch *gdbarch = get_current_arch ();
  struct type *type = builtin_type (gdbarch)->builtin_uint64;
  struct internalvar *var = lookup_internalvar (name);

  set_internalvar (var, value_from_longest (type, (LONGEST)val));
}

static void
cv_update_hw_vars (void)
{
  cv_set_uint32_var ("cuda_latest_launched_kernel_id",
                     cuda_latest_launched_kernel_id ());

  if (!cuda_current_focus::isDevice ())
    {
      cv_set_uint32_var ("cuda_num_devices", CUDA_INVALID);
      cv_set_uint32_var ("cuda_num_sm", CUDA_INVALID);
      cv_set_uint32_var ("cuda_num_warps", CUDA_INVALID);
      cv_set_uint32_var ("cuda_num_lanes", CUDA_INVALID);
      cv_set_uint32_var ("cuda_num_registers", CUDA_INVALID);
      cv_set_uint32_var ("cuda_num_uregisters", CUDA_INVALID);
      cv_set_uint32_var ("cuda_num_registers_allocated", CUDA_INVALID);
      cv_set_uint32_var ("cuda_shared_memory_size", CUDA_INVALID);
      return;
    }

  const auto &c = cuda_current_focus::get ().physical ();
  cv_set_uint32_var ("cuda_num_devices", cuda_state::get_num_devices ());
  cv_set_uint32_var ("cuda_num_sm", cuda_state::device_get_num_sms (c.dev ()));
  cv_set_uint32_var ("cuda_num_warps",
                     cuda_state::device_get_num_warps (c.dev ()));
  cv_set_uint32_var ("cuda_num_lanes",
                     cuda_state::device_get_num_lanes (c.dev ()));
  cv_set_uint32_var ("cuda_num_registers",
                     cuda_state::device_get_num_registers (c.dev ()));
  cv_set_uint32_var ("cuda_num_uregisters",
                     cuda_state::device_get_num_uregisters (c.dev ()));
  cv_set_uint32_var ("cuda_num_registers_allocated",
        	     cuda_state::warp_registers_allocated (c.dev (), c.sm (), c.wp ()));
  cv_set_uint32_var ("cuda_shared_memory_size",
        	     cuda_state::warp_shared_mem_size (c.dev (), c.sm (), c.wp ()));
}

static void
cv_update_focus_vars (void)
{
  if (!cuda_current_focus::isDevice ())
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

  const auto &current = cuda_current_focus::get ();
  const auto &p = current.physical ();
  const auto &l = current.logical ();
  auto kernel = cuda_state::warp_get_kernel (p.dev (), p.sm (), p.wp ());
  auto grid_dim = kernel_get_grid_dim (kernel);
  auto block_dim = kernel_get_block_dim (kernel);
  bool active
      = cuda_state::lane_valid (p.dev (), p.sm (), p.wp (), p.ln ())
        && cuda_state::lane_active (p.dev (), p.sm (), p.wp (), p.ln ());

  cv_set_uint32_var ("cuda_grid_dim_x", grid_dim.x);
  cv_set_uint32_var ("cuda_grid_dim_y", grid_dim.y);
  cv_set_uint32_var ("cuda_grid_dim_z", grid_dim.z);
  cv_set_uint32_var ("cuda_block_dim_x", block_dim.x);
  cv_set_uint32_var ("cuda_block_dim_y", block_dim.y);
  cv_set_uint32_var ("cuda_block_dim_z", block_dim.z);

  cv_set_uint32_var ("cuda_focus_device", p.dev ());
  cv_set_uint32_var ("cuda_focus_sm", p.sm ());
  cv_set_uint32_var ("cuda_focus_warp", p.wp ());
  cv_set_uint32_var ("cuda_focus_lane", p.ln ());
  cv_set_uint64_var ("cuda_focus_grid", l.gridId ());
  cv_set_uint64_var ("cuda_focus_kernel_id", l.kernelId ());
  cv_set_uint32_var ("cuda_focus_block_x", l.blockIdx ().x);
  cv_set_uint32_var ("cuda_focus_block_y", l.blockIdx ().y);
  cv_set_uint32_var ("cuda_focus_block_z", l.blockIdx ().z);
  cv_set_uint32_var ("cuda_focus_thread_x", l.threadIdx ().x);
  cv_set_uint32_var ("cuda_focus_thread_y", l.threadIdx ().y);
  cv_set_uint32_var ("cuda_focus_thread_z", l.threadIdx ().z);
  cv_set_uint32_var ("cuda_thread_active", active);
}

static void
cv_update_lineno_var (void)
{
  uint64_t pc = 0ULL;
  uint32_t lineno = 0;

  if (cuda_current_focus::isDevice ())
    {
      const auto &c = cuda_current_focus::get ().physical ();
      pc = cuda_state::lane_get_pc (c.dev (), c.sm (), c.wp (),
                                    c.ln ());
      lineno = find_pc_line (pc, 0).line;
    }

  cv_set_uint32_var ("cuda_thread_lineno", lineno);
}

static void
cv_update_call_depth_var (void)
{
  uint32_t depth = CUDA_INVALID;

  if (cuda_current_focus::isDevice ())
    {
      const auto &c = cuda_current_focus::get ().physical ();
      depth = cuda_state::lane_get_call_depth (c.dev (), c.sm (), c.wp (),
                                               c.ln ());
    }

  cv_set_uint32_var ("cuda_call_depth", depth);
}

static void
cv_update_syscall_depth_var (void)
{
  uint32_t depth = CUDA_INVALID;

  if (cuda_current_focus::isDevice ())
    {
      const auto &c = cuda_current_focus::get ().physical ();
      depth = cuda_state::lane_get_syscall_call_depth (c.dev (), c.sm (),
                                                       c.wp (), c.ln ());
    }

  cv_set_uint32_var ("cuda_syscall_call_depth", depth);
}

static void
cv_update_api_failures_vars (void)
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
cv_update_present_kernels_vars (void)
{
  uint32_t num_kernels;
  struct value *kernel_array, *blocks_array;

  num_kernels = cuda_convenience_get_present_blocks_kernels (&kernel_array,
                                                             &blocks_array);

  cv_set_uint32_var ("cuda_num_present_kernels", num_kernels);
  set_internalvar (lookup_internalvar ("cuda_present_kernel_ids"),
                   kernel_array);
  set_internalvar (lookup_internalvar ("cuda_present_block_idxs"),
                   blocks_array);
}

static void
cv_update_total_kernels_var (void)
{
  cv_set_uint32_var ("cuda_num_total_kernels",
                     cuda_state::get_num_present_kernels ());
}

struct cv_variable_group
{
  const char *group_name;
  bool enabled;
  void (*update_func) (void);
  const char *group_desc;
};

static struct cv_variable_group cv_var_groups[] = {
  { "hw_vars", false, cv_update_hw_vars,
    "HW specific variables, like number of sms, lanes, warps, etc" },
  { "focus", false, cv_update_focus_vars, "CUDA focus specific variables" },
  { "lineno", false, cv_update_lineno_var,
    "Line number that matches $pc in focus" },
  { "call_depth", false, cv_update_call_depth_var, "Call depth" },
  { "syscall_depth", false, cv_update_syscall_depth_var, "Systemcall depth" },
  { "api_failures", false, cv_update_api_failures_vars,
    "API failure error number and function name" },
  { "present_kernels", false, cv_update_present_kernels_vars,
    "Kernel and their ids currently present on GPUs" },
  { "total_kernels", false, cv_update_total_kernels_var,
    "Total number of kernels on GPUs" },
  { NULL, false, NULL, NULL },
};

int
cuda_enable_convenience_variables_group (char *name, bool enable)
{
  struct cv_variable_group *grp;
  int rc = false;

  for (grp = cv_var_groups; grp->group_name; grp++)
    if (!name || strcasecmp (name, grp->group_name) == 0)
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

  for (grp = cv_var_groups; grp->group_name; grp++)
    if (grp->enabled)
      grp->update_func ();

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
                 _ ("Specifies which convenience variables groups are "
                    "available for debugging.\n"));
  ptr += rc;
  size -= rc;
  rc = snprintf (ptr, size,
                 _ ("Groups names are: \"all\",\"none\" or comma separate "
                    "list of the folowing:\n"));
  ptr += rc;
  size -= rc;
  for (grp = cv_var_groups; grp->group_name; grp++)
    {
      rc = snprintf (ptr, size, " %*s : %s\n", 20, grp->group_name,
                     grp->group_desc);
      if (rc <= 0)
        break;
      ptr += rc;
      size -= rc;
    }
}
