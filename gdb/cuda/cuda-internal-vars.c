/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2025 NVIDIA Corporation
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

#include "cuda-api.h"
#include "cuda-coord-set.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "gdbsupport/gdb_string_view.h"
#include "gdbtypes.h"
#include "language.h"

#include <type_traits>

static inline void
check_cluster ()
{
  if (!cuda_current_focus::get ().logical ().hasCluster ())
    error (_ ("Current kernel has no cluster."));
}

static inline void
to_be_implemented ()
{
  error (_ ("This variable isn't supported yet."));
}

/* Physical coordinates */

static inline uint32_t
cuda_internal_var_laneid ()
{
  return cuda_current_focus::get ().physical ().ln ();
}

static inline uint32_t
cuda_internal_var_warpid ()
{
  return cuda_current_focus::get ().physical ().wp ();
}

static inline uint32_t
cuda_internal_var_nwarpid ()
{
  return cuda_state::device_get_num_warps (
      cuda_current_focus::get ().physical ().dev ());
}

static inline uint32_t
cuda_internal_var_smid ()
{
  return cuda_current_focus::get ().physical ().sm ();
}

static inline uint32_t
cuda_internal_var_nsmid ()
{
  return cuda_state::device_get_num_sms (
      cuda_current_focus::get ().physical ().dev ());
}

static inline uint32_t
cuda_internal_var_devid ()
{
  return cuda_current_focus::get ().physical ().dev ();
}

static inline uint32_t
cuda_internal_var_ndevid ()
{
  return cuda_state::get_num_devices ();
}

/* Logical coordinates */

static inline CuDim3
cuda_internal_var_tid ()
{
  return cuda_current_focus::get ().logical ().threadIdx ();
}

static inline CuDim3
cuda_internal_var_ntid ()
{
  return cuda_current_focus::get ().logical ().kernel ()->block_dim ();
}

static inline CuDim3
cuda_internal_var_ctaid ()
{
  return cuda_current_focus::get ().logical ().blockIdx ();
}

static inline CuDim3
cuda_internal_var_nctaid ()
{
  return cuda_current_focus::get ().logical ().kernel ()->grid_dim ();
}

static inline uint64_t
cuda_internal_var_gridid ()
{
  return cuda_current_focus::get ().logical ().gridId ();
}

static inline uint64_t
cuda_internal_var_kernelid ()
{
  return cuda_current_focus::get ().logical ().kernelId ();
}

static inline uint32_t
cuda_internal_var_nkernelid ()
{
  cuda_coord_set<cuda_coord_set_type::kernels, select_valid> kernels_itr (
      cuda_coords::wild ());
  return kernels_itr.size ();
}

/* Clusters */

static inline CuDim3
cuda_internal_var_cluster_ctaid ()
{
  check_cluster ();

  return cuda_current_focus::get ().logical ().clusterCtaIdx ();
}

static inline CuDim3
cuda_internal_var_cluster_nctaid ()
{
  check_cluster ();

  return cuda_current_focus::get ().logical ().clusterDim ();
}

static inline uint32_t
cuda_internal_var_cluster_ctarank ()
{
  check_cluster ();

  return cuda_current_focus::get ().logical ().clusterCtaRank ();
}

static inline uint32_t
cuda_internal_var_cluster_nctarank ()
{
  check_cluster ();

  const CuDim3 &dim = cuda_current_focus::get ().logical ().clusterDim ();
  return dim.x * dim.y * dim.z;
}

static inline CuDim3
cuda_internal_var_clusterid ()
{
  check_cluster ();

  return cuda_current_focus::get ().logical ().clusterIdx ();
}

static inline CuDim3
cuda_internal_var_nclusterid ()
{
  check_cluster ();

  const CuDim3 &grid_dim
      = cuda_current_focus::get ().logical ().kernel ()->grid_dim ();
  const CuDim3 &cluster_dim
      = cuda_current_focus::get ().logical ().clusterDim ();
  return { grid_dim.x / cluster_dim.x, grid_dim.y / cluster_dim.y,
	   grid_dim.z / cluster_dim.z };
}

static inline bool
cuda_internal_var_is_explicit_cluster ()
{
  return cuda_current_focus::get ().logical ().hasCluster ();
}

/* Masks */
static inline uint32_t
cuda_internal_var_lanemask_eq ()
{
  uint32_t mask = 1;
  mask <<= cuda_current_focus::get ().physical ().ln ();
  return mask;
}

static inline uint32_t
cuda_internal_var_lanemask_le ()
{
  uint32_t mask = cuda_internal_var_lanemask_eq ();
  mask |= (mask - 1);
  return mask;
}

static inline uint32_t
cuda_internal_var_lanemask_lt ()
{
  uint32_t mask = cuda_internal_var_lanemask_eq ();
  mask -= 1;
  return mask;
}

static inline uint32_t
cuda_internal_var_lanemask_ge ()
{
  return ~cuda_internal_var_lanemask_lt ();
}

static inline uint32_t
cuda_internal_var_lanemask_gt ()
{
  return ~cuda_internal_var_lanemask_le ();
}

/* Memory */

static inline uint32_t
cuda_internal_var_total_smem_size ()
{
  to_be_implemented (); // TODO: DTUD-3676

  return 0;
}

static inline uint32_t
cuda_internal_var_aggr_smem_size ()
{
  const cuda_coords_physical &c = cuda_current_focus::get ().physical ();
  return cuda_state::warp_shared_mem_size (c.dev (), c.sm (), c.wp ());
}

static inline uint32_t
cuda_internal_var_dynamic_smem_size ()
{
  to_be_implemented (); // TODO: DTUD-3676

  return 0;
}

/* Errors */

static struct value *
cuda_internal_var_api_failure_func_name ()
{
  if (!cuda_get_last_driver_api_error_func_name_size ())
    error (_ ("No error is available."));

  CORE_ADDR error_func_name_core_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_ADDR));

  if (!error_func_name_core_addr)
    error (
	_ ("Cannot retrieve the last driver API error function name addr."));

  uint64_t error_func_name_addr;
  if (target_read_memory (error_func_name_core_addr,
			  (gdb_byte *)&error_func_name_addr,
			  sizeof (uint64_t)))
    error (_ ("Failed to read target memory."));

  return value_from_pointer (
      builtin_type (cuda_get_gdbarch ())->builtin_data_ptr,
      error_func_name_addr);
}

/* Convenience variables */

static inline uint32_t
cuda_internal_var_thread_lineno ()
{
  const auto &c = cuda_current_focus::get ().physical ();
  uint64_t pc = cuda_state::lane_get_pc (c.dev (), c.sm (), c.wp (), c.ln ());
  return find_pc_line (pc, 0).line;
}

static inline bool
cuda_internal_var_thread_active ()
{
  const auto &c = cuda_current_focus::get ().physical ();
  return cuda_state::lane_valid (c.dev (), c.sm (), c.wp (), c.ln ())
	 && cuda_state::lane_active (c.dev (), c.sm (), c.wp (), c.ln ());
}

static inline uint32_t
cuda_internal_var_focus_block_x ()
{
  return cuda_current_focus::get ().logical ().blockIdx ().x;
}

static inline uint32_t
cuda_internal_var_focus_block_y ()
{
  return cuda_current_focus::get ().logical ().blockIdx ().y;
}

static inline uint32_t
cuda_internal_var_focus_block_z ()
{
  return cuda_current_focus::get ().logical ().blockIdx ().z;
}

static inline uint32_t
cuda_internal_var_focus_thread_x ()
{
  return cuda_current_focus::get ().logical ().threadIdx ().x;
}

static inline uint32_t
cuda_internal_var_focus_thread_y ()
{
  return cuda_current_focus::get ().logical ().threadIdx ().y;
}

static inline uint32_t
cuda_internal_var_focus_thread_z ()
{
  return cuda_current_focus::get ().logical ().threadIdx ().z;
}

static inline uint64_t
cuda_internal_var_latest_kernelid ()
{
  return cuda_state::last_launched_kernel_id ();
}

static inline uint32_t
cuda_internal_var_call_depth ()
{
  const auto &c = cuda_current_focus::get ().physical ();
  return cuda_state::lane_get_call_depth (c.dev (), c.sm (), c.wp (), c.ln ());
}

static inline uint32_t
cuda_internal_var_num_regs_allocated ()
{
  const auto &c = cuda_current_focus::get ().physical ();
  return cuda_state::warp_registers_allocated (c.dev (), c.sm (), c.wp ());
}

static inline uint32_t
cuda_internal_var_num_uregs ()
{
  const auto &c = cuda_current_focus::get ().physical ();
  return cuda_state::device_get_num_uregisters (c.dev ());
}

static inline std::vector<uint64_t>
cuda_internal_present_kernel_ids ()
{
  std::vector<uint64_t> res;
  cuda_coord_set<cuda_coord_set_type::kernels, select_valid> kernels_itr (
      cuda_coords::wild ());

  for (const auto &kern : kernels_itr)
    res.push_back (kern.logical ().kernelId ());

  return res;
}

static inline std::vector<std::vector<CuDim3>>
cuda_internal_present_block_idxs ()
{
  std::vector<std::vector<CuDim3>> res;

  /*
   * gdb value arrays require elements to be the same size, so we ensure
   * each kernel has the same number of blocks and set the blocks outside
   * the range to CUDA_INVALID.
   */
  uint32_t max_blocks = 0;
  /* This represents the array of iterators for every block in every kernel */
  std::vector<cuda_coord_set<cuda_coord_set_type::blocks, select_valid>>
      blocks_itr;
  cuda_coord_set<cuda_coord_set_type::kernels, select_valid> kernels_itr (
      cuda_coords::wild ());

  for (const auto &kern : kernels_itr)
    {
      /* Create an iterator over every block in the kernel */
      cuda_coords filter (
	  kern.physical ().dev (), CUDA_WILDCARD, CUDA_WILDCARD, CUDA_WILDCARD,
	  kern.logical ().kernelId (), CUDA_WILDCARD, CUDA_WILDCARD_DIM,
	  CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM);

      cuda_coord_set<cuda_coord_set_type::blocks, select_valid> blocks (
	  filter);
      /* Update max blocks */
      max_blocks = std::max ((uint32_t)blocks.size (), max_blocks);
      /* Save the blocks iterator */
      blocks_itr.push_back (std::move (blocks));
    }

  /* Iterate over every block iterator for every kernel*/
  for (auto &block_itr : blocks_itr)
    {
      std::vector<CuDim3> kernel_blocks;
      kernel_blocks.resize (max_blocks, CUDA_INVALID_DIM);

      int i = 0;
      for (const auto &block : block_itr)
	kernel_blocks[i++] = block.logical ().blockIdx ();

      res.push_back (kernel_blocks);
    }

  return res;
}

template <typename T>
static
    typename std::enable_if<std::is_integral<T>::value, struct value *>::type
    convert_to_gdb_value (T arg)
{
  struct gdbarch *arch = cuda_get_gdbarch ();
  if (std::is_signed<T>::value)
    {
      return value_from_longest (builtin_type (arch)->builtin_long_long, arg);
    }
  else
    {
      return value_from_ulongest (
	  builtin_type (arch)->builtin_unsigned_long_long, arg);
    }
}

template <typename T>
static struct value *
convert_to_gdb_value (std::vector<T> &arg)
{
  std::vector<struct value *> vals;

  for (auto &elt : arg)
    vals.push_back (convert_to_gdb_value (elt));

  return value_array (0, vals);
}

/* Passthrough for handlers returning their own struct value. */
[[maybe_unused]] static struct value *
convert_to_gdb_value (struct value *arg)
{
  return arg;
}

static struct value *
convert_to_gdb_value (CuDim3 arg)
{
  static type *dim3_type = lookup_struct ("dim3", nullptr);

  return value_from_contents (dim3_type,
			      reinterpret_cast<const gdb_byte *> (&arg));
}

static struct value *
convert_to_gdb_value (bool arg)
{
  static type *bool_type = lookup_typename (current_language, "bool", 0, 0);

  return value_from_contents (bool_type,
			      reinterpret_cast<const gdb_byte *> (&arg));
}

[[maybe_unused]] static struct value *
convert_to_gdb_value (const std::string &str)
{
  return value_cstring (str.c_str (), str.length (),
			builtin_type (cuda_get_gdbarch ())->builtin_char);
}

using cuda_internal_var_handler_fn = std::function<struct value *()>;
struct cuda_internal_var
{
  const char *name;
  cuda_internal_var_handler_fn handler;
  bool is_host;
  std::vector<gdb::string_view> aliases;
};

template <typename T>
static cuda_internal_var_handler_fn
wrap (const T &f)
{
  return [&] () {
    auto res = f ();
    return convert_to_gdb_value (res);
  };
}

/* === CUDA variable definition ===

  {name, wrap(<handler>), host, <list of aliases>}

  Where
  `name` is how the variable will be used in expressions as in `$name`
  `handler` is the function to call to evaluate the variable.
      wrap() will attempt to convert the return value of the handler into a gdb
  `value`. If you're introducing an unsupported type, add an overload to
  `convert_to_gdb_value`. `host` indicates if the variable should be accessible
  in host code. Default requires CUDA focus + initialized API. Optional value,
  default to false. `aliases` is a list of alternative names for this handler.
      This is used for backward compatibility with old convenience variables
  and future uses should be avoided. Optional value, default to empty.
*/
static cuda_internal_var cuda_internal_variables[] = {
  /* Physical coordinates */
  { "laneid", wrap (cuda_internal_var_laneid), false, { "cuda_focus_lane" } },
  { "warpid", wrap (cuda_internal_var_warpid), false, { "cuda_focus_warp" } },
  { "nwarpid", wrap (cuda_internal_var_nwarpid) },
  { "smid", wrap (cuda_internal_var_smid), false, { "cuda_focus_sm" } },
  { "nsmid", wrap (cuda_internal_var_nsmid), false, { "cuda_num_sms" } },
  { "devid", wrap (cuda_internal_var_devid), false, { "cuda_focus_device" } },
  { "ndevid", wrap (cuda_internal_var_ndevid) },

  /* Logical coordinates */
  { "tid", wrap (cuda_internal_var_tid) },
  { "ntid", wrap (cuda_internal_var_ntid) },
  { "ctaid", wrap (cuda_internal_var_ctaid) },
  { "nctaid", wrap (cuda_internal_var_nctaid) },
  { "gridid", wrap (cuda_internal_var_gridid), false, { "cuda_focus_grid" } },
  { "kernelid",
    wrap (cuda_internal_var_kernelid),
    false,
    { "cuda_focus_kernel_id" } },
  { "nkernelid",
    wrap (cuda_internal_var_nkernelid),
    true,
    { "cuda_num_total_kernels", "cuda_num_present_kernels" } },

  /* Clusters */
  { "cluster_ctaid", wrap (cuda_internal_var_cluster_ctaid) },
  { "cluster_nctaid", wrap (cuda_internal_var_cluster_nctaid) },
  { "cluster_ctarank", wrap (cuda_internal_var_cluster_ctarank) },
  { "cluster_nctarank", wrap (cuda_internal_var_cluster_nctarank) },
  { "clusterid", wrap (cuda_internal_var_clusterid) },
  { "nclusterid", wrap (cuda_internal_var_nclusterid) },
  { "is_explicit_cluster", wrap (cuda_internal_var_is_explicit_cluster) },

  /* Masks */
  { "lanemask_eq", wrap (cuda_internal_var_lanemask_eq) },
  { "lanemask_le", wrap (cuda_internal_var_lanemask_le) },
  { "lanemask_lt", wrap (cuda_internal_var_lanemask_lt) },
  { "lanemask_ge", wrap (cuda_internal_var_lanemask_ge) },
  { "lanemask_gt", wrap (cuda_internal_var_lanemask_gt) },

  /* Memory */
  { "total_smem_size", wrap (cuda_internal_var_total_smem_size) },
  { "aggr_smem_size",
    wrap (cuda_internal_var_aggr_smem_size),
    false,
    { "cuda_shared_memory_size" } },
  { "dynamic_smem_size", wrap (cuda_internal_var_dynamic_smem_size) },

  /* Errors */
  { "cuda_api_failure_func_name",
    wrap (cuda_internal_var_api_failure_func_name), true },
  { "cuda_api_failure_return_code", wrap (cuda_get_last_driver_api_error_code),
    true },

  /* Convenience variables */
  { "cuda_thread_active", wrap (cuda_internal_var_thread_active) },
  { "cuda_thread_lineno", wrap (cuda_internal_var_thread_lineno) },
  { "cuda_focus_block_x", wrap (cuda_internal_var_focus_block_x) },
  { "cuda_focus_block_y", wrap (cuda_internal_var_focus_block_y) },
  { "cuda_focus_block_z", wrap (cuda_internal_var_focus_block_z) },
  { "cuda_focus_thread_x", wrap (cuda_internal_var_focus_thread_x) },
  { "cuda_focus_thread_y", wrap (cuda_internal_var_focus_thread_y) },
  { "cuda_focus_thread_z", wrap (cuda_internal_var_focus_thread_z) },
  { "cuda_latest_launched_kernel_id", wrap (cuda_internal_var_latest_kernelid),
    true },
  { "cuda_call_depth", wrap (cuda_internal_var_call_depth) },
  { "cuda_num_registers_allocated",
    wrap (cuda_internal_var_num_regs_allocated) },
  { "cuda_num_uregisters", wrap (cuda_internal_var_num_uregs) },
  { "cuda_present_kernel_ids", wrap (cuda_internal_present_kernel_ids) },
  { "cuda_present_block_idxs", wrap (cuda_internal_present_block_idxs) },
};

static struct value *
cuda_internal_var_wrapper_host (struct gdbarch *arch, struct internalvar *var,
				void *data)
{
  cuda_internal_var_handler_fn target = *(cuda_internal_var_handler_fn *)data;

  return target ();
}

static struct value *
cuda_internal_var_wrapper (struct gdbarch *arch, struct internalvar *var,
			   void *data)
{
  if (!cuda_is_cuda_gdbarch (arch))
    error (_ ("CUDA focus is required to print this value."));

  if (!cuda_debugapi::api_state_initialized ())
    error (_ ("CUDA is not initialized."));

  return cuda_internal_var_wrapper_host (arch, var, data);
}

static const internalvar_funcs cuda_internal_var_funcs
    = { cuda_internal_var_wrapper, nullptr };

static const internalvar_funcs cuda_internal_var_funcs_host
    = { cuda_internal_var_wrapper_host, nullptr };

void _initialize_cuda_internal_vars ();

void
_initialize_cuda_internal_vars ()
{
  for (auto &v : cuda_internal_variables)
    {
      const internalvar_funcs *wrapper = v.is_host
					     ? &cuda_internal_var_funcs_host
					     : &cuda_internal_var_funcs;
      create_internalvar_type_lazy (v.name, wrapper, &v.handler);

      if (v.aliases.empty ())
	continue;

      for (gdb::string_view a : v.aliases)
	create_internalvar_type_lazy (a.data (), wrapper, &v.handler);
    }
}
