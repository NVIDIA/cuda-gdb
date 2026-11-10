/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2026 NVIDIA Corporation
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

/* CUDA built-in variables (threadIdx, blockIdx, blockDim, etc.) are
   published as gdbarch-owned, non-objfile-owned symbols with
   LOC_COMPUTED storage class.  Their values are computed at
   evaluation time from the current CUDA focus.  They are read-only;
   the read callback demotes the value to not_lval after populating
   its contents so the binding does not propagate through convenience
   variables (see cuda_builtin_lval_read).  */

#include "defs.h"

#include "cuda-builtins.h"

#include "cuda-coords.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "frame.h"
#include "gdbarch.h"
#include "gdbtypes.h"
#include "language.h"
#include "objfiles.h"
#include "value.h"

/* The CUDA tdep's builtin_symbols array is sized by a mirror constant
   to avoid pulling this header into every cuda-tdep.h consumer.  */
static_assert (CUDA_GDBARCH_NUM_BUILTINS == CUDA_BUILTIN_COUNT,
	       "CUDA_GDBARCH_NUM_BUILTINS / CUDA_BUILTIN_COUNT mismatch");

/* Order must match enum cuda_builtin_id.  */

const struct cuda_builtin_def cuda_builtin_defs[CUDA_BUILTIN_COUNT] = {
  { "threadIdx",  CUDA_BUILTIN_TYPE_UINT3 },   /* CUDA_BUILTIN_THREADIDX  */
  { "blockIdx",   CUDA_BUILTIN_TYPE_UINT3 },   /* CUDA_BUILTIN_BLOCKIDX   */
  { "clusterIdx", CUDA_BUILTIN_TYPE_UINT3 },   /* CUDA_BUILTIN_CLUSTERIDX */
  { "gridDim",    CUDA_BUILTIN_TYPE_DIM3  },   /* CUDA_BUILTIN_GRIDDIM    */
  { "blockDim",   CUDA_BUILTIN_TYPE_DIM3  },   /* CUDA_BUILTIN_BLOCKDIM   */
  { "clusterDim", CUDA_BUILTIN_TYPE_DIM3  },   /* CUDA_BUILTIN_CLUSTERDIM */
  { "warpSize",   CUDA_BUILTIN_TYPE_INT   },   /* CUDA_BUILTIN_WARPSIZE   */
};

/* Sizes of the gdb-side built-in types.  CUDA_BUILTIN_MAX_SIZE bounds
   the stack buffer in cuda_builtin_lval_read; the static_assert pins
   the CUDA Debugger API's CuDim3 to the dim3/uint3 layout so a future
   ABI change trips the build instead of silently corrupting reads.  */

static constexpr size_t CUDA_BUILTIN_DIM3_SIZE = 3 * sizeof (uint32_t);
static constexpr size_t CUDA_BUILTIN_INT_SIZE = sizeof (int32_t);
static constexpr size_t CUDA_BUILTIN_MAX_SIZE
  = std::max (CUDA_BUILTIN_DIM3_SIZE, CUDA_BUILTIN_INT_SIZE);

static_assert (sizeof (CuDim3) == CUDA_BUILTIN_DIM3_SIZE,
	       "CuDim3 must match the dim3/uint3 layout cuda-gdb publishes");

/* aclass index registered for the CUDA built-in symbols.  */

static int cuda_builtin_var_aclass;

/* Per-gdbarch cache for CUDA language types.  Kept in a
   registry rather than the CUDA tdep so the same types are reachable
   from host gdbarches too.  */

struct cuda_builtin_types
{
  struct type *dim3_type = nullptr;
  struct type *uint3_type = nullptr;
};

static const registry<gdbarch>::key<cuda_builtin_types>
  cuda_builtin_types_handle;

static cuda_builtin_types *
get_cuda_builtin_types (struct gdbarch *gdbarch)
{
  cuda_builtin_types *result = cuda_builtin_types_handle.get (gdbarch);
  if (result == nullptr)
    result = cuda_builtin_types_handle.emplace (gdbarch);
  return result;
}

/* Build a struct named NAME with the dim3 layout
   `{ unsigned int x, y, z; }'.  */

static struct type *
cuda_make_dim3_like_type (struct gdbarch *gdbarch, const char *name)
{
  struct type *field_type = builtin_type (gdbarch)->builtin_unsigned_int;
  const ULONGEST field_size = field_type->length ();
  /* The struct length and per-field bitpos must agree; CUDA targets
     always have 4-byte unsigned int.  */
  gdb_assert (3 * field_size == CUDA_BUILTIN_DIM3_SIZE);

  struct type *t = type_allocator (gdbarch).new_type ();

  /* NAME must outlive T; callers pass string literals.  */
  t->set_name (name);
  t->set_length (CUDA_BUILTIN_DIM3_SIZE);
  t->set_code (TYPE_CODE_STRUCT);
  t->alloc_fields (3);

  auto init_field = [&] (int idx, const char *fname)
    {
      struct field &fp = t->field (idx);
      fp.set_name (fname);
      fp.set_type (field_type);
      fp.set_loc_bitpos (idx * field_size * 8);
    };

  init_field (0, "x");
  init_field (1, "y");
  init_field (2, "z");

  return t;
}

/* See cuda-builtins.h.  */

struct type *
cuda_get_dim3_type (struct gdbarch *gdbarch)
{
  cuda_builtin_types *types = get_cuda_builtin_types (gdbarch);
  if (types->dim3_type == nullptr)
    types->dim3_type = cuda_make_dim3_like_type (gdbarch, "dim3");
  return types->dim3_type;
}

/* See cuda-builtins.h.  */

struct type *
cuda_get_uint3_type (struct gdbarch *gdbarch)
{
  cuda_builtin_types *types = get_cuda_builtin_types (gdbarch);
  if (types->uint3_type == nullptr)
    types->uint3_type = cuda_make_dim3_like_type (gdbarch, "uint3");
  return types->uint3_type;
}

/* Throw error if the inferior is not stopped on a CUDA device thread.  */

static void
require_device_focus (const char *builtin_name)
{
  if (!cuda_current_focus::isDevice ())
    error (_("CUDA built-in '%s' is only available when the inferior "
	     "is stopped on a CUDA device thread."),
	   builtin_name);
}

/* Size of the full underlying value for BUILTIN_ID, regardless of
   which sub-field a caller may be reading.  */

static size_t
cuda_builtin_full_size (enum cuda_builtin_id id)
{
  switch (cuda_builtin_defs[id].type_kind)
    {
    case CUDA_BUILTIN_TYPE_UINT3:
    case CUDA_BUILTIN_TYPE_DIM3:
      return CUDA_BUILTIN_DIM3_SIZE;
    case CUDA_BUILTIN_TYPE_INT:
      return CUDA_BUILTIN_INT_SIZE;
    }
  internal_error (_("cuda_builtin_full_size: bad type_kind %d"),
		  (int) cuda_builtin_defs[id].type_kind);
}

/* Fetch BUILTIN_ID's current value into BUF.  BUF must have room for
   cuda_builtin_full_size (id) bytes - the full underlying value, not
   any sub-field a caller may ultimately want.  */

static void
fetch_cuda_builtin (enum cuda_builtin_id id, gdb_byte *buf)
{
  const char *name = cuda_builtin_defs[id].name;
  require_device_focus (name);
  const auto &cur = cuda_current_focus::get ();

  switch (id)
    {
    case CUDA_BUILTIN_THREADIDX:
      {
	CuDim3 v = cur.logical ().threadIdx ();
	memcpy (buf, &v, sizeof v);
	return;
      }
    case CUDA_BUILTIN_BLOCKIDX:
      {
	CuDim3 v = cur.logical ().blockIdx ();
	memcpy (buf, &v, sizeof v);
	return;
      }
    case CUDA_BUILTIN_CLUSTERIDX:
      {
	/* Use cuda_state directly: the focus path can honor CUDA_IGNORE
	   values which we don't want to expose to the user.  */
	CuDim3 v = cuda_state::warp_get_cluster_idx (cur.physical ().dev (),
						     cur.physical ().sm (),
						     cur.physical ().wp ());
	memcpy (buf, &v, sizeof v);
	return;
      }
    case CUDA_BUILTIN_GRIDDIM:
      {
	uint64_t kernel_id = cur.logical ().kernelId ();
	auto kernel = cuda_state::find_kernel_by_kernel_id (kernel_id);
	if (kernel == nullptr)
	  error (_("CUDA built-in '%s': no kernel found for kernel id %llu."),
		 cuda_builtin_defs[id].name, (unsigned long long) kernel_id);
	CuDim3 v = kernel->grid_dim ();
	memcpy (buf, &v, sizeof v);
	return;
      }
    case CUDA_BUILTIN_BLOCKDIM:
      {
	uint64_t kernel_id = cur.logical ().kernelId ();
	auto kernel = cuda_state::find_kernel_by_kernel_id (kernel_id);
	if (kernel == nullptr)
	  error (_("CUDA built-in '%s': no kernel found for kernel id %llu."),
		 cuda_builtin_defs[id].name, (unsigned long long) kernel_id);
	CuDim3 v = kernel->block_dim ();
	memcpy (buf, &v, sizeof v);
	return;
      }
    case CUDA_BUILTIN_CLUSTERDIM:
      {
	/* See clusterIdx above re: bypassing CUDA_IGNORE.  */
	CuDim3 v = cuda_state::warp_get_cluster_dim (cur.physical ().dev (),
						     cur.physical ().sm (),
						     cur.physical ().wp ());
	memcpy (buf, &v, sizeof v);
	return;
      }
    case CUDA_BUILTIN_WARPSIZE:
      {
	int32_t v
	  = (int32_t) cuda_state::device_get_num_lanes (cur.physical ().dev ());
	memcpy (buf, &v, sizeof v);
	return;
      }
    case CUDA_BUILTIN_COUNT:
      break;
    }
  internal_error (_("fetch_cuda_builtin: bad id %d"), (int) id);
}

/* The lval_computed closure is the symbol pointer.  */

static enum cuda_builtin_id
builtin_id_of (struct value *v)
{
  struct symbol *sym = (struct symbol *) v->computed_closure ();
  return (enum cuda_builtin_id) sym->value_longest ();
}

/* Read callback for CUDA built-in lval_computed values.  V may be the
   full built-in or a sub-field carrying our lval_funcs/closure via
   primitive_field/set_component_location; fetch the full built-in
   once and copy out only the [v->offset (), v->offset () +
   v->type ()->length ()) slice V represents.

   Then demote V to not_lval so the binding does not survive into
   convenience-variable storage.  value_of_internalvar preserves
   lval_computed across internalvar storage to support write-through
   (e.g. $_siginfo.si_code = ...); without demoting, re-assignments
   to a convenience var holding a built-in slice would be routed back
   through cuda_builtin_lval_write.

   FIXME: mutating m_lval inside a read callback is a layering smudge;
   the cleaner fix is upstream (a read-only flag on lval_funcs, or
   value_of_internalvar honoring funcs->write == nullptr).  */

static void
cuda_builtin_lval_read (struct value *v)
{
  enum cuda_builtin_id id = builtin_id_of (v);

  gdb_byte full[CUDA_BUILTIN_MAX_SIZE];
  fetch_cuda_builtin (id, full);

  const size_t full_size = cuda_builtin_full_size (id);
  const LONGEST offset = v->offset ();
  const LONGEST length = v->type ()->length ();

  if (offset < 0
      || length < 0
      || (ULONGEST) offset + (ULONGEST) length > full_size)
    internal_error (_("cuda_builtin_lval_read: out-of-range slice "
		      "[%s, %s) for built-in '%s' of size %zu"),
		    plongest (offset), plongest (offset + length),
		    cuda_builtin_defs[id].name, full_size);

  memcpy (v->contents_raw ().data (), full + offset, length);

  v->set_lval (not_lval);
}

/* CUDA built-ins are read-only - they're computed from focus state
   and have no backing store to write to.  */

static void
cuda_builtin_lval_write (struct value *toval, struct value * /*fromval*/)
{
  error (_("CUDA built-in '%s' is read-only."),
	 cuda_builtin_defs[builtin_id_of (toval)].name);
}

static const struct lval_funcs cuda_builtin_lval_funcs = {
  cuda_builtin_lval_read,
  cuda_builtin_lval_write,
  /* is_optimized_out */ nullptr,
  /* indirect       */ nullptr,
  /* coerce_ref     */ nullptr,
  /* check_synthetic_pointer */ nullptr,
  /* copy_closure   */ nullptr,
  /* free_closure   */ nullptr,
};

static struct value *
cuda_builtin_read_variable (struct symbol *sym, const frame_info_ptr &)
{
  return value::allocate_computed (sym->type (), &cuda_builtin_lval_funcs,
				   (void *) sym);
}

static struct value *
cuda_builtin_read_variable_at_entry (struct symbol *sym,
				     const frame_info_ptr &)
{
  error (_("CUDA built-in '%s' has no entry value."), sym->print_name ());
}

/* Built-ins are fetched from CUDA focus state, not from any frame.  */

static enum symbol_needs_kind
cuda_builtin_get_symbol_read_needs (struct symbol *)
{
  return SYMBOL_NEEDS_NONE;
}

static void
cuda_builtin_describe_location (struct symbol *sym, CORE_ADDR,
				struct ui_file *stream)
{
  gdb_printf (stream, _("a CUDA built-in variable"));
}

static void
cuda_builtin_tracepoint_var_ref (struct symbol *sym, struct agent_expr *,
				 struct axs_value *)
{
  error (_("CUDA built-in '%s' cannot be used in tracepoint expressions."),
	 sym->print_name ());
}

static void
cuda_builtin_generate_c_location (struct symbol *sym, string_file *,
				  struct gdbarch *,
				  std::vector<bool> &,
				  CORE_ADDR, const char *)
{
  error (_("CUDA built-in '%s' has no C location."),
	 sym->print_name ());
}

static const struct symbol_computed_ops cuda_builtin_var_funcs = {
  cuda_builtin_read_variable,
  cuda_builtin_read_variable_at_entry,
  cuda_builtin_get_symbol_read_needs,
  cuda_builtin_describe_location,
  /* location_has_loclist = */ 0,
  cuda_builtin_tracepoint_var_ref,
  cuda_builtin_generate_c_location,
};

/* Allocate one built-in symbol on GDBARCH's obstack.  */

static void
add_one_builtin (struct gdbarch *gdbarch, cuda_gdbarch_tdep *tdep,
		 enum cuda_builtin_id id, struct type *type)
{
  const struct cuda_builtin_def &def = cuda_builtin_defs[id];

  struct symbol *sym
    = new (gdbarch_obstack (gdbarch)) struct symbol ();

  /* The name is a string literal; no demangler hashing needed.  */
  sym->m_name = def.name;
  sym->set_language (language_c, nullptr);
  sym->owner.arch = gdbarch;
  sym->set_is_objfile_owned (0);
  sym->set_section_index (0);
  sym->set_type (type);
  sym->set_domain (VAR_DOMAIN);
  sym->set_aclass_index (cuda_builtin_var_aclass);

  /* Stash the id so the read_variable callback can dispatch on it.  */
  sym->set_value_longest ((LONGEST) id);

  tdep->builtin_symbols[id] = sym;
}

/* Populate the CUDA built-in symbol cache on GDBARCH.  Idempotent.

   Not advertised in cuda-builtins.h: the only legitimate caller is
   cuda_get_gdbarch() in cuda-tdep.c, which forward-declares this so
   the "go through the getter" invariant stays intact (the helpers
   cuda_lookup_builtin_symbol / cuda_for_each_builtin_symbol both
   call cuda_get_gdbarch(), and observe a populated cache by the
   time the gdbarch is returned).  The local prototype below mirrors
   that extern declaration and silences -Wmissing-declarations at the
   definition site without exposing the symbol in the public header.  */

void cuda_init_builtin_symbols (struct gdbarch *gdbarch);

void
cuda_init_builtin_symbols (struct gdbarch *gdbarch)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);

  if (tdep->builtin_symbols[0] != nullptr)
    return;

  struct type *uint3_type = cuda_get_uint3_type (gdbarch);
  struct type *dim3_type = cuda_get_dim3_type (gdbarch);
  struct type *int_type = builtin_type (gdbarch)->builtin_int;

  for (int i = 0; i < CUDA_BUILTIN_COUNT; ++i)
    {
      enum cuda_builtin_id id = (enum cuda_builtin_id) i;
      struct type *type;
      switch (cuda_builtin_defs[id].type_kind)
	{
	case CUDA_BUILTIN_TYPE_UINT3:
	  type = uint3_type;
	  break;
	case CUDA_BUILTIN_TYPE_DIM3:
	  type = dim3_type;
	  break;
	case CUDA_BUILTIN_TYPE_INT:
	  type = int_type;
	  break;
	default:
	  internal_error (_("cuda_init_builtin_symbols: bad type_kind %d"),
			  (int) cuda_builtin_defs[id].type_kind);
	}
      add_one_builtin (gdbarch, tdep, id, type);
    }
}

/* See cuda-builtins.h.  */

struct symbol *
cuda_lookup_builtin_symbol (const char *name, domain_search_flags domain)
{
  if (name == nullptr)
    return nullptr;
  if ((domain & SEARCH_VAR_DOMAIN) == 0)
    return nullptr;

  struct gdbarch *gdbarch = cuda_get_gdbarch ();
  if (gdbarch == nullptr || !cuda_is_cuda_gdbarch (gdbarch))
    return nullptr;

  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  for (int i = 0; i < CUDA_BUILTIN_COUNT; ++i)
    if (strcmp (name, cuda_builtin_defs[i].name) == 0)
      return tdep->builtin_symbols[i];

  return nullptr;
}

/* See cuda-builtins.h.  */

void
cuda_for_each_builtin_symbol (std::function<void (struct symbol *)> f)
{
  struct gdbarch *gdbarch = cuda_get_gdbarch ();
  if (gdbarch == nullptr || !cuda_is_cuda_gdbarch (gdbarch))
    return;

  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  for (int i = 0; i < CUDA_BUILTIN_COUNT; ++i)
    f (tdep->builtin_symbols[i]);
}

void _initialize_cuda_builtins ();

void
_initialize_cuda_builtins ()
{
  cuda_builtin_var_aclass
    = register_symbol_computed_impl (LOC_COMPUTED, &cuda_builtin_var_funcs);
}
