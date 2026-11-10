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

#ifndef CUDA_BUILTINS_H
#define CUDA_BUILTINS_H

#include <functional>

#include "symtab.h"

struct gdbarch;
struct type;

/* Identifiers for CUDA built-in variables (threadIdx, blockIdx, ...).  */

enum cuda_builtin_id
{
  CUDA_BUILTIN_THREADIDX = 0,
  CUDA_BUILTIN_BLOCKIDX,
  CUDA_BUILTIN_CLUSTERIDX,
  CUDA_BUILTIN_GRIDDIM,
  CUDA_BUILTIN_BLOCKDIM,
  CUDA_BUILTIN_CLUSTERDIM,
  CUDA_BUILTIN_WARPSIZE,
  CUDA_BUILTIN_COUNT
};

/* Type kind for a CUDA built-in.

     - threadIdx, blockIdx, clusterIdx are `uint3'.
     - gridDim, blockDim, clusterDim are `dim3'.
     - warpSize is `int'.

   uint3 and dim3 have an identical 12-byte layout (three
   `unsigned int' fields), but the user-visible type names differ -
   `ptype threadIdx' should report `uint3', not `dim3'.  Extend this
   enum when adding a built-in whose type is none of the above.  */

enum cuda_builtin_type_kind
{
  CUDA_BUILTIN_TYPE_UINT3,
  CUDA_BUILTIN_TYPE_DIM3,
  CUDA_BUILTIN_TYPE_INT,
};

/* Descriptor for one CUDA built-in.  */

struct cuda_builtin_def
{
  const char *name;
  enum cuda_builtin_type_kind type_kind;
};

extern const struct cuda_builtin_def cuda_builtin_defs[CUDA_BUILTIN_COUNT];

/* Number of aclass slots reserved in symtab.c's symbol_impl[] for
   CUDA code.  Bump this if you add a call to register_symbol_*_impl
   in CUDA-only code paths.  */

#define CUDA_EXTRA_SYMBOL_IMPLS 1

/* Look up a CUDA built-in by NAME.  Returns nullptr if NAME is not a
   CUDA built-in, DOMAIN lacks SEARCH_VAR_DOMAIN, or there is no CUDA
   gdbarch yet (no CUDA inferior).  */

extern struct symbol *cuda_lookup_builtin_symbol
  (const char *name, domain_search_flags domain);

/* Invoke F on every CUDA built-in symbol.  No-op if there is no CUDA
   gdbarch yet.  */

extern void cuda_for_each_builtin_symbol
  (std::function<void (struct symbol *)> f);

/* Return the CUDA dim3 type for GDBARCH.  Available for any gdbarch
   (host or device) so user code that declares `dim3' in plain host
   functions also resolves.  */

extern struct type *cuda_get_dim3_type (struct gdbarch *gdbarch);

/* Return the CUDA uint3 type for GDBARCH.  Layout is identical to
   dim3 (three `unsigned int' fields), only the type name differs.
   Exposed for the same reason as `cuda_get_dim3_type': so the
   language definitions can publish `uint3' as a known type and so
   helpers that materialise a `CuDim3' as a uint3 can find it.  */

extern struct type *cuda_get_uint3_type (struct gdbarch *gdbarch);

#endif /* CUDA_BUILTINS_H */
