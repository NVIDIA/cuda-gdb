/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2013 NVIDIA Corporation
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

#ifndef _CUDA_TEXTURES_H
#define _CUDA_TEXTURES_H 1

#include "gdbtypes.h"
#include "symtab.h"
#include "objfiles.h"
#include "cudadebugger.h"


#define TEXTURE_DIM_MAX    4
#define TEXTURE_TYPE_NAME     "__texture_type__"
#define IS_TEXTURE_TYPE(type) ((type) && \
                               TYPE_CODE (type) == TYPE_CODE_TYPEDEF && \
                               !strcmp (TYPE_NAME (type), TEXTURE_TYPE_NAME))


struct value* cuda_texture_value_ptradd (struct type *type,
                                         struct value *base,
                                         LONGEST ofst);
struct value* cuda_texture_value_ind (struct type *type,
                                      struct value* base);
void cuda_texture_set_address_class (struct type *type);
bool cuda_texture_is_tex_ptr (struct type *type);
void cuda_texture_create_kernel_mapping (struct symbol *symbol,
                                         struct objfile *objfile);
void cuda_texture_read_tex_contents (CORE_ADDR addr, gdb_byte *buf);
void cuda_texture_dereference_tex_contents (CORE_ADDR addr, uint32_t *tex_id,
                                            uint32_t *dim, uint32_t **coords,
                                            bool *is_bindless);
void cuda_cleanup_tex_maps (void);

#endif
