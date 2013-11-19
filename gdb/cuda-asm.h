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


#ifndef _CUDA_ASM_H
#define _CUDA_ASM_H 1

#include "cuda-defs.h"

disasm_cache_t disasm_cache_create           (void);
void           disasm_cache_destroy          (disasm_cache_t disasm_cache);
void           disasm_cache_flush            (disasm_cache_t disasm_cache);
const char *   disasm_cache_find_instruction (disasm_cache_t disasm_cache,
                                              uint64_t pc, uint32_t
                                              *inst_size);

#endif
