/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2022-2024 NVIDIA Corporation
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

#ifndef _CUDA_FUNCTION_H
#define _CUDA_FUNCTION_H 1

#include "cudadebugger.h"
#include "cuda-defs.h"

#if CUDBG_API_VERSION_REVISION >= 132
void cuda_apply_function_load_updates (char *buffer, CUDBGLoadedFunctionInfo *info, uint32_t count);
#endif

#endif
