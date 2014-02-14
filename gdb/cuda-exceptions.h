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

#ifndef _CUDA_EXCEPTIONS_H
#define _CUDA_EXCEPTIONS_H 1

#include "cuda-defs.h"

extern cuda_exception_t cuda_exception;

bool cuda_exception_hit_p (cuda_exception_t exception);
void cuda_exception_reset (cuda_exception_t exception);

bool          cuda_exception_is_valid       (cuda_exception_t exception);
bool          cuda_exception_is_recoverable (cuda_exception_t exception);
uint32_t      cuda_exception_get_value      (cuda_exception_t exception);
cuda_coords_t cuda_exception_get_coords     (cuda_exception_t exception);

void          cuda_exception_print_message  (cuda_exception_t exception);
const char   *cuda_exception_type_to_name (CUDBGException_t);

#endif
