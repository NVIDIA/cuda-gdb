/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2014 NVIDIA Corporation
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

#ifndef _CUDA_ITERATOR_H
#define _CUDA_ITERATOR_H 1

#include "cuda-defs.h"
#include "cuda-coords.h"

typedef enum {
  CUDA_ITERATOR_TYPE_DEVICES = 0x001,
  CUDA_ITERATOR_TYPE_SMS     = 0x002,
  CUDA_ITERATOR_TYPE_WARPS   = 0x004,
  CUDA_ITERATOR_TYPE_LANES   = 0x008,
  CUDA_ITERATOR_TYPE_KERNELS = 0x010,
  CUDA_ITERATOR_TYPE_BLOCKS  = 0x020,
  CUDA_ITERATOR_TYPE_THREADS = 0x040,
} cuda_iterator_type;


void          cuda_iterator_destroy (cuda_iterator itr);
cuda_iterator cuda_iterator_create  (cuda_iterator_type type,
                                     cuda_coords_t *filter,
                                     cuda_select_t select_mask);

cuda_iterator cuda_iterator_start   (cuda_iterator itr);
cuda_iterator cuda_iterator_next    (cuda_iterator itr);
bool          cuda_iterator_end     (cuda_iterator itr);

cuda_coords_t cuda_iterator_get_current (cuda_iterator itr);
uint32_t      cuda_iterator_get_size    (cuda_iterator itr);

#endif
