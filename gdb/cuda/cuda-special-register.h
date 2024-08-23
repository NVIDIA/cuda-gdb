/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2024 NVIDIA Corporation
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

#ifndef _CUDA_SPECIAL_REGISTER_H
#define _CUDA_SPECIAL_REGISTER_H 1

#include "defs.h"
#include "cuda-defs.h"
#include "frame.h"
#include "cuda-regmap.h"

/*
 * Set of utility routines working on top of the cuda-regmap.h routines.
 * Handles everything related to what cuda-gdb called "the special register"
 * (everything that does not fit in a single SASS register).
 *
 * Note that the regmap object must always be passed as an input argument.
 */

/* returns true if regmap contains a special register */
bool cuda_special_register_p (regmap_t regmap);

/* read/write from/to special register to/from buffer */
void cuda_special_register_read  (regmap_t regmap,
                                  gdb_byte *buf);
void cuda_special_register_write (regmap_t regmap,
                                  const gdb_byte *buf);

/* read/write from/to special_register to/from value */
void cuda_special_register_to_value (regmap_t regmap,
                                     frame_info_ptr frame,
                                     gdb_byte *to);
void cuda_value_to_special_register (regmap_t regmap,
                                     frame_info_ptr frame,
                                     const gdb_byte *from);

/* print special register into a string */
void cuda_special_register_name (regmap_t regmap, char *buf, const int size);

#endif

