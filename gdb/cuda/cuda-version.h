/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2021-2024 NVIDIA Corporation
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

#ifndef _CUDA_VERSION_H
#define _CUDA_VERSION_H 1

#ifndef CUDA_VERSION
#define CUDA_VERSION 12080
#endif

const int cuda_major_version (void);
const int cuda_minor_version (void);
const char *cuda_current_year (void);

#endif

