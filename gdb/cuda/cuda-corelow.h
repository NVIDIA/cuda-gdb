/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2015-2025 NVIDIA Corporation
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

#ifndef _CUDA_CORELOW_H
#define _CUDA_CORELOW_H 1

extern void cuda_core_fetch_registers (struct regcache *regcache, int regno);
extern void cuda_core_load_api (const char *filename);
extern void cuda_core_free (void);
extern void cuda_core_initialize_events_exceptions (void);

#endif
