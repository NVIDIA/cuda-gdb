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

#ifndef _CUDA_LINUX_NAT_H
#define _CUDA_LINUX_NAT_H 1

#include "defs.h"
#include "bfd.h"
#include "elf-bfd.h"
#include "cudadebugger.h"

extern void cuda_sigtrap_restore_settings (void);
extern void cuda_nat_add_target (struct target_ops *t);
extern void cuda_sigtrap_set_silent (void);

#endif

