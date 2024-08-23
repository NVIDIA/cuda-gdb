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
 *
 * This is the master file where we update copyright information for
 * printing at runtime.
 *
 */

#ifdef GDBSERVER
#include "server.h"
#else
#include "defs.h"
#endif

#include "cuda-version.h"

/* Globals */

static const char nvidia_copyright_currrent_year[] = "2024";

const int cuda_major_version (void)
{
  return CUDA_VERSION / 1000;
}

const int cuda_minor_version (void)
{
  return (CUDA_VERSION % 1000) / 10;
}

const char *cuda_current_year (void)
{
  return nvidia_copyright_currrent_year;
}

