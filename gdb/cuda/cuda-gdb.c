/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2023 NVIDIA Corporation
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

#include "defs.h"
#include "cuda-gdb.h"
#include "cuda-version.h"

void
cuda_print_message_nvidia_version (struct ui_file *stream)
{
  fprintf_unfiltered (stream,
                      "NVIDIA (R) CUDA Debugger\n"
                      "CUDA Toolkit %d.%d release\n"
                      "Portions Copyright (C) 2007-%s NVIDIA Corporation\n",
		      cuda_major_version (), cuda_minor_version (),
		      cuda_current_year ());
}

