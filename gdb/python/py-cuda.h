/* Python interface for CUDA debugging

 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2024-2025 NVIDIA Corporation
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

#ifndef _PY_CUDA_H
#define _PY_CUDA_H 1

#ifdef HAVE_PYTHON

#include "python-internal.h"

PyMODINIT_FUNC gdbpy_cuda_init(void);

#endif

#endif
