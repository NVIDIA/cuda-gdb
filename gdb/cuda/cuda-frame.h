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

#ifndef _CUDA_FRAME_H
#define _CUDA_FRAME_H 1

#include "cuda-defs.h"
#include "frame-base.h"
#include "frame-unwind.h"
#include "frame.h"

extern const struct frame_unwind cuda_frame_unwind;
extern const struct frame_base cuda_frame_base;

const struct frame_unwind *cuda_frame_sniffer (frame_info_ptr next_frame);
const struct frame_base *
cuda_frame_base_sniffer (frame_info_ptr next_frame);

bool cuda_frame_p (frame_info_ptr next_frame);
bool cuda_frame_outermost_p (frame_info_ptr next_frame);

CORE_ADDR cuda_unwind_pc (struct gdbarch *gdbarch,
                          frame_info_ptr next_frame);

#endif
