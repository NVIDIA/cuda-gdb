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

#ifndef _CUDA_EVENTS_H
#define _CUDA_EVENTS_H 1

#include "cudadebugger.h"

typedef enum {
    CUDA_EVENT_INVALID,
    CUDA_EVENT_SYNC,
    CUDA_EVENT_ASYNC,
    CUDA_EVENT_MAX,
} cuda_event_kind_t;

void cuda_process_events (CUDBGEvent *event, cuda_event_kind_t kind);
void cuda_process_event  (CUDBGEvent *event);
void cuda_event_post_process (bool reset_bpt);

#endif
