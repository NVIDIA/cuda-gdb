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

#ifndef _CUDA_COMMANDS_H
#define _CUDA_COMMANDS_H 1

void run_info_cuda_command (void (*command)(const char *), const char *arg);

/*'info cuda' commands */
void info_cuda_devices_command         (const char *arg);
void info_cuda_sms_command             (const char *arg);
void info_cuda_warps_command           (const char *arg);
void info_cuda_lanes_command           (const char *arg);
void info_cuda_kernels_command         (const char *arg);
void info_cuda_contexts_command        (const char *arg);
void info_cuda_blocks_command          (const char *arg);
void info_cuda_threads_command         (const char *arg);
void info_cuda_launch_trace_command    (const char *arg);
void info_cuda_launch_children_command (const char *arg);
void info_cuda_managed_command         (const char *arg);
void info_cuda_line_command            (const char *arg);

/*cuda focus commands */
void cuda_command_switch (const char *switch_string);
void cuda_command_query  (const char *query_string);

#endif
