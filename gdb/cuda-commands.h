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

#ifndef _CUDA_COMMANDS_H
#define _CUDA_COMMANDS_H 1

void cuda_commands_initialize (void);
void run_info_cuda_command (void (*command)(char *), char *arg);

/*'info cuda' commands */
void info_cuda_devices_command         (char *arg);
void info_cuda_sms_command             (char *arg);
void info_cuda_warps_command           (char *arg);
void info_cuda_lanes_command           (char *arg);
void info_cuda_kernels_command         (char *arg);
void info_cuda_contexts_command        (char *arg);
void info_cuda_blocks_command          (char *arg);
void info_cuda_threads_command         (char *arg);
void info_cuda_launch_trace_command    (char *arg);
void info_cuda_launch_children_command (char *arg);

/*cuda focus commands */
void cuda_command_switch (char *switch_string);
void cuda_command_query  (char *query_string);

#endif

