/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2013 NVIDIA Corporation
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

#ifndef _REMOTE_CUDA_H
#define _REMOTE_CUDA_H 1

#include "target.h"
#include "remote.h"

extern void cuda_sigtrap_set_silent (void);
extern void cuda_sigtrap_restore_settings (void);
extern void cuda_print_memcheck_message (void);
extern void cuda_print_assert_message (void);

void cuda_remote_add_target (struct target_ops *t);
void set_cuda_remote_flag (bool connected);
void cuda_remote_version_handshake (const struct protocol_feature *feature,
                                    enum packet_support support,
                                    const char *version_string);
#endif
