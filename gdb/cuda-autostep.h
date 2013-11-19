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

#ifndef _CUDA_AUTOSTEP_H_
#define _CUDA_AUTOSTEP_H_ 1
#include <stdbool.h>

void cuda_handle_autostep (void);

bool cuda_get_autostep_pending (void);
void cuda_set_autostep_pending (bool pending);

bool cuda_get_autostep_stepping (void);
void cuda_set_autostep_stepping (bool stepping);

bool cuda_autostep_stop (void);

#endif
