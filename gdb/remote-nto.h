/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2017-2025 NVIDIA Corporation
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

/* CUDA added interfaces to the nto target */

#ifndef REMOTE_NTO_H
#define REMOTE_NTO_H
#ifdef NVIDIA_CUDA_GDB

#include "remote.h"
#include "nto-share/dsmsgs.h"

/* Return true if TARGET is a qnx target, otherwise,
   return false.  */
extern bool is_qnx_target (process_stratum_target *target);

/* Send BUF to the current remote target.  If BUF points to an empty
   string, either zero length, or the first character is the null
   character, then an error is thrown.  If the current target is not a
   remote target then an error is thrown.

   Calls CALLBACKS->sending() just before the packet is sent to the remote
   target, and calls CALLBACKS->received() with the reply once this is
   received from the remote target.  */

extern void send_qnx_packet (gdb::array_view<const char> &buf,
			     send_remote_packet_callbacks *callbacks);

#endif
#endif
