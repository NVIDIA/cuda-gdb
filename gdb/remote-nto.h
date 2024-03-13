/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2017-2023 NVIDIA Corporation
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

#ifndef REMOTE_NTO_H
#define REMOTE_NTO_H

#include "nto-share/dsmsgs.h"
#include "gdbsupport/byte-vector.h"
#include <stdbool.h>

typedef union
{
  unsigned char buf[DS_DATA_MAX_SIZE];
  DSMsg_union_t pkt;
  TSMsg_text_t text;
} DScomm_t;


/* Compatibility functions for the CUDA remote I/O */
struct remote_target;

int
qnx_getpkt (remote_target *ops, gdb::char_vector* buf, int forever);
int
qnx_getpkt_sane (gdb::char_vector* buf, int forever);
int
qnx_putpkt (remote_target *ops, const char *buf);
int
qnx_putpkt_binary (const char *buf, int cnt);

#endif
