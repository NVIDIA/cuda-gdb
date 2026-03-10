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

#ifndef CUDA_NTO_PROTOCOL_H
#define CUDA_NTO_PROTOCOL_H

#include "gdbsupport/common-defs.h"
#include "gdbsupport/array-view.h"
#include "remote-nto.h"

#include <unistd.h>

/* Each byte can be escaped, plus two frame markers (~),
   plus a checksum that can also possibly be escaped */
#define MAX_PACKET_SIZE (DS_DATA_MAX_SIZE * 2 + 4)

/* Copied from remote-nto.c */
typedef union
{
  unsigned char buf[DS_DATA_MAX_SIZE];
  DSMsg_union_t pkt;
  TSMsg_text_t text;
} DScomm_t;

enum endpoint {GDB, SRV, PDB};

void gdbserver_debug_print(enum endpoint from, enum endpoint to, unsigned char *buf);
int get_raw_pdebug_packet_size (unsigned char *src, int max_size);
int unpack_pdebug_packet (DScomm_t *packet, unsigned char *src);
int pack_pdebug_packet (unsigned char *dest, const DScomm_t *packet, int length);
int pack_cuda_packet (unsigned char *dest, char *src, int length);

void putpkt_pdebug (DScomm_t *packet, int length);
void getpkt_pdebug (DScomm_t *packet);

int qnx_write_inferior_memory (CORE_ADDR memaddr, const unsigned char *myaddr, int len);

#endif
