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

/*CUDA COMMAND PARSER
 *
 * Entry point for complex CUDA commands not handled by the usual GDB command
 * interface.
 *
 * At this point, 3 types of "commands" are supported by the CUDA parser, as
 * defined by the enum type command_t:
 *   - focus query (Ex: "cuda block lane device")
 *   - focus switch (Ex: "cuda block 2 thread 3")
 *   - conditional expressions for native conditional breakpoints, as either a
 *     AND-product or a OR-product of conditions. (Ex: threadIdx.x == 2 &&
 *     blockIdx.x == 3)
 *
 * The result of the parsing is passed as an array of request_t objects, where
 * each object represents a single request (or condition). A request applies to
 * a single type of coordinates (coord_type_t). It can be the device index, or
 * the blockIdx.x variable for instance. Given that coordinate, the request can
 * be of different types: same or exact. Same means that the value is the one in
 * the current focus. Exact means that the value is the one specified in the
 * value field of the request. Finally the cmp field indicates with comparison
 * operator to use for that coordinate, and the next field points to the next
 * request.
 *
 *   Example:
 *         type    = COORD_TYPE_BLOCK
 *         request = REQ_SAME, REQ_EXACT, REQ_NONE
 *         value   = { -1, 3, -1 }
 *         cmp     = CMP_LE
 *     translates to
 *         'block <= (n, 3)' where n is the current blockIdx.x
 *
 * With the switch and AND-product commands, ALL the requests must be met. With
 * the OR-product commands, ANY request must be met for the list of requests to
 * be valid.
 */

#ifndef _CUDA_PARSER_H
#define _CUDA_PARSER_H 1

#include "cudadebugger.h"
#include "defs.h"
#include "cuda-tdep.h"

typedef enum {
  CMP_NONE,       /* ignored */
  CMP_EQ,         /* == */
  CMP_NE,         /* != */
  CMP_LT,         /* < */
  CMP_GT,         /* > */
  CMP_LE,         /* <= */
  CMP_GE,         /* >= */
} compare_t;

typedef enum {
  CMD_NONE       = 0x001,    /* ignored */
  CMD_ERROR      = 0x002,    /* something went wrong */
  CMD_QUERY      = 0x004,    /* focus query command */
  CMD_SWITCH     = 0x008,    /* focus switch command */
  CMD_COND_AND   = 0x010,    /* AND-product of conditions */
  CMD_COND_OR    = 0x020,    /* OR-sum of hw conditions */
  CMD_FILTER     = 0x040,    /* filter for coordinates */
  CMD_FILTER_KERNEL= 0x080,    /* filter for kernel only*/
} command_t;

typedef enum {
  FILTER_TYPE_NONE        = 0x0000,    /* ignored */
  FILTER_TYPE_DEVICE      = 0x0001,    /* the device index */
  FILTER_TYPE_SM          = 0x0002,    /* the SM index */
  FILTER_TYPE_WARP        = 0x0004,    /* the warp index */
  FILTER_TYPE_LANE        = 0x0008,    /* the lane index */
  FILTER_TYPE_KERNEL      = 0x0010,    /* the kernel index */
  FILTER_TYPE_GRID        = 0x0020,    /* the grid index */
  FILTER_TYPE_BLOCK       = 0x0040,    /* the block index (blockIdx) */
  FILTER_TYPE_THREAD      = 0x0080,    /* the thread index (threadIdx) */
  FILTER_TYPE_BLOCKIDX_X  = 0x0100,    /* blockIdx.x */
  FILTER_TYPE_BLOCKIDX_Y  = 0x0200,    /* blockIdx.y */
  FILTER_TYPE_BLOCKIDX_Z  = 0x0400,    /* blockIdx.z */
  FILTER_TYPE_THREADIDX_X = 0x0800,    /* threadIdx.x */
  FILTER_TYPE_THREADIDX_Y = 0x1000,    /* threadIdx.y */
  FILTER_TYPE_THREADIDX_Z = 0x2000,    /* threadidx.z */
} filter_type_t;

typedef union {
    uint32_t          scalar;
    CuDim3            cudim3;
} request_value_t;

typedef struct request_st {
  filter_type_t       type;          /* the filter type: device, lane,... */
  request_value_t     value;         /* the request value (~0 means invalid/unspecified) */
  compare_t           cmp;           /* the comparison operator */
} request_t;

typedef struct {
  command_t  command;                /* the command type */
  uint32_t   num_requests;           /* the number of requests */
  uint32_t   max_requests;           /* the allocated number of requests */
  request_t *requests;               /* the pointer to the array of requests */
} cuda_parser_result_t;

extern command_t start_token;        /* Used internally by the parser */

void cuda_parser (const char * input, command_t command, cuda_parser_result_t **result,
                  cuda_coords_special_value_t dflt_value);
void cuda_parser_print (cuda_parser_result_t *result);

#endif
