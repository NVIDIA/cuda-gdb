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

#ifndef LIBCUDBGIPC_H
#define LIBCUDBGIPC_H 1

#include <stdarg.h>
#include <time.h>
#include <sys/time.h>

template<typename T>
inline static void
cudbgIpcReceive (char **ipc_buf, T *value)
{
  *value = *((T*)*ipc_buf);
  *ipc_buf += sizeof(T);
}

template<typename T>
inline static void
cudbgIpcReceive (char **ipc_buf,  T *values, size_t count)
{
  const size_t size = count * sizeof(*values);
  memcpy (values, *ipc_buf, size);
  *ipc_buf += size;
}

template<>
inline void
cudbgIpcReceive (char **ipc_buf,  void *buf, size_t size)
{
  memcpy (buf, *ipc_buf, size);
  *ipc_buf += size;
}

#define CUDBG_IPC_BEGIN_(cmd)                           \
do {                                                    \
  uint32_t _major = CUDBG_API_VERSION_MAJOR;            \
  uint32_t _command = cmd;                              \
  CUDBGResult _r;                                       \
  _r = cudbgipcAppend(&_major, sizeof(_major));         \
  if (_r != CUDBG_SUCCESS) return _r;                   \
  _r = cudbgipcAppend(&_command, sizeof(_command));     \
  if (_r != CUDBG_SUCCESS) return _r;                   \
} while (0)

#define CUDBG_IPC_APPEND_(d, s)                         \
do {                                                    \
  CUDBGResult _r = cudbgipcAppend(d, s);                \
  if (_r != CUDBG_SUCCESS) return _r;                   \
} while (0)

#define CUDBG_IPC_REQUEST_(d)                           \
do {                                                    \
  CUDBGResult _r = cudbgipcRequest(d, NULL);            \
  if (_r != CUDBG_SUCCESS) return _r;                   \
} while (0)

#define CUDBG_IPC_RECEIVE_(value, ipc_buf)              \
do {                                                    \
  cudbgIpcReceive(ipc_buf, value);                      \
} while (0)

#define CUDBG_IPC_RECEIVE_ARRAY_(value, size, ipc_buf)  \
do {                                                    \
  cudbgIpcReceive(ipc_buf, value, size);                \
} while (0)

/* This allows non-underscore macros to be redefined */
#define CUDBG_IPC_BEGIN CUDBG_IPC_BEGIN_
#define CUDBG_IPC_APPEND CUDBG_IPC_APPEND_
#define CUDBG_IPC_REQUEST CUDBG_IPC_REQUEST_
#define CUDBG_IPC_RECEIVE CUDBG_IPC_RECEIVE_
#define CUDBG_IPC_RECEIVE_ARRAY CUDBG_IPC_RECEIVE_ARRAY_

typedef struct CUDBGIPC_st {
    int from;
    int to;
    char name[256];
    int  fd;
    bool initialized;
    char *data;
    size_t dataSize;
} CUDBGIPC_t;

CUDBGResult cudbgipcAppend(const void *d, size_t size);
CUDBGResult cudbgipcRequest(void **d, size_t *size);
CUDBGResult cudbgipcCBWaitForData(void *data, size_t size);
CUDBGResult cudbgipcInitialize(void);
CUDBGResult cudbgipcFinalize(void);

/* Debugger API profiling collection typedefs/macros */
#define CUDBGIPC_API_STAT_MAX 256
typedef struct {
    const char *name;
    long times_called;
    double total_time;
    double min_time;
    double max_time;
} CUDBGIPCStat_t;

void cudbgipcStatsClear (void);
const CUDBGIPCStat_t *cudbgipcGetProfileStat(uint32_t);

#endif
