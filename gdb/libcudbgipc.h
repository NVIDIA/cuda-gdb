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

#ifndef LIBCUDBGIPC_H
#define LIBCUDBGIPC_H 1

#include <stdarg.h>
#include <time.h>
#include <sys/time.h>

#define CUDBG_IPC_APPEND(d, s)                     \
do {                                               \
  CUDBGResult r = cudbgipcAppend(d, s);            \
  if (r != CUDBG_SUCCESS) return r;                \
} while (0)

#define CUDBG_IPC_REQUEST(d)                       \
do {                                               \
  CUDBGResult r = cudbgipcRequest(d, NULL);        \
  if (r != CUDBG_SUCCESS) return r;                \
} while (0)

#define CUDBG_IPC_BEGIN(cmd)                       \
do {                                               \
  uint32_t major = CUDBG_API_VERSION_MAJOR;        \
  uint32_t command = cmd;                          \
  CUDBGResult r;                                   \
  r = cudbgipcAppend(&major, sizeof(major));       \
  if (r != CUDBG_SUCCESS) return r;                \
  r = cudbgipcAppend(&command, sizeof(command));   \
  if (r != CUDBG_SUCCESS) return r;                \
} while (0)

typedef struct CUDBGIPC_st {
    int from;
    int to;
    char name[256];
    int  fd;
    bool initialized;
    char *data;
    uint64_t dataSize;
} CUDBGIPC_t;

CUDBGResult cudbgipcAppend(void *d, uint32_t size);
CUDBGResult cudbgipcRequest(void **d, size_t *size);
CUDBGResult cudbgipcCBWaitForData(void *data, uint32_t size);
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

void cudbgipcStatsCollect (uint32_t, const char *, struct timespec *, struct timespec *);
const CUDBGIPCStat_t *cudbgipcGetProfileStat(uint32_t);

extern struct timespec cudbgipc_profile_start;

extern bool cuda_options_statistics_collection_enabled (void);

#define CUDBG_IPC_PROFILE_START()                       \
if (cuda_options_statistics_collection_enabled())       \
 {                                                      \
  struct timeval tv;                                    \
  gettimeofday (&tv, NULL);                             \
  cudbgipc_profile_start.tv_sec = tv.tv_sec;            \
  cudbgipc_profile_start.tv_nsec = tv.tv_usec*1000;     \
} while (0)

#define CUDBG_IPC_PROFILE_END(id,name)                                     \
if (cuda_options_statistics_collection_enabled())                          \
{                                                                          \
  struct timeval tv;                                                       \
  struct timespec profile_stop;                                            \
  gettimeofday (&tv, NULL);                                                \
  profile_stop.tv_sec = tv.tv_sec;                                         \
  profile_stop.tv_nsec = tv.tv_usec*1000;                                  \
  cudbgipcStatsCollect (id, name, &cudbgipc_profile_start, &profile_stop); \
} while (0)

#endif
