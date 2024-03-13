/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2013-2023 NVIDIA Corporation
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

#ifndef _CUDA_TDEP_SERVER_H
#define _CUDA_TDEP_SERVER_H 1

#ifndef GDBSERVER
#define GDBSERVER
#endif

#include "server.h"
#include "cuda/cuda-utils.h"
#include "cudadebugger.h"

#define CUDA_SYM(SYM)   \
  {             \
    _STRING_(SYM),       \
    0           \
  }

/*------------------------------ Global Variables ------------------------------*/

extern bool cuda_debugging_enabled;
extern bool cuda_initialized;
extern CUDBGAPI cudbgAPI;
extern CUDBGResult api_initialize_res;
extern CUDBGResult api_finalize_res;
extern CUDBGResult get_debugger_api_res;
extern CUDBGResult set_callback_api_res;

extern struct cuda_sym cuda_symbol_list[];
extern bool cuda_syms_looked_up;
extern bool cuda_launch_blocking;
extern bool cuda_software_preemption;
extern bool cuda_debug_general;
extern bool cuda_debug_libcudbg;
extern bool cuda_debug_notifications;
extern bool cuda_notify_youngest;
extern unsigned cuda_stop_signal;

extern ptid_t cuda_last_ptid;
extern struct target_waitstatus cuda_last_ws;
extern bool cuda_debugging_enabled;

struct cuda_trace_msg
{
  char buf [1024];
  struct cuda_trace_msg *next;
};

extern struct cuda_trace_msg *cuda_first_trace_msg;

extern struct cuda_trace_msg *cuda_last_trace_msg;

struct cuda_sym
{
  const char *name;
  CORE_ADDR addr;
}; 

/*-------------------------------- Prototypes ----------------------------------*/
#if defined(__QNXHOST__)
extern void ATTRIBUTE_NORETURN captured_main (int argc, char *argv[]);
#endif

void cuda_cleanup (void);
bool cuda_inferior_in_debug_mode (void);
bool cuda_initialize_target ();

CORE_ADDR cuda_get_symbol_address_from_cache (const char *name);

int  cuda_get_debugger_api (void);

void cuda_look_up_symbols (void);

int  cuda_get_symbol_cache_size (void);

bool cuda_options_statistics_collection_enabled (void);

void cuda_trace (const char *fmt, ...);

bool cuda_options_launch_blocking (void);

bool cuda_options_software_preemption (void);

bool cuda_options_debug_general (void);

bool cuda_options_debug_libcudbg (void);

bool cuda_options_debug_notifications (void);

bool cuda_options_notify_youngest (void);

bool cuda_check_pending_sigint (ptid_t ptid);

/* Linux vs. Mac OS X */
bool cuda_platform_supports_tid (void);
int  cuda_gdb_get_tid_or_pid (ptid_t ptid);

/* Session Management */
int         cuda_gdb_session_create (void);
void        cuda_gdb_session_destroy (void);
const char *cuda_gdb_session_get_dir (void);
uint32_t    cuda_gdb_session_get_id (void);

/* SIGTRAP vs SIGURG option */
unsigned cuda_options_stop_signal (void);
#endif
