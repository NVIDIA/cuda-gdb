/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2013 NVIDIA Corporation
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

#include "../cuda-utils.h"

#include "cudadebugger.h"
#include "server.h"

#define CUDA_SYM(SYM)   \
  {             \
    _STRING_(SYM),       \
    0           \
  }

/*------------------------------ Global Variables ------------------------------*/

extern bool cuda_debugging_enabled;
extern bool cuda_initialized;
extern bool all_cuda_symbols_looked_up;
CUDBGResult api_initialize_res;
CUDBGResult api_finalize_res;
CUDBGResult get_debugger_api_res;
CUDBGResult set_callback_api_res;

extern bool cuda_launch_blocking;
extern bool cuda_memcheck;
extern bool cuda_software_preemption;
extern bool cuda_debug_general;
extern bool cuda_debug_libcudbg;
extern bool cuda_debug_notifications;
extern bool cuda_notify_youngest;

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
void cuda_cleanup (void);
bool cuda_inferior_in_debug_mode (void);
bool cuda_initialize_target ();

CORE_ADDR cuda_get_symbol_address_from_cache (char*);

int  cuda_get_debugger_api (void);

int  cuda_get_symbol_cache_size (void);

void cuda_trace (char *fmt, ...);

bool cuda_options_memcheck ();

bool cuda_options_launch_blocking ();

bool cuda_options_software_preemption ();

bool cuda_options_debug_general ();

bool cuda_options_debug_libcudbg ();

bool cuda_options_debug_notifications ();

bool cuda_options_notify_youngest ();

bool cuda_check_pending_sigint (ptid_t ptid);

/* Linux vs. Mac OS X */
bool cuda_platform_supports_tid (void);
int  cuda_gdb_get_tid (ptid_t ptid);

/* Session Management */
int         cuda_gdb_session_create (void);
void        cuda_gdb_session_destroy (void);
const char *cuda_gdb_session_get_dir (void);
uint32_t    cuda_gdb_session_get_id (void);
#endif
