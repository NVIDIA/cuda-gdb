/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2013-2026 NVIDIA Corporation
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

#include <cstdarg>
#include <deque>
#include <string>

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
extern bool cuda_debug_general;
extern bool cuda_debug_libcudbg;
extern bool cuda_debug_notifications;
extern bool cuda_notify_youngest;
extern bool cuda_driver_logs;
extern bool cuda_printf_flushing;

extern ptid_t cuda_last_ptid;
extern struct target_waitstatus cuda_last_ws;

/* Queue of pending trace messages drained by `qnv.QUERY_TRACE_MESSAGE`.
Producers should call `cuda_enqueue_trace_message` instead of pushing to the
deque directly */
extern std::deque<std::string> cuda_trace_messages;

/* Format `"<prefix>" + vsprintf(fmt, ap)` and enqueue it as a single
   trace message. Truncates if the message is too long for the transport
   protocol. Caller retains ownership of `ap` and is responsible for its
   `va_end`.  */
void cuda_enqueue_trace_message (const char *prefix, const char *fmt,
				 va_list ap) ATTRIBUTE_PRINTF (2, 0);

struct cuda_sym
{
  const char *name;
  CORE_ADDR addr;
}; 

/*-------------------------------- Prototypes ----------------------------------*/
#if defined(__QNXHOST__)
extern void ATTRIBUTE_NORETURN captured_main (int argc, char *argv[]);
#endif

void cuda_gdb_setup (void);
void cuda_cleanup (void);
bool cuda_inferior_in_debug_mode (void);
bool cuda_initialize_target ();

/* CUDA cleanup state tracking for deferred connection teardown */
extern bool cuda_cleanup_completed;
extern bool cuda_exit_requested;

void cuda_set_driver_logging (bool enable);

CORE_ADDR cuda_get_symbol_address_from_cache (const char *name);

int  cuda_get_debugger_api (void);

void cuda_look_up_symbols (void);

int  cuda_get_symbol_cache_size (void);

bool cuda_options_statistics_collection_enabled (void);

void cuda_trace (const char *fmt, ...);

bool cuda_options_launch_blocking (void);

bool cuda_options_debug_general (void);

bool cuda_options_debug_libcudbg (void);

bool cuda_options_debug_notifications (void);

bool cuda_options_notify_youngest (void);

bool cuda_options_driver_logs (void);

bool cuda_options_printf_flushing (void);

bool cuda_check_pending_sigint (ptid_t ptid);

bool cuda_platform_supports_tid (void);
int  cuda_gdb_get_tid_or_pid (ptid_t ptid);

/* Session Management */
int         cuda_gdb_session_create (void);
void        cuda_gdb_session_destroy (void);
const char *cuda_gdb_session_get_dir (void);
uint32_t    cuda_gdb_session_get_id (void);

#endif
