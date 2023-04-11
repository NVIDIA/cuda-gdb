/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2023 NVIDIA Corporation
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

#ifndef _REMOTE_CUDA_H
#define _REMOTE_CUDA_H 1

#include "target.h"
#include "remote.h"

extern void cuda_sigtrap_set_silent (void);
extern void cuda_sigtrap_restore_settings (void);
extern void cuda_signal_set_silent (int, struct cuda_signal_info_st *);
extern void cuda_signal_restore_settings (int, struct cuda_signal_info_st *);
extern void cuda_print_memcheck_message (void);
extern void cuda_print_assert_message (void);

struct extended_remote_target;

void cuda_remote_kill (remote_target *ops);
void cuda_remote_mourn_inferior (remote_target *ops);
void cuda_remote_detach (remote_target *ops, inferior *arg0, int arg1);
void cuda_extended_remote_detach (extended_remote_target *ops, inferior *arg0, int arg1);
void cuda_remote_resume (remote_target *ops,
			 ptid_t arg0,
			 int TARGET_DEBUG_PRINTER (target_debug_print_step) arg1,
			 enum gdb_signal arg2);
ptid_t cuda_remote_wait (remote_target *ops,
			 ptid_t ptid, struct target_waitstatus *ws, int target_options);
void cuda_remote_store_registers (remote_target *ops,
				  struct regcache *regcache,
				  int regno);
int cuda_remote_insert_breakpoint (remote_target *ops, struct gdbarch *gdbarch,
				   struct bp_target_info *bp_tgt);
int cuda_remote_remove_breakpoint (remote_target *ops, struct gdbarch *gdbarch,
				   struct bp_target_info *bp_tgt,
				   enum remove_bp_reason reason);
enum target_xfer_status
cuda_remote_xfer_partial (remote_target *ops,
                          enum target_object object, const char *annex,
                          gdb_byte *readbuf, const gdb_byte *writebuf,
                          ULONGEST offset, ULONGEST len, ULONGEST *xfered_len);
struct gdbarch *cuda_remote_thread_architecture (remote_target *ops, ptid_t ptid);

void cuda_remote_prepare_to_store (remote_target *ops, struct regcache *regcache);

void set_cuda_remote_flag (bool connected);
#ifdef __QNXTARGET__
void cuda_version_handshake (const char *version_string);
void cuda_finalize_remote_target (void);
#else /* __QNXTARGET__ */
void cuda_remote_version_handshake (remote_target *remote,
				    const struct protocol_feature *feature,
                                    enum packet_support support,
                                    const char *version_string);
#endif /* __QNXTARGET__ */
#endif
