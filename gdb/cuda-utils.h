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

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H 1

#include "cuda-defs.h"

/* Utility functions for cuda-gdb */
#define CUDA_GDB_TMP_BUF_SIZE 1024
#define __S__(s)                #s
#define _STRING_(s)       __S__(s)

/* Initialize everything */
void cuda_utils_initialize (void);

/* Get the gdb temporary directory path */
extern const char* cuda_gdb_tmpdir_getdir (void);

/* Create a directory */
int cuda_gdb_dir_create (const char *dir_name, uint32_t permissions,
                         bool override_umask, bool *dir_exists);

/* Set file ownership to that of debugged process */
bool cuda_gdb_chown_to_pid_uid (int pid, const char *path);

/* Clean up files in a directory */
void cuda_gdb_dir_cleanup_files (char* dirpath);

/* Bypass non-fatal signals to the application during controlled application resume
 * Perform cleanup of the returned value to revert all changes in debugger behaviour
 * introduced by this routine
 */
struct cleanup *cuda_gdb_bypass_signals (void);

/* cuda debugging clock, incremented at each resume/wait cycle */
cuda_clock_t cuda_clock (void);
void         cuda_clock_increment (void);

void cuda_gdb_tmpdir_cleanup_self (void *unused);
void cuda_gdb_record_remove_all (void *unused);

/* CUDA register cache */
struct frame_info;
struct value;
struct type;
struct symbol;
void cuda_ptx_cache_store_register (struct frame_info *, int dwarf_regnum, struct value *);
struct value *cuda_ptx_cache_get_register (struct frame_info *, int dwarf_regnum, struct type *);
void cuda_ptx_cache_refresh (void);
void cuda_ptx_cache_local_vars_iterator (const char *, struct symbol *, void *);

/* CUDA - managed variables */
struct minimal_symbol;
void cuda_set_host_address_resident_on_gpu (bool);
bool cuda_is_host_address_resident_on_gpu (void);
bool cuda_managed_msymbol_p (struct minimal_symbol *);
void cuda_managed_memory_clean_regions (void);
bool cuda_is_value_managed_pointer (struct value *value);
bool cuda_is_uvm_used (void);
void cuda_set_uvm_used (bool);
#ifndef GDBSERVER
static inline void cuda_write_bool (CORE_ADDR addr, bool val)
{
  target_write_memory (addr, (char *)&val, 1);
}
bool cuda_managed_address_p (CORE_ADDR addr);
void cuda_managed_memory_add_region (CORE_ADDR begin, CORE_ADDR end);
#endif
#endif
