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

#ifndef _CUDA_AUTOSTEP_H_
#define _CUDA_AUTOSTEP_H_ 1
#include "cuda-iterator.h"
#include <stdbool.h>

bool cuda_get_autostep_pending (void);
void cuda_set_autostep_pending (bool pending);

/* Autostep helper functions and data structures.  */

/* Initialize the autostep state based on PC.  */
int cuda_initialize_autostep (CORE_ADDR pc);

/* Update autostep state based on the current state and PC.  */
int cuda_update_autostep_state (CORE_ADDR pc);

/* Stop autostepping and cleanup autostepping state.  */
int cuda_cleanup_autostep_state (void);

/* Print the exception data for the currently active autostep breakpoint.  */
void cuda_autostep_print_exception (void);

/* Device-specific autostep data.  */
struct device_astep_t
{
  /* Coordinates from the current iteration.  */
  cuda_coords cur_coords;
  /* Warp iterator data.  */
  cuda_iterator<cuda_iterator_type::threads, select_bkpt | select_valid> iter;
  /* Warp iterator current pos */
  cuda_iterator<cuda_iterator_type::threads,
                select_bkpt | select_valid>::iterator iter_pos;
  /* The current lane.  */
  int cur_ln;
  /* Lines information.  */
  int lines;
  /* Numberof steps.  */
  int nsteps;
  /* Size of an instruction.  */
  uint32_t inst_size;
};

/* Host-specific autostep data.  */
struct host_astep_t
{
  /* Placeholder for host-specific autostep data.  */
  bool placeholder;
};

struct autostep_state
{
  /* 1 if stepping instructions or 0 if stepping lines.  */
  int insn_stepping;
  /* The PC we started autostepping at.  */
  CORE_ADDR start_pc;
  /* The final PC we need to reach when we are done with autostepping.  */
  CORE_ADDR end_pc;
  /* The current PC we're stopped at.  */
  CORE_ADDR cur_pc;
  /* The line we started stepping at.  */
  struct symtab_and_line start_sal;
  /* The final line we need to reach when we are done autostepping.  */
  struct symtab_and_line stop_sal;
  /* The current sal.  */
  struct symtab_and_line cur_sal;
  /* Number of instructions we should step.  */
  int insns_to_step;
  /* Number of lines we should step.  */
  int lines_to_step;
  /* Number of lines we actually stepped so far.  */
  int lines_stepped;
  /* Number of instructions we actually stepped so far.  */
  int insns_stepped;
  /* Number of remaining steps/instructions/lines.  */
  int remaining;
  /* Additional host/device autostep data.  */
  device_astep_t device;
  autostep_state () = default;
  ~autostep_state () = default;
};

#endif
