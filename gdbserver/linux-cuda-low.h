/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2017-2024 NVIDIA Corporation
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

#ifndef CUDA_LINUX_LOW_H
#define CUDA_LINUX_LOW_H

#include "cuda-tdep-server.h"
#include "cuda/cuda-utils.h"
#include "cuda/cuda-notifications.h"

template<class BaseTarget>
class cuda_linux_process_target : public BaseTarget
{
public:

  cuda_linux_process_target()
  {
    /* Check the required CUDA debugger files are present */
    if (cuda_get_debugger_api ()) 
      {
	warning ("CUDA support disabled: could not obtain the CUDA debugger API\n");
	return;
      }

    cuda_debugging_enabled = true;
  }

  void mourn (process_info *proc) override
  {
    if (cuda_debugging_enabled)
      {
	cuda_cleanup ();
      }
    BaseTarget::mourn (proc);
  }

  ptid_t wait (ptid_t ptid, target_waitstatus *status,
	       target_wait_flags options) override
  {
    cuda_last_ptid = BaseTarget::wait (ptid, status, options);
    cuda_last_ws = *status;
    return cuda_last_ptid;
  }  

  void look_up_symbols () override
  {
    BaseTarget::look_up_symbols ();
    if (!cuda_debugging_enabled || cuda_syms_looked_up)
      return;
    cuda_look_up_symbols ();
  }

  void unexpected_stop (void) override
  {
    for_each_thread ([] (thread_info *thread)
    {
      struct lwp_info *lwp = get_thread_lwp (thread);
      linux_stop_lwp (lwp);
      /* Pretend we expected a stop, so that it gets reported to the client. */
      thread->last_resume_kind = resume_stop;
    });
  }
};

#endif
