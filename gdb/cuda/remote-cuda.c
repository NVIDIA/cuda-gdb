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

#include "defs.h"

#include <signal.h>
#include <stdbool.h>

#include "command.h"
#include "frame.h"
#include "gdbarch.h"
#include "gdbthread.h"
#include "inferior.h"
#include "regcache.h"
#include "remote.h"
#ifdef __QNXTARGET__
#include "remote-nto.h"
#endif

#include "cuda-convvars.h"
#include "cuda-events.h"
#include "cuda-exceptions.h"
#include "cuda-kernel.h"
#include "cuda-notifications.h"
#include "cuda-options.h"
#include "cuda-packet-manager.h"
#include "cuda-state.h"
#include "cuda-utils.h"
#include "libcudbgipc.h"
#include "remote-cuda.h"
#include "top.h"

#ifdef __QNXTARGET__
static bool symbols_are_set = false;
#endif

bool
cuda_remote_initialize_target ()
{
  CUDBGResult get_debugger_api_res;
  CUDBGResult set_callback_api_res;
  CUDBGResult api_initialize_res;
  uint32_t num_sms = 0;
  uint32_t num_warps = 0;
  uint32_t num_lanes = 0;
  uint32_t num_registers = 0;
  uint32_t dev_id = 0;
  bool driver_is_compatible;
  char *dev_type;
  char *sm_type;
#ifdef __QNXTARGET__
  struct cmd_list_element *alias = NULL;
  struct cmd_list_element *prefix_cmd = NULL;
  struct cmd_list_element *handle_cmd = NULL;
#endif

  if (cuda_initialized)
    return true;

#ifdef __QNXTARGET__
  /* If shared objects haven't been loaded yet, we won't find any symbol. */
  bool found = false;

  for (so_list *so : current_program_space->solibs ())
    {
      while (so)
	{
	  if (strstr (so->so_name, "libcuda.so") != NULL)
	    {
	      found = true;
	      break;
	    }
	  so = so->next;
	}

      if (found)
	break;
    }

  /* libcuda isn't loaded yet, we won't find any symbol. */
  if (!found)
    return false;

  /* Ignore signals that we use for notification passing on QNX.
     See cuda-notifications.c for details. */
  if (!lookup_cmd_composition ("handle", &alias, &prefix_cmd, &handle_cmd))
    {
      error (_ ("Failed to lookup the `handle` command."));
    }
  cmd_func (handle_cmd, "SIGEMT nostop noprint nopass", 0);
  cmd_func (handle_cmd, "SIGILL nostop noprint nopass", 0);

  /* Send the target the symbols it needs */
  if (!symbols_are_set)
    {
      /* First assume cuda-gdbserver is backwards compatible */
      cuda_remote_set_symbols (true,
			       &symbols_are_set);
      if (!symbols_are_set)
	{
	  warning (
	      _ ("Cannot access libcuda symbols, please set sysroot or "
		 "solib-search-path. CUDA debugging will not be available."));
	  /* If the above failed, assume it's incompatible and only send
	     the core symbols */
	  cuda_remote_set_symbols (false,
				   &symbols_are_set);
	  if (!symbols_are_set)
	    {
	      return false;
	    }
	}
    }
#endif

  /* Ask cuda-gdbserver to initialize. */
  uint32_t debugapi_major = 0;
  uint32_t debugapi_minor = 0;
  uint32_t debugapi_revision = 0;
  cuda_remote_initialize (&get_debugger_api_res, &set_callback_api_res,
			  &api_initialize_res, &cuda_initialized,
			  &cuda_debugging_enabled, &driver_is_compatible,
			  &debugapi_major, &debugapi_minor,
			  &debugapi_revision);

  cuda_debugapi::print_get_api_error (get_debugger_api_res);
  cuda_debugapi::handle_initialization_error (api_initialize_res);
  cuda_debugapi::handle_set_callback_api_error (set_callback_api_res);
  cuda_debugapi::set_api_version (debugapi_major, debugapi_minor,
				  debugapi_revision);

  if (!driver_is_compatible)
    {
      target_kill ();
      error (_ ("CUDA application cannot be debugged. The CUDA driver is not "
		"compatible."));
    }

  //FIXME: WAR for DTCGDB-3482
  //--------------------------------------------------------
  CUDBGAPI api = nullptr;
  CUDBGResult res = cudbgGetAPI (CUDBG_API_VERSION_MAJOR,
                                 CUDBG_API_VERSION_MINOR,
                                 CUDBG_API_VERSION_REVISION,
                                 &api);
  if (res == CUDBG_SUCCESS)
    cuda_debugapi::set_api (api);
  else
    cuda_debugapi::print_get_api_error (res);
  //--------------------------------------------------------

  if (!cuda_initialized)
    return false;

  cudbgipcInitialize ();
  cuda_state::initialize ();
  for (dev_id = 0; dev_id < cuda_state::get_num_devices (); dev_id++)
    {
      cuda_remote_query_device_spec (dev_id, &num_sms, &num_warps,
				     &num_lanes, &num_registers, &dev_type,
				     &sm_type);
      cuda_state::set_device_spec (dev_id, num_sms, num_warps, num_lanes,
				   num_registers, dev_type, sm_type);
    }
  cuda_remote_set_option ();
  cuda_update_report_driver_api_error_flags ();

  return true;
}

#ifdef __QNXTARGET__
void
cuda_finalize_remote_target (void)
{
  symbols_are_set = false;
}
#endif