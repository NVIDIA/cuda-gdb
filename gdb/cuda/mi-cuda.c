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

#include "defs.h"
#include <string.h>

#include "mi/mi-cmds.h"
#include "cuda-commands.h"

/* helper function to concatenate all the arguments into a single string with an
   extra space in-between each element, which will be the filter string for the
   info commands. */
static char*
concatenate_string (char **argv, int argc)
{
  int allocated = 0;
  int copied = 0;
  int size = 0;
  int i;
  char* result = NULL;

  for (i = 0; i < argc; ++i)
    {
      size = strlen (argv[i]);
      if (copied + size + 1 > allocated)
        {
          allocated += std::min (128, size + 1);
          result = (char *) xrealloc (result, allocated);
        }
      memcpy (result + copied, argv[i], size);
      *(result + copied + size) = ' ';
      copied += size + 1;
    }

  if (!result)
    {
      result = (char *) xrealloc (result, 1);
      *result = 0;
    }
  else
    *(result + copied - 1) = 0;

  return result;
}

void
mi_cmd_cuda_info_devices (const char *command, char **argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_devices_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_sms (const char *command, char **argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_sms_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_warps (const char *command, char **argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_warps_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_lanes (const char *command, char **argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_lanes_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_kernels (const char *command, char **argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_kernels_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_contexts (const char *command, char **argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_contexts_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_blocks (const char *command, char **argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_blocks_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_threads (const char *command, char **argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_threads_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_launch_trace (const char *command, char **argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_launch_trace_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_launch_children (const char *command, char **argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_launch_children_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_focus_query (const char *command, char **argv, int argc)
{
  char *query_string = concatenate_string (argv, argc);

  cuda_command_query (query_string);

  xfree (query_string);
}

void
mi_cmd_cuda_focus_switch (const char *command, char **argv, int argc)
{
  char *switch_string = concatenate_string (argv, argc);

  cuda_command_switch (switch_string);

  xfree (switch_string);
}
