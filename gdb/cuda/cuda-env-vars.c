/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2025 NVIDIA Corporation
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

#include "cuda-env-vars.h"
#include "ui-out.h"

void
cuda_check_env_vars (const char *ignored, int from_tty)
{
  const char *injectionPath;

  injectionPath = getenv ("CUDBG_INJECTION_PATH");
  if (injectionPath)
    {
      struct stat sb;
      if (stat (injectionPath, &sb) == -1)
	error (_ ("Cannot open library %s pointed by CUDBG_INJECTION_PATH: "
		  "file does not exist!"),
	       injectionPath);
    }
}
