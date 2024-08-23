/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2024 NVIDIA Corporation
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

#include "py-cuda.h"

#include "cuda/cuda-api.h"
#include "gdbsupport/gdb_unique_ptr.h"

#ifdef HAVE_PYTHON

#define DEFAULT_BUFFER_SIZE 1024

PyObject *gdbpy_cuda_execute_internal_command (PyObject *self, PyObject *args)
{
  const char *command;
  unsigned long long buffer_size = DEFAULT_BUFFER_SIZE;
  
  if (!PyArg_ParseTuple (args, "s|K", &command, &buffer_size))
    return nullptr;

  gdb::unique_xmalloc_ptr<char> buffer((char *) xmalloc (buffer_size));

  if (!cuda_debugapi::execute_internal_command (command, buffer.get(), buffer_size))
    {
      PyErr_SetString (PyExc_RuntimeError, "CUDA Debug API Error");
      return nullptr;
    }

  return Py_BuildValue ("s", buffer.get());
}

#endif
