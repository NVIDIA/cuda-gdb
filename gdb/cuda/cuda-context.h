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

#ifndef _CUDA_CONTEXT_H
#define _CUDA_CONTEXT_H 1

#include "defs.h"
#include "cuda-defs.h"

#include <unordered_map>

class cuda_module;

/*--------------------------------------------------------------------------*/

/* Context */

class cuda_context
{
public:
  cuda_context (uint32_t dev_id, uint64_t context_id);
  ~cuda_context ();

  uint64_t id () const
  { return m_id; }

  uint32_t dev_id () const
  { return m_dev_id; }

  void add_module (cuda_module* module);
  void remove_module (cuda_module* module);
  
  std::unordered_map<uint64_t, cuda_module*>& get_modules ()
  { return m_module_map; }

  void flush_disasm_caches ();

  // For internal debugging
  void print () const;

private:
  uint64_t    m_id;                    // the CUcontext handle
  uint32_t    m_dev_id;                // index of the parent device state

  // Map of modules loaded in this context
  std::unordered_map<uint64_t, cuda_module*> m_module_map;
};

#endif
