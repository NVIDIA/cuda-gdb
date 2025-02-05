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

#include "cuda-context.h"
#include "cuda-options.h"
#include "cuda-tdep.h"
#include "cuda-state.h"
#include "cuda-modules.h"

/******************************************************************************
 *
 *                                  Context
 *
 *****************************************************************************/


cuda_context::cuda_context (uint32_t dev_id, uint64_t context_id)
  : m_id (context_id), m_dev_id (dev_id)
{
  cuda_trace ("Context create dev_id %u context_id 0x%llx", m_dev_id, m_id);
}

cuda_context::~cuda_context ()
{
  cuda_trace ("Context destroy dev_id %u context_id 0x%llx", m_dev_id, m_id);
}

void
cuda_context::add_module (cuda_module* module)
{
  cuda_trace ("Context add module 0x%llx", module->id ());
  m_module_map[module->id ()] = module;
}

void
cuda_context::remove_module (cuda_module* module)
{
  cuda_trace ("Context remove module 0x%llx", module->id ());
  m_module_map.erase (module->id ());
}

void
cuda_context::print () const
{
  cuda_trace ("Context device %u context_id 0x%llx %llu modules",
              m_dev_id, m_id, m_module_map.size ());

  for (auto iter : m_module_map)
    iter.second->print ();
}

void
cuda_context::flush_disasm_caches ()
{
  for (auto iter : m_module_map)
    iter.second->flush_disasm_caches ();
}
