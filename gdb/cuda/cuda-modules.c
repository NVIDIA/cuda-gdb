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
#include "breakpoint.h"
#include "objfiles.h"
#include "source.h"

#include "cuda-defs.h"
#include "cuda-asm.h"
#include "cuda-elf-image.h"
#include "cuda-options.h"
#include "cuda-tdep.h"
#include "cuda-modules.h"
#include "cuda-kernel.h"


/******************************************************************************
 *
 *                                   Module
 *
 *****************************************************************************/

module_t
module_new (context_t context, uint64_t module_id, void *elf_image, uint64_t elf_image_size)
{
  module_t module;

  module = (module_t) xmalloc (sizeof *module);
  module->context    = context;
  module->module_id  = module_id;
  module->elf_image  = cuda_elf_image_new (elf_image, elf_image_size, module);
  module->disassembler = new cuda_disassembler ();
  module->next       = NULL;

  return module;
}

void
module_delete (module_t module)
{
  gdb_assert (module);

  if (cuda_elf_image_is_loaded (module->elf_image))
    cuda_elf_image_unload (module->elf_image);

  cuda_elf_image_delete (module->elf_image);

  delete module->disassembler;

  xfree (module);
}

void
module_print (module_t module)
{
  gdb_assert (module);

  cuda_trace ("      module_id 0x%llx", (unsigned long long) module->module_id);
}

context_t
module_get_context (module_t module)
{
  gdb_assert (module);

  return module->context;
}

uint64_t
module_get_id (module_t module)
{
  gdb_assert (module);

  return module->module_id;
}

elf_image_t
module_get_elf_image (module_t module)
{
  gdb_assert (module);

  return module->elf_image;
}

void
module_set_elf_image  (module_t module, elf_image_t elf_image)
{
  gdb_assert (module);

  module->elf_image = elf_image;
}
  
/******************************************************************************
 *
 *                                   Modules
 *
 *****************************************************************************/

struct modules_st {
  module_t    head;                 /* single-linked list of modules */
};

modules_t
modules_new (void)
{
  modules_t modules;

  modules = (modules_t) xmalloc (sizeof *modules);
  modules->head = NULL;

  return modules;
}

void
modules_delete (modules_t modules)
{
  module_t module;
  module_t next_module;

  gdb_assert (modules);

  module = modules->head;
  while (module)
    {
      next_module = module->next;
      kernels_terminate_module (module);
      module_delete (module);
      module = next_module;
    }
  xfree (modules);
}

void
modules_add (modules_t modules, module_t module)
{
  gdb_assert (modules);
  gdb_assert (module);

  module->next  = modules->head;
  modules->head = module;
}

void
modules_remove (modules_t modules, module_t target_module)
{
  module_t *pmodule;
  module_t next_module;

  gdb_assert (modules);
  gdb_assert (modules);

  pmodule = &modules->head;

  for (;pmodule && (*pmodule); pmodule = &((*pmodule)->next)) {
    if ((*pmodule) == target_module)
      {
        next_module = (*pmodule)->next;
        kernels_terminate_module (*pmodule);
        module_delete (*pmodule);
        *pmodule = next_module;
        break;
      }
  }
}

void
modules_print (modules_t modules)
{
  module_t module;

  gdb_assert (modules);

  for (module = modules->head; module; module = module->next)
    module_print (module);
}

module_t
modules_find_module_by_id (modules_t modules, uint64_t module_id)
{
  module_t module;

  for (module = modules->head; module; module = module->next)
    if (module->module_id == module_id)
      return module;

  return NULL;
}

module_t
modules_find_module_by_address (modules_t modules, CORE_ADDR addr)
{
  module_t    module = NULL;
  elf_image_t elf_image;

  gdb_assert (modules);

  for (module = modules->head; module; module = module->next)
  {
    elf_image = module_get_elf_image (module);
    if (cuda_elf_image_contains_address (elf_image, addr))
      return module;
  }

  return NULL;
}
