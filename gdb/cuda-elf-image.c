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

#include <sys/stat.h>

#include "defs.h"
#include "breakpoint.h"
#include "gdb_assert.h"
#include "source.h"

#include "cuda-context.h"
#include "cuda-elf-image.h"
#include "cuda-modules.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"

elf_image_t elf_image_chain = NULL;

struct elf_image_st {
  struct objfile    *objfile;     /* pointer to the ELF image as managed by GDB */
  char               objfile_path [CUDA_GDB_TMP_BUF_SIZE];
                                  /* file path of the ELF image in the tmp folder */
  uint64_t           size;        /* the size of the relocated ELF image */
  bool               loaded;      /* is the ELF image in memory? */
  bool               uses_abi;    /* does the ELF image uses the ABI to call functions */
  module_t           module;      /* the parent module */

  elf_image_t        prev;
  elf_image_t        next;
};


elf_image_t
cuda_elf_image_new (void *image, uint64_t size, module_t module)
{
  elf_image_t elf_image;

  elf_image = xmalloc (sizeof (*elf_image));
  elf_image->size     = size;
  elf_image->loaded   = false;
  elf_image->uses_abi = false;
  elf_image->module   = module;
  elf_image->prev     = NULL;
  elf_image->next     = NULL;

  if (elf_image_chain)
    {
      elf_image_chain->prev = elf_image;
      elf_image->next = elf_image_chain;
    }
  elf_image_chain = elf_image;

  cuda_elf_image_save (elf_image, image);

  return elf_image;
}

void
cuda_elf_image_delete (elf_image_t elf_image)
{
  if (elf_image->prev)
    elf_image->prev->next = elf_image->next;
  if (elf_image->next)
    elf_image->next->prev = elf_image->prev;
  if (elf_image_chain == elf_image)
    elf_image_chain = elf_image->next;

  gdb_assert (elf_image);
  xfree (elf_image);
}

struct objfile *
cuda_elf_image_get_objfile (elf_image_t elf_image)
{
  gdb_assert (elf_image);
  return elf_image->objfile;
}

uint64_t
cuda_elf_image_get_size (elf_image_t elf_image)
{
  gdb_assert (elf_image);
  return elf_image->size;
}

module_t
cuda_elf_image_get_module (elf_image_t elf_image)
{
  gdb_assert (elf_image);
  return elf_image->module;
}

elf_image_t
cuda_elf_image_get_next (elf_image_t elf_image)
{
  gdb_assert (elf_image);
  return elf_image->next;
}

bool
cuda_elf_image_is_loaded (elf_image_t elf_image)
{
  gdb_assert (elf_image);
  return elf_image->loaded;
}

bool
cuda_elf_image_contains_address (elf_image_t elf_image, CORE_ADDR addr)
{
  struct objfile       *objfile;
  struct obj_section   *osect = NULL;
  asection             *section = NULL;

  gdb_assert (elf_image);

  if (!cuda_elf_image_is_loaded (elf_image))
    return false;

  objfile = cuda_elf_image_get_objfile (elf_image);
  ALL_OBJFILE_OSECTIONS (objfile, osect)
    {
      section = osect->the_bfd_section;
      if (section && section->vma <= addr &&
          addr < section->vma + section->size)
        return true;
    }

  return false;
}

bool
cuda_elf_image_uses_abi (elf_image_t elf_image)
{
  gdb_assert (elf_image);

  return elf_image->uses_abi;
}

void cuda_decode_line_table (struct objfile *objfile);

/* This function gets the ELF image from a module load and saves it
 * onto the hard drive. */
void
cuda_elf_image_save (elf_image_t elf_image, void *image)
{
  int object_file_fd;
  struct stat object_file_stat;
  context_t context;
  uint64_t context_id;
  uint64_t module_id;
  uint64_t nbytes = 0;

  gdb_assert (elf_image);
  gdb_assert (!elf_image->loaded);

  context    = module_get_context (elf_image->module);
  context_id = context_get_id (context);
  module_id  = module_get_id (elf_image->module);
  snprintf (elf_image->objfile_path, sizeof (elf_image->objfile_path),
            "%s/elf.%llx.%llx.o.XXXXXX",
            cuda_gdb_session_get_dir (),
            (unsigned long long)context_id,
            (unsigned long long)module_id);

  object_file_fd = mkstemp (elf_image->objfile_path);
  if (object_file_fd == -1)
    error (_("Error: Failed to create device ELF symbol file!"));

  nbytes = write (object_file_fd, image, elf_image->size);
  close (object_file_fd);
  if (nbytes != elf_image->size)
    error (_("Error: Failed to write the ELF image file"));

  if (stat (elf_image->objfile_path, &object_file_stat))
    error (_("Error: Failed to stat device ELF symbol file!"));
  else if (object_file_stat.st_size != elf_image->size)
    error (_("Error: The device ELF file size is incorrect!"));
}

/* cuda_elf_image_load() reads the ELF image file into symbol table.
 * Native debugging calls cuda_elf_image_save() first. Remote debugging
 * only calls load() since the ELF image file has already fetched from server. */
void
cuda_elf_image_load (elf_image_t elf_image, bool is_system)
{
  bfd *abfd;
  struct objfile *objfile = NULL;
  const struct bfd_arch_info *arch_info;

  gdb_assert (elf_image);
  gdb_assert (!elf_image->loaded);

  /* auto breakpoints */
  VEC_truncate (CORE_ADDR, cuda_kernel_entry_addresses, 0);

  /* Open the object file and make sure to adjust its arch_info before reading
     its symbols. */
  abfd = symfile_bfd_open (elf_image->objfile_path);
  arch_info = bfd_lookup_arch (bfd_arch_m68k, 0);
  bfd_set_arch_info (abfd, arch_info);

  /* Load in the device ELF object file, forcing a symbol read and while
   * making sure that the breakpoints are not re-set automatically. */
  objfile = symbol_file_add_from_bfd (abfd, SYMFILE_DEFER_BP_RESET,
                                      NULL, 0, NULL);
  if (!objfile)
    error (_("Error: Failed to add symbols from device ELF symbol file!\n"));

  /* Identify this gdb objfile as being cuda-specific */
  objfile->cuda_objfile   = 1;
  objfile->gdbarch        = cuda_get_gdbarch ();
  /* CUDA - skip prologue - temporary */
  objfile->cuda_producer_is_open64 = cuda_producer_is_open64;

  /* CUDA - line info */
  if (!objfile->symtabs)
    cuda_decode_line_table (objfile);

  /* Initialize the elf_image object */
  elf_image->objfile  = objfile;
  elf_image->loaded   = true;
  elf_image->uses_abi = cuda_is_bfd_version_call_abi (objfile->obfd);
  cuda_trace ("loaded ELF image (name=%s, module=%p, abi=%d, objfile=%p)",
              objfile->name, elf_image->module,
              elf_image->uses_abi, objfile);

  cuda_resolve_breakpoints (0, elf_image);
  cuda_auto_breakpoints_add_locations (elf_image, is_system);
}

void
cuda_elf_image_unload (elf_image_t elf_image)
{
  struct objfile *objfile = elf_image->objfile;

  gdb_assert (elf_image->loaded);
  gdb_assert (objfile);
  gdb_assert (objfile->cuda_objfile);

  cuda_trace ("unloading ELF image (name=%s, module=%p)",
              objfile->name, elf_image->module);

  /* Make sure that all its users will be cleaned up. */
  clear_current_source_symtab_and_line ();
  clear_displays ();
  cuda_reset_invalid_breakpoint_location_section (objfile);
  free_objfile (objfile);

  elf_image->objfile = NULL;
  elf_image->loaded = false;
  elf_image->uses_abi = false;

  cuda_auto_breakpoints_remove_locations (elf_image);
  cuda_unresolve_breakpoints (elf_image);
}
