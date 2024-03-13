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
#include <sys/stat.h>

#include <map>

#include "breakpoint.h"
#include "source.h"

#include "cuda-context.h"
#include "cuda-elf-image.h"
#include "cuda-modules.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"

elf_image_t elf_image_chain = nullptr;
static elf_image_t current_elf_image = nullptr;
static std::multimap<elf_image_t, CORE_ADDR> cuda_kernel_entry_points;

struct elf_image_st {
  struct objfile    *objfile;     /* pointer to the ELF image as managed by GDB */
  char               objfile_path [CUDA_GDB_TMP_BUF_SIZE];
                                  /* file path of the ELF image in the tmp folder */
  uint64_t           size;        /* the size of the relocated ELF image */
  bool               loaded;      /* is the ELF image in memory? */
  bool               uses_abi;    /* does the ELF image uses the ABI to call functions */
  bool               system;      /* is this the system ELF image? */
  module_t           module;      /* the parent module */

  elf_image_t        prev;
  elf_image_t        next;
};

elf_image_t
cuda_elf_image_new (void *image, uint64_t size, module_t module)
{
  elf_image_t elf_image;

  elf_image = (elf_image_t) xmalloc (sizeof (*elf_image));
  elf_image->size     = size;
  elf_image->loaded   = false;
  elf_image->uses_abi = false;
  elf_image->system   = false;
  elf_image->module   = module;
  elf_image->prev     = nullptr;
  elf_image->next     = nullptr;

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
cuda_elf_image_is_system (elf_image_t elf_image)
{
  gdb_assert (elf_image);
  return elf_image->system;
}

bool
cuda_elf_image_contains_address (elf_image_t elf_image, CORE_ADDR addr)
{
  struct objfile       *objfile;
  struct obj_section   *osect = nullptr;
  asection             *section = nullptr;

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

void
cuda_elf_image_add_kernel_entry (CORE_ADDR addr)
{
  gdb_assert (current_elf_image);

  cuda_trace ("Adding entry point for ELF image %s pc=0x%llx",
  	      current_elf_image->objfile_path, (long long)addr);
  
  cuda_kernel_entry_points.emplace (current_elf_image, addr);
}

void
cuda_elf_image_auto_breakpoints_update_locations ()
{
  /* Only add forced locations if we encounter device side launches */
  if ( cuda_options_auto_breakpoints_forced_needed ())
    {
      /* Iterate over each entry in the multimap */
      for (auto& loc : cuda_kernel_entry_points)
	cuda_auto_breakpoints_forced_add_location (loc.first, loc.second);
      
      cuda_auto_breakpoints_update ();
      
      cuda_kernel_entry_points.clear ();
    }
  else
    {
      /* Always update the auto breakpoints in case the option changed. */
      cuda_auto_breakpoints_update ();
    }
}

void cuda_decode_line_table (struct objfile *objfile);

/* For very large cubins we may run into the Linux per-write-syscall
   limitation on the number of bytes read/written (which is just under 2G).
   Read/write in a loop to avoid that problem. */

static void
cuda_elf_image_rw (int fd, void *image, size_t len, bool read_direction)
{
  size_t remaining = len;
  char *ptr = (char *)image;

  while (remaining > 0)
    {
      ssize_t nbytes;

      if (read_direction)
	nbytes = read (fd, ptr, remaining);
      else
	nbytes = write (fd, ptr, remaining);

      if (nbytes < 0)
	error (_("Error: Failed to %s the ELF image file"),
	       read_direction ? "read" : "write");

      remaining -= (size_t) nbytes;
      ptr += nbytes;
    }
}

void
cuda_elf_image_read (const char *filename, void *image, size_t len)
{
  auto fd = open (filename, O_RDONLY);
  if (fd < 0)
    error (_("Could not open %s for reading"), filename);

  cuda_elf_image_rw (fd, image, len, true);

  close (fd);
}

void
cuda_elf_image_write (int fd, const void *image, size_t len)
{
  cuda_elf_image_rw (fd, (char *)image, len, false);
}

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

  cuda_elf_image_write (object_file_fd, image, elf_image->size);

  close (object_file_fd);

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
  struct objfile *objfile = nullptr;
  const struct bfd_arch_info *arch_info;

  gdb_assert (elf_image);
  gdb_assert (!elf_image->loaded);

  /* auto breakpoints */
  current_elf_image = elf_image;

  /* Open the object file and make sure to adjust its arch_info before reading
     its symbols. */
  gdb_bfd_ref_ptr abfd (symfile_bfd_open (elf_image->objfile_path));
  arch_info = bfd_lookup_arch (bfd_arch_m68k, 0);
  abfd->arch_info = arch_info;

  /* Load in the device ELF object file, forcing a symbol read and while
   * making sure that the breakpoints are not re-set automatically. */
  objfile = symbol_file_add_from_bfd (abfd, bfd_get_filename (abfd.get ()),
				      SYMFILE_DEFER_BP_RESET, nullptr, 0,
				      nullptr);
  if (!objfile)
    error (_("Error: Failed to add symbols from device ELF symbol file!\n"));

  /* Identify this gdb objfile as being cuda-specific */
  objfile->cuda_objfile   = true;
  objfile->per_bfd->gdbarch        = cuda_get_gdbarch ();
  /* CUDA - skip prologue - temporary */
  objfile->cuda_producer_is_open64 = cuda_producer_is_open64;

  /* Initialize the elf_image object */
  elf_image->objfile  = objfile;
  elf_image->loaded   = true;
  elf_image->system   = is_system;
  elf_image->uses_abi = cuda_is_bfd_version_call_abi (objfile->obfd.get ());
  cuda_trace ("loaded ELF image (name=%s, module=%p, abi=%d, objfile=%p)",
              objfile->original_name, elf_image->module,
              elf_image->uses_abi, objfile);

  /* FIXME: We probably shouldn't allow use of the elf_image we are constructing
   * until after we are certain it has been fully initialized. */

  /* This might change our ideas about frames already looked at.  */
  reinit_frame_cache ();

  /* CUDA - parse .debug_line if .debug_info doesn't exist.
   * This is to support -lineinfo compilation.
   * This requires a special case, as GDB's .debug_line parser is
   * deeply intertwined with the .debug_info parser.
   * See the bottom of dwarf2/read.c for cuda_decode_line_table()
   * where we hand-craft the datastructures to allow us to use
   * the embedded .debug_line parser.
   * 
   * This must happen after the reinit_frame_cache above.
   */
  if (!objfile->compunit_symtabs)
    cuda_decode_line_table (objfile);
  
  cuda_resolve_breakpoints (0, elf_image);

  cuda_elf_image_auto_breakpoints_update_locations ();

  current_elf_image = nullptr;

  cuda_trace ("ELF image load done");
}

void
cuda_elf_image_unload (elf_image_t elf_image)
{
  struct objfile *objfile = elf_image->objfile;

  gdb_assert (elf_image->loaded);
  gdb_assert (objfile);
  gdb_assert (objfile->cuda_objfile);

  cuda_trace ("unloading ELF image (name=%s, module=%p)",
              objfile->original_name, elf_image->module);

  /* Mark this objfile as being discarded.  */
  objfile->discarding = 1;

  /* Make sure that all its users will be cleaned up. */
  clear_current_source_symtab_and_line ();
  clear_displays ();
  cuda_reset_invalid_breakpoint_location_section (objfile);

  /* Unlink the ELF image in /tmp/cuda-dbg/ */
  unlink (objfile->original_name);

  /* Request the objfile be destroyed */
  objfile->unlink();

  elf_image->objfile = nullptr;
  elf_image->loaded = false;
  elf_image->uses_abi = false;

  /* Remove any still pending locations from the map for this key. */
  cuda_kernel_entry_points.erase (elf_image);

  /* Update the forced breakpoints to remove locations for this image. */
  cuda_auto_breakpoints_remove_locations (elf_image);

  /* Remove any user set breakpoints for this image. */
  cuda_unresolve_breakpoints (elf_image);
}

elf_image_t
cuda_get_elf_image_by_objfile (struct objfile *objfile)
{
  elf_image_t elf_image;
  CUDA_ALL_LOADED_ELF_IMAGES(elf_image)
    if (cuda_elf_image_get_objfile(elf_image) == objfile)
      return elf_image;
  return nullptr;
}

