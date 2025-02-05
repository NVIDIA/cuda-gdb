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
#include <sys/stat.h>

#include <map>
#include <vector>

#include "breakpoint.h"
#include "inferior.h"
#include "objfiles.h"
#include "source.h"

#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-defs.h"
#include "cuda-kernel.h"
#include "cuda-modules.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"

#include "elf/common.h"
#include "elf/external.h"

#define R_CUDA_NONE		0	// no relocation
#define R_CUDA_32		1	// 32bit specific address
#define R_CUDA_64		2	// 64bit specific address
#define R_CUDA_G32		3	// 32bit generic address
#define R_CUDA_G64		4	// 64bit generic address

static void
cuda_trace_module (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_ELF, fmt, ap);
  va_end (ap);
}

//
// Tracing as a macro so that the arguments are not evaluated unless the tracing domain is enabled
//
#define CUDA_TRACE_MODULE(fmt, ...)				\
  do {								\
    if (cuda_options_trace_domain_enabled (CUDA_TRACE_ELF))	\
      cuda_trace_module (fmt, ## __VA_ARGS__);			\
  } while (0)

extern void cuda_decode_line_table (struct objfile *objfile);

/******************************************************************************
 *
 *                                   Module
 *
 *****************************************************************************/


cuda_module* cuda_module::s_current_module = nullptr;

std::multimap<uint64_t, CORE_ADDR> cuda_module::s_kernel_entry_points;

cuda_module::cuda_module (uint64_t module_id,
			  CUDBGElfImageProperties properties,
			  cuda_context* context,
			  uint64_t elf_image_size)
  : m_id (module_id),
    m_properties (properties),
    m_context (context),
    m_objfile (nullptr),
    m_size (elf_image_size),
    m_loaded (false),
    m_functions_loaded (0),
    m_uses_abi (false),
    m_disassembler (nullptr)
{
  gdb_assert (m_id);
  gdb_assert (m_context);

  cuda_trace ("Module create context_id 0x%llx module_id 0x%llx total %llu modules",
	      m_context->id (), m_id, m_context->get_modules ().size () + 1);

  std::vector<uint8_t> buffer (m_size);
  cuda_debugapi::get_elf_image (m_context->dev_id (), module_id, true, buffer.data (), m_size);

  // Will initialize m_filename, but will not create the BFD / objfile.
  write_cubin (buffer.data ());
}

cuda_module::~cuda_module ()
{
  cuda_trace ("Module destroy context_id 0x%llx module_id 0x%llx total %llu modules",
	      m_context->id (), m_id, m_context->get_modules ().size () + 1);

  // If loaded, unload and unlink the objfile
  if (m_loaded)
    unload_objfile (true);

  // Remove any still pending locations from the kernel entry map for this module
  s_kernel_entry_points.erase (m_id);
}

cuda_module_disassembly_cache*
cuda_module::disassembler ()
{
  if (!m_disassembler)
    {
      const auto device
	  = cuda_state::device (m_context->dev_id ());
      const bool is_volta_plus = device->get_sm_version () >= 70;
      m_disassembler = std::make_unique<cuda_module_disassembly_cache> (
        device->get_insn_size (), is_volta_plus);
    }

  return m_disassembler.get ();
}

void
cuda_module::flush_disasm_caches ()
{
  if (m_disassembler)
    {
      m_disassembler->flush_elf_cache ();
      m_disassembler->flush_device_cache ();
    }
}

bool
cuda_module::contains_address (CORE_ADDR addr) const
{
  struct obj_section* osect = nullptr;
  ALL_OBJFILE_OSECTIONS (m_objfile, osect)
    {
      auto section = osect->the_bfd_section;
      if (section && (section->vma <= addr) && (addr < section->vma + section->size))
        return true;
    }

  return false;
}

void
cuda_module::add_kernel_entry (CORE_ADDR addr)
{
  gdb_assert (s_current_module);

  cuda_trace ("Adding entry point for ELF image %s pc=0x%llx",
  	      s_current_module->filename ().c_str (),
	      addr);
  
  s_kernel_entry_points.emplace (s_current_module->id (), addr);
}

void
cuda_module::auto_breakpoints_update_locations ()
{
  // Only add forced locations if we encounter device side launches
  if (cuda_options_auto_breakpoints_forced_needed ())
    {
      // Iterate over each entry in the multimap
      for (auto& loc : s_kernel_entry_points)
	{
	  auto module = cuda_state::find_module_by_id (loc.first);
	  if (!module)
	    cuda_trace ("auto_breakpoints_update_locations: no module with id 0x%llx", loc.first);
	  else
	    cuda_auto_breakpoints_forced_add_location (module, loc.second);
	}
      cuda_auto_breakpoints_update ();
      s_kernel_entry_points.clear ();
    }
  else
    {
      // Always update the auto breakpoints in case the option changed.
      cuda_auto_breakpoints_update ();
    }
}

void
cuda_module::read_cubin (void *image)
{
  auto fd = open (m_filename.c_str (), O_RDONLY);
  if (fd < 0)
    error ("Could not open %s for reading", m_filename.c_str ());

  size_t remaining = m_size;
  uint8_t* ptr = (uint8_t *)image;
  while (remaining > 0)
    {
      auto nbytes = read (fd, ptr, remaining);
      if (nbytes < 0)
	error ("Error: Failed to read the ELF image file");
      remaining -= (size_t) nbytes;
      ptr += nbytes;
    }
  close (fd);
}

void
cuda_module::write_buffer (int fd, const void *image, size_t len)
{
  size_t remaining = len;
  const uint8_t* ptr = (const uint8_t *)image;
  while (remaining > 0)
    {
      auto nbytes = write (fd, ptr, remaining);
      if (nbytes < 0)
	error ("Error: Failed to write the ELF image file");
      remaining -= (size_t) nbytes;
      ptr += nbytes;
    }
}

// This function gets the ELF image from a module load and saves it
// onto the hard drive (usually under /tmp/cuda-dbg).
// The temp filename is created on the first call to write_cubin()
// Following calls re-use the same filename.
void
cuda_module::write_cubin (const void *image)
{
  gdb_assert (!m_loaded);

  int fd;
  if (m_filename.size () == 0)
    {
      // Create a file in /tmp/cuda-dbg to hold the cubin
      char objfile_path [CUDA_GDB_TMP_BUF_SIZE];
      snprintf (objfile_path, sizeof (objfile_path),
		"%s/elf.%lx.%lx.o.XXXXXX",
		cuda_gdb_session_get_dir (),
		m_context->id (),
		m_id);
      fd = mkstemp (objfile_path);
      m_filename = objfile_path;
    }
  else
    fd = open (m_filename.c_str (), O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);

  if (fd < 0)
    error ("Error: Failed to create device ELF symbol file %s", m_filename.c_str ());

  write_buffer (fd, image, m_size);
  close (fd);

  struct stat object_file_stat;
  if (stat (m_filename.c_str (), &object_file_stat))
    error (_("Error: Failed to stat device ELF symbol file!"));

  if (object_file_stat.st_size != m_size)
    error (_("Error: The device ELF file size is incorrect!"));
}

// cuda_module::load_objfile() reads the ELF image file into symbol table.
// Native debugging calls cuda_module::save_buffer() first. Remote debugging
// only calls load() since the ELF image file has already fetched from server.
void
cuda_module::load_objfile ()
{
  gdb_assert (!m_loaded);

  // auto breakpoints
  s_current_module = this;

  // Open the object file and make sure to adjust its arch_info before reading
  // its symbols.
  gdb_bfd_ref_ptr abfd (symfile_bfd_open (m_filename.c_str ()));
  const auto arch_info = bfd_lookup_arch (bfd_arch_m68k, 0);
  abfd->arch_info = arch_info;

  // Load in the device ELF object file, forcing a symbol read and while
  // making sure that the breakpoints are not re-set automatically.
  m_objfile = symbol_file_add_from_bfd (abfd, bfd_get_filename (abfd.get ()),
					SYMFILE_DEFER_BP_RESET, nullptr, 0,
					nullptr);
  if (!m_objfile)
    error ("Error: Failed to add symbols from device ELF symbol file!\n");

  // Identify this gdb objfile as being cuda-specific
  m_objfile->cuda_objfile     = true;
  m_objfile->m_cuda_module    = this;

  m_objfile->per_bfd->gdbarch = cuda_get_gdbarch ();

  // CUDA - skip prologue - temporary
  m_objfile->cuda_producer_is_open64 = cuda_producer_is_open64;

  // cubin is now loaded
  m_loaded   = true;
  m_uses_abi = cuda_is_bfd_version_call_abi (m_objfile->obfd.get ());
  cuda_trace ("loaded ELF image (name=%s, module=0x%llx, abi=%d)",
              m_filename.c_str (), m_id, m_uses_abi);

  // This might change our ideas about frames already looked at.
  reinit_frame_cache ();

  // CUDA - parse .debug_line if .debug_info doesn't exist.
  // This is to support -lineinfo compilation.
  // This requires a special case, as GDB's .debug_line parser is
  // deeply intertwined with the .debug_info parser.
  // See the bottom of dwarf2/read.c for cuda_decode_line_table()
  // where we hand-craft the datastructures to allow us to use
  // the embedded .debug_line parser.
  //
  // This must happen after the reinit_frame_cache above.
  if (!m_objfile->compunit_symtabs)
    cuda_decode_line_table (m_objfile);
  
  // Don't try to update/set breakpoints for corefiles
  if (target_has_execution ())
    {
      cuda_resolve_breakpoints (0, this);
      auto_breakpoints_update_locations ();
    }

  s_current_module = nullptr;

  cuda_trace ("ELF image load done");
}

void
cuda_module::unload_objfile (bool unlink_file)
{
  gdb_assert (m_loaded);
  gdb_assert (m_objfile);
  gdb_assert (m_objfile->cuda_objfile);

  cuda_trace ("unloading ELF image (name=%s, module=0x%llx)",
              m_objfile->original_name, m_id);

  // Mark this objfile as being discarded.
  m_objfile->discarding = 1;

  // Make sure that all its users will be cleaned up.
  clear_current_source_symtab_and_line ();
  clear_displays ();
  cuda_reset_invalid_breakpoint_location_section (m_objfile);

  // Unlink the ELF image in /tmp/cuda-dbg/ if requested
  if (unlink_file)
    {
      unlink (m_filename.c_str ());
      m_filename = "";
    }

  // Request the objfile be destroyed
  m_objfile->unlink();
  m_objfile = nullptr;

  m_loaded = false;
  m_uses_abi = false;

  // Don't try to update/set breakpoints for corefiles
  if (target_has_execution ())
    {
      // Update the forced breakpoints to remove locations for this image.
      cuda_auto_breakpoints_remove_locations (this);

      // Remove any user set breakpoints for this image.
      cuda_unresolve_breakpoints (this);
    }
}

void
cuda_module::functions_loaded_event (uint32_t count)
{
  auto device_id = context ()->dev_id ();

  // Save a copy of the ELF image for updating below
  std::vector<uint8_t> buffer (m_size);

  // For debugging incremental vs. full reload
  bool incremental_updates = true;
  if (incremental_updates)
    {
      // Save a copy of the current cubin
      read_cubin (buffer.data ());

      // Unload the objfile, keeping the file in /tmp/cuda-dbg/...
      // in place for later writing.
      unload_objfile (false);

      // Fetch the incremental function load information
      // and apply the incremental relocations
      auto newly_loaded = count - functions_loaded ();
      if (cuda_debugapi::api_version ().m_revision >= 135)
	{
	  std::vector<CUDBGLoadedFunctionInfo> info (newly_loaded);
	  cuda_debugapi::get_loaded_function_info (device_id,
						   m_id,
						   info.data (),
						   functions_loaded (),
						   newly_loaded);
	  apply_function_load_updates (buffer.data (), info.data (), newly_loaded);
	}
      else
	{
	  // Older debugger backend - have to fetch the whole vector
	  // This should always work as we were sent a LFL event by the backend
	  // indicating support for this API
	  std::vector<CUDBGLoadedFunctionInfo> info (count);
	  cuda_debugapi::get_loaded_function_info (device_id, id (), info.data (), count);
	  apply_function_load_updates (buffer.data (),
				       &info[functions_loaded ()],
				       newly_loaded);
	}
      m_functions_loaded += newly_loaded;

      // For debugging, compare before and after incremental updates
      bool compare_cubins = false;
      if (compare_cubins)
	{
	  // Save a copy of the ELF image for comparison
	  std::vector<uint8_t> reference (m_size);

	  // Re-fetch the whole cubin
	  cuda_debugapi::get_elf_image (device_id, m_id, true, reference.data (), m_size);

	  // Compare
	  if (memcmp (buffer.data (), reference.data (), m_size))
	    error (_("%s: cubin buffers do not match after incremental update"), __FUNCTION__);
	}
    }
  else
    {
      // Unload the objfile
      unload_objfile (false);

      // Re-fetch the whole cubin
      cuda_debugapi::get_elf_image (device_id, m_id, true, buffer.data (), m_size);
    }

  // Write image back out to disk for cuda_module::load_objfile() to create a BFD from.
  write_cubin (buffer.data ());

  // Load the ELF image from the filesystem, creating a new objfile
  load_objfile ();

  // Flush the module disassembly caches as we've just updated the
  // ELF image
  flush_disasm_caches ();
}

void
cuda_module::print() const
{
  cuda_trace ("      module_id 0x%llx size %llu file %s",
	      m_id, m_size, m_filename.c_str ());
}

void
cuda_module::apply_function_load_updates (uint8_t *buffer,
					  CUDBGLoadedFunctionInfo *info,
					  uint32_t count)
{
  gdb_assert (buffer);
  gdb_assert (info);

  // Extract ELF Header information, section header table offset, number of sections, and section header size.
  Elf64_External_Ehdr *e_hdr = (Elf64_External_Ehdr *)buffer;
  uint64_t shoff = *(uint64_t *)e_hdr->e_shoff;
  uint64_t shnum = *(uint16_t *)e_hdr->e_shnum;
  uint16_t shentsize = *(uint16_t *)e_hdr->e_shentsize;
  gdb_assert (shentsize > 0);

  // Handle the case where shnum == 0 due to there being more than
  // SHN_LORESERVE sections.
  if (shnum == 0)
    {
      // The number of sections is stored in the sh_size field of the section
      // header at index 0.
      Elf64_External_Shdr *section = (Elf64_External_Shdr *)&buffer[shoff];
      shnum = *(uint64_t *)section->sh_size;
    }
  gdb_assert (count <= shnum);

  // Find the section string table
  uint64_t shstrndx = *(uint16_t *)e_hdr->e_shstrndx;
  if (shstrndx == SHN_XINDEX)
    {
      // The actual index of the section name string table section is contained in the
      // sh_link field of the section header at index 0.
      Elf64_External_Shdr *section = (Elf64_External_Shdr *)&buffer[shoff];
      shstrndx = *(uint64_t *)section->sh_link;
    }
  gdb_assert (shstrndx < shnum);

  Elf64_External_Shdr *shstrtab_section = (Elf64_External_Shdr *)&buffer[shoff + shstrndx * shentsize];
  uint64_t shstrtab_offset = *(uint64_t *)shstrtab_section->sh_offset;
  auto shstrtab = &buffer[shstrtab_offset];

  CUDA_TRACE_MODULE ("%u sections of %llu loaded", count, shnum);

  // Locate the strtab, symtab, and symtab section index sections.
  Elf64_External_Shdr *strtab = nullptr;
  Elf64_External_Shdr *symtab = nullptr;
  Elf64_External_Shdr *symtab_shndx = nullptr;
  for (int sectionidx = 0; sectionidx < shnum; ++sectionidx)
    {
      Elf64_External_Shdr *section = (Elf64_External_Shdr *)&buffer[shoff + sectionidx * shentsize];
      uint32_t section_type = *(uint32_t *)section->sh_type;
      uint32_t shstr_offset = *(uint32_t *)section->sh_name;
      const auto section_name = (const char *)&shstrtab[shstr_offset];

      if (section_type == SHT_SYMTAB)
	symtab = section;
      else if (section_type == SHT_SYMTAB_SHNDX)
	symtab_shndx = section;
      else if (section_type == SHT_STRTAB && !strcmp (section_name, ".strtab"))
	strtab = section;
    }

  // Need both a string table and a symbol table
  gdb_assert (strtab);
  gdb_assert (symtab);

  // Found the string table
  uint64_t strtab_offset = *(uint64_t *)strtab->sh_offset;

  // Found the symbol table
  uint64_t symtab_size = *(uint64_t *)symtab->sh_size;
  uint32_t symtab_entsize = *(uint32_t *)symtab->sh_entsize;
  uint64_t symtab_offset = *(uint64_t *)symtab->sh_offset;
  gdb_assert (symtab_entsize > 0);

  // The symbol table section index table is optional
  uint64_t symtab_shndx_entsize = 0;
  uint64_t symtab_shndx_offset = 0; 
  if (symtab_shndx)
    {
      symtab_shndx_entsize = *(uint32_t *)symtab_shndx->sh_entsize;
      gdb_assert (symtab_shndx_entsize > 0);

      symtab_shndx_offset = *(uint64_t *)symtab_shndx->sh_offset; 
      gdb_assert (symtab_shndx_offset > 0);
    }

  // Find the functions (sections) that are newly loaded this time.
  // This is based on the count of updated sections passed in.
  std::vector<uint64_t> section_addresses (shnum);
  for (int i = 0; i < count; ++i)
    {
      const uint64_t section_index = info[i].sectionIndex;
      Elf64_External_Shdr *section = (Elf64_External_Shdr *)&buffer[shoff + section_index * shentsize];

      CUDA_TRACE_MODULE ("relocate section %lld %s from 0x%llx to 0x%llx",
			 section_index,
			 (const char *)&shstrtab[*(uint32_t *)section->sh_name],
			 *(uint64_t *)section->sh_addr,
			 info[i].address);

      // Check for newly relocated sections
      const uint64_t section_address = *(uint64_t *)section->sh_addr;
      if (section_address)
	{
	  if (section_address != info[i].address)
	    CUDA_TRACE_MODULE("Skipping section %lld: mismatched addresses for section %lld %s: 0x%llx vs 0x%llx",
			      section_index,
			      (const char *)&shstrtab[*(uint32_t *)section->sh_name],
			      section_address,
			      info[i].address);
	  else
  	    CUDA_TRACE_MODULE ("Skipping section %lld %s: already relocated addr 0x%llx",
			       section_index,
			       (const char *)&shstrtab[*(uint32_t *)section->sh_name],
			       section_address);
	}
      else
        {
	  *(uint64_t *)section->sh_addr = info[i].address;
	  section_addresses[section_index] = info[i].address;

	  CUDA_TRACE_MODULE ("relocated section %d %s addr 0x%llx",
			     section_index,
			     (const char *)&shstrtab[*(uint32_t *)section->sh_name],
			     info[i].address);
	}
    }

  // Update symbol table values based on section addresses.
  std::vector<uint64_t> symbol_updated_address (symtab_size / symtab_entsize);
  for (int symidx = 0; symidx < symbol_updated_address.size (); ++symidx)
    {
      Elf64_External_Sym *symbol = (Elf64_External_Sym *)&buffer[symtab_offset + symidx * symtab_entsize];

      // Only relocate symbols which belong to newly located sections
      uint32_t sym_shndx = *(uint16_t *)symbol->st_shndx;

      // Check for st_shndx overflow.
      if (sym_shndx == SHN_XINDEX)
	{ 
	  // This cubin requires a symbol table section index table.
	  gdb_assert (symtab_shndx);
	  sym_shndx = *(uint32_t *)&buffer[symtab_shndx_offset + symidx * symtab_shndx_entsize];
	}
      if (sym_shndx >= section_addresses.size ())
	{
	  CUDA_TRACE_MODULE ("Symbol %s section number out of range %u >= %lu",
			     sym_shndx, section_addresses.size ());
	  continue;
	}

      if (section_addresses[sym_shndx])
	{
	  uint64_t value = *(uint64_t *)symbol->st_value;

	  const auto symbol_name = (const char *)&buffer[strtab_offset + *(uint32_t *)symbol->st_name];
	  CUDA_TRACE_MODULE ("relocating symbol %d %s section %d from 0x%llx plus 0x%llx to 0x%llx",
			     symidx,
			     symbol_name,
			     (uint32_t)sym_shndx,
			     value,
			     section_addresses[sym_shndx],
			     value + section_addresses[sym_shndx]);

	  value += section_addresses[sym_shndx];

	  *(uint64_t *)symbol->st_value = value;
	  symbol_updated_address[symidx] = value;
	}
    }

  // Apply relocations based on symbols just relocated
  for (uint32_t reloc_section_idx = 0; reloc_section_idx < shnum; ++reloc_section_idx)
    {
      Elf64_External_Shdr *reloc_section = (Elf64_External_Shdr *)&buffer[shoff + reloc_section_idx * shentsize];
      uint32_t reloc_section_type = *(uint32_t *)reloc_section->sh_type;

      if ((reloc_section_type == SHT_RELA) || (reloc_section_type == SHT_REL))
	{
	  bool is_rela = (reloc_section_type == SHT_RELA) ? true : false;
	  const auto reloc_section_name = (const char *)&shstrtab[*(uint32_t *)reloc_section->sh_name];

	  // The section we are relocating
	  uint32_t target_section_idx = *(uint32_t *)reloc_section->sh_info;
	  Elf64_External_Shdr *target_section = (Elf64_External_Shdr *)&buffer[shoff + target_section_idx * shentsize];
	  uint32_t target_section_flags = *(uint64_t *)target_section->sh_flags;
	  const auto target_section_name = (const char *)&shstrtab[*(uint32_t *)target_section->sh_name];

	  // The majority (all) of relocations in CODE / EXECINSTR sections are not supported
	  // Don't process relocations for these sections
	  if (target_section_flags & SHF_EXECINSTR)
	    {
	      CUDA_TRACE_MODULE ("Skipping EXECINSTR section %d %s reloc section %s",
				 target_section_idx,
				 target_section_name,
				 reloc_section_name);
	      continue;
	    }

	  CUDA_TRACE_MODULE ("Relocate %s section %s (%u) using reloc section %s (%u)",
			     is_rela ? "RELA" : "REL ",
			     target_section_name,
			     target_section_idx,
			     reloc_section_name,
			     reloc_section_idx);

	  auto target_section_ptr = &buffer[*(uint64_t *)target_section->sh_offset];

	  uint64_t reloc_section_size = *(uint64_t *)reloc_section->sh_size;
	  uint64_t reloc_section_offset = *(uint64_t *)reloc_section->sh_offset;
	  for (uint64_t offset = reloc_section_offset; offset < (reloc_section_offset + reloc_section_size); )
	    {
	      uint64_t r_offset = 0;
	      uint64_t r_info = 0;
	      uint64_t r_addend = 0;

	      if (is_rela)
		{
		  Elf64_External_Rela *rela = (Elf64_External_Rela *)&buffer[offset];

		  r_offset = *(uint64_t *)rela->r_offset;
		  r_info = *(uint64_t *)rela->r_info;
		  r_addend = *(uint64_t *)rela->r_addend;

		  offset += sizeof (Elf64_External_Rela);
		}
	      else
		{
		  Elf64_External_Rel *rel = (Elf64_External_Rel *)&buffer[offset];

		  r_offset = *(uint64_t *)rel->r_offset;
		  r_info = *(uint64_t *)rel->r_info;

		  // r_addend will be set below based on relocation type / size

		  offset += sizeof (Elf64_External_Rel);
		}

	      uint32_t symidx = ELF64_R_SYM (r_info);
	      if (symidx >= symbol_updated_address.size ())
		{
		  CUDA_TRACE_MODULE ("Skipping unknown symbol index %u of 0x%llu", symidx, symbol_updated_address.size ());
		  continue;
		}
	      
	      // Don't apply relocations for symbols whose sections haven't been loaded yet
	      if (!symbol_updated_address[symidx])
		continue;

	      auto target_ptr = &target_section_ptr[r_offset];
	      switch (ELF64_R_TYPE (r_info))
		{
		case R_CUDA_32:
		case R_CUDA_G32:
		  if (!is_rela)
		    r_addend = *(uint32_t *)target_ptr;
		  *(uint32_t *)target_ptr = (uint32_t)(symbol_updated_address[symidx] + r_addend);
		  break;

		case R_CUDA_64:
		case R_CUDA_G64:
		  if (!is_rela)
		    r_addend = *(uint64_t *)target_ptr;
		  *(uint64_t *)target_ptr = (uint64_t)(symbol_updated_address[symidx] + r_addend);
		  break;

		default:
		  // Unknown relocation, skip
		  CUDA_TRACE_MODULE ("unknown relocation type %d", (uint32_t)ELF64_R_TYPE (r_info));
		  break;
		}

	      // Log the relocation after it's been applied so that r_addend is correct for
	      // REL style relocations
	      if (cuda_options_trace_domain_enabled (CUDA_TRACE_ELF))
		{
		  Elf64_External_Sym *symbol = (Elf64_External_Sym *)&buffer[symtab_offset + symidx * symtab_entsize];
		  const auto symbol_name = (const char *)&buffer[strtab_offset + *(uint32_t *)symbol->st_name];

		  CUDA_TRACE_MODULE ("reloc %4s rtype %u tsect %u symidx %u r_off 0x%llx symval 0x%llx radd 0x%llx symbol %s",
				     is_rela ? "RELA" : "REL ",
				     (uint32_t)ELF64_R_TYPE (r_info),
				     (uint32_t)target_section_idx,
				     (uint32_t)symidx,
				     r_offset,
				     symbol_updated_address[symidx],
				     r_addend,
				     symbol_name);
		}
	    }
	}
    }
}
