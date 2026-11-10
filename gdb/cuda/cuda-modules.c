/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2026 NVIDIA Corporation
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

#include <cctype>
#include <cstring>
#include <map>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "bfd/elf-bfd.h"

#include "breakpoint.h"
#include "gdb_bfd.h"
#include "gdbcore.h"
#include "objfiles.h"
#include "source.h"
#include "symfile.h"

#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-events.h"
#include "cuda-modules.h"
#include "cuda-options.h"
#include "cuda-sass-json.h"
#include "cuda-state.h"
#include "cuda-stats.h"
#include "cuda-tdep.h"

#include "elf/common.h"
#include "elf/external.h"

#include "gdbsupport/scoped_fd.h"

#define R_CUDA_NONE 0 // no relocation
#define R_CUDA_32 1   // 32bit specific address
#define R_CUDA_64 2   // 64bit specific address
#define R_CUDA_G32 3  // 32bit generic address
#define R_CUDA_G64 4  // 64bit generic address

// Per-step timer for functions_loaded_event; off by default. When on,
// rows appear under FUNCTIONS_LOADED in `maint print cuda_stats events`.
#define CUDA_LFL_STEP_TIMER(step)                                             \
  std::optional<cuda_statistic::scoped_timer> step##_timer;                   \
  if (cuda_options_profile_lfl ())                                            \
  step##_timer.emplace (                                                      \
      get_cuda_event_statistics ().get_statistic ("FUNCTIONS_LOADED_" #step))

static void
cuda_trace_module (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_ELF, fmt, ap);
  va_end (ap);
}

//
// Tracing as a macro so that the arguments are not evaluated unless the
// tracing domain is enabled
//
#define CUDA_TRACE_MODULE(fmt, ...)                                           \
  do                                                                          \
    {                                                                         \
      if (cuda_options_trace_domain_enabled (CUDA_TRACE_ELF))                 \
	cuda_trace_module (fmt, ##__VA_ARGS__);                               \
    }                                                                         \
  while (0)

/******************************************************************************
 *
 *                                   Module
 *
 *****************************************************************************/

cuda_module *cuda_module::s_current_module = nullptr;

std::multimap<uint64_t, CORE_ADDR> cuda_module::s_kernel_entry_points;

cuda_rangemap<cuda_module *> cuda_module::s_cuda_objfile_section_ranges;

cuda_module::cuda_module (uint64_t module_id, uint64_t elf_handle,
			  CUDBGElfImageProperties properties,
			  cuda_context *context, uint64_t elf_image_size)
    : m_module_id (module_id), m_elf_handle (elf_handle),
      m_properties (properties), m_context (context),
      m_cubin_storage_stale (true), m_cubin_buffer (elf_image_size),
      m_objfile (nullptr), m_size (elf_image_size), m_loaded (false),
      m_functions_loaded (0), m_lazy_load_event_count (0), m_uses_abi (false),
      m_disassembler (nullptr)
{
  gdb_assert (m_module_id);
  gdb_assert (m_elf_handle);
  gdb_assert (m_context);

  // The +1 is for the current module being created.
  cuda_trace ("Module create context_id 0x%llx module_id 0x%llx elf_handle "
	      "0x%llx total %llu modules",
	      m_context->id (), m_module_id, m_elf_handle,
	      m_context->get_modules ().size () + 1);

  // Fetch the relocated ELF straight into m_cubin_buffer; load_objfile ()
  // serves it to BFD via gdb_bfd_openr_iovec instead of reading from disk.
  cuda_debugapi::get_elf_image (m_context->dev_id (), elf_handle, true,
				m_cubin_buffer.data (), m_size);

  m_file_path
      = string_printf ("%s/elf.%lx.%lx.%lx.o", cuda_gdb_session_get_dir (),
		       m_context->id (), m_module_id, m_elf_handle);

  // BFD reads from m_cubin_buffer; use a real /tmp name only under
  // preserve_tmp_data, which keeps that file current.
  if (cuda_options_preserve_tmp_data ())
    {
      m_bfd_name = m_file_path;
      write_cubin ();
    }
  else
    m_bfd_name = string_printf ("<cuda-cubin:0x%lx:0x%lx>", m_context->id (),
				m_elf_handle);
}

cuda_module::~cuda_module ()
{
  cuda_trace ("Module destroy context_id 0x%lx module_id 0x%lx elf_handle "
	      "0x%llx total %lu modules",
	      m_context->id (), m_module_id, m_elf_handle,
	      m_context->get_modules ().size ());

  // The /tmp file is unlinked by unload_objfile on cubin unload (unless
  // preserve_tmp_data). Only the kernel entry map needs cleanup here.
  s_kernel_entry_points.erase (m_elf_handle);
}

cuda_module_disassembly_cache *
cuda_module::disassembler ()
{
  if (!m_disassembler)
    m_disassembler = std::make_unique<cuda_module_disassembly_cache> (this);

  return m_disassembler.get ();
}

void
cuda_module::flush_disasm_caches (bool device_only)
{
  if (m_disassembler)
    m_disassembler->flush_cache (device_only);
}

void
cuda_module::add_kernel_entry (CORE_ADDR addr)
{
  gdb_assert (s_current_module);

  cuda_trace ("Adding entry point for ELF image %s pc=0x%llx",
	      s_current_module->bfd_name ().c_str (), addr);

  s_kernel_entry_points.emplace (s_current_module->elf_handle (), addr);
}

void
cuda_module::auto_breakpoints_update_locations ()
{
  // Only add forced locations if we encounter device side launches
  if (cuda_options_auto_breakpoints_forced_needed ())
    {
      // Iterate over each entry in the multimap
      for (auto &loc : s_kernel_entry_points)
	{
	  auto module = cuda_state::find_module_by_id (loc.first);
	  if (!module)
	    cuda_trace (
		"auto_breakpoints_update_locations: no module with id 0x%llx",
		loc.first);
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

cuda_module *
cuda_module::find_cuda_module_by_address (CORE_ADDR addr)
{
  auto ret = s_cuda_objfile_section_ranges.get (addr);
  if (ret)
    return *ret;
  return nullptr;
}

/* gdb_bfd_iovec_base that reads from a cuda_module's in-memory cubin
   buffer (cf. rocm_code_object_stream_memory in solib-rocm.c).

   Holds a non-owning pointer/size; safe because the BFD lifetime is
   bracketed by load_objfile/unload_objfile and the cuda_module owns the
   buffer.  */

namespace
{

class cuda_module_iovec final : public gdb_bfd_iovec_base
{
public:
  cuda_module_iovec (const uint8_t *data, uint64_t size)
      : m_data (data), m_size (size)
  {
  }

  file_ptr
  read (bfd *abfd, void *buf, file_ptr nbytes, file_ptr offset) override
  {
    if (nbytes < 0 || offset < 0 || (uint64_t)offset > m_size)
      {
	bfd_set_error (bfd_error_bad_value);
	return -1;
      }
    file_ptr available = (file_ptr)(m_size - (uint64_t)offset);
    if (nbytes > available)
      nbytes = available;
    memcpy (buf, m_data + offset, nbytes);
    return nbytes;
  }

  int
  stat (struct bfd *abfd, struct stat *sb) override
  {
    memset (sb, 0, sizeof (*sb));
    sb->st_size = (off_t)m_size;
    sb->st_mode = S_IFREG | 0444;
    return 0;
  }

private:
  const uint8_t *m_data;
  uint64_t m_size;
};

} // anonymous namespace

// Write m_cubin_buffer out to the /tmp/cuda-dbg/... file.
void
cuda_module::write_cubin ()
{
  scoped_fd fd (open (m_file_path.c_str (), O_WRONLY | O_CREAT | O_TRUNC,
		      S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH));
  if (fd.get () < 0)
    error ("Error: Failed to create device ELF symbol file %s",
	   m_file_path.c_str ());

  // Loop over write(): a single call can't transfer images above the
  // ~2G limit on Linux.
  const uint8_t *ptr = m_cubin_buffer.data ();
  size_t remaining = m_size;
  while (remaining > 0)
    {
      const ssize_t nbytes = write (fd.get (), ptr, remaining);
      if (nbytes < 0)
	error ("Error: Failed to write the ELF image file");
      remaining -= (size_t)nbytes;
      ptr += nbytes;
    }

  struct stat object_file_stat;
  if (stat (m_file_path.c_str (), &object_file_stat))
    error (_ ("Error: Failed to stat device ELF symbol file!"));

  if (object_file_stat.st_size != m_size)
    error (_ ("Error: The device ELF file size is incorrect!"));

  m_cubin_storage_stale = false;
}

/* Ensure the cubin exists on disk and return its path, for tools
   such as cuobjdump. The path is fixed at construction; the file is
   (re)written here only when stale.  */
const std::string &
cuda_module::sync_to_disk_and_get_file_path ()
{
  if (m_cubin_storage_stale)
    write_cubin ();
  return m_file_path;
}

/* Apply FUNC to each loaded CODE section (vma != 0). The BFD is rebuilt
   from the patched m_cubin_buffer on every LFL event, so vma reflects the
   current sh_addr; truly-unloaded sections keep vma == 0 and are skipped. */
template <typename Func>
inline void
cuda_module::foreach_cuda_objfile_section (Func func)
{
  for (auto osect : m_objfile->sections ())
    {
      asection *section = osect->the_bfd_section;
      if (!section)
	continue;

      // Skip sections that do not have code
      if (!(section->flags & SEC_CODE))
	continue;

      // Skip NOBITS sections
      if (elf_section_data (section)->this_hdr.sh_type == SHT_NOBITS)
	continue;

      // Skip sections that are unrelocated
      if (section->vma == 0)
	continue;

      // Process the section
      func (section);
    }
}

static std::string
compute_debug_sections (struct objfile *objfile)
{
  if (objfile == nullptr || objfile->obfd == nullptr)
    return "none";

  static const char *const sections[] = {
    ".debug_info",
    ".debug_line",
    ".debug_frame",
    ".nv_debug_line_sass",
    ".nv_debug_info_reg_sass",
  };

  bfd *abfd = objfile->obfd.get ();
  std::string result;
  for (const char *name : sections)
    {
      if (bfd_get_section_by_name (abfd, name) == nullptr)
	continue;
      if (!result.empty ())
	result += ' ';
      result += name;
    }

  return result.empty () ? "none" : result;
}

static std::string
format_virtual_sm (const std::optional<int64_t> &v)
{
  if (!v.has_value ())
    return std::string ();
  return string_printf ("compute_%ld", (long)*v);
}

static std::string
format_toolkit_version (const std::optional<std::string> &tool_version)
{
  /* Settled rule: extract "release X.Y" from tki_toolVersion (e.g. "Cuda
     compilation tools, release 13.4, V13.4.0" -> "13.4"). The integer
     nv_note_cuinfo_toolVersion is NOT decoded (encoding unconfirmed). */
  if (!tool_version.has_value ())
    return std::string ();
  static const std::regex re ("release ([0-9]+\\.[0-9]+)");
  std::smatch m;
  if (std::regex_search (*tool_version, m, re))
    return m[1].str ();
  return std::string ();
}

static std::string
format_compile_options (std::string options)
{
  /* Keep the detailed output on one line while preserving ordinary spacing
     in the recorded command line. */
  for (char &c : options)
    if (c != ' ' && std::isspace ((unsigned char)c))
      c = ' ';

  while (!options.empty () && options.back () == ' ')
    options.pop_back ();
  return options;
}

void
cuda_module::ensure_nvdisasm_metadata ()
{
  if (m_nv_metadata_state != nv_metadata_state::not_attempted)
    return;

  /* Pre-fill so getters never dereference a nullopt, even if a step throws. */
  m_nv_target_arch = std::string ();
  m_nv_virtual_sm = std::string ();
  m_nv_toolkit_version = std::string ();
  m_nv_tool_name = std::string ();
  m_nv_compile_options = std::string ();

  try
    {
      const std::string &path = sync_to_disk_and_get_file_path ();
      cuda_disasm_json::schema_metadata meta;
      if (cuda_read_module_metadata_json (path, meta))
	{
	  /* Target Architecture from the (required) SM.version, e.g. major=9
	     minor=0 -> "sm_90". Base arch only; accelerator suffix deferred. */
	  m_nv_target_arch = string_printf (
	      "sm_%ld%ld", (long)meta.m_sm_ver_major, (long)meta.m_sm_ver_minor);
	  m_nv_virtual_sm
	      = format_virtual_sm (meta.m_note_nv_cuda_virtual_sm_version);
	  m_nv_toolkit_version
	      = format_toolkit_version (meta.m_note_nv_toolkit_tool_version);
	  if (meta.m_note_nv_toolkit_tool_name.has_value ())
	    m_nv_tool_name = *meta.m_note_nv_toolkit_tool_name;
	  if (meta.m_note_nv_toolkit_tool_options.has_value ())
	    m_nv_compile_options
		= format_compile_options (*meta.m_note_nv_toolkit_tool_options);
	  m_nv_metadata_state = nv_metadata_state::ok;
	}
      else
	m_nv_metadata_state = nv_metadata_state::failed;
    }
  catch (const gdb_exception &e)
    {
      cuda_trace ("nvdisasm metadata extraction failed for module=0x%llx "
		  "elf_handle=0x%llx: %s",
		  m_module_id, m_elf_handle, e.what ());
      m_nv_metadata_state = nv_metadata_state::failed;
    }
}

const std::string &
cuda_module::get_nv_target_arch ()
{
  ensure_nvdisasm_metadata ();
  return *m_nv_target_arch;
}

const std::string &
cuda_module::get_nv_virtual_sm ()
{
  ensure_nvdisasm_metadata ();
  return *m_nv_virtual_sm;
}

const std::string &
cuda_module::get_nv_toolkit_version ()
{
  ensure_nvdisasm_metadata ();
  return *m_nv_toolkit_version;
}

const std::string &
cuda_module::get_nv_tool_name ()
{
  ensure_nvdisasm_metadata ();
  return *m_nv_tool_name;
}

const std::string &
cuda_module::get_nv_compile_options ()
{
  ensure_nvdisasm_metadata ();
  return *m_nv_compile_options;
}

// Read the ELF into the symbol table. Cubin bytes come from an iovec BFD
// over m_cubin_buffer, so loads and post-LFL reloads never touch /tmp.
// TODO: reconcile with the cuda_new_objfile observer in cuda-tdep.cu.
void
cuda_module::load_objfile ()
{
  gdb_assert (!m_loaded);

  // auto breakpoints
  s_current_module = this;

  gdb_assert (!m_bfd_name.empty ());

  auto opener = [this] (bfd *) -> gdb_bfd_iovec_base * {
    return new cuda_module_iovec (m_cubin_buffer.data (), m_size);
  };

  gdb_bfd_ref_ptr abfd
      = gdb_bfd_openr_iovec (m_bfd_name.c_str (), gnutarget, opener);
  if (abfd == nullptr)
    error ("Error: Failed to open device ELF image %s: %s",
	   m_bfd_name.c_str (), bfd_errmsg (bfd_get_error ()));

  // Explicit check: the iovec path skips the format auto-detection that
  // symfile_bfd_open would do.
  if (!bfd_check_format (abfd.get (), bfd_object))
    error ("Error: Device ELF image %s is not a valid object: %s",
	   m_bfd_name.c_str (), bfd_errmsg (bfd_get_error ()));

  const auto arch_info = bfd_lookup_arch (bfd_arch_m68k, 0);
  abfd->arch_info = arch_info;

  // Add the objfile's symbols without auto-resetting breakpoints.
  //
  // SYMFILE_NOT_FILENAME (default in-memory mode, synthetic
  // "<cuda-cubin:...>" name): sets OBJF_NOT_FILENAME so gdb won't lstat
  // the fake path, skipping two filesystem paths:
  //   - auto-load.c's "<name>-gdb.py" script lookup
  //   - dwarf2/index-write.c's .gdb_index cache (pointless: we rebuild
  //     from scratch every LFL event)
  // Under preserve_tmp_data the name is a real /tmp path, so leave it off.
  symfile_add_flags symfile_flags = SYMFILE_DEFER_BP_RESET;
  if (!cuda_options_preserve_tmp_data ())
    symfile_flags |= SYMFILE_NOT_FILENAME;

  m_objfile = symbol_file_add_from_bfd (abfd, bfd_get_filename (abfd.get ()),
					symfile_flags, nullptr, 0, nullptr);
  if (!m_objfile)
    error ("Error: Failed to add symbols from device ELF symbol file!\n");

  // Identify this gdb objfile as being cuda-specific
  m_objfile->cuda_objfile = true;

  m_objfile->per_bfd->gdbarch = cuda_get_gdbarch ();

  // Use the helper function to add sections
  foreach_cuda_objfile_section ([this] (asection *section) {
    // Add this section to the range map
    s_cuda_objfile_section_ranges.add (section->vma, section->size, this);
  });

  // cubin is now loaded
  m_loaded = true;
  m_uses_abi = cuda_is_bfd_version_call_abi (m_objfile->obfd.get ());
  cuda_trace (
      "loaded ELF image (name=%s, module=0x%llx, elf_handle=0x%llx, abi=%d)",
      m_bfd_name.c_str (), m_module_id, m_elf_handle, m_uses_abi);

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
  //
  // Derive the debug capability from the symbol tables GDB actually built,
  // not from raw ELF section presence: full DWARF if objfile_has_symbols(),
  // else line-info if the line-table decode reports it built a line program.
  // cuda_decode_line_table also makes -lineinfo cubins debuggable, so it must
  // run whenever there are no full symbols.
  bool full = objfile_has_symbols (m_objfile);
  bool line = !full && cuda_decode_line_table (m_objfile);
  m_debug_capability = full   ? cuda_debug_capability::full_debug
		       : line ? cuda_debug_capability::line_info
			      : cuda_debug_capability::none;
  m_debug_sections = compute_debug_sections (m_objfile);

  // Allow one nvdisasm retry against the newly relocated cubin: a prior
  // failure may have been due to the pre-relocation image. A successful
  // result is relocation-invariant and is kept. The retry is granted once
  // per module, not once per reload: FUNCTIONS_LOADED rebuilds the objfile
  // on every lazy function load, and a module that fails persistently would
  // otherwise spawn nvdisasm again after each one.
  if (m_nv_metadata_state == nv_metadata_state::failed && !m_nv_retry_consumed)
    {
      m_nv_retry_consumed = true;
      m_nv_metadata_state = nv_metadata_state::not_attempted;
    }

  // Don't try to update/set breakpoints for corefiles.
  if (target_has_execution ())
    {
      breakpoint_re_set ();
      // Need to manually update any forced auto breakpoint location
      // for this new objfile.
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

  cuda_trace (
      "unloading ELF image (name=%s, module=0x%llx, elf_handle=0x%llx)",
      m_objfile->original_name, m_module_id, m_elf_handle);

  // Mark this objfile as being discarded.
  m_objfile->discarding = true;

  // Use the helper function to remove sections from the range map.
  foreach_cuda_objfile_section ([] (asection *section) {
    /* Remove this section from the range map */
    s_cuda_objfile_section_ranges.remove_range (section->vma);
  });

  // Make sure that all its users will be cleaned up.
  clear_current_source_symtab_and_line (m_objfile);
  clear_displays ();

  // Unlink the /tmp cubin if requested; it may never have been
  // written, so ENOENT is fine.
  if (unlink_file && !cuda_options_preserve_tmp_data ())
    {
      unlink (m_file_path.c_str ());
      m_cubin_storage_stale = true;
    }

  // Tell the free_objfile observer if this is a permanent removal (stale
  // device breakpoint handles) or a temporary FUNCTIONS_LOADED rebuild
  // (handles still live).
  m_objfile->unlinked = unlink_file;

  // Request the objfile be destroyed
  m_objfile->unlink ();
  m_objfile = nullptr;

  m_loaded = false;
  m_uses_abi = false;
}

/* Fetch CUDBGLoadedFunctionInfo for this module's NEWLY_LOADED sections,
   hiding the API-version choice between the incremental (revision >= 135)
   and legacy whole-vector fetch. Returns a NEWLY_LOADED-length vector.  */

static std::vector<CUDBGLoadedFunctionInfo>
fetch_loaded_function_info (uint32_t device_id, uint64_t elf_handle,
			    uint64_t already_loaded, uint32_t total_count,
			    uint32_t newly_loaded)
{
  // Guard the std::copy below against overrun.
  gdb_assert (already_loaded <= total_count);
  gdb_assert (newly_loaded == total_count - already_loaded);

  std::vector<CUDBGLoadedFunctionInfo> result (newly_loaded);
  if (cuda_debugapi::api_version ().m_revision >= 135)
    {
      cuda_debugapi::get_loaded_function_info (
	  device_id, elf_handle, result.data (), already_loaded, newly_loaded);
    }
  else
    {
      // Older backend: no startIndex arg, so fetch the whole vector and
      // copy the newly-loaded tail.
      std::vector<CUDBGLoadedFunctionInfo> all (total_count);
      cuda_debugapi::get_loaded_function_info (device_id, elf_handle,
					       all.data (), total_count);
      std::copy (all.begin () + already_loaded, all.end (), result.begin ());
    }
  return result;
}

void
cuda_module::functions_loaded_event (uint32_t count)
{
  m_lazy_load_event_count++;

  auto device_id = context ()->dev_id ();
  auto already_loaded = functions_loaded ();

  // Prevent unsigned underflow of newly_loaded.
  gdb_assert (count >= already_loaded);
  auto newly_loaded = count - already_loaded;

  // Phase A: fetch per-section addresses for the new sections.
  std::vector<CUDBGLoadedFunctionInfo> info;
  {
    CUDA_LFL_STEP_TIMER (FETCH_INFO);
    info = fetch_loaded_function_info (device_id, m_elf_handle, already_loaded,
				       count, newly_loaded);
  }

  // Phase B: tear down the BFD/objfile before mutating its backing buffer.
  {
    CUDA_LFL_STEP_TIMER (UNLOAD);
    unload_objfile (false);
  }

  // Phase C: patch the cubin in place (sh_addr, then st_value for those
  // sections, then REL/RELA relocations). Safe: the BFD is down, so nothing
  // is reading m_cubin_buffer now.
  {
    CUDA_LFL_STEP_TIMER (APPLY_RELOCS);
    apply_function_load_updates (m_cubin_buffer.data (), info.data (),
				 newly_loaded);
  }

  m_functions_loaded += newly_loaded;

  // Phase D: refresh the /tmp cubin only under preserve_tmp_data; otherwise
  // sync_to_disk_and_get_file_path() writes it on demand when a tool needs it.
  {
    CUDA_LFL_STEP_TIMER (WRITE);
    if (cuda_options_preserve_tmp_data ())
      write_cubin ();
    else
      m_cubin_storage_stale = true;
  }

  // Phase E: reopen the iovec BFD over the patched buffer, re-parse DWARF.
  {
    CUDA_LFL_STEP_TIMER (LOAD);
    load_objfile ();
  }

  flush_disasm_caches (false);
}

void
cuda_module::print () const
{
  cuda_trace ("      module_id 0x%llx elf_handle 0x%llx size %llu file %s",
	      m_module_id, m_elf_handle, m_size, m_file_path.c_str ());
}

void
cuda_module::apply_function_load_updates (uint8_t *buffer,
					  CUDBGLoadedFunctionInfo *info,
					  uint32_t count)
{
  gdb_assert (buffer);
  gdb_assert (info);

  // Extract ELF Header information, section header table offset, number of
  // sections, and section header size.
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
      // The actual index of the section name string table section is contained
      // in the sh_link field of the section header at index 0.
      Elf64_External_Shdr *section = (Elf64_External_Shdr *)&buffer[shoff];
      shstrndx = *(uint64_t *)section->sh_link;
    }
  gdb_assert (shstrndx < shnum);

  Elf64_External_Shdr *shstrtab_section
      = (Elf64_External_Shdr *)&buffer[shoff + shstrndx * shentsize];
  uint64_t shstrtab_offset = *(uint64_t *)shstrtab_section->sh_offset;
  auto shstrtab = &buffer[shstrtab_offset];

  CUDA_TRACE_MODULE ("%u sections of %llu loaded", count, shnum);

  // Locate the strtab, symtab, and symtab section index sections.
  Elf64_External_Shdr *strtab = nullptr;
  Elf64_External_Shdr *symtab = nullptr;
  Elf64_External_Shdr *symtab_shndx = nullptr;
  for (int sectionidx = 0; sectionidx < shnum; ++sectionidx)
    {
      Elf64_External_Shdr *section
	  = (Elf64_External_Shdr *)&buffer[shoff + sectionidx * shentsize];
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
      // Skip out-of-range section indices to avoid overrunning buffer.
      if (section_index >= shnum)
	{
	  CUDA_TRACE_MODULE (
	      "Skipping out-of-range section index %llu (shnum %llu)",
	      (unsigned long long)section_index, (unsigned long long)shnum);
	  continue;
	}
      Elf64_External_Shdr *section
	  = (Elf64_External_Shdr *)&buffer[shoff + section_index * shentsize];

      CUDA_TRACE_MODULE (
	  "relocate section %lld %s from 0x%llx to 0x%llx", section_index,
	  (const char *)&shstrtab[*(uint32_t *)section->sh_name],
	  *(uint64_t *)section->sh_addr, info[i].address);

      // Check for newly relocated sections
      const uint64_t section_address = *(uint64_t *)section->sh_addr;
      if (section_address)
	{
	  if (section_address != info[i].address)
	    CUDA_TRACE_MODULE (
		"Skipping section %lld: mismatched addresses for section "
		"%lld "
		"%s: 0x%llx vs 0x%llx",
		section_index,
		(const char *)&shstrtab[*(uint32_t *)section->sh_name],
		section_address, info[i].address);
	  else
	    CUDA_TRACE_MODULE (
		"Skipping section %lld %s: already relocated addr 0x%llx",
		section_index,
		(const char *)&shstrtab[*(uint32_t *)section->sh_name],
		section_address);
	}
      else
	{
	  *(uint64_t *)section->sh_addr = info[i].address;
	  section_addresses[section_index] = info[i].address;

	  CUDA_TRACE_MODULE (
	      "relocated section %d %s addr 0x%llx", section_index,
	      (const char *)&shstrtab[*(uint32_t *)section->sh_name],
	      info[i].address);
	}
    }

  // Update symbol table values based on section addresses.
  std::vector<uint64_t> symbol_updated_address (symtab_size / symtab_entsize);
  for (int symidx = 0; symidx < symbol_updated_address.size (); ++symidx)
    {
      Elf64_External_Sym *symbol
	  = (Elf64_External_Sym
		 *)&buffer[symtab_offset + symidx * symtab_entsize];

      // Only relocate symbols which belong to newly located sections
      uint32_t sym_shndx = *(uint16_t *)symbol->st_shndx;

      // Check for st_shndx overflow.
      if (sym_shndx == SHN_XINDEX)
	{
	  // This cubin requires a symbol table section index table.
	  gdb_assert (symtab_shndx);
	  sym_shndx = *(uint32_t *)&buffer[symtab_shndx_offset
					   + symidx * symtab_shndx_entsize];
	}
      if (sym_shndx >= section_addresses.size ())
	{
	  CUDA_TRACE_MODULE ("Symbol %d section number out of range %u >= %lu",
			     symidx, sym_shndx, section_addresses.size ());
	  continue;
	}

      if (section_addresses[sym_shndx])
	{
	  uint64_t value = *(uint64_t *)symbol->st_value;

	  const auto symbol_name
	      = (const char
		     *)&buffer[strtab_offset + *(uint32_t *)symbol->st_name];
	  CUDA_TRACE_MODULE ("relocating symbol %d %s section %d from 0x%llx "
			     "plus 0x%llx to 0x%llx",
			     symidx, symbol_name, (uint32_t)sym_shndx, value,
			     section_addresses[sym_shndx],
			     value + section_addresses[sym_shndx]);

	  value += section_addresses[sym_shndx];

	  *(uint64_t *)symbol->st_value = value;
	  symbol_updated_address[symidx] = value;
	}
    }

  // Apply relocations based on symbols just relocated
  for (uint32_t reloc_section_idx = 0; reloc_section_idx < shnum;
       ++reloc_section_idx)
    {
      Elf64_External_Shdr *reloc_section
	  = (Elf64_External_Shdr
		 *)&buffer[shoff + reloc_section_idx * shentsize];
      uint32_t reloc_section_type = *(uint32_t *)reloc_section->sh_type;

      if ((reloc_section_type == SHT_RELA) || (reloc_section_type == SHT_REL))
	{
	  bool is_rela = (reloc_section_type == SHT_RELA) ? true : false;
	  const auto reloc_section_name
	      = (const char *)&shstrtab[*(uint32_t *)reloc_section->sh_name];

	  // The section we are relocating
	  uint32_t target_section_idx = *(uint32_t *)reloc_section->sh_info;
	  Elf64_External_Shdr *target_section
	      = (Elf64_External_Shdr
		     *)&buffer[shoff + target_section_idx * shentsize];
	  uint32_t target_section_flags
	      = *(uint64_t *)target_section->sh_flags;
	  const auto target_section_name
	      = (const char *)&shstrtab[*(uint32_t *)target_section->sh_name];

	  // The majority (all) of relocations in CODE / EXECINSTR sections are
	  // not supported Don't process relocations for these sections
	  if (target_section_flags & SHF_EXECINSTR)
	    {
	      CUDA_TRACE_MODULE (
		  "Skipping EXECINSTR section %d %s reloc section %s",
		  target_section_idx, target_section_name, reloc_section_name);
	      continue;
	    }

	  CUDA_TRACE_MODULE (
	      "Relocate %s section %s (%u) using reloc section %s (%u)",
	      is_rela ? "RELA" : "REL ", target_section_name,
	      target_section_idx, reloc_section_name, reloc_section_idx);

	  auto target_section_ptr
	      = &buffer[*(uint64_t *)target_section->sh_offset];

	  uint64_t reloc_section_size = *(uint64_t *)reloc_section->sh_size;
	  uint64_t reloc_section_offset
	      = *(uint64_t *)reloc_section->sh_offset;
	  for (uint64_t offset = reloc_section_offset;
	       offset < (reloc_section_offset + reloc_section_size);)
	    {
	      uint64_t r_offset = 0;
	      uint64_t r_info = 0;
	      uint64_t r_addend = 0;

	      if (is_rela)
		{
		  Elf64_External_Rela *rela
		      = (Elf64_External_Rela *)&buffer[offset];

		  r_offset = *(uint64_t *)rela->r_offset;
		  r_info = *(uint64_t *)rela->r_info;
		  r_addend = *(uint64_t *)rela->r_addend;

		  offset += sizeof (Elf64_External_Rela);
		}
	      else
		{
		  Elf64_External_Rel *rel
		      = (Elf64_External_Rel *)&buffer[offset];

		  r_offset = *(uint64_t *)rel->r_offset;
		  r_info = *(uint64_t *)rel->r_info;

		  // r_addend will be set below based on relocation type / size

		  offset += sizeof (Elf64_External_Rel);
		}

	      uint32_t symidx = ELF64_R_SYM (r_info);
	      if (symidx >= symbol_updated_address.size ())
		{
		  CUDA_TRACE_MODULE (
		      "Skipping unknown symbol index %u of 0x%llu", symidx,
		      symbol_updated_address.size ());
		  continue;
		}

	      // Don't apply relocations for symbols whose sections haven't
	      // been loaded yet
	      if (!symbol_updated_address[symidx])
		continue;

	      auto target_ptr = &target_section_ptr[r_offset];
	      switch (ELF64_R_TYPE (r_info))
		{
		case R_CUDA_32:
		case R_CUDA_G32:
		  if (!is_rela)
		    r_addend = *(uint32_t *)target_ptr;
		  *(uint32_t *)target_ptr
		      = (uint32_t)(symbol_updated_address[symidx] + r_addend);
		  break;

		case R_CUDA_64:
		case R_CUDA_G64:
		  if (!is_rela)
		    r_addend = *(uint64_t *)target_ptr;
		  *(uint64_t *)target_ptr
		      = (uint64_t)(symbol_updated_address[symidx] + r_addend);
		  break;

		default:
		  // Unknown relocation, skip
		  CUDA_TRACE_MODULE ("unknown relocation type %d",
				     (uint32_t)ELF64_R_TYPE (r_info));
		  break;
		}

	      // Log the relocation after it's been applied so that r_addend is
	      // correct for REL style relocations
	      if (cuda_options_trace_domain_enabled (CUDA_TRACE_ELF))
		{
		  Elf64_External_Sym *symbol
		      = (Elf64_External_Sym
			     *)&buffer[symtab_offset
				       + symidx * symtab_entsize];
		  const auto symbol_name
		      = (const char *)&buffer[strtab_offset
					      + *(uint32_t *)symbol->st_name];

		  CUDA_TRACE_MODULE (
		      "reloc %4s rtype %u tsect %u symidx %u r_off 0x%llx "
		      "symval 0x%llx radd 0x%llx symbol %s",
		      is_rela ? "RELA" : "REL ",
		      (uint32_t)ELF64_R_TYPE (r_info),
		      (uint32_t)target_section_idx, (uint32_t)symidx, r_offset,
		      symbol_updated_address[symidx], r_addend, symbol_name);
		}
	    }
	}
    }
}
