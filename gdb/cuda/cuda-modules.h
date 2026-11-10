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

#ifndef _CUDA_MODULES_H
#define _CUDA_MODULES_H 1

#include "cuda-context.h"
#include "cuda-defs.h"
#include "cudadebugger.h"

#include "cuda-rangemap.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include "gdbsupport/byte-vector.h"

class cuda_context;
class cuda_module_disassembly_cache;

// Debugging support level of a loaded cubin, derived from the symbol tables
// GDB actually built (not raw ELF section presence). line_info corresponds to
// -lineinfo (.debug_line only), full_debug to -G (full DWARF); unknown means
// the objfile is not yet loaded. Rendered to a string in cuda-commands.c.
enum class cuda_debug_capability
{
  unknown,
  none,
  line_info,
  full_debug
};

class cuda_module
{
public:
  cuda_module (uint64_t module_id, uint64_t elf_handle,
	       CUDBGElfImageProperties properties, cuda_context *context,
	       uint64_t elf_image_size);

  ~cuda_module ();

  uint64_t
  module_id () const
  {
    return m_module_id;
  }

  uint64_t
  elf_handle () const
  {
    return m_elf_handle;
  }

  cuda_context *
  context () const
  {
    return m_context;
  }

  uint32_t
  dev_id () const
  {
    return m_context->dev_id ();
  }

  uint64_t
  context_id () const
  {
    return m_context->id ();
  }

  struct objfile *
  objfile () const
  {
    return m_objfile;
  }

  uint64_t
  size () const
  {
    return m_size;
  }

  bool
  loaded () const
  {
    return m_loaded;
  }

  uint64_t
  functions_loaded () const
  {
    return m_functions_loaded;
  }

  uint64_t
  lazy_load_event_count () const
  {
    return m_lazy_load_event_count;
  }

  bool
  uses_abi () const
  {
    return m_uses_abi;
  }

  CUDBGElfImageProperties
  properties () const
  {
    return m_properties;
  }

  bool
  system () const
  {
    return m_properties & CUDBG_ELF_IMAGE_PROPERTIES_SYSTEM;
  }

  // /tmp/cuda-dbg path for tools using a real cubin file (e.g. cuobjdump,
  // nvdisasm). Writes the file here if stale; a no-op under preserve_tmp_data,
  // which already keeps it current eagerly.
  const std::string &sync_to_disk_and_get_file_path ();

  // Stable BFD/objfile identity: synthetic by default, the real /tmp path
  // under preserve_tmp_data.
  const std::string &
  bfd_name () const
  {
    return m_bfd_name;
  }

  cuda_debug_capability
  get_debug_capability () const
  {
    return m_debug_capability;
  }
  const std::string &
  get_debug_sections () const
  {
    return m_debug_sections;
  }

  // nvdisasm-derived display strings, lazily extracted once and cached.
  // Empty string means "attempted but unavailable" (renders <unavailable>).
  const std::string &get_nv_target_arch ();
  const std::string &get_nv_virtual_sm ();
  const std::string &get_nv_toolkit_version ();
  const std::string &get_nv_tool_name ();
  const std::string &get_nv_compile_options ();

  cuda_module_disassembly_cache *disassembler ();
  void flush_disasm_caches (bool device_only);

  // Open a BFD over m_cubin_buffer, parse DWARF, re-resolve breakpoints.
  void load_objfile ();
  void unload_objfile (bool unlink_file);

  void functions_loaded_event (uint32_t count);

  // Static class methods
  static void add_kernel_entry (CORE_ADDR addr);

  static void auto_breakpoints_update_locations ();

  static cuda_module *find_cuda_module_by_address (CORE_ADDR addr);

  // For internal debugging
  void print () const;

private:
  // Write m_cubin_buffer to the on-disk cubin at m_file_path.
  void write_cubin ();

  // Patch BUFFER (the cubin image, m_size bytes) for the COUNT sections in
  // INFO: assign sh_addr from the LFL addresses, update st_value for those
  // sections' symbols, then apply CUDA REL/RELA relocations.
  static void apply_function_load_updates (uint8_t *buffer,
					   CUDBGLoadedFunctionInfo *info,
					   uint32_t count);

  // Helper function to add sections to the range map using a functor
  template <typename Func>
  inline void foreach_cuda_objfile_section (Func func);

  // module_id
  uint64_t m_module_id;

  // elf_handle
  uint64_t m_elf_handle;

  // Property bitmask
  CUDBGElfImageProperties m_properties;

  // Context module is loaded in
  cuda_context *m_context;

  // BFD identity string: synthetic by default (BFD reads m_cubin_buffer);
  // the real /tmp path under preserve_tmp_data for out-of-band inspection.
  std::string m_bfd_name;

  // Deterministic /tmp/cuda-dbg path for subprocesses (cuobjdump,
  // nvdisasm). Fixed at construction even if the file is missing or stale.
  std::string m_file_path;

  // m_file_path's file is missing or stale vs m_cubin_buffer. Cleared by
  // write_cubin.
  bool m_cubin_storage_stale;

  // In-memory cubin image (m_size bytes), owned here and patched in place
  // by apply_function_load_updates on each LFL event. The iovec BFD reads
  // sections from here, so patches are visible on reopen without disk I/O.
  gdb::byte_vector m_cubin_buffer;

  // The underlying objfile - may be null at certain points
  struct objfile *m_objfile;

  // The size of the relocated ELF image
  uint64_t m_size;

  // Is the ELF image in memory?
  bool m_loaded;

  // How many functions have been loaded through LFL so far
  uint64_t m_functions_loaded;

  // How many CUDBG_EVENT_FUNCTIONS_LOADED events this module has encountered
  uint64_t m_lazy_load_event_count;

  // Debug support level, computed once per load_objfile from the symbol
  // tables GDB built. unknown until the objfile is loaded.
  cuda_debug_capability m_debug_capability = cuda_debug_capability::unknown;

  // Space-separated list of allowlisted debug sections present on the objfile
  // BFD, computed once per load_objfile. "none" when the list is empty.
  std::string m_debug_sections;

  // nvdisasm-derived display strings, lazily populated once. Empty string means
  // "attempted but unavailable" (renders as <unavailable>).
  std::optional<std::string> m_nv_target_arch;
  std::optional<std::string> m_nv_virtual_sm;
  std::optional<std::string> m_nv_toolkit_version;
  std::optional<std::string> m_nv_tool_name;
  std::optional<std::string> m_nv_compile_options;

  enum class nv_metadata_state
  {
    not_attempted,
    ok,
    failed
  };
  nv_metadata_state m_nv_metadata_state = nv_metadata_state::not_attempted;

  // A failure is retried once against the relocated cubin (see load_objfile).
  // Set when that retry is granted, so later reloads do not keep re-spawning
  // nvdisasm for a module that fails persistently.
  bool m_nv_retry_consumed = false;

  // Run nvdisasm once and fill the five optionals; safe to call repeatedly.
  void ensure_nvdisasm_metadata ();

  // Does the ELF image uses the ABI to call functions
  bool m_uses_abi;

  // Optional disassembler
  std::unique_ptr<cuda_module_disassembly_cache> m_disassembler;

  // Class members
  static cuda_module *s_current_module;
  static std::multimap<uint64_t, CORE_ADDR> s_kernel_entry_points;
  static cuda_rangemap<cuda_module *> s_cuda_objfile_section_ranges;
};

#endif
