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

#ifndef _CUDA_MODULES_H
#define _CUDA_MODULES_H 1

#include "cuda-context.h"
#include "cuda-defs.h"
#include "cudadebugger.h"

#include "cuda-rangemap.h"

#include <map>
#include <unordered_map>

class cuda_context;
class cuda_module_disassembly_cache;

class cuda_module
{
public:
  cuda_module (uint64_t module_id, CUDBGElfImageProperties properties,
	       cuda_context *context, uint64_t elf_image_size);

  ~cuda_module ();

  uint64_t
  id () const
  {
    return m_id;
  }

  cuda_context *
  context () const
  {
    return m_context;
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

  const std::string &
  filename () const
  {
    return m_filename;
  }

  cuda_module_disassembly_cache *disassembler ();
  void flush_disasm_caches ();

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
  // Read / write cubins using m_filename
  // Buffer is guarenteed / assumed to be m_size bytes in length
  void read_cubin (void *buffer);
  void write_cubin (const void *buffer);

  // Helper function that correctly handles writing cubins that
  // are larger than the max number of bytes that can be written
  // in a single call to write() (just under 2G on Linux).
  static void write_buffer (int fd, const void *image, uint64_t len);

  // buffer is assumed to be m_size bytes in length
  static void apply_function_load_updates (uint8_t *buffer,
					   CUDBGLoadedFunctionInfo *info,
					   uint32_t count);

  // Helper function to add sections to the range map using a functor
  template <typename Func>
  inline void foreach_cuda_objfile_section (Func func);

  // module_id
  uint64_t m_id;

  // Property bitmask
  CUDBGElfImageProperties m_properties;

  // Context module is loaded in
  cuda_context *m_context;

  // Path to the ELF/cubin file in /tmp
  std::string m_filename;

  // The underlying objfile - may be null at certain points
  struct objfile *m_objfile;

  // The size of the relocated ELF image
  uint64_t m_size;

  // Is the ELF image in memory?
  bool m_loaded;

  // How many functions have been loaded through LFL so far
  uint64_t m_functions_loaded;

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
