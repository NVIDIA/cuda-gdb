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


#ifndef _CUDA_ASM_H
#define _CUDA_ASM_H 1

#include "defs.h"
#include "cuda-defs.h"

#include <regex>
#include <string>
#include <unordered_map>

/******************************************************************************
 *
 *                             Disassembly Cache
 *
 *****************************************************************************/

class cuda_disassembler
{
public:
  cuda_disassembler ();
  
  bool disassemble (uint64_t pc, std::string& insn, uint32_t& insn_size);

  void flush_elf_cache (void) { m_elf_map.clear (); }
  void flush_device_cache (void) { m_device_map.clear (); }

 private:
  enum line_type
  {
    LINE_TYPE_UNKNOWN,
    LINE_TYPE_FUNC_HEADER,
    LINE_TYPE_OFFSET_INSN,
    LINE_TYPE_CODE_ONLY,
    LINE_TYPE_EOF
  };

  /* parser state */
  FILE *m_sass;
  struct objfile *m_objfile;
  uint64_t m_current_offset;

  std::string m_current_func;
  std::string m_current_insn;
  std::string m_current_line;

  /* Holder for pc->insn mappings from the cubin/ELF file.
     These may be updated by lazy function loading, so
     clear on resume. */
  std::unordered_map<uint64_t, std::string> m_elf_map;

  /* Holder for pc->insn mappings from the device.
     These are flushed before resuming the GPU, as the
     code may be modified after resuming. */
  std::unordered_map<uint64_t, std::string> m_device_map;

  /* regex for parsing various line types */
  std::regex m_offset_insn_line;
  std::regex m_code_line;
  std::regex m_func_header_line;

  std::string m_cuobjdump_path;

  void parse_disasm_output ();
  enum line_type parse_next_line ();

  bool populate_from_elf_image (uint64_t pc, std::string& insn);
  bool populate_from_device_memory (uint64_t pc, std::string& insn);

  const std::string find_cuobjdump ();
  const std::string find_executable (const std::string& name);

  static bool exists(const std::string& fname);
};

#endif
