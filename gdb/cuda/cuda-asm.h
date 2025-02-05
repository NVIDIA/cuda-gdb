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

#ifndef _CUDA_ASM_H
#define _CUDA_ASM_H 1

#include "gdbsupport/gdb_optional.h"
#include "objfiles.h"

#include "cuda-defs.h"

#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

/******************************************************************************
 *
 *                             Disassembly Cache
 *
 *****************************************************************************/

class cuda_module;

class cuda_instruction
{
public:
  /* Constructor for structured instructions from JSON output */
  inline cuda_instruction (const std::string &prefix,
			   const std::string &opcode,
			   const std::string &operands,
			   const std::string &extra)
      : m_predicate (prefix), m_opcode (opcode), m_operands (operands),
	m_extra (extra), m_is_control_flow (is_control_flow_value::unset)
  {
  }

  inline cuda_instruction (const std::string &prefix,
			   const std::string &opcode,
			   const std::string &operands,
			   const std::string &extra, bool is_control_flow)
      : m_predicate (prefix), m_opcode (opcode), m_operands (operands),
	m_extra (extra),
	m_is_control_flow (is_control_flow
			       ? is_control_flow_value::true_value
			       : is_control_flow_value::false_value)
  {
  }

  /* Constructor for raw instructions from text output */
  inline cuda_instruction (const std::string &raw_instruction)
      : m_predicate (), m_opcode (raw_instruction), m_operands (), m_extra (),
	m_is_control_flow (is_control_flow_value::unset)
  {
  }

  /* Move assignment operator */
  inline cuda_instruction& operator=(const cuda_instruction& other)
  {
    const_cast<std::string&>(m_predicate) = other.m_predicate;
    const_cast<std::string&>(m_opcode) = other.m_opcode;
    const_cast<std::string&>(m_operands) = other.m_operands;
    const_cast<std::string&>(m_extra) = other.m_extra;
    m_is_control_flow = other.m_is_control_flow;
    return *this;
  }

  cuda_instruction () = delete;

  /* Public API */
  /* Returns the human-readable string representation of the instruction in the
     CUDA GDB format. The output is expected to match the cuobjdump output:
     {opcode} {operands} {extra} IADD32I R1, R1, -0x8 ; CALL.ABS
     `(urf_kernel_set) ; */
  std::string to_string () const;

  bool is_control_flow (const bool skip_subroutines);

private:
  /* The instruction offset is not stored in the instruction object as it could
     be calculated at runtime given the instruction number in the function
     object and the SM arch. */
  /* predicate */
  const std::string m_predicate;
  /* opcode */
  const std::string m_opcode;
  /* operands */
  const std::string m_operands;
  /* instruction annotation */
  const std::string m_extra;
  /* is it a control flow instruction */
  enum class is_control_flow_value
  {
    unset = 0,
    true_value,
    false_value
  };
  is_control_flow_value m_is_control_flow;
  is_control_flow_value m_is_control_flow_skipping_subroutines;
  /* The list and the dictionary of additional string attributes are not stored
     in the instruction object. These string attributes shall be transformed by
     the JSON parser into domain specific fields. */

  bool eval_is_control_flow (const bool skip_subroutines);
};

class cuda_function
{
public:
  cuda_function (const std::string &name, const uint64_t startAddress,
		 const uint64_t length,
		 std::vector<cuda_instruction> &&instructions)
      : m_name (name), m_start_address (startAddress), m_length (length),
	m_instructions (std::move (instructions))
  {
  }

  inline const std::string &
  name () const
  {
    return m_name;
  }

  inline uint64_t
  start_address () const
  {
    return m_start_address;
  }

  inline uint64_t
  length () const
  {
    return m_length;
  }

  inline const std::vector<cuda_instruction> &
  instructions () const
  {
    return m_instructions;
  }

private:
  std::string m_name;
  uint64_t m_start_address;
  uint64_t m_length;
  std::vector<cuda_instruction> m_instructions;
};

class cuda_module_disassembly_cache
{
public:
  cuda_module_disassembly_cache (uint32_t insn_size, bool is_volta_plus)
      : m_insn_size (insn_size), m_is_volta_plus (is_volta_plus),
	m_cuobjdump_json (true)
  {
  }

  enum class disassembly_source
  {
    ELF = 0,
    DEVICE
  };

  /* Public API */
  gdb::optional<cuda_instruction> disassemble_instruction (uint64_t pc);

  inline void
  flush_elf_cache ()
  {
    m_elf_map.clear ();
  }

  inline void
  flush_device_cache ()
  {
    m_device_map.clear ();
  }

  inline uint32_t
  insn_size () const
  {
    return m_insn_size;
  }

private:
  /* helper methods */
  void add_function_to_cache (const cuda_function &function,
			      disassembly_source source);

  /* populate the cache */
  gdb::optional<cuda_instruction> populate_from_elf_image (uint64_t pc);
  gdb::optional<cuda_instruction> populate_from_device_memory (uint64_t pc);

  /* parse the disassembly output */
  bool parse_disasm_output_json (int fd);
  void parse_disasm_output (int fd, cuda_module* module);

  gdb::optional<cuda_instruction>
  disassemble_instruction (uint64_t pc, disassembly_source source);

  gdb::optional<cuda_instruction>
  cache_lookup (uint64_t pc, disassembly_source source) const;

  uint32_t m_insn_size;
  bool m_is_volta_plus;
  /* Set to false if cuobjdump doesn't support -json */
  bool m_cuobjdump_json;

  /* Holder for pc->insn mappings from the cubin/ELF file.
     These are flushed at the end of LFL event processing. */
  std::unordered_map<uint64_t, cuda_instruction> m_elf_map;

  /* Holder for pc->insn mappings from the device.
     These are flushed at the end of LFL event processing. */
  std::unordered_map<uint64_t, cuda_instruction> m_device_map;
};

#endif
