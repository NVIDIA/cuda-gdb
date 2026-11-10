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

#ifndef _CUDA_ASM_H
#define _CUDA_ASM_H 1

#include <cstdint>

#include "objfiles.h"

#include "cuda-context.h"
#include "cuda-state.h"
#include "cuda-tdep.h"

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class cuda_device;
class cuda_module;

/* Disassembly format / source combinations.
 * NVSAL disassembly is only available for SM 100 and later and from ELF
 * images. SASS disassembly is available from ELF images and device memory.
 */
// Default instruction size is 16 bytes.
constexpr uint32_t cuda_default_instruction_size = 16;

enum class cuda_instruction_attribute : uint32_t
{
  none = 0,
  control_flow = 1 << 0,
  subroutine_call = 1 << 1,
  inter_warp_barrier = 1 << 2,
  padding = 1 << 3,
};

constexpr cuda_instruction_attribute
operator| (cuda_instruction_attribute a, cuda_instruction_attribute b)
{
  return static_cast<cuda_instruction_attribute> (static_cast<uint32_t> (a)
						  | static_cast<uint32_t> (b));
}

constexpr cuda_instruction_attribute &
operator|= (cuda_instruction_attribute &a, cuda_instruction_attribute b)
{
  a = a | b;
  return a;
}

constexpr cuda_instruction_attribute
operator& (cuda_instruction_attribute a, cuda_instruction_attribute b)
{
  return static_cast<cuda_instruction_attribute> (static_cast<uint32_t> (a)
						  & static_cast<uint32_t> (b));
}

inline constexpr uint32_t cuda_disassembly_type_count
    = static_cast<uint32_t> (cuda_disassembly_type_t::sass_device) + 1;

/******************************************************************************
 *
 *                             Disassembly Cache
 *
 *****************************************************************************/

class cuda_disassembly_annotation
{
public:
  enum class annotation_type
  {
    NONE,
    STRING,
    INTEGER,
    BOOLEAN,
  };

  cuda_disassembly_annotation ()
      : m_name (""), m_type (annotation_type::NONE), m_string (""),
	m_integer (0), m_boolean (false)
  {
  }

  cuda_disassembly_annotation (const std::string &name,
			       const std::string &value)
      : m_name (name), m_type (annotation_type::STRING), m_string (value),
	m_integer (0), m_boolean (false)
  {
  }

  const std::string &
  name () const
  {
    return m_name;
  }

  annotation_type
  type () const
  {
    return m_type;
  }

  const std::string &
  string () const
  {
    gdb_assert (type () == annotation_type::STRING);
    return m_string;
  }

  int64_t
  integer () const
  {
    gdb_assert (type () == annotation_type::INTEGER);
    return m_integer;
  }

  bool
  boolean () const
  {
    gdb_assert (type () == annotation_type::BOOLEAN);
    return m_boolean;
  }

private:
  std::string m_name;
  annotation_type m_type;
  std::string m_string;
  int64_t m_integer;
  bool m_boolean;
};

class cuda_instruction_disassembly_info
{
public:
  cuda_instruction_disassembly_info (const std::string &predicate,
				     const std::string &opcode,
				     const std::string &operands,
				     const std::string &extra,
				     uint32_t insn_size)
      : m_valid (true), m_insn_size (insn_size), m_predicate (predicate),
	m_opcode (opcode), m_operands (operands), m_extra (extra)
  {
  }

  explicit cuda_instruction_disassembly_info (const std::string &opcode,
					      uint32_t insn_size)
      : m_valid (true), m_insn_size (insn_size), m_opcode (opcode)
  {
  }

  cuda_instruction_disassembly_info () : m_valid (false) {}

  /* Copy/Move constructor and assignment operator */
  cuda_instruction_disassembly_info (
      const cuda_instruction_disassembly_info &other)
      = default;
  cuda_instruction_disassembly_info &
  operator= (const cuda_instruction_disassembly_info &other)
      = default;
  cuda_instruction_disassembly_info (cuda_instruction_disassembly_info &&other)
      = default;
  cuda_instruction_disassembly_info &
  operator= (cuda_instruction_disassembly_info &&other)
      = default;

  bool
  is_valid () const
  {
    return m_valid;
  }

  void
  invalidate ()
  {
    m_valid = false;
  }

  uint32_t
  insn_size () const
  {
    return m_insn_size;
  }

  const std::string &
  predicate () const
  {
    gdb_assert (is_valid ());
    return m_predicate;
  }

  const std::string &
  opcode () const
  {
    gdb_assert (is_valid ());
    return m_opcode;
  }

  const std::string &
  operands () const
  {
    gdb_assert (is_valid ());
    return m_operands;
  }

  const std::string &
  extra () const
  {
    gdb_assert (is_valid ());
    return m_extra;
  }

  std::string to_string () const;

private:
  bool m_valid;
  /* instruction size */
  uint32_t m_insn_size;
  /* predicate */
  std::string m_predicate;
  /* opcode */
  std::string m_opcode;
  /* operands */
  std::string m_operands;
  /* instruction annotation */
  std::string m_extra;
};

class cuda_instruction
{
public:
  /* Constructor for empty instructions with no disassembly info */
  cuda_instruction (uint64_t pc = 0)
      : m_pc{ pc }, m_instruction_count{ 1 }, m_instruction_index{ 0 },
	m_info{}
  {
  }

  /* Constructor for structured instructions from JSON output.
     ATTRIBUTES come from JSON other-attributes when present; nullopt means
     omit them so eval_is_*() can classify from opcode text.  */
  cuda_instruction (uint64_t pc, uint32_t instruction_count,
		    uint32_t instruction_index,
		    std::optional<cuda_instruction_attribute> attributes,
		    cuda_disassembly_type_t type,
		    cuda_instruction_disassembly_info &&info)
      : m_pc{ pc }, m_instruction_count{ instruction_count },
	m_instruction_index{ instruction_index }, m_attributes{ attributes },
	m_info{}
  {
    gdb_assert (static_cast<uint32_t> (type) < cuda_disassembly_type_count);
    m_info[static_cast<uint32_t> (type)] = std::move (info);
  }

  /* Used for SASS Device disassembly only. Only the opcode is provided and the
     boolean flags for control flow, subroutine call, and barrier are left
     unset as they will be evaluated on demand later. */
  explicit cuda_instruction (uint64_t pc, const std::string &disassembly)
      : m_pc{ pc }, m_instruction_count{ 1 }, m_instruction_index{ 0 },
	m_info{}
  {
    m_info[static_cast<uint32_t> (cuda_disassembly_type_t::sass_device)]
	= cuda_instruction_disassembly_info (std::move (disassembly),
					     cuda_default_instruction_size);
  }

  /* Copy/Move constructor and assignment operator */
  cuda_instruction (const cuda_instruction &other) = default;
  cuda_instruction &operator= (const cuda_instruction &other) = default;
  cuda_instruction (cuda_instruction &&other) = default;
  cuda_instruction &operator= (cuda_instruction &&other) = default;

  /* Public API */

  // Returns the human-readable string representation of the instruction in the
  // CUDA GDB format.
  std::string to_string (cuda_disassembly_type_t type) const;

  uint64_t
  pc () const
  {
    return m_pc;
  }

  void
  set_pc (uint64_t pc)
  {
    m_pc = pc;
  }

  uint32_t
  count () const
  {
    return m_instruction_count;
  }

  uint32_t
  index () const
  {
    return m_instruction_index;
  }

  void
  set_count_index (uint32_t count, uint32_t index)
  {
    m_instruction_count = count;
    m_instruction_index = index;
  }

  void
  set_padding ()
  {
    if (m_attributes.has_value ())
      m_attributes = *m_attributes | cuda_instruction_attribute::padding;
    else
      m_attributes = cuda_instruction_attribute::padding;
  }

  bool
  is_padding () const
  {
    return m_attributes.has_value ()
	   && (*m_attributes & cuda_instruction_attribute::padding)
		  != cuda_instruction_attribute::none;
  }

  std::optional<cuda_instruction_attribute>
  attributes () const
  {
    return m_attributes;
  }

  void
  set_attributes (const cuda_instruction_attribute attributes)
  {
    m_attributes = attributes;
  }

  const std::vector<cuda_disassembly_annotation> &
  annotations () const
  {
    return m_annotations;
  }

  void
  add_annotation (const cuda_disassembly_annotation &annotation)
  {
    m_annotations.push_back (annotation);
  }

  cuda_disassembly_annotation *
  lookup_annotation (const std::string &name)
  {
    for (auto &annotation : m_annotations)
      if (annotation.name () == name)
	return &annotation;
    return nullptr;
  }

  /* Get the disassembly info for the given type */
  const cuda_instruction_disassembly_info *
  disassembly_info (const cuda_disassembly_type_t type) const
  {
    gdb_assert (static_cast<uint32_t> (type) < cuda_disassembly_type_count);
    return &m_info[static_cast<uint32_t> (type)];
  }

  const cuda_instruction_disassembly_info *
  first_valid_disassembly_info () const;

  /* Set the disassembly info for the given type */
  void
  set_disassembly_info (const cuda_disassembly_type_t type,
			const cuda_instruction_disassembly_info &info)
  {
    gdb_assert (static_cast<uint32_t> (type) < cuda_disassembly_type_count);
    m_info[static_cast<uint32_t> (type)] = info;
  }

  void
  invalidate_disassembly_info (const cuda_disassembly_type_t type)
  {
    gdb_assert (static_cast<uint32_t> (type) < cuda_disassembly_type_count);
    m_info[static_cast<uint32_t> (type)].invalidate ();
  }

  bool
  is_barrier ()
  {
    eval_and_set_attributes ();
    return m_attributes.has_value ()
	   && (*m_attributes & cuda_instruction_attribute::inter_warp_barrier)
		  != cuda_instruction_attribute::none;
  }

  bool
  is_control_flow ()
  {
    eval_and_set_attributes ();
    return m_attributes.has_value ()
	   && (*m_attributes & cuda_instruction_attribute::control_flow)
		  != cuda_instruction_attribute::none;
  }

  bool
  is_subroutine_call ()
  {
    eval_and_set_attributes ();
    return m_attributes.has_value ()
	   && (*m_attributes & cuda_instruction_attribute::subroutine_call)
		  != cuda_instruction_attribute::none;
  }

  bool
  is_placeholder () const
  {
    return first_valid_disassembly_info () == nullptr;
  }

private:
  /* Classify from opcode text when attributes were left unset (no JSON
     other-attributes, or DebugAPI SASS).  Uses any valid cached disassembly.
   */
  void eval_and_set_attributes ();
  bool eval_is_control_flow ();
  bool eval_is_subroutine_call ();
  bool eval_is_barrier ();

  /* PC of the instruction */
  uint64_t m_pc;

  /* Number of spanned SASS instructions (from NVSAL disassembly). */
  uint32_t m_instruction_count;

  /* Instruction index within the function */
  uint32_t m_instruction_index;

  /* Attributes such as control flow, subroutine call, and barrier. */
  std::optional<cuda_instruction_attribute> m_attributes;

  /* Annotations */
  std::vector<cuda_disassembly_annotation> m_annotations;

  /* Disassembly info for the different types (NVSAL ELF, SASS ELF, SASS
   * device)*/
  std::array<cuda_instruction_disassembly_info, cuda_disassembly_type_count>
      m_info;
};

class cuda_function
{
public:
  /* Constructor for a function with placeholder instructions */
  cuda_function (const std::string &name, const uint64_t start_address,
		 const uint64_t length);

  cuda_function (const std::string &name, const uint64_t start_address,
		 const uint64_t length,
		 std::vector<cuda_instruction> &&instructions);

  const std::string &
  name () const
  {
    return m_name;
  }

  uint64_t
  start_address () const
  {
    return m_start_address;
  }

  uint64_t
  end_address () const
  {
    return m_start_address + m_length;
  }

  uint64_t
  length () const
  {
    return m_length;
  }

  /* True if PC falls in this function's address range but past the capped
     instruction cache (disassembly was truncated).  */
  bool pc_beyond_cached_instructions (uint64_t pc) const;

  /* Grow the function's address span in place, extending the instruction
     vector with placeholders up to the per-function cache cap.  Preserves
     already-cached instructions, annotations, and disassembly status.  */
  void ensure_span (uint64_t span);

  /* Shrink the address span so end_address () <= END.  Drops trailing
     placeholder/instruction slots that fall past the new end.  */
  void clamp_end (uint64_t end);

  /* True if a previous NVSAL disassembly attempt for this function failed,
     so callers should skip straight to the SASS fallback. */
  std::optional<bool>
  disassembly_status (cuda_disassembly_type_t type) const
  {
    return m_disassembly_status[static_cast<uint32_t> (type)];
  }

  void
  set_disassembly_status (cuda_disassembly_type_t type,
			  std::optional<bool> status)
  {
    m_disassembly_status[static_cast<uint32_t> (type)] = status;
  }

  cuda_instruction *add_instruction (const cuda_instruction &instruction);

  const std::vector<cuda_instruction> &
  instructions () const
  {
    return m_instructions;
  }

  cuda_instruction *lookup_instruction (const uint64_t pc);

  cuda_instruction *lookup_lead_instruction (const uint64_t pc);

  uint64_t
  next_disassembly_pc (const uint64_t pc)
  {
    auto *lead_instruction = lookup_lead_instruction (pc);
    if (!lead_instruction)
      return pc + cuda_default_instruction_size;
    return pc + lead_instruction->count () * cuda_default_instruction_size;
  }

  const std::vector<cuda_disassembly_annotation> &
  annotations () const
  {
    return m_annotations;
  }

  void
  add_annotation (const cuda_disassembly_annotation &annotation)
  {
    m_annotations.push_back (annotation);
  }

  cuda_disassembly_annotation *
  lookup_annotation (const std::string &name)
  {
    for (auto &annotation : m_annotations)
      if (annotation.name () == name)
	return &annotation;
    return nullptr;
  }

  void flush_cache (bool device_only);

private:
  std::string m_name;
  uint64_t m_start_address;
  uint64_t m_length;

  /* Indicates whether different disassembly types have been attempted and
   * succeeded or failed. */
  std::array<std::optional<bool>, cuda_disassembly_type_count>
      m_disassembly_status;

  std::vector<cuda_instruction> m_instructions;

  /* Annotations */
  std::vector<cuda_disassembly_annotation> m_annotations;
};

class cuda_module_disassembly_cache
{
public:
  explicit cuda_module_disassembly_cache (cuda_module *module)
      : m_device (cuda_state::device (module->context ()->dev_id ())),
	m_module (module)
  {
  }

  /* Simple interface using global settings w/ fallback */

  const cuda_instruction_disassembly_info *disassemble (const uint64_t pc);

  /* Range-map lookup: returns the function whose [start, end) covers PC,
     or nullptr if PC falls in a gap or before the first known function. */
  cuda_function *lookup_function (const uint64_t pc);

  /* Lookup instruction in the cache */
  cuda_instruction *lookup_instruction (const uint64_t pc, bool disasm);

  void flush_cache (bool device_only);

  /* Get the disassembly type for the module */
  cuda_disassembly_type_t default_disassembly_type () const;

private:
  /* Insert FUNCTION into the range map keyed by start_address ().
     Replaces any existing entry with the same start address.  If another
     function already covers the start under a different key, reuses that
     entry (growing as needed) instead of inserting an overlap.  Clamps the
     new span so it does not cross the next mapped function's start.
     Returns a non-owning pointer to the stored cuda_function. */
  cuda_function *
  add_function_to_cache (std::unique_ptr<cuda_function> function);

  /* Cap SPAN so [start, start+span) does not cross the next function. */
  uint64_t clamp_span_to_next_function (uint64_t start, uint64_t span) const;

  /* Disassemble instruction with specific type - does not
   * fallback to SASS if NVSAL disassembly failed */
  const cuda_instruction_disassembly_info *
  disassemble (const uint64_t pc, const cuda_disassembly_type_t type);

  bool parse_nvsal_elf (const std::string &function_name, const uint64_t pc);
  bool parse_sass_elf (const std::string &function_name, const uint64_t pc);
  bool parse_sass_device (const std::string &function_name, const uint64_t pc);

  bool parse_json (int fd, cuda_disassembly_type_t type);

  bool function_bounds (const std::string &func_name, const uint64_t pc,
			uint64_t &start_pc, uint64_t &length);

  cuda_device *m_device;
  cuda_module *m_module;

  std::map<uint64_t, std::unique_ptr<cuda_function>> m_function_map;
};

/* Forward declaration to avoid pulling cuda-sass-json.h into this header. */
namespace cuda_disasm_json
{
struct schema_metadata;
}

/* Spawn `nvdisasm --emit-json -ndf -fun 1 <file_path>`, parse the full (small,
   -fun-bounded) JSON document, and copy the metadata into OUT. Returns true iff
   the parse succeeded AND nvdisasm exited 0. Does not throw. */
bool cuda_read_module_metadata_json (const std::string &file_path,
				     cuda_disasm_json::schema_metadata &out);

#endif
