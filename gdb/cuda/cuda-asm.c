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

#include "main.h"

#include "gdbsupport/common-utils.h"

#include "cuda-api.h"
#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-modules.h"
#include "cuda-options.h"
#include "cuda-sass-json.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-util-stream.h"

#include "block.h"
#include "symtab.h"

#include <algorithm>
#include <cstdint>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#if !defined(__QNX__)
#include <sys/prctl.h>
#endif
#include <sys/types.h>
#include <sys/wait.h>

extern char **environ;

#define MAX_BUFFER_SIZE 4096

#define CUDA_ERR_IF(_cond, _domain, _fmt, ...)                                \
  do                                                                          \
    {                                                                         \
      if (_cond)                                                              \
	{                                                                     \
	  cuda_trace_domain (_domain, _fmt, ##__VA_ARGS__);                   \
	  error (_fmt, ##__VA_ARGS__);                                        \
	}                                                                     \
    }                                                                         \
  while (0)

// Create an array of char pointers. The strings still own the pointers. Do not
// free/edit them.
static std::unique_ptr<const char *[]>
vector_to_argv (const std::vector<std::string> &args)
{
  auto argv = std::make_unique<const char *[]> (args.size () + 1);
  const char **p_argv = argv.get ();
  for (size_t i = 0; i < args.size (); ++i)
    {
      p_argv[i] = args[i].c_str ();
    }
  p_argv[args.size ()] = nullptr;
  return argv;
}

/* Search for executable in PATH, cuda-gdb launch folder or current folder */
static bool
exists (const std::string &fname)
{
  struct stat buf;
  return stat (fname.c_str (), &buf) == 0;
}

static constexpr size_t cuda_disassembly_max_fallback = 3;

/* Maximum number of instructions retained for a single CUDA function.
   Disassembly past this limit is dropped; lookups in the truncated tail
   print "<out-of-range>".  */
static constexpr size_t cuda_function_max_instructions = 1024 * 1024;

static const cuda_instruction_disassembly_info
    cuda_out_of_range_disassembly_info ("<out-of-range>",
					cuda_default_instruction_size);

/* Fill OUT_TYPES with the disassembly types to try for DEVICE, in order.
   Returns the number of entries written (1..cuda_disassembly_max_fallback). */
static size_t
cuda_disassembly_fallback_types (cuda_device *device,
				 cuda_disassembly_type_t *out_types,
				 size_t out_capacity)
{
  gdb_assert (out_capacity >= cuda_disassembly_max_fallback);

  const auto requested = cuda_options_disassembly_type ();
  const bool nvsal_capable = device->get_sm_version () >= 100;
  const bool can_use_device = target_has_execution ();
  size_t n = 0;

  switch (requested)
    {
    case cuda_disassembly_type_t::nvsal_elf:
      if (nvsal_capable)
	out_types[n++] = cuda_disassembly_type_t::nvsal_elf;
      out_types[n++] = cuda_disassembly_type_t::sass_elf;
      if (can_use_device)
	out_types[n++] = cuda_disassembly_type_t::sass_device;
      break;

    case cuda_disassembly_type_t::sass_elf:
      out_types[n++] = cuda_disassembly_type_t::sass_elf;
      if (can_use_device)
	out_types[n++] = cuda_disassembly_type_t::sass_device;
      break;

    case cuda_disassembly_type_t::sass_device:
      if (can_use_device)
	out_types[n++] = cuda_disassembly_type_t::sass_device;
      else
	/* Core dumps / no live target: device memory is unavailable. */
	out_types[n++] = cuda_disassembly_type_t::sass_elf;
      break;

    default:
      error ("Unknown disassembly type %u", static_cast<uint32_t> (requested));
    }

  return n;
}

/* Returns the current preferred disassembly type for a device. */
static cuda_disassembly_type_t
cuda_disassembly_type (cuda_device *device)
{
  cuda_disassembly_type_t types[cuda_disassembly_max_fallback];
  const size_t n = cuda_disassembly_fallback_types (
      device, types, cuda_disassembly_max_fallback);
  gdb_assert (n > 0);
  return types[0];
}

/******************************************************************************
 *
 *            CUDA Disassembly Info (one each for SASS and NVSAL)
 *
 *****************************************************************************/

std::string
cuda_instruction_disassembly_info::to_string () const
{
  gdb_assert (is_valid ());
  std::string str;

  if (!m_predicate.empty ())
    str += m_predicate + " ";

  str += m_opcode;

  if (!m_operands.empty ())
    str += " " + m_operands;

  if (!m_extra.empty ())
    str += " " + m_extra;

  return str;
}

static const char *
cuda_disassembly_type_to_string (cuda_disassembly_type_t type)
{
  switch (type)
    {
    case cuda_disassembly_type_t::nvsal_elf:
      return "NVSAL_ELF";
    case cuda_disassembly_type_t::sass_elf:
      return "SASS_ELF";
    case cuda_disassembly_type_t::sass_device:
      return "SASS_DEVICE";
    default:
      error ("Unknown disassembly type %u", static_cast<uint32_t> (type));
    }
}

/******************************************************************************
 *
 *                             CUDA Instruction
 *
 *****************************************************************************/

std::string
cuda_instruction::to_string (cuda_disassembly_type_t type) const
{
  gdb_assert (static_cast<uint32_t> (type) < cuda_disassembly_type_count);
  const auto *info = disassembly_info (type);
  if (!info || !info->is_valid ())
    error ("%s failed to get valid disassembly info for type %s", __func__,
	   cuda_disassembly_type_to_string (type));
  return info->to_string ();
}

void
cuda_instruction::eval_and_set_attributes ()
{
  if (!m_attributes.has_value ())
    {
      auto attributes = cuda_instruction_attribute::none;
      if (eval_is_control_flow ())
	attributes = attributes | cuda_instruction_attribute::control_flow;
      if (eval_is_subroutine_call ())
	attributes = attributes | cuda_instruction_attribute::subroutine_call;
      if (eval_is_barrier ())
	attributes
	    = attributes | cuda_instruction_attribute::inter_warp_barrier;
      if (attributes != cuda_instruction_attribute::none)
	m_attributes = attributes;
    }
}

const cuda_instruction_disassembly_info *
cuda_instruction::first_valid_disassembly_info () const
{
  /* Prefer NVSAL, then SASS ELF, then device text.  Used for placeholder
     detection and tracing; DebugAPI opcode matching uses SASS_DEVICE only.  */
  static constexpr cuda_disassembly_type_t order[] = {
    cuda_disassembly_type_t::nvsal_elf,
    cuda_disassembly_type_t::sass_elf,
    cuda_disassembly_type_t::sass_device,
  };

  for (auto type : order)
    {
      const auto *info = disassembly_info (type);
      if (info && info->is_valid ())
	return info;
    }
  return nullptr;
}

bool
cuda_instruction::eval_is_control_flow ()
{
  const auto &info
      = m_info[static_cast<size_t> (cuda_disassembly_type_t::sass_device)];
  if (!info.is_valid ())
    return false;

  const char *inst_str = info.opcode ().c_str ();

  /* Turing+:
   * https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#turing */
  /* BRXU - covered with BRX */
  /* JMXU - covered with JMX */
  if (strstr (inst_str, "BRA") != 0)
    return true;
  if (strstr (inst_str, "BRX") != 0)
    return true;
  if (strstr (inst_str, "JMP") != 0)
    return true;
  if (strstr (inst_str, "JMX") != 0)
    return true;
  if (strstr (inst_str, "RET") != 0)
    return true;
  if (strstr (inst_str, "BRK") != 0)
    return true;
  if (strstr (inst_str, "CONT") != 0)
    return true;
  if (strstr (inst_str, "SSY") != 0)
    return true;
  if (strstr (inst_str, "BPT") != 0)
    return true;
  if (strstr (inst_str, "EXIT") != 0)
    return true;
  if (strstr (inst_str, "SYNC") != 0)
    return true;
  if (strstr (inst_str, "BREAK") != 0)
    return true;
  /* BSYNC - covered with SYNC */
  if (strstr (inst_str, "KILL") != 0)
    return true;
  if (strstr (inst_str, "NANOSLEEP") != 0)
    return true;
  if (strstr (inst_str, "RTT") != 0)
    return true;
  if (strstr (inst_str, "WARPSYNC") != 0)
    return true;
  if (strstr (inst_str, "YIELD") != 0)
    return true;
  if (strstr (inst_str, "BMOV") != 0)
    return true;
  if (strstr (inst_str, "RPCMOV") != 0)
    return true;
  /* Hopper+:
   * https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#hopper */
  if (strstr (inst_str, "ACQBULK") != 0)
    return true;
  if (strstr (inst_str, "ENDCOLLECTIVE") != 0)
    return true;
  return false;
}

bool
cuda_instruction::eval_is_subroutine_call ()
{
  const auto &info
      = m_info[static_cast<size_t> (cuda_disassembly_type_t::sass_device)];
  if (!info.is_valid ())
    return false;

  const char *inst_str = info.opcode ().c_str ();

  if (strstr (inst_str, "CALL") != 0)
    return true;
  return false;
}

bool
cuda_instruction::eval_is_barrier ()
{
  const auto &info
      = m_info[static_cast<size_t> (cuda_disassembly_type_t::sass_device)];
  if (!info.is_valid ())
    return false;

  const char *inst_str = info.opcode ().c_str ();

  /* MEMBAR, DEPBAR, UCGABAR_* - covered with BAR */

  if (strstr (inst_str, "BAR") != 0)
    return true;
  return false;
}

/******************************************************************************
 *
 *                             CUDA Function
 *
 *****************************************************************************/

cuda_function::cuda_function (const std::string &name,
			      const uint64_t start_address,
			      const uint64_t length)
    : m_name (name), m_start_address (start_address), m_length (length)
{
  size_t count = length / cuda_default_instruction_size;
  if (count > cuda_function_max_instructions)
    {
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Function %s length 0x%lx exceeds max instruction "
			 "cache (%zu); truncating",
			 name.c_str (), length,
			 cuda_function_max_instructions);
      count = cuda_function_max_instructions;
    }
  m_instructions.resize (count);
  for (size_t index = 0; index < m_instructions.size (); index++)
    m_instructions[index].set_pc (start_address
				  + (index * cuda_default_instruction_size));
}

cuda_function::cuda_function (const std::string &name,
			      const uint64_t start_address,
			      const uint64_t length,
			      std::vector<cuda_instruction> &&instructions)
    : m_name (name), m_start_address (start_address), m_length (length),
      m_instructions (std::move (instructions))
{
  if (m_instructions.size () > cuda_function_max_instructions)
    {
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Function %s instruction list exceeds max cache "
			 "(%zu); dropping tail",
			 name.c_str (), cuda_function_max_instructions);
      m_instructions.resize (cuda_function_max_instructions);
    }
}

void
cuda_function::flush_cache (bool device_only)
{
  if (device_only)
    {
      for (auto &instruction : m_instructions)
	instruction.invalidate_disassembly_info (
	    cuda_disassembly_type_t::sass_device);
      return;
    }

  /* Keep the map entry but drop the instruction vector and length together
     so PCs are not reported as beyond a truncated cache.  Module-level full
     flushes clear m_function_map instead of calling this path.  */
  m_instructions.clear ();
  m_length = 0;
  m_disassembly_status.fill (std::nullopt);
}

bool
cuda_function::pc_beyond_cached_instructions (uint64_t pc) const
{
  if (pc < start_address () || pc >= end_address ()
      || (pc % cuda_default_instruction_size) != 0)
    return false;
  const auto index = (pc - start_address ()) / cuda_default_instruction_size;
  return index >= m_instructions.size ();
}

void
cuda_function::ensure_span (uint64_t span)
{
  if (span <= m_length)
    return;

  if (span % cuda_default_instruction_size != 0)
    span += cuda_default_instruction_size
	    - (span % cuda_default_instruction_size);

  m_length = span;

  const auto old_size = m_instructions.size ();
  const auto new_size = std::min (
      static_cast<size_t> (m_length / cuda_default_instruction_size),
      cuda_function_max_instructions);
  if (new_size <= old_size)
    return;

  m_instructions.resize (new_size);
  for (auto i = old_size; i < m_instructions.size (); ++i)
    m_instructions[i].set_pc (m_start_address
			      + i * cuda_default_instruction_size);
}

void
cuda_function::clamp_end (uint64_t end)
{
  if (end <= m_start_address)
    {
      m_length = 0;
      m_instructions.clear ();
      return;
    }

  uint64_t span = end - m_start_address;
  if (span % cuda_default_instruction_size != 0)
    span -= span % cuda_default_instruction_size;
  if (span >= m_length)
    return;

  m_length = span;
  const auto new_size = std::min (
      static_cast<size_t> (m_length / cuda_default_instruction_size),
      cuda_function_max_instructions);
  if (new_size < m_instructions.size ())
    m_instructions.resize (new_size);
}

cuda_instruction *
cuda_function::lookup_instruction (const uint64_t pc)
{
  if (pc < start_address () || pc >= end_address ()
      || (pc % cuda_default_instruction_size) != 0)
    return nullptr;
  const auto index = (pc - start_address ()) / cuda_default_instruction_size;
  if (index >= m_instructions.size ())
    return nullptr;
  gdb_assert (m_instructions[index].pc () == pc);
  return &m_instructions[index];
}

cuda_instruction *
cuda_function::add_instruction (const cuda_instruction &instruction)
{
  if (instruction.pc () < start_address ()
      || (instruction.pc () % cuda_default_instruction_size) != 0)
    return nullptr;

  /* Grow the function address range if JSON/device disassembly extends past
     the original bounds (e.g. short length in the JSON header).  */
  if (instruction.pc () >= end_address ())
    {
      m_length = instruction.pc () - m_start_address
		 + cuda_default_instruction_size;
      if (m_length % cuda_default_instruction_size != 0)
	m_length += cuda_default_instruction_size
		    - (m_length % cuda_default_instruction_size);
    }

  auto index
      = (instruction.pc () - start_address ()) / cuda_default_instruction_size;
  /* Drop instructions past the per-function cache cap.  */
  if (index >= cuda_function_max_instructions)
    return nullptr;

  if (index >= m_instructions.size ())
    {
      const auto old_size = m_instructions.size ();
      const auto new_size = std::min (
	  static_cast<size_t> (m_length / cuda_default_instruction_size),
	  cuda_function_max_instructions);
      m_instructions.resize (new_size);
      for (auto i = old_size; i < m_instructions.size (); ++i)
	m_instructions[i].set_pc (m_start_address
				  + i * cuda_default_instruction_size);
    }

  if (index >= m_instructions.size ())
    return nullptr;

  auto &existing = m_instructions[index];
  existing.set_pc (instruction.pc ());

  /* Preserve count/index/annotations when revisiting a PC that already has
     disassembly (e.g. parse_sass_device after NVSAL/SASS ELF). */
  bool had_disasm = false;
  for (uint32_t t = 0; t < cuda_disassembly_type_count; ++t)
    {
      const auto info = existing.disassembly_info (
	  static_cast<cuda_disassembly_type_t> (t));
      if (info && info->is_valid ())
	{
	  had_disasm = true;
	  break;
	}
    }

  for (uint32_t t = 0; t < cuda_disassembly_type_count; ++t)
    {
      const auto type = static_cast<cuda_disassembly_type_t> (t);
      const auto *info = instruction.disassembly_info (type);
      if (info && info->is_valid ())
	existing.set_disassembly_info (type, *info);
    }

  if (instruction.attributes ().has_value ())
    {
      if (existing.attributes ().has_value ())
	existing.set_attributes (*existing.attributes ()
				 | *instruction.attributes ());
      else
	existing.set_attributes (*instruction.attributes ());
    }

  if (!had_disasm)
    existing.set_count_index (instruction.count (), instruction.index ());

  for (const auto &annotation : instruction.annotations ())
    if (existing.lookup_annotation (annotation.name ()) == nullptr)
      existing.add_annotation (annotation);

  return &existing;
}

cuda_instruction *
cuda_function::lookup_lead_instruction (const uint64_t pc)
{
  if (pc_beyond_cached_instructions (pc))
    return nullptr;
  const auto *instruction = lookup_instruction (pc);
  if (!instruction)
    return nullptr;
  const auto index_pc
      = (pc - start_address ()) / cuda_default_instruction_size;
  const auto index_lead = index_pc - instruction->index ();
  gdb_assert (index_lead < m_instructions.size ());
  auto *lead_instruction = &m_instructions[index_lead];
  return lead_instruction;
}

/******************************************************************************
 *
 *                             CUDA Module Disassembly Cache
 *
 *****************************************************************************/

cuda_disassembly_type_t
cuda_module_disassembly_cache::default_disassembly_type () const
{
  return cuda_disassembly_type (m_device);
}

void
cuda_module_disassembly_cache::flush_cache (bool device_only)
{
  if (device_only)
    for (auto &iter : m_function_map)
      iter.second->flush_cache (device_only);
  else
    m_function_map.clear ();
}

cuda_function *
cuda_module_disassembly_cache::lookup_function (const uint64_t pc)
{
  /* upper_bound gives the first function whose start_address > pc;
     stepping back yields the candidate whose start_address <= pc.
     We then verify pc is actually within that function's range, since
     the candidate may sit before a gap. */
  auto it = m_function_map.upper_bound (pc);
  if (it == m_function_map.begin ())
    return nullptr;
  --it;
  cuda_function *func = it->second.get ();
  if (pc >= func->end_address ())
    return nullptr;
  return func;
}

cuda_instruction *
cuda_module_disassembly_cache::lookup_instruction (const uint64_t pc,
						   bool disasm)
{
  if (disasm && !disassemble (pc))
    return nullptr;
  auto function = lookup_function (pc);
  if (!function)
    return nullptr;
  auto instruction = function->lookup_instruction (pc);
  if (!instruction)
    return nullptr;
  if (instruction->is_placeholder ())
    return nullptr;
  return instruction;
}

/* Disassemble instruction with specific type - does not
 * fallback to SASS if NVSAL disassembly failed */
const cuda_instruction_disassembly_info *
cuda_module_disassembly_cache::disassemble (const uint64_t pc,
					    const cuda_disassembly_type_t type)
{
  auto function = lookup_function (pc);
  if (function && function->pc_beyond_cached_instructions (pc))
    return &cuda_out_of_range_disassembly_info;

  auto instruction = function ? function->lookup_instruction (pc) : nullptr;
  if (instruction)
    {
      const auto *info = instruction->disassembly_info (type);
      if (info && info->is_valid ())
	return info;
    }

  // Could not find requested disassembly info. For ELF-backed types, skip
  // retry if a previous attempt for this function already failed.
  if (type != cuda_disassembly_type_t::sass_device && function)
    {
      const auto status = function->disassembly_status (type);
      if (status.has_value () && !status.value ())
	return nullptr;
    }

  /* Need to disassemble the instruction (function) */

  // Demangling not needed here, use cuda_find_function_name_from_pc
  auto function_name = cuda_find_function_name_from_pc (pc, false);
  if (function_name.get ())
    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
		       "Attempting disassembly for function: %s",
		       function_name.get ());
  else
    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
		       "Could not find function name for pc 0x%lx, "
		       "disassembling whole module.",
		       pc);
  std::string function_name_str
      = function_name ? function_name.get () : std::string ();

  bool success = false;
  switch (type)
    {
    case cuda_disassembly_type_t::nvsal_elf:
      success = parse_nvsal_elf (function_name_str, pc);
      break;
    case cuda_disassembly_type_t::sass_elf:
      success = parse_sass_elf (function_name_str, pc);
      break;
    case cuda_disassembly_type_t::sass_device:
      success = parse_sass_device (function_name_str, pc);
      break;
    default:
      error ("Unknown disassembly type %u", static_cast<uint32_t> (type));
    }

  // Need to look up "function" again as we may have added it to the cache in
  // the parse_* function.
  function = lookup_function (pc);
  if (function && function->pc_beyond_cached_instructions (pc))
    return &cuda_out_of_range_disassembly_info;

  if ((type != cuda_disassembly_type_t::sass_device) && function)
    function->set_disassembly_status (type, std::optional<bool> (success));

  if (!success || !function
      || !(instruction = function->lookup_instruction (pc)))
    {
      // Either parsing failed or we didn't find the instruction in the cache
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Failed to disassemble instruction at pc 0x%lx for "
			 "type %s",
			 pc, cuda_disassembly_type_to_string (type));
      return nullptr;
    }

  return instruction->disassembly_info (type);
}

/* Simple interface using `set cuda disassembly` with type-specific fallback.
 */
const cuda_instruction_disassembly_info *
cuda_module_disassembly_cache::disassemble (const uint64_t pc)
{
  cuda_disassembly_type_t types[cuda_disassembly_max_fallback];
  const size_t n = cuda_disassembly_fallback_types (
      m_device, types, cuda_disassembly_max_fallback);

  for (size_t i = 0; i < n; ++i)
    {
      auto info = disassemble (pc, types[i]);
      if (info && info->is_valid ())
	return info;
    }

  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
		     "Failed to disassemble instruction at pc 0x%lx "
		     "(requested type %s)",
		     pc, cuda_disassembly_type_to_string (types[0]));
  return nullptr;
}

cuda_function *
cuda_module_disassembly_cache::add_function_to_cache (
    std::unique_ptr<cuda_function> function)
{
  const uint64_t key = function->start_address ();
  uint64_t end = function->end_address ();

  /* Avoid overlapping range-map entries when JSON and DebugAPI disagree on
     the function start.  Reuse the covering function and grow if needed.  */
  cuda_function *covering = lookup_function (key);
  if (covering != nullptr && covering->start_address () != key)
    {
      uint64_t needed = end - covering->start_address ();
      needed
	  = clamp_span_to_next_function (covering->start_address (), needed);
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Reusing function %s [0x%lx,0x%lx) instead of "
			 "overlapping insert for %s at 0x%lx",
			 covering->name ().c_str (),
			 covering->start_address (), covering->end_address (),
			 function->name ().c_str (), key);
      if (needed > covering->length ())
	covering->ensure_span (needed);
      return covering;
    }

  const uint64_t clamped_span
      = clamp_span_to_next_function (key, function->length ());
  if (clamped_span < function->length ())
    {
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Clamping function %s end from 0x%lx to 0x%lx to "
			 "avoid overlap with the next mapped function",
			 function->name ().c_str (), end, key + clamped_span);
      function->clamp_end (key + clamped_span);
    }

  auto [it, inserted]
      = m_function_map.insert_or_assign (key, std::move (function));
  return it->second.get ();
}

uint64_t
cuda_module_disassembly_cache::clamp_span_to_next_function (
    uint64_t start, uint64_t span) const
{
  if (span == 0)
    return 0;

  auto next = m_function_map.upper_bound (start);
  if (next == m_function_map.end ())
    return span;

  if (start + span <= next->first)
    return span;

  if (next->first <= start)
    return 0;

  uint64_t clamped = next->first - start;
  if (cuda_default_instruction_size != 0
      && (clamped % cuda_default_instruction_size) != 0)
    clamped -= clamped % cuda_default_instruction_size;
  return clamped;
}

class posix_spawn_file_actions
{
  posix_spawn_file_actions_t m_file_actions;
  bool m_initialized;

public:
  posix_spawn_file_actions () : m_initialized (false)
  {
    const int ret = posix_spawn_file_actions_init (&m_file_actions);
    if (ret != 0)
      error (_ ("Failed to initialize posix_spawn_file_actions: %s"),
	     safe_strerror (ret));
    m_initialized = true;
  }

  ~posix_spawn_file_actions ()
  {
    if (m_initialized)
      posix_spawn_file_actions_destroy (&m_file_actions);
  }

  DISABLE_COPY_AND_ASSIGN (posix_spawn_file_actions);

  posix_spawn_file_actions_t *
  get ()
  {
    if (!m_initialized)
      error (_ ("posix_spawn_file_actions not initialized"));
    return &m_file_actions;
  }

  /* Configure redirection of a pipe's write-end to a target FD in the child
     process, and ensure the unused ends are closed appropriately.
     Throws gdb_exception_error on failure. */
  void
  add_redirect (int child_read_fd, int child_write_fd, int target_fd)
  {
    if (!m_initialized)
      error (_ ("posix_spawn_file_actions not initialized"));

    /* Close parent's read-end in the child process. */
    int ret
	= posix_spawn_file_actions_addclose (&m_file_actions, child_read_fd);
    CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		 "disasm_process: add close failed: %s", safe_strerror (ret));

    /* Duplicate the pipe's write-end onto the requested FD. */
    ret = posix_spawn_file_actions_adddup2 (&m_file_actions, child_write_fd,
					    target_fd);
    CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		 "disasm_process: add dup2 failed: %s", safe_strerror (ret));

    /* Close the original write-end in the child after dup2. */
    ret = posix_spawn_file_actions_addclose (&m_file_actions, child_write_fd);
    CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		 "disasm_process: addclose failed for write end: %s",
		 safe_strerror (ret));
  }
};

class disasm_process
{
  pid_t m_pid;
  int m_stdout_rfd; // Read end for parent (stdout)
  int m_stderr_rfd; // Read end for parent (stderr)
  int m_stdout_wfd; // Write end for child (stdout)
  int m_stderr_wfd; // Write end for child (stderr)
  posix_spawn_file_actions
      m_file_actions; // Prepared file actions holding redirections
  bool m_quiet;
  bool m_is_nvsal_elf;

  /* Safely close a file descriptor */
  static void
  safe_close (int &fd)
  {
    if (fd != -1)
      {
	close (fd);
	fd = -1;
      }
  }

public:
  // Constructor: create pipes and pre-configure file actions so that exec()
  // only has to spawn the process.
  disasm_process (bool quiet, bool is_nvsal_elf)
      : m_pid (-1), m_stdout_rfd (-1), m_stderr_rfd (-1), m_stdout_wfd (-1),
	m_stderr_wfd (-1), m_file_actions (), m_quiet (quiet),
	m_is_nvsal_elf (is_nvsal_elf)
  {
    int stdout_fds[2] = { -1, -1 };
    int stderr_fds[2] = { -1, -1 };

    if (pipe (stdout_fds) == -1)
      CUDA_ERR_IF (true, CUDA_TRACE_DISASSEMBLER,
		   "disasm_process: stdout pipe failed: %s",
		   safe_strerror (errno));

    m_stdout_rfd = stdout_fds[0];
    m_stdout_wfd = stdout_fds[1];

    if (pipe (stderr_fds) == -1)
      {
	safe_close (m_stdout_rfd);
	safe_close (m_stdout_wfd);
	CUDA_ERR_IF (true, CUDA_TRACE_DISASSEMBLER,
		     "disasm_process: stderr pipe failed: %s",
		     safe_strerror (errno));
      }

    m_stderr_rfd = stderr_fds[0];
    m_stderr_wfd = stderr_fds[1];

    /* Set up redirection for stdout/stderr for the spawned
    process. Configures the file actions to dup the write end of the pipe to
    the target FD in the child and close unused FDs. */
    m_file_actions.add_redirect (m_stdout_rfd, m_stdout_wfd, STDOUT_FILENO);
    m_file_actions.add_redirect (m_stderr_rfd, m_stderr_wfd, STDERR_FILENO);
  }

  // Destructor: Ensures cleanup if wait() wasn't called explicitly
  ~disasm_process ()
  {
    try
      {
	if (m_pid > 0)
	  wait ();
      }
    catch (const gdb_exception &e)
      {
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   ("Ignoring exception in disasm_process dtor: %s"),
			   e.what ());
      }

    safe_close (m_stdout_rfd);
    safe_close (m_stderr_rfd);
    safe_close (m_stdout_wfd);
    safe_close (m_stderr_wfd);
  }

  DISABLE_COPY_AND_ASSIGN (disasm_process);

  /* Returns the file descriptor for reading the child's stdout. */
  int
  stdout_read_fd () const
  {
    return m_stdout_rfd;
  }

  /* Stop managing the stdout read descriptor after ownership transfers. */
  void
  disown_stdout_read_fd ()
  {
    gdb_assert (m_stdout_rfd != -1);
    m_stdout_rfd = -1;
  }

  /* Reads the entire contents of the child's stderr stream.
     Should be called after the process has likely finished writing stderr,
     typically before or during wait. */
  std::string
  get_stderr ()
  {
    if (m_stderr_rfd == -1)
      return std::string ();

    const int STDERR_BUFSZ = 512;
    std::string str;
    ssize_t nr;

    // Use non-blocking read to avoid hanging if stderr is large or process
    // hangs
    const int flags = fcntl (m_stderr_rfd, F_GETFL, 0);
    if (flags != -1)
      fcntl (m_stderr_rfd, F_SETFL, flags | O_NONBLOCK);

    gdb::unique_xmalloc_ptr<char> buff ((char *)xmalloc (STDERR_BUFSZ));
    str.reserve (STDERR_BUFSZ);

    do
      {
	nr = read (m_stderr_rfd, buff.get (), STDERR_BUFSZ);
	if (nr > 0)
	  str.append (buff.get (), nr);
	else if (nr < 0)
	  {
	    if (errno == EAGAIN || errno == EWOULDBLOCK)
	      // No more data available right now.
	      // We assume stderr is fully read if we hit this.
	      // For a robust solution, might need select/poll before read.
	      break;
	    else
	      {
		// Actual read error
		cuda_trace_domain (
		    CUDA_TRACE_DISASSEMBLER,
		    "disasm_process: failed to read from stderr: %s",
		    safe_strerror (errno));
		break;
	      }
	  }
      }
    while (nr > 0);

    // Restore original flags if changed
    if (flags != -1)
      fcntl (m_stderr_rfd, F_SETFL, flags);

    return str;
  }

  void
  log_message (const char *message, ...)
  {
    va_list va_args;
    va_start (va_args, message);
    if (m_quiet)
      cuda_vtrace_domain (CUDA_TRACE_DISASSEMBLER, message, va_args);
    else
      vwarning (message, va_args);
    va_end (va_args);
  }

  /* Executes the disassembler with the given parameters. Sets up pipes for
     stdout/stderr, spawns the process, and closes unnecessary FDs in the
     parent. Throws gdb_exception_error on failure. */
  void
  exec (const std::string &filename, const char *function_name)
  {
    // Check if already active by inspecting pid
    if (m_pid > 0)
      error ("disasm_process::exec called on already active object.");

    // Reset pid state only; file descriptors are prepared at construction.
    m_pid = -1;

    std::string disasm_binary;
    std::vector<std::string> disasm_args;
    if (m_is_nvsal_elf)
      {
	disasm_binary = "nvsal-dis";
	disasm_args.push_back (disasm_binary);
	disasm_args.push_back ("-json");
	disasm_args.push_back (filename);
      }
    else
      {
	disasm_binary = "cuobjdump";
	disasm_args.push_back (disasm_binary);
	disasm_args.push_back ("-json");
	disasm_args.push_back ("--dump-sass");

	if (function_name && function_name[0])
	  {
	    disasm_args.push_back ("--function");
	    disasm_args.push_back (function_name);
	  }

	disasm_args.push_back (filename);
      }
    if (cuda_options_trace_domain_enabled (CUDA_TRACE_DISASSEMBLER))
      {
	std::string args_str;
	for (const auto &arg : disasm_args)
	  {
	    args_str.append (" ");
	    args_str.append (arg);
	  }
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "disassembler command (ELF): %s",
			   args_str.c_str ());
      }

    try
      {
	auto argv = vector_to_argv (disasm_args);

	// First spawn attempt using PATH
	int ret = posix_spawnp (
	    &m_pid, disasm_binary.c_str (), m_file_actions.get (), nullptr,
	    const_cast<char *const *> (argv.get ()), environ);

	if (ret != 0)
	  {
	    // Fallback spawn attempt if first failed (e.g., not in PATH)
	    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			       "posix_spawnp failed for '%s' (errno %d: %s), "
			       "trying path relative to gdb",
			       disasm_binary.c_str (), ret,
			       safe_strerror (ret));

	    const std::string gdb_path = get_gdb_program_name ();
	    const auto slash_idx = gdb_path.rfind ("/");
	    CUDA_ERR_IF (slash_idx == std::string::npos,
			 CUDA_TRACE_DISASSEMBLER,
			 "Failed to find gdb directory for %s fallback: %s",
			 disasm_binary.c_str (), gdb_path.c_str ());

	    const std::string gdb_bin_dir = gdb_path.substr (0, slash_idx);
	    const std::string tool_path = gdb_bin_dir + "/" + disasm_binary;

	    // Check existence before attempting spawn with full path
	    CUDA_ERR_IF (!exists (tool_path), CUDA_TRACE_DISASSEMBLER,
			 "%s not found at '%s' for fallback spawn",
			 disasm_binary.c_str (), tool_path.c_str ());

	    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			       "Attempting fallback spawn with path: %s",
			       tool_path.c_str ());

	    argv.get ()[0] = tool_path.c_str ();
	    // Use posix_spawn, not spawnp, as we have the full path
	    ret = posix_spawn (
		&m_pid, tool_path.c_str (), m_file_actions.get (), nullptr,
		const_cast<char *const *> (argv.get ()), environ);
	  }

	// Final check after potentially two spawn attempts. If ret != 0, spawn
	// failed.
	CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		     "Failed to spawn %s (posix_spawn/p errno %d): %s",
		     disasm_binary.c_str (), ret, safe_strerror (ret));

	// If spawn succeeded:
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "%s spawned with pid %d",
			   disasm_binary.c_str (), static_cast<int> (m_pid));

	// Close the parent's write ends of the pipes now that spawn is done.
	// This is crucial - must happen *after* spawn but *before* wait.
	safe_close (m_stdout_wfd);
	safe_close (m_stderr_wfd);

	// Note: Parent's read ends (m_stdout_rfd, m_stderr_rfd) remain open
	// until wait or destructor.
      }
    catch (gdb_exception &e)
      {
	// Cleanup FDs if an exception occurred during setup
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "Exception during %s exec setup: %s",
			   disasm_binary.c_str (), e.what ());
	safe_close (m_stdout_rfd);
	safe_close (m_stdout_wfd);
	safe_close (m_stderr_rfd);
	safe_close (m_stderr_wfd);
	// Ensure state reflects inactivity if exec failed
	m_pid = -1;
	// Re-throw the exception
	throw;
      }
  }

  /* Additive sibling of exec(): spawns
     `nvdisasm --emit-json -ndf -fun 1 <file_path>` reusing this object's pipes
     and file-actions. Does NOT alter exec() or the SASS path. Throws on spawn
     failure (via CUDA_ERR_IF), matching exec(). */
  void
  exec_nvdisasm_metadata (const std::string &file_path)
  {
    if (m_pid > 0)
      error ("disasm_process::exec_nvdisasm_metadata called on active object.");
    m_pid = -1;

    std::string nvdisasm_str ("nvdisasm");
    std::vector<std::string> nvdisasm_args
	= { nvdisasm_str, "--emit-json", "-ndf", "-fun", "1", file_path };

    if (cuda_options_trace_domain_enabled (CUDA_TRACE_DISASSEMBLER))
      {
	std::string args_str;
	for (const auto &arg : nvdisasm_args)
	  {
	    args_str.append (" ");
	    args_str.append (arg);
	  }
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "nvdisasm metadata command:%s", args_str.c_str ());
      }

    try
      {
	auto argv = vector_to_argv (nvdisasm_args);

	// First spawn attempt using PATH.
	int ret = posix_spawnp (
	    &m_pid, nvdisasm_str.c_str (), m_file_actions.get (), NULL,
	    const_cast<char *const *> (argv.get ()), environ);

	if (ret != 0)
	  {
	    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			       "posix_spawnp failed for '%s' (errno %d: %s), "
			       "trying gdb bin dir fallback",
			       nvdisasm_str.c_str (), ret, safe_strerror (ret));

	    const std::string gdb_path = get_gdb_program_name ();
	    const auto slash_idx = gdb_path.find_last_of ('/');
	    CUDA_ERR_IF (slash_idx == std::string::npos, CUDA_TRACE_DISASSEMBLER,
			 "Failed to find gdb dir for nvdisasm fallback: %s",
			 gdb_path.c_str ());

	    const std::string gdb_bin_dir = gdb_path.substr (0, slash_idx);
	    const std::string nvdisasm_path = gdb_bin_dir + "/nvdisasm";
	    CUDA_ERR_IF (!exists (nvdisasm_path), CUDA_TRACE_DISASSEMBLER,
			 "nvdisasm not found at '%s' for fallback spawn",
			 nvdisasm_path.c_str ());

	    argv.get ()[0] = nvdisasm_path.c_str ();
	    ret = posix_spawn (
		&m_pid, nvdisasm_path.c_str (), m_file_actions.get (), NULL,
		const_cast<char *const *> (argv.get ()), environ);
	  }

	CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		     "Failed to spawn nvdisasm (posix_spawn/p errno %d): %s",
		     ret, safe_strerror (ret));

	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "nvdisasm spawned with pid %d", (int)m_pid);

	// Close the parent's write ends now that spawn is done (as exec()
	// does).
	safe_close (m_stdout_wfd);
	safe_close (m_stderr_wfd);
      }
    catch (gdb_exception &e)
      {
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "Exception during nvdisasm exec setup: %s",
			   e.what ());
	safe_close (m_stdout_rfd);
	safe_close (m_stdout_wfd);
	safe_close (m_stderr_rfd);
	safe_close (m_stderr_wfd);
	m_pid = -1;
	throw;
      }
  }

  /* Waits for the disasm process to terminate, cleans up resources (FDs),
     and returns the exit status of the process.
     Returns 0 on normal exit with status 0, non-zero exit status otherwise,
     returns -1 the process terminated abnormally, or throws an exception if
     waitpid fails. */
  int
  wait ()
  {
    // Check if inactive (pid <= 0 indicates not started or already
    // waited/cleaned up)
    if (m_pid <= 0)
      error ("disasm_process::wait called on inactive object.");

    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
		       "Waiting for disasm process pid %d",
		       static_cast<int> (m_pid));

    // Close parent's read ends now. This might signal the process if needed,
    // but primarily cleans up our side. It's safe to do before waitpid.
    safe_close (m_stdout_rfd);
    safe_close (m_stderr_rfd);

    int child_status = 0;
    int exit_status = -1; // Default to error/unknown status

    // Wait for the child process
    const pid_t wait_ret = waitpid (m_pid, &child_status, 0);

    if (wait_ret == -1)
      {
	// Error waiting, if NVSAL disassembly failed do not emit a warning as
	// there will be a fallback.
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "Failed to wait for disasm process pid %d: %s",
			   static_cast<int> (m_pid), safe_strerror (errno));
	log_message ("Failed to wait for disasm process pid %d: %s",
		     static_cast<int> (m_pid), safe_strerror (errno));
      }
    else if (wait_ret == m_pid)
      {
	// Process terminated
	if (WIFEXITED (child_status))
	  {
	    exit_status = WEXITSTATUS (child_status);
	    cuda_trace_domain (
		CUDA_TRACE_DISASSEMBLER,
		"disasm process pid %d exited normally with status %d",
		static_cast<int> (m_pid), exit_status);
	  }
	else
	  // Terminated abnormally (signal, etc.)
	  log_message (
	      "disasm process pid %d terminated abnormally (status %d)",
	      static_cast<int> (m_pid), child_status);
	// Keep exit_status = -1 for abnormal termination
      }
    else
      // Unexpected return from waitpid
      log_message ("waitpid returned unexpected pid %d (expected %d)",
		   static_cast<int> (wait_ret), static_cast<int> (m_pid));

    // Mark as inactive and cleanup pid
    m_pid = -1;

    return exit_status;
  }
};

bool
cuda_read_module_metadata_json (const std::string &file_path,
				cuda_disasm_json::schema_metadata &out)
{
  try
    {
      /* quiet: a metadata failure is reported by the return value and traced
	 here, so wait() should not also warn. Not an NVSAL ELF read: this
	 spawns nvdisasm directly rather than going through exec().  */
      disasm_process proc (true, false);
      proc.exec_nvdisasm_metadata (file_path);

      bool parsed = false;
      {
	/* Drain the ENTIRE (small, -fun-bounded) document: metadata +
	   functions + end. The parser/file_stream owns the stdout read fd and
	   is destroyed at the end of this scope, before wait(). */
	auto stream = std::make_unique<util_stream::file_stream> (
	    proc.stdout_read_fd ());
	proc.disown_stdout_read_fd ();
	/* -ndf emits no instructions, so the instruction size only has to be
	   the value the rest of the tree parses with.  */
	cuda_disasm_json::parser parser (std::move (stream),
					 cuda_default_instruction_size);
	parser.stream_parse_metadata (out);
	std::vector<cuda_disasm_json::schema_function> funcs;
	parser.stream_parse_functions (funcs); /* empty or one function */
	parser.stream_parse_end ();
	parsed = true;
      }

      const std::string errtext = proc.get_stderr ();
      if (!errtext.empty ())
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "nvdisasm metadata stderr: %s", errtext.c_str ());

      /* Reuse the existing normal wait(): returns the child's exit status
	 (0 == clean success, non-zero/-1 otherwise). */
      const int status = proc.wait ();
      return parsed && status == 0;
    }
  catch (const gdb_exception &e)
    {
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "nvdisasm metadata read failed: %s", e.what ());
      return false;
    }
}

bool
cuda_module_disassembly_cache::parse_nvsal_elf (
    const std::string &function_name, const uint64_t pc)
{
  const auto &file_path = m_module->sync_to_disk_and_get_file_path ();
  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
		     "NVSAL JSON disassembly found pc 0x%lx in %s", pc,
		     file_path.c_str ());

  disasm_process proc (true, true);

  try
    {
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Trying NVSAL JSON disassembly");

      proc.exec (file_path, function_name.c_str ());

      const bool parsed = parse_json (proc.stdout_read_fd (),
				      cuda_disassembly_type_t::nvsal_elf);

      // Read stderr *before* waiting
      std::string proc_err = proc.get_stderr ();
      if (!proc_err.empty ())
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "NVSAL JSON disassembly stderr: %s",
			   proc_err.c_str ());

      // Wait for process and check status
      const int status = proc.wait ();
      if (status == 0 && parsed)
	{
	  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			     "Successfully parsed NVSAL JSON disassembly.");
	  return true;
	}
      proc.log_message (
	  "Failed to parse NVSAL JSON disassembly (status: %d, parsed: %d)",
	  status, parsed);
    }
  catch (const gdb_exception &e) // Catch exceptions during exec/parse/wait
    {
      proc.log_message ("Exception during NVSAL JSON disassembly: %s",
			e.what ());
    }
  // proc goes out of scope here, destructor runs if needed
  return false;
}

bool
cuda_module_disassembly_cache::parse_sass_elf (
    const std::string &function_name, const uint64_t pc)
{
  const auto &file_path = m_module->sync_to_disk_and_get_file_path ();
  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
		     "SASS JSON disassembly found pc 0x%lx in %s", pc,
		     file_path.c_str ());

  disasm_process proc (false, false);

  try
    {
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Trying SASS JSON disassembly");

      proc.exec (file_path, function_name.c_str ());

      const bool parsed = parse_json (proc.stdout_read_fd (),
				      cuda_disassembly_type_t::sass_elf);

      // Read stderr *before* waiting
      std::string proc_err = proc.get_stderr ();
      if (!proc_err.empty ())
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "SASS JSON disassembly stderr: %s",
			   proc_err.c_str ());

      // Wait for process and check status
      const int status = proc.wait ();
      if (status == 0 && parsed)
	{
	  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			     "Successfully parsed SASS JSON disassembly.");
	  return true;
	}
      proc.log_message (
	  "Failed to parse SASS JSON disassembly (status: %d, parsed: %d)",
	  status, parsed);
    }
  catch (const gdb_exception &e) // Catch exceptions during exec/parse/wait
    {
      proc.log_message ("Exception during SASS JSON disassembly: %s",
			e.what ());
    }
  // proc goes out of scope here, destructor runs if needed

  /* Failed to disassemble */
  return false;
}

bool
cuda_module_disassembly_cache::function_bounds (const std::string &func_name,
						const uint64_t pc,
						uint64_t &start_pc,
						uint64_t &length)
{
  struct block_symbol bsym{};
  struct objfile *objfile = m_module->objfile ();
  gdb_assert (objfile != nullptr);
  bsym = lookup_global_symbol_from_objfile (
      objfile, GLOBAL_BLOCK, func_name.c_str (), SEARCH_FUNCTION_DOMAIN);
  if (bsym.symbol == nullptr)
    bsym = lookup_global_symbol_from_objfile (
	objfile, STATIC_BLOCK, func_name.c_str (), SEARCH_FUNCTION_DOMAIN);
  if (bsym.symbol != nullptr && bsym.block != nullptr)
    {
      /* Function symbols are LOC_BLOCK: value_address() is the block
	 pointer, not the code address.  Use the block bounds.  */
      start_pc = bsym.block->start ();
      length = bsym.block->end () - bsym.block->start ();
      if (pc < start_pc || pc >= start_pc + length)
	{
	  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			     "Symbol bounds [0x%lx, 0x%lx) for '%s' do not "
			     "contain pc 0x%lx; using single-instruction span",
			     start_pc, start_pc + length, func_name.c_str (),
			     pc);
	  start_pc = pc;
	  length = cuda_default_instruction_size;
	}
    }
  else
    {
      // Function not found, create a single-instruction placeholder
      // function
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Using DebugAPI disassembly for unknown "
			 "function at PC 0x%lx",
			 pc);
      start_pc = pc;
      length = cuda_default_instruction_size;
    }
  if (!start_pc || (start_pc % cuda_default_instruction_size != 0))
    {
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Invalid start PC for function '%s': 0x%lx",
			 func_name.c_str (), start_pc);
      return false;
    }
  if (!length || (length % cuda_default_instruction_size != 0))
    {
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Invalid length for function '%s': 0x%lx",
			 func_name.c_str (), length);
      return false;
    }
  return true;
}

/* SASS only */
bool
cuda_module_disassembly_cache::parse_sass_device (
    const std::string &function_name, const uint64_t pc)
{
  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
		     "populate (SASS/DebugAPI): pc 0x%lx", pc);
  try
    {
      uint32_t inst_size = 0;
      char buf[MAX_BUFFER_SIZE];

      buf[0] = '\0';
      cuda_debugapi::disassemble (m_device->dev_idx (), pc, &inst_size, buf,
				  sizeof (buf));
      if (buf[0] == '\0')
	return false;

      uint64_t start_pc = 0;
      uint64_t length = 0;
      std::string func_name = function_name;
      cuda_function *func = lookup_function (pc);
      if (!func)
	{
	  if (!function_bounds (func_name, pc, start_pc, length))
	    return false;

	  // We've not created a function for this PC, create a new one
	  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			     "Creating new function %s for PC 0x%lx start_pc "
			     "0x%lx length 0x%lx",
			     func_name.c_str (), pc, start_pc, length);
	  func = add_function_to_cache (std::make_unique<cuda_function> (
	      func_name.c_str (), start_pc, length));
	}
      gdb_assert (pc >= func->start_address () && pc < func->end_address ());
      cuda_instruction instruction (pc, buf);
      if (func->add_instruction (instruction) == nullptr)
	{
	  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			     "Failed to cache SASS/DebugAPI instruction at "
			     "pc 0x%lx for %s",
			     pc, func_name.c_str ());
	  return false;
	}

      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Disassembled SASS/DebugAPI: %s pc 0x%lx: %s",
			 func_name.c_str (), pc, buf);
      return true;
    }
  catch (const gdb_exception_error &exception)
    {
      cuda_trace_domain (
	  CUDA_TRACE_DISASSEMBLER,
	  "Exception disassembling SASS/DebugAPI device %u pc %lx (%s)",
	  m_device->dev_idx (), pc, exception.what ());
      return false;
    }
}

bool
cuda_module_disassembly_cache::parse_json (int fd,
					   cuda_disassembly_type_t type)
{
  using namespace cuda_disasm_json;

  parser parser (std::make_unique<util_stream::file_stream> (fd),
		 cuda_default_instruction_size);

  parser.set_function_consumer ([this,
				 type] (const schema_function &function) {
    /* Size the cached function from the instruction stream as well as the
       JSON length field.  Some dumps report a short/zero length while still
       emitting a full instruction list.  */
    uint64_t span = function.m_length;
    uint64_t insn_span = 0;
    for (const auto &json_insn : function.m_instructions)
      insn_span
	  += cuda_default_instruction_size * json_insn.m_count.value_or (1);
    if (insn_span > span)
      span = insn_span;

    /* Reuse only an exact start-address match.  Range lookup can return a
       different overlapping function and then the PC walk goes out of bounds
       on the first JSON instruction.  When the JSON span is larger than the
       cached entry, grow it in place so prior NVSAL/SASS variants and
       disassembly status are preserved.  */
    cuda_function *func = nullptr;
    auto it = m_function_map.find (function.m_start);
    if (it != m_function_map.end ())
      func = it->second.get ();
    if (!func)
      func = add_function_to_cache (std::make_unique<cuda_function> (
	  function.m_function_name, function.m_start, span));
    else
      {
	span = clamp_span_to_next_function (func->start_address (), span);
	if (func->length () < span)
	  func->ensure_span (span);
      }

    uint64_t pc = function.m_start;
    for (const auto &json_insn : function.m_instructions)
      {
	/* Prefer JSON attributes from nvdisasm (control-flow, subroutine-call,
	   inter-warp-barrier).  When other-attributes is omitted, stamp NONE
	   so ELF/JSON paths do not fall through to DebugAPI opcode matching
	   in eval_is_*().  Padding always sets the padding attribute.  */

	auto padding = json_insn.m_padding.value_or (0);
	const auto count = json_insn.m_count.value_or (1);
	gdb_assert (count > padding);
	const auto first_padding_idx = count - padding;

	for (auto idx = 0; idx < count;
	     idx++, pc += cuda_default_instruction_size)
	  {
	    // FIXME - disassembler doesn't emit "other-attributes" for all
	    // instructions in an expansion where there's also end padding.
	    auto is_padding = (idx >= first_padding_idx);
	    auto info = is_padding ? cuda_instruction_disassembly_info (
					 "", "<padding nop>", "", "",
					 cuda_default_instruction_size)
				   : cuda_instruction_disassembly_info (
					 json_insn.m_info.m_predicate,
					 json_insn.m_info.m_opcode,
					 json_insn.m_info.m_operands,
					 json_insn.m_info.m_extra,
					 json_insn.m_count.value_or (1)
					     * cuda_default_instruction_size);
	    std::optional<cuda_instruction_attribute> attributes;
	    if (is_padding)
	      attributes = cuda_instruction_attribute::padding;
	    else if (idx < json_insn.m_other_attributes.size ())
	      {
		const auto &schema_attrs = json_insn.m_other_attributes[idx];
		if (!cuda_disasm_json::validate_instruction_attributes (
			schema_attrs))
		  {
		    cuda_trace_domain (
			CUDA_TRACE_DISASSEMBLER,
			"Invalid instruction attributes 0x%x in %s output for "
			"instruction: index %d %s",
			static_cast<uint32_t> (schema_attrs.attributes ()),
			cuda_disassembly_type_to_string (type), idx,
			json_insn.m_info.m_opcode.c_str ());
		  }
		attributes = schema_attrs.attributes ();
	      }

	    auto insn = func->lookup_instruction (pc);
	    if (insn)
	      {
		// Already disassembled (possibly in another format), update
		// with this disassembly type info and JSON-derived attributes.
		insn->set_disassembly_info (type, std::move (info));
		if (attributes.has_value ())
		  {
		    if (insn->attributes ().has_value ())
		      insn->set_attributes (*insn->attributes ()
					    | *attributes);
		    else
		      insn->set_attributes (*attributes);
		  }
		// Update count and index if not already set
		if (type == cuda_disassembly_type_t::nvsal_elf)
		  insn->set_count_index ((idx == 0) ? count : 1, idx);
	      }
	    else if (func->pc_beyond_cached_instructions (pc)
		     || ((pc - function.m_start)
			 / cuda_default_instruction_size)
			    >= cuda_function_max_instructions)
	      {
		/* Past the per-function instruction cache cap — drop the
		   remainder of this function's disassembly.  */
		cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
				   "Dropping instructions past max cache "
				   "(%zu) for function %s at PC 0x%lx",
				   cuda_function_max_instructions,
				   function.m_function_name.c_str (), pc);
		return;
	      }
	    else
	      {
		cuda_instruction new_insn (pc, (idx == 0) ? count : 1, idx,
					   attributes, type, std::move (info));
		insn = func->add_instruction (new_insn);
		CUDA_ERR_IF (
		    !insn, CUDA_TRACE_DISASSEMBLER,
		    "Failed to add instruction to function %s at PC 0x%lx",
		    func->name ().c_str (), pc);
	      }
	  }
      }
  });

  schema_json schema;

  try
    {
      parser.stream_parse_metadata (schema.m_metadata);
      parser.stream_parse_functions (schema.m_functions);
      parser.stream_parse_end ();
    }
  catch (const gdb_exception &e)
    {
      if (e.error == errors::NOT_SUPPORTED_ERROR)
	{
	  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			     "Failed to parse disasm binary %s output: %s",
			     cuda_disassembly_type_to_string (type),
			     e.what ());
	  return false;
	}
      throw;
    }

  return true;
}
