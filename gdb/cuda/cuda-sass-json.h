/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2024-2026 NVIDIA Corporation
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

#ifndef _CUDA_SASS_JSON_H
#define _CUDA_SASS_JSON_H 1

#include "cuda-asm.h"
#include "cuda-util-stream.h"
#include "gdbsupport/common-exceptions.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

/* Unit test class declaration to befriend */
namespace selftests
{
class cuda_sass_json;
}

namespace cuda_disasm_json
{
struct schema_metadata
{
  /* Required properties */
  int64_t m_elf_layout_id;
  int64_t m_elf_ei_osabi;
  int64_t m_elf_ei_abiversion;

  int64_t m_schema_ver_major;
  int64_t m_schema_ver_minor;
  int64_t m_schema_ver_revision;

  int64_t m_sm_ver_major;
  int64_t m_sm_ver_minor;

  std::string m_producer;
  std::string m_description;

  /* Optional properties — .note.nv.tkinfo */
  std::optional<int64_t> m_note_nv_toolkit_version;
  std::optional<std::string> m_note_nv_toolkit_object_filename;
  std::optional<std::string> m_note_nv_toolkit_tool_name;
  std::optional<std::string> m_note_nv_toolkit_tool_version;
  std::optional<std::string> m_note_nv_toolkit_tool_branch;
  std::optional<std::string> m_note_nv_toolkit_tool_options;

  /* Optional properties — .note.nv.cuinfo (legacy .note.nv.cuver).
     The tag and the toolkit value are DISTINCT integers. */
  std::optional<int64_t> m_note_nv_cuda_version;        /* tag */
  std::optional<int64_t> m_note_nv_cuinfo_tool_version; /* nv_note_cuinfo_toolVersion */
  std::optional<int64_t> m_note_nv_cuda_virtual_sm_version;
};

struct schema_value
{
  enum class value_type
  {
    string,
    integer,
    boolean,
    null,
  };

  value_type m_type;
  std::string m_string;
  int64_t m_integer;
  bool m_boolean;

  bool
  is_string () const
  {
    return m_type == value_type::string;
  }
  bool
  is_integer () const
  {
    return m_type == value_type::integer;
  }
  bool
  is_boolean () const
  {
    return m_type == value_type::boolean;
  }
  bool
  is_null () const
  {
    return m_type == value_type::null;
  }

  std::string
  to_string () const
  {
    gdb_assert (m_type == value_type::string);
    return m_string;
  }
  int64_t
  to_integer () const
  {
    gdb_assert (m_type == value_type::integer);
    return m_integer;
  }
  bool
  to_boolean () const
  {
    gdb_assert (m_type == value_type::boolean);
    return m_boolean;
  }

  schema_value (std::string &&string)
      : m_type (value_type::string), m_string (std::move (string)),
	m_integer (0), m_boolean (false)
  {
  }

  schema_value (int64_t integer)
      : m_type (value_type::integer), m_string (), m_integer (integer),
	m_boolean (false)
  {
  }

  schema_value (bool boolean)
      : m_type (value_type::boolean), m_string (), m_integer (0),
	m_boolean (boolean)
  {
  }

  schema_value ()
      : m_type (value_type::null), m_string (), m_integer (0),
	m_boolean (false)
  {
  }
};

struct schema_annotation
{

  schema_value
  lookup (const std::string &name) const
  {
    for (const auto &[key, value] : m_values)
      {
	if (key == name)
	  return value;
      }

    // Return a null value if the key is not found
    return schema_value ();
  }

  void
  add (const std::string &name, const schema_value &value)
  {
    m_values.emplace_back (name, value);
  }

  std::vector<std::pair<std::string, schema_value>>
  values () const
  {
    return m_values;
  }

  std::vector<std::pair<std::string, schema_value>> m_values;
};

// Result of parsing an "other-attributes" object in an instruction.
class schema_instruction_attributes
{
public:
  schema_instruction_attributes ()
      : m_attributes (cuda_instruction_attribute::none),
	m_padding (std::nullopt)
  {
  }

  schema_instruction_attributes (cuda_instruction_attribute attributes,
				 std::optional<int> padding)
      : m_attributes (attributes), m_padding (padding)
  {
  }

  cuda_instruction_attribute
  attributes () const
  {
    return m_attributes;
  }

  void
  set_attributes (cuda_instruction_attribute attributes)
  {
    m_attributes = attributes;
  }

  bool
  is_barrier () const
  {
    return (m_attributes & cuda_instruction_attribute::inter_warp_barrier)
	   != cuda_instruction_attribute::none;
  }

  bool
  is_control_flow () const
  {
    return (m_attributes & cuda_instruction_attribute::control_flow)
	   != cuda_instruction_attribute::none;
  }

  bool
  is_subroutine_call () const
  {
    return (m_attributes & cuda_instruction_attribute::subroutine_call)
	   != cuda_instruction_attribute::none;
  }

  int
  has_padding () const
  {
    return m_padding.has_value ();
  }

  int
  padding () const
  {
    return m_padding.value_or (0);
  }

  void
  set_padding (int padding)
  {
    m_padding = std::make_optional (padding);
  }

private:
  cuda_instruction_attribute m_attributes;
  std::optional<int> m_padding;
};

struct schema_instruction_disassembly
{
  std::string m_predicate;
  std::string m_opcode;
  std::string m_operands;
  std::string m_extra;
};

struct schema_instruction
{
  uint64_t m_pc;
  schema_instruction_disassembly m_info;
  std::optional<uint32_t> m_count;
  std::optional<uint32_t> m_padding;
  std::optional<cuda_instruction_attribute> m_attributes;
  std::vector<schema_instruction_attributes> m_other_attributes;
  std::vector<std::string> m_other_flags;
  std::vector<schema_annotation> m_annotations;
};

struct schema_function
{
  std::string m_function_name;
  uint64_t m_start;
  uint64_t m_length;
  std::vector<std::string> m_other_attributes;
  std::vector<schema_instruction> m_instructions;
  std::vector<schema_annotation> m_annotations;
};

struct schema_json
{
  schema_metadata m_metadata;

  std::vector<schema_function> m_functions;
};

enum class token_type
{
  start_object,	       // {
  end_object,	       // }
  start_array,	       // [
  end_array,	       // ]
  element_separator,   // ,
  key_value_separator, // :
  string_value,
  integer_value,
  true_literal,	 // true
  false_literal, // false
  null_literal,	 // null

  end_of_input,
};

struct token
{
  token (token_type type, size_t start_pos, size_t end_pos);
  token (token_type type, size_t start_pos, size_t end_pos,
	 std::string &&string);
  token (token_type type, size_t start_pos, size_t end_pos, int64_t integer);
  token () = default;

  token_type m_type;
  size_t m_start_pos;
  size_t m_end_pos;

  std::string m_string;
  union
  {
    int64_t integer;
  } m_value;

  std::string to_string () const;
};

class lexer
{
public:
  lexer (std::unique_ptr<util_stream::istream> &&stream);
  token
  next ()
  {
    if (m_peeked)
      {
	m_peeked = false;
	return m_peeked_token;
      }

    return lex ();
  }

  token
  peek ()
  {
    if (!m_peeked)
      {
	m_peeked_token = lex ();
	m_peeked = true;
      }

    return m_peeked_token;
  }

private:
  std::unique_ptr<util_stream::istream> m_stream;

  bool m_peeked;
  token m_peeked_token;

  token lex ();

  inline token read_string (size_t start_pos);
  inline token read_number (size_t start_pos, int c);
  inline token read_true (size_t start_pos);
  inline token read_false (size_t start_pos);
  inline token read_null (size_t start_pos);

  inline void read_literal (const char *literal, const size_t start_pos);

  inline void skip_whitespaces ();

  inline int
  get_char ()
  {
    return m_stream->getc ();
  }
  inline int
  peek_char ()
  {
    return m_stream->peekc ();
  }

  // Utility functions
  static inline char escapeChar (const size_t position, const int c);
};

enum class parser_state
{
  at_start,

  at_metadata_start,
  at_metadata_end,

  at_functions_start,
  in_functions,
  at_functions_end,

  at_end,

  error,
};

class parser
{
public:
  typedef std::function<void (const schema_function &)> on_function_t;

  parser (std::unique_ptr<util_stream::istream> &&stream, uint32_t insn_size);

  /* Parse the entire json object */
  void parse_json (schema_json &schema);

  /* Advance the stream until the metadata object is parsed. May be useful if
   * we need to make some decisions based on the metadata before parsing the
   * whole object. */
  void stream_parse_metadata (schema_metadata &metadata);

  /* Advance the stream over the list of functions. Argument is left empty if
   * there is a function consumer set. */
  void stream_parse_functions (std::vector<schema_function> &functions);

  /* Check that the stream ends after the functions list */
  void stream_parse_end ();

  /* Returns the current state of the parser when parsing subsections
   * separately. */
  inline parser_state
  current_state () const
  {
    return m_state;
  }

  /* Sets a callback function that will be called every time a function is
   * parsed. Functions will not be stored in the schema object. */
  inline void
  set_function_consumer (on_function_t consumer)
  {
    m_on_function = consumer;
  }

private:
  enum class cuda_info_kind
  {
    cuinfo,
    cuver
  };

  lexer m_lexer;
  parser_state m_state;

  on_function_t m_on_function;

  uint32_t m_insn_size;

  void parse_metadata (schema_metadata &metadata);
  void parse_elf_metadata (schema_metadata &metadata);
  void parse_schema_version (schema_metadata &metadata);
  void parse_sm_metadata (schema_metadata &metadata);
  void parse_sm_version (schema_metadata &metadata);
  void parse_cuda_version (schema_metadata &metadata);
  void parse_toolkit_info (schema_metadata &metadata);
  template <cuda_info_kind Kind>
  void parse_cuda_info (schema_metadata &metadata);

  void parse_functions_list (std::vector<schema_function> &functions,
			     uint32_t insn_size);
  void parse_function (schema_function &function, uint32_t insn_size);
  void parse_instructions_list (const std::string &format,
				std::vector<schema_instruction> &instructions);
  void parse_instruction (const std::string &format,
			  std::vector<schema_instruction> &instructions);

  /* Returns true if field is part of the schema */
  bool parse_instruction_attribute_field (
      const std::string &format, schema_instruction &instruction,
      std::vector<cuda_instruction_attribute> &attributes,
      const std::string &field, int &fields_mask);

  void parse_string_list (std::vector<std::string> &list);
  void parse_instruction_attribute_object (schema_instruction &instruction);
  void parse_instruction_attributes (schema_instruction &instruction);
  void parse_instruction_attributes_dict (
      std::vector<std::pair<std::string, schema_value>> &attributes);
  void parse_annotations (std::vector<schema_annotation> &annotations);
  void parse_annotation (schema_annotation &annotation);
  void parse_value (schema_value &value);
  inline token next_expect (token_type type);

  inline void
  expect_state (parser_state state)
  {
    if (m_state != state)
      throw_error (errors::NOT_SUPPORTED_ERROR,
		   "Parser error: Expected state to be %s, but got: %s",
		   parser_state_to_string (state),
		   parser_state_to_string (m_state));
  }

  inline void
  set_state (parser_state state)
  {
    m_state = state;
  }

  void skip_value (const std::string &field);
  void skip_dictionary ();
  void skip_array ();

  inline token
  peek ()
  {
    return m_lexer.peek ();
  }

  inline token
  next ()
  {
    return m_lexer.next ();
  }

  // Utility functionss
  static inline void validate_fields_bitmask (int bitmask, int required,
					      const char *msg);
  static inline void update_bitmask_field (int &bitmask, const int field,
					   const token &token);

  static const char *parser_state_to_string (const parser_state state);

  friend class selftests::cuda_sass_json;
};

/* Validate JSON attribute relationships for a SASS instruction.
   - subroutine-call must imply control-flow (call is a subset of cf)
   - inter-warp-barrier is independent (cannot be control-flow or call)
   Returns true if validation passes, false if any constraint is violated. */
bool validate_instruction_attributes (
    const schema_instruction_attributes &attributes);

}

#endif
