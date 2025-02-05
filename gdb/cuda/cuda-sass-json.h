/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2024 NVIDIA Corporation
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

#include "cuda-util-stream.h"
#include "gdbsupport/common-exceptions.h"

#include <functional>
#include <memory>
#include <sstream>
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

  /* Optional properties */
  int64_t m_note_nv_toolkit_version;
  std::string m_note_nv_toolkit_object_filename;
  std::string m_note_nv_toolkit_tool_name;
  std::string m_note_nv_toolkit_tool_version;
  std::string m_note_nv_toolkit_tool_branch;
  std::string m_note_nv_toolkit_tool_options;

  int64_t m_note_nv_cuda_version;
  int64_t m_note_nv_cuda_virtual_sm_version;
};

struct schema_sass_instruction
{
  std::string m_predicate;
  std::string m_opcode;
  std::string m_operands;
  std::string m_extra;
  gdb::optional<bool> m_opt_is_control_flow;
  std::vector<std::pair<std::string, std::string>> m_other_attributes;
  std::vector<std::string> m_other_flags;
};

struct schema_function
{
  std::string m_function_name;
  uint64_t m_start;
  uint64_t m_length;
  std::vector<std::string> m_other_attributes;
  std::vector<schema_sass_instruction> m_instructions;
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

  parser (std::unique_ptr<util_stream::istream> &&stream);

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
  lexer m_lexer;
  parser_state m_state;

  on_function_t m_on_function;

  void parse_metadata (schema_metadata &metadata);
  void parse_elf_metadata (schema_metadata &metadata);
  void parse_schema_version (schema_metadata &metadata);
  void parse_sm_metadata (schema_metadata &metadata);
  void parse_sm_version (schema_metadata &metadata);
  void parse_cuda_version (schema_metadata &metadata);
  void parse_toolkit_info (schema_metadata &metadata);
  void parse_cuda_info (schema_metadata &metadata);

  void parse_functions_list (std::vector<schema_function> &functions);
  void parse_function (schema_function &function);
  void parse_sass_instructions_list (
      std::vector<schema_sass_instruction> &instructions);
  void parse_sass_instruction (schema_sass_instruction &instruction);
  /* Returns true if field is part of the schema */
  bool parse_sass_instruction_field (schema_sass_instruction &instruction,
				     const std::string &field,
				     int &fields_mask);

  void parse_string_list (std::vector<std::string> &list);
  void parse_other_attributes_dict (schema_sass_instruction &instruction);
  void parse_instruction_other_attribute_field (
      schema_sass_instruction &instruction, const std::string &field,
      const std::string &value);

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
  template <typename... Args>
  static inline void validate_fields_bitmask (int bitmask, int required,
					      const char *fmt, Args... args);
  static inline void update_bitmask_field (int &bitmask, const int field,
					   const token &token);

  static const char *parser_state_to_string (const parser_state state);

  friend class selftests::cuda_sass_json;
};
}

#endif
