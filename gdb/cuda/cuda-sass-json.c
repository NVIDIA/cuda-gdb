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

#include "defs.h"

#include "cuda-sass-json.h"
#include "cuda-tdep.h"

#include <stdarg.h>
#include <stdio.h>

#define PRINT_POS "zu"

#define EXPECT_KEY_EQUALS(_expected, _key, _token)                            \
  do                                                                          \
    {                                                                         \
      if ((_key) != (_expected))                                              \
	{                                                                     \
	  return false;                                                       \
	}                                                                     \
    }                                                                         \
  while (0)

using namespace util_stream;

namespace cuda_disasm_json
{
static const char *
token_type_to_string (const token_type type)
{
  switch (type)
    {
    case token_type::start_object:
      return "StartObject";
    case token_type::end_object:
      return "EndObject";
    case token_type::start_array:
      return "StartArray";
    case token_type::end_array:
      return "EndArray";
    case token_type::element_separator:
      return "ElementSeparator";
    case token_type::key_value_separator:
      return "KeyValueSeparator";
    case token_type::string_value:
      return "String";
    case token_type::integer_value:
      return "Integer";
    case token_type::true_literal:
      return "True";
    case token_type::false_literal:
      return "False";
    case token_type::null_literal:
      return "Null";
    case token_type::end_of_input:
      return "EndOfInput";
    default:
      return "Unknown";
    }
}

token::token (token_type type, size_t start_pos, size_t end_pos)
    : m_type (type), m_start_pos (start_pos), m_end_pos (end_pos), m_string (),
      m_value ()
{
}

token::token (token_type type, size_t start_pos, size_t end_pos,
	      std::string &&string)
    : m_type (type), m_start_pos (start_pos), m_end_pos (end_pos),
      m_string (std::move (string)), m_value ()
{
}

token::token (token_type type, size_t start_pos, size_t end_pos,
	      int64_t integer)
    : m_type (type), m_start_pos (start_pos), m_end_pos (end_pos), m_string (),
      m_value ({ .integer = integer })
{
}

std::string
token::to_string () const
{
  std::string base = "Token at position " + std::to_string (m_start_pos)
		     + " - " + std::to_string (m_end_pos) + ": ";
  switch (m_type)
    {
    case token_type::string_value:
      return base + std::string ("String: ") + m_string;
      break;
    case token_type::integer_value:
      return base + std::string ("Integer: ")
	     + std::to_string (m_value.integer);
      break;
    default:
      return base + token_type_to_string (m_type);
    }
}

lexer::lexer (std::unique_ptr<istream> &&stream)
    : m_stream (std::move (stream)), m_peeked (false), m_peeked_token ()
{
}

token
lexer::lex ()
{
  skip_whitespaces ();

  const size_t start_pos = m_stream->tell ();
  const int c = m_stream->getc ();

  switch (c)
    {
    // Value
    case '{':
      return token (token_type::start_object, start_pos, start_pos + 1);
    case '}':
      return token (token_type::end_object, start_pos, start_pos + 1);
    case '[':
      return token (token_type::start_array, start_pos, start_pos + 1);
    case ']':
      return token (token_type::end_array, start_pos, start_pos + 1);
    case ',':
      return token (token_type::element_separator, start_pos, start_pos + 1);
    case ':':
      return token (token_type::key_value_separator, start_pos, start_pos + 1);
    case '"':
      return read_string (start_pos);

    // Numbers (floats are not supported)
    case '-':
    case '0' ... '9':
      return read_number (start_pos, c);

    // Keywords (keep in mind we consumed the first character already)
    case 't':
      return read_true (start_pos);
    case 'f':
      return read_false (start_pos);
    case 'n':
      return read_null (start_pos);

    case '/':
      throw_error (errors::NOT_SUPPORTED_ERROR,
		   "Comments are not supported at %" PRINT_POS, start_pos);

    case EOF:
      return token (token_type::end_of_input, start_pos, start_pos + 1);
    default:
      throw_error (errors::NOT_SUPPORTED_ERROR,
		   "Unexpected character: %c(%d) at %" PRINT_POS, (char)c, c,
		   start_pos);
    }
}

inline token
lexer::read_string (const size_t start_pos)
{
  /* Reserve a large buffer to prevent string reallocations*/
  static constexpr size_t string_reserve = 64;

  std::string token_body;
  token_body.reserve (string_reserve);

  int c;
  while ((c = m_stream->getc ()) != '"')
    {
      if (c == EOF)
	throw_error (errors::NOT_SUPPORTED_ERROR,
		     "Unexpected end of input in string token at %" PRINT_POS,
		     start_pos);

      if (c == '\\')
	{
	  c = m_stream->getc ();
	  if (c == EOF)
	    throw_error (
		errors::NOT_SUPPORTED_ERROR,
		"Unexpected end of input in string token at %" PRINT_POS,
		start_pos);

	  token_body.push_back (escapeChar (m_stream->tell (), c));
	}
      else
	{
	  token_body.push_back (c);
	}
    }

  return token (token_type::string_value, start_pos, m_stream->tell (),
		std::move (token_body));
}

inline token
lexer::read_number (const size_t start_pos, int c)
{
  int64_t value = 0;
  bool negative = false;

  if (c == '-')
    {
      negative = true;
      c = get_char ();
    }

  switch (c)
    {
    case '0': // this allows leading zeros (not part of JSON strict syntax)
      break;
    case '1' ... '9':
      value = c - '0';
      break;
    default:
      throw_error (errors::NOT_SUPPORTED_ERROR,
		   "Unexpected character in number token at %" PRINT_POS
		   ": '%c'(%d)",
		   start_pos, (char)c, c);
    }

  for (c = peek_char (); c >= '0' && c <= '9'; c = peek_char ())
    {
      const bool overflows
	  = __builtin_mul_overflow (value, 10, &value)
	    || __builtin_add_overflow (
		value, c - '0', &value); // note: min value is rejected because
					 // the sign is changed at the end
      if (overflows)
	throw_error (errors::NOT_SUPPORTED_ERROR,
		     "Integer overflow at %" PRINT_POS, start_pos);

      /* Note: this consumes the previously added digit from the stream, after
       * then we can peek the next char. */
      get_char ();
    }

  const bool is_float = (c == '.' || c == 'e' || c == 'E');
  if (is_float)
    throw_error (errors::NOT_SUPPORTED_ERROR,
		 "Floats/Exponents are not supported at %" PRINT_POS,
		 start_pos);

  return token (token_type::integer_value, start_pos, m_stream->tell (),
		negative ? -value : value);
}

inline token
lexer::read_true (const size_t start_pos)
{
  read_literal ("rue", start_pos);

  return token (token_type::true_literal, start_pos, m_stream->tell ());
}

inline token
lexer::read_false (const size_t start_pos)
{
  read_literal ("alse", start_pos);

  return token (token_type::false_literal, start_pos, m_stream->tell ());
}

inline token
lexer::read_null (const size_t start_pos)
{
  read_literal ("ull", start_pos);

  return token (token_type::null_literal, start_pos, m_stream->tell ());
}

inline void
lexer::read_literal (const char *literal, const size_t start_pos)
{
  for (const char *c = literal; *c; c++)
    {
      if (get_char () != *c)
	{
	  throw_error (errors::NOT_SUPPORTED_ERROR,
		       "Unexpected character in literal token at %" PRINT_POS
		       ", expected %s",
		       start_pos, literal);
	}
    }
}

inline void
lexer::skip_whitespaces ()
{
  int c = peek_char ();
  while (c == ' ' || c == '\t' || c == '\n' || c == '\r')
    {
      get_char ();
      c = peek_char ();
    }
}

inline char
lexer::escapeChar (const size_t position, const int c)
{
  switch (c)
    {
    case '"':
      return '"';
    case '\\':
      return '\\';
    case '/':
      return '/';
    case 'b':
      return '\b';
    case 'f':
      return '\f';
    case 'n':
      return '\n';
    case 'r':
      return '\r';
    case 't':
      return '\t';
    case 'u':
      return 'u'; // unicode not decoded
    default:
      throw_error (errors::NOT_SUPPORTED_ERROR,
		   "Invalid escape character at %" PRINT_POS ": %c", position,
		   c);
    }
}

parser::parser (std::unique_ptr<istream> &&stream)
    : m_lexer (std::move (stream)), m_state (parser_state::at_start),
      m_on_function ()
{
}

void
parser::parse_json (schema_json &schema)
{
  expect_state (parser_state::at_start);

  next_expect (token_type::start_array);

  parse_metadata (schema.m_metadata);

  next_expect (token_type::element_separator);

  parse_functions_list (schema.m_functions);

  next_expect (token_type::end_array);

  next_expect (token_type::end_of_input);

  set_state (parser_state::at_end);
}

void
parser::stream_parse_metadata (schema_metadata &metadata)
{
  switch (m_state)
    {
    case parser_state::at_start:
      next_expect (token_type::start_array);
      break;
    case parser_state::at_metadata_start:
      break;
    default:
      throw_error (errors::NOT_SUPPORTED_ERROR, "%s: unexpected state: %s",
		   __func__, parser_state_to_string (m_state));
    }

  parse_metadata (metadata);

  set_state (parser_state::at_metadata_end);
}

void
parser::stream_parse_functions (std::vector<schema_function> &functions)
{
  switch (m_state)
    {
    case parser_state::at_start:
    case parser_state::at_metadata_start:
      throw_error (errors::NOT_SUPPORTED_ERROR, "cannot skip metadata");
    case parser_state::at_metadata_end:
      next_expect (token_type::element_separator);
      break;
    case parser_state::at_functions_start:
      break;
    default:
      throw_error (errors::NOT_SUPPORTED_ERROR, "%s: unexpected state: %s",
		   __func__, parser_state_to_string (m_state));
    }

  parse_functions_list (functions);

  set_state (parser_state::at_functions_end);
}

void
parser::stream_parse_end ()
{
  switch (m_state)
    {
    case parser_state::at_functions_end:
      next_expect (token_type::end_array);
      break;
    default:
      throw_error (errors::NOT_SUPPORTED_ERROR, "%s: unexpected state: %s",
		   __func__, parser_state_to_string (m_state));
    }

  next_expect (token_type::end_of_input);

  set_state (parser_state::at_end);
}

void
parser::parse_metadata (schema_metadata &metadata)
{
  enum class metadata_properties : int
  {
    elf = (1 << 0),
    sm = (1 << 1),
    schema_version = (1 << 2),
    producer = (1 << 3),
    description = (1 << 4),
    note_nv_toolkit_info = (1 << 5),
    note_nv_cuda_info = (1 << 6),
  };
  static constexpr int required_fields_mask
      = (int)metadata_properties::elf | (int)metadata_properties::sm
	| (int)metadata_properties::schema_version
	| (int)metadata_properties::producer
	| (int)metadata_properties::description;

  next_expect (token_type::start_object);

  token string_token;
  int fields_mask = 0;

  for (;;)
    {
      string_token = next_expect (token_type::string_value);
      next_expect (token_type::key_value_separator);

      if (string_token.m_string == "ELF")
	{
	  update_bitmask_field (fields_mask, (int)metadata_properties::elf,
				string_token);
	  parse_elf_metadata (metadata);
	}
      else if (string_token.m_string == "SchemaVersion")
	{
	  update_bitmask_field (fields_mask,
				(int)metadata_properties::schema_version,
				string_token);
	  parse_schema_version (metadata);
	}
      else if (string_token.m_string == "SM")
	{
	  update_bitmask_field (fields_mask, (int)metadata_properties::sm,
				string_token);
	  parse_sm_metadata (metadata);
	}
      else if (string_token.m_string == "Producer")
	{
	  update_bitmask_field (
	      fields_mask, (int)metadata_properties::producer, string_token);
	  string_token = next_expect (token_type::string_value);
	  metadata.m_producer = string_token.m_string;
	}
      else if (string_token.m_string == "Description")
	{
	  update_bitmask_field (fields_mask,
				(int)metadata_properties::description,
				string_token);
	  string_token = next_expect (token_type::string_value);
	  metadata.m_description = string_token.m_string;
	}
      else if (string_token.m_string == ".note.nv.tkinfo")
	{
	  update_bitmask_field (fields_mask,
				(int)metadata_properties::note_nv_toolkit_info,
				string_token);
	  parse_toolkit_info (metadata);
	}
      else if (string_token.m_string == ".note.nv.cuver")
	{
	  update_bitmask_field (fields_mask,
				(int)metadata_properties::note_nv_cuda_info,
				string_token);
	  parse_cuda_info (metadata);
	}
      else
	{
	  skip_value (string_token.m_string);
	}

      if (peek ().m_type == token_type::end_object)
	{
	  break;
	}

      next_expect (token_type::element_separator);
    }

  next_expect (token_type::end_object);

  validate_fields_bitmask (fields_mask, (int)required_fields_mask,
			   "Missing required fields in Metadata object");
}

void
parser::parse_elf_metadata (schema_metadata &metadata)
{
  enum class elf_properties : int
  {
    m_layout_id = (1 << 0),
    m_ei_osabi = (1 << 1),
    m_ei_abiversion = (1 << 2),
  };
  static constexpr int required_fields_mask
      = (int)elf_properties::m_layout_id | (int)elf_properties::m_ei_osabi
	| (int)elf_properties::m_ei_abiversion;

  next_expect (token_type::start_object);

  token string_token;
  token integer_token;
  int fields_mask = 0;

  for (;;)
    {
      string_token = next_expect (token_type::string_value);
      next_expect (token_type::key_value_separator);

      if (string_token.m_string == "layout-id")
	{
	  update_bitmask_field (fields_mask, (int)elf_properties::m_layout_id,
				string_token);
	  integer_token = next_expect (token_type::integer_value);
	  metadata.m_elf_layout_id = integer_token.m_value.integer;
	}
      else if (string_token.m_string == "ei_osabi")
	{
	  update_bitmask_field (fields_mask, (int)elf_properties::m_ei_osabi,
				string_token);
	  integer_token = next_expect (token_type::integer_value);
	  metadata.m_elf_ei_osabi = integer_token.m_value.integer;
	}
      else if (string_token.m_string == "ei_abiversion")
	{
	  update_bitmask_field (
	      fields_mask, (int)elf_properties::m_ei_abiversion, string_token);
	  integer_token = next_expect (token_type::integer_value);
	  metadata.m_elf_ei_abiversion = integer_token.m_value.integer;
	}
      else
	{
	  skip_value (string_token.m_string);
	}

      if (peek ().m_type == token_type::end_object)
	{
	  break;
	}

      next_expect (token_type::element_separator);
    }

  next_expect (token_type::end_object);

  validate_fields_bitmask (fields_mask, (int)required_fields_mask,
			   "Missing required fields in ELF object");
}

void
parser::parse_schema_version (schema_metadata &metadata)
{
  enum class schema_version_properties : int
  {
    major = (1 << 0),
    minor = (1 << 1),
    revision = (1 << 2),
  };
  static constexpr int required_fields_mask
      = (int)schema_version_properties::major
	| (int)schema_version_properties::minor
	| (int)schema_version_properties::revision;

  next_expect (token_type::start_object);

  token string_token;
  token integer_token;
  int fields_mask = 0;

  for (;;)
    {
      string_token = next_expect (token_type::string_value);
      next_expect (token_type::key_value_separator);

      if (string_token.m_string == "major")
	{
	  update_bitmask_field (fields_mask,
				(int)schema_version_properties::major,
				string_token);
	  integer_token = next_expect (token_type::integer_value);
	  metadata.m_schema_ver_major = integer_token.m_value.integer;
	}
      else if (string_token.m_string == "minor")
	{
	  update_bitmask_field (fields_mask,
				(int)schema_version_properties::minor,
				string_token);
	  integer_token = next_expect (token_type::integer_value);
	  metadata.m_schema_ver_minor = integer_token.m_value.integer;
	}
      else if (string_token.m_string == "revision")
	{
	  update_bitmask_field (fields_mask,
				(int)schema_version_properties::revision,
				string_token);
	  integer_token = next_expect (token_type::integer_value);
	  metadata.m_schema_ver_revision = integer_token.m_value.integer;
	}
      else
	{
	  skip_value (string_token.m_string);
	}

      if (peek ().m_type == token_type::end_object)
	{
	  break;
	}

      next_expect (token_type::element_separator);
    }

  next_expect (token_type::end_object);

  validate_fields_bitmask (fields_mask, (int)required_fields_mask,
			   "Missing required fields in SchemaVersion object");
}

void
parser::parse_sm_metadata (schema_metadata &metadata)
{
  enum class sm_properties : int
  {
    version = (1 << 0),
  };
  static constexpr int required_fields_mask = (int)sm_properties::version;

  next_expect (token_type::start_object);

  token string_token;
  int fields_mask = 0;

  for (;;)
    {
      string_token = next_expect (token_type::string_value);
      next_expect (token_type::key_value_separator);

      if (string_token.m_string == "version")
	{
	  update_bitmask_field (fields_mask, (int)sm_properties::version,
				string_token);
	  parse_sm_version (metadata);
	}
      else
	{
	  skip_value (string_token.m_string);
	}

      if (peek ().m_type == token_type::end_object)
	{
	  break;
	}

      next_expect (token_type::element_separator);
    }

  next_expect (token_type::end_object);

  validate_fields_bitmask (fields_mask, (int)required_fields_mask,
			   "Missing required fields in SM object");
}

void
parser::parse_sm_version (schema_metadata &metadata)
{
  enum class sm_version_properties : int
  {
    major = (1 << 0),
    minor = (1 << 1),
  };
  static constexpr int required_fields_mask
      = (int)sm_version_properties::major | (int)sm_version_properties::minor;

  next_expect (token_type::start_object);

  token string_token;
  token integer_token;
  int fields_mask = 0;

  for (;;)
    {
      string_token = next_expect (token_type::string_value);
      next_expect (token_type::key_value_separator);

      if (string_token.m_string == "major")
	{
	  update_bitmask_field (fields_mask, (int)sm_version_properties::major,
				string_token);
	  integer_token = next_expect (token_type::integer_value);
	  metadata.m_sm_ver_major = integer_token.m_value.integer;
	}
      else if (string_token.m_string == "minor")
	{
	  update_bitmask_field (fields_mask, (int)sm_version_properties::minor,
				string_token);
	  integer_token = next_expect (token_type::integer_value);
	  metadata.m_sm_ver_minor = integer_token.m_value.integer;
	}
      else
	{
	  skip_value (string_token.m_string);
	}

      if (peek ().m_type == token_type::end_object)
	{
	  break;
	}

      next_expect (token_type::element_separator);
    }

  next_expect (token_type::end_object);

  validate_fields_bitmask (fields_mask, (int)required_fields_mask,
			   "Missing required fields in SMVersion object");
}

void
parser::parse_toolkit_info (schema_metadata &metadata)
{
  enum class toolkit_info_properties : int
  {
    version = (1 << 0),
    object_filename = (1 << 1),
    tool_name = (1 << 2),
    tool_version = (1 << 3),
    tool_branch = (1 << 4),
    tool_options = (1 << 5),
  };
  static constexpr int required_fields_mask
      = (int)toolkit_info_properties::version
	| (int)toolkit_info_properties::object_filename
	| (int)toolkit_info_properties::tool_name
	| (int)toolkit_info_properties::tool_version
	| (int)toolkit_info_properties::tool_branch
	| (int)toolkit_info_properties::tool_options;

  next_expect (token_type::start_object);

  token string_token;
  token integer_token;
  int fields_mask = 0;

  for (;;)
    {
      string_token = next_expect (token_type::string_value);
      next_expect (token_type::key_value_separator);

      if (string_token.m_string == "tki_toolkitVersion")
	{
	  update_bitmask_field (fields_mask,
				(int)toolkit_info_properties::version,
				string_token);
	  integer_token = next_expect (token_type::integer_value);
	  metadata.m_note_nv_toolkit_version = integer_token.m_value.integer;
	}
      else if (string_token.m_string == "tki_objFname")
	{
	  update_bitmask_field (fields_mask,
				(int)toolkit_info_properties::object_filename,
				string_token);
	  string_token = next_expect (token_type::string_value);
	  metadata.m_note_nv_toolkit_object_filename = string_token.m_string;
	}
      else if (string_token.m_string == "tki_toolName")
	{
	  update_bitmask_field (fields_mask,
				(int)toolkit_info_properties::tool_name,
				string_token);
	  string_token = next_expect (token_type::string_value);
	  metadata.m_note_nv_toolkit_tool_name = string_token.m_string;
	}
      else if (string_token.m_string == "tki_toolVersion")
	{
	  update_bitmask_field (fields_mask,
				(int)toolkit_info_properties::tool_version,
				string_token);
	  string_token = next_expect (token_type::string_value);
	  metadata.m_note_nv_toolkit_tool_version = string_token.m_string;
	}
      else if (string_token.m_string == "tki_toolBranch")
	{
	  update_bitmask_field (fields_mask,
				(int)toolkit_info_properties::tool_branch,
				string_token);
	  string_token = next_expect (token_type::string_value);
	  metadata.m_note_nv_toolkit_tool_branch = string_token.m_string;
	}
      else if (string_token.m_string == "tki_toolOptions")
	{
	  update_bitmask_field (fields_mask,
				(int)toolkit_info_properties::tool_options,
				string_token);
	  string_token = next_expect (token_type::string_value);
	  metadata.m_note_nv_toolkit_tool_options = string_token.m_string;
	}
      else
	{
	  skip_value (string_token.m_string);
	}

      if (peek ().m_type == token_type::end_object)
	{
	  break;
	}

      next_expect (token_type::element_separator);
    }

  next_expect (token_type::end_object);

  validate_fields_bitmask (fields_mask, (int)required_fields_mask,
			   "Missing required fields in ToolkitInfo object");
}

void
parser::parse_cuda_info (schema_metadata &metadata)
{
  enum class cuda_info_properties : int
  {
    version = (1 << 0),
    virtual_sm_version = (1 << 1),
  };
  static constexpr int required_fields_mask
      = (int)cuda_info_properties::version
	| (int)cuda_info_properties::virtual_sm_version;

  next_expect (token_type::start_object);

  token string_token;
  token integer_token;
  int fields_mask = 0;

  for (;;)
    {
      string_token = next_expect (token_type::string_value);
      next_expect (token_type::key_value_separator);

      if (string_token.m_string == "nv_note_cuver")
	{
	  update_bitmask_field (
	      fields_mask, (int)cuda_info_properties::version, string_token);
	  integer_token = next_expect (token_type::integer_value);
	  metadata.m_note_nv_cuda_version = integer_token.m_value.integer;
	}
      else if (string_token.m_string == "nv_note_cu_virt_smver")
	{
	  update_bitmask_field (fields_mask,
				(int)cuda_info_properties::virtual_sm_version,
				string_token);
	  integer_token = next_expect (token_type::integer_value);
	  metadata.m_note_nv_cuda_virtual_sm_version
	      = integer_token.m_value.integer;
	}
      else
	{
	  skip_value (string_token.m_string);
	}

      if (peek ().m_type == token_type::end_object)
	{
	  break;
	}

      next_expect (token_type::element_separator);
    }

  next_expect (token_type::end_object);

  validate_fields_bitmask (fields_mask, (int)required_fields_mask,
			   "Missing required fields in CUver object");
}

void
parser::parse_functions_list (std::vector<schema_function> &functions)
{
  next_expect (token_type::start_array);

  functions.clear ();

  if (peek ().m_type != token_type::end_array)
    {
      for (;;)
	{
	  schema_function function;

	  parse_function (function);
	  if (m_on_function)
	    {
	      m_on_function (function);
	    }
	  else
	    {
	      functions.push_back (function);
	    }

	  if (peek ().m_type == token_type::end_array)
	    {
	      break;
	    }

	  next_expect (token_type::element_separator);
	}
    }

  next_expect (token_type::end_array);
}

void
parser::parse_function (schema_function &function)
{
  enum class function_properties : int
  {
    function_name = (1 << 0),
    start = (1 << 1),
    length = (1 << 2),
    sass_instructions = (1 << 3),
    other_attributes = (1 << 4),
  };
  static constexpr int required_fields_mask
      = (int)function_properties::function_name
	| (int)function_properties::start | (int)function_properties::length
	| (int)function_properties::sass_instructions;

  next_expect (token_type::start_object);

  token string_token;
  token integer_token;
  int fields_mask = 0;

  for (;;)
    {
      string_token = next_expect (token_type::string_value);
      next_expect (token_type::key_value_separator);

      if (string_token.m_string == "function-name")
	{
	  update_bitmask_field (fields_mask,
				(int)function_properties::function_name,
				string_token);
	  string_token = next_expect (token_type::string_value);
	  function.m_function_name = std::move (string_token.m_string);
	}
      else if (string_token.m_string == "start")
	{
	  update_bitmask_field (fields_mask, (int)function_properties::start,
				string_token);
	  integer_token = next_expect (token_type::integer_value);
	  function.m_start = integer_token.m_value.integer;
	}
      else if (string_token.m_string == "length")
	{
	  update_bitmask_field (fields_mask, (int)function_properties::length,
				string_token);
	  integer_token = next_expect (token_type::integer_value);
	  function.m_length = integer_token.m_value.integer;
	}
      else if (string_token.m_string == "sass-instructions")
	{
	  update_bitmask_field (fields_mask,
				(int)function_properties::sass_instructions,
				string_token);
	  parse_sass_instructions_list (function.m_instructions);
	}
      else if (string_token.m_string == "other-attributes")
	{
	  update_bitmask_field (fields_mask,
				(int)function_properties::other_attributes,
				string_token);
	  parse_string_list (function.m_other_attributes);
	}
      else
	{
	  skip_value (string_token.m_string);
	}

      if (peek ().m_type == token_type::end_object)
	{
	  break;
	}

      next_expect (token_type::element_separator);
    }

  next_expect (token_type::end_object);

  validate_fields_bitmask (fields_mask, (int)required_fields_mask,
			   "Missing required fields in Function object");
}

void
parser::parse_sass_instructions_list (
    std::vector<schema_sass_instruction> &instructions)
{
  next_expect (token_type::start_array);

  instructions.clear ();

  if (peek ().m_type != token_type::end_array)
    {
      for (;;)
	{
	  schema_sass_instruction instruction;

	  parse_sass_instruction (instruction);
	  instructions.push_back (instruction);

	  if (peek ().m_type == token_type::end_array)
	    {
	      break;
	    }

	  next_expect (token_type::element_separator);
	}
    }

  next_expect (token_type::end_array);
}

enum class sass_instruction_fields : int
{
  predicate = (1 << 0),
  opcode = (1 << 1),
  operands = (1 << 2),
  extra = (1 << 3),
  other_attributes = (1 << 4),
  other_flags = (1 << 5),
};

void
parser::parse_sass_instruction (schema_sass_instruction &instruction)
{
  static constexpr int required_fields_mask
      = (int)sass_instruction_fields::opcode;

  next_expect (token_type::start_object);

  token string_token;
  int fields_mask = 0;

  for (;;)
    {
      string_token = next_expect (token_type::string_value);
      next_expect (token_type::key_value_separator);

      parse_sass_instruction_field (
	  instruction, string_token.m_string,
	  fields_mask); // Check return if we want to log ignored fields

      if (peek ().m_type == token_type::end_object)
	{
	  break;
	}

      next_expect (token_type::element_separator);
    }

  next_expect (token_type::end_object);

  validate_fields_bitmask (
      fields_mask, (int)required_fields_mask,
      "Missing required fields in schema_sass_instruction object");
}

bool
parser::parse_sass_instruction_field (schema_sass_instruction &instruction,
				      const std::string &field,
				      int &fields_mask)
{
  // Fields accepted:
  // - predicate
  // - opcode
  // - operands
  // - extra
  // - other-attributes
  // - other-flags
  static constexpr size_t min_field_length = 4;
  static constexpr size_t min_field_length_others = 7;

  if (field.length () < min_field_length)
    {
      skip_value (field);
      return false;
    }

  {
    token string_token;

    switch (field[0])
      {
      case 'p': // p
	string_token = next_expect (token_type::string_value);
	instruction.m_predicate = std::move (string_token.m_string);
	return true;
      case 'o': // o
	switch (field[1])
	  {
	  case 'p': // op
	    switch (field[2])
	      {
	      case 'c': // opc
		EXPECT_KEY_EQUALS ("opcode", field, string_token);
		update_bitmask_field (fields_mask,
				      (int)sass_instruction_fields::opcode,
				      string_token);
		string_token = next_expect (token_type::string_value);
		instruction.m_opcode = std::move (string_token.m_string);
		return true;
	      case 'e': // ope
		EXPECT_KEY_EQUALS ("operands", field, string_token);
		update_bitmask_field (fields_mask,
				      (int)sass_instruction_fields::operands,
				      string_token);
		string_token = next_expect (token_type::string_value);
		instruction.m_operands = std::move (string_token.m_string);
		return true;
	      default:
		skip_value (field);
		break;
	      }
	    break;
	  case 't': // ot
	    if (field.length () < min_field_length_others)
	      {
		skip_value (field);
		break;
	      }
	    switch (field[6])
	      {
	      case 'a': // ot----a
		EXPECT_KEY_EQUALS ("other-attributes", field, string_token);
		update_bitmask_field (
		    fields_mask,
		    (int)sass_instruction_fields::other_attributes,
		    string_token);
		parse_other_attributes_dict (instruction);
		return true;
	      case 'f': // ot----f
		EXPECT_KEY_EQUALS ("other-flags", field, string_token);
		update_bitmask_field (
		    fields_mask, (int)sass_instruction_fields::other_flags,
		    string_token);
		parse_string_list (instruction.m_other_flags);
		return true;
	      default:
		skip_value (field);
		break;
	      }
	    break;
	  default:
	    skip_value (field);
	    break;
	  }
	break;
      case 'e': // e
	EXPECT_KEY_EQUALS ("extra", field, string_token);
	update_bitmask_field (fields_mask, (int)sass_instruction_fields::extra,
			      string_token);
	string_token = next_expect (token_type::string_value);
	instruction.m_extra = std::move (string_token.m_string);
	break;
      default:
	skip_value (field);
	break;
      }
  }

  return false;
}

void
parser::parse_string_list (std::vector<std::string> &strings)
{
  next_expect (token_type::start_array);

  strings.clear ();

  if (peek ().m_type != token_type::end_array)
    {
      for (;;)
	{
	  token string_token;
	  string_token = next_expect (token_type::string_value);
	  strings.push_back (std::move (string_token.m_string));

	  if (peek ().m_type == token_type::end_array)
	    {
	      break;
	    }

	  next_expect (token_type::element_separator);
	}
    }

  next_expect (token_type::end_array);
}

void
parser::parse_other_attributes_dict (schema_sass_instruction &instruction)
{
  next_expect (token_type::start_object);

  auto &dictionary = instruction.m_other_attributes;
  dictionary.clear ();

  if (peek ().m_type != token_type::end_object)
    {
      for (;;)
	{
	  token string_token;
	  string_token = next_expect (token_type::string_value);

	  next_expect (token_type::key_value_separator);

	  token value_token;
	  value_token = next_expect (token_type::string_value);

	  parse_instruction_other_attribute_field (
	      instruction, string_token.m_string, value_token.m_string);

	  dictionary.emplace_back (std::move (string_token.m_string),
				   std::move (value_token.m_string));

	  if (peek ().m_type == token_type::end_object)
	    {
	      break;
	    }

	  next_expect (token_type::element_separator);
	}
    }

  next_expect (token_type::end_object);
}

void
parser::parse_instruction_other_attribute_field (
    schema_sass_instruction &instruction, const std::string &field,
    const std::string &value)
{
  // Fields:
  // - control-flow

  if (field == "control-flow")
    {
      instruction.m_opt_is_control_flow = (value == "True");
    }
}

inline token
parser::next_expect (const token_type type)
{
  const auto token = next ();
  if (token.m_type != type)
    throw_error (errors::NOT_SUPPORTED_ERROR,
		 "Parser error: Expected token type %s, but got: %s",
		 token_type_to_string (type), token.to_string ().c_str ());
  return token;
}

void
parser::skip_value (const std::string &field)
{
  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "Skipping JSON field: %s\n",
		     field.c_str ());

  switch (peek ().m_type)
    {
    case token_type::start_object:
      return skip_dictionary ();
    case token_type::start_array:
      return skip_array ();
    case token_type::string_value:
    case token_type::integer_value:
    case token_type::true_literal:
    case token_type::false_literal:
    case token_type::null_literal:
      next ();
      break;
    default:
      throw_error (errors::NOT_SUPPORTED_ERROR,
		   "Parser error: Unexpected token type in value: %s",
		   token_type_to_string (peek ().m_type));
    }
}

void
parser::skip_dictionary ()
{
  uint64_t depth = 1;
  next_expect (token_type::start_object);

  while (depth > 0)
    {
      switch (next ().m_type)
	{
	case token_type::start_object:
	  depth++;
	  break;
	case token_type::end_object:
	  depth--;
	  break;
	default: // skip everything else to avoid unbound recursion
		 // This will ignore semantic errors in the dictionary
	  break;
	}
    }
}

void
parser::skip_array ()
{
  uint64_t depth = 1;
  next_expect (token_type::start_array);

  while (depth > 0)
    {
      switch (next ().m_type)
	{
	case token_type::start_array:
	  depth++;
	  break;
	case token_type::end_array:
	  depth--;
	  break;
	default: // skip everything else to avoid unbound recursion
		 // This will ignore semantic errors in the array
	  break;
	}
    }
}

template <typename... Args>
inline void
parser::validate_fields_bitmask (int bitmask, int required, const char *fmt,
				 Args... args)
{
  if ((bitmask & required) != required)
    throw_error (errors::NOT_SUPPORTED_ERROR, fmt, args...);
}

inline void
parser::update_bitmask_field (int &bitmask, const int field,
			      const token &token)
{
  if (bitmask & field)
    {
      throw_error (NOT_SUPPORTED_ERROR, "Duplicate field at pos %" PRINT_POS,
		   token.m_start_pos);
    }
  bitmask |= field;
}

const char *
parser::parser_state_to_string (const parser_state state)
{
  switch (state)
    {
    case parser_state::at_metadata_start:
      return "AtMetadataStart";
    case parser_state::at_functions_start:
      return "AtFunctionsStart";
    case parser_state::in_functions:
      return "InFunctions";
    case parser_state::at_end:
      return "AtEnd";
    default:
      return "Unknown";
    }
}
}