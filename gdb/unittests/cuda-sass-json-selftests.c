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

#include "gdbsupport/common-utils.h"
#include "gdbsupport/selftest.h"

#include <unistd.h>

#include <chrono>
#include <iostream>

#include "cuda/cuda-sass-json.h"

#define EXPECT_EXCEPTION(_statement, _error)                                  \
  do                                                                          \
    {                                                                         \
      try                                                                     \
	{                                                                     \
	  _statement;                                                         \
	}                                                                     \
      catch (const gdb_exception &e)                                          \
	{                                                                     \
	  SELF_CHECK (e.error == _error);                                     \
	  break;                                                              \
	}                                                                     \
      catch (...)                                                             \
	{                                                                     \
	  std::cerr << "Statement threw unexpected exception type\n";         \
	  SELF_CHECK (false && "Statement threw unexpected exception type");  \
	}                                                                     \
      std::cerr << "Statement did not throw\n";                               \
      SELF_CHECK (false && "Statement did not throw");                        \
    }                                                                         \
  while (0)

namespace selftests
{
using namespace cuda_disasm_json;

class cuda_sass_json
{
public:
  static const char *sass_cuda_12_7_json;

  static lexer
  make_lexer (const char *json_string)
  {
    auto stream = std::make_unique<util_stream::string_stream> (json_string);
    return lexer (std::move (stream));
  }

  static parser
  make_parser (const char *json_string)
  {
    auto stream = std::make_unique<util_stream::string_stream> (json_string);
    return parser (std::move (stream));
  }

  static parser
  make_parser (std::string &&json_string)
  {
    auto stream = std::make_unique<util_stream::string_stream> (
	std::move (json_string));
    return parser (std::move (stream));
  }

  static void
  test_cuda_sass_json_lexer_tokens ()
  {
    auto lexer = make_lexer ("12 { ]true false null \"abcd\"-54321\t\n:,}[");
    SELF_CHECK (lexer.peek ().m_type == token_type::integer_value);
    SELF_CHECK (lexer.next ().m_value.integer == 12);
    SELF_CHECK (lexer.next ().m_type == token_type::start_object);
    SELF_CHECK (lexer.next ().m_type == token_type::end_array);
    SELF_CHECK (lexer.next ().m_type == token_type::true_literal);
    SELF_CHECK (lexer.next ().m_type == token_type::false_literal);
    SELF_CHECK (lexer.next ().m_type == token_type::null_literal);
    SELF_CHECK (lexer.peek ().m_type == token_type::string_value);
    SELF_CHECK (lexer.next ().m_string == "abcd");
    SELF_CHECK (lexer.peek ().m_type == token_type::integer_value);
    SELF_CHECK (lexer.next ().m_value.integer == -54321);
    SELF_CHECK (lexer.next ().m_type == token_type::key_value_separator);
    SELF_CHECK (lexer.next ().m_type == token_type::element_separator);
    SELF_CHECK (lexer.next ().m_type == token_type::end_object);
    SELF_CHECK (lexer.next ().m_type == token_type::start_array);
    SELF_CHECK (lexer.next ().m_type == token_type::end_of_input);

    lexer = make_lexer ("");
    SELF_CHECK (lexer.next ().m_type == token_type::end_of_input);

    lexer = make_lexer ("01"); // Test that leading zeros are not allowed.
			       // Update if we want to disallow them.
    SELF_CHECK (lexer.peek ().m_type == token_type::integer_value);
    SELF_CHECK (lexer.next ().m_value.integer == 1);
  }

  static void
  test_cuda_sass_json_lexer_string_escapes ()
  {
    auto lexer = make_lexer (R"("\"\\\/\b\f\n\r\t")");
    SELF_CHECK (lexer.next ().m_string == "\"\\/\b\f\n\r\t");

    lexer = make_lexer (R"("\")");
    EXPECT_EXCEPTION (lexer.next (), NOT_SUPPORTED_ERROR);

    lexer = make_lexer (R"("\)");
    EXPECT_EXCEPTION (lexer.next (), NOT_SUPPORTED_ERROR);
  }

  static void
  test_cuda_sass_json_lexer_bad_tokens ()
  {
    auto lexer = make_lexer ("1\"aa");
    SELF_CHECK (lexer.next ().m_type == token_type::integer_value);
    EXPECT_EXCEPTION (lexer.next (), NOT_SUPPORTED_ERROR);

    lexer = make_lexer ("nulL");
    EXPECT_EXCEPTION (lexer.next (), NOT_SUPPORTED_ERROR);

    lexer = make_lexer ("tru");
    EXPECT_EXCEPTION (lexer.next (), NOT_SUPPORTED_ERROR);

    lexer = make_lexer ("fals");
    EXPECT_EXCEPTION (lexer.next (), NOT_SUPPORTED_ERROR);

    lexer = make_lexer ("$");
    EXPECT_EXCEPTION (lexer.next (), NOT_SUPPORTED_ERROR);

    lexer = make_lexer ("\"hmm \\");
    EXPECT_EXCEPTION (lexer.next (), NOT_SUPPORTED_ERROR);

    lexer = make_lexer (
	"9223372036854775807 -9223372036854775807 9223372036854775808");
    SELF_CHECK (lexer.next ().m_value.integer == 9223372036854775807);
    SELF_CHECK (lexer.next ().m_value.integer == -9223372036854775807);
    EXPECT_EXCEPTION (lexer.next (), NOT_SUPPORTED_ERROR);
  }

  static void
  test_cuda_sass_json_parser_schema_version ()
  {
    auto parser = make_parser (R"(
        {
            "revision": 100,
            "minor": 3,
            "major": 10
        }
    )");
    schema_metadata version;
    parser.parse_schema_version (version);
    SELF_CHECK (version.m_schema_ver_revision == 100);
    SELF_CHECK (version.m_schema_ver_minor == 3);
    SELF_CHECK (version.m_schema_ver_major == 10);
  }

  static void
  test_cuda_sass_json_parser_metadata ()
  {
    auto parser = make_parser (R"(
        {
            "ELF": {
                "layout-id": 4,
                "ei_osabi": 65,
                "ei_abiversion": 8
            },
            "SchemaVersion": {
                "major": 12,
                "minor": 7,
                "revision": 0
            },
            "SM": {
                "version": {
                    "major": 10,
                    "minor": 2
                }
            },
            "Producer": "disassembler",
            "Description": "metadata m_description",
            "additional-field": "this field is not part of the schema and should be ignored by the parser"
        }
    )");

    schema_metadata metadata;
    parser.parse_metadata (metadata);
    SELF_CHECK (metadata.m_elf_layout_id == 4);
    SELF_CHECK (metadata.m_elf_ei_osabi == 65);
    SELF_CHECK (metadata.m_elf_ei_abiversion == 8);
    SELF_CHECK (metadata.m_schema_ver_major == 12);
    SELF_CHECK (metadata.m_schema_ver_minor == 7);
    SELF_CHECK (metadata.m_schema_ver_revision == 0);
    SELF_CHECK (metadata.m_sm_ver_major == 10);
    SELF_CHECK (metadata.m_sm_ver_minor == 2);
    SELF_CHECK (metadata.m_producer == "disassembler");
    SELF_CHECK (metadata.m_description == "metadata m_description");
  }

  static void
  test_cuda_sass_json_parser_full_metadata ()
  {
    auto parser = make_parser (R"(
        {
            "ELF": {
                "layout-id": 4,
                "ei_osabi": 65,
                "ei_abiversion": 8
            },
            "SchemaVersion": {
                "major": 12,
                "minor": 7,
                "revision": 0
            },
            "SM": {
                "version": {
                    "major": 10,
                    "minor": 2
                }
            },
            "Producer": "disassembler",
            "Description": "metadata m_description",
            "additional-field": "this field is not part of the schema and should be ignored by the parser",
            ".note.nv.tkinfo": {
                "tki_toolkitVersion": 1,
                "tki_objFname": "obj",
                "tki_toolName": "tool",
                "tki_toolVersion": "1.0",
                "tki_toolBranch": "branch",
                "tki_toolOptions": "options
with
newlines"
            },
            ".note.nv.cuver": {
                "nv_note_cuver": 1,
                "nv_note_cu_virt_smver": 2
            }
            }
        }
        )");

    schema_metadata metadata;
    parser.parse_metadata (metadata);

    SELF_CHECK (metadata.m_note_nv_toolkit_version == 1);
    SELF_CHECK (metadata.m_note_nv_toolkit_object_filename == "obj");
    SELF_CHECK (metadata.m_note_nv_toolkit_tool_name == "tool");
    SELF_CHECK (metadata.m_note_nv_toolkit_tool_version == "1.0");
    SELF_CHECK (metadata.m_note_nv_toolkit_tool_branch == "branch");
    SELF_CHECK (metadata.m_note_nv_toolkit_tool_options
		== "options\nwith\nnewlines");
    SELF_CHECK (metadata.m_note_nv_cuda_version == 1);
    SELF_CHECK (metadata.m_note_nv_cuda_virtual_sm_version == 2);
  }

  static void
  test_cuda_sass_json_parser_bad_metadata ()
  {
    auto parser = make_parser (R"(
        {
            "ELF": {
                "layout-id": "4",
                "ei_osabi": 65,
                "ei_abiversion": 8
            }
        }
    )");

    schema_metadata metadata;
    EXPECT_EXCEPTION (parser.parse_metadata (metadata), NOT_SUPPORTED_ERROR);

    parser = make_parser (R"(
        {
            "ELF": []
        }
    )");

    EXPECT_EXCEPTION (parser.parse_metadata (metadata), NOT_SUPPORTED_ERROR);
  }

  static void
  _verify_cuda_sass_json_parse_sample_schema (schema_json schema)
  {
    SELF_CHECK (schema.m_metadata.m_elf_layout_id == 4);
    SELF_CHECK (schema.m_metadata.m_elf_ei_osabi == 65);
    SELF_CHECK (schema.m_metadata.m_elf_ei_abiversion == 8);
    SELF_CHECK (schema.m_metadata.m_schema_ver_major == 12);
    SELF_CHECK (schema.m_metadata.m_schema_ver_minor == 7);
    SELF_CHECK (schema.m_metadata.m_schema_ver_revision == 0);
    SELF_CHECK (schema.m_metadata.m_sm_ver_major == 10);
    SELF_CHECK (schema.m_metadata.m_sm_ver_minor == 2);
    SELF_CHECK (schema.m_metadata.m_producer == "disassembler");
    SELF_CHECK (schema.m_metadata.m_description == "metadata m_description");

    SELF_CHECK (schema.m_functions.size () == 1);
    SELF_CHECK (schema.m_functions[0].m_function_name == "my_function");
    SELF_CHECK (schema.m_functions[0].m_start == 4096);
    SELF_CHECK (schema.m_functions[0].m_length == 512);
    SELF_CHECK (schema.m_functions[0].m_other_attributes.size () == 2);
    SELF_CHECK (schema.m_functions[0].m_other_attributes[0] == "something");
    SELF_CHECK (schema.m_functions[0].m_other_attributes[1] == "else\n");

    SELF_CHECK (schema.m_functions[0].m_instructions.size () == 3);
    SELF_CHECK (schema.m_functions[0].m_instructions[0].m_predicate == "@!P0");
    SELF_CHECK (schema.m_functions[0].m_instructions[0].m_opcode == "MOV");
    SELF_CHECK (schema.m_functions[0].m_instructions[0].m_operands
		== "R0, R1");
    SELF_CHECK (schema.m_functions[0].m_instructions[0].m_extra == "");
    SELF_CHECK (
	schema.m_functions[0].m_instructions[0].m_other_attributes.size ()
	== 2);
    SELF_CHECK (
	schema.m_functions[0].m_instructions[0].m_other_attributes[0].first
	== "control-flow");
    SELF_CHECK (
	schema.m_functions[0].m_instructions[0].m_other_attributes[0].second
	== "True");
    SELF_CHECK (
	schema.m_functions[0].m_instructions[0].m_other_attributes[1].first
	== "another-attribute");
    SELF_CHECK (
	schema.m_functions[0].m_instructions[0].m_other_attributes[1].second
	== "some value");
    SELF_CHECK (schema.m_functions[0].m_instructions[0].m_other_flags.size ()
		== 2);
    SELF_CHECK (schema.m_functions[0].m_instructions[0].m_other_flags[0]
		== "control-flow-flag1");
    SELF_CHECK (schema.m_functions[0].m_instructions[0].m_other_flags[1]
		== "another-flag2");
    SELF_CHECK (schema.m_functions[0].m_instructions[1].m_predicate == "");
    SELF_CHECK (schema.m_functions[0].m_instructions[1].m_opcode == "NOP");
    SELF_CHECK (schema.m_functions[0].m_instructions[1].m_operands == "");
    SELF_CHECK (schema.m_functions[0].m_instructions[1].m_extra == "extra");

    SELF_CHECK (schema.m_functions[0].m_instructions[2].m_predicate == "");
    SELF_CHECK (schema.m_functions[0].m_instructions[2].m_opcode == "");
    SELF_CHECK (schema.m_functions[0].m_instructions[2].m_operands == "");
    SELF_CHECK (schema.m_functions[0].m_instructions[2].m_extra == "");
  }

  static void
  test_cuda_sass_json_stream_parse ()
  {
    auto parser = make_parser (sass_cuda_12_7_json);
    schema_json schema;

    SELF_CHECK (parser.current_state () == parser_state::at_start);
    parser.stream_parse_metadata (schema.m_metadata);
    SELF_CHECK (parser.current_state () == parser_state::at_metadata_end);
    parser.stream_parse_functions (schema.m_functions);
    SELF_CHECK (parser.current_state () == parser_state::at_functions_end);
    parser.stream_parse_end ();
    SELF_CHECK (parser.current_state () == parser_state::at_end);

    _verify_cuda_sass_json_parse_sample_schema (schema);
  }

  static void
  test_cuda_sass_json_fd_parse ()
  {
    int pipefd[2];
    SELF_CHECK (pipe (pipefd) != -1);

    parser parser (std::make_unique<util_stream::file_stream> (pipefd[0]));
    schema_json schema;

    SELF_CHECK (
	write (pipefd[1], sass_cuda_12_7_json, strlen (sass_cuda_12_7_json))
	== strlen (sass_cuda_12_7_json));
    SELF_CHECK (close (pipefd[1]) == 0);

    parser.parse_json (schema);

    _verify_cuda_sass_json_parse_sample_schema (schema);
  }

  static void
  test_cuda_sass_json_nesting_limits ()
  {
    static const size_t nested_level = 1000000;
    const char *fmt = R"(
    [
        {
            "random-field": %s
        }
    ]
    )";

    std::string many_dicts;
    for (int i = 0; i < nested_level; i++)
      {
	many_dicts += R"(
                        {
                        "key":
                )";
      }
    many_dicts += "\"value\"";
    for (int i = 0; i < nested_level; i++)
      {
	many_dicts += "}";
      }

    std::string many_arrays;
    for (int i = 0; i < nested_level; i++)
      {
	many_arrays += "[";
      }
    for (int i = 0; i < nested_level; i++)
      {
	many_arrays += "]";
      }

    std::string many_mixed;
    for (int i = 0; i < nested_level; i++)
      {
	many_mixed += "[{{";
      }
    for (int i = 0; i < nested_level; i++)
      {
	many_mixed += "}}]";
      }

    std::string json = string_printf (fmt, many_dicts.c_str ());
    auto parser = make_parser (std::move (json));
    schema_json schema;
    EXPECT_EXCEPTION (parser.stream_parse_metadata (schema.m_metadata),
		      NOT_SUPPORTED_ERROR); // should not stack overflow

    json = string_printf (fmt, many_arrays.c_str ());
    parser = make_parser (std::move (json));
    EXPECT_EXCEPTION (parser.stream_parse_metadata (schema.m_metadata),
		      NOT_SUPPORTED_ERROR); // should not stack overflow

    json = string_printf (fmt, many_mixed.c_str ());
    parser = make_parser (std::move (json));
    EXPECT_EXCEPTION (parser.stream_parse_metadata (schema.m_metadata),
		      NOT_SUPPORTED_ERROR); // should not stack overflow
  }

  static void
  test_cuda_sass_json_many_instructions ()
  {
    static const size_t num_instructions = 100000;
    std::string json = R"(
    [
        {
            "ELF": {
                "layout-id": 4,
                "ei_osabi": 65,
                "ei_abiversion": 8
            },
            "SchemaVersion": {
                "major": 12,
                "minor": 7,
                "revision": 0
            },
            "SM": {
                "version": {
                    "major": 10,
                    "minor": 2
                }
            },
            "Producer": "disassembler",
            "Description": "metadata m_description",
            "additional-field": "this field is not part of the schema and should be ignored by the parser"
        },
        [
            {
                "function-name": "my_function",
                "start": 4096,
                "length": 512,
                "other-attributes": ["something", "else\n"],
                "sass-instructions": [)";

    for (int i = 0; i < num_instructions; i++)
      {
	json += string_printf (R"(
                    {
                        "predicate": "%i",
                        "opcode": "%i",
                        "operands": "%i",
                        "extra": "",
                        "other-attributes": {
                            "control-flow": "%i",
                            "another-attribute": "%i"
                        },
                        "other-flags": [
                            "%i",
                            "another-flag2"
                        ]
                    })",
			       rand (), rand (), rand (), rand (), rand (),
			       rand ());

	if (i < num_instructions - 1)
	  json += ",";
      }

    json += R"(
                        ]
                }
        ]
    ])";

    std::cout << "Perf test: parsing " << num_instructions << " instructions"
	      << " / " << json.size () << " bytes" << std::endl;
    auto parser = make_parser (std::move (json));
    schema_json schema;
    auto start_time = std::chrono::high_resolution_clock::now ();
    parser.stream_parse_metadata (schema.m_metadata);
    auto metadata_time = std::chrono::high_resolution_clock::now ();
    parser.stream_parse_functions (schema.m_functions);
    auto functions_time = std::chrono::high_resolution_clock::now ();
    parser.stream_parse_end ();
    auto end_time = std::chrono::high_resolution_clock::now ();

    std::chrono::duration<float> metadata_elapsed = metadata_time - start_time;
    std::chrono::duration<float> functions_elapsed
	= functions_time - metadata_time;
    std::chrono::duration<float> total_elapsed = end_time - start_time;

    std::cout << "Metadata parsing time: " << metadata_elapsed.count () << "s"
	      << std::endl;
    std::cout << "Functions parsing time: " << functions_elapsed.count ()
	      << "s" << std::endl;
    std::cout << "Total parsing time: " << total_elapsed.count () << "s"
	      << std::endl;

    SELF_CHECK (schema.m_functions[0].m_instructions.size ()
		== num_instructions);
  }

  static void
  test_cuda_sass_json ()
  {
    test_cuda_sass_json_lexer_tokens ();
    test_cuda_sass_json_lexer_string_escapes ();
    test_cuda_sass_json_lexer_bad_tokens ();
    test_cuda_sass_json_parser_schema_version ();
    test_cuda_sass_json_parser_metadata ();
    test_cuda_sass_json_parser_full_metadata ();
    test_cuda_sass_json_parser_bad_metadata ();
    test_cuda_sass_json_stream_parse ();
    test_cuda_sass_json_nesting_limits ();
    test_cuda_sass_json_fd_parse ();
    test_cuda_sass_json_many_instructions ();
  }
};
}

void _initialize_cuda_sass_json_selftest ();
void
_initialize_cuda_sass_json_selftest ()
{
  selftests::register_test ("cuda_sass_json",
			    selftests::cuda_sass_json::test_cuda_sass_json);
}

const char *selftests::cuda_sass_json::sass_cuda_12_7_json =
    R"(
    [
        {
            "ELF": {
                "layout-id": 4,
                "ei_osabi": 65,
                "ei_abiversion": 8
            },
            "SchemaVersion": {
                "major": 12,
                "minor": 7,
                "revision": 0
            },
            "SM": {
                "version": {
                    "major": 10,
                    "minor": 2
                }
            },
            "Producer": "disassembler",
            "Description": "metadata m_description",
            "additional-field": "this field is not part of the schema and should be ignored by the parser"
        },
        [
            {
                "function-name": "my_function",
                "start": 4096,
                "length": 512,
                "other-attributes": ["something", "else\n"],
                "sass-instructions": [
                    {
                        "predicate": "@!P0",
                        "opcode": "MOV",
                        "operands": "R0, R1",
                        "extra": "",
                        "other-attributes": {
                            "control-flow": "True",
                            "another-attribute": "some value"
                        },
                        "other-flags": [
                            "control-flow-flag1",
                            "another-flag2"
                        ]
                    },
                    {
                        "opcode": "NOP",
                        "extra": "extra"
                    },
                    {
                        "opcode": ""
                    }
                ]
            }
        ]
    ]
)";
