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
#include "defs.h"

#include <ctype.h>

#include "block.h"
#include "complaints.h"
#include "exceptions.h"
#include "frame.h"
#include "main.h"

#include "cuda-api.h"
#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-elf-image.h"
#include "cuda-modules.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"

#define MAX_BUFFER_SIZE 4096

cuda_disassembler::cuda_disassembler ()
    : m_sass (NULL), m_objfile (NULL), m_current_offset ((uint64_t)-1LL),
      m_offset_insn_line ("[ \t]*/\\*([0-9a-f]+)\\*/[ \t]+(.*)[ \t]*;.*"),
      m_code_line ("[ \t]*/\\*[ \t]*0x([0-9a-f]+)[ \t]*\\*/.*"),
      m_func_header_line ("[ \t]*Function[ \t]*:[ \t]*([0-9A-Za-z_\\$]*)[ \t]*"),
      m_section_header_line ("[ \t]*([0-9A-Za-z_\\$\\.]*)[ \t]*:[ \t]*")
{
}

/* Search for executable in PATH, cuda-gdb launch folder or current folder */
bool
cuda_disassembler::exists (const std::string &fname)
{
  struct stat buf;
  return stat (fname.c_str (), &buf) == 0;
}

const std::string
cuda_disassembler::find_executable (const std::string &name)
{
  std::string path = getenv ("PATH");
  std::string::size_type pos = 0;
  std::string::size_type where = 0;
  while ((where = path.find (":", pos)) != std::string::npos)
    {
      std::string test_path = path.substr (pos, where) + "/" + name;
      if (exists (test_path))
        return test_path;
      pos = where + 1;
    }

  std::string gdb_path = std::string (get_gdb_program_name ());
  auto slash = gdb_path.rfind ("/");
  if (slash != std::string::npos)
    {
      std::string executable = gdb_path.substr (0, slash + 1) + name;
      if (exists (executable))
        return executable;
    }

  return name;
}

const std::string
cuda_disassembler::find_cuobjdump (void)
{
  if (m_cuobjdump_path.size () > 0)
    return m_cuobjdump_path;

  m_cuobjdump_path = find_executable ("cuobjdump");

  return m_cuobjdump_path;
}

//
//  Example cuobjdump Maxwell/Pascal output:
//
//              Function : cudaMalloc
//        .headerflags    @"EF_CUDA_SM61 EF_CUDA_PTX_SM(EF_CUDA_SM61)"
//                                                                          /* 0x007fbc0321e01fef */
//        /*0008*/                   IADD32I R1, R1, -0x8 ;                 /* 0x1c0fffffff870101 */
//        /*0010*/                   S2R R0, SR_LMEMHIOFF ;                 /* 0xf0c8000003770000 */
//        /*0018*/                   ISETP.GE.U32.AND P0, PT, R1, R0, PT ;  /* 0x5b6c038000070107 */
//                                                                          /* 0x007fbc03fde01fef */
//        /*0028*/               @P0 BRA 0x38 ;                             /* 0xe24000000080000f */
//        /*0030*/                   BPT.TRAP 0x1 ;                         /* 0xe3a00000001000c0 */
//        /*0038*/                   MOV R7, R7 ;                           /* 0x5c98078000770007 */
//                                                                          /* 0x007fbc03fde01fef */
//        /*0048*/                   MOV R6, R6 ;                           /* 0x5c98078000670006 */
//        /*0050*/                   MOV R5, R5 ;                           /* 0x5c98078000570005 */
//        /*0058*/                   MOV R4, R4 ;                           /* 0x5c98078000470004 */
//                                                                          /* 0x007fbc0321e01fef */
//
//  Example cuobjdump Volta+ output:
//
//		Function : fabsf
//	.headerflags    @"EF_CUDA_SM75 EF_CUDA_PTX_SM(EF_CUDA_SM75)"
//        /*0000*/                   MOV R4, R4 ;                           /* 0x0000000400047202 */
//                                                                          /* 0x003fde0000000f00 */
//        /*0010*/                   MOV R4, R4 ;                           /* 0x0000000400047202 */
//                                                                          /* 0x003fde0000000f00 */
//        /*0020*/                   FADD R4, -RZ, |R4| ;                   /* 0x40000004ff047221 */
//
//
//  Example cuobjdump .nv.uft output:
//                .nv.uft :
//        .headerflags    @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM90 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM90)"
//        /*0000*/                   CALL.ABS.NOINC 0x0 ;    /* 0x0000000000007943 */
//                                                           /* 0x003fde0003c00000 */
//        /*0010*/                   RET.ABS.NODEC R20 0x0 ; /* 0x0000000014007950 */
//                                                           /* 0x003fde0003e00000 */


enum cuda_disassembler::line_type
cuda_disassembler::parse_next_line (void)
{
  /* Initialize the parser state before proceeding */
  m_current_offset = (uint64_t)-1LL;

  /* clear parser state strings */
  m_current_func.clear ();
  m_current_line.clear ();
  m_current_section.clear ();

  /* Read the next line */
  char line_buffer[MAX_BUFFER_SIZE];
  if (!fgets (line_buffer, sizeof (line_buffer), m_sass))
    return LINE_TYPE_EOF;

  m_current_line = std::string (line_buffer);

  /* Look for a Function header */
  std::cmatch func_header;
  if (regex_search (line_buffer, func_header, m_func_header_line))
    {
      m_current_func = func_header.str (1);
      return LINE_TYPE_FUNC_HEADER;
    }

  /* Look for leading offset followed by an insn */
  std::cmatch offset_insn;
  if (regex_search (line_buffer, offset_insn, m_offset_insn_line))
    {

      /* extract the offset */
      const std::string &offset_str = offset_insn.str (1);
      m_current_offset = strtoull (offset_str.c_str (), NULL, 16);

      /* If necessary, trim mnemonic length */
      m_current_insn = offset_insn.str (2);
      return LINE_TYPE_OFFSET_INSN;
    }

  /* Look for a code-only line, nothing to extract */
  if (regex_search (line_buffer, m_code_line))
    return LINE_TYPE_CODE_ONLY;

  /* Look for a Section header - very permissive pattern, check last */
  std::cmatch section_header;
  if (regex_search (line_buffer, section_header, m_section_header_line))
    {
      m_current_section = section_header.str (1);
      return LINE_TYPE_SECTION_HEADER;
    }

  /* unknown line */
  return LINE_TYPE_UNKNOWN;
}

void
cuda_disassembler::parse_disasm_output (void)
{
  /* instruction encoding-only lines are 8 bytes each */
  const uint32_t disasm_line_size = 8;

  /* Maxwell/Pascal blank lines should be recorded,
     Volta+ should not. */
  auto sm_version = cuda_state::device_get_sm_version (
      cuda_current_focus::get ().physical ().dev ());
  const bool volta_plus = (sm_version >= 70);

  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "Volta+: %s",
                     volta_plus ? "true" : "false");

  /* parse the sass output and insert each instruction found */
  uint64_t last_pc = 0;
  uint64_t entry_pc = 0;
  while (true)
    {
      uint64_t pc = 0;

      /* parse the line and determine it's type */
      auto line_type = parse_next_line ();
      switch (line_type)
        {
        case LINE_TYPE_UNKNOWN:
          /* skip */
          cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "unknown-line: %s",
                             m_current_line.c_str ());
          continue;

        case LINE_TYPE_FUNC_HEADER:
          if (!m_current_func.empty ())
            {
              cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                                 "function header: %s",
                                 m_current_func.c_str ());
              /* Lookup the symbol to get the entry_pc value from the bound
               * minimal symbol */
              struct bound_minimal_symbol sym = lookup_minimal_symbol (
                  m_current_func.c_str (), NULL, m_objfile);
              if (sym.minsym == NULL)
                {
                  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                                     _ ("\"%s\" found in disassembly but has "
                                        "no minimal symbol"),
                                     m_current_func.c_str ());
                  complaint (_ ("\"%s\" found in disassembly but has no "
                                "minimal symbol"),
                             m_current_func.c_str ());
                }
              else
                {
                  entry_pc = BMSYMBOL_VALUE_ADDRESS (sym);
                  if (!entry_pc)
                    complaint (
                        _ ("\"%s\" exists in this program but entry_pc == 0"),
                        m_current_func.c_str ());
                  else if (!volta_plus && ((entry_pc & 0x1f) == 0x08))
                    entry_pc &= ~0x08;
                  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                                     "found \"%s\" at pc 0x%lx",
                                     m_current_func.c_str (), entry_pc);
                }
            }
          break;

        case LINE_TYPE_SECTION_HEADER:
          if (!m_current_section.empty ())
            {
	      // Check for known section names
	      std::string sym_name;
	      if (!m_current_section.compare (".nv.uft"))
		sym_name = "__UFT";
	      else
		{
                  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
				     "section header '%s' unknown", m_current_section.c_str ());
		  break;
		}
              cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                                 "section header: '%s' sym_name '%s'",
                                 m_current_section.c_str (), sym_name.c_str ());
              /* Lookup the symbol to get the entry_pc value from the bound
               * minimal symbol */
              struct bound_minimal_symbol sym = lookup_minimal_symbol (sym_name.c_str (), NULL, m_objfile);
              if (sym.minsym == NULL)
                {
                  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                                     "'%s' found in disassembly but has no minimal symbol for '%s'",
                                     m_current_section.c_str (),
				     sym_name.c_str ());
                  complaint ("'%s' found in disassembly but has no minimal symbol for '%s'",
			     m_current_section.c_str (),
			     sym_name.c_str ());
                }
              else
                {
                  entry_pc = BMSYMBOL_VALUE_ADDRESS (sym);
                  if (!entry_pc)
                    complaint ("'%s' exists in this program but entry_pc == 0", sym_name.c_str ());
                  else if (!volta_plus && ((entry_pc & 0x1f) == 0x08))
                    entry_pc &= ~0x08;
                  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "found '%s' at pc 0x%lx", sym_name.c_str (), entry_pc);
                }
            }
          break;

        case LINE_TYPE_OFFSET_INSN:
          cuda_trace_domain (
              CUDA_TRACE_DISASSEMBLER,
              "offset-insn: entry_pc 0x%lx offset 0x%lx pc 0x%lx insn: %s",
              entry_pc, m_current_offset, entry_pc + m_current_offset,
              m_current_insn.c_str ());
          if ((m_current_insn.size () > 0)
              && (m_current_offset != (uint64_t)-1LL) && entry_pc)
            {
              pc = entry_pc + m_current_offset;

              /* insert the disassembled instruction into the map */
              m_elf_map[pc] = m_current_insn;
              last_pc = pc;
              cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                                 "offset-insn: cache pc 0x%lx insn: %s", pc,
                                 m_current_insn.c_str ());
            }
          else
            cuda_trace_domain (
                CUDA_TRACE_DISASSEMBLER,
                "offset-insn: could not cache pc 0x%lx insn: %s", entry_pc,
                m_current_insn.c_str ());
          break;

        case LINE_TYPE_CODE_ONLY:
          cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                             "code-only: last_pc 0x%lx line_size %d", last_pc,
                             disasm_line_size);
          if (last_pc)
            {
              if (volta_plus)
                {
                  /* skip non-offset lines on Volta+, but still count them */
                  last_pc += disasm_line_size;
                  continue;
                }
              pc = last_pc + disasm_line_size;
            }
          else
            {
              /* first line is a non-offset/code-only line, use the entry pc */
              pc = entry_pc;
            }
          /* Insert the code-only line into the map */
          if (!pc)
            complaint (_ ("code-only line with pc of 0"));
          else
            m_elf_map[pc] = std::string ();
          last_pc = pc;
          break;

        case LINE_TYPE_EOF:
          /* We're done */
          cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "EOF");
          return;

        default:
          /* should never happen regardless of input */
          error ("unknown line-type encountered");
        }
    }
}

bool
cuda_disassembler::populate_from_elf_image (uint64_t pc, std::string &insn)
{
  /* collect all the necessary data */
  kernel_t kernel = cuda_current_focus::get ().logical ().kernel ();
  module_t module = kernel_get_module (kernel);
  elf_image_t elf_img = module_get_elf_image (module);
  m_objfile = cuda_elf_image_get_objfile (elf_img);
  const char *filename = m_objfile->original_name;

  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "populate (ELF): pc 0x%lx", pc);

  /* Generate the dissassembled code by using cuobjdump Can be
     per-function (faster, but may be invoked multiple times for a
     given file), or per-file (slower at first, but then
     faster). Defaults to trying per-function.
   */

  /* Use a temp gdb::unique_xmalloc_ptr<char> to make sure
     the buffer isn't freed before use. */

  gdb::unique_xmalloc_ptr<char> function_name;
  if (cuda_options_disassemble_per_function ())
    function_name = cuda_find_function_name_from_pc (pc, false);

  std::string command = find_cuobjdump ();
  command += std::string {" --dump-sass "};
  command += std::string {filename};
  if (function_name.get ())
    {
      command += std::string {" --function "};
      command += std::string {function_name.get ()};
    }

  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                     "disassembler command (ELF): pc 0x%lx: %s", pc, command.c_str ());
  m_sass = popen (command.c_str (), "r");
  if (!m_sass)
    error ("Cannot disassemble from the ELF image %s", filename);

  /* Parse the SASS output one line at a time */
  parse_disasm_output ();

  /* close the sass file */
  pclose (m_sass);

  /* we expect to always be able to diassemble at least one instruction */
  if (cuda_options_debug_strict () && m_elf_map.empty ())
    error ("Unable to disassemble a single device instruction from %s", filename);

  /* return false if still not found */
  if (m_elf_map.find (pc) == m_elf_map.end ())
    {
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                         "disasm from ELF: failed to disassemble instruction "
                         "at 0x%lx from %s",
                         pc, filename);
      return false;
    }

  /* it's in the cache, return it */
  insn = m_elf_map.at (pc);
  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "disasm (ELF): pc 0x%lx: %s", pc,
                     insn.c_str ());
  return true;
}

bool
cuda_disassembler::populate_from_device_memory (uint64_t pc, std::string &insn)
{
  uint32_t inst_size = 0;
  char buf[MAX_BUFFER_SIZE];

  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "populate (debugAPI): pc 0x%lx",
                     pc);

  buf[0] = '\0';
  cuda_debugapi::disassemble (cuda_current_focus::get ().physical ().dev (),
                              pc, &inst_size, buf, sizeof (buf));

  if (buf[0] == '\0')
    return false;

  m_device_map[pc] = std::string (buf);
  insn = m_device_map.at (pc);
  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                     "disasm (debugAPI): pc 0x%lx: %s", insn.c_str ());
  return true;
}

bool
cuda_disassembler::disassemble (uint64_t pc, std::string &insn,
                                uint32_t &inst_size)
{
  inst_size = 0;
  insn.clear ();

  /* If no CUDA yet, or we don't have device focus, we can't
     disassemble so return NULL */
  if (!cuda_initialized || !cuda_current_focus::isDevice ())
    return false;

  /* Get the instruction size if we don't already have it */
  inst_size = cuda_state::device_get_insn_size (
      cuda_current_focus::get ().physical ().dev ());

  /* Disassemble the instruction(s) */
  if (cuda_options_disassemble_from_elf_image ())
    {
      if (m_elf_map.find (pc) != m_elf_map.end ())
        {
          insn = m_elf_map.at (pc);
          cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                             "disasm from ELF (cached): pc 0x%lx: %s", pc,
                             insn.c_str ());
          return true;
        }

      /* No luck finding it in the ELF map, disassemble and update the map */
      return populate_from_elf_image (pc, insn);
    }

  /* If the pc is cached, return it's disassembly */
  if (m_device_map.find (pc) != m_device_map.end ())
    {
      insn = m_device_map.at (pc);
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
                         "disasm from debugAPI (cached): pc 0x%lx: %s", pc,
                         insn.c_str ());
      return true;
    }

  /* No luck finding it in the device map, disassemble and
     update the map directly */
  return populate_from_device_memory (pc, insn);
}
