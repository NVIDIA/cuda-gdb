/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2021 NVIDIA Corporation
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
#include "frame.h"
#include "exceptions.h"
#include "main.h"

#include "cuda-api.h"
#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-elf-image.h"
#include "cuda-modules.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-options.h"

#include <regex>
#include <unordered_map>

#define INSN_MAX_LENGTH 4096
#define LINE_MAX_LENGTH 4096
#define PATH_MAX_LENGTH 4096
#define COMMAND_MAX_LENGTH 4096

/******************************************************************************
 *
 *                             Disassembly Cache
 *
 *****************************************************************************/

class cuda_disasm_cache
{
public:
  cuda_disasm_cache ();
  
  const char *find_instruction (uint64_t pc, uint32_t *instruction_size);

  void maybe_flush_device_map (void);

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
  enum line_type m_current_line_type;
  char m_current_line[LINE_MAX_LENGTH];
  char m_current_func[LINE_MAX_LENGTH];
  char m_current_insn[INSN_MAX_LENGTH];

  /* Holder for pc->insn mappings from the cubin/ELF file.
     These are not flushed before resuming the GPU, as the
     ELF file is guarenteed not to change. */
  std::unordered_map<uint64_t, gdb::unique_xmalloc_ptr<char>> m_elf_map;

  /* Holder for pc->insn mappings from the device.
     These are flushed before resuming the GPU, as the
     code may be modified after resuming. */
  std::unordered_map<uint64_t, gdb::unique_xmalloc_ptr<char>> m_device_map;

  /* True if the device map should always be flushed before using
     the debugAPI to disassemble the instruction, this preserves the
     old behavior. Can be overridden by setting the
     CUDA_GDB_NO_FLUSH_DEVICE_DISASSEMBLY environment variable
     to a non-empty and non-zero value.

     set cuda disassemble_from device_memory
  */
  bool m_device_flush;

  /* regex for parsing various line types */
  std::regex m_offset_insn_line;
  std::regex m_code_line;
  std::regex m_func_header_line;

  void parse_disasm_output (void);
  void parse_next_line (void);

  const char *populate_from_elf_image (uint64_t pc);
  const char *populate_from_device_memory (uint64_t pc);

  uint32_t get_sm_major (const char *sm_type);
  uint32_t get_inst_size (uint64_t pc);

  char *find_cuobjdump (void);
  const char *find_executable (const char *name);
  bool exists(const char *fname);
  int gdb_program_dir_len (void);
};

cuda_disasm_cache::cuda_disasm_cache ()
: m_sass (NULL),
  m_objfile (NULL),
  m_current_offset ((uint64_t)-1LL),
  m_current_line_type (LINE_TYPE_UNKNOWN),
  m_device_flush (true),
  m_offset_insn_line ("/\\*([0-9a-f]+)\\*/[ \t]+(.*)[ \t]*;"),
  m_code_line ("/\\*[ \t]*0x([0-9a-f]+)[ \t]*\\*/"),
  m_func_header_line ("[ \t]*Function[ \t]*:[ \t]*([0-9A-Za-z_\\$]*)")
{
  memset (m_current_func, '\0', sizeof (m_current_func));
  memset (m_current_insn, '\0', sizeof (m_current_insn));
  memset (m_current_line, '\0', sizeof (m_current_line));

  /* Allow the environment variable to override the m_device_flush default */
  const char *env = getenv ("CUDA_GDB_NO_FLUSH_DEVICE_DISASSEMBLY");
  if (env && (strlen (env) > 0) && (strtol (env, NULL, 10) != 0))
    m_device_flush = false;
}

void
cuda_disasm_cache::maybe_flush_device_map (void)
{
  /* Only clear the device map as the ELF map is read-only */
  if (m_device_flush)
    m_device_map.clear ();
}

int
cuda_disasm_cache::gdb_program_dir_len (void)
{
  const char *gdb_program_name;
  int len = -1;
  int cnt;

  gdb_program_name = get_gdb_program_name();
  len = strlen(gdb_program_name);
  for (cnt=len; cnt > 1 && gdb_program_name[cnt-1] != '/'; cnt--)
    ;
  if (cnt > 1)
    len = cnt;

  return len;
}

/* Search for executable in PATH, cuda-gdb launch folder or current folder */
bool
cuda_disasm_cache::exists (const char *fname)
{
  struct stat buf;
  return stat (fname, &buf) == 0;
}

const char *
cuda_disasm_cache::find_executable (const char *name)
{
  static char return_path[PATH_MAX_LENGTH];
  const char *gdb_program_name;

  /* Save PATH to local variable because strtok() alters the string */
  char path[PATH_MAX_LENGTH];
  memset (path, '\0', sizeof (path));
  strncpy (path, getenv("PATH"), sizeof(path) - 1);

  for (char *dir = strtok (path, ":"); dir; dir = strtok (NULL, ":"))
    {
      snprintf (return_path, sizeof (return_path), "%s/%s", dir, name);
      if (exists (return_path))
        return return_path;
    }

  gdb_program_name = get_gdb_program_name();
  snprintf (return_path, sizeof (return_path), "%.*s%s",
            gdb_program_dir_len(), gdb_program_name, name);
  if (exists (return_path))
    return return_path;

  return name;
}

char *
cuda_disasm_cache::find_cuobjdump (void)
{
  static char cuobjdump_path[1024];
  static bool cuobjdump_path_initialized = false;

  if (cuobjdump_path_initialized)
    return cuobjdump_path;

  memset (cuobjdump_path, '\0', sizeof (cuobjdump_path));
  strncpy (cuobjdump_path, find_executable ("cuobjdump"), sizeof (cuobjdump_path) - 1);

  cuobjdump_path_initialized = true;

  return cuobjdump_path;
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
//        /*0000*/                   MOV R4, R4 ;                                        /* 0x0000000400047202 */
//                                                                                       /* 0x003fde0000000f00 */
//        /*0010*/                   MOV R4, R4 ;                                        /* 0x0000000400047202 */
//                                                                                       /* 0x003fde0000000f00 */
//        /*0020*/                   FADD R4, -RZ, |R4| ;                                /* 0x40000004ff047221 */
//

void
cuda_disasm_cache::parse_next_line (void)
{
  /* Initialize the parser state before proceeding */
  m_current_offset = (uint64_t)-1LL;
  m_current_line[0] = '\0';
  m_current_line_type = LINE_TYPE_UNKNOWN;

  /* prepare buffers for strncpy() */
  memset (m_current_func, '\0', sizeof (m_current_func));
  memset (m_current_insn, '\0', sizeof (m_current_insn));

  /* Read the next line */
  if (!fgets (m_current_line, sizeof (m_current_line), m_sass))
    {
      m_current_line_type = LINE_TYPE_EOF;
      return;
    }

  /* Look for a Function header */
  std::cmatch func_header;
  if (regex_search (m_current_line, func_header, m_func_header_line))
    {
      m_current_line_type = LINE_TYPE_FUNC_HEADER;

      const std::string &func_str = func_header.str (1);
      strncpy (m_current_func, func_str.c_str (), sizeof (m_current_func) - 1);
      return;
    }

  /* Look for leading offset followed by an insn */
  std::cmatch offset_insn;
  if (regex_search (m_current_line, offset_insn, m_offset_insn_line))
    {
      m_current_line_type = LINE_TYPE_OFFSET_INSN;

      /* extract the offset */
      const std::string &offset_str = offset_insn.str (1);
      m_current_offset = strtoull (offset_str.c_str (), NULL, 16);

      /* If necessary, trim mnemonic length */
      const std::string &mnemonic_str = offset_insn.str (2);
      strncpy (m_current_insn, mnemonic_str.c_str (), sizeof (m_current_insn) - 1);
      return;
    }
  
  /* Look for a code-only line, nothing to extract */
  if (regex_search (m_current_line, m_code_line))
    {
      m_current_line_type = LINE_TYPE_CODE_ONLY;
      return;
    }

  /* unknown line */
}

uint32_t
cuda_disasm_cache::get_sm_major (const char *sm_type)
{
  if (!sm_type || strlen(sm_type) < 4 || strncmp(sm_type, "sm_", 3) != 0)
    error ("unknown sm_type %s", sm_type ? sm_type : "(null)");

  return sm_type[3] - '0';
}

uint32_t
cuda_disasm_cache::get_inst_size (uint64_t pc)
{
  uint32_t inst_size = 0;
  uint32_t devId = cuda_current_device ();

#ifdef __QNXTARGET__
  // On QNX we cannot do target-side disassembly, so infer the instruction size
  // from the SM type as a workaround.
  int sm_major = get_sm_major (device_get_sm_type (devId));
  if (sm_major < 0)
    error ("Failed to get the SM type");

  if (sm_major <= 6)
    inst_size = 8;
  else if (sm_major >= 7)
    inst_size = 16;
  else
    error ("Unknown instruction size");
#else
  cuda_api_disassemble (devId, pc, &inst_size, NULL, 0);
#endif
  return inst_size;
}

void
cuda_disasm_cache::parse_disasm_output(void)
{
  /* instruction encoding-only lines are 8 bytes each */
  const uint32_t disasm_line_size = 8;

  /* Maxwell/Pascal blank lines should be recorded,
     Volta+ should not. */
  const int sm_major = get_sm_major (device_get_sm_type (cuda_current_device ()));
  const bool volta_plus = (sm_major >= 7);

  /* parse the sass output and insert each instruction found */
  uint64_t last_pc = 0;
  uint64_t entry_pc = 0;
  while (true)
    {
      /* parse the line and determine it's type */
      parse_next_line ();

      uint64_t pc = 0;
      switch (m_current_line_type)
        {
        case LINE_TYPE_UNKNOWN:
          /* skip */
          continue;

        case LINE_TYPE_FUNC_HEADER:
          /* We found the next Function header, stop processing this
             function, but return true to keep going */

          /* new lexical scope so lifetime of sym doesn't cross
             case/jump labels */
          {
            /* Lookup the symbol to get the entry_pc value from the bound minimal symbol */
            struct bound_minimal_symbol sym = lookup_minimal_symbol (m_current_func, NULL, m_objfile);
            if (sym.minsym == NULL)
              error (_("\"%s\" exists in this program but is not a function."), m_current_func);

            entry_pc = BMSYMBOL_VALUE_ADDRESS (sym);
            gdb_assert (entry_pc);
	    if (!volta_plus && ((entry_pc & 0x1f) == 0x08))
	      entry_pc &= ~0x08;
          }
          break;

        case LINE_TYPE_OFFSET_INSN:
          gdb_assert (m_current_offset != (uint64_t)-1LL);
          gdb_assert (entry_pc);

          pc = entry_pc + m_current_offset;

          /* insert the disassembled instruction into the map */
          m_elf_map[pc] = gdb::unique_xmalloc_ptr<char> (xstrdup (m_current_insn));
          last_pc = pc;
          break;

        case LINE_TYPE_CODE_ONLY:
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

          gdb_assert (pc);

          /* insert the blank line into the map */
          m_elf_map[pc] = gdb::unique_xmalloc_ptr<char> (xstrdup (""));
          last_pc = pc;
          break;

        case LINE_TYPE_EOF:
          /* We're done */
          return;

        default:
          /* should never happen regardless of input */
          error ("unknown line-type encountered");
        }
    }
}

const char *
cuda_disasm_cache::populate_from_elf_image (uint64_t pc)
{
  /* only read from the ELF image if pc isn't in the map */
  if (m_elf_map.find (pc) == m_elf_map.end ())
    {
      /* collect all the necessary data */
      kernel_t kernel         = cuda_current_kernel ();
      module_t module         = kernel_get_module (kernel);
      elf_image_t elf_img     = module_get_elf_image (module);
      m_objfile               = cuda_elf_image_get_objfile (elf_img);
      const char *filename          = m_objfile->original_name;

      /* Generate the dissassembled code by using cuobjdump Can be
         per-function (faster, but may be invoked multiple times for a
         given file), or per-file (slower at first, but then
         faster). Defaults to trying per-function.
       */

      const char *func_name = NULL;
      if (cuda_options_disassemble_per_function ())
        func_name = cuda_find_function_name_from_pc (pc, false);

      char command[COMMAND_MAX_LENGTH];
      if (func_name && (func_name[0] != '\0'))
        snprintf (command, sizeof (command), "%s --function '%s' --dump-sass '%s'",
                  find_cuobjdump(), func_name, filename);
      else
        snprintf (command, sizeof (command), "%s --dump-sass '%s'",
                  find_cuobjdump(), filename);

      m_sass = popen (command, "r");
      if (!m_sass)
        throw_error (GENERIC_ERROR, "Cannot disassemble from the ELF image.");

      /* Parse the SASS output one line at a time */
      parse_disasm_output ();

      /* close the sass file */
      pclose (m_sass);

      /* we expect to always be able to diassemble at least one instruction */
      if (cuda_options_debug_strict () && m_elf_map.empty ())
        throw_error (GENERIC_ERROR, "Unable to disassemble a single device instruction.");

      /* return the instruction or NULL if not found */
      if (m_elf_map.find (pc) == m_elf_map.end ())
        return NULL;
    }

  /* it's in the cache, return the char * */
  return m_elf_map[pc].get ();
}

const char *
cuda_disasm_cache::populate_from_device_memory (uint64_t pc)
{
  uint32_t inst_size;
  char buf[INSN_MAX_LENGTH];

  uint32_t devId = cuda_current_device ();

  buf[0] = 0;
  cuda_api_disassemble (devId, pc, &inst_size, buf, sizeof (buf));

  if (buf[0] == '\0')
    return NULL;

  m_device_map[pc] = gdb::unique_xmalloc_ptr<char> (xstrdup (buf));
  return m_device_map[pc].get ();
}

const char *
cuda_disasm_cache::find_instruction (uint64_t pc, uint32_t *instruction_size)
{

  /* If no CUDA yet, or we don't have device focus, we can't
     disassemble so return NULL */
  if (!cuda_initialized || !cuda_focus_is_device ())
    return NULL;

  /* Get/set the instruction size if we don't already have it */
  uint32_t devId = cuda_current_device ();
  uint32_t inst_size = device_get_inst_size (devId);
  if (!inst_size)
    {
      inst_size = get_inst_size (pc);
      if (!inst_size)
        throw_error (GENERIC_ERROR, "Cannot find the instruction size while disassembling.");
      device_set_inst_size (devId, inst_size);
    }

  /* Always return the instruction size */
  *instruction_size = inst_size;

  /* Disassemble the instruction(s) */
  if (cuda_options_disassemble_from_elf_image ())
    {
      if (m_elf_map.find (pc) != m_elf_map.end ())
        return m_elf_map[pc].get ();

      /* No luck finding it in the ELF map, disassemble the whole ELF
         file and update the map */
      return populate_from_elf_image (pc);
    }

  /* Maybe flush the device map before disassembling */
  maybe_flush_device_map ();

  /* If the pc is cached, return it's disassembly */
  if (m_device_map.find (pc) != m_device_map.end ())
    return m_device_map[pc].get ();

  /* No luck finding it in the device map, disassemble and
     update the map directly */
  return populate_from_device_memory (pc);
}

/*
  External APIs
*/

disasm_cache_t
disasm_cache_create (void)
{
  return (disasm_cache_t) new cuda_disasm_cache;
}

void
disasm_cache_destroy (disasm_cache_t disasm_cache)
{
  delete (cuda_disasm_cache *)disasm_cache;
}

void
disasm_cache_flush (disasm_cache_t disasm_cache)
{
  cuda_disasm_cache *cache = (cuda_disasm_cache *)disasm_cache;

  cache->maybe_flush_device_map ();
}

const char *
disasm_cache_find_instruction (disasm_cache_t disasm_cache,
                               uint64_t pc, uint32_t *instruction_size)
{
  cuda_disasm_cache *cache = (cuda_disasm_cache *)disasm_cache;

  return cache->find_instruction (pc, instruction_size);
}
