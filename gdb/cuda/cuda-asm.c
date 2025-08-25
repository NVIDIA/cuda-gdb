/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2025 NVIDIA Corporation
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

#include "gdbsupport/common-utils.h"

#include "cuda-api.h"
#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-kernel.h"
#include "cuda-modules.h"
#include "cuda-options.h"
#include "cuda-sass-json.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-util-stream.h"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#if !defined(__QNX__)
#include <sys/prctl.h>
#endif
#include <sys/types.h>
#include <sys/wait.h>

#include <fstream>
#include <sstream>

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
  p_argv[args.size ()] = NULL;
  return argv;
}

/* Search for executable in PATH, cuda-gdb launch folder or current folder */
static bool
exists (const std::string &fname)
{
  struct stat buf;
  return stat (fname.c_str (), &buf) == 0;
}

/******************************************************************************
 *
 *                             Disassembly Cache
 *
 *****************************************************************************/

std::string
cuda_instruction::to_string () const
{
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

bool
cuda_instruction::is_control_flow (const bool skip_subroutines)
{
  is_control_flow_value &cache_is_control_flow
      = skip_subroutines ? m_is_control_flow_skipping_subroutines
			 : m_is_control_flow;

  if (cache_is_control_flow != is_control_flow_value::unset)
    return cache_is_control_flow == is_control_flow_value::true_value;

  bool control_flow = eval_is_control_flow (skip_subroutines);
  cache_is_control_flow = control_flow ? is_control_flow_value::true_value
				       : is_control_flow_value::false_value;
  return control_flow;
}

bool
cuda_instruction::eval_is_control_flow (const bool skip_subroutines)
{
  if (m_opcode.size () == 0)
    return true;

  const char *inst_str = m_opcode.c_str ();

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
  if (strstr (inst_str, "CAL") != 0 && !skip_subroutines)
    return true;
  // JCAL - covered with CAL
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
  if (strstr (inst_str, "BAR") != 0)
    return true;
  if (strstr (inst_str, "SYNC") != 0)
    return true;
  if (strstr (inst_str, "BREAK") != 0)
    return true;
  /* BSYNC - covered with SYNC */
  /* CALL - covered with CAL */
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
  /* UCGABAR_* - covered with BAR */
  return false;
}

template <typename M, typename K, typename V>
static void inline map_insert_or_assign (M &map, const K &key, V &&value)
{
  auto it = map.find (key);
  if (it != map.end ())
    {
      it->second = value;
    }
  else
    {
      map.emplace (std::piecewise_construct, std::forward_as_tuple (key),
		   std::forward_as_tuple (value));
    }
}

gdb::optional<cuda_instruction>
cuda_module_disassembly_cache::disassemble_instruction (uint64_t pc)
{
  /* Check the setting of disassemble_from */
  auto source = cuda_options_disassemble_from_elf_image ()
		    ? disassembly_source::ELF
		    : disassembly_source::DEVICE;
  /* Disassemble the instruction */
  auto insn = disassemble_instruction (pc, source);

  /* Try to fallback to debug API if we failed to disassemble from ELF */
  if (!insn && source == disassembly_source::ELF)
    insn = disassemble_instruction (pc, disassembly_source::DEVICE);

  return insn;
}

gdb::optional<cuda_instruction>
cuda_module_disassembly_cache::disassemble_instruction (
    const uint64_t pc, const disassembly_source source)
{
  gdb::optional<cuda_instruction> insn (cache_lookup (pc, source));

  if (insn)
    return insn;

  switch (source)
    {
    case disassembly_source::ELF:
      return populate_from_elf_image (pc);
    case disassembly_source::DEVICE:
      return populate_from_device_memory (pc);
    default:
      error ("Unknown disassembly source");
    }
}

gdb::optional<cuda_instruction>
cuda_module_disassembly_cache::cache_lookup (
    const uint64_t pc, const disassembly_source source) const
{
  switch (source)
    {
    case disassembly_source::ELF:
      {
	auto it = m_elf_map.find (pc);
	if (it != m_elf_map.end ())
	  return it->second;
      }
      break;
    case disassembly_source::DEVICE:
      {
	auto it = m_device_map.find (pc);
	if (it != m_device_map.end ())
	  return it->second;
      }
      break;
    default:
      error ("Unknown disassembly source");
    }
  return gdb::optional<cuda_instruction> ();
}

void
cuda_module_disassembly_cache::add_function_to_cache (
    const cuda_function &function, const disassembly_source source)
{
  uint64_t pc = function.start_address ();
  switch (source)
    {
    case disassembly_source::ELF:
      for (const cuda_instruction &insn : function.instructions ())
	{
	  map_insert_or_assign (m_elf_map, pc, cuda_instruction (insn));
	  pc += m_insn_size;
	}
      break;
    case disassembly_source::DEVICE:
      for (const cuda_instruction &insn : function.instructions ())
	{
	  map_insert_or_assign (m_device_map, pc, cuda_instruction (insn));
	  pc += m_insn_size;
	}
      break;
    default:
      error ("Unknown disassembly source");
    }
}

class cuobjdump_process
{
  pid_t m_pid;
  int m_stdout_fd;
  int m_stderr_fd;

  /* Set up a spawn process to replace the target fd with a supplied pipe.
     Returns the read-end of that pipe. */
  int
  set_up_fd_redirect (posix_spawn_file_actions_t *file_actions, int pipe[2],
		      int target_fd)
  {
    int rfd = pipe[0];
    int wfd = pipe[1];

    /* Close read-end in child process */
    int ret = posix_spawn_file_actions_addclose (file_actions, rfd);
    CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		 "cuobjdump_process: add close failed: %s",
		 safe_strerror (ret));

    /* Redirect target fd to pipe write-end in child process.
       Original target fd will be closed by dup2. */
    ret = posix_spawn_file_actions_adddup2 (file_actions, wfd, target_fd);
    CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		 "cuobjdump_process: add dup2 failed: %s",
		 safe_strerror (ret));

    /* Close write-end in child process, since it has been dup2'd
       into the target fd we can close the duplicate one. */
    ret = posix_spawn_file_actions_addclose (file_actions, wfd);
    CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		 "cuobjdump_process: add close failed: %s",
		 safe_strerror (ret));

    return rfd;
  }

public:
  cuobjdump_process () : m_pid (-1), m_stdout_fd (-1), m_stderr_fd (-1) {}

  DISABLE_COPY_AND_ASSIGN (cuobjdump_process);

  int
  stdout_fd () const
  {
    return m_stdout_fd;
  }

  int
  stderr_fd () const
  {
    return m_stderr_fd;
  }

  std::string
  get_stderr ()
  {
    const int STDERR_BUFSZ = 512;
    std::string str;
    ssize_t nr;

    gdb::unique_xmalloc_ptr<char> buff ((char *)xmalloc (STDERR_BUFSZ));
    str.reserve (STDERR_BUFSZ);

    do
      {
	nr = read (m_stderr_fd, buff.get (), STDERR_BUFSZ);
	if (nr < 0)
	  {
	    cuda_trace_domain (
		CUDA_TRACE_DISASSEMBLER,
		"cuobjdump_process: failed to read from stderr: %s",
		safe_strerror (errno));
	    break;
	  }
	if (nr > 0)
	  str.append (buff.get (), nr);
      }
    while (nr > 0);

    return str;
  }

  void
  exec (uint64_t pc, const std::string &filename, const char *function_name,
	const bool generate_json)
  {
    std::string cuobjdump_str ("cuobjdump");
    std::vector<std::string> cuobjdump_args
	= { cuobjdump_str, "--dump-sass", filename };

    if (generate_json)
      cuobjdump_args.push_back ("-json");

    if (function_name && function_name[0])
      {
	cuobjdump_args.push_back ("--function");
	cuobjdump_args.push_back (function_name);
      }

    if (cuda_options_trace_domain_enabled (CUDA_TRACE_DISASSEMBLER))
      {
	// Build up and log the cuobjdump invocation command line
	std::string args;
	for (auto arg : cuobjdump_args)
	  {
	    args.append (" ");
	    args.append (arg);
	  }
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "disassembler command (ELF): pc 0x%lx: %s", pc,
			   args.c_str ());
      }

    int stdout_fds[2];
    int stderr_fds[2];
    posix_spawn_file_actions_t file_actions;

    CUDA_ERR_IF (pipe (stdout_fds) == -1 || pipe (stderr_fds) == -1,
		 CUDA_TRACE_DISASSEMBLER, "cuobjdump_process: pipe failed: %s",
		 safe_strerror (errno));

    int ret = posix_spawn_file_actions_init (&file_actions);
    CUDA_ERR_IF (
	ret != 0, CUDA_TRACE_DISASSEMBLER,
	"Failed to spawn cuobjdump. posix_spawn_file_actions_init failed: %s",
	safe_strerror (ret));

    m_stdout_fd
	= set_up_fd_redirect (&file_actions, stdout_fds, STDOUT_FILENO);
    m_stderr_fd
	= set_up_fd_redirect (&file_actions, stderr_fds, STDERR_FILENO);

    auto argv = vector_to_argv (cuobjdump_args);
    ret = posix_spawnp (&m_pid, cuobjdump_str.c_str (), &file_actions, NULL,
			const_cast<char *const *> (argv.get ()), environ);
    if (ret != 0)
      {
	const std::string gdb_path = get_gdb_program_name ();
	const auto slash = gdb_path.rfind ("/");
	CUDA_ERR_IF (slash == std::string::npos, CUDA_TRACE_DISASSEMBLER,
		     "Failed to spawn cuobjdump. could not find gdb path to "
		     "retry and posix_spawnp failed: %s",
		     safe_strerror (ret));

	const std::string gdb_dir = gdb_path.substr (0, slash);
	const std::string cuobjdump_path = gdb_dir + "/cuobjdump";
	bool cuobjdump_exists = exists (cuobjdump_path);
	CUDA_ERR_IF (!cuobjdump_exists, CUDA_TRACE_DISASSEMBLER,
		     "Failed to spawn cuobjdump. cuobjdump does not exist in "
		     "gdb path and posix_spawnp failed: %s",
		     safe_strerror (ret));

	argv.get ()[0] = cuobjdump_path.c_str ();
	ret = posix_spawnp (&m_pid, cuobjdump_path.c_str (), &file_actions,
			    NULL, const_cast<char *const *> (argv.get ()),
			    environ);
      }

    posix_spawn_file_actions_destroy (&file_actions);

    /* Close write-end in current process, we don't need it.
       This will also allow the read-end from this process to return EOF if no
       more data are available. */
    close (stdout_fds[1]);
    close (stderr_fds[1]);

    CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		 "Failed to spawn cuobjdump. posix_spawnp failed: %s",
		 safe_strerror (ret));
  }

  int
  cleanup ()
  {
    close (m_stdout_fd);
    close (m_stderr_fd);

    int child_status;
    pid_t wait_ret = waitpid (m_pid, &child_status, 0);
    CUDA_ERR_IF (wait_ret != m_pid, CUDA_TRACE_DISASSEMBLER,
		 "Failed to wait for cuobjdump: %s", safe_strerror (errno));

    CUDA_ERR_IF (!WIFEXITED (child_status), CUDA_TRACE_DISASSEMBLER,
		 "cuobjdump did not exit normally, status: %d", child_status);

    return WEXITSTATUS (child_status);
  }
};

gdb::optional<cuda_instruction>
cuda_module_disassembly_cache::populate_from_elf_image (const uint64_t pc)
{
  static constexpr bool generate_json_output = true;

  // If we couldn't find the cubin, we can't disassemble from the elf
  // image.
  const auto module = cuda_state::find_module_by_address (pc);
  if (!module)
    {
      warning ("Could not find cubin to disassemble for pc 0x%lx", pc);
      return gdb::optional<cuda_instruction> ();
    }
  const auto filename = module->filename ();
  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
		     "populate (ELF): found pc 0x%lx in %s", pc,
		     filename.c_str ());

  /* Generate the dissassembled code by using cuobjdump Can be
     per-function (faster, but may be invoked multiple times for a
     given file), or per-file (slower at first, but then
     faster). Defaults to trying per-function.
   */
  gdb::unique_xmalloc_ptr<char> function_name;
  if (cuda_options_disassemble_per_function ())
    function_name = cuda_find_function_name_from_pc (pc, false);

  cuobjdump_process proc;

  /* The legacy human readable output parser is used as a fallback if the JSON
     parser fails to parse the cuobjdump output */
  if (m_cuobjdump_json)
    {
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Trying to parse cuobjdump json output");

      proc.exec (pc, filename, function_name.get (), generate_json_output);

      /* parse the json output */

      const bool parsed = parse_disasm_output_json (proc.stdout_fd ());

      /* Dump stderr into trace logs */
      std::string proc_err = proc.get_stderr ();
      if (!proc_err.empty ())
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "cuobjdump_process error: %s", proc_err.c_str ());

      const bool success = (proc.cleanup () == 0);
      if (success)
	{
	  if (parsed)
	    return cache_lookup (pc, disassembly_source::ELF);

	  // This should not happen
	  warning (
	      "Failed to parse cuobjdump json output but cuobjdump succeeded");
	}

      /* cache the result and retry without json */
      m_cuobjdump_json = false;
    }

  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
		     "Failed to parse cuobjdump json output, falling back to "
		     "plaintext disassembly");

  proc.exec (pc, filename, function_name.get (), !generate_json_output);

  parse_disasm_output (proc.stdout_fd (), module);
  const bool success = (proc.cleanup () == 0);

  /* Dump stderr into trace logs */
  std::string proc_err = proc.get_stderr ();
  if (!proc_err.empty ())
    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "cuobjdump_process error: %s",
		       proc_err.c_str ());

  CUDA_ERR_IF (!success, CUDA_TRACE_DISASSEMBLER,
	       "Failed to cleanup cuobjdump");

  return cache_lookup (pc, disassembly_source::ELF);
}

gdb::optional<cuda_instruction>
cuda_module_disassembly_cache::populate_from_device_memory (const uint64_t pc)
{
  uint32_t inst_size = 0;
  char buf[MAX_BUFFER_SIZE];

  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "populate (debugAPI): pc 0x%lx",
		     pc);

  buf[0] = '\0';
  try
    {
      cuda_debugapi::disassemble (
	  cuda_current_focus::get ().physical ().dev (), pc, &inst_size, buf,
	  sizeof (buf));
    }
  catch (const gdb_exception_error &exception)
    {
      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			 "Exception disassembling device %u pc %lx (%s)",
			 cuda_current_focus::get ().physical ().dev (), pc,
			 exception.what ());
      return gdb::optional<cuda_instruction> ();
    }

  if (buf[0] == '\0')
    return gdb::optional<cuda_instruction> ();

  cuda_instruction instruction (buf);
  map_insert_or_assign (m_device_map, pc, cuda_instruction (instruction));
  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
		     "disasm (debugAPI): pc 0x%lx: %s", pc, buf);
  return instruction;
}

bool
cuda_module_disassembly_cache::parse_disasm_output_json (const int fd)
{
  using namespace cuda_disasm_json;

  parser parser (std::make_unique<util_stream::file_stream> (fd));

  parser.set_function_consumer ([this] (const schema_function &function) {
    std::vector<cuda_instruction> instructions;
    for (const auto &insn : function.m_instructions)
      {
	if (insn.m_opt_is_control_flow.has_value ())
	  instructions.push_back (cuda_instruction (
	      insn.m_predicate, insn.m_opcode, insn.m_operands, insn.m_extra,
	      *insn.m_opt_is_control_flow));
	else
	  instructions.push_back (cuda_instruction (
	      insn.m_predicate, insn.m_opcode, insn.m_operands, insn.m_extra));
      }

    add_function_to_cache (cuda_function (function.m_function_name,
					  function.m_start, function.m_length,
					  std::move (instructions)),
			   disassembly_source::ELF);
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
			     "Failed to parse cuobjdump json output: %s",
			     e.what ());
	  return false;
	}
      throw;
    }

  return true;
}

enum class disassembler_line_type
{
  unknown,
  function_header,
  section_header,
  offset_instruction,
  code_only,
  eof
};

static disassembler_line_type
parse_next_line (uint64_t &current_offset, std::string &current_insn,
		 std::string &current_func, std::string &current_line,
		 std::string &current_section, FILE *sass)
{
  static std::regex func_header_line (
      "[ \t]*Function[ \t]*:[ \t]*([0-9A-Za-z_\\$]*)[ \t]*");
  static std::regex offset_insn_line (
      "[ \t]*/\\*([0-9a-f]+)\\*/[ \t]+(.*)[ \t]*;.*");
  static std::regex code_line ("[ \t]*/\\*[ \t]*0x([0-9a-f]+)[ \t]*\\*/.*");
  static std::regex section_header_line (
      "[ \t]*([0-9A-Za-z_\\$\\.]*)[ \t]*:[ \t]*");
  /* Initialize the parser state before proceeding */
  current_offset = (uint64_t)-1LL;

  /* clear parser state strings */
  current_func.clear ();
  current_line.clear ();
  current_section.clear ();

  /* Read the next line */
  char line_buffer[MAX_BUFFER_SIZE];
  if (!fgets (line_buffer, sizeof (line_buffer), sass))
    return disassembler_line_type::eof;

  current_line = std::string (line_buffer);

  /* Look for a Function header */
  std::cmatch func_header;
  if (regex_search (line_buffer, func_header, func_header_line))
    {
      current_func = func_header.str (1);
      return disassembler_line_type::function_header;
    }

  /* Look for leading offset followed by an insn */
  std::cmatch offset_insn;
  if (regex_search (line_buffer, offset_insn, offset_insn_line))
    {

      /* extract the offset */
      const std::string &offset_str = offset_insn.str (1);
      current_offset = strtoull (offset_str.c_str (), NULL, 16);

      /* If necessary, trim mnemonic length */
      current_insn = offset_insn.str (2);
      return disassembler_line_type::offset_instruction;
    }

  /* Look for a code-only line, nothing to extract */
  if (regex_search (line_buffer, code_line))
    return disassembler_line_type::code_only;

  /* Look for a Section header - very permissive pattern, check last */
  std::cmatch section_header;
  if (regex_search (line_buffer, section_header, section_header_line))
    {
      current_section = section_header.str (1);
      return disassembler_line_type::section_header;
    }

  /* unknown line */
  return disassembler_line_type::unknown;
}

void
cuda_module_disassembly_cache::parse_disasm_output (const int fd,
						    cuda_module *module)
{
  FILE *sass = fdopen (fd, "r");
  CUDA_ERR_IF (!sass, CUDA_TRACE_DISASSEMBLER,
	       "Failed to open cuobjdump output fd: %s",
	       safe_strerror (errno));
  /* instruction encoding-only lines are 8 bytes each */
  const uint32_t disasm_line_size = 8;

  /* parse the sass output and insert each instruction found */
  uint64_t last_pc = 0;
  uint64_t entry_pc = 0;

  uint64_t current_offset;
  std::string current_insn;
  std::string current_func;
  std::string current_line;
  std::string current_section;
  while (true)
    {
      uint64_t pc = 0;

      /* parse the line and determine it's type */
      auto line_type
	  = parse_next_line (current_offset, current_insn, current_func,
			     current_line, current_section, sass);
      switch (line_type)
	{
	case disassembler_line_type::unknown:
	  /* skip */
	  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "unknown-line: %s",
			     current_line.c_str ());
	  continue;

	case disassembler_line_type::function_header:
	  if (!current_func.empty ())
	    {
	      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
				 "function header: %s", current_func.c_str ());
	      /* Lookup the symbol to get the entry_pc value from the bound
	       * minimal symbol */
	      struct bound_minimal_symbol sym = lookup_minimal_symbol (
		  current_func.c_str (), NULL, module->objfile ());
	      if (sym.minsym == NULL)
		{
		  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
				     _ ("\"%s\" found in disassembly but has "
					"no minimal symbol"),
				     current_func.c_str ());
		  complaint (_ ("\"%s\" found in disassembly but has no "
				"minimal symbol"),
			     current_func.c_str ());
		}
	      else
		{
		  entry_pc = sym.value_address ();
		  if (!entry_pc)
		    complaint (
			_ ("\"%s\" exists in this program but entry_pc == 0"),
			current_func.c_str ());
		  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
				     "found \"%s\" at pc 0x%lx",
				     current_func.c_str (), entry_pc);
		}
	    }
	  break;

	case disassembler_line_type::section_header:
	  if (!current_section.empty ())
	    {
	      // Check for known section names
	      std::string sym_name;
	      if (!current_section.compare (".nv.uft"))
		sym_name = "__UFT";
	      else
		{
		  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
				     "section header '%s' unknown",
				     current_section.c_str ());
		  break;
		}
	      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
				 "section header: '%s' sym_name '%s'",
				 current_section.c_str (), sym_name.c_str ());
	      /* Lookup the symbol to get the entry_pc value from the bound
	       * minimal symbol */
	      struct bound_minimal_symbol sym = lookup_minimal_symbol (
		  sym_name.c_str (), NULL, module->objfile ());
	      if (sym.minsym == NULL)
		{
		  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
				     "'%s' found in disassembly but has no "
				     "minimal symbol for '%s'",
				     current_section.c_str (),
				     sym_name.c_str ());
		  complaint ("'%s' found in disassembly but has no minimal "
			     "symbol for '%s'",
			     current_section.c_str (), sym_name.c_str ());
		}
	      else
		{
		  entry_pc = sym.value_address ();
		  if (!entry_pc)
		    complaint ("'%s' exists in this program but entry_pc == 0",
			       sym_name.c_str ());
		  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
				     "found '%s' at pc 0x%lx",
				     sym_name.c_str (), entry_pc);
		}
	    }
	  break;

	case disassembler_line_type::offset_instruction:
	  cuda_trace_domain (
	      CUDA_TRACE_DISASSEMBLER,
	      "offset-insn: entry_pc 0x%lx offset 0x%lx pc 0x%lx insn: %s",
	      entry_pc, current_offset, entry_pc + current_offset,
	      current_insn.c_str ());
	  if ((current_insn.size () > 0) && (current_offset != (uint64_t)-1LL)
	      && entry_pc)
	    {
	      pc = entry_pc + current_offset;

	      /* insert the disassembled instruction into the map */
	      map_insert_or_assign (m_elf_map, pc,
				    cuda_instruction (current_insn));
	      last_pc = pc;
	      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
				 "offset-insn: cache pc 0x%lx insn: %s", pc,
				 current_insn.c_str ());
	    }
	  else
	    cuda_trace_domain (
		CUDA_TRACE_DISASSEMBLER,
		"offset-insn: could not cache pc 0x%lx insn: %s", entry_pc,
		current_insn.c_str ());
	  break;

	case disassembler_line_type::code_only:
	  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			     "code-only: last_pc 0x%lx line_size %d", last_pc,
			     disasm_line_size);
	  if (last_pc)
	    {
	      /* skip non-offset lines, but still count them */
	      last_pc += disasm_line_size;
	      continue;
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
	    map_insert_or_assign (m_elf_map, pc, cuda_instruction (""));
	  last_pc = pc;
	  break;

	case disassembler_line_type::eof:
	  /* We're done */
	  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "EOF");
	  return;

	default:
	  /* should never happen regardless of input */
	  error ("unknown line-type encountered");
	}
    }
}
