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
vector_to_argv (const std::vector<std::string> &args) {
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
static bool exists (const std::string &fname)
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
cuda_instruction::is_barrier ()
{
  if (!m_is_barrier.has_value ())
    m_is_barrier = eval_is_barrier ();
  return *m_is_barrier;
}

bool
cuda_instruction::is_control_flow ()
{
  if (!m_is_control_flow.has_value ())
    m_is_control_flow = eval_is_control_flow ();
  return *m_is_control_flow;
}

bool
cuda_instruction::is_subroutine_call ()
{
  if (!m_is_subroutine_call.has_value ())
    m_is_subroutine_call = eval_is_subroutine_call ();
  return *m_is_subroutine_call;
}

bool
cuda_instruction::eval_is_control_flow () const
{
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
cuda_instruction::eval_is_subroutine_call () const
{
  if (strstr (m_opcode.c_str (), "CALL") != 0)
    return true;
  return false;
}

bool
cuda_instruction::eval_is_barrier () const
{
  /* MEMBAR, DEPBAR, UCGABAR_* - covered with BAR */
  if (strstr (m_opcode.c_str (), "BAR") != 0)
    return true;
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
		 "cuobjdump_process: add close failed: %s",
		 safe_strerror (ret));

    /* Duplicate the pipe's write-end onto the requested FD. */
    ret = posix_spawn_file_actions_adddup2 (&m_file_actions, child_write_fd,
					    target_fd);
    CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		 "cuobjdump_process: add dup2 failed: %s",
		 safe_strerror (ret));

    /* Close the original write-end in the child after dup2. */
    ret = posix_spawn_file_actions_addclose (&m_file_actions, child_write_fd);
    CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		 "cuobjdump_process: addclose failed for write end: %s",
		 safe_strerror (ret));
  }
};

class cuobjdump_process
{
  pid_t m_pid;
  int m_stdout_rfd; // Read end for parent (stdout)
  int m_stderr_rfd; // Read end for parent (stderr)
  int m_stdout_wfd; // Write end for child (stdout)
  int m_stderr_wfd; // Write end for child (stderr)
  posix_spawn_file_actions
      m_file_actions; // Prepared file actions holding redirections

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
  cuobjdump_process ()
      : m_pid (-1), m_stdout_rfd (-1), m_stderr_rfd (-1), m_stdout_wfd (-1),
	m_stderr_wfd (-1), m_file_actions ()
  {
    int stdout_fds[2] = { -1, -1 };
    int stderr_fds[2] = { -1, -1 };

    CUDA_ERR_IF (pipe (stdout_fds) == -1 || pipe (stderr_fds) == -1,
		 CUDA_TRACE_DISASSEMBLER, "cuobjdump_process: pipe failed: %s",
		 safe_strerror (errno));

    // Store read/write ends.
    m_stdout_rfd = stdout_fds[0];
    m_stdout_wfd = stdout_fds[1];
    m_stderr_rfd = stderr_fds[0];
    m_stderr_wfd = stderr_fds[1];

    /* Set up redirection for stdout/stderr for the spawned
    process. Configures the file actions to dup the write end of the pipe to
    the target FD in the child and close unused FDs. */
    m_file_actions.add_redirect (m_stdout_rfd, m_stdout_wfd, STDOUT_FILENO);
    m_file_actions.add_redirect (m_stderr_rfd, m_stderr_wfd, STDERR_FILENO);
  }

  // Destructor: Ensures cleanup if wait() wasn't called explicitly
  ~cuobjdump_process ()
  {
    try
      {
	if (m_pid > 0)
	  wait ();
      }
    catch (const gdb_exception &e)
      {
	cuda_trace_domain (
	    CUDA_TRACE_DISASSEMBLER,
	    ("Ignoring exception in cuobjdump_process dtor: %s"), e.what ());
      }

    safe_close (m_stdout_rfd);
    safe_close (m_stderr_rfd);
    safe_close (m_stdout_wfd);
    safe_close (m_stderr_wfd);
  }

  DISABLE_COPY_AND_ASSIGN (cuobjdump_process);

  /* Returns the file descriptor for reading the child's stdout. */
  int
  stdout_read_fd () const
  {
    return m_stdout_rfd;
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
		    "cuobjdump_process: failed to read from stderr: %s",
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

  /* Executes cuobjdump with the given parameters. Sets up pipes for
     stdout/stderr, spawns the process, and closes unnecessary FDs in the
     parent. Throws gdb_exception_error on failure. */
  void
  exec (uint64_t pc, const std::string &filename, const char *function_name,
	const bool generate_json)
  {
    // Check if already active by inspecting pid
    if (m_pid > 0)
      error ("cuobjdump_process::exec called on already active object.");

    // Reset pid state only; file descriptors are prepared at construction.
    m_pid = -1;

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
	std::string args_str;
	for (const auto &arg : cuobjdump_args)
	  {
	    args_str.append (" ");
	    args_str.append (arg);
	  }
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "disassembler command (ELF): pc 0x%lx:%s", pc,
			   args_str.c_str ());
      }

    try
      {
	auto argv = vector_to_argv (cuobjdump_args);

	// First spawn attempt using PATH
	int ret = posix_spawnp (
	    &m_pid, cuobjdump_str.c_str (), m_file_actions.get (), NULL,
	    const_cast<char *const *> (argv.get ()), environ);

	if (ret != 0)
	  {
	    // Fallback spawn attempt if first failed (e.g., not in PATH)
	    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			       "posix_spawnp failed for '%s' (errno %d: %s), "
			       "trying path relative to gdb",
			       cuobjdump_str.c_str (), ret,
			       safe_strerror (ret));

	    const std::string gdb_path = get_gdb_program_name ();
	    const auto slash_idx = gdb_path.rfind ("/");
	    CUDA_ERR_IF (
		slash_idx == std::string::npos, CUDA_TRACE_DISASSEMBLER,
		"Failed to find gdb directory for cuobjdump fallback: %s",
		gdb_path.c_str ());

	    const std::string gdb_bin_dir = gdb_path.substr (0, slash_idx);
	    const std::string cuobjdump_path = gdb_bin_dir + "/cuobjdump";

	    // Check existence before attempting spawn with full path
	    CUDA_ERR_IF (!exists (cuobjdump_path), CUDA_TRACE_DISASSEMBLER,
			 "cuobjdump not found at '%s' for fallback spawn",
			 cuobjdump_path.c_str ());

	    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			       "Attempting fallback spawn with path: %s",
			       cuobjdump_path.c_str ());

	    argv.get ()[0] = cuobjdump_path.c_str ();
	    // Use posix_spawn, not spawnp, as we have the full path
	    ret = posix_spawn (
		&m_pid, cuobjdump_path.c_str (), m_file_actions.get (), NULL,
		const_cast<char *const *> (argv.get ()), environ);
	  }

	// Final check after potentially two spawn attempts. If ret != 0, spawn
	// failed.
	CUDA_ERR_IF (ret != 0, CUDA_TRACE_DISASSEMBLER,
		     "Failed to spawn cuobjdump (posix_spawn/p errno %d): %s",
		     ret, safe_strerror (ret));

	// If spawn succeeded:
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "cuobjdump spawned with pid %d", (int)m_pid);

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
			   "Exception during cuobjdump exec setup: %s",
			   e.what ());
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

  /* Waits for the cuobjdump process to terminate, cleans up resources (FDs),
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
      error ("cuobjdump_process::wait called on inactive object.");

    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER, "Waiting for cuobjdump pid %d",
		       (int)m_pid);

    // Close parent's read ends now. This might signal the process if needed,
    // but primarily cleans up our side. It's safe to do before waitpid.
    safe_close (m_stdout_rfd);
    safe_close (m_stderr_rfd);

    int child_status = 0;
    int exit_status = -1; // Default to error/unknown status

    // Wait for the child process
    const pid_t wait_ret = waitpid (m_pid, &child_status, 0);

    if (wait_ret == -1)
      // Error waiting
      warning ("Failed to wait for cuobjdump pid %d: %s", (int)m_pid,
	       safe_strerror (errno));
    else if (wait_ret == m_pid)
      {
	// Process terminated
	if (WIFEXITED (child_status))
	  {
	    exit_status = WEXITSTATUS (child_status);
	    cuda_trace_domain (
		CUDA_TRACE_DISASSEMBLER,
		"cuobjdump pid %d exited normally with status %d", (int)m_pid,
		exit_status);
	  }
	else
	  // Terminated abnormally (signal, etc.)
	  warning ("cuobjdump pid %d terminated abnormally (status %d)",
		   (int)m_pid, child_status);
	// Keep exit_status = -1 for abnormal termination
      }
    else
      // Unexpected return from waitpid
      warning ("waitpid returned unexpected pid %d (expected %d)",
	       (int)wait_ret, (int)m_pid);

    // Mark as inactive and cleanup pid
    m_pid = -1;

    return exit_status;
  }
};

gdb::optional<cuda_instruction>
cuda_module_disassembly_cache::populate_from_elf_image (const uint64_t pc)
{
  // If we couldn't find the cubin, we can't disassemble from the elf
  // image.
  const auto module = cuda_state::find_module_by_address (pc);
  if (!module)
    {
      warning ("Could not find cubin to disassemble for pc 0x%lx", pc);
      return gdb::optional<cuda_instruction> ();
    }
  const auto &filename = module->filename (); // Use const ref
  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
		     "populate (ELF): found pc 0x%lx in %s", pc,
		     filename.c_str ());

  /* Determine if disassembly should be per-function */
  gdb::unique_xmalloc_ptr<char> function_name;
  if (cuda_options_disassemble_per_function ())
    {
      // Demangling not needed here, use cuda_find_function_name_from_pc
      function_name = cuda_find_function_name_from_pc (pc, false);
      if (function_name)
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "Attempting disassembly for function: %s",
			   function_name.get ());
      else
	cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			   "Could not find function name for pc 0x%lx, "
			   "disassembling whole module.",
			   pc);
    }

  bool disassembled_successfully = false;

  /* Try parsing JSON output first if enabled */
  if (m_cuobjdump_json)
    {
      cuobjdump_process proc;
      try
	{
	  cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			     "Trying cuobjdump with JSON output");

	  proc.exec (pc, filename, function_name.get (),
		     true); // generate_json = true

	  const bool parsed
	      = parse_disasm_output_json (proc.stdout_read_fd ());

	  // Read stderr *before* waiting
	  std::string proc_err = proc.get_stderr ();
	  if (!proc_err.empty ())
	    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			       "cuobjdump[JSON] stderr: %s",
			       proc_err.c_str ());

	  // Wait for process and check status
	  const int status = proc.wait ();
	  if (status == 0 && parsed)
	    {
	      cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
				 "Successfully parsed cuobjdump JSON output.");
	      disassembled_successfully = true;
	    }
	  else
	    {
	      warning ("Failed to parse cuobjdump JSON output or cuobjdump "
		       "failed (status: %d, parsed: %d)",
		       status, parsed);
	      // Disable JSON for future attempts if it failed
	      m_cuobjdump_json = false;
	    }
	}
      catch (const gdb_exception &e) // Catch exceptions during exec/parse/wait
	{
	  warning ("Exception during JSON disassembly: %s", e.what ());
	  m_cuobjdump_json
	      = false; // Disable JSON on exception
		       // proc destructor handles cleanup automatically
	}
      // proc goes out of scope here, destructor runs if needed
    }

  /* Fallback to plaintext if JSON is disabled or failed */
  if (!disassembled_successfully)
    {
      cuobjdump_process proc;
      try
	{
	  cuda_trace_domain (
	      CUDA_TRACE_DISASSEMBLER,
	      "Trying cuobjdump with plaintext output (JSON disabled)");

	  proc.exec (pc, filename, function_name.get (),
		     false); // generate_json = false

	  // Pass module needed by plaintext parser
	  parse_disasm_output (proc.stdout_read_fd (), module);

	  // Read stderr *before* waiting
	  std::string proc_err = proc.get_stderr ();
	  if (!proc_err.empty ())
	    cuda_trace_domain (CUDA_TRACE_DISASSEMBLER,
			       "cuobjdump[plaintext] stderr: %s",
			       proc_err.c_str ());

	  // Wait for process and check status
	  const int status = proc.wait ();
	  if (status == 0)
	    {
	      cuda_trace_domain (
		  CUDA_TRACE_DISASSEMBLER,
		  "Successfully parsed cuobjdump plaintext output.");
	      disassembled_successfully = true;
	    }
	  else
	    {
	      // Use CUDA_ERR_IF to match original fatal error behavior on
	      // fallback failure
	      CUDA_ERR_IF (
		  true, CUDA_TRACE_DISASSEMBLER,
		  "cuobjdump plaintext disassembly failed with status %d",
		  status);
	    }
	}
      catch (const gdb_exception &e) // Catch exceptions during exec/parse/wait
	{
	  // Match original fatal error behavior
	  error ("Exception during plaintext disassembly: %s", e.what ());
	  // proc destructor handles cleanup automatically
	}
      // proc goes out of scope here, destructor runs if needed
    }

  // return nullopt if disassembly failed: the pc is not in the cache
  return gdb::optional<cuda_instruction> ();
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
	  if (insn.m_opt_is_subroutine_call.has_value ())
	    {
	      instructions.emplace_back (insn.m_predicate, insn.m_opcode,
					 insn.m_operands, insn.m_extra,
					 *insn.m_opt_is_control_flow,
					 *insn.m_opt_is_subroutine_call);
	    }
	  else
	    {
	      instructions.emplace_back (insn.m_predicate, insn.m_opcode,
					 insn.m_operands, insn.m_extra,
					 *insn.m_opt_is_control_flow);
	    }
	else
	  instructions.emplace_back (insn.m_predicate, insn.m_opcode,
				     insn.m_operands, insn.m_extra);
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
			     "Failed to parse cuobjdump JSON output: %s",
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
