/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2015-2025 NVIDIA Corporation
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

#include "completer.h"
#include "gdbthread.h"
#include "inferior.h"
#include "process-stratum-target.h"
#include "readline/readline.h"
#include "regcache.h"
#include "target.h"

#include "cuda-api.h"
#include "cuda-context.h"
#include "cuda-coord-set.h"
#include "cuda-corelow.h"
#include "cuda-events.h"
#include "cuda-exceptions.h"
#include "cuda-linux-nat.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"

#include "../libcudacore/libcudacore.h"

/* The CUDA core file target */

static const target_info cuda_core_target_info
    = { "cudacore", N_ ("Local CUDA core dump file"),
	N_ ("Use CUDA core file as a target.\n\
Specify the filename to the core file.") };

class cuda_core_target final : public process_stratum_target
{
public:
  /* public data members */
  static CudaCore *m_cuda_core;
  /* public methods */
  cuda_core_target () = delete;
  cuda_core_target (const char *);
  ~cuda_core_target () override = default;

  /* Return a reference to this target's unique target_info
     object.  */
  const target_info &
  info () const override
  {
    return cuda_core_target_info;
  }

  void close () override;
  void detach (inferior *inf, int from_tty) override;
  void fetch_registers (struct regcache *, int) override;

  bool
  thread_alive (ptid_t ptid) override
  {
    return true;
  }

  std::string pid_to_str (ptid_t) override;

  bool
  has_memory () override
  {
    return true;
  }
  bool
  has_stack () override
  {
    return true;
  }
  bool
  has_registers () override
  {
    return true;
  }
  bool
  has_execution (inferior *inf) override
  {
    return false;
  }

private:
  void core_cleanup ();
};

CudaCore *cuda_core_target::m_cuda_core = nullptr;

cuda_core_target::cuda_core_target (const char *filename)
    : process_stratum_target ()
{
  cuda_core_load_api (filename);
}

std::string
cuda_core_target::pid_to_str (ptid_t ptid)
{
  struct inferior *inf;
  int pid;

  /* Try the LWPID field first.  */
  pid = ptid.lwp ();
  if (pid != 0)
    return normal_pid_to_str (ptid_t (pid));

  /* Otherwise, this isn't a "threaded" core -- use the PID field, but
   * only if it isn't a fake PID.  */
  inf = find_inferior_ptid (this, ptid);
  if (inf != NULL && !inf->fake_pid_p)
    return normal_pid_to_str (ptid);

  /* No luck.  We simply don't have a valid PID to print.  */
  return "<main task>";
}

void
cuda_core_target::fetch_registers (struct regcache *regcache, int regno)
{
  cuda_core_fetch_registers (regcache, regno);
}

void
cuda_core_fetch_registers (struct regcache *regcache, int regno)
{
  if (!cuda_is_cuda_gdbarch (regcache->arch ()) && regno >= 0)
    {
      /* Wrong architecture, this is likely the fake host thread
	 being "restored" and we're "reading" its stop PC.
	 cuda_core_target doesn't support cross-arch reg fetch,
	 unlike its counterpart cuda_nat_linux which can pass it
	 to its parent(native target)'s method.
	 Just invalidate the regno cache as REG_UNAVAILABLE.
	 Note: This won't work if all regs are requested (-1) */
      regcache->raw_supply (regno, nullptr);
      return;
    }

  if (!cuda_current_focus::isDevice ())
    return;

  // Lazily created by cuda_get_gdbarch(), should always be non-null
  struct gdbarch *gdbarch = cuda_get_gdbarch ();
  gdb_assert (gdbarch);

  // Read in all available registers
  for (auto regnum = 0; regnum < gdbarch_num_regs (gdbarch); regnum++)
    cuda_register_read (gdbarch, regcache, regnum);
}

#define CUDA_CORE_PID 966617

static void
cuda_core_register_tid (uint32_t tid)
{
  if (inferior_ptid != null_ptid)
    return;

  ptid_t ptid (CUDA_CORE_PID, tid, tid);
  struct thread_info *tp
      = add_thread (current_inferior ()->process_target (), ptid);
  switch_to_thread_no_regs (tp);
}

/*
 * This is called by both the cuda_core_target and the core_target.
 * For the latter, we don't want to install the entire cuda_core_target.
 */
void
cuda_core_load_api (const char *filename)
{
  CUDBGAPI api;

  printf_unfiltered (_ ("Opening GPU coredump: %s\n"), filename);

  gdb_assert (cuda_core_target::m_cuda_core == nullptr);

  cuda_core_target::m_cuda_core = cuCoreOpenByName (filename);
  if (cuda_core_target::m_cuda_core == nullptr)
    error ("Failed to read core file: %s", cuCoreErrorMsg ());
  api = cuCoreGetApi (cuda_core_target::m_cuda_core);
  if (api == NULL)
    error ("Failed to get debugger APIs: %s", cuCoreErrorMsg ());

  cuda_debugapi::set_api (api);
  cuda_debugapi::set_api_version (CUDBG_API_VERSION_MAJOR,
				  CUDBG_API_VERSION_MINOR,
				  CUDBG_API_VERSION_REVISION);

  /* Initialize the APIs */
  cuda_initialize ();
  if (!cuda_initialized)
    error ("Failed to initialize CUDA Core debugger API!");
}

void
cuda_core_free (void)
{
  if (cuda_core_target::m_cuda_core == nullptr)
    return;

  cuda_cleanup ();
  cuda_gdb_session_destroy ();
  cuCoreFree (cuda_core_target::m_cuda_core);
  cuda_core_target::m_cuda_core = nullptr;
}

void
cuda_core_initialize_events_exceptions (void)
{
  /* Flush registers cache */
  registers_changed ();

  /* Create session directory */
  if (cuda_gdb_session_create ())
    error ("Failed to create session directory");

  auto handle_ctx_create_event = [] (const CUDBGEvent &event) {
    // This loop will take a very long time on corefiles
    // with 1,000+ cubins. Give the user the opportunity to CTRL-C
    QUIT;

    switch (event.kind)
      {
      case CUDBG_EVENT_CTX_CREATE:
	cuda_core_register_tid (event.cases.contextCreate.tid);
	break;
      default:
	/* Do nothing */
	break;
      }
    return true;
  };

  /* Drain the event queue */
  cuda_process_events (CUDA_EVENT_SYNC, handle_ctx_create_event);

  // Read in all device state
  cuda_state::update_all_state (CUDBG_RESPONSE_TYPE_FULL);

  /* Figure out, where exception happened */
  cuda_exception ex;
  if (ex.has_exception ())
    {
      /* Exception detected, set focus to the exception */
      if (ex.has_coords ())
	{
	  cuda_coords c{ ex.coords () };
	  switch_to_cuda_thread (c);
	  cuda_current_focus::printFocus (false);
	}
      /* Print the exception */
      ex.print_message ();
    }
  else
    {
      /* No exception detected, check for fatal signals (SIGTRAP) */
      cuda_coord_set<cuda_coord_set_type::threads, select_valid | select_trap
						       | select_current_clock
						       | select_sngl>
	  coord{ cuda_coords::wild () };
      if (coord.size ())
	{
	  /* This is the first lane in the warp at a trap */
	  auto it = coord.begin ();
	  switch_to_cuda_thread (*it);
	  cuda_current_focus::printFocus (false);

	  cuda_set_signo (GDB_SIGNAL_TRAP);
	  gdb_printf (_ ("Program terminated with signal %s, %s.\n"),
		      gdb_signal_to_name (GDB_SIGNAL_TRAP),
		      gdb_signal_to_string (GDB_SIGNAL_TRAP));
	}
    }

  /* Fetch latest information about coredump grids */
  cuda_state::update_kernel_args ();
}

static void
cuda_core_target_open (const char *filename, int from_tty)
{
  struct inferior *inf = nullptr;
  gdbarch *old_gdbarch = nullptr;

  target_preopen (from_tty);

  if (filename == NULL)
    error (_ ("No core file specified."));

  gdb::unique_xmalloc_ptr<char> expanded_filename (tilde_expand (filename));

  cuda_core_target *target = new cuda_core_target (expanded_filename.get ());

  /* Own the target until it is sucessfully pushed. */
  target_ops_up target_holder (target);

  try
    {
      /* Push the target */
      current_inferior ()->push_target (std::move (target_holder));

      switch_to_no_thread ();

      /* flush register cache from a previous debug session. */
      registers_changed ();

      /* A CUDA corefile does not contain host process pid information.
       * We need to fake it here since we are only examining CUDA state.
       * Add the fake PID for the host thread. */
      inf = current_inferior ();
      inferior_appeared (inf, CUDA_CORE_PID);
      inf->fake_pid_p = true;
      thread_info *thread = add_thread_silent (target, ptid_t (CUDA_CORE_PID));
      switch_to_thread_no_regs (thread);

      /* Set debuggers architecture to CUDA */
      old_gdbarch = target_gdbarch ();
      set_target_gdbarch (cuda_get_gdbarch ());

      cuda_core_initialize_events_exceptions ();

      post_create_inferior (from_tty);

      /* If no exception found try to set focus to first valid thread */
      if (!cuda_current_focus::isDevice ())
	{
	  cuda_coord_set<cuda_coord_set_type::threads,
			 select_valid | select_sngl>
	      coord{ cuda_coords::wild () };
	  if (coord.size ())
	    {
	      /* Switch to the CUDA thread and print the focus update. */
	      switch_to_cuda_thread (*coord.begin ());
	      cuda_current_focus::printFocus (false);
	    }
	  else
	    {
	      /* Unable to set device focus. Give up, but allow the user to
		 debug global memory contents. */
	      cuda_current_focus::invalidate ();
	      warning ("No CUDA focus could be set");
	      if (old_gdbarch != nullptr)
		set_target_gdbarch (old_gdbarch);
	    }
	}

      /* Fetch all registers from core file. */
      target_fetch_registers (get_current_regcache (), -1);

      // Set up the frame cache
      reinit_frame_cache ();

      // Print the backtrace if we found something on the GPU to focus on
      if (cuda_current_focus::isDevice ())
	print_stack_frame (get_selected_frame (NULL), 1, SRC_AND_LOC, 1);

      // Disallow disassembling from `device_memory' in coredumps
      cuda_options_set_disassemble_from_elf_image ();

      /* Ensure the builtins objfile is created */
      cuda_init_cudart_symbols ();
    }
  catch (const gdb_exception_error &e)
    {
      if (e.reason < 0)
	{
	  if (inf != nullptr)
	    inf->pop_all_targets_at_and_above (process_stratum);

	  if (old_gdbarch != nullptr)
	    set_target_gdbarch (old_gdbarch);

	  registers_changed ();
	  reinit_frame_cache ();
	  cuda_cleanup ();

	  error (_ ("Could not open CUDA core file: %s"), e.what ());
	}
    }
}

void
cuda_core_target::close ()
{
  /* core_open will call detach and close but run_command will not call detach,
     hence we need to call cleanup here as well. */
  this->core_cleanup ();
  cuda_core_free ();

  /* If close got called, no more refs are pointing to this object. */
  delete this;
}

void
cuda_core_target::core_cleanup ()
{
  switch_to_no_thread ();
  exit_inferior (current_inferior ());

  clear_solib ();
  current_program_space->cbfd.reset (nullptr);
}

void
cuda_core_target::detach (inferior *inf, int from_tty)
{
  this->core_cleanup ();
  inf->unpush_target (this);
  registers_changed ();
  reinit_frame_cache ();

  if (from_tty)
    gdb_printf (_ ("No core file now.\n"));
}

void _initialize_cuda_corelow ();
void
_initialize_cuda_corelow ()
{
  add_target (cuda_core_target_info, cuda_core_target_open,
	      filename_completer);
}
