/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2014 NVIDIA Corporation
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
#include "inferior.h"
#include "gdbthread.h"
#include "arch-utils.h"
#include "regcache.h"

#include <block.h>

#include "cuda-autostep.h"
#include "cuda-state.h"
#include "cuda-iterator.h"
#include "cuda-frame.h"

/* When inside an autostep range, we go into single-step mode */
static bool autostep_stepping = false;
static bool autostep_pending = false;

/* From infcmd.c */
extern void delete_longjmp_breakpoint_cleanup (void *ignore);
extern void step_1 (int skip_subroutines, int single_inst, char *count_string);

/* Getters and setters */

bool
cuda_get_autostep_pending (void)
{
  return autostep_pending;
}

void
cuda_set_autostep_pending (bool pending)
{
  autostep_pending = pending;
}

bool
cuda_get_autostep_stepping (void)
{
  return autostep_stepping;
}

void
cuda_set_autostep_stepping (bool stepping)
{
  autostep_stepping = stepping;
}

bool
cuda_autostep_stop (void)
{
  struct thread_info *tp = NULL;

  if (target_has_execution && !ptid_equal (inferior_ptid, null_ptid))
    tp = inferior_thread ();

  return !tp || !tp->control.stop_step;
}

static void
autostep_report_exception_host (uint64_t before_pc)
{
  /* We know the exception must have been at the previous pc */
  struct gdbarch *gdbarch = get_current_arch ();
  struct type *type_uint32   = builtin_type (gdbarch)->builtin_uint32;
  struct type *type_data_ptr = builtin_type (gdbarch)->builtin_data_ptr;

  struct symtab_and_line before_sal = find_pc_line (before_pc, 0);

  printf_filtered (_("Autostep precisely caught exception at %s:%d (0x%" PRIx64 ")\n"),
    before_sal.symtab->filename, before_sal.line, before_pc);

  set_internalvar (lookup_internalvar ("autostep_exception_pc"),
    value_from_longest (type_data_ptr, (LONGEST) before_pc));
  set_internalvar (lookup_internalvar ("autostep_exception_line"),
    value_from_longest (type_uint32, (LONGEST) before_sal.line));
}

static void
autostep_report_exception_device (int before_ln, uint64_t before_pc,
  uint64_t after_pc)
{
  cuda_coords_t c;
  struct gdbarch *gdbarch = get_current_arch ();
  struct type *type_uint32   = builtin_type (gdbarch)->builtin_uint32;
  struct type *type_data_ptr = builtin_type (gdbarch)->builtin_data_ptr;

  cuda_print_message_focus (false);

  /* If the thread before stepping is also active, the exception didn't occur
     in a divergent thread */
  cuda_coords_get_current (&c);
  if (lane_is_active (c.dev, c.sm, c.wp, before_ln))
    {
      /* Exception in active lane. We know the exception must have been at the
         previous pc */

      struct symtab_and_line before_sal = find_pc_line (before_pc, 0);

      if (before_sal.symtab && before_sal.line)
        printf_filtered (_("Autostep precisely caught exception at %s:%d (0x%" PRIx64 ")\n"),
        before_sal.symtab->filename, before_sal.line, before_pc);
      else
        printf_filtered (_("Autostep precisely caught exception. (0x%" PRIx64 ")\n"), before_pc);

      set_internalvar (lookup_internalvar ("autostep_exception_pc"),
        value_from_longest (type_data_ptr, (LONGEST) before_pc));
      set_internalvar (lookup_internalvar ("autostep_exception_line"),
        value_from_longest (type_uint32, (LONGEST) before_sal.line));
    }
  else
    {
      /* Exception in divergent lane. We can't use the before_pc so we'll guess
         that the exception was at after_pc-8. This is true in all but some
         obscure cases. */

      uint64_t guess_pc;
      struct symtab_and_line guess_sal;

      cuda_api_get_adjusted_code_address (c.dev, after_pc, &guess_pc, CUDBG_ADJ_PREVIOUS_ADDRESS);

      guess_sal = find_pc_line (guess_pc, 0);

      printf_filtered (_("Autostep caught exception at instruction before 0x%" PRIx64 "\n"),
        after_pc);
      printf_filtered (_("This is probably %s:%d (0x%" PRIx64 ")\n"),
        guess_sal.symtab->filename, guess_sal.line, guess_pc);

      set_internalvar (lookup_internalvar ("autostep_exception_pc"),
        value_from_longest (type_data_ptr, (LONGEST) guess_pc));
      set_internalvar (lookup_internalvar ("autostep_exception_line"),
        value_from_longest (type_uint32, (LONGEST) guess_sal.line));
    }
}

/* Single steps all the host code if it is at an autostep */
static void
handle_autostep_host (void)
{
  struct breakpoint *astep, *overlap;
  uint64_t before_pc, after_pc;
  int remaining;
  bool single_inst;

  /* Check we're at the an autostep bp */
  before_pc = stop_pc;
  astep = cuda_find_autostep_by_addr (before_pc);
  if (astep == NULL)
    {
      cuda_set_autostep_pending (false);
      return;
    }

  /* Check if we single step or step by lines */
  single_inst = astep->cuda_autostep_length_type == cuda_autostep_insts;

  /* This suppresses printing the line/instruction after each step */
  cuda_set_autostep_stepping (true);

  /* Step until we are out of the autostep range */
  remaining = astep->cuda_autostep_length;

  while (remaining > 0)
    {
      before_pc = regcache_read_pc (get_current_regcache ());

      /* Clear pending flag to test if we encounter another autostep */
      cuda_set_autostep_pending (false);

      /* Basically does a next/nexti */
      step_1 (false, single_inst, NULL);

      if (cuda_autostep_stop ())
        {
          struct thread_info *tp = NULL;

          if (target_has_execution && !ptid_equal (inferior_ptid, null_ptid))
            tp = inferior_thread ();

          if (tp && signal_pass_state (tp->suspend.stop_signal))
            {
              /* This is an exception */
              autostep_report_exception_host (before_pc);
            }
          else
            {
              /* This is a breakpoint or the user stopped the program. */
              warning (_("Program stopped during an autostep range.\n"
                "Autostepping will not resume upon continuing.\n"));
            }

          break;
        }

      after_pc = regcache_read_pc (get_current_regcache ());

      /* Handle overlapping autosteps */
      if (cuda_get_autostep_pending ())
        {
          overlap = cuda_find_autostep_by_addr (after_pc);
          warning (_("Overlapping autostep %d ignored"), overlap->number);
        }

      remaining--;
    }

  /* Mark that autostepping has been handled */
  cuda_set_autostep_pending (false);
  cuda_set_autostep_stepping (false);
}

static uint64_t
find_end_pc(uint64_t pc)
{
  struct block *bl;
  struct minimal_symbol *msymbol;

  bl = block_for_pc ((CORE_ADDR)pc);
  if (bl)
       return BLOCK_END (bl);

  msymbol = lookup_minimal_symbol_by_pc ((CORE_ADDR)pc);
  if (msymbol)
       return SYMBOL_VALUE_ADDRESS (msymbol) + MSYMBOL_SIZE (msymbol);

  return (uint64_t)-1LL;
}

/* Single steps all the warps that are at an autostep */
static void
handle_autostep_device ()
{
  cuda_iterator iter;
  cuda_coords_t after_coords;
  struct breakpoint *astep, *overlap;
  uint64_t before_pc, after_pc, end_pc;
  int remaining;
  struct cleanup *old_cleanups;
  bool single_inst;
  cuda_coords_t filter;
  const char *sm_type;

  /* This suppresses printing of the line after each step */
  cuda_set_autostep_stepping (true);

  /* Iterate through all warps in current grid that are at a breakpoint */

  filter = CUDA_WILDCARD_COORDS;
  filter.valid = true;
  filter.gridId = CUDA_CURRENT;
  cuda_coords_evaluate_current (&filter, false);

  iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_WARPS, &filter,
    CUDA_SELECT_BKPT | CUDA_SELECT_VALID);
  old_cleanups = make_cleanup ((make_cleanup_ftype*)cuda_iterator_destroy,
    (void*)iter);

  for (cuda_iterator_start (iter);
       cuda_focus_is_device () && !cuda_iterator_end (iter);
       cuda_iterator_next (iter))
    {
      cuda_coords_t c = cuda_iterator_get_current (iter);

      /* Make sure the warp didn't become invalid */
      if (!warp_is_valid (c.dev, c.sm, c.wp))
        continue;

      /* Check we're at the autostep bp */
      before_pc = warp_get_active_virtual_pc (c.dev, c.sm, c.wp);
      astep = cuda_find_autostep_by_addr (before_pc);
      if (astep == NULL || astep->enable_state != bp_enabled)
        continue;

      /* Check that the device is Fermi or better */
      /* Must check here in case user re-enabled it */
      sm_type = device_get_sm_type (c.dev);
      if (strncmp (sm_type, "sm_1", 4) == 0)
        {
          warning ("Disabling autostep %d on device %d because autostep "
            "requires compute capability 2.0 or higher.", astep->number, c.dev);
          astep->enable_state = bp_disabled;
          continue;
        }

      /* Check if we single step or step by lines */
      single_inst = astep->cuda_autostep_length_type == cuda_autostep_insts;

      /* Set focus to current warp */
      c.ln = warp_get_lowest_active_lane (c.dev, c.sm, c.wp);
      cuda_coords_set_current_physical (c.dev, c.sm, c.wp, c.ln);

      /* Step until we are out of the autostep range */
      remaining = astep->cuda_autostep_length;

      while (remaining>0)
        {
          before_pc = warp_get_active_virtual_pc (c.dev, c.sm, c.wp);

          /* If pc is in the top frame - do not allow autostepping outside of kernel boundaries */
          if ( cuda_frame_outermost_p (get_next_frame (get_current_frame ())))
            {
              end_pc = find_end_pc (before_pc);
              if (before_pc >= end_pc)
                break;
            }

          /* Clear pending flag to test if we encounter another autostep */
          cuda_set_autostep_pending (false);

          /* Basically does a next/nexti */
          step_1 (false, single_inst, NULL);

          /* Make sure we can continue stepping this warp */
          if (!cuda_focus_is_device () || !warp_is_valid (c.dev, c.sm, c.wp))
            break;

          /* If warp changed, it means the original warp ran to completion so we
             should move on to next warp. */
          cuda_coords_get_current (&after_coords);
          if (after_coords.dev != c.dev || after_coords.sm != c.sm
              || after_coords.wp != c.wp)
            break;

          after_pc = warp_get_active_virtual_pc (c.dev, c.sm, c.wp);

          if (cuda_autostep_stop ())
            {
              struct thread_info *tp = NULL;

              if (target_has_execution && !ptid_equal (inferior_ptid, null_ptid))
                tp = inferior_thread ();

              if (tp && signal_pass_state (tp->suspend.stop_signal))
                {
                  /* This is an exception */
                  autostep_report_exception_device (c.ln, before_pc, after_pc);
                  cuda_set_autostep_pending (false);
                }
              else
                {
                  /* This is a breakpoint or the user stopped the program. */
                  warning (_("Program stopped during an autostep range.\n"
                   "This warp will not autostep upon resuming."));
                  /* Resume any pending warps but not current warp. */
                  cuda_set_autostep_pending (true);
                }

              cuda_set_autostep_stepping (false);

              do_cleanups (old_cleanups);
              return;
            }

          remaining--;

          /* Handle overlapping autosteps */
          if (cuda_get_autostep_pending ())
            {
              overlap = cuda_find_autostep_by_addr (after_pc);
              if (remaining > 0)
                {
                  warning (_("Overlapping autostep %d ignored"), overlap->number);
                }
              else
                {
                  /* The last astep just finished, go onto the next one */
                  astep = overlap;
                  remaining = overlap->cuda_autostep_length;
                }
            }
        }
    }

  /* Mark that autostepping has been handled */
  cuda_set_autostep_pending (false);
  cuda_set_autostep_stepping (false);

  do_cleanups (old_cleanups);
}

void
cuda_handle_autostep ()
{
  if (!cuda_get_autostep_pending ())
    return;

  if (cuda_focus_is_device ())
    handle_autostep_device ();
  else
    handle_autostep_host ();
}

