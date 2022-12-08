/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2015-2022 NVIDIA Corporation
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
#include "ui-out.h"
#include "gdbsupport/gdb_signals.h"
#include "symfile.h"
#include "demangle.h"
#include "cuda-coords.h"
#include "cuda-exceptions.h"
#include "cuda-iterator.h"
#include "cuda-options.h"
#include "cuda-state.h"

struct cuda_exception_st {
  bool valid;
  bool recoverable;
  bool error_pc_available;
  uint32_t value;
  cuda_coords_t coords;
};

// XXX temporary, until we get list of exceptions instead
struct cuda_exception_st cuda_exception_object;
cuda_exception_t cuda_exception = &cuda_exception_object;

bool
cuda_exception_is_valid (cuda_exception_t exception)
{
  gdb_assert (exception);
  return exception->valid;
}

bool
cuda_exception_is_recoverable (cuda_exception_t exception)
{
  gdb_assert (exception);
  gdb_assert (exception->valid);
  return exception->recoverable;
}

uint32_t
cuda_exception_get_value (cuda_exception_t exception)
{
  gdb_assert (exception);
  gdb_assert (exception->valid);
  return exception->value;
}

cuda_coords_t
cuda_exception_get_coords (cuda_exception_t exception)
{
  gdb_assert (exception);
  gdb_assert (exception->valid);
  gdb_assert (exception->coords.valid);
  return exception->coords;
}

void
cuda_exception_reset (cuda_exception_t exception)
{
  gdb_assert (exception);
  exception->valid = false;
  exception->recoverable = false;
  exception->value = 0U;
  exception->coords.valid = false;
}

static void
print_exception_origin (cuda_exception_t exception)
{
  cuda_coords_t c;
  uint64_t pc;
  bool precise;
  struct symtab_and_line sal;
  char *filename = NULL;
  struct ui_out *uiout = current_uiout;

  gdb_assert (cuda_exception_is_valid (exception));

  c = cuda_exception_get_coords (exception);
  precise = warp_has_error_pc (c.dev, c.sm, c.wp);

  if (!precise)
    {
      return;
    }

  uiout->text ("The exception was triggered at ");
  pc = warp_get_error_pc (c.dev, c.sm, c.wp);

  struct cuda_debug_inline_info *inline_info = NULL;

  struct obj_section *section = find_pc_overlay (pc);
  sal = find_pc_sect_line ((CORE_ADDR)pc, section, 0, &inline_info);

  if (sal.symtab && sal.line)
    {
      filename = strrchr ((char *)sal.symtab->filename, '/');
      if (filename)
        ++filename;
      else
        filename = (char *) sal.symtab->filename;
    }

  uiout->text ("PC ");
  uiout->field_fmt ("pc", "0x%llx",  (unsigned long long)pc);

  if (filename)
    {
      uiout->text         (" (");
      uiout->field_string ("filename"    , filename);
      uiout->text         (":");
      uiout->field_signed    ("line"        , sal.line);

      if (inline_info)
	{
	  gdb::unique_xmalloc_ptr<char> demangled = language_demangle (language_def (current_language->la_language),
								       inline_info->function,
								       DMGL_ANSI);

	  uiout->text         (" in ");
	  uiout->field_string ("inline_function", demangled.get () ? demangled.get () : inline_info->function);
	  uiout->text         (" inlined from ");
	  uiout->field_string ("inline_filename", lbasename (inline_info->filename));
	  uiout->text         (":");
	  uiout->field_signed    ("inline_line", inline_info->line);
	}
      uiout->text         (")");
    }

  uiout->text ("\n");
}

static void
print_exception_device (cuda_exception_t exception)
{
  cuda_coords_t c;

  gdb_assert (cuda_exception_is_valid (exception));

  c = cuda_exception_get_coords (exception);

  current_uiout->text ("The exception was triggered in device ");
  current_uiout->field_signed ("device", c.dev);
  current_uiout->text (".\n");
}

static void
print_assert_message (cuda_exception_t exception)
{
  current_uiout->text ("\nAssertion failed.\n");
}

static void
print_exception_name (cuda_exception_t exception)
{
  enum gdb_signal value;
  const char *name;

  gdb_assert (cuda_exception_is_valid (exception));

  value = (enum gdb_signal) cuda_exception_get_value (exception);
  name = gdb_signal_to_string (value);

  current_uiout->text ("\nCUDA Exception: ");
  current_uiout->field_string ("exception_name", name);
  current_uiout->text ("\n");
}

void
cuda_exception_print_message (cuda_exception_t exception)
{
  switch (cuda_exception_get_value (exception))
  {
    case GDB_SIGNAL_CUDA_WARP_ASSERT:
      print_assert_message (exception);
      print_exception_origin (exception);
      break;
    case GDB_SIGNAL_CUDA_LANE_ILLEGAL_ADDRESS:
    case GDB_SIGNAL_CUDA_LANE_NONMIGRATABLE_ATOMSYS:
      print_exception_origin (exception);
      break;
    case GDB_SIGNAL_CUDA_DEVICE_ILLEGAL_ADDRESS:
    case GDB_SIGNAL_CUDA_DEVICE_HARDWARE_STACK_OVERFLOW:
      print_exception_name (exception);
      print_exception_device (exception);
      break;
    case GDB_SIGNAL_CUDA_WARP_ILLEGAL_INSTRUCTION:
    case GDB_SIGNAL_CUDA_WARP_OUT_OF_RANGE_ADDRESS:
    case GDB_SIGNAL_CUDA_WARP_MISALIGNED_ADDRESS:
    case GDB_SIGNAL_CUDA_WARP_INVALID_ADDRESS_SPACE:
    case GDB_SIGNAL_CUDA_WARP_INVALID_PC:
    case GDB_SIGNAL_CUDA_WARP_HARDWARE_STACK_OVERFLOW:
    case GDB_SIGNAL_CUDA_WARP_ILLEGAL_ADDRESS:
#if (CUDBG_API_VERSION_REVISION >= 131)
    case GDB_SIGNAL_CUDA_CLUSTER_OUT_OF_RANGE_ADDRESS:
    case GDB_SIGNAL_CUDA_CLUSTER_BLOCK_NOT_PRESENT:
#endif
      print_exception_name (exception);
      print_exception_origin (exception);
      break;
    case GDB_SIGNAL_CUDA_LANE_SYSCALL_ERROR:
    case GDB_SIGNAL_CUDA_LANE_USER_STACK_OVERFLOW:
      print_exception_name (exception);
      print_exception_origin (exception);
      break;
    case GDB_SIGNAL_CUDA_UNKNOWN_EXCEPTION:
    default:
      print_exception_name (exception);
      break;
  }
}

bool
cuda_exception_hit_p (cuda_exception_t exception)
{
  CUDBGException_t exception_type = CUDBG_EXCEPTION_NONE;
  cuda_coords_t c = CUDA_INVALID_COORDS, filter = CUDA_WILDCARD_COORDS;
  cuda_api_warpmask tmp;
  cuda_iterator itr;

  /* Iteration should be limited to single sm if sstep is active */
  if (cuda_sstep_is_active())
    {
      filter.dev = cuda_sstep_dev_id();
      /* Performance optimization below is not applicable if software
         preemption is enabled, since physical coordinates can change
         between the steps, so the debugger must have to iterate over the
         whole device in order to find the exception */
      if (!cuda_options_software_preemption ())
        {
          filter.sm = cuda_sstep_sm_id();
          /* If only one bit is set in warp mask limit iteration to it */
          cuda_api_clear_mask(&tmp);
          cuda_api_set_bit(&tmp, cuda_sstep_wp_id(), 1);
          if(cuda_api_eq_mask(cuda_sstep_wp_mask(), &tmp))
            filter.wp = cuda_sstep_wp_id();
        }
    }

  if (filter.dev == CUDA_WILDCARD || device_has_exception (filter.dev))
    {
      itr = cuda_iterator_create (CUDA_ITERATOR_TYPE_THREADS, &filter,
                                  (cuda_select_t)(CUDA_SELECT_VALID | CUDA_SELECT_EXCPT | CUDA_SELECT_SNGL));
      cuda_iterator_start (itr);
      if (!cuda_iterator_end (itr))
        {
          c = cuda_iterator_get_current (itr);
          exception_type = lane_get_exception (c.dev, c.sm, c.wp, c.ln);
        }
      cuda_iterator_destroy (itr);
    }

  exception->coords = c;

  switch (exception_type)
    {
    case CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS:
      exception->value = GDB_SIGNAL_CUDA_LANE_ILLEGAL_ADDRESS;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_LANE_USER_STACK_OVERFLOW:
      exception->value = GDB_SIGNAL_CUDA_LANE_USER_STACK_OVERFLOW;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_DEVICE_HARDWARE_STACK_OVERFLOW:
      exception->value = GDB_SIGNAL_CUDA_DEVICE_HARDWARE_STACK_OVERFLOW;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION:
      exception->value = GDB_SIGNAL_CUDA_WARP_ILLEGAL_INSTRUCTION;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS:
      exception->value = GDB_SIGNAL_CUDA_WARP_OUT_OF_RANGE_ADDRESS;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS:
      exception->value = GDB_SIGNAL_CUDA_WARP_MISALIGNED_ADDRESS;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE:
      exception->value = GDB_SIGNAL_CUDA_WARP_INVALID_ADDRESS_SPACE;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_INVALID_PC:
      exception->value = GDB_SIGNAL_CUDA_WARP_INVALID_PC;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW:
      exception->value = GDB_SIGNAL_CUDA_WARP_HARDWARE_STACK_OVERFLOW;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS:
      exception->value = GDB_SIGNAL_CUDA_DEVICE_ILLEGAL_ADDRESS;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_ASSERT:
      exception->value = GDB_SIGNAL_CUDA_WARP_ASSERT;
      exception->valid = true;
      exception->recoverable = true;
      break;
    case CUDBG_EXCEPTION_LANE_SYSCALL_ERROR:
      exception->value = GDB_SIGNAL_CUDA_LANE_SYSCALL_ERROR;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS:
      exception->value = GDB_SIGNAL_CUDA_WARP_ILLEGAL_ADDRESS;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_LANE_NONMIGRATABLE_ATOMSYS:
      exception->value = GDB_SIGNAL_CUDA_LANE_NONMIGRATABLE_ATOMSYS;
      exception->valid = true;
      exception->recoverable = false;
      break;
#if (CUDBG_API_VERSION_REVISION >= 131)
    case CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT:
      exception->value = GDB_SIGNAL_CUDA_CLUSTER_BLOCK_NOT_PRESENT;
      exception->valid = true;
      exception->recoverable = false;
      break;
    case CUDBG_EXCEPTION_CLUSTER_OUT_OF_RANGE_ADDRESS:
      exception->value = GDB_SIGNAL_CUDA_CLUSTER_OUT_OF_RANGE_ADDRESS;
      exception->valid = true;
      exception->recoverable = false;
      break;
#endif
    case CUDBG_EXCEPTION_NONE:
      break;
    case CUDBG_EXCEPTION_UNKNOWN:
    default:
      /* If for some reason the device encounters an unknown exception, we
         still need to halt the chip and allow state inspection.  Just emit
         a warning indicating this is something that was unexpected, but
         handle it as a normal device exception. */
      warning ("Encountered unhandled device exception (%d)\n", exception_type);
      exception->value = GDB_SIGNAL_CUDA_UNKNOWN_EXCEPTION;
      exception->valid = true;
      exception->recoverable = false;
      break;
    }

  return exception_type != CUDBG_EXCEPTION_NONE;
}

const char *
cuda_exception_type_to_name (CUDBGException_t exception_type)
{
  switch (exception_type)
    {
    case CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_LANE_ILLEGAL_ADDRESS);
    case CUDBG_EXCEPTION_LANE_USER_STACK_OVERFLOW:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_LANE_USER_STACK_OVERFLOW);
    case CUDBG_EXCEPTION_DEVICE_HARDWARE_STACK_OVERFLOW:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_DEVICE_HARDWARE_STACK_OVERFLOW);
    case CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_WARP_ILLEGAL_INSTRUCTION);
    case CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_WARP_OUT_OF_RANGE_ADDRESS);
    case CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_WARP_MISALIGNED_ADDRESS);
    case CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_WARP_INVALID_ADDRESS_SPACE);
    case CUDBG_EXCEPTION_WARP_INVALID_PC:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_WARP_INVALID_PC);
    case CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_WARP_HARDWARE_STACK_OVERFLOW);
    case CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_DEVICE_ILLEGAL_ADDRESS);
    case CUDBG_EXCEPTION_WARP_ASSERT:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_WARP_ASSERT);
    case CUDBG_EXCEPTION_LANE_SYSCALL_ERROR:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_LANE_SYSCALL_ERROR);
    case CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_WARP_ILLEGAL_ADDRESS);
    case CUDBG_EXCEPTION_LANE_NONMIGRATABLE_ATOMSYS:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_LANE_NONMIGRATABLE_ATOMSYS);
#if (CUDBG_API_VERSION_REVISION >= 131)
    case CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_CLUSTER_BLOCK_NOT_PRESENT);
    case CUDBG_EXCEPTION_CLUSTER_OUT_OF_RANGE_ADDRESS:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_CLUSTER_OUT_OF_RANGE_ADDRESS);
#endif
    default:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_UNKNOWN_EXCEPTION);
    }
}
