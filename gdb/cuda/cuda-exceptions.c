/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2015-2023 NVIDIA Corporation
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

#include "cuda-coords.h"
#include "cuda-exceptions.h"
#include "cuda-iterator.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "demangle.h"
#include "gdbsupport/gdb_signals.h"
#include "symfile.h"
#include "ui-out.h"

void
cuda_exception::print_exception_origin () const
{
  char *filename = NULL;
  struct ui_out *uiout = current_uiout;

  gdb_assert (m_valid);

  const auto &c = m_coord.physical ();

  bool precise = cuda_state::warp_has_error_pc (c.dev (), c.sm (), c.wp ());

  if (!precise)
    return;

  uiout->text ("The exception was triggered at ");

  uint64_t pc = cuda_state::warp_get_error_pc (c.dev (), c.sm (), c.wp ());
  struct cuda_debug_inline_info *inline_info = NULL;
  struct obj_section *section = find_pc_overlay (pc);
  struct symtab_and_line sal
      = find_pc_sect_line ((CORE_ADDR)pc, section, 0, &inline_info);

  if (sal.symtab && sal.line)
    {
      filename = strrchr ((char *)sal.symtab->filename, '/');
      if (filename)
        ++filename;
      else
        filename = (char *)sal.symtab->filename;
    }

  uiout->text ("PC ");
  uiout->field_fmt ("pc", "0x%llx", (unsigned long long)pc);

  if (filename)
    {
      uiout->text (" (");
      uiout->field_string ("filename", filename);
      uiout->text (":");
      uiout->field_signed ("line", sal.line);

      if (inline_info)
        {
          gdb::unique_xmalloc_ptr<char> demangled = language_demangle (
              language_def (current_language->la_language),
              inline_info->function, DMGL_ANSI);

          uiout->text (" in ");
          uiout->field_string ("inline_function", demangled.get ()
                                                      ? demangled.get ()
                                                      : inline_info->function);
          uiout->text (" inlined from ");
          uiout->field_string ("inline_filename",
                               lbasename (inline_info->filename));
          uiout->text (":");
          uiout->field_signed ("inline_line", inline_info->line);
        }
      uiout->text (")");
    }

  uiout->text ("\n");
}

void
cuda_exception::print_exception_device () const
{
  gdb_assert (m_valid);

  const auto &c = m_coord.physical ();

  current_uiout->text ("The exception was triggered in device ");
  current_uiout->field_signed ("device", c.dev ());
  current_uiout->text (".\n");
}

void
cuda_exception::print_assert_message () const
{
  current_uiout->text ("\nAssertion failed.\n");
}

void
cuda_exception::print_exception_name () const
{
  gdb_assert (m_valid);

  const char *name = gdb_signal_to_string (m_gdb_sig);

  current_uiout->text ("\nCUDA Exception: ");
  current_uiout->field_string ("exception_name", name);
  current_uiout->text ("\n");
}

void
cuda_exception::printMessage () const
{
  gdb_assert (m_valid);

  switch (m_gdb_sig)
    {
    case GDB_SIGNAL_CUDA_WARP_ASSERT:
      print_assert_message ();
      print_exception_origin ();
      break;
    case GDB_SIGNAL_CUDA_LANE_ILLEGAL_ADDRESS:
    case GDB_SIGNAL_CUDA_LANE_NONMIGRATABLE_ATOMSYS:
      print_exception_origin ();
      break;
    case GDB_SIGNAL_CUDA_DEVICE_ILLEGAL_ADDRESS:
    case GDB_SIGNAL_CUDA_DEVICE_HARDWARE_STACK_OVERFLOW:
      print_exception_name ();
      print_exception_device ();
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
      print_exception_name ();
      print_exception_origin ();
      break;
    case GDB_SIGNAL_CUDA_LANE_SYSCALL_ERROR:
    case GDB_SIGNAL_CUDA_LANE_USER_STACK_OVERFLOW:
      print_exception_name ();
      print_exception_origin ();
      break;
    case GDB_SIGNAL_CUDA_UNKNOWN_EXCEPTION:
    default:
      print_exception_name ();
      break;
    }
}

cuda_exception::cuda_exception ()
    : m_valid{ false }, m_recoverable{ false }, m_gdb_sig{ GDB_SIGNAL_0 },
      m_coord{}
{
  uint32_t dev = CUDA_WILDCARD;
  uint32_t sm = CUDA_WILDCARD;
  uint32_t wp = CUDA_WILDCARD;
  CUDBGException_t exception = CUDBG_EXCEPTION_NONE;

  /* Iteration should be limited to single sm if sstep is active */
  if (cuda_sstep_is_active ())
    {
      dev = cuda_sstep_dev_id ();
      /* Performance optimization below is not applicable if software
         preemption is enabled, since physical coordinates can change
         between the steps, so the debugger must have to iterate over the
         whole device in order to find the exception */
      if (!cuda_options_software_preemption ())
        {
          sm = cuda_sstep_sm_id ();
          /* If only one bit is set in warp mask limit iteration to it */
          cuda_api_warpmask tmp;
          cuda_api_clear_mask (&tmp);
          cuda_api_set_bit (&tmp, cuda_sstep_wp_id (), 1);
          if (cuda_api_eq_mask (cuda_sstep_wp_mask (), &tmp))
            wp = cuda_sstep_wp_id ();
        }
    }

  /* Create the filter to iterate over */
  cuda_coords filter{ dev,
                      sm,
                      wp,
                      CUDA_WILDCARD,
                      CUDA_WILDCARD,
                      CUDA_WILDCARD,
                      CUDA_WILDCARD_DIM,
                      CUDA_WILDCARD_DIM,
                      CUDA_WILDCARD_DIM };

  if (dev == CUDA_WILDCARD || cuda_state::device_has_exception (dev))
    {
      cuda_iterator<cuda_iterator_type::threads,
                    select_valid | select_excpt | select_sngl>
          itr{ filter };
      if (itr.size ())
        {
          m_coord = *itr.begin ();
          const auto &c = m_coord.physical ();
          exception = cuda_state::lane_get_exception (c.dev (), c.sm (),
                                                      c.wp (), c.ln ());
        }
    }

  switch (exception)
    {
    case CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS:
      m_gdb_sig = GDB_SIGNAL_CUDA_LANE_ILLEGAL_ADDRESS;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_LANE_USER_STACK_OVERFLOW:
      m_gdb_sig = GDB_SIGNAL_CUDA_LANE_USER_STACK_OVERFLOW;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_DEVICE_HARDWARE_STACK_OVERFLOW:
      m_gdb_sig = GDB_SIGNAL_CUDA_DEVICE_HARDWARE_STACK_OVERFLOW;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_ILLEGAL_INSTRUCTION:
      m_gdb_sig = GDB_SIGNAL_CUDA_WARP_ILLEGAL_INSTRUCTION;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS:
      m_gdb_sig = GDB_SIGNAL_CUDA_WARP_OUT_OF_RANGE_ADDRESS;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_MISALIGNED_ADDRESS:
      m_gdb_sig = GDB_SIGNAL_CUDA_WARP_MISALIGNED_ADDRESS;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_INVALID_ADDRESS_SPACE:
      m_gdb_sig = GDB_SIGNAL_CUDA_WARP_INVALID_ADDRESS_SPACE;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_INVALID_PC:
      m_gdb_sig = GDB_SIGNAL_CUDA_WARP_INVALID_PC;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW:
      m_gdb_sig = GDB_SIGNAL_CUDA_WARP_HARDWARE_STACK_OVERFLOW;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_DEVICE_ILLEGAL_ADDRESS:
      m_gdb_sig = GDB_SIGNAL_CUDA_DEVICE_ILLEGAL_ADDRESS;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_ASSERT:
      m_gdb_sig = GDB_SIGNAL_CUDA_WARP_ASSERT;
      m_valid = true;
      m_recoverable = true;
      break;
    case CUDBG_EXCEPTION_LANE_SYSCALL_ERROR:
      m_gdb_sig = GDB_SIGNAL_CUDA_LANE_SYSCALL_ERROR;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS:
      m_gdb_sig = GDB_SIGNAL_CUDA_WARP_ILLEGAL_ADDRESS;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_LANE_NONMIGRATABLE_ATOMSYS:
      m_gdb_sig = GDB_SIGNAL_CUDA_LANE_NONMIGRATABLE_ATOMSYS;
      m_valid = true;
      m_recoverable = false;
      break;
#if (CUDBG_API_VERSION_REVISION >= 131)
    case CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT:
      m_gdb_sig = GDB_SIGNAL_CUDA_CLUSTER_BLOCK_NOT_PRESENT;
      m_valid = true;
      m_recoverable = false;
      break;
    case CUDBG_EXCEPTION_CLUSTER_OUT_OF_RANGE_ADDRESS:
      m_gdb_sig = GDB_SIGNAL_CUDA_CLUSTER_OUT_OF_RANGE_ADDRESS;
      m_valid = true;
      m_recoverable = false;
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
      warning ("Encountered unhandled device exception (%d)\n", exception);
      m_gdb_sig = GDB_SIGNAL_CUDA_UNKNOWN_EXCEPTION;
      m_valid = true;
      m_recoverable = false;
      break;
    }
}

const char *
cuda_exception::name () const
{
  gdb_assert (m_valid);
  return gdb_signal_to_string (m_gdb_sig);
}

const char *
cuda_exception::type_to_name (CUDBGException_t type)
{
  switch (type)
    {
    case CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_LANE_ILLEGAL_ADDRESS);
    case CUDBG_EXCEPTION_LANE_USER_STACK_OVERFLOW:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_LANE_USER_STACK_OVERFLOW);
    case CUDBG_EXCEPTION_DEVICE_HARDWARE_STACK_OVERFLOW:
      return gdb_signal_to_string (
          GDB_SIGNAL_CUDA_DEVICE_HARDWARE_STACK_OVERFLOW);
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
      return gdb_signal_to_string (
          GDB_SIGNAL_CUDA_WARP_HARDWARE_STACK_OVERFLOW);
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
      return gdb_signal_to_string (
          GDB_SIGNAL_CUDA_CLUSTER_OUT_OF_RANGE_ADDRESS);
#endif
    default:
      return gdb_signal_to_string (GDB_SIGNAL_CUDA_UNKNOWN_EXCEPTION);
    }
}
