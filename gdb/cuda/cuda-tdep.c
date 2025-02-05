/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2024 NVIDIA Corporation
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
#include <dlfcn.h>
#include <sys/stat.h>
#include <time.h>
#if !defined(__QNX__)
#include <sys/syscall.h>
#endif
#include <pthread.h>
#include <signal.h>
#if !defined(__QNX__)
#include <execinfo.h>
#endif
#include <string>
#include <unordered_map>
#include <vector>

#include "arch-utils.h"
#include "block.h"
#include "breakpoint.h"
#include "command.h"
#include "demangle.h"
#include "dis-asm.h"
#include "dummy-frame.h"
#include "dwarf2/frame.h"
#include "exceptions.h"
#include "exec.h"
#include "floatformat.h"
#include "frame-base.h"
#include "frame-unwind.h"
#include "frame.h"
#include "gdb/signals.h"
#include "gdbcmd.h"
#include "gdbcore.h"
#include "gdbthread.h"
#include "inferior.h"
#include "language.h"
#include "linux-tdep.h"
#include "main.h"
#include "objfiles.h"
#include "observable.h"
#include "osabi.h"
#include "regcache.h"
#include "reggroups.h"
#include "regset.h"
#include "source.h"
#include "symfile.h"
#include "symtab.h"
#include "target.h"
#include "user-regs.h"
#include "valprint.h"
#include "value.h"
#include "remote.h"

#include "elf-bfd.h"

#include "cuda-asm.h"
#include "cuda-autostep.h"
#include "cuda-context.h"
#include "cuda-frame.h"
#include "cuda-coord-set.h"
#include "cuda-modules.h"
#include "cuda-notifications.h"
#include "cuda-options.h"
#include "cuda-packet-manager.h"
#include "cuda-regmap.h"
#include "cuda-special-register.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"
#include "cudadebugger.h"
#include "gdb_bfd.h"
#include "mach-o.h"
#include "self-bt.h"

#define REGMAP_CLASS(x) (x >> 24)
#define REGMAP_REG(x) (x & 0xffffff)

/* Globals */
/* FIXME: Move these to the arch struct! */
CuDim3 gridDim = { 0, 0, 0 };
CuDim3 blockDim = { 0, 0, 0 };
bool cuda_initialized = false;
bool cuda_is_target_mourn_pending = false;
static bool inferior_in_debug_mode = false;
static char cuda_gdb_session_dir[CUDA_GDB_TMP_BUF_SIZE] = { 0 };
static uint32_t cuda_gdb_session_id = 0;
static std::unordered_map<uint64_t, int> cuda_ptx_virtual_map{};
static std::vector<std::string> cuda_ptx_virtual_str{};

int cuda_host_shadow_debug = 0;

bool
cuda_platform_supports_tid (void)
{
  /* Using TIDs on aarch64 was disabled due to DTCGDB-265.

     On aarch64, doing a tkill() right after pthread_join() results in a
     success, but no signal is delivered (or intercepted by GDB). This is
     reproducible with a standalone application without CUDA or GDB. Since
     cuda-gdb relies on signals for notification delivery, in some cases this
     would lead to hangs, because from the cuda-gdb point of view the signals
     have been "sent", but never "received". */
#if defined(linux) && defined(SYS_gettid) && !defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

int
cuda_gdb_get_tid_or_pid (ptid_t ptid)
{
  if (cuda_platform_supports_tid ())
    return ptid.tid ();
  else
    return ptid.pid ();
}

/* CUDA - skip prologue */
bool cuda_producer_is_open64;

/* CUDA - siginfo */
static int cuda_signo = 0;

static void
cuda_siginfo_trace (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_SIGINFO, fmt, ap);
  va_end (ap);
}

void
cuda_set_signo (int signo)
{
  cuda_signo = signo;

  cuda_siginfo_trace ("CUDA siginfo set to %d", signo);
}

int
cuda_get_signo (void)
{
  return cuda_signo;
}

/* Routines */

gdb::unique_xmalloc_ptr<char>
cuda_find_function_name_from_pc (CORE_ADDR pc, bool demangle)
{
  const char *name = nullptr;
  enum language lang = language_unknown;
  struct bound_minimal_symbol msymbol = lookup_minimal_symbol_by_pc (pc);

  /* find the mangled name */
  struct symbol *kernel = find_pc_function (pc);
  if (kernel && msymbol.minsym != nullptr
      && (msymbol.minsym->value_address (msymbol.objfile)
          > kernel->value_block ()->start ()))
    {
      name = msymbol.minsym->linkage_name ();
      lang = msymbol.minsym->language ();
    }
  else if (kernel)
    {
      name = kernel->linkage_name ();
      lang = kernel->language ();
    }
  else if (msymbol.minsym != NULL)
    {
      name = msymbol.minsym->linkage_name ();
      lang = msymbol.minsym->language ();
    }

  /* Return early, if name is not found */
  if (!name)
    return NULL;
  /* process the mangled name */
  else if (demangle)
    {
      gdb::unique_xmalloc_ptr<char> demangled
          = language_demangle (language_def (lang), name, DMGL_ANSI);
      if (demangled)
        return demangled;
      else
        return make_unique_xstrdup (name);
    }
  else
    return make_unique_xstrdup (name);
}

ATTRIBUTE_PRINTF (2, 0)
void
cuda_vtrace_domain (cuda_trace_domain_t domain, const char *fmt, va_list ap)
{
  if (!cuda_options_trace_domain_enabled (domain))
    return;

  // Try to keep stdout/stderr in sync as much as possible
  fflush (stdout);

  auto output = stdout;
  switch (domain)
    {
    case CUDA_TRACE_GENERAL:
      fprintf (output, "[CUDAGDB-GEN] ");
      break;
    case CUDA_TRACE_EVENT:
      fprintf (output, "[CUDAGDB-EVT] ");
      break;
    case CUDA_TRACE_BREAKPOINT:
      fprintf (output, "[CUDAGDB-BPT] ");
      break;
    case CUDA_TRACE_API:
      fprintf (output, "[CUDAGDB-API] ");
      break;
    case CUDA_TRACE_SIGINFO:
      fprintf (output, "[CUDAGDB-SIG] ");
      break;
    case CUDA_TRACE_DISASSEMBLER:
      fprintf (output, "[CUDAGDB-DISASM] ");
      break;
    case CUDA_TRACE_STATE:
      fprintf (output, "[CUDAGDB-STATE] ");
      break;
    case CUDA_TRACE_STATE_DECODE:
      fprintf (output, "[CUDAGDB-STATE-DECODE] ");
      break;
    default:
      fprintf (output, "[CUDAGDB] ");
      break;
    }
  vfprintf (output, fmt, ap);
  fprintf (output, "\n");
  fflush (output);
}

ATTRIBUTE_PRINTF (1, 2) void cuda_trace (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_GENERAL, fmt, ap);
  va_end (ap);
}

ATTRIBUTE_PRINTF (2, 3)
void
cuda_trace_domain (cuda_trace_domain_t domain, const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (domain, fmt, ap);
  va_end (ap);
}

bool
cuda_breakpoint_hit_p (cuda_coords &coords)
{
  /* With software preepmtion the physical coords might have changed. */
  if (cuda_options_software_preemption () && cuda_current_focus::isDevice ())
    {
      const auto &l = cuda_current_focus::get ().logical ();
      cuda_coords filter{ CUDA_WILDCARD,   CUDA_WILDCARD, CUDA_WILDCARD,
                          CUDA_WILDCARD,   l.kernelId (), l.gridId (),
                          l.clusterIdx (), l.blockIdx (), l.threadIdx () };
      cuda_coord_set<cuda_coord_set_type::threads,
                    select_valid | select_bkpt | select_trap
                        | select_current_clock | select_sngl>
          coord{ filter };
      if (coord.size ())
        {
          coords = *coord.begin ();
          return true;
        }
    }

  /* First check the current focus. */
  gdb::optional<cuda_coords> origin;
  if (cuda_current_focus::isDevice ())
    {
      origin = cuda_current_focus::get ();
      cuda_coord_set<cuda_coord_set_type::threads,
                    select_valid | select_bkpt | select_trap
                        | select_current_clock | select_sngl>
          coord{ cuda_current_focus::get () };
      if (coord.size ())
        {
          coords = *coord.begin ();
          return true;
        }
    }

  /* Now check for any breakpoint. */
  if (cuda_options_thread_selection_logical ())
    {
      cuda_coord_set<cuda_coord_set_type::threads,
                    select_valid | select_bkpt | select_trap
                        | select_current_clock,
                    cuda_coord_compare_type::logical>
          coord{ cuda_coords::wild (), origin };
      if (coord.size ())
        {
          coords = *coord.begin ();
          return true;
        }
    }
  else
    {
      cuda_coord_set<cuda_coord_set_type::threads,
                    select_valid | select_bkpt | select_trap
                        | select_current_clock | select_sngl,
                    cuda_coord_compare_type::physical>
          coord{ cuda_coords::wild (), origin };
      if (coord.size ())
        {
          coords = *coord.begin ();
          return true;
        }
    }

  return false;
}

/* Check if a dwarf2 register is a ptx virtual register string.
 * We look at the encoding to determine if it is.
 * Example:
 *                      as char[4]    as uint32_t
 *   device reg %r4 :      "\04r%"     0x00257234
 *   host reg  4    : "\0\0\0\004"     0x00000004
 */
uint64_t
cuda_check_dwarf2_reg_ptx_virtual_register (uint64_t dwarf2_reg)
{
  /* Ensure that the dwarf2_reg is greater than one character. Anything less
   * would be an invalid ascii string. */
  if (dwarf2_reg <= 0xff)
    return dwarf2_reg;

  /* Convert the uleb128 to a string. The order of characters
   * has to be reversed in order to be read as a standard string. */
  uint64_t reg_copy = dwarf2_reg;
  std::array<char, sizeof (uint64_t) + 1> raw_str;
  for (auto i = 0; i < sizeof (uint64_t); ++i)
    {
      raw_str[sizeof (uint64_t) - i - 1] = reg_copy & 0xff;
      reg_copy = reg_copy >> 8;
    }

  /* Null terminate the string */
  raw_str.back () = '\0';

  /* Advance past any '\0' */
  auto data_ptr = raw_str.begin ();
  while (*data_ptr == '\0' && data_ptr != raw_str.end ())
    ++data_ptr;

  /* Early return if we have an empty string */
  if (data_ptr == raw_str.end ())
    return dwarf2_reg;

  /* Create the possible ptx virtual register string */
  std::string ptx_reg_str{ data_ptr };

  /* Is this a ptx virtual register? */
  if (ptx_reg_str[0] == '%')
    {
      /* Check to see if we have already seen this ptx virtual register
       * string
       */
      auto elem = cuda_ptx_virtual_map.find (dwarf2_reg);
      if (elem != cuda_ptx_virtual_map.end ())
        {
          dwarf2_reg = (uint64_t)elem->second;
        }
      else
        {
          /* First time encountering this string - store it. */
          auto it = cuda_ptx_virtual_str.emplace (cuda_ptx_virtual_str.end (),
                                                  ptx_reg_str);
          /* Get the index of the ptx virtual register string in the vector.
           */
          int idx = std::distance (cuda_ptx_virtual_str.begin (), it);
          /* Enforce that we don't overflow the allowable range. */
          gdb_assert (idx < CUDA_PTX_VIRTUAL_TAG);
          /* Create the tagged variant of the idx */
          int tag = CUDA_PTX_VIRTUAL_ID (idx);
          /* Add this to the map and return. */
          cuda_ptx_virtual_map[dwarf2_reg] = tag;
          dwarf2_reg = (uint64_t)tag;
        }
    }

  return dwarf2_reg;
}

/* Check if a dwarf2 register is encoded as an ascii register string.
 * We look at the encoding to determine if it is.
 * Example:
 *                      as char[4]    as uint32_t
 *   device reg R4 :        "\04R"     0x00257234
 *   host reg  4    : "\0\0\0\004"     0x00000004
 */
uint64_t
cuda_check_dwarf2_reg_ascii_encoded_register (struct gdbarch *gdbarch,
					      uint64_t dwarf2_reg)
{
  /* Ensure that the dwarf2_reg is greater than one character. Anything less
   * would be an invalid ascii string. */
  if (dwarf2_reg <= 0xff)
    return dwarf2_reg;

  /* Convert the uleb128 to a string. The order of characters
   * has to be reversed in order to be read as a standard string. */
  uint64_t reg_copy = dwarf2_reg;
  std::array<char, sizeof (uint64_t) + 1> raw_str;
  for (auto i = 0; i < sizeof (uint64_t); ++i)
    {
      raw_str[sizeof (uint64_t) - i - 1] = reg_copy & 0xff;
      reg_copy = reg_copy >> 8;
    }

  /* Null terminate the string */
  raw_str.back () = '\0';

  /* Advance past any '\0' */
  auto data_ptr = raw_str.begin ();
  while (*data_ptr == '\0' && data_ptr != raw_str.end ())
    ++data_ptr;

  /* Early return if we have an empty string */
  if (data_ptr == raw_str.end ())
    return dwarf2_reg;

  /* Create the possible register string */
  std::string reg_str{ data_ptr };

  /* Is this an ascii encoded register? Today the compiler
   * only supports regular registers which start with 'R'. */
  if (reg_str[0] == 'R')
    {
      /* Convert to uint64_t and store in dwarf2_reg */
      dwarf2_reg = user_reg_map_name_to_regnum (gdbarch, reg_str.c_str (),
						reg_str.length ());
    }

  return dwarf2_reg;
}

// Return the name of register REGNUM.
static const char *
cuda_register_name (struct gdbarch *gdbarch, int regnum)
{
  static char buf[CUDA_GDB_TMP_BUF_SIZE];

  // Ignore registers not supported by this device
  buf[0] = '\0';

  gdb_assert (cuda_current_focus::isDevice ());
  const auto &c = cuda_current_focus::get ().physical ();

  // Explicitly check for RZ
  if (cuda_zero_register_p (gdbarch, regnum))
    return "RZ";

  // General purpose SASS registers
  if (cuda_regular_register_p (gdbarch, regnum))
    {
      const auto tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
      snprintf (buf, sizeof (buf), "R%d", regnum - tdep->first_regnum);
      return buf;
    }

  // The PC register
  if (cuda_pc_regnum_p (gdbarch, regnum))
    return "pc";

  // Error PC register
  if (cuda_error_pc_regnum_p (gdbarch, regnum))
    return "errorpc";

  // CC register
  if (cuda_cc_regnum_p (gdbarch, regnum))
    return "CC";

  // Invalid register
  if (cuda_invalid_regnum_p (gdbarch, regnum))
    return "(dummy internal register)";

  // The special CUDA register: stored in the regmap.
  if (cuda_special_regnum_p (gdbarch, regnum))
    {
      const auto regmap = regmap_get_search_result ();
      cuda_special_register_name (regmap, buf, sizeof (buf));
      return buf;
    }

  // Predicate registers
  if (cuda_pred_regnum_p (gdbarch, regnum))
    {
      const auto tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
      snprintf (buf, sizeof (buf), "P%d", regnum - tdep->first_pred_regnum);
      return buf;
    }

  // Uniform registers
  const auto num_uregs = cuda_state::device_get_num_uregisters (c.dev ());
  if (num_uregs > 0)
    {
      // Uniform zero register
      if (cuda_uniform_zero_register_p (gdbarch, regnum))
	return "URZ";

      // Uniform scalar registers
      if (cuda_uregnum_p (gdbarch, regnum))
	{
	  const auto tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
	  const auto uregnum = regnum - tdep->first_uregnum;
	  if (uregnum < num_uregs)
	    snprintf (buf, sizeof (buf), "UR%d", uregnum);
	  return buf;
	}

      // Uniform Predicate registers
      if (cuda_upred_regnum_p (gdbarch, regnum))
	{
	  const auto tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
	  snprintf (buf, sizeof (buf), "UP%d",
		    regnum - tdep->first_upred_regnum);
	  return buf;
	}
    }

  // (everything else) - zero length string
  return "";
}

static struct type *
cuda_register_type (struct gdbarch *gdbarch, int regnum)
{
  if (cuda_special_regnum_p (gdbarch, regnum))
    return builtin_type (gdbarch)->builtin_uint64;

  if (cuda_pc_regnum_p (gdbarch, regnum)
      || cuda_error_pc_regnum_p (gdbarch, regnum))
    return builtin_type (gdbarch)->builtin_func_ptr;

  return builtin_type (gdbarch)->builtin_uint32;
}

static regmap_t
cuda_get_physical_register (const char *reg_name)
{
  gdb_assert (cuda_current_focus::isDevice ());

  auto frame = get_selected_frame (NULL);
  auto virt_addr = get_frame_pc (frame);

  auto kernel = cuda_current_focus::get ().logical ().kernel ();
  auto module = kernel_get_module (kernel);

  auto symbol = find_pc_function ((CORE_ADDR)virt_addr);
  if (symbol)
    {
      CORE_ADDR func_start;

      find_pc_partial_function (virt_addr, NULL, &func_start, NULL);
      auto func_name = symbol->linkage_name ();
      auto addr = virt_addr - func_start;
      return regmap_table_search (module->objfile (), func_name, reg_name, addr);
    }

  return nullptr;
}

/*
 * Check whether the gdb architecture data structure knows of this register;
 * e.g., it is "Recognized by gdbarch."  If not, then it's probably a cuda
 * register and return -1.
 */
static int
cuda_decode_if_recognized (struct gdbarch *gdbarch, ULONGEST reg)
{
  int32_t decoded_reg;
  const int max_regs
      = gdbarch_num_regs (gdbarch) + gdbarch_num_pseudo_regs (gdbarch);

  /* The register is already decoded and ULONGEST is unsigned value */
  if (reg <= max_regs)
    return reg;

  /* The register is encoded with its register class. */
  if (!cuda_decode_physical_register (reg, &decoded_reg))
    return decoded_reg;

  /* Unrecognized register (probably cuda) */
  return -1;
}

/*
 * Return the regmap after decoding the register into a string
 */
static regmap_t
cuda_reg_string_to_regmap (struct gdbarch *gdbarch, int reg)
{
  if (reg < CUDA_PTX_VIRTUAL_TAG)
    return NULL;

  int idx = CUDA_PTX_VIRTUAL_IDX (reg);
  if ((idx < 0) || (idx > (cuda_ptx_virtual_str.size () - 1)))
    return NULL;

  return cuda_get_physical_register (cuda_ptx_virtual_str[idx].c_str ());
}

/*
 * Convert a CUDA DWARF register into a physical register index
 */
static int
cuda_dwarf2_reg_to_regnum (struct gdbarch *gdbarch, int reg)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  int32_t regno;
  uint32_t num_regs;
  regmap_t regmap;

  /* Check if the register is already decoded or encoded with reg class */
  regno = cuda_decode_if_recognized (gdbarch, reg);
  if (regno != -1)
    return regno;

  /* At this point, we know that the register is encoded as PTX register
   * string
   */
  regmap = cuda_reg_string_to_regmap (gdbarch, reg);
  if (!regmap)
    return -1;

  num_regs = regmap_get_num_entries (regmap);

  /* If we found a single SASS register, then we let cuda-gdb handle it
     normally. */
  if (num_regs == 1)
    {
      if (regmap_get_class (regmap, 0) == REG_CLASS_REG_FULL)
        {
          regno = regmap_get_register (regmap, 0);
          return regno;
        }
      else if (regmap_get_class (regmap, 0) == REG_CLASS_UREG_FULL)
        {
          regno = regmap_get_uregister (regmap, 0);
          return regno;
        }
    }

  /* Every situation that requires us to store data that cannot be
     represented as a single register index (regno). We keep hold of the data
     until the value is to be fetched. */
  if (cuda_special_register_p (regmap))
    {
      regno = tdep->special_regnum;
      return regno;
    }

  /* If no physical register was found or the register mapping is not
     useable, returns an invalid regnum to indicate it has been optimized
     out. */
  return tdep->invalid_lo_regnum;
}

/* Return the extrapolated gdb regnum.  This should only be called if no
 * other regnum can be found.
 *
 * This assumes that the extrapolated value is the second in the register
 * mapping.
 *
 * The user should be aware that extrapolated values are not 100% accurate.
 * The help message associated to "set cuda  value_extrapolation" mentions
 * this.
 */
int
cuda_reg_to_regnum_extrapolated (struct gdbarch *gdbarch, int reg)
{
  int regno;
  regmap_t regmap;

  if (!gdbarch)
    return -1;

  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);

  /* If this is a register that we recognize, use that */
  regno = cuda_decode_if_recognized (gdbarch, reg);
  if (regno != -1)
    return regno;

  /* Unrecognized, so turn the register into string and query with that */
  regmap = cuda_reg_string_to_regmap (gdbarch, reg);
  if (!regmap)
    return tdep->invalid_lo_regnum;

  /* We only deal with extrapolated values in this function */
  if (!regmap_is_extrapolated (regmap))
    return tdep->invalid_lo_regnum;

  /* Locate the assumed extrapolated value */
  if (regmap_get_num_entries (regmap) == 2)
    {
      if (regmap_get_class (regmap, 1) == REG_CLASS_REG_FULL)
        return (int)(regmap_get_register (regmap, 1) + tdep->first_regnum);

      if (regmap_get_class (regmap, 1) == REG_CLASS_UREG_FULL)
        return (int)(regmap_get_uregister (regmap, 1) + tdep->first_uregnum);
    }

  /* This will be treated as an "optimized out" register */
  return tdep->invalid_lo_regnum;
}

int
cuda_reg_to_regnum (struct gdbarch *gdbarch, int reg)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);

  if (REGMAP_CLASS (reg) == REG_CLASS_REG_FULL)
    return (REGMAP_REG (reg) == CUDA_REG_ZERO_REGISTER)
	       ? tdep->zero_regnum
	       : REGMAP_REG (reg) + tdep->first_regnum;

  if (REGMAP_CLASS (reg) == REG_CLASS_REG_PRED)
    return REGMAP_REG (reg) + tdep->first_pred_regnum;

  if (REGMAP_CLASS (reg) == REG_CLASS_UREG_FULL)
    return (REGMAP_REG (reg) == CUDA_UREG_ZERO_REGISTER)
	       ? tdep->zero_uregnum
	       : REGMAP_REG (reg) + tdep->first_uregnum;

  if (REGMAP_CLASS (reg) == REG_CLASS_UREG_PRED)
    return REGMAP_REG (reg) + tdep->first_upred_regnum;

  error (_ ("Invalid register."));

  return -1;
}

int cuda_regnum_to_reg (struct gdbarch * gdbarch, uint32_t regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);

  if (cuda_zero_register_p (gdbarch, regnum))
    return CUDA_REG_CLASS_AND_REGNO (REG_CLASS_REG_FULL,
				     CUDA_REG_ZERO_REGISTER);

  if (cuda_regular_register_p (gdbarch, regnum))
    return CUDA_REG_CLASS_AND_REGNO (REG_CLASS_REG_FULL,
				     regnum - tdep->first_regnum);

  if (cuda_pred_regnum_p (gdbarch, regnum))
    return CUDA_REG_CLASS_AND_REGNO (REG_CLASS_REG_PRED,
				     regnum - tdep->first_pred_regnum);

  if (cuda_uniform_zero_register_p (gdbarch, regnum))
    return CUDA_REG_CLASS_AND_REGNO (REG_CLASS_UREG_FULL,
				     CUDA_UREG_ZERO_REGISTER);

  if (cuda_uregnum_p (gdbarch, regnum))
    return CUDA_REG_CLASS_AND_REGNO (REG_CLASS_UREG_FULL,
				     regnum - tdep->first_uregnum);

  if (cuda_upred_regnum_p (gdbarch, regnum))
    return CUDA_REG_CLASS_AND_REGNO (REG_CLASS_UREG_PRED,
				     regnum - tdep->first_upred_regnum);

  error (_ ("Invalid register."));

  return CUDA_REG_CLASS_AND_REGNO (REG_CLASS_INVALID, 0);
}

void
cuda_register_read (struct gdbarch *gdbarch, struct regcache *regcache,
	      	    int regnum)
{
  gdb_assert (gdbarch);

  gdb_assert (cuda_current_focus::isDevice ());
  const auto &c = cuda_current_focus::get ().physical ();

  // PC register
  if (cuda_pc_regnum_p (gdbarch, regnum))
    {
      const uint64_t pc = cuda_state::lane_get_pc (c.dev (), c.sm (),
						   c.wp (), c.ln ());
      regcache->raw_supply (regnum, &pc);
      return;
    }

  // ErrorPC register
  if (cuda_error_pc_regnum_p (gdbarch, regnum))
    {
      if (cuda_state::warp_has_error_pc (c.dev (), c.sm (), c.wp ()))
	{
	  const uint64_t error_pc
	      = cuda_state::warp_get_error_pc (c.dev (), c.sm (), c.wp ());
	  regcache->raw_supply (regnum, &error_pc);
	}
      else
	regcache->raw_supply (regnum, NULL);
      return;
    }

  // CC register
  if (cuda_cc_regnum_p (gdbarch, regnum))
    {
      const uint32_t cc = cuda_state::lane_get_cc_register (c.dev (), c.sm (),
							    c.wp (), c.ln ());
      regcache->raw_supply (regnum, &cc);
      return;
    }

  // Single SASS register
  if (cuda_zero_register_p (gdbarch, regnum))
    {
      const uint32_t zero = 0;
      regcache->raw_supply (regnum, &zero);
      return;
    }

  if (cuda_regular_register_p (gdbarch, regnum))
    {
      const auto tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
      const uint32_t regval = cuda_state::lane_get_register (
	  c.dev (), c.sm (), c.wp (), c.ln (), regnum - tdep->first_regnum);
      regcache->raw_supply (regnum, &regval);
      return;
    }

  // Predicate register
  if (cuda_pred_regnum_p (gdbarch, regnum))
    {
      const auto tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
      const uint32_t pred = cuda_state::lane_get_predicate (
	  c.dev (), c.sm (), c.wp (), c.ln (),
	  regnum - tdep->first_pred_regnum);
      regcache->raw_supply (regnum, &pred);
      return;
    }

  // Uniform registers
  if (cuda_state::device_get_num_uregisters (c.dev ()) > 0)
    {
      if (cuda_uniform_zero_register_p (gdbarch, regnum))
	{
	  uint32_t zero = 0;
	  regcache->raw_supply (regnum, &zero);
	  return;
	}

      if (cuda_uregnum_p (gdbarch, regnum))
	{
	  const auto *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
	  const int uregnum = regnum - tdep->first_uregnum;
	  if (uregnum < cuda_state::device_get_num_uregisters (c.dev ()))
	    {
	      const uint32_t ureg = cuda_state::warp_get_uregister (
		  c.dev (), c.sm (), c.wp (), uregnum);
	      regcache->raw_supply (regnum, &ureg);
	    }
	  else
	    regcache->raw_supply (regnum, NULL);
	  return;
	}

      if (cuda_upred_regnum_p (gdbarch, regnum))
	{
	  const auto tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
	  const uint32_t upred = cuda_state::warp_get_upredicate (
	      c.dev (), c.sm (), c.wp (), regnum - tdep->first_upred_regnum);
	  regcache->raw_supply (regnum, &upred);
	  return;
	}
    }

  // Invalid register
  regcache->raw_supply (regnum, NULL);
}

enum register_status
cuda_pseudo_register_read (struct gdbarch *gdbarch,
			   readable_regcache *regcache, int regnum,
			   gdb_byte *buf)
{
  gdb_assert (cuda_current_focus::isDevice ());

  /* Any combination of SASS register, SP + offset, LMEM offset locations */
  if (cuda_special_regnum_p (gdbarch, regnum))
    {
      const auto regmap = regmap_get_search_result ();
      cuda_special_register_read (regmap, buf);
      return REG_VALID;
    }

  *(uint32_t *)buf = 0;
  return REG_UNAVAILABLE;
}

void
cuda_register_write (struct gdbarch *gdbarch, struct regcache *regcache,
		     int regnum, const gdb_byte *buf)
{
  gdb_assert (cuda_current_focus::isDevice ());
  const auto &c = cuda_current_focus::get ().physical ();

  // Filter out the registers that can't be written to
  if (cuda_invalid_regnum_p (gdbarch, regnum))
    error (_ ("Invalid register."));

  if (cuda_pc_regnum_p (gdbarch, regnum))
    error (_ ("PC register not writable"));

  if (cuda_error_pc_regnum_p (gdbarch, regnum))
    error (_ ("Error PC register not writable"));

  if (regnum == CUDA_REG_ZERO_REGISTER)
    error ("Cannot write to RZ register");

  if (cuda_uniform_zero_register_p (gdbarch, regnum))
    error ("Cannot write to URZ register");

  // CC register
  if (cuda_cc_regnum_p (gdbarch, regnum))
    {
      cuda_state::lane_set_cc_register (c.dev (), c.sm (), c.wp (), c.ln (),
					*(uint32_t *)buf);
      return;
    }

  // single SASS register
  if (cuda_regular_register_p (gdbarch, regnum))
    {
      cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
      cuda_state::lane_set_register (c.dev (), c.sm (), c.wp (), c.ln (),
				     regnum - tdep->first_regnum,
				     *(uint32_t *)buf);
      return;
    }

  // Predicate register
  if (cuda_pred_regnum_p (gdbarch, regnum))
    {
      cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
      cuda_state::lane_set_predicate (c.dev (), c.sm (), c.wp (), c.ln (),
				      regnum - tdep->first_pred_regnum,
				      *(uint32_t *)buf);
      return;
    }

  // Uniform registers (Turing+)
  if (cuda_state::device_get_num_uregisters (c.dev ()) > 0)
    {
      cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);

      // Uniform predicate register
      if (cuda_upred_regnum_p (gdbarch, regnum))
	{
	  cuda_state::warp_set_upredicate (c.dev (), c.sm (), c.wp (),
					   regnum - tdep->first_upred_regnum,
					   *(uint32_t *)buf);
	  return;
	}

      // Uniform registers
      if (cuda_uregnum_p (gdbarch, regnum))
	{
	  const auto uregnum = regnum - tdep->first_uregnum;
	  if (uregnum >= cuda_state::device_get_num_uregisters (c.dev ()))
	    error ("Register UR%u is out of range", uregnum);
	  cuda_state::warp_set_uregister (c.dev (), c.sm (), c.wp (), uregnum,
					  *(uint32_t *)buf);
	  return;
	}
    }

  error (_ ("Invalid register write"));
}

static void
cuda_pseudo_register_write (struct gdbarch *gdbarch, struct regcache *regcache,
			    int regnum, const gdb_byte *buf)
{
  // Invalid register
  if (cuda_invalid_regnum_p (gdbarch, regnum))
    {
      error (_ ("Invalid pseudo register."));
      return;
    }

  // Any combination of SASS register, SP + offset, LMEM offset locations
  if (cuda_special_regnum_p (gdbarch, regnum))
    {
      const auto regmap = regmap_get_search_result ();
      cuda_special_register_write (regmap, buf);
      return;
    }
  error ("Unknown pseudo register %u", regnum);
}

static int
cuda_register_reggroup_p (struct gdbarch *gdbarch, int regnum,
			  const struct reggroup *group)
{
  /* Do not include special and invalid registers in any group */
  if (cuda_invalid_regnum_p (gdbarch, regnum)
      || cuda_special_regnum_p (gdbarch, regnum))
    return 0;

  /* Include predicates and CC register in special and all register groups */
  if ((cuda_pred_regnum_p (gdbarch, regnum)
       || cuda_cc_regnum_p (gdbarch, regnum))
      && (group == system_reggroup || group == all_reggroup))
    return 1;

  return default_register_reggroup_p (gdbarch, regnum, group);
}

static int
cuda_print_insn (bfd_vma pc, disassemble_info *info)
{
  if (!cuda_current_focus::isDevice ())
    return 1;

  /* If this isn't a device address, don't bother */
  if (!cuda_is_device_code_address (pc))
    return 1;

  /* decode the instruction at the pc */
  auto kernel = cuda_current_focus::get ().logical ().kernel ();
  gdb_assert (kernel);

  auto module = kernel_get_module (kernel);
  gdb_assert (module);

  auto disassembler = module->disassembler ();
  gdb_assert (disassembler);

  const uint32_t inst_size = disassembler->insn_size ();
  auto inst = disassembler->disassemble_instruction (pc);
  if (!inst)
    info->fprintf_func (info->stream,
			"Cannot disassemble instruction at pc 0x%lx", pc);
  else
    info->fprintf_func (info->stream, "%s", inst->to_string ().c_str ());

  return inst_size;
}

bool
cuda_is_device_code_address (CORE_ADDR addr)
{
  struct obj_section *osect = NULL;
  asection *section = NULL;
  bool found_in_cpu_symtab = false;
  bool is_cuda_addr = false;

  /* Zero and (CORE_ADDR)-1 are CPU addresses */
  if (addr == 0 || addr == (CORE_ADDR)-1LL)
    return false;

  /*Iterate over all ELF sections */
  for (objfile *objfile : current_program_space->objfiles ())
    ALL_OBJFILE_OSECTIONS (objfile, osect)
    {
      section = osect->the_bfd_section;
      if (!section)
	continue;
      /* Check if addr belongs to CUDA ELF */
      if (objfile->cuda_objfile)
	{
	  /* Skip sections that do not have code */
	  if (!(section->flags & SEC_CODE))
	    continue;
	  /* skip sections that are unrelocated */
	  if (section->vma == 0)
	    continue;
	  if (section->vma > addr || section->vma + section->size <= addr)
	    continue;
	  return true;
	}

      /* Check if address belongs to one of the host ELFs */
      if (osect->addr () <= addr && osect->endaddr () > addr)
	found_in_cpu_symtab = true;
    }

  /* If address was found on CPU and wasn't found on any of the CUDA ELFs -
   * return false */
  if (found_in_cpu_symtab)
    return false;

  /* Fallback to backend API call */
  cuda_debugapi::is_device_code_address ((uint64_t)addr, &is_cuda_addr);
  return is_cuda_addr;
}

/* Returns true if obfd points to a CUDA ELF object file
   (checked by machine type).  Otherwise, returns false. */
bool
cuda_is_bfd_cuda (bfd *obfd)
{
  /* elf_header is a single element array in elf_obj_tdata,
     i.e. it can't be null if elf_obj_data is not null */
  return (obfd && elf_tdata (obfd)
	  && elf_tdata (obfd)->elf_header[0].e_machine == EM_CUDA);
}

/* Return the CUDA ELF ABI version.  If obfd points to a CUDA ELF
   object file and contains a valid CUDA ELF ABI version, it stores
   the ABI version in abi_version and returns true.  Otherwise, it
   returns false. */
bool
cuda_get_bfd_abi_version (bfd *obfd, unsigned int *abi_version)
{
  if (!cuda_is_bfd_cuda (obfd) || !abi_version)
    return false;

  // Just warn about unknown ABI versions
  unsigned int abiv = (elf_elfheader (obfd)->e_ident[EI_ABIVERSION]);

  static bool abi_warning_given = false;
  if (!abi_warning_given && (abiv > CUDA_ELFOSABIV_LATEST))
    {
      warning ("CUDA ELF Image contains unknown ABI version: %d", abiv);
      abi_warning_given = true;
    }

  *abi_version = abiv;

  return true;
}

/* Returns true if obfd points to a CUDA ELF object file that was
   compiled against the call frame ABI (the abi version is equal
   to CUDA_ELFOSABIV_ABI).  Returns false otherwise. */
bool
cuda_is_bfd_version_call_abi (bfd *obfd)
{
  unsigned int cuda_abi_version;
  bool is_cuda_abi;

  is_cuda_abi = cuda_get_bfd_abi_version (obfd, &cuda_abi_version);
  return (is_cuda_abi && cuda_abi_version >= CUDA_ELFOSABIV_ABI);
}

bool
cuda_current_active_elf_image_uses_abi (void)
{
  if (!cuda_state::current_context ())
    return false;

  if (!cuda_current_focus::isDevice ())
    return false;

  auto kernel = cuda_current_focus::get ().logical ().kernel ();
  gdb_assert (kernel);
  
  auto module = kernel_get_module (kernel);
  gdb_assert (module);

  gdb_assert (module->loaded ());
  return module->uses_abi ();
}

/* CUDA - breakpoints */
/* Like breakpoint_address_match, but gdbarch is a parameter. Required to
   evaluate gdbarch_has_global_breakpoints (gdbarch) in the right context. */
int
cuda_breakpoint_address_match (struct gdbarch *gdbarch,
			       const address_space *aspace1, CORE_ADDR addr1,
			       const address_space *aspace2, CORE_ADDR addr2)
{
  return ((gdbarch_has_global_breakpoints (gdbarch) || aspace1 == aspace2)
	  && addr1 == addr2);
}

CORE_ADDR
cuda_get_symbol_address (const char *name)
{
  struct bound_minimal_symbol msym = lookup_minimal_symbol (name, NULL, NULL);

  if (msym.minsym)
    return msym.minsym->value_address (msym.objfile);

  return 0;
}

uint64_t
cuda_get_last_driver_api_error_code (void)
{
  CORE_ADDR error_code_addr;
  uint64_t res;

  error_code_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_CODE));
  if (!error_code_addr)
    {
      /* This should never happen. If we hit the breakpoint we should have
       * the symbol */
      error (_ ("Cannot retrieve the last driver API error code.\n"));
    }

  target_read_memory (error_code_addr, (gdb_byte *)&res, sizeof (uint64_t));
  return res;
}

static uint64_t
cuda_get_last_driver_api_error_func_name_size (void)
{
  CORE_ADDR error_func_name_size_addr;
  uint64_t res;

  error_func_name_size_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_SIZE));
  if (!error_func_name_size_addr)
    {
      /* This should never happen. If we hit the breakpoint we should have
       * the symbol */
      error (_ ("Cannot determine the address of the last driver API error "
		"function name size.\n"));
    }

  target_read_memory (error_func_name_size_addr, (gdb_byte *)&res,
		      sizeof (uint64_t));
  return res;
}

void
cuda_get_last_driver_api_error_func_name (char **name)
{
  CORE_ADDR error_func_name_core_addr;
  uint64_t error_func_name_addr;
  char *func_name = NULL;
  uint32_t size = 0U;

  *name = NULL;
  size = cuda_get_last_driver_api_error_func_name_size ();
  if (!size)
    {
      /* This should never happen. If we hit the breakpoint we should have
       * the symbol */
      warning (_ ("Cannot retrieve the last driver API error function name "
		  "size.\n"));
      return;
    }

  func_name = (char *)xcalloc (sizeof *func_name, size);
  if (!func_name)
    {
      /* Buffer for function name string should be created successfully  */
      error (
	  _ ("Cannot allocate memory to save the reported function name.\n"));
    }

  error_func_name_core_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_ADDR));
  if (!error_func_name_core_addr)
    {
      /* This should never happen. If we hit the breakpoint we should have
       * the symbol */
      error (_ ("Cannot retrieve the last driver API error function name "
		"addr.\n"));
    }

  target_read_memory (error_func_name_core_addr,
		      (gdb_byte *)&error_func_name_addr, sizeof (uint64_t));
  target_read_memory (error_func_name_addr, (gdb_byte *)func_name, size);
  if (!func_name[0])
    {
      /* This should never happen. If we hit the breakpoint we should have
       * the symbol */
      error (_ ("Cannot retrieve the last driver API error function name.\n"));
    }
  *name = func_name;
}

bool
cuda_get_last_driver_api_error_source_name (std::string &source)
{
  if (cuda_debugapi::api_version ().m_revision < 147)
    {
      return false;
    }

  CORE_ADDR error_code_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE));
  if (!error_code_addr)
    {
      /* UMD revision does not match Debug API revision */
      cuda_trace (_ ("Driver API error source symbol is unavailable; "
        "API Version doesn't match actual UMD version."));
      return false;
    }

  uint32_t res;
  target_read_memory (error_code_addr, (gdb_byte *)&res, sizeof (res));
  
  switch (res)
    {
    case CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE_DRIVER:
      source = "Driver";
      break;
    case CUDBG_REPORTED_DRIVER_API_ERROR_SOURCE_RUNTIME:
      source = "Runtime";
      break;
    default:
      cuda_trace (_ ("Error source reporting unsupported, got: 0x%x."), res);
      return false;
    }

  return true;
}

static bool
cuda_get_last_driver_api_error_name_size (uint64_t &size)
{
  CORE_ADDR error_name_size_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_NAME_SIZE));
  if (!error_name_size_addr)
    {
      /* UMD revision does not match Debug API revision */
      cuda_trace (_ ("Driver API error source symbol is unavailable; "
        "API Version doesn't match actual UMD version."));
      return false;
    }

  target_read_memory (error_name_size_addr, (gdb_byte *)&size,
		      sizeof (size));
  return true;
}

bool
cuda_get_last_driver_api_error_name (std::string &name)
{
  if (cuda_debugapi::api_version ().m_revision < 147)
    {
      return false;
    }

  using char_t = char;

  uint64_t size;
  bool success = cuda_get_last_driver_api_error_name_size (size);
  if (!success)
    {
      cuda_trace (_ ("Cannot retrieve the last driver API error name size."));
      return false;
    }
  if (!size)
    {
      cuda_trace (_ ("Last Driver API error name is empty."));
      return false;
    }

  gdb::unique_xmalloc_ptr<char_t> buffer((char_t *)xcalloc (sizeof (char_t), size));
  if (!buffer)
    {
      /* Buffer for name string should be created successfully  */
      error (_ ("Cannot allocate memory to save the reported error name."));
    }

  CORE_ADDR error_name_core_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_NAME_ADDR));
  if (!error_name_core_addr)
    {
      /* This should never happen. If UMD revision was lower, 
      cuda_get_last_driver_api_error_name_size would have failed */
      error (_ ("Cannot retrieve the last driver API error name addr."));
    }

  uint64_t error_error_name_addr;
  target_read_memory (error_name_core_addr,
		      (gdb_byte *)&error_error_name_addr, sizeof (uint64_t));
  if (!error_name_core_addr)
    {
      cuda_trace (_ ("Last Driver API error name is null."));
      return false;
    }

  target_read_memory (error_error_name_addr, (gdb_byte *)buffer.get (), size);
  name = buffer.get ();

  return true;
}

static bool
cuda_get_last_driver_api_error_string_size (uint64_t &size)
{
  CORE_ADDR error_string_size_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_STRING_SIZE));
  if (!error_string_size_addr)
    {
      /* UMD revision does not match Debug API revision */
      cuda_trace (_ ("Driver API error string size symbol is unavailable; "
        "API Version doesn't match actual UMD version"));
      return false;
    }

  target_read_memory (error_string_size_addr, (gdb_byte *)&size,
		      sizeof (size));
  return true;
}

bool
cuda_get_last_driver_api_error_string (std::string &string)
{
  if (cuda_debugapi::api_version ().m_revision < 147)
    {
      return false;
    }

  using char_t = char;

  uint64_t size;
  bool success = cuda_get_last_driver_api_error_string_size (size);
  if (!success)
    {
      cuda_trace (_ ("Cannot retrieve the last driver API error string size."));
      return false;
    }
  if (!size)
    {
      cuda_trace (_ ("Last Driver API error string is empty."));
      return false;
    }

  gdb::unique_xmalloc_ptr<char_t> buffer((char_t *)xcalloc (sizeof (char_t), size));
  if (!buffer)
    {
      /* Buffer for error string should be created successfully  */
      error (_ ("Cannot allocate memory to save the reported error string."));
    }

  CORE_ADDR error_string_core_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_API_ERROR_STRING_ADDR));
  if (!error_string_core_addr)
    {
      /* This should never happen. If we hit the breakpoint we should have
       * the symbol */
      error (_ ("Cannot retrieve the last driver API error string addr."));
    }

  uint64_t error_error_string_addr;
  target_read_memory (error_string_core_addr,
		      (gdb_byte *)&error_error_string_addr, sizeof (uint64_t));
  if (!error_string_core_addr)
    {
      /* The backend does not report error names */
      cuda_trace (_ ("Last Driver API error string is null."));
      return false;
    }

  target_read_memory (error_error_string_addr, (gdb_byte *)buffer.get (), size);
  string = buffer.get ();

  return true;
}

uint64_t
cuda_get_last_driver_internal_error_code (void)
{
  CORE_ADDR error_code_addr;
  uint64_t res;

  error_code_addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE));
  if (!error_code_addr)
    {
      /* This should never happen. If we hit the breakpoint we should have
       * the symbol */
      error (_ ("Cannot retrieve the last driver internal error code.\n"));
    }

  target_read_memory (error_code_addr, (gdb_byte *)&res, sizeof (uint64_t));
  return res;
}

static void
cuda_sigpipe_handler (int signo)
{
  fprintf (stderr, "Error: A SIGPIPE has been received, this is likely due to "
		   "a crash from the CUDA backend.\n");
  fflush (stderr);

  cuda_cleanup ();
  exit (1);
}

void
cuda_signals_initialize (void)
{
  signal (SIGSEGV, segv_handler);
  signal (SIGPIPE, cuda_sigpipe_handler);
}

void
cuda_cleanup (void)
{
  cuda_trace ("cuda_cleanup");

  registers_changed ();
  cuda_auto_breakpoints_cleanup ();
  cuda_state::cleanup_breakpoints ();
  cuda_cleanup_cudart_symbols ();
  cuda_current_focus::invalidate ();
  cuda_state::finalize ();
  cuda_sstep_reset (false);
  cuda_set_device_launch_used (false);

  cuda_debugapi::finalize ();
  /* Notification reset must be called after notification thread has
   * been terminated, which is done as part of cuda_api_finalize() call. */
  cuda_notification_reset ();

  inferior_in_debug_mode = false;
  cuda_initialized = false;
  current_inferior ()->cuda_initialized = false;
}

void
cuda_final_cleanup (void *unused)
{
  if (cuda_initialized)
    cuda_debugapi::finalize ();
}

/* Initialize the CUDA debugger API and collect the static data about
   the devices. Once per application run. */
void
cuda_initialize (void)
{
  CORE_ADDR useExtDebuggerAddr = 0;
  uint32_t useExtDebugger = 0;

  // If the inferior has done this once already, just return
  if (current_inferior ()->cuda_initialized)
    {
      cuda_trace ("cuda_initialize: skipping as inferior already initialized");
      return;
    }

  cuda_initialized = !cuda_debugapi::initialize ();
  if (cuda_initialized)
    {
      cuda_trace ("cuda_initialize: initializing");

      // Mark the inferior as CUDA initialized when all the steps have
      // happened. This may require several calls to get this far. This is
      // sticky for the lifetime of the inferior
      current_inferior ()->cuda_initialized = true;

      cuda_state::initialize ();
      useExtDebuggerAddr
          = cuda_get_symbol_address (_STRING_ (CUDBG_USE_EXTERNAL_DEBUGGER));

      if (useExtDebuggerAddr)
        {
          auto res = target_read_memory (useExtDebuggerAddr,
                                         (gdb_byte *)&useExtDebugger,
                                         sizeof (useExtDebugger));
          // In case of error, make sure useExtDebugger is left unset
          if (res != 0)
            cuda_trace ("cuda_initialize: read of useExtDebugger failed: %d",
                        res);

          /* Value can either be 0 or 1, anything else is likely an invalid
           * read. */
          if (useExtDebugger > 1)
            error (_ ("Invalid value read for %s: %u\n"),
                   _STRING_ (CUDBG_USE_EXTERNAL_DEBUGGER), useExtDebugger);

          if (!useExtDebugger)
            printf ("Running on legacy stack.\n");
        }
    }
}

static bool
cuda_get_cudbg_api (void)
{
  CUDBGAPI api = nullptr;

  CUDBGResult res
      = cudbgGetAPI (CUDBG_API_VERSION_MAJOR, CUDBG_API_VERSION_MINOR,
		     CUDBG_API_VERSION_REVISION, &api);

  if (res == CUDBG_SUCCESS)
    cuda_debugapi::set_api (api);
  else
    cuda_debugapi::print_get_api_error (res);

  return (res != CUDBG_SUCCESS);
}

static void
kill_or_detach ()
{
  inferior *inf = current_inferior ();

  /* Leave core files alone.  */
  if (target_has_execution ())
    {
      if (inf->attach_flag)
        target_detach (inf, 0);
      else {
        target_kill ();
      }
    }
}

/* Can be removed when exposed through cudadebugger.h */
#define CUDBG_INJECTION_PATH_SIZE 4096
static void
cuda_initialize_injection ()
{
  CORE_ADDR injectionPathAddr;
  char *injectionPathEnv;
  void *injectionLib;
  const char *forceLegacy;

  forceLegacy = getenv ("CUDBG_USE_LEGACY_DEBUGGER");
  injectionPathEnv = getenv ("CUDBG_INJECTION_PATH");
  if ((forceLegacy && forceLegacy[0] == '1') || !injectionPathEnv)
    {
      /* No injection - cuda-gdb is the API client */
      return;
    }

  if (strlen (injectionPathEnv) >= CUDBG_INJECTION_PATH_SIZE)
    {
      kill_or_detach ();
      error (_("CUDBG_INJECTION_PATH must be no longer than %d: %s is %zd"),
             CUDBG_INJECTION_PATH_SIZE - 1, injectionPathEnv, strlen (injectionPathEnv));
    }

  injectionLib = dlopen (injectionPathEnv, RTLD_LAZY);

  if (injectionLib == NULL)
    {
      /* kill_or_detach() might clear dlerror, so copy it */
      char *dlerr = dlerror ();
      std::string err;
      if (dlerr)
        err = std::string(dlerr);
      else
        err = "unknown";

      kill_or_detach ();
      error (_("Cannot open library %s pointed by CUDBG_INJECTION_PATH: %s"),
             injectionPathEnv, err.c_str ());
    }

  dlclose (injectionLib);

  injectionPathAddr = cuda_get_symbol_address ("cudbgInjectionPath");
  if (!injectionPathAddr)
    {
      kill_or_detach ();
      error (_("No `cudbgInjectionPath` symbol in the CUDA driver"));
    }

  // If we can't write to the target, we can't have initialized the injection library
  auto res = target_write_memory (injectionPathAddr,
                                  (gdb_byte *)injectionPathEnv,
                                  strlen(injectionPathEnv) + 1);
  if (res != 0)
    cuda_trace ("cuda_initialize_injection: "
                "target_write_memory(injectionPath) failed %d",
                res);

  /* This message should be removed once we finalize the way the alternative
   * API backend is injected */
  printf ("CUDBG_INJECTION_PATH is set, forwarding it to the target (value: "
          "%s)\n",
          injectionPathEnv);
    
}

/* Tell the target application that it is being
 *  CUDA-debugged. Inferior must have been launched first.
 *  Returns true if initialization competed successfully, false otherwise
 */
bool
cuda_initialize_target (void)
{
  CORE_ADDR debugFlagAddr;
  CORE_ADDR rpcFlagAddr;
  CORE_ADDR gdbPidAddr;
  CORE_ADDR apiClientRevAddr;
  CORE_ADDR sessionIdAddr;
  CORE_ADDR launchblockingAddr;
  CORE_ADDR preemptionAddr;

  uint32_t apiClientPid;
  uint32_t apiClientRev = CUDBG_API_VERSION_REVISION;
  uint32_t sessionId;

  if (cuda_initialized)
    return true;

  // This flag is sticky
  if (current_inferior ()->cuda_initialized)
    return true;

  if (inferior_in_debug_mode)
    {
      cuda_initialize ();
      return true;
    }

  /* This is already done during cuda_linux_nat initialization, however some targets
     like cuda_core, will overwrite the loaded debugapi with their own. */
  if (cuda_get_cudbg_api ())
    error (_ ("Cannot get CUDA debugger API."));

  debugFlagAddr = cuda_get_symbol_address (_STRING_ (CUDBG_IPC_FLAG_NAME));
  if (!debugFlagAddr)
    return false;

  cuda_trace ("Initializing cuda target.\n");

  /* Initialize cuda utils, check if cuda-gdb lock is busy */
  cuda_signals_initialize ();
  cuda_debugapi::set_notify_new_event_callback (cuda_notification_notify);
  cuda_initialize ();
  sessionId = cuda_gdb_session_get_id ();

  /* When attaching or detaching, cuda_nat_attach() and
     cuda_nat_detach() control the setting of this flag,
     so don't touch it here. */
  if (CUDA_ATTACH_STATE_IN_PROGRESS != cuda_debugapi::get_attach_state ()
      && CUDA_ATTACH_STATE_DETACHING != cuda_debugapi::get_attach_state ())
    cuda_write_bool (debugFlagAddr, true);

  rpcFlagAddr = cuda_get_symbol_address (_STRING_ (CUDBG_RPC_ENABLED));
  gdbPidAddr = cuda_get_symbol_address (_STRING_ (CUDBG_APICLIENT_PID));
  apiClientRevAddr
      = cuda_get_symbol_address (_STRING_ (CUDBG_APICLIENT_REVISION));
  sessionIdAddr = cuda_get_symbol_address (_STRING_ (CUDBG_SESSION_ID));
  launchblockingAddr
      = cuda_get_symbol_address (_STRING_ (CUDBG_ENABLE_LAUNCH_BLOCKING));
  preemptionAddr
      = cuda_get_symbol_address (_STRING_ (CUDBG_ENABLE_PREEMPTION_DEBUGGING));

  if (!(rpcFlagAddr && gdbPidAddr && apiClientRevAddr && sessionIdAddr
        && launchblockingAddr && preemptionAddr))
    error (_ ("CUDA application cannot be debugged. The CUDA driver is not "
              "compatible."));

  cuda_initialize_injection ();

  apiClientPid = (uint32_t)getpid ();
  target_write_memory (gdbPidAddr, (gdb_byte *)&apiClientPid,
                       sizeof (apiClientPid));

  cuda_write_bool (rpcFlagAddr, true);
  target_write_memory (apiClientRevAddr, (gdb_byte *)&apiClientRev,
                       sizeof (apiClientRev));
  target_write_memory (sessionIdAddr, (gdb_byte *)&sessionId,
                       sizeof (sessionId));
  cuda_write_bool (launchblockingAddr, cuda_options_launch_blocking ());
  cuda_write_bool (preemptionAddr, cuda_options_software_preemption ());

  /* Setup our desired capabilities for the debugger backend. It is alright
   * if the older driver doesn't understand some of these flags. We will deal
   * with those situations after initialization. */
  CORE_ADDR capability_addr
      = cuda_get_symbol_address (_STRING_ (CUDBG_DEBUGGER_CAPABILITIES));
  if (capability_addr)
    {
      uint32_t capabilities = CUDBG_DEBUGGER_CAPABILITY_NONE;

      cuda_trace_domain (CUDA_TRACE_GENERAL,
			 "requesting CUDA lazy function loading support\n");
      capabilities |= CUDBG_DEBUGGER_CAPABILITY_LAZY_FUNCTION_LOADING;

      cuda_trace_domain (
	  CUDA_TRACE_GENERAL,
	  "requesting tracking of exceptions in exited warps\n");
      capabilities
	  |= CUDBG_DEBUGGER_CAPABILITY_REPORT_EXCEPTIONS_IN_EXITED_WARPS;

      cuda_trace_domain (
	  CUDA_TRACE_GENERAL,
	  "requesting no context push / pop events be delivered\n");
      capabilities
	  |= CUDBG_DEBUGGER_CAPABILITY_NO_CONTEXT_PUSH_POP_EVENTS;

      target_write_memory (capability_addr, (const gdb_byte *)&capabilities,
			   sizeof (capabilities));
    }

  cuda_update_report_driver_api_error_flags ();

  inferior_in_debug_mode = true;

  return true;
}

bool
cuda_inferior_in_debug_mode (void)
{
  return inferior_in_debug_mode;
}

type_instance_flags
cuda_address_class_type_flags (int byte_size, int addr_class)
{
  switch (addr_class)
    {
    case ptxCodeStorage:
      return TYPE_INSTANCE_FLAG_CUDA_CODE;
    case ptxConstStorage:
      return TYPE_INSTANCE_FLAG_CUDA_CONST;
    case ptxGenericStorage:
      return TYPE_INSTANCE_FLAG_CUDA_GENERIC;
    case ptxGlobalStorage:
      return TYPE_INSTANCE_FLAG_CUDA_GLOBAL;
    case ptxParamStorage:
      return TYPE_INSTANCE_FLAG_CUDA_PARAM;
    case ptxSharedStorage:
      return TYPE_INSTANCE_FLAG_CUDA_SHARED;
    case ptxLocalStorage:
      return TYPE_INSTANCE_FLAG_CUDA_LOCAL;
    case ptxRegStorage:
      return TYPE_INSTANCE_FLAG_CUDA_REG;
    case ptxURegStorage:
      return TYPE_INSTANCE_FLAG_CUDA_UREG;
    default:
      return 0;
    }
}

static const char *
cuda_address_class_type_flags_to_name (struct gdbarch *gdbarch,
                                       type_instance_flags type_flags)
{
  switch (type_flags & TYPE_INSTANCE_FLAG_CUDA_ALL)
    {
    case TYPE_INSTANCE_FLAG_CUDA_CODE:
      return "code";
    case TYPE_INSTANCE_FLAG_CUDA_CONST:
      return "constant";
    case TYPE_INSTANCE_FLAG_CUDA_GENERIC:
      return "generic";
    case TYPE_INSTANCE_FLAG_CUDA_GLOBAL:
      return "global";
    case TYPE_INSTANCE_FLAG_CUDA_PARAM:
      return "parameter";
    case TYPE_INSTANCE_FLAG_CUDA_SHARED:
      return "shared";
    case TYPE_INSTANCE_FLAG_CUDA_LOCAL:
      return "local";
    case TYPE_INSTANCE_FLAG_CUDA_REG:
      return "register";
    case TYPE_INSTANCE_FLAG_CUDA_MANAGED:
      return "managed_global";
    case TYPE_INSTANCE_FLAG_CUDA_UREG:
      return "uniform_register";
    default:
      return "unknown_segment";
    }
}

static bool
cuda_address_class_name_to_type_flags (struct gdbarch *gdbarch,
                                       const char *name,
                                       type_instance_flags *type_flags)
{
  if (strcmp (name, "code") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_CODE;
      return true;
    }

  if (strcmp (name, "constant") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_CONST;
      return true;
    }

  if (strcmp (name, "generic") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_GENERIC;
      return true;
    }

  if (strcmp (name, "global") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_GLOBAL;
      return true;
    }

  if (strcmp (name, "managed_global") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_MANAGED;
      return true;
    }

  if (strcmp (name, "parameter") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_PARAM;
      return true;
    }

  if (strcmp (name, "shared") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_SHARED;
      return true;
    }

  if (strcmp (name, "register") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_REG;
      return true;
    }

  if (strcmp (name, "local") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_LOCAL;
      return true;
    }

  if (strcmp (name, "uniform_register") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_UREG;
      return true;
    }

  return false;
}

void
cuda_print_lmem_address_type (void)
{
  struct gdbarch *gdbarch = get_current_arch ();
  const char *lmem_type = cuda_address_class_type_flags_to_name (
      gdbarch, TYPE_INSTANCE_FLAG_CUDA_LOCAL);

  gdb_printf ("(@%s unsigned *) ", lmem_type);
}

#define STO_CUDA_MANAGED 4  /* CUDA - managed variables */
#define STO_CUDA_ENTRY 0x10 /* CUDA - break_on_launch */

static void
cuda_elf_make_msymbol_special (asymbol *sym, struct minimal_symbol *msym)
{
  cuda_trace_domain (CUDA_TRACE_GENERAL, "symbol at 0x%016lx %s",
                     msym->value_raw_address (), msym->linkage_name ());

  /* managed variables */
  if (((elf_symbol_type *)sym)->internal_elf_sym.st_other == STO_CUDA_MANAGED)
    {
      msym->set_target_flag_1 (true);
      cuda_set_uvm_used (true);
    }

  /* break_on_launch */
  if (((elf_symbol_type *)sym)->internal_elf_sym.st_other == STO_CUDA_ENTRY)
    {
      /* Only insert loaded/relocated kernel entry points */
      auto addr = msym->value_raw_address ();
      if (addr)
	cuda_module::add_kernel_entry (addr);

      msym->set_value_address (addr);
    }
}

/* Temporary: intercept memory addresses when accessing known
   addresses pointing to CUDA RT variables. Returns 0 if found a CUDA
   RT variable, and 1 otherwise. */
static int
read_cudart_variable (uint64_t address, void *buffer, unsigned amount)
{
  if (address < CUDBG_BUILTINS_MAX)
    return 1;

  if (!cuda_current_focus::isDevice ())
    return 1;

  const auto &cur = cuda_current_focus::get ();

  if (CUDBG_THREADIDX_OFFSET <= address)
    {
      auto threadIdx = cur.logical ().threadIdx ();
      memcpy (buffer,
              (char *)&threadIdx + (int64_t)address - CUDBG_THREADIDX_OFFSET,
              amount);
    }
  else if (CUDBG_BLOCKIDX_OFFSET <= address)
    {
      auto blockIdx = cur.logical ().blockIdx ();
      memcpy (buffer,
              (char *)&blockIdx + (int64_t)address - CUDBG_BLOCKIDX_OFFSET,
              amount);
    }
  else if (CUDBG_CLUSTERIDX_OFFSET <= address)
    {
      // We could use clusterIdx here in cur, but that allows CUDA_IGNORE
      // which we don't want to expose to the user.
      auto clusterIdx = cuda_state::warp_get_cluster_idx (
          cur.physical ().dev (), cur.physical ().sm (),
          cur.physical ().wp ());
      memcpy (buffer,
              (char *)&clusterIdx + (int64_t)address - CUDBG_CLUSTERIDX_OFFSET,
              amount);
    }
  else if (CUDBG_GRIDDIM_OFFSET <= address)
    {
      auto kernel
          = kernels_find_kernel_by_kernel_id (cur.logical ().kernelId ());
      gdb_assert (kernel);
      auto gridDim = kernel_get_grid_dim (kernel);
      memcpy (buffer,
              (char *)&gridDim + (int64_t)address - CUDBG_GRIDDIM_OFFSET,
              amount);
    }
  else if (CUDBG_BLOCKDIM_OFFSET <= address)
    {
      auto kernel
          = kernels_find_kernel_by_kernel_id (cur.logical ().kernelId ());
      gdb_assert (kernel);
      auto blockDim = kernel_get_block_dim (kernel);
      memcpy (buffer,
              (char *)&blockDim + (int64_t)address - CUDBG_BLOCKDIM_OFFSET,
              amount);
    }
  else if (CUDBG_CLUSTERDIM_OFFSET <= address)
    {
      // We could use clusterDim here in cur, but that allows CUDA_IGNORE
      // which we don't want to expose to the user.
      auto clusterDim = cuda_state::warp_get_cluster_dim (
          cur.physical ().dev (), cur.physical ().sm (),
          cur.physical ().wp ());
      memcpy (buffer,
              (char *)&clusterDim + (int64_t)address - CUDBG_CLUSTERDIM_OFFSET,
              amount);
    }
  else if (CUDBG_WARPSIZE_OFFSET <= address)
    {
      auto num_lanes
          = cuda_state::device_get_num_lanes (cur.physical ().dev ());
      memcpy (buffer, &num_lanes, amount);
    }
  else
    return 1;

  return 0;
}

// Read from CUDA memory taking into account the type instance flags
// and device focus. If the address is not a CUDA address, or it's a
// generic pointer to memory that has to be read from the host, then
// return 1.  Otherwise, if the memory can be read from the GPU, do so
// and return 0 on success or raise an exception on error.
static int
cuda_read_memory_partial (CORE_ADDR address, type_instance_flags flags,
			  gdb_byte *buf, int len, uint64_t& hostaddr)
{
  gdb_assert (buf);

  hostaddr = address;

  // If CUDA isn't initialized, or if we don't have device focus,
  // return 1 so that the host path is tried next
  if (!cuda_debugging_enabled)
    return 1;

  cuda_set_host_address_resident_on_gpu (false);

  // Global memory accesses - does not require device focus
  if (flags & TYPE_INSTANCE_FLAG_CUDA_GLOBAL)
    {
      cuda_debugapi::read_global_memory ((uint64_t)address, buf, len);
      return 0;
    }

  // Then check for CUDA managed addresses
  if (cuda_managed_address_p (address))
    {
      cuda_debugapi::read_global_memory ((uint64_t)address, buf, len);
      cuda_set_host_address_resident_on_gpu (true);
      return 0;
    }

  if (cuda_current_focus::isDevice ())
    {
      const auto &c = cuda_current_focus::get ().physical ();

      // Generic addresses require some special handling
      if (!flags || (flags & TYPE_INSTANCE_FLAG_CUDA_GENERIC))
	{
	  /* If address of a built-in CUDA runtime variable, intercept it */
	  if (!read_cudart_variable (address, buf, len))
	    return 0;

	  try
	    {
	      // Will return true on success, false if it's a valid
	      // genric address but must be translated to a host
	      // address and accessed through the host. Other errors
	      // result in an exception being raised.
	      if (cuda_debugapi::read_generic_memory (c.dev (), c.sm (), c.wp (), c.ln (),
						      (uint64_t)address, buf, len))
		return 0;

	      // If we can translate the address, indicate that the host path should be tried next
	      // and return the translated address through hostaddr.
	      uint64_t mapped_addr = 0;
	      if (!cuda_debugapi::get_host_addr_from_device_addr (c.dev (),
								  (uint64_t)address, &mapped_addr))
		{
		  // Fallback to host access of hostaddr (==address)
		  return 1;
		}

	      // Replace the generic memory address with the mapped
	      // address from the host point of view.
	      hostaddr = mapped_addr;
	      return 1;
	    }
	  catch (const gdb_exception_error &e)
	    {
	      return 1;
	    }
	}

      if (flags & TYPE_INSTANCE_FLAG_CUDA_CODE)
	{
	  cuda_debugapi::read_code_memory (c.dev (),
					   (uint64_t)address, buf, len);
	  return 0;
	}

      if (flags & TYPE_INSTANCE_FLAG_CUDA_CONST)
	{
	  cuda_debugapi::read_const_memory (c.dev (),
					    (uint64_t)address, buf, len);
	  return 0;
	}

      if (flags & TYPE_INSTANCE_FLAG_CUDA_PARAM)
	{
	  cuda_debugapi::read_param_memory (c.dev (), c.sm (), c.wp (),
					    (uint64_t)address, buf, len);
	  return 0;
	}

      if (flags & TYPE_INSTANCE_FLAG_CUDA_SHARED)
	{
	  cuda_debugapi::read_shared_memory (c.dev (), c.sm (), c.wp (),
					     (uint64_t)address, buf, len);
	  return 0;
	}

      if (flags & TYPE_INSTANCE_FLAG_CUDA_LOCAL)
	{
	  cuda_debugapi::read_local_memory (c.dev (), c.sm (), c.wp (), c.ln (),
					    (uint64_t)address, buf, len);
	  return 0;
	}
    }

  return 1;
}

int
cuda_read_memory (CORE_ADDR address, type_instance_flags flags,
		  gdb_byte *buf, int len)
{
  gdb_assert (buf);

  cuda_trace ("cuda_read_memory (0x%lx, %u, 0x%08x)", (uint64_t)address, len, (uint32_t)flags);

  try
    {
      // If not a GPU address, try reading from the host
      uint64_t hostaddr = address;
      if (cuda_read_memory_partial (address, flags, buf, len, hostaddr))
	{
	  if (hostaddr == address)
	    return 1;
	  read_memory (hostaddr, buf, len);
	}
    }
  catch (gdb_exception_error& e)
    {
      cuda_trace ("Exception in read_memory(0x%lx, %d, 0x%08x)",
		  (uint64_t)address, len, (uint32_t)flags);
      throw;
    }

  if ((len == 4) || (len == 8))
    {
      uint64_t value = (len == 4) ? *(uint32_t *)buf : *(uint64_t *)buf;
      cuda_trace ("cuda_read_memory(0x%lx, %d, 0x%08x) = 0x%lx",
		  (uint64_t)address, len, (uint32_t)flags, value);
    }
  else
    cuda_trace ("cuda_read_memory(0x%lx, %d, 0x%08x)", (uint64_t)address, len, (uint32_t)flags);

  return 0;
}

// If there is an address class associated with this value, we've
// stored it in the type.  Check this here, and if set, read from the
// appropriate segment.
// Return 0 if the read was successful, 1 otherwise
int
cuda_read_memory (CORE_ADDR address, struct value *val, struct type *type, int len)
{
  gdb_assert (val);
  gdb_assert (type);
  
  /* No CUDA. Read the host memory */
  if (!cuda_debugging_enabled)
    return 1;

  const auto flags = TYPE_CUDA_ALL (type);
  gdb_byte *buf = value_contents_all_raw (val).data ();

  cuda_trace ("cuda_read_memory (0x%lx, %d, 0x%08x) [struct value]",
	      (uint64_t)address, len, (uint32_t)flags);

  if (!cuda_read_memory (address, flags, buf, len))
    return 0;
  
  // Check if the variable is on the stack (local memory). It happens
  // when not in the innermost frame.
  if (value_stack (val) && cuda_current_focus::isDevice ())
    {
      cuda_trace ("Trying to read from stack (local)");
      try
	{
	  if (!cuda_read_memory (address, TYPE_INSTANCE_FLAG_CUDA_LOCAL, buf, len))
	    return 0;
	}
      catch (gdb_exception_error &e)
	{
	}
    }

  // Default: read the host memory as usual.
  cuda_trace ("Falling back to host access");  
  return 1;
}

static CORE_ADDR
cuda_get_const_bank_address (uint32_t bank, uint32_t offset)
{
  if (!cuda_current_focus::isDevice ())
    {
      warning (_("A CUDA device isn't focused.\n"));
      return 0;
    }

  uint64_t addr = 0;
  const auto& c = cuda_current_focus::get ().physical ();

  if (cuda_debugapi::api_version ().m_revision >= 141)
    {
      const uint64_t gridId64 =
        cuda_current_focus::get ().logical ().gridId ();

      uint32_t size;
      cuda_debugapi::get_const_bank_address (c.dev (), gridId64,
                                             bank, &addr, &size);

      if (addr == 0 || offset >= size)
        throw_error (GENERIC_ERROR, "The requested value c[0x%x][0x%x] is not valid.", bank, offset);

      addr += offset;
    }
  else
    cuda_debugapi::get_const_bank_address (c.dev (), c.sm (), c.wp (),
                                              bank, offset, &addr);

  return addr;
}

static struct value *
cuda_get_const_bank_address_val (struct gdbarch *gdbarch,
                                 uint32_t bank, uint32_t offset)
{
  CORE_ADDR addr = cuda_get_const_bank_address (bank, offset);
  struct type *const_bank_uint = builtin_type (gdbarch)->builtin_unsigned_int;
  const_bank_uint->set_instance_flags(TYPE_INSTANCE_FLAG_CUDA_CONST);
  struct type *uint_ptr_type = lookup_pointer_type (const_bank_uint);

  return value_from_pointer(uint_ptr_type, addr);
}

// This is to preserve the symmetry of cuda_read/write_memory_partial.
static int
cuda_write_memory_partial (CORE_ADDR address, const gdb_byte *buf,
                           struct type *type, uint64_t& hostaddr)
{
  auto len = type->length ();

  hostaddr = address;

  /* No CUDA. Return 1. */
  if (!cuda_debugging_enabled)
    return 1;

  /* If address is marked as belonging to a CUDA memory segment, use the
     appropriate API call. */
  type_instance_flags flags
      = type ? TYPE_CUDA_ALL (type) : TYPE_INSTANCE_FLAG_CUDA_GENERIC;

  cuda_trace ("cuda_write_memory_partial (0x%lx, %u, 0x%08x)",
	      (uint64_t)address, (uint32_t)len, (uint32_t)flags);

  if (flags)
    {
      /* We can write global memory directly without cuda coords. */
      if (flags & TYPE_INSTANCE_FLAG_CUDA_GLOBAL)
        {
          cuda_debugapi::write_global_memory (address, buf, len);
          return 0;
        }

      /* Ensure we have device focus */
      if (!cuda_current_focus::isDevice ())
        return 1;

      const auto &c = cuda_current_focus::get ().physical ();
      if (flags & TYPE_INSTANCE_FLAG_CUDA_REG)
        {
          /* The following explains how we can come down this path, and why
             cuda_debugapi::write_local_memory is called when the address
             class indicates ptxRegStorage.

             We should only enter this case if we are:
                 1. debugging an application that is using the ABI
                 2. modifying a variable that is mapped to a register that
             has been saved on the stack
                 3. not modifying a variable for the _innermost_ device frame
                    (as this would follow the cuda_pseudo_register_write
             path).

             We can possibly add additional checks to ensure that address is
             within the permissable stack range, but
             cuda_debugapi::write_local_memory better return an appropriate
             error in that case anyway, so let's test the API.

             Note there is no corresponding case in
             cuda_read_memory_with_valtype, because _reading_ a previous
             frame's (saved) registers is all done directly by prev register
             methods (dwarf2-frame.c, cuda-tdep.c).

             As an alternative, we could intercept the value type prior to
             reaching this function and change it to ptxLocalStorage, but
             that can make debugging somewhat difficult. */
          gdb_assert (cuda_current_active_elf_image_uses_abi ());
          cuda_debugapi::write_local_memory (c.dev (), c.sm (), c.wp (),
                                             c.ln (), address, buf, len);
        }
      else if (flags & TYPE_INSTANCE_FLAG_CUDA_GENERIC)
	{
	  if (cuda_debugapi::write_generic_memory (c.dev (), c.sm (), c.wp (), c.ln (),
						   address, buf, len))
	    return 0;

	  // If we can translate the address, indicate that the host path should be tried next
	  // and return the translated address through hostaddr
	  uint64_t mapped_addr = 0;
	  if (cuda_debugapi::get_host_addr_from_device_addr (c.dev (), (uint64_t)address, &mapped_addr))
	    {
	      write_memory (mapped_addr, buf, len);
	      return 0;
	    }
	}
      else if (flags & TYPE_INSTANCE_FLAG_CUDA_PARAM)
        cuda_debugapi::write_param_memory (c.dev (), c.sm (), c.wp (), address,
                                           buf, len);
      else if (flags & TYPE_INSTANCE_FLAG_CUDA_SHARED)
        cuda_debugapi::write_shared_memory (c.dev (), c.sm (), c.wp (),
                                            address, buf, len);
      else if (flags & TYPE_INSTANCE_FLAG_CUDA_LOCAL)
        cuda_debugapi::write_local_memory (c.dev (), c.sm (), c.wp (), c.ln (),
                                           address, buf, len);
      else if (flags & TYPE_INSTANCE_FLAG_CUDA_CODE)
        error (_ ("Writing to code memory is not allowed."));
      else if (flags & TYPE_INSTANCE_FLAG_CUDA_CONST)
        error (_ ("Writing to constant memory is not allowed."));
      else
        error (_ ("Unknown storage specifier (write)  0x%x"),
               (unsigned int)flags);
      return 0;
    }
  return 1;
}

/* If there is an address class associated with this value, we've
   stored it in the type.  Check this here, and if set, write to the
   appropriate segment. */
void
cuda_write_memory (CORE_ADDR address, const gdb_byte *buf, struct type *type)
{
  auto len = type->length ();

  if ((len == 4) || (len == 8))
    {
      uint64_t value = (len == 4) ? *(uint32_t *)buf : *(uint64_t *)buf;
      cuda_trace ("cuda_write_memory(0x%lx, %u) 0x%lx", (uint64_t)address, (uint32_t)len, value);
    }
  else
    cuda_trace ("cuda_write_memory(0x%lx, %u)", (uint64_t)address, (uint32_t)len);

  /* No CUDA. Write the host memory */
  if (!cuda_debugging_enabled)
    {
      write_memory (address, buf, len);
      return;
    }

  /* Default: write the host memory as usual */
  try
    {
      uint64_t hostaddr = 0;
      if (!cuda_write_memory_partial (address, buf, type, hostaddr))
	return;

      /* Call the partial memory write, return on success */
      write_memory (hostaddr, buf, len);
    }
  catch (const gdb_exception_error &e)
    {
      /* CUDA - managed memory */
      if (!cuda_managed_address_p (address))
        throw;

      cuda_debugapi::write_global_memory ((uint64_t)address, buf, len);
    }
}

/* Single-Stepping

   The following data structures and routines are used as a framework to
   manage single-stepping with CUDA devices. It currently solves 2 issues

   1. When single-stepping a warp, we do not want to resume the host if we do
   not have to. The single-stepping framework allows for making GDB believe
   that everything was resumed and that a SIGTRAP was received after each
   step.

   2. When single-stepping a warp, other warps may be required to be stepped
   too. Out of convenience to the user, we want to keep single-stepping those
   other warps alongside the warp in focus. By doing so, stepping over a
   __syncthreads() instruction will bring all the warps in the same block to
   the next source line.

   This result is achieved by marking the warps we want to single-step with a
   warp mask. When the user issues a new command, the warp is initialized
   accordingly. If the command is a step command, we initialize the warp mask
   with the warp mask and let the mask grow over time as stepping occurs
   (there might be more than one step). If the command is not a step command,
   the warp mask is set empty and will remain that way. In that situation, if
   single-stepping is required, only the minimum number of warps will be
   single-stepped. */

static struct
{
  bool active;
  bool warned_on_divergent_stepping;
  bool grid_id_active;
  ptid_t ptid;
  cuda_coords coord;
  uint32_t before_lane_mask;
  uint64_t before_pc;
  cuda_api_warpmask warp_mask;
} cuda_sstep_info;

static int cuda_sstep_nsteps = 1;

void
cuda_sstep_set_nsteps (int nsteps)
{
  cuda_sstep_nsteps = nsteps;
}

bool
cuda_sstep_is_active (void)
{
  return cuda_sstep_info.active;
}

/* We only want to warn the user once about stepping
 * divergent threads. We use an observer to reset
 * if we have warned the user. */
static void
cuda_sstep_about_to_proceed (void)
{
  cuda_sstep_info.warned_on_divergent_stepping = false;
}

bool
cuda_print_divergent_stepping (void)
{
  if (cuda_sstep_info.warned_on_divergent_stepping)
    return false;
  cuda_sstep_info.warned_on_divergent_stepping = true;
  return true;
}

ptid_t
cuda_sstep_ptid (void)
{
  gdb_assert (cuda_sstep_info.active);
  return cuda_sstep_info.ptid;
}

uint32_t
cuda_sstep_dev_id (void)
{
  gdb_assert (cuda_sstep_info.active);
  return cuda_sstep_info.coord.physical ().dev ();
}

uint32_t
cuda_sstep_sm_id (void)
{
  gdb_assert (cuda_sstep_info.active);
  return cuda_sstep_info.coord.physical ().sm ();
}

uint32_t
cuda_sstep_wp_id (void)
{
  gdb_assert (cuda_sstep_info.active);
  return cuda_sstep_info.coord.physical ().wp ();
}

bool
cuda_sstep_lane_stepped (uint32_t ln_id)
{
  gdb_assert (cuda_sstep_info.active);
  return (cuda_sstep_info.before_lane_mask >> ln_id) & 1;
}

uint32_t
cuda_sstep_get_lowest_lane_stepped (void)
{
  uint32_t ln_id;
  gdb_assert (cuda_sstep_info.active);
  const auto nlanes = cuda_state::device_get_num_lanes (
      cuda_sstep_info.coord.physical ().dev ());
  for (ln_id = 0; ln_id < nlanes; ++ln_id)
    if ((cuda_sstep_info.before_lane_mask >> ln_id) & 1)
      break;
  return ln_id;
}

uint64_t
cuda_sstep_get_last_pc (void)
{
  gdb_assert (cuda_sstep_info.active);
  return cuda_sstep_info.before_pc;
}

cuda_api_warpmask *
cuda_sstep_wp_mask (void)
{
  gdb_assert (cuda_sstep_info.active);
  return &cuda_sstep_info.warp_mask;
}

uint64_t
cuda_sstep_grid_id (void)
{
  gdb_assert (cuda_sstep_info.active);
  return cuda_sstep_info.coord.logical ().gridId ();
}

void
cuda_sstep_set_ptid (ptid_t ptid)
{
  gdb_assert (cuda_sstep_info.active);
  cuda_sstep_info.ptid = ptid;
}

bool
cuda_find_next_control_flow_instruction (uint64_t pc, uint64_t range_start_pc,
                                         uint64_t range_end_pc, /* Exclusive */
                                         bool skip_subroutines,
                                         uint64_t &end_pc, uint32_t &inst_size)
{
  uint64_t adj_pc;
  kernel_t kernel = cuda_current_focus::get ().logical ().kernel ();
  std::string inst_str;

  inst_size = 0;

  cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                     "find next at pc=%lx start=%lx end=%lx", pc,
                     range_start_pc, range_end_pc);

  /* Check to see if we are not in the original source line range.
   * This can happen when branching to an inlined call.
   * In that case, lookup the start and end range for the new source line. */
  if ((pc < range_start_pc) || (pc >= range_end_pc))
    {
      struct symtab_and_line sal;

      sal = find_pc_line (pc, 0);
      range_start_pc = sal.pc;
      range_end_pc = sal.end;
      cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                         "adjust range start=%lx end=%lx", range_start_pc,
                         range_end_pc);
    }

  /* Iterate over instructions until the end of the step/next line range.
   * Break if instruction that can potentially alter program counter has been
   * encountered. */
  end_pc = pc;
  while (end_pc <= range_end_pc)
    {
      auto module = kernel_get_module (kernel);
      gdb_assert (module);

      auto disassembler = module->disassembler ();
      gdb_assert (disassembler);

      inst_size = disassembler->insn_size ();

      auto inst = disassembler->disassemble_instruction (end_pc);
      if (!inst)
        {
          cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                             "could not disassemble pc=0x%lx", end_pc);
          break;
        }

        inst_str = inst->to_string();
      cuda_trace_domain (CUDA_TRACE_BREAKPOINT, "%s: pc=0x%lx inst %.*s",
                         __func__, end_pc, 20, inst_str.c_str ());

      if (inst->is_control_flow (skip_subroutines))
        break;

      end_pc += inst_size;
    }

  /* The above loop might increment end_pc beyond step_range_end.
     In that case, adjust it to the step_range_end. */
  if (end_pc > range_end_pc)
    end_pc = range_end_pc;

  /* Adjust the end_pc to the exact instruction address */
  adj_pc = gdbarch_adjust_breakpoint_address (cuda_get_gdbarch (), end_pc);
  end_pc = adj_pc > range_end_pc
               ? range_end_pc - inst_size /* end_pc was pointing at SHINT and
                                             was adjusted beyond range_end_pc
                                             - reset to just before SHINT */
               : adj_pc;

  /* Will have stopped short if a control flow instruction is found */
  if (end_pc != range_end_pc)
    cuda_trace_domain (
        CUDA_TRACE_BREAKPOINT,
        "%s: next control point after %lx (up to %lx) is at %lx (inst %s)",
        __func__, pc, range_end_pc, end_pc, inst_str.c_str ());
  else
    cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                       "no control point found at end_pc=%lx", end_pc);

  /* end_pc is very close to range_end_pc if no control flow instruction was
   * found */
  return true;
}

static bool
cuda_sstep_fast (ptid_t ptid)
{
  if (!cuda_options_single_stepping_optimizations_enabled ())
    return false;

  /* No accelerated single stepping when we accuracy is expected */
  if (cuda_get_autostep_pending ())
    return false;

  struct thread_info *tp = inferior_thread ();
  if (!tp)
    return false;
  gdb_assert (tp->ptid == ptid);

  /* Skip if stepping just for one instruction */
  if (tp->control.step_range_end <= 1 && tp->control.step_range_start <= 1)
    return false;

  const auto &c = cuda_current_focus::get ().physical ();
  uint64_t pc = get_frame_pc (get_current_frame ());

  bool skip_subroutines = tp->control.step_over_calls == STEP_OVER_ALL;

  /* Defined by cuda_find_next_control_flow_instruction */
  uint64_t end_pc;
  uint32_t inst_size;
  auto found = cuda_find_next_control_flow_instruction (
      pc, tp->control.step_range_start, tp->control.step_range_end,
      skip_subroutines, end_pc, inst_size);
  if (!found)
    return false;

  /* Do not attempt to accelerate if stepping over less than 3 instructions
   */
  if ((end_pc <= pc) || ((end_pc - pc) < (3 * inst_size)))
    {
      cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                         "%s: advantage is not big enough: pc=0x%lx "
                         "end_pc=0x%lx inst_size = %u",
                         __func__, pc, end_pc, inst_size);
      return false;
    }

  cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                     "%s: trying to step from %lx to %lx", __func__, pc,
                     end_pc);

  /* If breakpoint is set at the current (or current active) PC - temporarily
   * unset it*/
  struct address_space *aspace = target_thread_address_space (ptid);
  uint64_t active_pc
      = cuda_state::warp_get_active_pc (c.dev (), c.sm (), c.wp ());

  if (breakpoint_here_p (aspace, pc))
    cuda_debugapi::unset_breakpoint (c.dev (), pc);

  if (active_pc != pc && breakpoint_here_p (aspace, active_pc))
    cuda_debugapi::unset_breakpoint (c.dev (), active_pc);

  /* Resume warp(s) until one of the lanes reaches end_pc */
  bool rc = cuda_state::resume_warps_until_pc (
      c.dev (), c.sm (), &cuda_sstep_info.warp_mask, end_pc);

  /* Reset the breakpoint if warps_resume_until call failed */
  if (!rc && breakpoint_here_p (aspace, pc))
    cuda_debugapi::set_breakpoint (c.dev (), pc);

  if (!rc && active_pc != pc && breakpoint_here_p (aspace, active_pc))
    cuda_debugapi::set_breakpoint (c.dev (), active_pc);

  return rc;
}

static bool
cuda_sstep_do_step (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,
                    uint32_t lane_id_hint, uint32_t nsteps, uint32_t flags,
                    cuda_api_warpmask &single_stepped_warp_mask)
{
  bool rc;

  rc = cuda_state::single_step_warp (dev_id, sm_id, wp_id, lane_id_hint,
                                     nsteps, flags, &single_stepped_warp_mask);
  if (rc || nsteps < 2)
    return rc;

  /* Fallback mode: if nsteps failed try single step */
  rc = cuda_state::single_step_warp (dev_id, sm_id, wp_id, lane_id_hint,
                                     1, flags, &single_stepped_warp_mask);
  return rc;
}

/**
 * cuda_sstep_execute(ptid_t) returns true if single-stepping was successful.
 * If false is returned, it is callee responsibility to cleanup CUDA sstep
 * state. (usually done by calling cuda_sstep_reset())
 */
bool
cuda_sstep_execute (ptid_t ptid)
{
  cuda_api_warpmask warp_mask;
  bool rc = true;

  /* Ensure we are currently not sstepping and we have device focus */
  gdb_assert (!cuda_sstep_info.active);
  gdb_assert (cuda_current_focus::isDevice ());

  /* Save nsteps locally and reset to default 1 */
  int nsteps = cuda_sstep_nsteps;
  cuda_sstep_set_nsteps (1);

  /* Get the current focus */
  const auto &coord = cuda_current_focus::get ();
  const auto &l = coord.logical ();
  const auto &p = coord.physical ();

  /* Save local info */
  bool grid_id_changed
      = cuda_sstep_info.grid_id_active && cuda_sstep_info.coord.valid ()
        && (cuda_sstep_info.coord.logical ().gridId () != l.gridId ());
  bool sstep_other_warps = cuda_api_has_bit (&cuda_sstep_info.warp_mask);
  /* Track the warp(s) we stepped */
  cuda_api_warpmask stepped_warp_mask;
  cuda_api_clear_mask (&stepped_warp_mask);

  /* Remember the single-step parameters to trick GDB */
  cuda_sstep_info.active = true;
  cuda_sstep_info.grid_id_active = true;
  cuda_sstep_info.ptid = ptid;
  cuda_sstep_info.coord = coord;
  gdb_assert (cuda_sstep_info.coord.isValidOnDevice ());
  cuda_sstep_info.before_lane_mask
      = cuda_state::warp_get_active_lanes_mask (p.dev (), p.sm (), p.wp ());
  cuda_sstep_info.before_pc
      = cuda_state::warp_get_active_pc (p.dev (), p.sm (), p.wp ());

  /* Do not try to single step if grid-id changed */
  if (grid_id_changed)
    {
      cuda_trace ("device %u sm %u: switched to new grid %llx while "
                  "single-stepping!\n",
                  p.dev (), p.sm (), (unsigned long long)l.gridId ());
      cuda_api_clear_mask (&cuda_sstep_info.warp_mask);
      cuda_api_set_bit (&cuda_sstep_info.warp_mask, p.wp (), 1);
      return true;
    }

  if (!sstep_other_warps)
    {
      cuda_api_clear_mask (&cuda_sstep_info.warp_mask);
      cuda_api_set_bit (&cuda_sstep_info.warp_mask, p.wp (), 1);
    }

  cuda_trace (
      "device %u sm %u: single-stepping warp mask %" WARP_MASK_FORMAT "\n",
      p.dev (), p.sm (), cuda_api_mask_string (&cuda_sstep_info.warp_mask));
  gdb_assert (cuda_api_get_bit (&cuda_sstep_info.warp_mask, p.wp ()));

  uint32_t wp_max = cuda_state::device_get_num_warps (p.dev ());

  if (cuda_options_software_preemption ())
    {
      /* If sw preemption is enabled, then only step
         the warp in focus.  Do not use the resulting
         warp_mask as it is invalid in between single
         step operations when this mode is enabled
         (these warps can/will land on different SMs,
         which is handled by invalidating state for
         all warps instead of using this mask -- see
         warp_single_step) */
      if (!cuda_sstep_fast (ptid))
        {
          uint32_t flags = 0;
          if (cuda_get_autostep_pending() || !cuda_options_step_divergent_lanes_enabled ())
            flags |= CUDBG_SINGLE_STEP_FLAGS_NO_STEP_OVER_WARP_BARRIERS;
          rc = cuda_sstep_do_step (p.dev (), p.sm (), p.wp (), p.ln (),
                                   nsteps, flags, warp_mask);
        }
    }
  else if (!cuda_sstep_fast (ptid))
    {
      /* Single-step all the warps in the warp mask. */
      for (uint32_t wp = 0; wp < wp_max; ++wp)
        if (cuda_api_get_bit (&cuda_sstep_info.warp_mask, wp)
            && cuda_state::sm_valid (p.dev (), p.sm ())
            && cuda_state::warp_valid (p.dev (), p.sm (), wp))
          {
            /* Set hint to an invalid lane, let the backend decide */
            constexpr uint32_t lane_id_hint = ~0;
            uint32_t flags = 0;
            if (cuda_get_autostep_pending() || !cuda_options_step_divergent_lanes_enabled ())
              flags |= CUDBG_SINGLE_STEP_FLAGS_NO_STEP_OVER_WARP_BARRIERS;
            rc = cuda_sstep_do_step (p.dev (), p.sm (), wp, lane_id_hint,
                                     nsteps, flags, warp_mask);
            if (!rc)
              break;

            cuda_api_or_mask (&stepped_warp_mask, &stepped_warp_mask,
                              &warp_mask);

            if (cuda_api_has_multiple_bits (&warp_mask))
              {
                /* warp_mask will have multiple bits set in case there was a
                   barrier instruction. In such case skip iterating through
                   the remaining valid warps as they are already synchronized
                 */
                break;
              }
          }
    }

  if (cuda_state::sm_valid (p.dev (), p.sm ()))
    {
      // Update the warp mask. It may have grown.
      cuda_api_cp_mask (&cuda_sstep_info.warp_mask, &stepped_warp_mask);

      // If any warps are marked invalid, but are in the warp_mask
      // clear them. This can happen if we stepped a warp over an exit
      cuda_api_and_mask (&cuda_sstep_info.warp_mask, &cuda_sstep_info.warp_mask,
			 cuda_state::sm_get_valid_warps_mask (p.dev (), p.sm ()));
    }
  else
    {
      // SM now empty, clear the cuda_sstep_info.warp_mask
      cuda_api_clear_mask (&cuda_sstep_info.warp_mask);
    }

  return rc;
}

void
cuda_sstep_initialize (bool stepping)
{
  cuda_api_clear_mask (&cuda_sstep_info.warp_mask);
  if (stepping && cuda_current_focus::isDevice ())
    cuda_api_set_bit (&cuda_sstep_info.warp_mask,
                      cuda_current_focus::get ().physical ().wp (), 1);
  cuda_sstep_info.grid_id_active = false;
}

void
cuda_sstep_reset (bool sstep)
{
  /*  When a subroutine is entered while stepping the device, cuda-gdb will
      insert a breakpoint and resume the device. When this happens, the focus
      may change due to the resume. This will cause the cached single step
     warp mask to be incorrect, causing an assertion failure. The fix here is
     to reset the warp mask when switching to a resume. This will cause
      single step execute to update the warp mask after performing the step.
   */
  if (!sstep && cuda_current_focus::isDevice () && cuda_sstep_is_active ())
    {
      cuda_api_clear_mask (&cuda_sstep_info.warp_mask);
      cuda_sstep_info.grid_id_active = false;
      cuda_sstep_info.coord = cuda_coords{};
    }

  cuda_sstep_info.active = false;
}

bool
cuda_sstep_kernel_has_terminated (void)
{
  gdb_assert (cuda_sstep_info.active);

  const auto &p = cuda_sstep_info.coord.physical ();

  /* If the warp we were stepping is still valid, the kernel has yet to
   * terminate. */
  if (cuda_state::sm_valid (p.dev (), p.sm ())
      && cuda_state::warp_valid (p.dev (), p.sm (), p.wp ()))
    return false;

  /* Check to see if the grid is still present on the device. */
  const auto &l = cuda_sstep_info.coord.logical ();
  cuda_coords filter{
    p.dev (),          CUDA_WILDCARD,     CUDA_WILDCARD,
    CUDA_WILDCARD,     CUDA_WILDCARD,     l.gridId (),
    CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM
  };
  cuda_coord_set<cuda_coord_set_type::kernels, select_valid | select_sngl> coord{
    filter
  };
  if (coord.size ())
    return false;

  /* Return true if we didn't find a valid warp for the grid */
  return true;
}

/* CUDA ABI return value convention:

   (size == 32-bits) .s32/.u32/.f32/.b32      -> R4
   (size == 64-bits) .s64/.u64/.f64/.b64      -> R4-R5
   (size <= 384-bits) .align N .b8 name[size] -> size <= 384-bits -> R4-R15
   (A) (size > 384-bits)  .align N .b8 name[size] -> size > 384-bits  ->
   Memory (B)

   For array case (B), the pointer to the memory location is passed as
   a parameter at the beginning of the parameter list.  Memory is allocated
   in the _calling_ function, which is the consumer of the return value.
*/
static enum return_value_convention
cuda_abi_return_value (struct gdbarch *gdbarch, struct value *function,
                       struct type *type, struct regcache *regcache,
                       gdb_byte *readbuf, const gdb_byte *writebuf)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  ULONGEST regnum = tdep->first_rv_regnum;
  int len = type->length ();

  /* The return value is in one or more registers. */
  if (len <= tdep->max_reg_rv_size)
    {
      /* Read/write all regs until we've satisfied len. */
      for (int i = 0; len > 0; i++, regnum++, len -= 4)
        {
          if (readbuf)
            {
	      ULONGEST regval = 0ULL;
              regcache_cooked_read_unsigned (regcache, regnum, &regval);
              uint32_t regval32 = (uint32_t)regval;
              memcpy (readbuf + i * 4, &regval32, std::min (len, 4));
            }
          if (writebuf)
            {
	      uint32_t regval32 = 0U;
              memcpy (&regval32, writebuf + i * 4, std::min (len, 4));
              ULONGEST regval = regval32;
              regcache_cooked_write_unsigned (regcache, regnum, regval);
            }
        }

      return RETURN_VALUE_REGISTER_CONVENTION;
    }

  /* The return value is in memory. */
  if (readbuf)
    {

      /* In the case of large return values, space has been allocated in
         memory to hold the value, and a pointer to that allocation is at the
         beginning of the parameter list.  We need to read the register that
         holds the address, and then read from that address to obtain the
         value. */
      ULONGEST addr;
      regcache_cooked_read_unsigned (regcache, regnum, &addr);
      const auto &c = cuda_current_focus::get ().physical ();
      cuda_debugapi::read_local_memory (c.dev (), c.sm (), c.wp (), c.ln (),
                                        addr, readbuf, len);
    }

  return RETURN_VALUE_ABI_RETURNS_ADDRESS;
}

static int
cuda_adjust_regnum (struct gdbarch *gdbarch, int regnum, int eh_frame_p)
{
  int adjusted_regnum = 0;

  /* If not a device register, nothing to adjust. This happens only when
     called by the DWARF2 frame sniffer when determining the type of the
     frame. It is then safe to bail out and not pass the request to the host
     adjust_regnum function, because, at that point, the type of the frame is
     not yet determinted. */
  if (!cuda_current_focus::isDevice ())
    return regnum;

  if (!cuda_decode_physical_register (regnum, &adjusted_regnum))
    return adjusted_regnum;

  return 0;
}

static CORE_ADDR
cuda_skip_prologue (struct gdbarch *gdbarch, CORE_ADDR pc)
{
  CORE_ADDR start_addr, end_addr, post_prologue_pc;

  /* CUDA - skip prologue - temporary
     Until we always have a prologue, even if empty.
     If Tesla kernel generated with open64, there is no prologue */
  {
    struct obj_section *osect = find_pc_section (pc);

    if (osect && osect->objfile
        && !cuda_is_bfd_version_call_abi (osect->objfile->obfd.get ())
        && osect->objfile->cuda_objfile
        && osect->objfile->cuda_producer_is_open64)
      return pc;
  }

  /* See if we can determine the end of the prologue via the symbol table.
     If so, then return either PC, or the PC after the prologue, whichever
     is greater.  */
  if (find_pc_partial_function (pc, NULL, &start_addr, &end_addr))
    {
      post_prologue_pc = skip_prologue_using_sal (gdbarch, start_addr);

      /* There is a bug in skip_prologue_using_sal(). The end PC returned by
         find_pc_sect_line() is off by one instruction. It's pointing to the
         first instruction of the next line instead of the last instruction
         of the current line. I cannot fix it there since the instruction
         size is unknown. But I can fix it here, which also has the advantage
         of not impacting the way gdb behaves with the host code. When that
         happens, it means that the function body is empty (foo(){};). In
         that case, we follow GDB policy and do not skip the prologue. It
         also allow us to no point to the last instruction of a device
         function. That instruction is not guaranteed to be ever executed,
         which makes setting breakpoints trickier. */
      if (post_prologue_pc > end_addr)
        post_prologue_pc = pc;

      /* If the post_prologue_pc does not make sense, return the given PC. */
      if (post_prologue_pc < pc)
        post_prologue_pc = pc;

      return post_prologue_pc;

      /* If we can't adjust the prologue from the symbol table, we may need
         to resort to instruction scanning.  For now, assume the entry above.
       */
    }

  /* If we can do no better than the original pc, then just return it. */
  return pc;
}

/* CUDA:  Determine whether or not the subprogram DIE requires
   a base address that should be added to all child DIEs.  This
   is determined based on the ABI version in the CUDA ELF image. */
CORE_ADDR
cuda_dwarf2_func_baseaddr (struct objfile *objfile, char *func_name)
{
  bool is_cuda_abi;
  unsigned int cuda_abi_version;

  /* See if this is a CUDA ELF object file and get its abi version.  Before
     CUDA_ELFOSABIV_RELOC, all DWARF DIE low/high PC attributes did not use
     relocators and were 0-based.  After CUDA_ELFOSABIV_RELOC, the low/high
     PC attributes use relocators and are no longer 0-based. */
  is_cuda_abi = cuda_get_bfd_abi_version (objfile->obfd.get (), &cuda_abi_version);
  if (is_cuda_abi
      && cuda_abi_version < CUDA_ELFOSABIV_RELOC) /* Not relocated */
    {
      if (func_name)
        {
          CORE_ADDR vma;
          if (cuda_find_func_text_vma_from_objfile (objfile, func_name, &vma))
            /* Return section base addr (vma) for this function */
            return vma;
        }
    }

  /* No base address */
  return 0;
}

/* Given an address string (function_name or filename:function_name or
   filename:lineno), find if there is matching address for it. If so, return
   it (prologue adjusted, if necessary) in func_addr. Returns true if the
   function is found, false otherwise. */
bool
cuda_find_pc_from_address_string (struct objfile *objfile, const char *func_name,
                                  CORE_ADDR &func_addr)
{
  std::string func_name_str{ func_name };
  struct gdbarch *gdbarch = get_current_arch ();
  const struct language_defn *lang = current_language;
  int dmgl_options = DMGL_ANSI | DMGL_PARAMS;
  gdb::unique_xmalloc_ptr<char> demangled;
  int lineno = 0;

  gdb_assert (objfile);
  gdb_assert (func_name);

  if (!cuda_is_bfd_cuda (objfile->obfd.get ()))
    return false;

  /* Test if a space exist in the function name
     If so, it's likely to be a pending conditional,
     so we going to ignore everything after the space */
  auto found_pos = func_name_str.find (' ');
  if (found_pos != std::string::npos)
    {
      func_name_str.erase (found_pos);
    }

  /* Test if a colon exists in the function name string.
     If so, name might be limited to symtab scope
     otherwise we can ignore everything before it
     so that we can search using the function name directly. */
  found_pos = func_name_str.find (':');
  if (found_pos != std::string::npos)
    {
      auto filename = func_name_str.substr (0, found_pos);
      auto lineno_str = func_name_str.substr (found_pos + 1);

      /* If all characters after colon are digits - it's a line number */
      if (lineno_str.find_first_not_of ("0123456789") == std::string::npos)
        {
          lineno = atoi (lineno_str.c_str ());
          for (compunit_symtab *cu : objfile->compunits ())
            {
              for (symtab *s : cu->filetabs ())
                {
                  if (compare_filenames_for_search (s->filename,
                                                    filename.c_str ()))
                    {
		      CORE_ADDR addr = 0;
                      if (find_line_pc (s, lineno, &addr))
                        {
                          func_addr = addr;
                          return true;
                        }
                    }
                }
            }
        }
    }

  /* We need to find the fully-qualified symbol name that func_name
     corresponds to (if any).  This will handle mangled symbol names,
     which is what will be used to lookup a CUDA device code symbol. */
  struct bound_minimal_symbol bmsym = lookup_minimal_symbol (func_name_str.c_str (), NULL, objfile);

  std::string sym_name_str;
  if (bmsym.minsym != NULL)
    sym_name_str = std::string{ bmsym.minsym->linkage_name () };
  else if ((demangled
            = language_demangle (lang, func_name_str.c_str (), dmgl_options)))
    {
      sym_name_str = std::string{ demangled.get () };
    }
  else
    sym_name_str = func_name_str;

  /* Look for functions - assigned from DWARF, this path will only
     find information for debug compilations. */
  lookup_name_info lookup_name (sym_name_str.c_str (),
                                symbol_name_match_type::SEARCH_NAME);

  for (compunit_symtab *cu : objfile->compunits ())
    {
      for (symtab *s : cu->filetabs ())
        {
	  for (const block *b : s->compunit ()->blockvector ()->blocks ())
            {
              if (!b->function ())
                continue;

              if (!SYMBOL_MATCHES_NATURAL_NAME (b->function (),
                                                sym_name_str.c_str ())
                  && !symbol_matches_search_name (b->function (), lookup_name))
                continue;

              func_addr = cuda_skip_prologue (gdbarch, b->start ());
              return true;
            }
        }
    }

  /* If we didn't find a function, then it could be a non-debug
     compilation, so look at the msymtab. */
  for (minimal_symbol *msym : objfile->msymbols ())
    {
      if (!sym_name_str.compare (msym->linkage_name ()))
        {
          func_addr = msym->value_address (objfile);
          return true;
        }
    }
  return false;
}

/* Given a raw function name (string), find its corresponding text section
   vma. Returns true if found and stores the address in vma.  Returns false
   otherwise. */
bool
cuda_find_func_text_vma_from_objfile (struct objfile *objfile, char *func_name,
                                      CORE_ADDR *vma)
{
  struct obj_section *osect = NULL;
  asection *section = NULL;
  char *text_seg_name = NULL;

  gdb_assert (objfile);
  gdb_assert (func_name);
  gdb_assert (vma);

  /* Construct CUDA text segment name */
  text_seg_name = (char *)xmalloc (strlen (CUDA_ELF_TEXT_PREFIX)
                                   + strlen (func_name) + 1);
  strcpy (text_seg_name, CUDA_ELF_TEXT_PREFIX);
  strcat (text_seg_name, func_name);

  ALL_OBJFILE_OSECTIONS (objfile, osect)
  {
    section = osect->the_bfd_section;
    if (section)
      {
        if (!strcmp (section->name, text_seg_name))
          {
            /* Found - store address in vma */
            xfree (text_seg_name);
            *vma = section->vma;
            return true;
          }
      }
  }

  /* Not found */
  xfree (text_seg_name);
  return false;
}

/* Returns 1 when a value is stored in more than one register (long, double).
   Works with assumption that the compiler allocates consecutive registers
   for those cases.  */
static int
cuda_convert_register_p (struct gdbarch *gdbarch, int regnum,
                         struct type *type)
{
  return (int)(cuda_pc_regnum_p (gdbarch, regnum)
               || cuda_special_regnum_p (gdbarch, regnum));
}

/* Read a value of type TYPE from register REGNUM in frame FRAME, and
   return its contents in TO.  */
static int
cuda_register_to_value (frame_info_ptr frame, int regnum,
                        struct type *type, gdb_byte *to, int *optimizep,
                        int *unavailablep)
{
  struct gdbarch *gdbarch = get_frame_arch (frame);
  regmap_t regmap;

  /* cuda_frame_prev_pc() should be used to read PC */
  gdb_assert (!cuda_pc_regnum_p (gdbarch, regnum));

  *optimizep = *unavailablep = 0;

  if (cuda_special_regnum_p (gdbarch, regnum))
    {
      regmap = regmap_get_search_result ();
      cuda_special_register_to_value (regmap, frame, to);
      return 1;
    }

  get_frame_register (frame, regnum, to);

  return 1;
}

/* Write the contents FROM of a value of type TYPE into register
   REGNUM in frame FRAME.  */
static void
cuda_value_to_register (frame_info_ptr frame, int regnum,
                        struct type *type, const gdb_byte *from)
{
  struct gdbarch *gdbarch = get_frame_arch (frame);
  regmap_t regmap;

  /* Could not write PC */
  gdb_assert (!cuda_pc_regnum_p (gdbarch, regnum));

  if (cuda_special_regnum_p (gdbarch, regnum))
    {
      regmap = regmap_get_search_result ();
      cuda_value_to_special_register (regmap, frame, from);
      return;
    }

  /* Ignore attempts to modify error pc */
  if (cuda_error_pc_regnum_p (gdbarch, regnum))
    return;

  put_frame_register (frame, regnum, from);
}

static const gdb_byte *
cuda_breakpoint_from_pc (struct gdbarch *gdbarch, CORE_ADDR *pc, int *len)
{
  return NULL;
}

/* Implement the breakpoint_kind_from_pc gdbarch method.  */

static int
cuda_breakpoint_kind_from_pc (struct gdbarch *gdbarch, CORE_ADDR *pcptr)
{
  /* A place holder of gdbarch method breakpoint_kind_from_pc.   */
  return 0;
}

static CORE_ADDR
cuda_adjust_breakpoint_address (struct gdbarch *gdbarch, CORE_ADDR bpaddr)
{
  uint64_t adjusted_addr = bpaddr;

  auto module = cuda_state::find_module_by_address (bpaddr);
  if (module)
    {
      // This can fail if the bpaddr is unrelocated. In that case, we return
      // the original address
      try
        {
          cuda_debugapi::get_adjusted_code_address (module->context ()->dev_id (),
						    bpaddr,
						    &adjusted_addr,
						    CUDBG_ADJ_CURRENT_ADDRESS);
        }
      catch (const gdb_exception &ex)
        {
          adjusted_addr = bpaddr;
        }
    }
  return (CORE_ADDR)adjusted_addr;
}

/* This function is suitable for architectures that don't
   extend/override the standard siginfo structure.  */

static struct type *
cuda_linux_get_siginfo_type (struct gdbarch *gdbarch)
{
  return linux_get_siginfo_type_with_fields (gdbarch, 0);
}

static struct gdbarch *
cuda_gdbarch_init (struct gdbarch_info info, struct gdbarch_list *arches)
{
  /* If there is already a candidate, use it.  */
  arches = gdbarch_list_lookup_by_info (arches, &info);
  if (arches != NULL)
    return arches->gdbarch;

  /* Allocate space for the new architecture.  */
  cuda_gdbarch_tdep *tdep = new cuda_gdbarch_tdep;
  struct gdbarch *gdbarch = gdbarch_alloc (&info, tdep);

  /* Data types.  */
  set_gdbarch_char_signed (gdbarch, 0);
  set_gdbarch_ptr_bit (gdbarch, tdep->ptr_size);
  set_gdbarch_addr_bit (gdbarch, 64);
  set_gdbarch_short_bit (gdbarch, 16);
  set_gdbarch_int_bit (gdbarch, 32);
  set_gdbarch_long_bit (gdbarch, 64);
  set_gdbarch_long_long_bit (gdbarch, 64);
  set_gdbarch_float_bit (gdbarch, 32);
  set_gdbarch_double_bit (gdbarch, 64);
  set_gdbarch_long_double_bit (gdbarch, 128);
  set_gdbarch_float_format (gdbarch, floatformats_ieee_single);
  set_gdbarch_double_format (gdbarch, floatformats_ieee_double);
  set_gdbarch_long_double_format (gdbarch, floatformats_ieee_double);

  /* Registers and Memory */
  set_gdbarch_num_regs (gdbarch, tdep->num_regs);
  set_gdbarch_num_pseudo_regs (gdbarch, tdep->num_pseudo_regs);

  set_gdbarch_pc_regnum (gdbarch, tdep->pc_regnum);
  set_gdbarch_ps_regnum (gdbarch, -1);
  set_gdbarch_sp_regnum (gdbarch, -1);
  set_gdbarch_fp0_regnum (gdbarch, -1);

  set_gdbarch_dwarf2_reg_to_regnum (gdbarch, cuda_dwarf2_reg_to_regnum);

  set_gdbarch_pseudo_register_write (gdbarch, cuda_pseudo_register_write);
  set_gdbarch_pseudo_register_read (gdbarch, cuda_pseudo_register_read);

  set_gdbarch_read_pc (gdbarch, NULL);
  set_gdbarch_write_pc (gdbarch, NULL);

  set_gdbarch_register_name (gdbarch, cuda_register_name);
  set_gdbarch_register_type (gdbarch, cuda_register_type);
  set_gdbarch_register_reggroup_p (gdbarch, cuda_register_reggroup_p);

  set_gdbarch_print_float_info (gdbarch, default_print_float_info);
  set_gdbarch_print_vector_info (gdbarch, NULL);

  set_gdbarch_convert_register_p (gdbarch, cuda_convert_register_p);
  set_gdbarch_register_to_value (gdbarch, cuda_register_to_value);
  set_gdbarch_value_to_register (gdbarch, cuda_value_to_register);

  /* Pointers and Addresses */
  set_gdbarch_fetch_pointer_argument (gdbarch, NULL);

  /* Address Classes */
  set_gdbarch_address_class_name_to_type_flags (
      gdbarch, cuda_address_class_name_to_type_flags);
  set_gdbarch_address_class_type_flags_to_name (
      gdbarch, cuda_address_class_type_flags_to_name);
  set_gdbarch_address_class_type_flags (gdbarch,
                                        cuda_address_class_type_flags);

  /* CUDA - managed variables */
  set_gdbarch_elf_make_msymbol_special (gdbarch,
                                        cuda_elf_make_msymbol_special);

  /* Register Representation */
  /* Frame Interpretation */
  set_gdbarch_skip_prologue (gdbarch, cuda_skip_prologue);
  set_gdbarch_inner_than (gdbarch, core_addr_lessthan);
  set_gdbarch_frame_align (gdbarch, NULL);
  set_gdbarch_frame_red_zone_size (gdbarch, 0);
  set_gdbarch_frame_args_skip (gdbarch, 0);
  set_gdbarch_unwind_pc (gdbarch, cuda_unwind_pc);
  set_gdbarch_unwind_sp (gdbarch, NULL);
  set_gdbarch_frame_num_args (gdbarch, NULL);
  set_gdbarch_return_value (gdbarch, cuda_abi_return_value);
  frame_unwind_append_unwinder (gdbarch, &cuda_frame_unwind);
  frame_base_append_sniffer (gdbarch, cuda_frame_base_sniffer);
  frame_base_set_default (gdbarch, &cuda_frame_base);
  dwarf2_append_unwinders (gdbarch);
  dwarf2_frame_set_adjust_regnum (gdbarch, cuda_adjust_regnum);

  /* Inferior Call Setup */
  set_gdbarch_dummy_id (gdbarch, NULL);
  set_gdbarch_push_dummy_call (gdbarch, NULL);

  set_gdbarch_skip_permanent_breakpoint (gdbarch, NULL);
  set_gdbarch_fast_tracepoint_valid_at (gdbarch, NULL);
  set_gdbarch_decr_pc_after_break (gdbarch, 0);
  set_gdbarch_max_insn_length (gdbarch, 8);

  /* Instructions */
  set_gdbarch_print_insn (gdbarch, cuda_print_insn);
  set_gdbarch_relocate_instruction (gdbarch, NULL);
  set_gdbarch_breakpoint_from_pc (gdbarch, cuda_breakpoint_from_pc);
  set_gdbarch_breakpoint_kind_from_pc (gdbarch, cuda_breakpoint_kind_from_pc);
  set_gdbarch_adjust_breakpoint_address (gdbarch,
                                         cuda_adjust_breakpoint_address);

  /* CUDA - no address space management */
  set_gdbarch_has_global_breakpoints (gdbarch, 1);

  /* We hijack the linux siginfo type for the CUDA target on both Mac & Linux
   */
  set_gdbarch_get_siginfo_type (gdbarch, cuda_linux_get_siginfo_type);

  return gdbarch;
}

static void
cuda_iterate_over_regset_sections (struct gdbarch *gdbarch,
                                   iterate_over_regset_sections_cb *cb,
                                   void *cb_data,
                                   const struct regcache *regcache)
{
}

struct gdbarch *
cuda_get_gdbarch (void)
{
  static struct gdbarch *cuda_gdbarch = nullptr;

  if (!cuda_gdbarch)
    {
      struct gdbarch_info info;
      info.bfd_arch_info = bfd_lookup_arch (bfd_arch_m68k, 0);
      cuda_gdbarch = gdbarch_find_by_info (info);

      /* Core file support. */
      set_gdbarch_iterate_over_regset_sections (
          cuda_gdbarch, cuda_iterate_over_regset_sections);
    }

  return cuda_gdbarch;
}

static struct value *
cuda_constant_bank_addr_internal_fn(struct gdbarch *gdbarch,
			const struct language_defn *language,
			void *cookie, int argc, struct value **argv)
{
  if (argc != 2)
    error (_("This function requires two parameters (bank, offset)."));

  if (!cuda_debugapi::api_state_initialized())
    error (_("API isn't initialized yet."));

  return cuda_get_const_bank_address_val (gdbarch,
					  value_as_long (argv[0]),
					  value_as_long (argv[1]));
}

void _initialize_cuda_tdep ();
void
_initialize_cuda_tdep ()
{
  gdbarch_register (bfd_arch_m68k, cuda_gdbarch_init);

  add_internal_function ("_cuda_const_bank", _("\
$_cuda_const_bank - returns the GPU address of an offset within a constant bank.\n\
Usage: $_cuda_const_bank(bank, offset)\n"),
			 cuda_constant_bank_addr_internal_fn, NULL);

  gdb::observers::about_to_proceed.attach (cuda_sstep_about_to_proceed,
					   "cuda_sstep");
}

bool
cuda_is_cuda_gdbarch (struct gdbarch *arch)
{
  if (gdbarch_bfd_arch_info (arch)->arch == bfd_arch_m68k)
    return true;
  else
    return false;
}

/********* Session Management **********/

int
cuda_gdb_session_create (void)
{
  int ret = 0;
  bool override_umask = false;
  bool dir_exists = false;

  /* Check if the previous session was cleaned up */
  if (cuda_gdb_session_dir[0] != '\0')
    error (
        _ ("The directory for the previous CUDA session was not cleaned up. "
           "Try deleting %s and retrying."),
        cuda_gdb_session_dir);

  cuda_gdb_session_id++;

  snprintf (cuda_gdb_session_dir, CUDA_GDB_TMP_BUF_SIZE, "%s/session%d",
            cuda_gdb_tmpdir_getdir (), cuda_gdb_session_id);

  cuda_trace ("creating new session %d", cuda_gdb_session_id);

  ret = cuda_gdb_dir_create (cuda_gdb_session_dir, S_IRWXU | S_IRWXG,
                             override_umask, &dir_exists);

  if (!ret && dir_exists)
    error (_ ("A stale CUDA session directory was found. "
              "Try deleting %s and retrying."),
           cuda_gdb_session_dir);
  else if (ret)
    error (_ ("Failed to create session directory: %s (ret=%d)."),
           cuda_gdb_session_dir, ret);

  /* Change session folder ownership if debugging as root */
  if (getuid () == 0)
    cuda_gdb_chown_to_pid_uid (inferior_ptid.pid (), cuda_gdb_session_dir);
  return ret;
}

void
cuda_gdb_session_destroy (void)
{
  cuda_gdb_dir_cleanup_files (cuda_gdb_session_dir);

#ifndef __QNXTARGET__
  rmdir (cuda_gdb_session_dir);
#endif /* __QNXTARGET__ */

  memset (cuda_gdb_session_dir, 0, CUDA_GDB_TMP_BUF_SIZE);
}

uint32_t
cuda_gdb_session_get_id (void)
{
  return cuda_gdb_session_id;
}

const char *
cuda_gdb_session_get_dir (void)
{
  return cuda_gdb_session_dir;
}

/* Find out if the provided address is a GPU address, and if so adjust it. */
void
cuda_adjust_device_code_address (CORE_ADDR addr,
                                 CORE_ADDR *adjusted_addr)
{
  auto module = cuda_state::find_module_by_address (addr);
  if (module)
    cuda_debugapi::get_adjusted_code_address (module->context ()->dev_id (),
					      addr, &addr,
					      CUDBG_ADJ_CURRENT_ADDRESS);
  *adjusted_addr = (CORE_ADDR)addr;
}

void
cuda_next_device_code_address (CORE_ADDR addr,
                               CORE_ADDR *adjusted_addr)
{
  auto module = cuda_state::find_module_by_address (addr);
  if (module)
    cuda_debugapi::get_adjusted_code_address (module->context ()->dev_id (),
					      addr, &addr,
					      CUDBG_ADJ_NEXT_ADDRESS);
  *adjusted_addr = (CORE_ADDR)addr;
}

void
cuda_update_report_driver_api_error_flags (void)
{
  CORE_ADDR addr;
  CUDBGReportDriverApiErrorFlags flags;

  update_cuda_api_error_breakpoint ();

  addr = cuda_get_symbol_address (
      _STRING_ (CUDBG_REPORT_DRIVER_API_ERROR_FLAGS));
  flags = cuda_options_api_failures_break_on_nonfatal ()
              ? CUDBG_REPORT_DRIVER_API_ERROR_FLAGS_NONE
              : CUDBG_REPORT_DRIVER_API_ERROR_FLAGS_SUPPRESS_NOT_READY;
  if (!addr)
    return;

  target_write_memory (addr, (gdb_byte *)&flags, sizeof (flags));
}

/* Determine if symbol refers to the __device_stub_ function corresponding
   to the potential host shadow function 'name'.
*/
/*
  Found a device stub function. Extract the shadow function name from
  the device stub mangled name and mark it as a shadow function. Can't
  use language_demangle here as the format appears non-standard.

  Example: _Z38__device_stub__Z9acos_main10acosParamsR10acosParams
  */

static bool
cuda_is_kernel_launch_stub (const std::string &linkage_name,
                            std::string &bare_name, std::string &mangled_name)
{
  bare_name.clear ();
  mangled_name.clear ();

  /* We only want __device_stub_ functions, and not the wrapper ones
     Check early here to avoid expensive demangling */
  const char *find = "__device_stub_";
  auto len = strlen (find);
  if (linkage_name.find (find) == std::string::npos)
    {
      if (cuda_host_shadow_debug > 2)
        cuda_trace ("shadow: failed to find __device_stub_ %s",
                    linkage_name.c_str ());
      return false;
    }

  /* Demangle to get the mangled name of the CUDA kernel/function */
  auto demangled = language_demangle (language_def (language_cplus),
                                      linkage_name.c_str (), DMGL_ANSI);
  if (!demangled)
    {
      if (cuda_host_shadow_debug)
        cuda_trace ("shadow: failed to demangle %s", linkage_name.c_str ());
      return false;
    }

  std::string demangled_str{ demangled.get () };

  if (cuda_host_shadow_debug)
    cuda_trace ("shadow: checking %s in %s", linkage_name.c_str (),
                demangled_str.c_str ());

  /* Find the first instance of find */
  auto pos = demangled_str.find (find);
  if (pos == std::string::npos)
    {
      if (cuda_host_shadow_debug)
        cuda_trace ("shadow: failed to find %s in %s", find,
                    demangled_str.c_str ());
      return false;
    }

  // Return the mangled kernel name extracted from the linkage_name
  mangled_name = demangled_str.substr (pos + len);

  // Try to demangle again to get the bare name
  auto demangled2 = language_demangle (language_def (language_cplus),
                                       mangled_name.c_str (), DMGL_ANSI);
  if (demangled2)
    bare_name = demangled2.get ();

  if (cuda_host_shadow_debug)
    cuda_trace ("shadow: bare %s mangled %s", bare_name.c_str (),
                mangled_name.c_str ());

  return true;
}

/* Search the given objfile to find host shadow functions.
   Return true if any found, false otherwise. */
static void
cuda_find_objfile_host_shadow_minsyms (
    struct objfile *objfile, std::vector<std::string> &cuda_device_stubs)
{
  if (cuda_host_shadow_debug)
    cuda_trace ("shadow: scanning %s", objfile->original_name);

  for (auto msymbol : objfile->msymbols ())
    {
      auto linkage_name = msymbol->linkage_name ();
      gdb_assert (linkage_name);

      std::string bare_name;
      std::string mangled_name;
      auto has_symbol_name
          = cuda_is_kernel_launch_stub (linkage_name, bare_name, mangled_name);
      if (has_symbol_name)
        {
          if (cuda_host_shadow_debug)
            cuda_trace ("shadow: update minsym '%s' bare '%s' mangled '%s'",
                        msymbol->linkage_name (), bare_name.c_str (),
                        mangled_name.c_str ());

          if (msymbol && !msymbol->cuda_host_shadow_checked)
            {
              msymbol->cuda_host_shadow = 1;
              msymbol->cuda_host_shadow_checked = 1;
            }

          if (!mangled_name.empty ())
            {
              auto mangled_minsym = lookup_minimal_symbol_text (
                  mangled_name.c_str (), objfile);
              if (mangled_minsym.minsym)
                {
                  if (!mangled_minsym.minsym->cuda_host_shadow_checked)
                    {
                      mangled_minsym.minsym->cuda_host_shadow = 1;
                      if (cuda_host_shadow_debug)
                        cuda_trace ("shadow: host mangled minsym found %s",
                                    mangled_name.c_str ());
                      // Only check the minsym once
                      mangled_minsym.minsym->cuda_host_shadow_checked = 1;
                    }
                }
              cuda_device_stubs.push_back (mangled_name);
            }

          if (!bare_name.empty ())
            {
              auto bare_minsym
                  = lookup_minimal_symbol_text (bare_name.c_str (), objfile);
              if (bare_minsym.minsym)
                {
                  if (!bare_minsym.minsym->cuda_host_shadow_checked)
                    {
                      bare_minsym.minsym->cuda_host_shadow = 1;
                      if (cuda_host_shadow_debug)
                        cuda_trace ("shadow: host bare minsym found %s",
                                    bare_name.c_str ());
                      /* Only check the minsym once once */
                      bare_minsym.minsym->cuda_host_shadow_checked = 1;
                    }
                }
              cuda_device_stubs.push_back (bare_name);
            }
        }
    }
}

/* Search the given objfile to find host shadow functions.
   Return true if any found, false otherwise. */
void
cuda_find_objfile_host_shadow_functions (struct objfile *objfile)
{
  if (cuda_host_shadow_debug > 1)
    cuda_trace ("shadow: checking objfile %s", objfile->original_name);

  /* If we've alreasdy scanned, we're done */
  if (objfile->cuda_host_shadow_scan_complete)
    return;

  /* host shadow functions are only on the CPU side. */
  if (objfile->cuda_objfile)
    return;

  std::vector<std::string> cuda_device_stubs;
  cuda_find_objfile_host_shadow_minsyms (objfile, cuda_device_stubs);

  /* If no matching minsyms were found, we're done scanning this objfile */
  if (!cuda_device_stubs.size ())
    {
      if (cuda_host_shadow_debug)
        cuda_trace ("shadow: no cuda_device_stubs");
      objfile->cuda_host_shadow_scan_complete = true;
      return;
    }

  /* At least one host shadow function */
  objfile->cuda_host_shadow_found = true;

  if (cuda_host_shadow_debug)
    cuda_trace ("shadow: processing host queue %d",
                (int)cuda_device_stubs.size ());

  bool all_symbols_found = true;
  for (auto &host_shadow_name : cuda_device_stubs)
    {
      auto host_shadow_sym = cuda_lookup_symbol_in_objfile_from_linkage_name (
          objfile, host_shadow_name.c_str (), language_cplus, VAR_DOMAIN);
      if (host_shadow_sym.symbol)
        {
          host_shadow_sym.symbol->cuda_host_shadow = 1;
          if (cuda_host_shadow_debug)
            cuda_trace ("shadow: cuda host symbol %s",
                        host_shadow_name.c_str ());
          continue;
        }
      else
        {
          all_symbols_found = false;
          if (cuda_host_shadow_debug)
            cuda_trace ("shadow: cuda host symbol %s not found",
                        host_shadow_name.c_str ());
        }
    }

  /* If we found all the symbols, mark the objfile as fully scanned so we
   * don't repeat */
  if (all_symbols_found)
    objfile->cuda_host_shadow_scan_complete = true;
}
