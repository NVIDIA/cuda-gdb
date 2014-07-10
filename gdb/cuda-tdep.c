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

#include <time.h>
#include <sys/stat.h>
#include <ctype.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <signal.h>
#ifndef __ANDROID__
#include <execinfo.h>
#endif

#include "defs.h"
#include "arch-utils.h"
#include "command.h"
#include "dummy-frame.h"
#include "dwarf2-frame.h"
#include "doublest.h"
#include "floatformat.h"
#include "frame.h"
#include "frame-base.h"
#include "frame-unwind.h"
#include "inferior.h"
#include "gdbcmd.h"
#include "gdbcore.h"
#include "objfiles.h"
#include "osabi.h"
#include "regcache.h"
#include "regset.h"
#include "symfile.h"
#include "symtab.h"
#include "target.h"
#include "dis-asm.h"
#include "source.h"
#include "block.h"
#include "gdb/signals.h"
#include "gdbthread.h"
#include "language.h"
#include "demangle.h"
#include "main.h"
#include "target.h"
#include "valprint.h"
#include "user-regs.h"
#include "linux-tdep.h"
#include "exec.h"
#include "value.h"
#include "exceptions.h"
#include "breakpoint.h"
#include "reggroups.h"

#include "gdb_assert.h"
#include "gdb_string.h"

#include "cuda-asm.h"
#include "cuda-autostep.h"
#include "cuda-context.h"
#include "cuda-elf-image.h"
#include "cuda-frame.h"
#include "cuda-iterator.h"
#include "cuda-modules.h"
#include "cuda-notifications.h"
#include "cuda-options.h"
#include "cuda-special-register.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"
#include "cuda-textures.h"
#include "libbfd.h"
#include "mach-o.h"
#include "cuda-regmap.h"

/*----------------------------------------- Globals ---------------------------------------*/

struct gdbarch *cuda_gdbarch = NULL;
CuDim3 gridDim  = { 0, 0, 0};
CuDim3 blockDim = { 0, 0, 0};
bool cuda_remote = false;
bool cuda_initialized = false;
bool cuda_is_target_mourn_pending = false;
static bool inferior_in_debug_mode = false;
static char cuda_gdb_session_dir[CUDA_GDB_TMP_BUF_SIZE] = {0};
static uint32_t cuda_gdb_session_id = 0;

extern initialize_file_ftype _initialize_cuda_tdep;

/* For Mac OS X */
bool cuda_platform_supports_tid (void)
{
#if defined(__ANDROID__) || (defined(linux) && defined(SYS_gettid))
    return true;
#else
    return false;
#endif
}

int
cuda_gdb_get_tid (ptid_t ptid)
{
  if (cuda_platform_supports_tid ())
    return TIDGET (ptid);
  else
    return PIDGET (ptid);
}


/* CUDA - skip prologue */
bool cuda_producer_is_open64;

/* CUDA architecture specific information.  */
struct gdbarch_tdep
{
  int num_regs;
  int num_pseudo_regs;

  /* Pointer size: 32-bits on i686, 64-bits on x86_64. So that we do not need
     to have 2 versions of the cuda-tdep.c file */
  int ptr_size;

  /* Registers */

  /* always */
  int pc_regnum;
  /* ABI only */
  int sp_regnum;
  int first_rv_regnum;
  int last_rv_regnum;
  int rz_regnum;
  int max_reg_rv_size;

  /* Predicate Registers */
  int first_pred_regnum;
  int last_pred_regnum;

  /* CC register */
  int cc_regnum;

  /* ErrorPC register */
  int error_pc_regnum;

  /* Pseudo-Registers */

  /* Special register to indicate to look at the regmap search result */
  int special_regnum;

  /* Register number to tell the debugger that no valid register was found.
     Used to avoid returning errors and having nice warning messages and
     consistent garbage return values instead (zero sounds good).
     invalid_lo_regnum is used for variables that aren't live and are
     stored in a single register.  The combination of invalid_lo_regnum
     and invalid_hi_regnum is needed when a variable isn't live and is
     stored in multiple registers. */
  int invalid_lo_regnum;
  int invalid_hi_regnum;
};


/* Predicates for checking if register belongs to specified register group */
static inline
bool cuda_special_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  return regnum == tdep->special_regnum;
}

static inline
bool cuda_invalid_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  return regnum == tdep->invalid_lo_regnum ||
         regnum == tdep->invalid_hi_regnum;
}

static inline
bool cuda_pred_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  return regnum >= tdep->first_pred_regnum &&
         regnum <= tdep->last_pred_regnum;
}

static inline
bool cuda_pc_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  return regnum == tdep->pc_regnum;
}

static inline
bool cuda_cc_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  return regnum == tdep->cc_regnum;
}

static inline
bool cuda_error_pc_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  return regnum == tdep->error_pc_regnum;
}

int
cuda_abi_sp_regnum (struct gdbarch *gdbarch)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  return tdep->sp_regnum;
}

int
cuda_special_regnum (struct gdbarch *gdbarch)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  return tdep->special_regnum;
}

int
cuda_pc_regnum (struct gdbarch *gdbarch)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  return tdep->pc_regnum;
}

/* CUDA - siginfo */
static int cuda_signo = 0;

static void
cuda_siginfo_trace (const char *fmt, ...)
{
  va_list ap;

  va_start (ap,fmt);
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


/*---------------------------------------- Routines ---------------------------------------*/

int
cuda_inferior_word_size (void)
{
#ifdef __linux__
  gdb_assert (0); // not implemented
#else
  unsigned long cputype = bfd_mach_o_cputype (exec_bfd);
  switch (cputype)
    {
      case BFD_MACH_O_CPU_TYPE_I386:
        return 4;
      case BFD_MACH_O_CPU_TYPE_X86_64:
        return 8;
      default:
        error (_("Unsupported CPU type: 0x%lx\n"), cputype);
    }
#endif

  gdb_assert (0);
  return -1;
}

const char *
cuda_find_function_name_from_pc (CORE_ADDR pc, bool demangle)
{
  char *demangled = NULL;
  const char *name = NULL;
  static char *unknown = "??";
  struct symbol *kernel = NULL;
  enum language lang = language_unknown;
  struct minimal_symbol *msymbol = lookup_minimal_symbol_by_pc (pc);

  /* find the mangled name */
  kernel = find_pc_function (pc);
  if (kernel && msymbol != NULL &&
      SYMBOL_VALUE_ADDRESS (msymbol) > BLOCK_START (SYMBOL_BLOCK_VALUE (kernel)))
    {
      name = SYMBOL_LINKAGE_NAME (msymbol);
      lang = SYMBOL_LANGUAGE (msymbol);
    }
  else if (kernel)
    {
      name = SYMBOL_CUDA_NAME (kernel);
      lang = SYMBOL_LANGUAGE (kernel);
    }
  else if (msymbol != NULL)
    {
      name = SYMBOL_LINKAGE_NAME (msymbol);
      lang = SYMBOL_LANGUAGE (msymbol);
    }

  /* process the mangled name */
  if (!name)
    return unknown;
  else if (demangle)
    {
      demangled = language_demangle (language_def (lang), name, DMGL_ANSI);
      if (demangled)
        return demangled;
      else
        return name;
    }
  else
    return name;
}

ATTRIBUTE_PRINTF(2, 0) void
cuda_vtrace_domain (cuda_trace_domain_t domain, const char *fmt, va_list ap)
{
  if (!cuda_options_trace_domain_enabled (domain))
    return;
  switch (domain)
    {
      case CUDA_TRACE_EVENT:
        fprintf (stderr, "[CUDAGDB-EVT] ");
        break;
      case CUDA_TRACE_BREAKPOINT:
        fprintf (stderr, "[CUDAGDB-BPT] ");
        break;
      case CUDA_TRACE_API:
        fprintf (stderr, "[CUDAGDB-API] ");
        break;
      case CUDA_TRACE_TEXTURES:
        fprintf (stderr, "[CUDAGDB-TEX] ");
        break;
      case CUDA_TRACE_GENERAL:
        fprintf (stderr, "[CUDAGDB-GEN] ");
        break;
      default:
        fprintf (stderr, "[CUDAGDB] ");
        break;
    }
    vfprintf (stderr, fmt, ap);
    fprintf (stderr, "\n");
    fflush (stderr);
}

ATTRIBUTE_PRINTF (1, 2) void
cuda_trace (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_GENERAL, fmt, ap);
  va_end (ap);
}

ATTRIBUTE_PRINTF (2, 3) void
cuda_trace_domain (cuda_trace_domain_t domain, const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (domain, fmt, ap);
  va_end (ap);
}

bool
cuda_breakpoint_hit_p (cuda_clock_t clock)
{
  cuda_iterator itr;
  cuda_coords_t c, filter = CUDA_WILDCARD_COORDS;
  bool breakpoint_hit = false;

  itr = cuda_iterator_create (CUDA_ITERATOR_TYPE_LANES, &filter,
                               CUDA_SELECT_VALID | CUDA_SELECT_BKPT);

  for (cuda_iterator_start (itr);
       !cuda_iterator_end (itr);
       cuda_iterator_next (itr))
    {
      /* if we hit a breakpoint at an earlier time, we do not report it again. */
      c = cuda_iterator_get_current (itr);
      if (lane_get_timestamp (c.dev, c.sm, c.wp, c.ln) < clock)
        continue;

      breakpoint_hit = true;
      break;
    }

  cuda_iterator_destroy (itr);

  return breakpoint_hit;
}

/* Return the name of register REGNUM. */
static const char*
cuda_register_name (struct gdbarch *gdbarch, int regnum)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  uint32_t device_num_regs, sp_regnum, offset;
  static const int size = CUDA_GDB_TMP_BUF_SIZE;
  static char buf[CUDA_GDB_TMP_BUF_SIZE];
  int d, i;
  bool high;
  regmap_t regmap;

  /* Ignore registers not supported by this device */
  device_num_regs = device_get_num_registers (cuda_current_device ());

  /* Single SASS register */
  if (regnum < device_num_regs)
    {
      snprintf (buf, sizeof (buf), "R%d", regnum);
      return buf;
    }

  /* The PC register */
  if (cuda_pc_regnum_p (gdbarch, regnum))
      return "pc";

  /* Invalid register */
  if (cuda_invalid_regnum_p (gdbarch, regnum))
      return "(dummy internal register)";

  /* The special CUDA register: stored in the regmap. */
  if (cuda_special_regnum_p (gdbarch, regnum))
    {
      regmap = regmap_get_search_result ();
      cuda_special_register_name (regmap, buf, size);
      return buf;
    }

  /* Predicate registers */
  if (cuda_pred_regnum_p (gdbarch, regnum))
    {
      snprintf (buf, sizeof (buf), "P%d", regnum - tdep->first_pred_regnum);
      return buf;
    }

  /* CC register */
  if (cuda_cc_regnum_p (gdbarch, regnum))
    return "CC";

  /* The Error PC register */
  if (cuda_error_pc_regnum_p (gdbarch, regnum))
      return "errorpc";


  /* (everything else) */
  return NULL;
}


static struct type *
cuda_register_type (struct gdbarch *gdbarch, int regnum)
{
  if (cuda_special_regnum_p (gdbarch, regnum))
    return builtin_type (gdbarch)->builtin_int64;

  if (cuda_pc_regnum_p (gdbarch, regnum) || cuda_error_pc_regnum_p (gdbarch, regnum))
    return builtin_type (gdbarch)->builtin_func_ptr;

  return builtin_type (gdbarch)->builtin_int32;
}

/*
 * Copy the DWARF register string that represents a CUDA register
 */
bool
cuda_get_dwarf_register_string (reg_t reg, char *deviceReg, size_t sz)
{
  static const int size = sizeof (ULONGEST);
  char buffer[size+1];
  char *p = NULL;
  bool isDeviceReg = false;
  int i;

  /* Is reg a virtual device register or a host register? Let's look
     at the encoding to determine it. Example:

     as char[4]      as uint32_t
     device reg %r4 :      "\04r%"     0x00257234
     host reg  4    : "\0\0\0\004"     0x00000004

     Therefore, as long as the host uses less than 256 registers, we
     can safely assume that if the uint32_t value of reg is larger
     than 0xff and the first character is a '%', we are dealing with
     a virtual device register (a virtual device register is made of
     at least 2 characters). */
  if (reg > 0xff)
    {
      /* if this is a device register, the register name string,
         originally encoded as ULEB128, has been decoded as an
         unsigned integer. The order of the characters has to be
         reversed in order to be read as a standard string. */
      memset (buffer, 0, size+1);
      for (i = 0; i < size; ++i)
        {
          buffer[size-1-i] = reg & 0xff;
          reg = reg >> 8;
        }

      /* find the first character of the string */
      p = buffer;
      while (*p == 0)
        ++p;

      /* copy the result if we are dealing with a device register. */
      if (p[0] == '%')
        {
          isDeviceReg = true;
          if (deviceReg)
            strncpy (deviceReg, p, sz);
        }
    }

  return !isDeviceReg;
}

static regmap_t
cuda_get_physical_register (char *reg_name)
{
  uint64_t start_pc, addr, virt_addr;
  kernel_t kernel;
  const char *func_name = NULL;
  module_t module;
  elf_image_t elf_image;
  struct objfile *objfile;
  regmap_t regmap = NULL;
  struct symbol *symbol = NULL;
  CORE_ADDR func_start;
  struct frame_info *frame = NULL;

  gdb_assert (cuda_focus_is_device ());

  frame = get_selected_frame (NULL);
  virt_addr = get_frame_pc (frame);

  kernel = cuda_current_kernel ();
  module = kernel_get_module (kernel);
  elf_image = module_get_elf_image (module);
  objfile = cuda_elf_image_get_objfile (elf_image);
  symbol = find_pc_function ((CORE_ADDR) virt_addr);

  if (symbol)
    {
      find_pc_partial_function (virt_addr, NULL, &func_start, NULL);
      func_name = SYMBOL_CUDA_NAME (symbol);
      addr = virt_addr - func_start;
      regmap = regmap_table_search (objfile, func_name, reg_name, addr);
    }

  return regmap;
}

/*
 * Convert a CUDA DWARF register into a physical register index
 * Also return whether mapping live range was extrapolated
 */
int
cuda_reg_to_regnum_ex (struct gdbarch *gdbarch, reg_t reg, bool *extrapolated)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  int max_regs = gdbarch_num_regs (gdbarch) + gdbarch_num_pseudo_regs (gdbarch);
  int32_t regno, decoded_reg;
  uint32_t num_regs;
  char reg_name[8+1];
  regmap_t regmap;

  if (extrapolated)
    *extrapolated = false;

  /* The register is already decoded and reg_t is unsigned value */
  if (reg <= max_regs)
    return reg;

  /* The register is encoded with its register class. */
  if (!cuda_decode_physical_register (reg, &decoded_reg))
    return decoded_reg;

  /* Unrecognized register. */
  if (cuda_get_dwarf_register_string (reg, reg_name, sizeof (reg_name)))
    return -1;

  /* At this point, we know that the register is encoded as PTX register string */
  regmap = cuda_get_physical_register (reg_name);
  if (!regmap)
    return -1;

  num_regs = regmap_get_num_entries (regmap);
  if (extrapolated)
    *extrapolated = regmap_is_extrapolated (regmap);

  /* If we found a single SASS register, then we let cuda-gdb handle it
     normally. */
  if (num_regs == 1 && regmap_get_class (regmap, 0) == REG_CLASS_REG_FULL)
    {
      regno = regmap_get_register (regmap, 0);
      return regno;
    }

  /* Every situation that requires us to store data that cannot be represented
     as a single register index (regno). We keep hold of the data until the
     value is to be fetched. */
  if (cuda_special_register_p (regmap))
    {
      regno = tdep->special_regnum;
      return regno;
    }

  /* If no physical register was found or the register mapping is not useable,
     returns an invalid regnum to indicate it has been optimized out. */
  return -1;
}

static int
cuda_reg_to_regnum (struct gdbarch *gdbarch, reg_t reg)
{
  return cuda_reg_to_regnum_ex (gdbarch, reg, NULL);
}

static enum register_status
cuda_pseudo_register_read (struct gdbarch *gdbarch,
                           struct regcache *regcache,
                           int regnum,
                           gdb_byte *buf)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  uint32_t dev, sm, wp, ln;
  regmap_t regmap;

  cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);

  /* Invalid Register */
  if (cuda_invalid_regnum_p (gdbarch, regnum))
    {
      *((uint32_t*)buf) = 0U;
      return REG_UNKNOWN;
    }

  /* Any combination of SASS register, SP + offset, LMEM offset locations */
  if (cuda_special_regnum_p (gdbarch, regnum))
    {
      regmap = regmap_get_search_result ();
      cuda_special_register_read (regmap, buf);
      return REG_VALID;
    }

  /* Predicate register */
  if (cuda_pred_regnum_p (gdbarch, regnum))
    {
      *buf = lane_get_predicate (dev, sm, wp, ln, regnum - tdep->first_pred_regnum);
      return REG_VALID;
    }

  /* CC register*/
  if (cuda_cc_regnum_p (gdbarch, regnum))
    {
      *buf = lane_get_cc_register (dev, sm, wp, ln);
      return REG_VALID;
    }

  /* ErrorPC register */
  if (cuda_error_pc_regnum_p (gdbarch, regnum))
    {
      if (!warp_has_error_pc (dev, sm, wp))
        return REG_UNAVAILABLE;
      *(CORE_ADDR *)buf = warp_get_error_pc (dev, sm, wp);
      return REG_VALID;
    }

  /* single SASS register */
  *buf = lane_get_register (dev, sm, wp, ln, regnum);
  return REG_VALID;
}


static void
cuda_pseudo_register_write (struct gdbarch *gdbarch,
                            struct regcache *regcache,
                            int regnum,
                            const gdb_byte *buf)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  uint32_t dev, sm, wp, ln, val;
  bool bval;
  regmap_t regmap;

  cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);

  /* invalid register */
  if (cuda_invalid_regnum_p (gdbarch, regnum))
    {
      error (_("Invalid register."));
      return;
    }

  /* Any combination of SASS register, SP + offset, LMEM offset locations */
  if (cuda_special_regnum_p (gdbarch, regnum))
    {
      regmap = regmap_get_search_result ();
      cuda_special_register_write (regmap, buf);
      return;
    }

  /* Predicate register */
  if (cuda_pred_regnum_p (gdbarch, regnum))
    {
      bval = *(bool *)buf;
      lane_set_predicate (dev, sm, wp, ln, regnum - tdep->first_pred_regnum, bval);
      return;
    }

  /* CC register */
  if (cuda_cc_regnum_p (gdbarch, regnum))
    {
      val = *(uint32_t *)buf;
      lane_set_cc_register (dev, sm, wp, ln, val);
      return;
    }

  /* single SASS register */
  val = *(uint32_t*)buf;
  lane_set_register (dev, sm, wp, ln, regnum, val);
}

static int
cuda_register_reggroup_p (struct gdbarch *gdbarch, int regnum,
                          struct reggroup *group)
{
  /* Do not include special and invalid registers in any group */
  if (cuda_invalid_regnum_p (gdbarch, regnum) ||
      cuda_special_regnum_p (gdbarch, regnum))
    return 0;

  /* Include predicates and CC register in special and all register groups */
  if (cuda_pred_regnum_p (gdbarch, regnum)  ||
      cuda_cc_regnum_p (gdbarch, regnum)    ||
      cuda_error_pc_regnum_p (gdbarch, regnum))
    return group == system_reggroup || group == all_reggroup;

  return default_register_reggroup_p (gdbarch, regnum, group);
}

static int
cuda_print_insn (bfd_vma pc, disassemble_info *info)
{
  uint32_t inst_size;
  bool is_device_address;
  kernel_t kernel;
  const char * inst;

  if (!cuda_focus_is_device ())
    return 1;

  /* If this isn't a device address, don't bother */
  if (!cuda_is_device_code_address (pc))
    return 1;

  /* decode the instruction at the pc */
  kernel = cuda_current_kernel ();
  inst = kernel_disassemble (kernel, pc, &inst_size);

  if (inst)
    info->fprintf_func (info->stream, "%s", inst);
  else
    info->fprintf_func (info->stream, "Cannot disassemble instruction");

  return inst_size;
}

bool
cuda_is_device_code_address (CORE_ADDR addr)
{
  struct objfile *objfile;
  struct obj_section *osect = NULL;
  asection           *section = NULL;
  bool found_in_cpu_symtab = false;
  bool is_cuda_addr = false;

  /* Zero and (CORE_ADDR)-1 are CPU addresses */
  if (addr == 0 || addr == (CORE_ADDR)-1LL)
    return false;

  /*Iterate over all ELF sections */
  ALL_OBJSECTIONS(objfile, osect)
    {
      section = osect->the_bfd_section;
      if (!section) continue;
      /* Check if addr belongs to CUDA ELF */
      if (objfile->cuda_objfile)
        {
          /* Skip sections that do not have code */
          if (!(section->flags & SEC_CODE))
            continue;
          if (section->vma > addr || section->vma + section->size < addr)
            continue;
          return true;
        }

      /* Check if address belongs to one of the host ELFs */
      if (obj_section_addr(osect) <= addr && obj_section_endaddr(osect) > addr)
        found_in_cpu_symtab = true;
    }

  /* If address was found on CPU and wasn't found on any of the CUDA ELFs - return false */
  if (found_in_cpu_symtab)
    return false;

  /* Fallback to backend API call */
  cuda_api_is_device_code_address ((uint64_t)addr, &is_cuda_addr);
  return is_cuda_addr;
}

/*------------------------------------------------------------------------- */

/* Returns true if obfd points to a CUDA ELF object file
   (checked by machine type).  Otherwise, returns false. */
bool
cuda_is_bfd_cuda (bfd *obfd)
{
  return (obfd &&
          obfd->tdata.elf_obj_data &&
          obfd->tdata.elf_obj_data->elf_header &&
          obfd->tdata.elf_obj_data->elf_header->e_machine == EM_CUDA);
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
  else
    {
      unsigned int abiv = (elf_elfheader (obfd)->e_ident[EI_ABIVERSION]);
      if (CUDA_ELFOSABIV_16BIT <= abiv && abiv <= CUDA_ELFOSABIV_LATEST)
        {
          *abi_version = abiv;
          return true;
        }
      else
        {
          static bool questionAsked = false;
          if (!questionAsked)
            {
              printf_filtered ("CUDA ELF Image contains unknown ABI version: %d\n", abiv);
              printf_filtered ("This might happen while debugging JITed code" \
                               "using latest driver with older tools.\n" \
                               "Further debugging might not be reliable.");
              target_terminal_ours ();
              if (!nquery ("Are you sure you want to continue? "))
                  fatal ("CUDA ELF Image contains unknown ABI version");
              target_terminal_inferior ();
            }
          *abi_version = abiv;
          questionAsked = true;
          return true;
        }
    }

  return false;
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
  kernel_t    kernel;
  module_t    module;
  elf_image_t elf_image;

  if (!get_current_context ())
    return false;

  if (!cuda_focus_is_device ())
    return false;

  kernel    = cuda_current_kernel ();
  module    = kernel_get_module (kernel);
  elf_image = module_get_elf_image (module);
  gdb_assert (cuda_elf_image_is_loaded (elf_image));
  return cuda_elf_image_uses_abi (elf_image);
}

/* CUDA - breakpoints */
/* Like breakpoint_address_match, but gdbarch is a parameter. Required to
   evaluate gdbarch_has_global_breakpoints (gdbarch) in the right context. */
int
cuda_breakpoint_address_match (struct gdbarch *gdbarch,
                               struct address_space *aspace1, CORE_ADDR addr1,
                               struct address_space *aspace2, CORE_ADDR addr2)
{
  return ((gdbarch_has_global_breakpoints (gdbarch)
           || aspace1 == aspace2)
          && addr1 == addr2);
}

CORE_ADDR
cuda_get_symbol_address (char *name)
{
  struct minimal_symbol *sym = lookup_minimal_symbol (name, NULL, NULL);

/* CUDA - Mac OS X specific */
#ifdef target_check_is_objfile_loaded
  struct objfile *objfile;

  /* CUDA - MAC OS X specific
     We need to check that the object file is actually loaded into
     memory, rather than accessing a cached set of symbols. */
  if (sym && sym->ginfo.bfd_section && sym->ginfo.bfd_section->owner)
    {
      objfile = find_objfile_by_name (sym->ginfo.bfd_section->owner->filename, 1); /* 1 = exact match */
      if (objfile && target_check_is_objfile_loaded (objfile))
        return SYMBOL_VALUE_ADDRESS (sym);
    }
#else
  if (sym)
    return SYMBOL_VALUE_ADDRESS (sym);
#endif

  return 0;
}

uint64_t
cuda_get_last_driver_api_error_code ()
{
  CORE_ADDR error_code_addr;
  uint64_t res;

  error_code_addr = cuda_get_symbol_address (_STRING_(CUDBG_REPORTED_DRIVER_API_ERROR_CODE));
  if (!error_code_addr)
    {
      /* This should never happen. If we hit the breakpoint we should have the symbol */
      error (_("Cannot retrieve the last driver API error code.\n"));
    }

  target_read_memory (error_code_addr, (char *)&res, sizeof (uint64_t));
  return res;
}

static uint64_t
cuda_get_last_driver_api_error_func_name_size ()
{
  CORE_ADDR error_func_name_size_addr;
  uint64_t res;

  error_func_name_size_addr = cuda_get_symbol_address (
    _STRING_(CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_SIZE));
  if (!error_func_name_size_addr)
    {
      /* This should never happen. If we hit the breakpoint we should have the symbol */
      error (_("Cannot retrieve the last driver API error function name size.\n"));
    }

  target_read_memory (error_func_name_size_addr, (char *)&res, sizeof (uint64_t));
  return res;
}

void
cuda_get_last_driver_api_error_func_name (char **name)
{
  CORE_ADDR error_func_name_core_addr;
  uint64_t error_func_name_addr;
  char *func_name = NULL;
  uint32_t size = 0U;

  size = cuda_get_last_driver_api_error_func_name_size ();
  if (!size)
    {
      /* This should never happen. If we hit the breakpoint we should have the symbol */
      error (_("Cannot retrieve the last driver API error function name size.\n"));
    }

  func_name = (char *)xcalloc(sizeof*func_name, size);
  if (!func_name)
    {
      /* Buffer for function name string should be created successfully  */
      error (_("Cannot allocate memory to save the reported function name.\n"));
    }

  error_func_name_core_addr = cuda_get_symbol_address (
    _STRING_(CUDBG_REPORTED_DRIVER_API_ERROR_FUNC_NAME_ADDR));
  if (!error_func_name_core_addr)
    {
      /* This should never happen. If we hit the breakpoint we should have the symbol */
      error (_("Cannot retrieve the last driver API error function name addr.\n"));
    }

  target_read_memory (error_func_name_core_addr, (char *)&error_func_name_addr, sizeof (uint64_t));
  target_read_memory (error_func_name_addr, func_name, size);
  if (!func_name[0])
    {
      /* This should never happen. If we hit the breakpoint we should have the symbol */
      error (_("Cannot retrieve the last driver API error function name.\n"));
    }
  *name = func_name;
}

uint64_t
cuda_get_last_driver_internal_error_code ()
{
  CORE_ADDR error_code_addr;
  uint64_t res;

  error_code_addr = cuda_get_symbol_address (_STRING_(CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE));
  if (!error_code_addr)
    {
      /* This should never happen. If we hit the breakpoint we should have the symbol */
      error (_("Cannot retrieve the last driver internal error code.\n"));
    }

  target_read_memory (error_code_addr, (char *)&res, sizeof (uint64_t));
  return res;
}

static void
cuda_signal_handler (int signo)
{
  void *buffer[100];
  int n;
#ifndef __ANDROID__
  psignal (signo, "Error: received unexpected signal");

  n = backtrace (buffer, sizeof buffer);
  fprintf (stderr, "BACKTRACE (%d frames):\n", n);
  backtrace_symbols_fd (buffer, n, STDERR_FILENO);
  fflush (stderr);
#else
  fprintf (stderr, "Error: received unexpected signal %d\n", signo);
#endif

  cuda_cleanup ();

  exit (1);
}

void
cuda_signals_initialize (void)
{
  signal (SIGSEGV, cuda_signal_handler);
  signal (SIGPIPE, cuda_signal_handler);
}

void
cuda_cleanup ()
{
  cuda_trace ("cuda_cleanup");

  set_current_context (NULL);
  cuda_auto_breakpoints_cleanup_breakpoints ();
  cuda_system_cleanup_breakpoints ();
  cuda_cleanup_cudart_symbols ();
  cuda_cleanup_tex_maps ();
  cuda_coords_invalidate_current ();
  cuda_system_cleanup_contexts ();
  if (cuda_initialized)
    cuda_system_finalize ();
  cuda_sstep_reset (false);

  /* In remote session, these functions are called on server side by cuda_linux_mourn() */
  if (!cuda_remote)
    {
      cuda_api_finalize ();
      /* Notification reset must be called after notification thread has
       * been terminated, which is done as part of cuda_api_finalize() call.
       */
      cuda_notification_reset ();

    }

  inferior_in_debug_mode = false;
  cuda_initialized = false;
}

void
cuda_final_cleanup (void *unused)
{
  if (cuda_initialized)
    cuda_api_finalize ();
}

void
cuda_initialize_driver_api_error_report (void)
{
  //CORE_ADDR functionAddr;
  CORE_ADDR functionEntryAddr =
    cuda_get_symbol_address (_STRING_(CUDBG_REPORT_DRIVER_API_ERROR));
  if (functionEntryAddr)
    {
      CORE_ADDR errorCodeAddr =
        cuda_get_symbol_address (_STRING_(CUDBG_REPORTED_DRIVER_API_ERROR_CODE));
      //target_read_memory (functionEntryAddr, (char *)&functionAddr, sizeof (CORE_ADDR));
      if (errorCodeAddr) 
        /* Create an internal breakpoint at the entry */
        create_cuda_driver_api_error_breakpoint (get_current_arch (),
                                                 functionEntryAddr);
      else
        warning (_("Cannot find memory location of driver API error code."));
    }
  else
    warning (_("Cannot find function entry for driver API error report."));
}

void
cuda_initialize_driver_internal_error_report (void)
{
  //CORE_ADDR functionAddr;
  CORE_ADDR functionEntryAddr =
    cuda_get_symbol_address (_STRING_(CUDBG_REPORT_DRIVER_INTERNAL_ERROR));
  if (functionEntryAddr)
    {
      CORE_ADDR errorCodeAddr =
        cuda_get_symbol_address (_STRING_(CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE));
      //target_read_memory (functionEntryAddr, (char *)&functionAddr, sizeof (CORE_ADDR));
      if (errorCodeAddr)
        /* Create an internal breakpoint at the entry */
        create_cuda_driver_internal_error_breakpoint (get_current_arch (),
                                                      functionEntryAddr);
      else
        warning (_("Cannot find memory location of driver internal error code."));
    }
  else
    warning (_("Cannot find function entry for driver internal error report."));
}

static void
cuda_initialize_uvm_detection (void)
{
  CORE_ADDR memAllocManagedAddr = cuda_get_symbol_address ("cuMemAllocManaged");
  if (!memAllocManagedAddr)
    {
      warning (_("Cannot find cuMemAllocManaged() routine address."));
      return;
    }
   create_cuda_uvm_breakpoint (get_current_arch(), memAllocManagedAddr);
}

/* Initialize the CUDA debugger API and collect the static data about
   the devices. Once per application run. */
static void
cuda_initialize (void)
{
  if (cuda_initialized)
    return;

  cuda_initialized = !cuda_api_initialize ();
  if (cuda_initialized)
    cuda_system_initialize ();
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
  CORE_ADDR memcheckAddr;
  CORE_ADDR launchblockingAddr;
  CORE_ADDR preemptionAddr;

  uint32_t pid;
  uint32_t apiClientRev = CUDBG_API_VERSION_REVISION;
  uint32_t sessionId;

  if (inferior_in_debug_mode)
  {
    cuda_initialize ();
    return true;
  }

  debugFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_IPC_FLAG_NAME));
  if (!debugFlagAddr)
    return false;

  /* Initialize cuda utils, check if cuda-gdb lock is busy */
  cuda_utils_initialize ();
  cuda_signals_initialize ();
  cuda_api_set_notify_new_event_callback (cuda_notification_notify);
  cuda_initialize ();
  sessionId = cuda_gdb_session_get_id ();

  /* When attaching or detaching, cuda_nat_attach() and
     cuda_nat_detach() control the setting of this flag,
     so don't touch it here. */
  if (CUDA_ATTACH_STATE_IN_PROGRESS != cuda_api_get_attach_state () &&
      CUDA_ATTACH_STATE_DETACHING != cuda_api_get_attach_state ())
    cuda_write_bool (debugFlagAddr, true);

  pid = getpid ();
  rpcFlagAddr        = cuda_get_symbol_address (_STRING_(CUDBG_RPC_ENABLED));
  gdbPidAddr         = cuda_get_symbol_address (_STRING_(CUDBG_APICLIENT_PID));
  apiClientRevAddr   = cuda_get_symbol_address (_STRING_(CUDBG_APICLIENT_REVISION));
  sessionIdAddr      = cuda_get_symbol_address (_STRING_(CUDBG_SESSION_ID));
  memcheckAddr       = cuda_get_symbol_address (_STRING_(CUDBG_ENABLE_INTEGRATED_MEMCHECK));
  launchblockingAddr = cuda_get_symbol_address (_STRING_(CUDBG_ENABLE_LAUNCH_BLOCKING));
  preemptionAddr     = cuda_get_symbol_address (_STRING_(CUDBG_ENABLE_PREEMPTION_DEBUGGING));

  if (!(rpcFlagAddr && gdbPidAddr && apiClientRevAddr &&
      sessionIdAddr && memcheckAddr && launchblockingAddr &&
      preemptionAddr))
    {
#ifdef __APPLE__
      kill (inferior_ptid.pid, SIGKILL);
#else
      kill (inferior_ptid.lwp, SIGKILL);
#endif
      error (_("CUDA application cannot be debugged. The CUDA driver is not compatible."));
    }

  target_write_memory (gdbPidAddr             , (char*)&pid, sizeof (pid));
  cuda_write_bool     (rpcFlagAddr            , true);
  target_write_memory (apiClientRevAddr       , (char*)&apiClientRev, sizeof(apiClientRev));
  target_write_memory (sessionIdAddr          , (char*)&sessionId, sizeof(sessionId));
  cuda_write_bool     (memcheckAddr           , cuda_options_memcheck ());
  cuda_write_bool     (launchblockingAddr     , cuda_options_launch_blocking ());
  cuda_write_bool     (preemptionAddr         , cuda_options_software_preemption ());
  cuda_update_report_driver_api_error_flags ();

  inferior_in_debug_mode = true;
  cuda_initialize_driver_api_error_report ();
  cuda_initialize_driver_internal_error_report ();
  cuda_initialize_uvm_detection ();
  return true;
}

bool
cuda_inferior_in_debug_mode (void)
{
  return inferior_in_debug_mode;
}

void
cuda_inferior_update_suspended_devices_mask (void)
{
  CORE_ADDR suspendedDevicesMaskAddr;
  unsigned int suspendedDevicesMask = cuda_system_get_suspended_devices_mask ();

  suspendedDevicesMaskAddr = cuda_get_symbol_address (_STRING_(CUDBG_DETACH_SUSPENDED_DEVICES_MASK));

  if (!suspendedDevicesMaskAddr)
    error (_("Failed to get suspended devices mask."));

  target_write_memory (suspendedDevicesMaskAddr, (char*)&suspendedDevicesMask,
                       sizeof(suspendedDevicesMask));
}

int
cuda_address_class_type_flags (int byte_size, int addr_class)
{
  switch (addr_class)
    {
      case ptxCodeStorage:    return TYPE_INSTANCE_FLAG_CUDA_CODE;
      case ptxConstStorage:   return TYPE_INSTANCE_FLAG_CUDA_CONST;
      case ptxGenericStorage: return TYPE_INSTANCE_FLAG_CUDA_GENERIC;
      case ptxGlobalStorage:  return TYPE_INSTANCE_FLAG_CUDA_GLOBAL;
      case ptxParamStorage:   return TYPE_INSTANCE_FLAG_CUDA_PARAM;
      case ptxSharedStorage:  return TYPE_INSTANCE_FLAG_CUDA_SHARED;
      case ptxTexStorage:     return TYPE_INSTANCE_FLAG_CUDA_TEX;
      case ptxLocalStorage:   return TYPE_INSTANCE_FLAG_CUDA_LOCAL;
      case ptxRegStorage:     return TYPE_INSTANCE_FLAG_CUDA_REG;
      default:                return 0;
    }
}

static const char *
cuda_address_class_type_flags_to_name (struct gdbarch *gdbarch, int type_flags)
{
  switch (type_flags & TYPE_INSTANCE_FLAG_CUDA_ALL)
    {
    case TYPE_INSTANCE_FLAG_CUDA_CODE:    return "code";
    case TYPE_INSTANCE_FLAG_CUDA_CONST:   return "constant";
    case TYPE_INSTANCE_FLAG_CUDA_GENERIC: return "generic";
    case TYPE_INSTANCE_FLAG_CUDA_GLOBAL:  return "global";
    case TYPE_INSTANCE_FLAG_CUDA_PARAM:   return "parameter";
    case TYPE_INSTANCE_FLAG_CUDA_SHARED:  return "shared";
    case TYPE_INSTANCE_FLAG_CUDA_TEX:     return "texture";
    case TYPE_INSTANCE_FLAG_CUDA_LOCAL:   return "local";
    case TYPE_INSTANCE_FLAG_CUDA_REG:     return "register";
    case TYPE_INSTANCE_FLAG_CUDA_MANAGED: return "managed_global";
    default:                              return "unknown_segment";
    }
}

static int
cuda_address_class_name_to_type_flags (struct gdbarch *gdbarch,
                                       const char *name,
                                       int *type_flags)
{
  if (strcmp (name, "code") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_CODE;
      return 1;
    }

  if (strcmp (name, "constant") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_CONST;
      return 1;
    }

  if (strcmp (name, "generic") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_GENERIC;
      return 1;
    }

  if (strcmp (name, "global") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_GLOBAL;
      return 1;
    }

  if (strcmp (name, "managed_global") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_MANAGED;
      return 1;
    }

  if (strcmp (name, "parameter") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_PARAM;
      return 1;
    }

  if (strcmp (name, "shared") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_SHARED;
      return 1;
    }

  if (strcmp (name, "texture") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_TEX;
      return 1;
    }

  if (strcmp (name, "register") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_REG;
      return 1;
    }

  if (strcmp (name, "local") == 0)
    {
      *type_flags = TYPE_INSTANCE_FLAG_CUDA_LOCAL;
      return 1;
    }

  return 0;
}

void
cuda_print_lmem_address_type (void)
{
  struct gdbarch *gdbarch = get_current_arch ();
  const char *lmem_type =
    cuda_address_class_type_flags_to_name (gdbarch,
                                           TYPE_INSTANCE_FLAG_CUDA_LOCAL);

  printf_filtered ("(@%s unsigned *) ", lmem_type);
}

#define STO_CUDA_MANAGED      4     /* CUDA - managed variables */
#define STO_CUDA_ENTRY     0x10     /* CUDA - break_on_launch */

VEC(CORE_ADDR) *cuda_kernel_entry_addresses = NULL;

static void
cuda_elf_make_msymbol_special (asymbol *sym, struct minimal_symbol *msym)
{
  /* managed variables */
  if (((elf_symbol_type *) sym)->internal_elf_sym.st_other == STO_CUDA_MANAGED)
    {
      MSYMBOL_TARGET_FLAG_1(msym) = 1;
      cuda_set_uvm_used (true);
    }

  /* break_on_launch */
  if (((elf_symbol_type *) sym)->internal_elf_sym.st_other == STO_CUDA_ENTRY)
    VEC_safe_push (CORE_ADDR, cuda_kernel_entry_addresses, SYMBOL_VALUE_ADDRESS(msym));
}

/* Temporary: intercept memory addresses when accessing known
   addresses pointing to CUDA RT variables. Returns 0 if found a CUDA
   RT variable, and 1 otherwise. */
static int
read_cudart_variable (uint64_t address, void * buffer, unsigned amount)
{
  CuDim3 thread_idx, block_dim;
  CuDim3 block_idx, grid_dim;
  uint32_t num_lanes;
  uint32_t dev_id, sm_id, wp_id, ln_id;
  struct {short x,y,z;} short_thread_idx, short_block_dim;
  struct {short x,y,z;} short_block_idx, short_grid_dim;
  kernel_t kernel;

  if (address < CUDBG_BUILTINS_MAX)
    return 1;

  if (!cuda_focus_is_device ())
    return 1;

  cuda_coords_get_current_physical (&dev_id, &sm_id, &wp_id, &ln_id);

  if (CUDBG_THREADIDX_OFFSET <= address)
    {
      thread_idx = lane_get_thread_idx (dev_id, sm_id, wp_id, ln_id);
      short_thread_idx.x = (short) thread_idx.x;
      short_thread_idx.y = (short) thread_idx.y;
      short_thread_idx.z = (short) thread_idx.z;
      memcpy (buffer, (char*)&short_thread_idx +
             (int64_t)address - CUDBG_THREADIDX_OFFSET, amount);
    }
  else if (CUDBG_BLOCKIDX_OFFSET <= address)
    {
      block_idx = warp_get_block_idx (dev_id, sm_id, wp_id);
      short_block_idx.x = (short) block_idx.x;
      short_block_idx.y = (short) block_idx.y;
      short_block_idx.z = (short) block_idx.z;
      memcpy (buffer, (char*)&short_block_idx
             + (int64_t)address - CUDBG_BLOCKIDX_OFFSET, amount);
    }
  else if (CUDBG_BLOCKDIM_OFFSET <= address)
    {
      kernel = warp_get_kernel (dev_id, sm_id, wp_id);
      block_dim = kernel_get_block_dim (kernel);
      short_block_dim.x = (short) block_dim.x;
      short_block_dim.y = (short) block_dim.y;
      short_block_dim.z = (short) block_dim.z;
      memcpy (buffer, (char*)&short_block_dim
             + (int64_t)address - CUDBG_BLOCKDIM_OFFSET, amount);
    }
  else if (CUDBG_GRIDDIM_OFFSET <= address)
    {
      kernel = warp_get_kernel (dev_id, sm_id, wp_id);
      grid_dim = kernel_get_grid_dim (kernel);
      short_grid_dim.x = (short) grid_dim.x;
      short_grid_dim.y = (short) grid_dim.y;
      short_grid_dim.z = (short) grid_dim.z;
      memcpy (buffer, (char*)&short_grid_dim
             + (int64_t)address - CUDBG_GRIDDIM_OFFSET, amount);
    }
  else if (CUDBG_WARPSIZE_OFFSET <= address)
    {
      dev_id    = cuda_current_device ();
      num_lanes = device_get_num_lanes (dev_id);
      memcpy (buffer, &num_lanes, amount);
    }
  else
    return 1;

  return 0;
}

/* Read LEN bytes of CUDA memory at address ADDRESS, placing the
   result in GDB's memory at BUF. Returns 0 on success, and 1
   otherwise. This is used only by partial_memory_read. */
int
cuda_read_memory_partial (CORE_ADDR address, gdb_byte *buf, int len, struct type *type)
{
  int flag;
  uint32_t dev, sm, wp, ln;
  uint32_t tex_id, dim;
  uint32_t *coords;
  bool is_bindless;

  /* No CUDA. Return 1 */
  if (!cuda_debugging_enabled)
    return 1;

  /* Sanity */
  gdb_assert (type);

  /* If address is marked as belonging to a CUDA memory segment, use the
     appropriate API call. */
  flag = TYPE_CUDA_ALL(type);
  if (flag)
    {
      cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);

      if (TYPE_CUDA_CODE(type))
        cuda_api_read_code_memory (dev, address, buf, len);
      else if (TYPE_CUDA_CONST(type))
        cuda_api_read_const_memory (dev, address, buf, len);
      else if (TYPE_CUDA_GENERIC(type))
        cuda_api_read_generic_memory (dev, sm, wp, ln, address, buf, len);
      else if (TYPE_CUDA_GLOBAL(type))
        cuda_api_read_generic_memory (dev, sm, wp, ln, address, buf, len);
      else if (TYPE_CUDA_PARAM(type))
        cuda_api_read_param_memory (dev, sm, wp, address, buf, len);
      else if (TYPE_CUDA_SHARED(type))
        cuda_api_read_shared_memory (dev, sm, wp, address, buf, len);
      else if (TYPE_CUDA_TEX(type))
        {
          cuda_texture_dereference_tex_contents (address, &tex_id, &dim, &coords, &is_bindless);
          if (is_bindless)
            cuda_api_read_texture_memory_bindless (dev, sm, wp, tex_id, dim, coords, buf, len);
          else
            cuda_api_read_texture_memory (dev, sm, wp, tex_id, dim, coords, buf, len);
        }
      else if (TYPE_CUDA_LOCAL(type))
        cuda_api_read_local_memory (dev, sm, wp, ln, address, buf, len);
      else
        error (_("Unknown storage specifier."));
      return 0;
    }

  return 1;
}

/* If there is an address class associated with this value, we've
   stored it in the type.  Check this here, and if set, read from the
   appropriate segment. */
void
cuda_read_memory (CORE_ADDR address, struct value *val, struct type *type, int len)
{
  volatile struct gdb_exception e;
  uint32_t dev, sm, wp, ln;
  gdb_byte *buf = value_contents_all_raw (val);

  /* No CUDA. Read the host memory */
  if (!cuda_debugging_enabled)
    goto bailout;

  cuda_set_host_address_resident_on_gpu (false);
  if (!cuda_focus_is_device())
    {
      if (!TYPE_CUDA_GLOBAL(type))
        goto bailout;
      cuda_api_read_global_memory ((uint64_t)address, buf, len);
      return;
    }

  /* Textures: read tex contents now and dereference the contents on the second
     call to cuda_read_memory. See below. */
  if (IS_TEXTURE_TYPE (type) || cuda_texture_is_tex_ptr (type))
    {
      cuda_texture_read_tex_contents (address, buf);
      return;
    }

  /* Call the partial memory read. Return on success */
  if (cuda_read_memory_partial (address, buf, len, type) == 0)
    return;

   /* The variable is on the stack. It happens when not in the innermost frame. */
  if (value_stack (val))
   {
      cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);
      cuda_api_read_local_memory (dev, sm, wp, ln, address, buf, len);
      return;
   }

  /* If address of a built-in CUDA runtime variable, intercept it */
  if (!read_cudart_variable (address, buf, len))
    return;

  /* Default: read the host memory as usual */
bailout:
  if (value_stack (val))
    {
      read_stack (address, buf, len);
      return;
    }
  TRY_CATCH (e, RETURN_MASK_ALL)
    {
      read_memory (address, buf, len);
    }
  if (e.reason == 0) return;
  if (!cuda_managed_address_p (address))
    throw_exception (e);
  cuda_api_read_global_memory ((uint64_t)address, buf, len);
  cuda_set_host_address_resident_on_gpu (true);
}

/* FIXME: This is to preserve the symmetry of cuda_read/write_memory_partial. */
int
cuda_write_memory_partial (CORE_ADDR address, const gdb_byte *buf, struct type *type)
{
  int flag;
  uint32_t dev, sm, wp, ln;
  int len = TYPE_LENGTH (type);

  /* No CUDA. Return 1. */
  if (!cuda_debugging_enabled)
    return 1;

  /* If address is marked as belonging to a CUDA memory segment, use the
     appropriate API call. */
  flag = TYPE_CUDA_ALL(type);
  if (flag)
    {
      cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);

      if (TYPE_CUDA_REG(type))
        {
          /* The following explains how we can come down this path, and why
             cuda_api_write_local_memory is called when the address class
             indicates ptxRegStorage.

             We should only enter this case if we are:
                 1. debugging an application that is using the ABI
                 2. modifying a variable that is mapped to a register that has
                    been saved on the stack
                 3. not modifying a variable for the _innermost_ device frame
                    (as this would follow the cuda_pseudo_register_write path).

             We can possibly add additional checks to ensure that address is
             within the permissable stack range, but cuda_api_write_local_memory
             better return an appropriate error in that case anyway, so let's
             test the API.

             Note there is no corresponding case in cuda_read_memory_with_valtype,
             because _reading_ a previous frame's (saved) registers is all done
             directly by prev register methods (dwarf2-frame.c, cuda-tdep.c).

             As an alternative, we could intercept the value type prior to
             reaching this function and change it to ptxLocalStorage, but that
             can make debugging somewhat difficult. */
          gdb_assert (cuda_current_active_elf_image_uses_abi ());
          cuda_api_write_local_memory (dev, sm, wp, ln, address, buf, len);
        }
      else if (TYPE_CUDA_GENERIC(type))
        cuda_api_write_generic_memory (dev, sm, wp, ln, address, buf, len);
      else if (TYPE_CUDA_GLOBAL(type))
        cuda_api_write_generic_memory (dev, sm, wp, ln, address, buf, len);
      else if (TYPE_CUDA_PARAM(type))
        cuda_api_write_param_memory (dev, sm, wp, address, buf, len);
      else if (TYPE_CUDA_SHARED(type))
        cuda_api_write_shared_memory (dev, sm, wp, address, buf, len);
      else if (TYPE_CUDA_LOCAL(type))
        cuda_api_write_local_memory (dev, sm, wp, ln, address, buf, len);
      else if (TYPE_CUDA_TEX(type))
        error (_("Writing to texture memory is not allowed."));
      else if (TYPE_CUDA_CODE(type))
        error (_("Writing to code memory is not allowed."));
      else if (TYPE_CUDA_CONST(type))
        error (_("Writing to constant memory is not allowed."));
      else
        error (_("Unknown storage specifier."));
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
  volatile struct gdb_exception e;
  int len = TYPE_LENGTH (type);

  /* No CUDA. Write the host memory */
  if (!cuda_debugging_enabled)
    {
      write_memory (address, buf, len);
      return;
    }

  /* Call the partial memory write, return on success */
  if (cuda_write_memory_partial (address, buf, type) == 0)
    return;

  /* Default: write the host memory as usual */
  TRY_CATCH (e, RETURN_MASK_ALL)
    {
      write_memory (address, buf, len);
    }
  if (e.reason == 0) return;
  /* CUDA - managed memory */
  if (!cuda_managed_address_p (address))
    throw_exception (e);
  cuda_api_write_global_memory ((uint64_t)address, buf, len);
}

/* Single-Stepping

   The following data structures and routines are used as a framework to manage
   single-stepping with CUDA devices. It currently solves 2 issues

   1. When single-stepping a warp, we do not want to resume the host if we do
   not have to. The single-stepping framework allows for making GDB believe
   that everything was resumed and that a SIGTRAP was received after each step.

   2. When single-stepping a warp, other warps may be required to be stepped
   too. Out of convenience to the user, we want to keep single-stepping those
   other warps alongside the warp in focus. By doing so, stepping over a
   __syncthreads() instruction will bring all the warps in the same block to
   the next source line.

   This result is achieved by marking the warps we want to single-step with a
   warp mask. When the user issues a new command, the warp is initialized
   accordingly. If the command is a step command, we initialize the warp mask
   with the warp mask and let the mask grow over time as stepping occurs (there
   might be more than one step). If the command is not a step command, the warp
   mask is set empty and will remain that way. In that situation, if
   single-stepping is required, only the minimum number of warps will be
   single-stepped. */

static struct {
  bool     active;
  ptid_t   ptid;
  uint32_t dev_id;
  uint32_t sm_id;
  uint32_t wp_id;
  uint64_t grid_id;
  bool     grid_id_valid;
  uint64_t warp_mask;
} cuda_sstep_info;

bool
cuda_sstep_is_active (void)
{
  return cuda_sstep_info.active;
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
  return cuda_sstep_info.dev_id;
}

uint32_t
cuda_sstep_sm_id (void)
{
  gdb_assert (cuda_sstep_info.active);
  return cuda_sstep_info.sm_id;
}

uint32_t
cuda_sstep_wp_id (void)
{
  gdb_assert (cuda_sstep_info.active);
  return cuda_sstep_info.wp_id;
}

uint64_t
cuda_sstep_wp_mask (void)
{
  gdb_assert (cuda_sstep_info.active);
  return cuda_sstep_info.warp_mask;
}

uint64_t
cuda_sstep_grid_id (void)
{
  gdb_assert (cuda_sstep_info.active);
  return cuda_sstep_info.grid_id;
}

void
cuda_sstep_set_ptid (ptid_t ptid)
{
  gdb_assert (cuda_sstep_info.active);
  cuda_sstep_info.ptid = ptid;
}

static bool
cuda_control_flow_instruction (const char *inst, bool skip_subroutines)
{
  if (!inst) return true;
  if (strstr(inst, "SSY")!=0) return true;
  if (strstr(inst, "BAR.SYNC")!=0) return true;
  if (strstr(inst, "BAR.RED")!=0) return true;
  if (strstr(inst, "BRA")!=0) return true;
  if (strstr(inst, "BRK")!=0) return true;
  if (strstr(inst, "NOP.S")!=0) return true;
  if (strstr(inst, "EXIT")!=0) return true;
  if (strstr(inst, "RET")!=0) return true;
  if (strstr(inst, "JMP")!=0) return true;
  if (strstr(inst, "JCAL")!=0 && !skip_subroutines) return true;

  return false;
}
static bool
cuda_sstep_fast (ptid_t ptid)
{
  uint32_t dev_id, sm_id,i;
  struct thread_info *tp = inferior_thread();
  kernel_t kernel = cuda_current_kernel ();
  struct address_space *aspace = NULL;
  const char *inst;
  uint32_t inst_size;
  uint64_t pc, end_pc, adj_pc;
  bool skip_subroutines, rc;

  if (!cuda_options_single_stepping_optimizations_enabled ())
    return false;

  /* No accelerated single stepping when we accuracy is expected */
  if (cuda_get_autostep_stepping ())
    {
      cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                         "Single stepping optimizations were disabled"
                         " due to autostep accuracy requirements");
      return false;
    }

  if (!tp)
    return false;
  gdb_assert (ptid_equal (tp->ptid, ptid));

  /* Skip if stepping just for one instruction */
  if (tp->control.step_range_end <= 1 && tp->control.step_range_start <= 1)
    return false;

  skip_subroutines = tp->control.step_over_calls == STEP_OVER_ALL;

  cuda_coords_get_current_physical (&dev_id, &sm_id, NULL, NULL);
  end_pc = pc = get_frame_pc (get_current_frame ());

  /* Iterate over instructions until the end of the step/next line range */
  /* Break if instruction that can potentially alter program counter has been encountered  */
  do {
    inst = kernel_disassemble (kernel, end_pc, &inst_size);
    cuda_trace_domain (CUDA_TRACE_BREAKPOINT, "%s: pc=0x%llx inst %.*s",
                       __func__, (long long) end_pc, 20, inst);
    if (cuda_control_flow_instruction(inst, skip_subroutines)) break;
    end_pc += inst_size;
  } while (end_pc < tp->control.step_range_end);

 /* The above loop might increment end_pc beyond step_range_end.
     In that case, adjust it to the step_range_end. */
  if (end_pc > tp->control.step_range_end)
    end_pc = tp->control.step_range_end;

  /* Do not attempt to accelerate if stepping over less than 3 instructions */
  if (end_pc <= pc || end_pc - pc < 3*inst_size) {
    cuda_trace_domain  (CUDA_TRACE_BREAKPOINT,
         "%s: Advantage is not big enough: pc=0x%llx end_pc=0x%llx inst_size = %u",
         __func__, (long long)pc, (long long)end_pc, (unsigned)inst_size);
    return false;
  }

  /* Adjust the address of breakpoint instruction is planted at*/
  adj_pc = gdbarch_adjust_breakpoint_address (cuda_get_gdbarch(), end_pc);
  end_pc = adj_pc > tp->control.step_range_end ? tp->control.step_range_end - inst_size : adj_pc ;

  /* Check again, if window is big enough */
  if (end_pc <= pc || end_pc - pc < 3*inst_size) {
    cuda_trace_domain  (CUDA_TRACE_BREAKPOINT,
         "%s: Advantage is not big enough: pc=0x%llx end_pc=0x%llx inst_size = %u",
         __func__, (long long)pc, (long long)end_pc, (unsigned)inst_size);
    return false;
  }


  cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
       "%s: trying to step from %llx to %llx (inst %.*s)",
       __func__, (long long)pc, (long long)end_pc,
       (inst == NULL || strlen(inst)> 20) ? 20 : (int)strlen(inst), inst);

  /* If breakpoint is set at the current PC - unset it*/
  aspace = target_thread_address_space (ptid);
  if (breakpoint_here_p (aspace, pc))
    cuda_api_unset_breakpoint (dev_id, pc);

  /* Resume warp(s) until one of the lanes reaches end_pc */
  rc = warps_resume_until (dev_id, sm_id, cuda_sstep_info.warp_mask, end_pc);

  /* Reset the breakpoint if warps_resume_until call failed */
  if (!rc && breakpoint_here_p (aspace, pc))
    cuda_api_set_breakpoint (dev_id, pc);

  return rc;
}

/**
 * cuda_sstep_execute(ptid_t) returns true if single-stepping was successful.
 * If false is returned, it is callee responsibility to cleanup CUDA sstep state.
 * (usually done by calling cuda_sstep_reset(). )
 */
bool
cuda_sstep_execute (ptid_t ptid)
{
  uint32_t dev_id, sm_id, wp_id, wp;
  uint64_t grid_id, warp_mask, stepped_warp_mask;
  bool     sstep_other_warps;
  bool     grid_id_changed;
  bool     rc = true;

  gdb_assert (!cuda_sstep_info.active);
  gdb_assert (cuda_focus_is_device ());

  cuda_coords_get_current_physical (&dev_id, &sm_id, &wp_id, NULL);
  grid_id           = warp_get_grid_id (dev_id, sm_id, wp_id);
  grid_id_changed   = cuda_sstep_info.grid_id_valid &&
                      cuda_sstep_info.grid_id != grid_id;
  sstep_other_warps = cuda_sstep_info.warp_mask != 0ULL;
  stepped_warp_mask = 0ULL;

  /* Remember the single-step parameters to trick GDB */
  cuda_sstep_info.active         = true;
  cuda_sstep_info.ptid           = ptid;
  cuda_sstep_info.dev_id         = dev_id;
  cuda_sstep_info.sm_id          = sm_id;
  cuda_sstep_info.wp_id          = wp_id;
  cuda_sstep_info.grid_id        = grid_id;
  cuda_sstep_info.grid_id_valid  = true;

  /* Do not try to single step if grid-id changed */
  if (grid_id_changed)
    {
        cuda_trace("device %u sm %u: switched to new grid %llx while single-stepping!\n",
                   dev_id, sm_id, (unsigned long long)grid_id);
        cuda_sstep_info.warp_mask = (1ULL << wp_id);
        return true;
    }

  if (!sstep_other_warps)
    cuda_sstep_info.warp_mask = (1ULL << wp_id);

  cuda_trace ("device %u sm %u: single-stepping warp mask 0x%llx\n",
              dev_id, sm_id, (unsigned long long)cuda_sstep_info.warp_mask);
  gdb_assert (cuda_sstep_info.warp_mask & (1ULL << wp_id));

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
        rc = warp_single_step (dev_id, sm_id, wp_id, &warp_mask);
    }
  else if (!cuda_sstep_fast (ptid))
    {
      /* Single-step all the warps in the warp mask. */
      for (wp = 0; wp < CUDBG_MAX_WARPS; ++wp)
        if (cuda_sstep_info.warp_mask & (1ULL << wp) &&
            warp_is_valid (dev_id, sm_id, wp))
          {
            rc = warp_single_step (dev_id, sm_id, wp, &warp_mask);
            if (!rc) break;
            stepped_warp_mask |= warp_mask;
          }
    }

  /* Update the warp mask. It may have grown. */
  cuda_sstep_info.warp_mask = stepped_warp_mask;


  /* If any warps are marked invalid, but are in the warp_mask
     clear them. This can happen if we stepped a warp over an exit */
  for (wp = 0; wp < CUDBG_MAX_WARPS; ++wp)
    if (cuda_sstep_info.warp_mask & (1ULL << wp) &&
        !warp_is_valid (dev_id, sm_id, wp))
      cuda_sstep_info.warp_mask &= ~(1ULL << wp);

  return rc;
}

void
cuda_sstep_initialize (bool stepping)
{
  if (stepping && cuda_focus_is_device ())
    cuda_sstep_info.warp_mask = (1ULL << cuda_current_warp ());
  else
    cuda_sstep_info.warp_mask = 0ULL;
  cuda_sstep_info.grid_id_valid = false;
}

void
cuda_sstep_reset (bool sstep)
{
/*  When a subroutine is entered while stepping the device, cuda-gdb will
    insert a breakpoint and resume the device. When this happens, the focus
    may change due to the resume. This will cause the cached single step warp
    mask to be incorrect, causing an assertion failure. The fix here is to
    reset the warp mask when switching to a resume. This will cause
    single step execute to update the warp mask after performing the step. */
  if (!sstep && cuda_focus_is_device () && cuda_sstep_is_active ())
    {
      cuda_sstep_info.warp_mask = 0ULL;
      cuda_sstep_info.grid_id_valid = false;
    }

  cuda_sstep_info.active = false;
}

bool
cuda_sstep_kernel_has_terminated (void)
{
  uint32_t dev_id, sm_id, wp_id;
  uint64_t grid_id;
  cuda_iterator itr;
  cuda_coords_t filter = CUDA_WILDCARD_COORDS;
  bool found_valid_warp;

  gdb_assert (cuda_sstep_info.active);

  dev_id  = cuda_sstep_info.dev_id;
  sm_id   = cuda_sstep_info.sm_id;
  wp_id   = cuda_sstep_info.wp_id;
  grid_id = cuda_sstep_info.grid_id;

  if (warp_is_valid (dev_id, sm_id, wp_id))
    return false;

  found_valid_warp = false;
  filter           = CUDA_WILDCARD_COORDS;
  filter.dev       = dev_id;
  filter.gridId    = grid_id;

  itr = cuda_iterator_create (CUDA_ITERATOR_TYPE_WARPS, &filter, CUDA_SELECT_VALID);
  for (cuda_iterator_start (itr); !cuda_iterator_end (itr); cuda_iterator_next (itr))
    {
      gdb_assert (cuda_iterator_get_current (itr).dev    == dev_id);
      gdb_assert (cuda_iterator_get_current (itr).gridId == grid_id);
      found_valid_warp = true;
      break;
    }
  cuda_iterator_destroy (itr);

  if (found_valid_warp)
    return false;

  return true;
}

/* CUDA ABI return value convention:

   (size == 32-bits) .s32/.u32/.f32/.b32      -> R4
   (size == 64-bits) .s64/.u64/.f64/.b64      -> R4-R5
   (size <= 384-bits) .align N .b8 name[size] -> size <= 384-bits -> R4-R15 (A)
   (size > 384-bits)  .align N .b8 name[size] -> size > 384-bits  -> Memory (B)

   For array case (B), the pointer to the memory location is passed as
   a parameter at the beginning of the parameter list.  Memory is allocated
   in the _calling_ function, which is the consumer of the return value.
*/
static enum return_value_convention
cuda_abi_return_value (struct gdbarch *gdbarch, struct value *function,
                       struct type *type, struct regcache *regcache,
                       gdb_byte *readbuf, const gdb_byte *writebuf)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  int regnum = tdep->first_rv_regnum;
  int len = TYPE_LENGTH (type);
  int i;
  uint32_t regval32 = 0U;
  ULONGEST regval   = 0ULL;
  ULONGEST addr;
  uint32_t dev, sm, wp, ln;

  /* The return value is in one or more registers. */
  if (len <= tdep->max_reg_rv_size)
    {
      /* Read/write all regs until we've satisfied len. */
      for (i = 0; len > 0; i++, regnum++, len -= 4)
        {
          if (readbuf)
            {
              regcache_cooked_read_unsigned (regcache, regnum, &regval);
              regval32 = (uint32_t) regval;
              memcpy (readbuf + i * 4, &regval32, min (len, 4));
            }
          if (writebuf)
            {
              memcpy (&regval32, writebuf + i * 4, min (len, 4));
              regval = regval32;
              regcache_cooked_write_unsigned (regcache, regnum, regval);
            }
        }

      return RETURN_VALUE_REGISTER_CONVENTION;
    }

  /* The return value is in memory. */
  if (readbuf)
  {

    /* In the case of large return values, space has been allocated in memory
       to hold the value, and a pointer to that allocation is at the beginning
       of the parameter list.  We need to read the register that holds the
       address, and then read from that address to obtain the value. */
    cuda_coords_get_current_physical (&dev, &sm, &wp, &ln);
    regcache_cooked_read_unsigned (regcache, regnum, &addr);
    cuda_api_read_local_memory (dev, sm, wp, ln, addr, readbuf, len);
  }

  return RETURN_VALUE_ABI_RETURNS_ADDRESS;
}

static int
cuda_adjust_regnum (struct gdbarch *gdbarch, int regnum, int eh_frame_p)
{
  int adjusted_regnum = 0;

  /* If not a device register, nothing to adjust. This happens only when called
     by the DWARF2 frame sniffer when determining the type of the frame. It is
     then safe to bail out and not pass the request to the host adjust_regnum
     function, because, at that point, the type of the frame is not yet
     determinted. */
  if (!cuda_focus_is_device ())
    return regnum;

  cuda_decode_physical_register (regnum, &adjusted_regnum);

  return adjusted_regnum;
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

    if (osect &&
        osect->objfile &&
        !cuda_is_bfd_version_call_abi (osect->objfile->obfd) &&
        osect->objfile->cuda_objfile &&
        osect->objfile->cuda_producer_is_open64)
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
         first instruction of the next line instead of the last instruction of
         the current line. I cannot fix it there since the instruction size is
         unknown. But I can fix it here, which also has the advantage of not
         impacting the way gdb behaves with the host code. When that happens,
         it means that the function body is empty (foo(){};). In that case, we
         follow GDB policy and do not skip the prologue. It also allow us to no
         point to the last instruction of a device function. That instruction
         is not guaranteed to be ever executed, which makes setting breakpoints
         trickier. */
      if (post_prologue_pc > end_addr)
        post_prologue_pc = pc;

      /* If the post_prologue_pc does not make sense, return the given PC. */
      if (post_prologue_pc < pc)
        post_prologue_pc = pc;

      return post_prologue_pc;

      /* If we can't adjust the prologue from the symbol table, we may need
         to resort to instruction scanning.  For now, assume the entry above. */
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
  is_cuda_abi = cuda_get_bfd_abi_version (objfile->obfd, &cuda_abi_version);
  if (is_cuda_abi && cuda_abi_version < CUDA_ELFOSABIV_RELOC) /* Not relocated */
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

/* Given an address string (function_name or filename:function_name or filename:lineno),
   find if there is matching address for it. If so, return it
   (prologue adjusted, if necessary) in func_addr.
   Returns true if the function is found, false otherwise. */
bool
cuda_find_pc_from_address_string (struct objfile *objfile,
                                  char *func_name,
                                  CORE_ADDR *func_addr)
{
  CORE_ADDR addr = 0;
  bool found = false;
  struct block *b;
  struct blockvector *bv;
  struct symtab *s = NULL;
  struct symtab *symtab_scope = NULL;
  struct minimal_symbol *msymbol = NULL;
  const char *sym_name = NULL;
  char *tmp_func_name = NULL;
  struct gdbarch *gdbarch = get_current_arch ();
  const struct language_defn *lang = current_language;
  int dmgl_options = DMGL_ANSI | DMGL_PARAMS;
  struct cleanup *cleanup_chain = NULL;
  char * demangled = NULL;
  bool is_lineno = false;

  gdb_assert (objfile);
  gdb_assert (func_name);
  gdb_assert (func_addr);

  if (!cuda_is_bfd_cuda (objfile->obfd))
    return false;

  cleanup_chain = make_cleanup (null_cleanup, NULL);

  /* Test if a space exist in the function name
     If so, it's likely to be a pending conditional,
     so we going to ignore everything after the space */
  if (strchr(func_name, ' ')!= NULL)
    {
      tmp_func_name = xstrdup (func_name);
      if (tmp_func_name)
        {
          func_name = tmp_func_name;
          make_cleanup (xfree, func_name);
          tmp_func_name = strchr(tmp_func_name, ' ');
          if (tmp_func_name)
            *tmp_func_name = 0;
       }
    }

  /* Test if a colon exists in the function name string.
     If so, name might be limited to symtab scope
     otherwise we can ignore everything before it
     so that we can search using the function name directly. */
  if ((tmp_func_name = strrchr (func_name, ':')))
    {
      unsigned long len = (unsigned long)tmp_func_name - (unsigned long)func_name;
      ALL_OBJFILE_SYMTABS (objfile, s)
          if ( s->filename && strncmp(s->filename, func_name, len) == 0)
            {
              symtab_scope = s;
              break;
            }
      func_name = tmp_func_name + 1;
      /* If all characters after colon are digits - it's a line number */
      while ( isdigit (*++tmp_func_name));
      is_lineno = *tmp_func_name == 0;
    }

  /* We need to find the fully-qualified symbol name that func_name
     corresponds to (if any).  This will handle mangled symbol names,
     which is what will be used to lookup a CUDA device code symbol. */
  if ((msymbol = lookup_minimal_symbol (func_name, NULL, objfile)))
    sym_name = msymbol->ginfo.name;
  else if ((demangled = language_demangle (lang, func_name, dmgl_options)))
    {
      sym_name = demangled;
      make_cleanup (xfree, (char *)sym_name);
    }
  else
    sym_name = func_name;

  /* Look for functions - assigned from DWARF, this path will only
     find information for debug compilations. */
  ALL_OBJFILE_SYMTABS (objfile, s)
    {
      int i;

      /* Check the symtab scope*/
      if (symtab_scope != NULL && s != symtab_scope)
        continue;

      /* If address is a line number try to find its pc from lineinfo */
      if (symtab_scope && is_lineno)
        {
          i = atoi (func_name);
          if (find_line_pc (s, i, &addr))
            {
               found = true;
               break;
            }
        }

      bv = BLOCKVECTOR (s);
      for (i = 0; i < BLOCKVECTOR_NBLOCKS (bv); i++)
        {
          b = BLOCKVECTOR_BLOCK (bv, i);
          if (!b || !b->function)
            continue;

          if (!SYMBOL_MATCHES_NATURAL_NAME (b->function, sym_name) &&
              !SYMBOL_MATCHES_SEARCH_NAME (b->function, sym_name))
            continue;

          found = true;
          addr = BLOCK_START (b);
          addr = cuda_skip_prologue (gdbarch, addr);
          break;
        }

      if (found)
        break;
    }

  /* If we didn't find a function, then it could be a non-debug
     compilation, so look at the msymtab. */
  if (!found)
    {
      ALL_OBJFILE_MSYMBOLS (objfile, msymbol)
        {
          if (!strcmp (sym_name, msymbol->ginfo.name))
            {
              found = true;
              addr = msymbol->ginfo.value.address;
            }
        }
    }

  /* Clean up before exiting. */
  do_cleanups (cleanup_chain);

  /* If we found the symbol, return its address in func_addr
     and return true. */
  if (found)
    {
      *func_addr = addr;
      return true;
    }

  return false;
}


/* Given a raw function name (string), find its corresponding text section vma.
   Returns true if found and stores the address in vma.  Returns false otherwise. */
bool
cuda_find_func_text_vma_from_objfile (struct objfile *objfile,
                                      char *func_name,
                                      CORE_ADDR *vma)
{
  struct obj_section *osect = NULL;
  asection *section = NULL;
  char *text_seg_name = NULL;

  gdb_assert (objfile);
  gdb_assert (func_name);
  gdb_assert (vma);

  /* Construct CUDA text segment name */
  text_seg_name = (char *) xmalloc (strlen (CUDA_ELF_TEXT_PREFIX) + strlen (func_name) + 1);
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
   Works with assumption that the compiler allocates consecutive registers for
   those cases.  */
static int
cuda_convert_register_p (struct gdbarch *gdbarch, int regnum, struct type *type)
{
  return (int)cuda_pc_regnum_p (gdbarch, regnum);
}

/* Read a value of type TYPE from register REGNUM in frame FRAME, and
   return its contents in TO.  */
static int
cuda_register_to_value (struct frame_info *frame, int regnum,
                        struct type *type, gdb_byte *to,
                        int *optimizep, int *unavailablep)
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
cuda_value_to_register (struct frame_info *frame, int regnum,
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
    {
      return;
    }

  put_frame_register (frame, regnum, from);
}

static const gdb_byte *
cuda_breakpoint_from_pc (struct gdbarch *gdbarch, CORE_ADDR *pc, int *len)
{
  return NULL;
}

static CORE_ADDR
cuda_adjust_breakpoint_address (struct gdbarch *gdbarch, CORE_ADDR bpaddr)
{
  uint64_t adjusted_addr = bpaddr;

  context_t context = cuda_system_find_context_by_addr (bpaddr);

  if (context)
    cuda_api_get_adjusted_code_address (context_get_device_id (context),
                                        bpaddr, &adjusted_addr, CUDBG_ADJ_CURRENT_ADDRESS);
  return (CORE_ADDR)adjusted_addr;
}

static struct gdbarch *
cuda_gdbarch_init (struct gdbarch_info info, struct gdbarch_list *arches)
{
  struct gdbarch      *gdbarch;
  struct gdbarch_tdep *tdep;

  /* If there is already a candidate, use it.  */
  arches = gdbarch_list_lookup_by_info (arches, &info);
  if (arches != NULL)
    return arches->gdbarch;

  /* Allocate space for the new architecture.  */
  tdep = XCALLOC (1, struct gdbarch_tdep);
  gdbarch = gdbarch_alloc (&info, tdep);

  /* Set extra CUDA architecture specific information */
  tdep->sp_regnum       = 1;   /* ABI only, SP is in R1 */
  tdep->first_rv_regnum = 4;   /* ABI only, First RV is in R4, also used to pass args */
  tdep->last_rv_regnum  = 15;  /* ABI only, Last RV is in R15, also used to pass args */
  tdep->rz_regnum       = 63;  /* ABI only, Zero is in R63 */
  tdep->pc_regnum       = 256; /* PC is after the last user register */

  tdep->num_regs           = 256 + 1;/* 256 general purpose registers + pc */
  tdep->num_pseudo_regs    = 12; /* 7 for predicates, 1 for CC, 4 for special/invalid */
  tdep->first_pred_regnum  = tdep->num_regs;
  tdep->last_pred_regnum   = tdep->num_regs + 6;
  tdep->cc_regnum          = tdep->last_pred_regnum + 1;
  tdep->error_pc_regnum    = tdep->cc_regnum + 1;
  tdep->special_regnum     = tdep->error_pc_regnum + 1;
  tdep->invalid_lo_regnum  = tdep->special_regnum + 1;
  tdep->invalid_hi_regnum  = tdep->invalid_lo_regnum + 1;

  tdep->max_reg_rv_size = (tdep->last_rv_regnum - tdep->first_rv_regnum + 1) * 4;
  tdep->ptr_size = TARGET_CHAR_BIT * sizeof (CORE_ADDR); /* 32 or 64 bits */

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
  set_gdbarch_num_regs        (gdbarch, tdep->num_regs);
  set_gdbarch_num_pseudo_regs (gdbarch, tdep->num_pseudo_regs);

  set_gdbarch_pc_regnum  (gdbarch, tdep->pc_regnum);
  set_gdbarch_ps_regnum  (gdbarch, -1);
  set_gdbarch_sp_regnum  (gdbarch, -1);
  set_gdbarch_fp0_regnum (gdbarch, -1);

  set_gdbarch_dwarf2_reg_to_regnum (gdbarch, cuda_reg_to_regnum);

  set_gdbarch_pseudo_register_write (gdbarch, cuda_pseudo_register_write);
  set_gdbarch_pseudo_register_read  (gdbarch, cuda_pseudo_register_read);

  set_gdbarch_read_pc  (gdbarch, NULL);
  set_gdbarch_write_pc (gdbarch, NULL);

  set_gdbarch_register_name (gdbarch, cuda_register_name);
  set_gdbarch_register_type (gdbarch, cuda_register_type);
  set_gdbarch_register_reggroup_p (gdbarch, cuda_register_reggroup_p);

  set_gdbarch_print_float_info     (gdbarch, NULL);
  set_gdbarch_print_vector_info    (gdbarch, NULL);

  set_gdbarch_convert_register_p  (gdbarch, cuda_convert_register_p);
  set_gdbarch_register_to_value   (gdbarch, cuda_register_to_value);
  set_gdbarch_value_to_register   (gdbarch, cuda_value_to_register);

  /* Pointers and Addresses */
  set_gdbarch_fetch_pointer_argument (gdbarch, NULL);

  /* Address Classes */
  set_gdbarch_address_class_name_to_type_flags(gdbarch,
                                               cuda_address_class_name_to_type_flags);
  set_gdbarch_address_class_type_flags_to_name(gdbarch,
                                               cuda_address_class_type_flags_to_name);
  set_gdbarch_address_class_type_flags (gdbarch,
                                        cuda_address_class_type_flags);

  /* CUDA - managed variables */
  set_gdbarch_elf_make_msymbol_special (gdbarch, cuda_elf_make_msymbol_special);

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

  set_gdbarch_regset_from_core_section (gdbarch, NULL);
  set_gdbarch_skip_permanent_breakpoint (gdbarch, NULL);
  set_gdbarch_fast_tracepoint_valid_at (gdbarch, NULL);
  set_gdbarch_decr_pc_after_break (gdbarch, 0);
  set_gdbarch_max_insn_length (gdbarch, 8);

  /* Instructions */
  set_gdbarch_print_insn (gdbarch, cuda_print_insn);
  set_gdbarch_relocate_instruction (gdbarch, NULL);
  set_gdbarch_breakpoint_from_pc   (gdbarch, cuda_breakpoint_from_pc);
  set_gdbarch_adjust_breakpoint_address (gdbarch, cuda_adjust_breakpoint_address);

  /* CUDA - no address space management */
  set_gdbarch_has_global_breakpoints (gdbarch, 1);

  /* We hijack the linux siginfo type for the CUDA target on both Mac & Linux */
  set_gdbarch_get_siginfo_type (gdbarch, linux_get_siginfo_type);

  return gdbarch;
}

struct gdbarch *
cuda_get_gdbarch (void)
{
  struct gdbarch_info info;

  if (!cuda_gdbarch)
    {
      gdbarch_info_init (&info);
      info.bfd_arch_info = bfd_lookup_arch (bfd_arch_m68k, 0);
      cuda_gdbarch = gdbarch_find_by_info (info);
    }

  return cuda_gdbarch;
}

void
_initialize_cuda_tdep (void)
{
  register_gdbarch_init (bfd_arch_m68k, cuda_gdbarch_init);
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
    error (_("The directory for the previous CUDA session was not cleaned up. "
             "Try deleting %s and retrying."), cuda_gdb_session_dir);

  cuda_gdb_session_id++;

  snprintf (cuda_gdb_session_dir, CUDA_GDB_TMP_BUF_SIZE,
            "%s/session%d", cuda_gdb_tmpdir_getdir (),
            cuda_gdb_session_id);

  cuda_trace ("creating new session %d", cuda_gdb_session_id);

  ret = cuda_gdb_dir_create (cuda_gdb_session_dir, S_IRWXU | S_IRWXG,
                             override_umask, &dir_exists);

  if (!ret && dir_exists)
    error (_("A stale CUDA session directory was found. "
             "Try deleting %s and retrying."), cuda_gdb_session_dir);
  else if (ret)
    error (_("Failed to create session directory: %s (ret=%d)."), cuda_gdb_session_dir, ret);

  /* Change session folder ownership if debugging as root */
  if (getuid() == 0)
    cuda_gdb_chown_to_pid_uid ( ptid_get_pid (inferior_ptid), cuda_gdb_session_dir);
  return ret;
}

void
cuda_gdb_session_destroy (void)
{
  cuda_gdb_dir_cleanup_files (cuda_gdb_session_dir);

  rmdir (cuda_gdb_session_dir);

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
cuda_adjust_device_code_address (CORE_ADDR original_addr, CORE_ADDR *adjusted_addr)
{
  context_t context = cuda_system_find_context_by_addr (original_addr);
  uint64_t addr = original_addr;

  if (context)
    cuda_api_get_adjusted_code_address (context_get_device_id (context),
                                        original_addr, &addr, CUDBG_ADJ_CURRENT_ADDRESS);
  *adjusted_addr = (CORE_ADDR)addr;
}

void
cuda_update_report_driver_api_error_flags (void)
{
  CORE_ADDR addr;
  CUDBGReportDriverApiErrorFlags flags;

  addr   = cuda_get_symbol_address (_STRING_(CUDBG_REPORT_DRIVER_API_ERROR_FLAGS));
  flags  = cuda_options_api_failures_break_on_nonfatal() ?
              CUDBG_REPORT_DRIVER_API_ERROR_FLAGS_NONE :
              CUDBG_REPORT_DRIVER_API_ERROR_FLAGS_SUPPRESS_NOT_READY;
  if (!addr)
    return;

   target_write_memory (addr, (char *)&flags, sizeof(flags));
}
