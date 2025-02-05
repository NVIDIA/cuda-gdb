/* Target-dependent code for QNX Neutrino aarch64.

   Copyright (C) 2014-2019 Free Software Foundation, Inc.

   Contributed by QNX Software Systems Ltd.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* NVIDIA CUDA Debugger CUDA-GDB
   Copyright (C) 2007-2024 NVIDIA Corporation
   Modified from the original GDB file referenced above by the CUDA-GDB
   team at NVIDIA <cudatools@nvidia.com>. */

#include "defs.h"

#include "frame.h"
#include "gdbcore.h"
#include "osabi.h"
#include "regcache.h"
#include "regset.h"
#include "target.h"
#include "trad-frame.h"
#include "tramp-frame.h"

#include "aarch64-tdep.h"
#include "nto-tdep.h"

#include "gdbsupport/gdb_assert.h"

#include "aarch64-nto-tdep.h"
#include "arch-utils.h"
#include "solib-nto.h"
#include "solib-svr4.h"
#include "solib.h"

#define AARCH64_GREGSZ 8U
#define AARCH64_FPREGSZ 16U
#define AARCH64_FPSRSZ 4U
#define AARCH64_FPCRSZ 4U
#define AARCH64_INSTRSZ 4U

static void
aarch64_nto_supply_gregset (struct regcache *regcache, const gdb_byte *gregs,
			    size_t len)
{
  int regno;

  for (regno = AARCH64_X0_REGNUM; regno <= AARCH64_CPSR_REGNUM; regno++)
    regcache->raw_supply (regno, gregs + AARCH64_GREGSZ * regno);
}

static void
aarch64_nto_supply_fpregset (struct regcache *regcache, const gdb_byte *fpregs,
			     size_t len)
{
  int regno;

  for (regno = AARCH64_V0_REGNUM; regno <= AARCH64_V31_REGNUM; regno++)
    regcache->raw_supply (
	regno, fpregs + AARCH64_FPREGSZ * (regno - AARCH64_V0_REGNUM));

  regcache->raw_supply (AARCH64_FPSR_REGNUM, fpregs + AARCH64_FPREGSZ * 32);
  regcache->raw_supply (AARCH64_FPCR_REGNUM,
			fpregs + AARCH64_FPREGSZ * 32 + AARCH64_FPSRSZ);
}

static void
aarch64_nto_supply_regset (struct regcache *regcache, int regset,
			  const gdb_byte *data, size_t len)
{
  switch (regset)
    {
    case NTO_REG_GENERAL:
      aarch64_nto_supply_gregset (regcache, data, len);
      break;
    case NTO_REG_FLOAT:
      aarch64_nto_supply_fpregset (regcache, data, len);
      break;
    default:
      gdb_assert (!"Unknown regset");
    }
}

static int
aarch64_nto_regset_id (int regno)
{
  if (regno == -1)
    return NTO_REG_END;
  else if (regno < AARCH64_V0_REGNUM)
    return NTO_REG_GENERAL;
  else if (regno < AARCH64_V0_REGNUM + 32 + 2) /* fpregs=32, fpsr/fpcr =2 */
    return NTO_REG_FLOAT;

  return -1; /* Error.  */
}

static int
aarch64_nto_register_area (int regset, unsigned cpuflags)
{
  if (regset == NTO_REG_GENERAL)
    return AARCH64_GREGSZ * AARCH64_V0_REGNUM;
  else if (regset == NTO_REG_FLOAT)
    return sizeof (AARCH64_FPU_REGISTERS);
  else
    warning (_ ("Only general and floatpoint registers supported on aarch64 "
		"for now\n"));
  return -1;
}

static int
aarch64_nto_regset_fill (const struct regcache *const regcache,
			const int regset, gdb_byte *const data, size_t len)
{
  if (regset == NTO_REG_GENERAL)
    {
      int regno;

      for (regno = 0; regno <= AARCH64_CPSR_REGNUM; regno++)
	regcache->raw_collect (regno, data + AARCH64_GREGSZ * regno);
      return 0;
    }
  else if (regset == NTO_REG_FLOAT)
    {
      int regno;

      for (regno = AARCH64_V0_REGNUM; regno <= AARCH64_V31_REGNUM; regno++)
	regcache->raw_collect (
	    regno, data + (regno - AARCH64_V0_REGNUM) * AARCH64_FPREGSZ);
      regcache->raw_collect (AARCH64_FPSR_REGNUM, data + 32 * AARCH64_FPREGSZ);
      regcache->raw_collect (AARCH64_FPCR_REGNUM,
			     data + 32 * AARCH64_FPREGSZ + AARCH64_FPSRSZ);
      return 0;
    }
  return -1; // Error
}

/* Implement the "init" method of struct tramp_frame.  */
// stack offset of the SIGSTACK_CONTEXT pointer
#define AARCH64_SIGFRAME_UCONTEXT_OFFSET  56

#define AARCH64_UCONTEXT_REG_OFFSET       0x30
#define AARCH64_MCONTEXT_FPREGS_OFFSET    272

/* create a pseudo frame that holds the 'call' to the signal handler and allows
 * to return to the original location at which the signal occurred */
static void
aarch64_nto_sigframe_init (const struct tramp_frame *self,
           frame_info_ptr this_frame,
           struct trad_frame_cache *this_cache,
           CORE_ADDR func)
{
  int i;

  // Get the stack frame
  CORE_ADDR sp = get_frame_register_unsigned (this_frame, AARCH64_SP_REGNUM);

  // Get the ucontext
  CORE_ADDR ucontext_addr=0;
  target_read_memory (sp + AARCH64_SIGFRAME_UCONTEXT_OFFSET, (gdb_byte *)&ucontext_addr, 4);

  // Get register information in the ucontext
  CORE_ADDR reg_addr = ucontext_addr + AARCH64_UCONTEXT_REG_OFFSET;

  nto_trace(0)("Initializing sigframe. uctx:%p, regs:%p\n", (void*)ucontext_addr, (void*)reg_addr);

  /* fill in the gregs */
  for (i = 0; i < AARCH64_X_REGS_NUM; i++)
    {
      trad_frame_set_reg_addr (this_cache, AARCH64_X0_REGNUM+i,
          reg_addr + i * AARCH64_GREGSZ);
    }

  /* fill in status registers */
  trad_frame_set_reg_addr (this_cache, AARCH64_PC_REGNUM,
     reg_addr + AARCH64_PC_REGNUM * AARCH64_GREGSZ);
  trad_frame_set_reg_addr (this_cache, AARCH64_CPSR_REGNUM,
     reg_addr + AARCH64_CPSR_REGNUM * AARCH64_GREGSZ);

  /* fill in FP Registers */
  for (i = 0; i < AARCH64_V_REGS_NUM; i++)
    {
      trad_frame_set_reg_addr (this_cache, AARCH64_V0_REGNUM + i,
           reg_addr
           + AARCH64_MCONTEXT_FPREGS_OFFSET
           + i * AARCH64_FPREGSZ);
    }

  trad_frame_set_reg_addr (this_cache, AARCH64_FPSR_REGNUM,
         reg_addr + AARCH64_MCONTEXT_FPREGS_OFFSET
         + 32 * AARCH64_FPREGSZ);
  trad_frame_set_reg_addr (this_cache, AARCH64_FPCR_REGNUM,
         reg_addr + AARCH64_MCONTEXT_FPREGS_OFFSET
         + 32 * AARCH64_FPREGSZ + 4);

  /* add frame to the list */
  trad_frame_set_id (this_cache, frame_id_build (sp, func));
}

/* Return whether THIS_FRAME corresponds to a QNX Neutrino sigtramp
   routine.  */
static int
aarch64_nto_sigframe_validate (const struct tramp_frame *self,
       frame_info_ptr this_frame,
       CORE_ADDR *pc)
{
  const char *name;
  find_pc_partial_function (*pc, &name, NULL, NULL);

  if (name && (strcmp ("__signalstub", name) == 0))
    return 1;

  return 0;
}

static const struct tramp_frame aarch64_nto_sigframe =
{
  /* The trampoline's type, some are signal trampolines, some are normal
     call-frame trampolines (aka thunks).  */
  SIGTRAMP_FRAME,
  /* The trampoline's entire instruction sequence.  It consists of a
     bytes/mask pair.  Search for this in the inferior at or around
     the frame's PC.  It is assumed that the PC is INSN_SIZE aligned,
     and that each element of TRAMP contains one INSN_SIZE
     instruction.  It is also assumed that INSN[0] contains the first
     instruction of the trampoline and hence the address of the
     instruction matching INSN[0] is the trampoline's "func" address.
     The instruction sequence is terminated by
     TRAMP_SENTINEL_INSN.  */
  AARCH64_INSTRSZ,
  {
    {0xaa0003f3, ULONGEST_MAX},       /* mov     x19, x0 */
    {0xf9401a63, ULONGEST_MAX},       /* ldr     x3, [x19,#SIGSTACK_HANDLER] */
    {0xf9401e62, ULONGEST_MAX},       /* ldr     x2, [x19,#SIGSTACK_CONTEXT] */
    {0x91000261, ULONGEST_MAX},       /* add     x1, x19, #SIGSTACK_SIGINFO  */
    {0xf9400260, ULONGEST_MAX},       /* ldr     x0, [x19, SIGSTACK_SIGNO]   */
    {0xd63f0060, ULONGEST_MAX},       /* blr     x3 */
    {0xaa1303e0, ULONGEST_MAX},       /* mov     x0, x19 */
    {0x14000000, 0x14000000},         /* b       SignalReturn */
    {TRAMP_SENTINEL_INSN, ULONGEST_MAX}
  },
  /* Initialize a trad-frame cache corresponding to the tramp-frame.
     FUNC is the address of the instruction TRAMP[0] in memory.  */
  aarch64_nto_sigframe_init,
  /* Return non-zero if the tramp-frame is valid for the PC requested.
     Adjust the PC to point to the address to check the instruction
     sequence against if required.  If this is NULL, then the tramp-frame
     is valid for any PC.  */
  aarch64_nto_sigframe_validate
};

static int
aarch64_nto_breakpoint_size (CORE_ADDR addr)
{
  return 0;
}

/* Register maps.  */

static const struct regcache_map_entry aarch64_nto_gregmap[] =
  {
    { 31, AARCH64_X0_REGNUM, 8 }, /* x0 ... x30 */
    { 1, AARCH64_SP_REGNUM, 8 },
    { 1, AARCH64_PC_REGNUM, 8 },
    { 1, AARCH64_CPSR_REGNUM, 8 },
    { 0 }
  };

static const struct regcache_map_entry aarch64_nto_fpregmap[] =
  {
    { 32, AARCH64_V0_REGNUM, 16 }, /* v0 ... v31 */
    { 1, AARCH64_FPSR_REGNUM, 4 },
    { 1, AARCH64_FPCR_REGNUM, 4 },
    { 0 }
  };

/* Register set definitions.  */

const struct regset aarch64_nto_gregset =
  {
    aarch64_nto_gregmap,
    regcache_supply_regset, regcache_collect_regset
  };

const struct regset aarch64_nto_fpregset =
  {
    aarch64_nto_fpregmap,
    regcache_supply_regset, regcache_collect_regset
  };

/* Implement the "regset_from_core_section" gdbarch method.  */

static void
aarch64_nto_iterate_over_regset_sections (struct gdbarch *gdbarch,
            iterate_over_regset_sections_cb *cb,
            void *cb_data,
            const struct regcache *regcache)
{
  cb (".reg", sizeof (AARCH64_CPU_REGISTERS), sizeof (AARCH64_CPU_REGISTERS), &aarch64_nto_gregset,
      NULL, cb_data);
  cb (".reg2", sizeof (AARCH64_FPU_REGISTERS), sizeof (AARCH64_FPU_REGISTERS), &aarch64_nto_fpregset,
      NULL, cb_data);
}

/* deriving the target capabilities from the cpuflags */
static const struct target_desc *
aarch64_nto_read_description(uint64_t cpuflags)
{
  aarch64_features features;
  /* Number of 128bit chunks in the SVE Z register
   * this is implementation dependant and since NTO seems to only
   * care about SIMD, we stick to the default 128bit SIMD register
   * layout if available */
  features.vq = (cpuflags & AARCH64_CPU_FLAG_SIMD)?0:1;
  /* pointer authentification support */
  features.pauth = (cpuflags & AARCH64_CPU_PAUTH);
  /* memory tagging support
   * not supported on NTO right now */
  features.mte   = false;
  return aarch64_read_description (features);
}

static struct nto_target_ops aarch64_nto_ops;

static void
init_aarch64_nto_ops (void)
{
  aarch64_nto_ops.regset_id = aarch64_nto_regset_id;
  aarch64_nto_ops.supply_gregset = aarch64_nto_supply_gregset;
  aarch64_nto_ops.supply_fpregset = aarch64_nto_supply_fpregset;
  aarch64_nto_ops.supply_altregset = nto_dummy_supply_regset;
  aarch64_nto_ops.supply_regset = aarch64_nto_supply_regset;
  aarch64_nto_ops.register_area = aarch64_nto_register_area;
  aarch64_nto_ops.regset_fill = aarch64_nto_regset_fill;
  aarch64_nto_ops.fetch_link_map_offsets =
      nto_generic_svr4_fetch_link_map_offsets;
  aarch64_nto_ops.breakpoint_size = aarch64_nto_breakpoint_size;
  aarch64_nto_ops.read_description = aarch64_nto_read_description;
}

static void
aarch64_nto_init_abi (struct gdbarch_info info, struct gdbarch *gdbarch)
{
  aarch64_gdbarch_tdep *tdep = gdbarch_tdep<aarch64_gdbarch_tdep> (gdbarch);
  struct nto_target_ops *nto_ops;
  nto_byte_order=BFD_ENDIAN_LITTLE;

  tdep->lowest_pc = 0x1000;

  nto_init_abi( info, gdbarch );

  init_aarch64_nto_ops ();

  /* Deal with our strange signals.  */
  nto_initialize_signals ();

  // our watchpoints trigger after the value has been written back
  set_gdbarch_have_nonsteppable_watchpoint(gdbarch, false);

  set_solib_svr4_fetch_link_map_offsets
    (gdbarch, nto_generic_svr4_fetch_link_map_offsets);

  nto_ops = get_nto_target_ops (gdbarch);
  *nto_ops = aarch64_nto_ops;

  set_gdbarch_so_ops (gdbarch, &nto_svr4_so_ops);

  set_gdbarch_get_siginfo_type (gdbarch, nto_get_siginfo_type);

  tramp_frame_prepend_unwinder (gdbarch, &aarch64_nto_sigframe);

  set_gdbarch_iterate_over_regset_sections (
      gdbarch, aarch64_nto_iterate_over_regset_sections);
}

void _initialize_aarch64_nto_tdep ();
void
_initialize_aarch64_nto_tdep ()
{
  gdbarch_register_osabi (bfd_arch_aarch64, 0, GDB_OSABI_QNXNTO,
			  aarch64_nto_init_abi);
}
