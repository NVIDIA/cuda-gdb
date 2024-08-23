/* Target-dependent code for QNX Neutrino aarch64.

   Copyright (C) 2014 Free Software Foundation, Inc.

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
#include "osabi.h"
#include "regset.h"
#include "regcache.h"
#include "target.h"

#include "nto-tdep.h"
#include "aarch64-tdep.h"

#include "gdbsupport/gdb_assert.h"

#include "nto-tdep.h"
#include "solib.h"
#include "solib-svr4.h"
#include "aarch64-nto-tdep.h"

#define AARCH64_GREGSZ 8U
#define AARCH64_FPREGSZ 16U
#define AARCH64_FPSRSZ 4U
#define AARCH64_FPCRSZ 4U

extern CORE_ADDR aarch64nto_sigcontext_addr (frame_info_ptr this_frame);
extern int aarch64nto_sigtramp_p (frame_info_ptr this_frame);

static void
aarch64nto_supply_gregset (struct regcache *regcache, const gdb_byte *gregs,
			   size_t len)
{
  int regno;

  for (regno = AARCH64_X0_REGNUM; regno <= AARCH64_CPSR_REGNUM; regno++)
    regcache->raw_supply (regno, gregs + AARCH64_GREGSZ * regno);
}

static void
aarch64nto_supply_fpregset (struct regcache *regcache, const gdb_byte *fpregs,
			    size_t len)
{
  int regno;

  for (regno = AARCH64_V0_REGNUM; regno <= AARCH64_V31_REGNUM; regno++)
    regcache->raw_supply (regno, fpregs + AARCH64_FPREGSZ
			  * (regno - AARCH64_V0_REGNUM));

  regcache->raw_supply (AARCH64_FPSR_REGNUM, fpregs + AARCH64_FPREGSZ * 32);
  regcache->raw_supply (AARCH64_FPCR_REGNUM, fpregs + AARCH64_FPREGSZ * 32 + AARCH64_FPSRSZ);
}

static void
aarch64nto_supply_regset (struct regcache *regcache, int regset,
			  const gdb_byte *data, size_t len)
{
  switch (regset)
    {
    case NTO_REG_GENERAL:
      aarch64nto_supply_gregset (regcache, data, len);
      break;
    case NTO_REG_FLOAT:
      aarch64nto_supply_fpregset (regcache, data, len);
      break;
    default:
      gdb_assert (!"Unknown regset");
    }
}

static int
aarch64nto_regset_id (int regno)
{
  if (regno == -1)
    return NTO_REG_END;
  else if (regno < AARCH64_V0_REGNUM)
    return NTO_REG_GENERAL;
  else if (regno < AARCH64_V0_REGNUM + 32 + 2) /* fpregs=32, fpsr/fpcr =2 */
    return NTO_REG_FLOAT;

  return -1;			/* Error.  */
}

static int
aarch64nto_register_area (int regset, unsigned cpuflags)
{
  if (regset == NTO_REG_GENERAL)
    return AARCH64_GREGSZ * AARCH64_V0_REGNUM;
  else if (regset == NTO_REG_FLOAT)
    return sizeof (AARCH64_CPU_REGISTERS);
  else
      warning(_("Only general and floatpoint registers supported on aarch64 for now\n"));
  return -1;
}

static int
aarch64nto_regset_fill (const struct regcache *const regcache,
			const int regset, gdb_byte *const data,
			size_t len)
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
	regcache->raw_collect (regno,
			       data + (regno - AARCH64_V0_REGNUM) * AARCH64_FPREGSZ);
      regcache->raw_collect (AARCH64_FPSR_REGNUM,
			     data + 32 * AARCH64_FPREGSZ);
      regcache->raw_collect (AARCH64_FPCR_REGNUM,
			     data + 32 * AARCH64_FPREGSZ + AARCH64_FPSRSZ);
      return 0;
    }
  return -1; // Error
}

/* Return whether THIS_FRAME corresponds to a QNX Neutrino sigtramp
   routine.  */

int
aarch64nto_sigtramp_p (frame_info_ptr this_frame)
{
  CORE_ADDR pc = get_frame_pc (this_frame);
  const char *name;

  find_pc_partial_function (pc, &name, NULL, NULL);

  return name && strcmp ("__signalstub", name) == 0;
}

/* Assuming THIS_FRAME is a QNX Neutrino sigtramp routine, return the
   address of the associated sigcontext structure.  */

CORE_ADDR
aarch64nto_sigcontext_addr (frame_info_ptr this_frame)
{
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  gdb_byte buf[AARCH64_GREGSZ];
  CORE_ADDR ptrctx;

  get_frame_register (this_frame, AARCH64_PC_REGNUM, buf);
  ptrctx = extract_unsigned_integer (buf, AARCH64_GREGSZ, byte_order);

  return ptrctx;
}

static int
aarch64nto_breakpoint_size (CORE_ADDR addr)
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
    regcache_supply_regset,
    regcache_collect_regset
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
  cb (".reg", sizeof (AARCH64_CPU_REGISTERS), sizeof (AARCH64_CPU_REGISTERS),
      &aarch64_nto_gregset, NULL, cb_data);
  cb (".reg2", sizeof (AARCH64_FPU_REGISTERS), sizeof (AARCH64_FPU_REGISTERS),
      &aarch64_nto_fpregset, NULL, cb_data);
}

static struct nto_target_ops aarch64_nto_ops;

static void
init_aarch64nto_ops (void)
{
  aarch64_nto_ops.regset_id = aarch64nto_regset_id;
  aarch64_nto_ops.supply_gregset = aarch64nto_supply_gregset;
  aarch64_nto_ops.supply_fpregset = aarch64nto_supply_fpregset;
  aarch64_nto_ops.supply_altregset = nto_dummy_supply_regset;
  aarch64_nto_ops.supply_regset = aarch64nto_supply_regset;
  aarch64_nto_ops.register_area = aarch64nto_register_area;
  aarch64_nto_ops.regset_fill = aarch64nto_regset_fill;
  aarch64_nto_ops.fetch_link_map_offsets =
    nto_generic_svr4_fetch_link_map_offsets;
  aarch64_nto_ops.breakpoint_size = aarch64nto_breakpoint_size;
}

static void
aarch64_nto_init_abi (struct gdbarch_info info, struct gdbarch *gdbarch)
{
  aarch64_gdbarch_tdep *tdep = gdbarch_tdep<aarch64_gdbarch_tdep> (gdbarch);
  static struct target_so_ops nto_svr4_so_ops;
  struct nto_target_ops *nto_ops;

  tdep->lowest_pc = 0x1000;

  init_aarch64nto_ops ();

  /* Deal with our strange signals.  */
  nto_initialize_signals ();

  set_solib_svr4_fetch_link_map_offsets
    (gdbarch, nto_generic_svr4_fetch_link_map_offsets);

  nto_ops = get_nto_target_ops (gdbarch);
  *nto_ops = aarch64_nto_ops;

  set_gdbarch_gdb_signal_to_target (gdbarch, nto_gdb_signal_to_target);
  set_gdbarch_gdb_signal_from_target (gdbarch, nto_gdb_signal_from_target);

  /* Initialize this lazily, to avoid an initialization order
     dependency on solib-svr4.c's _initialize routine.  */
  if (nto_svr4_so_ops.in_dynsym_resolve_code == NULL)
    {
      nto_svr4_so_ops = svr4_so_ops;

      /* Our loader handles solib relocations differently than svr4.  */
      nto_svr4_so_ops.relocate_section_addresses
        = nto_relocate_section_addresses;

      /* Supply a nice function to find our solibs.  */
      nto_svr4_so_ops.find_and_open_solib
        = nto_find_and_open_solib;

      /* Our linker code is in libc.  */
      nto_svr4_so_ops.in_dynsym_resolve_code
        = nto_in_dynsym_resolve_code;
    }
  set_gdbarch_so_ops (gdbarch, &nto_svr4_so_ops);

  set_gdbarch_get_siginfo_type (gdbarch, nto_get_siginfo_type);

  set_gdbarch_iterate_over_regset_sections
    (gdbarch, aarch64_nto_iterate_over_regset_sections);
}


void _initialize_aarch64nto_tdep ();
void
_initialize_aarch64nto_tdep ()
{
  gdbarch_register_osabi (bfd_arch_aarch64, 0,
			  GDB_OSABI_QNXNTO, aarch64_nto_init_abi);
}
