/* Darwin support for GDB, the GNU debugger.
   Copyright (C) 1997-2013 Free Software Foundation, Inc.

   Contributed by Apple Computer, Inc.

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

/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2013 NVIDIA Corporation
 * Modified from the original GDB file referenced above by the CUDA-GDB 
 * team at NVIDIA <cudatools@nvidia.com>.
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
#include "frame.h"
#include "inferior.h"
#include "gdbcore.h"
#include "target.h"
#include "floatformat.h"
#include "symtab.h"
#include "regcache.h"
#include "libbfd.h"
#include "objfiles.h"

#include "i387-tdep.h"
#include "amd64-tdep.h"
#include "osabi.h"
#include "ui-out.h"
#include "symtab.h"
#include "frame.h"
#include "gdb_assert.h"
#include "amd64-darwin-tdep.h"
#include "i386-darwin-tdep.h"
#include "solib.h"
#include "solib-darwin.h"
#include "dwarf2-frame.h"

/* Offsets into the struct x86_thread_state64 where we'll find the saved regs.
   From <mach/i386/thread_status.h> and amd64-tdep.h.  */
int amd64_darwin_thread_state_reg_offset[] =
{
  0 * 8,			/* %rax */
  1 * 8,			/* %rbx */
  2 * 8,			/* %rcx */
  3 * 8,			/* %rdx */
  5 * 8,			/* %rsi */
  4 * 8,			/* %rdi */
  6 * 8,			/* %rbp */
  7 * 8,			/* %rsp */
  8 * 8,			/* %r8 ...  */
  9 * 8,
  10 * 8,
  11 * 8,
  12 * 8,
  13 * 8,
  14 * 8,
  15 * 8,			/* ... %r15 */
  16 * 8,			/* %rip */
  17 * 8,			/* %rflags */
  18 * 8,			/* %cs */
  -1,				/* %ss */
  -1,				/* %ds */
  -1,				/* %es */
  19 * 8,			/* %fs */
  20 * 8			/* %gs */
};

const int amd64_darwin_thread_state_num_regs = 
  ARRAY_SIZE (amd64_darwin_thread_state_reg_offset);

/* Assuming THIS_FRAME is a Darwin sigtramp routine, return the
   address of the associated sigcontext structure.  */

static CORE_ADDR
amd64_darwin_sigcontext_addr (struct frame_info *this_frame)
{
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  CORE_ADDR rbx;
  gdb_byte buf[8];

  /* A pointer to the ucontext is passed as the fourth argument
     to the signal handler, which is saved in rbx.  */
  get_frame_register (this_frame, AMD64_RBX_REGNUM, buf);
  rbx = extract_unsigned_integer (buf, 8, byte_order);

  /* The pointer to mcontext is at offset 48.  */
  read_memory (rbx + 48, buf, 8);

  /* First register (rax) is at offset 16.  */
  return extract_unsigned_integer (buf, 8, byte_order) + 16;
}

/* CUDA - siginfo */
/* Darwin does not create a siginfo convenience variable, but the CUDA target
   does. Make sure we have on both the host and device side. */
static struct type *
amd64_darwin_get_siginfo_type (struct gdbarch *gdbarch)
{
  struct type *int_type, *uint_type, *long_type, *ulong_type;
  struct type *void_ptr_type, *pad_type;
  struct type *sigval_type;
  struct type *uid_type, *pid_type;
  struct type *siginfo_type;

  /* basic types */
  int_type = arch_integer_type (gdbarch, gdbarch_int_bit (gdbarch),
			 	0, "int");
  uint_type = arch_integer_type (gdbarch, gdbarch_int_bit (gdbarch),
				 1, "unsigned int");
  long_type = arch_integer_type (gdbarch, gdbarch_long_bit (gdbarch),
				 0, "long int");
  ulong_type = arch_integer_type (gdbarch, gdbarch_long_bit (gdbarch),
		                  0, "unsigned long int");
  void_ptr_type = lookup_pointer_type (builtin_type (gdbarch)->builtin_void);
  pad_type = init_vector_type (ulong_type, 7);

  /* union sigval */
  sigval_type = arch_composite_type (gdbarch, NULL, TYPE_CODE_UNION);
  TYPE_NAME (sigval_type) = xstrdup ("union sigval");
  append_composite_type_field (sigval_type, "sival_int", int_type);
  append_composite_type_field (sigval_type, "sival_ptr", void_ptr_type);

  /* __pid_t */
  pid_type = arch_type (gdbarch, TYPE_CODE_TYPEDEF, TYPE_LENGTH (int_type),
			xstrdup ("pid_t"));
  TYPE_TARGET_TYPE (pid_type) = int_type;
  TYPE_TARGET_STUB (pid_type) = 1;

  /* __uid_t */
  uid_type = arch_type (gdbarch, TYPE_CODE_TYPEDEF, TYPE_LENGTH (uint_type),
			xstrdup ("uid_t"));
  TYPE_TARGET_TYPE (uid_type) = uint_type;
  TYPE_TARGET_STUB (uid_type) = 1;

  /* struct siginfo */
  siginfo_type = arch_composite_type (gdbarch, NULL, TYPE_CODE_STRUCT);
  TYPE_NAME (siginfo_type) = xstrdup ("siginfo");
  append_composite_type_field (siginfo_type, "si_signo", int_type);
  append_composite_type_field (siginfo_type, "si_errno", int_type);
  append_composite_type_field (siginfo_type, "si_code", int_type);
  append_composite_type_field (siginfo_type, "si_pid", pid_type);
  append_composite_type_field (siginfo_type, "si_uid", uid_type);
  append_composite_type_field (siginfo_type, "si_status", int_type);
  append_composite_type_field (siginfo_type, "si_addr", void_ptr_type);
  append_composite_type_field (siginfo_type, "si_value", sigval_type);
  append_composite_type_field (siginfo_type, "si_band", long_type);
  append_composite_type_field (siginfo_type, "__pad", pad_type); 

  return siginfo_type;
}

static void
x86_darwin_init_abi_64 (struct gdbarch_info info, struct gdbarch *gdbarch)
{
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);

  amd64_init_abi (info, gdbarch);

  tdep->struct_return = reg_struct_return;

  dwarf2_frame_set_signal_frame_p (gdbarch, darwin_dwarf_signal_frame_p);

  tdep->sigtramp_p = i386_sigtramp_p;
  tdep->sigcontext_addr = amd64_darwin_sigcontext_addr;
  tdep->sc_reg_offset = amd64_darwin_thread_state_reg_offset;
  tdep->sc_num_regs = amd64_darwin_thread_state_num_regs;

  tdep->jb_pc_offset = 56;

  /* CUDA - siginfo */
  set_gdbarch_get_siginfo_type (gdbarch, amd64_darwin_get_siginfo_type);

  set_solib_ops (gdbarch, &darwin_so_ops);
}

/* -Wmissing-prototypes */
extern initialize_file_ftype _initialize_amd64_darwin_tdep;

void
_initialize_amd64_darwin_tdep (void)
{
  gdbarch_register_osabi (bfd_arch_i386, bfd_mach_x86_64,
                          GDB_OSABI_DARWIN, x86_darwin_init_abi_64);
}
