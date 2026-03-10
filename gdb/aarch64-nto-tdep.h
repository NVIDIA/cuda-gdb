/* Target-dependent code for QNX Neutrino x86_64.

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
   Copyright (C) 2007-2025 NVIDIA Corporation
   Modified from the original GDB file referenced above by the CUDA-GDB
   team at NVIDIA <cudatools@nvidia.com>. */


#ifndef AARCH64_NTO_TDEP_H
#define AARCH64_NTO_TDEP_H

#include <stdint.h>

#ifndef __QNXNTO__
typedef uint64_t _Uint64t;
typedef uint32_t _Uint32t;
#endif

/* From aarch64/context.h.  */
typedef struct aarch64_cpu_registers {
	_Uint64t	gpr[32];
	_Uint64t	elr;
	_Uint64t	pstate;
} AARCH64_CPU_REGISTERS __attribute__((__aligned__(16)));

/*
 * Register manipulation
 */
#define AARCH64_GET_REGIP(regs)			((regs)->elr)
#define AARCH64_GET_REGSP(regs)			((regs)->gpr[AARCH64_SP_REGNUM])
#define AARCH64_SET_REGIP(regs,v)		((regs)->elr = (v))
#define AARCH64_SET_REGSP(regs,v)		((regs)->gpr[AARCH64_SP_REGNUM] = (v))

#define AARCH64_FPVALID 0x1

typedef struct {
	_Uint64t	qlo;
	_Uint64t	qhi;
} aarch64_qreg_t;

typedef struct aarch64_fpu_registers {
	aarch64_qreg_t	reg[32];
	_Uint32t		fpsr;
	_Uint32t		fpcr;
} AARCH64_FPU_REGISTERS __attribute__((__aligned__(16)));

/* taken from <aarch64/sypage.h> */
#define AARCH32_CPU_FLAG_NEON		0x0040u		/* Neon Media Engine */
#define	AARCH64_CPU_FLAG_SIMD		(AARCH32_CPU_FLAG_NEON)
#define AARCH64_CPU_PAUTH		0x20000u

#endif /* aarch64-nto-tdep.h */
