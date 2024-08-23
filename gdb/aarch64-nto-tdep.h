/* Target-dependent code for QNX Neutrino x86_64.

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
 * Register names
 */
#define	AARCH64_REG_X0		0
#define	AARCH64_REG_X1		1
#define	AARCH64_REG_X2		2
#define	AARCH64_REG_X3		3
#define	AARCH64_REG_X4		4
#define	AARCH64_REG_X5		5
#define	AARCH64_REG_X6		6
#define	AARCH64_REG_X7		7
#define	AARCH64_REG_X8		8
#define	AARCH64_REG_X9		9
#define	AARCH64_REG_X10		10
#define	AARCH64_REG_X11		11
#define	AARCH64_REG_X12		12
#define	AARCH64_REG_X13		13
#define	AARCH64_REG_X14		14
#define	AARCH64_REG_X15		15
#define	AARCH64_REG_X16		16
#define	AARCH64_REG_X17		17
#define	AARCH64_REG_X18		18
#define	AARCH64_REG_X19		19
#define	AARCH64_REG_X20		20
#define	AARCH64_REG_X21		21
#define	AARCH64_REG_X22		22
#define	AARCH64_REG_X23		23
#define	AARCH64_REG_X24		24
#define	AARCH64_REG_X25		25
#define	AARCH64_REG_X26		26
#define	AARCH64_REG_X27		27
#define	AARCH64_REG_X28		28
#define	AARCH64_REG_X29		29
#define	AARCH64_REG_X30		30
#define	AARCH64_REG_X31		31

/*
 * Register name aliases
 */
#define	AARCH64_REG_LR		30
#define	AARCH64_REG_SP		31

/*
 * Register manipulation
 */
#define AARCH64_GET_REGIP(regs)			((regs)->elr)
#define AARCH64_GET_REGSP(regs)			((regs)->gpr[AARCH64_REG_SP])
#define AARCH64_SET_REGIP(regs,v)		((regs)->elr = (v))
#define AARCH64_SET_REGSP(regs,v)		((regs)->gpr[AARCH64_REG_SP] = (v))

/*
 * Register mappings for AARCH32 state
 */
#define	AARCH32_REG_R0		0
#define	AARCH32_REG_R1		1
#define	AARCH32_REG_R2		2
#define	AARCH32_REG_R3		3
#define	AARCH32_REG_R4		4
#define	AARCH32_REG_R5		5
#define	AARCH32_REG_R6		6
#define	AARCH32_REG_R7		7
#define	AARCH32_REG_R8		8
#define	AARCH32_REG_R9		9
#define	AARCH32_REG_R10		10
#define	AARCH32_REG_R11		11
#define	AARCH32_REG_R12		12
#define	AARCH32_REG_R13		13
#define	AARCH32_REG_R14		14

#define	AARCH32_REG_SP		AARCH32_REG_R13
#define	AARCH32_REG_LR		AARCH32_REG_R14

typedef struct {
	_Uint64t	qlo;
	_Uint64t	qhi;
} aarch64_qreg_t;

typedef struct aarch64_fpu_registers {
	aarch64_qreg_t	reg[32];
	_Uint32t		fpsr;
	_Uint32t		fpcr;
} AARCH64_FPU_REGISTERS __attribute__((__aligned__(16)));

#endif /* aarch64-nto-tdep.h */
