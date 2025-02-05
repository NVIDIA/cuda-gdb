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


#ifndef AMD64_NTO_TDEP_H
#define AMD64_NTO_TDEP_H

#include <stdint.h>

typedef struct fsave_area_64 {
    uint32_t fpu_control_word;
    uint32_t fpu_status_word;
    uint32_t fpu_tag_word;
    uint32_t fpu_ip;
    uint32_t fpu_cs;
    uint32_t fpu_op;
    uint32_t fpu_ds;
    uint8_t  st_regs[80];
} X86_64_NDP_REGISTERS;

#endif /* amd64-nto-tdep.h */
