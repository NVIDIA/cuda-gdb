/* Common code for x86 siginfo handling.

   Copyright (C) 2015 Free Software Foundation, Inc.

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

#ifndef X86_SIGINFO_H
#define X86_SIGINFO_H 1


enum amd64_siginfo_fixup_mode
{
  X32_FIXUP = 1,
  COMPAT32_FIXUP = 2,
  UNKNOWN
};

int
amd64_linux_siginfo_fixup_low (siginfo_t *native, gdb_byte *inf,
			       int direction,
			       enum amd64_siginfo_fixup_mode mode);


#endif
