# Copyright (C) 2022-2022 Free Software Foundation, Inc.
#
#   This file is part of GDB.
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

import gdb

objfile = None
symtab = None

class JITRegisterCode(gdb.Breakpoint):
    def stop(self):

        global objfile 
        global symtab

        frame = gdb.newest_frame()
        name = frame.read_var('name').string()
        code = int(frame.read_var('code'))
        size = int(frame.read_var('size'))

        objfile = gdb.Objfile(name)
        symtab = gdb.Symtab(objfile, "py-jit.c")
        symtab.add_block(name, code, code + size)
        symtab.set_linetable([
            gdb.LineTableEntry(29, code, True),
            gdb.LineTableEntry(30, code+3, True),            
            gdb.LineTableEntry(31, code+6, True)
        ])

        return False # Continue execution


JITRegisterCode("jit_register_code", internal=True)
