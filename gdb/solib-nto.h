/* SVR4 shared library extensions for QNX Neutrino to use in GDB

   Copyright (C) 2000-2019 Free Software Foundation, Inc.

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

#ifndef SOLIB_NTO_H_
#define SOLIB_NTO_H_

#include "nto-tdep.h"
#include "nto-share/qnx_linkmap_note.h"

#ifndef PATH_MAX
#define PATH_MAX 1024
#endif
#define NOTE_GNU_BUILD_ID_NAME  ".note.gnu.build-id"

struct so_list * nto_solist_from_qnx_linkmap_note (void);
int nto_cmp_host_to_target_word (bfd *, CORE_ADDR, CORE_ADDR);
int nto_so_validate (const struct so_list *);

extern struct target_so_ops nto_svr4_so_ops;

/* flag to control Build-id checking */
extern bool nto_allow_mismatched_debuginfo;

#endif /* SOLIB_NTO_H_ */
