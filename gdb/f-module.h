/* Fortran 90 module support in GDB.

   Copyright (C) 2013 Allinea Software Ltd.

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



#if !defined (F_MODULE_H)
#define F_MODULE_H 1

struct block;
struct modtab_entry;
struct partial_symtab;
struct symbol;
struct symtab;

/* Open a new record of this Fortran module.

   Squirrels away a copy of the associated partial_symtab for later use
   when (if) the object-file's symbols have not been fully initialised.  */

extern void f_module_announce (struct objfile *objfile,
			       const char *name);

/* Make this the "current" Fortran module.

   All Fortran symbols encountered will be marked as being part of this
   module until f_module_leave () is called.  */

extern void f_module_enter (struct objfile *objfile,
			    const char *name);

/* Exit from this Fortran module.

   After this call, there will be no "current" Fortran module. Any
   further Fortran symbols encountered will not be associated with
   the (now) previous Fortran module.  */

extern void f_module_leave (void);

/* Associate this symbol with the current Fortran module.  */

extern void f_module_sym_add (const struct symbol *sym); 

#endif /* !defined(F_MODULE_H) */
