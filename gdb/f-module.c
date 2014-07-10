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

#include "defs.h"
#include "symtab.h"
#include "psympriv.h"
#include "f-module.h"

#include "block.h"
#include "command.h"
#include "gdb_string.h"
#include "hashtab.h"
#include "source.h"
#include "value.h"
#include "cp-support.h"
#include "objfiles.h"

#include <ctype.h>

static const struct objfile_data *f_module_objfile_data_key;

/* A linked-list of pointers to this module's symbols.  */

struct modtab_symbol
{
  const struct symbol *sym;
  struct modtab_symbol *next;
};

/* Record of a particular module.

   Created with a pointer to the associated partial_symtab. This is used
   to populate (if needed) the list of symbols for this module the first
   time information is requested by the user.  */

struct modtab_entry
{
  char *name;
  struct modtab_symbol *sym_list;
};

/* Keep a record of the "current" module.

   Used when adding newly discovered symbols into the encompassing
   module.  */

static struct modtab_entry *open_module = NULL;
static struct objfile *open_module_objfile = NULL;

/* Hash function for a module-name.

   This is a copy of the code from htab_hash_string(), except that it
   converts every char in the module-name to lower-case before computing
   the hash value.

   This is done to cope with Fortran's case-insensitivity ...  */

static hashval_t
htab_hash_modname (const void *p)
{
  const struct modtab_entry *modptr = (const struct modtab_entry *) p;
  const unsigned char *str = (const unsigned char *) modptr->name;
  hashval_t r = 0;
  unsigned char c;

  while ((c = tolower(*str++)) != 0)
    r = r * 67 + c - 113;

  return r;
}

/* Equality function for a module-name.

   Used by hash-table code to detect collisions of the hashing-function.  */

static int
modname_eq (const struct modtab_entry *lhs, const struct modtab_entry *rhs)
{
  /* Case-insensitive string compare ...  */

  return strcasecmp (lhs->name, rhs->name) == 0;
}

/* Initialise the module hash-table.  */

static struct htab *
modtab_init (struct objfile *objfile)
{
  struct htab *modtab = objfile_data (objfile, f_module_objfile_data_key);
  if (!modtab)
    {
#if defined(USE_MMALLOC)
      modtab = htab_create_alloc_ex (256, htab_hash_modname,
				     (int (*) (const void *, const void *)) modname_eq,
				     NULL,  objfile->md, xmcalloc, xmfree);
#else
      modtab = htab_create (256, htab_hash_modname,
			   (int (*) (const void *, const void *)) modname_eq,
			   NULL);
#endif
      set_objfile_data (objfile, f_module_objfile_data_key, modtab);
    }
  return modtab;				 
}

/* Open a new record of this Fortran module.

   Squirrels away a copy of the associated partial_symtab for later use
   when (if) the object-file's symbols have not been fully initialised.  */

void
f_module_announce (struct objfile *objfile, const char *name)
{  
  struct modtab_entry *candidate, **slot;
  struct htab *modtab = modtab_init (objfile);

  candidate = (struct modtab_entry *) obstack_alloc (&objfile->objfile_obstack, sizeof (struct modtab_entry));
  candidate->name = obstack_copy0 (&objfile->objfile_obstack, name, strlen (name));
  candidate->sym_list = NULL;

  slot = (struct modtab_entry **) htab_find_slot (modtab, candidate, INSERT);

  if (*slot == NULL)
    {
      *slot = candidate;
    }
  else
    {
      /* There will be a DW_TAG_module for every source file that used a module
         so silently ignore duplicates.  */
      obstack_free (&objfile->objfile_obstack, candidate);
      /* candidate->name is freed automatically - see docs for obstack_free.  */
    }
}

/* Find a given Fortran modtab entry.  */

static struct modtab_entry *
f_module_lookup (struct objfile *objfile, const char *module_name)
{
  struct modtab_entry search_entry, *found_entry;
  struct htab *modtab = objfile_data (objfile, f_module_objfile_data_key);
  if (!modtab)
    return NULL;
  
  /* const-correctness gone to pot.  */
  search_entry.name = (char *) module_name;
  found_entry = htab_find (modtab, &search_entry);

  return found_entry;
}

/* Make this the "current" Fortran module.

   All Fortran symbols encountered will be marked as being part of this
   module until f_module_leave () is called.  */

void
f_module_enter (struct objfile *objfile, const char *name)
{
  struct modtab_entry *found_entry;

  if (open_module)
    {
      printf_filtered ("f_module_enter: attempt to nest a module\n");
      return;
    }

  found_entry = f_module_lookup (objfile, name);

  if (found_entry)
    {
      open_module = found_entry;
      open_module_objfile = objfile;
    }
  else
    {
      printf_filtered ("f_module_enter: module present in symtab but not psymtab\n");
    }
}

/* Exit from this Fortran module.

   After this call, there will be no "current" Fortran module. Any
   further Fortran symbols encountered will not be associated with
   the (now) previous Fortran module.  */

void
f_module_leave ()
{
  if (!open_module)
    {
      printf_filtered ("f_module_leave(): not currently in a module\n");
    }

  open_module = NULL;
}
  
/* Associate this symbol with the current Fortran module.  */

void
f_module_sym_add (const struct symbol *sym)
{
  if (!open_module)
    {
      printf_filtered ("f_module_sym_add: not currently in a module\n");
      return;
    }

  /* Some compilers create a DW_TAG_subprogram entry with the same
     name as the encompassing module. Ignore.  */
  if (strcasecmp (open_module->name, sym->ginfo.name) == 0)
    {
      return;
    }

  {
    struct modtab_symbol *msym;

    msym = (struct modtab_symbol *) obstack_alloc (&open_module_objfile->objfile_obstack, sizeof (struct modtab_symbol));
    msym->sym = sym;
    msym->next = open_module->sym_list;
    open_module->sym_list = msym;
  }
}

static int module_name_matcher (const char *name, void *arg)
{
  return strstr (name, "::") != NULL;
}

static int file_name_matcher (const char *name, void *arg, int basenames)
{
  return 1;
}

/* Pretty-print a module's name.

   Called during the walk of the hash-table of all modules.

   If this is NOT the first time it has been called (during this
   particular "walk"), put a nice comma into the output.  */

static int
print_module_name (void **slot, void *arg)
{
  const struct modtab_entry **module = (const struct modtab_entry **) slot;
  int *first = (int *) arg;

  if (*first)
    {
      *first = 0;
    }
  else
    {
      printf_filtered (", ");
    }

  printf_filtered ("%s", (*module)->name);

  return 1;
}

/* User-command. Retrieve and print a list of all Fortran modules.  */

static void
modules_info (char *ignore, int from_tty)
{
  int first = 1;
  struct objfile *objfile;

  printf_filtered ("All defined modules:\n\n");

  ALL_OBJFILES(objfile)
  {
    struct htab *modtab = objfile_data (objfile, f_module_objfile_data_key);
    if (modtab)
      htab_traverse_noresize (modtab, print_module_name, &first);
  }

  printf_filtered ("\n");
}

static void
print_module_symbols_ex (const struct modtab_entry *module,
                         const enum search_domain kind, struct objfile *objfile)
{
  const struct modtab_symbol *sym_list;

  printf_filtered ("\nModule %s:\n", module->name);

  /* Expand any symtabs with module symbols in.  */
  objfile->sf->qf->expand_symtabs_matching (objfile,
					  file_name_matcher,
					  module_name_matcher,
					  kind,
					  NULL);

  /* Careful! Need to grab the sym_list AFTER converting the psymtab
     into a symtab (above).  */
  sym_list = module->sym_list;

  while (sym_list)
    {
      const struct symbol *sym = sym_list->sym;

      if ((kind == VARIABLES_DOMAIN && SYMBOL_CLASS (sym) != LOC_TYPEDEF &&
           SYMBOL_CLASS (sym) != LOC_BLOCK) ||
          (kind == FUNCTIONS_DOMAIN && SYMBOL_CLASS (sym) == LOC_BLOCK))
        {
	  const char *fullname;

          type_print (SYMBOL_TYPE (sym),
                      (SYMBOL_CLASS (sym) == LOC_TYPEDEF
                       ? "" : SYMBOL_PRINT_NAME (sym)),
                      gdb_stdout, 0);

	  fullname = symtab_to_fullname (sym->symtab);
          printf_filtered (";%s;%d;\n",
                           (fullname ?
                            fullname :
                            ""),
                           sym->line);
        }

      sym_list = sym_list->next;
    }
}

struct print_module_symbols_data
{
  struct objfile *objfile;
  enum search_domain kind;
};

/* Pretty-print a module's symbols.

   Passed in the module concerned and an indication as to the sort of
   information (either variables or functions) required.

   If the full symbol-information for the module's object file has not
   yet been read, will cause that to happen first.  */

static int
print_module_symbols (void **slot, void *arg)
{
  const struct modtab_entry *module = *((const struct modtab_entry **) slot);
  const struct print_module_symbols_data *data = (const struct print_module_symbols_data *) arg;
  const enum search_domain kind = data->kind;
  struct objfile *objfile = data->objfile;

  print_module_symbols_ex (module, kind, objfile);

  return 1;
}

/* Retrieve and print info for a named Fortran module.

   If no module named, retrieve and print for ALL modules.

   Needs to be told the sort of information required. Currently,
   this is either functions or variables.  */

static void
module_symbol_info (char *module_name, enum search_domain kind, int from_tty)
{
  static const char *classnames[] =
    {
      "variable",
      "function"
    };

  if (module_name)
    {
      const struct modtab_entry *found_entry;
      struct objfile *objfile;

      printf_filtered ("All defined module %ss for \"%s\":\n",
                       classnames[(int) (kind - VARIABLES_DOMAIN)],
                       module_name);

      ALL_OBJFILES(objfile)
      {
	found_entry = f_module_lookup (objfile, module_name);

	if (found_entry)
	  print_module_symbols_ex (found_entry, kind, objfile);
      }
    }
  else
    {
      struct objfile *objfile;

      printf_filtered ("All defined module %ss:\n",
                       classnames[(int) (kind - VARIABLES_DOMAIN)]);

      ALL_OBJFILES(objfile)
      {
	struct htab *modtab = objfile_data (objfile, f_module_objfile_data_key);
	if (modtab)
	  {
	    struct print_module_symbols_data data;
	    data.objfile = objfile;
	    data.kind = kind;
	    htab_traverse_noresize (modtab, print_module_symbols, &data);
	  }
      }
    }

  printf_filtered ("\n");
}

/* User-command. Retrieve and print a list of all functions for the
   named Fortran module.
   
   If no module named, do all modules.  */

static void
module_functions (char *module_name, int from_tty)
{
  module_symbol_info (module_name, FUNCTIONS_DOMAIN, from_tty);
}

/* User-command. Retrieve and print a list of all variables for the
   named Fortran module.
   
   If no module named, do all modules.  */

static void
module_variables (char *module_name, int from_tty)
{
  module_symbol_info (module_name, VARIABLES_DOMAIN, from_tty);
}

static void
f_module_per_objfile_free (struct objfile *objfile, void *d)
{
  struct htab *modtab = (struct htab *)d;
  
  if (modtab)
    htab_delete (modtab);
}

/* Initialise the Fortran module code.  */

extern void _initialize_f_module (void);

void
_initialize_f_module (void)
{
  f_module_objfile_data_key
    = register_objfile_data_with_cleanup (NULL, f_module_per_objfile_free);

  add_info ("modules", modules_info,
            _("All Fortran 90 modules."));
  add_info ("module_functions", module_functions,
            _("All global functions for the named Fortran 90 module."));
  add_info ("module_variables", module_variables,
            _("All global variables for the named Fortran 90 module."));
}
