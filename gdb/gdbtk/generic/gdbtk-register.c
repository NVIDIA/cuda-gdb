/* Tcl/Tk command definitions for Insight - Registers
   Copyright 2001 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

#include "defs.h"
#include "frame.h"
#include "value.h"

#include <tcl.h>
#include "gdbtk.h"
#include "gdbtk-cmds.h"

/* This contains the previous values of the registers, since the last call to
   gdb_changed_register_list.  */

static char *old_regs;

static int gdb_changed_register_list (ClientData, Tcl_Interp *, int,
				      Tcl_Obj * CONST[]);
static int gdb_fetch_registers (ClientData, Tcl_Interp *, int,
				Tcl_Obj * CONST[]);
static int gdb_regnames (ClientData, Tcl_Interp *, int, Tcl_Obj * CONST[]);
static int get_pc_register (ClientData, Tcl_Interp *, int, Tcl_Obj * CONST[]);
static void get_register (int, void *);
static void get_register_name (int, void *);
static int map_arg_registers (int, Tcl_Obj * CONST[],
			      void (*)(int, void *), void *);
static void register_changed_p (int, void *);
static void setup_architecture_data (void);

int
Gdbtk_Register_Init (Tcl_Interp *interp)
{
  Tcl_CreateObjCommand (interp, "gdb_changed_register_list", gdbtk_call_wrapper,
			gdb_changed_register_list, NULL);
  Tcl_CreateObjCommand (interp, "gdb_fetch_registers", gdbtk_call_wrapper,
			gdb_fetch_registers, NULL);
  Tcl_CreateObjCommand (interp, "gdb_regnames", gdbtk_call_wrapper, gdb_regnames,
			NULL);
  Tcl_CreateObjCommand (interp, "gdb_pc_reg", gdbtk_call_wrapper, get_pc_register,
			NULL);

  /* Register/initialize any architecture specific data */
  setup_architecture_data ();
  register_gdbarch_swap (&old_regs, sizeof (old_regs), NULL);
  register_gdbarch_swap (NULL, 0, setup_architecture_data);

  return TCL_OK;
}

/* This implements the tcl command "gdb_changed_register_list"
 * It takes a list of registers, and returns a list of
 * the registers on that list that have changed since the last
 * time the proc was called.
 *
 * Tcl Arguments:
 *    A list of registers.
 * Tcl Result:
 *    A list of changed registers.
 */
static int
gdb_changed_register_list (clientData, interp, objc, objv)
     ClientData clientData;
     Tcl_Interp *interp;
     int objc;
     Tcl_Obj *CONST objv[];
{
  objc--;
  objv++;

  return map_arg_registers (objc, objv, register_changed_p, NULL);
}

/* This implements the tcl command gdb_fetch_registers
 * Pass it a list of register names, and it will
 * return their values as a list.
 *
 * Tcl Arguments:
 *    format: The format string for printing the values
 *    args: the registers to look for
 * Tcl Result:
 *    A list of their values.
 */
static int
gdb_fetch_registers (clientData, interp, objc, objv)
     ClientData clientData;
     Tcl_Interp *interp;
     int objc;
     Tcl_Obj *CONST objv[];
{
  int format, result;

  if (objc < 2)
    {
      Tcl_WrongNumArgs (interp, 1, objv, "format ?register1 register2 ...?");
      return TCL_ERROR;
    }
  objc -= 2;
  objv++;
  format = *(Tcl_GetStringFromObj (objv[0], NULL));
  objv++;

  if (objc != 1)
    result_ptr->flags |= GDBTK_MAKES_LIST;    /* Output the results as a list */
  result = map_arg_registers (objc, objv, get_register, (void *) format);
  if (objc != 1)
    result_ptr->flags &= ~GDBTK_MAKES_LIST;

  return result;
}

/* This implements the TCL command `gdb_regnames'.  Its syntax is:

   gdb_regnames [-numbers] [REGNUM ...]

   Return a list containing the names of the registers whose numbers
   are given by REGNUM ... .  If no register numbers are given, return
   all the registers' names.

   Note that some processors have gaps in the register numberings:
   even if there is no register numbered N, there may still be a
   register numbered N+1.  So if you call gdb_regnames with no
   arguments, you can't assume that the N'th element of the result is
   register number N.

   Given the -numbers option, gdb_regnames returns, not a list of names,
   but a list of pairs {NAME NUMBER}, where NAME is the register name,
   and NUMBER is its number.  */
static int
gdb_regnames (clientData, interp, objc, objv)
     ClientData clientData;
     Tcl_Interp *interp;
     int objc;
     Tcl_Obj *CONST objv[];
{
  int numbers = 0;

  objc--;
  objv++;

  if (objc >= 1)
    {
      char *s = Tcl_GetStringFromObj (objv[0], NULL);
      if (STREQ (s, "-numbers"))
	numbers = 1;
      objc--;
      objv++;
    }

  return map_arg_registers (objc, objv, get_register_name, &numbers);
}

/* This implements the tcl command get_pc_reg
 * It returns the value of the PC register
 *
 * Tcl Arguments:
 *    None
 * Tcl Result:
 *    The value of the pc register.
 */
static int
get_pc_register (clientData, interp, objc, objv)
     ClientData clientData;
     Tcl_Interp *interp;
     int objc;
     Tcl_Obj *CONST objv[];
{
  char *buff;

  xasprintf (&buff, "0x%llx", (long long) read_register (PC_REGNUM));
  Tcl_SetStringObj (result_ptr->obj_ptr, buff, -1);
  free(buff);
  return TCL_OK;
}

static void
get_register (regnum, fp)
     int regnum;
     void *fp;
{
  struct type *reg_vtype;
  char raw_buffer[MAX_REGISTER_RAW_SIZE];
  char virtual_buffer[MAX_REGISTER_VIRTUAL_SIZE];
  int format = (int) fp;
  int optim;

  if (format == 'N')
    format = 0;

  /* read_relative_register_raw_bytes returns a virtual frame pointer
     (FRAME_FP (selected_frame)) if regnum == FP_REGNUM instead
     of the real contents of the register. To get around this,
     use get_saved_register instead. */
  get_saved_register (raw_buffer, &optim, (CORE_ADDR *) NULL, selected_frame,
		      regnum, (enum lval_type *) NULL);
  if (optim)
    {
      Tcl_ListObjAppendElement (NULL, result_ptr->obj_ptr,
				Tcl_NewStringObj ("Optimized out", -1));
      return;
    }

  /* Convert raw data to virtual format if necessary.  */

  reg_vtype = REGISTER_VIRTUAL_TYPE (regnum);
  if (REGISTER_CONVERTIBLE (regnum))
    {
      REGISTER_CONVERT_TO_VIRTUAL (regnum, reg_vtype,
      				   raw_buffer, virtual_buffer);
    }
  else
    memcpy (virtual_buffer, raw_buffer, REGISTER_VIRTUAL_SIZE (regnum));

  if (format == 'r')
    {
      int j;
      char *ptr, buf[1024];

      strcpy (buf, "0x");
      ptr = buf + 2;
      for (j = 0; j < REGISTER_RAW_SIZE (regnum); j++)
	{
	  register int idx = TARGET_BYTE_ORDER == BIG_ENDIAN ? j
	  : REGISTER_RAW_SIZE (regnum) - 1 - j;
	  sprintf (ptr, "%02x", (unsigned char) raw_buffer[idx]);
	  ptr += 2;
	}
      fputs_filtered (buf, gdb_stdout);
    }
  else
    if ((TYPE_CODE (reg_vtype) == TYPE_CODE_UNION)
        && (strcmp (FIELD_NAME (TYPE_FIELD (reg_vtype, 0)), REGISTER_NAME (regnum)) == 0))
      {
        val_print (FIELD_TYPE (TYPE_FIELD (reg_vtype, 0)), virtual_buffer, 0, 0,
	           gdb_stdout, format, 1, 0, Val_pretty_default);
      }
    else
      val_print (REGISTER_VIRTUAL_TYPE (regnum), virtual_buffer, 0, 0,
	         gdb_stdout, format, 1, 0, Val_pretty_default);

}

static void
get_register_name (regnum, argp)
     int regnum;
     void *argp;
{
  /* Non-zero if the caller wants the register numbers, too.  */
  int numbers = * (int *) argp;
  Tcl_Obj *name = Tcl_NewStringObj (REGISTER_NAME (regnum), -1);
  Tcl_Obj *elt;

  if (numbers)
    {
      /* Build a tuple of the form "{REGNAME NUMBER}", and append it to
	 our result.  */
      Tcl_Obj *array[2];

      array[0] = name;
      array[1] = Tcl_NewIntObj (regnum);
      elt = Tcl_NewListObj (2, array);
    }
  else
    elt = name;

  Tcl_ListObjAppendElement (NULL, result_ptr->obj_ptr, elt);
}

/* This is a sort of mapcar function for operations on registers */

static int
map_arg_registers (objc, objv, func, argp)
     int objc;
     Tcl_Obj *CONST objv[];
     void (*func) (int regnum, void *argp);
     void *argp;
{
  int regnum, numregs;

  /* Note that the test for a valid register must include checking the
     REGISTER_NAME because NUM_REGS may be allocated for the union of
     the register sets within a family of related processors.  In this
     case, some entries of REGISTER_NAME will change depending upon
     the particular processor being debugged.  */

  numregs = NUM_REGS + NUM_PSEUDO_REGS;
  
  if (objc == 0)		/* No args, just do all the regs */
    {
      for (regnum = 0;
	   regnum < numregs;
	   regnum++)
	{
	  if (REGISTER_NAME (regnum) == NULL
	      || *(REGISTER_NAME (regnum)) == '\0')
	    continue;
	  
	  func (regnum, argp);
	}
      
      return TCL_OK;
    }

  /* Else, list of register #s, just do listed regs */
  for (; objc > 0; objc--, objv++)
    {
      if (Tcl_GetIntFromObj (NULL, *objv, &regnum) != TCL_OK)
	{
	  result_ptr->flags |= GDBTK_IN_TCL_RESULT;
	  return TCL_ERROR;
	}

      if (regnum >= 0
	  && regnum < numregs
	  && REGISTER_NAME (regnum) != NULL
	  && *REGISTER_NAME (regnum) != '\000')
	func (regnum, argp);
      else
	{
	  Tcl_SetStringObj (result_ptr->obj_ptr, "bad register number", -1);
	  return TCL_ERROR;
	}
    }

  return TCL_OK;
}

static void
register_changed_p (regnum, argp)
     int regnum;
     void *argp;		/* Ignored */
{
  char raw_buffer[MAX_REGISTER_RAW_SIZE];

  if (read_relative_register_raw_bytes (regnum, raw_buffer))
    return;

  if (memcmp (&old_regs[REGISTER_BYTE (regnum)], raw_buffer,
	      REGISTER_RAW_SIZE (regnum)) == 0)
    return;

  /* Found a changed register.  Save new value and return its number. */

  memcpy (&old_regs[REGISTER_BYTE (regnum)], raw_buffer,
	  REGISTER_RAW_SIZE (regnum));

  Tcl_ListObjAppendElement (NULL, result_ptr->obj_ptr, Tcl_NewIntObj (regnum));
}

static void
setup_architecture_data ()
{
  /* don't trust REGISTER_BYTES to be zero. */
  old_regs = xmalloc (REGISTER_BYTES + 1);
  memset (old_regs, 0, REGISTER_BYTES + 1);
}

