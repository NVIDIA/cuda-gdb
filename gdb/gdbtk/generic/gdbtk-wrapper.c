/* longjmp-free interface between gdb and gdbtk.
   Copyright (C) 1999, 2000, 2002 Free Software Foundation, Inc.

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
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

#include "defs.h"
#include "frame.h"
#include "value.h"
#include "varobj.h"
#include "block.h"
#include "exceptions.h"
#include "language.h"
#include "gdbtk-wrapper.h"

/*
 * Wrapper functions exported to the world
 */

gdb_result GDB_value_fetch_lazy (value_ptr);

gdb_result GDB_evaluate_expression (struct expression *, value_ptr *);

gdb_result GDB_type_print (value_ptr, char *, struct ui_file *, int);

gdb_result GDB_val_print (struct type *type, char *valaddr,
			  CORE_ADDR address, struct ui_file *stream,
			  int format, int deref_ref, int recurse,
			  enum val_prettyprint pretty);

gdb_result GDB_value_equal (value_ptr, value_ptr, int *);

gdb_result GDB_parse_exp_1 (char **stringptr, struct block *block, int comma,
			    struct expression **result);

gdb_result GDB_evaluate_type (struct expression *exp, value_ptr * result);

gdb_result GDB_block_for_pc (CORE_ADDR pc, struct block **result);

gdb_result GDB_block_innermost_frame (struct block *block,
				      struct frame_info **result);

gdb_result GDB_reinit_frame_cache (void);

gdb_result GDB_value_ind (value_ptr val, value_ptr * rval);

gdb_result GDB_value_slice (value_ptr val, int low, int num,
			    value_ptr * rval);

gdb_result GDB_value_coerce_array (value_ptr val, value_ptr * rval);

gdb_result GDB_value_struct_elt (value_ptr * argp, value_ptr * args,
				 char *name, int *static_memfunc,
				 char *err, value_ptr * rval);

gdb_result GDB_value_cast (struct type *type, value_ptr val,
			   value_ptr * rval);

gdb_result GDB_get_frame_block (struct frame_info *fi, struct block **rval);

gdb_result GDB_get_prev_frame (struct frame_info *fi,
			       struct frame_info **result);

gdb_result GDB_get_next_frame (struct frame_info *fi,
			       struct frame_info **result);

gdb_result GDB_find_relative_frame (struct frame_info *fi,
				    int *start, struct frame_info **result);

gdb_result GDB_get_current_frame (struct frame_info **result);

gdb_result GDB_varobj_update (struct varobj **varp,
			      struct varobj ***changelist, int explicit,
			      int *result);

/*
 * Private functions for this file
 */
static gdb_result call_wrapped_function (catch_errors_ftype *,
					 struct gdb_wrapper_arguments *);

static int wrap_type_print (char *);

static int wrap_evaluate_expression (char *);

static int wrap_value_fetch_lazy (char *);

static int wrap_val_print (char *);

static int wrap_value_equal (char *);

static int wrap_parse_exp_1 (char *opaque_arg);

static int wrap_evaluate_type (char *opaque_arg);

static int wrap_block_for_pc (char *opaque_arg);

static int wrap_block_innermost_frame (char *opaque_arg);

static int wrap_reinit_frame_cache (char *opaque_arg);

static int wrap_value_ind (char *opaque_arg);

static int wrap_value_slice (char *opaque_arg);

static int wrap_value_coerce_array (char *opaque_arg);

static int wrap_value_struct_elt (char *opaque_arg);

static int wrap_value_cast (char *opaque_arg);

static int wrap_get_frame_block (char *opaque_arg);

static int wrap_get_prev_frame (char *opaque_arg);

static int wrap_get_next_frame (char *opaque_arg);

static int wrap_find_relative_frame (char *opaque_arg);

static int wrap_get_current_frame (char *opaque_arg);

static int wrap_varobj_update (char *opaque_arg);

static gdb_result
call_wrapped_function (catch_errors_ftype *fn, struct gdb_wrapper_arguments *arg)
{
  if (!catch_errors (fn, (char *) &arg, "", RETURN_MASK_ERROR))
    {
      /* An error occurred */
      return GDB_ERROR;
    }

  return GDB_OK;
}

gdb_result
GDB_type_print (value_ptr val, char *varstring,
		struct ui_file *stream, int show)
{
  struct gdb_wrapper_arguments args;

  args.args[0].ptr = val;
  args.args[1].ptr = varstring;
  args.args[2].ptr = stream;
  args.args[3].integer = show;
  return call_wrapped_function ((catch_errors_ftype *) wrap_type_print, &args);
}

static int
wrap_type_print (char *a)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) a;
  value_ptr val = (value_ptr) (*args)->args[0].ptr;
  char *varstring = (*args)->args[1].ptr;
  struct ui_file *stream = (struct ui_file *) (*args)->args[2].ptr;
  int show = (*args)->args[3].integer;
  type_print (value_type (val), varstring, stream, show);
  return 1;
}

gdb_result
GDB_val_print (struct type *type,
	       char *valaddr,
	       CORE_ADDR address,
	       struct ui_file *stream,
	       int format,
	       int deref_ref,
	       int recurse,
	       enum val_prettyprint pretty)
{
  struct gdb_wrapper_arguments args;

  args.args[0].ptr = type;
  args.args[1].ptr = valaddr;
  args.args[2].ptr = &address;
  args.args[3].ptr = stream;
  args.args[4].integer = format;
  args.args[5].integer = deref_ref;
  args.args[6].integer = recurse;
  args.args[7].integer = pretty;

  return call_wrapped_function ((catch_errors_ftype *) wrap_val_print, &args);
}

static int
wrap_val_print (char *a)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) a;
  struct type *type;
  const gdb_byte *valaddr;
  CORE_ADDR address;
  struct ui_file *stream;
  int format;
  int deref_ref;
  int recurse;
  enum val_prettyprint pretty;

  type = (struct type *) (*args)->args[0].ptr;
  valaddr = (gdb_byte *) (*args)->args[1].ptr;
  address = *(CORE_ADDR *) (*args)->args[2].ptr;
  stream = (struct ui_file *) (*args)->args[3].ptr;
  format = (*args)->args[4].integer;
  deref_ref = (*args)->args[5].integer;
  recurse = (*args)->args[6].integer;
  pretty = (enum val_prettyprint) (*args)->args[7].integer;

  val_print (type, valaddr, 0, address, stream, format, deref_ref,
	     recurse, pretty, current_language);
  return 1;
}

gdb_result
GDB_value_fetch_lazy (value_ptr value)
{
  struct gdb_wrapper_arguments args;

  args.args[0].ptr = value;
  return call_wrapped_function ((catch_errors_ftype *) wrap_value_fetch_lazy, &args);
}

static int
wrap_value_fetch_lazy (char *a)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) a;

  value_fetch_lazy ((value_ptr) (*args)->args[0].ptr);
  return 1;
}

gdb_result
GDB_evaluate_expression (struct expression *exp, value_ptr *value)
{
  struct gdb_wrapper_arguments args;
  gdb_result result;
  args.args[0].ptr = exp;

  result = call_wrapped_function ((catch_errors_ftype *) wrap_evaluate_expression, &args);
  if (result != GDB_OK)
    return result;

  *value = (value_ptr) args.result.ptr;
  return GDB_OK;
}

static int
wrap_evaluate_expression (char *a)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) a;

  (*args)->result.ptr =
    evaluate_expression ((struct expression *) (*args)->args[0].ptr);
  return 1;
}

gdb_result
GDB_value_equal (val1, val2, result)
     value_ptr val1;
     value_ptr val2;
     int *result;
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = val1;
  args.args[1].ptr = val2;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_value_equal, &args);
  if (r != GDB_OK)
    return r;

  *result = args.result.integer;
  return GDB_OK;
}

static int
wrap_value_equal (char *a)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) a;
  value_ptr val1, val2;

  val1 = (value_ptr) (*args)->args[0].ptr;
  val2 = (value_ptr) (*args)->args[1].ptr;

  (*args)->result.integer = value_equal (val1, val2);
  return 1;
}

gdb_result
GDB_parse_exp_1 (char **stringptr, struct block *block,
		 int comma, struct expression **result)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = stringptr;
  args.args[1].ptr = block;
  args.args[2].integer = comma;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_parse_exp_1, &args);
  if (r != GDB_OK)
    return r;

  *result = (struct expression *) args.result.ptr;
  return GDB_OK;
}

static int
wrap_parse_exp_1 (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  struct block *block;
  char **stringptr;
  int comma;

  stringptr = (char **) (*args)->args[0].ptr;
  block = (struct block *) (*args)->args[1].ptr;
  comma = (*args)->args[2].integer;

  (*args)->result.ptr = parse_exp_1 (stringptr, block, comma);
  return 1;
}

gdb_result
GDB_evaluate_type (struct expression *exp, value_ptr *result)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = exp;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_evaluate_type, &args);
  if (r != GDB_OK)
    return r;

  *result = (value_ptr) args.result.ptr;
  return GDB_OK;
}

static int
wrap_evaluate_type (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  struct expression *exp;

  exp = (struct expression *) (*args)->args[0].ptr;
  (*args)->result.ptr = evaluate_type (exp);
  return 1;
}

gdb_result
GDB_block_for_pc (CORE_ADDR pc, struct block **result)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = &pc;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_block_for_pc, &args);
  if (r != GDB_OK)
    return r;

  *result = (struct block *) args.result.ptr;
  return GDB_OK;
}

static int
wrap_block_for_pc (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  CORE_ADDR pc;

  pc = *(CORE_ADDR *) (*args)->args[0].ptr;
  (*args)->result.ptr = block_for_pc (pc);
  return 1;
}

gdb_result
GDB_block_innermost_frame (struct block *block, struct frame_info **result)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = block;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_block_innermost_frame, &args);
  if (r != GDB_OK)
    return r;

  *result = (struct frame_info *) args.result.ptr;
  return GDB_OK;
}

static int
wrap_block_innermost_frame (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  struct block *block;

  block = (struct block *) (*args)->args[0].ptr;
  (*args)->result.ptr = block_innermost_frame (block);
  return 1;
}

gdb_result
GDB_reinit_frame_cache ()
{
  gdb_result r;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_reinit_frame_cache, NULL);
  if (r != GDB_OK)
    return r;

  return GDB_OK;
}

static int
wrap_reinit_frame_cache (char *opaque_arg)
{
  reinit_frame_cache ();
  return 1;
}

gdb_result
GDB_value_ind (value_ptr val, value_ptr *rval)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = val;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_value_ind, &args);
  if (r != GDB_OK)
    return r;

  *rval = (value_ptr) args.result.ptr;
  return GDB_OK;
}

static int
wrap_value_ind (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  value_ptr val;

  val = (value_ptr) (*args)->args[0].ptr;
  (*args)->result.ptr = value_ind (val);
  return 1;
}

gdb_result
GDB_value_slice (value_ptr val, int low, int num, value_ptr *rval)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = val;
  args.args[1].integer = low;
  args.args[2].integer = num;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_value_slice, &args);
  if (r != GDB_OK)
    return r;

  *rval = (value_ptr) args.result.ptr;
  return GDB_OK;
}

static int
wrap_value_slice (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  value_ptr val;
  int low, num;

  val = (value_ptr) (*args)->args[0].ptr;
  low = (*args)->args[1].integer;
  num = (*args)->args[2].integer;
  (*args)->result.ptr = value_slice (val, low, num);
  return 1;
}

gdb_result
GDB_value_coerce_array (val, rval)
     value_ptr val;
     value_ptr *rval;
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = val;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_value_coerce_array,
			     &args);
  if (r != GDB_OK)
    return r;

  *rval = (value_ptr) args.result.ptr;
  return GDB_OK;
}

static int
wrap_value_coerce_array (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  value_ptr val;

  val = (value_ptr) (*args)->args[0].ptr;
  (*args)->result.ptr = value_coerce_array (val);
  return 1;
}

gdb_result
GDB_value_struct_elt (value_ptr *argp,
		      value_ptr *args,
		      char *name,
		      int *static_memfunc,
		      char *err,
		      value_ptr *rval)
{
  struct gdb_wrapper_arguments argss;
  gdb_result r;

  argss.args[0].ptr = argp;
  argss.args[1].ptr = args;
  argss.args[2].ptr = name;
  argss.args[3].ptr = static_memfunc;
  argss.args[4].ptr = err;
  r = call_wrapped_function ((catch_errors_ftype *) wrap_value_struct_elt, &argss);
  if (r != GDB_OK)
    return r;

  *rval = (value_ptr) argss.result.ptr;
  return GDB_OK;
}

static int
wrap_value_struct_elt (char *opaque_arg)
{
  struct gdb_wrapper_arguments **argss = (struct gdb_wrapper_arguments **) opaque_arg;
  value_ptr *argp, *args;
  char *name;
  int *static_memfunc;
  char *err;

  argp = (value_ptr *) (*argss)->args[0].ptr;
  args = (value_ptr *) (*argss)->args[1].ptr;
  name = (char *) (*argss)->args[2].ptr;
  static_memfunc = (int *) (*argss)->args[3].ptr;
  err = (char *) (*argss)->args[4].ptr;

  (*argss)->result.ptr = value_struct_elt (argp, args, name, static_memfunc, err);
  return 1;
}

gdb_result
GDB_value_cast (struct type *type, value_ptr val, value_ptr *rval)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = type;
  args.args[1].ptr = val;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_value_cast, &args);
  if (r != GDB_OK)
    return r;

  *rval = (value_ptr) args.result.ptr;
  return GDB_OK;
}

static int
wrap_value_cast (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  value_ptr val;
  struct type *type;

  type = (struct type *) (*args)->args[0].ptr;
  val = (value_ptr) (*args)->args[1].ptr;
  (*args)->result.ptr = value_cast (type, val);

  return 1;
}

gdb_result
GDB_get_frame_block (struct frame_info *fi, struct block **rval)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = fi;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_get_frame_block, &args);
  if (r != GDB_OK)
    return r;

  *rval = (struct block *) args.result.ptr;
  return GDB_OK;
}

static int
wrap_get_frame_block (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  struct frame_info *fi;

  fi = (struct frame_info *) (*args)->args[0].ptr;
  (*args)->result.ptr = get_frame_block (fi, NULL);

  return 1;
}

gdb_result
GDB_get_prev_frame (struct frame_info *fi, struct frame_info **result)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = fi;
  r = call_wrapped_function ((catch_errors_ftype *) wrap_get_prev_frame, &args);
  if (r != GDB_OK)
    return r;

  *result = (struct frame_info *) args.result.ptr;
  return GDB_OK;
}

static int
wrap_get_prev_frame (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  struct frame_info *fi = (struct frame_info *) (*args)->args[0].ptr;

  (*args)->result.ptr = get_prev_frame (fi);
  return 1;
}

gdb_result
GDB_get_next_frame (struct frame_info *fi, struct frame_info **result)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = fi;
  r = call_wrapped_function ((catch_errors_ftype *) wrap_get_next_frame, &args);
  if (r != GDB_OK)
    return r;

  *result = (struct frame_info *) args.result.ptr;
  return GDB_OK;
}

static int
wrap_get_next_frame (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  struct frame_info *fi = (struct frame_info *) (*args)->args[0].ptr;

  (*args)->result.ptr = get_next_frame (fi);
  return 1;
}

gdb_result
GDB_find_relative_frame (struct frame_info *fi, int *start,
			 struct frame_info **result)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = fi;
  args.args[1].ptr = start;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_find_relative_frame, 
			     &args);
  if (r != GDB_OK)
    return r;

  *result = (struct frame_info *) args.result.ptr;
  return GDB_OK;
}

static int
wrap_find_relative_frame (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;
  struct frame_info *fi = (struct frame_info *) (*args)->args[0].ptr;
  int *start = (int *) (*args)->args[1].ptr;

  (*args)->result.ptr = find_relative_frame (fi, start);
  return 1;
}

gdb_result
GDB_get_current_frame (struct frame_info **result)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_get_current_frame, 
			     &args);
  if (r != GDB_OK)
    return r;

  *result = (struct frame_info *) args.result.ptr;
  return GDB_OK;
}

static int
wrap_get_current_frame (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args = (struct gdb_wrapper_arguments **) opaque_arg;

  (*args)->result.ptr = get_current_frame ();
  return 1;
}

gdb_result
GDB_varobj_update (struct varobj **varp, struct varobj ***changelist,
		   int explicit, int *result)
{
  struct gdb_wrapper_arguments args;
  gdb_result r;

  args.args[0].ptr = varp;
  args.args[1].ptr = changelist;
  args.args[2].integer = explicit;

  r = call_wrapped_function ((catch_errors_ftype *) wrap_varobj_update, &args);
  if (r != GDB_OK)
    return r;

  *result = args.result.integer;
  return GDB_OK;
}

static int wrap_varobj_update (char *opaque_arg)
{
  struct gdb_wrapper_arguments **args
    = (struct gdb_wrapper_arguments **) opaque_arg;
  struct varobj **varp = (struct varobj **) (*args)->args[0].ptr;
  struct varobj ***changelist = (struct varobj ***) (*args)->args[1].ptr;
  int explicit = (*args)->args[2].integer;

  (*args)->result.integer = varobj_update (varp, changelist, explicit);
  return 1;
}
