/* Fortran language support routines for GDB, the GNU debugger.

   Copyright (C) 1993-2013 Free Software Foundation, Inc.

   Contributed by Motorola.  Adapted from the C parser by Farooq Butt
   (fmbutt@engage.sps.mot.com).

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

#include "defs.h"
#include "gdb_string.h"
#include "symtab.h"
#include "gdbtypes.h"
#include "expression.h"
#include "parser-defs.h"
#include "language.h"
#include "f-lang.h"
#include "f-module.h"
#include "valprint.h"
#include "value.h"
#include "cp-support.h"
#include "charset.h"
#include "c-lang.h"
#include "dwarf2loc.h"          /* dl added this for the baton stuff */


/* Local functions */

extern void _initialize_f_language (void);

static void f_printchar (int c, struct type *type, struct ui_file * stream);
static void f_emit_char (int c, struct type *type,
			 struct ui_file * stream, int quoter);

/* Return the encoding that should be used for the character type
   TYPE.  */

static const char *
f_get_encoding (struct type *type)
{
  const char *encoding;

  switch (TYPE_LENGTH (type))
    {
    case 1:
      encoding = target_charset (get_type_arch (type));
      break;
    case 4:
      if (gdbarch_byte_order (get_type_arch (type)) == BFD_ENDIAN_BIG)
	encoding = "UTF-32BE";
      else
	encoding = "UTF-32LE";
      break;

    default:
      error (_("unrecognized character type"));
    }

  return encoding;
}

/* Print the character C on STREAM as part of the contents of a literal
   string whose delimiter is QUOTER.  Note that that format for printing
   characters and strings is language specific.
   FIXME:  This is a copy of the same function from c-exp.y.  It should
   be replaced with a true F77 version.  */

static void
f_emit_char (int c, struct type *type, struct ui_file *stream, int quoter)
{
  const char *encoding = f_get_encoding (type);

  generic_emit_char (c, type, stream, quoter, encoding);
}

/* Implementation of la_printchar.  */

static void
f_printchar (int c, struct type *type, struct ui_file *stream)
{
  fputs_filtered ("'", stream);
  LA_EMIT_CHAR (c, type, stream, '\'');
  fputs_filtered ("'", stream);
}

/* Print the character string STRING, printing at most LENGTH characters.
   Printing stops early if the number hits print_max; repeat counts
   are printed as appropriate.  Print ellipses at the end if we
   had to stop before printing LENGTH characters, or if FORCE_ELLIPSES.
   FIXME:  This is a copy of the same function from c-exp.y.  It should
   be replaced with a true F77 version.  */

static void
f_printstr (struct ui_file *stream, struct type *type, const gdb_byte *string,
	    unsigned int length, const char *encoding, int force_ellipses,
	    const struct value_print_options *options)
{
  const char *type_encoding = f_get_encoding (type);

  if (TYPE_LENGTH (type) == 4)
    fputs_filtered ("4_", stream);

  if (!encoding || !*encoding)
    encoding = type_encoding;

  generic_printstr (stream, type, string, length, encoding,
		    force_ellipses, '\'', 0, options);
}

static void
reset_lengths(struct type* type) 
{
  struct type* range_type;
  int high_bound; 
  int low_bound; 
  if (TYPE_CODE(TYPE_TARGET_TYPE(type)) == TYPE_CODE_ARRAY || 
     TYPE_CODE(TYPE_TARGET_TYPE(type)) == TYPE_CODE_STRING) 
    reset_lengths(TYPE_TARGET_TYPE(type));
  range_type = TYPE_FIELD_TYPE(type, 0);
  high_bound = TYPE_HIGH_BOUND (range_type);
  low_bound = TYPE_LOW_BOUND (range_type);
  TYPE_LENGTH(type) = 
    TYPE_LENGTH (TYPE_TARGET_TYPE(type)) * (high_bound - low_bound + 1);
}

static void 
setup_array_bounds (struct value *v, struct value *objptr, struct frame_info *frame)
{
  struct type *tmp_type = value_type (v);

  if (!value_address (objptr))
    return;
  
  if (frame == NULL)
    frame = deprecated_safe_get_selected_frame ();

  while (TYPE_CODE(tmp_type)== TYPE_CODE_ARRAY
	 || TYPE_CODE(tmp_type)== TYPE_CODE_STRING) 
   {
      /* check the array bounds -- could be dynamic */
      struct type* range_type = TYPE_FIELD_TYPE(tmp_type, 0);

      if (TYPE_CODE(range_type) == TYPE_CODE_RANGE)
	{
	  if (TYPE_BOUND_BATON_FUNCTION(range_type)) 
	    {
	      if (frame) {
		/* one of DL's hacked types */
		int high_bound;
		int low_bound;
		
		if (TYPE_LOW_BOUND_BATON(range_type)) 
		  {
		    void *baton;
		    long r;
		    /* recalculate */
		    CORE_ADDR(*f)(void*, struct value *, void*);
		    f = (CORE_ADDR(*)(void*, struct value *, void*))
		      TYPE_BOUND_BATON_FUNCTION(range_type);
		    
		    baton = TYPE_LOW_BOUND_BATON(range_type);
		    
		    r = (int) f(baton, objptr, frame);
		    
 		    if (r > 1000L * 1000L* 1000L || r < -100000L) r = 1; /* FIXME */ 
		    TYPE_LOW_BOUND(range_type) = (int) r;
		    TYPE_LOW_BOUND_UNDEFINED (range_type) = 0;
		  }
		if (TYPE_HIGH_BOUND_BATON(range_type)) 
		  {
		    /* recalculate */
		    void *baton;
		    long r;
		    CORE_ADDR(*f)(void*, struct value *, void*);
		    f = (CORE_ADDR(*)(void*, struct value *, void*))
		      TYPE_BOUND_BATON_FUNCTION(range_type);
		    
		    baton = TYPE_HIGH_BOUND_BATON(range_type);
		    
		    r = (int) f(baton, objptr, frame);
 		    if (r > 1000L * 1000L* 1000L || r < -100000L) r = 1; /* FIXME */ 
		    TYPE_HIGH_BOUND(range_type) = (int) r;
		    TYPE_HIGH_BOUND_UNDEFINED (range_type) = 0;
		  }
		if (TYPE_COUNT_BOUND_BATON(range_type)) 
		  { 
		    /* the DW_AT_count field */
		    void *baton;
		    CORE_ADDR r;
		    /* recalculate */
		    CORE_ADDR(*f)(void*, struct value *, void*);
		    f = (CORE_ADDR(*)(void*, struct value *, void*))
		      TYPE_BOUND_BATON_FUNCTION(range_type);
		    
		    baton = TYPE_COUNT_BOUND_BATON(range_type);
		    
		    r = (int) f(baton, objptr, frame);
		    
 		    if (r > 1000L * 1000L* 1000L) r = 1; /* FIXME */ 
		    /* note.. upper bound is inclusive */
		    TYPE_HIGH_BOUND(range_type) = ((int) r) + TYPE_LOW_BOUND(range_type) - 1;
		  }
		if (TYPE_LOW_BOUND(range_type) > 
		    TYPE_HIGH_BOUND(range_type)) 
		  {
		    TYPE_LOW_BOUND(range_type) = 
		      TYPE_HIGH_BOUND(range_type) + 1;
		    TYPE_HIGH_BOUND_UNDEFINED (range_type) = 0;
		    TYPE_LOW_BOUND_UNDEFINED (range_type) = 0;
		  }
	      }
	    }
	}
      tmp_type = TYPE_TARGET_TYPE(tmp_type);  
    }
  tmp_type = value_type (v);
  
  if (TYPE_CODE(tmp_type) == TYPE_CODE_ARRAY
      || TYPE_CODE(tmp_type) == TYPE_CODE_STRING)
    {
      reset_lengths(tmp_type);
    }
}

static CORE_ADDR
get_new_address (struct value *v, struct value *objptr, struct frame_info *frame)
{
  struct type *type = value_type (v);
  CORE_ADDR r = value_address (v);

  if (frame == NULL)
    frame = deprecated_safe_get_selected_frame ();

  /* array memory location is dynamic : eg. f90 */
  if (TYPE_CODE (type) == TYPE_CODE_ARRAY && TYPE_NFIELDS (type) == 2)
    {
      struct array_location_batons *baton_holder
	= (struct array_location_batons *) TYPE_FIELD_TYPE (type, 1);

      void *baton_evaluation_function;

      int alloced = -1;

      CORE_ADDR (*f) (void *, struct value *, void *);
      f = (CORE_ADDR (*)(void *, struct value *, void *))
	baton_holder->baton_evaluation_function;

      if (baton_holder->allocated_baton)
	{
	  alloced = (int) f (baton_holder->allocated_baton, objptr, frame);
         }

      if (!alloced)
        {
          r = (CORE_ADDR) 0;
        }

      if (r && baton_holder->intel_location_baton)
        {
	  r = f (baton_holder->intel_location_baton, objptr, frame);
	}


      if (r && baton_holder->pgi_lbase_baton)
	{
	  struct type *tmp_type = NULL;
	  int elem_skip = 1;
	  int lbase = 0;

	  if (baton_holder->pgi_elem_skip_baton)
	    {
	      elem_skip = (int) f (baton_holder->pgi_elem_skip_baton,
				   objptr, frame);
	    }
	  if (elem_skip == 0)
	    return (CORE_ADDR) 0;	/* seen with PGI for non-allocated arrays */

	  /* 
	     pgi uses lbase_baton to allow you to define an alloc array
	     in a subset of an existing array

	     the logic is to add l_base_baton, then to take subscript of
	     index and add the section offset of that index.  Ticket 4243 in
	     tracker has the pgi spec.

	     but, gdb at this point just wants to know where the array starts, it
	     can handle the rest -- assuming the 'section offset' is the obvious
	     place to find the ith index..

	   */

	  lbase = ((int) f (baton_holder->pgi_lbase_baton, objptr, frame));

	  r = r + (lbase - 1) * elem_skip;

	  if (frame == NULL)
	    frame = deprecated_safe_get_selected_frame ();

	  tmp_type = type;

	  while (TYPE_CODE (tmp_type) == TYPE_CODE_PTR)
	    tmp_type = TYPE_TARGET_TYPE (tmp_type);

	  while (TYPE_CODE (tmp_type) == TYPE_CODE_ARRAY
		 || TYPE_CODE (tmp_type) == TYPE_CODE_STRING)
	    {
	      /* check the array bounds -- could be dynamic */
	      struct type *range_type = TYPE_FIELD_TYPE (tmp_type, 0);

	      if (TYPE_CODE (range_type) == TYPE_CODE_RANGE && r)
		{
		  if (TYPE_BOUND_BATON_FUNCTION (range_type))
		    {
		      if (frame)
			{
			  int stride = 1;
			  int soffset = 0;
			  int lstride = 0;
                         int low = 1;

			  void *baton;
			  CORE_ADDR (*f) (void *, struct value *, void *);
			  f = TYPE_BOUND_BATON_FUNCTION (range_type);
 

			  low = TYPE_LOW_BOUND (range_type);
			  if (TYPE_LOW_BOUND_BATON (range_type))
			    {
			      baton = 
				TYPE_LOW_BOUND_BATON (range_type);
			      low = f (baton, objptr, frame);
			    }
			  stride = TYPE_STRIDE_VALUE(range_type);
			  if (TYPE_STRIDE_BATON(range_type))
			    {
			      baton =
				(void *)
				TYPE_STRIDE_BATON(range_type);
			      stride = f (baton, objptr, frame);
			    }
			  soffset = TYPE_SOFFSET_VALUE (range_type);
			  if (TYPE_SOFFSET_BATON (range_type))
			    {
			      baton =
				(void *)
				TYPE_SOFFSET_BATON (range_type);
			      soffset = f (baton, objptr, frame);
			    }
			  lstride = TYPE_LSTRIDE_VALUE (range_type);
			  if (TYPE_LSTRIDE_BATON (range_type))
			    {
			      baton =
				(void *)
				TYPE_LSTRIDE_BATON (range_type);
			      lstride = f (baton, objptr, frame);
			    }

			  r += (low * stride + soffset) * lstride * elem_skip;
			}
		    }
		}
	      tmp_type = TYPE_TARGET_TYPE (tmp_type);
	    }
	  tmp_type = type;
	}
    }

  return r;
}

struct value *
f_fixup_value (struct value *v, struct frame_info *frame)
{
  struct type *type;
  CORE_ADDR address;
  struct value *objptr = v;
  
  if (!v)
    return v;

  type = value_type (v);
  
  /* Automatically follow pointers.  */
  if (current_language->la_language == language_fortran)
    {
      while (TYPE_CODE (type) == TYPE_CODE_PTR)
	{
	  address = unpack_pointer (type, value_contents (v));

	  /* If address is invalid then leave the pointer intact.  */
	  if (address < 65536)
	    break;
	  type = TYPE_TARGET_TYPE (type);      
	  v = value_at_lazy (type, address);
	}
    }

  /* Fix the size and address of dynamic arrays.  */
  if (VALUE_LVAL (v) == lval_memory
      && (TYPE_CODE (type)== TYPE_CODE_ARRAY
	  || TYPE_CODE (type)== TYPE_CODE_STRING))
    {
      /* refresh the calculated array bounds DL */
      if (value_lazy (v))
	setup_array_bounds(v, objptr, frame);	
  
      /* Handle the situation where it is dynamic */
      if (TYPE_CODE (type) == TYPE_CODE_ARRAY)
	{
	  address = get_new_address (v, objptr, frame);
	  if (address != value_address (v))
	    {
	      set_value_address (v, address);
	      set_value_offset (v, 0);
	    }
	}
    }

  return v;
}

/* Table of operators and their precedences for printing expressions.  */

static const struct op_print f_op_print_tab[] =
{
  {"+", BINOP_ADD, PREC_ADD, 0},
  {"+", UNOP_PLUS, PREC_PREFIX, 0},
  {"-", BINOP_SUB, PREC_ADD, 0},
  {"-", UNOP_NEG, PREC_PREFIX, 0},
  {"*", BINOP_MUL, PREC_MUL, 0},
  {"/", BINOP_DIV, PREC_MUL, 0},
  {"DIV", BINOP_INTDIV, PREC_MUL, 0},
  {"MOD", BINOP_REM, PREC_MUL, 0},
  {"=", BINOP_ASSIGN, PREC_ASSIGN, 1},
  {".OR.", BINOP_LOGICAL_OR, PREC_LOGICAL_OR, 0},
  {".AND.", BINOP_LOGICAL_AND, PREC_LOGICAL_AND, 0},
  {".NOT.", UNOP_LOGICAL_NOT, PREC_PREFIX, 0},
  {".EQ.", BINOP_EQUAL, PREC_EQUAL, 0},
  {".NE.", BINOP_NOTEQUAL, PREC_EQUAL, 0},
  {".LE.", BINOP_LEQ, PREC_ORDER, 0},
  {".GE.", BINOP_GEQ, PREC_ORDER, 0},
  {".GT.", BINOP_GTR, PREC_ORDER, 0},
  {".LT.", BINOP_LESS, PREC_ORDER, 0},
  {"**", UNOP_IND, PREC_PREFIX, 0},
  {"@", BINOP_REPEAT, PREC_REPEAT, 0},
  {"ABS", UNOP_FABS, PREC_BUILTIN_FUNCTION, 0},
  {"AIMAG", UNOP_CIMAG, PREC_BUILTIN_FUNCTION, 0},
  {"REALPART", UNOP_CREAL, PREC_BUILTIN_FUNCTION, 0},
  {"CMPLX", BINOP_CMPLX, PREC_BUILTIN_FUNCTION, 0},
  {"IEEE_IS_INF", UNOP_IEEE_IS_INF, PREC_BUILTIN_FUNCTION, 0},
  {"IEEE_IS_FINITE", UNOP_IEEE_IS_FINITE, PREC_BUILTIN_FUNCTION, 0},
  {"IEEE_IS_NAN", UNOP_IEEE_IS_NAN, PREC_BUILTIN_FUNCTION, 0},
  {"IEEE_IS_NORMAL", UNOP_IEEE_IS_NORMAL, PREC_BUILTIN_FUNCTION, 0},
  {"CEILING", UNOP_CEIL, PREC_BUILTIN_FUNCTION, 0},
  {"FLOOR", UNOP_FLOOR, PREC_BUILTIN_FUNCTION, 0},
  {"MOD", BINOP_FMOD, PREC_BUILTIN_FUNCTION, 0},
  {NULL, 0, 0, 0}
};

enum f_primitive_types {
  f_primitive_type_character,
  f_primitive_type_logical,
  f_primitive_type_logical_s1,
  f_primitive_type_logical_s2,
  f_primitive_type_logical_s8,
  f_primitive_type_integer,
  f_primitive_type_integer_s2,
  f_primitive_type_integer_s8,
  f_primitive_type_real,
  f_primitive_type_real_s8,
  f_primitive_type_real_s16,
  f_primitive_type_complex_s8,
  f_primitive_type_complex_s16,
  f_primitive_type_void,
  nr_f_primitive_types
};

static void
f_language_arch_info (struct gdbarch *gdbarch,
		      struct language_arch_info *lai)
{
  const struct builtin_f_type *builtin = builtin_f_type (gdbarch);

  lai->string_char_type = builtin->builtin_character;
  lai->primitive_type_vector
    = GDBARCH_OBSTACK_CALLOC (gdbarch, nr_f_primitive_types + 1,
                              struct type *);

  lai->primitive_type_vector [f_primitive_type_character]
    = builtin->builtin_character;
  lai->primitive_type_vector [f_primitive_type_logical]
    = builtin->builtin_logical;
  lai->primitive_type_vector [f_primitive_type_logical_s1]
    = builtin->builtin_logical_s1;
  lai->primitive_type_vector [f_primitive_type_logical_s2]
    = builtin->builtin_logical_s2;
  lai->primitive_type_vector [f_primitive_type_logical_s8]
    = builtin->builtin_logical_s8;
  lai->primitive_type_vector [f_primitive_type_integer_s8]
    = builtin->builtin_integer_s8;    
  lai->primitive_type_vector [f_primitive_type_real]
    = builtin->builtin_real;
  lai->primitive_type_vector [f_primitive_type_real_s8]
    = builtin->builtin_real_s8;
  lai->primitive_type_vector [f_primitive_type_real_s16]
    = builtin->builtin_real_s16;
  lai->primitive_type_vector [f_primitive_type_complex_s8]
    = builtin->builtin_complex_s8;
  lai->primitive_type_vector [f_primitive_type_complex_s16]
    = builtin->builtin_complex_s16;
  lai->primitive_type_vector [f_primitive_type_void]
    = builtin->builtin_void;

  lai->bool_type_symbol = "logical";
  lai->bool_type_default = builtin->builtin_logical_s2;
}

/* Remove the modules separator :: from the default break list.  */

static char *
f_word_break_characters (void)
{
  static char *retval;

  if (!retval)
    {
      char *s;

      retval = xstrdup (default_word_break_characters ());
      s = strchr (retval, ':');
      if (s)
	{
	  char *last_char = &s[strlen (s) - 1];

	  *s = *last_char;
	  *last_char = 0;
	}
    }
  return retval;
}

/* Consider the modules separator :: as a valid symbol name character
   class.  */

static VEC (char_ptr) *
f_make_symbol_completion_list (char *text, char *word, enum type_code code)
{
  return default_make_symbol_completion_list_break_on (text, word, ":", code);
}

const struct language_defn f_language_defn =
{
  "fortran",
  language_fortran,
  range_check_on,
  case_sensitive_off,
  array_column_major,
  macro_expansion_no,
  &exp_descriptor_standard,
  f_parse,			/* parser */
  f_error,			/* parser error function */
  null_post_parser,
  f_printchar,			/* Print character constant */
  f_printstr,			/* function to print string constant */
  f_emit_char,			/* Function to print a single character */
  f_print_type,			/* Print a type using appropriate syntax */
  default_print_typedef,	/* Print a typedef using appropriate syntax */
  f_val_print,			/* Print a value using appropriate syntax */
  c_value_print,		/* FIXME */
  default_read_var_value,	/* la_read_var_value */
  NULL,				/* Language specific skip_trampoline */
  NULL,                    	/* name_of_this */
  cp_lookup_symbol_nonlocal,	/* lookup_symbol_nonlocal */
  basic_lookup_transparent_type,/* lookup_transparent_type */
  NULL,				/* Language specific symbol demangler */
  NULL,				/* Language specific
				   class_name_from_physname */
  f_op_print_tab,		/* expression operators for printing */
  0,				/* arrays are first-class (not c-style) */
  1,				/* String lower bound */
  f_word_break_characters,
  f_make_symbol_completion_list,
  f_language_arch_info,
  default_print_array_index,
  default_pass_by_reference,
  default_get_string,
  NULL,				/* la_get_symbol_name_cmp */
  iterate_over_symbols,
  LANG_MAGIC
};

static void *
build_fortran_types (struct gdbarch *gdbarch)
{
  struct builtin_f_type *builtin_f_type
    = GDBARCH_OBSTACK_ZALLOC (gdbarch, struct builtin_f_type);

  builtin_f_type->builtin_void
    = arch_type (gdbarch, TYPE_CODE_VOID, 1, "VOID");

  builtin_f_type->builtin_character
    = arch_integer_type (gdbarch, TARGET_CHAR_BIT, 0, "character");

  builtin_f_type->builtin_logical_s1
    = arch_boolean_type (gdbarch, TARGET_CHAR_BIT, 1, "logical*1");

  builtin_f_type->builtin_integer_s2
    = arch_integer_type (gdbarch, gdbarch_short_bit (gdbarch), 0,
			 "integer*2");
  builtin_f_type->builtin_integer_s8
    = arch_integer_type (gdbarch, gdbarch_long_long_bit (gdbarch), 0,
			 "integer*8");

  builtin_f_type->builtin_logical_s2
    = arch_boolean_type (gdbarch, gdbarch_short_bit (gdbarch), 1,
			 "logical*2");

  builtin_f_type->builtin_logical_s8
    = arch_boolean_type (gdbarch, gdbarch_long_long_bit (gdbarch), 1,
			 "logical*8");

  builtin_f_type->builtin_integer
    = arch_integer_type (gdbarch, gdbarch_int_bit (gdbarch), 0,
			 "integer");

  builtin_f_type->builtin_logical
    = arch_boolean_type (gdbarch, gdbarch_int_bit (gdbarch), 1,
			 "logical*4");

  builtin_f_type->builtin_real
    = arch_float_type (gdbarch, gdbarch_float_bit (gdbarch),
		       "real", NULL);
  builtin_f_type->builtin_real_s8
    = arch_float_type (gdbarch, gdbarch_double_bit (gdbarch),
		       "real*8", NULL);
  builtin_f_type->builtin_real_s16
    = arch_float_type (gdbarch, gdbarch_long_double_bit (gdbarch),
		       "real*16", NULL);

  builtin_f_type->builtin_complex_s8
    = arch_complex_type (gdbarch, "complex*8",
			 builtin_f_type->builtin_real);
  builtin_f_type->builtin_complex_s16
    = arch_complex_type (gdbarch, "complex*16",
			 builtin_f_type->builtin_real_s8);
  builtin_f_type->builtin_complex_s32
    = arch_complex_type (gdbarch, "complex*32",
			 builtin_f_type->builtin_real_s16);

  return builtin_f_type;
}

static struct gdbarch_data *f_type_data;

const struct builtin_f_type *
builtin_f_type (struct gdbarch *gdbarch)
{
  return gdbarch_data (gdbarch, f_type_data);
}

void
_initialize_f_language (void)
{
  f_type_data = gdbarch_data_register_post_init (build_fortran_types);

  add_language (&f_language_defn);
}
