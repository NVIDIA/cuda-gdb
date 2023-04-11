/*
 * Cuda language support
 *
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2023 NVIDIA Corporation
 * Written by CUDA-GDB team at NVIDIA <cudatools@nvidia.com>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#include "defs.h"
#include "gdbtypes.h"
#include "symtab.h"
#include "expression.h"
#include "parser-defs.h"
#include "language.h"
#include "varobj.h"
#include "target-float.h"
#include "gdbarch.h"
#include "cuda-lang.h"

#include <math.h>

/* A helper function for UNOP_ISINF.  */

struct value *
eval_op_cuda_isinf (struct type *expect_type, struct expression *exp,
		    enum noside noside,
		    enum exp_opcode opcode,
		    struct value *arg1)
{
  struct type *type = value_type (arg1);
  if (type->code () != TYPE_CODE_FLT)
    error (_("argument to isinf must be of type float"));
  double val
    = target_float_to_host_double (value_contents (arg1).data (),
				   type);
  type = language_bool_type (exp->language_defn, exp->gdbarch);
  return value_from_longest (type, isinf (val));
}

/* A helper function for UNOP_ISNAN.  */

struct value *
eval_op_cuda_isnan (struct type *expect_type, struct expression *exp,
		    enum noside noside,
		    enum exp_opcode opcode,
		    struct value *arg1)
{
  struct type *type = value_type (arg1);
  if (type->code () != TYPE_CODE_FLT)
    error (_("argument to isnan must be of type float"));
  double val
    = target_float_to_host_double (value_contents (arg1).data (),
				   type);
  type = language_bool_type (exp->language_defn, exp->gdbarch);
  return value_from_longest (type, isnan (val));
}

/* A helper function for UNOP_ISFINITE.  */

struct value *eval_op_cuda_isfinite (struct type *expect_type, struct expression *exp,
				     enum noside noside,
				     enum exp_opcode opcode,
				     struct value *arg1)
{
  struct type *type = value_type (arg1);
  if (type->code () != TYPE_CODE_FLT)
    error (_("argument to isfinite must be of type float"));
  double val
    = target_float_to_host_double (value_contents (arg1).data (),
				   type);
  type = language_bool_type (exp->language_defn, exp->gdbarch);
  return value_from_longest (type, isfinite (val));
}

/* A helper function for UNOP_ISNORMAL.  */

struct value *eval_op_cuda_isnormal (struct type *expect_type, struct expression *exp,
				     enum noside noside,
				     enum exp_opcode opcode,
				     struct value *arg1)
{
  struct type *type = value_type (arg1);
  if (type->code () != TYPE_CODE_FLT)
    error (_("argument to isnormal must be of type float"));
  double val
    = target_float_to_host_double (value_contents (arg1).data (),
				   type);
  type = language_bool_type (exp->language_defn, exp->gdbarch);
  return value_from_longest (type, isnormal (val));
}

/* A helper function for UNOP_CREAL.  */

struct value *eval_op_cuda_creal (struct type *expect_type, struct expression *exp,
				  enum noside noside,
				  enum exp_opcode opcode,
				  struct value *arg1)
{
  struct type *type = value_type (arg1);
  if (type->code () != TYPE_CODE_COMPLEX)
    error (_("argument to creal must be of type complex"));
  return value_real_part (arg1);
}

/* A helper function for UNOP_CIMAG.  */

struct value *eval_op_cuda_cimag (struct type *expect_type, struct expression *exp,
				  enum noside noside,
				  enum exp_opcode opcode,
				  struct value *arg1)
{
  struct type *type = value_type (arg1);
  if (type->code () != TYPE_CODE_COMPLEX)
    error (_("argument to cimag must be of type complex"));
  return value_imaginary_part (arg1);
}

/* A helper function for UNOP_ABS/UNOP_FABS.  */

struct value *eval_op_cuda_abs (struct type *expect_type, struct expression *exp,
				enum noside noside,
				enum exp_opcode opcode,
				struct value *arg1)
{
  struct type *type = value_type (arg1);
  switch (type->code ())
    {
    case TYPE_CODE_FLT:
      {
	double d
	  = fabs (target_float_to_host_double (value_contents (arg1).data (),
		  value_type (arg1)));
        return value_from_host_double (type, d);
      }
    case TYPE_CODE_INT:
      {
        LONGEST l = value_as_long (arg1);
        l = llabs (l);
        return value_from_longest (type, l);
      }
    }
  error (_("%s of type %s not supported"), 
	 type->code () == TYPE_CODE_FLT ? "fabs":"abs", 
	 TYPE_SAFE_NAME (type));
}

/* A helper function for UNOP_CEIL.  */

struct value *eval_op_cuda_ceil (struct type *expect_type, struct expression *exp,
				 enum noside noside,
				 enum exp_opcode opcode,
				 struct value *arg1)
{
  struct type *type = value_type (arg1);
  if (type->code () != TYPE_CODE_FLT)
    error (_("argument to ceil must be of type float"));
  double val
    = target_float_to_host_double (value_contents (arg1).data (),
                                   value_type (arg1));
  val = ceil (val);
  return value_from_host_double (type, val);
}

/* A helper function for UNOP_FLOOR.  */

struct value *eval_op_cuda_floor (struct type *expect_type, struct expression *exp,
				  enum noside noside,
				  enum exp_opcode opcode,
				  struct value *arg1)
{
  struct type *type = value_type (arg1);
  if (type->code () != TYPE_CODE_FLT)
    error (_("argument to floor must be of type float"));
  double val
    = target_float_to_host_double (value_contents (arg1).data (),
                                   value_type (arg1));
  val = floor (val);
  return value_from_host_double (type, val);
}

/* A helper function for BINOP_FMOD.  */

struct value *eval_op_cuda_fmod (struct type *expect_type, struct expression *exp,
				 enum noside noside,
				 enum exp_opcode opcode,
				 struct value *arg1, struct value *arg2)
{
  struct type *type = value_type (arg1);
  if (type->code () != TYPE_CODE_FLT)
    error (_("argument to mod must be of type float"));
  if (type->code () != value_type (arg2)->code ())
    error (_("non-matching types for parameters to mod"));

  double d1
    = target_float_to_host_double (value_contents (arg1).data (),
				   value_type (arg1));
  double d2
    = target_float_to_host_double (value_contents (arg2).data (),
				   value_type (arg2));
  double d3 = fmod (d1, d2);
  return value_from_host_double (type, d3);
}

