/*
 * CUDA language support definitions for GDB, the GNU debugger 
 *
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2024 NVIDIA Corporation
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

#ifndef CUDA_LANG_H
#define CUDA_LANG_H

#include "expop.h"

extern struct value *eval_op_cuda_isinf (struct type *expect_type, struct expression *exp,
					 enum noside noside,
					 enum exp_opcode opcode,
					 struct value *arg1);
extern struct value *eval_op_cuda_isnan (struct type *expect_type, struct expression *exp,
					 enum noside noside,
					 enum exp_opcode opcode,
					 struct value *arg1);
extern struct value *eval_op_cuda_isfinite (struct type *expect_type, struct expression *exp,
					    enum noside noside,
					    enum exp_opcode opcode,
					    struct value *arg1);
extern struct value *eval_op_cuda_isnormal (struct type *expect_type, struct expression *exp,
					    enum noside noside,
					    enum exp_opcode opcode,
					    struct value *arg1);
extern struct value *eval_op_cuda_creal (struct type *expect_type, struct expression *exp,
					 enum noside noside,
					 enum exp_opcode opcode,
					 struct value *arg1);
extern struct value *eval_op_cuda_cimag (struct type *expect_type, struct expression *exp,
					 enum noside noside,
					 enum exp_opcode opcode,
					 struct value *arg1);
extern struct value *eval_op_cuda_abs (struct type *expect_type, struct expression *exp,
				       enum noside noside,
				       enum exp_opcode opcode,
				       struct value *arg1);
extern struct value *eval_op_cuda_ceil (struct type *expect_type, struct expression *exp,
					enum noside noside,
					enum exp_opcode opcode,
					struct value *arg1);
extern struct value *eval_op_cuda_floor (struct type *expect_type, struct expression *exp,
					 enum noside noside,
					 enum exp_opcode opcode,
					 struct value *arg1);
extern struct value *eval_op_cuda_fmod (struct type *expect_type, struct expression *exp,
					enum noside noside,
					enum exp_opcode opcode,
					struct value *arg1, struct value *arg2);

namespace expr
{

using cuda_isinf_operation = unop_operation<UNOP_ISINF, eval_op_cuda_isinf>;
using cuda_isnan_operation = unop_operation<UNOP_ISNAN, eval_op_cuda_isnan>;
using cuda_isfinite_operation = unop_operation<UNOP_ISFINITE, eval_op_cuda_isfinite>;
using cuda_isnormal_operation = unop_operation<UNOP_ISNORMAL, eval_op_cuda_isnormal>;
using cuda_creal_operation = unop_operation<UNOP_CREAL, eval_op_cuda_creal>;
using cuda_cimag_operation = unop_operation<UNOP_CIMAG, eval_op_cuda_cimag>;
using cuda_abs_operation = unop_operation<UNOP_ABS, eval_op_cuda_abs>;
using cuda_fabs_operation = unop_operation<UNOP_FABS, eval_op_cuda_abs>;
using cuda_ceil_operation = unop_operation<UNOP_CEIL, eval_op_cuda_ceil>;
using cuda_floor_operation = unop_operation<UNOP_FLOOR, eval_op_cuda_floor>;
using cuda_fmod_operation = binop_operation<BINOP_FMOD, eval_op_cuda_fmod>;

}/* namespace expr */

#endif /* CUDA_LANG_H */
