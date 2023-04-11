/*
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

#ifndef _CUDA_REGMAP_H
#define _CUDA_REGMAP_H 1

#include "defs.h"
#include "cudadebugger.h"
#include "symtab.h"
#include "objfiles.h"
#include "cuda-defs.h"

regmap_t      regmap_get_search_result  (void);
const char *  regmap_get_func_name      (regmap_t regmap);
const char *  regmap_get_reg_name       (regmap_t regmap);
uint64_t      regmap_get_addr           (regmap_t regmap);
uint32_t      regmap_get_num_entries    (regmap_t regmap);
uint32_t      regmap_get_location_index (regmap_t regmap, uint32_t idx);
CUDBGRegClass regmap_get_class          (regmap_t regmap, uint32_t idx);
uint32_t      regmap_get_register       (regmap_t regmap, uint32_t idx);
uint32_t      regmap_get_sp_register    (regmap_t regmap, uint32_t idx);
uint32_t      regmap_get_sp_offset      (regmap_t regmap, uint32_t idx);
uint32_t      regmap_get_offset         (regmap_t regmap, uint32_t idx);
uint32_t      regmap_get_half_register  (regmap_t regmap, uint32_t idx,
                                         bool *in_higher_16_bits);
uint32_t      regmap_get_uregister      (regmap_t regmap, uint32_t idx);
uint32_t      regmap_get_half_uregister (regmap_t regmap, uint32_t idx,
					 bool *in_higher_16_bits);

bool regmap_is_readable     (regmap_t regmap);
bool regmap_is_writable     (regmap_t regmap);
bool regmap_is_extrapolated (regmap_t regmap);

regmap_t regmap_table_search (struct objfile *objfile, const char *func_name,
                              const char *reg_name, uint64_t addr);
void     regmap_table_print  (struct objfile *objfile);

int      cuda_decode_physical_register (uint64_t reg, int32_t *result);

#endif
