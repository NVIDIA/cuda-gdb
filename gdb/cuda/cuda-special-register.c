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

#include "defs.h"

#include "cuda-api.h"
#include "cuda-coords.h"
#include "cuda-special-register.h"
#include "cuda-tdep.h"
#include "cuda-state.h"

static void cuda_special_register_read_entry (regmap_t regmap,
                                              uint32_t entry_idx,
                                              uint32_t *buf);
static void cuda_special_register_write_entry (regmap_t regmap,
                                               uint32_t entry_idx,
                                               const uint32_t *buf);

/* Returns true if the contents of regmap have matching values.
 
   There may be more than one instance per location index.  All those
   locations contain the same value. This is the expected behavior when the
   application is compiled with -G. Otherwise, anything is possible and we
   must bail out.
  
   When there are multiple locations for the same value, we read & write all
   of them to keep things consistent.
 */
static bool
cuda_special_register_has_matching_values (regmap_t regmap)
{
  uint32_t location_index, max_location_index;
  uint32_t i, num_entries;
  uint32_t raw_value, ref_raw_value;
  bool ref_raw_value_available;

  gdb_assert (regmap);

  /* No entries always have matching values */
  num_entries = regmap_get_num_entries (regmap);
  if (num_entries == 0)
    return true;

  /* Find max location index */
  max_location_index = ~0U;
  for (i = 0; i < num_entries; ++i)
    {
      location_index = regmap_get_location_index (regmap, i);
      if (max_location_index == ~0U ||  location_index > max_location_index)
        max_location_index = location_index;
    }
  gdb_assert (max_location_index != ~0U);

  /* Check all the values with the same location index are identical */
  for (location_index = 0; location_index <= max_location_index; ++location_index)
    {
      ref_raw_value_available = false;

      for (i = 0; i < num_entries; ++i)
        {
          if (regmap_get_location_index (regmap, i) != location_index)
            continue;

          if (!ref_raw_value_available)
            {
              ref_raw_value = 0U;
              cuda_special_register_read_entry (regmap, i, &ref_raw_value);
              ref_raw_value_available = true;
              continue;
            }

          raw_value = 0U;
          cuda_special_register_read_entry (regmap, i, &raw_value);
          if (raw_value != ref_raw_value)
            return false;
        }
    }

  return true;
}

bool
cuda_special_register_p (regmap_t regmap)
{
  uint32_t num_regs;
  CUDBGRegClass reg_class;

  gdb_assert (regmap);

  /* Special register is made of at least one register */
  num_regs = regmap_get_num_entries (regmap);
  if (num_regs == 0)
    return false;

  /* Only 1 full register does not require using the special register framework */
  if (num_regs == 1)
    {
      reg_class = regmap_get_class (regmap, 0);
      if (reg_class == REG_CLASS_REG_FULL || reg_class == REG_CLASS_UREG_FULL)
	return false;
    }

  /* All the necessary chunks must be present */
  if (!regmap_is_readable (regmap))
    return false;

  /* All the instances of the same chunk must have matching values */
  if (!cuda_special_register_has_matching_values (regmap))
    return false;

  return true;
}

static void
cuda_special_register_read_entry (regmap_t regmap,
                                  uint32_t entry_idx,
                                  uint32_t *buf)
{
  uint32_t stack_addr, offset, sz, regnum;
  int sp_regnum, tmp;
  bool high;

  gdb_assert (regmap);
  gdb_assert (buf);
  gdb_assert (regmap_is_readable (regmap));

  const auto& c = cuda_current_focus::get ().physical ();

  sz = sizeof *buf;

  switch (regmap_get_class (regmap, entry_idx))
  {
    case REG_CLASS_REG_FULL:
      regnum = regmap_get_register (regmap, entry_idx);
      *buf = cuda_state::lane_get_register (c.dev (), c.sm (), c.wp (), c.ln (), regnum);
      break;

    case REG_CLASS_MEM_LOCAL:
      gdb_assert (!cuda_current_active_elf_image_uses_abi ());
      offset = regmap_get_offset (regmap, entry_idx);
      cuda_debugapi::read_local_memory (c.dev (), c.sm (), c.wp (), c.ln (), offset, buf, sz);
      break;

    case REG_CLASS_LMEM_REG_OFFSET:
      gdb_assert (cuda_current_active_elf_image_uses_abi ());
      sp_regnum = regmap_get_sp_register (regmap, entry_idx);
      offset = regmap_get_sp_offset (regmap, entry_idx);
      stack_addr = cuda_state::lane_get_register (c.dev (), c.sm (), c.wp (), c.ln (), sp_regnum);
      cuda_debugapi::read_local_memory (c.dev (), c.sm (), c.wp (), c.ln (), stack_addr + offset, buf, sz);
      break;

    case REG_CLASS_REG_HALF:
      regnum = regmap_get_half_register (regmap, entry_idx, &high);
      tmp = cuda_state::lane_get_register (c.dev (), c.sm (), c.wp (), c.ln (), regnum);
      *buf = high ? tmp >> 16 : tmp & 0xffff;
      break;

    case REG_CLASS_UREG_FULL:
      regnum = regmap_get_uregister (regmap, entry_idx);
      *buf = cuda_state::warp_get_uregister (c.dev (), c.sm (), c.wp (), regnum);
      break;

    case REG_CLASS_UREG_HALF:
      regnum = regmap_get_half_uregister (regmap, entry_idx, &high);
      tmp = cuda_state::warp_get_uregister (c.dev (), c.sm (), c.wp (), regnum);
      *buf = high ? tmp >> 16 : tmp & 0xffff;
      break;

    case REG_CLASS_REG_CC:
    case REG_CLASS_REG_PRED:
    case REG_CLASS_REG_ADDR:
    case REG_CLASS_UREG_PRED:
      error (_("CUDA Register Class 0x%x not supported yet.\n"),
             regmap_get_class (regmap, entry_idx));
      break;

    default:
      gdb_assert (0);
  }
}

static void
cuda_special_register_write_entry (regmap_t regmap,
                                   uint32_t entry_idx,
                                   const uint32_t *buf)
{
  uint32_t stack_addr, offset, sz, old_val, new_val, regnum;
  void *ptr;
  int sp_regnum;
  bool high;

  gdb_assert (regmap);
  gdb_assert (buf);

  const auto& c = cuda_current_focus::get ().physical ();

  ptr = (void*)buf;
  sz = sizeof *buf;

  switch (regmap_get_class (regmap, entry_idx))
  {
    case REG_CLASS_REG_FULL:
      regnum = regmap_get_register (regmap, entry_idx);
      cuda_state::lane_set_register (c.dev (), c.sm (), c.wp (), c.ln (), regnum, *buf);
      break;

    case REG_CLASS_MEM_LOCAL:
      gdb_assert (!cuda_current_active_elf_image_uses_abi ());
      offset = regmap_get_offset (regmap, entry_idx);
      cuda_debugapi::write_local_memory (c.dev (), c.sm (), c.wp (), c.ln (), offset, ptr, sz);
      break;

    case REG_CLASS_LMEM_REG_OFFSET:
      gdb_assert (cuda_current_active_elf_image_uses_abi ());
      sp_regnum = regmap_get_sp_register (regmap, entry_idx);
      offset = regmap_get_sp_offset (regmap, entry_idx);
      stack_addr = cuda_state::lane_get_register (c.dev (), c.sm (), c.wp (), c.ln (), sp_regnum);
      cuda_debugapi::write_local_memory (c.dev (), c.sm (), c.wp (), c.ln (), stack_addr + offset, ptr, sz);
      break;

    case REG_CLASS_REG_HALF:
      regnum = regmap_get_half_register (regmap, entry_idx, &high);
      old_val = cuda_state::lane_get_register (c.dev (), c.sm (), c.wp (), c.ln (), regnum);
      if (high)
        new_val = (*buf << 16) | (old_val & 0x0000ffff);
      else
        new_val = (old_val & 0xffff0000) | (*buf);
      cuda_state::lane_set_register (c.dev (), c.sm (), c.wp (), c.ln (), regnum, new_val);
      break;

    case REG_CLASS_REG_CC:
    case REG_CLASS_REG_PRED:
    case REG_CLASS_REG_ADDR:
      error (_("CUDA Register Class 0x%x not supported yet.\n"),
             regmap_get_class (regmap, entry_idx));
      break;

    default:
      gdb_assert (0);
  }
}

void
cuda_special_register_read (regmap_t regmap, gdb_byte *buf)
{
  uint32_t idx, i, *ptr;

  gdb_assert (regmap);
  gdb_assert (buf);

  if (!regmap_is_readable (regmap))
    error (_("Read request impossible. Insufficient debug information."));

  for (i = 0; i < regmap_get_num_entries (regmap); ++i)
    {
      idx = regmap_get_location_index (regmap, i);
      ptr = &((uint32_t*)buf)[idx];
      cuda_special_register_read_entry (regmap, i, ptr);
    }
}

void
cuda_special_register_write (regmap_t regmap, const gdb_byte *buf)
{
  uint32_t idx, i, *ptr;

  gdb_assert (regmap);
  gdb_assert (buf);

  if (!regmap_is_writable (regmap))
    error (_("Write request impossible. Insufficient debug information."));

  for (i = 0; i < regmap_get_num_entries (regmap); ++i)
    {
      idx = regmap_get_location_index (regmap, i);
      ptr = &((uint32_t*)buf)[idx];
      cuda_special_register_write_entry (regmap, i, ptr);
    }
}

void
cuda_special_register_to_value (regmap_t regmap, frame_info_ptr frame,
                                gdb_byte *to)
{
  int i = 0, regnum = 0;
  bool high = false;
  uint32_t *p = (uint32_t *)to;
  uint32_t sp_regnum = 0, offset = 0, stack_addr = 0, val32 = 0, idx = 0;

  gdb_assert (regmap);
  gdb_assert (frame);
  gdb_assert (to);
  gdb_assert (regmap_is_readable (regmap));

  const auto& c = cuda_current_focus::get ().physical ();

  for (i = 0; i < regmap_get_num_entries (regmap); ++i)
    {
      idx = regmap_get_location_index (regmap, i);

      switch (regmap_get_class (regmap, i))
        {
          case REG_CLASS_REG_FULL:
            regnum = regmap_get_register (regmap, i);
            get_frame_register (frame, regnum, (gdb_byte*)&p[idx]);
            break;

          case REG_CLASS_MEM_LOCAL:
            offset = regmap_get_offset (regmap, i);
            cuda_debugapi::read_local_memory (c.dev (), c.sm (), c.wp (), c.ln (), offset,
                                        (void*)&p[idx], sizeof (p[idx]));
            break;

          case REG_CLASS_LMEM_REG_OFFSET:
            sp_regnum = regmap_get_sp_register (regmap, i);
            offset = regmap_get_sp_offset (regmap, i);
            get_frame_register (frame, sp_regnum, (gdb_byte*)&stack_addr);
            cuda_debugapi::read_local_memory (c.dev (), c.sm (), c.wp (), c.ln (), stack_addr + offset,
                                        (void*)&p[idx], sizeof (p[idx]));
            break;

          case REG_CLASS_REG_HALF:
            regnum = regmap_get_half_register (regmap, i, &high);
            get_frame_register (frame, regnum, (gdb_byte*)&val32);
            p[idx] = high ? val32 >> 16 : val32 & 0xffff;
            break;

          case REG_CLASS_UREG_FULL:
            regnum = regmap_get_uregister (regmap, i);
            get_frame_register (frame, regnum, (gdb_byte*)&p[idx]);
            break;

          case REG_CLASS_UREG_HALF:
            regnum = regmap_get_half_uregister (regmap, i, &high);
            get_frame_register (frame, regnum, (gdb_byte*)&val32);
            p[idx] = high ? val32 >> 16 : val32 & 0xffff;
            break;

          case REG_CLASS_REG_CC:
          case REG_CLASS_REG_PRED:
          case REG_CLASS_REG_ADDR:
          case REG_CLASS_UREG_PRED:
            error (_("CUDA Register Class 0x%x not supported yet.\n"),
                   regmap_get_class (regmap, i));
            break;

          default:
            gdb_assert (0);
        }
    }
}

void
cuda_value_to_special_register (regmap_t regmap, frame_info_ptr frame,
                                const gdb_byte *from)
{
  int i = 0, regnum = 0;
  bool high = false;
  uint32_t *p = (uint32_t *)from;
  uint32_t sp_regnum = 0, offset = 0, stack_addr = 0, val32 = 0, idx = 0;

  gdb_assert (regmap);
  gdb_assert (frame);
  gdb_assert (from);

  if (!regmap_is_writable (regmap))
    error (_("Write request impossible. Insufficient debug information."));

  const auto& c = cuda_current_focus::get ().physical ();

  for (i = 0; i < regmap_get_num_entries (regmap); ++i)
    {
      idx = regmap_get_location_index (regmap, i);

      switch (regmap_get_class (regmap, i))
        {
          case REG_CLASS_REG_FULL:
            regnum = regmap_get_register (regmap, i);
            put_frame_register (frame, regnum, (gdb_byte*)&p[idx]);
            break;

          case REG_CLASS_MEM_LOCAL:
            offset = regmap_get_offset (regmap, i);
            cuda_debugapi::write_local_memory (c.dev (), c.sm (), c.wp (), c.ln (), offset,
                                        (void*)&p[idx], sizeof (p[idx]));
            break;

          case REG_CLASS_LMEM_REG_OFFSET:
            sp_regnum = regmap_get_sp_register (regmap, i);
            offset = regmap_get_sp_offset (regmap, i);
            get_frame_register (frame, sp_regnum, (gdb_byte*)&stack_addr);
            cuda_debugapi::write_local_memory (c.dev (), c.sm (), c.wp (), c.ln (), stack_addr + offset,
                                         (void*)&p[idx], sizeof (p[idx]));
            break;

          case REG_CLASS_REG_HALF:
            regnum = regmap_get_half_register (regmap, i, &high);
            get_frame_register (frame, regnum, (gdb_byte*)&val32);
            val32 = high ? (val32 & 0xffff)     | (p[idx] << 16)
                         : (val32 & 0xffff0000) | (p[idx] & 0xffff);
            put_frame_register (frame, regnum, (gdb_byte*)&val32);
            break;

          case REG_CLASS_UREG_FULL:
            regnum = regmap_get_uregister (regmap, i);
            put_frame_register (frame, regnum, (gdb_byte*)&p[idx]);
            break;

          case REG_CLASS_UREG_HALF:
            regnum = regmap_get_half_uregister (regmap, i, &high);
            get_frame_register (frame, regnum, (gdb_byte*)&val32);
            val32 = high ? (val32 & 0xffff)     | (p[idx] << 16)
                         : (val32 & 0xffff0000) | (p[idx] & 0xffff);
            put_frame_register (frame, regnum, (gdb_byte*)&val32);
            break;

          case REG_CLASS_REG_CC:
          case REG_CLASS_REG_PRED:
          case REG_CLASS_REG_ADDR:
	  case REG_CLASS_UREG_PRED:
            error (_("CUDA Register Class 0x%x not supported yet.\n"),
                   regmap_get_class (regmap, i));
            break;

          default:
            gdb_assert (0);
        }
    }
}

void
cuda_special_register_name (regmap_t regmap, char *buf, const int size)
{
  uint32_t sp_regnum, offset, regnum;
  int d, i;
  bool high;

  gdb_assert (regmap);
  gdb_assert (buf);

  buf[0] = '\0';

  d = 0;
  for (i = 0; i < regmap_get_num_entries (regmap); ++i)
    {
      if (i > 0)
        d += snprintf (buf + d, size - 1 - d, "/$");

      switch (regmap_get_class (regmap, i))
        {
          case REG_CLASS_REG_FULL:
            regnum = regmap_get_register (regmap, i);
            d += snprintf (buf + d, size - 1 - d, "R%d", regnum);
            break;

          case REG_CLASS_MEM_LOCAL:
            offset = regmap_get_offset (regmap, i);
            d += snprintf (buf + d, size - 1 - d, "(spilled @ 0x%x)", offset);
            break;

          case REG_CLASS_LMEM_REG_OFFSET:
            sp_regnum = regmap_get_sp_register (regmap, i);
            offset = regmap_get_sp_offset (regmap, i);
            d += snprintf (buf + d, size - 1 - d, "(spilled @ [R%d]+0x%x)",
                           sp_regnum,  offset);
            break;

          case REG_CLASS_REG_HALF:
            regnum = regmap_get_half_register (regmap, i, &high);
            d += snprintf (buf + d, size - 1 - d, "R%d.%s", regnum, high ? "hi" : "lo");
            break;

          case REG_CLASS_UREG_FULL:
            regnum = regmap_get_uregister (regmap, i);
            d += snprintf (buf + d, size - 1 - d, "UR%d", regnum);
            break;

          case REG_CLASS_UREG_HALF:
            regnum = regmap_get_half_uregister (regmap, i, &high);
            d += snprintf (buf + d, size - 1 - d, "UR%d.%s", regnum, high ? "hi" : "lo");
            break;

          case REG_CLASS_REG_CC:
          case REG_CLASS_REG_PRED:
          case REG_CLASS_REG_ADDR:
	  case REG_CLASS_UREG_PRED:
            error (_("CUDA Register Class 0x%x not supported yet.\n"),
                   regmap_get_class (regmap, i));
            break;

          default:
            gdb_assert (0);
        }
    }
}
