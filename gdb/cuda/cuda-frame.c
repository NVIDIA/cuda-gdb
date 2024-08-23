/*
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

#include "defs.h"

#include "arch-utils.h"
#include "block.h"
#include "cuda-frame.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "dummy-frame.h"
#include "dwarf2/frame.h"
#include "frame-base.h"
#include "regcache.h"
#include "user-regs.h"
#include "value.h"

static CORE_ADDR cuda_frame_prev_pc (frame_info_ptr next_frame);

struct cuda_frame_cache
{
  CORE_ADDR base;
  CORE_ADDR pc;
};

/* Returns true if the frame corresponds to a CUDA device function. */
bool
cuda_frame_p (frame_info_ptr next_frame)
{
  if (cuda_current_focus::isDevice ())
    return true;
  else
    return false;
}

/* Returns true if this frame, the previous frame of next_frame, is inlined.

   The implementation is slightly convoluted because this is an unwind routine.
   Therefore, we cannot assume anything about this frame. We can only look
   down, to the next frame and below (and their corresponding blocks).
 */
static bool
cuda_frame_inlined_p (frame_info_ptr next_frame)
{
  int num_inlined_frames = 0;
  CORE_ADDR pc = 0;
  const struct block *block = NULL;
  const struct block *this_frame_block = NULL;
  const struct block *outermost_block = NULL;
  frame_info_ptr frame = NULL;
  enum frame_type frame_type = SENTINEL_FRAME;

  /* Count the number of inlined frames between this frame and the next
     non-inlined frame. */
  num_inlined_frames = 0;
  for (frame = next_frame; frame; frame = get_next_frame (frame))
    {
      frame_type = get_frame_type (frame);
      if (frame_type != INLINE_FRAME)
        break;
      ++num_inlined_frames;
    }

  /* Find the function block for this frame by skipping the function blocks
     corresponding to inlined functions. Also find the outermost block in that
     block chain. */
  pc = cuda_frame_prev_pc (next_frame);
  for (block = block_for_pc (pc); block; block = block->superblock ())
    {
      if (!block->function ())
        continue;

      --num_inlined_frames;
      if (num_inlined_frames == -1)
        this_frame_block = block;

      outermost_block = block;
    }

  /* If there is no debug info, bail out. */
  if (!outermost_block)
    return false;

  /* This frame is inlined if it is not the caller function frame in this block
     chain. */
  return (this_frame_block != outermost_block);
}

static bool
cuda_abi_frame_outermost_p (frame_info_ptr next_frame)
{
  int call_stack_depth = 0;
  int syscall_call_depth = 0;
  int normal_frame_depth = 0;
  frame_info_ptr frame = NULL;
  enum frame_type frame_type = SENTINEL_FRAME;

  /* check for valid frame */
  if (!next_frame || (frame_relative_level (next_frame) < 0))
    return false;
  /* An inlined frame cannot be the outermost frame. */
  if (cuda_frame_inlined_p (next_frame))
    return false;

  /* Find the call stack depth, i.e. the maximum number of normal frames,
     syscall frames included, we are expected to find in the call stack. */
  const auto &c = cuda_current_focus::get ().physical ();
  call_stack_depth
      = cuda_state::lane_get_call_depth (c.dev (), c.sm (), c.wp (), c.ln ());

  /* To compute the normal frame depth of this frame, count the number of
     next normal frames. */
  normal_frame_depth = 0;
  for (frame = next_frame; frame; frame = get_next_frame (frame))
    {
      frame_type = get_frame_type (frame);
      switch (frame_type)
        {
        case INLINE_FRAME:
          continue;
        case NORMAL_FRAME:
          ++normal_frame_depth;
          continue;
        default:
          break;
        }
    }

  /* The syscall frames must be taken into account, even if they are hidden. */
  if (cuda_options_hide_internal_frames ())
    {
      syscall_call_depth = cuda_state::lane_get_syscall_call_depth (
          c.dev (), c.sm (), c.wp (), c.ln ());
      normal_frame_depth += syscall_call_depth;
    }
  if (normal_frame_depth)
    --normal_frame_depth;

  /* A normal frame is the outermost frame if it is the Nth normal frame in the
     stack, where N is the call stack depth. */
  return normal_frame_depth == call_stack_depth;
}

static bool
cuda_noabi_frame_outermost_p (frame_info_ptr next_frame)
{
  if (cuda_frame_inlined_p (next_frame))
    return false;

  return true;
}

/* Returns true if the current frame (next_frame->prev) is the
   outermost device frame. */
bool
cuda_frame_outermost_p (frame_info_ptr next_frame)
{
  if (!cuda_frame_p (next_frame))
    return false;

  if (cuda_current_active_elf_image_uses_abi ())
    return cuda_abi_frame_outermost_p (next_frame);
  else
    return cuda_noabi_frame_outermost_p (next_frame);
}

static CORE_ADDR
cuda_abi_frame_cache_base (frame_info_ptr this_frame)
{
  struct gdbarch *gdbarch = get_frame_arch (this_frame);
  int sp_regnum = cuda_abi_sp_regnum (gdbarch);
  gdb_byte buf[8];

  memset (buf, 0, sizeof (buf));
  frame_unwind_register (this_frame, sp_regnum, buf);
  return extract_unsigned_integer (buf, sizeof buf, BFD_ENDIAN_LITTLE);
}

static CORE_ADDR
cuda_noabi_frame_cache_base (frame_info_ptr this_frame)
{
  return 0;
}

static struct cuda_frame_cache *
cuda_frame_cache (frame_info_ptr this_frame, void **this_cache)
{
  struct cuda_frame_cache *cache;

  gdb_assert (cuda_frame_p (this_frame));

  if (*this_cache)
    return (struct cuda_frame_cache *)*this_cache;

  cache = FRAME_OBSTACK_ZALLOC (struct cuda_frame_cache);
  *this_cache = cache;

  cache->pc = get_frame_func (this_frame);

  if (cuda_current_active_elf_image_uses_abi ())
    cache->base = cuda_abi_frame_cache_base (this_frame);
  else
    cache->base = cuda_noabi_frame_cache_base (this_frame);

  return cache;
}

/* cuda_frame_base_address is not ABI-dependent, since it only
   queries the base field from the frame cache.  It is the frame
   cache itself which is constructed uniquely for ABI/non-ABI
   compilations. */
static CORE_ADDR
cuda_frame_base_address (frame_info_ptr this_frame, void **this_cache)
{
  struct cuda_frame_cache *cache;
  CORE_ADDR base;

  gdb_assert (cuda_frame_p (this_frame));

  if (*this_cache)
    return ((struct cuda_frame_cache *)(*this_cache))->base;

  cache = cuda_frame_cache (this_frame, this_cache);
  base = cache->base;
  return base;
}

static struct frame_id
cuda_abi_frame_id_build (frame_info_ptr this_frame, void **this_cache,
                         struct frame_id *this_id)
{
  struct cuda_frame_cache *cache;
  int call_depth = 0;
  int syscall_call_depth = 0;
  int this_level;

  cache = cuda_frame_cache (this_frame, this_cache);
  this_level = frame_relative_level (this_frame);

  const auto &c = cuda_current_focus::get ().physical ();
  call_depth
      = cuda_state::lane_get_call_depth (c.dev (), c.sm (), c.wp (), c.ln ());
  syscall_call_depth = cuda_state::lane_get_syscall_call_depth (
      c.dev (), c.sm (), c.wp (), c.ln ());

  /* With the ABI, we can have multiple device frames. */
  if (this_level < call_depth)
    {
      /* When we have syscall frames, we will build them as special frames,
         as the API will always return only the PC to the first non syscall
         frame. Thus all frames less the syscall_call_depth will be identical
         to the frame at the syscall call depth */
      if ((this_level < syscall_call_depth)
          && !cuda_options_hide_internal_frames ())
        return frame_id_build_special (cache->base, cache->pc,
                                       syscall_call_depth + this_level);
      else
        return frame_id_build (cache->base, cache->pc);
    }
  else
    return frame_id_build_special (cache->base, cache->pc, 1);
}

static struct frame_id
cuda_noabi_frame_id_build (frame_info_ptr this_frame, void **this_cache,
                           struct frame_id *this_id)
{
  struct cuda_frame_cache *cache;
  struct frame_id frame_id;

  cache = cuda_frame_cache (this_frame, this_cache);
  frame_id = frame_id_build (cache->base, cache->pc);

  return frame_id;
}

static enum unwind_stop_reason
cuda_frame_unwind_stop_reason (frame_info_ptr this_frame,
                               void **this_cache)
{
  /*NS: TODO*/
  return UNWIND_NO_REASON;
}

static void
cuda_frame_this_id (frame_info_ptr this_frame, void **this_cache,
                    struct frame_id *this_id)
{
  int this_level = frame_relative_level (this_frame);

  gdb_assert (cuda_frame_p (this_frame));

  if (cuda_current_active_elf_image_uses_abi ())
    *this_id = cuda_abi_frame_id_build (this_frame, this_cache, this_id);
  else
    *this_id = cuda_noabi_frame_id_build (this_frame, this_cache, this_id);

  frame_debug_printf ("{ cuda_frame_this_id (frame=%d) -> this_id=%s }",
                      this_level, this_id->to_string ().c_str ());
}

static CORE_ADDR
cuda_abi_frame_prev_pc (frame_info_ptr next_frame)
{
  uint64_t pc;
  int syscall_call_depth = 0;
  int call_depth = 0;
  int level = 0;
  int num_normal_frames = 0;
  frame_info_ptr frame = NULL;
  enum frame_type frame_type = SENTINEL_FRAME;

  const auto &c = cuda_current_focus::get ().physical ();
  call_depth
      = cuda_state::lane_get_call_depth (c.dev (), c.sm (), c.wp (), c.ln ());

  /* compute the frame level, excluded the inlined frames */
  num_normal_frames = 0;
  for (frame = next_frame; frame; frame = get_next_frame (frame))
    {
      frame_type = get_frame_type (frame);
      switch (frame_type)
        {
        case INLINE_FRAME:
          continue;
        case NORMAL_FRAME:
          ++num_normal_frames;
          continue;
        default:
          break;
        }
    }
  level += num_normal_frames;

  /* remember to skip the syscall frames if required */
  if (cuda_options_hide_internal_frames ())
    {
      syscall_call_depth = cuda_state::lane_get_syscall_call_depth (
          c.dev (), c.sm (), c.wp (), c.ln ());
      level += syscall_call_depth;
    }

  if (level == 0)
    pc = cuda_state::lane_get_pc (c.dev (), c.sm (), c.wp (), c.ln ());
  else if (level <= call_depth)
    pc = cuda_state::lane_get_return_address (
        c.dev (), c.sm (), c.wp (), c.ln (), level - 1);
  else
    pc = 0;

  return (CORE_ADDR)pc;
}

static CORE_ADDR
cuda_noabi_frame_prev_pc (frame_info_ptr next_frame)
{
  const auto &c = cuda_current_focus::get ().physical ();
  uint64_t pc
      = cuda_state::lane_get_pc (c.dev (), c.sm (), c.wp (), c.ln ());

  return (CORE_ADDR)pc;
}

CORE_ADDR
cuda_frame_prev_pc (frame_info_ptr next_frame)
{
  gdb_assert (cuda_frame_p (next_frame));

  if (cuda_current_active_elf_image_uses_abi ())
    return cuda_abi_frame_prev_pc (next_frame);
  else
    return cuda_noabi_frame_prev_pc (next_frame);
}

/* When unwinding registers stored on CUDA ABI frames, we use
   this function to hook in the dwarf2 frame unwind routines. */
static struct value *
cuda_abi_hook_dwarf2_frame_prev_register (frame_info_ptr next_frame,
                                          void **this_cache, int regnum)
{
  struct gdbarch *gdbarch = get_frame_arch (next_frame);
  int sp_regnum = cuda_abi_sp_regnum (gdbarch);
  void *dwarfcache = NULL;
  struct frame_base *dwarf2_base_finder = NULL;
  struct value *value = NULL;
  CORE_ADDR sp = 0;
  int dwarf2 = 0;

  /* If we have a dwarf2 base finder, then we will use it to know what the
     value of the stack pointer is.  See dwarf2-frame.c */
  if (regnum == sp_regnum)
    {
      dwarf2_base_finder
          = (struct frame_base *)dwarf2_frame_base_sniffer (next_frame);
      if (dwarf2_base_finder)
        {
          sp = dwarf2_base_finder->this_base (next_frame, &dwarfcache);
          value = frame_unwind_got_address (next_frame, regnum, sp);
        }
    }

  /* If we have a dwarf2 unwinder, then we will use it to know where to look
     for the value of all CUDA registers.  See dwarf2-frame.c */
  dwarf2 = dwarf2_frame_unwind.sniffer (&dwarf2_frame_unwind, next_frame,
                                        (void **)&dwarfcache);
  if (!value && dwarf2)
    value = dwarf2_frame_unwind.prev_register (next_frame,
                                               (void **)&dwarfcache, regnum);

  return value;
}

/* With the ABI, prev_register needs assistance from the dwarf2 frame
   unwinder to decode the storage location of a given regnum for a
   given frame.  Non-debug compilations will not have this assistance,
   so we check for a proper dwarf2 unwinder to make sure.  The PC can
   be decoded without dwarf2 assistance thanks to the device's runtime
   stack. */
static struct value *
cuda_abi_frame_prev_register (frame_info_ptr next_frame, void **this_cache,
                              int regnum)
{
  struct gdbarch *gdbarch = get_frame_arch (next_frame);
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);
  CORE_ADDR pc = 0;
  struct value *value = NULL;
  frame_info_ptr frame = NULL;
  enum frame_type frame_type = SENTINEL_FRAME;
  bool read_register_from_device = false;

  /* read directly from the device when dealing with the innermost inlined
     frames or the innermost normal frame. */
  read_register_from_device = true;
  for (frame = next_frame; frame; frame = get_next_frame (frame))
    {
      frame_type = get_frame_type (frame);
      if (frame_type != INLINE_FRAME)
        {
          read_register_from_device = false;
          break;
        }
    }

  if (regnum == pc_regnum)
    {
      pc = cuda_frame_prev_pc (next_frame);
      value = frame_unwind_got_address (next_frame, regnum, pc);
    }
  else if (read_register_from_device)
    value = frame_unwind_got_register (next_frame, regnum, regnum);
  /* Stack frame unwinding using .debug_frame information is buggy in the case
   * of syscall user stubs, fallback on directly reading from the inner frame
   * in this case */
  else if (!get_frame_id_p (next_frame)
           || !get_frame_id (next_frame).special_addr_p
           || get_frame_id (next_frame).special_addr != 1)
    value = cuda_abi_hook_dwarf2_frame_prev_register (next_frame, this_cache,
                                                      regnum);

  /* Try to fetch the register value from the inner frame.  */
  if (!value)
    value = get_frame_register_value (next_frame, regnum);

  /* Last resort: if no value found, use the register for the innermost frame. */
  if (!value)
    value = frame_unwind_got_register (get_next_frame (frame), regnum, regnum);

  return value;
}

/* Without the ABI, prev_register only needs to read current values from
   the register file (with the exception of PC, which requires special
   handling for inserted dummy frames) */
static struct value *
cuda_noabi_frame_prev_register (frame_info_ptr next_frame,
                                void **this_cache, int regnum)
{
  struct gdbarch *gdbarch = get_frame_arch (next_frame);
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);
  CORE_ADDR pc = 0;

  if (regnum == pc_regnum)
    {
      pc = cuda_frame_prev_pc (next_frame);
      return frame_unwind_got_address (next_frame, regnum, pc);
    }
  else
    return frame_unwind_got_register (next_frame, regnum, regnum);
}

static struct value *
cuda_frame_prev_register (frame_info_ptr next_frame, void **this_cache,
                          int regnum)
{
  int next_level = frame_relative_level (next_frame);
  struct gdbarch *gdbarch = get_frame_arch (next_frame);
  struct value *value = NULL;

  if (cuda_current_active_elf_image_uses_abi ())
    value = cuda_abi_frame_prev_register (next_frame, this_cache, regnum);
  else
    value = cuda_noabi_frame_prev_register (next_frame, this_cache, regnum);

  if (frame_debug)
    {
      gdb_printf (gdb_stdlog,
                          "{ cuda_frame_prev_register "
                          "(frame=%d,regnum=%d(%s),...) ",
                          next_level, regnum,
                          user_reg_map_regnum_to_name (gdbarch, regnum));
      gdb_printf (gdb_stdlog, "->");
      gdb_printf (gdb_stdlog, " *bufferp=");
      if (value == NULL)
        gdb_printf (gdb_stdlog, "<NULL>");
      else
        {
          int i;
          gdb::array_view<const gdb_byte> buf = value_contents (value);
          gdb_printf (gdb_stdlog, "[");
          for (i = 0; i < register_size (gdbarch, regnum); i++)
            gdb_printf (gdb_stdlog, "%02x", buf[i]);
          gdb_printf (gdb_stdlog, "]");
        }
      gdb_printf (gdb_stdlog, " }\n");
    }

  return value;
}

/* cuda_frame_sniffer_check is not ABI-dependent at the moment.
   Ideally, there will be 2 separate sniffers, and we can remove
   switching internally within each of the frame functions. */
static int
cuda_frame_sniffer_check (const struct frame_unwind *self,
                          frame_info_ptr next_frame,
                          void **this_prologue_cache)
{
  bool is_cuda_frame;
  int next_level = frame_relative_level (next_frame);

  is_cuda_frame = cuda_frame_p (next_frame) && self == &cuda_frame_unwind;

  if (frame_debug)
    gdb_printf (gdb_stdlog,
                        "{ cuda_frame_sniffer_check "
                        "(frame = %d) -> %d }\n",
                        next_level, is_cuda_frame);
  return is_cuda_frame;
}

const struct frame_base *
cuda_frame_base_sniffer (frame_info_ptr next_frame)
{
  const struct frame_base *base = NULL;
  int next_level = frame_relative_level (next_frame);

  if (cuda_frame_p (next_frame))
    base = &cuda_frame_base;

  if (frame_debug)
    gdb_printf (gdb_stdlog,
                        "{ cuda_frame_base_sniffer "
                        "(frame=%d) -> %d }\n",
                        next_level, !!base);

  return base;
}

const struct frame_unwind *
cuda_frame_sniffer (frame_info_ptr next_frame)
{
  const struct frame_unwind *unwind = NULL;
  int next_level = frame_relative_level (next_frame);

  if (cuda_frame_p (next_frame))
    unwind = &cuda_frame_unwind;

  if (frame_debug)
    gdb_printf (gdb_stdlog,
                        "{ cuda_frame_sniffer (frame=%d) -> %d }\n",
                        next_level, !!unwind);

  return unwind;
}

CORE_ADDR
cuda_unwind_pc (struct gdbarch *gdbarch, frame_info_ptr next_frame)
{
  CORE_ADDR pc;

  pc = cuda_frame_prev_pc (next_frame);

  frame_debug_printf ("{ cuda_unwind_pc (next_frame=n/a) -> %s }",
                      hex_string (pc));

  return pc;
}

const struct frame_unwind cuda_frame_unwind = {
  "cuda prologue",
  NORMAL_FRAME,
  cuda_frame_unwind_stop_reason,
  cuda_frame_this_id,
  cuda_frame_prev_register,
  NULL,
  cuda_frame_sniffer_check,
  NULL,
  NULL,
};

const struct frame_base cuda_frame_base
    = { &cuda_frame_unwind, cuda_frame_base_address, cuda_frame_base_address,
        cuda_frame_base_address };
