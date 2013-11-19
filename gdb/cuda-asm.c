/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2013 NVIDIA Corporation
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

#include <string.h>

#include "defs.h"
#include "frame.h"
#include "gdb_assert.h"
#include "exceptions.h"

#include "cuda-api.h"
#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-elf-image.h"
#include "cuda-modules.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-options.h"

/******************************************************************************
 *
 *                        One PC-Instruction Mapping
 *
 *****************************************************************************/

typedef struct inst_st      *inst_t;

struct inst_st {
  uint64_t      pc;      /* the PC of the disassembled instruction */
  char         *text;    /* the dissassembled instruction */
  uint32_t      size;    /* size of the instruction in bytes */
  inst_t next;           /* the next element in the list */
};

static inst_t
inst_create (uint64_t pc, const char *text, inst_t next)
{
  inst_t inst;
  int len;

  gdb_assert (text);

  len = strlen (text);
  inst = xmalloc (sizeof *inst);
  inst->text= xmalloc (len + 1);

  inst->pc   = pc;
  inst->text= strncpy (inst->text, text, len + 1);
  inst->next = next;
  inst->size = 0;

  return inst;
}

static void
inst_destroy (inst_t inst)
{
  xfree (inst->text);
  xfree (inst);
}


/******************************************************************************
 *
 *                             Disassembly Cache
 *
 *****************************************************************************/

struct disasm_cache_st {
  bool       cached;            /* have we already tried to populate this disasm_cache */
  uint64_t   entry_pc;          /* entry PC of the function being cached */
  inst_t     head;              /* head of the list of instructions */
};

disasm_cache_t
disasm_cache_create (void)
{
  disasm_cache_t disasm_cache;

  disasm_cache = xmalloc (sizeof *disasm_cache);

  disasm_cache->cached     = false;
  disasm_cache->entry_pc   = 0ULL;
  disasm_cache->head       = NULL;

  return disasm_cache;
}

void
disasm_cache_flush (disasm_cache_t disasm_cache)
{
  inst_t inst = disasm_cache->head;
  inst_t next_inst;

  while (inst)
    {
      next_inst = inst->next;
      inst_destroy (inst);
      inst = next_inst;
    }

  disasm_cache->cached     = false;
  disasm_cache->entry_pc   = 0ULL;
  disasm_cache->head       = NULL;
}

void
disasm_cache_destroy (disasm_cache_t disasm_cache)
{
  disasm_cache_flush (disasm_cache);

  xfree (disasm_cache);
}

extern char *gdb_program_name;

static int
gdb_program_dir_len (void)
{
  static int len = -1;
  int cnt;

  if (len >=0)
    return len;

  len = strlen(gdb_program_name);
  for (cnt=len; cnt > 1 && gdb_program_name[cnt-1] != '/'; cnt--);
  if (cnt > 1)
    len = cnt;

  return len;
}

static void
disasm_cache_populate_from_elf_image (disasm_cache_t disasm_cache, uint64_t pc)
{
  kernel_t kernel;
  module_t module;
  elf_image_t elf_img;
  struct objfile *objfile;
  uint32_t devId;
  uint64_t ofst = 0, entry_pc = 0;
  FILE *sass;
  char command[1024], line[1024], header[1024];
  char *filename;
  const char *function_name;
  char *function_base_name;
  char text[80], *semi_colon;
  CORE_ADDR pc1, pc2;
  inst_t prev_inst = NULL;
  bool header_found = false;

  /* early exit if already cached */
  entry_pc = get_pc_function_start (pc);
  if (disasm_cache->cached && disasm_cache->entry_pc == entry_pc)
    return;
  disasm_cache_flush (disasm_cache);
  disasm_cache->cached = true;
  disasm_cache->entry_pc = entry_pc;

  /* collect all the necessary data */
  kernel      = cuda_current_kernel ();
  module      = kernel_get_module (kernel);
  elf_img     = module_get_elf_image (module);
  objfile     = cuda_elf_image_get_objfile (elf_img);
  filename    = objfile->name;
  function_name = cuda_find_function_name_from_pc (pc, false);

  /* generate the dissassembled code using cuobjdump if available */
  snprintf (command, sizeof (command), "%.*scuobjdump --function %s --dump-sass %s",
        system ("which cuobjdump >/dev/null") == 0 ? 0: gdb_program_dir_len(),
        gdb_program_name, function_name, filename);
  sass = popen (command, "r");

  if (!sass)
    throw_error (GENERIC_ERROR, "Cannot disassemble from the ELF image.");

  /* discard the function arguments if specified */
  function_base_name = strdup (function_name);
  function_base_name = strtok (function_base_name, "(");

  /* find the wanted function in the cuobjdump output */
  snprintf (header, sizeof (header), "\t\tFunction : %s\n", function_base_name);
  while (!header_found && fgets (line, sizeof (line), sass) != NULL)
    if (strncmp (line, header, strlen (header)) == 0)
      header_found = true;
  xfree (function_base_name);

  /* return if failed to detect the function header */
  if (!header_found)
    {
      pclose (sass);
      throw_error (GENERIC_ERROR, "Cannot find the function header while disassembling.");
    }

  /* parse the sass output and insert each instruction individually */
  while (fgets (line, sizeof (line), sass) != NULL)
    {
      /* stop reading at the first white line */
      if (strcmp (line, "\n") == 0)
        break;

      /* read the instruction line and ignore others */
      memset (text, 0, sizeof text);

      if (sscanf (line, " /*%"PRIx64"*/ %79c", &ofst, text) != 2)
        continue;

      /* discard the ';' and everything afterwards */
      semi_colon = strchr (text, ';');
      if (semi_colon)
        *semi_colon = 0;

      /* add the instruction to the cache at the found offset */
      prev_inst = disasm_cache->head;
      disasm_cache->head = inst_create (entry_pc + ofst, text, disasm_cache->head);

      /* update the size of the previous instruction */
      if (prev_inst)
        prev_inst->size = entry_pc + ofst - prev_inst->pc;
    }

  /* update the instruction size of the last instruction */
  if (disasm_cache->head)
    {
      pc1 = get_pc_function_start (entry_pc + ofst);
      pc2 = get_pc_function_start (entry_pc + ofst +4);
      disasm_cache->head->size = (pc1 != 0 && pc2 != 0 && pc1 == pc2) ? 8 : 4;
    }

  /* close the sass file */
  pclose (sass);

  /* we expect to always being able to diassemble at least one instruction */
  if (cuda_options_debug_strict () && !disasm_cache->head)
    throw_error (GENERIC_ERROR, "Unable to disassemble a single device instruction.");
}

static void
disasm_cache_read_from_device_memory (disasm_cache_t disasm_cache, uint64_t pc, uint32_t *inst_size)
{
  char buf[512];
  uint32_t devId;

  /* no caching */
  disasm_cache_flush (disasm_cache);

  if (!cuda_initialized)
    return;

  buf[0] = 0;
  devId = cuda_current_device ();
  cuda_api_disassemble (devId, pc, inst_size, buf, sizeof (buf));

  disasm_cache->head = inst_create (pc, buf, disasm_cache->head);
  disasm_cache->head->size = *inst_size;

  return;
}

const char *
disasm_cache_find_instruction (disasm_cache_t disasm_cache,
                               uint64_t pc, uint32_t *inst_size)
{
  inst_t inst = NULL;

  if (!cuda_focus_is_device ())
    return NULL;

  /* compute the disassembled instruction */
  if (cuda_options_disassemble_from_elf_image ())
    disasm_cache_populate_from_elf_image (disasm_cache, pc);
  else
    disasm_cache_read_from_device_memory (disasm_cache, pc, inst_size);


  /* find the cached disassembled instruction. */
  for (inst = disasm_cache->head; inst; inst = inst->next)
    if (inst->pc == pc)
      break;

  /* return the instruction or NULL if not found */
  if (inst && inst->text && *inst->text != 0)
    {
      *inst_size = inst->size;
      return inst->text;
    }
  else
    {
      *inst_size = 4;
      return NULL;
    }
}
