/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2014 NVIDIA Corporation
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

#include "stdlib.h"
#include "stdio.h"
#include "config.h"
#include "bfd.h"
#include "elf-bfd.h"
#include "libbfd.h"
#include "defs.h"
#include "block.h"
#include "value.h"

#include "cuda-defs.h"
#include "cuda-coords.h"
#include "cuda-api.h"
#include "cuda-textures.h"
#include "cuda-options.h"
#include "cuda-kernel.h"
#include "cuda-state.h"


#define EIFMT_SVAL         4
#define EIATTR_IMAGE_SLOT  2
#define EF_CUDA_SM_MASK    0xff
#define EF_CUDA_SM30       30
#define CUDA_ELF_NV_INFO_PREFIX ".nv.info."

/*Mapping from kernel (mangled) name to texture id for each texture symbol */
typedef struct cuda_tex_map_t {
  const char *kernel_name;
  uint32_t texid;
  uint32_t symindex;                //global symtab index of the texture symbol
  bool use_symindex;
  struct cuda_tex_map_t *next;      //pointer to next map
  struct cuda_tex_map_t *next_maps; //pointer to next set of maps (for cleanup)
} cuda_tex_map_t;

/*Texture memory access always needs two components for address.
   The address of the texture, and the coordinates within the texture. 
   Whiling reading texture memory, should ALWAYS use texture_address to 
   hold the two components. */
struct {
  enum {from_value_ind = 0, from_value_ptradd} last_called;
  uint32_t address;
  uint32_t dim;
  uint32_t coords[TEXTURE_DIM_MAX];
  bool     is_bindless;
} texture_address;

typedef struct {
  uint8_t format;
  uint8_t attr;
  uint16_t size;
} cuda_nv_info_t;

typedef struct {
  uint32_t index;
  uint32_t slot;
} cuda_image_slot_t;

typedef union {
  char *c;
  cuda_nv_info_t *info;
  cuda_image_slot_t *slot;
} cuda_magic_cast_t;

/* a list of tex map created */
cuda_tex_map_t *cuda_tex_maps = NULL;

static void
cuda_textures_trace (char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_TEXTURES, fmt, ap);
  va_end (ap);
}

/* Create a (mangled) kernel name to texture id mapping
   for a given symbol table index */
static cuda_tex_map_t*
cuda_create_tex_map (struct objfile *objfile,
                     uint32_t symindex)
{
  Elf_Internal_Shdr *shdr = NULL;
  Elf_Internal_Ehdr *elf_header = objfile->obfd->tdata.elf_obj_data->elf_header;
  asection *section = NULL;
  uint32_t i;
  uint32_t prefixlen = strlen (CUDA_ELF_NV_INFO_PREFIX);
  void *buf = NULL;
  cuda_magic_cast_t trav, range;
  cuda_nv_info_t *info;
  cuda_image_slot_t *slot;
  cuda_tex_map_t *rVal = NULL, *prev_map = NULL, *new_map;
  file_ptr prev_location, result;

  gdb_assert (objfile);

  if ((EF_CUDA_SM_MASK & elf_header->e_flags) >= EF_CUDA_SM30)
    {
       /* For SM_30 and higher, we only need to store the symindex. */
       new_map = (cuda_tex_map_t*)malloc (sizeof (*new_map));
       new_map->kernel_name = 0;
       new_map->texid = 0;
       new_map->symindex = symindex;
       new_map->use_symindex = true;
       new_map->next = NULL;
       new_map->next_maps = NULL;
       rVal = new_map;
       goto done;
    }

  /* Find all the ".nv.info.*" sections */
  for (i = 0; i < objfile->obfd->tdata.elf_obj_data->num_elf_sections; ++i)
    {
      shdr = objfile->obfd->tdata.elf_obj_data->elf_sect_ptr[i];
      section = shdr->bfd_section;
      if (!section || strncmp (section->name, CUDA_ELF_NV_INFO_PREFIX, prefixlen))
        continue;

      /* buffer the matched section in memory */
      buf = realloc (buf, shdr->sh_size);
      prev_location = objfile->obfd->iovec->btell (objfile->obfd);
      if (prev_location == -1)
        continue;
      result = (file_ptr)objfile->obfd->iovec->bseek (objfile->obfd, (file_ptr)shdr->sh_offset, 0);
      if (result == -1)
        continue;
      result = objfile->obfd->iovec->bread (objfile->obfd, buf, (file_ptr)shdr->sh_size);
      if (result == -1 || result < (file_ptr)shdr->sh_size)
        continue;

      /* traverse through the section and find all the mapping */
      trav.c = (char*)buf;
      range.c = (char*)buf + shdr->sh_size;
      while (trav.c < range.c)
        {
          info = trav.info;
          ++trav.info;
          if (info->format == EIFMT_SVAL)
            {
              slot = trav.slot;
              trav.c += info->size;
              if (info->attr == EIATTR_IMAGE_SLOT && slot->index == symindex)
                {
                  new_map = (cuda_tex_map_t*)malloc (sizeof (*new_map));
                  new_map->kernel_name = section->name + prefixlen;
                  new_map->texid = slot->slot;
                  new_map->symindex = symindex;
                  new_map->use_symindex = false;
                  new_map->next = NULL;
                  new_map->next_maps = NULL;
                  if (prev_map)
                    {
                      prev_map->next = new_map;
                      prev_map = new_map;
                    }
                  else
                    {
                      rVal = new_map;
                      prev_map = new_map;
                    }
                }
            }
        }
      /* rewind the iostream to where it was */
      objfile->obfd->iovec->bseek (objfile->obfd, prev_location, 0);
    }

done:
  if (buf)
    free (buf);

  if (!rVal)
    return NULL;

  /* maintain a list of all tex maps created */
  rVal->next_maps = cuda_tex_maps;
  cuda_tex_maps = rVal;

  return rVal;
}

void
cuda_cleanup_tex_maps (void)
{
  cuda_tex_map_t *prev_map = NULL;
  cuda_tex_map_t *next_set_of_maps = NULL;
  while (cuda_tex_maps)
    {
      next_set_of_maps = cuda_tex_maps->next_maps;

      while (cuda_tex_maps)
        {
          prev_map = cuda_tex_maps;
          cuda_tex_maps = cuda_tex_maps->next;
          free (prev_map);
        }

      cuda_tex_maps = next_set_of_maps;
    }
  /* cuda_tex_maps should be NULL from here on */
}

static bool
cuda_find_tex_id (cuda_tex_map_t *mapping,
                  uint32_t *texid)
{
  kernel_t kernel = NULL;
  struct symbol *symbol = NULL;
  struct minimal_symbol *msymbol;
  const char *name = NULL;
  uint64_t pc;

  gdb_assert (texid);

  if (mapping && mapping->use_symindex)
    {
      *texid = mapping->symindex;
      return true;
    }

  /* Get the current kernel's name. We want the mangled name,
     cuda_current_kernel_name gives demangled. */
  kernel = cuda_current_kernel ();
  pc = kernel_get_virt_code_base (kernel);
  msymbol = lookup_minimal_symbol_by_pc (pc);
  symbol = find_pc_function (pc);

  if (symbol && msymbol != NULL &&
      SYMBOL_VALUE_ADDRESS (msymbol) > BLOCK_START (SYMBOL_BLOCK_VALUE (symbol)))
      name = SYMBOL_LINKAGE_NAME (msymbol);
  else if (symbol)
      name = SYMBOL_CUDA_NAME (symbol);
  else if (msymbol != NULL)
      name = SYMBOL_LINKAGE_NAME (msymbol);

  cuda_textures_trace ("current kernel name: %s", name);

  if (!name)
    return false;

  while (mapping)
    {
      if (mapping->kernel_name && !strcmp_iw (name, mapping->kernel_name))
        {
          *texid = mapping->texid;
          return true;
        }
      else
        mapping = mapping->next;
    }

  return false;
}

static void
cuda_tex_append_coords (CORE_ADDR address,
                        uint32_t new_coords)
{
  /* In order to match C array and texture layout, the user has to
     give the coords in the reverse order as that of Tex2D. The coords
     is being reversed in here in order to match what the hardware
     expects. 
   */
  uint32_t i;
  for (i = TEXTURE_DIM_MAX - 1; i > 0; --i)
    texture_address.coords[i] = texture_address.coords[i - 1];

  if (address != (CORE_ADDR)(uintptr_t)&texture_address)
    {
      texture_address.address = address;
      texture_address.dim = 1;
    }
  else
    texture_address.dim++;
  texture_address.coords[0] = new_coords;
}

/* The relocated ELF image contains the symbol table index for the texture
   variable in the location attribute. Use the symtol table index to map the
   kernel name to it and save the mapping in place of the symbol table index. */
void
cuda_texture_create_kernel_mapping (struct symbol *symbol,
                                    struct objfile *objfile)
{
  uint32_t symbol_index = 0;
  cuda_tex_map_t *map = NULL;

  gdb_assert (symbol);

  cuda_textures_trace ("create mapping for symbol %s",
                       SYMBOL_LINKAGE_NAME (symbol));

  if (IS_TEXTURE_TYPE (symbol->type))
    {
      symbol_index = (uint32_t)symbol->ginfo.value.address;
      map = cuda_create_tex_map (objfile, symbol_index);
      symbol->ginfo.value.address = (CORE_ADDR)(uintptr_t)map;
    }
}

/* Returns true if the final target type of type resides in texture memory */
bool
cuda_texture_is_tex_ptr (struct type *type)
{
  CHECK_TYPEDEF (type);
  if (TYPE_CODE (type) != TYPE_CODE_PTR)
    return false;

  while (type && TYPE_CODE (type) == TYPE_CODE_PTR)
    type = TYPE_TARGET_TYPE (type);

  if (!type)
    return false;

  if (!TYPE_CUDA_TEX (type))
    return false;

  return true;
}

struct value*
cuda_texture_value_ptradd (struct type *type, struct value *base, LONGEST ofst)
{
  struct value *value = NULL;
  CORE_ADDR base_addr = 0;
  CORE_ADDR tex_addr_ptr = 0;

  cuda_textures_trace ("appending coords with offset %ld "
                       "from value_ptradd to type %p",
                       ofst, type);

  base_addr    = value_as_address (base);
  tex_addr_ptr = (CORE_ADDR)(uintptr_t)&texture_address;

  /* texture address requires two components: the address for the texture and
     the coordinates within the texture. */
  if (base_addr == tex_addr_ptr)
    /* value_as_address truncates the pointer, use tex_addr_ptr instead */
    cuda_tex_append_coords (tex_addr_ptr, ofst);
  else
    /* call value_as_address to get tex id */
    cuda_tex_append_coords (base_addr, ofst);

  texture_address.last_called = from_value_ptradd;

  value = value_from_pointer (type, tex_addr_ptr);

  return value;
}

struct value*
cuda_texture_value_ind (struct type *type, struct value* base)
{
  struct value *value = NULL;
  CORE_ADDR base_addr = 0;
  CORE_ADDR tex_addr_ptr = 0;

  cuda_textures_trace ("appending coords from value_ind to type %p", type);

  base_addr    = value_as_address (base);
  tex_addr_ptr = (CORE_ADDR)(uintptr_t)&texture_address;

  /* Do not mess up the address if value_add has already taken care of */
  if (texture_address.last_called != from_value_ptradd)
  {
    if (base_addr== tex_addr_ptr)
      /* value_as_address truncates the pointer, use tex_addr_ptr instead */
      cuda_tex_append_coords (tex_addr_ptr, 0);
    else
      /* call value_as_address to get tex id */
      cuda_tex_append_coords (base_addr, 0);
  }

  texture_address.last_called = from_value_ind;

  value = value_at_lazy (type, tex_addr_ptr);

  return value;
}

/* Walk through all the nested pointers of a texture type and set their address
   class to texture memory. */
void
cuda_texture_set_address_class (struct type *type)
{
  struct type *target_type = NULL;
  int instance_flags = 0;

  gdb_assert (IS_TEXTURE_TYPE (type));

  cuda_textures_trace ("set address class of type %p to texture", type);

  target_type = TYPE_TARGET_TYPE (type);
  while (target_type &&
         TYPE_CODE (target_type) == TYPE_CODE_PTR &&
         TYPE_TARGET_TYPE (target_type))
  {
    target_type = TYPE_TARGET_TYPE (target_type);
    instance_flags = TYPE_INSTANCE_FLAGS (target_type);
    instance_flags &= ~TYPE_INSTANCE_FLAG_ADDRESS_CLASS_ALL;
    instance_flags |= TYPE_INSTANCE_FLAG_CUDA_TEX;
    TYPE_INSTANCE_FLAGS (target_type) = instance_flags;
  }
}

/* Formats like "p *(@texture float*)texVar" will call cuda_read_memory twice.
   The first call is to read the content of texVar itself. The second call is
   to dereference the content of texVar. */
void
cuda_texture_read_tex_contents (CORE_ADDR addr, gdb_byte *buf)
{
  cuda_tex_map_t *kernel_to_tex_id = NULL;
  uint32_t tex_id = 0;

  cuda_textures_trace ("read tex contents for address %lx", addr);

  if (addr == (CORE_ADDR)(uintptr_t)&texture_address)
    *(CORE_ADDR*)buf = addr;
  else
  {
    /* The first call. "address" contains the kernel name to tex_id mapping */
    kernel_to_tex_id = (cuda_tex_map_t*)(uintptr_t)addr;
    if (!cuda_find_tex_id (kernel_to_tex_id, &tex_id))
      error (_("Texture not found in current kernel."));

    *(uint32_t*)buf = tex_id;
    texture_address.is_bindless = kernel_to_tex_id->use_symindex;
  }

  cuda_textures_trace ("read tex contents sets buf to %lx", *(uint64_t*)buf);
}

void
cuda_texture_dereference_tex_contents (CORE_ADDR addr, uint32_t *tex_id,
                                       uint32_t *dim, uint32_t **coords,
                                       bool *is_bindless)
{
  if (addr != (CORE_ADDR)(uintptr_t)&texture_address)
    error (_("Error occured whiling reading texture memory."));

  *tex_id = texture_address.address;
  *dim    = texture_address.dim;
  *coords = texture_address.coords;
  *is_bindless = texture_address.is_bindless;

  cuda_textures_trace ("dereference tex contents for address %lx "
                       "to tex_id %d dim %d coords (%d,%d,%d,%d)",
                       addr, *tex_id, *dim, *dim > 0 ? coords[0] : 0,
                       *dim > 1 ? coords[1] : 0, *dim > 2 ? coords[2] : 0,
                       *dim > 3 ? coords[3] : 0);
}
