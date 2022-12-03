/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2022 NVIDIA Corporation
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
#include "cuda-functions.h"
#include "cuda-tdep.h"

#include "elf/common.h"
#include "elf/external.h"

#include <vector>

#define R_CUDA_NONE                   0      // no relocation
#define R_CUDA_32                     1      // 32bit specific address
#define R_CUDA_64                     2      // 64bit specific address
#define R_CUDA_G32                    3      // 32bit generic address
#define R_CUDA_G64                    4      // 64bit generic address

#if CUDBG_API_VERSION_REVISION >= 132

static void
cuda_trace_loading (const char *fmt, ...)
{

  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_EVENT, fmt, ap);
  va_end (ap);
}

void
cuda_apply_function_load_updates (char *buffer, CUDBGLoadedFunctionInfo *info, uint32_t count)
{
  cuda_trace_loading ("%s count %u", __FUNCTION__, count);

  gdb_assert (buffer);
  gdb_assert (info);

  Elf64_External_Ehdr *e_hdr = (Elf64_External_Ehdr *)buffer;

  /* Update section table addresses */

  auto shoff = *(uint64_t *)e_hdr->e_shoff;
  auto shnum = *(uint16_t *)e_hdr->e_shnum;
  auto shentsize = *(uint16_t *)e_hdr->e_shentsize;

  cuda_trace_loading ("%d sections", shnum);

  std::vector<uint64_t> section_addresses;
  for (int i = 0; i < shnum; ++i)
    section_addresses.push_back (0);

  /* Only update section_addresses for newly loaded functions.
     We detect these by looking for a value of 0 in the section table. */

  /* Find the functions (sections) that are newly loaded this time */
  for (int i = 0; i < count; ++i)
    {
      section_addresses[info[i].sectionIndex] = info[i].address;
      cuda_trace_loading ("relocate section %d to 0x%lx",
			  info[i].sectionIndex, info[i].address);
    }

  /* Relocate the sections */
  for (int sectionidx = 0; sectionidx < shnum; ++sectionidx)
    {
      auto section = (Elf64_External_Shdr *)&buffer[shoff + sectionidx * shentsize];

      /* Check for newly relocated section */
      auto section_address = *(uint64_t *)section->sh_addr;
      if (section_addresses[sectionidx])
	{
	  if (section_address)
	    {
	      cuda_trace_loading ("skipping already relocated section %d addr 0x%lx",
				  sectionidx, section_addresses[sectionidx]);
	      /* Remove from the updated sections vector */
	      section_addresses[sectionidx] = 0;
	      continue;
	    }
	  *(uint64_t *)section->sh_addr = section_addresses[sectionidx];
	  cuda_trace_loading ("relocated section %d addr 0x%lx",
			      sectionidx, section_addresses[sectionidx]);
	}
    }


  /* Find the symtab and strtab sections */
  auto shstrndx = *(uint16_t *)e_hdr->e_shstrndx;
  auto shstrtab_section = (Elf64_External_Shdr *)&buffer[shoff + shstrndx * shentsize];
  auto shstrtab_offset = *(uint64_t *)shstrtab_section->sh_offset;
  auto shstrtab = (char *)&buffer[shstrtab_offset];

  Elf64_External_Shdr *strtab = nullptr;
  Elf64_External_Shdr *symtab = nullptr;
  for (int sectionidx = 0; sectionidx < shnum; ++sectionidx)
    {
      auto section = (Elf64_External_Shdr *)&buffer[shoff + sectionidx * shentsize];
      auto section_type = *(uint32_t *)section->sh_type;

      auto shstr_offset = *(uint32_t *)section->sh_name;
      auto section_name = &shstrtab[shstr_offset];

      if (section_type == SHT_SYMTAB)
	symtab = section;
      else if (!strcmp (section_name, ".strtab"))
	strtab = section;
    }

  /* Need both a string table and a symbol table */
  gdb_assert (strtab);
  gdb_assert (symtab);

  /* Found the string table */
  auto strtab_offset = *(uint64_t *)strtab->sh_offset;

  /* Found the symbol table */
  auto symtab_size = *(uint64_t *)symtab->sh_size;
  auto symtab_entsize = *(uint32_t *)symtab->sh_entsize;
  auto symtab_offset = *(uint64_t *)symtab->sh_offset;

  /* Update symbol table values based on section addresses. */
  std::vector<uint64_t> symbol_updated_address;
  for (int symidx = 0; symidx < symtab_size/symtab_entsize; ++symidx)
    {
      auto symbol = (Elf64_External_Sym *)&buffer[symtab_offset + symidx * symtab_entsize];

      /* Only relocate symbols which belong to newly located sections */
      auto sym_shndx = *(uint16_t *)symbol->st_shndx;
      if (section_addresses[sym_shndx])
	{
	  auto value = *(uint64_t *)symbol->st_value;
	  value += section_addresses[sym_shndx];

	  auto symbol_name = &buffer[strtab_offset + *(uint32_t *)symbol->st_name];
	  cuda_trace_loading ("relocating symbol %d section %d address 0x%lx name %s",
			      sym_shndx, (uint32_t)sym_shndx, value, symbol_name);

	  *(uint64_t *)symbol->st_value = value;

	  symbol_updated_address.push_back (value);
	}
      else
	symbol_updated_address.push_back (0);
    }

  /* Apply relocations based on symbols just relocated */
  for (int sectionidx = 0; sectionidx < shnum; ++sectionidx)
    {
      auto reloc_section = (Elf64_External_Shdr *)&buffer[shoff + sectionidx * shentsize];
      auto reloc_section_type = *(uint32_t *)reloc_section->sh_type;
      auto reloc_section_name = &shstrtab[*(uint32_t *)reloc_section->sh_name];

      if ((reloc_section_type == SHT_RELA) || (reloc_section_type == SHT_REL))
	{
	  bool is_rela = (reloc_section_type == SHT_RELA) ? true : false;

	  /* The section we are relocating */
	  auto target_section_idx = *(uint32_t *)reloc_section->sh_info;
	  auto target_section = (Elf64_External_Shdr *)&buffer[shoff + target_section_idx * shentsize];
	  auto target_section_ptr = &buffer[*(uint64_t *)target_section->sh_offset];

	  auto target_section_name = &shstrtab[*(uint32_t *)target_section->sh_name];
	  cuda_trace_loading ("relocate section %s %d %s with %s",
			      is_rela ? "RELA" : "REL ",
			      sectionidx,
			      target_section_name,
			      reloc_section_name);

	  uint64_t reloc_section_size = *(uint64_t *)reloc_section->sh_size;
	  uint64_t reloc_section_offset = *(uint64_t *)reloc_section->sh_offset;
	  for (uint64_t offset = reloc_section_offset; offset < (reloc_section_offset + reloc_section_size); )
	    {
	      uint64_t r_offset = 0;
	      uint64_t r_info = 0;
	      uint64_t r_addend = 0;

	      if (is_rela)
		{
		  auto rela = (Elf64_External_Rela *)&buffer[offset];

		  r_offset = *(uint64_t *)rela->r_offset;
		  r_info = *(uint64_t *)rela->r_info;
		  r_addend = *(uint64_t *)rela->r_addend;

		  offset += sizeof (Elf64_External_Rela);
		}
	      else
		{
		  auto rel = (Elf64_External_Rel *)&buffer[offset];

		  r_offset = *(uint64_t *)rel->r_offset;
		  r_info = *(uint64_t *)rel->r_info;

		  /* r_addend set below based on relocation size */

		  offset += sizeof (Elf64_External_Rel);
		}

	      auto reltype = ELF64_R_TYPE (r_info);

	      /* Skip relocations for symbols we didn't just update */
	      auto symidx = ELF64_R_SYM (r_info);
	      if (symidx >= symtab_size/symtab_entsize)
		{
		  cuda_trace_loading ("skipping unknown symbol %d", symidx);
		  continue;
		}
	      
	      auto symbol = (Elf64_External_Sym *)&buffer[symtab_offset + symidx * symtab_entsize];
	      auto symbol_name = &buffer[strtab_offset + *(uint32_t *)symbol->st_name];

	      if (!symbol_updated_address[symidx])
		{
		  cuda_trace_loading ("skipping unrelocated symbol %d %s", symidx, symbol_name);
		  continue;
		}

	      cuda_trace_loading ("\trel type %d idx %d offset 0x%lx ovalue 0x%lx value 0x%lx symbol %s",
				  (uint32_t)reltype, symidx, r_offset,
				  symbol_updated_address[symidx],
				  symbol_updated_address[symidx] + r_addend,
				  symbol_name);

	      auto target_ptr = (char *)&target_section_ptr[r_offset];
	      switch (reltype)
		{
		case R_CUDA_32:
		case R_CUDA_G32:
		  if (!is_rela)
		    r_addend = *(uint32_t *)&target_section_ptr[r_offset];
		  *(uint32_t *)target_ptr = (uint32_t)(symbol_updated_address[symidx] + r_addend);
		  break;

		case R_CUDA_64:
		case R_CUDA_G64:
		  if (!is_rela)
		    r_addend = *(uint64_t *)&target_section_ptr[r_offset];
		  *(uint64_t *)target_ptr = (uint64_t)(symbol_updated_address[symidx] + r_addend);
		  break;

		default:
		  /* Unknown relocation, skip */
		  cuda_trace_loading ("unknown relocation type %d", reltype);
		  break;
		}
	    }
	}
    }
}
#endif
