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

#include "cuda-coords.h"
#include "cuda-options.h"
#include "cuda-regmap.h"
#include "cuda-state.h"
#include "obstack.h"

#include <chrono>
#include <unordered_map>
#include <vector>

/*
   The PTX to SASS register map table is made of a series of entries,
   one per function. Each function entry is made of a list of register
   mappings, from a PTX register to a SASS register. The table size is
   saved in the first 32 bits.

     | fct name | number of entries |
       | idx | ptx_reg | sass_reg | start | end |
       | idx | ptx_reg | sass_reg | start | end |
       ...
       | idx | ptx_reg | sass_reg | start | end |
     | fct name | number of entries |
       | idx | ptx_reg | sass_reg | start | end |
       ...
     ...

   A PTX reg is mapped to one more SASS registers. If a PTX register
   is mapped to more than one SASS register, multiple entries are
   required and the 'idx' field is incremented by 1 for each one of
   them. The 'start' and 'end' addresses indicate the physical address
   between which the mapping is valid.

   The 8 high bits of a sass_reg are the register class (see cudadebugger.h).
   The low 24 bits are either the register index, or the offset in local
   memory, or the stack pointer register index and the offset.
 */

/* Raw value decoding */
#define REGMAP_CLASS(x) (x >> 24)
#define REGMAP_REG(x) (x & 0xffffff)
#define REGMAP_OFST(x) (x & 0xffffff)
#define REGMAP_SP_REG(x) ((x >> 16) & 0xff)
#define REGMAP_SP_OFST(x) (x & 0xffff)

/* Structure/global variables for storing search results*/
#define REGMAP_ENTRIES_ALLOC 64

struct regmap_st
{
  struct
  {
    const char *func_name; /* the kernel name */
    const char *reg_name;  /* the PTX register name */
    uint64_t addr;         /* the kernel-relative PC address */
  } input;
  struct
  {
    uint32_t num_entries;           /* # entries in the other fields */
    uint32_t num_entries_allocated; /* number of entries allocated for
				       raw_value[] */
    uint32_t *raw_value;            /* see REGMAP_* macros above */
    uint32_t max_location_index;    /* max loc index across all addrs */
    uint32_t *location_index;       /* location index for raw value */
    bool extrapolated; /* Indicates that regmap was extrapolated */
  } output;
};

static struct regmap_st cuda_regmap_st;
regmap_t cuda_regmap = &cuda_regmap_st;

/* Structures used to parse on-disk register mapping representation */
typedef struct
{
  union
  {
    char *byte;
    uint32_t *uint32;
  } u;
  char *byte_end;
} regmap_iterator_t;

/* Structures describing register mapping in-memory representations */
#define PTX_RNAME_LEN 8
typedef struct
{
  char rname[PTX_RNAME_LEN]; // PTX register name without trailing %
  uint32_t target;
  uint32_t start, end;
  uint32_t extended_end;
  uint32_t idx;
} regmap_map_t;

typedef struct
{
  char *name;
  uint32_t maps_no;
  regmap_map_t map[0];
} regmap_func_t;

typedef struct cuda_regmap_table
{
  struct objfile *owner;
  struct obstack *obstack;
  uint32_t num_funcs;
  regmap_func_t *func[0];
} regmap_table_t;

// For in-depth debugging
static int cuda_debug_regmap = 0;

/* Results query routines */
regmap_t
regmap_get_search_result (void)
{
  return cuda_regmap;
}

const char *
regmap_get_func_name (regmap_t regmap)
{
  gdb_assert (regmap);
  gdb_assert (regmap->input.func_name);

  return regmap->input.func_name;
}

const char *
regmap_get_reg_name (regmap_t regmap)
{
  gdb_assert (regmap);
  gdb_assert (regmap->input.reg_name);

  return regmap->input.reg_name;
}

uint64_t
regmap_get_addr (regmap_t regmap)
{
  gdb_assert (regmap);

  return regmap->input.addr;
}

uint32_t
regmap_get_num_entries (regmap_t regmap)
{
  gdb_assert (regmap);

  return regmap->output.num_entries;
}

bool
regmap_is_extrapolated (regmap_t regmap)
{
  gdb_assert (regmap);

  return regmap->output.extrapolated;
}

uint32_t
regmap_get_location_index (regmap_t regmap, uint32_t idx)
{
  gdb_assert (regmap);
  gdb_assert (idx < regmap->output.num_entries);

  return regmap->output.location_index[idx];
}

CUDBGRegClass
regmap_get_class (regmap_t regmap, uint32_t idx)
{
  gdb_assert (regmap);
  gdb_assert (idx < regmap->output.num_entries);

  return (CUDBGRegClass)REGMAP_CLASS (regmap->output.raw_value[idx]);
}

uint32_t
regmap_get_half_register (regmap_t regmap, uint32_t idx,
			  bool *in_higher_16_bits)
{
  uint32_t raw_register = 0;

  gdb_assert (in_higher_16_bits);
  gdb_assert (regmap);
  gdb_assert (idx < regmap->output.num_entries);
  gdb_assert (REGMAP_CLASS (regmap->output.raw_value[idx])
	      == REG_CLASS_REG_HALF);

  raw_register = REGMAP_REG (regmap->output.raw_value[idx]);
  *in_higher_16_bits = raw_register & 1;
  return raw_register / 2;
}

uint32_t
regmap_get_register (regmap_t regmap, uint32_t idx)
{
  gdb_assert (regmap);
  gdb_assert (idx < regmap->output.num_entries);
  gdb_assert (REGMAP_CLASS (regmap->output.raw_value[idx])
	      == REG_CLASS_REG_FULL);

  return REGMAP_REG (regmap->output.raw_value[idx]);
}

uint32_t
regmap_get_half_uregister (regmap_t regmap, uint32_t idx,
			   bool *in_higher_16_bits)
{
  uint32_t raw_register = 0;

  gdb_assert (in_higher_16_bits);
  gdb_assert (regmap);
  gdb_assert (idx < regmap->output.num_entries);
  gdb_assert (REGMAP_CLASS (regmap->output.raw_value[idx])
	      == REG_CLASS_UREG_HALF);

  raw_register = REGMAP_REG (regmap->output.raw_value[idx]);
  *in_higher_16_bits = raw_register & 1;
  return raw_register / 2;
}

uint32_t
regmap_get_uregister (regmap_t regmap, uint32_t idx)
{
  gdb_assert (regmap);
  gdb_assert (idx < regmap->output.num_entries);
  gdb_assert (REGMAP_CLASS (regmap->output.raw_value[idx])
	      == REG_CLASS_UREG_FULL);

  return REGMAP_REG (regmap->output.raw_value[idx]);
}

uint32_t
regmap_get_sp_register (regmap_t regmap, uint32_t idx)
{
  gdb_assert (regmap);
  gdb_assert (idx < regmap->output.num_entries);
  gdb_assert (REGMAP_CLASS (regmap->output.raw_value[idx])
	      == REG_CLASS_LMEM_REG_OFFSET);

  return REGMAP_SP_REG (regmap->output.raw_value[idx]);
}

uint32_t
regmap_get_sp_offset (regmap_t regmap, uint32_t idx)
{
  gdb_assert (regmap);
  gdb_assert (idx < regmap->output.num_entries);
  gdb_assert (REGMAP_CLASS (regmap->output.raw_value[idx])
	      == REG_CLASS_LMEM_REG_OFFSET);

  return REGMAP_SP_OFST (regmap->output.raw_value[idx]);
}

uint32_t
regmap_get_offset (regmap_t regmap, uint32_t idx)
{
  gdb_assert (regmap);
  gdb_assert (idx < regmap->output.num_entries);
  gdb_assert (REGMAP_CLASS (regmap->output.raw_value[idx])
	      == REG_CLASS_MEM_LOCAL);

  return REGMAP_OFST (regmap->output.raw_value[idx]);
}

/*
 * Determine if the value indicated by this register map is readable and
 * writable.
 */
static void
regmap_find_access_permissions (regmap_t regmap, bool *read, bool *write)
{
  uint32_t num_chunks, chunk;
  uint32_t num_entries, i;
  uint32_t num_instances, expected_num_instances;

  gdb_assert (regmap);
  gdb_assert (write);
  gdb_assert (read);

  /* No entry means nothing to read or write */
  num_entries = regmap_get_num_entries (regmap);
  if (num_entries == 0)
    {
      *read = false;
      *write = false;
      return;
    }

  /* Compute the number of chunks */
  num_chunks = regmap->output.max_location_index + 1;
  gdb_assert (num_chunks <= regmap->output.num_entries_allocated + 1);

  /* Iterate over each chunk to determine the permissions. */
  *read = true;
  *write = regmap_is_extrapolated (regmap) ? false : true;
  expected_num_instances = ~0U;
  for (chunk = 0; chunk < num_chunks; ++chunk)
    {
      /* Count the number of instances for this chunk */
      num_instances = 0;
      for (i = 0; i < num_entries; ++i)
	if (regmap_get_location_index (regmap, i) == chunk)
	  ++num_instances;

      /* Chunk 0, which always exists, is used as the reference */
      if (chunk == 0)
	expected_num_instances = num_instances;

      /* Not readable or writable if one chunk is missing */
      if (num_instances == 0)
	{
	  *read = false;
	  *write = false;
	}

      /* Writeable if same number of instances for all the chunks */
      if (num_instances != expected_num_instances)
	*write = false;
    }
}

bool
regmap_is_readable (regmap_t regmap)
{
  bool read = false;
  bool write = false;

  regmap_find_access_permissions (regmap, &read, &write);

  return read;
}

bool
regmap_is_writable (regmap_t regmap)
{
  bool read = false;
  bool write = false;

  regmap_find_access_permissions (regmap, &read, &write);

  return write;
}

/* Initialize regmap buffer iterator */
static void
regmap_iterator_start (regmap_iterator_t *itr, char *ptr, uint32_t size)
{
  gdb_assert (itr);

  memset (itr, 0, sizeof (*itr));
  itr->u.byte = ptr;
  itr->byte_end = ptr + size;
}

/* Create a copy of regmap buffer iterator */
static void
regmap_iterator_copy (const regmap_iterator_t *in, regmap_iterator_t *out)
{
  gdb_assert (in);
  gdb_assert (out);

  memset (out, 0, sizeof (*out));
  out->u.byte = in->u.byte;
  out->byte_end = in->byte_end;
}

/* Advance iterator position by offs bytes */
static inline void
regmap_iterator_move (regmap_iterator_t *itr, uint32_t offs)
{
  gdb_assert (itr);
  itr->u.byte += offs;
}

/* Returns true if iterator reached end of regmap section */
static bool
regmap_iterator_end (regmap_iterator_t *itr)
{
  gdb_assert (itr);
  return itr->u.byte >= itr->byte_end;
}

/* Returns pointer to a start of null terminated string
 * and advances iterator to next element.
 */
static inline char *
regmap_iterator_read_string (regmap_iterator_t *itr)
{
  char *ptr;
  gdb_assert (itr);
  ptr = itr->u.byte;
  regmap_iterator_move (itr, strlen (ptr) + 1);
  return ptr;
}

/* Returns uint32 value and advances iterator to next element */
static inline uint32_t
regmap_iterator_read_uint32 (regmap_iterator_t *itr)
{
  gdb_assert (itr);
  return *itr->u.uint32++;
}

/* read one function entry and returns the address of its first register entry
 */
static void
regmap_parse_ondisk_func (regmap_iterator_t *itr, char **fname,
			  uint32_t *num_entries, uint32_t *func_size)
{
  uint32_t i, entries;
  char *name;
  regmap_iterator_t q;

  gdb_assert (itr);

  name = regmap_iterator_read_string (itr);
  entries = regmap_iterator_read_uint32 (itr);

  /* Because the size of the register entries is variables, we must read all
     the register entries to compute the function entry size. Sigh. */
  regmap_iterator_copy (itr, &q);
  for (i = 0; i < entries; i++)
    {
      regmap_iterator_read_uint32 (&q); /* read idx */
      regmap_iterator_read_string (&q); /* read ptx_reg name */
      regmap_iterator_read_uint32 (&q); /* read value */
      regmap_iterator_read_uint32 (&q); /* read live range start */
      regmap_iterator_read_uint32 (&q); /* read live range end */
    }

  /* Return parsed values */
  if (fname)
    *fname = name;
  if (num_entries)
    *num_entries = entries;
  if (func_size)
    *func_size = q.u.byte - itr->u.byte;
}

/* reads reg_sass elf section into the xmalloced buffer */
static void
regmap_read_elf_section (struct objfile *objfile, char **buffer,
			 uint32_t *buffer_size)
{
  static const char section_name[] = ".nv_debug_info_reg_sass";
  bfd *abfd;
  asection *asection;
  char *ptr;
  uint32_t size;

  gdb_assert (objfile);
  gdb_assert (buffer);
  gdb_assert (buffer_size);

  /* default value */
  *buffer = NULL;
  *buffer_size = 0;

  /* find the proper section */
  abfd = objfile->obfd.get ();
  asection = bfd_get_section_by_name (abfd, section_name);
  if (!asection)
    return;

  /* Seek to a section start inside bfd object */
  if (bfd_seek (abfd, asection->filepos, SEEK_SET) != 0)
    return;

  /* allocate space to read the section */
  size = bfd_section_size (asection);
  ptr = (char *)xmalloc (size);
  if (!ptr)
    return;

  /* read the section */
  if (bfd_bread (ptr, size, abfd) != size)
    {
      xfree (ptr);
      return;
    }

  *buffer = ptr;
  *buffer_size = size;
}

// Extend live range of mapping until the register is used for something else
static void
regmap_extend_liverange (regmap_func_t *func)
{
  regmap_map_t *map;
  regmap_map_t *map_end = &func->map[func->maps_no];
  uint32_t end_max = 0;
  uint32_t min_entries = 0;
  uint32_t max_entries = 0;
  uint32_t compares = 0;

  const auto start {std::chrono::high_resolution_clock::now ()};

  std::unordered_map<uint64_t, std::vector<regmap_map_t *>> reg_to_map;

  // Find the max end for this function
  // Build the map of register names to a vector of ranges
  // PERF: replace this with a lookup of the function's PC range instead
  for (map = &func->map[0]; map < map_end; map++)
    {
      reg_to_map[*(uint64_t *)map->rname].emplace_back (map);

      if (end_max < map->end)
	end_max = map->end;
    }

  // Scan all the register map entries
  for (const auto& reg_vector : reg_to_map) 
    {
      // For this register, compare all the mappings with each other,
      // extending ranges as necessary
      for (auto& map : reg_vector.second)
	{
	  // Initialize the extended end to be the end of the function
	  map->extended_end = end_max;

	  // Compare all the "other" mappings against this mapping
	  for (const auto& map1 : reg_vector.second)
	    {
	      compares++;
	      // Nothing to do for comparing a map entry with itself.
	      if (map == map1)
		continue;
	      if (map1->start >= map->end)
		{
		  // This mapping is past the one we're comparing
		  // against, update map->extended_end to cover the
		  // gap
		  map->extended_end
		    = (map1->start < map->extended_end ? map1->start
		       : map->extended_end);
		}
	      else if (map1->end > map->end)
		{
		  // Overlapping range with an end further out, don't extend our range
		  map->extended_end = map->end;
		  break;
		}
	    }
	}
      // Update min/max counts
      const auto entries = reg_vector.second.size();
      if (max_entries < entries)
	max_entries = entries;
      if (!min_entries || (min_entries > entries))
	min_entries = entries;
    }

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_GENERAL) && reg_to_map.size ())
    {
      const auto end {std::chrono::high_resolution_clock::now ()};
      const std::chrono::duration<double> elapsed_seconds {end - start};
      const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds> (elapsed_seconds).count ();
      cuda_trace ("%s: %u maps %lu regs (min %u / max %u / avg %lu) %u compares (%lu us)",
		  func->name,
		  func->maps_no,
		  reg_to_map.size (),
		  min_entries,
		  max_entries,
		  func->maps_no / reg_to_map.size (),
		  compares,
		  microseconds);
    }
}

// Load regmap function record into cacheable in-memory representation
static int
regmap_load_func (regmap_iterator_t *itr, regmap_table_t *table)
{
  char *rname, *fname;
  uint32_t num_entries = 0;
  regmap_func_t *func;
  regmap_map_t *map;
  int cnt;
  uint32_t alloc_size;

  gdb_assert (itr);
  gdb_assert (table);

  regmap_parse_ondisk_func (itr, &fname, &num_entries, NULL);
  alloc_size = sizeof (regmap_func_t) + num_entries * sizeof (regmap_map_t);
  func = (regmap_func_t *)obstack_alloc (table->obstack, alloc_size);
  if (!func)
    return -1;
  memset (func, 0, alloc_size);

  func->name = (char *)obstack_alloc (table->obstack, strlen (fname) + 1);
  if (!func->name)
    {
      obstack_free (table->obstack, func);
      return -1;
    }
  strcpy (func->name, fname);

  // Read register map entries
  for (cnt = 0; cnt < num_entries; cnt++)
    {
      map = &func->map[func->maps_no];
      /* Map index is encoded as two lower bits for 32-bit word in PTX->SASS
       * register map */
      map->idx = regmap_iterator_read_uint32 (itr) & 3;
      rname = regmap_iterator_read_string (itr);
      map->target = regmap_iterator_read_uint32 (itr);
      map->start = regmap_iterator_read_uint32 (itr);
      map->end = regmap_iterator_read_uint32 (itr);
      gdb_assert (rname);
      if (rname[0] != '%' || strlen (rname + 1) >= sizeof (uint64_t))
	{
	  // This could be called 10-100 thousand times or more
	  // Only log with extended debugging enabled
	  if (cuda_debug_regmap)
	    cuda_trace ("%s: skipping regmap %s->0x%08x from 0x%08x->0x%08x",
			__func__, rname, map->target, map->start, map->end);
	  continue;
	}
      /* Append a new entry */
      strcpy (map->rname, rname + 1);
      map->extended_end = map->end;
      func->maps_no++;
    }

  regmap_extend_liverange (func);

  table->func[table->num_funcs++] = func;
  return 0;
}

/* Loads PTX->SASS register mapping table into the structure allocated on
 * objfile's heap */
static regmap_table_t *
regmap_load_table (struct objfile *objfile)
{
  char *buffer;
  uint32_t buffer_size;
  regmap_iterator_t itr;
  regmap_table_t *table;

  struct obstack *obstack;
  uint32_t cnt, alloc_size, func_size;

  gdb_assert (objfile);
  obstack = &objfile->objfile_obstack;

  /* If table was read already - return it */
  if (objfile->cuda_regmap)
    return objfile->cuda_regmap;

  regmap_read_elf_section (objfile, &buffer, &buffer_size);

  /* Count number of functions and ranges in this regmap table
   * If section is empty or can not be found, it is still safe to call
   * regmap_iterator_start() and regmap_iterator_end() would always return true
   */
  regmap_iterator_start (&itr, buffer, buffer_size);
  for (cnt = 0; !regmap_iterator_end (&itr); cnt++)
    {
      regmap_parse_ondisk_func (&itr, NULL, NULL, &func_size);
      regmap_iterator_move (&itr, func_size);
    }

  alloc_size = sizeof (regmap_table_t) + cnt * sizeof (regmap_func_t *);
  table = (regmap_table_t *)obstack_alloc (obstack, alloc_size);
  if (!table)
    goto err;

  memset (table, 0, alloc_size);
  table->owner = objfile;
  table->obstack = obstack;
  objfile->cuda_regmap = table;

  regmap_iterator_start (&itr, buffer, buffer_size);
  while (!regmap_iterator_end (&itr))
    if (regmap_load_func (&itr, table))
      goto err;

  xfree (buffer);

  return table;

err:
  objfile->cuda_regmap = NULL;

  xfree (buffer);

  if (!table)
    return NULL;

  for (cnt = 0; cnt < table->num_funcs; cnt++)
    if (table->func[cnt])
      {
	if (table->func[cnt]->name)
	  obstack_free (obstack, table->func[cnt]->name);
	obstack_free (obstack, table->func[cnt]);
      }

  obstack_free (obstack, table);

  return NULL;
}

/* Print whole regmap section */
void
regmap_table_print (struct objfile *objfile)
{
  regmap_table_t *table;
  regmap_func_t *func;
  regmap_map_t *map, *map_end;
  uint32_t cnt;

  gdb_assert (objfile);
  table = regmap_load_table (objfile);
  gdb_assert (table);

  for (func = table->func[0], cnt = 0; cnt < table->num_funcs;
       func = table->func[++cnt])
    {
      printf ("Function: %s (%u entries)\n", func->name, func->maps_no);
      for (map = &func->map[0], map_end = &func->map[func->maps_no];
	   map < map_end; map++)
	if (map->end == map->extended_end)
	  printf ("\t%10s (idx: %d) -> 0x%x, range 0x%08x - 0x%08x\n",
		  map->rname, map->idx, map->target, map->start, map->end);
	else
	  printf ("\t%10s (idx: %d) -> 0x%x, range 0x%08x - 0x%08x (could be "
		  "extended to 0x%08x)\n",
		  map->rname, map->idx, map->target, map->start, map->end,
		  map->extended_end);
    }
}

static void
regmap_append (regmap_t regmap, uint32_t idx, uint32_t target)
{
  /* Expand the vectors as needed */
  if (regmap->output.num_entries >= regmap->output.num_entries_allocated)
    {
      regmap->output.num_entries_allocated += REGMAP_ENTRIES_ALLOC;
      regmap->output.raw_value = (uint32_t *)xrealloc (
	  regmap->output.raw_value,
	  regmap->output.num_entries_allocated * sizeof (uint32_t));
      regmap->output.location_index = (uint32_t *)xrealloc (
	  regmap->output.location_index,
	  regmap->output.num_entries_allocated * sizeof (uint32_t));
    }

  regmap->output.raw_value[regmap->output.num_entries] = target;
  regmap->output.location_index[regmap->output.num_entries] = idx;
  regmap->output.num_entries++;
}

/* Generate regmap_t entry for given PTX register at given address in a given
 * function */
regmap_t
regmap_table_search (struct objfile *objfile, const char *func_name,
		     const char *reg_name, uint64_t addr)
{
  const char *tmp;
  uint32_t func_name_len;
  uint32_t cnt;
  regmap_table_t *table;
  regmap_func_t *func;
  regmap_map_t *map, *map_end;

  gdb_assert (objfile);
  gdb_assert (func_name);
  gdb_assert (reg_name && reg_name[0] == '%');

  /* Copy the function name to filter out the parameters, if any */
  func_name_len = strlen (func_name);
  tmp = strchr (func_name, '(');
  if (tmp)
    func_name_len = (unsigned long)tmp - (unsigned long)func_name;

  /* Initialize the search */
  cuda_regmap->input.func_name = func_name;
  cuda_regmap->input.reg_name = reg_name;
  cuda_regmap->input.addr = addr;
  cuda_regmap->output.num_entries = 0;
  cuda_regmap->output.extrapolated = false;
  cuda_regmap->output.max_location_index = ~0U;

  /* allocate initial vectors if needed */
  if (!cuda_regmap->output.num_entries_allocated)
    {
      cuda_regmap->output.num_entries_allocated = REGMAP_ENTRIES_ALLOC;
      cuda_regmap->output.raw_value = (uint32_t *)xcalloc (
	  cuda_regmap->output.num_entries_allocated, sizeof (uint32_t));
      cuda_regmap->output.location_index = (uint32_t *)xcalloc (
	  cuda_regmap->output.num_entries_allocated, sizeof (uint32_t));
    }
  else
    {
      /* Clear vectors (to make debugging easier) */
      memset (cuda_regmap->output.raw_value, 0,
	      cuda_regmap->output.num_entries_allocated * sizeof (uint32_t));
      memset (cuda_regmap->output.location_index, 0,
	      cuda_regmap->output.num_entries_allocated * sizeof (uint32_t));
    }

  /* Search in each function */
  table = regmap_load_table (objfile);
  if (!table || table->num_funcs == 0)
    return cuda_regmap;

  for (func = table->func[cnt = 0]; cnt < table->num_funcs;
       func = table->func[++cnt])
    {
      if (strncmp (func->name, func_name, func_name_len) != 0
	  || func->name[func_name_len] != 0)
	continue;
      for (map = &func->map[0], map_end = &func->map[func->maps_no];
	   map < map_end; map++)
	{
	  /* Discard this register reg if the register name does not match */
	  if (strcmp (map->rname, reg_name + 1) != 0)
	    continue;

	  /* Save the maximum location index encountered for this register name
	   */
	  if (cuda_regmap->output.max_location_index == ~0U
	      || map->idx > cuda_regmap->output.max_location_index)
	    cuda_regmap->output.max_location_index = map->idx;

	  /* Discard this register reg if the address if out of range/extended
	   * range */
	  if (addr < map->start
	      || addr > (cuda_options_value_extrapolation_enabled ()
			     ? map->extended_end
			     : map->end))
	    continue;

	  /* Ignore upper 64-bit of 128-bit PTX registers */
	  if (map->idx > 1)
	    continue;

	  /* Save the found element in the regmap object */
	  regmap_append (cuda_regmap, map->idx, map->target);

	  /* Mark output as extrapolated */
	  if (cuda_options_value_extrapolation_enabled () && addr > map->end)
	    cuda_regmap->output.extrapolated = true;
	}
    }

  return cuda_regmap;
}

/* See if reg is a properly encoded CUDA physical register.  Currently only
   used by the DWARF2 frame reader (see dwarf2-frame.c) to decode CFA
   instructions that take a ULEB128-encoded register as an argument.  More
   noteably, it neither overrides nor is tied to a gdbarch register method.

   NOTE:  This is the raw backend encoding of a physical register, inclusive
   of the reg class and reg # (not the ULEB128-encoded virtual PTX register
   name).
 */
int
cuda_decode_physical_register (uint64_t reg, int32_t *result)
{
  uint32_t dev_id = cuda_current_focus::get ().physical ().dev ();
  uint32_t num_regs = cuda_state::device_get_num_registers (dev_id);
  ULONGEST last_regnum = num_regs - 1;

  if (reg < last_regnum)
    {
      *result = (int32_t)reg;
      return 0;
    }

  if (REGMAP_CLASS (reg) == REG_CLASS_REG_FULL)
    {
      *result = (int32_t)REGMAP_REG (reg);
      return 0;
    }

  return -1;
}
