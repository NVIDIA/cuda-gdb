/* NVIDIA CUDA Debugger CUDA-GDB
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

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <signal.h>
#if !defined(__QNX__)
# include <execinfo.h>
#endif
#include <dlfcn.h>
#include <link.h>

#include "bfd.h"
#include "cuda-tdep.h"
#include "self-bt.h"

extern struct r_debug _r_debug;
static asymbol **symlist;
static long sym_count;

struct bt_frame_info
{
  void *addr;
  char *obj;
  char *symbol;
  ptrdiff_t offset;
};

static void clean_frame_info_struct (struct bt_frame_info *btfi)
{
  free (btfi->obj);
  free (btfi->symbol);
}

static void exit_error (void)
{
  fprintf (stderr, "An error has occured while resolving the backtrace.\n");
  fflush (stderr);
  cuda_cleanup ();
  exit (1);
}

static void bfd_symbol_extraction (bfd *curr_exe)
{
  long sym_size = bfd_get_symtab_upper_bound (curr_exe);

  if (sym_size <= 0)
    exit_error ();

  symlist = (asymbol**)malloc (sym_size);

  if (!symlist)
    exit_error ();

  sym_count = bfd_canonicalize_symtab (curr_exe, symlist);
}

static void bfd_deinitialize (bfd *curr_exe)
{
  if (curr_exe)
    bfd_close (curr_exe);
  free (symlist);
}

static bfd *bfd_initialize (void)
{
  bfd *curr_exe = NULL;

  bfd_init ();
  curr_exe = bfd_openr ("/proc/self/exe", NULL);

  if (!curr_exe)
    exit_error ();

  if (!bfd_check_format (curr_exe, bfd_object))
    exit_error ();

  bfd_symbol_extraction (curr_exe);

  return curr_exe;
}

/* Try to get the symbol name by locating the ELF symbols in memory and find the
 * closest symbol to our bt address. */
static void resolve_with_bfd (struct bt_frame_info *btfi)
{
  long i;
  symbol_info sym = {0};
  unsigned long target = 0;

  if ((uint64_t)btfi->addr < _r_debug.r_map->l_addr)
    return;

  /* AFAIK the first entry of the link map is the executable itself */
  target = (uint64_t)btfi->addr - _r_debug.r_map->l_addr;

  /* TODO: The search could be optimized, do we really want it? */
  for (i = 0; i < sym_count; ++i)
    {
      symbol_info si = {0};

      /* Discard section start labels (e.g _start and .text may coincide)*/
      if (symlist[i]->flags & BSF_SECTION_SYM)
        continue;

      bfd_symbol_info (symlist[i], &si);

      if (!si.value || si.value > target)
        continue;

      if (si.value > sym.value)
        sym = si;
    }

  /* If we get past _end, the symbol belongs to another section (should have
   * been resolved earlier using dladdr(3)) */
  if (!sym.name || !strcmp (sym.name, "_end"))
    return;

  btfi->symbol = strdup (sym.name);
  btfi->offset = target - sym.value;
}

/* Try to get the symbol name using dladdr(3) in case the symbol is exported in
 * the dynamic table. */
static void resolve_with_dladdr (struct bt_frame_info *btfi)
{
  Dl_info tmp = {0};

  if (!btfi->addr)
    return;

  if (!dladdr (btfi->addr, &tmp))
    return;

  if (tmp.dli_sname)
    btfi->symbol = strdup (tmp.dli_sname);

  if (tmp.dli_fname)
    btfi->obj = strdup (tmp.dli_fname);

  if (tmp.dli_saddr)
    btfi->offset = (char*)btfi->addr - (char*)tmp.dli_saddr;
}

static void resolve (struct bt_frame_info *btfi)
{
  resolve_with_dladdr (btfi);

  if (!btfi->symbol)
    resolve_with_bfd (btfi);

  if (btfi->symbol)
    {
      char *demangled = bfd_demangle (NULL, btfi->symbol, 0);

      if (!demangled)
        return;

      free (btfi->symbol);
      btfi->symbol = demangled;
    }
}

static void print_bt_info (struct bt_frame_info *btfi)
{
  size_t obj_len = btfi->obj ? strlen (btfi->obj) : 0;

  if (obj_len > 15)
    fprintf (stderr, "...%12s| ", btfi->obj + obj_len - 12);
  else
    fprintf (stderr, "%15s| ", btfi->obj ? btfi->obj : "unknown");

  fprintf (stderr, "%s", btfi->symbol ? btfi->symbol : "?????");
  if (btfi->symbol)
    fprintf (stderr, "() +0x%tx", btfi->offset);

  fprintf (stderr, "\n");
  fflush (stderr);
}

void segv_handler (__attribute__((unused)) int signo)
{
  int i;
  int count = 0;
  void *buffer[100] = {0};

#if !defined(__QNX__)
  fprintf (stderr, "\ncuda-gdb has received a SIGSEGV and will attempt to get its own backtrace.\n\n");

  count = backtrace (buffer, 100);
  bfd *curr = bfd_initialize ();

  for (i = 0; i < count; ++i)
    {
      struct bt_frame_info btfi = {.addr = buffer[i]};

      resolve (&btfi);
      print_bt_info (&btfi);

      clean_frame_info_struct (&btfi);
    }

  if (curr)
    bfd_deinitialize (curr);
#else
  fprintf (stderr, "\ncuda-gdb has received a SIGSEGV.\n\n");
#endif

  fflush (stderr);
  cuda_cleanup ();
  exit (1);
}
