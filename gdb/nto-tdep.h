/* nto-tdep.h - QNX Neutrino target header.

   Copyright (C) 2003-2023 Free Software Foundation, Inc.

   Contributed by QNX Software Systems Ltd.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* NVIDIA CUDA Debugger CUDA-GDB
   Copyright (C) 2007-2024 NVIDIA Corporation
   Modified from the original GDB file referenced above by the CUDA-GDB
   team at NVIDIA <cudatools@nvidia.com>. */

#ifndef NTO_TDEP_H
#define NTO_TDEP_H

#include "solist.h"
#include "osabi.h"
#include "regset.h"
#include "gdbthread.h"

/* for struct tidinfo */
#include "nto-share/dsmsgs.h"

/* taken from <sys/neutrino.h> */
#define _NTO_THREAD_NAME_MAX 100
/* taken from <sys/siginfo.h> */
#define SI_NOINFO 127

/* Target operations defined for Neutrino targets (<target>-nto-tdep.c).  */

struct nto_target_ops
{
/* Given a register, return an id that represents the Neutrino
   regset it came from.  If reg == -1 update all regsets.  */
  int (*regset_id) (int);

  void (*supply_gregset) (struct regcache *, const gdb_byte *, size_t len);

  void (*supply_fpregset) (struct regcache *, const gdb_byte *, size_t len);

  void (*supply_altregset) (struct regcache *, const gdb_byte *, size_t len);

/* Given a regset, tell gdb about registers stored in data.  */
  void (*supply_regset) (struct regcache *, int, const gdb_byte *, size_t len);

/* Given a regset and offset, return the size of the register area or
   -1 on error */
  int (*register_area) (int, unsigned);

/* Build the Neutrino register set info into the data buffer.
   Return -1 if unknown regset, 0 otherwise.  */
  int (*regset_fill) (const struct regcache *, int, gdb_byte *, size_t);

/* Gives the fetch_link_map_offsets function exposure outside of
   solib-svr4.c so that we can override relocate_section_addresses().  */
  struct link_map_offsets *(*fetch_link_map_offsets) (void);

/* Used by nto_elf_osabi_sniffer to determine if we're connected to an
   Neutrino target.  */
  enum gdb_osabi (*is_nto_target) (bfd *abfd);

/* Used on arm to determine breakpoint size: thumb/arm.  */
  int (*breakpoint_size) (CORE_ADDR addr);

/* Variant specific directory extension. e.g. -spe, -v7... */
  const char *(*variant_directory_suffix)(void);

/* Read description. */
  const struct target_desc *(*read_description) (uint64_t cpuflags);
};

nto_target_ops *get_nto_target_ops (struct gdbarch *gdbarch);
#define target_nto_gdbarch_data (get_nto_target_ops (target_gdbarch ()))

extern int nto_internal_debugging;
extern int nto_stop_on_thread_events;

#define nto_regset_id (target_nto_gdbarch_data->regset_id)

#define nto_supply_gregset (target_nto_gdbarch_data->supply_gregset)

#define nto_supply_fpregset (target_nto_gdbarch_data->supply_fpregset)

#define nto_supply_altregset (target_nto_gdbarch_data->supply_altregset)

#define nto_supply_regset (target_nto_gdbarch_data->supply_regset)

#define nto_register_area (target_nto_gdbarch_data->register_area)

#define nto_regset_fill (target_nto_gdbarch_data->regset_fill)

#define nto_breakpoint_size (target_nto_gdbarch_data->breakpoint_size)

#define nto_variant_directory_suffix (target_nto_gdbarch_data->variant_directory_suffix)

#define ntoops_read_description (target_nto_gdbarch_data->read_description)

#define nto_trace(level) \
  if ((nto_internal_debugging & 0xFF) <= (level)) {} else \
    printf_unfiltered ("nto: "); \
  if ((nto_internal_debugging & 0xFF) <= (level)) {} else \
    printf_unfiltered

#define NTO_ALL_REGS (-1)
#define RAW_SUPPLY_IF_NEEDED(regcache, whichreg, dataptr) \
  {if (!(NTO_ALL_REGS == regno || regno == (whichreg))) {} \
    else regcache->raw_supply (whichreg, dataptr); }

/* Keep this consistant with neutrino syspage.h.  */
enum
{
  CPUTYPE_X86,
  CPUTYPE_PPC,
  CPUTYPE_MIPS,
  CPUTYPE_SPARE,
  CPUTYPE_ARM,
  CPUTYPE_SH,
  CPUTYPE_X86_64,
  CPUTYPE_AARCH64,
  CPUTYPE_UNKNOWN
};

enum
{
  OSTYPE_QNX4,
  OSTYPE_NTO
};

/* These correspond to the DSMSG_* versions in dsmsgs.h.  */
enum
{
  NTO_REG_GENERAL,
  NTO_REG_FLOAT,
  NTO_REG_SYSTEM,
  NTO_REG_ALT,
  NTO_REG_END
};

typedef char qnx_reg64[8];

typedef struct _debug_regs
{
  qnx_reg64 padding[1024];
} nto_regset_t;

struct nto_thread_info : public private_thread_info
{
  unsigned char state = 0;
  unsigned char flags = 0;
  void *siginfo; // cached from core file read

public:
  void fill( const struct tidinfo *data ) {
    state=data->state;
    flags=data->flags;
  }
};

/* Per-inferior data, common for both procfs and remote.  */
struct nto_inferior_data
{
  /* Is program loaded? */
  bool has_memory;

  /* Does target have stack available? */
  bool has_stack;

  /* Is it being executed? */
  bool has_execution;

  /* Does it have registers? */
  bool has_registers;

  /* Last stopped flags result from wait function */
  unsigned int stopped_flags;

  /* Last known stopped PC */
  CORE_ADDR stopped_pc;

  /* In case of a fork, remember child pid. */
  int child_pid;

  /* In case of a fork, is it a vfork? */
  bool vfork;

  /* bind_func address needed to determine if we are in
   * dynsym code */
  CORE_ADDR bind_func_addr;

  /* Size of __bind_func symbol */
  size_t bind_func_sz;

  /* Similar to bind_func, we want to look it up only once */
  CORE_ADDR resolve_func_addr;

  /* To avoid repeatedly looking up symbols, mark here
   * that the lookup has been done.  If it is done,
   * then bind_func_ptr will not be re-calculated,
   * even if it is still zero (meaning original attempt
   * failed).
   */
  int bind_func_p;
};

/* Generic functions in nto-tdep.c.  */

void nto_init_solib_absolute_prefix (void);

char **nto_parse_redirection (char *start_argv[], const char **in,
			      const char **out, const char **err);

int nto_map_arch_to_cputype (const char *);


enum gdb_osabi nto_elf_osabi_sniffer (bfd *abfd);

void nto_initialize_signals (void);

/* Dummy function for initializing nto_target_ops on targets which do
   not define a particular regset.  */
void nto_dummy_supply_regset (struct regcache *regcache, const gdb_byte *regs,
			      size_t len);

const char *nto_extra_thread_info ( struct thread_info * );

const char *nto_thread_name( gdbarch *arch, struct thread_info *ti );

struct thread_info *nto_find_thread (ptid_t ptid);

struct link_map_offsets* nto_generic_svr4_fetch_link_map_offsets (void);

LONGEST nto_read_auxv_from_initial_stack (CORE_ADDR inital_stack,
					  gdb_byte *readbuf,
					  LONGEST len, size_t sizeof_auxv_t);

std::string nto_pid_to_str ( ptid_t);

struct nto_inferior_data *nto_inferior_data (struct inferior *inf);

struct type *nto_get_siginfo_type (struct gdbarch *);

void nto_get_siginfo_from_procfs_status (const void *status, void *siginfo);

bool nto_stopped_by_watchpoint ( );

#ifdef NVIDIA_BUGFIX
const char *nto_target (void);
#else
char *nto_target (void);
#endif

#define IS_64BIT() (gdbarch_bfd_arch_info (target_gdbarch ())->bits_per_word == 64)

extern int nto_gdb_signal_to_target (struct gdbarch *gdbarch,
				     enum gdb_signal signal);
extern enum gdb_signal nto_gdb_signal_from_target (struct gdbarch *gdbarch,
						   int nto_signal);

std::string nto_core_pid_to_str (struct gdbarch *gdbarch, ptid_t ptid);

extern void
nto_init_abi (struct gdbarch_info info, struct gdbarch *gdbarch);

inferior *nto_add_inferior (process_stratum_target *target, int pid);

extern enum bfd_endian nto_byte_order;

#endif /* NTO_TDEP_H */
