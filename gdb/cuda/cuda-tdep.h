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

#ifndef _CUDA_TDEP_H
#define _CUDA_TDEP_H 1

#include "gdbarch.h"
#include "cuda-api.h"
#include "cuda-coords.h"

#include <vector>

struct inferior;
struct gdbarch;

/* CUDA - skip prologue
   REMOVE ONCE TRANSITION TESLA KERNELS HAVE PROLOGUES ALL THE TIME */
extern bool cuda_producer_is_open64;

/* CUDA ELF Specification */

#define EV_CURRENT 1
#define ELFOSABI_CUDA 0x33
#define CUDA_ELFOSABIV_16BIT   0 /* 16-bit ctaid.x size */
#define CUDA_ELFOSABIV_32BIT   1 /* 32-bit ctaid.x size */
#define CUDA_ELFOSABIV_RELOC   2 /* ELFOSABIV_32BIT + All relocators in DWARF*/
#define CUDA_ELFOSABIV_ABI     3   /* ELFOSABIV_RELOC + Calling Convention */
#define CUDA_ELFOSABIV_SYSCALL 4 /* ELFOSABIV_ABI + Improved syscall relocation */
#define CUDA_ELFOSABIV_SEPCOMP 5 /* ELFOSABIV_SYSCALL + new caller-callee save conventions */
#define CUDA_ELFOSABIV_ABI3    6 /* ELFOSABIV_SEPCOMP + fixes */
#define CUDA_ELFOSABIV_ABI4    7 /* ELFOSABIV_ABI3 + runtime JIT link */
#define CUDA_ELFOSABIV_STD_ELF 8 /* Standard ELF revision */
#define CUDA_ELFOSABIV_LATEST  CUDA_ELFOSABIV_STD_ELF

#define CUDA_ELF_TEXT_PREFIX                                                  \
  ".text." /* CUDA ELF text section format: ".text.KERNEL" */

// Maximum numbers of registers and predicates, both regular and uniform
// Do not count the "virtual" zero register as a regular register
#define CUDA_REG_MAX_REGISTERS  255u
#define CUDA_REG_ZERO_REGISTER  255u
#define CUDA_REG_MAX_PREDICATES 8u

// Do not count the "virtual" zero register as a uniform register
#define CUDA_UREG_MAX_REGISTERS  255u
#define CUDA_UREG_MAX_PREDICATES 8u

// Encoded register for the zero register
#define CUDA_UREG_ZERO_REGISTER  255u

/*Return values that exceed 384-bits in size are returned in memory.
   (R4-R15 = 12 4-byte registers = 48-bytes = 384-bits that can be
   used to return values in registers). */
#define CUDA_ABI_MAX_REG_RV_SIZE 48 /* Size in bytes */

#define CUDA_REG_CLASS_AND_REGNO(cl, regnum)                                  \
  (((cl) << 24) | ((regnum)&0x00ffffff))

/* Used to convert dwarf2 regno to identifier that fits inside INT_MAX. This is
 * a gdb upstream requirement for DW_OP_regx. */
/* We don't want to use the sign bit as our tag. A requirement is that
 * registers must be positive integers only. */
#define CUDA_PTX_VIRTUAL_TAG (1 << 30)
#define CUDA_PTX_VIRTUAL_ID(idx) ((idx) | CUDA_PTX_VIRTUAL_TAG)
/* We want everything but our tag bit and the sign bit to be 1 */
#define CUDA_PTX_VIRTUAL_IDX(id) ((id) & (CUDA_PTX_VIRTUAL_TAG - 1))

/* Type Declarations */

#define CUDA_MAX_NUM_RESIDENT_BLOCKS_PER_GRID 256
#define CUDA_MAX_NUM_RESIDENT_THREADS_PER_BLOCK 1024
#define CUDA_MAX_NUM_RESIDENT_THREADS                                         \
  (CUDA_MAX_NUM_RESIDENT_BLOCKS_PER_GRID                                      \
   * CUDA_MAX_NUM_RESIDENT_THREADS_PER_BLOCK)

typedef enum return_value_convention rvc_t;

/* Global Variables */

extern bool cuda_debugging_enabled;
struct gdbarch *cuda_get_gdbarch (void);
bool cuda_is_cuda_gdbarch (struct gdbarch *);

/* Offsets of the CUDA built-in variables */
#define CUDBG_BUILTINS_BASE ((CORE_ADDR)0)
#define CUDBG_THREADIDX_OFFSET (CUDBG_BUILTINS_BASE - 12)
#define CUDBG_BLOCKIDX_OFFSET (CUDBG_THREADIDX_OFFSET - 12)
#define CUDBG_CLUSTERIDX_OFFSET (CUDBG_BLOCKIDX_OFFSET - 12)
#define CUDBG_GRIDDIM_OFFSET (CUDBG_CLUSTERIDX_OFFSET - 12)
#define CUDBG_BLOCKDIM_OFFSET (CUDBG_GRIDDIM_OFFSET - 12)
#define CUDBG_CLUSTERDIM_OFFSET (CUDBG_BLOCKDIM_OFFSET - 12)
#define CUDBG_WARPSIZE_OFFSET (CUDBG_CLUSTERDIM_OFFSET - 4)
#define CUDBG_BUILTINS_MAX (CUDBG_WARPSIZE_OFFSET)

/* CUDA architecture specific information.  */
struct cuda_gdbarch_tdep : gdbarch_tdep_base
{
  /* Pointer size: 32-bits on i686, 64-bits on x86_64. So that we do not need
     to have 2 versions of the cuda-tdep file */
  static constexpr int ptr_size = TARGET_CHAR_BIT * sizeof (CORE_ADDR);

  // PC
  static constexpr int pc_regnum = 0;
  
  // ErrorPC register
  static constexpr int error_pc_regnum = pc_regnum + 1;

  // Regular registers
  static constexpr int first_regnum = error_pc_regnum + 1;
  static constexpr int last_regnum = first_regnum + (CUDA_REG_MAX_REGISTERS - 1);
  static constexpr int zero_regnum = last_regnum + 1;

  // Predicate Registers
  static constexpr int first_pred_regnum = zero_regnum + 1;
  static constexpr int last_pred_regnum = first_pred_regnum + (CUDA_REG_MAX_PREDICATES - 1);

  // Uniform Registers
  static constexpr int first_uregnum = last_pred_regnum + 1;
  static constexpr int last_uregnum = first_uregnum + (CUDA_UREG_MAX_REGISTERS - 1);
  static constexpr int zero_uregnum = last_uregnum + 1;
  static constexpr int first_upred_regnum = zero_uregnum + 1;
  static constexpr int last_upred_regnum = first_upred_regnum + (CUDA_UREG_MAX_PREDICATES - 1);

  // CC register
  static constexpr int cc_regnum = last_upred_regnum + 1;

  // Regular registers, predicates, uniform registers, uniform predicates, PC
  static constexpr int num_regs = cc_regnum + 1;

  // Pseudo-Registers
  static constexpr int first_pseudo_regnum = num_regs;

  // Special register to indicate to look at the regmap search result
  static constexpr int special_regnum = first_pseudo_regnum;

  // Register number to tell the debugger that no valid register was found.
  // Used to avoid returning errors and having nice warning messages and
  // consistent garbage return values instead (zero sounds good).
  // invalid_lo_regnum is used for variables that aren't live and are
  // stored in a single register.  The combination of invalid_lo_regnum
  // and invalid_hi_regnum is needed when a variable isn't live and is
  // stored in multiple registers.
  static constexpr int invalid_lo_regnum = special_regnum + 1;
  static constexpr int invalid_hi_regnum = invalid_lo_regnum + 1;
  static constexpr int last_pseudo_regnum = invalid_hi_regnum;

  // Pseudo reg count
  static constexpr int num_pseudo_regs = last_pseudo_regnum - first_pseudo_regnum + 1;

  // ABI only
  static constexpr int sp_regnum = first_regnum + 1;       // ABI only, SP is in R1
  static constexpr int first_rv_regnum = first_regnum + 4; // ABI only, First RV is in R4, also used to pass args
  static constexpr int last_rv_regnum = first_regnum + 15; // ABI only, Last RV is in R15, also used to pass args
  static constexpr int max_reg_rv_size = (last_rv_regnum - first_rv_regnum + 1) * 4;
};

/* Predicates for checking if register belongs to specified register group */
static inline bool
cuda_regular_register_p (struct gdbarch *gdbarch, int regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return regnum >= tdep->first_regnum && regnum <= tdep->last_regnum;
}

static inline bool
cuda_zero_register_p (struct gdbarch *gdbarch, int regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return regnum == tdep->zero_regnum;
}

static inline bool
cuda_special_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return regnum == tdep->special_regnum;
}

static inline bool
cuda_invalid_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return regnum == tdep->invalid_lo_regnum
         || regnum == tdep->invalid_hi_regnum;
}

static inline bool
cuda_is_regnum_valid (struct gdbarch *gdbarch, int regnum)
{
  return !cuda_invalid_regnum_p (gdbarch, regnum);
}

static inline bool
cuda_pred_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return regnum >= tdep->first_pred_regnum && regnum <= tdep->last_pred_regnum;
}

static inline bool
cuda_upred_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return regnum >= tdep->first_upred_regnum
         && regnum <= tdep->last_upred_regnum;
}

static inline bool
cuda_uniform_zero_register_p (struct gdbarch *gdbarch, int regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return regnum == tdep->zero_uregnum;
}

static inline bool
cuda_uregnum_p (struct gdbarch *gdbarch, int regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return regnum >= tdep->first_uregnum && regnum <= tdep->last_uregnum;
}

static inline bool
cuda_pc_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return regnum == tdep->pc_regnum;
}

static inline bool
cuda_cc_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return regnum == tdep->cc_regnum;
}

static inline bool
cuda_error_pc_regnum_p (struct gdbarch *gdbarch, int regnum)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return regnum == tdep->error_pc_regnum;
}

static inline int
cuda_abi_sp_regnum (struct gdbarch *gdbarch)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return tdep->sp_regnum;
}

static inline int
cuda_special_regnum (struct gdbarch *gdbarch)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return tdep->special_regnum;
}

static inline int
cuda_pc_regnum (struct gdbarch *gdbarch)
{
  cuda_gdbarch_tdep *tdep = gdbarch_tdep<cuda_gdbarch_tdep> (gdbarch);
  return tdep->pc_regnum;
}

// CUDART symbols
struct cuda_cudart_symbols_st
{
  struct objfile *objfile;
};

// Signal handling information
struct cuda_signal_info_st
{
  bool saved;
  uint32_t print;
  uint32_t stop;
};

/* Prototypes to avoid implicit declarations (hack-hack) */

extern bool cuda_initialized;

struct partial_symtab;
void switch_to_cuda_thread (const cuda_coords &coords);
int cuda_thread_select (char *, int);
void cuda_update_cudart_symbols (void);
void cuda_cleanup_cudart_symbols (void);

/* Prototypes */

int cuda_startup (void);
void cuda_kill (void);
void cuda_cleanup (void);
void cuda_final_cleanup (void *unused);
bool cuda_initialize_target (void);
void cuda_initialize (void);
bool cuda_inferior_in_debug_mode (void);
void cuda_load_device_info (char *, struct partial_symtab *);
void cuda_signals_initialize (void);
void cuda_update_report_driver_api_error_flags (void);

gdb::unique_xmalloc_ptr<char> cuda_find_function_name_from_pc (CORE_ADDR pc,
                                                               bool demangle);
bool cuda_breakpoint_hit_p (cuda_coords &coords);

uint64_t cuda_get_last_driver_api_error_code (void);
void cuda_get_last_driver_api_error_func_name (char **name);
bool cuda_get_last_driver_api_error_source_name (std::string &source);
bool cuda_get_last_driver_api_error_name (std::string &name);
bool cuda_get_last_driver_api_error_string (std::string &string);
uint64_t cuda_get_last_driver_internal_error_code (void);

/* Debugging */
typedef enum
{
  CUDA_TRACE_GENERAL,
  CUDA_TRACE_EVENT,
  CUDA_TRACE_BREAKPOINT,
  CUDA_TRACE_API,
  CUDA_TRACE_SIGINFO,
  CUDA_TRACE_DISASSEMBLER,
  CUDA_TRACE_ELF,
  CUDA_TRACE_STATE,
  CUDA_TRACE_STATE_DECODE,
} cuda_trace_domain_t;

void cuda_vtrace_domain (cuda_trace_domain_t, const char *, va_list);
void cuda_trace_domain (cuda_trace_domain_t domain, const char *, ...);
void cuda_trace (const char *, ...);

/*Single-Stepping */
bool cuda_sstep_is_active (void);
bool cuda_print_divergent_stepping (void);
uint32_t cuda_sstep_dev_id (void);
uint64_t cuda_sstep_grid_id (void);
uint32_t cuda_sstep_wp_id (void);
uint32_t cuda_sstep_sm_id (void);
uint64_t cuda_sstep_get_last_pc (void);
bool cuda_sstep_lane_stepped (uint32_t ln_id);
uint32_t cuda_sstep_get_lowest_lane_stepped (void);
cuda_api_warpmask *cuda_sstep_wp_mask (void);
ptid_t cuda_sstep_ptid (void);
void cuda_sstep_set_ptid (ptid_t ptid);
void cuda_sstep_set_nsteps (int nsteps);
void cuda_sstep_initialize (bool stepping);
bool cuda_sstep_execute (ptid_t ptid);
void cuda_sstep_reset (bool sstep);
bool cuda_sstep_kernel_has_terminated (void);

/*Registers */
uint64_t cuda_check_dwarf2_reg_ptx_virtual_register (uint64_t dwarf2_reg);
uint64_t cuda_check_dwarf2_reg_ascii_encoded_register (struct gdbarch *gdbarch, uint64_t dwarf2_reg);
int cuda_reg_to_regnum_extrapolated (struct gdbarch *gdbarch, int reg);

int cuda_reg_to_regnum (struct gdbarch *gdbarch, int reg);
int cuda_regnum_to_reg (struct gdbarch *gdbarch, uint32_t regnum);
bool cuda_is_regnum_valid (struct gdbarch *gdbarch, int regnum);
enum register_status cuda_pseudo_register_read (struct gdbarch *gdbarch,
                                                readable_regcache *regcache,
                                                int regnum, gdb_byte *buf);
void cuda_register_read (struct gdbarch *gdbarch, struct regcache *regcache,
                         int regnum);
void cuda_register_write (struct gdbarch *gdbarch, struct regcache *regcache,
                          int regnum, const gdb_byte *buf);

/*Storage addresses and names */
void cuda_print_lmem_address_type (void);
type_instance_flags cuda_address_class_type_flags (int byte_size,
                                                   int addr_class);

/*ABI/BFD/ELF/DWARF/objfile calls */
bool cuda_is_bfd_cuda (bfd *obfd);
bool cuda_is_bfd_version_call_abi (bfd *obfd);
bool cuda_get_bfd_abi_version (bfd *obfd, unsigned int *abi_version);
bool cuda_current_active_elf_image_uses_abi (void);
CORE_ADDR cuda_dwarf2_func_baseaddr (struct objfile *objfile, char *func_name);
bool cuda_find_pc_from_address_string (struct objfile *objfile,
                                       const char *func_name, CORE_ADDR &func_addr);
bool cuda_find_func_text_vma_from_objfile (struct objfile *objfile,
                                           char *func_name, CORE_ADDR *vma);
bool cuda_is_device_code_address (CORE_ADDR addr);
int cuda_abi_sp_regnum (struct gdbarch *);
int cuda_special_regnum (struct gdbarch *);
int cuda_pc_regnum (struct gdbarch *);
CORE_ADDR cuda_get_symbol_address (const char *name);
int cuda_dwarf2_addr_size (struct objfile *objfile);
void cuda_decode_line_table (struct objfile *objfile);

/*Segmented memory reads/writes */
int cuda_read_memory (CORE_ADDR address, type_instance_flags flags,
		      gdb_byte *buf, int len);

int cuda_read_memory (CORE_ADDR address, struct value *val, struct type *type,
                      int len);

int cuda_write_memory_partial (CORE_ADDR address, const gdb_byte *buf,
                               struct type *type);
void cuda_write_memory (CORE_ADDR address, const gdb_byte *buf,
                        struct type *type);

/*Breakpoints */
void cuda_resolve_breakpoints (int bp_number_from, cuda_module* module);
void cuda_unresolve_breakpoints (cuda_module* module);
void cuda_reset_invalid_breakpoint_location_section (struct objfile *objfile);
int cuda_breakpoint_address_match (struct gdbarch *gdbarch,
                                   const address_space *aspace1,
                                   CORE_ADDR addr1,
                                   const address_space *aspace2,
                                   CORE_ADDR addr2);
void cuda_adjust_host_pc (ptid_t r);
void cuda_adjust_device_code_address (CORE_ADDR original_addr,
                                      CORE_ADDR *adjusted_addr);
void cuda_next_device_code_address (CORE_ADDR original_addr,
                                    CORE_ADDR *adjusted_addr);
bool cuda_find_next_control_flow_instruction (
    uint64_t pc, uint64_t range_start_pc, uint64_t range_end_pc,
    bool skip_subroutines, uint64_t &end_pc, uint32_t &inst_size);

bool cuda_platform_supports_tid (void);
int cuda_gdb_get_tid_or_pid (ptid_t ptid);
int cuda_get_signo (void);
void cuda_set_signo (int signo);

/* Session Management */
int cuda_gdb_session_create (void);
void cuda_gdb_session_destroy (void);
const char *cuda_gdb_session_get_dir (void);
uint32_t cuda_gdb_session_get_id (void);

/* Attach support */
void cuda_do_detach (struct inferior *inf);

/* Support for detecting host shadow functions */
void cuda_find_objfile_host_shadow_functions (struct objfile *objfile);

#endif
