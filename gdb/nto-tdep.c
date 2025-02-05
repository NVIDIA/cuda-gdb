/* nto-tdep.c - general QNX Neutrino target functionality.

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

#include "defs.h"
#include <sys/stat.h>
#include "nto-tdep.h"
#include "top.h"
#include "inferior.h"
#include "infrun.h"
#include "gdbarch.h"
#include "bfd.h"
#include "elf-bfd.h"
#include "solib-svr4.h"
#include "gdbcore.h"
#include "gdb/signals.h"
#include "gdbcmd.h"
#include "objfiles.h"
#include "source.h"
#include "gdbsupport/pathstuff.h"
#include "observable.h"
#include "minsyms.h"
#include "linux-nat.h"
#include "solib.h"
#include "auxv.h"

#define QNX_NOTE_NAME	"QNX"
#define QNX_INFO_SECT_NAME "QNX_info"

#ifdef __QNXNTO__
#include <sys/debug.h>
typedef debug_process_t nto_procfs_info;
typedef siginfo_t nto_siginfo_t;
typedef struct _debug_thread_info64 nto_procfs_status;
#define nto_si_pid  si_pid
#define nto_si_uid  si_uid
#define nto_si_value  si_value
#define nto_si_utime  si_utime
#define nto_si_status  si_status
#define nto_si_stime  si_stime
#define nto_si_fltno  si_fltno
#define nto_si_fltip  si_fltip
#define nto_si_addr  si_addr
#else
/* We can't use _NTO_HDR_ as that relies on information about the
 * target CPU and we don't know about that yet */
#include "nto-share/debug.h"
#endif

int nto_internal_debugging=0;
int nto_stop_on_thread_events=0;
bool nto_allow_mismatched_debuginfo=false;

static const registry<inferior>::key<struct nto_inferior_data>
  nto_inferior_data_reg;

enum bfd_endian nto_byte_order=BFD_ENDIAN_LITTLE;

/*
 * fetch signal info from the core
 */
static LONGEST
nto_core_xfer_siginfo (struct gdbarch *gdbarch, gdb_byte *readbuf,
       ULONGEST offset, ULONGEST len) {
  struct thread_info *ti;
  unsigned maxlen;

  gdb_assert( readbuf );
  maxlen=nto_get_siginfo_type(gdbarch)->length ();

  ti = nto_find_thread (inferior_ptid);
  if (!ti) {
    warning ("Thread with tid %ld not found.\n", inferior_ptid.lwp());
    return TARGET_XFER_EOF;
  }

  if (ti->priv.get() == NULL) {
    warning (_("%s (%p) does not have thread private data - siginfo not available\n"),
       nto_pid_to_str(inferior_ptid).c_str(), ti);
    return TARGET_XFER_EOF;
  }

  if ((offset + len) > maxlen ) {
    if (offset <= maxlen )
      len = maxlen - offset;
    else {
      len = 0;
    }
  }

  struct nto_thread_info *priv=(struct nto_thread_info *)ti->priv.get();
  memcpy (readbuf, (char *)priv->siginfo + offset, len);
  return len ? len : TARGET_XFER_E_IO;
}

/* Per-architecture data key.  */
struct nto_gdbarch_data
{
  struct type *siginfo_type;
};
static const registry<gdbarch>::key<nto_gdbarch_data> nto_gdbarch_data_data;
/* Get or initialize the nto gdbarch data.  */
static nto_gdbarch_data *
get_nto_gdbarch_data (struct gdbarch *gdbarch)
{
  nto_gdbarch_data *result = nto_gdbarch_data_data.get (gdbarch);
  if (result == nullptr)
    result = nto_gdbarch_data_data.emplace (gdbarch);
  return result;
}

struct type *
nto_get_siginfo_type (struct gdbarch *gdbarch)
{
  struct nto_gdbarch_data *nto_gdbarch_data;
  struct type *int_type, *uint_type, *long_type, *void_ptr_type;
  struct type *uid_type, *pid_type;
  struct type *sigval_type, *clock_type;
  struct type *siginfo_type, *sidata_type;
  struct type *siproc_type, *sipdata_type, *type;

  nto_gdbarch_data = get_nto_gdbarch_data (gdbarch);
  if (nto_gdbarch_data->siginfo_type != NULL)
    return nto_gdbarch_data->siginfo_type;

  int_type = arch_integer_type (gdbarch, gdbarch_int_bit (gdbarch),
				 0, "int");
  uint_type = arch_integer_type (gdbarch, gdbarch_int_bit (gdbarch),
				 1, "unsigned int");
  long_type = arch_integer_type (gdbarch, gdbarch_long_bit (gdbarch),
				 0, "long");
  void_ptr_type = lookup_pointer_type (builtin_type (gdbarch)->builtin_void);

  /* union sigval */
  sigval_type = arch_composite_type (gdbarch, NULL, TYPE_CODE_UNION);
  sigval_type->set_name ("union sigval");
  append_composite_type_field (sigval_type, "sival_int", int_type);
  append_composite_type_field_aligned (sigval_type, "sival_ptr",
				       void_ptr_type, long_type->length ());

  /* pid_t */
  pid_type = arch_type (gdbarch, TYPE_CODE_TYPEDEF,
			gdbarch_int_bit (gdbarch), "pid_t");
  pid_type->set_target_type (int_type);
  pid_type->set_is_stub (true);

  /* uid_t */
  uid_type = arch_type (gdbarch, TYPE_CODE_TYPEDEF,
			gdbarch_int_bit (gdbarch), "uid_t");
  uid_type->set_target_type (int_type);
  uid_type->set_is_stub (true);

  /* clock_t */
  clock_type = arch_type (gdbarch, TYPE_CODE_TYPEDEF,
			  gdbarch_int_bit (gdbarch), "clock_t");
  clock_type->set_target_type (uint_type);
  clock_type->set_is_stub (true);

  /* __data */
  sidata_type = arch_composite_type (gdbarch, NULL, TYPE_CODE_UNION);

  /* __pad */
  append_composite_type_field (sidata_type, "__pad",
			       init_vector_type (int_type, 7));

  /* __data.__proc */
  siproc_type = arch_composite_type (gdbarch, NULL, TYPE_CODE_STRUCT);
  append_composite_type_field (siproc_type, "__pid", pid_type);

  /* __data.__pdata */
  sipdata_type = arch_composite_type (gdbarch, NULL, TYPE_CODE_UNION);

  /* __data.__pdata.__kill */
  type = arch_composite_type (gdbarch, NULL, TYPE_CODE_STRUCT);
  append_composite_type_field (type, "__uid", uid_type);
  append_composite_type_field (type, "__value", sigval_type);
  append_composite_type_field (sipdata_type, "__kill", type);

  /* __data.__pdata.__chld */
  type = arch_composite_type (gdbarch, NULL, TYPE_CODE_STRUCT);
  append_composite_type_field (type, "__utime", clock_type);
  append_composite_type_field (type, "__status", int_type);
  append_composite_type_field (type, "__stime", clock_type);
  append_composite_type_field_aligned (sipdata_type, "__chld", type,
				       long_type->length ());
  append_composite_type_field_aligned (siproc_type, "__pdata", sipdata_type,
				       long_type->length ());
  append_composite_type_field_aligned (sidata_type, "__proc", siproc_type,
				       long_type->length ());

  /* __data.__fault */
  type = arch_composite_type (gdbarch, NULL, TYPE_CODE_STRUCT);
  append_composite_type_field (type, "__fltno", int_type);
  append_composite_type_field_aligned (type, "__fltip", void_ptr_type,
				       long_type->length ());
  append_composite_type_field_aligned (type, "__addr", void_ptr_type,
				       long_type->length ());
  append_composite_type_field_aligned (sidata_type, "__fault", type,
				       long_type->length ());

  /* struct siginfo */
  siginfo_type = arch_composite_type (gdbarch, NULL, TYPE_CODE_STRUCT);
  siginfo_type->set_name ("siginfo");
  append_composite_type_field (siginfo_type, "si_signo", int_type);
  append_composite_type_field (siginfo_type, "si_code", int_type);
  append_composite_type_field (siginfo_type, "si_errno", int_type);
  append_composite_type_field_aligned (siginfo_type,
				       "__data", sidata_type,
				       long_type->length ());

  nto_gdbarch_data->siginfo_type = siginfo_type;

  return siginfo_type;
}

const char *
nto_target (void)
{
  char *p = getenv ("QNX_TARGET");

#ifdef __CYGWIN__
  static char buf[PATH_MAX];
  if (p)
    cygwin_conv_path (CCP_WIN_A_TO_POSIX, p, buf, PATH_MAX);
  else
    cygwin_conv_path (CCP_WIN_A_TO_POSIX, gdb_sysroot.c_str (), buf, PATH_MAX);
  return buf;
#else
  return p ? p : gdb_sysroot.c_str ();
#endif
}

static Elf_Internal_Phdr *
find_load_phdr_2 (bfd *abfd, unsigned int p_filesz,
		  unsigned int p_memsz, unsigned int p_flags,
		  unsigned int p_align)
{
  Elf_Internal_Phdr *phdr;
  unsigned int i;

  nto_trace(0)("%s()\n", __func__);

  if (!abfd || !elf_tdata (abfd))
    return NULL;

  phdr = elf_tdata (abfd)->phdr;
  for (i = 0; i < elf_elfheader (abfd)->e_phnum; i++, phdr++)
    {
      if (phdr->p_type == PT_LOAD && phdr->p_flags == p_flags
	  && phdr->p_memsz == p_memsz && phdr->p_filesz == p_filesz
	  && phdr->p_align == p_align)
	return phdr;
    }

  nto_trace(0)("  No phdr found for %s\n", abfd->filename ? abfd->filename:"[no file]");
  return NULL;
}

/*
 * do not try to resolve dynamic symbols when stepping on static library
 * functions.
 */
static CORE_ADDR
nto_ldqnx2_skip_solib_resolver (struct gdbarch *gdbarch, CORE_ADDR pc)
{
  struct nto_inferior_data *const inf_data
    = nto_inferior_data (current_inferior ());

  nto_trace(0)("nto_ldqnx2_skip_solib_resolver()\n");
  // TODO: Proper cleanup of inf. data
  if (inf_data->bind_func_p == 0)
    {
      /* On Neutrino with libc 6.5.0 and later, lazy binding is performed
       * by a function called */
      struct bound_minimal_symbol resolver
          = lookup_bound_minimal_symbol ("__resolve_func");

      if (resolver.minsym && resolver.objfile)
        {
	  struct bound_minimal_symbol bind_f
              = lookup_minimal_symbol ("__bind_func", NULL, resolver.objfile);

          if (bind_f.minsym)
            {
              inf_data->bind_func_p = 1;
              inf_data->bind_func_addr = bind_f.value_address ();
	      gdb_assert (bind_f.minsym);
              inf_data->bind_func_sz = bind_f.minsym->size ();
              inf_data->resolve_func_addr = resolver.value_address ();
            }
        }
    }

  if (inf_data->bind_func_p)
    {
      if (inf_data->resolve_func_addr == pc)
        return frame_unwind_caller_pc (get_current_frame ());
    }

    return 0;
}

char **
nto_parse_redirection (char *pargv[], const char **pin, const char **pout, 
		       const char **perr)
{
  char **argv;
  const char *in, *out, *err;
  const char *p;
  int argc, i, n;

  for (n = 0; pargv[n]; n++);
  if (n == 0)
    return NULL;
  in = "";
  out = "";
  err = "";

  argv = XCNEWVEC (char *, n + 1);
  argc = n;
  for (i = 0, n = 0; n < argc; n++)
    {
      p = pargv[n];
      if (*p == '>')
	{
	  p++;
	  if (*p)
	    out = p;
	  else
	    out = pargv[++n];
	}
      else if (*p == '<')
	{
	  p++;
	  if (*p)
	    in = p;
	  else
	    in = pargv[++n];
	}
      else if (*p++ == '2' && *p++ == '>')
	{
	  if (*p == '&' && *(p + 1) == '1')
	    err = out;
	  else if (*p)
	    err = p;
	  else
	    err = pargv[++n];
	}
      else
	argv[i++] = pargv[n];
    }
  *pin = in;
  *pout = out;
  *perr = err;
  return argv;
}

struct link_map_offsets *
nto_generic_svr4_fetch_link_map_offsets (void)
{
  static struct link_map_offsets lmo32;
  static struct link_map_offsets *lmp32 = NULL;
  static struct link_map_offsets lmo64;
  static struct link_map_offsets *lmp64 = NULL;

  if (lmp32 == NULL)
    {
      lmp32 = &lmo32;

      /* r_debug structure.  */
      lmo32.r_version_offset = 0;
      lmo32.r_version_size = 4;
      lmo32.r_map_offset = 4;
      lmo32.r_brk_offset = 8;
      lmo32.r_state_offset = 12;
      lmo32.r_state_size = 4;
      lmo32.r_ldsomap_offset = 20;

      /* Link map.  */
      lmo32.link_map_size = 32;	/* The actual size is 552 bytes, but
				   this is all we need.  */
      lmo32.l_addr_offset = 0;
      lmo32.l_name_offset = 4;
      lmo32.l_ld_offset = 8;
      lmo32.l_next_offset = 12;
      lmo32.l_prev_offset = 16;
      lmo32.l_path_offset = 28;
    }
  if (lmp64 == NULL)
    {
      lmp64 = &lmo64;

      /* r_debug structure.  */
      lmo64.r_version_offset = 0;
      lmo64.r_version_size = 4;
      lmo64.r_map_offset = 8;
      lmo64.r_brk_offset = 16;
      lmo64.r_state_offset = 24;
      lmo64.r_state_size = 4;
      lmo64.r_ldsomap_offset = 32;
      /* Link map.  */
      lmo64.link_map_size = 64;
      lmo64.l_addr_offset = 0;
      lmo64.l_name_offset = 8;
      lmo64.l_ld_offset = 16;
      lmo64.l_next_offset = 24;
      lmo64.l_prev_offset = 32;
      lmo64.l_path_offset = 56;
    }
  if (IS_64BIT())
    return lmp64;
  else
    return lmp32;
}

void
nto_dummy_supply_regset (struct regcache *regcache, const gdb_byte *regs,
			 size_t len)
{
  nto_trace(0)("NTO Dummy supply regset called!\n");
  /* Do nothing.  */
}

/*
 * check if the current section is "QNX_info", this means the object is
 * QNX Neutrino.
 */
static void
nto_sniff_abi_note_section (bfd *abfd, asection *sect, void *obj)
{
  const char *sectname;
  unsigned int sectsize;
  /* Buffer holding the section contents.  */
  char *note;
  unsigned int namelen;
  const char *name;
  const unsigned sizeof_Elf_Nhdr = 12;

  sectname = bfd_section_name (sect);
  sectsize = bfd_section_size (sect);

  if (sectsize > 128)
    sectsize = 128;

  if (sectname != NULL && strstr (sectname, QNX_INFO_SECT_NAME) != NULL)
    *(enum gdb_osabi *) obj = GDB_OSABI_QNXNTO;
  else if (sectname != NULL && strstr (sectname, "note") != NULL
	   && sectsize > sizeof_Elf_Nhdr)
    {
      note = XNEWVEC (char, sectsize);
      bfd_get_section_contents (abfd, sect, note, 0, sectsize);
      namelen = (unsigned int) bfd_h_get_32 (abfd, note);
      name = note + sizeof_Elf_Nhdr;
      if (sectsize >= namelen + sizeof_Elf_Nhdr
	  && namelen == sizeof (QNX_NOTE_NAME)
	  && 0 == strcmp (name, QNX_NOTE_NAME))
	*(enum gdb_osabi *) obj = GDB_OSABI_QNXNTO;

      XDELETEVEC (note);
    }
}

/*
 * inject the OS ABI sniffer that checks sections for signs that the
 * target object is in fact QNX Neutrino
 */
enum gdb_osabi
nto_elf_osabi_sniffer (bfd *abfd)
{
  enum gdb_osabi osabi = GDB_OSABI_UNKNOWN;

  bfd_map_over_sections (abfd,
			 nto_sniff_abi_note_section,
			 &osabi);

  return osabi;
}

/* CUDA FIXME: Getting the remote thread state is 
 * broken/not implemented with cuda-gdb qnx. 
 * It looks like this may happen in nto-procfs.c
 * but we don't use that interface for remote debugging.
 * We use pdebug.
 */
static const char * const nto_thread_state_str[] =
{
  "DEAD",		/* 0  0x00 */
  "RUNNING",	/* 1  0x01 */
  "READY",	/* 2  0x02 */
  "STOPPED",	/* 3  0x03 */
  "SEND",		/* 4  0x04 */
  "RECEIVE",	/* 5  0x05 */
  "REPLY",	/* 6  0x06 */
  "STACK",	/* 7  0x07 */
  "WAITTHREAD",	/* 8  0x08 */
  "WAITPAGE",	/* 9  0x09 */
  "SIGSUSPEND",	/* 10 0x0a */
  "SIGWAITINFO",	/* 11 0x0b */
  "NANOSLEEP",	/* 12 0x0c */
  "MUTEX",	/* 13 0x0d */
  "CONDVAR",	/* 14 0x0e */
  "JOIN",		/* 15 0x0f */
  "INTR",		/* 16 0x10 */
  "SEM",		/* 17 0x11 */
  "WAITCTX",	/* 18 0x12 */
  "NET_SEND",	/* 19 0x13 */
  "NET_REPLY"	/* 20 0x14 */
};

/*
 * returns the state of the thread as a string
 * This is not a getter for the nto_thread_info extension!
 */
const char *
nto_extra_thread_info ( struct thread_info *ti )
{
  struct nto_thread_info *priv;
  if ( ( ti != NULL ) && ( ti->priv.get() != NULL) ) {
    priv=(struct nto_thread_info *)ti->priv.get();
    if( priv->state < ARRAY_SIZE (nto_thread_state_str) ) {
      return (char *)nto_thread_state_str [priv->state];
    }
  }
  return "";
}

/*
 * make sure the signals used by the OS are never blocked
 */
void
nto_initialize_signals (void)
{
  int sig;
  for ( sig = GDB_SIGNAL_SELECT; sig <= GDB_SIGNAL_PROCNTO_64; sig++ )
    {
      signal_stop_update  (sig, 0);
      signal_print_update (sig, 0);
      signal_pass_update  (sig, 1);
    }
}

static void
show_nto_debug (struct ui_file *file, int from_tty,
                struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("QNX NTO debug level is %d.\n"), nto_internal_debugging);
}

static int
nto_print_tidinfo_callback (struct thread_info *tp, void *data)
{
  char star = ' ';
  int tid = 0;
  int state = 0;
  int flags = 0;
  struct nto_thread_info *priv;

  if (tp)
    {
      if ( tp->ptid == inferior_ptid )
        star = '*';

      if (tp->priv.get())
        {
          priv=(struct nto_thread_info *)tp->priv.get();
          state = priv->state;
          flags = priv->flags;
        }

      tid = tp->ptid.lwp();

      gdb_printf ("%c%d\t%d\t%d\n", star, tid, state, flags);
    }

  return 0;
}

static void
nto_info_tidinfo_command (const char *args, int from_tty)
{
  const char *execfile = get_exec_file (0);
  nto_trace (0) ("%s (args=%s, from_tty=%d)\n", __func__,
		  args ? args : "(null)", from_tty);

  target_update_thread_list ();
  gdb_printf("Threads for pid %d (%s)\nTid:\tState:\tFlags:\n",
		  inferior_ptid.pid (), execfile ? execfile : "");

  iterate_over_threads (nto_print_tidinfo_callback, NULL);
}

std::string
nto_pid_to_str (ptid_t ptid)
{
  if (ptid.lwp () == 0)
    return string_printf ("pid %d", ptid.pid ());

  return string_printf ("pid %d tid %ld", ptid.pid (), ptid.lwp ());
}

std::string
nto_core_pid_to_str (struct gdbarch *gdbarch, ptid_t ptid)
{
  return nto_pid_to_str (ptid);
}

void
nto_get_siginfo_from_procfs_status (const void *const ps, void *siginfo)
{
  nto_siginfo_t *dst = (nto_siginfo_t *)siginfo;
  const nto_procfs_status * const status = (nto_procfs_status *)ps;
  const nto_siginfo_t *src = (nto_siginfo_t *)(&status->info);

  struct type *ptr_t = builtin_type (target_gdbarch ())->builtin_data_ptr;

  nto_trace(1)("nto_get_siginfo_from_procfs_status()\n");

  memset (dst, 0, sizeof (nto_siginfo_t));

  dst->si_signo = extract_signed_integer((gdb_byte *) &src->si_signo,
      sizeof(src->si_signo), nto_byte_order);
  dst->si_code = extract_signed_integer((gdb_byte *) &src->si_code,
      sizeof(src->si_code), nto_byte_order);
  if (dst->si_code == SI_NOINFO)
    return;

  dst->si_errno = extract_signed_integer((gdb_byte *) &src->si_errno,
      sizeof(src->si_errno), nto_byte_order);

  if (dst->si_code <= 0) // SI_FROMUSER
    {
      dst->nto_si_pid = extract_signed_integer((gdb_byte *) &src->nto_si_pid,
	  sizeof(src->nto_si_pid), nto_byte_order);
      dst->nto_si_uid = extract_signed_integer((gdb_byte *) &src->nto_si_uid,
	  sizeof(src->nto_si_uid), nto_byte_order);
      dst->nto_si_value.sival_ptr = (void *) extract_typed_address(
	  (gdb_byte *) &src->nto_si_value, ptr_t);
    }
  else if (dst->si_signo
      == gdbarch_gdb_signal_to_target(target_gdbarch(), GDB_SIGNAL_CHLD))
    {
      /* child termination */
      dst->nto_si_pid = extract_signed_integer((gdb_byte *) &src->nto_si_pid,
	  sizeof(src->nto_si_pid), nto_byte_order);
      dst->nto_si_utime = extract_unsigned_integer(
	  (gdb_byte *) &src->nto_si_utime, sizeof(src->nto_si_utime),
	  nto_byte_order);
      dst->nto_si_status = extract_signed_integer(
	  (gdb_byte *) &src->nto_si_status, sizeof(src->nto_si_status),
	  nto_byte_order);
      dst->nto_si_stime = extract_unsigned_integer(
	  (gdb_byte *) &src->nto_si_stime, sizeof(src->nto_si_stime),
	  nto_byte_order);
    }
  else
    {
      dst->nto_si_fltno = extract_signed_integer((gdb_byte *) &src->nto_si_fltno,
	  sizeof(src->nto_si_fltno), nto_byte_order);
      dst->nto_si_fltip = (void *)extract_typed_address((gdb_byte *) &src->nto_si_fltip,
	  ptr_t);
      dst->nto_si_addr = (void *)extract_typed_address((gdb_byte *) &src->nto_si_addr,
	  ptr_t);
    }
}

/* Read AUXV from initial_stack.  */
LONGEST
nto_read_auxv_from_initial_stack (CORE_ADDR initial_stack, gdb_byte *readbuf,
				  LONGEST len, size_t sizeof_auxv_t)
{
  gdb_byte targ32[4]; /* For 32 bit target values.  */
  gdb_byte targ64[8]; /* For 64 bit target values.  */
  CORE_ADDR data_ofs = 0;
  ULONGEST anint;
  LONGEST len_read = 0;
  gdb_byte *buff;
  int ptr_size;

  nto_trace(1) ("%s ()\n", __func__);

  if (sizeof_auxv_t == 16)
    ptr_size = 8;
  else
    ptr_size = 4;

  /* Skip over argc, argv and envp... Comment from ldd.c:

   The startup frame is set-up so that we have:
   auxv
   NULL
   ...
   envp2
   envp1 <----- void *frame + (argc + 2) * sizeof(char *)
   NULL
   ...
   argv2
   argv1
   argc  <------ void * frame

   On entry to ldd, frame gives the address of argc on the stack.  */
  /* Read argc. 4 bytes on both 64 and 32 bit arches and luckily little
   * endian. So we just read first 4 bytes.  */
  if (target_read_memory (initial_stack + data_ofs, targ32, 4) != 0)
    return 0;

  anint = extract_unsigned_integer (targ32, sizeof(targ32), nto_byte_order);

  /* Size of pointer is assumed to be 4 bytes (32 bit arch.) */
  data_ofs += (anint + 2) * ptr_size; /* + 2 comes from argc itself and
   NULL terminating pointer in
   argv.  */

  /* Now loop over env table:  */
  anint = 0;
  while (target_read_memory (initial_stack + data_ofs, targ64, ptr_size) == 0)
    {
      if (extract_unsigned_integer (targ64, ptr_size, nto_byte_order) == 0)
	anint = 1; /* Keep looping until non-null entry is found.  */
      else if (anint)
	break;
      data_ofs += ptr_size;
    }
  initial_stack += data_ofs;

  memset (readbuf, 0, len);
  buff = readbuf;
  while (len_read <= len - sizeof_auxv_t)
    {
      if (target_read_memory (initial_stack + len_read, buff, sizeof_auxv_t)
	  == 0)
	{
	  /* Both 32 and 64 bit structures have int as the first field.  */
	  const ULONGEST a_type = extract_unsigned_integer (buff,
							    sizeof(targ32),
							    nto_byte_order);
	  if (a_type == AT_NULL)
	    break;
	  buff += sizeof_auxv_t;
	  len_read += sizeof_auxv_t;
	}
      else
	break;
    }
  return len_read;
}

bool
nto_stopped_by_watchpoint ( )
{
  /* NOTE: nto_stopped_by_watchpoint will be called ONLY while we are
     stopped due to a SIGTRAP.  This assumes gdb works in 'all-stop' mode;
     future gdb versions will likely run in 'non-stop' mode in which case
     we will have to store/examine statuses per thread in question.
     Until then, this will work fine.  */

  struct inferior *inf = current_inferior ();
  struct nto_inferior_data *inf_data;

  gdb_assert (inf != NULL);

  inf_data = nto_inferior_data (inf);

  return inf_data->stopped_flags
   & (_DEBUG_FLAG_TRACE_RD
      | _DEBUG_FLAG_TRACE_WR
      | _DEBUG_FLAG_TRACE_MODIFY);
}

static void
nto_solib_loaded_listener (struct so_list *solib)
{
  /* Check if the libraries match.
     We compare all PT_LOAD segments.  */
  CORE_ADDR mem_phdr_addr;
  CORE_ADDR phdr_offs_addr;
  gdb_byte offs_buf[8];

  unsigned int offsetof_e_phoff;
  unsigned int sizeof_e_phoff;
  unsigned int sizeof_Elf_Phdr;
  const unsigned int offsetof_p_type = 0;
  unsigned int offsetof_p_filesz;
  unsigned int offsetof_p_memsz;
  unsigned int offsetof_p_flags;
  unsigned int offsetof_p_align;
  const unsigned int sizeof_p_type = 4;
  const unsigned sizeof_p_flags = 4;
  unsigned int sizeof_p_filesz;
  unsigned int sizeof_p_memsz;
  unsigned int sizeof_p_align;
  nto_trace (0) ("nto_solib_loaded_listener(%s)\n", solib->so_name[0]?solib->so_name:"<unnamed>");

  switch (gdbarch_bfd_arch_info (target_gdbarch ())->bits_per_word)
    {
    case 64:
      offsetof_e_phoff = 32;
      sizeof_e_phoff = 8;
      sizeof_Elf_Phdr = 56;
      offsetof_p_filesz = 32;
      offsetof_p_memsz = 40;
      offsetof_p_flags = 4;
      offsetof_p_align = 48;
      sizeof_p_filesz = 8;
      sizeof_p_memsz = 8;
      sizeof_p_align = 8;
      break;
    case 32:
      offsetof_e_phoff = 28;
      sizeof_e_phoff = 4;
      sizeof_Elf_Phdr = 32;
      offsetof_p_filesz = 16;
      offsetof_p_memsz = 20;
      offsetof_p_flags = 24;
      offsetof_p_align = 28;
      sizeof_p_filesz = sizeof_p_memsz = sizeof_p_align = 4;
      break;
    default:
      gdb_assert (!"Unsupported number of bits per word");
    }

  phdr_offs_addr = solib->addr_low + offsetof_e_phoff;

  if (target_read_memory (phdr_offs_addr, offs_buf, sizeof_e_phoff))
    {
      nto_trace (0) ("Could not read memory.\n");
      return;
    }

  mem_phdr_addr =
      solib->addr_low
      + extract_typed_address (offs_buf,
             builtin_type (target_gdbarch ())->builtin_data_ptr);

  while (1)
    {
      gdb_byte phdr_buf[sizeof_Elf_Phdr];
      /* We compare phdr fields: p_type, p_flags, p_aign, p_filesz, p_memsz */
      unsigned int p_type;
      unsigned int p_filesz;
      unsigned int p_memsz;
      unsigned int p_flags;
      unsigned int p_align;
      Elf_Internal_Phdr *file_phdr;

      if (target_read_memory (mem_phdr_addr, phdr_buf, sizeof_Elf_Phdr))
        {
          nto_trace (0) ("Could not read phdr at 0x%lx\n", mem_phdr_addr);
          return;
        }

      p_type = extract_unsigned_integer (&phdr_buf[offsetof_p_type],
           sizeof_p_type, nto_byte_order);
      if (p_type == PT_LOAD)
        {
          p_filesz = extract_unsigned_integer (&phdr_buf[offsetof_p_filesz],
                sizeof_p_filesz, nto_byte_order);
          p_memsz = extract_unsigned_integer (&phdr_buf[offsetof_p_memsz],
                sizeof_p_memsz, nto_byte_order);
          p_flags = extract_unsigned_integer (&phdr_buf[offsetof_p_flags],
                sizeof_p_flags, nto_byte_order);
          p_align = extract_unsigned_integer (&phdr_buf[offsetof_p_align],
                sizeof_p_align, nto_byte_order);

          if (solib->symbols_loaded)
            {
              file_phdr = find_load_phdr_2 (solib->abfd,
                  p_filesz, p_memsz, p_flags, p_align);
              if (file_phdr == NULL)
                {
                  /* This warning is being parsed by the IDE, the
                   * format should not change without consultations with
                   * IDE team.  */
                  warning ("Host file %s does not match target file %s.",
                      solib->so_name, solib->so_original_name);
                  break;
                }
            }
        }

      if (p_type == PT_NULL)
        break;

      mem_phdr_addr += sizeof_Elf_Phdr;
    }
}

/* Return nto_inferior_data for the given INFERIOR.  If not yet created,
   construct it.  */

struct nto_inferior_data *
nto_inferior_data (struct inferior *const inferior)
{
  struct inferior *const inf = inferior ? inferior : current_inferior ();
  struct nto_inferior_data *inf_data;

  gdb_assert (inf != NULL);

  inf_data = nto_inferior_data_reg.get (inf);
  if (inf_data == NULL)
    inf_data = nto_inferior_data_reg.emplace (inf);

  return inf_data;
}

/* Per-architecture data key.  */
static const registry<gdbarch>::key<nto_target_ops> nto_gdbarch_ops_data;
/* Get or initialize the nto target ops.  */
nto_target_ops *
get_nto_target_ops (struct gdbarch *gdbarch)
{
  nto_target_ops *result = nto_gdbarch_ops_data.get (gdbarch);
  if (result == nullptr)
    result = nto_gdbarch_ops_data.emplace (gdbarch);
  return result;
}

/*
 * translates a GDB signal code into an NTO signal.
 * Signal mappings differ from OS to OS and sometimes even between
 * architectures, so GDB has it's own signal list that needs
 * to be translated to the actual target and back.
 */
int
nto_gdb_signal_to_target (struct gdbarch *gdbarch, enum gdb_signal signal)
{
  switch (signal)
    {
    /* signals that can be sent with slay */
    case GDB_SIGNAL_0:
      return 0;
    case GDB_SIGNAL_HUP:
      return 1; /* SIGHUP */
    case GDB_SIGNAL_INT:
      return 2;   /* interrupt */
    case GDB_SIGNAL_QUIT:
      return 3;   /* quit */
    case GDB_SIGNAL_ILL:
      return 4;   /* illegal instruction (not reset when caught) */
    case GDB_SIGNAL_TRAP:
      return 5;   /* trace trap (not reset when caught) */
    case GDB_SIGNAL_ABRT:
      return 6;   /* used by abort */
    case GDB_SIGNAL_EMT:
      return 7;   /* EMT instruction */
    case GDB_SIGNAL_FPE:
      return 8;   /* floating point exception */
    case GDB_SIGNAL_KILL:
      return 9;   /* kill (cannot be caught or ignored) */
    case GDB_SIGNAL_BUS:
      return 10;  /* bus error */
    case GDB_SIGNAL_SEGV:
      return 11;  /* segmentation violation */
    case GDB_SIGNAL_SYS:
      return 12;  /* bad argument to system call */
    case GDB_SIGNAL_PIPE:
      return 13;  /* write on pipe with no reader */
    case GDB_SIGNAL_ALRM:
      return 14;  /* real-time alarm clock */
    case GDB_SIGNAL_TERM:
      return 15;  /* software termination signal from kill */
    case GDB_SIGNAL_USR1:
      return 16;  /* user defined signal 1 */
    case GDB_SIGNAL_USR2:
      return 17;  /* user defined signal 2 */
    case GDB_SIGNAL_CHLD:
      return 18;  /* death of child */
    case GDB_SIGNAL_PWR:
      return 19;  /* power-fail restart */
    case GDB_SIGNAL_WINCH:
      return 20;  /* window change */
    case GDB_SIGNAL_URG:
      return 21;  /* urgent condition on I/O channel */
    case GDB_SIGNAL_POLL:
    case GDB_SIGNAL_IO:
      return 22;  /* System V name for SIGIO */
    case GDB_SIGNAL_STOP:
      return 23;  /* sendable stop signal not from tty */
    case GDB_SIGNAL_TSTP:
      return 24;  /* stop signal from tty */
    case GDB_SIGNAL_CONT:
      return 25;  /* continue a stopped process */
    case GDB_SIGNAL_TTIN:
      return 26;  /* attempted background tty read */
    case GDB_SIGNAL_TTOU:
      return 27;  /* attempted background tty write */
    case GDB_SIGNAL_VTALRM:
      return 28;  /* virtual timer expired */
    case GDB_SIGNAL_PROF:
      return 29;  /* profiling timer expired */
    case GDB_SIGNAL_XCPU:
      return 30;  /* exceded cpu limit */
    case GDB_SIGNAL_XFSZ:
      return 31;  /* exceded file size limit */
    case GDB_SIGNAL_DOOM:
      return 32;
      /* NTO does not define signals between 33 and 40 *
       * filled up with non-posix RT signals           */
    case GDB_SIGNAL_REALTIME_33:
      return 33;
    case GDB_SIGNAL_REALTIME_34:
    return 34;
    case GDB_SIGNAL_REALTIME_35:
    return 35;
    case GDB_SIGNAL_REALTIME_36:
    return 36;
    case GDB_SIGNAL_REALTIME_37:
    return 37;
    case GDB_SIGNAL_REALTIME_38:
    return 38;
    case GDB_SIGNAL_REALTIME_39:
    return 39;
    case GDB_SIGNAL_REALTIME_40:
      return 40;
    /* POSIX RT-signals ->  */
    case GDB_SIGNAL_REALTIME_41:
      return 41;
    case GDB_SIGNAL_REALTIME_42:
      return 42;
    case GDB_SIGNAL_REALTIME_43:
      return 43;
    case GDB_SIGNAL_REALTIME_44:
      return 44;
    case GDB_SIGNAL_REALTIME_45:
      return 45;
    case GDB_SIGNAL_REALTIME_46:
      return 46;
    case GDB_SIGNAL_REALTIME_47:
      return 47;
    case GDB_SIGNAL_REALTIME_48:
      return 48;
    case GDB_SIGNAL_REALTIME_49:
      return 49;
    case GDB_SIGNAL_REALTIME_50:
      return 50;
    case GDB_SIGNAL_REALTIME_51:
      return 51;
    case GDB_SIGNAL_REALTIME_52:
      return 52;
    case GDB_SIGNAL_REALTIME_53:
      return 53;
    case GDB_SIGNAL_REALTIME_54:
      return 54;
    case GDB_SIGNAL_REALTIME_55:
      return 55;
    case GDB_SIGNAL_REALTIME_56:
      return 56;
    /* Special purpose signals reserved by the Kernel! ->  */
    case GDB_SIGNAL_SELECT:
      return 57;
    case GDB_SIGNAL_PHOTON:
      return 58;
    case GDB_SIGNAL_PROCNTO_59:
      return 59;
    case GDB_SIGNAL_PROCNTO_60:
      return 60;
    case GDB_SIGNAL_PROCNTO_61:
      return 61;
    case GDB_SIGNAL_PROCNTO_62:
      return 62;
    case GDB_SIGNAL_PROCNTO_63:
      return 63;
    case GDB_SIGNAL_PROCNTO_64:
      return 64;
    default:
      return 0;
    }
}

/*
 * reverse function to nto_gdb_signal_to_target above
 */
enum gdb_signal
nto_gdb_signal_from_target (struct gdbarch *gdbarch, int nto_signal)
{
  int i;
  for (i = GDB_SIGNAL_0; i < GDB_SIGNAL_LAST; ++i)
    {
      int tgtsig = nto_gdb_signal_to_target (gdbarch, (enum gdb_signal)i);
      if (tgtsig == nto_signal)
        return (enum gdb_signal)i;
    }
  /* Give debug message as GDB_SIGNAL_UNKNOWN (143) can be confusing */
  nto_trace (0) ("Unknown NTO signal %i, stay within [1..64]\n", nto_signal );
  return GDB_SIGNAL_UNKNOWN;
}

/*
 * helper to find a thread by lwp
 */
struct thread_info *
nto_find_thread (ptid_t ptid)
{
  struct thread_info *res=NULL;

  for (thread_info *tp : current_inferior ()->threads ())
    {
      if ( tp->ptid.pid() == ptid.pid() )
        {
          if ( tp->ptid.lwp() == ptid.lwp() )
            {
              res=tp;
              break;
            }
        }
    }
  return res;
}

/*
 * When a new thread is found in a core file, this hook gets called to tweak
 * the information and make GDB realize it is in fact a thread and not a
 * process..
 *
 * first we need to get the section for the thread: .qnx_core_status/<tid>
 * from there we can get all the states and extended information
 * Registers follow the default .reg/<tid> notation, so no extra work here
 */
static void
nto_new_thread_listener( thread_info *ti ) {
  /* no assert as this will also be triggered on remote and
   * self-hosted environments */
  if (!core_bfd) return;

  char sectname[64];
  struct nto_thread_info *priv=NULL;
  unsigned int sectsize=0;
  nto_procfs_status ps;

  nto_trace(1)( "nto_new_thread_listener for gdb_tid_%d - %d:%ld\n",
      ti->global_num, ti->ptid.pid(), ti->ptid.lwp() );

  /* fetch the proper segment from the core */
  xsnprintf( sectname, 63, ".qnx_core_status/%ld", ti->ptid.lwp() );
  asection *tsect=bfd_get_section_by_name( core_bfd, sectname );

  if( !tsect ) {
    warning("Could not find section %s in core!", sectname );
    return;
  }

  sectsize = bfd_section_size (tsect);

  if (sectsize > sizeof (ps) ) sectsize = sizeof (ps);

  if (bfd_seek ( core_bfd, tsect->filepos, SEEK_SET) != 0) {
    warning ( "Cannot read %s from core!", sectname);
    return;
  }

  if( bfd_bread ((gdb_byte *)&ps, sectsize, core_bfd ) != sectsize ) {
    warning ( "could not read %d bytes from section %s!", sectsize, sectname);
    return;
  }

  /* we always create a new one as this should never be called on an
   * already existing thread but even if, the reset() frees any
   * existing information */
  priv=new nto_thread_info();
  ti->priv.reset(priv);

  priv->state = extract_unsigned_integer ((gdb_byte *)&ps.state,
        sizeof (ps.state), nto_byte_order);
  priv->flags = extract_unsigned_integer ((gdb_byte *)&ps.flags,
        sizeof (ps.flags), nto_byte_order);
  priv->siginfo = malloc (sizeof (ps.info));

  if (priv->siginfo == NULL) {
    warning ("Out of memory.\n");
    return;
  }

  /* fill siginfo as well.. */
  nto_get_siginfo_from_procfs_status (&ps, priv->siginfo);
}

/*
 * since GDB expects the AUXV to be in the .auxv section while QNX puts it into
 * .qnx_core_info section we need to supply this information manually after
 * the inferior is created.
 *
 * TODO: this /should/ be emulated in BFD!
 *
 * What happens is that auxv_inferior_data() is called to fill the auxv buffer
 * for each identified thread. That one calls the
 * xfer_partial( TARGET_OBJECT_AUXV ) if the auxv for the inferior has not yet
 * been buffered and this fails for QNX generated cores. However if we fill
 * the buffer manually before, then the  original call shall happily return
 * with the expected auxv data.
 *
 * See also solib-nto.c for the special handling of shared libraries in a
 * core file.
 *
 * Calling this as late as possible to avoid the auxv getting cleaned up.
 */
static void
nto_inferior_appeared_listener( inferior *inf ) {
  /* no assert as this will also be triggered on remote and self-hosted
   * environments */
  if( !core_bfd ) {
      return;
  }
  nto_trace (0) ("%s ()\n", __func__);
#ifdef NVIDIA_CUDA_GDB
  /* CUDA: Changed this slightly from upstream QNX to provide better encapsulation. */
  gdb::optional<gdb::byte_vector>& auxv_data = qnx_get_auxv_data (inf);
#else
  const struct inferior_key<auxv_info> auxv_inferior_data = get_auxv_inferior_data ();
  struct auxv_info *info = auxv_inferior_data.emplace (inf);
#endif
  nto_procfs_info pinfo;
  int len;

  /* buffer can hold 20 auxv_t entries - taken from nto-procfs.c */
  const size_t sizeof_tempbuf = 20 * ( IS_64BIT()? 16 : 8 );
  gdb_byte tempbuf[sizeof_tempbuf];
  const char *sectname=".qnx_core_info";
  unsigned int sectsize;
  CORE_ADDR initial_stack;

  /* fetch the proper segment from the core */
  asection *tsect=bfd_get_section_by_name( core_bfd, sectname );

  if( !tsect ) {
    warning("Could not find section %s in core!", sectname );
    return;
  }

  sectsize = bfd_section_size (tsect);

  if (sectsize > sizeof (pinfo))
    sectsize = sizeof (pinfo);

  if (bfd_seek (core_bfd, tsect->filepos, SEEK_SET) != 0) {
    warning ("Could not jump to procinfo in %s!", sectname );
    return;
  }

  len = bfd_bread ((gdb_byte *)&pinfo, sectsize, core_bfd);
  if (len != sectsize) {
    warning ("Could not read procinfo from %s!", sectname );
    return;
  }

  initial_stack = extract_unsigned_integer ((gdb_byte *)&pinfo.initial_stack,
      sizeof (pinfo.initial_stack), nto_byte_order);

  nto_trace(0)("Initial stack at 0x%lx\n",initial_stack);

  len = nto_read_auxv_from_initial_stack
    (initial_stack, tempbuf, sizeof_tempbuf, IS_64BIT()? 16 : 8);
  gdb::byte_vector databuf(len);
  memcpy( databuf.data(), tempbuf, len );

#ifdef NVIDIA_CUDA_GDB
  auxv_data = databuf;
#else
  info->data=databuf;
#endif

  // Make sure the symbol relocations match the corefile
  if (current_program_space->exec_bfd () != NULL) {
      // if a file was given, forget everything about it
      // symbols will get loaded implicitly later in the corefile handling
      symbol_file_clear(0);
  }
  else {
      // no file was given, load all the symbols explicitly
      solib_add (NULL, 0, auto_solib_add);
  }
}

/* Add PID to GDB's inferior table */
inferior *
nto_add_inferior ( process_stratum_target *target, int pid )
{
  struct inferior *inf;

  if (gdbarch_has_global_solist (target_gdbarch ()))
    {
      /* If the target shares code across all inferiors, then every
	 attach adds a new inferior.  */
      inf = add_inferior (pid);

      /* ... and every inferior is bound to the same program space.
	 However, each inferior may still have its own address
	 space.  */
      inf->aspace = maybe_new_address_space ();
      inf->pspace = current_program_space;
    }
  else
    {
      /* In the traditional debugging scenario, there's a 1-1 match
	 between program/address spaces.  We simply bind the inferior
	 to the program space's address space.  */
      inf = current_inferior ();

      /* However, if the current inferior is already bound to a
	 process, find some other empty inferior.  */
      if (inf->pid != 0)
	{
	  inf = nullptr;
	  for (inferior *it : all_inferiors ())
	    if (it->pid == 0)
	      {
		inf = it;
		break;
	      }
	}
      if (inf == nullptr)
	{
	  /* Since all inferiors were already bound to a process, add
	     a new inferior.  */
	  inf = add_inferior_with_spaces ();
	}
      switch_to_inferior_no_thread (inf);
      inf->push_target (target);
      inferior_appeared (inf, pid);
    }

  return inf;
}

/*
 * generic NTO ABI function that do not depend on a specific architecture
 */
void
nto_init_abi (struct gdbarch_info info, struct gdbarch *gdbarch)
{
  nto_trace (0) ("%s ()\n", __func__);
  /* all threads are found and inferior is announced, now get the AUXV */
  gdb::observers::inferior_appeared.attach (nto_inferior_appeared_listener, "nto-tdep");

  /* corelow found a new thread, add extra thread info */
  gdb::observers::new_thread.attach(nto_new_thread_listener, "nto-tdep");

  set_gdbarch_core_pid_to_str (gdbarch, nto_core_pid_to_str);
  set_gdbarch_core_xfer_siginfo (gdbarch, nto_core_xfer_siginfo);
  set_gdbarch_gdb_signal_from_target (gdbarch, nto_gdb_signal_from_target);
  set_gdbarch_gdb_signal_to_target (gdbarch, nto_gdb_signal_to_target);
  set_gdbarch_get_siginfo_type (gdbarch, nto_get_siginfo_type);
  set_gdbarch_skip_solib_resolver( gdbarch, nto_ldqnx2_skip_solib_resolver);
}

void _initialize_nto_tdep ();
void
_initialize_nto_tdep ()
{
  nto_trace (0) ("%s ()\n", __func__);

  /* this not needed on self hosted platforms */
#ifndef __QNXNTO__
  if( getenv("QNX_TARGET") == NULL ) {
    error("QNX environment is not set!");
  }
#endif

  add_setshow_zinteger_cmd ("nto-debug", class_maintenance,
                            &nto_internal_debugging, _("\
Set QNX NTO internal debugging."), _("\
Show QNX NTO internal debugging."), _("\
When non-zero, nto specific debug info is\n\
displayed. Different information is displayed\n\
for different positive values."),
                            NULL,
                            &show_nto_debug, /* FIXME: i18n: QNX NTO internal 
                                     debugging is %s.  */
                            &setdebuglist, &showdebuglist);

  add_setshow_zinteger_cmd ("nto-stop-on-thread-events", class_support,
                            &nto_stop_on_thread_events, _("\
Stop on thread events ."), _("\
Show stop on thread events setting."), _("\
When set to 1, stop on thread created and thread destroyed events.\n"),
                            NULL,
                            NULL,
                            &setlist, &showlist);

  add_setshow_boolean_cmd ("nto-allow-mismatched-debuginfo", class_support,
			   &nto_allow_mismatched_debuginfo, _ ("\
Set the flag to allow mismatching build-ids in symbol files."),
			   _ ("\
Show nto-allow-mismatch-debuginfo flag."),
			   _ ("\
If set, GDB will use symbol files for executables even if the build-id does \
not match."),
			   NULL, NULL, &setlist, &showlist);

  add_info ("tidinfo", nto_info_tidinfo_command, "List threads for current process." );

  gdb::observers::solib_loaded.attach (nto_solib_loaded_listener, "nto-tdep");
}
