/*
 * $QNXtpLicenseC:
 * Copyright 2005-2021 QNX Software Systems. All Rights Reserved.
 *
 * This source code may contain confidential information of QNX Software
 * Systems (QSS) and its licensors.  Any use, reproduction, modification,
 * disclosure, distribution or transfer of this software, or any software
 * that includes or is based upon any of this code, is prohibited unless
 * expressly authorized by QSS by written agreement.  For more information
 * (including whether this source code file has been published) please
 * email licensing@qnx.com. $
*/

/*

   This file was derived from remote.c. It communicates with a
   target talking the Neutrino remote debug protocol.
   See nto-share/dsmsgs.h for details.

   Copyright 2003 Free Software Foundation, Inc.

   Contributed by QNX Software Systems Ltd.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

/* NVIDIA CUDA Debugger CUDA-GDB
   Copyright (C) 2017-2024 NVIDIA Corporation
   Modified from the original GDB file referenced above by the CUDA-GDB
   team at NVIDIA <cudatools@nvidia.com>. */

#include "defs.h"
#include "exceptions.h"
#include <fcntl.h>
#include <signal.h>

#include <string.h>
#include "terminal.h"
#include "inferior.h"
#include "target.h"
#include "target-descriptions.h"
#include "gdbcmd.h"
#include "objfiles.h"
#include "gdbthread.h"
#include "completer.h"
#include "cli/cli-decode.h"
#include "regcache.h"
#include "gdbcore.h"
#include "serial.h"
#include "readline/readline.h"
#include "infrun.h"
#include "remote.h"
#include "elf-bfd.h"
#include "elf/common.h"

#ifdef NVIDIA_CUDA_GDB
#include "cuda/cuda-linux-nat-template.h"
#include "remote-nto.h"
#endif

#ifdef NVIDIA_BUGFIX
#include <queue>
#endif

#include "environ.h"

#include <time.h>

#include "nto-share/dsmsgs.h"
#include "nto-tdep.h"

#ifndef __MINGW32__
#include <termios.h>
#else
#define SIGKILL 9
#endif

#include "source.h"

#ifdef __QNXNTO__

#include <sys/debug.h>
typedef debug_thread_t nto_procfs_status;
typedef debug_process_t nto_procfs_info;
typedef siginfo_t nto_siginfo_t;
#else
#include "nto-share/debug.h"
#endif

#include "solib.h"

#ifdef __CYGWIN__
#include <sys/cygwin.h>
#endif

#ifndef EOK
#define EOK 0
#endif

#ifndef O_BINARY
#define O_BINARY 0
#endif

#define QNX_READ_MODE  0x0
#define QNX_WRITE_MODE  0x301
#define QNX_WRITE_PERMS  0x1ff

/* The following define does a cast to const gdb_byte * type.  */

#define EXTRACT_SIGNED_INTEGER(ptr, len, byte_order) \
  extract_signed_integer ((const gdb_byte *)(ptr), len, byte_order)
#define EXTRACT_UNSIGNED_INTEGER(ptr, len, byte_order) \
  extract_unsigned_integer ((const gdb_byte *)(ptr), len, byte_order)


typedef union
{
  unsigned char buf[DS_DATA_MAX_SIZE];
  DSMsg_union_t pkt;
  TSMsg_text_t text;
} DScomm_t;


#ifdef __MINGW32__
/* Name collision with a symbol declared in Winsock2.h.  */
#define recv recvb
#endif
#ifdef NVIDIA_CUDA_GDB
/* This variable was internal to nto_send_init. We need it in the outer scope
   to be able to set it to the value we need, since we have two hosts connecting
   to the pdebug target - cuda-gdb and cuda-gdbserver and `mid` must be kept
   in sync between them. */
static unsigned char mid;
#endif

static bool nto_force_hwbp = false;
/* names of the expected Kernel notifications for readable trace output */
static const char* const _DSMSGS[12] {
  "DSMSG_NOTIFY_PIDLOAD",    /* 0 */
  "DSMSG_NOTIFY_TIDLOAD",    /* 1 */
  "DSMSG_NOTIFY_DLLLOAD",    /* 2 */
  "DSMSG_NOTIFY_PIDUNLOAD",  /* 3 */
  "DSMSG_NOTIFY_TIDUNLOAD",  /* 4 */
  "DSMSG_NOTIFY_DLLUNLOAD",  /* 5 */
  "DSMSG_NOTIFY_BRK",        /* 6 */
  "DSMSG_NOTIFY_STEP",       /* 7 */
  "DSMSG_NOTIFY_SIGEV",      /* 8 */
  "DSMSG_NOTIFY_STOPPED",    /* 9 */
  "DSMSG_NOTIFY_FORK",       /* 10 */
  "DSMSG_NOTIFY_EXEC"        /* 11 */
};

static const target_info pdebug_target_info = {
  "qnx",
  N_("Remote serial target in pdebug-specific protocol"),
  N_("Debug a remote machine using the legacy QNX Debugging Protocol.\n\
  Specify the device it is connected to (e.g. /dev/ser1, <rmt_host>:<port>)\n\
  or `pty' to launch `pdebug' for debugging.")
};

#ifdef NVIDIA_CUDA_GDB
struct nto_remote_inferior_data
{
  nto_remote_inferior_data() : remote_exe(), remote_cwd(), auxv() {}
  ~nto_remote_inferior_data ()
  {
    if (auxv) xfree(auxv);
    auxv = nullptr;
    remote_exe.clear ();
    remote_cwd.clear ();
  }
  /* File to be executed on remote.  */
  std::string remote_exe;

  /* Current working directory on remote.  */
  std::string remote_cwd;

  /* Cached auxiliary vector */
  gdb_byte *auxv;
};
#else
struct nto_remote_inferior_data
{
  /* File to be executed on remote.  */
  char *remote_exe;

  /* Current working directory on remote.  */
  char *remote_cwd;

  /* Cached auxiliary vector */
  gdb_byte *auxv;
};
#endif

/*
 * todo it may make sense to put the session data in here too..
 */

class pdebug_target : public process_stratum_target
{
public:

  pdebug_target () = default;
  ~pdebug_target () override;

  const target_info &info () const override
  { return pdebug_target_info; }

  /* Open a remote connection.  */
  static void pdebug_open (const char *, int);
  void close () override;

  bool can_attach () override {
    return true;
  }
  void attach ( const char*, int ) override;
  void post_attach( int ) override;
  void detach (inferior *, int) override;
  void disconnect (const char *, int) override;
  void resume (ptid_t, int TARGET_DEBUG_PRINTER (target_debug_print_step), enum gdb_signal) override;
  ptid_t wait (ptid_t, struct target_waitstatus *, target_wait_flags) override;
  void fetch_registers (struct regcache *, int) override;
  void store_registers (struct regcache *, int) override;
  void prepare_to_store (struct regcache *) override;
  void files_info () override;
  int insert_breakpoint (struct gdbarch *, struct bp_target_info *) override;
  int remove_breakpoint (struct gdbarch *, struct bp_target_info *, enum remove_bp_reason) override;
  int can_use_hw_breakpoint (enum bptype, int, int) override;
  int insert_hw_breakpoint (struct gdbarch *, struct bp_target_info *) override;
  int remove_hw_breakpoint (struct gdbarch *, struct bp_target_info *) override;
  int remove_watchpoint (CORE_ADDR, int, enum target_hw_bp_type, struct expression *) override;
  int insert_watchpoint (CORE_ADDR, int, enum target_hw_bp_type, struct expression *) override;
  bool stopped_by_watchpoint () override {
    return nto_stopped_by_watchpoint();
  }
#if 0
  void terminal_init () override;
  void terminal_inferior () override;
  void terminal_ours_for_output () override;
  void terminal_info (const char *, int) override;
#endif
  void kill () override;
  void load (const char *args, int from_tty ) override {
    return generic_load (args, from_tty);
  }
  bool can_create_inferior () override {
    return true;
  }
  int insert_fork_catchpoint (int) override;
  int remove_fork_catchpoint (int) override;
  int insert_vfork_catchpoint (int) override;
  int remove_vfork_catchpoint (int) override;
  void follow_fork (inferior *, ptid_t, target_waitkind, bool, bool) override;
  int insert_exec_catchpoint (int) override;
  int remove_exec_catchpoint (int) override;
  void create_inferior (const char *, const std::string &, char **, int) override;
  void mourn_inferior () override;
  bool can_run () override;
  bool thread_alive (ptid_t ptid) override;
  void update_thread_list () override;
  std::string pid_to_str (ptid_t ptid) override {
    return nto_pid_to_str (ptid);
  }
  const char *extra_thread_info (thread_info *ti ) override {
    return nto_extra_thread_info( ti );
  }
  bool has_all_memory () override;
  bool has_memory () override;
  bool has_stack () override;
  bool has_registers () override;
  bool can_async_p () override {
    /* Not yet. */
    return false;
  };
  bool supports_non_stop () override {
    /* Not yet. */
    return false;
  }
  const struct target_desc *read_description () override;
  bool supports_multi_process () override {
    return true;
  }
  int verify_memory (const gdb_byte *data, CORE_ADDR memaddr, ULONGEST size) override;
  enum target_xfer_status xfer_partial (enum target_object object,
                const char *annex,
                gdb_byte *readbuf,
                const gdb_byte *writebuf,
                ULONGEST offset, ULONGEST len,
                ULONGEST *xfered_len) override;

public: /* pdebug specific methods */


  int       nto_insert_breakpoint (CORE_ADDR, gdb_byte *);
  int       nto_remove_breakpoint (CORE_ADDR addr, gdb_byte *contents_cache);
  ptid_t    nto_parse_notify (const DScomm_t *recv, struct target_waitstatus *status);
  int	    nto_send_env (const char *env);
  int       nto_send_arg (const char *arg);
  int       nto_set_thread (ptid_t);
  int       nto_thread_alive (ptid_t);
  int       nto_set_thread_alive (ptid_t);
  int       nto_read_procfsinfo (nto_procfs_info*);
  int       nto_read_procfsstatus (nto_procfs_status*);
  void      nto_detach_pid (int pid);

};

/* static functions that must work without a pdebug_target instance */
static int       nto_close_1 (void);
static void      nto_send_init (DScomm_t *, unsigned, unsigned, unsigned);
static unsigned  nto_send_recv (const DScomm_t *, DScomm_t *, unsigned, int);
static int       nto_start_remote (void);
/* todo turn nto_remote_inferior_data into a proper class */
static struct nto_remote_inferior_data
                *nto_get_remote_inferior_data (struct inferior *inf);
static int       nto_attach_only (const int pid, DScomm_t *const recv);
static int       nto_detach_only (const int pid);

/* command extensions */
static void      nto_add_commands (void);
static int       nto_fileopen (char *fname, int mode, int perms);
static void      nto_fileclose (int);
static int       nto_fileread (char *buf, int size);
static int       nto_filewrite (char *buf, int size);

/* CTRL-C handling */
static void      nto_interrupt (int signo);
static void      nto_interrupt_query (void);
static void      nto_interrupt_twice (int);
static void      nto_interrupt_retry (int);

#ifdef NVIDIA_CUDA_GDB
/* CUDA: This is needed for newer gdb versions */
static const registry<inferior>::key<struct nto_remote_inferior_data> nto_remote_inferior_data_key;
#else
static const struct inferior_data *nto_remote_inferior_data_reg;
#endif

#ifdef __MINGW32__
static void
alarm (int sig)
{
  /* Do nothing, this is windows.  */
}

#define sleep(x) Sleep(1000 * (x))

#endif

struct pdebug_session
{
  /* Number of seconds to wait for a timeout on the remote side.  */
  /* CUDA: by analogy with remote.c, increase max timeout from 10 to 600 seconds
     to account for potentially slow codepaths while single-stepping through CUDA code.
     Bug for tracking proper fix: 2035133 */
  int timeout = 600;

  /* Whether to inherit environment from remote pdebug or host gdb.  */
  bool inherit_env = true;

  /* File to be executed on remote.  Assigned to new inferiors.  */
#ifdef NVIDIA_CUDA_GDB
  std::string remote_exe {};
#else
  char *remote_exe;
#endif

  /* Current working directory on remote.  Assigned to new inferiors.  */
#ifdef NVIDIA_CUDA_GDB
  std::string remote_cwd {};
#else
  char *remote_cwd;
#endif

  /* Descriptor for I/O to remote machine.  Initialize it to NULL so that
     nto_open knows that we don't have a file open when the program
     starts.  */
  struct serial *desc = nullptr;

  /* NTO CPU type of the remote machine.  */
  int cputype = -1;

  /* NTO CPU ID of the remote machine.  */
  unsigned cpuid = 0;

  /* Communication channels to the remote.  */
  unsigned channelrd = SET_CHANNEL_DEBUG;
  unsigned channelwr = SET_CHANNEL_DEBUG;

  /* The version of the protocol used by the pdebug we connect to.
     Set in nto_start_remote().  */
  int target_proto_major = 0;
  int target_proto_minor = 0;

  /* Communication buffer used by to_resume and to_wait. Nothing else
   * should be using it, all other operations should use their own
   * buffers allocated on the stack or heap.  */
  DScomm_t recv;

#ifdef NVIDIA_BUGFIX
  std::queue<DScomm_t> pending;
#endif
};

#ifdef NVIDIA_CUDA_GDB
  /* CUDA: Moved initializers into struct definition */
struct pdebug_session only_session;
#else
struct pdebug_session only_session = {
  10,
  false,
  NULL,
  NULL,
  NULL,
  -1,
  0,
  SET_CHANNEL_DEBUG,
  SET_CHANNEL_DEBUG,
  0, /* target_proto_major */
  0, /* target_proto_minor */
};
#endif

/* Remote session (connection) to a QNX target. */
struct pdebug_session *current_session = &only_session;

/* Flag for whether upload command sets the current session's remote_exe.  */
static bool upload_sets_exec = true;

/* control variables that are now local and no longer global */
static int watchdog = 0;
static void
show_watchdog (struct ui_file *file, int from_tty,
	       struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("Watchdog timer is %s.\n"), value);
}

/* Shadow detach_fork from infrun.c */
static bool nto_detach_fork = false;

/* These define the version of the protocol implemented here.  */
#define HOST_QNX_PROTOVER_MAJOR  0
#define HOST_QNX_PROTOVER_MINOR  7

/* HOST_QNX_PROTOVER 0.8 - 64 bit capable structures.  */

/* Stuff for dealing with the packets which are part of this protocol.  */

#define MAX_TRAN_TRIES 2
#define MAX_RECV_TRIES 2

#define FRAME_CHAR  0x7e
#define ESC_CHAR  0x7d

static unsigned char nak_packet[] =
  { FRAME_CHAR, SET_CHANNEL_NAK, 0, FRAME_CHAR };
static unsigned char ch_reset_packet[] =
  { FRAME_CHAR, SET_CHANNEL_RESET, 0xff, FRAME_CHAR };
static unsigned char ch_debug_packet[] =
  { FRAME_CHAR, SET_CHANNEL_DEBUG, 0xfe, FRAME_CHAR };
static unsigned char ch_text_packet[] =
  { FRAME_CHAR, SET_CHANNEL_TEXT, 0xfd, FRAME_CHAR };

#define SEND_NAK         serial_write(current_session->desc,nak_packet,sizeof(nak_packet))
#define SEND_CH_RESET    serial_write(current_session->desc,ch_reset_packet,sizeof(ch_reset_packet))
#define SEND_CH_DEBUG    serial_write(current_session->desc,ch_debug_packet,sizeof(ch_debug_packet))
#define SEND_CH_TEXT     serial_write(current_session->desc,ch_text_packet,sizeof(ch_text_packet))

/* Pdebug returns errno values on Neutrino that do not correspond to right
   errno values on host side.  */

#define NTO_ENAMETOOLONG        78
#define NTO_ELIBACC             83
#define NTO_ELIBBAD             84
#define NTO_ELIBSCN             85
#define NTO_ELIBMAX             86
#define NTO_ELIBEXEC            87
#define NTO_EILSEQ              88
#define NTO_ENOSYS              89

#if defined(__QNXNTO__) || defined (__SOLARIS__)
#define errnoconvert(x) x
#elif defined(__linux__) || defined (__CYGWIN__) || defined (__MINGW32__) || defined(__APPLE__)

struct errnomap_t { int nto; int other; };

static int
errnoconvert(int x)
{
  struct errnomap_t errnomap[] = {
    #if defined (__linux__)
      {NTO_ENAMETOOLONG, ENAMETOOLONG}, {NTO_ELIBACC, ELIBACC},
      {NTO_ELIBBAD, ELIBBAD}, {NTO_ELIBSCN, ELIBSCN}, {NTO_ELIBMAX, ELIBMAX},
      {NTO_ELIBEXEC, ELIBEXEC}, {NTO_EILSEQ, EILSEQ}, {NTO_ENOSYS, ENOSYS}
    #elif defined(__CYGWIN__)
      {NTO_ENAMETOOLONG, ENAMETOOLONG}, {NTO_ENOSYS, ENOSYS}
    #elif defined(__MINGW32__)
      /* The closest mappings from mingw's errno.h.  */
      {NTO_ENAMETOOLONG, ENAMETOOLONG}, {NTO_ELIBACC, ESRCH},
      {NTO_ELIBEXEC, ENOEXEC}, {NTO_EILSEQ, EILSEQ}, {NTO_ENOSYS, ENOSYS}
    #elif defined(__APPLE__)
      {NTO_ENAMETOOLONG, ENAMETOOLONG}, {NTO_ELIBACC, ESRCH},
      {NTO_ELIBBAD, ESRCH}, {NTO_ELIBSCN, ENOEXEC}, {NTO_ELIBMAX, EPERM},
      {NTO_ELIBEXEC, ENOEXEC}, {NTO_EILSEQ, EILSEQ}, {NTO_ENOSYS, ENOSYS}

    #endif
  };
  int i;

  for (i = 0; i < sizeof(errnomap) / sizeof(errnomap[0]); i++)
    if (errnomap[i].nto == x) return errnomap[i].other;
  return x;
}

#define errnoconvert(x) errnoconvert(x)
#else
#error errno mapping not setup for this host
#endif /* __QNXNTO__ */

/* Get a pointer to the current remote target.  If not connected to a
   remote target, return NULL.  */
static pdebug_target *
get_current_target ()
{
  target_ops *proc_target = current_inferior ()->process_target ();
  return dynamic_cast<pdebug_target *> (proc_target);
}

#ifdef NVIDIA_CUDA_GDB
/* Return true if TARGET is a qnx target, otherwise,
   return false.  */
bool 
is_qnx_target (process_stratum_target *target)
{
  return dynamic_cast<pdebug_target *> (target) != NULL;
}
#endif

static void remote_unpush_target (pdebug_target *target);

/*
 * close the current connection (if any)
 * clean up all inferiors
 * and print an error message
 */
static void pdebug_error(const char* msg, ...)
{
  if(current_session->desc) {
      serial_close (current_session->desc);
      /* do not free current_session->desc, this is done in serial_close! */
      current_session->desc=NULL;
  }
  remote_unpush_target (get_current_target());

  va_list ap;
  va_start (ap, msg);
  verror (msg, ap);
  va_end (ap);
}

/* add a new thread and fill in the qnx thread information */
static thread_info*
nto_add_thread (process_stratum_target *targ, int pid, int tid)
{
  DScomm_t ttran, trecv;
  struct tidinfo *ptidinfo;
  struct nto_thread_info *priv = new struct nto_thread_info ();
  ptid_t ptid = ptid_t (pid, tid, 0);
  struct thread_info *ti = nto_find_thread(ptid);
  if (ti != NULL)
    {
      warning("Thread %s already exists!", nto_pid_to_str(ptid).c_str());
      return ti;
    }

  /* also update full thread info */
  nto_send_init (&ttran, DStMsg_select, DSMSG_SELECT_QUERY, SET_CHANNEL_DEBUG);
  ttran.pkt.select.pid = pid;
  ttran.pkt.select.pid = EXTRACT_SIGNED_INTEGER (&ttran.pkt.select.pid,
						       sizeof(int32_t),
						       nto_byte_order);
  ttran.pkt.select.tid = tid;
  ttran.pkt.select.tid = EXTRACT_SIGNED_INTEGER (&ttran.pkt.select.tid,
						       sizeof(int32_t),
						       nto_byte_order);
  nto_send_recv (&ttran, &trecv, sizeof (ttran.pkt.select), 0);
  ptidinfo = (struct tidinfo *) trecv.pkt.okdata.data;
  if ((trecv.pkt.hdr.cmd == DSrMsg_okdata) && (ptidinfo->tid == tid))
    {
      priv->fill(ptidinfo);
    }
  else
    {
      warning("Could not get threadinfo for pid:%d tid:%d!", pid, tid);
    }
  return add_thread_with_info (targ, ptid_t (pid, tid, 0), priv);
}

/* Call FUNC wrapped in a TRY/CATCH that swallows all GDB
   exceptions. */
static int
catch_errors (int (*func) ())
{
  try
    {
      return func( );
    }
  catch (const gdb_exception_error &ex)
    {
      exception_print (gdb_stderr, ex);
    }

  return 1;
}

/* Send a packet to the remote machine.  Also sets channelwr and informs
   target if channelwr has changed.  */
static int
putpkt (const DScomm_t *const tran, const unsigned len)
{
  int i;
  unsigned char csum = 0;
  unsigned char buf2[DS_DATA_MAX_SIZE * 2];
  unsigned char *p;

  /* Copy the packet into buffer BUF2, encapsulating it
     and giving it a checksum.  */

  p = buf2;
  *p++ = FRAME_CHAR;

  nto_trace (1) ("putpkt() - cmd %d, subcmd %d, mid %d\n",
       tran->pkt.hdr.cmd, tran->pkt.hdr.subcmd,
       tran->pkt.hdr.mid);

  if (remote_debug)
    printf_unfiltered ("Sending packet (len %d): ", len);

  for (i = 0; i < len; i++)
    {
      unsigned char c = tran->buf[i];

      if (remote_debug)
        printf_unfiltered ("%2.2x", c);
      csum += c;

      switch (c)
        {
        case FRAME_CHAR:
        case ESC_CHAR:
          if (remote_debug)
            printf_unfiltered ("[escape]");
          *p++ = ESC_CHAR;
          c ^= 0x20;
          break;
        }
      *p++ = c;
    }

  csum ^= 0xff;

  if (remote_debug)
    {
      printf_unfiltered ("%2.2x\n", csum);
      gdb_flush (gdb_stdout);
    }
  switch (csum)
    {
    case FRAME_CHAR:
    case ESC_CHAR:
      *p++ = ESC_CHAR;
      csum ^= 0x20;
      break;
    }
  *p++ = csum;
  *p++ = FRAME_CHAR;

  /* GP added - June 17, 1999.  There used to be only 'channel'.
     Now channelwr and channelrd keep track of the state better.
     If channelwr is not in the right state, notify target and set channelwr.  */
  if (current_session->channelwr != tran->pkt.hdr.channel)
    {
      switch (tran->pkt.hdr.channel)
        {
        case SET_CHANNEL_TEXT:
          SEND_CH_TEXT;
          break;
        case SET_CHANNEL_DEBUG:
          SEND_CH_DEBUG;
          break;
        }
      current_session->channelwr = tran->pkt.hdr.channel;
    }

  if (serial_write (current_session->desc, (char *)buf2, p - buf2))
    perror_with_name ("putpkt: write failed");

  return len;
}
#if 0
static void
sig_io (int signal)
{
  char buff[1000];
  fd_set  read_set;
  struct timeval timeout_value;

  FD_ZERO (&read_set);
  FD_SET (0, &read_set);

  timeout_value.tv_sec = 0;
  timeout_value.tv_usec = 0;

  if (select (1, &read_set, NULL, NULL, &timeout_value) > 0)
    {
      int read_count = read (STDIN_FILENO, buff, 1);

      if (read_count < 0)
  {
    gdb_printf (gdb_stderr, _("Error reading stdin: %d(%s)\n"),
            errno, safe_strerror (errno));
    return;
  }
      if (read_count == 0)
  {
    gdb_printf (gdb_stderr, _("EOF\n"));
    return;
  }
      gdb_printf (gdb_stderr, _("'%c'"), buff[0]);
    }
}
#endif

/* Read a single character from the remote end, masking it down to 8 bits.  */
static int
readchar (int timeout)
{
  int ch;
#if 0
//  void (*def_sig)(int) = signal(SIGIO, sig_io);
  struct sigaction sa;

  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = sig_io;

  if (sigaction(SIGIO, &sa, NULL) < 0) {
    fprintf(stderr,"[ERROR]: sigaction error\n");
    exit(1);
  }

  if (fcntl(0, F_SETOWN, getpid()) < 0) {
    fprintf(stderr,"[ERROR]: fcntl F_SETOWN error\n");
    exit(1);
  }

  if (fcntl(0, F_SETFL, O_NONBLOCK|O_ASYNC) < 0) {
    fprintf(stderr,"[ERROR]: fcntl error for O_NONBLOCK|O_ASYNC\n");
    exit(1);
  }
#endif
  ch = serial_readchar (current_session->desc, timeout);

  switch (ch)
    {
    case SERIAL_EOF:
      pdebug_error ("Remote connection closed");
      break;
    case SERIAL_ERROR:
      pdebug_error ("Remote communication error (%i: %s)", errno, strerror(errno));
      break;
    case SERIAL_TIMEOUT:
      return ch;
    }

  return ch & 0xff;
}

/* Come here after finding the start of the frame.  Collect the rest into BUF,
   verifying the checksum, length, and handling run-length compression.
   Returns 0 on any error, 1 on success.  */
static int
read_frame (unsigned char *const buf, const size_t bufsz)
{
  unsigned char csum;
  unsigned char *bp = buf;
  unsigned char modifier = 0;
  int c;

  if (remote_debug)
    gdb_printf ("Receiving data: ");

  csum = 0;

  memset (bp, -1, bufsz);
  for (;;)
    {
      c = readchar (current_session->timeout);

      switch (c)
  {
  case SERIAL_TIMEOUT:
    gdb_puts ("Timeout in mid-packet, retrying\n");
    return -1;
  case ESC_CHAR:
    modifier = 0x20;
    continue;
  case FRAME_CHAR:
    if (bp == buf)
      continue;    /* Ignore multiple start frames.  */
    if (csum != 0xff)  /* Checksum error.  */
      return -1;
    return bp - buf - 1;
  default:
    c ^= modifier;
    if (remote_debug)
      gdb_printf ("%2.2x", c);
    csum += c;
    *bp++ = c;
    break;
  }
      modifier = 0;
    }
  return 1;
}

/* Read a packet from the remote machine, with error checking,
   and store it in recv.buf.
   If FOREVER, wait forever rather than timing out; this is used
   while the target is executing user code.  */
static int
getpkt (DScomm_t *const recv, const int forever)
{
  int c;
  int tries;
  int timeout;
  unsigned len;

  if (remote_debug)
    printf_unfiltered ("getpkt(%d)\n", forever);

  if (forever)
    {
      timeout = watchdog > 0 ? watchdog : -1;
    }
  else
    {
      timeout = current_session->timeout;
    }

  for (tries = 0; tries < MAX_RECV_TRIES; tries++)
    {
      /* This can loop forever if the remote side sends us characters
         continuously, but if it pauses, we'll get a zero from readchar
         because of timeout.  Then we'll count that as a retry.

         Note that we will only wait forever prior to the start of a packet.
         After that, we expect characters to arrive at a brisk pace.  They
         should show up within nto_timeout intervals.  */
      do
  {
    c = readchar (timeout);

    if (c == SERIAL_TIMEOUT)
      {
        /* Watchdog went off.  Kill the target.  */
        if (forever && watchdog > 0)
    {
      pdebug_error ("Watchdog has expired.  Target detached.");
    }
        gdb_puts ("Timed out.\n");
        return -1;
      }
  }
      while (c != FRAME_CHAR);

      /* We've found the start of a packet, now collect the data.  */
      len = read_frame (recv->buf, sizeof recv->buf);

      if (remote_debug)
  gdb_printf ("\n");

      if (len >= sizeof (struct DShdr))
  {
    if (recv->pkt.hdr.channel)  /* If hdr.channel is not 0, then hdr.channel is supported.  */
      current_session->channelrd = recv->pkt.hdr.channel;

    if (remote_debug)
      {
        printf_unfiltered ("getpkt() - len %d, channelrd %d,", len,
         current_session->channelrd);
        switch (current_session->channelrd)
    {
    case SET_CHANNEL_DEBUG:
      printf_unfiltered (" cmd = %d, subcmd = %d, mid = %d\n",
             recv->pkt.hdr.cmd, recv->pkt.hdr.subcmd,
             recv->pkt.hdr.mid);
      break;
    case SET_CHANNEL_TEXT:
      printf_unfiltered (" text message\n");
      break;
    case SET_CHANNEL_RESET:
      printf_unfiltered (" set_channel_reset\n");
      break;
    default:
      printf_unfiltered (" unknown channel!\n");
      break;
    }
      }
    return len;
  }
      if (len >= 1)
  {
    /* Packet too small to be part of the debug protocol,
       must be a transport level command.  */
    if (recv->buf[0] == SET_CHANNEL_NAK)
      {
        /* Our last transmission didn't make it - send it again.  */
        current_session->channelrd = SET_CHANNEL_NAK;
        return -1;
      }
    if (recv->buf[0] <= SET_CHANNEL_TEXT)
      current_session->channelrd = recv->buf[0];

    if (remote_debug)
      {
        printf_unfiltered ("set channelrd to %d\n",
         current_session->channelrd);
      }
    --tries;    /* Doesn't count as a retry.  */
    continue;
  }
      SEND_NAK;
    }

  /* We have tried hard enough, and just can't receive the packet.  Give up.  */
  printf_unfiltered ("Ignoring packet error, continuing...");
  return 0;
}

static void
nto_send_init (DScomm_t *const tran, unsigned cmd, const unsigned subcmd, const unsigned chan)
{
#ifndef NVIDIA_CUDA_GDB
  /* CUDA: see explanation above. We need to keep this in sync for cuda/cuda-gdbserver. */
  static unsigned char mid;
#endif

  gdb_assert (tran != NULL);

  nto_trace (2) ("    nto_send_init(cmd %d, subcmd %d)\n", cmd,
       subcmd);

  if (nto_byte_order == BFD_ENDIAN_BIG)
    cmd |= DSHDR_MSG_BIG_ENDIAN;

  memset (tran, 0, sizeof (DScomm_t));

  tran->pkt.hdr.cmd = cmd;  /* TShdr.cmd.  */
  tran->pkt.hdr.subcmd = subcmd;  /* TShdr.console.  */
  tran->pkt.hdr.mid = ((chan == SET_CHANNEL_DEBUG) ? mid++ : 0);  /* TShdr.spare1.  */
  tran->pkt.hdr.channel = chan;  /* TShdr.channel.  */
}


#if 0
/* Send text to remote debug daemon - Pdebug.
 *
 * TODO: currently unused, see CLT-1252
 */

void
nto_outgoing_text (char *buf, int nbytes)
{
  DScomm_t tran;

  TSMsg_text_t *msg;

  msg = (TSMsg_text_t *) & tran;

  msg->hdr.cmd = TSMsg_text;
  msg->hdr.console = 0;
  msg->hdr.spare1 = 0;
  msg->hdr.channel = SET_CHANNEL_TEXT;

  memcpy (msg->text, buf, nbytes);

  putpkt (&tran, nbytes + offsetof (TSMsg_text_t, text));
}
#endif

/* Display some text that came back across the text channel.  */

static int
nto_incoming_text (TSMsg_text_t *const text, const int len)
{
  int textlen;
  const size_t buf_sz = TS_TEXT_MAX_SIZE + 1;
  char buf[buf_sz];

  textlen = len - offsetof (TSMsg_text_t, text);
  if (textlen <= 0)
    return 0;

  switch (text->hdr.cmd)
    {
    case TSMsg_text:
      snprintf (buf, buf_sz, "%s", text->text);
      buf[textlen] = '\0';
      //ui_file_write (gdb_stdtarg, buf, textlen);
      gdb_puts (buf, gdb_stdtarg);
      return 0;
    default:
      return -1;
    }
}


/* Send env. string. Send multipart if env string too long and
   our protocol version allows multipart env string.

   Returns > 0 if successful, 0 on error.  */

int
pdebug_target::nto_send_env (const char *env)
{
  int len; /* Length including zero terminating char.  */
  int totlen = 0;
  DScomm_t tran, recv;

  gdb_assert (env != NULL);
  len = strlen (env) + 1;
  if (current_session->target_proto_major > 0
      || current_session->target_proto_minor >= 2)
    {
  while (len > DS_DATA_MAX_SIZE)
    {
      nto_send_init (&tran, DStMsg_env, DSMSG_ENV_SETENV_MORE,
         SET_CHANNEL_DEBUG);
      memcpy (tran.pkt.env.data, env + totlen,
        DS_DATA_MAX_SIZE);
      if (!nto_send_recv (&tran, &recv, offsetof (DStMsg_env_t, data) +
         DS_DATA_MAX_SIZE, 1))
        {
    /* An error occured.  */
    return 0;
        }
      len -= DS_DATA_MAX_SIZE;
      totlen += DS_DATA_MAX_SIZE;
    }
    }
  else if (len > DS_DATA_MAX_SIZE)
    {
      /* Not supported by this protocol version.  */
      printf_unfiltered
  ("** Skipping env var \"%.40s .....\" <cont>\n", env);
      printf_unfiltered
  ("** Protovers under 0.2 do not handle env vars longer than %d\n",
    DS_DATA_MAX_SIZE - 1);
      return 0;
    }
  nto_send_init (&tran, DStMsg_env, DSMSG_ENV_SETENV, SET_CHANNEL_DEBUG);
  memcpy (tran.pkt.env.data, env + totlen, len);
  return nto_send_recv (&tran, &recv, offsetof (DStMsg_env_t, data) + len, 1);
}


/* Send an argument to inferior. Unfortunately, DSMSG_ENV_ADDARG
   does not support multipart strings limiting the length
   of single argument to DS_DATA_MAX_SIZE.  */

int
pdebug_target::nto_send_arg (const char *arg)
{
  int len;
  DScomm_t tran, recv;

  gdb_assert (arg != NULL);

  len = strlen(arg) + 1;
  if (len > DS_DATA_MAX_SIZE)
    {
      printf_unfiltered ("Argument too long: %.40s...\n", arg);
      return 0;
    }
  nto_send_init (&tran, DStMsg_env, DSMSG_ENV_ADDARG, SET_CHANNEL_DEBUG);
  memcpy (tran.pkt.env.data, arg, len);
  return nto_send_recv (&tran, &recv, offsetof (DStMsg_env_t, data) + len, 1);
}

/* Send the command in tran.buf to the remote machine,
   and read the reply into recv.buf.  */

static unsigned
nto_send_recv (const DScomm_t *const tran, DScomm_t *const recv,
	       const unsigned len, const int report_errors)
{
  int rlen;
  unsigned tries;
#ifdef NVIDIA_BUGFIX
  bool stashed = false;
#endif

  if (current_session->desc == NULL)
    {
      errno = ENOTCONN;
      return 0;
    }

  for (tries = 0;; tries++)
    {
      if (tries >= MAX_TRAN_TRIES)
	{
	  unsigned char err = DSrMsg_err;

	  printf_unfiltered ("Remote exhausted %d retries.\n", tries);
	  if (nto_byte_order == BFD_ENDIAN_BIG)
	    err |= DSHDR_MSG_BIG_ENDIAN;
	  recv->pkt.hdr.cmd = err;
	  recv->pkt.err.err = EIO;
	  recv->pkt.err.err = EXTRACT_SIGNED_INTEGER (&recv->pkt.err.err,
						      4, nto_byte_order);
	  rlen = sizeof (recv->pkt.err);
	  /* connection is considered dead */
	  current_session->desc = NULL;
	  break;
	}
#ifdef NVIDIA_BUGFIX
      if (!stashed)
        putpkt (tran, len);
      else
        stashed = false;
#else
      putpkt (tran, len);
#endif
      for (;;)
	{
	  rlen = getpkt (recv, 0);
	  if ((current_session->channelrd != SET_CHANNEL_TEXT)
	      || (rlen == -1))
	    break;
	  nto_incoming_text (&recv->text, rlen);
	}
      if (rlen == -1)    /* Getpkt returns -1 if MsgNAK received.  */
	{
	  printf_unfiltered ("MsgNak received - resending\n");
	  continue;
	}
      if ((rlen >= 0) && (recv->pkt.hdr.mid == tran->pkt.hdr.mid))
	break;
#ifdef NVIDIA_CUDA_GDB
      if ((rlen >= 0) && (recv->pkt.hdr.cmd == DSrMsg_okcuda))
	{
	  /* We need to keep mid in sync. cuda-gdbserver always returns
	     the latest unused mid in okcuda packets. */
	  mid = recv->pkt.hdr.mid;
	  const_cast<DScomm_t *>(tran)->pkt.hdr.mid = mid;
	  break;
	}
#endif
#ifdef NVIDIA_BUGFIX
      if (recv->pkt.hdr.cmd == DShMsg_notify)
        {
	  nto_trace (1) ("Received notify message\n");
	  current_session->pending.push (*recv);
	  stashed = true;
	  /* Let's not consider this an attempt. */
	  tries--;
	  continue;
	}
#endif

      nto_trace (1) ("mid mismatch!\n");

    }
  /* Getpkt() sets channelrd to indicate where the message came from.
     now we switch on the channel (/type of message) and then deal
     with it.  */
  switch (current_session->channelrd)
  {
    case SET_CHANNEL_DEBUG:
      if (((recv->pkt.hdr.cmd & DSHDR_MSG_BIG_ENDIAN) != 0))
	{
	  char buff[sizeof(tran->buf)];

	  sprintf (buff, "set endian big");
	  if (nto_byte_order != BFD_ENDIAN_BIG)
	    execute_command (buff, 0);
	}
      else
	{
	  char buff[sizeof(tran->buf)];

	  sprintf (buff, "set endian little");
	  if (nto_byte_order != BFD_ENDIAN_LITTLE)
	    execute_command (buff, 0);
	}
      recv->pkt.hdr.cmd &= ~DSHDR_MSG_BIG_ENDIAN;
      if (recv->pkt.hdr.cmd == DSrMsg_err)
	{
	  /* the actual errno */
	  errno = errnoconvert (EXTRACT_SIGNED_INTEGER (&recv->pkt.err.err, 4,
							nto_byte_order));
	  if (report_errors)
	    {
	      /* any error reported by pdebug */
	      switch (EXTRACT_SIGNED_INTEGER(&recv->pkt.hdr.subcmd,
					     sizeof(uint8_t), nto_byte_order))
	      {
		case PDEBUG_ENOERR:
		  break;
		case PDEBUG_ENOPTY:
		  perror_with_name ("Remote (no ptys available)");
		  break;
		case PDEBUG_ETHREAD:
		  perror_with_name ("Remote (thread start error)");
		  break;
		case PDEBUG_ECONINV:
		  perror_with_name ("Remote (invalid console number)");
		  break;
		case PDEBUG_ESPAWN:
		  perror_with_name ("Remote (spawn error)");
		  break;
		case PDEBUG_EPROCFS:
		  perror_with_name ("Remote (procfs [/proc] error)");
		  break;
		case PDEBUG_EPROCSTOP:
		  perror_with_name ("Remote (devctl PROC_STOP error)");
		  break;
		case PDEBUG_EQPSINFO:
		  perror_with_name ("Remote (psinfo error)");
		  break;
		case PDEBUG_EQMEMMODEL:
		  perror_with_name
		  ("Remote (invalid memory model [not flat] )");
		  break;
		case PDEBUG_EQPROXY:
		  perror_with_name ("Remote (proxy error)");
		  break;
		case PDEBUG_EQDBG:
		  perror_with_name ("Remote (__nto_debug_* error)");
		  break;
		default:
		  if( errno != EOK )
		    {
		      perror_with_name ("Remote");
		    }
	      }
	    }
	}
      break;
    case SET_CHANNEL_TEXT:
    case SET_CHANNEL_RESET:
      break;
  }
  return rlen;
}

int
pdebug_target::nto_set_thread (ptid_t ptid)
{
  DScomm_t tran, recv;
  long int th = ptid.lwp();
  if( th == 0 )
      th=1;

  nto_trace (0) ("nto_set_thread(%s)\n", nto_pid_to_str (ptid).c_str());

  if (find_inferior_ptid (this, ptid) == NULL)
    {
      nto_trace (0) ( "  no inferior for %s yet!\n", nto_pid_to_str (ptid).c_str());
      return 0;
    }

  nto_send_init (&tran, DStMsg_select, DSMSG_SELECT_SET, SET_CHANNEL_DEBUG);
  tran.pkt.select.pid = ptid.pid();
  tran.pkt.select.pid = EXTRACT_SIGNED_INTEGER ((gdb_byte*)&tran.pkt.select.pid, 4,
            nto_byte_order);
  tran.pkt.select.tid = EXTRACT_SIGNED_INTEGER (&th, 4, nto_byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.select), 1);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      nto_trace (0) ("  thread %li does not exist (%i - %s)\n", th, recv.pkt.err.err, strerror(recv.pkt.err.err));
      return 0;
    }

  switch_to_thread(this, ptid);

  return 1;
}

/*
 * Checks if the given thread is alive,  RECV will contain returned_tid.
 * NOTE: Make sure this stays like that since we will use this side effect in
 * other functions to determine first thread alive (for example, after attach).
 * Returns
 *  tid - thread id
 *        this is either the requested thread or a higher number of the next
 *        active thread
 *  -1  - the process has no more active threads
 */
int
pdebug_target::nto_thread_alive (ptid_t th)
{
  DScomm_t tran, recv;
  int returned_tid;

  nto_trace(1) ("nto_thread_alive(%s)\n", nto_pid_to_str (th).c_str());

  /* this can happen in IDE sessions when the IDE requests data before the
   * process has fully spawned so it must not be an error. */
  if (th.pid() == 0)
    {
      nto_trace(0)("  No pid to find a thread for!");
      return 0;
    }

  nto_send_init (&tran, DStMsg_select, DSMSG_SELECT_QUERY, SET_CHANNEL_DEBUG);
  tran.pkt.select.pid = th.pid ();
  tran.pkt.select.pid = EXTRACT_SIGNED_INTEGER(&tran.pkt.select.pid,
					       sizeof(int32_t), nto_byte_order);
  tran.pkt.select.tid = th.lwp ();
  tran.pkt.select.tid = EXTRACT_SIGNED_INTEGER(&tran.pkt.select.tid,
					       sizeof(int32_t), nto_byte_order);
  nto_send_recv (&tran, &recv, sizeof(tran.pkt.select), 0);

  if (recv.pkt.hdr.cmd == DSrMsg_okdata)
    {
      /* Data is tidinfo.
       Note: tid returned might not be the same as requested.
       If it is not, then requested thread is dead.  */
      const struct tidinfo * const ptidinfo =
	  (struct tidinfo *) recv.pkt.okdata.data;
      returned_tid = EXTRACT_SIGNED_INTEGER(&ptidinfo->tid, sizeof(int16_t),
					    nto_byte_order);
    }
  else if (recv.pkt.hdr.cmd == DSrMsg_okstatus)
    {
      /* This is the old behaviour. It doesn't really tell us
       what is the status of the thread, but rather answers question:
       "Does the thread exist?". Note that a thread might have already
       exited but has not been joined yet; we will show it here as
       alive and well. Not completely correct.  */
      returned_tid = EXTRACT_SIGNED_INTEGER(&recv.pkt.okstatus.status,
					    sizeof(int32_t), nto_byte_order);
    }
  else
    {
      /* no more threads available */
      nto_trace (0) ("  No threads available (%i - %s)\n", recv.pkt.err.err, strerror(recv.pkt.err.err));
      returned_tid = -1;
    }

  return returned_tid;
}

/* Return nonzero if the thread th is still alive on the remote system. */
bool
pdebug_target::thread_alive (ptid_t th)
{
  int alive;
  alive = (nto_thread_alive (th) == th.lwp());
  nto_trace(0) ("pdebug_thread_alive(%s) is %s\n", nto_pid_to_str (th).c_str(),
		alive ? "alive" : "dead");
  return alive;
}

/*
 * sets a non-dead thread for the given process/thread.
 * By default the given thread is checked if it is alive. If not, the process
 * is scanned for an active thread.
 * If the given thread is 0, then the currently active thread is checked. This
 * is done to avoid switching from the current thread and sending following
 * messages to the wrong thread. returns the tid of the active
 * thread or 0 if the process has no active threads.
 */
int
pdebug_target::nto_set_thread_alive (ptid_t th)
{
  int alive;

  /* check thread id, if this is 0 then a message to the process is about to
   * be sent so the thread is changed to the current thread id so the context
   * does not change for the following messages. */
  if (th.lwp () == 0)
      th = ptid_t (th.pid (), inferior_ptid.lwp ()?inferior_ptid.lwp ():1, 0);

  nto_trace(0) ("nto_set_thread_alive(%s)\n", nto_pid_to_str (th).c_str());
  alive = nto_thread_alive (th);

  if (alive != th.lwp())
    {
      /* was the original check already for the whole process? */
      if (th.lwp() != 1)
	{
	  /* try with thread1 */
	  th = ptid_t (th.pid (), 1, 0);
	  alive = nto_thread_alive (th);
	}
    }

  if (alive > 0)
    {
      nto_set_thread (ptid_t (th.pid (), alive, 0));
    }
  else
    {
      nto_trace (0) ("  no thread alive for %s\n", nto_pid_to_str (th).c_str());
      alive = 0;
    }

  return alive;
}

/* Clean up connection to a remote debugger.  */
static int
nto_close_1 ( )
{
  DScomm_t tran, recv;

  nto_send_init (&tran, DStMsg_disconnect, 0, SET_CHANNEL_DEBUG);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.disconnect), 0);
  serial_close (current_session->desc);

  return 0;
}

void
pdebug_target::close ( )
{
  nto_trace (0) ("pdebug_close\n");

  /* close the connection if it's still alive */
  if (current_session->desc)
    {
      catch_errors ( nto_close_1 );
      current_session->desc = NULL;
    }

  /* Make sure we leave stdin registered in the event loop.  */
  terminal_ours ();

  /* We don't have a connection to the remote stub anymore.  Get rid
     of all the inferiors and their threads we were controlling.
     Reset inferior_ptid to null_ptid first, as otherwise has_stack_frame
     will be unable to find the thread corresponding to (pid, 0, 0).  */
  inferior_ptid = null_ptid;

  /* todo align with current TLS layout */
  trace_reset_local_state ();

  remote_unpush_target ( this );
  delete this;
}


/* Reads procfs_info structure for the given process.

   Returns 1 on success, 0 otherwise.  */

int
pdebug_target::nto_read_procfsinfo (nto_procfs_info *pinfo)
{
  DScomm_t tran, recv;

  gdb_assert (pinfo != NULL && !! "pinfo must not be NULL\n");
  nto_send_init (&tran, DStMsg_procfsinfo, 0, SET_CHANNEL_DEBUG);
  tran.pkt.procfsinfo.pid = inferior_ptid.pid();
  tran.pkt.procfsinfo.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.procfsinfo.pid,
                4, nto_byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.procfsinfo), 0);
  if (recv.pkt.hdr.cmd == DSrMsg_okdata)
    {
      memcpy (pinfo, recv.pkt.okdata.data, sizeof (*pinfo));
      return 1;
    }
  else
    {
      nto_trace (0) ("DStMsg_procfsinfo not supported by the target.\n");
    }
  return 0;
}


/* Reads procfs_status structure for the given process.

   Returns 1 on success, 0 otherwise.  */

int
pdebug_target::nto_read_procfsstatus (nto_procfs_status *pstatus)
{
  DScomm_t tran, recv;

  gdb_assert (pstatus != NULL && !! "pstatus must not be NULL\n");
  nto_send_init (&tran, DStMsg_procfsstatus, 0, SET_CHANNEL_DEBUG);
  tran.pkt.procfsstatus.pid = inferior_ptid.pid();
  tran.pkt.procfsstatus.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.procfsstatus.pid,
                4, nto_byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.procfsstatus), 0);
  if (recv.pkt.hdr.cmd == DSrMsg_okdata)
    {
      memcpy (pstatus, recv.pkt.okdata.data, sizeof (*pstatus));
      return 1;
    }
  else
    {
      nto_trace (0) ("DStMsg_procfsstatus not supported by the target.\n");
    }
  return 0;
}


/* This is a 'hack' to reset internal state maintained by gdb. It is
   unclear why it doesn't do it automatically, but the same hack can be
   seen in linux, so I guess it is o.k. to use it here too.  */
extern void nullify_last_target_wait_ptid (void);

/*
 * fetch the current environment names on the target
 * While I would prefer to fetch the environment as a whole from the target, this would
 * be very prone of data overflows, DS_DATA_MAX_SIZE is 1k and that could already be not
 * enough for a convoluted LD_LIBRARY_PATH, so we prepare to fetch them one by one.
 */
static int
fetch_envvars ()
{
  int len;
  int rlen;
  int vars=0;
  DScomm_t tran, recv, vtran, vrecv;
  char *varname=NULL;

  /* drop current host environment */
  current_inferior()->environment.clear();

  nto_send_init (&tran, DStMsg_targenv, DSMSG_TARGENV_GETNAMES, SET_CHANNEL_DEBUG);
  rlen = nto_send_recv (&tran, &recv, sizeof (tran.pkt.env), 0);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      warning("Could not read target environments!");
      return 0;
    }

  if( recv.pkt.env.data[0] == 0 )
    {
      nto_trace(0)("Target returned no environments\n");
      return 0;
    }

  varname=recv.pkt.env.data;
  for(len=0; len <= rlen; len++)
    {
      if(varname[len]==0)
	{
	  vars++;
	  if(varname[len+1]==0)
	    break;
	}
    }

  nto_trace(0)("Target returned %i environments\n", vars);

  for( len = 0; len < vars; len++ )
    {
      nto_send_init (&vtran, DStMsg_targenv, DSMSG_TARGENV_GETVALUE, SET_CHANNEL_DEBUG);
      strcpy(vtran.pkt.env.data, varname);
      rlen = nto_send_recv (&vtran, &vrecv, sizeof (vtran.pkt.env), 0);

      if (vrecv.pkt.hdr.cmd == DSrMsg_err)
	{
	  warning("Could not read environment %s from target!", varname);
	}
      else
	{
	  current_inferior()->environment.set (varname, vrecv.pkt.env.data);
	}
      varname=varname+strlen(varname)+1;
    }

  return vars;
}

static int
nto_start_remote ( )
{
  int orig_target_endian;
  DScomm_t tran, recv;

  nto_trace (0) ("nto_start_remote\n" );

  for (;;)
    {
      orig_target_endian = (nto_byte_order == BFD_ENDIAN_BIG);

      /* Reset remote pdebug.  */
      SEND_CH_RESET;

      nto_send_init (&tran, DStMsg_connect, 0, SET_CHANNEL_DEBUG);

      tran.pkt.connect.major = HOST_QNX_PROTOVER_MAJOR;
      tran.pkt.connect.minor = HOST_QNX_PROTOVER_MINOR;

      nto_send_recv (&tran, &recv, sizeof (tran.pkt.connect), 0);

      if (recv.pkt.hdr.cmd != DSrMsg_err)
  break;
      if (orig_target_endian == (nto_byte_order == BFD_ENDIAN_BIG))
  break;
      /* Send packet again, with opposite endianness.  */
    }
  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
       pdebug_error ("Connection failed: %ld.",
       (long) EXTRACT_SIGNED_INTEGER (&recv.pkt.err.err, 4, nto_byte_order));
    }
  /* NYI: need to size transmit/receive buffers to allowed size in connect response.  */

  printf_unfiltered ("Remote target is %s-endian\n",
         (nto_byte_order == BFD_ENDIAN_BIG) ? "big" : "little");

  /* Try to query pdebug for their version of the protocol.  */
  nto_send_init (&tran, DStMsg_protover, 0, SET_CHANNEL_DEBUG);
  tran.pkt.protover.major = HOST_QNX_PROTOVER_MAJOR;
  tran.pkt.protover.minor = HOST_QNX_PROTOVER_MINOR;
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.protover), 0);
  if ((recv.pkt.hdr.cmd == DSrMsg_err)
      && (EXTRACT_SIGNED_INTEGER (&recv.pkt.err.err, 4, nto_byte_order)
    == EINVAL))  /* Old pdebug protocol version 0.0.  */
    {
      current_session->target_proto_major = 0;
      current_session->target_proto_minor = 0;
    }
  else if (recv.pkt.hdr.cmd == DSrMsg_okstatus)
    {
      current_session->target_proto_major =
  EXTRACT_SIGNED_INTEGER (&recv.pkt.okstatus.status, 4, nto_byte_order);
      current_session->target_proto_minor =
  EXTRACT_SIGNED_INTEGER (&recv.pkt.okstatus.status, 4, nto_byte_order);
      current_session->target_proto_major =
  (current_session->target_proto_major >> 8) & DSMSG_PROTOVER_MAJOR;
      current_session->target_proto_minor =
  current_session->target_proto_minor & DSMSG_PROTOVER_MINOR;
    }
  else
    {
      pdebug_error ("Connection failed (Protocol Version Query): %ld.",
       (long) EXTRACT_SIGNED_INTEGER (&recv.pkt.err.err, 4, nto_byte_order));
    }

  nto_trace (0) ("Pdebug protover %d.%d, GDB protover %d.%d\n",
       current_session->target_proto_major,
       current_session->target_proto_minor,
       HOST_QNX_PROTOVER_MAJOR, HOST_QNX_PROTOVER_MINOR);

#ifdef NVIDIA_CUDA_GDB
  /* Fail if remote is pdebug or a different CUDA version in cuda-gdbserver */
  cuda_qnx_version_handshake ();
#endif

  /* If we had an inferior running previously, gdb will have some internal
     states which we need to clear to start fresh.  */
  registers_changed ();
  nullify_last_target_wait_ptid ();
  inferior_ptid = null_ptid;

  if( current_session->target_proto_minor > 7 )
    if( fetch_envvars() == 0)
      {
	warning("Could not read target environment! Enabling nto-inherit-env.");
	current_session->inherit_env=1;
      }

  return 1;
}

/* Remove any of the remote.c targets from target stack.  Upper targets depend
   on it so remove them first.  */

static void
remote_unpush_target ( pdebug_target *target )
{
  /* We have to unpush the target from all inferiors, even those that
     aren't running.  */
  scoped_restore_current_inferior restore_current_inferior;

  for (inferior *inf : all_inferiors (target))
    {
      switch_to_inferior_no_thread (inf);
      inf->pop_all_targets_at_and_above (process_stratum);
      generic_mourn_inferior ();
    }

  /* Don't rely on target_close doing this when the target is popped
     from the last remote inferior above, because something may be
     holding a reference to the target higher up on the stack, meaning
     target_close won't be called yet.  We lost the connection to the
     target, so clear these now, otherwise we may later throw
     TARGET_CLOSE_ERROR while trying to tell the remote target to
     close the file.  */
//  fileio_handles_invalidate_target (target);
}


static void
nto_semi_init (void)
{
  DScomm_t tran, recv;

  nto_send_init (&tran, DStMsg_disconnect, 0, SET_CHANNEL_DEBUG);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.disconnect), 0);

  inferior_ptid = null_ptid;

  if (!catch_errors (nto_start_remote) )
    {
      reinit_frame_cache ();
      remote_unpush_target (get_current_target());
      nto_trace (2) ("nto_semi_init() - pop_target\n");
    }
}

static int nto_open_interrupted = 0;

static void
nto_open_break (int signo)
{
  nto_trace(0)("SIGINT in serial open\n");
  nto_open_interrupted = 1;
}

/* Open a connection to a remote debugger.
   NAME is the filename used for communication.  */
void
pdebug_target::pdebug_open (const char *name, int from_tty)
{
  int tries = 0;
  void (*ofunc) (int);

  nto_trace (0) ("pdebug_open(name '%s', from_tty %d)\n", name,
       from_tty);

  nto_open_interrupted = 0;
  if (name == 0)
    error
      ("To open a remote debug connection, you need to specify what serial\ndevice is attached to the remote system (e.g. /dev/ttya).");

  /* If we're connected to a running target, target_preopen will kill it.
     Ask this question first, before target_preopen has a chance to kill
     anything.  */
  if ( ( current_session->desc != NULL ) && !have_inferiors ())
    {
      if (from_tty
    && !query (_("Already connected to a remote target.  Disconnect? ")))
  error (_("Still connected."));
    }

  target_preopen (from_tty);

  ofunc = signal(SIGINT, nto_open_break);

  while (tries < MAX_TRAN_TRIES && !nto_open_interrupted)
  {
    current_session->desc = serial_open (name);

    if (nto_open_interrupted)
      break;

    /* Give the target some time to come up. When we are connecting
       immediately after disconnecting from the remote, pdebug
       needs some time to start listening to the port. */
    if (!current_session->desc)
      {
        tries++;
        sleep (1);
      }
    else
        break;
  }

  signal(SIGINT, ofunc);

  if (nto_open_interrupted)
    {
      return;
    }

  if (!current_session->desc)
    {
      perror_with_name (name);
    }

  if (baud_rate != -1)
    {
      if (serial_setbaudrate (current_session->desc, baud_rate))
  {
    serial_close (current_session->desc);
    perror_with_name (name);
  }
    }

  serial_raw (current_session->desc);

  /* If there is something sitting in the buffer we might take it as a
     response to a command, which would be bad.  */
  serial_flush_input (current_session->desc);

  if (from_tty)
    {
      gdb_puts ("Remote debugging using ");
      gdb_puts (name);
      gdb_puts ("\n");
    }

#ifdef NVIDIA_CUDA_GDB
  pdebug_target *target = dynamic_cast<pdebug_target *>(new cuda_nat_linux<pdebug_target> {});
#else
  pdebug_target *target=new pdebug_target();
#endif
  current_inferior ()->push_target (target);  /* Switch to using remote target now.  */

  nto_add_commands ();
  nto_trace (3) ("pdebug_open() - push_target\n");

  inferior_ptid = null_ptid;

  /* Start the remote connection; if error (0), discard this target.
     In particular, if the user quits, be sure to discard it
     (we'd be in an inconsistent state otherwise).  */
  if (!catch_errors ( nto_start_remote ))
    {
      remote_unpush_target (target);

      nto_trace (0) ("pdebug_open() - pop_target\n");
    }
  else
    {
      /*
       * GDB expects fork handling to be triggered by a special breakpoint.
       * NTO sends a special message on fork, this means the system and
       * especially parent and child process states are not what GDB expects.
       * When detaching from the non-followed process, this makes no
       * difference but when GDB is asked to attach to both processes the
       * flow differs from the expectations and the inferiors get out of sync
       * leading into an unrecoverable mess.
       *
       * So we make sure that detach-on-fork is disabled and shadow the
       * original control variable so that future changes only change the local
       * variable while the GDB setting stays immutable.
       * This way 'show detach-on-fork' will always return 'on' and GDB will
       * act upon it.
       *
       * This MUST be removed once detach-on-fork handling works properly.
       */
/* CUDA: Disabled for now. */
#if 0
      if (nto_detach_fork == 0)
	{
	  struct cmd_list_element *prev_cmd = setlist;
	  struct cmd_list_element *cmd=NULL;

	  // get the setter for detach-on-fork. Yes this will fail if
	  // detach-on-fork is the first entry in the setter list but in
	  // infrun.c a lot of setshows are defined before detach-on-fork
	  while (prev_cmd->next != NULL)
	    {
	      if (strcmp (prev_cmd->next->name, "detach-on-fork") == 0)
		{
		  cmd=prev_cmd->next;
		  break;
		}
	      prev_cmd=prev_cmd->next;
	    }

	  if (cmd != NULL)
	    {
	      if (cmd->var.get () == false)
		{
		  warning("Disabling detach-on-fork is not supported for remote targets, re-enabling it!");
		  cmd->var.set (true);
		}
	      // replace the GDB control flag with our own in the setter only
	      cmd->var=&nto_detach_fork;
	      nto_detach_fork=1;
	      gdb_printf (gdb_stdlog, "Disabled 'set detach-on-fork' for remote targets\n");
	    }
	  else
	    {
	      error("Could not find 'set detach-on-fork' command!");
	    }
	}
#endif
    }
}

/* Perform remote attach.
 *
 * Use caller provided recv as the reply may be used by the caller. */

static int
nto_attach_only (const int pid, DScomm_t *const recv)
{
  DScomm_t tran;
  nto_trace (0) ("nto_attach_only(%i)\n", pid);

  nto_send_init (&tran, DStMsg_attach, 0, SET_CHANNEL_DEBUG);
  tran.pkt.attach.pid = pid;
  tran.pkt.attach.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.attach.pid, 4,
            nto_byte_order);
  nto_send_recv (&tran, recv, sizeof (tran.pkt.attach), 0);

  if (recv->pkt.hdr.cmd != DSrMsg_okdata)
    {
      error (_("Failed to attach"));
      return 0;
    }
  return 1;
}

/**
 * detach debug channel from pid
 *
 * this just frees the process on the target and does not affect
 * GDB's handling/knowledge of the process. Mainly used to clean up
 * after a fork()
 */
static int
nto_detach_only (const int pid) {
  DScomm_t tran, recv;
  nto_trace (0) ("nto_detach_only(%i)\n", pid);

  nto_send_init (&tran, DStMsg_detach, 0, SET_CHANNEL_DEBUG);
  tran.pkt.detach.pid = pid;
  tran.pkt.detach.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.detach.pid, 4, nto_byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.detach), 1);

  return (recv.pkt.hdr.cmd == DSrMsg_okdata);
}

/* Attaches to a process on the target side.  Arguments are as passed
   to the `attach' command by the user.  This routine can be called
   when the target is not on the target-stack, if the target_can_run
   routine returns 1; in that case, it must push itself onto the stack.
   Upon exit, the target should be ready for normal operations, and
   should be ready to deliver the status of the process immediately
   (without waiting) to an upcoming target_wait call.  */
void
pdebug_target::attach (const char *args, int from_tty)
{
  ptid_t ptid;
  struct inferior *inf;
  struct nto_inferior_data *inf_data;
  DScomm_t tran, *recv = &current_session->recv;

  /* if someone wants to attach, throw away any old session first -
   * even if attaching fails lateron
   */
  if (inferior_ptid != null_ptid)
    nto_semi_init ();

  nto_trace (0) ("pdebug_attach(args '%s', from_tty %d)\n",
       args ? args : "(null)", from_tty);

  if (!args)
    error_no_arg ("process-id to attach");

  ptid = ptid_t (atoi (args),1,0);

  objfile *symfile_objfile = current_program_space->symfile_object_file;

  if (symfile_objfile != NULL) {
    exec_file_attach (symfile_objfile->original_name, from_tty);
  }
  else {
    const int pid = ptid.pid();
    struct dspidlist *pidlist = (struct dspidlist *)recv->pkt.okdata.data;

    /* Look for the binary executable name */
    nto_send_init (&tran, DStMsg_pidlist, DSMSG_PIDLIST_SPECIFIC, SET_CHANNEL_DEBUG);
    tran.pkt.pidlist.pid = EXTRACT_UNSIGNED_INTEGER (&pid, 4, nto_byte_order);
    tran.pkt.pidlist.tid = 0;
    nto_send_recv (&tran, recv, sizeof (tran.pkt.pidlist), 0);
    if (only_session.recv.pkt.hdr.cmd == DSrMsg_okdata) {
      exec_file_attach (pidlist->name, from_tty);
  }
  }

  if (from_tty) {
    printf_unfiltered ("Attaching to %s\n", nto_pid_to_str (ptid).c_str());
    gdb_flush (gdb_stdout);
  }

  if (!nto_attach_only (ptid.pid(), recv))
    return;

  gdb_flush (gdb_stdout);

  /* Hack this in here, since we will bypass the notify.  */
  current_session->cputype =
  EXTRACT_SIGNED_INTEGER (&recv->pkt.notify.un.pidload.cputype, 2,
    nto_byte_order);
  current_session->cpuid =
  EXTRACT_SIGNED_INTEGER (&recv->pkt.notify.un.pidload.cpuid, 4,
    nto_byte_order);

  inf = current_inferior ();
  inf->attach_flag = true;
  inf->removable = false;

  /* Remove LD_LIBRARY_PATH. In the future, we should fetch
   * it from the target and setup correctly prepended with
   * QNX_TARGET/<CPU> */
  inf->environment.set("LD_LIBRARY_PATH", "");

  inferior_ptid  = ptid;
  inferior_appeared (inf, inferior_ptid.pid());
  /* add and switch to the current thread */
  thread_info *tp = add_thread (this, inferior_ptid);
  switch_to_thread_no_regs (tp);
  /* update all thread information */
  update_thread_list();

 /* NYI: add symbol information for process.  */
  /* Turn the PIDLOAD into a STOPPED notification so that when gdb
     calls nto_wait, we won't cycle around.
     recv refers to a global structure here! */
  recv->pkt.hdr.cmd = DShMsg_notify;
  recv->pkt.hdr.subcmd = DSMSG_NOTIFY_STOPPED;
  recv->pkt.notify.pid = ptid.pid();
  recv->pkt.notify.tid = ptid.lwp();
  recv->pkt.notify.pid = EXTRACT_SIGNED_INTEGER (&recv->pkt.notify.pid, 4,
                 nto_byte_order);
  recv->pkt.notify.tid = EXTRACT_SIGNED_INTEGER (&recv->pkt.notify.tid, 4,
                 nto_byte_order);

  inf_data = nto_inferior_data (inf);
  inf_data->has_execution = 1;
  inf_data->has_stack = 1;
  inf_data->has_registers = 1;
  inf_data->has_memory = 1;
}

void
pdebug_target::post_attach (int pid)
{
  nto_trace (0) ("%s pid:%d\n", __func__, pid);
  if (current_program_space->exec_bfd () != NULL)
    solib_create_inferior_hook(0);
}

/* This takes a program previously attached to and detaches it.  After
   this is done, GDB can be used to debug some other program.  We
   better not have left any breakpoints in the target program or it'll
   die when it hits one.  */
void
pdebug_target::detach (inferior *inf, int from_tty)
{
  gdb_assert (inf != NULL);

  nto_trace (0) ("pdebug_detach(%d from_tty %d)\n", inf->pid, from_tty);

  if (from_tty)
    {
      const char *exec_file = get_exec_file (0);
      printf_unfiltered ("Detaching from program: %s %d\n", exec_file==NULL?"":exec_file,
       inf->pid);
      gdb_flush (gdb_stdout);
    }

  nto_detach_only(inf->pid);
  detach_inferior(inf);
  mourn_inferior ();
  inferior_ptid = null_ptid;
}

/* implementation of disconnect so mi-target-disconnect may work */
void
pdebug_target::disconnect ( const char *args, int from_tty)
{
  if (args) {
    error (_("Argument given to \"disconnect\" while remote debugging."));
  }

  /* clean up current inferior */
  mourn_inferior ();

  /* Make sure we unpush even the extended remote targets.  Calling
     nto_mourn_inferior won't unpush, and remote_mourn won't
     unpush if there is more than one inferior left. unpush calls close()
     on the target, so the connection will be cleaned up */
  remote_unpush_target (this);

  if (from_tty) {
    gdb_puts ("Ending remote debugging.\n");
  }
}

/* Tell the remote machine to resume.  */
void
pdebug_target::resume (ptid_t ptid, int step,
      enum gdb_signal sig)
{
  DScomm_t tran, *const recv = &current_session->recv;
  int signo;
  const int runone = ptid.lwp() > 0;
  unsigned sizeof_pkt;
  ptid_t restore = inferior_ptid;

  nto_trace (0) ("pdebug_resume(pid %d, tid %ld, step %d, sig %d)\n",
     ptid.pid(), ptid.lwp(),
     step, nto_gdb_signal_to_target (target_gdbarch (), sig));

  if (inferior_ptid == null_ptid)
    return;

  gdb_assert (inferior_ptid.pid() == current_inferior ()->pid);

  if (!nto_set_thread_alive (ptid))
    {
      error("Process %d has no active threads!", ptid.pid() );
    }

  /* The HandleSig stuff is part of the new protover 0.1, but has not
     been implemented in all pdebugs that reflect that version.  If
     the HandleSig comes back with an error, then revert to protover 0.0
     behaviour, regardless of actual protover.
     The handlesig msg sends the signal to pass, and a char array
     'signals', which is the list of signals to notice.  */
  nto_send_init (&tran, DStMsg_handlesig, 0, SET_CHANNEL_DEBUG);
  tran.pkt.handlesig.sig_to_pass
    = nto_gdb_signal_to_target (target_gdbarch (), sig);
  tran.pkt.handlesig.sig_to_pass =
    EXTRACT_SIGNED_INTEGER (&tran.pkt.handlesig.sig_to_pass, 4, nto_byte_order);
  for (signo = 0; signo < QNXNTO_NSIG; signo++)
    {
      if (signal_stop_state (nto_gdb_signal_from_target (
           target_gdbarch (), signo)) == 0
    && signal_print_state (nto_gdb_signal_from_target (
         target_gdbarch (), signo)) == 0
    && signal_pass_state (nto_gdb_signal_from_target (
        target_gdbarch (), signo)) == 1)
  {
    tran.pkt.handlesig.signals[signo] = 0;
  }
      else
  {
    tran.pkt.handlesig.signals[signo] = 1;
  }
    }
  nto_send_recv (&tran, recv, sizeof (tran.pkt.handlesig), 0);
  if (recv->pkt.hdr.cmd == DSrMsg_err)
    if (sig != GDB_SIGNAL_0)
      {
  nto_send_init (&tran, DStMsg_kill, 0, SET_CHANNEL_DEBUG);
  tran.pkt.kill.signo = nto_gdb_signal_to_target (target_gdbarch (), sig);
  tran.pkt.kill.signo =
    EXTRACT_SIGNED_INTEGER (&tran.pkt.kill.signo, 4, nto_byte_order);
  nto_send_recv (&tran, recv, sizeof (tran.pkt.kill), 1);
      }

  nto_send_init (&tran, DStMsg_run, (step || runone) ? DSMSG_RUN_COUNT
           : DSMSG_RUN, SET_CHANNEL_DEBUG);
  tran.pkt.run.step.count = 1;
  tran.pkt.run.step.count =
  EXTRACT_UNSIGNED_INTEGER (&tran.pkt.run.step.count, 4, nto_byte_order);
      sizeof_pkt = sizeof (tran.pkt.run);
  nto_send_recv (&tran, recv, sizeof_pkt, 1);

  /* switch back to original inferior if needed */
  if (restore != inferior_ptid)
    {
      switch_to_thread (this, restore);
    }
}

static void (*ofunc) (int);
#ifndef __MINGW32__
static void (*ofunc_alrm) (int);
#endif

/* Yucky but necessary globals used to track state in nto_wait() as a
   result of things done in nto_interrupt(), nto_interrupt_twice(),
   and nto_interrupt_retry().  */
static sig_atomic_t SignalCount = 0;  /* Used to track ctl-c retransmits.  */
static sig_atomic_t InterruptedTwice = 0;  /* Set in nto_interrupt_twice().  */
static sig_atomic_t WaitingForStopResponse = 0;  /* Set in nto_interrupt(), cleared in nto_wait().  */

#define QNX_TIMER_TIMEOUT 5
#define QNX_CTL_C_RETRIES 3

static void
nto_interrupt_retry (int signo)
{
  SignalCount++;
  if (SignalCount >= QNX_CTL_C_RETRIES)  /* Retry QNX_CTL_C_RETRIES times after original transmission.  */
    {
      printf_unfiltered
  ("CTL-C transmit - 3 retries exhausted.  Ending debug session.\n");
      WaitingForStopResponse = 0;
      SignalCount = 0;
      target_mourn_inferior (inferior_ptid);
      //deprecated_throw_reason (RETURN_QUIT);
      quit ();
    }
  else
    {
      nto_interrupt (SIGINT);
    }
}


/* Ask the user what to do when an interrupt is received.  */
static void
nto_interrupt_query (void)
{
  alarm (0);
  signal (SIGINT, ofunc);
#ifndef __MINGW32__
  signal (SIGALRM, ofunc_alrm);
#endif
  target_terminal::ours ();
  InterruptedTwice = 0;

  if (query
      ("Interrupted while waiting for the program.\n Give up (and stop debugging it)? "))
    {
      SignalCount = 0;
      target_mourn_inferior (inferior_ptid);
      //deprecated_throw_reason (RETURN_QUIT);
      quit ();
    }
  target_terminal::inferior ();
#ifndef __MINGW32__
  signal (SIGALRM, nto_interrupt_retry);
#endif
  signal (SIGINT, nto_interrupt_twice);
  alarm (QNX_TIMER_TIMEOUT);
}


/* The user typed ^C twice.  */
static
void nto_interrupt_twice (int signo)
{
  InterruptedTwice = 1;
}

/* Send ^C to target to halt it.  Target will respond, and send us a
   packet.  */

/* GP - Dec 21, 2000.  If the target sends a NotifyHost at the same time as
   GDB sends a DStMsg_stop, then we would get into problems as both ends
   would be waiting for a response, and not the sent messages.  Now, we put
   the pkt and set the global flag 'WaitingForStopResponse', and return.
   This then goes back to the the main loop in nto_wait() below where we
   now check against the debug message received, and handle both.
   All retries of the DStMsg_stop are handled via SIGALRM and alarm(timeout).  */
static void
nto_interrupt (int signo)
{
  DScomm_t tran;

  nto_trace (0) ("nto_interrupt(signo %d)\n", signo);

  /* If this doesn't work, try more severe steps.  */
  signal (signo, nto_interrupt_twice);
#ifndef __MINGW32__
  signal (SIGALRM, nto_interrupt_retry);
#endif

  WaitingForStopResponse = 1;

  nto_send_init (&tran, DStMsg_stop, DSMSG_STOP_PIDS, SET_CHANNEL_DEBUG);
  putpkt (&tran, sizeof (tran.pkt.stop));

  /* Set timeout.  */
  alarm (QNX_TIMER_TIMEOUT);
}

/* Wait until the remote machine stops, then return,
   storing status in STATUS just as `wait' would.
   Returns "pid".  */
/* TODO: check options.. */
ptid_t
pdebug_target::wait (ptid_t ptid, struct target_waitstatus *status, target_wait_flags options)
{
  DScomm_t *const recv = &current_session->recv;
  ptid_t returned_ptid = ptid_t(current_inferior ()->pid, 1);

  nto_trace (0) ("pdebug_wait pid %d, inferior pid %d tid %ld\n",
     ptid.pid(), inferior_ptid.pid(), ptid.lwp());

  status->set_stopped (GDB_SIGNAL_0);

  nto_inferior_data (NULL)->stopped_flags = 0;

  if (recv->pkt.hdr.cmd != DShMsg_notify)
    {
      int len;
      char waiting_for_notify;

      waiting_for_notify = 1;
      SignalCount = 0;
      InterruptedTwice = 0;

      ofunc = (void (*)(int)) signal (SIGINT, nto_interrupt);
#ifndef __MINGW32__
      ofunc_alrm = (void (*)(int)) signal (SIGALRM, nto_interrupt_retry);
#endif
      for (;;)
  {
#ifdef NVIDIA_BUGFIX
    if (current_session->pending.empty ())
      {
#endif
    len = getpkt (recv, 1);
    if (len < 0)    /* Error - probably received MSG_NAK.  */
      {
        if (WaitingForStopResponse)
    {
      /* We do not want to get SIGALRM while calling it's handler
         the timer is reset in the handler.  */
      alarm (0);
#ifndef __MINGW32__
      nto_interrupt_retry (SIGALRM);
#else
      nto_interrupt_retry (0);
#endif
      continue;
    }
        else
    {
      /* Turn off the alarm, and reset the signals, and return.  */
      alarm (0);
      signal (SIGINT, ofunc);
#ifndef __MINGW32__
      signal (SIGALRM, ofunc_alrm);
#endif
      warning(" wait got an alarm!");
      return null_ptid;
    }
      }
#ifdef NVIDIA_BUGFIX
      }
    else
      {
	*recv = current_session->pending.front ();
	current_session->channelrd = recv->pkt.hdr.channel;
	current_session->pending.pop ();
      }
#endif
    if (current_session->channelrd == SET_CHANNEL_TEXT)
      nto_incoming_text (&recv->text, len);
    else      /* DEBUG CHANNEL.  */
      {
        recv->pkt.hdr.cmd &= ~DSHDR_MSG_BIG_ENDIAN;
        /* If we have sent the DStMsg_stop due to a ^C, we expect
           to get the response, so check and clear the flag
           also turn off the alarm - no need to retry,
           we did not lose the packet.  */
        if ((WaitingForStopResponse) && (recv->pkt.hdr.cmd == DSrMsg_ok))
    {
      WaitingForStopResponse = 0;
      status->set_stopped (GDB_SIGNAL_INT);
      alarm (0);
      if (!waiting_for_notify)
        break;
    }
        /* Else we get the Notify we are waiting for.  */
        else if (recv->pkt.hdr.cmd == DShMsg_notify)
    {
      DScomm_t tran;

      waiting_for_notify = 0;
      /* Send an OK packet to acknowledge the notify.  */
      nto_send_init (&tran, DSrMsg_ok, recv->pkt.hdr.mid,
         SET_CHANNEL_DEBUG);
      tran.pkt.hdr.mid = recv->pkt.hdr.mid;
      putpkt (&tran, sizeof (tran.pkt.ok));

      returned_ptid = nto_parse_notify (recv, status);

      if (!WaitingForStopResponse)
        break;
    }
      }
  }
      gdb_flush (gdb_stdtarg);
      gdb_flush (gdb_stdout);
      alarm (0);

      /* Hitting Ctl-C sends a stop request, a second ctl-c means quit,
         so query here, after handling the results of the first ctl-c
         We know we were interrupted twice because the yucky global flag
         'InterruptedTwice' is set in the handler, and cleared in
         nto_interrupt_query().  */
      if (InterruptedTwice)
	nto_interrupt_query ();

      signal (SIGINT, ofunc);
#ifndef __MINGW32__
      signal (SIGALRM, ofunc_alrm);
#endif
    }

  recv->pkt.hdr.cmd = DSrMsg_ok;  /* To make us wait the next time.  */
  nto_trace(0)("pdebug_wait: returning %s\n", nto_pid_to_str(returned_ptid).c_str());
  return returned_ptid;
}

ptid_t
pdebug_target::nto_parse_notify (const DScomm_t * const recv, struct target_waitstatus *status)
{
  int pid, tid;
  CORE_ADDR stopped_pc = 0;
  struct inferior *inf;
  struct nto_inferior_data *inf_data;

  inf = current_inferior ();

  gdb_assert(inf != NULL);

  inf_data = nto_inferior_data (inf);

  gdb_assert(inf_data != NULL);

  nto_trace(0) (
      "nto_parse_notify(status) - %s\n",
      recv->pkt.hdr.subcmd <= DSMSG_NOTIFY_EXEC ?
	  _DSMSGS[recv->pkt.hdr.subcmd] : "DSMSG_UNKNOWN");

  pid = EXTRACT_SIGNED_INTEGER(&recv->pkt.notify.pid, 4, nto_byte_order);
  tid = EXTRACT_SIGNED_INTEGER(&recv->pkt.notify.tid, 4, nto_byte_order);

  switch (recv->pkt.hdr.subcmd)
    {
    /* process death */
    case DSMSG_NOTIFY_PIDUNLOAD:
      {
      const int32_t * const pstatus = &recv->pkt.notify.un.pidunload_v3.status;
      const int faulted = recv->pkt.notify.un.pidunload_v3.faulted;

      if (faulted)
	{
	  auto sig = nto_gdb_signal_from_target (
	      target_gdbarch (),
	      EXTRACT_SIGNED_INTEGER(pstatus, 4, nto_byte_order));
	  if (sig)
	    status->set_signalled (sig); /* Abnormal death.  */
	  else
	    status->set_exited (sig); /* Normal death.  */
	}
      else
	{
	  /* Normal death, possibly with exit value.  */
	  status->set_exited (EXTRACT_SIGNED_INTEGER(pstatus, 4,
						     nto_byte_order));
	}
      }
      inf_data->has_execution = 0;
      inf_data->has_stack = 0;
      inf_data->has_registers = 0;
      inf_data->has_memory = 0;
      break;

    /* stepped on a breakpoint */
    case DSMSG_NOTIFY_BRK:
      inf_data->stopped_flags = EXTRACT_UNSIGNED_INTEGER(
	      &recv->pkt.notify.un.brk.flags, 4, nto_byte_order);
      stopped_pc = EXTRACT_UNSIGNED_INTEGER(&recv->pkt.notify.un.brk.ip,
						8, nto_byte_order);
      inf_data->stopped_pc = stopped_pc;
      status->set_stopped (GDB_SIGNAL_TRAP);
      break;

    /* took a step */
    case DSMSG_NOTIFY_STEP:
      stopped_pc = EXTRACT_UNSIGNED_INTEGER(
	      &recv->pkt.notify.un.step.ip, sizeof(uint64_t), nto_byte_order);
      inf_data->stopped_pc = stopped_pc;
      status->set_stopped (GDB_SIGNAL_TRAP);
      break;

    /* received a signal */
    case DSMSG_NOTIFY_SIGEV:
#ifdef NVIDIA_CUDA_GDB
  {
    auto sig = nto_gdb_signal_from_target (
	      target_gdbarch (),
	      EXTRACT_SIGNED_INTEGER(&recv->pkt.notify.un.sigev.signo, 4,
				     nto_byte_order));
    /* CUDA - We need to check if we have received an urgent message (aka sync event).
       This happens when we receive a SIGEMT or SIGILL. We switch between the two to
       prevent QNX from killing the app due to the same signal being received twice in
       a row.
       FIXME: Currently we will get signaled before init has finished. We are unable
       to detect that our event notification was received with cuda_notification_received ().
       As a result, we always assume SIGEMT or SIGILL are notifications. This is not
       desirable. */
    if (sig == GDB_SIGNAL_EMT || sig == GDB_SIGNAL_ILL)
      status->set_stopped (sig);
    else
      status->set_signalled (sig);
  }
#else
      status->set_signalled (nto_gdb_signal_from_target (
	      target_gdbarch (),
	      EXTRACT_SIGNED_INTEGER(&recv->pkt.notify.un.sigev.signo, 4,
				     nto_byte_order)));
#endif
      break;

    /* a new process was created */
    case DSMSG_NOTIFY_PIDLOAD:
      current_session->cputype = EXTRACT_SIGNED_INTEGER(
	      &recv->pkt.notify.un.pidload.cputype, sizeof(uint16_t), nto_byte_order);
      current_session->cpuid = EXTRACT_SIGNED_INTEGER(
	      &recv->pkt.notify.un.pidload.cpuid, sizeof(uint32_t), nto_byte_order);

      inf_data->has_execution = 1;
      inf_data->has_stack = 1;
      inf_data->has_registers = 1;
      inf_data->has_memory = 1;
      status->set_loaded ();

      break;

    /* a new thread was created */
    case DSMSG_NOTIFY_TIDLOAD:
      {
	if (nto_stop_on_thread_events)
	  status->set_stopped (GDB_SIGNAL_0);
	else
	  status->set_spurious ();
	tid = EXTRACT_UNSIGNED_INTEGER(
		&recv->pkt.notify.un.thread_event.tid, sizeof(int32_t),
		nto_byte_order);

	nto_trace(0) ("New thread event: tid %d\n", tid);

	nto_add_thread(this, pid, tid);

	if (status->kind () == TARGET_WAITKIND_SPURIOUS)
	  tid = inferior_ptid.lwp ();
      }
      break;

    /* thread death */
    case DSMSG_NOTIFY_TIDUNLOAD:
      {
	ptid_t cur = ptid_t (pid, tid, 0);
	const int32_t * const ptid = &recv->pkt.notify.un.thread_event.tid;
	const int tid_exited = EXTRACT_SIGNED_INTEGER(ptid, 4, nto_byte_order);

	nto_trace(0) ("Thread destroyed: tid: %d active: %d\n", tid_exited,
		      tid);

	if (nto_stop_on_thread_events)
	  status->set_stopped (GDB_SIGNAL_0);
	else
	  status->set_spurious ();
	/* Must determine an alive thread for this to work. */
	if (inferior_ptid != cur)
	  {
	    switch_to_thread (this, cur);
	  }
      }
      break;

    /* process forked */
    case DSMSG_NOTIFY_FORK:
      {
        int32_t child_pid = recv->pkt.notify.un.fork_event.pid;
        nto_trace(0)("fork - parent: %d child: %d\n", pid, child_pid);
        inf_data->child_pid = EXTRACT_SIGNED_INTEGER(&child_pid, 4, nto_byte_order);
	status->set_forked (ptid_t(child_pid, 1, 0));

        /* immediately attach to the child so it doesn't run away between handling the
         * fork notification and follow_fork() on slow targets this is not needed as
         * GDB is fast enough to attach in time on it's own.
         * Do not yet create an inferior as that will happen automatically */
        DScomm_t attrecv;
        nto_attach_only (child_pid, &attrecv);
      }
    break;

    /* process called exec() */
    case DSMSG_NOTIFY_EXEC:
      /* Notification format: pidload. */
      nto_trace(0)("DSMSG_NOTIFY_EXEC %d, %s\n", pid, recv->pkt.notify.un.pidload.name);
      status->set_execd (make_unique_xstrdup ( recv->pkt.notify.un.pidload.name ));
      break;

    /* DLL changes - no explicit action, just continue */
    case DSMSG_NOTIFY_DLLLOAD:
    case DSMSG_NOTIFY_DLLUNLOAD:
      status->set_spurious ();
      break;

    /* process stopped */
    case DSMSG_NOTIFY_STOPPED:
      status->set_stopped (GDB_SIGNAL_0);
      break;

    default:
      warning ("Unexpected notify type %d", recv->pkt.hdr.subcmd);
      break;
    }

  nto_trace(0) ("  current inferior: %d pid: %d\n", current_inferior ()->pid, pid);

  /* set the current context to the inferior that received the signal */

  static inferior *newinf = find_inferior_pid(this, pid);
  if (newinf == NULL)
    {
      nto_trace(0) (" create new inferior for pid %d\n", pid);
      newinf = nto_add_inferior(this, pid);
    }

  /* No thread is set explicitly, go back to the original thread */
  if (tid == 0)
    {
      tid=inferior_ptid.lwp ();
      /* No previous thread, fetch an existing thread */
      if (tid == 0)
	tid = nto_thread_alive(ptid_t(pid, 1, 0));
    }

  /* the process must have an existing thread, if not we're in trouble */
  if (tid == -1)
    error("Process %d has no threads!", pid);

  nto_trace(0) ("  nto_parse_notify end: pid=%d, tid=%d  ip=%s\n", pid,
		tid, paddress (target_gdbarch (), stopped_pc));

  return ptid_t(pid, tid);
}

static unsigned nto_get_cpuflags (void)
{
  static int read_done = 0;
  static unsigned cpuflags = 0;
  DScomm_t tran, recv;


  if (!read_done)
    {
      read_done = 1;

      nto_send_init (&tran, DStMsg_cpuinfo, 0, SET_CHANNEL_DEBUG);
      nto_send_recv (&tran, &recv, sizeof (tran.pkt.cpuinfo), 1);

      if (recv.pkt.hdr.cmd != DSrMsg_err)
  {
    struct dscpuinfo foo;
    memcpy (&foo, recv.pkt.okdata.data, sizeof (struct dscpuinfo));
    cpuflags = EXTRACT_SIGNED_INTEGER (&foo.cpuflags, 4, nto_byte_order);
  }
    }
  return cpuflags;
}

/* Fetch the regset, returning true if successful.  If supply is true,
   then supply these registers to gdb as well.  */
static int
fetch_regs (struct regcache *regcache, int regset, int supply)
{
  int len;
  int rlen;
  DScomm_t tran, recv;

  len = nto_register_area (regset, nto_get_cpuflags ());
  if (len < 1)
    return 0;

  /* Ugly hack to keep i386-nto-tdep.c cleaner. */
  if (gdbarch_bfd_arch_info (target_gdbarch ()) != NULL
      && strcmp (gdbarch_bfd_arch_info (target_gdbarch ())->arch_name, "i386")
   == 0
      && regset == NTO_REG_FLOAT && len > 512
      && current_session->target_proto_major == 0
      && current_session->target_proto_minor < 5)
    {
      /* Pre-avx support context was at most 512 bytes. With avx,
       * it can be more. Old pdebug (pre 0.5) would EINVAL if kernel
       * says regset is different from gdb's idea; older kernel also
       * returned exactly 512. */
      len = 512;
    }

  nto_send_init (&tran, DStMsg_regrd, regset, SET_CHANNEL_DEBUG);
  tran.pkt.regrd.offset = 0;  /* Always get whole set.  */
  tran.pkt.regrd.size = EXTRACT_SIGNED_INTEGER (&len, 2,
            nto_byte_order);

  rlen = nto_send_recv (&tran, &recv, sizeof (tran.pkt.regrd), 0);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    return 0;

  if (supply)
    nto_supply_regset (regcache, regset, recv.pkt.okdata.data, rlen);
  return 1;
}

/* Read register REGNO, or all registers if REGNO == -1, from the contents
   of REGISTERS.  */
void
pdebug_target::fetch_registers (struct regcache *regcache, int regno)
{
  int regset;
  ptid_t ptid=regcache->ptid ();

  nto_trace (0) ("pdebug_fetch_registers(regcache %p ,regno %d) for %s\n",
     regcache, regno, nto_pid_to_str(ptid).c_str());

  if (ptid == null_ptid)
    {
      nto_trace (0) ("  ptid is null_ptid, can not fetch registers\n");
      return;
    }

  if (!nto_set_thread (ptid))
    return;

  if (regno == -1)
    {        /* Get all regsets.  */
      for (regset = NTO_REG_GENERAL; regset < NTO_REG_END; regset++)
  {
    fetch_regs (regcache, regset, 1);
  }
    }
  else
    {
      regset = nto_regset_id (regno);
      fetch_regs (regcache, regset, 1);
    }
}

/* Prepare to store registers.  Don't have to do anything.  */
void
pdebug_target::prepare_to_store (struct regcache *regcache)
{
   nto_trace (0) ("pdebug_prepare_to_store()\n");
}


/* Store register REGNO, or all registers if REGNO == -1, from the contents
   of REGISTERS.  */
void
pdebug_target::store_registers (struct regcache *regcache, int regno)
{
  int len, regset, regno_regset;
  DScomm_t tran, recv;
  ptid_t ptid=regcache->ptid ();

  nto_trace (0) ("pdebug_store_registers(regno %d)\n", regno);

  gdb_assert (ptid != null_ptid);

  if (!nto_set_thread (ptid))
    return;

  regno_regset = nto_regset_id (regno);

  for (regset = NTO_REG_GENERAL; regset < NTO_REG_END; regset++)
    {
      if (regno_regset != NTO_REG_END && regno_regset != regset)
  continue;

      len = nto_register_area (regset, nto_get_cpuflags ());
      if (len < 1)
  continue;

      nto_send_init (&tran, DStMsg_regwr, regset, SET_CHANNEL_DEBUG);
      tran.pkt.regwr.offset = 0;
      if (nto_regset_fill (regcache, regset, tran.pkt.regwr.data, len) == -1)
  continue;

      nto_send_recv (&tran, &recv, offsetof (DStMsg_regwr_t, data) + len, 1);
    }
}

/* Use of the data cache *used* to be disabled because it loses for looking at
   and changing hardware I/O ports and the like.  Accepting `volatile'
   would perhaps be one way to fix it.  Another idea would be to use the
   executable file for the text segment (for all SEC_CODE sections?
   For all SEC_READONLY sections?).  This has problems if you want to
   actually see what the memory contains (e.g. self-modifying code,
   clobbered memory, user downloaded the wrong thing).

   Because it speeds so much up, it's now enabled, if you're playing
   with registers you turn it off (set remotecache 0).  */

/* Write memory data directly to the remote machine.
   This does not inform the data cache; the data cache uses this.
   MEMADDR is the address in the remote memory space.
   MYADDR is the address of the buffer in our space.
   LEN is the number of bytes.

   Returns number of bytes transferred, or 0 for error.  */
static enum target_xfer_status
nto_write_bytes (CORE_ADDR memaddr, const gdb_byte *myaddr, int len,
     ULONGEST *const xfered_len)
{
  long long addr;
  DScomm_t tran, recv;

  nto_trace (0) ("nto_write_bytes(to %s, from %p, len %d)\n",
     paddress (target_gdbarch (), memaddr), myaddr, len);

  /* NYI: need to handle requests bigger than largest allowed packet.  */
  nto_send_init (&tran, DStMsg_memwr, 0, SET_CHANNEL_DEBUG);
  addr = memaddr;
  tran.pkt.memwr.addr = EXTRACT_UNSIGNED_INTEGER (&addr, 8,
              nto_byte_order);
  memcpy (tran.pkt.memwr.data, myaddr, len);
  nto_send_recv (&tran, &recv, offsetof (DStMsg_memwr_t, data) + len, 0);

  switch (recv.pkt.hdr.cmd)
    {
    case DSrMsg_ok:
      *xfered_len = len;
      break;
    case DSrMsg_okstatus:
      *xfered_len = EXTRACT_SIGNED_INTEGER (&recv.pkt.okstatus.status, 4,
              nto_byte_order);
      break;
    default:
      return TARGET_XFER_E_IO;
    }

  if (*xfered_len < 0)
    return TARGET_XFER_E_IO;

  return (*xfered_len)? TARGET_XFER_OK : TARGET_XFER_EOF;
}

/* Read memory data directly from the remote machine.
   This does not use the data cache; the data cache uses this.
   MEMADDR is the address in the remote memory space.
   MYADDR is the address of the buffer in our space.
   LEN is the number of bytes.

   Returns number of bytes transferred, or 0 for error.  */
static enum target_xfer_status
nto_read_bytes (CORE_ADDR memaddr, gdb_byte *myaddr, int len,
    ULONGEST *const xfered_len)
{
  int rcv_len, tot_len, ask_len;
  long long addr;

  if (remote_debug)
    {
      printf_unfiltered ("nto_read_bytes(from %s, to %p, len %d)\n",
       paddress (target_gdbarch (), memaddr), myaddr, len);
    }

  tot_len = rcv_len = ask_len = 0;

  /* NYI: Need to handle requests bigger than largest allowed packet.  */
  do
    {
      DScomm_t tran, recv;
      nto_send_init (&tran, DStMsg_memrd, 0, SET_CHANNEL_DEBUG);
      addr = memaddr + tot_len;
      tran.pkt.memrd.addr = EXTRACT_UNSIGNED_INTEGER (&addr, 8,
                  nto_byte_order);
      ask_len =
  ((len - tot_len) >
   DS_DATA_MAX_SIZE) ? DS_DATA_MAX_SIZE : (len - tot_len);
      tran.pkt.memrd.size = EXTRACT_SIGNED_INTEGER (&ask_len, 2,
                nto_byte_order);
      rcv_len = nto_send_recv (&tran, &recv, sizeof (tran.pkt.memrd), 0) - sizeof (recv.pkt.hdr);
      if (rcv_len <= 0)
  break;
      if (recv.pkt.hdr.cmd == DSrMsg_okdata)
  {
    memcpy (myaddr + tot_len, recv.pkt.okdata.data, rcv_len);
    tot_len += rcv_len;
  }
      else
  break;
    }
  while (tot_len != len);

  *xfered_len = tot_len;

  return (tot_len? TARGET_XFER_OK : TARGET_XFER_EOF);
}

enum target_xfer_status
pdebug_target::xfer_partial (enum target_object object,
      const char *annex, gdb_byte *readbuf,
      const gdb_byte *writebuf, const ULONGEST offset,
      const ULONGEST len, ULONGEST *const xfered_len)
{
  const unsigned arch_len
    = gdbarch_bfd_arch_info (target_gdbarch ())->bits_per_word;

  if (arch_len != 32 && arch_len != 64)
    {
      return TARGET_XFER_E_IO;
    }

  if (object == TARGET_OBJECT_MEMORY)
    {
      if (readbuf != NULL)
  return nto_read_bytes (offset, readbuf, len, xfered_len);
      else if (writebuf != NULL)
  return nto_write_bytes (offset, writebuf, len, xfered_len);
    }
  else if (object == TARGET_OBJECT_AUXV
     && readbuf)
    {
      /* For 32-bit architecture, size of auxv_t is 8 bytes.  */
      const unsigned int sizeof_auxv_t = (arch_len == 32)? 8 : 16;
      const unsigned int sizeof_tempbuf = 20 * sizeof_auxv_t;
      int tempread = 0;
      gdb_byte *tempbuf = (gdb_byte *)alloca (sizeof_tempbuf);
      nto_procfs_info procfs_info;

      if (!tempbuf)
        return TARGET_XFER_E_IO;

      /* We first try to read auxv using initial stack.  The problem is, older
         pdebug-s don't support reading procfs_info.  */

      if (nto_read_procfsinfo (&procfs_info))
  {
    struct inferior *const inf = current_inferior ();
    struct nto_remote_inferior_data *inf_rdata;

    inf_rdata = nto_get_remote_inferior_data (inf);

    if (inf_rdata->auxv == NULL)
      {
        const CORE_ADDR initial_stack
    = EXTRACT_SIGNED_INTEGER (&procfs_info.initial_stack,
            arch_len / 8, nto_byte_order);

        inf_rdata->auxv = (gdb_byte *)xcalloc (1, sizeof_tempbuf);
        tempread = nto_read_auxv_from_initial_stack (initial_stack,
                 inf_rdata->auxv,
                 sizeof_tempbuf,
                 sizeof_auxv_t);
      }
    else
      {
        tempread = sizeof_tempbuf;
      }
    tempbuf = inf_rdata->auxv;
  }
      tempread = (tempread<len?tempread:len) - offset;
      memcpy (readbuf, tempbuf + offset, tempread);
      *xfered_len = tempread;
      return tempread? TARGET_XFER_OK : TARGET_XFER_EOF;
    }  /* TARGET_OBJECT_AUXV */
  else if (object == TARGET_OBJECT_SIGNAL_INFO
     && readbuf)
    {
      nto_procfs_status status;
      nto_siginfo_t siginfo;
      LONGEST mylen = len;

      if ((offset + mylen) > sizeof (nto_siginfo_t))
  {
    if (offset < sizeof (nto_siginfo_t))
      mylen = sizeof (nto_siginfo_t) - offset;
    else
      return TARGET_XFER_EOF;
  }

      if (!nto_read_procfsstatus (&status))
  return TARGET_XFER_E_IO;

      // does byte order translation
      nto_get_siginfo_from_procfs_status (&status, &siginfo);
      memcpy (readbuf, (gdb_byte *)&siginfo + offset, mylen);
      *xfered_len = len;
      return len? TARGET_XFER_OK : TARGET_XFER_EOF;
    }
  return this->beneath()->xfer_partial (object, annex, readbuf,
            writebuf, offset, len, xfered_len);
}

void
pdebug_target::files_info ( )
{
  nto_trace (0) ("pdebug_files_info( )\n" );

  gdb_puts ("Debugging a target via pdebug.\n");
}


static int
nto_kill_1 ( )
{
  DScomm_t tran, recv;

  nto_trace (0) ("nto_kill_1\n");

  if (inferior_ptid != null_ptid)
    {
      nto_send_init (&tran, DStMsg_kill, DSMSG_KILL_PID, SET_CHANNEL_DEBUG);
      tran.pkt.kill.signo = SIGKILL;
      tran.pkt.kill.signo = EXTRACT_SIGNED_INTEGER (&tran.pkt.kill.signo,
                4, nto_byte_order);
      nto_send_recv (&tran, &recv, sizeof (tran.pkt.kill), 0);

      nto_detach_only (inferior_ptid.pid());
    }

  return 0;
}

void
pdebug_target::kill ( )
{
  // struct target_waitstatus wstatus;
  // ptid_t ptid;

  nto_trace (0) ("pdebug_kill()\n");

  remove_breakpoints ();
  // get_last_target_status (&this, &ptid, &wstatus);

  /* Use catch_errors so the user can quit from gdb even when we aren't on
     speaking terms with the remote system.  */
  catch_errors( nto_kill_1 );

  mourn_inferior ( );

  return;
}

void
pdebug_target::mourn_inferior ( )
{
  struct inferior *inf = current_inferior ();
  struct nto_inferior_data *inf_data;
  struct nto_remote_inferior_data *inf_rdata;

  nto_trace (0) ("pdebug_mourn_inferior()\n");

  gdb_assert (inf != NULL);

  inf_data = nto_inferior_data (inf);

  gdb_assert (inf_data != NULL);

  inf_rdata = nto_get_remote_inferior_data (inf);

  xfree (inf_rdata->auxv);
  inf_rdata->auxv = NULL;

  nto_detach_only (inferior_ptid.pid());

  generic_mourn_inferior ();
  inf_data->has_execution = 0;
  inf_data->has_stack = 0;
  inf_data->has_registers = 0;
  inf_data->has_memory = 0;
}

static int
nto_fd_raw (int fd)
{
#ifndef __MINGW32__
  struct termios termios_p;

  if (tcgetattr (fd, &termios_p))
    return (-1);

  termios_p.c_cc[VMIN] = 1;
  termios_p.c_cc[VTIME] = 0;
  termios_p.c_lflag &= ~(ECHO | ICANON | ISIG | ECHOE | ECHOK | ECHONL);
  termios_p.c_oflag &= ~(OPOST);
  return (tcsetattr (fd, TCSADRAIN, &termios_p));
#else
  return 0;
#endif
}

void
pdebug_target::create_inferior (const char *exec_file, const std::string &args, char **env, int from_tty)
{
  DScomm_t tran, recv;
  unsigned argc;
  unsigned envc;
  char **start_argv, **argv;
  char **pargv;
  char *p;
  int fd;
  struct target_waitstatus status;
  const char *in, *out, *err;
  int errors = 0;
  struct inferior *const inf = current_inferior ();
  struct nto_inferior_data *inf_data;
  struct nto_remote_inferior_data *inf_rdata;

  gdb_assert (inf != NULL);

  inf_data = nto_inferior_data (inf);

  gdb_assert (inf_data != NULL);

  inf_rdata = nto_get_remote_inferior_data (inf);

  gdb_assert (inf_rdata != NULL);

  inf_data->stopped_flags = 0;

  remove_breakpoints ();

#ifdef NVIDIA_CUDA_GDB
  if (inf_rdata->remote_exe.length () == 0)
    {
      inf_rdata->remote_exe = current_session->remote_exe;
    }

  if (inf_rdata->remote_cwd.length () == 0)
    {
      inf_rdata->remote_cwd = current_session->remote_cwd;
    }

  if (inf_rdata->remote_exe.length ())
    {
      exec_file = inf_rdata->remote_exe.c_str ();
      gdb_printf (gdb_stdout, "Remote: %s\n", exec_file);
    }
#else
  if (inf_rdata->remote_exe == NULL && current_session->remote_exe != NULL)
    {
      inf_rdata->remote_exe = xstrdup (current_session->remote_exe);
    }

  if (inf_rdata->remote_cwd == NULL && current_session->remote_cwd != NULL)
    {
      inf_rdata->remote_cwd = xstrdup (current_session->remote_cwd);
    }

  if (inf_rdata->remote_exe && inf_rdata->remote_exe[0] != '\0')
    {
      exec_file = inf_rdata->remote_exe;
      gdb_printf (gdb_stdout, "Remote: %s\n", exec_file);
    }
#endif
  if (current_session->desc == NULL)
    pdebug_open ("pty", 0);

  if (inferior_ptid != null_ptid)
    nto_semi_init ();

  nto_trace (0) ("pdebug_create_inferior(exec_file '%s', args '%s', environ)\n",
       exec_file ? exec_file : "(null)",
       args.empty() ? "\"\"":args.c_str() );

  nto_send_init (&tran, DStMsg_env, DSMSG_ENV_CLEARENV, SET_CHANNEL_DEBUG);
  nto_send_recv (&tran, &recv, sizeof (DStMsg_env_t), 1);

  if (!current_session->inherit_env)
    {
      for (envc = 0; *env; env++, envc++)
        errors += !nto_send_env (*env);
      if (errors)
        warning ("Error(s) occured while sending environment variables.\n");
    }

#ifdef NVIDIA_CUDA_GDB
  if (inf_rdata->remote_cwd.length ())
    {
      nto_send_init (&tran, DStMsg_cwd, DSMSG_CWD_SET, SET_CHANNEL_DEBUG);
      strcpy ((char *)tran.pkt.cwd.path, inf_rdata->remote_cwd.c_str ());
      nto_send_recv (&tran, &recv, offsetof (DStMsg_cwd_t, path)
    + strlen ((const char *)tran.pkt.cwd.path) + 1, 1);
    }
#else
  if (inf_rdata->remote_cwd != NULL)
    {
      nto_send_init (&tran, DStMsg_cwd, DSMSG_CWD_SET, SET_CHANNEL_DEBUG);
      strcpy ((char *)tran.pkt.cwd.path, inf_rdata->remote_cwd);
      nto_send_recv (&tran, &recv, offsetof (DStMsg_cwd_t, path)
    + strlen ((const char *)tran.pkt.cwd.path) + 1, 1);
    }
#endif

  nto_send_init (&tran, DStMsg_env, DSMSG_ENV_CLEARARGV, SET_CHANNEL_DEBUG);
  nto_send_recv (&tran, &recv, sizeof (DStMsg_env_t), 1);

  pargv = buildargv (args.c_str());
  if (pargv == NULL)
    malloc_failure (0);
  start_argv = nto_parse_redirection (pargv, &in, &out, &err);

  if (in[0])
    {
      if ((fd = open (in, O_RDONLY)) == -1)
  perror (in);
      else
  nto_fd_raw (fd);
    }

  if (out[0])
    {
      if ((fd = open (out, O_WRONLY)) == -1)
  perror (out);
      else
  nto_fd_raw (fd);
    }

  if (err[0])
    {
      if ((fd = open (err, O_WRONLY)) == -1)
  perror (err);
      else
  nto_fd_raw (fd);
    }

  in = "@0";
  out = "@1";
  err = "@2";
  argc = 0;
  if (exec_file != NULL)
    {
      errors = !nto_send_arg (exec_file);
      /* Send it twice - first as cmd, second as argv[0]. */
      if (!errors)
  errors = !nto_send_arg (exec_file);

      if (errors)
  {
    error ("Failed to send executable file name.\n");
    goto freeargs;
  }
    }
  else if (*start_argv == NULL)
    {
      error ("No executable specified.");
      errors = 1;
      goto freeargs;
    }
  else
    {
      /* Send arguments (starting from index 1, argv[0] has already been
         sent above. */
	  objfile *symfile_objfile = current_program_space->symfile_object_file;
      if (symfile_objfile != NULL)
  exec_file_attach (symfile_objfile->original_name, 0);

      exec_file = *start_argv;

      errors = !nto_send_arg (*start_argv);

      if (errors)
  {
    error ("Failed to send argument.\n");
    goto freeargs;
  }
    }

  errors = 0;
  for (argv = start_argv; *argv && **argv; argv++, argc++)
    {
      errors |= !nto_send_arg (*argv);
    }

  if (errors)
    {
      error ("Error(s) encountered while sending arguments.\n");
    }

freeargs:
  freeargv (pargv);
  free (start_argv);
  if (errors)
    return;

  /* NYI: msg too big for buffer.  */
  if (current_session->inherit_env)
    nto_send_init (&tran, DStMsg_load, DSMSG_LOAD_DEBUG | DSMSG_LOAD_INHERIT_ENV,
       SET_CHANNEL_DEBUG);
  else
    nto_send_init (&tran, DStMsg_load, DSMSG_LOAD_DEBUG, SET_CHANNEL_DEBUG);

  p = tran.pkt.load.cmdline;

  tran.pkt.load.envc = 0;
  tran.pkt.load.argc = 0;

  strcpy (p, exec_file);
  p += strlen (p);
  *p++ = '\0';      /* load_file */

  strcpy (p, in);
  p += strlen (p);
  *p++ = '\0';      /* stdin */

  strcpy (p, out);
  p += strlen (p);
  *p++ = '\0';      /* stdout */

  strcpy (p, err);
  p += strlen (p);
  *p++ = '\0';      /* stderr */

  nto_send_recv (&tran, &recv, offsetof (DStMsg_load_t, cmdline) + p - tran.pkt.load.cmdline + 1,
      1);

  if (recv.pkt.hdr.cmd != DSrMsg_okdata)
    error ("Could not run executable on target!");

  /* Comes back as an DSrMsg_okdata, but it's really a DShMsg_notify. */
  inferior_ptid  = nto_parse_notify (&recv, &status);
  gdb_assert (status.kind () == TARGET_WAITKIND_LOADED);
  inferior_appeared (inf, inferior_ptid.pid());
  thread_info *tp = nto_add_thread (this, inferior_ptid.pid (), 1);
  switch_to_thread_no_regs (tp);
  inf->attach_flag = true;
  inf->removable = true;
}

int
pdebug_target::nto_insert_breakpoint (CORE_ADDR addr, gdb_byte *contents_cache)
{
  DScomm_t tran, recv;
  size_t sizeof_pkt;

  nto_trace (0) ("nto_insert_breakpoint(addr %s, contents_cache %p) pid:%d\n",
                 paddress (target_gdbarch (), addr), contents_cache, inferior_ptid.pid());

  nto_send_init (&tran, DStMsg_brk, DSMSG_BRK_EXEC, SET_CHANNEL_DEBUG);

  tran.pkt.brk.size = nto_breakpoint_size (addr);
  tran.pkt.brk.addr = EXTRACT_UNSIGNED_INTEGER (&addr, 8, nto_byte_order);
  sizeof_pkt = sizeof (tran.pkt.brk);

  nto_send_recv (&tran, &recv, sizeof_pkt, 0);
  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      nto_trace (0) ("  could not set breakpoint\n");
    }
  return recv.pkt.hdr.cmd == DSrMsg_err;
}

/* To be called from breakpoint.c through
  current_target.to_insert_breakpoint.  */
int
pdebug_target::insert_breakpoint (struct gdbarch *gdbarch,
        struct bp_target_info *bp_tg_inf)
{
  if( nto_force_hwbp )
    {
      // change breakpoint nature
      bp_tg_inf->kind=bp_loc_hardware_breakpoint;
      if (!insert_hw_breakpoint(gdbarch, bp_tg_inf))
	return 0;

      if (!query (_("Could not force HW breakpoint. Disable nto-force-hwbp?")))
	return 1;

      // revert breakpoint nature and reset force flag
      bp_tg_inf->kind=bp_loc_software_breakpoint;
      nto_force_hwbp = 0;
    }

  if (bp_tg_inf == 0)
    {
      internal_error(_("Target info invalid."));
    }

  /* Must select appropriate inferior. */
  if (!nto_set_thread_alive (inferior_ptid))
    {
      return 1;
    }

  bp_tg_inf->placed_address = bp_tg_inf->reqstd_address;

  return nto_insert_breakpoint ( bp_tg_inf->placed_address,
                bp_tg_inf->shadow_contents);
}


int
pdebug_target::nto_remove_breakpoint (CORE_ADDR addr, gdb_byte *contents_cache)
{
  DScomm_t tran, recv;

  nto_trace (0)  ("nto_remove_breakpoint(addr %s, contents_cache %p) (pid %d)\n",
                 paddress (target_gdbarch (), addr), contents_cache, inferior_ptid.pid());

  /* Must select appropriate inferior. */
  if (!nto_set_thread_alive (inferior_ptid))
    {
      return 1;
    }

  nto_send_init (&tran, DStMsg_brk, DSMSG_BRK_EXEC, SET_CHANNEL_DEBUG);
  tran.pkt.brk.addr = EXTRACT_UNSIGNED_INTEGER (&addr,
      sizeof(tran.pkt.brk.addr), nto_byte_order);
  tran.pkt.brk.size = -1;
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.brk), 0);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      nto_trace(1)("  errno=%d (%s)\n", recv.pkt.err.err, strerror(recv.pkt.err.err));
    }
  return recv.pkt.hdr.cmd == DSrMsg_err;
}

int
pdebug_target::remove_breakpoint (struct gdbarch *gdbarch,
        struct bp_target_info *bp_tg_inf,
        enum remove_bp_reason reason)
{
  nto_trace (0) ("%s ( bp_tg_inf=%p )\n", __func__, bp_tg_inf);

  if (bp_tg_inf == 0)
    {
      internal_error (_("Target info invalid."));
    }

  return nto_remove_breakpoint ( bp_tg_inf->placed_address,
        bp_tg_inf->shadow_contents);
}

int
pdebug_target::remove_hw_breakpoint (struct gdbarch *gdbarch,
        struct bp_target_info *bp_tg_inf)
{
  nto_trace (0) ("%s ( bp_tg_inf=%p )\n", __func__, bp_tg_inf);

  if (bp_tg_inf == 0)
    {
      internal_error (_("Target info invalid."));
    }

  return nto_remove_breakpoint (bp_tg_inf->placed_address,
        bp_tg_inf->shadow_contents);
}

#if defined(__CYGWIN__) || defined(__MINGW32__)
char
*slashify (const char *buf)
{
  int i = 0;
  static char *retv;
  retv=(char*)malloc(strlen(buf)+1);

  while (buf[i]) {
    if (buf[i] == '\\') {
      retv[i]='/';
    }
    else {
      retv[i]=buf[i];
    }
    i++;
  }
  retv[i]=0;
  return retv;
}
#endif

static void
upload_command (const char *args, int fromtty)
{
#if defined(__CYGWIN__)
  char cygbuf[PATH_MAX];
#endif
  int fd;
  int len;
  char buf[DS_DATA_MAX_SIZE];
  char *from, *to;
  char **argv;
  gdb::unique_xmalloc_ptr<char> filename_opened;

  // see source.c, openp and exec.c, file_command for more details.
  //
  struct inferior *inf = current_inferior ();
  struct nto_remote_inferior_data *inf_rdata;

  gdb_assert (inf != NULL);

  inf_rdata = nto_get_remote_inferior_data (inf);

  if (args == 0)
    {
      printf_unfiltered ("You must specify a filename to send.\n");
      return;
    }

#if defined(__CYGWIN__) || defined(__MINGW32__)
  /* todo do we really?
   * We need to convert back slashes to forward slashes for DOS
     style paths, else buildargv will remove them.  */
  char *wargs=slashify (args);
  argv = buildargv( wargs );
  free(wargs);
#else
  argv = buildargv (args);
#endif

  if (argv == NULL)
    malloc_failure (0);

  if (*argv == NULL)
    error (_("No source file name was specified"));

#if defined(__CYGWIN__)
  cygwin_conv_to_posix_path (argv[0], cygbuf);
  from = cygbuf;
#else
  from = argv[0];
#endif
  to = argv[1] ? argv[1] : from;

  from = tilde_expand (*argv);

  if ((fd = openp (NULL, OPF_TRY_CWD_FIRST, from,
                   O_RDONLY | O_BINARY, &filename_opened)) < 0)
    {
      printf_unfiltered ("Unable to open '%s': %s\n", from, strerror (errno));
      return;
    }

  nto_trace(0) ("Opened %s for reading\n", filename_opened.get());

  if (nto_fileopen (to, QNX_WRITE_MODE, QNX_WRITE_PERMS) == -1)
    {
      printf_unfiltered ("Remote was unable to open '%s': %s\n", to,
       strerror (errno));
      close (fd);
      filename_opened.release();
      xfree (from);
      return;
    }

  while ((len = read (fd, buf, sizeof buf)) > 0)
    {
      if (nto_filewrite (buf, len) == -1)
  {
    printf_unfiltered ("Remote was unable to complete write: %s\n",
           strerror (errno));
    goto exit;
  }
    }
  if (len == -1)
    {
      printf_unfiltered ("Local read failed: %s\n", strerror (errno));
      goto exit;
    }

  /* Everything worked so set remote exec file.  */
  if (upload_sets_exec)
    {
#ifdef NVIDIA_CUDA_GDB
      inf_rdata->remote_exe = std::string {to};
      if (only_session.remote_exe.length () == 0)
	only_session.remote_exe = std::string {to};
#else
      xfree (inf_rdata->remote_exe);
      inf_rdata->remote_exe = xstrdup (to);
      if (only_session.remote_exe == NULL)
        only_session.remote_exe = xstrdup (to);
#endif
    }

exit:
  nto_fileclose (fd);
  filename_opened.release();
  xfree (from);
  close (fd);
}

static void
download_command (const char *args, int fromtty)
{
#if defined(__CYGWIN__)
  char cygbuf[PATH_MAX];
#endif
  int fd;
  int len;
  char buf[DS_DATA_MAX_SIZE];
  char *from, *to;
  char **argv;

  if (args == 0)
    {
      printf_unfiltered ("You must specify a filename to get.\n");
      return;
    }

#if defined(__CYGWIN__) || defined(__MINGW32__)
  char *wargs=slashify (args);
  argv = buildargv( wargs );
  free(wargs);
#else
  argv = buildargv (args);
#endif

  if (argv == NULL)
    malloc_failure (0);

  from = argv[0];
#if defined(__CYGWIN__)
  if (argv[1])
    {
      cygwin_conv_to_posix_path (argv[1], cygbuf);
      to = cygbuf;
    }
  else
    to = from;
#else
  to = argv[1] ? argv[1] : from;
#endif

  if ((fd = open (to, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY,
        S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH )) == -1)
    {
      printf_unfiltered ("Unable to open '%s': %s\n", to, strerror (errno));
      goto cleanup;
    }

  if (nto_fileopen (from, QNX_READ_MODE, 0) == -1)
    {
      printf_unfiltered ("Remote was unable to open '%s': %s\n", from,
       strerror (errno));
      close (fd);
      goto cleanup;
    }

  while ((len = nto_fileread (buf, sizeof buf)) > 0)
    {
      if (write (fd, buf, len) == -1)
  {
    printf_unfiltered ("Local write failed: %s\n", strerror (errno));
    close (fd);
    goto cleanup;
  }
    }

  if (len == -1)
    printf_unfiltered ("Remote read failed: %s\n", strerror (errno));
  nto_fileclose (fd);
  close (fd);

cleanup:
  freeargv (argv);
}

static void
nto_add_commands (void)
{
  struct cmd_list_element *c;

  c =
    add_com ("upload", class_obscure, upload_command,
       "Send a file to the target (upload {local} [{remote}])");
  set_cmd_completer (c, filename_completer);
  add_com ("download", class_obscure, download_command,
     "Get a file from the target (download {remote} [{local}])");
}

static int nto_remote_fd = -1;

static int
nto_fileopen (char *fname, int mode, int perms)
{
  DScomm_t tran, recv;

  if (nto_remote_fd != -1)
    {
      printf_unfiltered
  ("Remote file currently open, it must be closed before you can open another.\n");
      errno = EAGAIN;
      return -1;
    }

  nto_send_init (&tran, DStMsg_fileopen, 0, SET_CHANNEL_DEBUG);
  strcpy (tran.pkt.fileopen.pathname, fname);
  tran.pkt.fileopen.mode = EXTRACT_SIGNED_INTEGER (&mode, 4,
               nto_byte_order);
  tran.pkt.fileopen.perms = EXTRACT_SIGNED_INTEGER (&perms, 4,
                nto_byte_order);
  nto_send_recv (&tran, &recv, sizeof tran.pkt.fileopen, 0);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      errno = errnoconvert (EXTRACT_SIGNED_INTEGER (&recv.pkt.err.err,
                4, nto_byte_order));
      return -1;
    }
  return nto_remote_fd = 0;
}

static void
nto_fileclose (int fd)
{
  DScomm_t tran, recv;
  unsigned sizeof_pkt;

  if (nto_remote_fd == -1)
    return;

  nto_send_init (&tran, DStMsg_fileclose, 0, SET_CHANNEL_DEBUG);
  tran.pkt.fileclose.mtime = 0;
  sizeof_pkt = sizeof (tran.pkt.fileclose);
  nto_send_recv (&tran, &recv, sizeof_pkt, 1);
  nto_remote_fd = -1;
}

static int
nto_fileread (char *buf, int size)
{
  int len;
  DScomm_t tran, recv;

  nto_send_init (&tran, DStMsg_filerd, 0, SET_CHANNEL_DEBUG);
  tran.pkt.filerd.size = EXTRACT_SIGNED_INTEGER (&size, 2,
             nto_byte_order);
  len = nto_send_recv (&tran, &recv, sizeof tran.pkt.filerd, 0);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      errno = errnoconvert (recv.pkt.err.err);
      return -1;
    }

  len -= sizeof recv.pkt.okdata.hdr;
  memcpy (buf, recv.pkt.okdata.data, len);
  return len;
}

static int
nto_filewrite (char *buf, int size)
{
  int len, siz;
  DScomm_t tran, recv;

  for (siz = size; siz > 0; siz -= len, buf += len)
    {
      len =
  siz < sizeof tran.pkt.filewr.data ? siz : sizeof tran.pkt.filewr.data;
      nto_send_init (&tran, DStMsg_filewr, 0, SET_CHANNEL_DEBUG);
      memcpy (tran.pkt.filewr.data, buf, len);
      nto_send_recv (&tran, &recv, sizeof (tran.pkt.filewr.hdr) + len, 0);

      if (recv.pkt.hdr.cmd == DSrMsg_err)
  {
    errno = errnoconvert (recv.pkt.err.err);
    return size - siz;
  }
    }
  return size;
}

bool
pdebug_target::can_run ( )
{
  nto_trace (0) ("%s ()\n", __func__);
  return false;
}

int
pdebug_target::can_use_hw_breakpoint (enum bptype type, int cnt,
         int othertype)
{
  /* generally NTO does support HW breakpoints */
  return 1;
}

bool
pdebug_target::has_registers ( )
{
  struct inferior *inf = current_inferior();
  if (!inf) return 0;

  return nto_inferior_data (inf)->has_registers;
}

bool
pdebug_target::has_memory ( )
{
  struct inferior *inf = current_inferior();
  if (!inf) return 0;

  return nto_inferior_data (inf)->has_memory;
}

bool
pdebug_target::has_stack ( )
{
  struct inferior *inf = current_inferior();
  if (!inf) return 0;

  return nto_inferior_data (inf)->has_stack;
}

bool
pdebug_target::has_all_memory ( )
{
  struct inferior *inf = current_inferior();
  if (!inf) return 0;

  return nto_inferior_data (inf)->has_memory;
}

int
pdebug_target::insert_fork_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

int
pdebug_target::remove_fork_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

int
pdebug_target::insert_vfork_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

int
pdebug_target::remove_vfork_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

int
pdebug_target::insert_exec_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

int
pdebug_target::remove_exec_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

/*
 * Target hook for follow_fork.
 *
 * On entry the current inferior is the followed one
 *
 * this does some special handling after a fork happened, depending on
 * the fork control switches fork-follow-mode and detach-on-fork:
 *
 * no special support for vfork() as this has been deprecated and is currently
 * just a wrapped fork() call. See <process.h>
 *
 * note: disabling detach_fork is currently not supported even though the stub
 *       code is here!
 */
void
pdebug_target::follow_fork (inferior *child_inf, ptid_t child_ptid,
			    target_waitkind fork_kind, bool follow_child,
			    bool detach_fork)
{
  nto_trace (0) ("follow_fork(%ld, %s, %s)\n", child_ptid.lwp (),
      follow_child?"child":"parent", detach_fork?"detach":"stay attached");

  nto_trace(0) ("  current inferior: %i\n", current_inferior()->pid);
  nto_trace(0) ("  current thread: %s\n", nto_pid_to_str(inferior_thread()->ptid).c_str());

  pid_t     child_pid=-1;

  if (follow_child)
    {
      child_inf = child_inf;
      child_pid = child_inf->pid;
    }
  else
    {
      child_pid = child_ptid.pid ();
      child_inf  = find_inferior_pid(this, child_pid);
    }

  if (child_inf != NULL)
    {
      struct nto_inferior_data *inf_data = nto_inferior_data (child_inf);
      child_inf->attach_flag = false;
      child_inf->removable = true;
      inf_data->has_execution = 1;
      inf_data->has_stack = 1;
      inf_data->has_registers = 1;
      inf_data->has_memory = 1;
      nto_add_thread(this, child_pid, 1);
    }

  if (detach_fork && !follow_child)
    {
      /* Detach the fork child if it was attached */
      if (child_inf != NULL)
        {
	  nto_trace(0) ("  detaching from child inferior\n");
          detach(child_inf,0);
        }
      else
	{
	  nto_trace(0) ("  just sending detach msg to child\n");
	  nto_detach_only (child_pid);
	}
    }

  update_thread_list();
}

int
pdebug_target::verify_memory (const gdb_byte *data,
       CORE_ADDR memaddr, ULONGEST size)
{
  // TODO: This should be more optimal, similar to remote.c
  // implementation and pass address, size and crc32 to pdebug
  // so it can perform crc32 there and save network traffic
  gdb_byte *const buf = (gdb_byte *)xmalloc (size);
  int match;

  nto_trace (0) ("%s (addr:%s, size:%llu)\n", __func__, paddress (target_gdbarch (), memaddr), (long long unsigned)size);

  if (target_read_memory (memaddr, buf, size) != 0)
    {
      warning (_("Error reading memory"));
      return -1;
    }

  match = (memcmp (buf, data, size) == 0);
  xfree (buf);
  return match;
}

const struct target_desc *
pdebug_target::read_description ( )
{
  if (ntoops_read_description)
    return ntoops_read_description (nto_get_cpuflags ());
  else
    return NULL;
}

#if 0
/* Terminal handling.  */
void
pdebug_target::terminal_init ( )
{
}

void
pdebug_target::terminal_inferior ( )
{
}

void
pdebug_target::terminal_ours_for_output ( )
{
}

void
pdebug_target::terminal_ours ( )
{
}

struct void
nto_terminal_save_ours ( )
{
}

void
pdebug_target::terminal_info (const char *arg, int c)
{
}

#endif

static void
update_threadnames (void)
{
  DScomm_t tran, recv;
  struct dstidnames *tidnames = (struct dstidnames *) recv.pkt.okdata.data;
  int cur_pid;
  unsigned int numleft;

  nto_trace (0) ("%s ()\n", __func__);

  cur_pid = inferior_ptid.pid();
  if(!cur_pid) {
    gdb_printf(gdb_stderr, "No inferior.\n");
    return;
  }

  do {
    unsigned int i, numtids;
    char *name;

    nto_send_init (&tran, DStMsg_tidnames, 0, SET_CHANNEL_DEBUG);
    nto_send_recv (&tran, &recv, sizeof(tran.pkt.tidnames), 0);
    if (recv.pkt.hdr.cmd == DSrMsg_err) {
      errno = errnoconvert (EXTRACT_SIGNED_INTEGER (&recv.pkt.err.err, sizeof(recv.pkt.err.err), nto_byte_order));
      if (errno != EINVAL)  {
        warning ("Could not retrieve tidnames (%d - %s)\n", errno, strerror(errno) );
    }
      return;
    }

    numtids = EXTRACT_UNSIGNED_INTEGER (&tidnames->numtids, sizeof(tidnames->numtids), nto_byte_order);
    numleft = EXTRACT_UNSIGNED_INTEGER (&tidnames->numleft, sizeof(tidnames->numleft), nto_byte_order);

    /* the namelist is a mess of the format:
     * <nul*>tid<nul>tidname<nul>tid<nul>tidname<nul>... */
    name=(char *)tidnames->data;
    /* skip to the first tid */
    while ((*name<'0') || (*name>'9'))
	name++;
    /* extract and set names */
    for(i = 0 ; i < numtids ; i++) {
      struct thread_info *ti;
      ptid_t ptid;
      int tid;
      int namelen;
      char *end;

      tid = strtol(name, &end, 10);
      if (tid == 0)
	warning("Received a name for thread 0!");

      name = end + 1; /* Skip the null terminator to the actual name */
      namelen = strlen(name);

      nto_trace (0) ("Thread %d name: %s\n", tid, name);
      ptid = ptid_t (cur_pid, tid, 0);
      ti = nto_find_thread (ptid);
      if(ti)
	{
	  if(ti->name () == NULL)
	    ti->set_name (make_unique_xstrdup(name));
	}
      else
	{
	  warning("Thread %d (%s) does not exist!", tid, name);
	}
      name += namelen + 1;
    }
  } while(numleft > 0);
}

void
pdebug_target::update_thread_list ( )
{
  DScomm_t tran, recv;
  int cur_pid, start_tid = 1, total_tids = 0, num_tids;
  struct dspidlist *pidlist = (struct dspidlist *) recv.pkt.okdata.data;
  struct tidinfo *tip;
  char subcmd;

  nto_trace (0) ("%s ()\n", __func__);

  cur_pid = inferior_ptid.pid();
  if(!cur_pid){
    gdb_printf(gdb_stderr, "No inferior.\n");
    return;
  }
  subcmd = DSMSG_PIDLIST_SPECIFIC;

  do {
    /* fetch threads for cur_pid starting with thread start_tid */
    nto_send_init (&tran, DStMsg_pidlist, subcmd, SET_CHANNEL_DEBUG );
    tran.pkt.pidlist.pid = EXTRACT_UNSIGNED_INTEGER (&cur_pid, 4, nto_byte_order);
    tran.pkt.pidlist.tid = EXTRACT_UNSIGNED_INTEGER (&start_tid, 4, nto_byte_order);
    nto_send_recv (&tran, &recv, sizeof(tran.pkt.pidlist), 0);

    /* transmit/msg error */
    if (recv.pkt.hdr.cmd == DSrMsg_err) {
      errno = errnoconvert (EXTRACT_SIGNED_INTEGER (&recv.pkt.err.err, 4, nto_byte_order));
      return;
    }

    /* no transmit error but no valid data available */
    if (recv.pkt.hdr.cmd != DSrMsg_okdata) {
      errno = EOK;
      nto_trace (1) ("msg not DSrMsg_okdata!\n");
      return;
    }

    /* get the number of active threads for pid */
    num_tids = EXTRACT_UNSIGNED_INTEGER (&pidlist->num_tids, 4, nto_byte_order);

    /* unfortunately pdebug does not use dspidlist->tids but just appends the
     * tidinfos to the process header.. */
    tip = (struct tidinfo *) &pidlist->name[(strlen(pidlist->name) + 1 + 3) & ~3];

    /* iterate through the tidinfos */
    for ( ;tip->tid != 0; tip++ )
    {
      struct thread_info *new_thread;
      ptid_t ptid;
      tip->tid =  EXTRACT_UNSIGNED_INTEGER (&tip->tid, 2,
              nto_byte_order);
      ptid = ptid_t(cur_pid, tip->tid, 0);

      if (tip->tid < 0) {
        warning ("TID %d < 0\n", tip->tid );
        continue;
      }

      /* check if we know this thread already */
      new_thread = nto_find_thread (ptid);

      /* it's a new, non-dead thread */
      if(!new_thread && tip->state != 0) {
        new_thread = add_thread (this, ptid);
        new_thread->priv.reset(new nto_thread_info());
      }

      /* update thread info */
      if ( new_thread ) {
        /* create thread info if the thread was found and announced by GDB */
        if( new_thread->priv.get() == NULL ) {
          new_thread->priv.reset(new nto_thread_info());
        }
        ((struct nto_thread_info *)new_thread->priv.get())->fill(tip);
      }
      total_tids++;
    }

    subcmd = DSMSG_PIDLIST_SPECIFIC_TID;
    start_tid = total_tids + 1;
  } while(total_tids < num_tids);

  update_threadnames ();
}

#ifdef __MINGW32__
char *
strcasestr(const char *const haystack, const char *const needle)
{
  char buff_h[1024];
  char buff_n[1024];
  const char *p;
  int i;

  p = haystack;
  for (p = haystack, i = 0; *p && i < sizeof(buff_h)-1; ++p, ++i)
    buff_h[i] = toupper(*p);
  buff_h[i] = '\0';
  for (p = needle, i = 0; *p && i < sizeof(buff_n)-1; ++p, ++i)
    buff_n[i] = toupper(*p);
  buff_n[i] = '\0';

  return strstr(buff_h, buff_n);
}
#endif /* __MINGW32__ */

static void
nto_pidlist (const char *args, int from_tty)
{
  DScomm_t tran, recv;
  struct dspidlist *pidlist = (struct dspidlist *) recv.pkt.okdata.data;
  struct tidinfo *tip;
  char specific_tid_supported = 0;
  int pid, start_tid, total_tid;
  char subcmd;

  start_tid = 1;
  total_tid = 0;
  pid = 1;
  subcmd = DSMSG_PIDLIST_BEGIN;

  /* Send a DSMSG_PIDLIST_SPECIFIC_TID to see if it is supported.  */
  nto_send_init (&tran, DStMsg_pidlist, DSMSG_PIDLIST_SPECIFIC_TID,
     SET_CHANNEL_DEBUG);
  tran.pkt.pidlist.pid = EXTRACT_SIGNED_INTEGER (&pid, 4,
             nto_byte_order);
  tran.pkt.pidlist.tid = EXTRACT_SIGNED_INTEGER (&start_tid, 4,
             nto_byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.pidlist), 0);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    specific_tid_supported = 0;
  else
    specific_tid_supported = 1;

  while (1)
    {
      nto_send_init (&tran, DStMsg_pidlist, subcmd, SET_CHANNEL_DEBUG);
      tran.pkt.pidlist.pid = EXTRACT_SIGNED_INTEGER (&pid, 4,
                 nto_byte_order);
      tran.pkt.pidlist.tid = EXTRACT_SIGNED_INTEGER (&start_tid, 4,
                 nto_byte_order);
      nto_send_recv (&tran, &recv, sizeof (tran.pkt.pidlist), 0);
      if (recv.pkt.hdr.cmd == DSrMsg_err)
  {
    errno = errnoconvert (EXTRACT_SIGNED_INTEGER
          (&recv.pkt.err.err, 4,
           nto_byte_order));
    return;
  }
      if (recv.pkt.hdr.cmd != DSrMsg_okdata)
  {
    errno = EOK;
    return;
  }

      for (tip =
     (struct tidinfo *) &pidlist->name[(strlen (pidlist->name) + 1 + 3) & ~3];
     tip->tid != 0; tip++)
  {
    if ((args != NULL && strcasestr(pidlist->name, args) != NULL)
        || args == NULL)
      gdb_printf ("%s - %ld/%ld\n", pidlist->name,
           (long) EXTRACT_SIGNED_INTEGER (&pidlist->pid,
                  4, nto_byte_order),
           (long) EXTRACT_SIGNED_INTEGER (&tip->tid, 2,
                  nto_byte_order));
    total_tid++;
  }
      pid = EXTRACT_SIGNED_INTEGER (&pidlist->pid, 4, nto_byte_order);
      if (specific_tid_supported)
  {
    if (total_tid < EXTRACT_SIGNED_INTEGER
          (&pidlist->num_tids, 4, nto_byte_order))
      {
        subcmd = DSMSG_PIDLIST_SPECIFIC_TID;
        start_tid = total_tid + 1;
        continue;
      }
  }
      start_tid = 1;
      total_tid = 0;
      subcmd = DSMSG_PIDLIST_NEXT;
    }
  return;
}

static struct dsmapinfo *
nto_mapinfo (unsigned addr, int first, int elfonly)
{
  DScomm_t tran, recv;
  struct dsmapinfo map;
  static struct dsmapinfo dmap;
  DStMsg_mapinfo_t *mapinfo = (DStMsg_mapinfo_t *) & tran.pkt;
  char subcmd;

  if (core_bfd != NULL)
    {        /* Have to implement corefile mapinfo.  */
      errno = EOK;
      return NULL;
    }

  subcmd = addr ? DSMSG_MAPINFO_SPECIFIC :
    first ? DSMSG_MAPINFO_BEGIN : DSMSG_MAPINFO_NEXT;
  if (elfonly)
    subcmd |= DSMSG_MAPINFO_ELF;

  nto_send_init (&tran, DStMsg_mapinfo, subcmd, SET_CHANNEL_DEBUG);
  mapinfo->addr = EXTRACT_UNSIGNED_INTEGER (&addr, sizeof(mapinfo->addr), nto_byte_order);
  nto_send_recv (&tran, &recv, sizeof (*mapinfo), 0);
  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      errno = errnoconvert (EXTRACT_SIGNED_INTEGER
            (&recv.pkt.err.err, 4, nto_byte_order));
      return NULL;
    }
  if (recv.pkt.hdr.cmd != DSrMsg_okdata)
    {
      errno = EOK;
      return NULL;
    }

  memset (&dmap, 0, sizeof (dmap));
  memcpy (&map, &recv.pkt.okdata.data[0], sizeof (map));
  dmap.ino = EXTRACT_UNSIGNED_INTEGER (&map.ino, 8, nto_byte_order);
  dmap.dev = EXTRACT_SIGNED_INTEGER (&map.dev, 4, nto_byte_order);

  dmap.text.addr = EXTRACT_UNSIGNED_INTEGER (&map.text.addr, 4,
               nto_byte_order);
  dmap.text.size = EXTRACT_SIGNED_INTEGER (&map.text.size, 4,
             nto_byte_order);
  dmap.text.flags = EXTRACT_SIGNED_INTEGER (&map.text.flags, 4,
              nto_byte_order);
  dmap.text.debug_vaddr =
    EXTRACT_UNSIGNED_INTEGER (&map.text.debug_vaddr, 4,
            nto_byte_order);
  dmap.text.offset = EXTRACT_UNSIGNED_INTEGER (&map.text.offset, 8,
                 nto_byte_order);
  dmap.data.addr = EXTRACT_UNSIGNED_INTEGER (&map.data.addr, 4,
               nto_byte_order);
  dmap.data.size = EXTRACT_SIGNED_INTEGER (&map.data.size, 4,
             nto_byte_order);
  dmap.data.flags = EXTRACT_SIGNED_INTEGER (&map.data.flags, 4,
              nto_byte_order);
  dmap.data.debug_vaddr =
    EXTRACT_UNSIGNED_INTEGER (&map.data.debug_vaddr, 4, nto_byte_order);
  dmap.data.offset = EXTRACT_UNSIGNED_INTEGER (&map.data.offset, 8,
                 nto_byte_order);

  strcpy (dmap.name, map.name);

  return &dmap;
}

static void
nto_meminfo (const char *args, int from_tty)
{
  struct dsmapinfo *dmp;
  int first = 1;

  while ((dmp = nto_mapinfo (0, first, 0)) != NULL)
    {
      first = 0;
      gdb_printf ("%s\n", dmp->name);
      gdb_printf ("\ttext=%08x bytes @ 0x%08x\n", dmp->text.size,
           dmp->text.addr);
      gdb_printf ("\t\tflags=%08x\n", dmp->text.flags);
      gdb_printf ("\t\tdebug=%08x\n", dmp->text.debug_vaddr);
      gdb_printf ("\t\toffset=%016" PRIx64 "\n", dmp->text.offset);
      if (dmp->data.size)
  {
    gdb_printf ("\tdata=%08x bytes @ 0x%08x\n", dmp->data.size,
         dmp->data.addr);
    gdb_printf ("\t\tflags=%08x\n", dmp->data.flags);
    gdb_printf ("\t\tdebug=%08x\n", dmp->data.debug_vaddr);
    gdb_printf ("\t\toffset=%016" PRIx64 "\n", dmp->data.offset);
  }
      gdb_printf ("\tdev=0x%x\n", dmp->dev);
      gdb_printf ("\tino=0x%" PRIx64 "\n", dmp->ino);
    }
}

/*
 * for some reason this was not designed analogue to the
 * nto_to_insert_breakpoint nto_insert_breakpoint pair
 */
int
pdebug_target::insert_hw_breakpoint (struct gdbarch *gdbarch,
        struct bp_target_info *bp_tg_inf)
{
  DScomm_t tran, recv;
  size_t sizeof_pkt;

  nto_trace (0) ("pdebug_insert_hw_breakpoint(addr %s) pid:%d\n",
     paddress (gdbarch, bp_tg_inf->reqstd_address),
     inferior_ptid.pid());

  /* if target info is unset, something really bad is going on! */
  if ( bp_tg_inf == NULL ) {
    internal_error(_("Target info invalid."));
  }

  /* Must select appropriate inferior. */
  if (!nto_set_thread_alive (inferior_ptid))
    {
      return 1;
    }

  /* expect that all will succeed */
  bp_tg_inf->placed_address = bp_tg_inf->reqstd_address;

  nto_send_init (&tran, DStMsg_brk, DSMSG_BRK_EXEC | DSMSG_BRK_HW,
     SET_CHANNEL_DEBUG);

  tran.pkt.brk.size = 0;
  tran.pkt.brk.addr = EXTRACT_SIGNED_INTEGER (&bp_tg_inf->placed_address,
      sizeof(tran.pkt.brk.addr), nto_byte_order);
  sizeof_pkt = sizeof (tran.pkt.brk);

  nto_send_recv (&tran, &recv, sizeof_pkt, 1);
  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      bp_tg_inf->placed_address = 0;
      nto_trace (0) ("Unable to set HW breakpoint\n");
    }

  return recv.pkt.hdr.cmd == DSrMsg_err;
}

static int
nto_hw_watchpoint (CORE_ADDR addr, int len, enum target_hw_bp_type type)
{
  DScomm_t tran, recv;
  unsigned subcmd;

  nto_trace (0) ("nto_hw_watchpoint(addr %s, len %x, type %x)\n",
     paddress (target_gdbarch (), addr), len, type);

  switch (type)
    {
    case 1:      /* Read.  */
      subcmd = DSMSG_BRK_RD;
      break;
    case 2:      /* Read/Write.  */
      subcmd = DSMSG_BRK_WR;
      break;
    default:      /* Modify.  */
      subcmd = DSMSG_BRK_MODIFY;
    }
  subcmd |= DSMSG_BRK_HW;

  nto_send_init (&tran, DStMsg_brk, subcmd, SET_CHANNEL_DEBUG);
  tran.pkt.brk.addr = EXTRACT_UNSIGNED_INTEGER (&addr,
      sizeof(tran.pkt.brk.addr), nto_byte_order);
  tran.pkt.brk.size = EXTRACT_SIGNED_INTEGER (&len,
      sizeof(tran.pkt.brk.size), nto_byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.brk), 0);
  return recv.pkt.hdr.cmd == DSrMsg_err ? -1 : 0;
}

int
pdebug_target::remove_watchpoint (CORE_ADDR addr, int len,
        enum target_hw_bp_type type, struct expression *exp)
{
  return nto_hw_watchpoint (addr, -1, type);
}

int
pdebug_target::insert_watchpoint (CORE_ADDR addr, int len,
        enum target_hw_bp_type type, struct expression *exp)
{
  return nto_hw_watchpoint (addr, len, type);
}

pdebug_target::~pdebug_target ()
{
#if 0
	struct remote_state *rs = get_remote_state ();

  /* Check for NULL because we may get here with a partially
     constructed target/connection.  */
  if (rs->remote_desc == nullptr)
    return;

  serial_close (rs->remote_desc);

  /* We are destroying the remote target, so we should discard
     everything of this target.  */
  discard_pending_stop_replies_in_queue ();

  if (rs->remote_async_inferior_event_token)
    delete_async_event_handler (&rs->remote_async_inferior_event_token);
#else
  if(current_session->desc) {
      serial_close (current_session->desc);
      /* do not free current_session->desc, this is done in serial_close! */
      current_session->desc=NULL;
  }
  remote_unpush_target (get_current_target());
#endif
}

#ifndef NVIDIA_CUDA_GDB
static void
nto_remote_inferior_data_cleanup (struct inferior *const inf, void *const dat)
{
  struct nto_remote_inferior_data *const inf_rdata
    = (struct nto_remote_inferior_data *const) dat;

  if (dat)
    {
      xfree (inf_rdata->auxv);
      inf_rdata->auxv = NULL;
      xfree (inf_rdata->remote_exe);
      inf_rdata->remote_exe = NULL;
      xfree (inf_rdata->remote_cwd);
      inf_rdata->remote_cwd = NULL;
    }
  xfree (dat);
}
#endif

static struct nto_remote_inferior_data *
nto_get_remote_inferior_data (struct inferior *const inf)
{
  struct nto_remote_inferior_data *inf_data;

  gdb_assert (inf != NULL);

#ifdef NVIDIA_CUDA_GDB
  /* CUDA: Changed this to work with newer versions of gdb. */
  inf_data = nto_remote_inferior_data_key.get (inf);
  if (inf_data == nullptr)
    {
      inf_data = nto_remote_inferior_data_key.emplace (inf);
    }
#else
  inf_data = (struct nto_remote_inferior_data *)inferior_data (inf,
         nto_remote_inferior_data_reg);
  if (inf_data == NULL)
    {
      inf_data = XCNEW (struct nto_remote_inferior_data);
      set_inferior_data (inf, nto_remote_inferior_data_reg, inf_data);
    }
#endif

  return inf_data;
}

static void
set_nto_exe (const char *args, int from_tty,
       struct cmd_list_element *c)
{
  struct inferior *const inf = current_inferior ();
  struct nto_remote_inferior_data *const inf_rdat
    = nto_get_remote_inferior_data (inf);

#ifdef NVIDIA_CUDA_GDB
  inf_rdat->remote_exe = current_session->remote_exe;
#else
  xfree (inf_rdat->remote_exe);
  inf_rdat->remote_exe = xstrdup (current_session->remote_exe);
#endif
}

static void
show_nto_exe (struct ui_file *file, int from_tty,
              struct cmd_list_element *c, const char *value)
{
  struct inferior *const inf = current_inferior ();
  struct nto_remote_inferior_data *const inf_rdat
    = nto_get_remote_inferior_data (inf);

  deprecated_show_value_hack (file, from_tty, c, inf_rdat->remote_exe.c_str ());
}

static void
set_nto_cwd (const char *args, int from_tty, struct cmd_list_element *c)
{
  struct inferior *const inf = current_inferior ();
  struct nto_remote_inferior_data *const inf_rdat
    = nto_get_remote_inferior_data (inf);

#ifdef NVIDIA_CUDA_GDB
  inf_rdat->remote_cwd = current_session->remote_cwd;
#else
  xfree (inf_rdat->remote_cwd);
  inf_rdat->remote_cwd = xstrdup (current_session->remote_cwd);
#endif
}

static void
show_nto_cwd (struct ui_file *file, int from_tty,
              struct cmd_list_element *c, const char *value)
{
  struct inferior *const inf = current_inferior ();
  struct nto_remote_inferior_data *const inf_rdat
    = nto_get_remote_inferior_data (inf);

#ifdef NVIDIA_CUDA_GDB
  deprecated_show_value_hack (file, from_tty, c, inf_rdat->remote_cwd.c_str ());
#else
  deprecated_show_value_hack (file, from_tty, c, inf_rdat->remote_cwd);
#endif
}

#ifdef NVIDIA_CUDA_GDB
/* CUDA extensions for sending our custom CUDA packets to the server. */
void
send_qnx_packet (gdb::array_view<const char> &buf,
		 send_remote_packet_callbacks *callbacks)
{
  if (buf.size () == 0 || buf.data ()[0] == '\0')
    error (_("a remote packet must not be empty"));

  if (buf.size () > DS_DATA_MAX_SIZE)
    {
      std::string arg ((char *)buf.data (), buf.size ());
      printf_unfiltered ("Packet too long: %.40s...\n", arg.c_str ());
    }

  if (get_current_target () == nullptr)
    error (_("QNX packets can only be sent to a pdebug target"));

  callbacks->sending (buf);

  DScomm_t tran, recv;
  nto_send_init (&tran, DStMsg_cuda, 0, SET_CHANNEL_DEBUG);
  memcpy (tran.pkt.cuda.data, buf.data (), buf.size ());
  int bytes = nto_send_recv (&tran, &recv,
			     offsetof (DStMsg_cuda_t, data) + buf.size (), 1);

  if (bytes < 0)
    error (_("error while fetching packet from remote target"));
  else if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      nto_trace (1) ("  errno=%d (%s)\n", recv.pkt.err.err,
		     strerror (recv.pkt.err.err));
      error (_("error packet received from remote target"));
    }
  else if (recv.pkt.hdr.cmd != DSrMsg_okcuda)
    error (_("unexpected packet received from remote target"));

  bytes -= sizeof (struct DShdr);
  gdb::array_view<const char> view ((const char *)recv.pkt.cuda.data, bytes);
  callbacks->received (view);
}
#endif

void _initialize_remote_nto ();
void
_initialize_remote_nto ()
{
  /*
   * remote_g_packet_data_handle = gdbarch_data_register_pre_init (remote_g_packet_data_init);
   * remote_pspace_data = register_program_space_data_with_cleanup (NULL, remote_pspace_data_cleanup);
   * gdb::observers::new_objfile.attach (remote_new_objfile);
   */

  add_target ( pdebug_target_info, pdebug_target::pdebug_open);

#ifndef NVIDIA_CUDA_GDB
  nto_remote_inferior_data_reg
    = register_inferior_data_with_cleanup (NULL, nto_remote_inferior_data_cleanup);
#endif

  add_setshow_zinteger_cmd ("watchdog", class_maintenance, &watchdog, _("\
Set watchdog timer."), _("\
Show watchdog timer."), _("\
When non-zero, this timeout is used instead of waiting forever for a target\n\
to finish a low-level step or continue operation.  If the specified amount\n\
of time passes without a response from the target, an error occurs."),
			    NULL,
			    show_watchdog,
			    &setlist, &showlist);

  add_setshow_zinteger_cmd ("nto-timeout", no_class,
          &only_session.timeout, _("\
Set timeout value for communication with the remote."), _("\
Show timeout value for communication with the remote."), _("\
The remote will timeout after nto-timeout seconds."),
          NULL, NULL, &setlist, &showlist);

  add_setshow_boolean_cmd ("nto-inherit-env", no_class,
         &only_session.inherit_env, _("\
Set if the inferior should inherit environment from pdebug or gdb."), _("\
Show nto-inherit-env value."), _("\
If nto-inherit-env is off, the process spawned on the remote \
will have its environment set by gdb.  Otherwise, it will inherit its \
environment from pdebug."), NULL, NULL,
        &setlist, &showlist);

  add_setshow_string_cmd ("nto-cwd", class_support, &only_session.remote_cwd,
        _("\
Set the working directory for the remote process."), _("\
Show current working directory for the remote process."), _("\
Working directory for the remote process. This directory must be \
specified before remote process is run."),
        &set_nto_cwd, &show_nto_cwd, &setlist, &showlist);

  add_setshow_string_cmd ("nto-executable", class_files,
        &only_session.remote_exe, _("\
Set the binary to be executed on the remote QNX Neutrino target."), _("\
Show currently set binary to be executed on the remote QNX Neutrino target."),
      _("\
Binary to be executed on the remote QNX Neutrino target when "\
"'run' command is used."),
        &set_nto_exe, &show_nto_exe, &setlist, &showlist);

  add_setshow_boolean_cmd ("nto-force-hwbp",  class_breakpoint,
			   &nto_force_hwbp, _("\
Set if hardware breakpoints should be forced."), _("\
Show nto-force-hwbp value."), _("\
If nto-force-hwbp is on, all breakpoints being set will be forced \
to become hardware breakpoints. Otherwise the default behaviour \
is used (see: breakpoint auto-hw)."), NULL, NULL,
	&setlist, &showlist);

  add_setshow_boolean_cmd ("upload-sets-exec", class_files,
         &upload_sets_exec, _("\
Set the flag for upload to set nto-executable."), _("\
Show nto-executable flag."), _("\
If set, upload will set nto-executable. Otherwise, nto-executable \
will not change."),
         NULL, NULL, &setlist, &showlist);

  add_info ("pidlist", nto_pidlist, _("List processes on the target.  Optional argument will filter out process names not containing (case insensitive) argument string."));
  add_info ("meminfo", nto_meminfo, "memory information");
}
