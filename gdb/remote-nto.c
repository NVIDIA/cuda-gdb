/*
 * $QNXtpLicenseC:  
 * Copyright 2005,2007, QNX Software Systems. All Rights Reserved.
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
   Copyright (C) 2017-2023 NVIDIA Corporation
   Modified from the original GDB file referenced above by the CUDA-GDB
   team at NVIDIA <cudatools@nvidia.com>. */

#include "defs.h"
#include "exceptions.h"
#include <fcntl.h>
#include <signal.h>
#include <termios.h>
/* CUDA: unblocking gdb 10.1 upgrade */
#include <queue>

#include "remote-nto.h"
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
#include "cuda/cuda-packet-manager.h"
#include "cuda/remote-cuda.h"
#include "source.h"

#include "infrun.h"

#include "elf-bfd.h"
#include "elf/common.h"

#include "environ.h"

#include <time.h>

#include "nto-share/dsmsgs.h"
#include "nto-tdep.h"

#ifdef __QNXNTO__
#if 0
#include <sys/elf_notes.h>
#warning FIXME
#if __PTR_BITS__ == 32
#define Elf32_Phdr Elf32_External_Phdr
#elif __PTR_BITS__ == 64
#define Elf64_Phdr Elf64_External_Phdr
#else
#error __PTR_BITS__ not setup correctly
#endif
#define __ELF_H_INCLUDED /* Needed for our link.h to avoid including elf.h.  */
#include <sys/link.h>
#endif

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

#ifdef __MINGW32__
#define	ENOTCONN	57		/* Socket is not connected */
#endif
#
#ifndef EOK
#define EOK 0
#endif

#ifndef O_BINARY
#define O_BINARY 0
#endif

#define QNX_READ_MODE	0x0
#define QNX_WRITE_MODE	0x301
#define QNX_WRITE_PERMS	0x1ff

/* The following define does a cast to const gdb_byte * type.  */

#define EXTRACT_SIGNED_INTEGER(ptr, len, byte_order) \
  extract_signed_integer ((const gdb_byte *)(ptr), len, byte_order)
#define EXTRACT_UNSIGNED_INTEGER(ptr, len, byte_order) \
  extract_unsigned_integer ((const gdb_byte *)(ptr), len, byte_order)

#ifdef __MINGW32__
/* Name collision with a symbol declared in Winsock2.h.  */
#define recv recvb
#endif

static int putpkt (const DScomm_t *tran, unsigned);

static int readchar (int timeout);

static int getpkt (DScomm_t *recv, int forever);

static int getpkt_with_retry (DScomm_t *recv, int forever, int max_tries);

static unsigned nto_send_recv (const DScomm_t *tran, DScomm_t *recv, unsigned, int);

static enum target_xfer_status nto_write_bytes (
    CORE_ADDR memaddr, const gdb_byte *myaddr, int len, ULONGEST *xfered_len);

static enum target_xfer_status nto_read_bytes (
    CORE_ADDR memaddr, gdb_byte *myaddr, int len, ULONGEST *xfered_len);

static ptid_t nto_parse_notify (const DScomm_t *recv, remote_target *,
				struct target_waitstatus *status);

void nto_outgoing_text (char *buf, int nbytes);

static int nto_incoming_text (TSMsg_text_t *text, int len);

static void nto_send_init (DScomm_t *tran, unsigned cmd, unsigned subcmd, unsigned chan);

static int nto_send_env (const char *env);

static int nto_send_arg (const char *arg);

static int nto_fd_raw (int fd);

static void nto_interrupt (int signo);

static void nto_interrupt_twice (int signo);

static void interrupt_query (void);

static void upload_command (const char *args, int from_tty);

static void download_command (const char *args, int from_tty);

static void nto_add_commands (void);

static void nto_remove_commands (void);

static int nto_fileopen (char *fname, int mode, int perms);

static void nto_fileclose (int);

static int nto_fileread (char *buf, int size);

static int nto_filewrite (char *buf, int size);

static void nto_pidlist (const char *args, int from_tty);

static struct dsmapinfo *nto_mapinfo (unsigned addr, int first, int elfonly);

static void nto_meminfo (const char *args, int from_tty);

static void nto_remote_inferior_data_cleanup (struct inferior *inf, void *dat);


template <class parent>
class qnx_remote_target : public parent
{
public:
  qnx_remote_target () = default;
  ~qnx_remote_target () override = default;

  /* Open a remote connection.  */
  static void open (const char *, int);

  virtual void close () override;

  virtual void load (const char *arg0, int arg1) override;

  virtual void attach (const char *arg0, int arg1) override;

  virtual void post_attach (int arg0) override;

  virtual void detach (inferior *arg0, int arg1) override;

  virtual void resume (ptid_t arg0,
		       int TARGET_DEBUG_PRINTER () arg1,
		       enum gdb_signal arg2) override;

  virtual ptid_t wait (ptid_t arg0, struct target_waitstatus *arg1,
		       target_wait_flags target_options) override;

  virtual void fetch_registers (struct regcache *arg0, int arg1) override;

  virtual void store_registers (struct regcache *arg0, int arg1) override;

  virtual void prepare_to_store (struct regcache *arg0) override;

  virtual enum target_xfer_status xfer_partial (enum target_object object,
						const char *annex,
						gdb_byte *readbuf,
						const gdb_byte *writebuf,
						ULONGEST offset, ULONGEST len,
						ULONGEST *xfered_len) override;

  virtual void files_info () override;

  virtual int can_use_hw_breakpoint (enum bptype arg0, int arg1, int arg2) override;

  virtual int insert_breakpoint (struct gdbarch *arg0,
				 struct bp_target_info *arg1) override;
  
  virtual int remove_breakpoint (struct gdbarch *arg0,
				 struct bp_target_info *arg1,
				 enum remove_bp_reason arg2) override;

  virtual int insert_hw_breakpoint (struct gdbarch *arg0,
				    struct bp_target_info *arg1) override;

  virtual int remove_hw_breakpoint (struct gdbarch *arg0,
				    struct bp_target_info *arg1) override;

  virtual int remove_watchpoint (CORE_ADDR arg0, int arg1,
				 enum target_hw_bp_type arg2, struct expression *arg3) override;

  virtual int insert_watchpoint (CORE_ADDR arg0, int arg1,
				 enum target_hw_bp_type arg2, struct expression *arg3) override;

  virtual bool stopped_by_watchpoint () override;

  virtual void kill () override;

  virtual bool can_create_inferior () override;

  virtual void create_inferior (const char *arg0, const std::string &arg1,
				char **arg2, int arg3) override;

  virtual void mourn_inferior () override;

  virtual void pass_signals (gdb::array_view<const unsigned char>) override
  { /* QPassSignals used by remote_target::pass_signals is not supported on QNX */ }

  virtual void program_signals (gdb::array_view<const unsigned char>) override
  { /* QProgramSignals used by remote_target::program_signals is not supported on QNX */ }

  virtual void update_thread_list () override;

  virtual bool can_run () override;

  virtual bool thread_alive (ptid_t ptid) override;

  virtual void stop (ptid_t) override
  { /* in GDB 7.12, nto_ops.to_stop was set to 0 */ }

  virtual bool has_all_memory () override;

  virtual bool has_memory () override;

  virtual bool has_stack () override;

  virtual bool has_registers () override;

  virtual bool has_execution (inferior *) override;

  virtual std::string pid_to_str (ptid_t) override;

  virtual const char *thread_name (struct thread_info *ti) override;

#ifndef NVIDIA_CUDA_GDB
  virtual const char *extra_thread_info (thread_info *arg0) override;
#endif

  virtual int insert_fork_catchpoint (int arg0) override;

  virtual int remove_fork_catchpoint (int arg0) override;
  
  virtual int insert_vfork_catchpoint (int arg0) override;
  
  virtual int remove_vfork_catchpoint (int arg0) override;

  virtual void follow_fork (inferior *child_inf, ptid_t child_ptid,
			    target_waitkind fork_kind, bool follow_child,
			    bool detach_fork) override;

  virtual int insert_exec_catchpoint (int arg0) override;

  virtual int remove_exec_catchpoint (int arg0) override;

  virtual bool supports_multi_process () override;

  virtual int verify_memory (const gdb_byte *data,
			     CORE_ADDR memaddr, ULONGEST size) override;

  virtual bool is_async_p (void) override;

  virtual bool can_async_p (void) override;

  virtual int get_trace_status (struct trace_status *ts) override;

  virtual bool supports_non_stop (void) override;

  virtual const struct target_desc *read_description () override;

  virtual void remote_check_symbols () override;

  int hw_watchpoint (CORE_ADDR addr, int len, enum target_hw_bp_type type);

  int remove_breakpoint (CORE_ADDR addr, gdb_byte *contents_cache);

  int do_insert_breakpoint (struct gdbarch *arg0,
			    struct bp_target_info *arg1);

};

struct nto_remote_inferior_data
{
  nto_remote_inferior_data() : remote_exe(), remote_cwd(), auxv() {}
  /* File to be executed on remote.  */
  std::string remote_exe;

  /* Current working directory on remote.  */
  std::string remote_cwd;

  /* Cached auxiliary vector */
  gdb_byte *auxv;
};

static struct nto_remote_inferior_data *nto_remote_inferior_data (
    struct inferior *inf);

static const struct inferior_data *nto_remote_inferior_data_reg;

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
  std::string remote_exe {};

  /* Current working directory on remote.  Assigned to new inferiors.  */
  std::string remote_cwd {};

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

  /* CUDA: unblocking gdb 10.1 upgrade */
  std::queue<DScomm_t> pending;
};

struct pdebug_session only_session;

/* Remote session (connection) to a QNX target. */
struct pdebug_session *current_session = &only_session;

/* Flag for whether upload command sets the current session's remote_exe.  */
static bool upload_sets_exec = true;

/* These define the version of the protocol implemented here.  */
#define HOST_QNX_PROTOVER_MAJOR	0
#define HOST_QNX_PROTOVER_MINOR	7

/* HOST_QNX_PROTOVER 0.8 - 64 bit capable structures.  */

/* Stuff for dealing with the packets which are part of this protocol.  */

#define MAX_TRAN_TRIES 3
#define MAX_RECV_TRIES 3

#define FRAME_CHAR	0x7e
#define ESC_CHAR	0x7d

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

/* This variable was internal to nto_send_init. We need it in the outer scope
   to be able to set it to the value we need, since we have two hosts connecting
   to the pdebug target - cuda-gdb and cuda-gdbserver and `mid` must be kept
   in sync between them. */
static unsigned char mid;

static int
supports64bit(void)
{
  return (current_session->target_proto_major > 0
	  || current_session->target_proto_minor >= 7);
}

/* Compatibility functions for the CUDA remote I/O */
int
qnx_getpkt (remote_target *ops, gdb::char_vector* buf, int forever)
{
  int length;
  DScomm_t recv;

  while (true)
    {
      /* cuda-gdbserver cannot deal with retries */
      length = getpkt_with_retry (&recv, forever, 1);
      if (length <= 0 || current_session->channelrd != SET_CHANNEL_TEXT)
        {
          break;
        }
      nto_incoming_text (&recv.text, length);
    }

  if (length == -1 || recv.pkt.hdr.cmd == DSrMsg_err)
    {
      return -1;
    }
  else if (recv.pkt.hdr.cmd == DSrMsg_okcuda)
    {
      /* We need to keep mid in sync. cuda-gdbserver always returns
         the latest unused mid in okcuda packets. */
      mid = recv.pkt.hdr.mid;

      length -= sizeof (struct DShdr);
      if (length + 1 > buf->size ())
        {
          error ("Buffer is too small");
        }
      memcpy (buf->data (), recv.pkt.cuda.data, length);
      buf->at (length) = '\0';
      return length;
    }
  else if (length == 0)
    {
      /* Empty packet, signaling unsupported packet */
      buf->at (length) = '\0';
      return length;
    }
  else
    {
      error ("Received invalid CUDA response packet");
    }
}

/* Compatibility functions for the CUDA remote I/O */
int
qnx_getpkt_sane (gdb::char_vector* buf, int forever)
{
  return qnx_getpkt (cuda_get_current_remote_target (), buf, forever);
}

int
qnx_putpkt (remote_target *ops, const char *buf)
{
  return qnx_putpkt_binary (buf, strlen (buf));
}

int
qnx_putpkt_binary (const char *buf, int cnt)
{
  DScomm_t tran;

  nto_send_init (&tran, DStMsg_cuda, 0, SET_CHANNEL_DEBUG);
  memcpy (tran.pkt.cuda.data, buf, cnt);
  putpkt (&tran, offsetof (DStMsg_cuda_t, data) + cnt);

  return 0;
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

  if (!current_session->desc)
    perror_with_name ("Remote connection closed.");

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
	  fprintf_unfiltered (gdb_stderr, _("Error reading stdin: %d(%s)\n"),
			      errno, safe_strerror (errno));
	  return;
	}
      if (read_count == 0)
	{
	  fprintf_unfiltered (gdb_stderr, _("EOF\n"));
	  return;
	}
      fprintf_unfiltered (gdb_stderr, _("'%c'"), buff[0]);
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
      error ("Remote connection closed");
    case SERIAL_ERROR:
      perror_with_name ("Remote communication error");
    case SERIAL_TIMEOUT:
      return ch;
    default:
      return ch & 0xff;
    }
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
    printf_filtered ("Receiving data: ");

  csum = 0;

  memset (bp, -1, bufsz);
  for (;;)
    {
      c = readchar (current_session->timeout);

      switch (c)
	{
	case SERIAL_TIMEOUT:
	  puts_filtered ("Timeout in mid-packet, retrying\n");
	  return -1;
	case ESC_CHAR:
	  modifier = 0x20;
	  continue;
	case FRAME_CHAR:
	  if (bp == buf)
	    continue;		/* Ignore multiple start frames.  */
	  if (csum != 0xff)	/* Checksum error.  */
	    return -1;
	  return bp - buf - 1;
	default:
	  c ^= modifier;
	  if (remote_debug)
	    printf_filtered ("%2.2x", c);
	  csum += c;
	  *bp++ = c;
	  break;
	}
      modifier = 0;
    }
}

/* Read a packet from the remote machine, with error checking,
   and store it in recv.buf.
   If FOREVER, wait forever rather than timing out; this is used
   while the target is executing user code.  */
static int
getpkt (DScomm_t *const recv, const int forever)
{
  return getpkt_with_retry (recv, forever, MAX_RECV_TRIES);
}

static int
getpkt_with_retry (DScomm_t *const recv, const int forever, const int max_tries)
{
  int c;
  int tries;
  int timeout;
  unsigned len;

  if (!current_session->desc)
    perror_with_name ("Remote connection closed.");

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

  for (tries = 0; tries < max_tries; tries++)
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
		  target_mourn_inferior (inferior_ptid);
		  error ("Watchdog has expired.  Target detached.");
		}
	      puts_filtered ("Timed out.\n");
	      return -1;
	    }
	}
      while (c != FRAME_CHAR);

      /* We've found the start of a packet, now collect the data.  */
      len = read_frame (recv->buf, sizeof recv->buf);

      if (remote_debug)
	printf_filtered ("\n");

      if (len >= sizeof (struct DShdr))
	{
	  if (recv->pkt.hdr.channel)	/* If hdr.channel is not 0, then hdr.channel is supported.  */
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
	  --tries;		/* Doesn't count as a retry.  */
	  continue;
	}
      /* Empty packet, likely a reply to an unsupported packet */
      if (!len)
        {
          return 0;
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
  gdb_assert (tran != NULL);

  nto_trace (2) ("    nto_send_init(cmd %d, subcmd %d)\n", cmd,
			 subcmd);

  if (gdbarch_byte_order (target_gdbarch ()) == BFD_ENDIAN_BIG)
    cmd |= DSHDR_MSG_BIG_ENDIAN;

  memset (tran, 0, sizeof (DScomm_t));

  tran->pkt.hdr.cmd = cmd;	/* TShdr.cmd.  */
  tran->pkt.hdr.subcmd = subcmd;	/* TShdr.console.  */
  tran->pkt.hdr.mid = ((chan == SET_CHANNEL_DEBUG) ? mid++ : 0);	/* TShdr.spare1.  */
  tran->pkt.hdr.channel = chan;	/* TShdr.channel.  */
}


/* Send text to remote debug daemon - Pdebug.  */

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
      fputs_unfiltered (buf, gdb_stdtarg);
      return 0;
    default:
      return -1;
    }
}


/* Send env. string. Send multipart if env string too long and
   our protocol version allows multipart env string. 

   Returns > 0 if successful, 0 on error.  */

static int
nto_send_env (const char *env)
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

static int
nto_send_arg (const char *arg)
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
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  int rlen;
  /* CUDA: unblocking gdb 10.1 upgrade */
  int stashed = 0;
  unsigned tries;

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
	  if (gdbarch_byte_order (target_gdbarch ()) == BFD_ENDIAN_BIG)
	    err |= DSHDR_MSG_BIG_ENDIAN;
	  recv->pkt.hdr.cmd = err;
	  recv->pkt.err.err = EIO;
	  recv->pkt.err.err = EXTRACT_SIGNED_INTEGER (&recv->pkt.err.err,
						      4, byte_order);
	  rlen = sizeof (recv->pkt.err);
	  break;
	}

      /* CUDA: unblocking gdb 10.1 upgrade */
      /* Last cycled stashed a packet, and decremented tries, we don't need
       * another putpkt */
      if (!stashed)
	putpkt (tran, len);
      else
	stashed = 0;

      for (;;)
	{
	  rlen = getpkt (recv, 0);
	  if ((current_session->channelrd != SET_CHANNEL_TEXT)
	      || (rlen == -1))
	    break;
	  nto_incoming_text (&recv->text, rlen);
	}
      if (rlen == -1)		/* Getpkt returns -1 if MsgNAK received.  */
	{
	  printf_unfiltered ("MsgNak received - resending\n");
	  continue;
	}
      if ((rlen >= 0) && (recv->pkt.hdr.mid == tran->pkt.hdr.mid))
	break;

      /* CUDA: unblocking gdb 10.1 upgrade */
      if (recv->pkt.hdr.cmd == DShMsg_notify)
      {
	/* We received a notification, let's stash it for now. */
	current_session->pending.push(*recv);
	stashed = 1;

	/* Let's not consider this an attempt. */
	tries--;
      }
      else
      {
	error ("mid mismatch! tran_mid=%d, recv_mid=%d, unexpected_cmd=%d", tran->pkt.hdr.mid, recv->pkt.hdr.mid, recv->pkt.hdr.cmd);
      }
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
	  if (gdbarch_byte_order (target_gdbarch ()) != BFD_ENDIAN_BIG)
	    execute_command (buff, 0);
	}
      else
	{
	  char buff[sizeof(tran->buf)];

	  sprintf (buff, "set endian little");
	  if (gdbarch_byte_order (target_gdbarch ()) != BFD_ENDIAN_LITTLE)
	    execute_command (buff, 0);
	}
      recv->pkt.hdr.cmd &= ~DSHDR_MSG_BIG_ENDIAN;
      if (recv->pkt.hdr.cmd == DSrMsg_err)
	{
	  errno = errnoconvert (EXTRACT_SIGNED_INTEGER (&recv->pkt.err.err, 4,
							byte_order));
	  if (report_errors)
	    {
	      switch (recv->pkt.hdr.subcmd)
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
		  perror_with_name ("Remote");
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

static int
set_thread (const int th)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  DScomm_t tran, recv;

  nto_trace (0) ("set_thread(th %d pid %d, prev tid %ld)\n", th,
		 inferior_ptid.pid (), inferior_ptid.tid ());

  nto_send_init (&tran, DStMsg_select, DSMSG_SELECT_SET, SET_CHANNEL_DEBUG);
  tran.pkt.select.pid = inferior_ptid.pid ();
  tran.pkt.select.pid = EXTRACT_SIGNED_INTEGER ((gdb_byte*)&tran.pkt.select.pid, 4,
						byte_order);
  tran.pkt.select.tid = EXTRACT_SIGNED_INTEGER (&th, 4, byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.select), 1);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      nto_trace (0) ("Thread %d does not exist\n", th);
      return 0;
    }

  return 1;
}


/* Return nonzero if the thread TH is still alive on the remote system.  
   RECV will contain returned_tid. NOTE: Make sure this stays like that
   since we will use this side effect in other functions to determine
   first thread alive (for example, after attach).  */
template <class parent>
bool
qnx_remote_target<parent>::thread_alive (ptid_t th)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  int alive = 0;
  DScomm_t tran, recv;

  nto_trace (0) ("nto_thread_alive -- pid %d, tid %ld \n",
		 th.pid (), th.tid ());

  nto_send_init (&tran, DStMsg_select, DSMSG_SELECT_QUERY, SET_CHANNEL_DEBUG);
  tran.pkt.select.pid = th.pid ();
  tran.pkt.select.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.select.pid, 4,
						byte_order);
  tran.pkt.select.tid = th.tid ();
  tran.pkt.select.tid = EXTRACT_SIGNED_INTEGER (&tran.pkt.select.tid, 4,
						byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.select), 0);
  if (recv.pkt.hdr.cmd == DSrMsg_okdata)
    {
      /* Data is tidinfo. 
	Note: tid returned might not be the same as requested.
	If it is not, then requested thread is dead.  */
      uintptr_t ptidinfoaddr = (uintptr_t) &recv.pkt.okdata.data;
      struct tidinfo *ptidinfo = (struct tidinfo *) ptidinfoaddr;
      int returned_tid = EXTRACT_SIGNED_INTEGER (&ptidinfo->tid, 2,
						 byte_order);
      alive = (th.tid () == returned_tid) && ptidinfo->state;
    }
  else if (recv.pkt.hdr.cmd == DSrMsg_okstatus)
    {
      /* This is the old behaviour. It doesn't really tell us
      what is the status of the thread, but rather answers question:
      "Does the thread exist?". Note that a thread might have already
      exited but has not been joined yet; we will show it here as 
      alive an well. Not completely correct.  */
      int returned_tid = EXTRACT_SIGNED_INTEGER (&recv.pkt.okstatus.status, 4,
						 byte_order);
      alive = (th.tid () == returned_tid);
    }

  nto_trace (0) ("Thread %lu is alive = %d\n", th.tid (), alive);
  /* In case of a failure, return 0. This will happen when requested
    thread is dead and there is no alive thread with the larger tid.  */
  return alive;
}

static ptid_t
nto_get_thread_alive (remote_target *ops, ptid_t th)
{
  DScomm_t tran, recv;
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  int returned_tid;

  nto_send_init (&tran, DStMsg_select, DSMSG_SELECT_QUERY, SET_CHANNEL_DEBUG);
  tran.pkt.select.pid = th.pid ();
  tran.pkt.select.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.select.pid, 4,
						byte_order);
  tran.pkt.select.tid = th.tid ();
  tran.pkt.select.tid = EXTRACT_SIGNED_INTEGER (&tran.pkt.select.tid, 4,
						byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.select), 0);

  if (recv.pkt.hdr.cmd == DSrMsg_okdata)
    {
      /* Data is tidinfo. 
	Note: tid returned might not be the same as requested.
	If it is not, then requested thread is dead.  */
      const struct tidinfo *const ptidinfo = (struct tidinfo *) recv.pkt.okdata.data;
      returned_tid = EXTRACT_SIGNED_INTEGER (&ptidinfo->tid, 2,
					     byte_order);

      if (!ptidinfo->state)
	{
	  return nto_get_thread_alive (ops, ptid_t (th.pid (), 0,
						    returned_tid+1));
	}
    }
  else if (recv.pkt.hdr.cmd == DSrMsg_okstatus)
    {
      /* This is the old behaviour. It doesn't really tell us
      what is the status of the thread, but rather answers question:
      "Does the thread exist?". Note that a thread might have already
      exited but has not been joined yet; we will show it here as 
      alive and well. Not completely correct.  */
      returned_tid = EXTRACT_SIGNED_INTEGER (&recv.pkt.okstatus.status,
					     4, byte_order);
    }
  else
    return minus_one_ptid;

  return ptid_t (th.pid (), th.lwp (), returned_tid);
}

/* Clean up connection to a remote debugger.  */
template <class parent>
void
qnx_remote_target<parent>::close (void)
{
  nto_trace (0) ("nto_close\n");

  if (current_session->desc)
    {
      try
	{
	  DScomm_t tran, recv;

	  nto_send_init (&tran, DStMsg_disconnect, 0, SET_CHANNEL_DEBUG);
	  nto_send_recv (&tran, &recv, sizeof (tran.pkt.disconnect), 0);
	  serial_close (current_session->desc);

	  /* CUDA - set cuda_remote flag to be false */
	  set_cuda_remote_flag (false);
	}
      catch (const gdb_exception_error &ex)
	{
	  exception_fprintf (gdb_stderr, ex, "Error in nto_close\n");
	}
	
      current_session->desc = NULL;
      nto_remove_commands ();
    }
}

/* Reads procfs_info structure for the given process.

   Returns 1 on success, 0 otherwise.  */

static int
nto_read_procfsinfo (nto_procfs_info *pinfo)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  DScomm_t tran, recv;

  gdb_assert (pinfo != NULL && !! "pinfo must not be NULL\n");
  nto_send_init (&tran, DStMsg_procfsinfo, 0, SET_CHANNEL_DEBUG);
  tran.pkt.procfsinfo.pid = inferior_ptid.pid ();
  tran.pkt.procfsinfo.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.procfsinfo.pid,
						    4, byte_order);
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

static int
nto_read_procfsstatus (nto_procfs_status *pstatus)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  DScomm_t tran, recv;

  gdb_assert (pstatus != NULL && !! "pstatus must not be NULL\n");
  nto_send_init (&tran, DStMsg_procfsstatus, 0, SET_CHANNEL_DEBUG);
  tran.pkt.procfsstatus.pid = inferior_ptid.pid ();
  tran.pkt.procfsstatus.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.procfsstatus.pid,
						    4, byte_order);
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

static int
nto_start_remote (remote_target *ops)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  int orig_target_endian;
  DScomm_t tran, recv;

  nto_trace (0) ("nto_start_remote\n");

  for (;;)
    {
      orig_target_endian = (gdbarch_byte_order (target_gdbarch ()) == BFD_ENDIAN_BIG);

      /* Reset remote pdebug.  */
      SEND_CH_RESET;

      nto_send_init (&tran, DStMsg_connect, 0, SET_CHANNEL_DEBUG);

      tran.pkt.connect.major = HOST_QNX_PROTOVER_MAJOR;
      tran.pkt.connect.minor = HOST_QNX_PROTOVER_MINOR;

      nto_send_recv (&tran, &recv, sizeof (tran.pkt.connect), 0);

      if (recv.pkt.hdr.cmd != DSrMsg_err)
	break;
      if (orig_target_endian == (gdbarch_byte_order (target_gdbarch ()) == BFD_ENDIAN_BIG))
	break;
      /* Send packet again, with opposite endianness.  */
    }
  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      error ("Connection failed: %ld.",
	     (long) EXTRACT_SIGNED_INTEGER (&recv.pkt.err.err, 4, byte_order));
    }
  /* NYI: need to size transmit/receive buffers to allowed size in connect response.  */

  printf_unfiltered ("Remote target is %s-endian\n",
		     (gdbarch_byte_order (target_gdbarch ()) ==
		      BFD_ENDIAN_BIG) ? "big" : "little");

  /* Try to query pdebug for their version of the protocol.  */
  nto_send_init (&tran, DStMsg_protover, 0, SET_CHANNEL_DEBUG);
  tran.pkt.protover.major = HOST_QNX_PROTOVER_MAJOR;
  tran.pkt.protover.minor = HOST_QNX_PROTOVER_MINOR;
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.protover), 0);
  if ((recv.pkt.hdr.cmd == DSrMsg_err)
      && (EXTRACT_SIGNED_INTEGER (&recv.pkt.err.err, 4, byte_order)
	  == EINVAL))	/* Old pdebug protocol version 0.0.  */
    {
      current_session->target_proto_major = 0;
      current_session->target_proto_minor = 0;
    }
  else if (recv.pkt.hdr.cmd == DSrMsg_okstatus)
    {
      current_session->target_proto_major =
	EXTRACT_SIGNED_INTEGER (&recv.pkt.okstatus.status, 4, byte_order);
      current_session->target_proto_minor =
	EXTRACT_SIGNED_INTEGER (&recv.pkt.okstatus.status, 4, byte_order);
      current_session->target_proto_major =
	(current_session->target_proto_major >> 8) & DSMSG_PROTOVER_MAJOR;
      current_session->target_proto_minor =
	current_session->target_proto_minor & DSMSG_PROTOVER_MINOR;
    }
  else
    {
      error ("Connection failed (Protocol Version Query): %ld.",
	     (long) EXTRACT_SIGNED_INTEGER (&recv.pkt.err.err, 4, byte_order));
    }

  nto_trace (0) ("Pdebug protover %d.%d, GDB protover %d.%d\n",
			 current_session->target_proto_major,
			 current_session->target_proto_minor,
			 HOST_QNX_PROTOVER_MAJOR, HOST_QNX_PROTOVER_MINOR);

  /* Fail if remote is pdebug or a different CUDA version in cuda-gdbserver */
  cuda_qnx_version_handshake (ops);

  /* If we had an inferior running previously, gdb will have some internal
     states which we need to clear to start fresh.  */
  registers_changed ();
  nullify_last_target_wait_ptid ();
  inferior_ptid = null_ptid;

  return 1;
}

static void
nto_semi_init (remote_target *ops)
{
  DScomm_t tran, recv;

  nto_send_init (&tran, DStMsg_disconnect, 0, SET_CHANNEL_DEBUG);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.disconnect), 0);

  inferior_ptid = null_ptid;

  try
    {
      nto_start_remote (ops);
    }
  catch (const gdb_exception_error &ex)
    {
      exception_print (gdb_stderr, ex);
      reinit_frame_cache ();
      remote_unpush_target (ops);
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
template <class parent>
void
qnx_remote_target<parent>::open (const char *name, int from_tty)
{
  int tries = 0;
  void (*ofunc) (int);

  nto_trace (0) ("nto_open(name '%s', from_tty %d)\n", name, from_tty);

  nto_open_interrupted = 0;
  if (name == 0)
    error
      ("To open a remote debug connection, you need to specify what serial\ndevice is attached to the remote system (e.g. /dev/ttya).");

  target_preopen (from_tty);

  ofunc = signal(SIGINT, nto_open_break);

  while (tries < MAX_TRAN_TRIES && !nto_open_interrupted)
  {
    current_session->desc = remote_serial_open (name);

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
      puts_filtered ("Remote debugging using ");
      puts_filtered (name);
      puts_filtered ("\n");
    }
  /* CUDA - set cuda_remote flag to be true */
  set_cuda_remote_flag (true);

  remote_target *remote = cuda_new_remote_target ();

  struct remote_state *rs = remote->get_remote_state ();
  rs->remote_desc = current_session->desc;

  current_inferior ()->push_target (remote);	/* Switch to using remote target now.  */
  nto_add_commands ();
  nto_trace (3) ("nto_open() - push_target\n");

  inferior_ptid = null_ptid;

  /* Start the remote connection; if error (0), discard this target.
     In particular, if the user quits, be sure to discard it
     (we'd be in an inconsistent state otherwise).  */
  try
    {
      nto_start_remote (remote);
    }
  catch (const gdb_exception_error &ex)
    {
      remote_unpush_target (remote);

      nto_trace (0) ("nto_open() - pop_target\n");
    }

  /* CUDA - Initialize the remote target */
  cuda_remote_attach ();
}

/* Perform remote attach.
 *
 * Use caller provided recv as the reply may be used by the caller. */

static int
nto_attach_only (const int pid, DScomm_t *const recv)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  DScomm_t tran;

  nto_send_init (&tran, DStMsg_attach, 0, SET_CHANNEL_DEBUG);
  tran.pkt.attach.pid = pid;
  tran.pkt.attach.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.attach.pid, 4,
						byte_order);
  nto_send_recv (&tran, recv, sizeof (tran.pkt.attach), 0);

  if (recv->pkt.hdr.cmd != DSrMsg_okdata)
    {
      error (_("Failed to attach"));
      return 0;
    }
  return 1;
}

/* Attaches to a process on the target side.  Arguments are as passed
   to the `attach' command by the user.  This routine can be called
   when the target is not on the target-stack, if the target_can_run
   routine returns 1; in that case, it must push itself onto the stack.
   Upon exit, the target should be ready for normal operations, and
   should be ready to deliver the status of the process immediately
   (without waiting) to an upcoming target_wait call.  */
template <class parent>
void
qnx_remote_target<parent>::attach (const char *args, int from_tty)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  ptid_t ptid;
  struct inferior *inf;
  struct nto_inferior_data *inf_data;
  DScomm_t tran, *recv = &current_session->recv;

  if (inferior_ptid != null_ptid)
    nto_semi_init (this);

  nto_trace (0) ("nto_attach(args '%s', from_tty %d)\n",
			 args ? args : "(null)", from_tty);

  if (!args)
    error_no_arg ("process-id to attach");

  ptid = ptid_t (atoi (args));

  if (current_program_space->symfile_object_file != NULL)
   exec_file_attach (current_program_space->symfile_object_file->original_name, from_tty);

  if (from_tty)
    {
      printf_unfiltered ("Attaching to %s\n", target_pid_to_str (ptid).c_str ());
      gdb_flush (gdb_stdout);
    }

  if (!nto_attach_only (ptid.pid (), recv))
    return;

  /* Hack this in here, since we will bypass the notify.  */
  if (supports64bit())
    {
      current_session->cputype =
	EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._64.un.pidload.cputype, 2,
				byte_order);
      current_session->cpuid =
	EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._64.un.pidload.cpuid, 4,
				byte_order);
    }
  else
    {
      current_session->cputype =
	EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._32.un.pidload.cputype, 2,
				byte_order);
      current_session->cpuid =
	EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._32.un.pidload.cpuid, 4,
				byte_order);
    }
#ifdef QNX_SET_PROCESSOR_TYPE
  QNX_SET_PROCESSOR_TYPE (current_session->cpuid);	/* For mips.  */
#endif
  /* Get thread info as well.  */
  //ptid = nto_get_thread_alive (ptid);
  inferior_ptid = ptid_t (EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._32.pid, 4,
						  byte_order),
			  0,
			  EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._32.tid, 4,
						  byte_order));
  inf = current_inferior ();
  inf->attach_flag = 1;

  /* Remove LD_LIBRARY_PATH. In the future, we should fetch
   * it from the target and setup correctly prepended with
   * QNX_TARGET/<CPU> */
  inf->environment.set ("LD_LIBRARY_PATH", "");

  inferior_appeared (inf, ptid.pid ());

  if (current_program_space->symfile_object_file == NULL)
    {
      const int pid = ptid.pid ();
      struct dspidlist *pidlist = (struct dspidlist *)recv->pkt.okdata.data;

      /* Look for the binary executable name */
      nto_send_init (&tran, DStMsg_pidlist, DSMSG_PIDLIST_SPECIFIC,
		     SET_CHANNEL_DEBUG);
      tran.pkt.pidlist.pid = EXTRACT_UNSIGNED_INTEGER (&pid, 4, byte_order);
      tran.pkt.pidlist.tid = 0;
      nto_send_recv (&tran, recv, sizeof (tran.pkt.pidlist), 0);
      if (only_session.recv.pkt.hdr.cmd == DSrMsg_okdata)
	{
	  exec_file_attach (pidlist->name, from_tty);
	}
    }

  /* Initalize thread list.  */
  update_thread_list ();

 /* NYI: add symbol information for process.  */
  /* Turn the PIDLOAD into a STOPPED notification so that when gdb
     calls nto_wait, we won't cycle around.  */
  recv->pkt.hdr.cmd = DShMsg_notify;
  recv->pkt.hdr.subcmd = DSMSG_NOTIFY_STOPPED;
  recv->pkt.notify._32.pid = ptid.pid ();
  recv->pkt.notify._32.tid = ptid.tid ();
  recv->pkt.notify._32.pid = EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._32.pid, 4,
						     byte_order);
  recv->pkt.notify._32.tid = EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._32.tid, 4,
						     byte_order);

  inf_data = nto_inferior_data (inf);
  inf_data->has_execution = 1;
  inf_data->has_stack = 1;
  inf_data->has_registers = 1;
  inf_data->has_memory = 1;
}

template <class parent>
void
qnx_remote_target<parent>::post_attach (int pid)
{
  nto_trace (0) ("%s pid:%d\n", __func__, pid);
#ifdef SOLIB_CREATE_INFERIOR_HOOK
  if (current_program_space->exec_bfd ())
    SOLIB_CREATE_INFERIOR_HOOK (pid);
#endif
}

/* This takes a program previously attached to and detaches it.  After
   this is done, GDB can be used to debug some other program.  We
   better not have left any breakpoints in the target program or it'll
   die when it hits one.  */
template <class parent>
void
qnx_remote_target<parent>::detach (inferior *inf, int from_tty)
{
  DScomm_t tran, recv;
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  struct nto_inferior_data *inf_data;

  gdb_assert (inf != NULL);

  nto_trace (0) ("nto_detach()\n");

  if (from_tty)
    {
      const char *exec_file = get_exec_file (0);
      if (exec_file == 0)
	exec_file = "";

      printf_unfiltered ("Detaching from program: %s %d\n", exec_file,
			 inferior_ptid.pid ());
      gdb_flush (gdb_stdout);
    }
#if 0 /* XXX - where does args come from now? */
  if (args)
    {
      int sig = nto_gdb_signal_to_target (target_gdbarch (),
					  (enum gdb_signal)atoi (args));

      nto_send_init (&tran, DStMsg_kill, 0, SET_CHANNEL_DEBUG);
      tran.pkt.kill.signo = EXTRACT_SIGNED_INTEGER (&sig, 4, byte_order);
      nto_send_recv (&tran, &recv, sizeof (tran.pkt.kill), 1);
    }
#endif

  nto_send_init (&tran, DStMsg_detach, 0, SET_CHANNEL_DEBUG);
  tran.pkt.detach.pid = inferior_ptid.pid ();
  tran.pkt.detach.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.detach.pid, 4, byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.detach), 1);
  target_mourn_inferior (inferior_ptid);
  inferior_ptid = null_ptid;

  inf_data = nto_inferior_data (inf);
  inf_data->has_execution = 0;
  inf_data->has_stack = 0;
  inf_data->has_registers = 0;
  inf_data->has_memory = 0;
}


/* Tell the remote machine to resume.  */
template <class parent>
void
qnx_remote_target<parent>::resume (ptid_t ptid, int step, enum gdb_signal sig)
{
  DScomm_t tran, *const recv = &current_session->recv;
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  int signo;
  const int runone = ptid.tid () > 0;
  unsigned sizeof_pkt;

  nto_trace (0) ("nto_resume(pid %d, tid %ld, step %d, sig %d)\n",
		 ptid.pid (), ptid.tid (),
		 step, nto_gdb_signal_to_target (target_gdbarch (), sig));

  if (inferior_ptid == null_ptid)
    return;

  gdb_assert (inferior_ptid.pid () == current_inferior ()->pid);

  /* Select requested thread.  If minus_one_ptid is given, or selecting
     requested thread fails, select tid 1.  If tid 1 does not exist,
     first next available will be selected.  */
  if (ptid != minus_one_ptid)
    {
      ptid_t ptid_alive = nto_get_thread_alive (this,
						ptid_t (ptid.pid (), 0, inferior_ptid.tid ()));

      /* If returned thread is minus_one_ptid, then requested thread is
	 dead and there are no alive threads with tid > ptid_get_tid (ptid).
	 Try with first alive with tid >= 1.  */
      if (ptid_alive == minus_one_ptid)
	{
	  nto_trace (0) ("Thread %ld does not exist. Trying with tid >= 1\n",
			 ptid.tid ());
	  ptid_alive = nto_get_thread_alive (this, ptid_t (ptid.pid (), 0, 1));
	  nto_trace (1) ("First next tid found is: %ld\n", 
			 ptid_alive.tid ());
	}
      if (ptid_alive != minus_one_ptid)
	{
	  if (!set_thread (ptid_alive.tid ()))
	    {
	      nto_trace (0) ("Failed to set thread: %ld\n", 
			     ptid_alive.tid ());
	    }
	}
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
    EXTRACT_SIGNED_INTEGER (&tran.pkt.handlesig.sig_to_pass, 4, byte_order);
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
	  EXTRACT_SIGNED_INTEGER (&tran.pkt.kill.signo, 4, byte_order);
	nto_send_recv (&tran, recv, sizeof (tran.pkt.kill), 1);
      }

#if 0
  // aarch has single step but also does hardwar stepping.
  if (gdbarch_software_single_step_p (target_gdbarch ()))
    {
      /* Do not interfere with gdb logic. */
      nto_send_init (&tran, DStMsg_run, DSMSG_RUN,
		     SET_CHANNEL_DEBUG);
    }
  else
#endif
    {
      nto_send_init (&tran, DStMsg_run, (step || runone) ? DSMSG_RUN_COUNT
							 : DSMSG_RUN,
		     SET_CHANNEL_DEBUG);
    }
  if (supports64bit())
    {
      tran.pkt.run.step.count = 1;
      tran.pkt.run.step.count =
	EXTRACT_UNSIGNED_INTEGER (&tran.pkt.run.step.count, 4, byte_order);
      sizeof_pkt = sizeof (tran.pkt.run);
    }
  else
    {
      tran.pkt.run32.step.count = 1;
      tran.pkt.run32.step.count =
	EXTRACT_UNSIGNED_INTEGER (&tran.pkt.run32.step.count, 4, byte_order);
      sizeof_pkt = sizeof (tran.pkt.run32);
    }
  nto_send_recv (&tran, recv, sizeof_pkt, 1);
}

static void (*ofunc) (int);
#ifndef __MINGW32__
static void (*ofunc_alrm) (int);
#endif

/* Yucky but necessary globals used to track state in nto_wait() as a
   result of things done in nto_interrupt(), nto_interrupt_twice(),
   and nto_interrupt_retry().  */
static sig_atomic_t SignalCount = 0;	/* Used to track ctl-c retransmits.  */
static sig_atomic_t InterruptedTwice = 0;	/* Set in nto_interrupt_twice().  */
static sig_atomic_t WaitingForStopResponse = 0;	/* Set in nto_interrupt(), cleared in nto_wait().  */

#define QNX_TIMER_TIMEOUT 5
#define QNX_CTL_C_RETRIES 3

static void
nto_interrupt_retry (int signo)
{
  SignalCount++;
  if (SignalCount >= QNX_CTL_C_RETRIES)	/* Retry QNX_CTL_C_RETRIES times after original transmission.  */
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
interrupt_query (void)
{
  alarm (0);
  signal (SIGINT, ofunc);
#ifndef __MINGW32__
  signal (SIGALRM, ofunc_alrm);
#endif
  current_inferior ()->top_target ()->terminal_ours ();
  InterruptedTwice = 0;

  if (query
      ("Interrupted while waiting for the program.\n Give up (and stop debugging it)? "))
    {
      SignalCount = 0;
      target_mourn_inferior (inferior_ptid);
      //deprecated_throw_reason (RETURN_QUIT);
      quit ();
    }
  current_inferior ()->top_target ()->terminal_inferior ();
#ifndef __MINGW32__
  signal (SIGALRM, nto_interrupt_retry);
#endif
  signal (SIGINT, nto_interrupt_twice);
  alarm (QNX_TIMER_TIMEOUT);
}


/* The user typed ^C twice.  */
static void
nto_interrupt_twice (int signo)
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
template <class parent>
ptid_t
qnx_remote_target<parent>::wait (ptid_t ptid, struct target_waitstatus *status, target_wait_flags target_options)
{
  DScomm_t *const recv = &current_session->recv;
  ptid_t returned_ptid = null_ptid;

  nto_trace (0) ("nto_wait pid %d, tid %ld\n",
		 ptid.pid (), ptid.tid ());

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
	  /* CUDA: unblocking gdb 10.1 upgrade */
	  if (current_session->pending.empty())
	  {
	    len = getpkt (recv, 1);
	    if (len < 0)		/* Error - probably received MSG_NAK.  */
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
		return null_ptid;
	      }
	    }
	  }
	  else
	  {
	    *recv = current_session->pending.front();
	    current_session->pending.pop();
	    current_session->channelrd = recv->pkt.hdr.channel;
	  }

	  if (current_session->channelrd == SET_CHANNEL_TEXT)
	    nto_incoming_text (&recv->text, len);
	  else			/* DEBUG CHANNEL.  */
	    {
	      recv->pkt.hdr.cmd &= ~DSHDR_MSG_BIG_ENDIAN;
	      /* If we have sent the DStMsg_stop due to a ^C, we expect
	         to get the response, so check and clear the flag
	         also turn off the alarm - no need to retry,
	         we did not lose the packet.  */
	      if ((WaitingForStopResponse) && (recv->pkt.hdr.cmd == DSrMsg_ok))
		{
		  WaitingForStopResponse = 0;
		  status->set_signalled (GDB_SIGNAL_INT);
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
		  /* Handle old pdebug protocol behavior, where out of order msgs get dropped
		     version 0.0 does this, so we must resend after a notify.  */
		  if ((current_session->target_proto_major == 0)
		      && (current_session->target_proto_minor == 0))
		    {
		      if (WaitingForStopResponse)
			{
			  alarm (0);

			  /* Change the command to something other than notify
			     so we don't loop in here again - leave the rest of
			     the packet alone for nto_parse_notify() below!!!  */
			  recv->pkt.hdr.cmd = DSrMsg_ok;
			  nto_interrupt (SIGINT);
			}
		    }
		  returned_ptid = nto_parse_notify (recv, this, status);

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
         interrupt_query().  */
      if (InterruptedTwice)
	interrupt_query ();

      signal (SIGINT, ofunc);
#ifndef __MINGW32__
      signal (SIGALRM, ofunc_alrm);
#endif
    }

  recv->pkt.hdr.cmd = DSrMsg_ok;	/* To make us wait the next time.  */
  return returned_ptid;
}

static ptid_t
nto_parse_notify (const DScomm_t *const recv, remote_target *ops,
		  struct target_waitstatus *status)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  int pid, tid;
  CORE_ADDR stopped_pc = 0;
  struct inferior *inf;
  struct nto_inferior_data *inf_data;

  inf = current_inferior ();

  gdb_assert (inf != NULL);

  inf_data = nto_inferior_data (inf);

  gdb_assert (inf_data != NULL);

  nto_trace (0) ("nto_parse_notify(status) - subcmd %d\n",
			 recv->pkt.hdr.subcmd);

  pid = EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._32.pid, 4, byte_order);
  tid = EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._32.tid, 4, byte_order);
  if (tid == 0)
    tid = 1;

  switch (recv->pkt.hdr.subcmd)
    {
    case DSMSG_NOTIFY_PIDUNLOAD:
      /* Added a new struct pidunload_v3 to the notify.un.  This includes a
         faulted flag so we can tell if the status value is a signo or an
         exit value.  See dsmsgs.h, protoverminor bumped to 3. GP Oct 31 2002.  */
      if (current_session->target_proto_major > 0
	  || current_session->target_proto_minor >= 3)
	{
	  const int32_t *const pstatus = (supports64bit()) ? &recv->pkt.notify._64.un.pidunload_v3.status
				       : &recv->pkt.notify._32.un.pidunload_v3.status;

	  const int faulted = (supports64bit()) ? recv->pkt.notify._64.un.pidunload_v3.faulted
						: recv->pkt.notify._32.un.pidunload_v3.faulted;
	  if (faulted)
	    {
	      auto value =
		nto_gdb_signal_from_target
		  (target_gdbarch (), EXTRACT_SIGNED_INTEGER
				    (pstatus, 4, byte_order));
	      if (value)
		status->set_signalled (value);	/* Abnormal death.  */
	      else
		status->set_exited (value);	/* Normal death.  */
	    }
	  else
	    {
	      auto value =
		EXTRACT_SIGNED_INTEGER (pstatus, 4, byte_order);
	      status->set_exited (value);	/* Normal death, possibly with exit value.  */
	    }
	}
      else
	{
	  /* Only supported on 32-bit pdebugs, old ones. */
	  auto value=
	    nto_gdb_signal_from_target (target_gdbarch (),
					EXTRACT_SIGNED_INTEGER
					  (&recv->pkt.notify._32.un.pidunload.status,
					   4, byte_order));
	  if (value)
	    status->set_signalled (value);	/* Abnormal death.  */
	  else
	    status->set_exited (value);		/* Normal death.  */
	  /* Current inferior is gone, switch to something else */
	}
      inf_data->has_execution = 0;
      inf_data->has_stack = 0;
      inf_data->has_registers = 0;
      inf_data->has_memory = 0;
      break;
    case DSMSG_NOTIFY_BRK:
      if (supports64bit())
	{
	  inf_data->stopped_flags =
	    EXTRACT_UNSIGNED_INTEGER (&recv->pkt.notify._64.un.brk.flags, 4,
				      byte_order);
	  stopped_pc = EXTRACT_UNSIGNED_INTEGER (&recv->pkt.notify._64.un.brk.ip,
						 8, byte_order);
	}
      else
	{
	  inf_data->stopped_flags =
	    EXTRACT_UNSIGNED_INTEGER (&recv->pkt.notify._32.un.brk.flags, 4,
				      byte_order);
	  stopped_pc = EXTRACT_UNSIGNED_INTEGER (&recv->pkt.notify._32.un.brk.ip,
						 4, byte_order);
	}
      inf_data->stopped_pc = stopped_pc;
      /* NOTE: We do not have New thread notification. This will cause
	 gdb to think that breakpoint stop is really a new thread event if
	 it happens to be in a thread unknown prior to this stop.
	 We add new threads here to be transparent to the rest 
	 of the gdb.  */
      if (current_session->target_proto_major == 0 &&
	  current_session->target_proto_minor < 4)
	{
	  update_thread_list ();
	}
      /* Fallthrough.  */
    case DSMSG_NOTIFY_STEP:
      /* NYI: could update the CPU's IP register here.  */
      status->set_stopped (GDB_SIGNAL_TRAP);
      break;
    case DSMSG_NOTIFY_SIGEV:
      if (supports64bit())
	{
	  auto sig =
	    nto_gdb_signal_from_target (target_gdbarch (),
					EXTRACT_SIGNED_INTEGER
					  (&recv->pkt.notify._64.un.sigev.signo,
					   4, byte_order));
	  status->set_stopped (sig);
	}
      else
	{
	  auto sig =
	    nto_gdb_signal_from_target (target_gdbarch (),
					EXTRACT_SIGNED_INTEGER
					  (&recv->pkt.notify._32.un.sigev.signo,
					   4, byte_order));
	  status->set_stopped (sig);
	}
      break;
    case DSMSG_NOTIFY_PIDLOAD:
      if (supports64bit())
	{
	  current_session->cputype =
	    EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._64.un.pidload.cputype, 2,
				    byte_order);
	  current_session->cpuid =
	    EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._64.un.pidload.cpuid, 4,
				    byte_order);
	}
      else
	{
	  current_session->cputype =
	    EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._32.un.pidload.cputype, 2,
				    byte_order);
	  current_session->cpuid =
	    EXTRACT_SIGNED_INTEGER (&recv->pkt.notify._32.un.pidload.cpuid, 4,
				    byte_order);
	}
#ifdef QNX_SET_PROCESSOR_TYPE
      QNX_SET_PROCESSOR_TYPE (current_session->cpuid);	/* For mips.  */
#endif
      inf_data->has_execution = 1;
      inf_data->has_stack = 1;
      inf_data->has_registers = 1;
      inf_data->has_memory = 1;
      status->set_loaded ();
      break;
    case DSMSG_NOTIFY_TIDLOAD:
      {
	struct nto_thread_info *priv;

	if (nto_stop_on_thread_events)
	  status->set_stopped (GDB_SIGNAL_0);
	else
	  status->set_spurious ();
	if (supports64bit())
	  {
	    tid = EXTRACT_UNSIGNED_INTEGER (&recv->pkt.notify._64.un.thread_event.tid,
					    4, byte_order);
	  }
	else
	  {
	    tid = EXTRACT_UNSIGNED_INTEGER (&recv->pkt.notify._32.un.thread_event.tid,
					    4, byte_order);
	  }
	nto_trace (0) ("New thread event: tid %d\n", tid);

	priv = new nto_thread_info;
	priv->tid = tid;
//    priv->starting_ip = stopped_pc;
	add_thread_with_info (ops, ptid_t (pid, 0, tid), priv);
	if (status->kind () == TARGET_WAITKIND_SPURIOUS)
	  tid = inferior_ptid.tid ();
      }
      break;
    case DSMSG_NOTIFY_TIDUNLOAD:
      {
	ptid_t cur = ptid_t (pid, 0, tid);
	const int32_t *const ptid = (supports64bit()) ? &recv->pkt.notify._64.un.thread_event.tid
						: &recv->pkt.notify._32.un.thread_event.tid;
	const int tid_exited = EXTRACT_SIGNED_INTEGER (ptid, 4, byte_order);

	nto_trace (0) ("Thread destroyed: tid: %d active: %d\n", tid_exited,
		       tid);

	if (nto_stop_on_thread_events)
	  status->set_stopped (GDB_SIGNAL_0);
	else
	  status->set_spurious ();
	/* Must determine an alive thread for this to work. */
	switch_to_thread (ops, cur);
      }
      break;
#ifdef _DEBUG_WHAT_VFORK
    case DSMSG_NOTIFY_FORK:
      {
	int32_t child_pid = (supports64bit ()) ? recv->pkt.notify._64.un.fork_event.pid
					      : recv->pkt.notify._32.un.fork_event.pid;
	uint32_t vfork = (supports64bit ()) ? recv->pkt.notify._64.un.fork_event.vfork
					    : recv->pkt.notify._32.un.fork_event.vfork;

	nto_trace (0) ("DSMSG_NOTIFY_FORK %d\n", pid);
	inf_data->child_pid
	  = EXTRACT_SIGNED_INTEGER (&child_pid, 4, byte_order);
		nto_trace (0) ("inf data child pid: %d\n", inf_data->child_pid);
	auto child_ptid = ptid_t (inf_data->child_pid, 0, 1);
	inf_data->vfork
	  = EXTRACT_SIGNED_INTEGER (&vfork, 4, byte_order) & _DEBUG_WHAT_VFORK;
	if (inf_data->vfork)
	  status->set_vforked (child_ptid);
	else
	  status->set_forked (child_ptid);
	nto_trace (0) ("child_pid=%d\n", status->child_ptid ().pid ());
      }
      break;
    case DSMSG_NOTIFY_EXEC:
      {
	/* Notification format: pidload. */
	nto_trace (0) ("DSMSG_NOTIFY_EXEC %d, %s\n", pid,
		       supports64bit()?recv->pkt.notify._64.un.pidload.name:
				       recv->pkt.notify._32.un.pidload.name);
	auto pathname = make_unique_xstrdup (
				supports64bit()?recv->pkt.notify._64.un.pidload.name:
				recv->pkt.notify._32.un.pidload.name);
	status->set_execd (std::move (pathname));
      }
      break;
#endif
    case DSMSG_NOTIFY_DLLLOAD:
    case DSMSG_NOTIFY_DLLUNLOAD:
      status->set_spurious ();
      break;
    case DSMSG_NOTIFY_STOPPED:
      status->set_stopped (GDB_SIGNAL_0);
      break;
    default:
      warning ("Unexpected notify type %d", recv->pkt.hdr.subcmd);
      break;
    }
  nto_trace (0) ("nto_parse_notify: pid=%d, tid=%d ip=0x%s\n",
		 pid, tid, paddress (target_gdbarch (), stopped_pc));
  return ptid_t (pid, 0, tid);
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
	  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
	  memcpy (&foo, recv.pkt.okdata.data, sizeof (struct dscpuinfo));
	  cpuflags = EXTRACT_SIGNED_INTEGER (&foo.cpuflags, 4, byte_order);
	}
    }
  return cpuflags;
}

/* Fetch the regset, returning true if successful.  If supply is true,
   then supply these registers to gdb as well.  */
static int
fetch_regs (struct regcache *regcache, int regset, int supply)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
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
  tran.pkt.regrd.offset = 0;	/* Always get whole set.  */
  tran.pkt.regrd.size = EXTRACT_SIGNED_INTEGER (&len, 2,
						byte_order);

  rlen = nto_send_recv (&tran, &recv, sizeof (tran.pkt.regrd), 0);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    return 0;

  if (supply)
    nto_supply_regset (regcache, regset, recv.pkt.okdata.data, rlen);
  return 1;
}

/* Read register REGNO, or all registers if REGNO == -1, from the contents
   of REGISTERS.  */
template <class parent>
void
qnx_remote_target<parent>::fetch_registers (struct regcache *regcache, int regno)
{
  int regset;
  /* CUDA: unblocking gdb 10.1 upgrade */
  bool switch_back_to_null = false;

  nto_trace (0) ("nto_fetch_registers(regcache %p ,regno %d)\n",
		 regcache, regno);

  if (inferior_ptid == null_ptid)
    {
      /* CUDA: unblocking gdb 10.1 upgrade */
      nto_trace (0) ("ptid is null_ptid, forcing the temporary inferior switch\n");
      inferior_ptid = regcache->ptid();
      switch_back_to_null = true;
    }

  /* CUDA: unblocking gdb 10.1 upgrade */
  if (!set_thread (inferior_ptid.tid ()))
    {
      if (switch_back_to_null)
	{
	  inferior_ptid = null_ptid;
	}
      return;
    }

  if (regno == -1)
    {				/* Get all regsets.  */
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

  /* CUDA: unblocking gdb 10.1 upgrade */
  if (switch_back_to_null)
  {
    inferior_ptid = null_ptid;
  }
}

/* Prepare to store registers.  Don't have to do anything.  */
template <class parent>
void
qnx_remote_target<parent>::prepare_to_store (struct regcache *regcache)
{
   nto_trace (0) ("nto_prepare_to_store()\n");
}


/* Store register REGNO, or all registers if REGNO == -1, from the contents
   of REGISTERS.  */
template <class parent>
void
qnx_remote_target<parent>::store_registers (struct regcache *regcache, int regno)
{
  int len, regset, regno_regset;
  DScomm_t tran, recv;

  nto_trace (0) ("nto_store_registers(regno %d)\n", regno);

  if (inferior_ptid == null_ptid)
    return;

  if (!set_thread (inferior_ptid.tid ()))
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
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  long long addr;
  DScomm_t tran, recv;

  nto_trace (0) ("nto_write_bytes(to %s, from %p, len %d)\n",
		 paddress (target_gdbarch (), memaddr), myaddr, len);

  /* NYI: need to handle requests bigger than largest allowed packet.  */
  nto_send_init (&tran, DStMsg_memwr, 0, SET_CHANNEL_DEBUG);
  addr = memaddr;
  tran.pkt.memwr.addr = EXTRACT_UNSIGNED_INTEGER (&addr, 8,
						  byte_order);
  memcpy (tran.pkt.memwr.data, myaddr, len);
  nto_send_recv (&tran, &recv, offsetof (DStMsg_memwr_t, data) + len, 0);

  switch (recv.pkt.hdr.cmd)
    {
    case DSrMsg_ok:
      *xfered_len = len;
      break;
    case DSrMsg_okstatus:
      *xfered_len = EXTRACT_SIGNED_INTEGER (&recv.pkt.okstatus.status, 4,
					    byte_order);
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
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
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
						      byte_order);
      ask_len =
	((len - tot_len) >
	 DS_DATA_MAX_SIZE) ? DS_DATA_MAX_SIZE : (len - tot_len);
      tran.pkt.memrd.size = EXTRACT_SIGNED_INTEGER (&ask_len, 2,
						    byte_order);
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

/* Read AUXV from note.  */
static void
nto_core_read_auxv_from_note (bfd *abfd, asection *sect, void *pauxv_buf)
{
  struct auxv_buf *auxv_buf = (struct auxv_buf *)pauxv_buf;
  const char *sectname;
  unsigned int sectsize;
  const char qnx_core_info[] = ".qnx_core_info/";
  const unsigned int qnx_sectnamelen = 14;/* strlen (qnx_core_status).  */
  const char warning_msg[] = "Unable to read %s section from core.\n";
  nto_procfs_info info;
  int len;
  CORE_ADDR initial_stack;
  enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());

  sectname = bfd_section_name (sect);
  sectsize = bfd_section_size (sect);
  if (sectsize > sizeof (info))
    sectsize = sizeof (info);

  if (strncmp (sectname, qnx_core_info, qnx_sectnamelen) != 0)
    return;

  if (bfd_seek (abfd, sect->filepos, SEEK_SET) != 0)
    {
      warning (warning_msg, sectname);
      return;
    }
  len = bfd_bread ((gdb_byte *)&info, sectsize, abfd);
  if (len != sectsize)
    {
      warning (warning_msg, sectname);
      return;
    }
  initial_stack = extract_unsigned_integer
    ((gdb_byte *)&info.initial_stack, sizeof (info.initial_stack), byte_order);

  auxv_buf->len_read = nto_read_auxv_from_initial_stack
    (initial_stack, auxv_buf->readbuf, auxv_buf->len, IS_64BIT()? 16 : 8);

}

template <class parent>
enum target_xfer_status
qnx_remote_target<parent>::xfer_partial (enum target_object object,
					 const char *annex, gdb_byte *readbuf,
					 const gdb_byte *writebuf, const ULONGEST offset,
					 const ULONGEST len, ULONGEST *const xfered_len)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
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
      ULONGEST tempread = 0;
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

	  inf_rdata = nto_remote_inferior_data (inf);

	  if (inf_rdata->auxv == NULL)
	    {
	      const CORE_ADDR initial_stack
		= EXTRACT_SIGNED_INTEGER (&procfs_info.initial_stack,
					  arch_len / 8, byte_order);

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
      else if (current_program_space->exec_bfd ()
	       && current_program_space->exec_bfd ()->tdata.elf_obj_data != NULL
	       && current_program_space->exec_bfd ()->tdata.elf_obj_data->phdr != NULL)
	{
	  /* Fallback for older pdebug-s. They do not support
	     procfsinfo transfer, so we have to read auxv from
	     executable file.  */
	  uint64_t phdr = 0;
	  unsigned int phnum = 0;
	  gdb_byte *buff = readbuf;

	  /* Simply copy what we have in exec_bfd to the readbuf.  */
	  while (current_program_space->exec_bfd ()->tdata.elf_obj_data->phdr[phnum].p_type != PT_NULL)
	    {
	      if (current_program_space->exec_bfd ()->tdata.elf_obj_data->phdr[phnum].p_type == PT_PHDR)
		phdr = current_program_space->exec_bfd ()->tdata.elf_obj_data->phdr[phnum].p_vaddr;
	      phnum++;
	    }

	  /* Create artificial auxv, with AT_PHDR, AT_PHENT and AT_PHNUM
	     elements.  */
	  *(int*)buff = AT_PHNUM;
	  *(int*)buff = extract_signed_integer (buff, sizeof (int),
						byte_order);
	  buff += arch_len / 8; /* Includes 4 byte padding for auxv64_t */
	  if (arch_len == 32)
	    *(unsigned *)buff = EXTRACT_SIGNED_INTEGER (&phnum,
							sizeof (phnum),
							byte_order);
	  else
	    *(uint64_t *)buff = EXTRACT_SIGNED_INTEGER (&phnum,
							sizeof (phnum),
							byte_order);

	  buff += arch_len / 8;
	  *(int*)buff = AT_PHENT;
	  *(int*)buff = extract_signed_integer (buff, sizeof (int),
						byte_order);
	  buff += arch_len / 8;
	  if (arch_len == 32)
	    {
	      *(int*)buff = 0x20; /* sizeof(Elf32_Phdr) */
	      *(int*)buff = extract_signed_integer (buff, sizeof (int),
						    byte_order);
	    }
	  else
	    {
	      *(uint64_t*)buff = 56; /* sizeof(Elf64_Phdr) */
	      *(uint64_t*)buff = extract_signed_integer (buff,
							 sizeof (uint64_t),
							 byte_order);
	    }
	  buff += arch_len / 8;

	  *(int*)buff = AT_PHDR;
	  *(int*)buff = extract_signed_integer (buff, sizeof (int),
						byte_order);
	  buff += arch_len / 8;
	  if (arch_len == 32)
	    {
	      *(int*)buff = phdr;
	      *(int*)buff = extract_signed_integer (buff, sizeof (int),
						byte_order);
	    }
	  else
	    {
	      *(uint64_t*)buff = phdr;
	      *(uint64_t*)buff = extract_signed_integer (buff,
							 sizeof (uint64_t),
							 byte_order);
	    }
	  buff += arch_len / 8;
	  tempread = (int)(buff - readbuf);
	}
      tempread = std::min (tempread, len) - offset;
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
  if (parent::beneath ())
    return parent::beneath ()->xfer_partial (object, annex, readbuf,
					     writebuf, offset, len, xfered_len);
  return TARGET_XFER_E_IO;
}

template <class parent>
void
qnx_remote_target<parent>::files_info ()
{
  nto_trace (0) ("nto_files_info()\n");

  puts_filtered ("Debugging a target over a serial line.\n");
}

template <class parent>
void
qnx_remote_target<parent>::kill (void)
{
  struct target_waitstatus wstatus;
  ptid_t ptid;
  process_stratum_target *wait_target;

  nto_trace (0) ("nto_kill()\n");

  remove_breakpoints ();
  get_last_target_status (&wait_target, &ptid, &wstatus);

  /* Use catch_errors so the user can quit from gdb even when we aren't on
     speaking terms with the remote system.  */
  try
    {
      DScomm_t tran, recv;
      const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());

      if (inferior_ptid != null_ptid)
	{
	  nto_send_init (&tran, DStMsg_kill, DSMSG_KILL_PID, SET_CHANNEL_DEBUG);
	  tran.pkt.kill.signo = 9;	/* SIGKILL  */
	  tran.pkt.kill.signo = EXTRACT_SIGNED_INTEGER (&tran.pkt.kill.signo,
							4, byte_order);
	  nto_send_recv (&tran, &recv, sizeof (tran.pkt.kill), 0);
#if 0
	  nto_send_init (&tran, DStMsg_detach, 0, SET_CHANNEL_DEBUG);
	  tran.pkt.detach.pid = inferior_ptid.pid ();
	  tran.pkt.detach.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.detach.pid,
							4, byte_order);
	  nto_send_recv (&tran, &recv, sizeof (tran.pkt.detach), 1);
#endif
	}
    }
  catch (const gdb_exception_error &e)
    {
    }

  target_mourn_inferior (inferior_ptid);

  return;
}

template <class parent>
void
qnx_remote_target<parent>::mourn_inferior (void)
{
  DScomm_t tran, recv;
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  struct inferior *inf = current_inferior ();
  struct nto_inferior_data *inf_data;
  struct nto_remote_inferior_data *inf_rdata;

  gdb_assert (inf != NULL);

  inf_data = nto_inferior_data (inf);

  gdb_assert (inf_data != NULL);

  inf_rdata = nto_remote_inferior_data (inf);

  nto_trace (0) ("nto_mourn_inferior()\n");

  xfree (inf_rdata->auxv);
  inf_rdata->auxv = NULL;

  nto_send_init (&tran, DStMsg_detach, 0, SET_CHANNEL_DEBUG);
  tran.pkt.detach.pid = inferior_ptid.pid ();
  tran.pkt.detach.pid = EXTRACT_SIGNED_INTEGER (&tran.pkt.detach.pid,
						4, byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.detach), 1);

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

template <class parent>
bool
qnx_remote_target<parent>::can_create_inferior (void)
{
  return true;
}

template <class parent>
void
qnx_remote_target<parent>::create_inferior (const char *exec_file, const std::string &args,
					    char **env, int from_tty)
{
  DScomm_t tran, recv;
  unsigned argc;
  unsigned envc;
  char **start_argv, **argv, **pargv,  *p;
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

  inf_rdata = nto_remote_inferior_data (inf);

  gdb_assert (inf_rdata != NULL);

  inf_data->stopped_flags = 0;

  remove_breakpoints ();

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
      fprintf_unfiltered (gdb_stdout, "Remote: %s\n", exec_file);
    }

  if (current_session->desc == NULL)
    qnx_remote_target<parent>::open ("pty", 0);

  if (inferior_ptid != null_ptid)
    nto_semi_init (this);

  nto_trace (0) ("nto_create_inferior(exec_file '%s', environ)\n",
		 exec_file ? exec_file : "(null)");

  nto_send_init (&tran, DStMsg_env, DSMSG_ENV_CLEARENV, SET_CHANNEL_DEBUG);
  nto_send_recv (&tran, &recv, sizeof (DStMsg_env_t), 1);

  if (!current_session->inherit_env)
    {
      for (envc = 0; *env; env++, envc++)
	errors += !nto_send_env (*env);
      if (errors)
	warning ("Error(s) occured while sending environment variables.\n");
    }

  if (inf_rdata->remote_cwd.length ())
    {
      nto_send_init (&tran, DStMsg_cwd, DSMSG_CWD_SET, SET_CHANNEL_DEBUG);
      strcpy ((char *)tran.pkt.cwd.path, inf_rdata->remote_cwd.c_str ());
      nto_send_recv (&tran, &recv, offsetof (DStMsg_cwd_t, path)
		+ strlen ((const char *)tran.pkt.cwd.path) + 1, 1);
    }

  nto_send_init (&tran, DStMsg_env, DSMSG_ENV_CLEARARGV, SET_CHANNEL_DEBUG);
  nto_send_recv (&tran, &recv, sizeof (DStMsg_env_t), 1);

  pargv = buildargv (args.c_str ());
  if (pargv == NULL)
    malloc_failure (0);
  start_argv = nto_parse_redirection (pargv, &in, &out, &err);

  if (in[0])
    {
      if ((fd = ::open (in, O_RDONLY)) == -1)
	perror (in);
      else
	nto_fd_raw (fd);
    }

  if (out[0])
    {
      if ((fd = ::open (out, O_WRONLY)) == -1)
	perror (out);
      else
	nto_fd_raw (fd);
    }

  if (err[0])
    {
      if ((fd = ::open (err, O_WRONLY)) == -1)
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
      if (current_program_space->symfile_object_file != NULL)
	exec_file_attach (current_program_space->symfile_object_file->original_name, 0);

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
  freeargv ((char **)pargv);
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
  *p++ = '\0';			/* load_file */

  strcpy (p, in);
  p += strlen (p);
  *p++ = '\0';			/* stdin */

  strcpy (p, out);
  p += strlen (p);
  *p++ = '\0';			/* stdout */

  strcpy (p, err);
  p += strlen (p);
  *p++ = '\0';			/* stderr */

  nto_send_recv (&tran, &recv, offsetof (DStMsg_load_t, cmdline) + p - tran.pkt.load.cmdline + 1,
	    1);
  /* Comes back as an DSrMsg_okdata, but it's really a DShMsg_notify. */
  if (recv.pkt.hdr.cmd == DSrMsg_okdata)
    {
      struct inferior *inf;

      inferior_ptid  = nto_parse_notify (&recv, this, &status);
      inf = current_inferior ();
      inferior_appeared (inf, inferior_ptid.pid ());
      thread_info *thr = add_thread_silent (this, inferior_ptid);
      switch_to_thread (thr);
      inf->attach_flag = 1;
    }

  /* NYI: add the symbol info somewhere?  */
#ifdef SOLIB_CREATE_INFERIOR_HOOK
  if (current_program_space->exec_bfd ())
    SOLIB_CREATE_INFERIOR_HOOK (pid);
#endif
}

template <class parent>
int
qnx_remote_target<parent>::do_insert_breakpoint (struct gdbarch *gdbarch, struct bp_target_info *bp_tgt)
{
  CORE_ADDR addr = bp_tgt->placed_address;
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  DScomm_t tran, recv;
  size_t sizeof_pkt;

  nto_trace (0) ("nto_insert_breakpoint(addr %s) pid:%d\n", 
                 paddress (target_gdbarch (), addr),
		 inferior_ptid.pid ());

  nto_send_init (&tran, DStMsg_brk, DSMSG_BRK_EXEC, SET_CHANNEL_DEBUG);

  if (supports64bit())
    {
      tran.pkt.brk.size = nto_breakpoint_size (addr);
      tran.pkt.brk.addr = EXTRACT_UNSIGNED_INTEGER (&addr, 8, byte_order);
      sizeof_pkt = sizeof (tran.pkt.brk);
    }
  else
    {
      tran.pkt.brk32.size = nto_breakpoint_size (addr);
      tran.pkt.brk32.addr = EXTRACT_UNSIGNED_INTEGER (&addr, 4,
						      byte_order);
      sizeof_pkt = sizeof (tran.pkt.brk32);
    }
  nto_send_recv (&tran, &recv, sizeof_pkt, 0);
  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      nto_trace (0) ("FAIL\n");
    }
  return recv.pkt.hdr.cmd == DSrMsg_err;
}

/* To be called from breakpoint.c through
  current_target.to_insert_breakpoint.  */

template <class parent>
int
qnx_remote_target<parent>::insert_breakpoint (struct gdbarch *gdbarch,
					      struct bp_target_info *bp_tg_inf)
{
  if (bp_tg_inf == 0)
    {
      internal_error(__FILE__, __LINE__, _("Target info invalid."));
    }

  /* Must select appropriate inferior.  Due to our pdebug protocol,
     the following looks convoluted.  But in reality all we are doing is
     making sure pdebug selects an existing thread in the inferior_ptid.
     We need to switch pdebug internal current prp pointer.   */
  if (!set_thread (nto_get_thread_alive (NULL, ptid_t (inferior_ptid.pid ())).tid ()))
    {
      nto_trace (0) ("Could not set (pid,tid):(%d,%ld)\n",
		     inferior_ptid.pid (), inferior_ptid.tid ());
      return 0;
    }

  bp_tg_inf->placed_address = bp_tg_inf->reqstd_address;

  return do_insert_breakpoint (gdbarch, bp_tg_inf);
}


template <class parent>
int
qnx_remote_target<parent>::remove_breakpoint (CORE_ADDR addr, gdb_byte *contents_cache)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  DScomm_t tran, recv;

  nto_trace (0)	("nto_remove_breakpoint(addr %s, contents_cache %p) (pid %d)\n",
                 paddress (target_gdbarch (), addr), contents_cache,
		 inferior_ptid.pid ());

  /* Must select appropriate inferior.  Due to our pdebug protocol,
     the following looks convoluted.  But in reality all we are doing is
     making sure pdebug selects an existing thread in the inferior_ptid.
     We need to swithc pdebug internal current prp pointer.   */
  if (!set_thread (nto_get_thread_alive (NULL, ptid_t (inferior_ptid.pid ())).tid ()))
    {
      nto_trace (0) ("Could not set (pid,tid):(%d,%ld)\n",
		     inferior_ptid.pid (), inferior_ptid.tid ());
      return 0;
    }

  /* This got changed to send DSMSG_BRK_EXEC with a size of -1
     nto_send_init(DStMsg_brk, DSMSG_BRK_REMOVE, SET_CHANNEL_DEBUG).  */
  nto_send_init (&tran, DStMsg_brk, DSMSG_BRK_EXEC, SET_CHANNEL_DEBUG);
  if (supports64bit())
    {
      tran.pkt.brk.addr = EXTRACT_UNSIGNED_INTEGER (&addr, 8, byte_order);
      tran.pkt.brk.size = -1;
    }
  else
    {
      tran.pkt.brk32.addr = EXTRACT_UNSIGNED_INTEGER (&addr, 4,
						      byte_order);
      tran.pkt.brk32.size = -1;
    }
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.brk), 0);
  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      nto_trace (0) ("FAIL\n");
    }
  return recv.pkt.hdr.cmd == DSrMsg_err;
}

template <class parent>
int
qnx_remote_target<parent>::remove_breakpoint (struct gdbarch *gdbarch,
					      struct bp_target_info *bp_tg_inf,
					      enum remove_bp_reason reason)
{
  nto_trace (0) ("%s ( bp_tg_inf=%p )\n", __func__, bp_tg_inf);

  if (bp_tg_inf == 0)
    {
      internal_error (__FILE__, __LINE__, _("Target info invalid."));
    }

  return remove_breakpoint (bp_tg_inf->placed_address,
			    bp_tg_inf->shadow_contents);
}

template <class parent>
int
qnx_remote_target<parent>::remove_hw_breakpoint (struct gdbarch *gdbarch,
						 struct bp_target_info *bp_tg_inf)
{
  nto_trace (0) ("%s ( bp_tg_inf=%p )\n", __func__, bp_tg_inf);

  if (bp_tg_inf == 0)
    {
      internal_error (__FILE__, __LINE__, _("Target info invalid."));
    }

  return remove_breakpoint (bp_tg_inf->placed_address,
			    bp_tg_inf->shadow_contents);
}

#if defined(__CYGWIN__) || defined(__MINGW32__)
static void
slashify (char *buf)
{
  int i = 0;
  while (buf[i])
    {
      /* Not sure why we would want to leave an escaped '\', but seems
         safer.  */
      if (buf[i] == '\\')
	{
	  if (buf[i + 1] == '\\')
	    i++;
	  else
	    buf[i] = '/';
	}
      i++;
    }
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
  struct inferior *inf = current_inferior ();
  struct nto_remote_inferior_data *inf_rdata;

  gdb_assert (inf != NULL);

  inf_rdata = nto_remote_inferior_data (inf);

  if (args == 0)
    {
      printf_unfiltered ("You must specify a filename to send.\n");
      return;
    }

#if defined(__CYGWIN__) || defined(__MINGW32__)
  /* We need to convert back slashes to forward slashes for DOS
     style paths, else buildargv will remove them.  */
  slashify (args);
#endif
  argv = buildargv (args);

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

  /* full file name. Things like $cwd will be expanded.
     see source.c, openp and exec.c, file_command for more details. */
  gdb::unique_xmalloc_ptr<char> filename_opened;
  
  if ((fd = openp (NULL, OPF_TRY_CWD_FIRST, from,
                   O_RDONLY | O_BINARY, &filename_opened)) < 0)
    {
      printf_unfiltered ("Unable to open '%s': %s\n", from, strerror (errno));
      return;
    }

  nto_trace(0) ("Opened %s for reading\n", filename_opened.get ());

  if (nto_fileopen (to, QNX_WRITE_MODE, QNX_WRITE_PERMS) == -1)
    {
      printf_unfiltered ("Remote was unable to open '%s': %s\n", to,
			 strerror (errno));
      close (fd);
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
      inf_rdata->remote_exe = std::string {to};
      if (only_session.remote_exe.length () == 0)
	only_session.remote_exe = std::string {to};
    }

exit:
  nto_fileclose (fd);
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
  slashify (args);
#endif

  argv = buildargv (args);
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

  if ((fd = open (to, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0666)) == -1)
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
  freeargv ((char **)argv);
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

static void
nto_remove_commands (void)
{
  //extern struct cmd_list_element *cmdlist;

//  delete_cmd ("upload", &cmdlist);
// FIXME  delete_cmd ("download", &cmdlist);
}

static int nto_remote_fd = -1;

static int
nto_fileopen (char *fname, int mode, int perms)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
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
						   byte_order);
  tran.pkt.fileopen.perms = EXTRACT_SIGNED_INTEGER (&perms, 4,
						    byte_order);
  nto_send_recv (&tran, &recv, sizeof tran.pkt.fileopen, 0);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      errno = errnoconvert (EXTRACT_SIGNED_INTEGER (&recv.pkt.err.err,
						    4, byte_order));
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
  if (supports64bit())
    {
      tran.pkt.fileclose.mtime = 0;
      sizeof_pkt = sizeof (tran.pkt.fileclose);
    }
  else
    {
      tran.pkt.fileclose32.mtime = 0;
      sizeof_pkt = sizeof (tran.pkt.fileclose32);
    }
  nto_send_recv (&tran, &recv, sizeof_pkt, 1);
  nto_remote_fd = -1;
}

static int
nto_fileread (char *buf, int size)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  int len;
  DScomm_t tran, recv;

  nto_send_init (&tran, DStMsg_filerd, 0, SET_CHANNEL_DEBUG);
  tran.pkt.filerd.size = EXTRACT_SIGNED_INTEGER (&size, 2,
						 byte_order);
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

template <class parent>
bool
qnx_remote_target<parent>::can_run (void)
{
  nto_trace (0) ("%s ()\n", __func__);
  return 0;
}

template <class parent>
int
qnx_remote_target<parent>::can_use_hw_breakpoint (enum bptype type, int cnt, int othertype)
{
  return 1;
}

template <class parent>
bool
qnx_remote_target<parent>::has_registers (void)
{
  struct inferior *inf;

  inf = find_inferior_pid (this, inferior_ptid.pid ());
  if (!inf) return 0;

  return nto_inferior_data (inf)->has_registers;
}

template <class parent>
const char *
qnx_remote_target<parent>::thread_name (struct thread_info *ti)
{
  if (ti && ti->priv) {
    const std::string& name = static_cast<nto_thread_info *> (ti->priv.get())->name;
    return name.empty() ? NULL : name.c_str();
  }
  return NULL;
}

#ifndef NVIDIA_CUDA_GDB
/* CUDA FIXME: Getting the remote thread state is
 * broken/not implemented with cuda-gdb qnx.
 * It looks like this may happen in nto-procfs.c
 * but we don't use that interface for remote debugging.
 * We use pdebug.
 */
static const char *nto_thread_state_str[] =
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

template <class parent>
const char *
qnx_remote_target<parent>::extra_thread_info (struct thread_info *ti)
{
  if (ti != NULL && ti->priv != NULL)
    {
      nto_thread_info *priv = get_nto_thread_info (ti);

      if (priv->state < ARRAY_SIZE (nto_thread_state_str))
	return nto_thread_state_str [priv->state];
    }
  return "";
}
#endif

template <class parent>
std::string
qnx_remote_target<parent>::pid_to_str (ptid_t ptid)
{
  int pid, tid;
  struct thread_info *ti;
  std::string thread_id;

  pid = ptid.pid ();
  tid = ptid.tid ();

  ti = find_thread_ptid (this, ptid);
  nto_thread_info *info = ti ? (struct nto_thread_info *)ti->priv.get() : NULL;

  if (ti && info && info->name.length ())
    {
      thread_id = string_printf(" tid %d name \"%s\"", tid, info->name.c_str ());
    }
  else if (tid > 0)
    thread_id = string_printf(" tid %d", tid);

  return string_printf("pid %d%s", pid, thread_id.c_str ());
}

template <class parent>
bool
qnx_remote_target<parent>::has_execution (inferior *inf)
{
  return nto_inferior_data (inf)->has_execution;
}

template <class parent>
bool
qnx_remote_target<parent>::has_memory (void)
{
  struct inferior *inf;

  inf = find_inferior_pid (this, inferior_ptid.pid ());
  if (!inf) return 0;

  return nto_inferior_data (inf)->has_memory;
}

template <class parent>
bool
qnx_remote_target<parent>::has_stack (void)
{
  struct inferior *inf;

  inf = find_inferior_pid (this, inferior_ptid.pid ());
  if (!inf) return 0;

  return nto_inferior_data (inf)->has_stack;
}

template <class parent>
bool
qnx_remote_target<parent>::has_all_memory (void)
{
  struct inferior *inf;

  inf = find_inferior_pid (this, inferior_ptid.pid ());
  if (!inf) return 0;

  return nto_inferior_data (inf)->has_memory;
}

template <class parent>
int
qnx_remote_target<parent>::insert_fork_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

template <class parent>
int
qnx_remote_target<parent>::remove_fork_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

template <class parent>
int
qnx_remote_target<parent>::insert_vfork_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

template <class parent>
int
qnx_remote_target<parent>::remove_vfork_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

template <class parent>
int
qnx_remote_target<parent>::insert_exec_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

template <class parent>
int
qnx_remote_target<parent>::remove_exec_catchpoint (int pid)
{
  nto_trace (0) ("%s", __func__);
  return 0;
}

/* GDB makes assumption that after VFORK/FORK we are attached to both.
   This is not the case on QNX but to make gdb behave, we do attach
   to both anyway, regardless of the scenario. */
template <class parent>
void
qnx_remote_target<parent>::follow_fork (inferior *child_inf, ptid_t child_ptid,
					target_waitkind fork_kind, bool follow_child,
					bool detach_fork)
{
  struct inferior *const parent_inf = current_inferior ();
  int parent_pid;

  nto_trace (1) ("%s follow_child: %d\n", __func__, follow_child);

  gdb_assert (parent_inf != NULL);
  gdb_assert (parent_inf->pid == inferior_ptid.pid ());

  parent_pid = parent_inf->pid;

  nto_trace (0) ("Child pid: %d %s\n", child_ptid.pid (), target_waitkind_str (fork_kind));

  if (follow_child)
    {
      /* Attach to the child process, then detach from the parent. */
      struct nto_inferior_data *inf_data;
      struct nto_remote_inferior_data *c_inf_rdata, *p_inf_rdata;
      DScomm_t recv;

      /* To appease testsuite and be inline with linux. */
      if (info_verbose || nto_internal_debugging)
	{
	  fprintf_filtered (gdb_stdlog,
			    _("Attaching after process %d "
			      "%s to child process %d.\n"),
			      parent_pid, target_waitkind_str (fork_kind),
			      child_ptid.pid ());
	}

      if (!nto_attach_only (child_ptid.pid (), &recv))
	error (_("Could not attach to %d\n"), child_ptid.pid ());

      child_inf->attach_flag = parent_inf->attach_flag;
      copy_terminal_info (child_inf, parent_inf);
      child_inf->gdbarch = parent_inf->gdbarch;
      copy_inferior_target_desc_info (child_inf, parent_inf);
      child_inf->set_args (parent_inf->args ());

      /* Always clone pspace so that inferiors don't get their
       * associated file names messed up. */
      child_inf->aspace = new_address_space ();
      child_inf->pspace = new program_space (child_inf->aspace);
      child_inf->removable = 1;
      child_inf->symfile_flags = SYMFILE_NO_READ;
      /* Following the child. */
      set_current_program_space (child_inf->pspace);
      clone_program_space (child_inf->pspace, parent_inf->pspace);

      if (fork_kind == TARGET_WAITKIND_VFORKED)
	{
	  child_inf->vfork_parent = parent_inf;
	  child_inf->pending_detach = 0;
	  parent_inf->vfork_child = child_inf;
	  parent_inf->pending_detach = detach_fork;
	  parent_inf->waiting_for_vfork_done = 1;
	  /* Wait for child event EXEC or PIDUNLOAD */
	  parent_inf->pspace->breakpoints_not_allowed = 1;

	  inferior_ptid = child_ptid;
	  add_thread (this, inferior_ptid);
	}
      else
	{
	  if (detach_fork)
	    {
	      target_detach (NULL, 0);
	      inferior_ptid = child_ptid;
	      add_thread (this, inferior_ptid);
	    }
	  else
	    {
	      inferior_ptid = child_ptid;
	      add_thread (this, inferior_ptid);

	      child_inf->aspace = new_address_space ();
	      child_inf->pspace = new program_space (child_inf->aspace);
	      child_inf->removable = 1;
	      child_inf->symfile_flags = SYMFILE_NO_READ;
	      set_current_program_space (child_inf->pspace);
	      clone_program_space (child_inf->pspace, parent_inf->pspace);
	      solib_create_inferior_hook (0);
	    }
	}

      inf_data = nto_inferior_data (child_inf);
      inf_data->has_execution = 1;
      inf_data->has_stack = 1;
      inf_data->has_registers = 1;
      inf_data->has_memory = 1;

      p_inf_rdata = nto_remote_inferior_data (parent_inf);
      c_inf_rdata = nto_remote_inferior_data (child_inf);

      if (p_inf_rdata->remote_exe.length () == 0)
	c_inf_rdata->remote_exe = p_inf_rdata->remote_exe;
      if (p_inf_rdata->remote_cwd.length ())
	c_inf_rdata->remote_cwd = p_inf_rdata->remote_cwd;
    }
  else /* !follow_child */
    {
      struct nto_inferior_data *inf_data;

      if (!detach_fork)
	{
	  const ptid_t old_inferior_ptid = inferior_ptid;
	  struct program_space *const old_program_space
	    = current_program_space;
	  DScomm_t recv;

	  nto_trace (0)("%s: parent, attach child\n", __func__);
	  if (fork_kind == TARGET_WAITKIND_VFORKED)
	      error (_("Can not attach to vforked child and not follow it\n"));

	  if (!nto_attach_only (child_ptid.pid (), &recv))
	      error (_("Could not attach to %d\n"), child_ptid.pid ());

	  child_inf->attach_flag = parent_inf->attach_flag;
	  copy_terminal_info (child_inf, parent_inf);

	  inferior_ptid = nto_get_thread_alive (this, child_ptid);
	  add_thread (this, inferior_ptid);

	  child_inf->aspace = new_address_space ();
	  child_inf->pspace = new program_space (child_inf->aspace);
	  child_inf->removable = 1;
	  set_current_program_space (child_inf->pspace);
	  clone_program_space (child_inf->pspace, parent_inf->pspace);

	  solib_create_inferior_hook (0);

	  inf_data = nto_inferior_data (child_inf);
	  inf_data->has_execution = 1;
	  inf_data->has_stack = 1;
	  inf_data->has_registers = 1;
	  inf_data->has_memory = 1;

	  /* Restore */
	  set_current_program_space (old_program_space);
	  inferior_ptid = old_inferior_ptid;
	  set_thread (inferior_ptid.tid ());
	} else {
	  /* To appease testsuite and be inline with linux. */
	  if (info_verbose || nto_internal_debugging)
	    {
	      current_inferior ()->top_target ()->terminal_ours ();
	      fprintf_filtered (gdb_stdlog,
				"Detaching after %s from "
				"child process %d.\n",
				target_waitkind_str (fork_kind),
				child_ptid.pid ());
	    }
	}
    }
}

template <class parent>
bool
qnx_remote_target<parent>::supports_multi_process (void)
{
  return 1;
}

template <class parent>
int
qnx_remote_target<parent>::verify_memory (const gdb_byte *data, CORE_ADDR memaddr, ULONGEST size)
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

template <class parent>
void
qnx_remote_target<parent>::load (const char *args, int from_tty)
{
  generic_load (args, from_tty);
}

template <class parent>
bool
qnx_remote_target<parent>::is_async_p (void)
{
  /* Not yet. */
  return false;
}

template <class parent>
bool
qnx_remote_target<parent>::can_async_p (void)
{
  /* Not yet. */
  return 0;
}

template <class parent>
int
qnx_remote_target<parent>::get_trace_status (struct trace_status *ts)
{
   /* in 7.12 it dispatched to tdefault_get_trace_status */
  return -1;
}

template <class parent>
void
qnx_remote_target<parent>::remote_check_symbols (void)
{
  /* qSymbol used by remote_target::remote_check_symbols is not supported on QNX */
}

template <class parent>
bool
qnx_remote_target<parent>::supports_non_stop (void)
{
  /* Not yet. */
  return 0;
}

template <class parent>
const struct target_desc *
qnx_remote_target<parent>::read_description (void)
{
  if (ntoops_read_description)
    return ntoops_read_description (nto_get_cpuflags ());
  else
    return NULL;
}

static const target_info qnx_remote_target_info = {
  "qnx",
  N_("Remote serial target using the QNX Debugging Protocol"),
  N_("Debug a remote machine using the QNX Debugging Protocol.\n"
     "Specify the device it is connected to (e.g. /dev/ser1, <rmt_host>:<port>)\n"
     "or `pty' to launch `pdebug' for debugging."),
};

static void
update_threadnames (remote_target *ops)
{
  DScomm_t tran, recv;
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  struct dstidnames *tidnames = (struct dstidnames *) recv.pkt.okdata.data;
  int cur_pid;
  unsigned int numleft;

  nto_trace (0) ("%s ()\n", __func__);

  cur_pid = inferior_ptid.pid ();
  if(!cur_pid)
    {
      fprintf_unfiltered(gdb_stderr, "No inferior.\n");
      return;
    }

  do
    {
      unsigned int i, numtids;
      char *buf;

      nto_send_init (&tran, DStMsg_tidnames, 0, SET_CHANNEL_DEBUG);
      nto_send_recv (&tran, &recv, sizeof(tran.pkt.tidnames), 0);
      if (recv.pkt.hdr.cmd == DSrMsg_err)
        {
	  errno = errnoconvert (EXTRACT_SIGNED_INTEGER
				  (&recv.pkt.err.err, 4,
				   byte_order));
	  if (errno != EINVAL) /* Not old pdebug, but something else.  */
	    {
	      warning ("Warning: could not retrieve tidnames (errno=%d)\n",
		       errno);
	    }
	  return;
	}

      numtids = EXTRACT_UNSIGNED_INTEGER (&tidnames->numtids, 4,
					  byte_order);
      numleft = EXTRACT_UNSIGNED_INTEGER (&tidnames->numleft, 4,
					  byte_order);
      buf = (char *)tidnames + sizeof(*tidnames);
      for(i = 0 ; i < numtids ; i++)
	{
	  struct thread_info *ti;
	  struct nto_thread_info *priv;
	  ptid_t ptid;
	  int tid;
	  int namelen;
	  char *tmp;

	  tid = strtol(buf, &tmp, 10);
	  buf = tmp + 1; /* Skip the null terminator.  */
	  namelen = strlen(buf);

	  nto_trace (0) ("Thread %d name: %s\n", tid, buf);

	  ptid = ptid_t (cur_pid, 0, tid);
	  ti = find_thread_ptid (ops, ptid);
	  if (ti)
	    {
	      priv = get_nto_thread_info (ti);
	      if (priv)
		priv->name = buf;
	    }
	  buf += namelen + 1;
	}
    } while(numleft > 0);
}

template <class parent>
void
qnx_remote_target<parent>::update_thread_list (void)
{
  DScomm_t tran, recv;
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  int cur_pid, start_tid = 1, total_tids = 0, num_tids;
  struct dspidlist *pidlist = (struct dspidlist *) recv.pkt.okdata.data;
  struct tidinfo *tip;
  char subcmd;

  nto_trace (0) ("%s ()\n", __func__);

  cur_pid = inferior_ptid.pid ();
  if(!cur_pid){
    fprintf_unfiltered(gdb_stderr, "No inferior.\n");
    return;
  }
  subcmd = DSMSG_PIDLIST_SPECIFIC;

  do {
    nto_send_init (&tran, DStMsg_pidlist, subcmd, SET_CHANNEL_DEBUG );
    tran.pkt.pidlist.pid = EXTRACT_UNSIGNED_INTEGER (&cur_pid, 4,
						     byte_order);
    tran.pkt.pidlist.tid = EXTRACT_UNSIGNED_INTEGER (&start_tid, 4,
						     byte_order);
    nto_send_recv (&tran, &recv, sizeof(tran.pkt.pidlist), 0);
    if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      errno = errnoconvert (EXTRACT_SIGNED_INTEGER
			      (&recv.pkt.err.err, 4, byte_order));
      return;
    }
    if (recv.pkt.hdr.cmd != DSrMsg_okdata)
    {
      errno = EOK;
      nto_trace (1) ("msg not DSrMsg_okdata!\n");
      return;
    }
    num_tids = EXTRACT_UNSIGNED_INTEGER (&pidlist->num_tids, 4,
					 byte_order);
    for (tip =
	 (struct tidinfo *) &pidlist->name[(strlen(pidlist->name) + 1 + 3) & ~3];
	 tip->tid != 0; tip++ )
    {
      struct thread_info *new_thread;
      ptid_t ptid;

      tip->tid =  EXTRACT_UNSIGNED_INTEGER (&tip->tid, 2,
					    byte_order);
      ptid = ptid_t(cur_pid, 0, tip->tid);

      if (tip->tid < 0)
	{
	  //warning ("TID < 0\n");
	  continue;
	}

      new_thread = find_thread_ptid (this, ptid);
      if (!new_thread && tip->state != 0)
        new_thread = add_thread (this, ptid);
      if (new_thread && !new_thread->priv)
	{
	  nto_thread_info *priv = new nto_thread_info;

	  priv->tid = tip->tid;
	  new_thread->priv = std::unique_ptr<private_thread_info> (priv);
	}
      total_tids++;
    }
    subcmd = DSMSG_PIDLIST_SPECIFIC_TID;
    start_tid = total_tids + 1;
  } while(total_tids < num_tids);

  update_threadnames (this);
}

void
nto_pidlist (const char *args, int from_tty)
{
  DScomm_t tran, recv;
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
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
						 byte_order);
  tran.pkt.pidlist.tid = EXTRACT_SIGNED_INTEGER (&start_tid, 4,
						 byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.pidlist), 0);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    specific_tid_supported = 0;
  else
    specific_tid_supported = 1;

  while (1)
    {
      nto_send_init (&tran, DStMsg_pidlist, subcmd, SET_CHANNEL_DEBUG);
      tran.pkt.pidlist.pid = EXTRACT_SIGNED_INTEGER (&pid, 4,
						     byte_order);
      tran.pkt.pidlist.tid = EXTRACT_SIGNED_INTEGER (&start_tid, 4,
						     byte_order);
      nto_send_recv (&tran, &recv, sizeof (tran.pkt.pidlist), 0);
      if (recv.pkt.hdr.cmd == DSrMsg_err)
	{
	  errno = errnoconvert (EXTRACT_SIGNED_INTEGER
				  (&recv.pkt.err.err, 4,
				   byte_order));
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
	    printf_filtered ("%s - %ld/%ld\n", pidlist->name,
			     (long) EXTRACT_SIGNED_INTEGER (&pidlist->pid,
							    4, byte_order),
			     (long) EXTRACT_SIGNED_INTEGER (&tip->tid, 2,
							    byte_order));
	  total_tid++;
	}
      pid = EXTRACT_SIGNED_INTEGER (&pidlist->pid, 4, byte_order);
      if (specific_tid_supported)
	{
	  if (total_tid < EXTRACT_SIGNED_INTEGER
			    (&pidlist->num_tids, 4, byte_order))
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
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  struct dsmapinfo map;
  static struct dsmapinfo dmap;
  DStMsg_mapinfo_t *mapinfo = (DStMsg_mapinfo_t *) & tran.pkt;
  char subcmd;

  if (core_bfd != NULL)
    {				/* Have to implement corefile mapinfo.  */
      errno = EOK;
      return NULL;
    }

  subcmd = addr ? DSMSG_MAPINFO_SPECIFIC :
    first ? DSMSG_MAPINFO_BEGIN : DSMSG_MAPINFO_NEXT;
  if (elfonly)
    subcmd |= DSMSG_MAPINFO_ELF;

  nto_send_init (&tran, DStMsg_mapinfo, subcmd, SET_CHANNEL_DEBUG);
  mapinfo->addr = EXTRACT_UNSIGNED_INTEGER (&addr, 8/*FIXME*/, byte_order);
  nto_send_recv (&tran, &recv, sizeof (*mapinfo), 0);
  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      errno = errnoconvert (EXTRACT_SIGNED_INTEGER
			      (&recv.pkt.err.err, 4, byte_order));
      return NULL;
    }
  if (recv.pkt.hdr.cmd != DSrMsg_okdata)
    {
      errno = EOK;
      return NULL;
    }

  memset (&dmap, 0, sizeof (dmap));
  memcpy (&map, &recv.pkt.okdata.data[0], sizeof (map));
  dmap.ino = EXTRACT_UNSIGNED_INTEGER (&map.ino, 8, byte_order);
  dmap.dev = EXTRACT_SIGNED_INTEGER (&map.dev, 4, byte_order);

  dmap.text.addr = EXTRACT_UNSIGNED_INTEGER (&map.text.addr, 4,
					     byte_order);
  dmap.text.size = EXTRACT_SIGNED_INTEGER (&map.text.size, 4,
					   byte_order);
  dmap.text.flags = EXTRACT_SIGNED_INTEGER (&map.text.flags, 4,
					    byte_order);
  dmap.text.debug_vaddr =
    EXTRACT_UNSIGNED_INTEGER (&map.text.debug_vaddr, 4,
			      byte_order);
  dmap.text.offset = EXTRACT_UNSIGNED_INTEGER (&map.text.offset, 8,
					       byte_order);
  dmap.data.addr = EXTRACT_UNSIGNED_INTEGER (&map.data.addr, 4,
					     byte_order);
  dmap.data.size = EXTRACT_SIGNED_INTEGER (&map.data.size, 4,
					   byte_order);
  dmap.data.flags = EXTRACT_SIGNED_INTEGER (&map.data.flags, 4,
					    byte_order);
  dmap.data.debug_vaddr =
    EXTRACT_UNSIGNED_INTEGER (&map.data.debug_vaddr, 4, byte_order);
  dmap.data.offset = EXTRACT_UNSIGNED_INTEGER (&map.data.offset, 8,
					       byte_order);

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
      printf_filtered ("%s\n", dmp->name);
      printf_filtered ("\ttext=%08x bytes @ 0x%08x\n", dmp->text.size,
		       dmp->text.addr);
      printf_filtered ("\t\tflags=%08x\n", dmp->text.flags);
      printf_filtered ("\t\tdebug=%08x\n", dmp->text.debug_vaddr);
      printf_filtered ("\t\toffset=%016llx\n", dmp->text.offset);
      if (dmp->data.size)
	{
	  printf_filtered ("\tdata=%08x bytes @ 0x%08x\n", dmp->data.size,
			   dmp->data.addr);
	  printf_filtered ("\t\tflags=%08x\n", dmp->data.flags);
	  printf_filtered ("\t\tdebug=%08x\n", dmp->data.debug_vaddr);
	  printf_filtered ("\t\toffset=%016llx\n", dmp->data.offset);
	}
      printf_filtered ("\tdev=0x%x\n", dmp->dev);
      printf_filtered ("\tino=0x%llx\n", dmp->ino);
    }
}

template <class parent>
int
qnx_remote_target<parent>::insert_hw_breakpoint (struct gdbarch *gdbarch,
						 struct bp_target_info *bp_tg_inf)
{
  DScomm_t tran, recv;
  const enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);

  nto_trace (0) ("nto_insert_hw_breakpoint(addr %s, contents_cache %p)\n",
		 paddress (gdbarch, bp_tg_inf->placed_address),
		 bp_tg_inf->shadow_contents);

  if (bp_tg_inf == NULL)
    return -1;

  nto_send_init (&tran, DStMsg_brk, DSMSG_BRK_EXEC | DSMSG_BRK_HW,
		 SET_CHANNEL_DEBUG);
  if (supports64bit())
    {
      tran.pkt.brk.size = nto_breakpoint_size (bp_tg_inf->placed_address);
      tran.pkt.brk.addr
	= EXTRACT_SIGNED_INTEGER (&bp_tg_inf->placed_address, 4,
				  byte_order);
    }
  else
    {
      tran.pkt.brk32.size = nto_breakpoint_size (bp_tg_inf->placed_address);
      tran.pkt.brk32.addr
	= EXTRACT_SIGNED_INTEGER (&bp_tg_inf->placed_address, 4,
				  byte_order);
    }
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.brk), 0);
  return recv.pkt.hdr.cmd == DSrMsg_err;
}

template <class parent>
int
qnx_remote_target<parent>::hw_watchpoint (CORE_ADDR addr, int len, enum target_hw_bp_type type)
{
  DScomm_t tran, recv;
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  unsigned subcmd;

  nto_trace (0) ("nto_hw_watchpoint(addr %s, len %x, type %x)\n",
		 paddress (target_gdbarch (), addr), len, type);

  switch (type)
    {
    case 1:			/* Read.  */
      subcmd = DSMSG_BRK_RD;
      break;
    case 2:			/* Read/Write.  */
      subcmd = DSMSG_BRK_WR;
      break;
    default:			/* Modify.  */
      subcmd = DSMSG_BRK_MODIFY;
    }
  subcmd |= DSMSG_BRK_HW;

  nto_send_init (&tran, DStMsg_brk, subcmd, SET_CHANNEL_DEBUG);
  if (supports64bit())
    {
      tran.pkt.brk.addr = EXTRACT_UNSIGNED_INTEGER (&addr, 8,
						    byte_order);
      tran.pkt.brk.size = EXTRACT_SIGNED_INTEGER (&len, 4, byte_order);
    }
  else
    {
      tran.pkt.brk32.addr = EXTRACT_UNSIGNED_INTEGER (&addr, 4,
						      byte_order);
      tran.pkt.brk32.size = EXTRACT_SIGNED_INTEGER (&len, 4, byte_order);
    }
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.brk), 0);
  return recv.pkt.hdr.cmd == DSrMsg_err ? -1 : 0;
}

template <class parent>
int
qnx_remote_target<parent>::remove_watchpoint (CORE_ADDR addr, int len,
					      enum target_hw_bp_type type, struct expression *exp)
{
  return hw_watchpoint (addr, -1, type);
}

template <class parent>
int
qnx_remote_target<parent>::insert_watchpoint (CORE_ADDR addr, int len,
					      enum target_hw_bp_type type, struct expression *exp)
{
  return hw_watchpoint (addr, len, type);
}

template <class parent>
bool
qnx_remote_target<parent>::stopped_by_watchpoint (void)
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

#if 0
static struct tidinfo *
nto_thread_info (int pid, short tid)
{
  const enum bfd_endian byte_order = gdbarch_byte_order (target_gdbarch ());
  struct dspidlist *pidlist = (void *) recv.pkt.okdata.data;
  struct tidinfo *tip;

  nto_send_init (&tran, DStMsg_pidlist, DSMSG_PIDLIST_SPECIFIC_TID,
		 SET_CHANNEL_DEBUG);
  tran.pkt.pidlist.tid = EXTRACT_SIGNED_INTEGER (&tid, 2,
						 byte_order);
  tran.pkt.pidlist.pid = EXTRACT_SIGNED_INTEGER (&pid, 4,
						 byte_order);
  nto_send_recv (&tran, &recv, sizeof (tran.pkt.pidlist), 0);

  if (recv.pkt.hdr.cmd == DSrMsg_err)
    {
      nto_send_init (&tran, DStMsg_pidlist, DSMSG_PIDLIST_SPECIFIC,
		     SET_CHANNEL_DEBUG);
      tran.pkt.pidlist.pid = EXTRACT_SIGNED_INTEGER (&pid, 4,
						     byte_order);
      nto_send_recv (&tran, &recv, sizeof (tran.pkt.pidlist), 0);
      if (recv.pkt.hdr.cmd == DSrMsg_err)
	{
	  errno = errnoconvert (recv.pkt.err.err);
	  return NULL;
	}
    }

  /* Tidinfo structures are 4-byte aligned and start after name.  */
  for (tip = (void *) &pidlist->name[(strlen (pidlist->name) + 1 + 3) & ~3];
       tip->tid != 0; tip++)
    {
      if (tid == EXTRACT_SIGNED_INTEGER (&tip->tid, 2, byte_order))
	return tip;
    }

  return NULL;
}
#endif


static void
nto_remote_inferior_data_cleanup (struct inferior *const inf, void *const dat)
{
  struct nto_remote_inferior_data *const inf_rdata
    = (struct nto_remote_inferior_data *const) dat;

  if (dat)
    {
      xfree (inf_rdata->auxv);
      inf_rdata->auxv = NULL;
      inf_rdata->remote_exe.clear ();
      inf_rdata->remote_cwd.clear ();
    }
  xfree (dat);
}


static struct nto_remote_inferior_data *
nto_remote_inferior_data (struct inferior *const inf)
{
  struct nto_remote_inferior_data *inf_data;

  gdb_assert (inf != NULL);

  inf_data = (struct nto_remote_inferior_data *)inferior_data (inf,
	       nto_remote_inferior_data_reg);
  if (inf_data == NULL)
    {
      inf_data = new struct nto_remote_inferior_data;
      set_inferior_data (inf, nto_remote_inferior_data_reg, inf_data);
    }

  return inf_data;
}

static void
set_nto_exe (const char *args, int from_tty,
	     struct cmd_list_element *c)
{
  struct inferior *const inf = current_inferior ();
  struct nto_remote_inferior_data *const inf_rdat
    = nto_remote_inferior_data (inf);

  inf_rdat->remote_exe = current_session->remote_exe;
}

static void
show_nto_exe (struct ui_file *file, int from_tty,
              struct cmd_list_element *c, const char *value)
{
  struct inferior *const inf = current_inferior ();
  struct nto_remote_inferior_data *const inf_rdat
    = nto_remote_inferior_data (inf);

  deprecated_show_value_hack (file, from_tty, c, inf_rdat->remote_exe.c_str ());
}

static void
set_nto_cwd (const char *args, int from_tty, struct cmd_list_element *c)
{
  struct inferior *const inf = current_inferior ();
  struct nto_remote_inferior_data *const inf_rdat
    = nto_remote_inferior_data (inf);

  inf_rdat->remote_cwd = current_session->remote_cwd;
}

static void
show_nto_cwd (struct ui_file *file, int from_tty,
              struct cmd_list_element *c, const char *value)
{
  struct inferior *const inf = current_inferior ();
  struct nto_remote_inferior_data *const inf_rdat
    = nto_remote_inferior_data (inf);

  deprecated_show_value_hack (file, from_tty, c, inf_rdat->remote_cwd.c_str ());
}

void _initialize_nto ();
void
_initialize_nto ()
{
  add_target (qnx_remote_target_info, qnx_remote_target<remote_target>::open);

  nto_remote_inferior_data_reg
    = register_inferior_data_with_cleanup (NULL, nto_remote_inferior_data_cleanup);

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
