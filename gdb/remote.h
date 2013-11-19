/* Remote target communications for serial-line targets in custom GDB protocol
   Copyright (C) 1999-2013 Free Software Foundation, Inc.

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

#ifndef REMOTE_H
#define REMOTE_H

#include "remote-notif.h"

struct target_desc;

enum packet_support
{
  PACKET_SUPPORT_UNKNOWN = 0,
  PACKET_ENABLE,
  PACKET_DISABLE
};

/* This type describes each known response to the qSupported
   packet.  */
struct protocol_feature
{
  /* The name of this protocol feature.  */
  const char *name;

  /* The default for this protocol feature.  */
  enum packet_support default_support;

  /* The function to call when this feature is reported, or after
     qSupported processing if the feature is not supported.
     The first argument points to this structure.  The second
     argument indicates whether the packet requested support be
     enabled, disabled, or probed (or the default, if this function
     is being called at the end of processing and this feature was
     not reported).  The third argument may be NULL; if not NULL, it
     is a NUL-terminated string taken from the packet following
     this feature's name and an equals sign.  */
  void (*func) (const struct protocol_feature *, enum packet_support,
                const char *);

  /* The corresponding packet for this feature.  Only used if
     FUNC is remote_supported_packet.  */
  int packet;
};

/* Read a packet from the remote machine, with error checking, and
   store it in *BUF.  Resize *BUF using xrealloc if necessary to hold
   the result, and update *SIZEOF_BUF.  If FOREVER, wait forever
   rather than timing out; this is used (in synchronous mode) to wait
   for a target that is is executing user code to stop.  */

extern void getpkt (char **buf, long *sizeof_buf, int forever);
extern int getpkt_sane (char **buf, long *sizeof_buf, int forever);

/* Send a packet to the remote machine, with error checking.  The data
   of the packet is in BUF.  The string in BUF can be at most PBUFSIZ
   - 5 to account for the $, # and checksum, and for a possible /0 if
   we are debugging (remote_debug) and want to print the sent packet
   as a string.  */

extern int putpkt (char *buf);
extern int putpkt_binary (char *buf, int cnt);
int remote_escape_output (const unsigned char *, int, unsigned char *, int *, int);
int remote_unescape_input (const unsigned char *, int, unsigned char *, int);

extern int hex2bin (const char *hex, gdb_byte *bin, int count);

extern int bin2hex (const gdb_byte *bin, char *hex, int count);

extern char *unpack_varlen_hex (char *buff, ULONGEST *result);

extern void async_remote_interrupt_twice (void *arg);

void register_remote_g_packet_guess (struct gdbarch *gdbarch, int bytes,
				     const struct target_desc *tdesc);
void register_remote_support_xml (const char *);

void remote_file_put (const char *local_file, const char *remote_file,
		      int from_tty);
void remote_file_get (const char *remote_file, const char *local_file,
		      int from_tty);
void remote_file_delete (const char *remote_file, int from_tty);

bfd *remote_bfd_open (const char *remote_file, const char *target);

int remote_filename_p (const char *filename);
int remote_query_attached (int pid);

long get_remote_packet_size (void);
extern int remote_register_number_and_offset (struct gdbarch *gdbarch,
					      int regnum, int *pnum,
					      int *poffset);

extern void remote_notif_get_pending_events (struct notif_client *np);
#endif
