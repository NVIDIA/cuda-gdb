/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2017-2026 NVIDIA Corporation
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

#include "cuda-nto-protocol.h"
#include "cuda-tdep-server.h"
#include "cuda/cuda-packet-format.h"
#include "gdbsupport/array-view.h"
#include "gdbsupport/common-debug.h"
#include "gdbsupport/common-defs.h"
#include "gdbsupport/scope-exit.h"
#include "gdbsupport/scoped_fd.h"
#include "server.h"

#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/* QNX SDP806 gnulib defines select -> rpl_select, which clobbers the
   DSMsg_union_t `select' member used below.  We never call select() here, so
   drop the macro.  QNX-only, as the collision is specific to QNX headers.  */
#ifdef __QNX__
#undef select
#endif

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/* This intentionally has external linkage: inferior_pid is accessed from
   gdb/cuda/cuda-notifications.c, which is compiled into cuda-gdbserver under
   __QNXHOST__.  Do NOT move this definition into an unnamed namespace.  */
uint32_t inferior_pid;

namespace
{

constexpr size_t POLL_FD_PDEBUG = 0;
constexpr size_t POLL_FD_HOST = 1;
constexpr size_t POLL_FD_COUNT = 2;
constexpr int POLL_TIMEOUT_MS = 500;
constexpr int PDEBUG_CONNECT_ATTEMPTS = 50;
constexpr unsigned int PDEBUG_CONNECT_RETRY_US = 100000;

/* NUL-terminated by construction so it can be passed safely to C APIs
   (setenv, access, ...) without an extra std::string copy.  */
constexpr char RAMDISK_PATH[] = "/ramdisk";

/* Kept open by the gdbserver process so the watchdog can detect when
   cuda-gdbserver exits and should terminate pdebug.  */
scoped_fd pdebug_watchdog_keepalive_fd;

constexpr char NO_RESPONSE = '\xff';

#define cuda_nto_debug_printf(debug_enabled, fmt, ...)                        \
  debug_prefixed_printf_cond (debug_enabled, "cuda-nto-server", fmt,          \
			      ##__VA_ARGS__)

void check_poll_revents (const pollfd &fd, const char *name);

/* Write the full VIEW to FD, retrying on EINTR and advancing past partial
   writes.  Calls error() on a hard syscall failure or if write() makes no
   progress (which would otherwise livelock).  */
void
write_all (int fd, gdb::array_view<const gdb_byte> view)
{
  const int length = static_cast<int> (view.size ());
  while (!view.empty ())
    {
      ssize_t n = write (fd, view.data (), view.size ());
      if (n < 0)
	{
	  if (errno == EINTR)
	    continue;
	  error ("Failed to write to fd %d: %s", fd, strerror (errno));
	}
      if (n == 0)
	error ("Failed to write to fd %d: write made no progress "
	       "after %d of %d bytes",
	       fd, length - static_cast<int> (view.size ()), length);
      view = view.slice (n);
    }
}

void
set_close_on_exec (int fd)
{
  const int flags = fcntl (fd, F_GETFD);
  if (flags < 0 || fcntl (fd, F_SETFD, flags | FD_CLOEXEC) < 0)
    error ("Failed to set close-on-exec on pipe: %s", strerror (errno));
}

void
ensure_directory_exists (const char *path)
{
  struct stat st;
  if (stat (path, &st) == 0)
    {
      if (!S_ISDIR (st.st_mode))
	error ("%s exists but is not a directory", path);
      return;
    }

  const int saved_errno = errno;
  if (saved_errno != ENOENT)
    error ("Cannot inspect %s: %s", path, strerror (saved_errno));

  if (mkdir (path, S_IRWXU | S_IRWXG | S_IRWXO) == 0)
    return;

  const int mkdir_errno = errno;
  if (mkdir_errno != EEXIST)
    error ("Cannot create %s: %s", path, strerror (mkdir_errno));

  if (stat (path, &st) != 0 || !S_ISDIR (st.st_mode))
    error ("%s exists but is not a directory", path);
}

/* Mirror gdb's currently focused pid (from DStMsg_select on the host stream)
   into ::inferior_pid, used for QNX CUDA notifications.  */
void update_inferior_pid_from_host_frame (gdb_byte *buf, int raw_length,
					  bool debug_enabled);

class cuda_nto_server
{
public:
  cuda_nto_server (scoped_fd &&pdebug_fd, scoped_fd &&host_fd,
		   bool debug_enabled)
      : m_pdebug_fd (std::move (pdebug_fd)), m_host_fd (std::move (host_fd)),
	m_debug_enabled (debug_enabled)
  {
    gdb_assert (s_instance == nullptr);
    s_instance = this;
  }

  ~cuda_nto_server () { s_instance = nullptr; }

  static cuda_nto_server &
  get_nto_server ()
  {
    gdb_assert (s_instance != nullptr);
    return *s_instance;
  }

  int
  write_inferior_memory (CORE_ADDR addr, gdb::array_view<const gdb_byte> src)
  {
    return qnx_write_inferior_memory (m_pdebug_fd.get (), m_debug_enabled,
				      m_mid++, addr, src.data (),
				      static_cast<ssize_t> (src.size ()));
  }

  void run () ATTRIBUTE_NORETURN;

private:
  DISABLE_COPY_AND_ASSIGN (cuda_nto_server);

  void handle_pdebug_packet (gdb_byte *buf, int length);
  void handle_host_packet (gdb_byte *buf, int length);

  scoped_fd m_pdebug_fd;
  scoped_fd m_host_fd;
  bool m_debug_enabled = false;
  uint8_t m_mid = 0;
  char m_pending_response = NO_RESPONSE;

  static cuda_nto_server *s_instance;
};

cuda_nto_server *cuda_nto_server::s_instance = nullptr;

void
create_ramdisk ()
{
  ensure_directory_exists (RAMDISK_PATH);

  if (system ("devb-ram disk name=ram ram nodinit blk ramdisk=13m,cache=512k")
      || system ("waitfor /dev/ram0"))
    error ("Failed to create a ram disk\n");
  if (system ("mkqnx6fs -q /dev/ram0"))
    error ("Failed to create the ram disk file system\n");
  const std::string mount_cmd
      = std::string ("mount -t qnx6 /dev/ram0 ") + RAMDISK_PATH;
  if (system (mount_cmd.c_str ()))
    error ("Failed to mount the ram disk\n");
}

void
ensure_functional_tmpdir ()
{
  const char *env_tmpdir = getenv ("TMPDIR");
  std::string tmpdir;

  if (env_tmpdir != nullptr && env_tmpdir[0] != '\0'
      && std::string_view (env_tmpdir) != RAMDISK_PATH)
    {
      warning ("TMPDIR is set to %s, skipping using the ram disk", env_tmpdir);
      tmpdir = env_tmpdir;
    }
  else
    {
      /* Create the ram disk if its mount point does not yet exist. */
      if (access (RAMDISK_PATH, F_OK) != 0)
	{
	  create_ramdisk ();
	  fprintf (stderr, "Created ram disk mounted at %s\n", RAMDISK_PATH);
	}
      else
	fprintf (stderr, "Using ram disk mounted at %s\n", RAMDISK_PATH);

      fprintf (stderr, "Setting TMPDIR to %s\n", RAMDISK_PATH);
      if (setenv ("TMPDIR", RAMDISK_PATH, 1) != 0)
	error ("%s\n", strerror (errno));
      tmpdir = RAMDISK_PATH;
    }

  /* Probe tmpdir by creating (and then removing) a unique directory and
     a FIFO inside it.  mkdtemp replaces the trailing XXXXXX in place
     with a unique suffix.  */
  std::string dir = tmpdir;
  if (!dir.empty () && dir.back () != '/')
    dir += '/';
  dir += "cuda-gdbserver-fifo.XXXXXX";
  if (mkdtemp (dir.data ()) == nullptr)
    error ("%s\n"
	   "Failed to create a temporary directory in %s, program cannot "
	   "continue.\n",
	   strerror (errno), tmpdir.c_str ());
  SCOPE_EXIT { (void)rmdir (dir.c_str ()); };

  const std::string fifo = dir + "/test_fifo";
  if (mkfifo (fifo.c_str (), S_IRUSR | S_IWUSR) != 0)
    error ("%s\n"
	   "Failed to create a pipe in %s, program cannot continue.\n",
	   strerror (errno), tmpdir.c_str ());
  SCOPE_EXIT { (void)unlink (fifo.c_str ()); };
}

void
launch_pdebug (char *argv[], int first_arg, int argc, int port)
{
  /* Must outlive pdebug_argv: its c_str() is referenced from the vector
     and is read by execvp below.  */
  const std::string pdebug_port = std::to_string (port);

  /* pdebug argv: program name, passed-in args, "-f", port, NULL terminator.
     "-f" prevents pdebug from daemonizing and changing its PID, which is
     required by the watchdog logic below.  */
  std::vector<const char *> pdebug_argv;
  pdebug_argv.reserve ((argc - first_arg) + 4);
  pdebug_argv.push_back ("pdebug");
  pdebug_argv.insert (pdebug_argv.end (), argv + first_arg, argv + argc);
  pdebug_argv.push_back ("-f");
  pdebug_argv.push_back (pdebug_port.c_str ());
  pdebug_argv.push_back (nullptr);

  /* In order to reliably kill the forked pdebug, we first fork a watchdog
     process that reads from a pipe that's never written to.  The read
     returns when the other end of the pipe is closed, i.e. when the parent
     (cuda-qnx-gdbserver) exits for any reason.  We fork pdebug from the
     watchdog and kill it after the pipe is closed.  pdebug must be run with
     -f to prevent daemonization and changing of its PID.  */
  std::array<int, 2> fd{ -1, -1 };
  if (pipe (fd.data ()) != 0)
    error ("Failed to create pipe: %s", strerror (errno));
  scoped_fd read_fd (fd[0]);
  scoped_fd write_fd (fd[1]);
  set_close_on_exec (read_fd.get ());
  set_close_on_exec (write_fd.get ());

  switch (fork ())
    {
    case -1:
      error ("Fork failed: %s", strerror (errno));
    case 0:
      {
	/* Watchdog process.  */
	write_fd = scoped_fd ();
	if (setpgid (0, 0) != 0)
	  fprintf (stderr, "Failed to set watchdog process group: %s\n",
		   strerror (errno));

	const pid_t pdebug_pid = fork ();
	switch (pdebug_pid)
	  {
	  case -1:
	    fprintf (stderr, "Failed to fork pdebug: %s\n", strerror (errno));
	    _exit (EXIT_FAILURE);
	  case 0:
	    /* pdebug process.  */
	    read_fd = scoped_fd ();
	    execvp ("pdebug", const_cast<char **> (pdebug_argv.data ()));
	    fprintf (stderr, "Failed to exec pdebug: %s\n", strerror (errno));
	    _exit (EXIT_FAILURE);
	  }

	for (;;)
	  {
	    char dummy;
	    const ssize_t n = read (read_fd.get (), &dummy, 1);
	    if (n < 0 && errno == EINTR)
	      continue;
	    if (n <= 0)
	      break;
	  }
	if (kill (pdebug_pid, SIGTERM) != 0 && errno != ESRCH)
	  fprintf (stderr, "Failed to terminate pdebug: %s\n",
		   strerror (errno));
	/* Reap pdebug so it isn't left as a zombie under the watchdog.  */
	int pdebug_status;
	while (waitpid (pdebug_pid, &pdebug_status, 0) < 0 && errno == EINTR)
	  ;
	_exit (EXIT_SUCCESS);
      }
    default:
      /* gdbserver process.  */
      read_fd = scoped_fd ();
      pdebug_watchdog_keepalive_fd = std::move (write_fd);
      fprintf (stderr, "cuda-gdbserver started\n");
      break;
    }
}

scoped_fd
connect_pdebug_local (uint16_t port)
{
  sockaddr_in pdebug_addr{};
  pdebug_addr.sin_family = AF_INET;
  pdebug_addr.sin_port = htons (port);
  pdebug_addr.sin_addr.s_addr = htonl (INADDR_LOOPBACK);

  for (int attempt = 0; attempt < PDEBUG_CONNECT_ATTEMPTS; ++attempt)
    {
      scoped_fd pdebug_fd (socket (AF_INET, SOCK_STREAM, 0));
      if (pdebug_fd.get () < 0)
	error ("Failed to create pdebug socket: %s", strerror (errno));

      if (connect (pdebug_fd.get (),
		   reinterpret_cast<sockaddr *> (&pdebug_addr),
		   sizeof (pdebug_addr))
	  == 0)
	return pdebug_fd;

      const int saved_errno = errno;
      if ((saved_errno != ECONNREFUSED && saved_errno != EINTR
	   && saved_errno != ETIMEDOUT && saved_errno != ENOENT)
	  || attempt + 1 == PDEBUG_CONNECT_ATTEMPTS)
	error ("Cannot connect to pdebug: %s", strerror (saved_errno));

      usleep (PDEBUG_CONNECT_RETRY_US);
    }

  gdb_assert_not_reached (
      "pdebug connection retry loop should return or throw");
}

scoped_fd
accept_host_connection (uint16_t port)
{
  scoped_fd server_fd (socket (AF_INET, SOCK_STREAM, 0));
  if (server_fd.get () < 0)
    error ("Failed to create socket: %s", strerror (errno));

  const int reuse_addr = 1;
  if (setsockopt (server_fd.get (), SOL_SOCKET, SO_REUSEADDR, &reuse_addr,
		  sizeof (reuse_addr))
      != 0)
    warning ("Failed to set SO_REUSEADDR: %s", strerror (errno));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons (port);
  addr.sin_addr.s_addr = INADDR_ANY;
  if (bind (server_fd.get (), reinterpret_cast<sockaddr *> (&addr),
	    sizeof (addr))
      < 0)
    error ("Failed to bind socket: %s", strerror (errno));
  if (listen (server_fd.get (), 5) < 0)
    error ("Failed to listen on socket: %s", strerror (errno));

  sockaddr_in host_addr{};
  int accepted_fd;
  do
    {
      socklen_t client_addr_len = sizeof (host_addr);
      accepted_fd = accept (server_fd.get (),
			    reinterpret_cast<sockaddr *> (&host_addr),
			    &client_addr_len);
    }
  while (accepted_fd < 0 && errno == EINTR);

  scoped_fd host_fd (accepted_fd);
  if (host_fd.get () < 0)
    error ("Failed to accept incoming connection: %s", strerror (errno));

  return host_fd;
}

void
update_inferior_pid_from_host_frame (gdb_byte *buf, int raw_length,
				     bool debug_enabled)
{
  if (raw_length < 3)
    return;

  const unsigned char cmd_raw = buf[1] & ~DSHDR_MSG_BIG_ENDIAN;
  if (cmd_raw != DStMsg_attach && cmd_raw != DStMsg_select
      && cmd_raw != DStMsg_detach && cmd_raw != DStMsg_disconnect)
    return;

  /* DSMSG_SELECT_QUERY inspects a pid's threads; only SET changes focus.  */
  if (cmd_raw == DStMsg_select && buf[2] != DSMSG_SELECT_SET)
    return;

  DScomm_t packet;
  const int decoded_length = unpack_pdebug_packet (&packet, buf);
  if (decoded_length < static_cast<int> (sizeof (DShdr)))
    return;

  if (packet.pkt.hdr.channel != 0
      && packet.pkt.hdr.channel != SET_CHANNEL_DEBUG)
    return;

  const auto set_inferior_pid
      = [debug_enabled] (uint32_t new_pid, const char *reason) {
	  if (new_pid == inferior_pid)
	    return;
	  inferior_pid = new_pid;
	  if (!debug_enabled)
	    return;
	  if (new_pid == 0)
	    cuda_nto_debug_printf (debug_enabled, "Inferior pid cleared (%s)",
				   reason);
	  else
	    cuda_nto_debug_printf (debug_enabled, "Inferior pid: %u (%s)",
				   new_pid, reason);
	};

  switch (packet.pkt.hdr.cmd & ~DSHDR_MSG_BIG_ENDIAN)
    {
    case DStMsg_attach:
      {
	if (decoded_length < static_cast<int> (sizeof (DStMsg_attach_t)))
	  {
	    warning ("Truncated DStMsg_attach request");
	    return;
	  }
	const int32_t new_pid = packet.pkt.attach.pid;
	if (new_pid > 0)
	  set_inferior_pid (static_cast<uint32_t> (new_pid), "DStMsg_attach");
	return;
      }
    case DStMsg_select:
      {
	if (packet.pkt.hdr.subcmd != DSMSG_SELECT_SET)
	  return;
	if (decoded_length < static_cast<int> (sizeof (DStMsg_select_t)))
	  {
	    warning ("Truncated DStMsg_select request");
	    return;
	  }
	const int32_t new_pid = packet.pkt.select.pid;
	if (new_pid > 0)
	  set_inferior_pid (static_cast<uint32_t> (new_pid), "DStMsg_select");
	return;
      }
    case DStMsg_detach:
      {
	if (decoded_length < static_cast<int> (sizeof (DStMsg_detach_t)))
	  {
	    warning ("Truncated DStMsg_detach request");
	    return;
	  }
	const int32_t detach_pid = packet.pkt.detach.pid;
	if (detach_pid > 0
	    && static_cast<uint32_t> (detach_pid) == inferior_pid)
	  set_inferior_pid (0, "DStMsg_detach");
	return;
      }
    case DStMsg_disconnect:
      set_inferior_pid (0, "DStMsg_disconnect");
      return;
    default:
      return;
    }
}

void
cuda_nto_server::handle_pdebug_packet (gdb_byte *buf, int length)
{
  if (m_debug_enabled)
    gdbserver_debug_print (PDB, GDB, buf + 1);

  if (m_pending_response != NO_RESPONSE)
    {
      DScomm_t response;

      unpack_pdebug_packet (&response, buf);
      switch (m_pending_response)
	{
	case DStMsg_load:
	  inferior_pid = response.pkt.notify.pid;
	  if (m_debug_enabled)
	    cuda_nto_debug_printf (m_debug_enabled,
				   "Inferior pid: %u (DStMsg_load)",
				   inferior_pid);
	  break;
	default:
	  error ("Unhandled response");
	  break;
	}
      m_pending_response = NO_RESPONSE;
    }

  write_all (m_host_fd.get (), gdb::make_array_view (buf, length));
}

void
cuda_nto_server::handle_host_packet (gdb_byte *buf, int length)
{
  int raw_length;

  /* There can be several packets from the host in the buffer.
     We need to process each of them in turn.  */
  do
    {
      raw_length = get_raw_pdebug_packet_size (buf, length);
      const unsigned char cmd_raw = buf[1] & ~DSHDR_MSG_BIG_ENDIAN;

      if (cmd_raw == DStMsg_cuda)
	{
	  bool null_packet = false;
	  int packet_length;
	  DScomm_t cuda_packet;

	  if (m_debug_enabled)
	    gdbserver_debug_print (GDB, SRV, buf + 1);

	  packet_length = unpack_pdebug_packet (&cuda_packet, buf);
	  m_mid = cuda_packet.pkt.hdr.mid;

	  char *const packet_start
	      = reinterpret_cast<char *> (cuda_packet.pkt.cuda.data);
	  packet_length
	      -= packet_start - reinterpret_cast<char *> (cuda_packet.buf);
	  if (packet_length < 0)
	    error ("Truncated DStMsg_cuda packet");

	  const std::string_view request (packet_start,
					  static_cast<size_t> (packet_length));
	  if (startswith (request, cuda_packet_prefix))
	    {
	      cuda_packet_decoder decoder (request);
	      cuda_packet_encoder encoder (QNX_CUDA_PACKET_DATA_SIZE + 1);
	      handle_cuda_packet (decoder, encoder);

	      const std::string_view response = encoder.view ();
	      if (response.size () > QNX_CUDA_PACKET_DATA_SIZE)
		error ("CUDA response length exceeds QNX pdebug capacity");
	      response.copy (packet_start, response.size ());
	      packet_start[response.size ()] = '\0';
	      packet_length = pack_cuda_packet (
		  buf, packet_start, static_cast<int> (response.size ()),
		  m_mid++);
	    }
	  else if (startswith (request, "vCUDA"))
	    {
	      int new_packet_len = 0;
	      const int handled
		  = handle_vCuda (request,
				  gdb::make_array_view (
				      packet_start, QNX_CUDA_PACKET_DATA_SIZE),
				  &new_packet_len);
	      if (handled)
		{
		  if (new_packet_len < 0
		      || static_cast<size_t> (new_packet_len)
			     > QNX_CUDA_PACKET_DATA_SIZE)
		    error (
			"vCUDA response length exceeds QNX pdebug capacity");
		  packet_length = pack_cuda_packet (buf, packet_start,
						    new_packet_len, m_mid++);
		}
	      else
		{
		  packet_length = pack_pdebug_packet (buf, nullptr, 0);
		  null_packet = true;
		}
	    }
	  else
	    {
	      if (m_debug_enabled)
		{
		  char *p = packet_start;
		  for (; isalpha (*p) || *p == ':'; ++p)
		    ;

		  warning ("Unsupported operation: %.*s",
			   static_cast<int> (p - packet_start), packet_start);
		}

	      /* Send an empty packet to notify the host this packet isn't
		 supported.  */
	      packet_length = pack_pdebug_packet (buf, nullptr, 0);
	      null_packet = true;
	    }

	  if (m_debug_enabled && !null_packet)
	    gdbserver_debug_print (SRV, GDB, buf + 1);

	  write_all (m_host_fd.get (),
		     gdb::make_array_view (buf, packet_length));
	}
      else
	{
	  if (m_debug_enabled)
	    gdbserver_debug_print (GDB, PDB, buf + 1);

	  update_inferior_pid_from_host_frame (buf, raw_length,
					       m_debug_enabled);
	  if (cmd_raw == DStMsg_load)
	    m_pending_response = DStMsg_load;

	  write_all (m_pdebug_fd.get (),
		     gdb::make_array_view (buf, raw_length));
	}

      buf += raw_length;
      length -= raw_length;
    }
  while (length > 0);
}

void ATTRIBUTE_NORETURN
cuda_nto_server::run ()
{
  std::array<pollfd, POLL_FD_COUNT> fds = {};
  fds[POLL_FD_PDEBUG].fd = m_pdebug_fd.get ();
  fds[POLL_FD_PDEBUG].events = POLLIN;
  fds[POLL_FD_HOST].fd = m_host_fd.get ();
  fds[POLL_FD_HOST].events = POLLIN;

  fprintf (stderr, "cuda-gdbserver starting main loop\n");
  for (;;)
    {
      std::array<gdb_byte, MAX_PACKET_SIZE> buf;
      int n = poll (fds.data (), fds.size (), POLL_TIMEOUT_MS);
      if (n < 0)
	{
	  if (errno == EINTR)
	    continue;
	  error ("poll failed: %s", strerror (errno));
	}
      if (n == 0)
	continue;

      check_poll_revents (fds[POLL_FD_PDEBUG], "pdebug");
      check_poll_revents (fds[POLL_FD_HOST], "host");

      if (fds[POLL_FD_PDEBUG].revents & POLLIN)
	{
	  do
	    n = read (m_pdebug_fd.get (), buf.data (), buf.size ());
	  while (n < 0 && errno == EINTR);
	  if (n == 0)
	    error ("pdebug exited");
	  if (n < 0)
	    error ("Failed to read from pdebug socket: %s", strerror (errno));

	  handle_pdebug_packet (buf.data (), n);
	}

      if (fds[POLL_FD_HOST].revents & POLLIN)
	{
	  do
	    n = read (m_host_fd.get (), buf.data (), buf.size ());
	  while (n < 0 && errno == EINTR);
	  if (n == 0)
	    error ("Host exited");
	  if (n < 0)
	    error ("Failed to read from host socket: %s", strerror (errno));

	  handle_host_packet (buf.data (), n);
	}
    }
}

/* Normally this would be initialized when constructing the target object.
   For cuda nto we don't make use of the target.  This initializer mirrors
   the cuda_linux_process_target constructor.  */
void
initialize_cuda_remote ()
{
  /* Check the required CUDA debugger files are present */
  if (cuda_get_debugger_api ())
    {
      warning (
	  "CUDA support disabled. Could not obtain the CUDA debugger API");
      return;
    }

  cuda_debugging_enabled = true;
}

ATTRIBUTE_PRINTF (1, 2) void show_usage (const char *err, ...)
{
  FILE *stream = stdout;

  if (err != nullptr)
    {
      stream = stderr;
      va_list args;
      va_start (args, err);
      fprintf (stream, "ERROR: ");
      vfprintf (stream, err, args);
      fprintf (stream, "\n");
      va_end (args);
    }

  fprintf (stream,
	   "Usage: cuda-gdbserver PORT [OPTIONS] [--pdebug-args ARGS]\n"
	   "Options:\n"
	   "  -h, --help        - This help\n"
	   "  -d, --debug       - Print debug info\n"
	   "      --pdebug-args - Pass any additional arguments to pdebug\n");
  exit (err != nullptr ? EXIT_FAILURE : EXIT_SUCCESS);
}

int
parse_port (const char *arg)
{
  char *end = nullptr;
  errno = 0;
  const long port = strtol (arg, &end, 10);

  if (arg[0] == '\0' || *end != '\0' || errno == ERANGE || port <= 0
      || port > 0xfffe)
    show_usage ("invalid port %s", arg);

  return static_cast<int> (port);
}

void
check_poll_revents (const pollfd &fd, const char *name)
{
  if (fd.revents & (POLLERR | POLLNVAL))
    error ("%s socket error (revents=0x%x)", name,
	   static_cast<unsigned int> (fd.revents));

  if ((fd.revents & POLLHUP) && !(fd.revents & POLLIN))
    error ("%s socket closed", name);
}

} // namespace

/* QNX-host implementation of the writer declared in cuda-tdep-server.cc.  */
extern int cuda_write_inferior_memory (CORE_ADDR memaddr,
				       gdb::array_view<const gdb_byte> src);

int
cuda_write_inferior_memory (CORE_ADDR memaddr,
			    gdb::array_view<const gdb_byte> src)
{
  return cuda_nto_server::get_nto_server ().write_inferior_memory (memaddr,
								   src);
}

void ATTRIBUTE_NORETURN
captured_main (int argc, char *argv[])
{
  if (argc < 2)
    show_usage ("missing PORT argument.");

  const int port = parse_port (argv[1]);

  bool debug_enabled = false;
  int first_pdebug_arg = argc;
  for (int i = 2; i < argc; ++i)
    {
      const std::string_view arg = argv[i];
      if (arg == "-h" || arg == "--help")
	show_usage (nullptr);
      else if (arg == "-d" || arg == "--debug")
	debug_enabled = true;
      else if (arg == "--pdebug-args")
	{
	  first_pdebug_arg = i + 1;
	  break;
	}
      else
	show_usage ("invalid option %s", argv[i]);
    }

  /* TODO: find a better way to create pipes on QNX, see bug 1949586 */
  ensure_functional_tmpdir ();

  cuda_gdb_setup ();

  launch_pdebug (argv, first_pdebug_arg, argc, port + 1);

  initialize_cuda_remote ();
  fprintf (stderr, "cuda-gdbserver cuda initialized\n");

  scoped_fd pdebug_fd (
      connect_pdebug_local (static_cast<uint16_t> (port + 1)));
  scoped_fd host_fd (accept_host_connection (static_cast<uint16_t> (port)));
  cuda_nto_server server (std::move (pdebug_fd), std::move (host_fd),
			  debug_enabled);
  server.run ();
}
