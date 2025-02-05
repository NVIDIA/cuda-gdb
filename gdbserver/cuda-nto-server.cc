/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2017-2024 NVIDIA Corporation
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

#include "gdbsupport/common-defs.h"
#include "gdbsupport/common-exceptions.h"
#include "server.h"
#include "cuda-nto-protocol.h"
#include "cuda-tdep-server.h"
#include "cuda/cuda-utils.h"
#include "cuda/cuda-notifications.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <limits.h>
#include <netinet/in.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define POLL_FD_COUNT 2
#define POLL_FD_PDEBUG 0
#define POLL_FD_HOST 1
#define POLL_TIMEOUT_MS 500

#define RAMDISK_PATH "/ramdisk"
#define NO_RESPONSE 0xff

int pdebug_sockfd; /* used in cuda-nto-protocol.c */
extern int mid;
bool qnx_gdbserver_debug = false;
char processResponse = NO_RESPONSE;
uint32_t inferior_pid;

static void
create_ramdisk (void)
{
  int result;

  result =
    system ("devb-ram disk name=ram ram nodinit blk ramdisk=13m,cache=512k") ||
    system ("waitfor /dev/ram0");
  if (result)
    {
      error ("Failed to create a ram disk\n");
    }
  result = system ("mkqnx6fs -q /dev/ram0");
  if (result)
    {
      error ("Failed to create the ram disk file system\n");
    }
  result = system ("mount -t qnx6 /dev/ram0 " RAMDISK_PATH);
  if (result)
    {
      error ("Failed to mount the ram disk\n");
    }
}

static void
ensure_functional_tmpdir (void)
{
  const char *tmpdir;
  char test_path[PATH_MAX];

  tmpdir = getenv ("TMPDIR");
  if (tmpdir != NULL && tmpdir[0] != '\0' && strcmp (tmpdir, RAMDISK_PATH) != 0)
    {
      printf ("warning: TMPDIR is set to %s, skipping using the ram disk\n", tmpdir);
    }
  else
    {
      /* Create ram disk if its path doesn't exist, i.e. if access() returns -1 */
      if (access (RAMDISK_PATH, F_OK))
        {
          create_ramdisk ();
          printf ("Created ram disk mounted at " RAMDISK_PATH "\n");
        }
      else
        {
          printf ("Using ram disk mounted at " RAMDISK_PATH "\n");
        }

      printf ("Setting TMPDIR to " RAMDISK_PATH "\n");
      if (setenv ("TMPDIR", RAMDISK_PATH, 1))
        {
          error ("%s\n", strerror (errno));
        }
      tmpdir = RAMDISK_PATH;
    }

  snprintf (test_path, PATH_MAX, "%s/test_fifo", tmpdir);
  if (mkfifo (test_path, S_IRUSR | S_IWUSR))
    {
      error ("%s\n" \
             "Failed to create a pipe in %s, program cannot continue.\n",
             strerror (errno), tmpdir);
    }
  else
    {
      unlink (test_path);
    }
}

static void
launch_pdebug (char *argv[], int first_arg, int argc, int port)
{
  /* Passed-in args + program name + port + "-f" + NULL */
  int pdebug_argc = argc - first_arg + 4;
  std::string pdebug_exec("pdebug");
  std::string pdebug_port;
  std::string pdebug_arg1("-f");
  char *pdebug_argv[pdebug_argc];
  pid_t pdebug_pid;
  int fd[2];

  pdebug_argv[0] = const_cast<char *>(pdebug_exec.c_str ());
  for (int i = 1; i <= pdebug_argc - 4; i++)
    {
      pdebug_argv[i] = argv[first_arg + i - 1];
    }
  pdebug_port = std::to_string(port);
  pdebug_argv[pdebug_argc - 3] = const_cast<char *>(pdebug_arg1.c_str()); /* -f: prevent from daemonizing */
  pdebug_argv[pdebug_argc - 2] = const_cast<char *>(pdebug_port.c_str());
  pdebug_argv[pdebug_argc - 1] = NULL;

  /* In order to reliably kill the forked pdebug, we first fork
     a watchdog process that reads from a pipe that's never written to.
     The read returns when the other end of the pipe is closed,
     i.e. when the parent (cuda-qnx-gdbserver) exits for any reason.
     We fork pdebug from the watchdog and kill it after the pipe is closed.
     pdebug must be run with -f to prevent daemonization and changing of its PID. */
  pipe (fd);
  switch (fork ())
    {
    case -1:
      error ("Fork failed");
    case 0: /* watchdog process */
      close (fd[1]);
      setpgid (0, 0); /* put child into its own group */

      pdebug_pid = fork ();
      switch (pdebug_pid)
        {
        case -1:
          error ("Fork failed");
        case 0: /* pdebug process */
          close (fd[0]);
          execvp ("pdebug", pdebug_argv);
          error ("Exec failed");
        }

      {
        char buf[1];
        read (fd[0], buf, 1);
      }
      kill (pdebug_pid, SIGTERM);
      exit (EXIT_SUCCESS);
    default: /* gdbserver process */
      sleep (1);
      printf ("cuda-gdbserver started\n");
    }
}

/* Sets global pdebug_sockfd and returns hostfd */
static int
connect_all (uint16_t port, uint16_t pdebug_port)
{
  int server_sockfd, hostfd, res;
  struct sockaddr_in addr, host_addr, pdebug_addr;
  socklen_t client_addr_len;
  struct addrinfo hint, *addrinfo = NULL, *addr_iter;

  pdebug_sockfd = socket (AF_INET, SOCK_STREAM, 0);
  if (pdebug_sockfd < 0)
    {
      error ("Failed to create socket");
    }

  memset (&hint, 0, sizeof (hint));
  hint.ai_family = AF_INET;

  res = getaddrinfo ("localhost", NULL, &hint, &addrinfo);
  if (res != 0)
    {
      error ("Cannot find pdebug host: %s", strerror (errno));
    }

  memset (&pdebug_addr, 0, sizeof (pdebug_addr));
  pdebug_addr.sin_family = AF_INET;
  pdebug_addr.sin_port = htons (pdebug_port);

  addr_iter = addrinfo;

  while (addr_iter) {
    if (addr_iter->ai_family == AF_INET) {
      addr = *(sockaddr_in*)addr_iter->ai_addr;
      memcpy (&pdebug_addr.sin_addr.s_addr, &addr.sin_addr.s_addr, sizeof (addr.sin_addr.s_addr));
      break;
    }
    addr_iter = addr_iter->ai_next;
  }
  freeaddrinfo (addrinfo);

  if (!addr_iter)
    {
      error ("Cannot find pdebug host");
    }

  if (connect (pdebug_sockfd, (struct sockaddr *) &pdebug_addr, sizeof (pdebug_addr)) < 0)
    {
      error ("Cannot connect to pdebug");
    }

  server_sockfd = socket (AF_INET, SOCK_STREAM, 0);
  if (server_sockfd < 0)
    {
      error ("Failed to create socket");
    }
  memset (&addr, 0, sizeof (addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons (port);
  addr.sin_addr.s_addr = INADDR_ANY;
  if (bind (server_sockfd, (struct sockaddr *) &addr, sizeof (addr)) < 0)
    {
      error ("Failed to bind socket");
    }
  listen (server_sockfd, 5);
  client_addr_len = sizeof (host_addr);
  hostfd = accept (server_sockfd, (struct sockaddr*) &host_addr, &client_addr_len);
  if (hostfd < 0)
    {
      error ("Failed to accept incoming connection");
    }

  return hostfd;
}

static void
handle_pdebug_packet (unsigned char *buf, int length, int hostfd)
{
  if (qnx_gdbserver_debug)
    {
      gdbserver_debug_print(PDB, GDB, buf+1);
    }

  if (processResponse != NO_RESPONSE)
    {
      DScomm_t response;

      unpack_pdebug_packet (&response, buf);
      switch (processResponse)
        {
        case DStMsg_load:
          inferior_pid = response.pkt.notify.pid;
          printf ("Inferior pid: %d\n", inferior_pid);
          break;
        default:
          error ("Unhandled response");
          break;
        }
      processResponse = NO_RESPONSE;
    }

  write (hostfd, buf, length);
}

static void
handle_host_packet (unsigned char *buf, int length, int hostfd)
{
  int raw_length;

  /* There can be several packets from the host in the buffer.
     We need to process each of them in turn. */
  do
    {
      raw_length = get_raw_pdebug_packet_size (buf, length);
      if (buf[1] == DStMsg_cuda)
        {
          int null_packet = 0;
          int packet_length;
          DScomm_t cuda_packet;
          char *packet_start;

          if (qnx_gdbserver_debug)
            {
              gdbserver_debug_print(GDB, SRV, buf+1);
            }
          packet_length = unpack_pdebug_packet (&cuda_packet, buf);
          mid = cuda_packet.pkt.hdr.mid; /* Get the latest unused mid */
          /* The following protocol is text-based, we will convert the
             byte-stream into a string. */
          packet_start = (char *) cuda_packet.pkt.cuda.data;
          packet_length -= packet_start - (char *) cuda_packet.buf;
          if (strncmp (packet_start, "qnv.", 4) == 0)
            {
              handle_cuda_packet (packet_start);
              packet_length = pack_cuda_packet (buf, packet_start, 0);
            }
          else if (strncmp (packet_start, "vCUDA", 5) == 0)
            {
              handle_vCuda (packet_start, packet_length, &packet_length);
              packet_length = pack_cuda_packet (buf, packet_start, packet_length);
            }
          else
            {
              if (qnx_gdbserver_debug)
                {
                  char *p = packet_start;
                  for (; isalpha(*p) || *p == ':'; ++p);

                  warning ("Unsupported operation: %.*s", (int)(p - packet_start), packet_start);
                }

              /* Send an empty packet to notify the host this packet isn't
               * supported */
              packet_length = pack_pdebug_packet (buf, NULL, 0);
              null_packet = 1;
            }

          if (qnx_gdbserver_debug && !null_packet)
            {
              gdbserver_debug_print(SRV, GDB, buf+1);
            }
          write (hostfd, buf, packet_length);
        }
      else
        {
          if (qnx_gdbserver_debug)
            {
              gdbserver_debug_print(GDB, PDB, buf+1);
            }

          if (buf[1] == DStMsg_load)
            {
              processResponse = buf[1];
            }

          write (pdebug_sockfd, buf, raw_length);
        }

      buf += raw_length;
      length -= raw_length;
    }
  while (length > 0);
}

/*
 * Normally this will be initialized when constructing the target object.
 * For cuda nto we don't make use of the target. This initializer should
 * mirror the cuda_linux_process_target constructor.
 */
static void
initialize_cuda_remote (void)
{
  /* Check the required CUDA debugger files are present */
  if (cuda_get_debugger_api ())
    {
      printf ("warning: CUDA support disabled. Could not obtain the CUDA debugger API\n");
      return;
    }

  cuda_debugging_enabled = true;
}

static void
show_usage (const char *error, ...)
{
  va_list args;
  FILE *stream = stdout;

  if (error)
    {
      stream = stderr;
      va_start (args, error);
      fprintf (stream, "ERROR: ");
      vfprintf (stream, error, args);
      fprintf (stream, "\n");
      va_end (args);
    }
  fprintf (stream,
          "Usage: cuda-gdbserver PORT [OPTIONS] [--pdebug-args ARGS]\n" \
          "Options:\n" \
          "  -h, --help        - This help\n" \
          "  -d, --debug       - Print debug info\n" \
          "      --pdebug-args - Pass any additional arguments to pdebug\n");
  exit (error ? EXIT_FAILURE : EXIT_SUCCESS);
}

void ATTRIBUTE_NORETURN
captured_main (int argc, char *argv[])
{
  int port;
  int first_pdebug_arg = argc;
  int hostfd;
  struct pollfd fds[POLL_FD_COUNT];

  if (argc < 2)
    {
      show_usage ("missing PORT argument.");
    }

  port = atoi (argv[1]);

  if (port == 0 || port >= 0xffff)
    {
      show_usage ("invalid port %s", argv[1]);
    }

  for (int i = 2; i < argc; i++)
    {
      if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
          show_usage (NULL);
        }
      if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0)
        {
          qnx_gdbserver_debug = true;
          continue;
        }
      if (strcmp(argv[i], "--pdebug-args") == 0)
        {
          first_pdebug_arg = i + 1;
          break;
        }
      show_usage ("invalid option %s", argv[i]);
    }

  /* TODO: find a better way to create pipes on QNX, see bug 1949586 */
  ensure_functional_tmpdir ();

  /* We use the gdb initializers for some of the CUDA sources we share between
   * gdb and gdbserver. See gdb/make-init-c for more info. There is no
   * equivalent concept for gdbserver today. We need to explicitly call the
   * intializers once per execution. */
  static bool cuda_called_initializers = false;
  extern void _initialize_cuda_notification ();
  extern void _initialize_cuda_utils ();
  if (!cuda_called_initializers)
    {
      cuda_called_initializers = true;
      _initialize_cuda_notification ();
      _initialize_cuda_utils ();
    }

  launch_pdebug (argv, first_pdebug_arg, argc, port + 1);

  initialize_cuda_remote ();
  printf ("cuda-gdbserver cuda initialized\n");

  hostfd = connect_all ((uint16_t) port, (uint16_t) (port + 1));

  fds[POLL_FD_PDEBUG].fd = pdebug_sockfd;
  fds[POLL_FD_PDEBUG].events = POLLIN;
  fds[POLL_FD_HOST].fd = hostfd;
  fds[POLL_FD_HOST].events = POLLIN;

  printf ("cuda-gdbserver starting main loop\n");
  for (;;)
    {
      unsigned char buf[MAX_PACKET_SIZE];
      int n;

      n = poll (fds, POLL_FD_COUNT, POLL_TIMEOUT_MS);
      if (n > 0)
        {
          if (fds[POLL_FD_PDEBUG].revents & POLLIN)
            {
              n = read (pdebug_sockfd, buf, MAX_PACKET_SIZE);
              if (n == 0)
                {
                  error ("pdebug exited");
                  break;
                }
              if (n < 0)
                {
                  error ("Failed to read from pdebug socket");
                }

              handle_pdebug_packet (buf, n, hostfd);
            }
          if (fds[POLL_FD_HOST].revents & POLLIN)
            {
              n = read (hostfd, buf, MAX_PACKET_SIZE);
              if (n == 0)
                {
                  error ("Host exited");
                  break;
                }
              if (n < 0)
                {
                  error ("Failed to read from host socket");
                }

              handle_host_packet (buf, n, hostfd);
            }
        }
    }

  throw_quit ("Quit");
}
