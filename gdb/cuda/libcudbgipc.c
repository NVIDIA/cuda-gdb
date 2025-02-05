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

#ifdef GDBSERVER
#include "server.h"
#include "cuda-tdep-server.h"
#else
#include "defs.h"
#include "cuda/cuda-options.h"
#include "cuda/cuda-tdep.h"
#include "gdbsupport/ptid.h"
#include "inferior.h"
#endif

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include "gdbsupport/rsp-low.h"

#include "cuda/cuda-utils.h"
#include "cuda/libcudbg.h"
#include "cuda/libcudbgipc.h"
#include <cudadebugger.h>

#include <sstream>

/*Forward declarations */
static ATTRIBUTE_PRINTF(1, 2) void cudbgipc_trace(const char *fmt, ...);

/*Globals */
CUDBGNotifyNewEventCallback cudbgDebugClientCallback = NULL;
CUDBGIPC_t commOut;
CUDBGIPC_t commIn;
CUDBGIPC_t commCB;
static bool cudbgPreInitComplete = false;
pthread_t callbackEventThreadHandle;
pthread_t cudagdbMainThreadHandle;

static void *
cudbgCallbackHandler(void *arg)
{
    CUDBGCBMSG_t data;
    CUDBGResult res;
    CUDBGEventCallbackData cbData;
    sigset_t sigset;

    /* SIGCHLD signals must be caught by the main thread */
    sigemptyset (&sigset);
    sigaddset (&sigset, SIGCHLD);
    sigprocmask (SIG_BLOCK, &sigset, NULL);

    for (;;) {
        res = cudbgipcCBWaitForData(&data, sizeof data);
        if (res != CUDBG_SUCCESS) {
            cudbgipc_trace ("failure while waiting for callback data! (res = %d)", res);
            break;
        }
        if (data.terminate) {
            cudbgipc_trace ("Callback handler thread received termination data.\n");
            break;
        }
        if (cudbgDebugClientCallback)
          {
            cbData.tid = data.tid;
            cbData.timeout = data.timeout;
            cudbgDebugClientCallback(&cbData);
          }
    }

    return NULL;
}

static CUDBGResult
cudbgipcCreate(CUDBGIPC_t *ipc, int from, int to, int flags)
{
    const char *env = getenv("CUDA_GDB_IPC_OPEN_NONBLOCKING");
    int timeout_in_seconds = env ? atoi(env) : 0;

    snprintf(ipc->name, sizeof (ipc->name), "%s/pipe.%d.%d",
             cuda_gdb_session_get_dir (), from, to);

    /* If the inferior hasn't been properly set up for cuda
       debugging yet, the fifo should not exist (it is stale).
       Unlink it, and carry on. */
    if (access(ipc->name, F_OK) == 0) {
        if (!cuda_inferior_in_debug_mode()) {
            cudbgipc_trace("Found stale fifo (%s), unlinking...\n", ipc->name);
            if (unlink(ipc->name) && errno != ENOENT)
                return CUDBG_ERROR_COMMUNICATION_FAILURE;
        }
    }

    if ((flags & O_WRONLY) == O_WRONLY) {
        if (access(ipc->name, F_OK) == -1)
            return CUDBG_ERROR_UNINITIALIZED;
    }
    else if (mkfifo(ipc->name, S_IRGRP | S_IWGRP | S_IRUSR | S_IWUSR) && errno != EEXIST) {
        cudbgipc_trace("Failed to create fifo (from=%u, to=%u, file=%s, errno=%d)",
                       from, to, ipc->name, errno);
        return CUDBG_ERROR_COMMUNICATION_FAILURE;
    }

    /* If cuda-gdb is launched as root, make pipes are writeable by UID of debugged process */
#ifndef GDBSERVER
    if (getuid() == 0) {
      int pid = (int) inferior_ptid.pid ();
      if (pid > 0 && !cuda_gdb_chown_to_pid_uid (pid, ipc->name)) {
	cudbgipc_trace("Changing ownership to pid %d uid failed for %s, errno=%d",
                       pid, ipc->name, errno);
	return CUDBG_ERROR_COMMUNICATION_FAILURE;
      }
    }
#endif

    if (timeout_in_seconds > 0)
        flags |= O_NONBLOCK;

    do {
        if ((ipc->fd = open(ipc->name, flags)) >= 0)
            break;

        if (errno == ENXIO) {
            sleep(1);
        }
        else {
            cudbgipc_trace("Pipe opening failure (from=%u, to=%u, flags=%x, file=%s, errno=%d)",
                           ipc->from, ipc->to, flags, ipc->name, errno);
            return CUDBG_ERROR_COMMUNICATION_FAILURE;
        }
	--timeout_in_seconds;
    } while (timeout_in_seconds >= 0);

    if (timeout_in_seconds < 0) {
        cudbgipc_trace("Pipe opening timeout (from=%u, to=%u, flags=%x, file=%s, errno=%d)",
                       ipc->from, ipc->to, flags, ipc->name, errno);
        return CUDBG_ERROR_COMMUNICATION_FAILURE;
    }

    if ((flags & O_WRONLY) == O_WRONLY) {
        /* If opening for write, unlink it instantly */
        if (unlink(ipc->name) && errno != ENOENT) {
            cudbgipc_trace("Cannot unlink fifo (from=%u, to=%u, file=%s, errno=%d)",
                           ipc->from, ipc->to, ipc->name, errno);
            return CUDBG_ERROR_COMMUNICATION_FAILURE;
        }
    }

    /* Initialize message */
    ipc->dataSize = sizeof(ipc->dataSize);
    ipc->data     = (char *) malloc(sizeof(ipc->dataSize));
    if (!ipc->data)
        return CUDBG_ERROR_OS_RESOURCES;
    memset(ipc->data, 0, ipc->dataSize);

    /* Indicate successful initialization */
    ipc->from        = from;
    ipc->to          = to;
    ipc->initialized = true;

    return CUDBG_SUCCESS;
}

static CUDBGResult
cudbgipcDestroy(CUDBGIPC_t *ipc)
{
    gdb_assert (ipc->name);

    if (close(ipc->fd) == -1) {
        cudbgipc_trace("Failed to close ipc (from=%u, to=%u, errno=%u)",
                       ipc->from, ipc->to, errno);
        return CUDBG_ERROR_COMMUNICATION_FAILURE;
    }

    /* not an error if file does not exist */
    if (unlink(ipc->name) && errno != ENOENT) {
        cudbgipc_trace("Cannot unlink fifo (from=%u, to=%u, file=%s, errno=%d)",
                       ipc->from, ipc->to, ipc->name, errno);
        return CUDBG_ERROR_COMMUNICATION_FAILURE;
    }

    memset(ipc->name, 0, sizeof (ipc->name));
    free(ipc->data);
    ipc->from = 0;
    ipc->to = 0;
    ipc->initialized = false;

    return CUDBG_SUCCESS;
}

static CUDBGResult
cudbgipcInitializeCommIn(void)
{
    CUDBGResult res;

    res = cudbgipcCreate(&commIn,
                         LIBCUDBG_PIPE_ENDPOINT_RPCD,
                         LIBCUDBG_PIPE_ENDPOINT_DEBUG_CLIENT,
#if __QNXHOST__
                         /* On QNX readonly pipes always return EOF when there is no data
                            so create a fake writer too */
                         O_RDWR | O_NONBLOCK
#else
                         O_RDONLY | O_NONBLOCK
#endif
                         );

    if (res != CUDBG_SUCCESS)
        return res;

    cudbgipc_trace("initialized commIn (from = %d, to = %d)", commIn.from, commIn.to);
    return CUDBG_SUCCESS;
}

static CUDBGResult
cudbgipcInitializeCommOut(void)
{
CUDBGResult res;

    res = cudbgipcCreate(&commOut,
                         LIBCUDBG_PIPE_ENDPOINT_DEBUG_CLIENT,
                         LIBCUDBG_PIPE_ENDPOINT_RPCD,
                         O_WRONLY);
    if (res != CUDBG_SUCCESS)
        return res;

    cudbgipc_trace("initialized commOut (from = %d, to = %d)", commOut.from, commOut.to);
    return CUDBG_SUCCESS;
}

static CUDBGResult
cudbgipcInitializeCommCB(void)
{
    CUDBGResult res;

    res = cudbgipcCreate(&commCB,
                         LIBCUDBG_PIPE_ENDPOINT_RPCD_CB,
                         LIBCUDBG_PIPE_ENDPOINT_DEBUG_CLIENT_CB,
#if __QNXHOST__
                         /* On QNX readonly pipes always return EOF when there is no data
                            so create a fake writer too */
                         O_RDWR | O_NONBLOCK
#else
                         O_RDONLY | O_NONBLOCK
#endif
                         );

    if (res != CUDBG_SUCCESS)
        return res;

    cudbgipc_trace("initialized commCB (from = %d, to = %d)", commCB.from, commCB.to);

    return CUDBG_SUCCESS;
}

#ifndef GDBSERVER
/*
 * CUDADBG API call RSP wrapper protocol
 * Following RSP commands are used to proxy CUDADBG API from cuda-gdb to cuda-gdbserver:
 *  * - vCuda;w;sz;$(req) -> w;sz;$(reply)
 */
#include "cuda-packet-manager.h"
#include "remote.h"
#include <array>

/* Set of callbacks used for send_remote_packet */
struct cudbgipc_remote_callbacks : public send_remote_packet_callbacks
{
public:
  /* Initialize the output buffer with the start of cuda packet string */
  cudbgipc_remote_callbacks () 
  : m_res (CUDBG_SUCCESS), m_processing_multi_packet (false)
  { reset_out_buffer (); }

  ~cudbgipc_remote_callbacks () = default;

  /* Append the string to the output buffer */
  void
  append_request (gdb::char_vector &buf)
  {
    gdb_assert (has_space (buf.size ()));
    m_out_buffer.insert (m_out_buffer.end (), buf.begin (), buf.end ());
  }

  CUDBGResult
  send_request (void)
  {
    /* Null terminate outBuffer */
    m_out_buffer.push_back ('\0');
    gdb::array_view<const char> view (m_out_buffer.data (), m_out_buffer.size ());
    send_remote_packet (view, this);
    return m_res;
  }

  void
  sending (gdb::array_view<const char> &buf) override
  {
    cudbgipc_trace ("Sending remote packet request: %s", buf.data ());
  }

  void
  received (gdb::array_view<const char> &buf) override
  {
    cudbgipc_trace ("Received response: %s", buf.data ());

    /* Reset the buffer if we are not processing multi-packet repsonse */
    if (!m_processing_multi_packet)
      m_in_buffer.clear ();

    /* Check for the multi packet request */
    m_processing_multi_packet = (memcmp (buf.data (), "MP;", strlen ("MP;")) == 0);

    /* Check for integer literal */
    if (buf[0] == 'E')
      {
	cudbgipc_trace ("Received error response!");
	m_res = (CUDBGResult)atoi (buf.data () + 1);
      }
    else if (m_processing_multi_packet || memcmp (buf.data (), "OK;", strlen ("OK;")) == 0)
      {
	/* Resize the in buffer */
	auto in_buffer_size = m_in_buffer.size ();
	m_in_buffer.resize (in_buffer_size + buf.size ());
	/* Unescape the recv buffer into in buffer */
	int recv_bytes = remote_unescape_input (
	    (const gdb_byte *)buf.data () + 3, buf.size () - 3,
	    (gdb_byte *)m_in_buffer.data () + in_buffer_size,
	    m_in_buffer.size () - in_buffer_size);
	/* Resize the in buffer downwards */
	m_in_buffer.resize (in_buffer_size + recv_bytes);
	/* If a multi-packet reply received, send a request for more data */
	if (m_processing_multi_packet)
	  {
	    /* Build the request string */
	    std::ostringstream more_data_stream;
	    more_data_stream << "vCUDARetr;" << m_in_buffer.size ();
	    std::string more_data = more_data_stream.str ();
	    /* We don't want to copy the null terminator here */
	    m_out_buffer.clear ();
	    m_out_buffer.insert (m_out_buffer.begin (), more_data.begin (), more_data.end ());
	    /* Send it */
	    send_request ();
	  }
      }
    else
      {
	/* Unknown packet! */
	cudbgipc_trace ("Received invalid response!");
      }
    /* Reset the out buffer */
    reset_out_buffer ();
  }

  /* TODO: Rethink this getter */
  void *
  get_result_buffer (void)
  {
    return (void *)m_in_buffer.data ();
  }

  size_t
  get_result_size (void)
  {
    return m_in_buffer.size ();
  }

private:
  void
  reset_out_buffer (void)
  {
    m_out_buffer.clear ();
    std::string req = "vCUDA;";
    m_out_buffer.insert (m_out_buffer.begin (), req.begin (), req.end ());
  }

  /* Check if there is size remaining for the request */
  /* Guard against integer overflow */
  /* FIXME: gdbserver should handle sizes larger than INT_MAX bytes */
  bool
  has_space (size_t size)
  {
    return m_out_buffer.size () + size <= INT_MAX;
  }

  gdb::char_vector m_out_buffer;
  gdb::char_vector m_in_buffer;
  CUDBGResult m_res;
  bool m_processing_multi_packet = false;
};
static cudbgipc_remote_callbacks remote_callbacks;

static CUDBGResult
cudbgipcAppendRemote (const void *d, size_t size)
{
  int out_len, bytes_processed;

  /* Escape the input data */
  gdb::char_vector escaped_data (size * 2 + 1);
  out_len = remote_escape_output ((const gdb_byte *)d, size, 1,
				  (gdb_byte *)escaped_data.data (),
				  &bytes_processed, escaped_data.size ());
  gdb_assert (bytes_processed == size);
  escaped_data.resize (out_len);

  /* Append the request */
  remote_callbacks.append_request (escaped_data);

  return CUDBG_SUCCESS;
}

static CUDBGResult
cudbgipcRequestRemote (void **d, size_t *size)
{
  auto res = remote_callbacks.send_request ();
  if (res == CUDBG_SUCCESS)
    {
      *d = remote_callbacks.get_result_buffer ();
      if (size)
	*size = remote_callbacks.get_result_size ();
    }

  return res;
}
#endif

static CUDBGResult
cudbgipcPush(CUDBGIPC_t *out)
{
    ssize_t writeCount = 0;
    size_t offset = 0;
    char *buf;

    gdb_assert (out);

    /* Push out the header (size) */
    memcpy(out->data, (char*)&out->dataSize, sizeof(out->dataSize));
    buf = out->data;

    /* Push out the data */
    for (offset = 0, writeCount = 0; offset < out->dataSize; offset += writeCount) {
        writeCount = write(out->fd, buf + offset, out->dataSize - offset);
        if (writeCount < 0) {
            /* Forward SIGINT received during syscall to main thread signal handler */
            if (errno == EINTR && pthread_self() != cudagdbMainThreadHandle)
              pthread_kill (cudagdbMainThreadHandle, SIGINT);

            if (errno != EAGAIN && errno != EINTR) {
                cudbgipc_trace("Fifo write error (from=%u, to=%u, out->dataSize=%zu, offset=%zu, errno=%d)",
                               out->from, out->to, out->dataSize, offset, errno);
                return CUDBG_ERROR_COMMUNICATION_FAILURE;
            }
            writeCount = 0;
        }
    }

    memset(out->data, 0, sizeof(out->dataSize));
    out->dataSize = sizeof(out->dataSize);
    return CUDBG_SUCCESS;
}

static CUDBGResult
cudbgipcRead(CUDBGIPC_t *in, void *buf, size_t size)
{
    ssize_t readCount = 0;
    size_t offset = 0;

    gdb_assert (in);

    for (offset = 0; offset < size; offset += readCount) {
        readCount = read(in->fd, (char*)buf + offset, size - offset);
        if (readCount == 0) {
            cudbgipc_trace("EOF reached");
            return CUDBG_ERROR_COMMUNICATION_FAILURE;
        }
        if (readCount < 0) {
            /* Forward SIGINT received during syscall to main thread signal handler */
            if (errno == EINTR && pthread_self() != cudagdbMainThreadHandle)
              pthread_kill (cudagdbMainThreadHandle, SIGINT);

            if (errno != EAGAIN && errno != EINTR) {
                cudbgipc_trace("Fifo read error (from=%u, to=%u, size=%zu, offset=%zu, errno=%d)",
                               in->from, in->to, size, offset, errno);
                return CUDBG_ERROR_COMMUNICATION_FAILURE;
            }
            readCount = 0;
        }
    }

    return CUDBG_SUCCESS;
}

static CUDBGResult
cudbgipcPull(CUDBGIPC_t *in)
{
    CUDBGResult res;

    /* Obtain the size */
    res = cudbgipcRead(in, &in->dataSize, sizeof in->dataSize);
    if (res != CUDBG_SUCCESS) {
        cudbgipc_trace("failed to read size (res=%d)", res);
        return res;
    }

    /* Allocate memory given the size */
    if ((in->data = (char *) realloc(in->data, in->dataSize)) == 0) {
        cudbgipc_trace("Memory reallocation failed (res=%d)", res);
        return CUDBG_ERROR_COMMUNICATION_FAILURE;
    }
    /* Guard against incorrect preamble read */
    if (in->dataSize == 0)
      {
	cudbgipc_trace("Read zero sized preamble");
	return CUDBG_ERROR_COMMUNICATION_FAILURE;
      }
    memset(in->data, 0, in->dataSize);

    /* Obtain the data */
    res = cudbgipcRead(in, in->data, in->dataSize - sizeof in->dataSize);
    if (res != CUDBG_SUCCESS) {
        cudbgipc_trace("failed to read data (res=%d)", res);
        return res;
    }

    return CUDBG_SUCCESS;
}

static CUDBGResult
cudbgipcWait(CUDBGIPC_t *in)
{
   fd_set readFDS;
   fd_set errFDS;
   int ret;

   if (!in->initialized)
       return CUDBG_ERROR_COMMUNICATION_FAILURE;

   /* wait for data to be available for reading */
   FD_ZERO(&readFDS);
   FD_ZERO(&errFDS);
   FD_SET(in->fd, &readFDS);
   FD_SET(in->fd, &errFDS);
   do {
       ret = select(in->fd + 1, &readFDS, NULL, &errFDS, NULL);

       /* Forward SIGINT received during syscall to main thread signal handler */
       if (ret < 0 && errno == EINTR && pthread_self() != cudagdbMainThreadHandle)
         pthread_kill (cudagdbMainThreadHandle, SIGINT);
   } while (ret == -1 && errno == EINTR);

   if (ret == -1) {
       cudbgipc_trace("Select error (from=%u, to=%u, errno=%u)", in->from, in->to, errno);
       return CUDBG_ERROR_COMMUNICATION_FAILURE;
   }

   if (FD_ISSET(in->fd, &errFDS)) {
       cudbgipc_trace("Select error on in->fd (from=%u, to=%u, errno=%u)", in->from, in->to, errno);
       return CUDBG_ERROR_COMMUNICATION_FAILURE;
   }

   return CUDBG_SUCCESS;
}

static CUDBGResult
cudbgipcAppendLocal(const void *d, size_t size)
{
    CUDBGResult res;
    size_t dataSize = 0;
    void *data = NULL;

    if (!commOut.initialized) {
        res = cudbgipcInitializeCommOut();
        if (res != CUDBG_SUCCESS)
            return res;
    }

    dataSize = commOut.dataSize + size;
    /* Guard against integer overflow */
    if (commOut.dataSize > dataSize)
      return CUDBG_ERROR_COMMUNICATION_FAILURE;
    if ((data = realloc(commOut.data, dataSize)) == NULL)
        return CUDBG_ERROR_COMMUNICATION_FAILURE;

    memcpy(((char *)data) + commOut.dataSize, d, size);

    commOut.data = (char *)data;
    commOut.dataSize = dataSize;

    return CUDBG_SUCCESS;
}

CUDBGResult
cudbgipcAppend(const void *d, size_t size)
{
#ifndef GDBSERVER
    if (is_remote_target (current_inferior ()->process_target ()))
        return cudbgipcAppendRemote (d, size);
#endif
    return cudbgipcAppendLocal (d, size);
}

CUDBGResult
cudbgipcRequest(void **d, size_t *size)
{
    CUDBGResult res;

#ifndef GDBSERVER
    if (is_remote_target (current_inferior ()->process_target ()))
        return cudbgipcRequestRemote (d, size);
#endif
    res = cudbgipcPush(&commOut);
    if (res != CUDBG_SUCCESS) {
        cudbgipc_trace("cudbgipcRequest push failed (res=%d)", res);
        return res;
    }

    res = cudbgipcWait(&commIn);
    if (res != CUDBG_SUCCESS) {
        cudbgipc_trace("cudbgipcRequest wait failed (res=%d)", res);
        return res;
    }

    res = cudbgipcPull(&commIn);
    if (res != CUDBG_SUCCESS) {
        cudbgipc_trace("cudbgipcRequest pull failed (res=%d)", res);
        return res;
    }

    *(uintptr_t **)d = (uintptr_t *)commIn.data;
    if (size) *size = commIn.dataSize;

    return CUDBG_SUCCESS;
}

CUDBGResult
cudbgipcCBWaitForData(void *d, size_t size)
{
    CUDBGResult res;

    if (!commCB.initialized) {
        res = cudbgipcInitializeCommCB();
        if (res != CUDBG_SUCCESS) {
            cudbgipc_trace("failed to initialize cb fifo (res=%d)", res);
            return res;
        }
    }

    res = cudbgipcWait(&commCB);
    if (res != CUDBG_SUCCESS) {
        cudbgipc_trace("CB wait for data failed (res=%d)", res);
        return res;
    }

    res = cudbgipcRead(&commCB, d, size);
    if (res != CUDBG_SUCCESS) {
        cudbgipc_trace("CB read data failed (res=%d)", res);
        return res;
    }

    return CUDBG_SUCCESS;
}

CUDBGResult
cudbgipcInitialize(void)
{
    CUDBGResult res;
    int ret;

    if (cudbgPreInitComplete)
        return CUDBG_SUCCESS;

    ret = cuda_gdb_session_create ();
    if (ret)
        return CUDBG_ERROR_COMMUNICATION_FAILURE;

    res = cudbgipcInitializeCommIn();
    if (res != CUDBG_SUCCESS)
        return CUDBG_ERROR_COMMUNICATION_FAILURE;

    cudagdbMainThreadHandle = pthread_self();
    if (pthread_create(&callbackEventThreadHandle, NULL, cudbgCallbackHandler, NULL))
        return CUDBG_ERROR_COMMUNICATION_FAILURE;
    cudbgPreInitComplete = true;

    return CUDBG_SUCCESS;
}

CUDBGResult
cudbgipcFinalize(void)
{
    CUDBGResult res;

#ifndef GDBSERVER
    if (is_remote_target (current_inferior ()->process_target ()))
        return CUDBG_SUCCESS;
#endif

    if (pthread_join(callbackEventThreadHandle, NULL)) {
        cudbgipc_trace ("post finalize error joining with callback thread\n");
        return CUDBG_ERROR_INTERNAL;
    }

    res = cudbgipcDestroy(&commOut);
    if (res != CUDBG_SUCCESS) {
        cudbgipc_trace ("post finalize error finalizing ipc (res = %d)\n", res);
        return res;
    }

    res = cudbgipcDestroy(&commIn);
    if (res != CUDBG_SUCCESS) {
        cudbgipc_trace ("post finalize error finalizing ipc (res = %d)\n", res);
        return res;
    }

    res = cudbgipcDestroy(&commCB);
    if (res != CUDBG_SUCCESS) {
        cudbgipc_trace ("post finalize error finalizing ipc (res = %d)\n", res);
        return res;
    }

    cuda_gdb_session_destroy ();

    cudbgPreInitComplete = false;
    cudbgDebugClientCallback = NULL;

    return CUDBG_SUCCESS;
}

ATTRIBUTE_PRINTF(1, 2) void cudbgipc_trace(const char *fmt, ...)
{
#ifdef GDBSERVER
  struct cuda_trace_msg *msg;
#endif
  va_list ap;

  if (!cuda_options_debug_libcudbg())
    return;

  va_start (ap, fmt);
#ifdef GDBSERVER
  msg = (struct cuda_trace_msg *) xmalloc (sizeof (*msg));
  if (!cuda_first_trace_msg)
    cuda_first_trace_msg = msg;
  else
    cuda_last_trace_msg->next = msg;
  sprintf (msg->buf, "[CUDAGDB] libcudbg ipc ");
  vsnprintf (msg->buf + strlen (msg->buf), sizeof (msg->buf), fmt, ap);
  msg->next = NULL;
  cuda_last_trace_msg = msg;
#else
  fprintf (stderr, "[CUDAGDB] libcudbg ipc ");
  vfprintf (stderr, fmt, ap);
  fprintf (stderr, "\n");
  fflush (stderr);
#endif
}

