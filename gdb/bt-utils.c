/* Copyright (C) 2021-2024 Free Software Foundation, Inc.

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

#include "bt-utils.h"
#include "command.h"
#include "cli/cli-cmds.h"
#include "ui.h"
#include "cli/cli-decode.h"
#ifdef NVIDIA_CUDA_GDB
#include "demangle.h"
#include <string.h>
#include "ui-style.h"
#endif

/* See bt-utils.h.  */

void
gdb_internal_backtrace_set_cmd (const char *args, int from_tty,
				cmd_list_element *c)
{
  gdb_assert (c->type == set_cmd);
  gdb_assert (c->var.has_value ());
  gdb_assert (c->var->type () == var_boolean);

#ifndef GDB_PRINT_INTERNAL_BACKTRACE
  if (c->var->get<bool> ())
    {
      c->var->set<bool> (false);
      error (_("support for this feature is not compiled into GDB"));
    }
#endif
}

#ifdef GDB_PRINT_INTERNAL_BACKTRACE
#ifdef GDB_PRINT_INTERNAL_BACKTRACE_USING_LIBBACKTRACE

/* Callback used by libbacktrace if it encounters an error.  */

static void
libbacktrace_error (void *data, const char *errmsg, int errnum)
{
  /* A negative errnum indicates no debug info was available, just
     skip printing a backtrace in this case.  */
  if (errnum < 0)
    return;

  const auto sig_write = [] (const char *msg) -> void
  {
    gdb_stderr->write_async_safe (msg, strlen (msg));
  };

  sig_write ("error creating backtrace: ");
  sig_write (errmsg);
  if (errnum > 0)
    {
      char buf[20];
      snprintf (buf, sizeof (buf), ": %d", errnum);
      buf[sizeof (buf) - 1] = '\0';

      sig_write (buf);
    }
  sig_write ("\n");
}

#ifdef NVIDIA_CUDA_GDB
/* Async-safe demangling support for libbacktrace.  */

/* Static buffer for demangled names. This is used to provide async-safe
   demangling in signal handlers. The buffer is large enough for most
   demangled C++ names. */
static char demangle_buffer[2048];
static size_t demangle_buffer_used = 0;

/* Callback for cplus_demangle_v3_callback that writes to our static buffer
   in an async-safe manner. */
static void
demangle_callback (const char *s, size_t len, void *opaque)
{
  size_t *buffer_offset = (size_t *) opaque;
  size_t available = sizeof (demangle_buffer) - *buffer_offset - 1;

  if (len > available)
    len = available;

  if (len > 0)
    {
      memcpy (demangle_buffer + *buffer_offset, s, len);
      *buffer_offset += len;
    }
}

/* Attempt to demangle a symbol name in an async-safe manner.
   Returns the demangled name or the original name if demangling fails. */
static const char *
async_safe_demangle (const char *mangled)
{
  /* Only attempt to demangle C++ names */
  if (mangled == nullptr || mangled[0] != '_' || mangled[1] != 'Z')
    return mangled;

  /* Reset the buffer */
  demangle_buffer_used = 0;

  /* Try to demangle using the callback interface which is async-safe */
  int result = cplus_demangle_v3_callback (mangled, 
                                          DMGL_PARAMS | DMGL_ANSI,
                                          demangle_callback,
                                          &demangle_buffer_used);

  if (result && demangle_buffer_used > 0 
      && demangle_buffer_used < sizeof (demangle_buffer))
    {
      demangle_buffer[demangle_buffer_used] = '\0';
      return demangle_buffer;
    }

  /* Demangling failed, return the original name */
  return mangled;
}

static void
write_default (const char *text)
{
  gdb_stderr->write_async_safe (text, strlen (text));
}

static void
write_style (const char *text, const ui_file_style &style)
{
  gdb_stderr->emit_style_escape (style);
  write_default (text);
  gdb_stderr->reset_style ();
}

static int
libbacktrace_print (void *data, uintptr_t pc, const char *filename,
			   int lineno, const char *function)
{
  static ui_file_style address_style (ui_file_style::BLUE, ui_file_style::NONE);
  static ui_file_style function_style (ui_file_style::YELLOW, ui_file_style::NONE);
  static ui_file_style filename_style (ui_file_style::GREEN, ui_file_style::NONE);

  char buf[19];

  snprintf (buf, sizeof (buf), "0x%016" PRIxPTR, pc);
  buf[sizeof (buf) - 1] = '\0';

  write_style (buf, address_style);

  write_default ("| ");

  if (function == nullptr)
    write_style ("???", function_style);
  else
    {
      const char *display_name = async_safe_demangle (function);
      write_style (display_name, function_style);
    }

  if (filename != nullptr)
    {
      /* Remove leading "../" from filename */
      const char *display_filename = filename;
      while (strncmp (display_filename, "../", 3) == 0)
        display_filename += 3;

      write_default (" at ");
      write_style (display_filename, filename_style);
      if (lineno > 0)
        {
          write_default (":");
          snprintf (buf, sizeof (buf), "%d", lineno);
          buf[sizeof (buf) - 1] = '\0';
          write_default (buf);
        }
    }
  write_default ("\n");

  return function != nullptr && strcmp (function, "main") == 0;
}
#else

/* Callback used by libbacktrace to print a single stack frame.  */

static int
libbacktrace_print (void *data, uintptr_t pc, const char *filename,
		    int lineno, const char *function)
{
  const auto sig_write = [] (const char *msg) -> void
  {
    gdb_stderr->write_async_safe (msg, strlen (msg));
  };

  /* Buffer to print addresses and line numbers into.  An 8-byte address
     with '0x' prefix and a null terminator requires 20 characters.  This
     also feels like it should be enough to represent line numbers in most
     files.  We are also careful to ensure we don't overflow this buffer.  */
  char buf[20];

  snprintf (buf, sizeof (buf), "0x%" PRIxPTR " ", pc);
  buf[sizeof (buf) - 1] = '\0';
  sig_write (buf);
  sig_write (function == nullptr ? "???" : function);
  if (filename != nullptr)
    {
      sig_write ("\n\t");
      sig_write (filename);
      sig_write (":");
      snprintf (buf, sizeof (buf), "%d", lineno);
      buf[sizeof (buf) - 1] = '\0';
      sig_write (buf);
    }
  sig_write ("\n");

  return function != nullptr && strcmp (function, "main") == 0;
}
#endif /* NVIDIA_CUDA_GDB */

/* Write a backtrace to GDB's stderr in an async safe manner.  This is a
   backtrace of GDB, not any running inferior, and is to be used when GDB
   crashes or hits some other error condition.  */

static void
gdb_internal_backtrace_1 ()
{
  static struct backtrace_state *state = nullptr;

  if (state == nullptr)
    state = backtrace_create_state (nullptr, 0, libbacktrace_error, nullptr);

  backtrace_full (state, 0, libbacktrace_print, libbacktrace_error, nullptr);
}

#elif defined GDB_PRINT_INTERNAL_BACKTRACE_USING_EXECINFO

/* See the comment on previous version of this function.  */

static void
gdb_internal_backtrace_1 ()
{
  const auto sig_write = [] (const char *msg) -> void
  {
    gdb_stderr->write_async_safe (msg, strlen (msg));
  };

  /* Allow up to 25 frames of backtrace.  */
  void *buffer[25];
  int frames = backtrace (buffer, ARRAY_SIZE (buffer));

  backtrace_symbols_fd (buffer, frames, gdb_stderr->fd ());
  if (frames == ARRAY_SIZE (buffer))
    sig_write (_("Backtrace might be incomplete.\n"));
}

#else
#error "unexpected internal backtrace policy"
#endif

static const char *str_backtrace = "----- Backtrace -----\n";
static const char *str_backtrace_unavailable = "Backtrace unavailable\n";

#endif /* GDB_PRINT_INTERNAL_BACKTRACE */

/* See bt-utils.h.  */

void
gdb_internal_backtrace_init_str ()
{
#ifdef GDB_PRINT_INTERNAL_BACKTRACE
  str_backtrace = _("----- Backtrace -----\n");
  str_backtrace_unavailable = _("Backtrace unavailable\n");
#endif
}

/* See bt-utils.h.  */

void
gdb_internal_backtrace ()
{
  if (current_ui == nullptr)
    return;

#ifdef GDB_PRINT_INTERNAL_BACKTRACE
  const auto sig_write = [] (const char *msg) -> void
  {
    gdb_stderr->write_async_safe (msg, strlen (msg));
  };

  sig_write (str_backtrace);

  if (gdb_stderr->fd () > -1)
    gdb_internal_backtrace_1 ();
  else
    sig_write (str_backtrace_unavailable);

  sig_write ("---------------------\n");
#endif
}
