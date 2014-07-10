/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2014 NVIDIA Corporation
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

#include "defs.h"
#include "top.h"
#include "command.h"
#include "frame.h"
#include "environ.h"
#include "inferior.h"
#include "gdbcmd.h"

#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-convvars.h"
#include "cuda-packet-manager.h"
#include "libcudbgipc.h"
#include "objfiles.h"
#include "cuda-regmap.h"
#include "cuda-tdep.h"

/*List of set/show cuda commands */
struct cmd_list_element *setcudalist;
struct cmd_list_element *showcudalist;

/*List of set/show debug cuda commands */
struct cmd_list_element *setdebugcudalist;
struct cmd_list_element *showdebugcudalist;

/*
 * cuda prefix
 */
static void
set_cuda (char *arg, int from_tty)
{
  printf_unfiltered (_("\"set cuda\" must be followed by the name of a cuda subcommand.\n"));
  help_list (setcudalist, "set cuda ", -1, gdb_stdout);
}

static void
show_cuda (char *args, int from_tty)
{
  cmd_show_list (showcudalist, from_tty, "");
}

static void
cuda_options_initialize_cuda_prefix (void)
{
  add_prefix_cmd ("cuda", no_class, set_cuda,
                  _("Generic command for setting gdb cuda variables"),
                  &setcudalist, "set cuda ", 0, &setlist);

  add_prefix_cmd ("cuda", no_class, show_cuda,
                  _("Generic command for showing gdb cuda variables"),
                  &showcudalist, "show cuda ", 0, &showlist);
}

/*
 * set debug cuda
 */
static void
set_debug_cuda (char *arg, int from_tty)
{
  printf_unfiltered (_("\"set debug cuda\" must be followed by the name of a debug cuda subcommand.\n"));
  help_list (setdebugcudalist, "set debug cuda ", -1, gdb_stdout);
}

static void
show_debug_cuda (char *args, int from_tty)
{
  cmd_show_list (showdebugcudalist, from_tty, "");
}

static void
cuda_options_initialize_debug_cuda_prefix (void)
{
  add_prefix_cmd ("cuda", no_class, set_debug_cuda,
                  _("Generic command for setting gdb cuda debugging flags"),
                  &setdebugcudalist, "set debug cuda ", 0, &setdebuglist);

  add_prefix_cmd ("cuda", no_class, show_debug_cuda,
                  _("Generic command for showing gdb cuda debugging flags"),
                  &showdebugcudalist, "show debug cuda ", 0, &showdebuglist);
}

/*
 * set debug cuda trace
 */
static char *cuda_debug_trace_string = NULL;
static int cuda_debug_trace_flags  = 0;
static struct {
  const char *name;
  cuda_trace_domain_t domain;
  const char *description;
} cuda_debug_trace_names[] = {
  {"general", CUDA_TRACE_GENERAL, "show/hide general debug trace of the internal CUDA-specific functions"},
  {"event", CUDA_TRACE_EVENT, "show/hide CUDA event trace messages"},
  {"breakpoint", CUDA_TRACE_BREAKPOINT, "show/hide CUDA-specific breakpoint handling trace messages"},
  {"api", CUDA_TRACE_API, "show/hide CUDA Debugger API trace messages"},
  {"textures", CUDA_TRACE_TEXTURES, "show/hide trace of CUDA texture accesses"},
  {"siginfo", CUDA_TRACE_SIGINFO, "When enabled, update $_siginfo if the application is signalled by a CUDA exception"},
  {NULL, 0, NULL},
};

bool
cuda_options_trace_domain_enabled (cuda_trace_domain_t domain)
{
  return (cuda_debug_trace_flags>>domain)&1;
}

static void
cuda_options_set_trace_domain (cuda_trace_domain_t domain, bool enabled)
{
  if (enabled)
    cuda_debug_trace_flags |= 1<<domain;
  else
    cuda_debug_trace_flags &= ~(1<<domain);
}


static void
cuda_show_debug_trace (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  int cnt;

  fprintf_filtered (file, _("CUDA general debug trace:"));
  for (cnt = 0; cuda_debug_trace_names[cnt].name; cnt++)
    fprintf_filtered (file, "%s %s: %s",
            cnt ? "," : "",
            cuda_debug_trace_names[cnt].name,
            cuda_options_trace_domain_enabled (cuda_debug_trace_names[cnt].domain) ? "on" : "off" );
  fprintf_filtered (file, "\n");
}

static void
cuda_set_debug_trace (char *args, int from_tty,
                        struct cmd_list_element *c)
{
  char *flags = NULL;
  char *ptr,*flg;
  int cnt, rc;

  cuda_debug_trace_flags = 0;

  if (cuda_debug_trace_string)
    flags = xstrdup (cuda_debug_trace_string);

  /* Enable all trace logs */
  if (flags && (strcasecmp (flags, "all") == 0 ||
                strcasecmp (flags, "on") == 0 ||
                atoi(flags)>0))
    {
      cuda_debug_trace_flags = -1;
      goto out;
    }

  if (!flags)
     goto out;

  if (strcmp (flags, "0") == 0 ||
      strcasecmp (flags, "none") == 0 ||
      strcasecmp (flags, "off") == 0)
    goto out;

  for (ptr=flags;ptr;)
    {
      flg = strsep(&ptr,",");

      for (cnt = 0; cuda_debug_trace_names[cnt].name; cnt++)
        if (strcasecmp (flg, cuda_debug_trace_names[cnt].name) == 0)
          {
            cuda_options_set_trace_domain (cuda_debug_trace_names[cnt].domain, 1);
            break;
          }

      if (cuda_debug_trace_names[cnt].name == NULL)
        printf_unfiltered (_("Unknown debug trace domain \"%s\"\n"), flg);
    }

out:
  if (flags)
    xfree (flags);

  if (cuda_remote)
    cuda_remote_set_option ();
}

static void
cuda_build_debug_trace_help_message (char *ptr, size_t size)
{
  int rc, cnt;

  rc = snprintf (ptr, size, _("Specifies which trace messages should be printed.\n"));
  ptr += rc; size -= rc;

  rc = snprintf (ptr, size,
    _("Trace event domains are: \"all\",\"none\" or comma separate list of the following:\n"));
  ptr += rc; size -= rc;

  for (cnt=0; cuda_debug_trace_names[cnt].name; cnt++)
    {
        rc = snprintf (ptr, size, " %*s : %s\n",
             20, cuda_debug_trace_names[cnt].name, cuda_debug_trace_names[cnt].description);
        if (rc <= 0) break;
        ptr += rc; size -= rc;
    }
}

static void
cuda_options_initialize_debug_trace (void)
{
  static char dt_help_string[1024];

  cuda_build_debug_trace_help_message ( dt_help_string, sizeof(dt_help_string));

  add_setshow_string_cmd ("trace", class_maintenance, &cuda_debug_trace_string,
                           _("Set debug trace of the internal CUDA-specific functions"),
                           _("Show debug trace of internal CUDA-specific functions."),
                           dt_help_string,
                           cuda_set_debug_trace, cuda_show_debug_trace,
                           &setdebugcudalist, &showdebugcudalist);
}

bool
cuda_options_debug_general ()
{
  return cuda_options_trace_domain_enabled (CUDA_TRACE_GENERAL);
}

/*
 * set debug cuda notifications
 */
static int cuda_debug_notifications;

static void
cuda_show_debug_notifications (struct ui_file *file, int from_tty,
                               struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("CUDA notifications debug trace is %s.\n"), value);
}

static void
cuda_set_debug_notifications (char *args, int from_tty,
                              struct cmd_list_element *c)
{
  if (cuda_remote)
    cuda_remote_set_option ();
}

static void
cuda_options_initialize_debug_notifications ()
{
  cuda_debug_notifications = false;

  add_setshow_boolean_cmd ("notifications", class_maintenance, &cuda_debug_notifications,
                           _("Set debug trace of the CUDA notification functions"),
                           _("Show debug trace of the CUDA notification functions."),
                           _("When non-zero, internal debugging of the CUDA notification functions is enabled."),
                           cuda_set_debug_notifications, cuda_show_debug_notifications,
                           &setdebugcudalist, &showdebugcudalist);
}

bool
cuda_options_debug_notifications ()
{
  return cuda_debug_notifications;
}


/*
 * set debug cuda libcudbg
 */
static int cuda_debug_libcudbg;

static void
cuda_show_debug_libcudbg (struct ui_file *file, int from_tty,
                          struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("CUDA libcudbg debug trace is %s.\n"), value);
}

static void
cuda_set_debug_libcudbg (char *args, int from_tty,
                         struct cmd_list_element *c)
{
  if (cuda_remote)
    cuda_remote_set_option ();
}

static void
cuda_options_initialize_debug_libcudbg ()
{
  cuda_debug_libcudbg = false;

  add_setshow_boolean_cmd ("libcudbg", class_maintenance, &cuda_debug_libcudbg,
                           _("Set debug trace of the CUDA RPC client functions"),
                           _("Show debug trace of the CUDA RPC client functions."),
                           _("When non-zero, internal debugging of the CUDA RPC client functions is enabled."),
                           cuda_set_debug_libcudbg, cuda_show_debug_libcudbg,
                           &setdebugcudalist, &showdebugcudalist);
}

bool
cuda_options_debug_libcudbg ()
{
  return cuda_debug_libcudbg;
}

/*
 * set debug cuda extra convenience variables
 */
static char *cuda_debug_convenience_vars = NULL;

static void
cuda_show_debug_convenience_vars (struct ui_file *file, int from_tty,
                                  struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("The following extra convenience variables are used: %s.\n"),
                    value && *value != 0 ? value : "none");
}

static void
cuda_set_debug_convenience_vars (char *args, int from_tty,
                                 struct cmd_list_element *c)
{
  char *groups = NULL;
  char *ptr,*grp;
  int cnt, rc;

  if (cuda_debug_convenience_vars)
    groups = xstrdup (cuda_debug_convenience_vars);
  /* Enable all variables*/
  if (groups && (strcasecmp(groups,"all")==0 || atoi(groups)>0))
    {
      cuda_enable_convenience_variables_group (NULL, true);
      xfree (groups);
      return;
    }

  cuda_enable_convenience_variables_group (NULL, false);
  if (!groups)
     return;
  if (groups && (strcmp(groups,"0")==0 || strcasecmp(groups,"none")==0))
    {
      xfree (groups);
      return;
    }
  for (ptr=groups;ptr;)
    {
      grp = strsep(&ptr,",");
      rc = cuda_enable_convenience_variables_group (grp, true);
      if (!rc)
        printf_unfiltered (_("Unknown variable group name \"%s\"\n"), grp);
    }
  xfree (groups);
}

static void
cuda_options_initialize_debug_convenience_vars (void)
{

  static char cv_help_string[1024];

  cuda_build_covenience_variables_help_message ( cv_help_string, sizeof(cv_help_string));

  add_setshow_string_cmd ("convenience_vars", class_maintenance, &cuda_debug_convenience_vars,
                            _("Set use of extra convenience variables used for debugging."),
                            _("Show use of extra convenience variables used for debugging."),
                            cv_help_string,
                            cuda_set_debug_convenience_vars, cuda_show_debug_convenience_vars,
                            &setdebugcudalist, &showdebugcudalist);
}

/*
 * set debug cuda strict
 */
static int cuda_debug_strict;

static void
cuda_show_debug_strict (struct ui_file *file, int from_tty,
                        struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("The debugger strict execution mode is %s.\n"), value);
}

static void
cuda_options_initialize_debug_strict ()
{
  cuda_debug_strict = false;

  add_setshow_boolean_cmd ("strict", class_maintenance, &cuda_debug_strict,
                           _("Set debugger execution mode to normal or strict."),
                           _("Show the debugger execution mode."),
                           _("When non-zero, the debugger will produce error messages instead of warnings. For testing purposes only."),
                           NULL, cuda_show_debug_strict,
                           &setdebugcudalist, &showdebugcudalist);
}

bool
cuda_options_debug_strict ()
{
  return cuda_debug_strict;
}

/*
 * set cuda memcheck
 */
int cuda_memcheck_auto;
enum auto_boolean cuda_memcheck;

static void
cuda_show_cuda_memcheck (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("CUDA Memory Checker (cudamemcheck) is %s.\n"), value);
}

static void
cuda_options_initialize_memcheck ()
{
  cuda_memcheck = AUTO_BOOLEAN_AUTO;

  add_setshow_auto_boolean_cmd ("memcheck", class_cuda, &cuda_memcheck,
                                _("Turn on/off CUDA Memory Checker next time the inferior application is run."),
                                _("Show if CUDA Memory Checker is turned on/off."),
                                _("When enabled, CUDA Memory Checker checks for out-of-bounds memory accesses. "
                                  "When auto, CUDA Memory Checker is enabled if CUDA_MEMCHECK environment variable is set."),
                                NULL, cuda_show_cuda_memcheck,
                                &setcudalist, &showcudalist);
}

bool
cuda_options_memcheck (void)
{
  struct gdb_environ *env = current_inferior ()->environment;

  /* Memcheck auto value is determined by the CUDA_MEMCHECK env, which
   * can only be evaluated when the inferior is running. */
  cuda_memcheck_auto = env && !!get_in_environ (env, "CUDA_MEMCHECK");

  return cuda_memcheck == AUTO_BOOLEAN_TRUE ||
        (cuda_memcheck == AUTO_BOOLEAN_AUTO && cuda_memcheck_auto);
}

/*
 * set cuda coalescing
 */
int cuda_coalescing_auto;
enum auto_boolean cuda_coalescing;

static void
cuda_set_coalescing (char *args, int from_tty, struct cmd_list_element *c)
{
  printf_filtered ("Coalescing of the CUDA commands output is %s.\n",
                   cuda_options_coalescing () ? "on" : "off");
}

static void
cuda_show_coalescing (struct ui_file *file, int from_tty,
                      struct cmd_list_element *c, const char *value)
{
  printf_filtered ("Coalescing of the CUDA commands output is %s.\n",
                   cuda_options_coalescing () ? "on" : "off");
}

static void
cuda_options_initialize_coalescing ()
{
  cuda_coalescing_auto = true;
  cuda_coalescing      = AUTO_BOOLEAN_AUTO;

  add_setshow_auto_boolean_cmd ("coalescing", class_cuda, &cuda_coalescing,
                                _("Turn on/off coalescing of the CUDA commands output."),
                                _("Show if coalescing of the CUDA commands output is turned on/off."),
                                _("When enabled, the output of the CUDA commands will be coalesced when possible."),
                                cuda_set_coalescing, cuda_show_coalescing,
                                &setcudalist, &showcudalist);
}

bool
cuda_options_coalescing (void)
{
  return cuda_coalescing == AUTO_BOOLEAN_TRUE ||
         (cuda_coalescing == AUTO_BOOLEAN_AUTO && cuda_coalescing_auto);
}

/*
 * set cuda notify youngest|random
 */
const char  cuda_notify_youngest[]    = "youngest";
const char  cuda_notify_random[]      = "random";

const char *cuda_notify_enums[] = {
  cuda_notify_youngest,
  cuda_notify_random,
  NULL
};

const char *cuda_notify;

static void
cuda_show_notify (struct ui_file *file, int from_tty,
                  struct cmd_list_element *c, const char *value)
{
  printf_filtered ("CUDA notifications will be sent by default to thread: %s.\n", value);
}

static void
cuda_set_notify (char *args, int from_tty, struct cmd_list_element *c)
{
}

static void
cuda_options_initialize_notify (void)
{
  cuda_notify = cuda_notify_youngest;

  add_setshow_enum_cmd ("notify", class_cuda,
                        cuda_notify_enums, &cuda_notify,
                        _("Thread to notify about CUDA events when no other known candidate."),
                        _("Show which thread will be notified when a CUDA event occurs and no other thread is specified."),
                        _("When no thread is specified by CUDA event, the following thread will be notified:\n"
                          "  youngest : the thread with the smallest thread id (default)\n"
                          "  random   : the first valid thread cuda-gdb can find\n"),
                        cuda_set_notify, cuda_show_notify,
                        &setcudalist, &showcudalist);
}

bool
cuda_options_notify_youngest (void)
{
  return cuda_notify == cuda_notify_youngest;
}

bool
cuda_options_notify_random (void)
{
  return cuda_notify == cuda_notify_random;
}

/*
 * set cuda break_on_launch
 */
const char  cuda_break_on_launch_none[]        = "none";
const char  cuda_break_on_launch_application[] = "application";
const char  cuda_break_on_launch_system[]      = "system";
const char  cuda_break_on_launch_all[]         = "all";

const char *cuda_break_on_launch_enums[] = {
  cuda_break_on_launch_none,
  cuda_break_on_launch_application,
  cuda_break_on_launch_system,
  cuda_break_on_launch_all,
  NULL
};

static const char *cuda_break_on_launch;

/* Forward Declearation */
static int cuda_defer_kernel_launch_notifications;

static void
cuda_show_break_on_launch (struct ui_file *file, int from_tty,
                           struct cmd_list_element *c, const char *value)
{
  printf_filtered ("Break on every kernel launch is set to '%s'.\n", value);
}

static void
cuda_set_break_on_launch (char *args, int from_tty, struct cmd_list_element *c)
{
  cuda_auto_breakpoints_update_breakpoints ();
}

static void
cuda_options_initialize_break_on_launch (void)
{
  cuda_break_on_launch = cuda_break_on_launch_none;

  add_setshow_enum_cmd ("break_on_launch", class_cuda,
                        cuda_break_on_launch_enums, &cuda_break_on_launch,
                        _("Automatically set a breakpoint at the entrance of kernels."),
                        _("Show if the debugger stops the application on kernel launches."),
                        _("When enabled, a breakpoint is hit on kernel launches:\n"
                          "  none        : no breakpoint is set (default)\n"
                          "  application : a breakpoint is set at the entrance of all the application kernels\n"
                          "  system      : a breakpoint is set at the entrance of all the system kernels\n"
                          "  all         : a breakpoint is set at the entrance of all kernels"),
                        cuda_set_break_on_launch, cuda_show_break_on_launch,
                        &setcudalist, &showcudalist);
}

void
cuda_options_disable_break_on_launch (void)
{
  cuda_break_on_launch = cuda_break_on_launch_none;

  cuda_set_break_on_launch (NULL, 0, NULL);
}

bool
cuda_options_break_on_launch_system (void)
{
  return (cuda_break_on_launch == cuda_break_on_launch_system ||
          cuda_break_on_launch == cuda_break_on_launch_all);
}

bool
cuda_options_break_on_launch_application (void)
{
  return (cuda_break_on_launch == cuda_break_on_launch_application ||
          cuda_break_on_launch == cuda_break_on_launch_all);
}

/*
 * set cuda disassemble_from
 */
const char  cuda_disassemble_from_device_memory [] = "device_memory";
const char  cuda_disassemble_from_elf_image[]      = "elf_image";

const char *cuda_disassemble_from_enums[] = {
  cuda_disassemble_from_device_memory,
  cuda_disassemble_from_elf_image,
  NULL
};

const char *cuda_disassemble_from;

static void
cuda_show_disassemble_from (struct ui_file *file, int from_tty,
                            struct cmd_list_element *c, const char *value)
{
  printf_filtered ("CUDA code is disassembled from %s.\n", value);
}

static void
cuda_set_disassemble_from (char *args, int from_tty, struct cmd_list_element *c)
{
  cuda_system_flush_disasm_cache ();
}

static void
cuda_options_initialize_disassemble_from (void)
{
  cuda_disassemble_from = cuda_disassemble_from_elf_image;

  add_setshow_enum_cmd ("disassemble_from", class_cuda,
                        cuda_disassemble_from_enums, &cuda_disassemble_from,
                        _("Choose whether to disassemble from the device memory "
                          "(slow) or the ELF image (fast)."),
                        _("Show where the device code is disassembled from."),
                        _("Choose where the device code is disassembled from:\n"
                          "  device_memory : the device code memory (slow)\n"
                          "  elf_image     : the device ELF image on the host (fast)\n"),
                        cuda_set_disassemble_from, cuda_show_disassemble_from,
                        &setcudalist, &showcudalist);
}

bool
cuda_options_disassemble_from_device_memory (void)
{
  return cuda_disassemble_from == cuda_disassemble_from_device_memory;
}

bool
cuda_options_disassemble_from_elf_image (void)
{
  return cuda_disassemble_from == cuda_disassemble_from_elf_image;
}

/*
 * set cuda hide_internal_frames
 */
int cuda_hide_internal_frames;

static void
cuda_set_hide_internal_frames (char *args, int from_tty, struct cmd_list_element *c)
{
  // force rebuilding frame stack to see the change
  reinit_frame_cache ();
}

static void
cuda_show_hide_internal_frames (struct ui_file *file, int from_tty,
                                struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("Hiding of CUDA internal frames is %s.\n"), value);
}

static void
cuda_options_initialize_hide_internal_frames ()
{
  cuda_hide_internal_frames = 1;

  add_setshow_boolean_cmd ("hide_internal_frame", class_cuda, &cuda_hide_internal_frames,
                           _("Set hiding of the internal CUDA frames when printing the call stack"),
                           _("Show hiding of the internal CUDA frames when printing the call stack."),
                           _("When non-zero, internal CUDA frames are omitted when printing the call stack."),
                           cuda_set_hide_internal_frames, cuda_show_hide_internal_frames,
                           &setcudalist, &showcudalist);
}

bool
cuda_options_hide_internal_frames (void)
{
  return cuda_hide_internal_frames;
}

/*
 * set cuda show_kernel_events
 */
const char  cuda_show_kernel_events_none[]        = "none";
const char  cuda_show_kernel_events_application[] = "application";
const char  cuda_show_kernel_events_system[]      = "system";
const char  cuda_show_kernel_events_all[]         = "all";

const char *cuda_show_kernel_events_enums[] = {
  cuda_show_kernel_events_none,
  cuda_show_kernel_events_application,
  cuda_show_kernel_events_system,
  cuda_show_kernel_events_all,
  NULL
};

static const char *cuda_show_kernel_events;
static unsigned int cuda_show_kernel_events_depth;

static void
cuda_show_show_kernel_events (struct ui_file *file, int from_tty,
                              struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("Show CUDA kernel events is set to '%s'.\n"), value);
}

static void
cuda_set_show_kernel_events (char *args, int from_tty, struct cmd_list_element *c)
{
  cuda_auto_breakpoints_update_breakpoints ();
}

static void
cuda_show_show_kernel_events_depth (struct ui_file *file, int from_tty,
                              struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("Show CUDA kernel events depth is set to %s.\n"), value);
}

static void
cuda_options_initialize_show_kernel_events (void)
{
  cuda_show_kernel_events = cuda_show_kernel_events_none;

  add_setshow_enum_cmd ("kernel_events", class_cuda,
                        cuda_show_kernel_events_enums, &cuda_show_kernel_events,
                        _("Turn on/off kernel events (launch/termination) output messages."),
                        _("Show kernel events."),
                        _("When enabled, the kernel launch and termination events are displayed:\n"
                          "  none        : no kernel events are displayed\n"
                          "  application : application kernels events are displayed\n"
                          "  system      : system kernel events are displayed\n"
                          "  all         : all kernel events are displayed"),
                        cuda_set_show_kernel_events, cuda_show_show_kernel_events,
                        &setcudalist, &showcudalist);

  cuda_show_kernel_events_depth = 1; /* Only host kernel events by default */

  add_setshow_uinteger_cmd ("kernel_events_depth", class_cuda,
                            &cuda_show_kernel_events_depth,
                            _("Set the maximum depth of nested kernels event notifications."),
                            _("Show the maximum depth of nested kernels event notifications."),
                            _("Controls the maximum depth of the kernels after which no kernel event notifications will be displayed.\n"
                              "A value of zero means that there is no maximum and that all the kernel notifications are displayed.\n"
                              "A value of one means that the debugger will display kernel event notifications only for kernels launched from the CPU (default)."),
                            NULL, cuda_show_show_kernel_events_depth,
                            &setcudalist, &showcudalist);
}

unsigned int
cuda_options_show_kernel_events_depth (void)
{
  return cuda_show_kernel_events_depth;
}

bool
cuda_options_show_kernel_events_system (void)
{
  return (cuda_show_kernel_events == cuda_show_kernel_events_system ||
          cuda_show_kernel_events == cuda_show_kernel_events_all);
}

bool
cuda_options_show_kernel_events_application (void)
{
  return (cuda_show_kernel_events == cuda_show_kernel_events_application ||
          cuda_show_kernel_events == cuda_show_kernel_events_all);
}

/* set cuda defer_kernel_launch_notifications */
static int cuda_defer_kernel_launch_notifications = 0;

static void
cuda_show_defer_kernel_launch_notifications (struct ui_file *file, int from_tty,
                                            struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("CUDA defer kernel launch notifications is %s.\n"), value);
}

static void
cuda_set_defer_kernel_launch_notifications (char *args, int from_tty, struct cmd_list_element *c)
{
  printf_filtered ("Warning: CUDA defer kernel launch notifications has been deprecated and has no effect.\n");
}

static void
cuda_options_initialize_defer_kernel_launch_notifications (void)
{
  cuda_defer_kernel_launch_notifications = 0;
  add_setshow_boolean_cmd ("defer_kernel_launch_notifications", class_cuda,
                            &cuda_defer_kernel_launch_notifications,
                            _("Turn on/off deferral of kernel launch messages (deprecated)."),
                            _("Show whether kernel launch notifications are deferred."),
                            _("When turned on, the kernel launch and termination events are deferred until the debugger hits a breakpoint or an exception.\n"
                              "This option has been deprecated and has no effect any more."),
                            cuda_set_defer_kernel_launch_notifications,
                            cuda_show_defer_kernel_launch_notifications,
                            &setcudalist, &showcudalist);
}

/*
 * set cuda show_context_events
 */
int cuda_show_context_events;

static void
cuda_show_show_context_events (struct ui_file *file, int from_tty,
                               struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("Show CUDA context events is %s.\n"), value);
}

static void
cuda_options_initialize_show_context_events (void)
{
  cuda_show_context_events = 0;

  add_setshow_boolean_cmd ("context_events", class_cuda, &cuda_show_context_events,
                           _("Turn on/off context events (push/pop/create/destroy) output messages."),
                           _("Show context events."),
                           _("When turned on, push/pop/create/destroy context events are displayed."),
                           NULL,
                           cuda_show_show_context_events,
                           &setcudalist, &showcudalist);
}

bool
cuda_options_show_context_events (void)
{
  return cuda_show_context_events;
}

/*
 * set cuda launch_blocking
 */
int cuda_launch_blocking;

static void
cuda_set_launch_blocking (char *args, int from_tty, struct cmd_list_element *c)
{
  if (cuda_launch_blocking)
      printf_filtered ("On the next run, the CUDA kernel launches will be blocking.\n");
  else
      printf_filtered ("On the next run, the CUDA kernel launches will be non-blocking.\n");
}

static void
cuda_show_launch_blocking (struct ui_file *file, int from_tty,
                           struct cmd_list_element *c, const char *value)
{
  if (cuda_launch_blocking)
    fprintf_filtered (file, _("On the next run, the CUDA kernel launches will be blocking.\n"));
  else
    fprintf_filtered (file, _("On the next run, the CUDA kernel launches will be non-blocking.\n"));
}

static void
cuda_options_initialize_launch_blocking (void)
{
  cuda_launch_blocking = 0;

  add_setshow_boolean_cmd ("launch_blocking", class_cuda, &cuda_launch_blocking,
                           _("Turn on/off CUDA kernel launch blocking (effective starting from the next run)"),
                           _("Show whether CUDA kernel launches are blocking."),
                           _("When turned on, CUDA kernel launches are blocking (effective starting from the next run."),
                           cuda_set_launch_blocking,
                           cuda_show_launch_blocking,
                           &setcudalist, &showcudalist);
}

bool
cuda_options_launch_blocking (void)
{
  return cuda_launch_blocking;
}

/*
 * set cuda thread_selection
 */
const char  cuda_thread_selection_policy_logical[]  = "logical";
const char  cuda_thread_selection_policy_physical[] = "physical";
const char *cuda_thread_selection_policy_enums[]    = {
  cuda_thread_selection_policy_logical,
  cuda_thread_selection_policy_physical,
  NULL
};
const char *cuda_thread_selection_policy;

static void
show_cuda_thread_selection_policy (struct ui_file *file, int from_tty,
                                   struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("CUDA thread selection policy is %s.\n"), value);
}

static void
cuda_options_initialize_thread_selection (void)
{
  cuda_thread_selection_policy = cuda_thread_selection_policy_logical;

  add_setshow_enum_cmd ("thread_selection", class_cuda,
                        cuda_thread_selection_policy_enums, &cuda_thread_selection_policy,
                        _("Set the automatic thread selection policy to use when the current thread cannot be selected.\n"),
                        _("Show the automatic thread selection policy to use when the current thread cannot be selected.\n"),
                        _("logical  == the thread with the lowest logical coordinates (blockIdx/threadIdx) is selected\n"
                          "physical == the thread with the lowest physical coordinates (dev/sm/wp/ln) is selected."),
                        NULL, show_cuda_thread_selection_policy,
                        &setcudalist, &showcudalist);
}

bool
cuda_options_thread_selection_logical (void)
{
  return cuda_thread_selection_policy == cuda_thread_selection_policy_logical;
}

bool
cuda_options_thread_selection_physical (void)
{
  return cuda_thread_selection_policy == cuda_thread_selection_policy_physical;
}

static void
show_cuda_copyright_command (char *ignore, int from_tty)
{
  print_gdb_version (gdb_stdout);
  printf_filtered ("\n");
}

static void
cuda_options_initialize_copyright (void)
{
  add_cmd ("copyright", class_cuda, show_cuda_copyright_command,
           _("Copyright for GDB with CUDA support."),
           &showcudalist);
}

/*
 * set cuda api_failures
 */
const char  cuda_api_failures_option_ignore[]       = "ignore";
const char  cuda_api_failures_option_stop[]         = "stop";
const char  cuda_api_failures_option_hide[]         = "hide";
const char  cuda_api_failures_option_stop_all[]     = "stop_all";
const char  cuda_api_failures_option_ignore_all[]   = "ignore_all";

const char *cuda_api_failures_options_enums[] = {
    cuda_api_failures_option_ignore,
    cuda_api_failures_option_stop,
    cuda_api_failures_option_hide,
    cuda_api_failures_option_stop_all,
    cuda_api_failures_option_ignore_all,
    NULL
};

const char *cuda_api_failures_option;

static void
cuda_show_api_failures (struct ui_file *file, int from_tty,
                          struct cmd_list_element *c, const char *value)
{
  printf_filtered ("api_failures is set to '%s'.\n", value);
}

static void
cuda_set_api_failures  (char *args, int from_tty, struct cmd_list_element *c)
{
  cuda_update_report_driver_api_error_flags ();
}

static void
cuda_options_initialize_api_failures (void)
{
  cuda_api_failures_option = cuda_api_failures_option_ignore;

  add_setshow_enum_cmd ("api_failures", class_cuda,
                        cuda_api_failures_options_enums, &cuda_api_failures_option,
                        _("Set the api_failures to ignore/stop/hide on CUDA driver API call errors."),
                        _("Show if cuda-gdb ignores/stops/hides on CUDA driver API call errors."),
                        _("  ignore     : Warning message is printed for every fatal CUDA API call failure (default)\n"
                          "  stop       : The application is stopped when a CUDA API call returns a fatal error\n"
                          "  ignore_all : Warning message is printed for every CUDA API call failure\n"
                          "  stop_all   : The application is stopped when a CUDA API call returns any error\n"
                          "  hide       : CUDA API call failures are not reported."),
                        cuda_set_api_failures, cuda_show_api_failures,
                        &setcudalist, &showcudalist);
}

bool
cuda_options_api_failures_ignore(void)
{
  return (cuda_api_failures_option == cuda_api_failures_option_ignore) ||
         (cuda_api_failures_option == cuda_api_failures_option_ignore_all);
}

bool
cuda_options_api_failures_stop(void)
{
  return (cuda_api_failures_option == cuda_api_failures_option_stop) ||
         (cuda_api_failures_option == cuda_api_failures_option_stop_all);
}

bool
cuda_options_api_failures_hide(void)
{
  return (cuda_api_failures_option == cuda_api_failures_option_hide);
}

bool
cuda_options_api_failures_break_on_nonfatal(void)
{
  return (cuda_api_failures_option == cuda_api_failures_option_ignore_all) ||
         (cuda_api_failures_option == cuda_api_failures_option_stop_all);
}

/*
 * set cuda software_preemption
 */
int cuda_software_preemption_auto;
enum auto_boolean cuda_software_preemption;

static void
cuda_show_cuda_software_preemption (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("Software preemption debugging is %s.\n"), value);
}

static void
cuda_options_initialize_software_preemption ()
{
  cuda_software_preemption = AUTO_BOOLEAN_AUTO;

  add_setshow_auto_boolean_cmd ("software_preemption", class_cuda, &cuda_software_preemption,
                                _("Turn on/off CUDA software preemption debugging the next time the inferior application is run."),
                                _("Show if CUDA software preemption debugging is turned on/off."),
                                _("When enabled, upon suspending the inferior application, the debugger frees the GPU for use by other applications.  This option is currently limited to devices with compute capability sm_35."),
                                NULL, cuda_show_cuda_software_preemption,
                                &setcudalist, &showcudalist);
}

bool
cuda_options_software_preemption (void)
{
  struct gdb_environ *env = current_inferior ()->environment;
  char *cuda_dsp = env ? get_in_environ (env, "CUDA_DEBUGGER_SOFTWARE_PREEMPTION") : NULL;

  /* Software preemption auto value is determined by the
     CUDA_DEBUGGER_SOFTWARE_PREEMPTION env var */
  cuda_software_preemption_auto = cuda_dsp && strcmp(cuda_dsp,"1")==0;

  return cuda_software_preemption == AUTO_BOOLEAN_TRUE ||
        (cuda_software_preemption == AUTO_BOOLEAN_AUTO && cuda_software_preemption_auto);

  return cuda_software_preemption == AUTO_BOOLEAN_TRUE;
}

/*
 * set cuda gpu_busy_check
 */
enum auto_boolean cuda_gpu_busy_check;

static void
cuda_show_cuda_gpu_busy_check (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("GPU busy check is %s.\n"), value);
}

static void
cuda_options_initialize_gpu_busy_check ()
{
  cuda_gpu_busy_check = AUTO_BOOLEAN_TRUE;

  add_setshow_auto_boolean_cmd ("gpu_busy_check", class_cuda, &cuda_gpu_busy_check,
                                _("Turn on/off GPU busy check the next time the inferior application is run. (Mac only)"),
                                _("Show if GPU busy check on Darwin is turned on/off."),
                                _("When enabled, cuda-gdb will attempt to detect if any GPU to be used is already used for graphics."),
                                NULL, cuda_show_cuda_gpu_busy_check,
                                &setcudalist, &showcudalist);
}

bool
cuda_options_gpu_busy_check (void)
{
  return cuda_gpu_busy_check == AUTO_BOOLEAN_TRUE ||
         cuda_gpu_busy_check == AUTO_BOOLEAN_AUTO;
}


/*
 * set cuda variable_value_cache
 */
enum auto_boolean cuda_variable_value_cache_enabled;

static void
cuda_show_cuda_variable_value_cache_enabled (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("Variable value cache enabled is %s.\n"), value);
}

static void
cuda_options_initialize_variable_value_cache_enabled ()
{
  cuda_variable_value_cache_enabled = AUTO_BOOLEAN_TRUE;

  add_setshow_auto_boolean_cmd ("ptx_cache", class_cuda, &cuda_variable_value_cache_enabled,
                                _("Turn on/off GPU variable value cache"),
                                _("Show if GPU variable value cache is is turned on/off."),
                                _("When enabled, cuda-gdb will cache the last known values of PTX registers mapped to local variables for a current lane."),
                                NULL, cuda_show_cuda_variable_value_cache_enabled,
                                &setcudalist, &showcudalist);
}

bool
cuda_options_variable_value_cache_enabled (void)
{
  return cuda_variable_value_cache_enabled == AUTO_BOOLEAN_TRUE ||
         cuda_variable_value_cache_enabled == AUTO_BOOLEAN_AUTO;
}

static void
cuda_print_statistics (char *args, int from_tty)
{
  uint32_t cnt;
  const CUDBGIPCStat_t *st;
  struct cleanup *table_cleanup, *row_cleanup;
  struct ui_out *uiout = current_uiout;

  /* column headers */
  const char *header_name = "API name";
  const char *header_calls = "Number of calls";
  const char *header_avg = "Average call time(usec)";
  const char *header_min = "Min call time(usec)";
  const char *header_max = "Max call time(usec)";

  int name_width = strlen(header_name);
  int row_no = 0;
  double total = 0;

  for (cnt = 0; cnt < CUDBGIPC_API_STAT_MAX; cnt++)
    {
      st = cudbgipcGetProfileStat (cnt);
      if (!st) continue;

      total += st->total_time;
      name_width = max(name_width,strlen(st->name));
      row_no++;
    }

  table_cleanup = make_cleanup_ui_out_table_begin_end (uiout, 5, row_no, "CUDBGAPIStatTable");
  ui_out_table_header (uiout, name_width,           ui_left,   "name",  header_name);
  ui_out_table_header (uiout, strlen(header_calls), ui_center, "calls", header_calls);
  ui_out_table_header (uiout, strlen(header_avg),   ui_center, "avg",   header_avg);
  ui_out_table_header (uiout, strlen(header_min),   ui_center, "min",   header_min);
  ui_out_table_header (uiout, strlen(header_max),   ui_center, "max",   header_max);
  ui_out_table_body (uiout);

  for (cnt=0; cnt < CUDBGIPC_API_STAT_MAX; cnt++)
    {
      st = cudbgipcGetProfileStat (cnt);
      if (!st) continue;

      row_cleanup = make_cleanup_ui_out_tuple_begin_end (uiout, "CUDBGAPIStatRow");
      ui_out_field_string (uiout, "name", st->name);
      ui_out_field_int (uiout, "calls", st->times_called);
      ui_out_field_int (uiout, "avg", (int)(st->total_time/st->times_called));
      ui_out_field_int (uiout, "min", (int)st->min_time);
      ui_out_field_int (uiout, "max", (int)st->max_time);
      ui_out_text (uiout, "\n");
      do_cleanups (row_cleanup);
    }

  do_cleanups (table_cleanup);

  printf_unfiltered ("Total time spend in CUDBG API is %f sec\n", total*1e-6);
}


static int cuda_gpu_collect_stats = 0;

static void
cuda_show_cuda_gpu_collect_stats (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("CUDA debugger API statistics collection is %s.\n"), value);
}


bool
cuda_options_statistics_collection_enabled (void)
{
  return cuda_gpu_collect_stats != 0;
}

static void
cuda_options_initialize_stats (void)
{
  add_cmd ("cuda_stats", class_maintenance, cuda_print_statistics,
           _("Print statistics about CUDA Debugger API."),
           &maintenanceprintlist);

  add_setshow_boolean_cmd ("collect_stats", class_cuda, &cuda_gpu_collect_stats,
                           _("Turn on/off CUDA Debugger API statistics collection"),
                           _("Show if CUDA Debugger API statistics collection is enabled."),
                           _("When enabled, cuda-gdb will collect debugger API call statistics."),
                           NULL, cuda_show_cuda_gpu_collect_stats,
                           &setcudalist, &showcudalist);
}

/*
 * maintenance print cuda_regmap
 */
int cuda_value_extrapolation = 0;

static void
cuda_show_cuda_value_extrapolation (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("CUDA value extrapolation is %s.\n"), value);
}

bool
cuda_options_value_extrapolation_enabled (void)
{
  return cuda_value_extrapolation != 0;
}

static void
cuda_print_regmap (char *args, int from_tty)
{

  struct objfile *objfile;

  ALL_OBJFILES (objfile)
    {
      if (!objfile->cuda_objfile)
        continue;
      printf( "Objfile %s \n", objfile->name);
      regmap_table_print (objfile);
    }
}

static void
cuda_options_initialize_value_extrapolation (void)
{
  add_cmd ("cuda_regmap", class_maintenance, cuda_print_regmap,
           _("Print GPUs register map table"),
           &maintenanceprintlist);

  add_setshow_boolean_cmd ("value_extrapolation", class_cuda, &cuda_value_extrapolation,
                           _("Turn on/off CUDA register value extrapolation"),
                           _("Show if CUDA register value extrapolation is enabled."),
                           _("When enabled, cuda-gdb will attempt to extrapolate the value"
                             " of variables that would otherwise marked as 'optimized out'."
                             " The extrapolation is based on the last location where the"
                             " variable was known to be stored (register or memory location)."
                             " The extrapolated value is NOT guaranteed to be correct and should be read with extra care."),
                           NULL, cuda_show_cuda_value_extrapolation,
                           &setcudalist, &showcudalist);
}

/*
 * set cuda single_stepping_optimizations
 */
int cuda_gpu_single_stepping_optimizations = 1;

static void
cuda_show_single_stepping_optimizations (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _("CUDA single stepping optimizations are %s.\n"), value);
}

bool
cuda_options_single_stepping_optimizations_enabled (void)
{
  return cuda_gpu_single_stepping_optimizations != 0;
}

static void
cuda_options_initialize_single_stepping_optimization (void)
{
#ifdef __ANDROID__
  cuda_gpu_single_stepping_optimizations = 0;
#endif
  add_setshow_boolean_cmd ("single_stepping_optimizations", class_cuda, &cuda_gpu_single_stepping_optimizations,
                           _("Turn on/off CUDA single-stepping optimizations"),
                           _("Show if CUDA single-stepping optimizations are enabled."),
                           _("When enabled, cuda-gdb will use optimized methods to"
                             " single-step the program, when verifiably correct."
                             " Those optimizations accelerate single-stepping most of the time."),
                           NULL, cuda_show_single_stepping_optimizations,
                           &setcudalist, &showcudalist);

}

/*Initialization */
void
cuda_options_initialize ()
{
  cuda_options_initialize_cuda_prefix ();
  cuda_options_initialize_debug_cuda_prefix ();
  cuda_options_initialize_debug_trace ();
  cuda_options_initialize_debug_notifications ();
  cuda_options_initialize_debug_libcudbg ();
  cuda_options_initialize_debug_convenience_vars ();
  cuda_options_initialize_debug_strict ();
  cuda_options_initialize_memcheck ();
  cuda_options_initialize_coalescing ();
  cuda_options_initialize_break_on_launch ();
  cuda_options_initialize_api_failures ();
  cuda_options_initialize_disassemble_from ();
  cuda_options_initialize_hide_internal_frames ();
  cuda_options_initialize_show_kernel_events ();
  cuda_options_initialize_defer_kernel_launch_notifications ();
  cuda_options_initialize_show_context_events ();
  cuda_options_initialize_launch_blocking ();
  cuda_options_initialize_thread_selection ();
  cuda_options_initialize_copyright ();
  cuda_options_initialize_notify ();
  cuda_options_initialize_software_preemption ();
  cuda_options_initialize_gpu_busy_check ();
  cuda_options_initialize_variable_value_cache_enabled ();
  cuda_options_initialize_stats ();
  cuda_options_initialize_value_extrapolation ();
  cuda_options_initialize_single_stepping_optimization();
}
