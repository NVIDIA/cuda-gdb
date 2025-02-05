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

#include "defs.h"
#include "top.h"
#include "command.h"
#include "frame.h"
#include "environ.h"
#include "inferior.h"
#include "gdbcmd.h"
#include "remote.h"

#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-convvars.h"
#include "cuda-packet-manager.h"
#include "libcudbgipc.h"
#include "objfiles.h"
#include "cuda-regmap.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"

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
set_cuda (const char *arg, int from_tty)
{
  printf_unfiltered (_("\"set cuda\" must be followed by the name of a cuda subcommand.\n"));
  help_list (setcudalist, "set cuda ", (enum command_class) -1, gdb_stdout);
}

static void
show_cuda (const char *args, int from_tty)
{
  cmd_show_list (showcudalist, from_tty);
}

static void
cuda_options_initialize_cuda_prefix (void)
{
  add_prefix_cmd ("cuda", no_class, set_cuda,
                  _("Generic command for setting gdb cuda variables"),
                  &setcudalist, 0, &setlist);

  add_prefix_cmd ("cuda", no_class, show_cuda,
                  _("Generic command for showing gdb cuda variables"),
                  &showcudalist, 0, &showlist);
}

/*
 * set debug cuda
 */
static void
set_debug_cuda (const char *arg, int from_tty)
{
  printf_unfiltered (_("\"set debug cuda\" must be followed by the name of a debug cuda subcommand.\n"));
  help_list (setdebugcudalist, "set debug cuda ", (enum command_class) -1, gdb_stdout);
}

static void
show_debug_cuda (const char *args, int from_tty)
{
  cmd_show_list (showdebugcudalist, from_tty);
}

static void
cuda_options_initialize_debug_cuda_prefix (void)
{
  add_prefix_cmd ("cuda", no_class, set_debug_cuda,
                  _("Generic command for setting gdb cuda debugging flags"),
                  &setdebugcudalist, 0, &setdebuglist);

  add_prefix_cmd ("cuda", no_class, show_debug_cuda,
                  _("Generic command for showing gdb cuda debugging flags"),
                  &showdebugcudalist, 0, &showdebuglist);
}

/*
 * set debug cuda trace
 */
static std::string cuda_debug_trace_string;
static uint32_t cuda_debug_trace_flags = 0;
static struct {
  const char *name;
  cuda_trace_domain_t domain;
  const char *description;
} cuda_debug_trace_names[] = {
  {"general", CUDA_TRACE_GENERAL, "show/hide general debug trace of the internal CUDA-specific functions"},
  {"disasm", CUDA_TRACE_DISASSEMBLER, "When enabled trace GPU disassembly operations"},
  {"elf", CUDA_TRACE_ELF, "show/hide CUDA ELF file processing trace messages"},
  {"event", CUDA_TRACE_EVENT, "show/hide CUDA event trace messages"},
  {"breakpoint", CUDA_TRACE_BREAKPOINT, "show/hide CUDA-specific breakpoint handling trace messages"},
  {"api", CUDA_TRACE_API, "show/hide CUDA Debugger API trace messages"},
  {"siginfo", CUDA_TRACE_SIGINFO, "When enabled, update $_siginfo if the application is signalled by a CUDA exception"},
  {"state", CUDA_TRACE_STATE, "show/hide CUDA state trace messages"},
  {"decode", CUDA_TRACE_STATE_DECODE, "show/hide CUDA state device update decoding messages"},
  {NULL, (cuda_trace_domain_t) 0, NULL},
};

bool
cuda_options_trace_domain_enabled (cuda_trace_domain_t domain)
{
  return (cuda_debug_trace_flags>>domain)&1;
}

uint32_t
cuda_options_trace_domain_enabled_flags ()
{
  return cuda_debug_trace_flags;
}

void
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

  gdb_printf (file, _("CUDA general debug trace:"));
  for (cnt = 0; cuda_debug_trace_names[cnt].name; cnt++)
    gdb_printf (file, "%s %s: %s",
            cnt ? "," : "",
            cuda_debug_trace_names[cnt].name,
            cuda_options_trace_domain_enabled (cuda_debug_trace_names[cnt].domain) ? "on" : "off" );
  gdb_printf (file, "\n");
}

static void
cuda_set_debug_trace (const char *args, int from_tty,
		      struct cmd_list_element *c)
{
  char *flags = NULL;
  char *ptr,*flg;
  int cnt;

  cuda_debug_trace_flags = 0;

  if (cuda_debug_trace_string.length ())
    flags = xstrdup (cuda_debug_trace_string.c_str());

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

  if (is_remote_target (current_inferior ()->process_target ()))
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
cuda_options_debug_general (void)
{
  return cuda_options_trace_domain_enabled (CUDA_TRACE_GENERAL);
}

/*
 * set debug cuda notifications
 */
static bool cuda_debug_notifications = false;

static void
cuda_show_debug_notifications (struct ui_file *file, int from_tty,
                               struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("CUDA notifications debug trace is %s.\n"), value);
}

static void
cuda_set_debug_notifications (const char *args, int from_tty,
                              struct cmd_list_element *c)
{
  if (is_remote_target (current_inferior ()->process_target ()))
    cuda_remote_set_option ();
}

static void
cuda_options_initialize_debug_notifications (void)
{
  add_setshow_boolean_cmd ("notifications", class_maintenance, &cuda_debug_notifications,
                           _("Set debug trace of the CUDA notification functions"),
                           _("Show debug trace of the CUDA notification functions."),
                           _("When non-zero, internal debugging of the CUDA notification functions is enabled."),
                           cuda_set_debug_notifications, cuda_show_debug_notifications,
                           &setdebugcudalist, &showdebugcudalist);
}

bool
cuda_options_debug_notifications (void)
{
  return cuda_debug_notifications;
}


/*
 * set debug cuda libcudbg
 */
static bool cuda_debug_libcudbg = false;

static void
cuda_show_debug_libcudbg (struct ui_file *file, int from_tty,
                          struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("CUDA libcudbg debug trace is %s.\n"), value);
}

static void
cuda_set_debug_libcudbg (const char *args, int from_tty,
                         struct cmd_list_element *c)
{
  if (is_remote_target (current_inferior ()->process_target ()))
    cuda_remote_set_option ();
}

static void
cuda_options_initialize_debug_libcudbg (void)
{
  add_setshow_boolean_cmd ("libcudbg", class_maintenance, &cuda_debug_libcudbg,
                           _("Set debug trace of the CUDA RPC client functions"),
                           _("Show debug trace of the CUDA RPC client functions."),
                           _("When non-zero, internal debugging of the CUDA RPC client functions is enabled."),
                           cuda_set_debug_libcudbg, cuda_show_debug_libcudbg,
                           &setdebugcudalist, &showdebugcudalist);
}

bool
cuda_options_debug_libcudbg (void)
{
  return cuda_debug_libcudbg;
}

/*
 * set debug cuda extra convenience variables
 */
static std::string cuda_debug_convenience_vars;

static void
cuda_show_debug_convenience_vars (struct ui_file *file, int from_tty,
                                  struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("The following extra convenience variables are used: %s.\n"),
                    value && *value != 0 ? value : "none");
}

static void
cuda_set_debug_convenience_vars (const char *args, int from_tty,
                                 struct cmd_list_element *c)
{
  char *groups = NULL;
  char *ptr,*grp;
  int rc;

  if (cuda_debug_convenience_vars.length ())
    groups = xstrdup (cuda_debug_convenience_vars.c_str());
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
static bool cuda_debug_strict = false;

static void
cuda_show_debug_strict (struct ui_file *file, int from_tty,
                        struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("The debugger strict execution mode is %s.\n"), value);
}

static void
cuda_options_initialize_debug_strict (void)
{
  add_setshow_boolean_cmd ("strict", class_maintenance, &cuda_debug_strict,
                           _("Set debugger execution mode to normal or strict."),
                           _("Show the debugger execution mode."),
                           _("When non-zero, the debugger will produce error messages instead of warnings. For testing purposes only."),
                           NULL, cuda_show_debug_strict,
                           &setdebugcudalist, &showdebugcudalist);
}

bool
cuda_options_debug_strict (void)
{
  return cuda_debug_strict;
}

/*
 * set cuda memcheck
 * REMOVED: Support has been removed for this feature.
 * We still want to print a message to the user informing of this.
 */
static void
cuda_set_cuda_memcheck (auto_boolean enabled)
{
  error (_("Support for CUDA Memory Checker has been removed. Use the standalone Compute Sanitizer instead."));
}

static auto_boolean
cuda_get_cuda_memcheck (void)
{
  return AUTO_BOOLEAN_FALSE;
}

static void
cuda_show_cuda_memcheck (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("Support for CUDA Memory Checker has been removed. Use the standalone Compute Sanitizer instead.\n"));
}

static void
cuda_options_initialize_memcheck (void)
{
  add_setshow_auto_boolean_cmd ("memcheck", class_cuda,
				_("Turn on/off CUDA Memory Checker next time the inferior application is run. (removed)"),
				_("Show if CUDA Memory Checker is turned on/off. (removed)"),
				_("Support for CUDA Memory Checker has been removed. Use the standalone Compute Sanitizer instead."),
				cuda_set_cuda_memcheck,
				cuda_get_cuda_memcheck,
				cuda_show_cuda_memcheck,
				&setcudalist, &showcudalist);
}

/*
 * set cuda coalescing
 */
static bool cuda_coalescing_auto = true;
static enum auto_boolean cuda_coalescing = AUTO_BOOLEAN_AUTO;

static void
cuda_set_coalescing (const char *args, int from_tty, struct cmd_list_element *c)
{
  gdb_printf ("Coalescing of the CUDA commands output is %s.\n",
                   cuda_options_coalescing () ? "on" : "off");
}

static void
cuda_show_coalescing (struct ui_file *file, int from_tty,
                      struct cmd_list_element *c, const char *value)
{
  gdb_printf ("Coalescing of the CUDA commands output is %s.\n",
                   cuda_options_coalescing () ? "on" : "off");
}

static void
cuda_options_initialize_coalescing (void)
{
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
static const char  cuda_notify_youngest[]    = "youngest";
static const char  cuda_notify_random[]      = "random";

static const char *cuda_notify_enums[] = {
  cuda_notify_youngest,
  cuda_notify_random,
  NULL
};

static const char *cuda_notify = cuda_notify_youngest;

static void
cuda_show_notify (struct ui_file *file, int from_tty,
                  struct cmd_list_element *c, const char *value)
{
  gdb_printf ("CUDA notifications will be sent by default to thread: %s.\n", value);
}

static void
cuda_set_notify (const char *args, int from_tty, struct cmd_list_element *c)
{
  if (is_remote_target (current_inferior ()->process_target ()))
    cuda_remote_set_option ();
}

static void
cuda_options_initialize_notify (void)
{
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
static const char  cuda_break_on_launch_none[]        = "none";
static const char  cuda_break_on_launch_application[] = "application";
static const char  cuda_break_on_launch_system[]      = "system";
static const char  cuda_break_on_launch_all[]         = "all";

static const char *cuda_break_on_launch_enums[] = {
  cuda_break_on_launch_none,
  cuda_break_on_launch_application,
  cuda_break_on_launch_system,
  cuda_break_on_launch_all,
  NULL
};

static const char *cuda_break_on_launch = cuda_break_on_launch_none;

static void
cuda_show_break_on_launch (struct ui_file *file, int from_tty,
                           struct cmd_list_element *c, const char *value)
{
  gdb_printf ("Break on every kernel launch is set to '%s'.\n", value);
}

static void
cuda_set_break_on_launch (const char *args, int from_tty, struct cmd_list_element *c)
{
  /* Update to receive KERNEL_READY events. */
  cuda_options_force_set_launch_notification_update ();

  /* Update kernel entry bpts if needed. */
  cuda_module::auto_breakpoints_update_locations ();
}

static void
cuda_options_initialize_break_on_launch (void)
{
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
static const char cuda_disassemble_from_device_memory [] = "device_memory";
static const char cuda_disassemble_from_elf_image[]      = "elf_image";

static const char *cuda_disassemble_from_enums[] = {
  cuda_disassemble_from_device_memory,
  cuda_disassemble_from_elf_image,
  NULL
};

static const char *cuda_disassemble_from = cuda_disassemble_from_elf_image;

static void
cuda_show_disassemble_from (struct ui_file *file, int from_tty,
                            struct cmd_list_element *c, const char *value)
{
  gdb_printf ("CUDA code is disassembled from %s.\n", value);
}

static void
cuda_set_disassemble_from (const char *args, int from_tty, struct cmd_list_element *c)
{
  cuda_state::flush_disasm_caches ();
}

static void
cuda_options_initialize_disassemble_from (void)
{
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
 * set cuda disassemble_per
 */
static const char  cuda_disassemble_per_function[] = "function";
static const char  cuda_disassemble_per_file [] = "file";

static const char *cuda_disassemble_per_enums[] = {
  cuda_disassemble_per_function,
  cuda_disassemble_per_file,
  NULL
};

static const char *cuda_disassemble_per = cuda_disassemble_per_function;

static void
cuda_show_disassemble_per (struct ui_file *file, int from_tty,
                          struct cmd_list_element *c, const char *value)
{
  gdb_printf ("CUDA code is disassembled per %s.\n", value);
}

static void
cuda_set_disassemble_per (const char *args, int from_tty, struct cmd_list_element *c)
{
}

static void
cuda_options_initialize_disassemble_per (void)
{
  add_setshow_enum_cmd ("disassemble_per", class_cuda,
                        cuda_disassemble_per_enums, &cuda_disassemble_per,
                        _("Choose whether to disassemble a function at a time (fast) "
                          "or file at a time (slow)."),
                        _("Show if disassembly is per-function or per-file."),
                        _("Choose how disassembly is performed:\n"
                          "  function : done on a per-function basis (fast))\n"
                          "  file     : done on a per-file basis (slower at first, then faster)\n"),
                        cuda_set_disassemble_per, cuda_show_disassemble_per,
                        &setcudalist, &showcudalist);
}

bool
cuda_options_disassemble_per_function (void)
{
  return cuda_disassemble_per == cuda_disassemble_per_function;
}

bool
cuda_options_disassemble_per_file (void)
{
  return cuda_disassemble_per == cuda_disassemble_per_file;
}

/*
 * set cuda hide_internal_frames
 */
static bool cuda_hide_internal_frames = true;

static void
cuda_set_hide_internal_frames (const char *args, int from_tty, struct cmd_list_element *c)
{
  // force rebuilding frame stack to see the change
  reinit_frame_cache ();
}

static void
cuda_show_hide_internal_frames (struct ui_file *file, int from_tty,
                                struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("Hiding of CUDA internal frames is %s.\n"), value);
}

static void
cuda_options_initialize_hide_internal_frames (void)
{
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
static const char  cuda_show_kernel_events_none[]        = "none";
static const char  cuda_show_kernel_events_application[] = "application";
static const char  cuda_show_kernel_events_system[]      = "system";
static const char  cuda_show_kernel_events_all[]         = "all";

static const char *cuda_show_kernel_events_enums[] = {
  cuda_show_kernel_events_none,
  cuda_show_kernel_events_application,
  cuda_show_kernel_events_system,
  cuda_show_kernel_events_all,
  NULL
};

static const char *cuda_show_kernel_events = cuda_show_kernel_events_none;
/* Only host kernel events by default */
static unsigned int cuda_show_kernel_events_depth = 1;

static void
cuda_show_show_kernel_events (struct ui_file *file, int from_tty,
                              struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("Show CUDA kernel events is set to '%s'.\n"), value);
}

void
cuda_options_force_set_launch_notification_update (void)
{
  /* Only use KERNEL_READY events if we are using auto breakpoints or kernel ready events
     and we don't need to use the forced method. */
  if ((cuda_options_auto_breakpoints_needed () ||
       (cuda_show_kernel_events != cuda_show_kernel_events_none)) &&
      (!cuda_options_auto_breakpoints_forced_needed ()))
    cuda_debugapi::set_kernel_launch_notification_mode (CUDBG_KNL_LAUNCH_NOTIFY_EVENT);
  else
    cuda_debugapi::set_kernel_launch_notification_mode (CUDBG_KNL_LAUNCH_NOTIFY_DEFER);
}

static void
cuda_set_show_kernel_events (const char *args, int from_tty, struct cmd_list_element *c)
{
  /* Update to receive KERNEL_READY events. */
  cuda_options_force_set_launch_notification_update ();

  /* Update kernel entry bpts if needed. */
  cuda_module::auto_breakpoints_update_locations ();
}

static void
cuda_show_show_kernel_events_depth (struct ui_file *file, int from_tty,
                              struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("Show CUDA kernel events depth is set to %s.\n"), value);
}

static void
cuda_options_initialize_show_kernel_events (void)
{
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

  add_setshow_uinteger_cmd ("kernel_events_depth", class_cuda,
                            &cuda_show_kernel_events_depth,
                            _("Set the maximum depth of nested kernels event notifications."),
                            _("Show the maximum depth of nested kernels event notifications."),
                            _("Controls the maximum depth of the kernels after which no kernel event notifications will be displayed.\n"
                              "A value of zero means that there is no maximum and that all the kernel notifications are displayed.\n"
                              "A value of one means that the debugger will display kernel event notifications only for kernels launched from the CPU (default)."),
                            cuda_set_show_kernel_events, cuda_show_show_kernel_events_depth,
                            &setcudalist, &showcudalist);

  cuda_options_force_set_launch_notification_update ();
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


bool
cuda_options_auto_breakpoints_needed (void)
{
  return (cuda_break_on_launch != cuda_break_on_launch_none);
}

bool
cuda_options_auto_breakpoints_forced_needed (void)
{
  return ((cuda_show_kernel_events_depth > 1 &&
           cuda_show_kernel_events != cuda_show_kernel_events_none) ||
          (cuda_options_auto_breakpoints_needed () && 
	   cuda_is_device_launch_used ()));
}

/*
 * set cuda show_context_events
 */
static bool cuda_show_context_events = false;

static void
cuda_show_show_context_events (struct ui_file *file, int from_tty,
                               struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("Show CUDA context events is %s.\n"), value);
}

static void
cuda_options_initialize_show_context_events (void)
{
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
static bool cuda_launch_blocking = false;

static void
cuda_set_launch_blocking (const char *args, int from_tty, struct cmd_list_element *c)
{
  if (cuda_launch_blocking)
      gdb_printf ("On the next run, the CUDA kernel launches will be blocking.\n");
  else
      gdb_printf ("On the next run, the CUDA kernel launches will be non-blocking.\n");
}

static void
cuda_show_launch_blocking (struct ui_file *file, int from_tty,
                           struct cmd_list_element *c, const char *value)
{
  if (cuda_launch_blocking)
    gdb_printf (file, _("On the next run, the CUDA kernel launches will be blocking.\n"));
  else
    gdb_printf (file, _("On the next run, the CUDA kernel launches will be non-blocking.\n"));
}

static void
cuda_options_initialize_launch_blocking (void)
{
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
static const char  cuda_thread_selection_policy_logical[]  = "logical";
static const char  cuda_thread_selection_policy_physical[] = "physical";
static const char *cuda_thread_selection_policy_enums[]    = {
  cuda_thread_selection_policy_logical,
  cuda_thread_selection_policy_physical,
  NULL
};
static const char *cuda_thread_selection_policy = cuda_thread_selection_policy_logical;

static void
show_cuda_thread_selection_policy (struct ui_file *file, int from_tty,
                                   struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("CUDA thread selection policy is %s.\n"), value);
}

static void
cuda_options_initialize_thread_selection (void)
{
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
show_cuda_copyright_command (const char *ignore, int from_tty)
{
  print_gdb_version (gdb_stdout, false);
  gdb_printf ("\n");
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
static const char  cuda_api_failures_option_ignore[]       = "ignore";
static const char  cuda_api_failures_option_stop[]         = "stop";
static const char  cuda_api_failures_option_hide[]         = "hide";
static const char  cuda_api_failures_option_stop_all[]     = "stop_all";
static const char  cuda_api_failures_option_ignore_all[]   = "ignore_all";

static const char *cuda_api_failures_options_enums[] = {
    cuda_api_failures_option_ignore,
    cuda_api_failures_option_stop,
    cuda_api_failures_option_hide,
    cuda_api_failures_option_stop_all,
    cuda_api_failures_option_ignore_all,
    NULL
};

static const char *cuda_api_failures_option = cuda_api_failures_option_ignore;

static void
cuda_show_api_failures (struct ui_file *file, int from_tty,
                          struct cmd_list_element *c, const char *value)
{
  gdb_printf ("api_failures is set to '%s'.\n", value);
}

static void
cuda_set_api_failures  (const char *args, int from_tty, struct cmd_list_element *c)
{
  cuda_update_report_driver_api_error_flags ();
}

static void
cuda_options_initialize_api_failures (void)
{
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
static bool cuda_software_preemption_auto = false;
static enum auto_boolean cuda_software_preemption = AUTO_BOOLEAN_AUTO;

static void
cuda_show_cuda_software_preemption (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("Software preemption debugging is %s.\n"), value);
}

static void
cuda_options_initialize_software_preemption (void)
{
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
  struct gdb_environ &env = current_inferior ()->environment;
  const char *cuda_dsp = env.get ("CUDA_DEBUGGER_SOFTWARE_PREEMPTION");

  /* Software preemption auto value is determined by the
     CUDA_DEBUGGER_SOFTWARE_PREEMPTION env var */
  cuda_software_preemption_auto = cuda_dsp && strcmp(cuda_dsp,"1")==0;

  return cuda_software_preemption == AUTO_BOOLEAN_TRUE ||
        (cuda_software_preemption == AUTO_BOOLEAN_AUTO && cuda_software_preemption_auto);
}

/*
 * set cuda variable_value_cache
 */
static enum auto_boolean cuda_variable_value_cache_enabled = AUTO_BOOLEAN_TRUE;

static void
cuda_show_cuda_variable_value_cache_enabled (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("Variable value cache enabled is %s.\n"), value);
}

static void
cuda_options_initialize_variable_value_cache_enabled (void)
{
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

/*
 * set cuda api stat collection
 */
static bool cuda_gpu_collect_stats = true;

static void
cuda_show_cuda_gpu_collect_stats (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("CUDA debugger API statistics collection is %s.\n"), value);
}

bool
cuda_options_statistics_collection_enabled (void)
{
  return cuda_gpu_collect_stats;
}

static void
cuda_print_statistics (const char *args, int from_tty)
{
  struct ui_out *uiout = current_uiout;

  /* column headers */
  const char *header_name = "API name";
  const char *header_calls = "Number of calls";
  const char *header_avg = "Average call time(usec)";
  const char *header_min = "Min call time(usec)";
  const char *header_max = "Max call time(usec)";
  const char *header_total = "Total call time(usec)";

  size_t name_width = strlen(header_name);
  int row_no = 0;
  std::chrono::microseconds total_time = std::chrono::microseconds::zero ();

  auto preprocess_stats = [&](const cuda_api_stat &stat)
  {
    if (stat.times_called == 0)
      return;
    name_width = std::max(name_width, stat.name.length());
    ++row_no;
    total_time += stat.total_time;
  };
  cuda_debugapi::for_each_api_stat (preprocess_stats);

  ui_out_emit_table table_cleanup (uiout, 6, row_no, "CUDBGAPIStatTable");
  uiout->table_header (name_width,           ui_left,  "name",  header_name);
  uiout->table_header (strlen(header_calls), ui_right, "calls", header_calls);
  uiout->table_header (strlen(header_avg),   ui_right, "avg",   header_avg);
  uiout->table_header (strlen(header_min),   ui_right, "min",   header_min);
  uiout->table_header (strlen(header_max),   ui_right, "max",   header_max);
  uiout->table_header (strlen(header_total), ui_right, "total", header_total);
  uiout->table_body ();

  auto process_stats = [&] (const cuda_api_stat &stat) {
    if (stat.times_called == 0)
      return;
    ui_out_emit_tuple row_cleanup (uiout, "CUDBGAPIStatRow");
    uiout->field_string ("name", stat.name.c_str ());
    uiout->field_signed ("calls", stat.times_called);
    uiout->field_signed ("avg", stat.total_time.count () / stat.times_called);
    uiout->field_signed ("min", stat.min_time.count ());
    uiout->field_signed ("max", stat.max_time.count ());
    uiout->field_signed ("total", stat.total_time.count ());
    uiout->text ("\n");
  };
  cuda_debugapi::for_each_api_stat (process_stats);

  printf_unfiltered ("Total time spent in CUDBG API is %.6f sec\n", std::chrono::duration<double>(total_time).count ());
}

static void
cuda_reset_statistics (const char *args, int from_tty)
{
  cuda_debugapi::reset_api_stat ();
}

static void
cuda_options_initialize_stats (void)
{
  add_cmd ("cuda_stats", class_maintenance, cuda_print_statistics,
           _("Print statistics about CUDA Debugger API."),
           &maintenanceprintlist);

  add_cmd ("reset_cuda_stats", class_maintenance, cuda_reset_statistics,
	   _("Reset collected statistics about CUDA Debugger API."),
	   &maintenancelist);

  add_setshow_boolean_cmd ("cuda_stats", class_maintenance, &cuda_gpu_collect_stats,
                           _("Turn on/off CUDA Debugger API statistics collection"),
                           _("Show if CUDA Debugger API statistics collection is enabled."),
                           _("When enabled, cuda-gdb will collect debugger API call statistics."),
                           NULL, cuda_show_cuda_gpu_collect_stats,
                           &maintenance_set_cmdlist, &maintenance_show_cmdlist);
}

/*
 * maintenance print cuda_regmap
 */
static bool cuda_value_extrapolation = false;

static void
cuda_show_cuda_value_extrapolation (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("CUDA value extrapolation is %s.\n"), value);
}

bool
cuda_options_value_extrapolation_enabled (void)
{
  return cuda_value_extrapolation != 0;
}

static void
cuda_print_regmap (const char *args, int from_tty)
{
  for (objfile *objfile : current_program_space->objfiles ())
    {
      if (!objfile->cuda_objfile)
        continue;
      printf( "Objfile %s \n", objfile->original_name);
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
static bool cuda_gpu_single_stepping_optimizations = true;

static void
cuda_show_single_stepping_optimizations (struct ui_file *file, int from_tty,
                         struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("CUDA single stepping optimizations are %s.\n"), value);
}

bool
cuda_options_single_stepping_optimizations_enabled (void)
{
  return cuda_gpu_single_stepping_optimizations != false;
}

static void
cuda_options_initialize_single_stepping_optimization (void)
{
  add_setshow_boolean_cmd ("single_stepping_optimizations", class_cuda, &cuda_gpu_single_stepping_optimizations,
                           _("Turn on/off CUDA single-stepping optimizations"),
                           _("Show if CUDA single-stepping optimizations are enabled."),
                           _("When enabled, cuda-gdb will use optimized methods to"
                             " single-step the program, when verifiably correct."
                             " Those optimizations accelerate single-stepping most of the time."),
                           NULL, cuda_show_single_stepping_optimizations,
                           &setcudalist, &showcudalist);

}

/*
 * set cuda step_divergent_lanes
 */
static bool cuda_step_divergent_lanes = true;

static void
cuda_show_step_divergent_lanes (struct ui_file *file, int from_tty,
				struct cmd_list_element *c, const char *value)
{
  gdb_printf (file,
	      _ ("CUDA stepping of divergent lanes while instruction single "
		 "stepping is %s.\n"),
	      value);
}

bool
cuda_options_step_divergent_lanes_enabled (void)
{
  return cuda_step_divergent_lanes;
}

static void
cuda_options_initialize_step_divergent_lanes (void)
{
  add_setshow_boolean_cmd (
      "step_divergent_lanes", class_cuda, &cuda_step_divergent_lanes,
      _ ("Turn on/off CUDA stepping of divergent lanes while instruction "
	 "single stepping"),
      _ ("Show if CUDA stepping of divergent lanes while instruction "
	 "single stepping is enabled."),
      _ ("When on(default), cuda-gdb will repeatedly step the warp in "
	 "focus until the CUDA thread that is focused on is active. "
	 "When off, the warp in focus will only be stepped once and "
	 "the focused cuda thread will be changed to the nearest active lane "
	 "in the warp."),
      NULL, cuda_show_step_divergent_lanes, &setcudalist, &showcudalist);
}

static const char cuda_stop_signal_sigurg[]	= "SIGURG";
static const char cuda_stop_signal_sigtrap[]	= "SIGTRAP";
static const char *cuda_stop_signal_enum[] = {
	cuda_stop_signal_sigurg,
	cuda_stop_signal_sigtrap,
	NULL
};

static const char *cuda_stop_signal_string = cuda_stop_signal_sigurg;

static unsigned cuda_stop_signal = GDB_SIGNAL_URG;

static void
cuda_show_stop_signal (struct ui_file *file, int from_tty,
                       struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("CUDA stop signal is %s\n"), value);
}

static void
cuda_set_stop_signal (const char *args, int from_tty, struct cmd_list_element *c)
{
  if (strcasecmp(cuda_stop_signal_string,"SIGURG")==0)
     cuda_stop_signal = GDB_SIGNAL_URG;
  if (strcasecmp(cuda_stop_signal_string,"SIGTRAP")==0)
     cuda_stop_signal = GDB_SIGNAL_TRAP;

  if (is_remote_target (current_inferior ()->process_target ()))
    cuda_remote_set_option ();
}

unsigned cuda_options_stop_signal (void)
{
  return cuda_stop_signal;
}

static void
cuda_options_initialize_stop_signal (void)
{
  add_setshow_enum_cmd ("stop_signal", class_cuda,
                          cuda_stop_signal_enum, &cuda_stop_signal_string,
                          _("Set signal used to notify the debugger of the CUDA event"),
                          _("Show signal used to notify the debugger of the CUDA event"),
                          _("Could be set to SIGURG (default) or SIGTRAP (legacy)"),
                          cuda_set_stop_signal, cuda_show_stop_signal,
                          &setcudalist, &showcudalist);
}

/*
 * set cuda device_resume_on_cpu_dynamic_function_call
 */
static bool cuda_device_resume_on_cpu_dynamic_function_call = true;

static void
cuda_show_no_device_resume_on_cpu_dynamic_function_call (struct ui_file *file, int from_tty,
                                                         struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _("CUDA device resume during dynamic function call on host is %s.\n"), value);
}

static void
cuda_options_initialize_device_resume_on_cpu_dynamic_function_call (void)
{
  add_setshow_boolean_cmd ("device_resume_on_cpu_dynamic_function_call", class_maintenance, &cuda_device_resume_on_cpu_dynamic_function_call,
                           _("Turn on/off resuming device during dynamic function call on host"),
                           _("Show if resuming device during dynamic function call on host is enabled"),
                           _("When non-zero, CUDA device is resumed during dynamic function call on host."),
                           NULL, cuda_show_no_device_resume_on_cpu_dynamic_function_call,
                           &setcudalist, &showcudalist);
}

bool
cuda_options_device_resume_on_cpu_dynamic_function_call (void)
{
  return cuda_device_resume_on_cpu_dynamic_function_call;
}

/*Initialization */
void _initialize_cuda_options ();
void
_initialize_cuda_options ()
{
  cuda_options_initialize_cuda_prefix ();
  cuda_options_initialize_debug_cuda_prefix ();
  cuda_options_initialize_debug_trace ();
  cuda_options_initialize_debug_notifications ();
  cuda_options_initialize_debug_libcudbg ();
  cuda_options_initialize_debug_convenience_vars ();
  cuda_options_initialize_debug_strict ();
  cuda_options_initialize_coalescing ();
  cuda_options_initialize_break_on_launch ();
  cuda_options_initialize_api_failures ();
  cuda_options_initialize_disassemble_from ();
  cuda_options_initialize_disassemble_per ();
  cuda_options_initialize_hide_internal_frames ();
  cuda_options_initialize_show_kernel_events ();
  cuda_options_initialize_show_context_events ();
  cuda_options_initialize_launch_blocking ();
  cuda_options_initialize_thread_selection ();
  cuda_options_initialize_copyright ();
  cuda_options_initialize_notify ();
  cuda_options_initialize_software_preemption ();
  cuda_options_initialize_variable_value_cache_enabled ();
  cuda_options_initialize_stats ();
  cuda_options_initialize_value_extrapolation ();
  cuda_options_initialize_single_stepping_optimization ();
  cuda_options_initialize_step_divergent_lanes ();
  cuda_options_initialize_stop_signal ();
  cuda_options_initialize_device_resume_on_cpu_dynamic_function_call ();
  // Support removed - print notification
  cuda_options_initialize_memcheck ();
}
