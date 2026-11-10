/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2026 NVIDIA Corporation
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

#include <string.h>

#include "cuda-commands.h"
#include "cuda-logs.h"
#include "mi/mi-cmds.h"
#include "mi/mi-getopt.h"
#include "ui-out.h"

/* helper function to concatenate all the arguments into a single string with
   an extra space in-between each element, which will be the filter string for
   the info commands. */
static char *
concatenate_string (const char *const *argv, int argc)
{
  int allocated = 0;
  int copied = 0;
  int size = 0;
  int i;
  char *result = NULL;

  for (i = 0; i < argc; ++i)
    {
      size = strlen (argv[i]);
      if (copied + size + 1 > allocated)
	{
	  allocated += std::min (128, size + 1);
	  result = (char *)xrealloc (result, allocated);
	}
      memcpy (result + copied, argv[i], size);
      *(result + copied + size) = ' ';
      copied += size + 1;
    }

  if (!result)
    {
      result = (char *)xrealloc (result, 1);
      *result = 0;
    }
  else
    *(result + copied - 1) = 0;

  return result;
}

/* Report an option that appeared more than once.  Rule and setting
   mutations happen only after all options have been parsed, so this and all
   other syntax errors leave the current CUDA log configuration unchanged.  */
static void
mi_cuda_reject_duplicate_option (const char *command, const char *option,
				 bool already_seen)
{
  if (already_seen)
    error (_ ("%s: Duplicate option %s"), command, option);
}

/* Reject any unparsed operands.  mi_getopt stops at the first operand (or
   after "--"), while these commands accept options only.  */
static void
mi_cuda_reject_operands (const char *command, int argc,
			 const char *const *argv, int oind)
{
  if (oind != argc)
    error (_ ("%s: Unexpected argument: %s"), command, argv[oind]);
}

/* Validate a command that accepts neither options nor operands.  A lone
   "--" is the standard mi_getopt end-of-options marker and is accepted.  */
static void
mi_cuda_require_no_arguments (const char *command, int argc,
			      const char *const *argv)
{
  static const mi_opt opts[] = {
    { 0, 0, 0 },
  };
  int oind = 0;
  const char *oarg = nullptr;

  mi_getopt (command, argc, argv, opts, &oind, &oarg);
  mi_cuda_reject_operands (command, argc, argv, oind);
}

void
mi_cmd_cuda_log_set (const char *, const char *const *argv, int argc)
{
  enum option
    {
      ENABLED_OPT,
      LEVEL_OPT,
    };
  static const mi_opt opts[] = {
    { "-enabled", ENABLED_OPT, 1 },
    { "-level", LEVEL_OPT, 1 },
    { 0, 0, 0 },
  };

  const char *enabled = nullptr;
  const char *level = nullptr;
  bool enabled_seen = false;
  bool level_seen = false;
  int oind = 0;
  const char *oarg = nullptr;

  while (true)
    {
      const int opt
	  = mi_getopt ("-cuda-log-set", argc, argv, opts, &oind, &oarg);
      if (opt < 0)
	break;

      switch ((enum option) opt)
	{
	case ENABLED_OPT:
	  mi_cuda_reject_duplicate_option ("-cuda-log-set", "--enabled",
					   enabled_seen);
	  enabled_seen = true;
	  enabled = oarg;
	  break;
	case LEVEL_OPT:
	  mi_cuda_reject_duplicate_option ("-cuda-log-set", "--level",
					   level_seen);
	  level_seen = true;
	  level = oarg;
	  break;
	}
    }

  mi_cuda_reject_operands ("-cuda-log-set", argc, argv, oind);
  if (enabled_seen == level_seen)
    error (_ ("-cuda-log-set: Exactly one of --enabled or --level is "
	       "required"));

  if (enabled_seen)
    {
      if (strcmp (enabled, "on") == 0)
	cuda_options_set_logs_enabled (true);
      else if (strcmp (enabled, "off") == 0)
	cuda_options_set_logs_enabled (false);
      else
	error (_ ("-cuda-log-set: Invalid value for --enabled: %s"), enabled);
    }
  else
    cuda_options_set_log_level_name (level);
}

void
mi_cmd_cuda_log_status (const char *, const char *const *argv,
			int argc)
{
  mi_cuda_require_no_arguments ("-cuda-log-status", argc, argv);

  const cuda_log_rule_counts counts = cuda_options_log_rule_counts ();
  current_uiout->field_string ("log",
			       cuda_options_logs_enabled () ? "on" : "off");
  current_uiout->field_string ("log-level",
			       cuda_options_log_level_name ());
  current_uiout->field_unsigned ("rule-count", counts.total);
  current_uiout->field_unsigned ("enabled-rule-count", counts.enabled);
}

void
mi_cmd_cuda_log_add (const char *, const char *const *argv, int argc)
{
  enum option
    {
      OS_THREAD_OPT,
      SEVERITY_OPT,
      REGEX_OPT,
    };
  static const mi_opt opts[] = {
    { "-os-thread", OS_THREAD_OPT, 1 },
    { "-severity", SEVERITY_OPT, 1 },
    { "-regex", REGEX_OPT, 1 },
    { 0, 0, 0 },
  };

  const char *os_thread = "any";
  const char *severity = "error";
  const char *regex = "";
  bool os_thread_seen = false;
  bool severity_seen = false;
  bool regex_seen = false;
  int oind = 0;
  const char *oarg = nullptr;

  while (true)
    {
      const int opt
	  = mi_getopt ("-cuda-log-add", argc, argv, opts, &oind, &oarg);
      if (opt < 0)
	break;

      switch ((enum option) opt)
	{
	case OS_THREAD_OPT:
	  mi_cuda_reject_duplicate_option ("-cuda-log-add", "--os-thread",
					   os_thread_seen);
	  os_thread_seen = true;
	  os_thread = oarg;
	  break;
	case SEVERITY_OPT:
	  mi_cuda_reject_duplicate_option ("-cuda-log-add", "--severity",
					   severity_seen);
	  severity_seen = true;
	  severity = oarg;
	  break;
	case REGEX_OPT:
	  mi_cuda_reject_duplicate_option ("-cuda-log-add", "--regex",
					   regex_seen);
	  regex_seen = true;
	  regex = oarg;
	  break;
	}
    }

  mi_cuda_reject_operands ("-cuda-log-add", argc, argv, oind);
  const uint32_t rule_id
      = cuda_options_log_rule_add (os_thread, severity, regex);
  current_uiout->field_unsigned ("rule-id", rule_id);
}

void
mi_cmd_cuda_log_delete (const char *, const char *const *argv,
			int argc)
{
  enum option
    {
      RULE_ID_OPT,
      DELETE_ALL_OPT,
    };
  static const mi_opt opts[] = {
    { "-rule-id", RULE_ID_OPT, 1 },
    { "-delete-all", DELETE_ALL_OPT, 0 },
    { 0, 0, 0 },
  };

  const char *rule_id = nullptr;
  bool rule_id_seen = false;
  bool delete_all_seen = false;
  int oind = 0;
  const char *oarg = nullptr;

  while (true)
    {
      const int opt
	  = mi_getopt ("-cuda-log-delete", argc, argv, opts, &oind, &oarg);
      if (opt < 0)
	break;

      switch ((enum option) opt)
	{
	case RULE_ID_OPT:
	  mi_cuda_reject_duplicate_option ("-cuda-log-delete", "--rule-id",
					   rule_id_seen);
	  rule_id_seen = true;
	  rule_id = oarg;
	  break;
	case DELETE_ALL_OPT:
	  mi_cuda_reject_duplicate_option ("-cuda-log-delete", "--delete-all",
					   delete_all_seen);
	  delete_all_seen = true;
	  break;
	}
    }

  mi_cuda_reject_operands ("-cuda-log-delete", argc, argv, oind);
  if (rule_id_seen == delete_all_seen)
    error (_ ("-cuda-log-delete: Exactly one of --rule-id or --delete-all "
	       "is required"));

  size_t deleted_count;
  if (delete_all_seen)
    deleted_count = cuda_options_log_rules_delete_all ();
  else
    {
      cuda_options_log_rule_delete (rule_id);
      deleted_count = 1;
    }
  current_uiout->field_unsigned ("deleted-count", deleted_count);
}

static void
mi_cuda_log_set_rule_enabled (const char *command,
			      const char *const *argv, int argc, bool enabled)
{
  enum option
    {
      RULE_ID_OPT,
    };
  static const mi_opt opts[] = {
    { "-rule-id", RULE_ID_OPT, 1 },
    { 0, 0, 0 },
  };

  const char *rule_id = nullptr;
  bool rule_id_seen = false;
  int oind = 0;
  const char *oarg = nullptr;

  while (true)
    {
      const int opt = mi_getopt (command, argc, argv, opts, &oind, &oarg);
      if (opt < 0)
	break;

      switch ((enum option) opt)
	{
	case RULE_ID_OPT:
	  mi_cuda_reject_duplicate_option (command, "--rule-id", rule_id_seen);
	  rule_id_seen = true;
	  rule_id = oarg;
	  break;
	}
    }

  mi_cuda_reject_operands (command, argc, argv, oind);
  if (!rule_id_seen)
    error (_ ("%s: Option --rule-id is required"), command);

  cuda_options_log_rule_set_enabled (rule_id, enabled);
}

void
mi_cmd_cuda_log_enable (const char *, const char *const *argv,
			int argc)
{
  mi_cuda_log_set_rule_enabled ("-cuda-log-enable", argv, argc, true);
}

void
mi_cmd_cuda_log_disable (const char *, const char *const *argv,
			 int argc)
{
  mi_cuda_log_set_rule_enabled ("-cuda-log-disable", argv, argc, false);
}

void
mi_cmd_cuda_info_log (const char *, const char *const *argv, int argc)
{
  mi_cuda_require_no_arguments ("-cuda-info-log", argc, argv);
  cuda_log_rules_print (current_uiout);
}

void
mi_cmd_cuda_info_barriers (const char *command, const char *const *argv,
			   int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_barriers_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_devices (const char *command, const char *const *argv,
			  int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_devices_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_sms (const char *command, const char *const *argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_sms_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_warps (const char *command, const char *const *argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_warps_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_lanes (const char *command, const char *const *argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_lanes_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_kernels (const char *command, const char *const *argv,
			  int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_kernels_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_contexts (const char *command, const char *const *argv,
			   int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_contexts_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_clusters (const char *command, const char *const *argv,
			   int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_clusters_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_blocks (const char *command, const char *const *argv,
			 int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_blocks_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_threads (const char *command, const char *const *argv,
			  int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_threads_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_managed (const char *command, const char *const *argv,
			  int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_managed_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_info_line (const char *command, const char *const *argv, int argc)
{
  char *filter = concatenate_string (argv, argc);

  run_info_cuda_command (info_cuda_line_command, filter);

  xfree (filter);
}

void
mi_cmd_cuda_focus_query (const char *command, const char *const *argv,
			 int argc)
{
  char *query_string = concatenate_string (argv, argc);

  cuda_command_query (query_string);

  xfree (query_string);
}

void
mi_cmd_cuda_focus_switch (const char *command, const char *const *argv,
			  int argc)
{
  char *switch_string = concatenate_string (argv, argc);

  cuda_command_switch (switch_string);

  xfree (switch_string);
}
