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

#include "cli/cli-decode.h"
#include "cli/cli-option.h"
#include "cli/cli-setshow.h"
#include "cli/cli-utils.h"
#include "command.h"
#include "inferior.h"
#include "remote.h"
#include "ui-out.h"

#include "gdbsupport/buildargv.h"
#include "gdbsupport/function-view.h"

#include <algorithm>
#include <cstring>
#include <inttypes.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda-api.h"
#include "cuda-commands.h"
#include "cuda-logs.h"
#include "cuda-options.h"
#include "cuda-packet-manager.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"

/* Top-level "cuda" prefix command list, defined in cuda-commands.c.  */
extern struct cmd_list_element *cudalist;

static std::string cuda_log_add_help;
static struct cmd_list_element *cuda_log_set_setting_cmd;

static constexpr char cuda_log_rules_unsupported_build_msg[]
    = N_ ("CUDA log rules are not supported by this build.");

#if CUDBG_API_VERSION_REVISION >= 193
struct cuda_log_rule
{
  uint32_t rule_id;
  bool enabled;
  uint64_t thread_id;
  CUDBGCudaLogLevelFilter severity;
  std::string regex;
};

/* User ids are 1-based; backend indexes are 0-based plus the async tail.  */
static std::vector<cuda_log_rule> cuda_log_rules;
static std::vector<uint32_t> cuda_log_rule_backend_ids;
static uint32_t cuda_next_log_rule_id = 1;
/* Last installed ruleset.  Only records tagged with it may stop.  */
static uint32_t cuda_active_log_ruleset_index = 0;
static bool cuda_log_rules_dirty = false;
#endif

#if CUDBG_API_VERSION_REVISION >= 193
static void
cuda_clear_log_rule_backend_state (void)
{
  cuda_active_log_ruleset_index = 0;
  cuda_log_rule_backend_ids.clear ();
}

static std::string
cuda_options_log_rule_thread_to_string (uint64_t thread_id)
{
  if (thread_id == 0)
    return "any";

  return string_printf ("0x%" PRIx64, thread_id);
}

static const char *
cuda_options_log_rule_severity_name (CUDBGCudaLogLevelFilter severity)
{
  switch (severity)
    {
    case CUDBG_CUDA_LOG_LEVEL_FILTER_ANY:
      return "any";
    case CUDBG_CUDA_LOG_LEVEL_FILTER_ERROR:
      return "error";
    case CUDBG_CUDA_LOG_LEVEL_FILTER_WARNING:
      return "warning";
    default:
      return "unknown";
    }
}

/* Display strings for a rule.  Presentation only.  */
struct cuda_log_rule_display
{
  uint32_t rule_id;
  const char *enabled;	/* "yes" / "no" (static storage).  */
  std::string thread;	/* "any" or "0x<tid>".  */
  const char *severity; /* "any" / "error" / "warning" (static storage).  */
  std::string regex;	/* "<any>" or the rule's regex.  */
};

/* Parse VALUE as an unsigned integer through gdb's get_ulongest, so it accepts
   decimal/hex/octal and $convenience/$history refs like other numeric command
   arguments.  Rejects a leading sign so negative input can't wrap;
   get_ulongest throws on empty input or trailing junk.  */
static uint64_t
cuda_options_parse_unsigned (std::string_view value)
{
  if (!value.empty () && (value.front () == '-' || value.front () == '+'))
    error (_ ("Expected an unsigned integer: %.*s"), (int)value.size (),
	   value.data ());

  std::string text{ value };
  const char *p = text.c_str ();
  return get_ulongest (&p);
}

static uint64_t
cuda_options_log_rule_parse_thread_id (std::string_view value)
{
  if (value == "any")
    return 0;

  const uint64_t thread_id = cuda_options_parse_unsigned (value);
  if (thread_id == 0)
    error (_ ("Use 'any' to match any thread; 0 is not a valid "
	      "OS thread id."));

  return thread_id;
}

static CUDBGCudaLogLevelFilter
cuda_options_log_rule_parse_severity (std::string_view value)
{
  if (value == "any")
    return CUDBG_CUDA_LOG_LEVEL_FILTER_ANY;
  if (value == "error")
    return CUDBG_CUDA_LOG_LEVEL_FILTER_ERROR;
  if (value == "warning")
    return CUDBG_CUDA_LOG_LEVEL_FILTER_WARNING;

  error (_ ("Unknown CUDA log rule severity: %.*s"), (int)value.size (),
	 value.data ());
}

static uint32_t
cuda_options_log_rule_parse_id (std::string_view value)
{
  const uint64_t rule_id = cuda_options_parse_unsigned (value);

  constexpr uint64_t max_rule_id = UINT32_MAX;
  if (rule_id > max_rule_id)
    error (_ ("CUDA log rule id out of range: %.*s "
	      "(parsed as %" PRIu64 ", max %u)"),
	   (int)value.size (), value.data (), rule_id, (unsigned)UINT32_MAX);

  if (rule_id == 0)
    error (_ ("CUDA log rule id 0 is reserved; "
	      "valid ids start at 1."));

  return static_cast<uint32_t> (rule_id);
}

static std::vector<cuda_log_rule>::iterator
cuda_find_log_rule (uint32_t rule_id)
{
  return std::find_if (cuda_log_rules.begin (), cuda_log_rules.end (),
		       [rule_id] (const cuda_log_rule &rule) {
			 return rule.rule_id == rule_id;
		       });
}

struct cuda_log_rules_snapshot
{
  std::vector<cuda_log_rule> rules;
  uint32_t next_rule_id;
};

/* Apply a host edit, push it, and roll back the host edit on error.
   INSTALL is strict; CLEANUP is best-effort.  */
static void
cuda_log_rules_apply (cuda_log_rule_apply_policy policy,
		      gdb::function_view<void ()> apply_host_change)
{
  const bool install = (policy == cuda_log_rule_apply_policy::install);
  if (install)
    {
      /* Remote/QNX cannot stop on log rules.  */
      if (is_remote_target (current_inferior ()->process_target ()))
	error (_ ("CUDA log rules are not supported on remote targets."));

      if (!cuda_options_logs_enabled ())
	error (_ ("CUDA log rules require CUDA logs "
		  "collection; enable it with \"set cuda log on\"."));
    }

  cuda_log_rules_snapshot snapshot = { cuda_log_rules, cuda_next_log_rule_id };
  try
    {
      apply_host_change ();
      cuda_update_log_rules (policy);
    }
  catch (const gdb_exception_error &e)
    {
      cuda_trace ("cuda_log_rules_apply: rollback after error: %s", e.what ());
      cuda_log_rules = std::move (snapshot.rules);
      cuda_next_log_rule_id = snapshot.next_rule_id;
      throw;
    }
}

#endif /* CUDBG_API_VERSION_REVISION >= 193 */

uint32_t
cuda_options_log_rule_add (std::string_view thread,
			   std::string_view severity, std::string_view regex)
{
#if CUDBG_API_VERSION_REVISION >= 193
  const uint64_t thread_id
      = cuda_options_log_rule_parse_thread_id (thread);
  const CUDBGCudaLogLevelFilter severity_filter
      = cuda_options_log_rule_parse_severity (severity);

  uint32_t rule_id = 0;
  cuda_log_rules_apply (cuda_log_rule_apply_policy::install, [&] () {
    rule_id = cuda_next_log_rule_id;
    cuda_log_rules.push_back (
	{ rule_id, true, thread_id, severity_filter, std::string{ regex } });
    cuda_next_log_rule_id = rule_id + 1;
  });
  return rule_id;
#else
  (void) thread;
  (void) severity;
  (void) regex;
  error ("%s", _ (cuda_log_rules_unsupported_build_msg));
#endif
}

uint32_t
cuda_options_log_rule_delete (std::string_view rule_id_value)
{
#if CUDBG_API_VERSION_REVISION >= 193
  const uint32_t rule_id = cuda_options_log_rule_parse_id (rule_id_value);
  if (cuda_find_log_rule (rule_id) == cuda_log_rules.end ())
    error (_ ("Unknown CUDA log rule id %u"), rule_id);

  cuda_log_rules_apply (cuda_log_rule_apply_policy::cleanup, [&] () {
    cuda_log_rules.erase (cuda_find_log_rule (rule_id));
  });
  return rule_id;
#else
  (void) rule_id_value;
  error ("%s", _ (cuda_log_rules_unsupported_build_msg));
#endif
}

size_t
cuda_options_log_rules_delete_all (void)
{
#if CUDBG_API_VERSION_REVISION >= 193
  const size_t deleted_count = cuda_log_rules.size ();
  if (deleted_count != 0)
    cuda_log_rules_apply (cuda_log_rule_apply_policy::cleanup,
			  [] () { cuda_log_rules.clear (); });
  return deleted_count;
#else
  error ("%s", _ (cuda_log_rules_unsupported_build_msg));
#endif
}

uint32_t
cuda_options_log_rule_set_enabled (std::string_view rule_id_value,
				   bool enabled)
{
#if CUDBG_API_VERSION_REVISION >= 193
  const uint32_t rule_id = cuda_options_log_rule_parse_id (rule_id_value);
  auto it = cuda_find_log_rule (rule_id);
  if (it == cuda_log_rules.end ())
    error (_ ("Unknown CUDA log rule id %u"), rule_id);

  if (it->enabled != enabled)
    {
      const cuda_log_rule_apply_policy policy
	  = enabled ? cuda_log_rule_apply_policy::install
		    : cuda_log_rule_apply_policy::cleanup;
      cuda_log_rules_apply (
	  policy, [&] () { cuda_find_log_rule (rule_id)->enabled = enabled; });
    }

  return rule_id;
#else
  (void) rule_id_value;
  (void) enabled;
  error ("%s", _ (cuda_log_rules_unsupported_build_msg));
#endif
}

cuda_log_rule_counts
cuda_options_log_rule_counts (void)
{
#if CUDBG_API_VERSION_REVISION >= 193
  const size_t enabled = std::count_if (
      cuda_log_rules.begin (), cuda_log_rules.end (),
      [] (const cuda_log_rule &rule) { return rule.enabled; });
  return { cuda_log_rules.size (), enabled };
#else
  return { 0, 0 };
#endif
}

struct cuda_log_add_params
{
  std::string thread = "any";
  const char *severity = "error";
};

static const char *const cuda_log_rule_severity_enums[]
    = { "error", "warning", "any", nullptr };

static const gdb::option::option_def cuda_log_add_option_defs[] = {
  gdb::option::string_option_def<cuda_log_add_params>{
      "thread", [] (cuda_log_add_params *params) { return &params->thread; },
      nullptr,
      "Set the OS thread id to match, or 'any' to match all threads "
      "(default: any)." },

  gdb::option::enum_option_def<cuda_log_add_params>{
      "severity", cuda_log_rule_severity_enums,
      [] (cuda_log_add_params *params) { return &params->severity; }, nullptr,
      "Set the CUDA log severity to match (default: error)." },
};

static inline gdb::option::option_def_group
make_cuda_log_add_option_def_group (cuda_log_add_params *params)
{
  return { { cuda_log_add_option_defs }, params };
}

static void
cuda_log_add_command_completer (cmd_list_element *ignore,
				completion_tracker &tracker, const char *text,
				const char * /* word */)
{
  const auto def_grp = make_cuda_log_add_option_def_group (nullptr);
  gdb::option::complete_options (
      tracker, &text, gdb::option::PROCESS_OPTIONS_UNKNOWN_IS_OPERAND,
      def_grp);
}

/* cuda log add [-thread <id|any>] [-severity <error|warning|any>] [regex] */
static void
cuda_log_add_command (const char *args, int from_tty)
{
#if CUDBG_API_VERSION_REVISION >= 193
  cuda_log_add_params params;
  const auto def_grp = make_cuda_log_add_option_def_group (&params);
  args = args ? args : "";
  gdb::option::process_options (
      &args, gdb::option::PROCESS_OPTIONS_UNKNOWN_IS_OPERAND, def_grp);

  args = args == nullptr ? "" : skip_spaces (args);
  const std::string regex = args;

  const uint32_t rule_id
      = cuda_options_log_rule_add (params.thread, params.severity, regex);
  const cuda_log_rule &rule = *cuda_find_log_rule (rule_id);

  const std::string thread
      = cuda_options_log_rule_thread_to_string (rule.thread_id);
  if (regex.empty ())
    gdb_printf (
	_ ("CUDA log rule %u added: thread=%s severity=%s regex=<any>\n"),
	rule_id, thread.c_str (),
	cuda_options_log_rule_severity_name (rule.severity));
  else
    gdb_printf (
	_ ("CUDA log rule %u added: thread=%s severity=%s regex=\"%s\"\n"),
	rule_id, thread.c_str (),
	cuda_options_log_rule_severity_name (rule.severity), regex.c_str ());
#else
  error ("%s", _ (cuda_log_rules_unsupported_build_msg));
#endif
}

/* cuda log delete [rule_id]  (no id deletes all rules) */
static void
cuda_log_delete_command (const char *args, int from_tty)
{
#if CUDBG_API_VERSION_REVISION >= 193
  gdb_argv argv (args ? args : "");

  if (argv.count () == 0)
    {
      const size_t deleted_count = cuda_options_log_rules_delete_all ();
      gdb_printf (_ ("Deleted %zu CUDA log rule%s.\n"), deleted_count,
		  deleted_count == 1 ? "" : "s");
      return;
    }

  if (argv.count () != 1)
    error (_ ("Usage: cuda log delete [rule_id]"));

  const uint32_t rule_id = cuda_options_log_rule_delete (argv[0]);
  gdb_printf (_ ("CUDA log rule %u deleted.\n"), rule_id);
#else
  error ("%s", _ (cuda_log_rules_unsupported_build_msg));
#endif
}

static void
cuda_log_set_rule_enabled (const char *args, bool enabled)
{
#if CUDBG_API_VERSION_REVISION >= 193
  gdb_argv argv (args ? args : "");
  if (argv.count () != 1)
    error (_ ("Usage: cuda log %s <rule_id>"), enabled ? "enable" : "disable");

  const uint32_t rule_id
      = cuda_options_log_rule_set_enabled (argv[0], enabled);

  gdb_printf (_ ("CUDA log rule %u %s.\n"), rule_id,
	      enabled ? "enabled" : "disabled");
#else
  error ("%s", _ (cuda_log_rules_unsupported_build_msg));
#endif
}

static void
cuda_log_enable_command (const char *args, int from_tty)
{
  cuda_log_set_rule_enabled (args, true);
}

static void
cuda_log_disable_command (const char *args, int from_tty)
{
  cuda_log_set_rule_enabled (args, false);
}

#if CUDBG_API_VERSION_REVISION >= 193
static void
cuda_log_rules_emit_table (struct ui_out *uiout)
{
  size_t id_w = strlen ("Id");
  size_t enabled_w = strlen ("Enabled");
  size_t thread_w = strlen ("Thread");
  size_t severity_w = strlen ("Severity");
  size_t regex_w = strlen ("Regex");

  std::vector<cuda_log_rule_display> rows;
  rows.reserve (cuda_log_rules.size ());

  for (const cuda_log_rule &rule : cuda_log_rules)
    {
      cuda_log_rule_display row = {
	rule.rule_id,
	rule.enabled ? "yes" : "no",
	cuda_options_log_rule_thread_to_string (rule.thread_id),
	cuda_options_log_rule_severity_name (rule.severity),
	rule.regex.empty () ? "<any>" : rule.regex,
      };

      id_w = std::max (id_w, std::to_string (row.rule_id).size ());
      enabled_w = std::max (enabled_w, strlen (row.enabled));
      thread_w = std::max (thread_w, row.thread.size ());
      severity_w = std::max (severity_w, strlen (row.severity));
      regex_w = std::max (regex_w, row.regex.size ());
      rows.push_back (std::move (row));
    }

  const int nr_rows = static_cast<int> (rows.size ());

  ui_out_emit_table table_emitter (uiout, 5, nr_rows,
				   "InfoCudaLogRulesTable");
  uiout->table_header (id_w, ui_left, "rule-id", "Id");
  uiout->table_header (enabled_w, ui_left, "enabled", "Enabled");
  uiout->table_header (thread_w, ui_left, "os-thread", "Thread");
  uiout->table_header (severity_w, ui_left, "severity", "Severity");
  uiout->table_header (regex_w, ui_left, "regex", "Regex");
  uiout->table_body ();

  for (const cuda_log_rule_display &row : rows)
    {
      ui_out_emit_tuple row_emitter (uiout, "InfoCudaLogRulesRow");
      uiout->field_fmt ("rule-id", "%u", row.rule_id);
      uiout->field_string ("enabled", row.enabled);
      uiout->field_string ("os-thread", row.thread.c_str ());
      uiout->field_string ("severity", row.severity);
      uiout->field_string ("regex", row.regex.c_str ());
      uiout->text ("\n");
    }
}
#endif

void
cuda_log_rules_print (struct ui_out *uiout)
{
#if CUDBG_API_VERSION_REVISION >= 193
  if (cuda_log_rules.empty () && !uiout->is_mi_like_p ())
    {
      uiout->message (_ ("No CUDA log rules configured.\n"));
      return;
    }

  if (!uiout->is_mi_like_p ())
    uiout->message (_ ("CUDA log rules:\n"));

  cuda_log_rules_emit_table (uiout);
#else
  (void) uiout;
  error ("%s", _ (cuda_log_rules_unsupported_build_msg));
#endif
}

/* info cuda log (canonical listing).  */
void
info_cuda_log_command (const char *arg)
{
  cuda_log_rules_print (current_uiout);
}

uint32_t
cuda_options_log_rule_id_for_backend_index (uint32_t backend_index)
{
#if CUDBG_API_VERSION_REVISION >= 193
  if (backend_index >= cuda_log_rule_backend_ids.size ())
    return 0;

  return cuda_log_rule_backend_ids[backend_index];
#else
  return 0;
#endif
}

uint32_t
cuda_options_log_active_ruleset_index (void)
{
#if CUDBG_API_VERSION_REVISION >= 193
  return cuda_active_log_ruleset_index;
#else
  return 0;
#endif
}

void
cuda_options_reset_log_rule_runtime_state (void)
{
#if CUDBG_API_VERSION_REVISION >= 193
  /* Drop per-session backend bookkeeping; configured rules and the id counter
     persist so a re-run re-applies them.  */
  cuda_clear_log_rule_backend_state ();
  cuda_log_rules_dirty = true;
#endif
}

void
cuda_update_log_rules (cuda_log_rule_apply_policy policy)
{
#if CUDBG_API_VERSION_REVISION >= 193
  size_t enabled_count = 0;
  for (const cuda_log_rule &rule : cuda_log_rules)
    if (rule.enabled)
      ++enabled_count;

  /* Ruleset 0 (async-all) is the default: with no enabled rules and nothing
     to clear, skip the tail and the setCudaLogRules round-trip.  */
  if (enabled_count == 0 && cuda_log_rule_backend_ids.empty ())
    {
      cuda_active_log_ruleset_index = 0;
      cuda_log_rules_dirty = false;
      return;
    }

  if (is_remote_target (current_inferior ()->process_target ()))
    {
      /* CUDA log rules are not plumbed to remote/QNX; keep host bookkeeping
	 and skip the backend round-trip.  */
      if (enabled_count > 0)
	warning (_ ("CUDA log rules do not trigger on remote targets."));
      cuda_clear_log_rule_backend_state ();
      cuda_log_rules_dirty = false;
      return;
    }

  if (!cuda_initialized || !cuda_debugapi::api_state_initialized ())
    {
      cuda_clear_log_rule_backend_state ();
      cuda_log_rules_dirty = true;
      return;
    }

  std::vector<CUDBGCudaLogRule> new_rules;
  std::vector<std::string> new_regex_storage;
  std::vector<uint32_t> new_backend_rule_ids;

  new_rules.reserve (enabled_count + 1);
  new_regex_storage.reserve (enabled_count);
  new_backend_rule_ids.reserve (enabled_count);

  for (const cuda_log_rule &rule : cuda_log_rules)
    {
      if (!rule.enabled)
	continue;

      new_regex_storage.emplace_back (rule.regex);

      CUDBGCudaLogRule api_rule{};
      api_rule.osThreadIdFilter = rule.thread_id;
      api_rule.logLevelFilter = rule.severity;
      api_rule.messageFilterRegex = new_regex_storage.back ().empty ()
					? nullptr
					: new_regex_storage.back ().c_str ();
      api_rule.action = CUDBG_CUDA_LOG_RULE_ACTION_SEND_SYNC;
      new_rules.push_back (api_rule);
      new_backend_rule_ids.push_back (rule.rule_id);
    }

  /* Catch-all async tail: setCudaLogRules replaces the whole filter and drops
     unmatched logs, so this preserves async delivery for non-matching logs. */
  CUDBGCudaLogRule tail_rule{};
  tail_rule.osThreadIdFilter = 0;
  tail_rule.logLevelFilter = CUDBG_CUDA_LOG_LEVEL_FILTER_ANY;
  tail_rule.messageFilterRegex = nullptr;
  tail_rule.action = CUDBG_CUDA_LOG_RULE_ACTION_SEND_ASYNC;
  new_rules.push_back (tail_rule);

  cuda_log_rules_dirty = true;

  const auto prev_backend_rule_ids = cuda_log_rule_backend_ids;
  const uint32_t prev_active_ruleset_index = cuda_active_log_ruleset_index;

  uint32_t new_ruleset_index = 0;
  cuda_consume_and_print_logs ();
  CUDBGResult res = cuda_debugapi::set_cuda_log_rules (
      new_rules.data (), static_cast<uint32_t> (new_rules.size ()),
      &new_ruleset_index);

  if (res == CUDBG_ERROR_NOT_SUPPORTED)
    {
      cuda_log_rule_backend_ids = std::move (prev_backend_rule_ids);
      cuda_active_log_ruleset_index = prev_active_ruleset_index;

      if (policy == cuda_log_rule_apply_policy::install)
	error (_ ("CUDA log rules are not supported by the "
		  "current driver/backend; rule was not installed."));
      else
	warning (_ ("CUDA log rules are not supported by the "
		    "current driver/backend."));
    }
  else
    {
      cuda_log_rule_backend_ids = std::move (new_backend_rule_ids);
      /* Discard stop matches from stale id tables.  */
      cuda_active_log_ruleset_index = new_ruleset_index;
    }
  cuda_log_rules_dirty = false;
#endif
}

void
cuda_maybe_update_log_rules (void)
{
#if CUDBG_API_VERSION_REVISION >= 193
  if (!cuda_log_rules_dirty)
    return;

  if (!cuda_initialized || !cuda_debugapi::api_state_initialized ())
    return;

  try
    {
      cuda_update_log_rules (cuda_log_rule_apply_policy::cleanup);
    }
  catch (const gdb_exception_error &e)
    {
      cuda_trace ("cuda_maybe_update_log_rules: %s", e.what ());
    }
#endif
}

/* CUDA log collection: set cuda log on|off (legacy: driver_logs).  */
static bool cuda_logs = true;

static void
cuda_set_logs (const char *args, int from_tty, struct cmd_list_element *c)
{
  cuda_options_set_logs_enabled (cuda_logs);
}

static void
cuda_set_log_command (const char *args, int from_tty)
{
  gdb_assert (cuda_log_set_setting_cmd != nullptr);
  do_set_command (args, from_tty, cuda_log_set_setting_cmd);
}

static void
cuda_show_logs (struct ui_file *file, int from_tty, struct cmd_list_element *c,
		const char *value)
{
  gdb_printf (file, _ ("CUDA logs collection is %s.\n"), value);
}

bool
cuda_options_logs_enabled (void)
{
  return cuda_logs;
}

void
cuda_options_set_logs_enabled (bool enabled)
{
#if CUDBG_API_VERSION_REVISION >= 193
  if (!enabled)
    {
      const bool has_enabled_log_rules = std::any_of (
	  cuda_log_rules.begin (), cuda_log_rules.end (),
	  [] (const cuda_log_rule &rule) { return rule.enabled; });
      if (has_enabled_log_rules)
	warning (_ ("Disabling CUDA logs prevents enabled CUDA log rules from "
		    "triggering."));
    }
#endif

  cuda_logs = enabled;

  if (is_remote_target (current_inferior ()->process_target ()))
    cuda_remote_set_option ();
  else
    cuda_set_log_capability (cuda_logs);

#if CUDBG_API_VERSION_REVISION >= 193
  /* Re-propagate rules now (same replay as cuda_initialize).  */
  cuda_update_log_rules (cuda_log_rule_apply_policy::cleanup);
#endif /* CUDBG_API_VERSION_REVISION >= 193 */
}

/* CUDA log display level: set cuda log level (legacy: driver_log_level).
   cuda_log_level_setting always points at a cuda_log_level_enums entry
   so add_setshow_enum_cmd never sees a short-lived string.  */
static const char *const cuda_log_level_enums[]
    = { "error", "warning", "extra", NULL };
static const char *cuda_log_level_setting = "error";
static cuda_log_level_t cuda_log_level = cuda_log_level_t::error;

void
cuda_options_set_log_level_name (const char *value)
{
  const std::string_view sv{ value };
  if (sv == "error")
    {
      cuda_log_level = cuda_log_level_t::error;
      cuda_log_level_setting = cuda_log_level_enums[0];
    }
  else if (sv == "warning")
    {
      cuda_log_level = cuda_log_level_t::warning;
      cuda_log_level_setting = cuda_log_level_enums[1];
    }
  else if (sv == "extra")
    {
      cuda_log_level = cuda_log_level_t::extra;
      cuda_log_level_setting = cuda_log_level_enums[2];
    }
  else
    error (_ ("Unknown CUDA log level: %s"), value);
}

const char *
cuda_options_log_level_name (void)
{
  return cuda_log_level_setting;
}

static void
cuda_set_log_level (const char *args, int from_tty, struct cmd_list_element *c)
{
  cuda_options_set_log_level_name (cuda_log_level_setting);
}

static void
cuda_show_log_level (struct ui_file *file, int from_tty,
		     struct cmd_list_element *c, const char *value)
{
  gdb_printf (file, _ ("CUDA logs level is set to '%s'.\n"), value);
}

cuda_log_level_t
cuda_options_log_level (void)
{
  return cuda_log_level;
}

void _initialize_cuda_log ();
void
_initialize_cuda_log ()
{
  static struct cmd_list_element *set_cuda_log_cmdlist;
  static struct cmd_list_element *show_cuda_log_cmdlist;
  static struct cmd_list_element *hidden_set_cuda_log_cmdlist;

  add_prefix_cmd (
      "log", class_cuda, cuda_set_log_command,
      _ ("Set whether collection and display of CUDA logs is enabled."),
      &set_cuda_log_cmdlist, 1, &setcudalist);

  set_show_commands show_log_cmds = add_setshow_boolean_cmd (
      "log", class_cuda, &cuda_logs,
      _ ("Set whether collection and display of CUDA logs is enabled."),
      _ ("Show whether collection and display of CUDA logs is enabled."),
      _ ("When turned on, CUDA logs are collected from the driver "
	 "and displayed by the debugger."),
      cuda_set_logs, cuda_show_logs, &hidden_set_cuda_log_cmdlist,
      &showcudalist);
  cuda_log_set_setting_cmd = show_log_cmds.set;
  show_log_cmds.show->subcommands = &show_cuda_log_cmdlist;

  /* set/show cuda log level (canonical).  */
  set_show_commands level_cmds = add_setshow_enum_cmd (
      "level", class_cuda, cuda_log_level_enums, &cuda_log_level_setting,
      _ ("Set the minimum severity level for CUDA logs to display."),
      _ ("Show the minimum severity level for CUDA logs to display."),
      _ ("Valid options are:\n"
	 "  error   - Print only CUDA driver errors (default).\n"
	 "  warning - Print CUDA driver errors and warnings.\n"
	 "  extra - Same as 'warning', but adds printing of logging thread IDs"
	 " and log timestamps.\n"
	 " A thread ID is the return value of pthread_self() printed in hex"
	 " and timestamps are in local time formatted as HH:MM:SS.ms."),
      cuda_set_log_level, cuda_show_log_level, &set_cuda_log_cmdlist,
      &show_cuda_log_cmdlist);

  /* "cuda log <verb>": rule management.  Attach to the top-level cuda
     command list defined by cuda-commands.c.  */
  static struct cmd_list_element *cuda_log_cmdlist;
  add_basic_prefix_cmd ("log", class_cuda, _ ("CUDA driver log rules."),
			&cuda_log_cmdlist, 0, &cudalist);
  const auto add_def_grp = make_cuda_log_add_option_def_group (nullptr);
  cuda_log_add_help = gdb::option::build_help (_ ("Add a CUDA log rule.\n"
						  "Usage: cuda log add "
						  "[-thread <id|any>] "
						  "[-severity "
						  "<error|warning|any>] "
						  "[regex]\n"
						  "\n"
						  "Options:\n"
						  "%OPTIONS%"),
					       add_def_grp);
  cmd_list_element *log_add_cmd
      = add_cmd ("add", class_cuda, cuda_log_add_command,
		 cuda_log_add_help.c_str (), &cuda_log_cmdlist);
  set_cmd_completer_handle_brkchars (log_add_cmd,
				     cuda_log_add_command_completer);
  add_cmd ("delete", class_cuda, cuda_log_delete_command,
	   _ ("Delete CUDA log rules.\n"
	      "Usage: cuda log delete            (deletes all rules)\n"
	      "       cuda log delete <rule_id>"),
	   &cuda_log_cmdlist);
  add_cmd ("enable", class_cuda, cuda_log_enable_command,
	   _ ("Enable a CUDA log rule.\n"
	      "Usage: cuda log enable <rule_id>"),
	   &cuda_log_cmdlist);
  add_cmd ("disable", class_cuda, cuda_log_disable_command,
	   _ ("Disable a CUDA log rule.\n"
	      "Usage: cuda log disable <rule_id>"),
	   &cuda_log_cmdlist);

  /* Legacy "set/show cuda driver_log_level": hidden alias of the canonical
     "set/show cuda log level".  */
  cmd_list_element *legacy_level_set = add_alias_cmd (
      "driver_log_level", level_cmds.set, class_cuda, 0, &setcudalist);
  deprecate_cmd (legacy_level_set, "set cuda log level");
  cmd_list_element *legacy_level_show = add_alias_cmd (
      "driver_log_level", level_cmds.show, class_cuda, 0, &showcudalist);
  deprecate_cmd (legacy_level_show, "show cuda log level");

  /* Legacy "set/show cuda driver_logs": keep a boolean-only compatibility
     command, so "driver_logs" cannot reach "cuda log" subcommands.  */
  set_show_commands driver_logs_cmds = add_setshow_boolean_cmd (
      "driver_logs", class_cuda, &cuda_logs,
      _ ("Turn on/off collection and display of CUDA logs."),
      _ ("Show if collection and display of CUDA logs is enabled."),
      _ ("When turned on, CUDA logs are collected from the driver "
	 "and displayed by the debugger."),
      cuda_set_logs, cuda_show_logs, &setcudalist, &showcudalist);
  deprecate_cmd (driver_logs_cmds.set, "set cuda log on|off");
  deprecate_cmd (driver_logs_cmds.show, "show cuda log");
}
