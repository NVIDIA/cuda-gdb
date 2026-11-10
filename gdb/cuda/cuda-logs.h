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

#ifndef _CUDA_LOGS_H
#define _CUDA_LOGS_H 1

#include <cstddef>
#include <cstdint>
#include <string_view>

struct ui_out;

/* CUDA log display severity (set cuda log level).  */
enum class cuda_log_level_t
{
  error,
  warning,
  extra,
};

enum class cuda_log_rule_apply_policy
{
  install,
  cleanup
};

/* CUDA log collection (set cuda log on|off).  */
bool cuda_options_logs_enabled (void);
void cuda_options_set_logs_enabled (bool enabled);

/* CUDA log display level (set cuda log level).  */
cuda_log_level_t cuda_options_log_level (void);
const char *cuda_options_log_level_name (void);
void cuda_options_set_log_level_name (const char *value);

/* Host-side CUDA log rule counts.  ENABLED is a subset of TOTAL; neither
   count implies that the rules are installed in the current backend.  */
struct cuda_log_rule_counts
{
  size_t total;
  size_t enabled;
};

/* Command-independent CUDA log rule management shared by CLI and MI.
   Rule mutations preserve the existing transactional install/cleanup
   behavior.  */
uint32_t cuda_options_log_rule_add (std::string_view thread,
				    std::string_view severity,
				    std::string_view regex);
uint32_t cuda_options_log_rule_delete (std::string_view rule_id);
size_t cuda_options_log_rules_delete_all (void);
uint32_t cuda_options_log_rule_set_enabled (std::string_view rule_id,
					     bool enabled);
cuda_log_rule_counts cuda_options_log_rule_counts (void);

/* Print the configured rules using UIOUT.  MI receives a table even when no
   rules are configured; CLI retains its explanatory empty-state message.  */
void cuda_log_rules_print (struct ui_out *uiout);

uint32_t cuda_options_log_rule_id_for_backend_index (uint32_t backend_index);
/* Active ruleset index from the last setCudaLogRules; records matched under a
   superseded ruleset are discarded.  */
uint32_t cuda_options_log_active_ruleset_index (void);
/* Drop per-session bookkeeping (ruleset index, backend-id table, support
   state) on inferior teardown.  */
void cuda_options_reset_log_rule_runtime_state (void);
/* Push the enabled host rule list to the backend.  INSTALL is strict;
   CLEANUP is best-effort for delete/disable/init/replay paths.  */
void cuda_update_log_rules (
    cuda_log_rule_apply_policy policy = cuda_log_rule_apply_policy::cleanup);
/* Propagate pending rule updates to the backend.  */
void cuda_maybe_update_log_rules (void);

#endif
