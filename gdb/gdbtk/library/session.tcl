# Local preferences functions for GDBtk.
# Copyright 2000 Red Hat, Inc.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License (GPL) as published by
# the Free Software Foundation; either version 2 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# An internal function used when saving sessions.  Returns a string
# that can be used to recreate all pertinent breakpoint state.
proc SESSION_serialize_bps {} {
  set result {}

  foreach bp_num [gdb_get_breakpoint_list] {
    lassign [gdb_get_breakpoint_info $bp_num] file function line_number \
      address type enabled disposition ignore_count command_list \
      condition thread hit_count user_specification

    switch -glob -- $type {
      "breakpoint" -
      "hw breakpoint" {
	if {$disposition == "delete"} {
	  set cmd tbreak
	} else {
	  set cmd break
	}

	append cmd " "
	if {$user_specification != ""} {
	  append cmd "$user_specification"
	} elseif {$file != ""} {
	  # BpWin::bp_store uses file tail here, but I think that is
	  # wrong.
	  append cmd "$file:$line_number"
	} else {
	  append cmd "*$address"
	}
      }
      "watchpoint" -
      "hw watchpoint" {
	set cmd watch
	if {$user_specification != ""} {
	  append cmd " $user_specification"
	} else {
	  # There's nothing sensible to do.
	  continue
	}
      }

      "catch*" {
	# FIXME: Don't know what to do.
	continue
      }

      default {
	# Can't serialize anything other than those listed above.
	continue
      }
    }

    lappend result [list $cmd $enabled $condition $command_list]
  }

  return $result
}

# An internal function used when loading sessions.  It takes a
# breakpoint string and recreates all the breakpoints.
proc SESSION_recreate_bps {specs} {
  foreach spec $specs {
    lassign $spec create enabled condition commands

    # Create the breakpoint
    gdb_cmd $create

    # Below we use `\$bpnum'.  This means we don't have to figure out
    # the number of the breakpoint when doing further manipulations.

    if {! $enabled} {
      gdb_cmd "disable \$bpnum"
    }

    if {$condition != ""} {
      gdb_cmd "cond \$bpnum $condition"
    }

    if {[llength $commands]} {
      lappend commands end
      gdb_cmd "commands \$bpnum\n[join $commands \n]"
    }
  }
}

#
# This procedure decides what makes up a gdb `session'.  Roughly a
# session is whatever the user found useful when debugging a certain
# executable.
#
# Eventually we should expand this procedure to know how to save
# window placement and contents.  That requires more work.
#
proc session_save {} {
  global gdb_exe_name gdb_target_name
  global gdb_current_directory gdb_source_path

  # gdb sessions are named after the executable.
  set name $gdb_exe_name
  set key gdb/session/$name

  # We fill a hash and then use that to set the actual preferences.

  # Always set the exe. name in case we later decide to change the
  # interpretation of the session key.
  set values(executable) $gdb_exe_name

  # Some simple state the user wants.
  set values(args) [gdb_get_inferior_args]
  set values(dirs) $gdb_source_path
  set values(pwd) $gdb_current_directory
  set values(target) $gdb_target_name

  # Breakpoints.
  set values(breakpoints) [SESSION_serialize_bps]

  # Recompute list of recent sessions.  Trim to no more than 5 sessions.
  set recent [concat [list $name] \
		[lremove [pref getd gdb/recent-projects] $name]]
  if {[llength $recent] > 5} then {
    set recent [lreplace $recent 5 end]
  }
  pref setd gdb/recent-projects $recent

  foreach k [array names values] {
    pref setd $key/$k $values($k)
  }
  pref setd $key/all-keys [array names values]
}

#
# Load a session saved with session_save.  NAME is the pretty name of
# the session, as returned by session_list.
#
proc session_load {name} {
  global gdb_target_name

  # gdb sessions are named after the executable.
  set key gdb/session/$name

  # Fetch all keys for this session into an array.
  foreach k [pref getd $key/all-keys] {
    set values($k) [pref getd $key/$k]
  }

  if {[info exists values(dirs)]} {
    # FIXME: short-circuit confirmation.
    gdb_cmd "directory"
    gdb_cmd "directory $values(dirs)"
  }

  if {[info exists values(pwd)]} {
    gdb_cmd "cd $values(pwd)"
  }

  if {[info exists values(args)]} {
    gdb_set_inferior_args $values(args)
  }

  if {[info exists values(executable)]} {
    gdb_clear_file
    set_exe_name $values(executable)
    set_exe
  }

  if {[info exists values(breakpoints)]} {
    SESSION_recreate_bps $values(breakpoints)
  }

  if {[info exists values(target)]} {
    debug "Restoring Target: $values(target)"
    set gdb_target_name $values(target)
  }
}

#
# Delete a session.  NAME is the internal name of the session.
#
proc session_delete {name} {
  # FIXME: we can't yet fully define this because the libgui
  # preference code doesn't supply a delete method.
  set recent [lremove [pref getd gdb/recent-projects] $name]
  pref setd gdb/recent-projects $recent
}

#
# Return a list of all known sessions.  This returns the `pretty name'
# of the session -- something suitable for a menu.
#
proc session_list {} {
  set newlist {}
  set result {}
  foreach name [pref getd gdb/recent-projects] {
    set exe [pref getd gdb/session/$name/executable]
    # Take this opportunity to prune the list.
    if {[file exists $exe]} then {
      lappend newlist $name
      lappend result $exe
    } else {
      # FIXME: if we could delete keys we would delete all keys
      # associated with NAME now.
    }
  }
  pref setd gdb/recent-projects $newlist
  return $result
}
