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
  global gdb_current_directory gdb_source_path gdb_inferior_args

  # gdb sessions are named after the executable.
  set name $gdb_exe_name
  set key gdb/session/$name

  # We fill a hash and then use that to set the actual preferences.

  # Always set the exe. name in case we later decide to change the
  # interpretation of the session key.
  set values(executable) $gdb_exe_name

  # Some simple state the user wants.  FIXME: these should have
  # dedicated commands instead of using `gdb_cmd'.
  set values(args) $gdb_inferior_args
  set values(dirs) $gdb_source_path
  set values(pwd) $gdb_current_directory

  set values(target) $gdb_target_name

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
    gdb_cmd "set args $values(args)"
  }

  if {[info exists values(executable)]} {
    gdb_clear_file
    set_exe_name $values(executable)
    set_exe
  }

  # FIXME: handle target
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
