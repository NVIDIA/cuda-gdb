#! /bin/sh

# NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2014 NVIDIA Corporation
# Modified from the original GDB file referenced above by the CUDA-GDB 
# team at NVIDIA <cudatools@nvidia.com>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

# Similar to gdb.sh (original), but without system search paths and
# architecture translation.

# gdb is setgid procmod and dyld will truncate any DYLD_FRAMEWORK_PATH etc
# settings on exec.  The user is really trying to set these things
# in their process, not gdb.  So we smuggle it over the setgid border in
# GDB_DYLD_* where it'll be laundered inside gdb before invoking the inferior.

unset GDB_DYLD_FRAMEWORK_PATH
unset GDB_DYLD_FALLBACK_FRAMEWORK_PATH
unset GDB_DYLD_LIBRARY_PATH
unset GDB_DYLD_FALLBACK_LIBRARY_PATH
unset GDB_DYLD_ROOT_PATH
unset GDB_DYLD_PATHS_ROOT
unset GDB_DYLD_IMAGE_SUFFIX
unset GDB_DYLD_INSERT_LIBRARIES
[ -n "$DYLD_FRAMEWORK_PATH" ] && GDB_DYLD_FRAMEWORK_PATH="$DYLD_FRAMEWORK_PATH"
[ -n "$DYLD_FALLBACK_FRAMEWORK_PATH" ] && GDB_DYLD_FALLBACK_FRAMEWORK_PATH="$DYLD_FALLBACK_FRAMEWORK_PATH"
[ -n "$DYLD_LIBRARY_PATH" ] && GDB_DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH"
[ -n "$DYLD_FALLBACK_LIBRARY_PATH" ] && GDB_DYLD_FALLBACK_LIBRARY_PATH="$DYLD_FALLBACK_LIBRARY_PATH"
[ -n "$DYLD_ROOT_PATH" ] && GDB_DYLD_ROOT_PATH="$DYLD_ROOT_PATH"
[ -n "$DYLD_PATHS_ROOT" ] && GDB_DYLD_PATHS_ROOT="$DYLD_PATHS_ROOT"
[ -n "$DYLD_IMAGE_SUFFIX" ] && GDB_DYLD_IMAGE_SUFFIX="$DYLD_IMAGE_SUFFIX"
[ -n "$DYLD_INSERT_LIBRARIES" ] && GDB_DYLD_INSERT_LIBRARIES="$DYLD_INSERT_LIBRARIES"
export GDB_DYLD_FRAMEWORK_PATH
export GDB_DYLD_FALLBACK_FRAMEWORK_PATH
export GDB_DYLD_LIBRARY_PATH
export GDB_DYLD_FALLBACK_LIBRARY_PATH
export GDB_DYLD_ROOT_PATH
export GDB_DYLD_PATHS_ROOT
export GDB_DYLD_IMAGE_SUFFIX
export GDB_DYLD_INSERT_LIBRARIES

# dyld will warn if any of these are set and the user invokes a setgid program
# like gdb.
unset DYLD_FRAMEWORK_PATH
unset DYLD_FALLBACK_FRAMEWORK_PATH
unset DYLD_LIBRARY_PATH
unset DYLD_FALLBACK_LIBRARY_PATH
unset DYLD_ROOT_PATH
unset DYLD_PATHS_ROOT
unset DYLD_IMAGE_SUFFIX
unset DYLD_INSERT_LIBRARIES

# Try to find cuda-binary-gdb on PATH, if it fails, search for it in
# the same folder as gdb wrapper script
CUDA_BINARY_GDB=$(dirname $0)/cuda-binary-gdb
type -p $CUDA_BINARY_GDB &> /dev/null
if [ $? -ne 0 ]; then
    echo "Unable to start cuda-binary-gdb: cannot find binary in '$(dirname $0)'"
    exit 1
fi

exec "$CUDA_BINARY_GDB" "$@"

