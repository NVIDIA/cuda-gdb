AC_INIT(Make-rules)
AC_CONFIG_AUX_DIR(../../..)
AM_INIT_AUTOMAKE(insight, 1.0)
AM_MAINTAINER_MODE
AC_PROG_MAKE_SET
AC_CANONICAL_HOST
AC_PROG_CC
AC_OBJEXT
AC_EXEEXT

# Find Tcl, Tk, etc
SC_PATH_TCLCONFIG
SC_LOAD_TCLCONFIG
SC_PATH_TKCONFIG
SC_LOAD_TKCONFIG

# Special in-tree hackery
here=`pwd`
cd ${srcdir}/../../..
topdir=`pwd`
cd ${here}

if test "${TCL_SRC_DIR}" = "${topdir}/tcl"; then
  TCL_INCLUDES="-I${TCL_SRC_DIR}/generic"
  TCL_LIBRARY="${TCL_BUILD_LIB_SPEC}"
else
  TCL_INCLUDES="${TCL_INCLUDE_SPEC}"
  TCL_LIBRARY="${TCL_LIB_SPEC}"
fi

AC_SUBST(TCL_DBGX)
AC_SUBST(TCL_SHLIB_CFLAGS)
AC_SUBST(TCL_SHLIB_LD)
AC_SUBST(TCL_SHLIB_SUFFIX)
AC_SUBST(TCL_INCLUDES)
AC_SUBST(TCL_LIBRARY)

# Make sure TCL_SHLIB_SUFFIX is set
if test x$TCL_SHLIB_SUFFIX = x ; then
  case "${host}" in
    *cygwin*) TCL_SHLIB_SUFFIX=".dll" ;;
    *) TCL_SHILB_SUFFIX=".so" ;;
  esac
fi
AC_SUBST(TCL_SHLIB_SUFFIX)

# Since we're not using autoconf > 2.1x, we cannot use AC_CONFIG_FILES.
make_subdirs=""

# See if -mwin32 works for cygwin
case "${host}" in
  *cygwin*)
    ocflags=${CFLAGS}
    CFLAGS="${CFLAGS} -mwin32"
    AC_TRY_COMPILE(,,,CFLAGS="${ocflags}")
    ;;
  *) ;;
esac

# note toplevel plugin build directory
plugin_builddir=`pwd`

#
# Plugins
#

# Supported hosts

# Only supported/tested on linux, solaris, cygwin
supported=yes
case "${host}" in
  *cygwin*) ;;
  *solaris*) ;;
  *linux*) ;;
  *) supported=no ;;
esac

# Host-specific configury
case "${host}" in
  *cygwin*)
    AC_CHECK_TOOL(NM, nm, nm)
    AC_CHECK_TOOL(AS, as, as)
    AC_CHECK_TOOL(LD, ld, ld)
    AC_MSG_CHECKING("for libcygwin.a")
    LIBCYGWIN_A=`$CC -print-file-name=libcygwin.a`
    AC_MSG_RESULT($LIBCYGWIN_A)
    AC_SUBST(LIBCYGWIN_A)
    AC_CHECK_TOOL(DLLTOOL, dlltool)
    ;;
esac

# Plugins supported by this configuration
if test x${supported} = xyes; then
  plugins="rhabout"
  case "${target}" in
    *cygwin*) ;;
    *linux*) ;;
    *i?86*)
      plugins="${plugins} intel-pentium"
      ;;
  esac

  for i in $plugins ; do
    make_subdirs="${make_subdirs} $i"
  done
fi

AC_SUBST(make_subdirs)
AC_SUBST(plugin_builddir)

AC_OUTPUT(Makefile rhabout/Makefile:rhabout/Makefile.in:Make-rules rhabout/rhabout.tcl intel-pentium/Makefile:intel-pentium/Makefile.in:Make-rules intel-pentium/intel-pentium.tcl)
