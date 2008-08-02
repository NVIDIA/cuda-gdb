dnl Process this file with autoconf to produce configure.

AC_INIT(src/subcommand.h)
AM_INIT_AUTOMAKE(libgui, 0.0)
AM_CONFIG_HEADER(config.h)
AM_MAINTAINER_MODE
AC_CANONICAL_HOST
AC_PROG_CC
AC_EXEEXT
AC_OBJEXT
AC_ARG_ENABLE(install-libgui, \
  [  --enable-install-libgui Install libgui.a and library header files])
AM_CONDITIONAL(CROSS_COMPILING, test x$cross_compiling = xyes)
AM_CONDITIONAL(INSTALL_LIBGUI, test x$enable_install_libgui = xyes)
AC_CHECK_TOOL(RANLIB, ranlib, :)
AC_CHECK_TOOL(AR, ar, :)

AC_FUNC_ALLOCA
AC_HAVE_HEADERS(stddef.h stdlib.h getopt.h unistd.h fcntl.h sys/file.h sys/wait.h string.h strings.h)
AC_CHECK_FUNCS(raise)

AC_ARG_ENABLE(ide, [  --enable-ide            Enable IDE support],
[case "${enableval}" in
  yes) ide=yes ;;
  no)  ide=no ;;
  *)   AC_MSG_ERROR(bad value ${enableval} for enable-ide option) ;;
esac],
[ide=no])
AM_CONDITIONAL(IDE_ENABLED, test x$ide = xyes)
if test x$ide = xyes; then
  AC_DEFINE(IDE_ENABLED)
fi

# If we have random, assume we have srandom.  If we have drand48,
# assume we have srand48.  If we have rand, assume we have srand.
AC_CHECK_FUNCS(random drand48 rand, break)

dnl Tcl ensures that strncasecmp is provided everywhere.  But in some
dnl situations we might need to declare it.  We check for that case
dnl here.
AC_EGREP_CPP(strncasecmp, [#include <string.h>],
  AC_DEFINE(HAVE_STRNCASECMP_DECL))


AC_CACHE_CHECK([for cygwin32], ide_cv_os_cygwin32,
[AC_EGREP_CPP(lose, [
#ifdef __CYGWIN32__
lose
#endif],[ide_cv_os_cygwin32=yes],[ide_cv_os_cygwin32=no])])
ac_win_build="no"
if test x$ide_cv_os_cygwin32 = xyes; then
  ac_win_build="yes"
fi

case "${host}" in
*-*-cygwin*)
        touch ac$$.c
        if ${CC} -c -mwin32 ac$$.c >/dev/null 2>&1; then
            case "$LIBGUI_CFLAGS" in
                *-mwin32*) ;;
                *) LIBGUI_CFLAGS="-mwin32 $LIBGUI_CFLAGS" ;;
            esac
        fi
        rm -f ac$$.o ac$$.c
        ;;
esac
case "${host}" in
*-*-cygwin*)
	LIBGUI_CFLAGS="-DWIN32 $LIBGUI_CFLAGS" ;;
esac
AC_SUBST(LIBGUI_CFLAGS) 

tmp="`cd $srcdir/library; pwd`"
if test x"$ac_cv_prog_CC" = xcl ; then
  tmp2="`cygpath --windows $tmp`"
  LIBGUI_LIBRARY_DIR="`echo $tmp2 | sed -e s#\\\\\\\\#/#g`"
else
  LIBGUI_LIBRARY_DIR=$tmp
fi
AC_SUBST(LIBGUI_LIBRARY_DIR)

# Check for Tcl and Tk.
SC_PATH_TCLCONFIG
SC_LOAD_TCLCONFIG
SC_PATH_TKCONFIG
SC_LOAD_TKCONFIG

here=`pwd`
cd ${srcdir}/..
topdir=`pwd`
cd ${here}

if test "${TCL_SRC_DIR}" = "${topdir}/tcl"; then
  # Using in-tree Tcl/Tk
  case "${host}" in
    *-*-cygwin*) platDir="win" ;;
    *) platDir="unix" ;;
  esac

  TCL_INCLUDES="-I${TCL_SRC_DIR}/generic -I${TCL_SRC_DIR}/${platDir}"
  TCL_LIBRARY=${TCL_BUILD_LIB_SPEC}
  TK_INCLUDES="-I${TK_SRC_DIR}/generic -I${TK_SRC_DIR}/${platDir}"
  TK_LIBRARY=${TK_BUILD_LIB_SPEC}
else
  # Using installed Tcl/Tk
  CY_AC_TCL_PRIVATE_HEADERS
  CY_AC_TK_PRIVATE_HEADERS
  TCL_INCLUDES="${TCL_INCLUDE_SPEC} ${TCL_PRIVATE_INCLUDE}"
  TCL_LIBRARY=${TCL_LIB_SPEC}
  TK_INCLUDES="${TK_INCLUDE_SPEC} ${TK_PRIVATE_INCLUDE}"
  TK_LIBRARY=${TK_LIB_SPEC}
fi

AC_SUBST(TCL_DEFS)
AC_SUBST(TCL_INCLUDES)
AC_SUBST(TK_DEFS)
AC_SUBST(TK_INCLUDES)
AC_SUBST(TK_XINCLUDES)
AC_SUBST(TCL_LIBS)
AC_SUBST(TK_LIBRARY)
AC_SUBST(TCL_LIBRARY)
AC_SUBST(TK_LIBS)

AM_CONDITIONAL(TCL_SHARED, test x$TCL_SHARED_BUILD = x1)

AC_SUBST(RPATH_ENVVAR)

AC_OUTPUT([Makefile library/Makefile src/Makefile])



