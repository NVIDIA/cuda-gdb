sinclude(../config/tcl.m4)

dnl Find the location of the private Tcl headers
dnl When Tcl is installed, this is TCL_INCLUDE_SPEC/tcl-private/generic
dnl When Tcl is in the build tree, this is not needed.
dnl
dnl Note: you must use first use SC_LOAD_TCLCONFIG!
AC_DEFUN([CY_AC_TCL_PRIVATE_HEADERS], [
  AC_MSG_CHECKING([for Tcl private headers])
  private_dir=""
  dir=`echo ${TCL_INCLUDE_SPEC}/tcl-private/generic | sed -e s/-I//`
  if test -f ${dir}/tclInt.h ; then
    private_dir=${dir}
  fi

  if test x"${private_dir}" = x; then
    AC_ERROR(could not find private Tcl headers)
  else
    TCL_PRIVATE_INCLUDE="-I${private_dir}"
    AC_MSG_RESULT(${private_dir})
  fi
])

dnl Find the location of the private Tk headers
dnl When Tk is installed, this is TK_INCLUDE_SPEC/tk-private/generic
dnl When Tk is in the build tree, this not needed.
dnl
dnl Note: you must first use SC_LOAD_TKCONFIG
AC_DEFUN([CY_AC_TK_PRIVATE_HEADERS], [
  AC_MSG_CHECKING([for Tk private headers])
  private_dir=""
  dir=`echo ${TK_INCLUDE_SPEC}/tk-private/generic | sed -e s/-I//`
  if test -f ${dir}/tkInt.h; then
    private_dir=${dir}
  fi

  if test x"${private_dir}" = x; then
    AC_ERROR(could not find Tk private headers)
  else
    TK_PRIVATE_INCLUDE="-I${private_dir}"
    AC_MSG_RESULT(${private_dir})
  fi
])
