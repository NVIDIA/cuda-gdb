/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2015-2025 NVIDIA Corporation
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

/* Ensure we define all dynlib imports in this source file by setting
 * the following macro. We only do this once to ensure a single definition. */
#define GDBDYN_DEFINE_VARIABLES 1

#include "defs.h"
#include "arch-utils.h"
#include "command.h"
#include "ui-out.h"
#include "cli/cli-script.h"
#include "gdbcmd.h"
#include "progspace.h"
#include "objfiles.h"
#include "value.h"
#include "language.h"
#include "python.h"
#include "python-internal.h"
#include "gdbsupport/scope-exit.h"
#include <dlfcn.h>
#include <regex>
#include <string>
#include <vector>

static bool python_initialized = false;
static void *libpython_handle = NULL;

/* Allow user to disable dlopen of python. */
bool cuda_disable_python = false;

/* Name of the library we were able to find/dlopen */
static std::string libpython_full_name;

static std::string python_internal_error {};

const char *
get_python_init_error ()
{
  if (python_internal_error.length() == 0)
    return nullptr;
  return python_internal_error.c_str();
}

/* base libpython name */
static const std::string libpython_base = "libpython";

/* Possible PEP-3149 flag names */
static const std::vector<std::string> libpython_flags = {
  "",
  "d", "m", "u",
  "dm", "du", "mu",
  "dmu"
};

/* Possible soname endings */
static const std::vector<std::string> libpython_endings = {
  ".so", ".so.1", ".so.1.0"
};

/* Scans through the know list of PEP-3149 flags looking for
 * a libpython*.so* file based on the passed in libname.
 * return nullptr if none of the derived library names are found.
 * If the library is found, save it's name for later printing.
 */
static void *
load_libpython (const std::string& libname)
{
  libpython_full_name.clear ();

  /* Start with the possible flags. We also include the empty string here. */
  for (const auto& it : libpython_flags)
    {
      const auto fname = libname + it;
      /* Add each possible ending - This is probably overkill. */
      for (const auto& it2 : libpython_endings)
	{
	  const auto qualified_name = fname + it2;
	  void *handle = dlopen (qualified_name.c_str (), RTLD_NOW | RTLD_GLOBAL);
	  /* Return the handle if we were able to open the library */
	  if (handle)
	    {
	      libpython_full_name = qualified_name;
	      return handle;
	    }
	}
    }
  return nullptr;
}

void
python_print_library ()
{
  static bool libpython_name_printed = false;

  if (libpython_handle && !libpython_name_printed)
    {
      extern std::string libpython_full_name;
      printf_unfiltered ("Using python library %s\n", libpython_full_name.c_str ());
      libpython_name_printed = true;
    }
}

bool
is_python_available (void)
{
  /* Check to see if python is explicitly disabled via command line. */
  if (cuda_disable_python)
    return false;

  if (python_initialized)
    return libpython_handle != NULL;

  python_initialized = true;

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 6
  // Special hack for python3.6 to search for
  // "compatible enough" later minor versions if 3.6
  // can't be found. We know 3.10 is incompatible, so
  // stop at 3.9
  std::string libpython_name;
  const auto major = std::to_string (PY_MAJOR_VERSION);
  for (auto minor = PY_MINOR_VERSION; minor <= 9; ++minor)
    {
      const auto name = libpython_base + major + "." + std::to_string (minor);
      libpython_handle = load_libpython (name);
      if (libpython_handle)
	{
	  libpython_name = name;
	  break;
	}
    }

  // Print the not-found message for the version we compiled with
  if (!libpython_handle)
    libpython_name = libpython_base + major + "." + std::to_string (PY_MINOR_VERSION);
#else    

  // Only check for the python we built against
  const auto major = std::to_string (PY_MAJOR_VERSION);
  const auto minor = std::to_string (PY_MINOR_VERSION);
  auto libpython_name = libpython_base + major + "." + minor;
  libpython_handle = load_libpython (libpython_name);

#endif

  if (!libpython_handle)
    {
      /* Failed to locate a compatible libpython. */
      python_internal_error = std::string{"Unable to locate compatible " + libpython_name + " library. Python integration disabled."};
      return false;
    }

/* Types and functions that are accessed via direct function pointer need to be
 * checked to ensure they have been resolved. If we fail to resolve, we need to
 * prevent python from being enabled to avoid potential null pointer dereferences. */
#define RESOLVE_AND_CHECK(varname, symname)				\
  varname = reinterpret_cast<decltype(varname)>(dlsym (libpython_handle, symname));			\
  if (!varname)								\
    {									\
      python_internal_error = "Failed to find " symname " in " + libpython_full_name; \
      goto err_out;							\
    }

  /* Resolve types, exceptions, and functions that are accessed via direct function pointer */
  RESOLVE_AND_CHECK(gdbdyn_IgnoreEnvironmentFlag, "Py_IgnoreEnvironmentFlag");
  RESOLVE_AND_CHECK(gdbdyn_DontWriteBytecodeFlag, "Py_DontWriteBytecodeFlag");

  RESOLVE_AND_CHECK(gdbdyn_None, "_Py_NoneStruct");
  RESOLVE_AND_CHECK(gdbdyn_True, "_Py_TrueStruct");
  RESOLVE_AND_CHECK(gdbdyn_False, "_Py_FalseStruct");
  RESOLVE_AND_CHECK(gdbdyn_NotImplemented, "_Py_NotImplementedStruct");

  RESOLVE_AND_CHECK(gdbdyn_FloatType, "PyFloat_Type");
  RESOLVE_AND_CHECK(gdbdyn_BoolType, "PyBool_Type");
  RESOLVE_AND_CHECK(gdbdyn_SliceType, "PySlice_Type");

  RESOLVE_AND_CHECK(pgdbpyExc_AttributeError, "PyExc_AttributeError");
  RESOLVE_AND_CHECK(pgdbpyExc_IndexError, "PyExc_IndexError");
  RESOLVE_AND_CHECK(pgdbpyExc_IOError, "PyExc_IOError");
  RESOLVE_AND_CHECK(pgdbpyExc_KeyError, "PyExc_KeyError");
  RESOLVE_AND_CHECK(pgdbpyExc_KeyboardInterrupt, "PyExc_KeyboardInterrupt");
  RESOLVE_AND_CHECK(pgdbpyExc_MemoryError, "PyExc_MemoryError");
  RESOLVE_AND_CHECK(pgdbpyExc_NameError, "PyExc_NameError");
  RESOLVE_AND_CHECK(pgdbpyExc_NotImplementedError, "PyExc_NotImplementedError");
  RESOLVE_AND_CHECK(pgdbpyExc_OverflowError, "PyExc_OverflowError");
  RESOLVE_AND_CHECK(pgdbpyExc_RuntimeError, "PyExc_RuntimeError");
  RESOLVE_AND_CHECK(pgdbpyExc_StopIteration, "PyExc_StopIteration");
  RESOLVE_AND_CHECK(pgdbpyExc_SystemError, "PyExc_SystemError");
  RESOLVE_AND_CHECK(pgdbpyExc_TypeError, "PyExc_TypeError");
  RESOLVE_AND_CHECK(pgdbpyExc_ValueError, "PyExc_ValueError");

  RESOLVE_AND_CHECK(pgdbdyn_OSReadlineTState, "_PyOS_ReadlineTState");
  RESOLVE_AND_CHECK(pgdbdyn_PyOS_ReadlineFunctionPointer, "PyOS_ReadlineFunctionPointer");
  RESOLVE_AND_CHECK(gdbdyn_PyType_GenericNew, "PyType_GenericNew");

/* Most functions are accessed via wrappers or indirectly. Since we can check
 * for nullptr references here, load failure is not an error. */
#define RESOLVE(varname,symname)				\
  varname = reinterpret_cast<decltype (varname)>(dlsym (libpython_handle, symname));

  /* Resolve called functions */
  RESOLVE(gdbdyn_Arg_UnpackTuple, "PyArg_UnpackTuple");
  RESOLVE(gdbdyn_BuildValue, "Py_BuildValue");
  RESOLVE(gdbdyn_ErrFormat, "PyErr_Format");
  RESOLVE(gdbdyn_FatalError, "Py_FatalError");
  RESOLVE(gdbdyn_PyObject_CallFunctionObjArgs, "PyObject_CallFunctionObjArgs");
  RESOLVE(gdbdyn_PyObject_CallMethodObjArgs, "PyObject_CallMethodObjArgs");
  RESOLVE(gdbdyn_PyArg_ParseTuple, "PyArg_ParseTuple");
  RESOLVE(gdbdyn_PyArg_ParseTuple_SizeT, "PyArg_ParseTuple_SizeT");
  RESOLVE(gdbdyn_PyArg_ParseTupleAndKeywords, "PyArg_ParseTupleAndKeywords");
  RESOLVE(gdbdyn_PyArg_VaParseTupleAndKeywords, "PyArg_VaParseTupleAndKeywords");
  RESOLVE(gdbdyn_PyUnicode_FromFormat, "PyUnicode_FromFormat");
  RESOLVE(gdbdyn_PyBuffer_Release, "PyBuffer_Release");
  RESOLVE(gdbdyn_PyBytes_AsString, "PyBytes_AsString");
  RESOLVE(gdbdyn_PyBytes_FromString, "PyBytes_FromString");
  RESOLVE(gdbdyn_PyBytes_AsStringAndSize, "PyBytes_AsStringAndSize");
  RESOLVE(gdbdyn_PyBytes_FromStringAndSize, "PyBytes_FromStringAndSize");
  RESOLVE(gdbdyn_PyMemoryView_FromObject, "PyMemoryView_FromObject");
  RESOLVE(gdbdyn_PyBool_FromLong, "PyBool_FromLong");
  RESOLVE(gdbdyn_PyCallable_Check, "PyCallable_Check");
  RESOLVE(gdbdyn_PyDict_New, "PyDict_New");
  RESOLVE(gdbdyn_PyDict_SetItem, "PyDict_SetItem");
  RESOLVE(gdbdyn_PyDict_SetItemString, "PyDict_SetItemString");
  RESOLVE(gdbdyn_PyDict_Keys, "PyDict_Keys");
  RESOLVE(gdbdyn_PyDict_Next, "PyDict_Next");
  RESOLVE(gdbdyn_PyErr_Clear, "PyErr_Clear");
  RESOLVE(gdbdyn_PyErr_ExceptionMatches, "PyErr_ExceptionMatches");
  RESOLVE(gdbdyn_PyErr_Fetch, "PyErr_Fetch");
  RESOLVE(gdbdyn_PyErr_GivenExceptionMatches, "PyErr_GivenExceptionMatches");
  RESOLVE(gdbdyn_PyErr_Occurred, "PyErr_Occurred");
  RESOLVE(gdbdyn_PyErr_Print, "PyErr_Print");
  RESOLVE(gdbdyn_PyErr_Restore, "PyErr_Restore");
  RESOLVE(gdbdyn_PyErr_SetFromErrno, "PyErr_SetFromErrno");
  RESOLVE(gdbdyn_PyErr_SetInterrupt, "PyErr_SetInterrupt");
  RESOLVE(gdbdyn_PyErr_SetNone, "PyErr_SetNone");
  RESOLVE(gdbdyn_PyErr_SetObject, "PyErr_SetObject");
  RESOLVE(gdbdyn_PyErr_SetString, "PyErr_SetString");
  RESOLVE(gdbdyn_PyErr_NewException, "PyErr_NewException");
  RESOLVE(gdbdyn_PyEval_InitThreads, "PyEval_InitThreads");
  RESOLVE(gdbdyn_PyEval_ReleaseLock, "PyEval_ReleaseLock");
  RESOLVE(gdbdyn_PyEval_RestoreThread, "PyEval_RestoreThread");
  RESOLVE(gdbdyn_PyEval_SaveThread, "PyEval_SaveThread");
  RESOLVE(gdbdyn_PyFloat_AsDouble, "PyFloat_AsDouble");
  RESOLVE(gdbdyn_PyFloat_FromDouble, "PyFloat_FromDouble");
  RESOLVE(gdbdyn_PyGILState_Ensure, "PyGILState_Ensure");
  RESOLVE(gdbdyn_PyGILState_Release, "PyGILState_Release");
  RESOLVE(gdbdyn_PyImport_AddModule, "PyImport_AddModule");
  RESOLVE(gdbdyn_PyImport_ImportModule, "PyImport_ImportModule");
  RESOLVE(gdbdyn_PyImport_GetModuleDict, "PyImport_GetModuleDict");
  RESOLVE(gdbdyn_PyImport_AppendInittab, "PyImport_AppendInittab");
  RESOLVE(gdbdyn_PyImport_ExtendInittab, "PyImport_ExtendInittab");
  RESOLVE(gdbdyn_PyBuffer_FillInfo, "PyBuffer_FillInfo");
  RESOLVE(gdbdyn_PyInt_AsLong, "PyInt_AsLong");
  RESOLVE(gdbdyn_PyInt_GetMax, "PyInt_GetMax");
  RESOLVE(gdbdyn_PyInt_FromLong, "PyInt_FromLong");
  RESOLVE(gdbdyn_PyIter_Next, "PyIter_Next");
  RESOLVE(gdbdyn_PyList_Append, "PyList_Append");
  RESOLVE(gdbdyn_PyList_AsTuple, "PyList_AsTuple");
  RESOLVE(gdbdyn_PyList_GetItem, "PyList_GetItem");
  RESOLVE(gdbdyn_PyList_SetItem, "PyList_SetItem");
  RESOLVE(gdbdyn_PyList_Insert, "PyList_Insert");
  RESOLVE(gdbdyn_PyList_New, "PyList_New");
  RESOLVE(gdbdyn_PyList_Size, "PyList_Size");
  RESOLVE(gdbdyn_PyLong_AsLongLong, "PyLong_AsLongLong");
  RESOLVE(gdbdyn_PyLong_AsLongLongAndOverflow, "PyLong_AsLongLongAndOverflow");
  RESOLVE(gdbdyn_PyLong_AsLong, "PyLong_AsLong");
  RESOLVE(gdbdyn_PyLong_AsUnsignedLongLong, "PyLong_AsUnsignedLongLong");
  RESOLVE(gdbdyn_PyLong_AsSsize_t, "PyLong_AsSsize_t");
  RESOLVE(gdbdyn_PyLong_FromLong, "PyLong_FromLong");
  RESOLVE(gdbdyn_PyLong_FromLongLong, "PyLong_FromLongLong");
  RESOLVE(gdbdyn_PyLong_FromUnsignedLong, "PyLong_FromUnsignedLong");
  RESOLVE(gdbdyn_PyLong_FromUnsignedLongLong, "PyLong_FromUnsignedLongLong");
  RESOLVE(gdbdyn_PyMem_Malloc, "PyMem_Malloc");
  RESOLVE(gdbdyn_PyMem_RawMalloc, "PyMem_RawMalloc");
  RESOLVE(gdbdyn_PyModule_AddIntConstant, "PyModule_AddIntConstant");
  RESOLVE(gdbdyn_PyModule_AddObject, "PyModule_AddObject");
  RESOLVE(gdbdyn_PyModule_AddStringConstant, "PyModule_AddStringConstant");
  RESOLVE(gdbdyn_PyModule_GetDict, "PyModule_GetDict");
  RESOLVE(gdbdyn_PyNumber_Long, "PyNumber_Long");
  RESOLVE(gdbdyn_PyOS_InterruptOccurred, "PyOS_InterruptOccurred");
  RESOLVE(gdbdyn_PyObject_AsReadBuffer, "PyObject_AsReadBuffer");
  RESOLVE(gdbdyn_PyObject_Repr, "PyObject_Repr");
  RESOLVE(gdbdyn_PyObject_CheckReadBuffer, "PyObject_CheckReadBuffer");
  RESOLVE(gdbdyn_PyObject_GenericGetAttr, "PyObject_GenericGetAttr");
  RESOLVE(gdbdyn_PyObject_GenericSetAttr, "PyObject_GenericSetAttr");
  RESOLVE(gdbdyn_PyObject_GetAttr, "PyObject_GetAttr");
  RESOLVE(gdbdyn_PyObject_GetAttrString, "PyObject_GetAttrString");
  RESOLVE(gdbdyn_PyObject_GetBuffer, "PyObject_GetBuffer");
  RESOLVE(gdbdyn_PyObject_GetIter, "PyObject_GetIter");
  RESOLVE(gdbdyn_PyObject_HasAttr, "PyObject_HasAttr");
  RESOLVE(gdbdyn_PyObject_HasAttrString, "PyObject_HasAttrString");
  RESOLVE(gdbdyn_PyObject_IsTrue, "PyObject_IsTrue");
  RESOLVE(gdbdyn_PyObject_IsInstance, "PyObject_IsInstance");
  RESOLVE(gdbdyn_PyObject_RichCompareBool, "PyObject_RichCompareBool");
  RESOLVE(gdbdyn_PyObject_SetAttrString, "PyObject_SetAttrString");
  RESOLVE(gdbdyn_PyObject_Str, "PyObject_Str");
  RESOLVE(gdbdyn_PyObject_Size, "PyObject_Size");
  RESOLVE(gdbdyn_PyRun_SimpleString, "PyRun_SimpleString");
  RESOLVE(gdbdyn_PyRun_StringFlags, "PyRun_StringFlags");
  RESOLVE(gdbdyn_PyRun_InteractiveLoop, "PyRun_InteractiveLoop");
  RESOLVE(gdbdyn_PyRun_SimpleFile, "PyRun_SimpleFile");
  RESOLVE(gdbdyn_PySequence_Check, "PySequence_Check");
  RESOLVE(gdbdyn_PySequence_Concat, "PySequence_Concat");
  RESOLVE(gdbdyn_PySequence_DelItem, "PySequence_DelItem");
  RESOLVE(gdbdyn_PySequence_GetItem, "PySequence_GetItem");
  RESOLVE(gdbdyn_PySequence_Index, "PySequence_Index");
  RESOLVE(gdbdyn_PySequence_List, "PySequence_List");
  RESOLVE(gdbdyn_PySequence_Size, "PySequence_Size");
  RESOLVE(gdbdyn_PySys_GetObject, "PySys_GetObject");
  RESOLVE(gdbdyn_PySys_SetPath, "PySys_SetPath");
  RESOLVE(gdbdyn_PyThreadState_Get, "PyThreadState_Get");
  RESOLVE(gdbdyn_PyThreadState_Swap, "PyThreadState_Swap");
  RESOLVE(gdbdyn_PyTuple_GetItem, "PyTuple_GetItem");
  RESOLVE(gdbdyn_PyTuple_New, "PyTuple_New");
  RESOLVE(gdbdyn_PyTuple_SetItem, "PyTuple_SetItem");
  RESOLVE(gdbdyn_PyTuple_Size, "PyTuple_Size");
  RESOLVE(gdbdyn_PyType_IsSubtype, "PyType_IsSubtype");
  RESOLVE(gdbdyn_PyType_Ready, "PyType_Ready");
  RESOLVE(gdbdyn_Py_Finalize, "Py_Finalize");
  RESOLVE(gdbdyn_Py_Initialize, "Py_Initialize");
  RESOLVE(gdbdyn_PyModule_Create2, "PyModule_Create2");
  RESOLVE(gdbdyn_PyObject_Call, "PyObject_Call");
  RESOLVE(gdbdyn_PyObject_CallObject, "PyObject_CallObject");
  RESOLVE(gdbdyn_Py_SetProgramName, "Py_SetProgramName");
  RESOLVE(gdbdyn__PyObject_New, "_PyObject_New");
  RESOLVE(gdbdyn_PyCode_New, "PyCode_New");
  RESOLVE(gdbdyn_PyFrame_New, "PyFrame_New");
  RESOLVE(gdbdyn_PyUnicode_FromString, "PyUnicode_FromString");
  RESOLVE(gdbdyn_PyUnicode_CompareWithASCIIString, "PyUnicode_CompareWithASCIIString");
  RESOLVE(gdbdyn_PyUnicode_Decode, "PyUnicode_Decode");
  RESOLVE(gdbdyn_PyUnicode_AsEncodedString, "PyUnicode_AsEncodedString");
  RESOLVE(gdbdyn_PyUnicode_AsASCIIString, "PyUnicode_AsASCIIString");
  RESOLVE(gdbdyn_PyUnicode_FromEncodedObject, "PyUnicode_FromEncodedObject");
  RESOLVE(gdbdyn_PySlice_GetIndicesEx, "PySlice_GetIndicesEx");
  
  // Introduced in python3.10
  RESOLVE(gdbdyn_PyObject_CheckBuffer, "PyObject_CheckBuffer");
  RESOLVE(gdbdyn_Py_DecRef, "Py_DecRef");

  /* Ensure to disable writing of .pyc files before init. */
  if (gdbdyn_DontWriteBytecodeFlag != NULL)
    *gdbdyn_DontWriteBytecodeFlag = 1;

  return true;
err_out:
  dlclose (libpython_handle);
  libpython_handle = NULL;
  return false;
}
