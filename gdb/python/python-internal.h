/* Gdb/Python header for private use by Python module.

   Copyright (C) 2008-2023 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#ifndef PYTHON_PYTHON_INTERNAL_H
#define PYTHON_PYTHON_INTERNAL_H

#include "extension.h"
#include "extension-priv.h"

/* These WITH_* macros are defined by the CPython API checker that
   comes with the Python plugin for GCC.  See:
   https://gcc-python-plugin.readthedocs.org/en/latest/cpychecker.html
   The checker defines a WITH_ macro for each attribute it
   exposes.  Note that we intentionally do not use
   'cpychecker_returns_borrowed_ref' -- that idiom is forbidden in
   gdb.  */

#ifdef WITH_CPYCHECKER_TYPE_OBJECT_FOR_TYPEDEF_ATTRIBUTE
#define CPYCHECKER_TYPE_OBJECT_FOR_TYPEDEF(ARG)		\
  __attribute__ ((cpychecker_type_object_for_typedef (ARG)))
#else
#define CPYCHECKER_TYPE_OBJECT_FOR_TYPEDEF(ARG)
#endif

#ifdef WITH_CPYCHECKER_SETS_EXCEPTION_ATTRIBUTE
#define CPYCHECKER_SETS_EXCEPTION __attribute__ ((cpychecker_sets_exception))
#else
#define CPYCHECKER_SETS_EXCEPTION
#endif

#ifdef WITH_CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION_ATTRIBUTE
#define CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION		\
  __attribute__ ((cpychecker_negative_result_sets_exception))
#else
#define CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION
#endif

/* /usr/include/features.h on linux systems will define _POSIX_C_SOURCE
   if it sees _GNU_SOURCE (which config.h will define).
   pyconfig.h defines _POSIX_C_SOURCE to a different value than
   /usr/include/features.h does causing compilation to fail.
   To work around this, undef _POSIX_C_SOURCE before we include Python.h.

   Same problem with _XOPEN_SOURCE.  */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

/* On sparc-solaris, /usr/include/sys/feature_tests.h defines
   _FILE_OFFSET_BITS, which pyconfig.h also defines.  Same work
   around technique as above.  */
#undef _FILE_OFFSET_BITS

/* A kludge to avoid redefinition of snprintf on Windows by pyerrors.h.  */
#if defined(_WIN32) && defined(HAVE_DECL_SNPRINTF)
#define HAVE_SNPRINTF 1
#endif

/* Another kludge to avoid compilation errors because MinGW defines
   'hypot' to '_hypot', but the C++ headers says "using ::hypot".  */
#ifdef __MINGW32__
# define _hypot hypot
#endif

/* Request clean size types from Python.  */
#define PY_SSIZE_T_CLEAN

/* Include the Python header files using angle brackets rather than
   double quotes.  On case-insensitive filesystems, this prevents us
   from including our python/python.h header file.  */
#include <Python.h>
#include <frameobject.h>
#include "py-ref.h"

#define Py_TPFLAGS_CHECKTYPES 0

/* If Python.h does not define WITH_THREAD, then the various
   GIL-related functions will not be defined.  However,
   PyGILState_STATE will be.  */
#ifndef WITH_THREAD
#define PyGILState_Ensure() ((PyGILState_STATE) 0)
#define PyGILState_Release(ARG) ((void)(ARG))
#define PyEval_InitThreads()
#define PyThreadState_Swap(ARG) ((void)(ARG))
#define PyEval_ReleaseLock()
#endif

/* Python supplies HAVE_LONG_LONG and some `long long' support when it
   is available.  These defines let us handle the differences more
   cleanly.  */
#ifdef HAVE_LONG_LONG

#define GDB_PY_LL_ARG "L"
#define GDB_PY_LLU_ARG "K"
typedef PY_LONG_LONG gdb_py_longest;
typedef unsigned PY_LONG_LONG gdb_py_ulongest;
#define gdb_py_long_as_ulongest PyLong_AsUnsignedLongLong

#else /* HAVE_LONG_LONG */

#define GDB_PY_LL_ARG "L"
#define GDB_PY_LLU_ARG "K"
typedef long gdb_py_longest;
typedef unsigned long gdb_py_ulongest;
#define gdb_py_long_as_ulongest PyLong_AsUnsignedLong

#endif /* HAVE_LONG_LONG */

#if PY_VERSION_HEX < 0x03020000
typedef long Py_hash_t;
#endif

/* PyMem_RawMalloc appeared in Python 3.4.  For earlier versions, we can just
   fall back to PyMem_Malloc.  */

#if PY_VERSION_HEX < 0x03040000
#define PyMem_RawMalloc PyMem_Malloc
#endif


#ifdef NVIDIA_PYTHON_DYNLIB
/* The following defines interfaces used by the dlopen mechanism to
 * support python without directly linking against libpython. */

/* Redefine NULL to nullptr to avoid warnings due to passing 
 * NULL to long int with the generated wrappers. */
#undef NULL
#define NULL nullptr

/* dynlib functions */
extern bool is_python_available ();
extern const char *get_python_init_error ();
extern void python_print_library ();

/* The following macro is used to allow us to maintain a single
 * list of function prototypes. By default these will be extern.
 * They are defined in the python-dynobj.c source file. */
#ifdef GDBDYN_DEFINE_VARIABLES
#define GDBDYN_EXTERN
#else
#define GDBDYN_EXTERN extern
#endif

/* Dynamic reference to flags */
GDBDYN_EXTERN int *gdbdyn_IgnoreEnvironmentFlag;
GDBDYN_EXTERN int *gdbdyn_DontWriteBytecodeFlag;

/* Dynamic reference to constants */
GDBDYN_EXTERN PyObject *gdbdyn_None;
GDBDYN_EXTERN PyObject *gdbdyn_True;
GDBDYN_EXTERN PyObject *gdbdyn_False;
GDBDYN_EXTERN PyObject *gdbdyn_NotImplemented;

GDBDYN_EXTERN PyTypeObject *gdbdyn_FloatType;
GDBDYN_EXTERN PyTypeObject *gdbdyn_BoolType;
GDBDYN_EXTERN PyTypeObject *gdbdyn_SliceType;
GDBDYN_EXTERN PyTypeObject *gdbdyn_UnicodeType;

/* Dynamic reference to exception types */
GDBDYN_EXTERN PyObject **pgdbpyExc_AttributeError;
GDBDYN_EXTERN PyObject **pgdbpyExc_IndexError;
GDBDYN_EXTERN PyObject **pgdbpyExc_IOError;
GDBDYN_EXTERN PyObject **pgdbpyExc_KeyError;
GDBDYN_EXTERN PyObject **pgdbpyExc_KeyboardInterrupt;
GDBDYN_EXTERN PyObject **pgdbpyExc_MemoryError;
GDBDYN_EXTERN PyObject **pgdbpyExc_NotImplementedError;
GDBDYN_EXTERN PyObject **pgdbpyExc_OverflowError;
GDBDYN_EXTERN PyObject **pgdbpyExc_RuntimeError;
GDBDYN_EXTERN PyObject **pgdbpyExc_StopIteration;
GDBDYN_EXTERN PyObject **pgdbpyExc_SystemError;
GDBDYN_EXTERN PyObject **pgdbpyExc_TypeError;
GDBDYN_EXTERN PyObject **pgdbpyExc_ValueError;
GDBDYN_EXTERN PyObject **pgdbpyExc_NameError;

GDBDYN_EXTERN PyThreadState **pgdbdyn_OSReadlineTState;

/* Dynamic functions exposed as function pointers */
GDBDYN_EXTERN char *(**pgdbdyn_PyOS_ReadlineFunctionPointer) (FILE*, FILE*, const char*);

/* Dynamic reference to functions */
GDBDYN_EXTERN int (*gdbdyn_Arg_UnpackTuple)(PyObject *, const char *, Py_ssize_t, Py_ssize_t, ...);
GDBDYN_EXTERN PyObject * (*gdbdyn_ErrFormat) (PyObject *, const char *, ...);
GDBDYN_EXTERN PyObject * (*gdbdyn_BuildValue) (const char *, ...);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyObject_CallFunctionObjArgs) (PyObject *,...);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyObject_CallMethodObjArgs) (PyObject *, PyObject *,...);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyObject_CallMethod) (PyObject *o, const char *m, const char *format, ...);
GDBDYN_EXTERN int (*gdbdyn_PyArg_ParseTuple) (PyObject *obj, const char *, ...);
GDBDYN_EXTERN int (*gdbdyn_PyArg_ParseTuple_SizeT) (PyObject *obj, const char *, ...);
GDBDYN_EXTERN int (*gdbdyn_PyArg_ParseTupleAndKeywords) (PyObject *obj, PyObject *, const char *, char **, ...);
GDBDYN_EXTERN int (*gdbdyn_PyArg_VaParseTupleAndKeywords) (PyObject *obj, PyObject *, const char *, char **, va_list);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyUnicode_FromFormat) (const char *format, ...);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyBool_FromLong) (long);
GDBDYN_EXTERN char * (*gdbdyn_PyBytes_AsString) (PyObject *o);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyBytes_FromString) (const char *v);
GDBDYN_EXTERN int (*gdbdyn_PyBytes_AsStringAndSize) (PyObject *obj, char **buffer, Py_ssize_t *Length);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyBytes_FromStringAndSize) (const char *v, Py_ssize_t len);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyMemoryView_FromObject) (PyObject *);
GDBDYN_EXTERN int (*gdbdyn_PyBuffer_FillInfo) (Py_buffer *view, PyObject *exporter, void *buf, Py_ssize_t len, int readonly, int flags);
GDBDYN_EXTERN int (*gdbdyn_PyCallable_Check) (PyObject *o);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyDict_New) (void);
GDBDYN_EXTERN int (*gdbdyn_PyDict_SetItem) (PyObject *p, PyObject *key, PyObject *val);
GDBDYN_EXTERN int (*gdbdyn_PyDict_SetItemString) (PyObject *dp, const char *key, PyObject *item);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyDict_Keys) (PyObject *p);
GDBDYN_EXTERN int (*gdbdyn_PyDict_Next) (PyObject *, Py_ssize_t *, PyObject **, PyObject **);
GDBDYN_EXTERN void (*gdbdyn_PyErr_Clear) (void);
GDBDYN_EXTERN int (*gdbdyn_PyErr_ExceptionMatches) (PyObject *);
GDBDYN_EXTERN void (*gdbdyn_PyErr_Fetch) (PyObject **, PyObject **, PyObject **);
GDBDYN_EXTERN int (*gdbdyn_PyErr_GivenExceptionMatches) (PyObject *, PyObject *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyErr_Occurred) (void);
GDBDYN_EXTERN void (*gdbdyn_PyErr_Print) (void);
GDBDYN_EXTERN void (*gdbdyn_PyErr_Restore) (PyObject *, PyObject *, PyObject *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyErr_SetFromErrno) (PyObject *);
GDBDYN_EXTERN void (*gdbdyn_PyErr_SetInterrupt) (void);
GDBDYN_EXTERN void (*gdbdyn_PyErr_SetNone) (PyObject *type);
GDBDYN_EXTERN void (*gdbdyn_PyErr_SetObject) (PyObject *, PyObject *);
GDBDYN_EXTERN void (*gdbdyn_PyErr_SetString) (PyObject *, const char *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyErr_NewException)(const char *name, PyObject *base, PyObject *dict);
GDBDYN_EXTERN void (*gdbdyn_PyEval_InitThreads) (void);
GDBDYN_EXTERN void (*gdbdyn_PyEval_ReleaseLock) (void);
GDBDYN_EXTERN void (*gdbdyn_PyEval_RestoreThread) (PyThreadState *);
GDBDYN_EXTERN PyThreadState * (*gdbdyn_PyEval_SaveThread) (void);
GDBDYN_EXTERN double (*gdbdyn_PyFloat_AsDouble) (PyObject *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyFloat_FromDouble) (double);
GDBDYN_EXTERN PyFrameObject * (*gdbdyn_PyFrame_New)(PyThreadState *, PyCodeObject *, PyObject *, PyObject *);
GDBDYN_EXTERN PyGILState_STATE (*gdbdyn_PyGILState_Ensure) (void);
GDBDYN_EXTERN void (*gdbdyn_PyGILState_Release) (PyGILState_STATE);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyImport_AddModule) (const char *name);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyImport_ImportModule) (const char *name);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyImport_GetModuleDict) (void);
GDBDYN_EXTERN int (*gdbdyn_PyImport_AppendInittab) (const char *name, PyObject *(*initfunc)(void));
GDBDYN_EXTERN int (*gdbdyn_PyImport_ExtendInittab) (struct _inittab *newtab);
GDBDYN_EXTERN long (*gdbdyn_PyInt_AsLong) (PyObject *);
GDBDYN_EXTERN long (*gdbdyn_PyInt_GetMax) (void);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyInt_FromLong) (long);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyIter_Next) (PyObject *);
GDBDYN_EXTERN int (*gdbdyn_PyIter_Check) (PyObject *);
GDBDYN_EXTERN int (*gdbdyn_PyList_Append) (PyObject *, PyObject *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyList_AsTuple) (PyObject *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyList_GetItem) (PyObject *, Py_ssize_t);
GDBDYN_EXTERN int (*gdbdyn_PyList_SetItem) (PyObject *, Py_ssize_t, PyObject *);
GDBDYN_EXTERN int (*gdbdyn_PyList_Insert) (PyObject *, Py_ssize_t, PyObject *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyList_New) (Py_ssize_t size);
GDBDYN_EXTERN Py_ssize_t (*gdbdyn_PyList_Size) (PyObject *);
GDBDYN_EXTERN PY_LONG_LONG (*gdbdyn_PyLong_AsLongLong) (PyObject *);
GDBDYN_EXTERN long (*gdbdyn_PyLong_AsLong) (PyObject *);
GDBDYN_EXTERN unsigned PY_LONG_LONG (*gdbdyn_PyLong_AsUnsignedLongLong) (PyObject *);
GDBDYN_EXTERN Py_ssize_t (*gdbdyn_PyLong_AsSsize_t) (PyObject *pylong);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyLong_FromLong) (long);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyLong_FromLongLong) (PY_LONG_LONG);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyLong_FromUnsignedLong) (unsigned long);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyLong_FromUnsignedLongLong) (unsigned PY_LONG_LONG);
GDBDYN_EXTERN void * (*gdbdyn_PyMem_RawMalloc) (size_t);
GDBDYN_EXTERN void * (*gdbdyn_PyMem_Malloc) (size_t);
GDBDYN_EXTERN int (*gdbdyn_PyModule_AddIntConstant) (PyObject *, const char *, long);
GDBDYN_EXTERN int (*gdbdyn_PyModule_AddObject) (PyObject *, const char *, PyObject *);
GDBDYN_EXTERN int (*gdbdyn_PyModule_AddStringConstant) (PyObject *, const char *, const char *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyModule_GetDict) (PyObject *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyNumber_Long) (PyObject *o);
GDBDYN_EXTERN int (*gdbdyn_PyOS_InterruptOccurred) (void);
GDBDYN_EXTERN int (*gdbdyn_PyObject_AsReadBuffer) (PyObject *obj, const void **, Py_ssize_t *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyObject_Repr) (PyObject *obj);
GDBDYN_EXTERN int (*gdbdyn_PyObject_CheckReadBuffer) (PyObject *obj);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyObject_GenericGetAttr) (PyObject *, PyObject *);
GDBDYN_EXTERN int (*gdbdyn_PyObject_GenericSetAttr)(PyObject *arg1, PyObject *arg2, PyObject *arg3);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyObject_GetAttr) (PyObject *, PyObject *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyObject_GetAttrString) (PyObject *, const char *);
GDBDYN_EXTERN int (*gdbdyn_PyObject_GetBuffer) (PyObject *exporter, Py_buffer *view, int flags);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyObject_GetIter) (PyObject *);
GDBDYN_EXTERN int (*gdbdyn_PyObject_HasAttr) (PyObject *, PyObject *);
GDBDYN_EXTERN int (*gdbdyn_PyObject_HasAttrString) (PyObject *, const char *);
GDBDYN_EXTERN int (*gdbdyn_PyObject_IsTrue) (PyObject *);
GDBDYN_EXTERN int (*gdbdyn_PyObject_IsInstance) (PyObject *inst, PyObject *cls);
GDBDYN_EXTERN int (*gdbdyn_PyObject_RichCompareBool) (PyObject *, PyObject *, int);
GDBDYN_EXTERN int (*gdbdyn_PyObject_SetAttrString) (PyObject *, const char *, PyObject *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyObject_Str) (PyObject *);
GDBDYN_EXTERN Py_ssize_t (*gdbdyn_PyObject_Size) (PyObject *o);
GDBDYN_EXTERN int (*gdbdyn_PyRun_SimpleString) (const char *command);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyRun_StringFlags) (const char *, int, PyObject *, PyObject *, PyCompilerFlags *);
GDBDYN_EXTERN int (*gdbdyn_PyRun_InteractiveLoop) (FILE *fp, const char *filename);
GDBDYN_EXTERN int (*gdbdyn_PyRun_SimpleFile) (FILE *fp, const char *filename);
GDBDYN_EXTERN int (*gdbdyn_PySequence_Check) (PyObject *o);
GDBDYN_EXTERN PyObject * (*gdbdyn_PySequence_Concat) (PyObject *o1, PyObject *o2);
GDBDYN_EXTERN int (*gdbdyn_PySequence_DelItem) (PyObject *o, Py_ssize_t i);
GDBDYN_EXTERN PyObject * (*gdbdyn_PySequence_GetItem) (PyObject *o, Py_ssize_t i);
GDBDYN_EXTERN Py_ssize_t (*gdbdyn_PySequence_Index) (PyObject *o, PyObject *value);
GDBDYN_EXTERN PyObject * (*gdbdyn_PySequence_List) (PyObject *o);
GDBDYN_EXTERN Py_ssize_t (*gdbdyn_PySequence_Size) (PyObject *o);
GDBDYN_EXTERN PyObject * (*gdbdyn_PySys_GetObject) (char *);
GDBDYN_EXTERN void (*gdbdyn_PySys_SetPath) (wchar_t *);
GDBDYN_EXTERN PyThreadState * (*gdbdyn_PyThreadState_Get) (void);
GDBDYN_EXTERN PyThreadState * (*gdbdyn_PyThreadState_Swap) (PyThreadState *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyTuple_GetItem) (PyObject *, Py_ssize_t);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyTuple_New) (Py_ssize_t size);
GDBDYN_EXTERN int (*gdbdyn_PyTuple_SetItem) (PyObject *, Py_ssize_t, PyObject *);
GDBDYN_EXTERN Py_ssize_t (*gdbdyn_PyTuple_Size) (PyObject *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyType_GenericNew)(PyTypeObject *, PyObject *, PyObject *);
GDBDYN_EXTERN int (*gdbdyn_PyType_IsSubtype) (PyTypeObject *, PyTypeObject *);
GDBDYN_EXTERN int (*gdbdyn_PyType_Ready) (PyTypeObject *);
GDBDYN_EXTERN void (*gdbdyn_Py_Finalize) (void);
GDBDYN_EXTERN void (*gdbdyn_Py_Initialize) (void);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyModule_Create2)(PyModuleDef *, int);
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 6)
GDBDYN_EXTERN void (*gdbdyn_Py_SetProgramName) (wchar_t *);
#else
GDBDYN_EXTERN void (*gdbdyn_Py_SetProgramName) (const wchar_t *);
#endif
GDBDYN_EXTERN PyObject * (*gdbdyn__PyObject_New) (PyTypeObject *);
GDBDYN_EXTERN PyCodeObject * (*gdbdyn_PyCode_New) (int, int, int, int, int,
    PyObject *, PyObject *, PyObject *, PyObject *,
    PyObject *, PyObject *, PyObject *, PyObject *,
    int, PyObject *);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyObject_CallObject) (PyObject *callable_object, PyObject *args);
GDBDYN_EXTERN PyObject * (*gdbdyn_PyObject_Call)(PyObject *callable_object, PyObject *args, PyObject *kw);
GDBDYN_EXTERN PyObject* (*gdbdyn_PyUnicode_Decode)(const char *s, Py_ssize_t size, const char *encoding, const char *errors);
GDBDYN_EXTERN PyObject* (*gdbdyn_PyUnicode_AsEncodedString)(register PyObject *unicode, const char *encoding, const char *errors);
GDBDYN_EXTERN PyObject* (*gdbdyn_PyUnicode_FromEncodedObject)(register PyObject *obj, const char *encoding, const char *errors);
GDBDYN_EXTERN PyObject* (*gdbdyn_PyUnicode_FromString)(const char *string);
GDBDYN_EXTERN int (*gdbdyn_PyUnicode_CompareWithASCIIString)(PyObject *uni, const char *string);
GDBDYN_EXTERN PyObject* (*gdbdyn_PyUnicode_AsASCIIString)(PyObject *uni);
GDBDYN_EXTERN void (*gdbdyn_PyBuffer_Release) (Py_buffer *buf);
GDBDYN_EXTERN int (*gdbdyn_PySlice_GetIndicesEx) (PyObject *slice, Py_ssize_t length, Py_ssize_t *start,
    Py_ssize_t *stop, Py_ssize_t *step, Py_ssize_t *slicelength);

/* Redefine python constants. We check for nullptr during initialization. */
#define Py_IgnoreEnvironmentFlag (*gdbdyn_IgnoreEnvironmentFlag)
#define Py_DontWriteBytecodeFlag (*gdbdyn_DontWriteBytecodeFlag)

#define _Py_NoneStruct (*gdbdyn_None)
#define _Py_TrueStruct (*gdbdyn_True)
#define _Py_FalseStruct (*gdbdyn_False)
#define _Py_NotImplementedStruct (*gdbdyn_NotImplemented)

#define PyFloat_Type (*gdbdyn_FloatType)
#define PyBool_Type (*gdbdyn_BoolType)
#define PySlice_Type (*gdbdyn_SliceType)

#define PyExc_AttributeError (*pgdbpyExc_AttributeError)
#define PyExc_IndexError (*pgdbpyExc_IndexError)
#define PyExc_IOError (*pgdbpyExc_IOError)
#define PyExc_KeyError (*pgdbpyExc_KeyError)
#define PyExc_KeyboardInterrupt (*pgdbpyExc_KeyboardInterrupt)
#define PyExc_MemoryError (*pgdbpyExc_MemoryError)
#define PyExc_NotImplementedError (*pgdbpyExc_NotImplementedError)
#define PyExc_OverflowError (*pgdbpyExc_OverflowError)
#define PyExc_RuntimeError (*pgdbpyExc_RuntimeError)
#define PyExc_StopIteration (*pgdbpyExc_StopIteration)
#define PyExc_SystemError (*pgdbpyExc_SystemError)
#define PyExc_TypeError (*pgdbpyExc_TypeError)
#define PyExc_ValueError (*pgdbpyExc_ValueError)
#define PyExc_NameError (*pgdbpyExc_NameError) 

#define _PyOS_ReadlineTState (*pgdbdyn_OSReadlineTState)

/* Redefine python functions exposed as function pointers. We check for nullptr during
 * initialization. */
#undef PyType_GenericNew
#define PyType_GenericNew (*gdbdyn_PyType_GenericNew)
#undef PyOS_ReadlineFunctionPointer
#define PyOS_ReadlineFunctionPointer (*pgdbdyn_PyOS_ReadlineFunctionPointer)

/* Wrap python functions to ensure they are only
 * called when python is enabled. */
template<typename Func, Func func, typename... Args>
static inline auto
gdbdyn_pywrapper (Args&&... args)
{
  if (!is_python_available ())
    {
      error (_("Attempted python call but python is uninitialized."));
    } 
  if ((*func) == nullptr)
    {
      error (_("Attempted to call a python binding that dlsym failed to load."));
    }
  return (*func)(std::forward<Args>(args)...);
}
#define GDBDYN_PYWRAPPER(name) gdbdyn_pywrapper<decltype(&name), &name>

/* Redefine python functions */
#undef PyArg_UnpackTuple
#define PyArg_UnpackTuple GDBDYN_PYWRAPPER(gdbdyn_Arg_UnpackTuple)
#undef PyErr_Format
#define PyErr_Format GDBDYN_PYWRAPPER(gdbdyn_ErrFormat)
#undef Py_BuildValue
#define Py_BuildValue GDBDYN_PYWRAPPER(gdbdyn_BuildValue)
#undef PyObject_CallFunctionObjArgs
#define PyObject_CallFunctionObjArgs GDBDYN_PYWRAPPER(gdbdyn_PyObject_CallFunctionObjArgs)
#undef PyObject_CallMethodObjArgs
#define PyObject_CallMethodObjArgs GDBDYN_PYWRAPPER(gdbdyn_PyObject_CallMethodObjArgs)
#undef PyObject_CallMethod
#define PyObject_CallMethod GDBDYN_PYWRAPPER(gdbdyn_PyObject_CallMethod)
#undef PyArg_ParseTuple
#define PyArg_ParseTuple GDBDYN_PYWRAPPER(gdbdyn_PyArg_ParseTuple)
#undef PyArg_ParseTuple_SizeT
#define PyArg_ParseTuple_SizeT GDBDYN_PYWRAPPER(gdbdyn_PyArg_ParseTuple_SizeT)
#undef PyArg_ParseTupleAndKeywords
#define PyArg_ParseTupleAndKeywords GDBDYN_PYWRAPPER(gdbdyn_PyArg_ParseTupleAndKeywords)
#undef PyArg_VaParseTupleAndKeywords
#define PyArg_VaParseTupleAndKeywords GDBDYN_PYWRAPPER(gdbdyn_PyArg_VaParseTupleAndKeywords)
#undef PyUnicode_FromFormat
#define PyUnicode_FromFormat GDBDYN_PYWRAPPER(gdbdyn_PyUnicode_FromFormat)
#undef PyBool_FromLong
#define PyBool_FromLong GDBDYN_PYWRAPPER(gdbdyn_PyBool_FromLong)
#undef PyBytes_AsString
#define PyBytes_AsString GDBDYN_PYWRAPPER(gdbdyn_PyBytes_AsString)
#undef PyBytes_FromString
#define PyBytes_FromString GDBDYN_PYWRAPPER(gdbdyn_PyBytes_FromString)
#undef PyBytes_AsStringAndSize
#define PyBytes_AsStringAndSize GDBDYN_PYWRAPPER(gdbdyn_PyBytes_AsStringAndSize)
#undef PyBytes_FromStringAndSize
#define PyBytes_FromStringAndSize GDBDYN_PYWRAPPER(gdbdyn_PyBytes_FromStringAndSize)
#undef PyMemoryView_FromObject
#define PyMemoryView_FromObject GDBDYN_PYWRAPPER(gdbdyn_PyMemoryView_FromObject)
#undef PyBuffer_FillInfo
#define PyBuffer_FillInfo GDBDYN_PYWRAPPER(gdbdyn_PyBuffer_FillInfo)
#undef PyCallable_Check
#define PyCallable_Check GDBDYN_PYWRAPPER(gdbdyn_PyCallable_Check)
#undef PyDict_Keys
#define PyDict_Keys GDBDYN_PYWRAPPER(gdbdyn_PyDict_Keys)
#undef PyDict_New
#define PyDict_New GDBDYN_PYWRAPPER(gdbdyn_PyDict_New)
#undef PyDict_SetItem
#define PyDict_SetItem GDBDYN_PYWRAPPER(gdbdyn_PyDict_SetItem)
#undef PyDict_SetItemString
#define PyDict_SetItemString GDBDYN_PYWRAPPER(gdbdyn_PyDict_SetItemString)
#undef PyDict_Next
#define PyDict_Next GDBDYN_PYWRAPPER(gdbdyn_PyDict_Next)
#undef PyErr_Clear
#define PyErr_Clear GDBDYN_PYWRAPPER(gdbdyn_PyErr_Clear)
#undef PyErr_ExceptionMatches
#define PyErr_ExceptionMatches GDBDYN_PYWRAPPER(gdbdyn_PyErr_ExceptionMatches)
#undef PyErr_Fetch
#define PyErr_Fetch GDBDYN_PYWRAPPER(gdbdyn_PyErr_Fetch)
#undef PyErr_GivenExceptionMatches
#define PyErr_GivenExceptionMatches GDBDYN_PYWRAPPER(gdbdyn_PyErr_GivenExceptionMatches)
#undef PyErr_Occurred
#define PyErr_Occurred GDBDYN_PYWRAPPER(gdbdyn_PyErr_Occurred)
#undef PyErr_Print
#define PyErr_Print GDBDYN_PYWRAPPER(gdbdyn_PyErr_Print)
#undef PyErr_Restore
#define PyErr_Restore GDBDYN_PYWRAPPER(gdbdyn_PyErr_Restore)
#undef PyErr_SetFromErrno
#define PyErr_SetFromErrno GDBDYN_PYWRAPPER(gdbdyn_PyErr_SetFromErrno)
#undef PyErr_SetInterrupt
#define PyErr_SetInterrupt GDBDYN_PYWRAPPER(gdbdyn_PyErr_SetInterrupt)
#undef PyErr_SetNone
#define PyErr_SetNone GDBDYN_PYWRAPPER(gdbdyn_PyErr_SetNone)
#undef PyErr_SetObject
#define PyErr_SetObject GDBDYN_PYWRAPPER(gdbdyn_PyErr_SetObject)
#undef PyErr_SetString
#define PyErr_SetString GDBDYN_PYWRAPPER(gdbdyn_PyErr_SetString)
#undef PyErr_NewException
#define PyErr_NewException GDBDYN_PYWRAPPER(gdbdyn_PyErr_NewException)
#undef PyEval_InitThreads
#define PyEval_InitThreads GDBDYN_PYWRAPPER(gdbdyn_PyEval_InitThreads)
#undef PyEval_ReleaseLock
#define PyEval_ReleaseLock GDBDYN_PYWRAPPER(gdbdyn_PyEval_ReleaseLock)
#undef PyEval_RestoreThread
#define PyEval_RestoreThread GDBDYN_PYWRAPPER(gdbdyn_PyEval_RestoreThread)
#undef PyEval_SaveThread
#define PyEval_SaveThread GDBDYN_PYWRAPPER(gdbdyn_PyEval_SaveThread)
#undef PyFloat_AsDouble
#define PyFloat_AsDouble GDBDYN_PYWRAPPER(gdbdyn_PyFloat_AsDouble)
#undef PyFloat_FromDouble
#define PyFloat_FromDouble GDBDYN_PYWRAPPER(gdbdyn_PyFloat_FromDouble)
#undef PyFrame_New
#define PyFrame_New GDBDYN_PYWRAPPER(gdbdyn_PyFrame_New)
#undef PyGILState_Ensure
#define PyGILState_Ensure GDBDYN_PYWRAPPER(gdbdyn_PyGILState_Ensure)
#undef PyGILState_Release
#define PyGILState_Release GDBDYN_PYWRAPPER(gdbdyn_PyGILState_Release)
#undef PyImport_AddModule
#define PyImport_AddModule GDBDYN_PYWRAPPER(gdbdyn_PyImport_AddModule)
#undef PyImport_ImportModule
#define PyImport_ImportModule GDBDYN_PYWRAPPER(gdbdyn_PyImport_ImportModule)
#undef PyImport_GetModuleDict
#define PyImport_GetModuleDict GDBDYN_PYWRAPPER(gdbdyn_PyImport_GetModuleDict)
#undef PyImport_AppendInittab
#define PyImport_AppendInittab GDBDYN_PYWRAPPER(gdbdyn_PyImport_AppendInittab)
#undef PyImport_ExtendInittab
#define PyImport_ExtendInittab GDBDYN_PYWRAPPER(gdbdyn_PyImport_ExtendInittab)
#undef PyInt_AsLong
#define PyInt_AsLong GDBDYN_PYWRAPPER(gdbdyn_PyInt_AsLong)
#undef PyInt_GetMax
#define PyInt_GetMax GDBDYN_PYWRAPPER(gdbdyn_PyInt_GetMax)
#undef PyInt_FromLong
#define PyInt_FromLong GDBDYN_PYWRAPPER(gdbdyn_PyInt_FromLong)
#undef PyIter_Next
#define PyIter_Next GDBDYN_PYWRAPPER(gdbdyn_PyIter_Next)
#undef PyIter_Check
#define PyIter_Check GDBDYN_PYWRAPPER(gdbdyn_PyIter_Check)
#undef PyList_Append
#define PyList_Append GDBDYN_PYWRAPPER(gdbdyn_PyList_Append)
#undef PyList_AsTuple
#define PyList_AsTuple GDBDYN_PYWRAPPER(gdbdyn_PyList_AsTuple)
#undef PyList_GetItem
#define PyList_GetItem GDBDYN_PYWRAPPER(gdbdyn_PyList_GetItem)
#undef PyList_SetItem
#define PyList_SetItem GDBDYN_PYWRAPPER(gdbdyn_PyList_SetItem)
#undef PyList_Insert
#define PyList_Insert GDBDYN_PYWRAPPER(gdbdyn_PyList_Insert)
#undef PyList_New
#define PyList_New GDBDYN_PYWRAPPER(gdbdyn_PyList_New)
#undef PyList_Size
#define PyList_Size GDBDYN_PYWRAPPER(gdbdyn_PyList_Size)
#undef PyLong_AsLongLong
#define PyLong_AsLongLong GDBDYN_PYWRAPPER(gdbdyn_PyLong_AsLongLong)
#undef PyLong_AsLong
#define PyLong_AsLong GDBDYN_PYWRAPPER(gdbdyn_PyLong_AsLong)
#undef PyLong_AsUnsignedLongLong
#define PyLong_AsUnsignedLongLong GDBDYN_PYWRAPPER(gdbdyn_PyLong_AsUnsignedLongLong)
#undef PyLong_AsSsize_t
#define PyLong_AsSsize_t GDBDYN_PYWRAPPER(gdbdyn_PyLong_AsSsize_t)
#undef PyLong_FromLong
#define PyLong_FromLong GDBDYN_PYWRAPPER(gdbdyn_PyLong_FromLong)
#undef PyLong_FromLongLong
#define PyLong_FromLongLong GDBDYN_PYWRAPPER(gdbdyn_PyLong_FromLongLong)
#undef PyLong_FromUnsignedLong
#define PyLong_FromUnsignedLong GDBDYN_PYWRAPPER(gdbdyn_PyLong_FromUnsignedLong)
#undef PyLong_FromUnsignedLongLong
#define PyLong_FromUnsignedLongLong GDBDYN_PYWRAPPER(gdbdyn_PyLong_FromUnsignedLongLong)
#undef PyMem_RawMalloc
#define PyMem_RawMalloc GDBDYN_PYWRAPPER(gdbdyn_PyMem_RawMalloc)
#undef PyMem_Malloc
#define PyMem_Malloc GDBDYN_PYWRAPPER(gdbdyn_PyMem_Malloc)
#undef PyModule_AddIntConstant
#define PyModule_AddIntConstant GDBDYN_PYWRAPPER(gdbdyn_PyModule_AddIntConstant)
#undef PyModule_AddObject
#define PyModule_AddObject GDBDYN_PYWRAPPER(gdbdyn_PyModule_AddObject)
#undef PyModule_AddStringConstant
#define PyModule_AddStringConstant GDBDYN_PYWRAPPER(gdbdyn_PyModule_AddStringConstant)
#undef PyModule_GetDict
#define PyModule_GetDict GDBDYN_PYWRAPPER(gdbdyn_PyModule_GetDict)
#undef PyNumber_Long
#define PyNumber_Long GDBDYN_PYWRAPPER(gdbdyn_PyNumber_Long)
#undef PyOS_InterruptOccurred
#define PyOS_InterruptOccurred GDBDYN_PYWRAPPER(gdbdyn_PyOS_InterruptOccurred)
#undef PyObject_AsReadBuffer
#define PyObject_AsReadBuffer GDBDYN_PYWRAPPER(gdbdyn_PyObject_AsReadBuffer)
#undef PyObject_Repr
#define PyObject_Repr GDBDYN_PYWRAPPER(gdbdyn_PyObject_Repr)
#undef PyObject_CheckReadBuffer
#define PyObject_CheckReadBuffer GDBDYN_PYWRAPPER(gdbdyn_PyObject_CheckReadBuffer)
#undef PyObject_GenericGetAttr
#define PyObject_GenericGetAttr GDBDYN_PYWRAPPER(gdbdyn_PyObject_GenericGetAttr)
#undef PyObject_GenericSetAttr
#define PyObject_GenericSetAttr GDBDYN_PYWRAPPER(gdbdyn_PyObject_GenericSetAttr)
#undef PyObject_GetAttr
#define PyObject_GetAttr GDBDYN_PYWRAPPER(gdbdyn_PyObject_GetAttr)
#undef PyObject_GetAttrString
#define PyObject_GetAttrString GDBDYN_PYWRAPPER(gdbdyn_PyObject_GetAttrString)
#undef PyObject_GetBuffer
#define PyObject_GetBuffer GDBDYN_PYWRAPPER(gdbdyn_PyObject_GetBuffer)
#undef PyObject_GetIter
#define PyObject_GetIter GDBDYN_PYWRAPPER(gdbdyn_PyObject_GetIter)
#undef PyObject_HasAttr
#define PyObject_HasAttr GDBDYN_PYWRAPPER(gdbdyn_PyObject_HasAttr)
#undef PyObject_HasAttrString
#define PyObject_HasAttrString GDBDYN_PYWRAPPER(gdbdyn_PyObject_HasAttrString)
#undef PyObject_IsTrue
#define PyObject_IsTrue GDBDYN_PYWRAPPER(gdbdyn_PyObject_IsTrue)
#undef PyObject_IsInstance
#define PyObject_IsInstance GDBDYN_PYWRAPPER(gdbdyn_PyObject_IsInstance)
#undef PyObject_RichCompareBool
#define PyObject_RichCompareBool GDBDYN_PYWRAPPER(gdbdyn_PyObject_RichCompareBool)
#undef PyObject_SetAttrString
#define PyObject_SetAttrString GDBDYN_PYWRAPPER(gdbdyn_PyObject_SetAttrString)
#undef PyObject_Str
#define PyObject_Str GDBDYN_PYWRAPPER(gdbdyn_PyObject_Str)
#undef PyObject_Size
#define PyObject_Size GDBDYN_PYWRAPPER(gdbdyn_PyObject_Size)
#undef PyRun_SimpleString
#define PyRun_SimpleString GDBDYN_PYWRAPPER(gdbdyn_PyRun_SimpleString)
#undef PyRun_StringFlags
#define PyRun_StringFlags GDBDYN_PYWRAPPER(gdbdyn_PyRun_StringFlags)
#undef PyRun_InteractiveLoop
#define PyRun_InteractiveLoop GDBDYN_PYWRAPPER(gdbdyn_PyRun_InteractiveLoop)
#undef PyRun_SimpleFile
#define PyRun_SimpleFile GDBDYN_PYWRAPPER(gdbdyn_PyRun_SimpleFile)
#undef PySequence_Check
#define PySequence_Check GDBDYN_PYWRAPPER(gdbdyn_PySequence_Check)
#undef PySequence_Concat
#define PySequence_Concat GDBDYN_PYWRAPPER(gdbdyn_PySequence_Concat)
#undef PySequence_DelItem
#define PySequence_DelItem GDBDYN_PYWRAPPER(gdbdyn_PySequence_DelItem)
#undef PySequence_GetItem
#define PySequence_GetItem GDBDYN_PYWRAPPER(gdbdyn_PySequence_GetItem)
#undef PySequence_Index
#define PySequence_Index GDBDYN_PYWRAPPER(gdbdyn_PySequence_Index)
#undef PySequence_List
#define PySequence_List GDBDYN_PYWRAPPER(gdbdyn_PySequence_List)
#undef PySequence_Size
#define PySequence_Size GDBDYN_PYWRAPPER(gdbdyn_PySequence_Size)
#undef PySys_GetObject
#define PySys_GetObject GDBDYN_PYWRAPPER(gdbdyn_PySys_GetObject)
#undef PySys_SetPath
#define PySys_SetPath GDBDYN_PYWRAPPER(gdbdyn_PySys_SetPath)
#undef PyThreadState_Get
#define PyThreadState_Get GDBDYN_PYWRAPPER(gdbdyn_PyThreadState_Get)
#undef PyThreadState_Swap
#define PyThreadState_Swap GDBDYN_PYWRAPPER(gdbdyn_PyThreadState_Swap)
#undef PyTuple_GetItem
#define PyTuple_GetItem GDBDYN_PYWRAPPER(gdbdyn_PyTuple_GetItem)
#undef PyTuple_New
#define PyTuple_New GDBDYN_PYWRAPPER(gdbdyn_PyTuple_New)
#undef PyTuple_SetItem
#define PyTuple_SetItem GDBDYN_PYWRAPPER(gdbdyn_PyTuple_SetItem)
#undef PyTuple_Size
#define PyTuple_Size GDBDYN_PYWRAPPER(gdbdyn_PyTuple_Size)
#undef PyType_IsSubtype
#define PyType_IsSubtype GDBDYN_PYWRAPPER(gdbdyn_PyType_IsSubtype)
#undef PyType_Ready
#define PyType_Ready GDBDYN_PYWRAPPER(gdbdyn_PyType_Ready)
#undef Py_Finalize
#define Py_Finalize GDBDYN_PYWRAPPER(gdbdyn_Py_Finalize)
#undef Py_Initialize
#define Py_Initialize GDBDYN_PYWRAPPER(gdbdyn_Py_Initialize)
#undef PyModule_Create2
#define PyModule_Create2 GDBDYN_PYWRAPPER(gdbdyn_PyModule_Create2)
#undef Py_SetProgramName
#define Py_SetProgramName GDBDYN_PYWRAPPER(gdbdyn_Py_SetProgramName)
#undef _PyObject_New
#define _PyObject_New GDBDYN_PYWRAPPER(gdbdyn__PyObject_New)
#undef PyCode_New
#define PyCode_New GDBDYN_PYWRAPPER(gdbdyn_PyCode_New)
#undef PyObject_CallObject
#define PyObject_CallObject GDBDYN_PYWRAPPER(gdbdyn_PyObject_CallObject)
#undef PyObject_Call
#define PyObject_Call GDBDYN_PYWRAPPER(gdbdyn_PyObject_Call)
#undef PyUnicode_Decode
#define PyUnicode_Decode GDBDYN_PYWRAPPER(gdbdyn_PyUnicode_Decode)
#undef PyUnicode_AsEncodedString
#define PyUnicode_AsEncodedString GDBDYN_PYWRAPPER(gdbdyn_PyUnicode_AsEncodedString)
#undef PyUnicode_FromEncodedObject
#define PyUnicode_FromEncodedObject GDBDYN_PYWRAPPER(gdbdyn_PyUnicode_FromEncodedObject)
#undef PyUnicode_FromString
#define PyUnicode_FromString GDBDYN_PYWRAPPER(gdbdyn_PyUnicode_FromString)
#undef PyUnicode_CompareWithASCIIString
#define PyUnicode_CompareWithASCIIString GDBDYN_PYWRAPPER(gdbdyn_PyUnicode_CompareWithASCIIString)
#undef PyUnicode_AsASCIIString
#define PyUnicode_AsASCIIString GDBDYN_PYWRAPPER(gdbdyn_PyUnicode_AsASCIIString)
#undef PyBuffer_Release
#define PyBuffer_Release GDBDYN_PYWRAPPER(gdbdyn_PyBuffer_Release)
#undef PySlice_GetIndicesEx
#define PySlice_GetIndicesEx GDBDYN_PYWRAPPER(gdbdyn_PySlice_GetIndicesEx)

/* Fix template expansion failures due to implicit casts used in gdb code. */
static inline PyObject*
gdb_PyUnicode_Decode (const char *s, Py_ssize_t size, const char *encoding, const char *errors)
{
  return PyUnicode_Decode (s, size, encoding, errors);
}
#undef PyUnicode_Decode
#define PyUnicode_Decode gdb_PyUnicode_Decode

static inline PyObject*
gdb_PyUnicode_AsEncodedString (PyObject *unicode, const char *encoding, const char *errors)
{
  return PyUnicode_AsEncodedString (unicode, encoding, errors);
}
#undef PyUnicode_AsEncodedString
#define PyUnicode_AsEncodedString gdb_PyUnicode_AsEncodedString

static inline PyObject*
gdb_PyRun_String (const char *str, int s, PyObject *g, PyObject *l)
{
  return PyRun_StringFlags (str, s, g, l, nullptr);
}
#undef PyRun_String
#define PyRun_String gdb_PyRun_String

static inline PyObject* 
gdb_PyRun_StringFlags (const char *str, int s, PyObject *g, PyObject *l, PyCompilerFlags *f)
{
  return PyRun_StringFlags (str, s, g, l, f);
}
#undef PyRun_StringFlags
#define PyRun_StringFlags gdb_PyRun_StringFlags

static inline PyObject*
gdb_PyObject_Call(PyObject *callable_object, PyObject *args, PyObject *kw)
{
  return PyObject_Call (callable_object, args, kw);
}
#undef PyObject_Call
#define PyObject_Call gdb_PyObject_Call

static inline PyObject* 
gdb_PyObject_CallObject (PyObject *callable_object, PyObject *args)
{
  return PyObject_CallObject (callable_object, args);
}
#undef PyObject_CallObject
#define PyObject_CallObject gdb_PyObject_CallObject

static inline PyObject*
gdb_PyBool_FromLong (long l)
{
  return PyBool_FromLong (l);
}
#undef PyBool_FromLong
#define PyBool_FromLong gdb_PyBool_FromLong

#endif /* NVIDIA_PYTHON_DYNLIB */

/* PyObject_CallMethod's 'method' and 'format' parameters were missing
   the 'const' qualifier before Python 3.4.  Hence, we wrap the
   function in our own version to avoid errors with string literals.
   Note, this is a variadic template because PyObject_CallMethod is a
   varargs function and Python doesn't have a "PyObject_VaCallMethod"
   variant taking a va_list that we could defer to instead.  */

template<typename... Args>
static inline PyObject *
gdb_PyObject_CallMethod (PyObject *o, const char *method, const char *format,
			 Args... args) /* ARI: editCase function */
{
  return PyObject_CallMethod (o,
			      const_cast<char *> (method),
			      const_cast<char *> (format),
			      args...);
}

#undef PyObject_CallMethod
#define PyObject_CallMethod gdb_PyObject_CallMethod

/* The 'name' parameter of PyErr_NewException was missing the 'const'
   qualifier in Python <= 3.4.  Hence, we wrap it in a function to
   avoid errors when compiled with -Werror.  */

static inline PyObject*
gdb_PyErr_NewException (const char *name, PyObject *base, PyObject *dict)
{
  return PyErr_NewException (const_cast<char *> (name), base, dict);
}

#ifdef NVIDIA_PYTHON_DYNLIB
/* Silence warning */
#undef PyErr_NewException
#endif
#define PyErr_NewException gdb_PyErr_NewException

/* PySys_GetObject's 'name' parameter was missing the 'const'
   qualifier before Python 3.4.  Hence, we wrap it in a function to
   avoid errors when compiled with -Werror.  */

static inline PyObject *
gdb_PySys_GetObject (const char *name)
{
  return PySys_GetObject (const_cast<char *> (name));
}

#ifdef NVIDIA_PYTHON_DYNLIB
/* Silence warning */
#undef PySys_GetObject
#endif
#define PySys_GetObject gdb_PySys_GetObject

/* PySys_SetPath was deprecated in Python 3.11.  Disable the deprecated
   code for Python 3.10 and newer.  */
#if PY_VERSION_HEX < 0x030a0000

/* PySys_SetPath's 'path' parameter was missing the 'const' qualifier
   before Python 3.6.  Hence, we wrap it in a function to avoid errors
   when compiled with -Werror.  */

# define GDB_PYSYS_SETPATH_CHAR wchar_t

static inline void
gdb_PySys_SetPath (const GDB_PYSYS_SETPATH_CHAR *path)
{
  PySys_SetPath (const_cast<GDB_PYSYS_SETPATH_CHAR *> (path));
}

#ifdef NVIDIA_PYTHON_DYNLIB
/* Silence warning */
#undef PySys_SetPath
#endif
#define PySys_SetPath gdb_PySys_SetPath
#endif

/* Wrap PyGetSetDef to allow convenient construction with string
   literals.  Unfortunately, PyGetSetDef's 'name' and 'doc' members
   are 'char *' instead of 'const char *', meaning that in order to
   list-initialize PyGetSetDef arrays with string literals (and
   without the wrapping below) would require writing explicit 'char *'
   casts.  Instead, we extend PyGetSetDef and add constexpr
   constructors that accept const 'name' and 'doc', hiding the ugly
   casts here in a single place.  */

struct gdb_PyGetSetDef : PyGetSetDef
{
  constexpr gdb_PyGetSetDef (const char *name_, getter get_, setter set_,
			     const char *doc_, void *closure_)
    : PyGetSetDef {const_cast<char *> (name_), get_, set_,
		   const_cast<char *> (doc_), closure_}
  {}

  /* Alternative constructor that allows omitting the closure in list
     initialization.  */
  constexpr gdb_PyGetSetDef (const char *name_, getter get_, setter set_,
			     const char *doc_)
    : gdb_PyGetSetDef {name_, get_, set_, doc_, NULL}
  {}

  /* Constructor for the sentinel entries.  */
  constexpr gdb_PyGetSetDef (std::nullptr_t)
    : gdb_PyGetSetDef {NULL, NULL, NULL, NULL, NULL}
  {}
};

/* The 'keywords' parameter of PyArg_ParseTupleAndKeywords has type
   'char **'.  However, string literals are const in C++, and so to
   avoid casting at every keyword array definition, we'll need to make
   the keywords array an array of 'const char *'.  To avoid having all
   callers add a 'const_cast<char **>' themselves when passing such an
   array through 'char **', we define our own version of
   PyArg_ParseTupleAndKeywords here with a corresponding 'keywords'
   parameter type that does the cast in a single place.  (This is not
   an overload of PyArg_ParseTupleAndKeywords in order to make it
   clearer that we're calling our own function instead of a function
   that exists in some newer Python version.)  */

static inline int
gdb_PyArg_ParseTupleAndKeywords (PyObject *args, PyObject *kw,
				 const char *format, const char **keywords, ...)
{
  va_list ap;
  int res;

  va_start (ap, keywords);
  res = PyArg_VaParseTupleAndKeywords (args, kw, format,
				       const_cast<char **> (keywords),
				       ap);
  va_end (ap);

  return res;
}

/* In order to be able to parse symtab_and_line_to_sal_object function
   a real symtab_and_line structure is needed.  */
#include "symtab.h"

/* Also needed to parse enum var_types. */
#include "command.h"
#include "breakpoint.h"

enum gdbpy_iter_kind { iter_keys, iter_values, iter_items };

struct block;
struct value;
struct language_defn;
struct program_space;
struct bpstat;
struct inferior;

extern int gdb_python_initialized;

extern PyObject *gdb_module;
extern PyObject *gdb_python_module;
extern PyTypeObject value_object_type
    CPYCHECKER_TYPE_OBJECT_FOR_TYPEDEF ("value_object");
extern PyTypeObject block_object_type
    CPYCHECKER_TYPE_OBJECT_FOR_TYPEDEF("block_object");
extern PyTypeObject symbol_object_type
    CPYCHECKER_TYPE_OBJECT_FOR_TYPEDEF ("symbol_object");
extern PyTypeObject event_object_type
    CPYCHECKER_TYPE_OBJECT_FOR_TYPEDEF ("event_object");
extern PyTypeObject breakpoint_object_type
    CPYCHECKER_TYPE_OBJECT_FOR_TYPEDEF ("breakpoint_object");
extern PyTypeObject frame_object_type
    CPYCHECKER_TYPE_OBJECT_FOR_TYPEDEF ("frame_object");
extern PyTypeObject thread_object_type
    CPYCHECKER_TYPE_OBJECT_FOR_TYPEDEF ("thread_object");

/* Ensure that breakpoint_object_type is initialized and return true.  If
   breakpoint_object_type can't be initialized then set a suitable Python
   error and return false.

   This function needs to be called from any gdbpy_initialize_* function
   that wants to reference breakpoint_object_type.  After all the
   gdbpy_initialize_* functions have been called then breakpoint_object_type
   is guaranteed to have been initialized, and this function does not need
   calling before referencing breakpoint_object_type.  */

extern bool gdbpy_breakpoint_init_breakpoint_type ();

struct gdbpy_breakpoint_object
{
  PyObject_HEAD

  /* The breakpoint number according to gdb.  */
  int number;

  /* The gdb breakpoint object, or NULL if the breakpoint has been
     deleted.  */
  struct breakpoint *bp;

  /* 1 is this is a FinishBreakpoint object, 0 otherwise.  */
  int is_finish_bp;
};

/* Require that BREAKPOINT be a valid breakpoint ID; throw a Python
   exception if it is invalid.  */
#define BPPY_REQUIRE_VALID(Breakpoint)                                  \
    do {                                                                \
      if ((Breakpoint)->bp == NULL)                                     \
	return PyErr_Format (PyExc_RuntimeError,                        \
			     _("Breakpoint %d is invalid."),            \
			     (Breakpoint)->number);                     \
    } while (0)

/* Require that BREAKPOINT be a valid breakpoint ID; throw a Python
   exception if it is invalid.  This macro is for use in setter functions.  */
#define BPPY_SET_REQUIRE_VALID(Breakpoint)                              \
    do {                                                                \
      if ((Breakpoint)->bp == NULL)                                     \
	{                                                               \
	  PyErr_Format (PyExc_RuntimeError, _("Breakpoint %d is invalid."), \
			(Breakpoint)->number);                          \
	  return -1;                                                    \
	}                                                               \
    } while (0)


/* Variables used to pass information between the Breakpoint
   constructor and the breakpoint-created hook function.  */
extern gdbpy_breakpoint_object *bppy_pending_object;


struct thread_object
{
  PyObject_HEAD

  /* The thread we represent.  */
  struct thread_info *thread;

  /* The Inferior object to which this thread belongs.  */
  PyObject *inf_obj;
};

struct inferior_object;

extern struct cmd_list_element *set_python_list;
extern struct cmd_list_element *show_python_list;

/* extension_language_script_ops "methods".  */

/* Return true if auto-loading Python scripts is enabled.
   This is the extension_language_script_ops.auto_load_enabled "method".  */

extern bool gdbpy_auto_load_enabled (const struct extension_language_defn *);

/* extension_language_ops "methods".  */

extern enum ext_lang_rc gdbpy_apply_val_pretty_printer
  (const struct extension_language_defn *,
   struct value *value,
   struct ui_file *stream, int recurse,
   const struct value_print_options *options,
   const struct language_defn *language);
extern enum ext_lang_bt_status gdbpy_apply_frame_filter
  (const struct extension_language_defn *,
   frame_info_ptr frame, frame_filter_flags flags,
   enum ext_lang_frame_args args_type,
   struct ui_out *out, int frame_low, int frame_high);
extern void gdbpy_preserve_values (const struct extension_language_defn *,
				   struct objfile *objfile,
				   htab_t copied_types);
extern enum ext_lang_bp_stop gdbpy_breakpoint_cond_says_stop
  (const struct extension_language_defn *, struct breakpoint *);
extern int gdbpy_breakpoint_has_cond (const struct extension_language_defn *,
				      struct breakpoint *b);

extern enum ext_lang_rc gdbpy_get_matching_xmethod_workers
  (const struct extension_language_defn *extlang,
   struct type *obj_type, const char *method_name,
   std::vector<xmethod_worker_up> *dm_vec);


PyObject *gdbpy_history (PyObject *self, PyObject *args);
PyObject *gdbpy_add_history (PyObject *self, PyObject *args);
extern PyObject *gdbpy_history_count (PyObject *self, PyObject *args);
PyObject *gdbpy_convenience_variable (PyObject *self, PyObject *args);
PyObject *gdbpy_set_convenience_variable (PyObject *self, PyObject *args);
PyObject *gdbpy_breakpoints (PyObject *, PyObject *);
PyObject *gdbpy_frame_stop_reason_string (PyObject *, PyObject *);
PyObject *gdbpy_lookup_symbol (PyObject *self, PyObject *args, PyObject *kw);
PyObject *gdbpy_lookup_global_symbol (PyObject *self, PyObject *args,
				      PyObject *kw);
PyObject *gdbpy_lookup_static_symbol (PyObject *self, PyObject *args,
				      PyObject *kw);
PyObject *gdbpy_lookup_static_symbols (PyObject *self, PyObject *args,
					   PyObject *kw);
PyObject *gdbpy_start_recording (PyObject *self, PyObject *args);
PyObject *gdbpy_current_recording (PyObject *self, PyObject *args);
PyObject *gdbpy_stop_recording (PyObject *self, PyObject *args);
PyObject *gdbpy_newest_frame (PyObject *self, PyObject *args);
PyObject *gdbpy_selected_frame (PyObject *self, PyObject *args);
PyObject *gdbpy_lookup_type (PyObject *self, PyObject *args, PyObject *kw);
int gdbpy_is_field (PyObject *obj);
PyObject *gdbpy_create_lazy_string_object (CORE_ADDR address, long length,
					   const char *encoding,
					   struct type *type);
PyObject *gdbpy_inferiors (PyObject *unused, PyObject *unused2);
PyObject *gdbpy_create_ptid_object (ptid_t ptid);
PyObject *gdbpy_selected_thread (PyObject *self, PyObject *args);
PyObject *gdbpy_selected_inferior (PyObject *self, PyObject *args);
PyObject *gdbpy_string_to_argv (PyObject *self, PyObject *args);
PyObject *gdbpy_parameter_value (const setting &var);
gdb::unique_xmalloc_ptr<char> gdbpy_parse_command_name
  (const char *name, struct cmd_list_element ***base_list,
   struct cmd_list_element **start_list);
PyObject *gdbpy_register_tui_window (PyObject *self, PyObject *args,
				     PyObject *kw);

PyObject *symtab_and_line_to_sal_object (struct symtab_and_line sal);
PyObject *symtab_to_symtab_object (struct symtab *symtab);
PyObject *symbol_to_symbol_object (struct symbol *sym);
PyObject *block_to_block_object (const struct block *block,
				 struct objfile *objfile);
PyObject *value_to_value_object (struct value *v);
PyObject *value_to_value_object_no_release (struct value *v);
PyObject *type_to_type_object (struct type *);
PyObject *frame_info_to_frame_object (frame_info_ptr frame);
PyObject *symtab_to_linetable_object (PyObject *symtab);
gdbpy_ref<> pspace_to_pspace_object (struct program_space *);
PyObject *pspy_get_printers (PyObject *, void *);
PyObject *pspy_get_frame_filters (PyObject *, void *);
PyObject *pspy_get_frame_unwinders (PyObject *, void *);
PyObject *pspy_get_xmethods (PyObject *, void *);

gdbpy_ref<> objfile_to_objfile_object (struct objfile *);
PyObject *objfpy_get_printers (PyObject *, void *);
PyObject *objfpy_get_frame_filters (PyObject *, void *);
PyObject *objfpy_get_frame_unwinders (PyObject *, void *);
PyObject *objfpy_get_xmethods (PyObject *, void *);
PyObject *gdbpy_lookup_objfile (PyObject *self, PyObject *args, PyObject *kw);

PyObject *gdbarch_to_arch_object (struct gdbarch *gdbarch);
PyObject *gdbpy_all_architecture_names (PyObject *self, PyObject *args);

PyObject *gdbpy_new_register_descriptor_iterator (struct gdbarch *gdbarch,
						  const char *group_name);
PyObject *gdbpy_new_reggroup_iterator (struct gdbarch *gdbarch);

gdbpy_ref<thread_object> create_thread_object (struct thread_info *tp);
gdbpy_ref<> thread_to_thread_object (thread_info *thr);;
gdbpy_ref<inferior_object> inferior_to_inferior_object (inferior *inf);

PyObject *gdbpy_buffer_to_membuf (gdb::unique_xmalloc_ptr<gdb_byte> buffer,
				  CORE_ADDR address, ULONGEST length);

struct process_stratum_target;
gdbpy_ref<> target_to_connection_object (process_stratum_target *target);
PyObject *gdbpy_connections (PyObject *self, PyObject *args);

const struct block *block_object_to_block (PyObject *obj);
struct symbol *symbol_object_to_symbol (PyObject *obj);
struct value *value_object_to_value (PyObject *self);
struct value *convert_value_from_python (PyObject *obj);
struct type *type_object_to_type (PyObject *obj);
struct symtab *symtab_object_to_symtab (PyObject *obj);
struct symtab_and_line *sal_object_to_symtab_and_line (PyObject *obj);
frame_info_ptr frame_object_to_frame_info (PyObject *frame_obj);
struct gdbarch *arch_object_to_gdbarch (PyObject *obj);

/* Convert Python object OBJ to a program_space pointer.  OBJ must be a
   gdb.Progspace reference.  Return nullptr if the gdb.Progspace is not
   valid (see gdb.Progspace.is_valid), otherwise return the program_space
   pointer.  */

extern struct program_space *progspace_object_to_program_space (PyObject *obj);

void gdbpy_initialize_gdb_readline (void);
int gdbpy_initialize_auto_load (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_values (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_frames (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_instruction (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_btrace (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_record (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_symtabs (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_commands (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_symbols (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_symtabs (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_blocks (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_types (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_functions (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_pspace (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_objfile (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_breakpoints (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_breakpoint_locations ()
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_finishbreakpoints (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_lazy_string (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_linetable (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_parameters (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_thread (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_inferior (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_eventregistry (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_event (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_arch (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_registers ()
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_xmethods (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_unwind (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_tui ()
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_membuf ()
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_connection ()
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
int gdbpy_initialize_micommands (void)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;
void gdbpy_finalize_micommands ();
int gdbpy_initialize_disasm ()
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;

PyMODINIT_FUNC gdbpy_events_mod_func ();

/* A wrapper for PyErr_Fetch that handles reference counting for the
   caller.  */
class gdbpy_err_fetch
{
public:

  gdbpy_err_fetch ()
  {
    PyObject *error_type, *error_value, *error_traceback;

    PyErr_Fetch (&error_type, &error_value, &error_traceback);
    m_error_type.reset (error_type);
    m_error_value.reset (error_value);
    m_error_traceback.reset (error_traceback);
  }

  /* Call PyErr_Restore using the values stashed in this object.
     After this call, this object is invalid and neither the to_string
     nor restore methods may be used again.  */

  void restore ()
  {
    PyErr_Restore (m_error_type.release (),
		   m_error_value.release (),
		   m_error_traceback.release ());
  }

  /* Return the string representation of the exception represented by
     this object.  If the result is NULL a python error occurred, the
     caller must clear it.  */

  gdb::unique_xmalloc_ptr<char> to_string () const;

  /* Return the string representation of the type of the exception
     represented by this object.  If the result is NULL a python error
     occurred, the caller must clear it.  */

  gdb::unique_xmalloc_ptr<char> type_to_string () const;

  /* Return true if the stored type matches TYPE, false otherwise.  */

  bool type_matches (PyObject *type) const
  {
    return PyErr_GivenExceptionMatches (m_error_type.get (), type);
  }

  /* Return a new reference to the exception value object.  */

  gdbpy_ref<> value ()
  {
    return m_error_value;
  }

private:

  gdbpy_ref<> m_error_type, m_error_value, m_error_traceback;
};

/* Called before entering the Python interpreter to install the
   current language and architecture to be used for Python values.
   Also set the active extension language for GDB so that SIGINT's
   are directed our way, and if necessary install the right SIGINT
   handler.  */
class gdbpy_enter
{
 public:

  /* Set the ambient Python architecture to GDBARCH and the language
     to LANGUAGE.  If GDBARCH is nullptr, then the architecture will
     be computed, when needed, using get_current_arch; see the
     get_gdbarch method.  If LANGUAGE is not nullptr, then the current
     language at time of construction will be saved (to be restored on
     destruction), and the current language will be set to
     LANGUAGE.  */
  explicit gdbpy_enter (struct gdbarch *gdbarch = nullptr,
			const struct language_defn *language = nullptr);

  ~gdbpy_enter ();

  DISABLE_COPY_AND_ASSIGN (gdbpy_enter);

  /* Return the current gdbarch, as known to the Python layer.  This
     is either python_gdbarch (which comes from the most recent call
     to the gdbpy_enter constructor), or, if that is nullptr, the
     result of get_current_arch.  */
  static struct gdbarch *get_gdbarch ();

  /* Called only during gdb shutdown.  This sets python_gdbarch to an
     acceptable value.  */
  static void finalize ();

 private:

  /* The current gdbarch, according to Python.  This can be
     nullptr.  */
  static struct gdbarch *python_gdbarch;

  struct active_ext_lang_state *m_previous_active;
  PyGILState_STATE m_state;
  struct gdbarch *m_gdbarch;
  const struct language_defn *m_language;

  /* An optional is used here because we don't want to call
     PyErr_Fetch too early.  */
  gdb::optional<gdbpy_err_fetch> m_error;
};

/* Like gdbpy_enter, but takes a varobj.  This is a subclass just to
   make constructor delegation a little nicer.  */
class gdbpy_enter_varobj : public gdbpy_enter
{
 public:

  /* This is defined in varobj.c, where it can access varobj
     internals.  */
  gdbpy_enter_varobj (const struct varobj *var);

};

/* The opposite of gdb_enter: this releases the GIL around a region,
   allowing other Python threads to run.  No Python APIs may be used
   while this is active.  */
class gdbpy_allow_threads
{
public:

  gdbpy_allow_threads ()
    : m_save (PyEval_SaveThread ())
  {
    gdb_assert (m_save != nullptr);
  }

  ~gdbpy_allow_threads ()
  {
    PyEval_RestoreThread (m_save);
  }

  DISABLE_COPY_AND_ASSIGN (gdbpy_allow_threads);

private:

  PyThreadState *m_save;
};

/* Use this after a TRY_EXCEPT to throw the appropriate Python
   exception.  */
#define GDB_PY_HANDLE_EXCEPTION(Exception)	\
  do {						\
    if (Exception.reason < 0)			\
      {						\
	gdbpy_convert_exception (Exception);	\
	return NULL;				\
      }						\
  } while (0)

/* Use this after a TRY_EXCEPT to throw the appropriate Python
   exception.  This macro is for use inside setter functions.  */
#define GDB_PY_SET_HANDLE_EXCEPTION(Exception)				\
    do {								\
      if (Exception.reason < 0)						\
	{								\
	  gdbpy_convert_exception (Exception);				\
	  return -1;							\
	}								\
    } while (0)

int gdbpy_print_python_errors_p (void);
void gdbpy_print_stack (void);
void gdbpy_print_stack_or_quit ();
void gdbpy_handle_exception () ATTRIBUTE_NORETURN;

/* A wrapper around calling 'error'.  Prefixes the error message with an
   'Error occurred in Python' string.  Use this in C++ code if we spot
   something wrong with an object returned from Python code.  The prefix
   string gives the user a hint that the mistake is within Python code,
   rather than some other part of GDB.

   This always calls error, and never returns.  */

void gdbpy_error (const char *fmt, ...)
  ATTRIBUTE_NORETURN ATTRIBUTE_PRINTF (1, 2);

gdbpy_ref<> python_string_to_unicode (PyObject *obj);
gdb::unique_xmalloc_ptr<char> unicode_to_target_string (PyObject *unicode_str);
gdb::unique_xmalloc_ptr<char> python_string_to_target_string (PyObject *obj);
gdbpy_ref<> python_string_to_target_python_string (PyObject *obj);
gdb::unique_xmalloc_ptr<char> python_string_to_host_string (PyObject *obj);
gdbpy_ref<> host_string_to_python_string (const char *str);
int gdbpy_is_string (PyObject *obj);
gdb::unique_xmalloc_ptr<char> gdbpy_obj_to_string (PyObject *obj);

int gdbpy_is_lazy_string (PyObject *result);
void gdbpy_extract_lazy_string (PyObject *string, CORE_ADDR *addr,
				struct type **str_type,
				long *length,
				gdb::unique_xmalloc_ptr<char> *encoding);

int gdbpy_is_value_object (PyObject *obj);

/* Note that these are declared here, and not in python.h with the
   other pretty-printer functions, because they refer to PyObject.  */
gdbpy_ref<> apply_varobj_pretty_printer (PyObject *print_obj,
					 struct value **replacement,
					 struct ui_file *stream,
					 const value_print_options *opts);
gdbpy_ref<> gdbpy_get_varobj_pretty_printer (struct value *value);
gdb::unique_xmalloc_ptr<char> gdbpy_get_display_hint (PyObject *printer);
PyObject *gdbpy_default_visualizer (PyObject *self, PyObject *args);

PyObject *gdbpy_print_options (PyObject *self, PyObject *args);
void gdbpy_get_print_options (value_print_options *opts);
extern const struct value_print_options *gdbpy_current_print_options;

void bpfinishpy_pre_stop_hook (struct gdbpy_breakpoint_object *bp_obj);
void bpfinishpy_post_stop_hook (struct gdbpy_breakpoint_object *bp_obj);

extern PyObject *gdbpy_doc_cst;
extern PyObject *gdbpy_children_cst;
extern PyObject *gdbpy_to_string_cst;
extern PyObject *gdbpy_display_hint_cst;
extern PyObject *gdbpy_enabled_cst;
extern PyObject *gdbpy_value_cst;

/* Exception types.  */
extern PyObject *gdbpy_gdb_error;
extern PyObject *gdbpy_gdb_memory_error;
extern PyObject *gdbpy_gdberror_exc;

extern void gdbpy_convert_exception (const struct gdb_exception &)
    CPYCHECKER_SETS_EXCEPTION;

int get_addr_from_python (PyObject *obj, CORE_ADDR *addr)
    CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;

gdbpy_ref<> gdb_py_object_from_longest (LONGEST l);
gdbpy_ref<> gdb_py_object_from_ulongest (ULONGEST l);
int gdb_py_int_as_long (PyObject *, long *);

PyObject *gdb_py_generic_dict (PyObject *self, void *closure);

int gdb_pymodule_addobject (PyObject *module, const char *name,
			    PyObject *object)
  CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION;

struct varobj_iter;
struct varobj;
std::unique_ptr<varobj_iter> py_varobj_get_iterator
     (struct varobj *var,
      PyObject *printer,
      const value_print_options *opts);

/* Deleter for Py_buffer unique_ptr specialization.  */

struct Py_buffer_deleter
{
  void operator() (Py_buffer *b) const
  {
    PyBuffer_Release (b);
  }
};

/* A unique_ptr specialization for Py_buffer.  */
typedef std::unique_ptr<Py_buffer, Py_buffer_deleter> Py_buffer_up;

/* Parse a register number from PYO_REG_ID and place the register number
   into *REG_NUM.  The register is a register for GDBARCH.

   If a register is parsed successfully then *REG_NUM will have been
   updated, and true is returned.  Otherwise the contents of *REG_NUM are
   undefined, and false is returned.  When false is returned, the
   Python error is set.

   The PYO_REG_ID object can be a string, the name of the register.  This
   is the slowest approach as GDB has to map the name to a number for each
   call.  Alternatively PYO_REG_ID can be an internal GDB register
   number.  This is quick but should not be encouraged as this means
   Python scripts are now dependent on GDB's internal register numbering.
   Final PYO_REG_ID can be a gdb.RegisterDescriptor object, these objects
   can be looked up by name once, and then cache the register number so
   should be as quick as using a register number.  */

extern bool gdbpy_parse_register_id (struct gdbarch *gdbarch,
				     PyObject *pyo_reg_id, int *reg_num);

/* Return true if OBJ is a gdb.Architecture object, otherwise, return
   false.  */

extern bool gdbpy_is_architecture (PyObject *obj);

/* Return true if OBJ is a gdb.Progspace object, otherwise, return false.  */

extern bool gdbpy_is_progspace (PyObject *obj);

/* Take DOC, the documentation string for a GDB command defined in Python,
   and return an (possibly) modified version of that same string.

   When a command is defined in Python, the documentation string will
   usually be indented based on the indentation of the surrounding Python
   code.  However, the documentation string is a literal string, all the
   white-space added for indentation is included within the documentation
   string.

   This indentation is then included in the help text that GDB displays,
   which looks odd out of the context of the original Python source code.

   This function analyses DOC and tries to figure out what white-space
   within DOC was added as part of the indentation, and then removes that
   white-space from the copy that is returned.

   If the analysis of DOC fails then DOC will be returned unmodified.  */

extern gdb::unique_xmalloc_ptr<char> gdbpy_fix_doc_string_indentation
  (gdb::unique_xmalloc_ptr<char> doc);

/* Implement the 'print_insn' hook for Python.  Disassemble an instruction
   whose address is ADDRESS for architecture GDBARCH.  The bytes of the
   instruction should be read with INFO->read_memory_func as the
   instruction being disassembled might actually be in a buffer.

   Used INFO->fprintf_func to print the results of the disassembly, and
   return the length of the instruction in octets.

   If no instruction can be disassembled then return an empty value.  */

extern gdb::optional<int> gdbpy_print_insn (struct gdbarch *gdbarch,
					    CORE_ADDR address,
					    disassemble_info *info);

#endif /* PYTHON_PYTHON_INTERNAL_H */
