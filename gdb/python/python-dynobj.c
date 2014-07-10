/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2014 NVIDIA Corporation
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
#include <dlfcn.h>

#if HAVE_LIBPYTHON2_4
#define CONSTCHAR char
#else
#define CONSTCHAR const char
#endif

#define STRINGIFY2(name) #name
#define STRINGIFY(name) STRINGIFY2(name)

bool python_initialized = false;
static void *libpython_handle = NULL;

/* Dynamic references to constants */
PyObject *gdbpy_None = NULL;
PyObject *gdbpy_True = NULL;
PyObject *gdbpy_Zero = NULL;
PyObject *gdbpy_NotImplemented = NULL;
PyTypeObject *gdbpy_FloatType = NULL;
PyTypeObject *gdbpy_BoolType = NULL;
PyTypeObject *gdbpy_IntType = NULL;
PyTypeObject *gdbpy_LongType = NULL;
PyTypeObject *gdbpy_StringType = NULL;
PyTypeObject *gdbpy_ListType = NULL;
PyTypeObject *gdbpy_TupleType = NULL;
PyTypeObject *gdbpy_UnicodeType = NULL;

/* Dynamic reference to exception types */
PyObject **pgdbpyExc_AttributeError = NULL;
PyObject **pgdbpyExc_IOError = NULL;
PyObject **pgdbpyExc_KeyError = NULL;
PyObject **pgdbpyExc_KeyboardInterrupt  = NULL;
PyObject **pgdbpyExc_MemoryError = NULL;
PyObject **pgdbpyExc_NotImplementedError  = NULL;
PyObject **pgdbpyExc_OverflowError  = NULL;
PyObject **pgdbpyExc_RuntimeError = NULL;
PyObject **pgdbpyExc_StopIteration = NULL;
PyObject **pgdbpyExc_SystemError = NULL;
PyObject **pgdbpyExc_TypeError = NULL;
PyObject **pgdbpyExc_ValueError = NULL;

PyThreadState **pgdbpy_OSReadlineTState = NULL;
char * (**pgdbpyOS_ReadlineFunctionPointer) (FILE *, FILE *, char *) = NULL;


/* Imported functions */
int (*gdbpy_Arg_UnpackTuple) (PyObject *, const char *, Py_ssize_t, Py_ssize_t, ...) = NULL;
PyObject * (*gdbpy_ErrFormat)(PyObject *, const char *, ...) = NULL;
PyObject * (*gdbpy_BuildValue) (const char *, ...) = NULL;
PyObject * (*gdbpy_ObjectCallFunctionObjArgs) (PyObject *,...) = NULL;
PyObject * (*gdbpy_ObjectCallMethodObjArgs) (PyObject *, PyObject *,...) = NULL;
PyObject * (*gdbpy_ObjectCallMethod)(PyObject *o, char *m, char *format, ...) = NULL;
int (*gdbpy_ArgParseTuple) (PyObject *obj, const char *, ...) = NULL;
int (*gdbpy_ArgParseTupleAndKeywords) (PyObject *obj, PyObject *, const char *, char **, ...) = NULL;
PyObject * (*gdbpy_StringFromFormat) (const char *, ...) = NULL;


static PyObject * (*gdb_PyBool_FromLong) (long) = NULL;
static PyObject * (*gdb_PyBuffer_FromReadWriteObject) (PyObject *base, Py_ssize_t offset, Py_ssize_t size) = NULL;
static int (*gdb_PyCallable_Check) (PyObject *o) = NULL;
static PyObject * (*gdb_PyDict_New) (void) = NULL;
static int (*gdb_PyDict_SetItemString) (PyObject *dp, const char *key, PyObject *item) = NULL;
static void (*gdb_PyErr_Clear) (void) = NULL;
static int (*gdb_PyErr_ExceptionMatches) (PyObject *) = NULL;
static void (*gdb_PyErr_Fetch) (PyObject **, PyObject **, PyObject **) = NULL;
static int (*gdb_PyErr_GivenExceptionMatches) (PyObject *, PyObject *) = NULL;
static PyObject * (*gdb_PyErr_Occurred) (void) = NULL;
static void (*gdb_PyErr_Print) (void) = NULL;
static void (*gdb_PyErr_Restore) (PyObject *, PyObject *, PyObject *) = NULL;
static PyObject * (*gdb_PyErr_SetFromErrno) (PyObject *) = NULL;
static void (*gdb_PyErr_SetInterrupt) (void) = NULL;
static void (*gdb_PyErr_SetObject) (PyObject *, PyObject *) = NULL;
static void (*gdb_PyErr_SetString) (PyObject *, const char *) = NULL;
static PyObject * (*gdb_PyErr_NewException)(char *name, PyObject *base, PyObject *dict) = NULL;
static void (*gdb_PyEval_InitThreads) (void) = NULL;
static void (*gdb_PyEval_ReleaseLock) (void) = NULL;
static void (*gdb_PyEval_RestoreThread) (PyThreadState *) = NULL;
static PyThreadState * (*gdb_PyEval_SaveThread) (void) = NULL;
static double (*gdb_PyFloat_AsDouble) (PyObject *) = NULL;
static PyObject * (*gdb_PyFloat_FromDouble) (double) = NULL;
static PyFrameObject * (*gdb_PyFrame_New)(PyThreadState *, PyCodeObject *, PyObject *, PyObject *) = NULL;
static PyGILState_STATE (*gdb_PyGILState_Ensure) (void) = NULL;
static void (*gdb_PyGILState_Release) (PyGILState_STATE) = NULL;
static PyObject * (*gdb_PyImport_AddModule) (const char *name) = NULL;
static PyObject * (*gdb_PyImport_ImportModule) (const char *name) = NULL;
static long (*gdb_PyInt_AsLong) (PyObject *) = NULL;
static PyObject * (*gdb_PyInt_FromLong) (long) = NULL;
static long (*gdb_PyInt_GetMax) (void) = NULL;
static PyObject * (*gdb_PyIter_Next) (PyObject *) = NULL;
static int (*gdb_PyList_Append) (PyObject *, PyObject *) = NULL;
static PyObject * (*gdb_PyList_AsTuple) (PyObject *) = NULL;
static PyObject * (*gdb_PyList_GetItem) (PyObject *, Py_ssize_t) = NULL;
static int (*gdb_PyList_Insert) (PyObject *, Py_ssize_t, PyObject *) = NULL;
static PyObject * (*gdb_PyList_New) (Py_ssize_t size) = NULL;
static Py_ssize_t (*gdb_PyList_Size) (PyObject *) = NULL;
static PY_LONG_LONG (*gdb_PyLong_AsLongLong) (PyObject *) = NULL;
static unsigned PY_LONG_LONG (*gdb_PyLong_AsUnsignedLongLong) (PyObject *) = NULL;
static PyObject * (*gdb_PyLong_FromLong) (long) = NULL;
static PyObject * (*gdb_PyLong_FromLongLong) (PY_LONG_LONG) = NULL;
static PyObject * (*gdb_PyLong_FromUnsignedLong) (unsigned long) = NULL;
static PyObject * (*gdb_PyLong_FromUnsignedLongLong) (unsigned PY_LONG_LONG) = NULL;
static void * (*gdb_PyMem_Malloc) (size_t) = NULL;
static int (*gdb_PyModule_AddIntConstant) (PyObject *, const char *, long) = NULL;
static int (*gdb_PyModule_AddObject) (PyObject *, const char *, PyObject *) = NULL;
static int (*gdb_PyModule_AddStringConstant) (PyObject *, const char *, const char *) = NULL;
static PyObject * (*gdb_PyModule_GetDict) (PyObject *) = NULL;
static PyObject * (*gdb_PyNumber_Long) (PyObject *o) = NULL;
static int (*gdb_PyOS_InterruptOccurred) (void) = NULL;
static int (*gdb_PyObject_AsReadBuffer) (PyObject *obj, const void **, Py_ssize_t *) = NULL;
static int (*gdb_PyObject_CheckReadBuffer) (PyObject *obj) = NULL;
static PyObject * (*gdb_PyObject_GenericGetAttr) (PyObject *, PyObject *) = NULL;
static int (*gdb_PyObject_GenericSetAttr)(PyObject *arg1, PyObject *arg2, PyObject *arg3) = NULL;
static PyObject * (*gdb_PyObject_GetAttr) (PyObject *, PyObject *) = NULL;
static PyObject * (*gdb_PyObject_GetAttrString) (PyObject *, const char *) = NULL;
static PyObject * (*gdb_PyObject_GetIter) (PyObject *) = NULL;
static int (*gdb_PyObject_HasAttr) (PyObject *, PyObject *) = NULL;
static int (*gdb_PyObject_HasAttrString) (PyObject *, const char *) = NULL;
static int (*gdb_PyObject_IsTrue) (PyObject *) = NULL;
static int (*gdb_PyObject_RichCompareBool) (PyObject *, PyObject *, int) = NULL;
static int (*gdb_PyObject_SetAttrString) (PyObject *, const char *, PyObject *) = NULL;
static PyObject * (*gdb_PyObject_Str) (PyObject *) = NULL;
static int (*gdb_PyRun_InteractiveLoopFlags) (FILE *, const char *, PyCompilerFlags *) = NULL;
static int (*gdb_PyRun_SimpleFileExFlags) (FILE *, const char *, int, PyCompilerFlags *) = NULL;
static int (*gdb_PyRun_SimpleStringFlags) (const char *, PyCompilerFlags *) = NULL;
static PyObject * (*gdb_PyRun_StringFlags)(const char *, int, PyObject *, PyObject *, PyCompilerFlags *) = NULL;
static int (*gdb_PySequence_Check) (PyObject *o) = NULL;
static int (*gdb_PySequence_DelItem) (PyObject *o, Py_ssize_t i) = NULL;
static PyObject * (*gdb_PySequence_GetItem) (PyObject *o, Py_ssize_t i) = NULL;
static Py_ssize_t (*gdb_PySequence_Index) (PyObject *o, PyObject *value) = NULL;
static PyObject * (*gdb_PySequence_List) (PyObject *o) = NULL;
static Py_ssize_t (*gdb_PySequence_Size) (PyObject *o) = NULL;
static char * (*gdb_PyString_AsString) (PyObject *) = NULL;
static PyObject * (*gdb_PyString_Decode) (const char *, Py_ssize_t, const char *, const char *) = NULL;
static PyObject * (*gdb_PyString_FromString) (const char *) = NULL;
static Py_ssize_t (*gdb_PyString_Size) (PyObject *) = NULL;
static PyObject * (*gdb_PySys_GetObject) (char *) = NULL;
static void (*gdb_PySys_SetPath) (char *) = NULL;
static PyThreadState * (*gdb_PyThreadState_Get) (void) = NULL;
static PyThreadState * (*gdb_PyThreadState_Swap) (PyThreadState *) = NULL;
static PyObject * (*gdb_PyTuple_GetItem) (PyObject *, Py_ssize_t) = NULL;
static PyObject * (*gdb_PyTuple_New) (Py_ssize_t size) = NULL;
static int (*gdb_PyTuple_SetItem) (PyObject *, Py_ssize_t, PyObject *) = NULL;
static Py_ssize_t (*gdb_PyTuple_Size) (PyObject *) = NULL;
static PyObject * (*gdb_PyType_GenericNew)(PyTypeObject *, PyObject *, PyObject *) = NULL;
static int (*gdb_PyType_IsSubtype) (PyTypeObject *, PyTypeObject *) = NULL;
static int (*gdb_PyType_Ready) (PyTypeObject *) = NULL;
static void (*gdb_Py_Finalize) (void) = NULL;
static int (*gdb_Py_FlushLine) (void) = NULL;
static void (*gdb_Py_Initialize) (void) = NULL;
static PyObject * (*gdb_Py_InitModule4)(const char *, PyMethodDef *, const char *, PyObject *, int) = NULL;
static PyObject * (*gdb_Py_InitModule4_64)(const char *, PyMethodDef *, const char *, PyObject *, int) = NULL;
static void (*gdb_Py_SetProgramName) (char *) = NULL;
static PyObject * (*gdb__PyObject_New) (PyTypeObject *) = NULL;
static PyCodeObject * (*gdb_PyCode_New) (int, int, int, int,
           PyObject *, PyObject *, PyObject *, PyObject *,
           PyObject *, PyObject *, PyObject *, PyObject *, int, PyObject *) = NULL;
static PyObject * (*gdb_PyObject_CallObject) (PyObject *callable_object, PyObject *args) = NULL;
static PyObject * (*gdb_PyObject_Call)(PyObject *callable_object, PyObject *args, PyObject *kw) = NULL;
static PyObject* (*gdb_PyUnicode_Decode)(const char *s, Py_ssize_t size, const char *encoding, const char *errors) = NULL;
static PyObject* (*gdb_PyUnicode_AsEncodedString)(register PyObject *unicode, const char *encoding, const char *errors) = NULL;
static PyObject* (*gdb_PyUnicode_FromEncodedObject)(register PyObject *obj, const char *encoding, const char *errors) = NULL;



bool
is_python_available (void) {

  int i;
  static const char *libpython_names[] = {
#if HAVE_LIBPYTHON2_4
                     "libpython2.4.so.1.0", "libpython2.4.so.1",
#elif !defined(__APPLE__)
                     "libpython2.7.so.1", "libpython2.6.so.1.0",
                     "libpython2.6.so.1", "libpython2.5.so.1.0",
                     "libpython2.5.so.1",
#else
                     "libpython2.7.dylib",
                     "Python.framework/Versions/2.7/Python",
                     "/System/Library/Frameworks/Python.framework/Versions/2.7/Python",
#endif
                     NULL };

  if (python_initialized)
    return libpython_handle != NULL;

  for ( i = 0; libpython_names[i] && !libpython_handle ; ++i)
    libpython_handle = dlopen ( libpython_names[i], RTLD_NOW | RTLD_GLOBAL);

  python_initialized = true;
  if (!libpython_handle)
    return false;

#define RESOLVE_AND_CHECK(varname,symname)            \
  varname = dlsym (libpython_handle, symname);        \
  if (!varname)                                       \
    {                                                 \
      fprintf (stderr, "Symbol %s could not be found" \
               " in python library!\n", symname);     \
      goto err_out;                                   \
    }
  /* Resolve types and exceptions */
  RESOLVE_AND_CHECK(gdbpy_None,                    "_Py_NoneStruct");
  RESOLVE_AND_CHECK(gdbpy_True,                    "_Py_TrueStruct");
  RESOLVE_AND_CHECK(gdbpy_Zero,                    "_Py_ZeroStruct");
  RESOLVE_AND_CHECK(gdbpy_FloatType,               "PyFloat_Type");
  RESOLVE_AND_CHECK(gdbpy_BoolType,                "PyBool_Type");
  RESOLVE_AND_CHECK(gdbpy_IntType,                 "PyInt_Type");
  RESOLVE_AND_CHECK(gdbpy_LongType,                "PyLong_Type");
  RESOLVE_AND_CHECK(gdbpy_StringType,              "PyString_Type");
  RESOLVE_AND_CHECK(gdbpy_ListType,                "PyList_Type");
  RESOLVE_AND_CHECK(gdbpy_TupleType,               "PyTuple_Type");
  RESOLVE_AND_CHECK(gdbpy_UnicodeType,             "PyUnicode_Type");
  RESOLVE_AND_CHECK(gdbpy_NotImplemented,          "_Py_NotImplementedStruct");

  RESOLVE_AND_CHECK(pgdbpyExc_AttributeError,      "PyExc_AttributeError");
  RESOLVE_AND_CHECK(pgdbpyExc_IOError,             "PyExc_IOError");
  RESOLVE_AND_CHECK(pgdbpyExc_KeyError,            "PyExc_KeyError");
  RESOLVE_AND_CHECK(pgdbpyExc_KeyboardInterrupt,   "PyExc_KeyboardInterrupt");
  RESOLVE_AND_CHECK(pgdbpyExc_MemoryError,         "PyExc_MemoryError");
  RESOLVE_AND_CHECK(pgdbpyExc_NotImplementedError, "PyExc_NotImplementedError");
  RESOLVE_AND_CHECK(pgdbpyExc_OverflowError,       "PyExc_OverflowError");
  RESOLVE_AND_CHECK(pgdbpyExc_RuntimeError,        "PyExc_RuntimeError");
  RESOLVE_AND_CHECK(pgdbpyExc_StopIteration,       "PyExc_StopIteration");
  RESOLVE_AND_CHECK(pgdbpyExc_SystemError,         "PyExc_SystemError");
  RESOLVE_AND_CHECK(pgdbpyExc_TypeError,           "PyExc_TypeError");
  RESOLVE_AND_CHECK(pgdbpyExc_ValueError,          "PyExc_ValueError");
  RESOLVE_AND_CHECK(pgdbpy_OSReadlineTState,           "_PyOS_ReadlineTState");
  RESOLVE_AND_CHECK(pgdbpyOS_ReadlineFunctionPointer, "PyOS_ReadlineFunctionPointer");

  /* Resolve variadic functions */
  RESOLVE_AND_CHECK(gdbpy_Arg_UnpackTuple,           "PyArg_UnpackTuple");
  RESOLVE_AND_CHECK(gdbpy_ErrFormat,                 "PyErr_Format");
  RESOLVE_AND_CHECK(gdbpy_BuildValue,                STRINGIFY(Py_BuildValue));
  RESOLVE_AND_CHECK(gdbpy_ObjectCallFunctionObjArgs, "PyObject_CallFunctionObjArgs");
  RESOLVE_AND_CHECK(gdbpy_ObjectCallMethodObjArgs,   "PyObject_CallMethodObjArgs");
  RESOLVE_AND_CHECK(gdbpy_ObjectCallMethod,          STRINGIFY(PyObject_CallMethod));
  RESOLVE_AND_CHECK(gdbpy_ArgParseTuple,             STRINGIFY(PyArg_ParseTuple));
  RESOLVE_AND_CHECK(gdbpy_ArgParseTupleAndKeywords,  STRINGIFY(PyArg_ParseTupleAndKeywords));
  RESOLVE_AND_CHECK(gdbpy_StringFromFormat,          "PyString_FromFormat");

  /* Resolve indirectly called functions */
  gdb_PyBool_FromLong = dlsym (libpython_handle, "PyBool_FromLong");
  gdb_PyBuffer_FromReadWriteObject = dlsym (libpython_handle, "PyBuffer_FromReadWriteObject");
  gdb_PyCallable_Check = dlsym (libpython_handle, "PyCallable_Check");
  gdb_PyDict_New = dlsym (libpython_handle, "PyDict_New");
  gdb_PyDict_SetItemString = dlsym (libpython_handle, "PyDict_SetItemString");
  gdb_PyErr_Clear = dlsym (libpython_handle, "PyErr_Clear");
  gdb_PyErr_ExceptionMatches = dlsym (libpython_handle, "PyErr_ExceptionMatches");
  gdb_PyErr_Fetch = dlsym (libpython_handle, "PyErr_Fetch");
  gdb_PyErr_GivenExceptionMatches = dlsym (libpython_handle, "PyErr_GivenExceptionMatch");
  gdb_PyErr_Occurred = dlsym (libpython_handle, "PyErr_Occurred");
  gdb_PyErr_Print = dlsym (libpython_handle, "PyErr_Print");
  gdb_PyErr_Restore = dlsym (libpython_handle, "PyErr_Restore");
  gdb_PyErr_SetFromErrno = dlsym (libpython_handle, "PyErr_SetFromErrno");
  gdb_PyErr_SetInterrupt = dlsym (libpython_handle, "PyErr_SetInterrupt");
  gdb_PyErr_SetObject = dlsym (libpython_handle, "PyErr_SetObject");
  gdb_PyErr_SetString = dlsym (libpython_handle, "PyErr_SetString");
  gdb_PyErr_NewException = dlsym (libpython_handle, "PyErr_NewException");
  gdb_PyEval_InitThreads = dlsym (libpython_handle, "PyEval_InitThreads");
  gdb_PyEval_ReleaseLock = dlsym (libpython_handle, "PyEval_ReleaseLock");
  gdb_PyEval_RestoreThread = dlsym (libpython_handle, "PyEval_RestoreThread");
  gdb_PyEval_SaveThread = dlsym (libpython_handle, "PyEval_SaveThread");
  gdb_PyFloat_AsDouble = dlsym (libpython_handle, "PyFloat_AsDouble");
  gdb_PyFloat_FromDouble = dlsym (libpython_handle, "PyFloat_FromDouble");
  gdb_PyGILState_Ensure = dlsym (libpython_handle, "PyGILState_Ensure");
  gdb_PyGILState_Release = dlsym (libpython_handle, "PyGILState_Release");
  gdb_PyImport_AddModule = dlsym (libpython_handle, "PyImport_AddModule");
  gdb_PyImport_ImportModule = dlsym (libpython_handle, "PyImport_ImportModule");
  gdb_PyInt_AsLong = dlsym (libpython_handle, "PyInt_AsLong");
  gdb_PyInt_FromLong = dlsym (libpython_handle, "PyInt_FromLong");
  gdb_PyInt_GetMax = dlsym (libpython_handle, "PyInt_GetMax");
  gdb_PyIter_Next = dlsym (libpython_handle, "PyIter_Next");
  gdb_PyList_Append = dlsym (libpython_handle, "PyList_Append");
  gdb_PyList_AsTuple = dlsym (libpython_handle, "PyList_AsTuple");
  gdb_PyList_GetItem = dlsym (libpython_handle, "PyList_GetItem");
  gdb_PyList_Insert = dlsym (libpython_handle, "PyList_Insert");
  gdb_PyList_New = dlsym (libpython_handle, "PyList_New");
  gdb_PyList_Size = dlsym (libpython_handle, "PyList_Size");
  gdb_PyLong_AsLongLong = dlsym (libpython_handle, "PyLong_AsLongLong");
  gdb_PyLong_AsUnsignedLongLong = dlsym (libpython_handle, "PyLong_AsUnsignedLongLong");
  gdb_PyLong_FromLong = dlsym (libpython_handle, "PyLong_FromLong");
  gdb_PyLong_FromLongLong = dlsym (libpython_handle, "PyLong_FromLongLong");
  gdb_PyLong_FromUnsignedLong = dlsym (libpython_handle, "PyLong_FromUnsignedLong");
  gdb_PyLong_FromUnsignedLongLong = dlsym (libpython_handle, "PyLong_FromUnsignedLongLong");
  gdb_PyMem_Malloc = dlsym (libpython_handle, "PyMem_Malloc");
  gdb_PyModule_AddIntConstant = dlsym (libpython_handle, "PyModule_AddIntConstant");
  gdb_PyModule_AddObject = dlsym (libpython_handle, "PyModule_AddObject");
  gdb_PyModule_AddStringConstant = dlsym (libpython_handle, "PyModule_AddStringConstant");
  gdb_PyModule_GetDict = dlsym (libpython_handle, "PyModule_GetDict");
  gdb_PyNumber_Long = dlsym (libpython_handle, "PyNumber_Long");
  gdb_PyOS_InterruptOccurred = dlsym (libpython_handle, "PyOS_InterruptOccurred");
  gdb_PyObject_AsReadBuffer = dlsym (libpython_handle, "PyObject_AsReadBuffer");
  gdb_PyObject_CheckReadBuffer = dlsym (libpython_handle, "PyObject_CheckReadBuffer");
  gdb_PyObject_GenericGetAttr = dlsym (libpython_handle, "PyObject_GenericGetAttr");
  gdb_PyObject_GenericSetAttr = dlsym (libpython_handle, "PyObject_GenericSetAttr");
  gdb_PyObject_GetAttr = dlsym (libpython_handle, "PyObject_GetAttr");
  gdb_PyObject_GetAttrString = dlsym (libpython_handle, "PyObject_GetAttrString");
  gdb_PyObject_GetIter = dlsym (libpython_handle, "PyObject_GetIter");
  gdb_PyObject_HasAttr = dlsym (libpython_handle, "PyObject_HasAttr");
  gdb_PyObject_HasAttrString = dlsym (libpython_handle, "PyObject_HasAttrString");
  gdb_PyObject_IsTrue = dlsym (libpython_handle, "PyObject_IsTrue");
  gdb_PyObject_RichCompareBool = dlsym (libpython_handle, "PyObject_RichCompareBool");
  gdb_PyObject_SetAttrString = dlsym (libpython_handle, "PyObject_SetAttrString");
  gdb_PyObject_Str = dlsym (libpython_handle, "PyObject_Str");
  gdb_PyRun_InteractiveLoopFlags = dlsym (libpython_handle, "PyRun_InteractiveLoopFlags");
  gdb_PyRun_StringFlags = dlsym (libpython_handle, "PyRun_StringFlags");
  gdb_PyRun_SimpleFileExFlags = dlsym (libpython_handle, "PyRun_SimpleFileExFlags");
  gdb_PyRun_SimpleStringFlags = dlsym (libpython_handle, "PyRun_SimpleStringFlags");
  gdb_PySequence_Check = dlsym (libpython_handle, "PySequence_Check");
  gdb_PySequence_DelItem = dlsym (libpython_handle, "PySequence_DelItem");
  gdb_PySequence_GetItem = dlsym (libpython_handle, "PySequence_GetItem");
  gdb_PySequence_Index = dlsym (libpython_handle, "PySequence_Index");
  gdb_PySequence_List = dlsym (libpython_handle, "PySequence_List");
  gdb_PySequence_Size = dlsym (libpython_handle, "PySequence_Size");
  gdb_PyString_AsString = dlsym (libpython_handle, "PyString_AsString");
  gdb_PyString_Decode = dlsym (libpython_handle, "PyString_Decode");
  gdb_PyString_FromString = dlsym (libpython_handle, "PyString_FromString");
  gdb_PyString_Size = dlsym (libpython_handle, "PyString_Size");
  gdb_PySys_GetObject = dlsym (libpython_handle, "PySys_GetObject");
  gdb_PySys_SetPath = dlsym (libpython_handle, "PySys_SetPath");
  gdb_PyThreadState_Get = dlsym (libpython_handle, "PyThreadState_Get");
  gdb_PyThreadState_Swap = dlsym (libpython_handle, "PyThreadState_Swap");
  gdb_PyTuple_GetItem = dlsym (libpython_handle, "PyTuple_GetItem");
  gdb_PyTuple_New = dlsym (libpython_handle, "PyTuple_New");
  gdb_PyTuple_SetItem = dlsym (libpython_handle, "PyTuple_SetItem");
  gdb_PyTuple_Size = dlsym (libpython_handle, "PyTuple_Size");
  gdb_PyType_GenericNew = dlsym (libpython_handle, "PyType_GenericNew");
  gdb_PyType_IsSubtype = dlsym (libpython_handle, "PyType_IsSubtype");
  gdb_PyType_Ready = dlsym (libpython_handle, "PyType_Ready");
  gdb_Py_Finalize = dlsym (libpython_handle, "Py_Finalize");
  gdb_Py_FlushLine = dlsym (libpython_handle, "Py_FlushLine");
  gdb_Py_Initialize = dlsym (libpython_handle, "Py_Initialize");
  gdb_Py_InitModule4 = dlsym (libpython_handle, "Py_InitModule4");
  gdb_Py_InitModule4_64 = dlsym (libpython_handle, "Py_InitModule4_64");
  gdb_PyObject_Call = dlsym (libpython_handle, "PyObject_Call");
  gdb_PyObject_CallObject = dlsym (libpython_handle, "PyObject_CallObject");
  gdb_Py_SetProgramName = dlsym (libpython_handle, "Py_SetProgramName");
  gdb__PyObject_New = dlsym (libpython_handle, "_PyObject_New");
  gdb_PyCode_New = dlsym (libpython_handle, "PyCode_New");
  gdb_PyFrame_New = dlsym (libpython_handle, "PyFrame_New");
#ifdef __APPLE__
  gdb_PyUnicode_Decode = dlsym (libpython_handle, "PyUnicodeUCS2_Decode");
  gdb_PyUnicode_AsEncodedString = dlsym (libpython_handle, "PyUnicodeUCS2_AsEncodedString");
  gdb_PyUnicode_FromEncodedObject = dlsym (libpython_handle, "PyUnicodeUCS2_FromEncodedObject");
#else
  gdb_PyUnicode_Decode = dlsym (libpython_handle, "PyUnicodeUCS4_Decode");
  gdb_PyUnicode_AsEncodedString = dlsym (libpython_handle, "PyUnicodeUCS4_AsEncodedString");
  gdb_PyUnicode_FromEncodedObject = dlsym (libpython_handle, "PyUnicodeUCS4_FromEncodedObject");
#endif
  return true;
err_out:
  dlclose (libpython_handle);
  libpython_handle = NULL;
  return false;
}

#define PYWRAPPER(rtype,name)                                                 \
rtype                                                                         \
name (void)                                                                   \
{                                                                             \
  if (!is_python_available () || gdb_## name == NULL)                         \
    {                                                                         \
      warning ("%s: called while Python is not available!", __FUNCTION__);    \
      return (rtype) 0;                                                       \
    }                                                                         \
  return gdb_## name ();                                                      \
}

#define PYWRAPPER_ARG1(rtype,name, atype)                                     \
rtype                                                                         \
name (atype arg1)                                                             \
{                                                                             \
  if (!is_python_available () || gdb_## name == NULL)                         \
    {                                                                         \
      warning ("%s: called while Python is not available!", __FUNCTION__);    \
      return (rtype) 0;                                                       \
    }                                                                         \
  return gdb_## name (arg1);                                                  \
}

#define PYWRAPPER_ARG2(rtype,name, atype, btype)                              \
rtype                                                                         \
name (atype arg1, btype arg2)                                                 \
{                                                                             \
  if (!is_python_available () || gdb_## name == NULL)                         \
    {                                                                         \
      warning ("%s: called while Python is not available!", __FUNCTION__);    \
      return (rtype) 0;                                                       \
    }                                                                         \
  return gdb_## name (arg1, arg2);                                            \
}

#define PYWRAPPER_ARG3(rtype,name, atype, btype, ctype)                       \
rtype                                                                         \
name (atype arg1, btype arg2, ctype arg3)                                     \
{                                                                             \
  if (!is_python_available () || gdb_## name == NULL)                         \
    {                                                                         \
      warning ("%s: called while Python is not available!", __FUNCTION__);    \
      return (rtype) 0;                                                       \
    }                                                                         \
  return gdb_## name (arg1, arg2, arg3);                                      \
}

#define PYWRAPPER_ARG4(rtype,name, atype, btype, ctype, dtype)                \
rtype                                                                         \
name (atype arg1, btype arg2, ctype arg3, dtype arg4)                         \
{                                                                             \
  if (!is_python_available () || gdb_## name == NULL)                         \
    {                                                                         \
      warning ("%s: called while Python is not available!", __FUNCTION__);    \
      return (rtype) 0;                                                       \
    }                                                                         \
  return gdb_## name (arg1, arg2, arg3, arg4);                                \
}

#define PYWRAPPERVOID(name)                                                   \
void                                                                          \
name (void)                                                                   \
{                                                                             \
  if (is_python_available () && gdb_## name != NULL)                          \
    gdb_## name ();                                                           \
  else                                                                        \
    warning ("%s: called while Python is not available!", __FUNCTION__);      \
}

#define PYWRAPPERVOID_ARG1(name, atype)                                       \
void                                                                          \
name (atype arg1)                                                             \
{                                                                             \
  if (is_python_available () && gdb_## name != NULL)                          \
    gdb_## name (arg1);                                                       \
  else                                                                        \
    warning ("%s: called while Python is not available!", __FUNCTION__);      \
}

#define PYWRAPPERVOID_ARG2(name, atype, btype)                                \
void                                                                          \
name (atype arg1, btype arg2)                                                 \
{                                                                             \
  if (is_python_available () && gdb_## name != NULL)                          \
    gdb_## name (arg1, arg2);                                                 \
  else                                                                        \
    warning ("%s: called while Python is not available!", __FUNCTION__);      \
}

#define PYWRAPPERVOID_ARG3(name, atype, btype, ctype)                         \
void                                                                          \
name (atype arg1, btype arg2, ctype arg3)                                     \
{                                                                             \
  if (is_python_available () && gdb_## name != NULL)                          \
    gdb_## name (arg1, arg2, arg3);                                           \
  else                                                                        \
    warning ("%s: called while Python is not available!", __FUNCTION__);      \
}

PYWRAPPER_ARG1(PyObject *, PyBool_FromLong, long)
PYWRAPPER_ARG1(int, PyCallable_Check, PyObject *)
PYWRAPPER (PyObject *, PyDict_New)
PYWRAPPER_ARG3(int, PyDict_SetItemString, PyObject *, const char *, PyObject *)
PYWRAPPER_ARG1(int, PyErr_ExceptionMatches, PyObject *)
PYWRAPPER_ARG2(int, PyErr_GivenExceptionMatches, PyObject *, PyObject *)
PYWRAPPER (PyObject *, PyErr_Occurred)
PYWRAPPER_ARG1(PyObject *, PyErr_SetFromErrno, PyObject *)
PYWRAPPERVOID(PyErr_Clear)
PYWRAPPERVOID_ARG3(PyErr_Fetch, PyObject **, PyObject **, PyObject **)
PYWRAPPERVOID(PyErr_Print)
PYWRAPPERVOID_ARG3(PyErr_Restore, PyObject *, PyObject *, PyObject *)
PYWRAPPERVOID(PyErr_SetInterrupt)
PYWRAPPERVOID_ARG2(PyErr_SetObject, PyObject *, PyObject *)
PYWRAPPERVOID_ARG2(PyErr_SetString, PyObject *, const char *)
PYWRAPPERVOID(PyEval_InitThreads)
PYWRAPPERVOID(PyEval_ReleaseLock)
PYWRAPPERVOID_ARG1(PyEval_RestoreThread, PyThreadState *)
PYWRAPPER (PyThreadState *, PyEval_SaveThread);
PYWRAPPER_ARG1(double, PyFloat_AsDouble, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyFloat_FromDouble, double)
PYWRAPPER (PyGILState_STATE, PyGILState_Ensure)
PYWRAPPERVOID_ARG1(PyGILState_Release, PyGILState_STATE)
PYWRAPPER_ARG1(PyObject *, PyImport_AddModule, CONSTCHAR *)
PYWRAPPER_ARG1(PyObject *,PyImport_ImportModule, CONSTCHAR *)
PYWRAPPER_ARG1(long, PyInt_AsLong, PyObject *)
PYWRAPPER_ARG1(PyObject *,PyInt_FromLong, long)
PYWRAPPER (long, PyInt_GetMax)
PYWRAPPER_ARG1(PyObject *, PyIter_Next, PyObject *)
PYWRAPPER_ARG2(int, PyList_Append, PyObject *, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyList_AsTuple, PyObject *)
PYWRAPPER_ARG2(PyObject *,PyList_GetItem, PyObject *, Py_ssize_t)
PYWRAPPER_ARG3(int, PyList_Insert, PyObject *, Py_ssize_t, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyList_New, Py_ssize_t)
PYWRAPPER_ARG1(Py_ssize_t, PyList_Size, PyObject *)
PYWRAPPER_ARG1(PY_LONG_LONG, PyLong_AsLongLong, PyObject *)
PYWRAPPER_ARG1(unsigned PY_LONG_LONG, PyLong_AsUnsignedLongLong, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyLong_FromLong, long)
PYWRAPPER_ARG1(PyObject *, PyLong_FromLongLong, PY_LONG_LONG)
PYWRAPPER_ARG1(PyObject *, PyLong_FromUnsignedLong, unsigned long)
PYWRAPPER_ARG1(PyObject *, PyLong_FromUnsignedLongLong, unsigned PY_LONG_LONG)
PYWRAPPER_ARG1(void *, PyMem_Malloc, size_t)
PYWRAPPER_ARG3(int, PyModule_AddIntConstant, PyObject *, CONSTCHAR *, long)
PYWRAPPER_ARG3(int, PyModule_AddObject, PyObject *, CONSTCHAR *, PyObject *)
PYWRAPPER_ARG3(int, PyModule_AddStringConstant, PyObject *, CONSTCHAR *, CONSTCHAR *)
PYWRAPPER_ARG1(PyObject *, PyModule_GetDict, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyNumber_Long, PyObject *)
PYWRAPPER (int, PyOS_InterruptOccurred)
PYWRAPPER_ARG1(int, PyObject_CheckReadBuffer, PyObject *)
PYWRAPPER_ARG2(PyObject *, PyObject_GenericGetAttr, PyObject *, PyObject *)
PYWRAPPER_ARG2(PyObject *, PyObject_GetAttr, PyObject *, PyObject *)
PYWRAPPER_ARG2(PyObject *, PyObject_GetAttrString, PyObject *, CONSTCHAR *)
PYWRAPPER_ARG1(PyObject *, PyObject_GetIter, PyObject *)
PYWRAPPER_ARG2(int, PyObject_HasAttr, PyObject *, PyObject *)
PYWRAPPER_ARG2(int, PyObject_HasAttrString, PyObject *, CONSTCHAR *)
PYWRAPPER_ARG1(int, PyObject_IsTrue, PyObject *)
PYWRAPPER_ARG3(int, PyObject_RichCompareBool, PyObject *, PyObject *, int)
PYWRAPPER_ARG3(int, PyObject_SetAttrString, PyObject *, CONSTCHAR *, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyObject_Str, PyObject *)
PYWRAPPER_ARG3(int, PyRun_InteractiveLoopFlags, FILE *, const char *, PyCompilerFlags *)
PYWRAPPER_ARG4(int, PyRun_SimpleFileExFlags, FILE *, const char *, int, PyCompilerFlags *)
PYWRAPPER_ARG2(int, PyRun_SimpleStringFlags, const char *, PyCompilerFlags *)
PYWRAPPER_ARG1(int, PySequence_Check, PyObject *);
PYWRAPPER_ARG2(int, PySequence_DelItem, PyObject *, Py_ssize_t)
PYWRAPPER_ARG2(PyObject *, PySequence_GetItem, PyObject *, Py_ssize_t)
PYWRAPPER_ARG2(Py_ssize_t, PySequence_Index, PyObject *, PyObject *)
PYWRAPPER_ARG1(PyObject *, PySequence_List, PyObject *)
PYWRAPPER_ARG1(Py_ssize_t, PySequence_Size, PyObject *)
PYWRAPPER_ARG1(char *, PyString_AsString, PyObject *)
PYWRAPPER_ARG1(PyObject *, PyString_FromString, const char *)
PYWRAPPER_ARG1(Py_ssize_t, PyString_Size, PyObject *)
PYWRAPPER_ARG1(PyObject *, PySys_GetObject, char *)
PYWRAPPERVOID_ARG1(PySys_SetPath,char *)
PYWRAPPER (PyThreadState *, PyThreadState_Get);
PYWRAPPER_ARG1(PyThreadState *, PyThreadState_Swap, PyThreadState *)
PYWRAPPER_ARG2(PyObject *, PyTuple_GetItem, PyObject *, Py_ssize_t)
PYWRAPPER_ARG1(PyObject *, PyTuple_New, Py_ssize_t)
PYWRAPPER_ARG3(int, PyTuple_SetItem, PyObject *, Py_ssize_t, PyObject *)
PYWRAPPER_ARG1(Py_ssize_t, PyTuple_Size, PyObject *)
PYWRAPPER_ARG2(int, PyType_IsSubtype, PyTypeObject *, PyTypeObject *)
PYWRAPPER_ARG1(int, PyType_Ready, PyTypeObject *)
PYWRAPPERVOID(Py_Finalize)
PYWRAPPER(int, Py_FlushLine)
PYWRAPPERVOID(Py_Initialize)
PYWRAPPERVOID_ARG1(Py_SetProgramName, char *)
PYWRAPPER_ARG1(PyObject *, _PyObject_New, PyTypeObject *)
PYWRAPPER_ARG2(PyObject *, PyObject_CallObject, PyObject *, PyObject *)
PYWRAPPER_ARG3(PyObject *, PyObject_Call, PyObject *, PyObject *, PyObject *)
PYWRAPPER_ARG3(PyObject *, PyErr_NewException, char *, PyObject *, PyObject *)
PYWRAPPER_ARG4(PyObject *, PyString_Decode, const char *, Py_ssize_t, const char *, const char *)
PYWRAPPER_ARG3(PyObject *, PyType_GenericNew, PyTypeObject *, PyObject *, PyObject *)
PYWRAPPER_ARG3(int, PyObject_AsReadBuffer, PyObject *, const void **, Py_ssize_t *)
PYWRAPPER_ARG3(int, PyObject_GenericSetAttr, PyObject *, PyObject *, PyObject *)
PYWRAPPER_ARG3(PyObject *, PyBuffer_FromReadWriteObject, PyObject *, Py_ssize_t, Py_ssize_t)
PYWRAPPER_ARG4 (PyObject *, PyUnicode_Decode, const char *, Py_ssize_t, const char *, const char *)
PYWRAPPER_ARG3 (PyObject *, PyUnicode_FromEncodedObject, register PyObject *, const char *, const char *)
PYWRAPPER_ARG3 (PyObject *,PyUnicode_AsEncodedString, register PyObject *, const char *, const char *)
PYWRAPPER_ARG4(PyFrameObject *, PyFrame_New, PyThreadState *,PyCodeObject *, PyObject *, PyObject *)

PyCodeObject *
PyCode_New(int a, int b, int c, int d,
           PyObject *e, PyObject *f, PyObject *g, PyObject *h,
           PyObject *i, PyObject *j, PyObject *k, PyObject *l, int m, PyObject *n)
{
  if (!is_python_available () || gdb_PyCode_New == NULL)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return NULL;
    }
  return gdb_PyCode_New(a, b, c, d, e, f, g, h, i, j, k, l, m, n);
}

PyObject *
PyRun_StringFlags(const char *arg1, int arg2, PyObject *arg3, PyObject *arg4, PyCompilerFlags *arg5)
{
  if (!is_python_available () || gdb_PyRun_StringFlags == NULL)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return NULL;
    }
  return gdb_PyRun_StringFlags (arg1, arg2, arg3, arg4, arg5);
}

PyObject*
Py_InitModule4(CONSTCHAR *name, PyMethodDef *methods, CONSTCHAR *doc, PyObject *self, int apiver)
{
  if (!is_python_available () || (gdb_Py_InitModule4 == NULL && gdb_Py_InitModule4_64 == NULL))
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return NULL;
    }
  /* For 64-bit processes, entry point name have changed
   * in more recent version of libpython 
   */
  if (gdb_Py_InitModule4_64 != NULL)
    return gdb_Py_InitModule4_64 (name, methods, doc, self, apiver);
  return gdb_Py_InitModule4 (name, methods, doc, self, apiver);
}


#if HAVE_LIBPYTHON2_4
PyObject *
PyRun_String(const char *arg1, int arg2, PyObject *arg3, PyObject *arg4)
{
  if (!is_python_available () || gdb_PyRun_StringFlags == 0)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return NULL;
    }
  return gdb_PyRun_StringFlags (arg1, arg2, arg3, arg4, NULL);
}

int
PyRun_SimpleString(const char * arg1) {
  if (!is_python_available () || gdb_PyRun_SimpleStringFlags == NULL)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return 0;
    }
  return gdb_PyRun_SimpleStringFlags(arg1, NULL);
}

int
PyRun_InteractiveLoop(FILE * arg1,const char * arg2) {
  if (!is_python_available () || gdb_PyRun_InteractiveLoopFlags == NULL)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return 0;
    }
  return gdb_PyRun_InteractiveLoopFlags(arg1, arg2, NULL);
}

int
PyRun_SimpleFile(FILE * arg1,const char * arg2) {
  if (!is_python_available () || gdb_PyRun_SimpleFileExFlags == NULL)
    {
      warning ("%s: called while Python is not available!", __FUNCTION__);
      return 0;
    }
  return gdb_PyRun_SimpleFileExFlags(arg1, arg2, 0, NULL);
}
#endif
