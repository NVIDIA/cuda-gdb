/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2024-2025 NVIDIA Corporation
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

#include "py-cuda.h"

#include "cuda/cuda-api.h"
#include "cuda/cuda-coord-set.h"
#include "cuda/cuda-coords.h"
#include "cuda/cuda-state.h"
#include "gdbsupport/gdb_unique_ptr.h"

#ifdef HAVE_PYTHON

#include "structmember.h"

#define DEFAULT_BUFFER_SIZE 1024

// Python3.6 doesn't have Py_RETURN_RICHCOMPARE, so fake it here.
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 6

#define Py_RETURN_RICHCOMPARE(a, b, op)                                       \
  return gdbpy_return_richcompare (a, b, op)

// We only use boolean values, so we can simplify the implementation.
static PyObject *
gdbpy_return_richcompare (bool a, bool b, int op)
{
  switch (op)
    {
    case Py_EQ:
      return PyBool_FromLong (a == b);
    case Py_NE:
      return PyBool_FromLong (a != b);
    case Py_LT:
    case Py_LE:
    case Py_GE:
    case Py_GT:
    default:
      PyErr_Format (PyExc_RuntimeError, "Unsupported boolean comparison op %u",
		    op);
      return nullptr;
    }
}
#endif

static PyObject *gdbpy_cuda_cu_dim3_create (const CuDim3 &dim3);

static PyObject *
gdbpy_cuda_coords_logical_create (const cuda_coords_logical &coords);
static PyObject *
gdbpy_cuda_coords_physical_create (const cuda_coords_physical &coords);

// Functions for wrapping the C++ types into Python types
// Using templates to avoid implicit type conversions
template <typename T> PyObject *to_python (T value);

template <>
PyObject *
to_python (bool value)
{
  return PyBool_FromLong (value);
}

template <>
PyObject *
to_python (uint32_t value)
{
  return Py_BuildValue ("I", value);
}

template <>
PyObject *
to_python (uint64_t value)
{
  return Py_BuildValue ("K", value);
}

template <>
PyObject *
to_python (int64_t value)
{
  return Py_BuildValue ("L", value);
}

template <>
PyObject *
to_python (const char *value)
{
  return Py_BuildValue ("s", value);
}

template <>
PyObject *
to_python (CuDim3 value)
{
  return gdbpy_cuda_cu_dim3_create (value);
}

// Identity function
template <typename T>
static T
identity (T value)
{
  return value;
}

/* gdb.cuda module components */

static PyObject *gdbpy_cuda_execute_internal_command (PyObject *self,
						      PyObject *args);
static PyObject *gdbpy_cuda_get_focus_physical (PyObject *self,
						PyObject *args);
static PyObject *gdbpy_cuda_set_focus_physical (PyObject *self,
						PyObject *args);
static PyObject *gdbpy_cuda_get_focus_logical (PyObject *self, PyObject *args);
static PyObject *gdbpy_cuda_set_focus_logical (PyObject *self, PyObject *args);
static PyObject *gdbpy_cuda_get_devices (PyObject *self, PyObject *args);
static PyObject *gdbpy_cuda_read_global_memory (PyObject *self,
						PyObject *args,
						PyObject *kwargs);

static PyObject *gdbpy_cuda_get_device (PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *gdbpy_cuda_get_sm (PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *gdbpy_cuda_get_warp (PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *gdbpy_cuda_get_lane (PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef gdbpy_cuda_methods[] = {
  { "execute_internal_command", gdbpy_cuda_execute_internal_command,
    METH_VARARGS, "execute internal command" },
  { "get_focus_physical", gdbpy_cuda_get_focus_physical, METH_NOARGS,
    "Returns the current focus as cuda.CoordsPhysical" },
  { "set_focus_physical", gdbpy_cuda_set_focus_physical, METH_VARARGS,
    "Sets the current focus to the passed in physical coordinates" },
  { "get_focus_logical", gdbpy_cuda_get_focus_logical, METH_NOARGS,
    "Returns the current focus as cuda.CoordsLogical" },
  { "set_focus_logical", gdbpy_cuda_set_focus_logical, METH_VARARGS,
    "Sets the current focus to the passed in logical coordinates" },
  { "devices", gdbpy_cuda_get_devices, METH_NOARGS,
    "Returns the list of devices as list of cuda.Device" },
  { "device", (PyCFunction) gdbpy_cuda_get_device, METH_VARARGS | METH_KEYWORDS,
    "Returns the specified cuda.Device" },
  { "sm", (PyCFunction) gdbpy_cuda_get_sm, METH_VARARGS | METH_KEYWORDS,
    "Returns the specified cuda.Sm" },
  { "warp", (PyCFunction) gdbpy_cuda_get_warp, METH_VARARGS | METH_KEYWORDS,
    "Returns the specified cuda.Warp" },
  { "lane", (PyCFunction) gdbpy_cuda_get_lane, METH_VARARGS | METH_KEYWORDS,
    "Returns the specified cuda.Lane" },
  { "read_global_memory", (PyCFunction) gdbpy_cuda_read_global_memory, METH_VARARGS | METH_KEYWORDS,
    "Read global memory" },
  { nullptr, nullptr, 0, nullptr },
};

static PyModuleDef gdbpy_cuda_module = {
  PyModuleDef_HEAD_INIT,
  "cuda",
  "CUDA Python Integration Module",
  -1,
  gdbpy_cuda_methods,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static bool
gdbpy_cuda_install_type_in_module (PyObject *module, const char *name,
				   PyTypeObject *type)
{
  if (PyType_Ready (type) < 0)
    return false;

  Py_INCREF (type);
  if (PyModule_AddObject (module, name, (PyObject *)type) < 0)
    {
      Py_DECREF (type);
      return false;
    }
  return true;
}

static PyObject *
gdbpy_cuda_read_global_memory (PyObject *self, PyObject *args, PyObject *kwargs)
{
  uint64_t size = 0;
  uint64_t address = 0;
  static const char *kwlist[] = { "address", "size", nullptr };

  if (!PyArg_ParseTupleAndKeywords (args, kwargs, "KK", (char **)kwlist,
				    &address, &size))
    return nullptr;

  try
    {
      gdbpy_ref<> buffer (PyBytes_FromStringAndSize (nullptr, size));
      if (buffer == nullptr)
	return nullptr;

      cuda_debugapi::read_global_memory (address, PyBytes_AS_STRING (buffer.get ()),
					 size);
      return buffer.release ();
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

/* gdb.cuda.CuDim3 type */

typedef struct
{
  PyObject_HEAD uint32_t x;
  uint32_t y;
  uint32_t z;
} gdbpy_cuda_cu_dim3_object;

static PyTypeObject gdbpy_cuda_cu_dim3_type
    = { PyVarObject_HEAD_INIT (nullptr, 0) };

static PyObject *
gdbpy_cuda_cu_dim3_get_x (PyObject *self, void *closure)
{
  const auto dim3 = (gdbpy_cuda_cu_dim3_object *)self;

  return PyLong_FromUnsignedLong (dim3->x);
}

static PyObject *
gdbpy_cuda_cu_dim3_get_y (PyObject *self, void *closure)
{
  const auto dim3 = (gdbpy_cuda_cu_dim3_object *)self;

  return PyLong_FromUnsignedLong (dim3->y);
}

static PyObject *
gdbpy_cuda_cu_dim3_get_z (PyObject *self, void *closure)
{
  const auto dim3 = (gdbpy_cuda_cu_dim3_object *)self;

  return PyLong_FromUnsignedLong (dim3->z);
}

static gdb_PyGetSetDef gdbpy_cuda_cu_dim3_getset[] = {
  { "x", gdbpy_cuda_cu_dim3_get_x, nullptr, "x", nullptr },
  { "y", gdbpy_cuda_cu_dim3_get_y, nullptr, "y", nullptr },
  { "z", gdbpy_cuda_cu_dim3_get_z, nullptr, "z", nullptr },
  { nullptr },
};

static int
gdbpy_cuda_cu_dim3_init (PyObject *self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = { "x", "y", "z", nullptr };

  // PyArg_ParseTupleAndKeywords does not modify fields corresponding
  // to missing arguments, so we need to initialize them to 0
  auto dim3 = (gdbpy_cuda_cu_dim3_object *)self;
  dim3->x = 0;
  dim3->y = 0;
  dim3->z = 0;

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|III", (char **)kwlist,
				    &dim3->x, &dim3->y, &dim3->z))
    return -1;
  return 0;
}

static PyObject *
gdbpy_cuda_cu_dim3_create (const CuDim3 &dim3)
{
  gdbpy_ref<gdbpy_cuda_cu_dim3_object> self (PyObject_New (gdbpy_cuda_cu_dim3_object, &gdbpy_cuda_cu_dim3_type));
  if (self == nullptr)
    return nullptr;

  self->x = dim3.x;
  self->y = dim3.y;
  self->z = dim3.z;

  return (PyObject *)self.release ();
}

static PyObject *
gdbpy_cuda_cu_dim3_richcompare (PyObject *self, PyObject *other, int opid)
{
  if (opid != Py_EQ && opid != Py_NE)
    {
      PyErr_SetString (PyExc_TypeError, "Invalid comparison");
      return nullptr;
    }
  if (other == Py_None)
    {
      if (opid == Py_EQ)
	Py_RETURN_FALSE;
      if (opid == Py_NE)
	Py_RETURN_TRUE;
    }
  if (!PyObject_TypeCheck (self, &gdbpy_cuda_cu_dim3_type))
    {
      PyErr_SetString (PyExc_TypeError, "Invalid type");
      return nullptr;
    }
  if (!PyObject_TypeCheck (other, &gdbpy_cuda_cu_dim3_type))
    {
      PyErr_SetString (PyExc_TypeError, "Invalid type");
      return nullptr;
    }

  auto a = (gdbpy_cuda_cu_dim3_object *)self;
  auto b = (gdbpy_cuda_cu_dim3_object *)other;
  Py_RETURN_RICHCOMPARE ((a->x == b->x) && (a->y == b->y) && (a->z == b->z),
			 true, opid);
}

static PyObject *
gdbpy_cuda_cu_dim3_repr (PyObject *self)
{
  const auto dim3 = (gdbpy_cuda_cu_dim3_object *)self;

  return PyUnicode_FromFormat ("(%u,%u,%u)", dim3->x, dim3->y, dim3->z);
}

static bool
gdbpy_cuda_cu_dim3_type_init (PyObject *module)
{
  gdbpy_cuda_cu_dim3_type.tp_name = "cuda.CuDim3";
  gdbpy_cuda_cu_dim3_type.tp_doc
      = PyDoc_STR ("CUDA 3 dimensional coordinates");
  gdbpy_cuda_cu_dim3_type.tp_basicsize = sizeof (gdbpy_cuda_cu_dim3_object);
  gdbpy_cuda_cu_dim3_type.tp_itemsize = 0;
  gdbpy_cuda_cu_dim3_type.tp_flags = Py_TPFLAGS_DEFAULT;
  gdbpy_cuda_cu_dim3_type.tp_new = PyType_GenericNew;
  gdbpy_cuda_cu_dim3_type.tp_init = gdbpy_cuda_cu_dim3_init;
  gdbpy_cuda_cu_dim3_type.tp_getset = gdbpy_cuda_cu_dim3_getset;
  gdbpy_cuda_cu_dim3_type.tp_repr = gdbpy_cuda_cu_dim3_repr;
  gdbpy_cuda_cu_dim3_type.tp_richcompare = gdbpy_cuda_cu_dim3_richcompare;

  return gdbpy_cuda_install_type_in_module (module, "CuDim3",
					    &gdbpy_cuda_cu_dim3_type);
}

//
// cuda.Device / cuda.Sm / cuda.Warp / cuda.Lane types
//

// Type structs
static PyTypeObject gdbpy_cuda_device_type
    = { PyVarObject_HEAD_INIT (nullptr, 0) };

static PyTypeObject gdbpy_cuda_sm_type
    = { PyVarObject_HEAD_INIT (nullptr, 0) };

static PyTypeObject gdbpy_cuda_warp_type
    = { PyVarObject_HEAD_INIT (nullptr, 0) };

static PyTypeObject gdbpy_cuda_lane_type
    = { PyVarObject_HEAD_INIT (nullptr, 0) };

// Class structs

typedef struct
{
  PyObject_HEAD uint32_t dev;
} gdbpy_cuda_device_object;

typedef struct
{
  PyObject_HEAD uint32_t dev;
  uint32_t sm;
} gdbpy_cuda_sm_object;

typedef struct
{
  PyObject_HEAD uint32_t dev;
  uint32_t sm;
  uint32_t wp;
} gdbpy_cuda_warp_object;

typedef struct
{
  PyObject_HEAD uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
} gdbpy_cuda_lane_object;

//
// Validity checkers
//
// Initialization functions
static PyObject *gdbpy_cuda_device_create (uint32_t dev);
static PyObject *gdbpy_cuda_sm_create (uint32_t dev, uint32_t sm);
static PyObject *gdbpy_cuda_warp_create (uint32_t dev, uint32_t sm,
					 uint32_t wp);
static PyObject *gdbpy_cuda_lane_create (uint32_t dev, uint32_t sm,
					 uint32_t wp, uint32_t ln);

// Access functions
static gdbpy_cuda_device_object *gdbpy_cuda_device (PyObject *self);
static gdbpy_cuda_sm_object *gdbpy_cuda_sm (PyObject *self);
static gdbpy_cuda_warp_object *gdbpy_cuda_warp (PyObject *self);
static gdbpy_cuda_lane_object *gdbpy_cuda_lane (PyObject *self);

static bool
gdbpy_check_device (uint32_t dev)
{
  if (dev >= cuda_state::get_num_devices ())
    {
      PyErr_Format (PyExc_RuntimeError, "Invalid Device <cuda.Device dev%u>",
		    dev);
      return false;
    }
  return true;
}

static bool
gdbpy_check_sm (uint32_t dev, uint32_t sm)
{
  if (!gdbpy_check_device (dev)
      || (sm >= cuda_state::device_get_num_sms (dev)))
    {
      PyErr_Format (PyExc_RuntimeError, "Invalid SM <cuda.Sm dev%u.sm%u>", dev,
		    sm);
      return false;
    }
  return true;
}

static bool
gdbpy_check_warp (uint32_t dev, uint32_t sm, uint32_t wp)
{
  if (!gdbpy_check_sm (dev, sm)
      || (wp >= cuda_state::device_get_num_warps (dev)))
    {
      PyErr_Format (PyExc_RuntimeError,
		    "Invalid Warp <cuda.Warp dev%u.sm%u.wp%u>", dev, sm, wp);
      return false;
    }
  return true;
}

static bool
gdbpy_check_lane (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln)
{
  if (!gdbpy_check_warp (dev, sm, wp)
      || (ln >= cuda_state::device_get_num_lanes (dev)))
    {
      PyErr_Format (PyExc_RuntimeError,
		    "Invalid Lane <cuda.Lane dev%u.sm%u.wp%u.ln%u>", dev, sm,
		    wp, ln);
      return false;
    }
  return true;
}

static gdbpy_cuda_device_object *
gdbpy_cuda_device (PyObject *self)
{
  if (!PyObject_TypeCheck (self, &gdbpy_cuda_device_type))
    {
      PyErr_SetString (PyExc_TypeError, "Invalid type");
      return nullptr;
    }

  auto device = (gdbpy_cuda_device_object *)self;
  return gdbpy_check_device (device->dev) ? device : nullptr;
}

static gdbpy_cuda_sm_object *
gdbpy_cuda_sm (PyObject *self)
{
  if (!PyObject_TypeCheck (self, &gdbpy_cuda_sm_type))
    {
      PyErr_SetString (PyExc_TypeError, "Invalid type");
      return nullptr;
    }

  auto sm = (gdbpy_cuda_sm_object *)self;
  return gdbpy_check_sm (sm->dev, sm->sm) ? sm : nullptr;
}

static gdbpy_cuda_warp_object *
gdbpy_cuda_warp (PyObject *self)
{
  if (!PyObject_TypeCheck (self, &gdbpy_cuda_warp_type))
    {
      PyErr_SetString (PyExc_TypeError, "Invalid type");
      return nullptr;
    }

  auto warp = (gdbpy_cuda_warp_object *)self;
  return gdbpy_check_warp (warp->dev, warp->sm, warp->wp) ? warp : nullptr;
}

static gdbpy_cuda_lane_object *
gdbpy_cuda_lane (PyObject *self)
{
  if (!PyObject_TypeCheck (self, &gdbpy_cuda_lane_type))
    {
      PyErr_SetString (PyExc_TypeError, "Invalid type");
      return nullptr;
    }

  auto lane = (gdbpy_cuda_lane_object *)self;
  return gdbpy_check_lane (lane->dev, lane->sm, lane->wp, lane->ln) ? lane
								    : nullptr;
}

// Attribute wrappers for Device / SM / Warp / Lane

template <typename T> using CudaDeviceGetter = T (*) (uint32_t dev);

template <typename T, typename U, CudaDeviceGetter<U> Getter>
static PyObject *
cuda_device_attr_wrapper (PyObject *self, void *closure)
{
  try
    {
      const auto obj = gdbpy_cuda_device (self);
      if (obj == nullptr)
	return nullptr;

      return to_python (static_cast<T> (Getter (obj->dev)));
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

// The simple version of the wrapper
template <typename T, CudaDeviceGetter<T> Getter>
static PyObject *
cuda_device_attr_wrapper (PyObject *self, void *closure)
{
  return cuda_device_attr_wrapper<T, T, Getter> (self, closure);
}

// Attribute wrapper for CUDA warp attributes
template <typename T>
using CudaWarpGetter = T (*) (uint32_t dev, uint32_t sm, uint32_t wp);

// For accessing attributes that need casting to get the right version of
// to_python()
template <typename T, typename U, CudaWarpGetter<U> Getter>
static PyObject *
cuda_warp_attr_wrapper (PyObject *self, void *closure)
{
  try
    {
      const auto obj = gdbpy_cuda_warp (self);
      if (obj == nullptr)
	return nullptr;

      return to_python (static_cast<T> (Getter (obj->dev, obj->sm, obj->wp)));
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

// The simple version of the wrapper
template <typename T, CudaWarpGetter<T> Getter>
static PyObject *
cuda_warp_attr_wrapper (PyObject *self, void *closure)
{
  return cuda_warp_attr_wrapper<T, T, Getter> (self, closure);
}

// Attribute wrapper for CUDA lane attributes
template <typename T>
using CudaLaneGetter
    = T (*) (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln);

// For accessing attributes that need casting to get the right version of
// to_python()
template <typename T, typename U, CudaLaneGetter<U> Getter>
static PyObject *
cuda_lane_attr_wrapper (PyObject *self, void *closure)
{
  try
    {
      const auto obj = gdbpy_cuda_lane (self);
      if (obj == nullptr)
	return nullptr;

      return to_python (
	  static_cast<T> (Getter (obj->dev, obj->sm, obj->wp, obj->ln)));
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

// The simple version of the wrapper
template <typename T, CudaLaneGetter<T> Getter>
static PyObject *
cuda_lane_attr_wrapper (PyObject *self, void *closure)
{
  return cuda_lane_attr_wrapper<T, T, Getter> (self, closure);
}

// Getters for Device / SM / Warp / Lane IDs
template <typename T>
static PyObject *
cuda_get_device_id (PyObject *self, void *closure)
{
  return Py_BuildValue ("I", ((T *)self)->dev);
}

template <typename T>
static PyObject *
cuda_get_sm_id (PyObject *self, void *closure)
{
  return Py_BuildValue ("I", ((T *)self)->sm);
}

template <typename T>
static PyObject *
cuda_get_warp_id (PyObject *self, void *closure)
{
  return Py_BuildValue ("I", ((T *)self)->wp);
}

template <typename T>
static PyObject *
cuda_get_lane_id (PyObject *self, void *closure)
{
  return Py_BuildValue ("I", ((T *)self)->ln);
}

//
// cuda.Device type
//
// Exposing cuda_device attributes to python
// At the moment we only expose read-only attributes
//

// Table of cuda_device attributes
static gdb_PyGetSetDef gdbpy_cuda_device_getset[] = {
  { "device_id", cuda_get_device_id<gdbpy_cuda_device_object>, nullptr,
    "Device ID", nullptr },
  { "has_exception",
    cuda_device_attr_wrapper<bool, cuda_state::device_has_exception>, nullptr,
    "Has exception", nullptr },
  { "instruction_size",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_insn_size>,
    nullptr, "Instruction size", nullptr },
  { "name",
    cuda_device_attr_wrapper<const char *, cuda_state::device_get_device_name>,
    nullptr, "Name", nullptr },
  { "num_kernels",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_num_kernels>,
    nullptr, "Num kernels", nullptr },
  { "num_lanes",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_num_lanes>,
    nullptr, "Num lanes", nullptr },
  { "num_predicates",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_num_predicates>,
    nullptr, "Num predicates", nullptr },
  { "num_registers",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_num_registers>,
    nullptr, "Num registers", nullptr },
  { "num_sms",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_num_sms>,
    nullptr, "Num SMs", nullptr },
  { "num_upredicates",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_num_upredicates>,
    nullptr, "Num uniform predicates", nullptr },
  { "num_uregisters",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_num_uregisters>,
    nullptr, "Num uniform registers", nullptr },
  { "num_warps",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_num_warps>,
    nullptr, "Num warps", nullptr },
  { "pci_bus_id",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_pci_bus_id>,
    nullptr, "PCI Bus ID", nullptr },
  { "pci_device_id",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_pci_dev_id>,
    nullptr, "PCI Device ID", nullptr },
  { "sm_type",
    cuda_device_attr_wrapper<const char *, cuda_state::device_get_sm_type>,
    nullptr, "SM type", nullptr },
  { "sm_version",
    cuda_device_attr_wrapper<uint32_t, cuda_state::device_get_sm_version>,
    nullptr, "SM version", nullptr },
  { "type",
    cuda_device_attr_wrapper<const char *, cuda_state::device_get_device_type>,
    nullptr, "Device type", nullptr },
  { nullptr },
};

static PyObject *
gdbpy_cuda_device_sms_filtered (PyObject *self,
				bool (*filter) (uint32_t dev, uint32_t sm))
{
  const auto device = gdbpy_cuda_device (self);
  if (!device)
    return nullptr;

  gdbpy_ref<> list (PyList_New (0));
  if (list == nullptr)
    return nullptr;

  const auto num_sms = cuda_state::device_get_num_sms (device->dev);
  for (auto i = 0; i < num_sms; i++)
    {
      // Skip those we want to filter out
      if (filter && !filter (device->dev, i))
	continue;

      gdbpy_ref<> sm ((PyObject *)gdbpy_cuda_sm_create (device->dev, i));
      if (sm == nullptr)
	return nullptr;
      if (PyList_Append (list.get (), sm.get ()) == -1)
        return nullptr;

      cuda_trace_domain (CUDA_TRACE_PYTHON,
			 "Added SM <dev%u.sm%u> @ %p to list at %p",
			 device->dev, i, sm.get (), list.get ());
    }
  return list.release ();
}

static PyObject *
gdbpy_cuda_device_method_sms (PyObject *self, PyObject *args)
{
  return gdbpy_cuda_device_sms_filtered (self, nullptr);
}

static PyObject *
gdbpy_cuda_device_method_active_sms (PyObject *self, PyObject *args)
{
  return gdbpy_cuda_device_sms_filtered (
      self, [] (uint32_t dev, uint32_t sm) -> bool {
	return cuda_state::device_get_active_sms_mask (dev)[sm];
      });
}

static PyMethodDef gdbpy_cuda_device_methods[] = {
  { "sms", gdbpy_cuda_device_method_sms, METH_NOARGS,
    "Returns the list of cuda.Sm objects for a cuda.Device" },
  { "active_sms", gdbpy_cuda_device_method_active_sms, METH_NOARGS,
    "Returns the list of active cuda.Sm objects for a cuda.Device" },
  { nullptr, nullptr, 0, nullptr },
};

static PyObject *
gdbpy_cuda_device_method_richcompare (PyObject *self, PyObject *other,
				      int opid)
{
  if (opid != Py_EQ && opid != Py_NE)
    {
      PyErr_SetString (PyExc_TypeError, "Invalid comparison");
      return nullptr;
    }
  if (other == Py_None)
    {
      if (opid == Py_EQ)
	Py_RETURN_FALSE;
      if (opid == Py_NE)
	Py_RETURN_TRUE;
    }

  const auto obj_self = gdbpy_cuda_device (self);
  if (obj_self == nullptr)
    return nullptr;

  const auto obj_other = gdbpy_cuda_device (other);
  if (obj_other == nullptr)
    return nullptr;

  Py_RETURN_RICHCOMPARE (obj_self->dev == obj_other->dev, true, opid);
}

static PyObject *
gdbpy_cuda_device_method_repr (PyObject *self)
{
  const auto device = (gdbpy_cuda_device_object *)self;

  return PyUnicode_FromFormat ("<cuda.Device dev%u>", device->dev);
}

static PyObject *
gdbpy_cuda_device_create (uint32_t dev)
{
  if (!gdbpy_check_device (dev))
    return nullptr;

  auto *self = PyObject_New (gdbpy_cuda_device_object, &gdbpy_cuda_device_type);
  if (self == nullptr)
    return nullptr;

  self->dev = dev;

  cuda_trace_domain (CUDA_TRACE_PYTHON, "cuda_device_create <dev%u> @ %p", dev, self);
  return (PyObject *)self;
}

static void
gdbpy_cuda_device_finalize (PyObject *self)
{
  auto obj = (gdbpy_cuda_device_object *)self;
  cuda_trace_domain (CUDA_TRACE_PYTHON, "cuda_device_finalize <dev%u> @ %p",
		     obj->dev, obj);
}

static bool
gdbpy_cuda_device_type_init (PyObject *module)
{
  auto flags = Py_TPFLAGS_DEFAULT;

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 10
  // Don't allow users to create their own instances
  flags |= Py_TPFLAGS_DISALLOW_INSTANTIATION;
#endif

  gdbpy_cuda_device_type.tp_name = "cuda.Device";
  gdbpy_cuda_device_type.tp_doc = PyDoc_STR ("CUDA Device");
  gdbpy_cuda_device_type.tp_basicsize = sizeof (gdbpy_cuda_device_object);
  gdbpy_cuda_device_type.tp_itemsize = 0;
  gdbpy_cuda_device_type.tp_flags = flags;
  gdbpy_cuda_device_type.tp_new = PyType_GenericNew;
  gdbpy_cuda_device_type.tp_getset = gdbpy_cuda_device_getset;
  gdbpy_cuda_device_type.tp_methods = gdbpy_cuda_device_methods;
  gdbpy_cuda_device_type.tp_repr = gdbpy_cuda_device_method_repr;
  gdbpy_cuda_device_type.tp_finalize = gdbpy_cuda_device_finalize;
  gdbpy_cuda_device_type.tp_richcompare = gdbpy_cuda_device_method_richcompare;

  return gdbpy_cuda_install_type_in_module (module, "Device",
					    &gdbpy_cuda_device_type);
}

//
// cuda.Sm type
//
// Exposing cuda_sm attributes to python
//

static PyObject *
cuda_sm_get_attribute (PyObject *self,
		       PyObject *(*closure) (uint32_t dev, uint32_t sm))
{
  try
    {
      auto obj = gdbpy_cuda_sm (self);
      if (!obj)
	return nullptr;

      return closure (obj->dev, obj->sm);
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

static PyObject *
gdbpy_cuda_sm_get_errorpc (PyObject *self, void *closure)
{
  return cuda_sm_get_attribute (
      self, [] (uint32_t dev, uint32_t sm) -> PyObject * {
	// Return None if there is no error pc
	if (!cuda_state::sm_has_error_pc (dev, sm))
	  Py_RETURN_NONE;
	return Py_BuildValue ("K", cuda_state::sm_get_error_pc (dev, sm));
      });
}

static PyObject *
gdbpy_cuda_sm_get_exception (PyObject *self, void *closure)
{
  return cuda_sm_get_attribute (
      self, [] (uint32_t dev, uint32_t sm) -> PyObject * {
	// Return None if there is no exception
	if (!cuda_state::sm_has_exception (dev, sm))
	  Py_RETURN_NONE;
	return Py_BuildValue ("I", cuda_state::sm_get_exception (dev, sm));
      });
}

// cuda.Sm attributes table

static gdb_PyGetSetDef gdbpy_cuda_sm_getset[] = {
  { "device_id", cuda_get_device_id<gdbpy_cuda_sm_object>, nullptr, "Device ID",
    nullptr },
  { "sm_id", cuda_get_sm_id<gdbpy_cuda_sm_object>, nullptr, "Sm ID", nullptr },
  { "errorpc", gdbpy_cuda_sm_get_errorpc, nullptr, "Error pc", nullptr },
  { "exception", gdbpy_cuda_sm_get_exception, nullptr, "Exception", nullptr },
  { nullptr },
};

static PyObject *
gdbpy_cuda_sm_method_warps (PyObject *self, PyObject *args)
{
  const auto sm = gdbpy_cuda_sm (self);
  if (!sm)
    return nullptr;

  const auto num_warps = cuda_state::device_get_num_warps (sm->dev);
  gdbpy_ref<> list (PyList_New (num_warps));
  if (list == nullptr)
    return nullptr;

  for (auto i = 0; i < num_warps; i++)
    {
      auto warp = gdbpy_cuda_warp_create (sm->dev, sm->sm, i);
      if (warp == nullptr)
	return nullptr;
      if (PyList_SetItem (list.get (), i, warp) == -1)
	{
	  Py_DECREF (warp);
	  return nullptr;
	}
    }

  return list.release ();
}

static PyMethodDef gdbpy_cuda_sm_methods[] = {
  { "warps", gdbpy_cuda_sm_method_warps, METH_NOARGS,
    "Returns the list of cuda.Warp objects for a cuda.Sm" },
  { nullptr, nullptr, 0, nullptr },
};

static PyObject *
gdbpy_cuda_sm_method_richcompare (PyObject *self, PyObject *other, int opid)
{
  if (opid != Py_EQ && opid != Py_NE)
    {
      PyErr_SetString (PyExc_TypeError, "Invalid comparison");
      return nullptr;
    }
  if (other == Py_None)
    {
      if (opid == Py_EQ)
	Py_RETURN_FALSE;
      if (opid == Py_NE)
	Py_RETURN_TRUE;
    }

  const auto obj_self = gdbpy_cuda_sm (self);
  if (obj_self == nullptr)
    return nullptr;

  const auto obj_other = gdbpy_cuda_sm (other);
  if (obj_other == nullptr)
    return nullptr;

  Py_RETURN_RICHCOMPARE ((obj_self->dev == obj_other->dev)
			     && (obj_self->sm == obj_other->sm),
			 true, opid);
}

static PyObject *
gdbpy_cuda_sm_method_repr (PyObject *self)
{
  gdbpy_cuda_sm_object *sm = (gdbpy_cuda_sm_object *)self;

  return PyUnicode_FromFormat ("<cuda.Sm dev%u.sm%u>", sm->dev, sm->sm);
}

static PyObject *
gdbpy_cuda_sm_create (uint32_t dev, uint32_t sm)
{
  if (!gdbpy_check_sm (dev, sm))
    return nullptr;

  auto *self = PyObject_New (gdbpy_cuda_sm_object, &gdbpy_cuda_sm_type);
  if (self == nullptr)
    return nullptr;

  self->dev = dev;
  self->sm = sm;

  cuda_trace_domain (CUDA_TRACE_PYTHON, "cuda_sm_create <dev%u.sm%u> @ %p",
		     dev, sm, self);

  return (PyObject *)self;
}

static void
gdbpy_cuda_sm_finalize (PyObject *self)
{
  auto obj = (gdbpy_cuda_sm_object *)self;
  cuda_trace_domain (CUDA_TRACE_PYTHON, "cuda_sm_finalize <dev%u.sm%u> @ %p",
		     obj->dev, obj->sm, obj);
}

static bool
gdbpy_cuda_sm_type_init (PyObject *module)
{
  auto flags = Py_TPFLAGS_DEFAULT;

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 10
  // Don't allow users to create their own instances
  flags |= Py_TPFLAGS_DISALLOW_INSTANTIATION;
#endif

  gdbpy_cuda_sm_type.tp_name = "cuda.Sm";
  gdbpy_cuda_sm_type.tp_doc = PyDoc_STR ("CUDA SM");
  gdbpy_cuda_sm_type.tp_basicsize = sizeof (gdbpy_cuda_sm_object);
  gdbpy_cuda_sm_type.tp_itemsize = 0;
  gdbpy_cuda_sm_type.tp_flags = flags;
  gdbpy_cuda_sm_type.tp_new = PyType_GenericNew;
  gdbpy_cuda_sm_type.tp_getset = gdbpy_cuda_sm_getset;
  gdbpy_cuda_sm_type.tp_methods = gdbpy_cuda_sm_methods;
  gdbpy_cuda_sm_type.tp_repr = gdbpy_cuda_sm_method_repr;
  gdbpy_cuda_sm_type.tp_finalize = gdbpy_cuda_sm_finalize;
  gdbpy_cuda_sm_type.tp_richcompare = gdbpy_cuda_sm_method_richcompare;

  return gdbpy_cuda_install_type_in_module (module, "Sm", &gdbpy_cuda_sm_type);
}

//
// cuda.Warp type
//
// Exposing cuda_warp attributes to python
//

static PyObject *
cuda_warp_get_attribute (PyObject *self,
			 PyObject *(*closure) (uint32_t dev, uint32_t sm,
					       uint32_t wp))
{
  try
    {
      auto obj = gdbpy_cuda_warp (self);
      if (!obj)
	return nullptr;

      // If the warp isn't currently valid, return None
      if (!cuda_state::warp_valid (obj->dev, obj->sm, obj->wp))
	Py_RETURN_NONE;

      return closure (obj->dev, obj->sm, obj->wp);
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

static PyObject *
gdbpy_cuda_warp_get_valid (PyObject *self, void *closure)
{
  try
    {
      const auto obj = gdbpy_cuda_warp (self);
      if (!obj)
	return nullptr;

      if (!cuda_state::warp_valid (obj->dev, obj->sm, obj->wp))
	Py_RETURN_FALSE;

      Py_RETURN_TRUE;
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

static PyObject *
gdbpy_cuda_warp_get_errorpc (PyObject *self, void *closure)
{
  return cuda_warp_get_attribute (
      self, [] (uint32_t dev, uint32_t sm, uint32_t wp) -> PyObject * {
	// Return None if the warp is invalid or there is no error pc
	if (!cuda_state::warp_has_error_pc (dev, sm, wp))
	  Py_RETURN_NONE;
	return Py_BuildValue ("K",
			      cuda_state::warp_get_error_pc (dev, sm, wp));
      });
}

static PyObject *
gdbpy_cuda_warp_get_cluster_exception_block_idx (PyObject *self, void *closure)
{
  return cuda_warp_get_attribute (
      self, [] (uint32_t dev, uint32_t sm, uint32_t wp) -> PyObject * {
	if (!cuda_state::warp_has_cluster_exception_target_block_idx (dev, sm,
								      wp))
	  Py_RETURN_NONE;
	return gdbpy_cuda_cu_dim3_create (
	    cuda_state::warp_get_cluster_exception_target_block_idx (dev, sm,
								     wp));
      });
}

static gdb_PyGetSetDef gdbpy_cuda_warp_getset[] = {
  { "device_id", cuda_get_device_id<gdbpy_cuda_warp_object>, nullptr, "Device ID",
    nullptr },
  { "sm_id", cuda_get_sm_id<gdbpy_cuda_warp_object>, nullptr, "Sm ID",
    nullptr },
  { "warp_id", cuda_get_warp_id<gdbpy_cuda_warp_object>, nullptr, "Warp ID",
    nullptr },
  { "active_pc",
    cuda_warp_attr_wrapper<uint64_t, cuda_state::warp_get_active_pc>, nullptr,
    "Active pc", nullptr },
  { "is_broken", cuda_warp_attr_wrapper<bool, cuda_state::warp_broken>, nullptr,
    "Broken", nullptr },
  { "errorpc", gdbpy_cuda_warp_get_errorpc, nullptr, "Error pc", nullptr },
  { "grid_id",
    cuda_warp_attr_wrapper<int64_t, uint64_t, cuda_state::warp_get_grid_id>,
    nullptr, "Grid ID", nullptr },
  { "registers_allocated",
    cuda_warp_attr_wrapper<uint32_t, cuda_state::warp_registers_allocated>,
    nullptr, "Registers allocated", nullptr },
  { "shared_memory_size",
    cuda_warp_attr_wrapper<uint32_t, cuda_state::warp_shared_mem_size>,
    nullptr, "Shared memory size", nullptr },
  { "is_valid", gdbpy_cuda_warp_get_valid, nullptr, "Valid", nullptr },
  { "block_idx",
    cuda_warp_attr_wrapper<CuDim3, const CuDim3 &,
			   cuda_state::warp_get_block_idx>,
    nullptr, "Block Index", nullptr },
  { "cluster_dim",
    cuda_warp_attr_wrapper<CuDim3, const CuDim3 &,
			   cuda_state::warp_get_cluster_dim>,
    nullptr, "Cluster Dimension", nullptr },
  { "cluster_idx",
    cuda_warp_attr_wrapper<CuDim3, const CuDim3 &,
			   cuda_state::warp_get_cluster_idx>,
    nullptr, "Cluster Index", nullptr },
  { "cluster_exception_block_idx",
    gdbpy_cuda_warp_get_cluster_exception_block_idx, nullptr,
    "Cluster exception block index", nullptr },
  { nullptr },
};

static PyObject *
gdbpy_cuda_warp_lanes_filtered (PyObject *self,
				bool (*filter) (uint32_t dev, uint32_t sm,
						uint32_t wp, uint32_t ln))
{
  auto warp = gdbpy_cuda_warp (self);
  if (!warp)
    return nullptr;

  const auto num_lanes = cuda_state::device_get_num_lanes (warp->dev);
  gdbpy_ref<> list (PyList_New (0));
  if (list == nullptr)
    return nullptr;

  for (auto i = 0; i < num_lanes; i++)
    {
      // Skip those we want to filter out
      if (filter && !filter (warp->dev, warp->sm, warp->wp, i))
	continue;

      gdbpy_ref<> lane ((PyObject *)gdbpy_cuda_lane_create (warp->dev, warp->sm, warp->wp, i));
      if (lane == nullptr)
	return nullptr;
      if (PyList_Append (list.get (), lane.get ()) == -1)
	return nullptr;
    }
  return list.release ();
}

static PyObject *
gdbpy_cuda_warp_method_lanes (PyObject *self, PyObject *args)
{
  return gdbpy_cuda_warp_lanes_filtered (self, nullptr);
}

static PyObject *
gdbpy_cuda_warp_method_active_lanes (PyObject *self, PyObject *args)
{
  return gdbpy_cuda_warp_lanes_filtered (
      self, [] (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln) -> bool {
	return cuda_state::warp_valid (dev, sm, wp)
	       && cuda_state::lane_active (dev, sm, wp, ln);
      });
}

static PyObject *
gdbpy_cuda_warp_method_divergent_lanes (PyObject *self, PyObject *args)
{
  return gdbpy_cuda_warp_lanes_filtered (
      self, [] (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln) -> bool {
	return cuda_state::warp_valid (dev, sm, wp)
	       && cuda_state::lane_divergent (dev, sm, wp, ln);
      });
}

static PyObject *
gdbpy_cuda_warp_method_valid_lanes (PyObject *self, PyObject *args)
{
  return gdbpy_cuda_warp_lanes_filtered (
      self, [] (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln) -> bool {
	return cuda_state::warp_valid (dev, sm, wp)
	       && cuda_state::lane_valid (dev, sm, wp, ln);
      });
}

static PyObject *
gdbpy_cuda_warp_method_read_shared_memory (PyObject *self, PyObject *args, PyObject *kwargs)
{
  auto warp = gdbpy_cuda_warp (self);
  if (!warp)
    return nullptr;

  uint64_t size = 0;
  uint64_t address = 0;
  static const char *kwlist[] = { "address", "size", nullptr };

  if (!PyArg_ParseTupleAndKeywords (args, kwargs, "KK", (char **)kwlist, &address, &size))
    return nullptr;

  try
    {
      gdbpy_ref<> buffer (PyBytes_FromStringAndSize (nullptr, size));
      if (buffer == nullptr)
	return nullptr;
      cuda_debugapi::read_shared_memory (
	  warp->dev, warp->sm, warp->wp, address,
	  PyBytes_AS_STRING (buffer.get ()), size);
      return buffer.release ();
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

static PyMethodDef gdbpy_cuda_warp_methods[] = {
  { "lanes", gdbpy_cuda_warp_method_lanes, METH_NOARGS,
    "Returns the list of cuda.Lane objects for a cuda.Warp" },
  { "active_lanes", gdbpy_cuda_warp_method_active_lanes, METH_NOARGS,
    "Returns the list of the active cuda.Lane objects for a cuda.Warp" },
  { "divergent_lanes", gdbpy_cuda_warp_method_divergent_lanes, METH_NOARGS,
    "Returns the list of the divergent cuda.Lane objects for a cuda.Warp" },
  { "valid_lanes", gdbpy_cuda_warp_method_valid_lanes, METH_NOARGS,
    "Returns the list of the valid cuda.Lane objects for a cuda.Warp" },
  { "read_shared_memory", (PyCFunction) gdbpy_cuda_warp_method_read_shared_memory,
    METH_VARARGS | METH_KEYWORDS, "Read shared memory from a cuda.Warp" },
  { nullptr, nullptr, 0, nullptr },
};

static PyObject *
gdbpy_cuda_warp_method_richcompare (PyObject *self, PyObject *other, int opid)
{
  if (opid != Py_EQ && opid != Py_NE)
    {
      PyErr_SetString (PyExc_TypeError, "Invalid comparison");
      return nullptr;
    }
  if (other == Py_None)
    {
      if (opid == Py_EQ)
	Py_RETURN_FALSE;
      if (opid == Py_NE)
	Py_RETURN_TRUE;
    }

  const auto obj_self = gdbpy_cuda_warp (self);
  if (obj_self == nullptr)
    return nullptr;

  const auto obj_other = gdbpy_cuda_warp (other);
  if (obj_other == nullptr)
    return nullptr;

  Py_RETURN_RICHCOMPARE ((obj_self->dev == obj_other->dev)
			     && (obj_self->sm == obj_other->sm)
			     && (obj_self->wp == obj_other->wp),
			 true, opid);
}

static PyObject *
gdbpy_cuda_warp_method_repr (PyObject *self)
{
  const auto warp = (gdbpy_cuda_warp_object *)self;

  return PyUnicode_FromFormat ("<cuda.Warp dev%u.sm%u.wp%u>", warp->dev,
			       warp->sm, warp->wp);
}

static PyObject *
gdbpy_cuda_warp_create (uint32_t dev, uint32_t sm, uint32_t wp)
{
  if (!gdbpy_check_warp (dev, sm, wp))
    return nullptr;

  auto *self = PyObject_New (gdbpy_cuda_warp_object, &gdbpy_cuda_warp_type);
  if (self == nullptr)
    return nullptr;

  self->dev = dev;
  self->sm = sm;
  self->wp = wp;

  cuda_trace_domain (CUDA_TRACE_PYTHON,
		     "cuda_warp_create <dev%u.sm%u.wp%u> @ %p", dev, sm, wp, self);
  return (PyObject *)self;
}

static void
gdbpy_cuda_warp_finalize (PyObject *self)
{
  auto obj = (gdbpy_cuda_warp_object *)self;
  cuda_trace_domain (CUDA_TRACE_PYTHON,
		     "cuda_warp_finalize <dev%u.sm%u.wp%u> @ %p", obj->dev,
		     obj->sm, obj->wp, obj);
}

static bool
gdbpy_cuda_warp_type_init (PyObject *module)
{
  auto flags = Py_TPFLAGS_DEFAULT;

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 10
  // Don't allow users to create their own instances
  flags |= Py_TPFLAGS_DISALLOW_INSTANTIATION;
#endif

  gdbpy_cuda_warp_type.tp_name = "cuda.Warp";
  gdbpy_cuda_warp_type.tp_doc = PyDoc_STR ("CUDA Warp");
  gdbpy_cuda_warp_type.tp_basicsize = sizeof (gdbpy_cuda_warp_object);
  gdbpy_cuda_warp_type.tp_itemsize = 0;
  gdbpy_cuda_warp_type.tp_flags = flags;
  gdbpy_cuda_warp_type.tp_new = PyType_GenericNew;
  gdbpy_cuda_warp_type.tp_getset = gdbpy_cuda_warp_getset;
  gdbpy_cuda_warp_type.tp_methods = gdbpy_cuda_warp_methods;

  gdbpy_cuda_warp_type.tp_repr = gdbpy_cuda_warp_method_repr;
  gdbpy_cuda_warp_type.tp_finalize = gdbpy_cuda_warp_finalize;
  gdbpy_cuda_warp_type.tp_richcompare = gdbpy_cuda_warp_method_richcompare;

  return gdbpy_cuda_install_type_in_module (module, "Warp",
					    &gdbpy_cuda_warp_type);
}

//
// cuda.Lane type
//
// Exposing cuda_lane attributes to python
//

static PyObject *
cuda_lane_get_attribute (PyObject *self,
			 PyObject *(*closure) (uint32_t dev, uint32_t sm,
					       uint32_t wp, uint32_t ln))
{
  try
    {
      const auto obj = gdbpy_cuda_lane (self);
      if (!obj)
	return nullptr;

      // If the warp isn't currently valid, return None
      if (!cuda_state::warp_valid (obj->dev, obj->sm, obj->wp))
	Py_RETURN_NONE;

      // If the lane isn't currently valid, return None
      if (!cuda_state::lane_valid (obj->dev, obj->sm, obj->wp, obj->ln))
	Py_RETURN_NONE;

      return closure (obj->dev, obj->sm, obj->wp, obj->ln);
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

static PyObject *
gdbpy_cuda_lane_get_valid (PyObject *self, void *closure)
{
  try
    {
      const auto obj = gdbpy_cuda_lane (self);
      if (!obj)
	return nullptr;

      if (!cuda_state::warp_valid (obj->dev, obj->sm, obj->wp)
	  || !cuda_state::lane_valid (obj->dev, obj->sm, obj->wp, obj->ln))
	Py_RETURN_FALSE;

      Py_RETURN_TRUE;
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

static PyObject *
gdbpy_cuda_lane_get_exception (PyObject *self, void *closure)
{
  return cuda_lane_get_attribute (
      self,
      [] (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln) -> PyObject * {
	const auto exception
	    = cuda_state::lane_get_exception (dev, sm, wp, ln);
	if (exception == CUDBG_EXCEPTION_NONE)
	  Py_RETURN_NONE;
	return Py_BuildValue ("I", exception);
      });
}

// cuda.Lane attributes table

static gdb_PyGetSetDef gdbpy_cuda_lane_getset[] = {
  { "device_id", cuda_get_device_id<gdbpy_cuda_lane_object>, nullptr, "Device ID",
    nullptr },
  { "sm_id", cuda_get_sm_id<gdbpy_cuda_lane_object>, nullptr, "Sm ID",
    nullptr },
  { "warp_id", cuda_get_warp_id<gdbpy_cuda_lane_object>, nullptr, "Warp ID",
    nullptr },
  { "lane_id", cuda_get_lane_id<gdbpy_cuda_lane_object>, nullptr, "Lane ID",
    nullptr },
  { "is_valid", gdbpy_cuda_lane_get_valid, nullptr, "Valid", nullptr },
  { "is_active", cuda_lane_attr_wrapper<bool, cuda_state::lane_active>, nullptr,
    "Active", nullptr },
  { "is_divergent", cuda_lane_attr_wrapper<bool, cuda_state::lane_active>,
    nullptr, "Active", nullptr },
  { "pc", cuda_lane_attr_wrapper<uint64_t, cuda_state::lane_get_pc>, nullptr,
    "PC", nullptr },
  { "exception", gdbpy_cuda_lane_get_exception, nullptr, "Exception",
    nullptr },
  { "cc_register",
    cuda_lane_attr_wrapper<uint32_t, cuda_state::lane_get_cc_register>,
    nullptr, "CC register", nullptr },
  { "thread_idx",
    cuda_lane_attr_wrapper<const CuDim3 &, cuda_state::lane_get_thread_idx>,
    nullptr, "Thread index", nullptr },
  { nullptr },
};

static PyObject *
gdbpy_cuda_lane_method_richcompare (PyObject *self, PyObject *other, int opid)
{
  if (opid != Py_EQ && opid != Py_NE)
    {
      PyErr_SetString (PyExc_TypeError, "Invalid comparison");
      return nullptr;
    }
  if (other == Py_None)
    {
      if (opid == Py_EQ)
	Py_RETURN_FALSE;
      if (opid == Py_NE)
	Py_RETURN_TRUE;
    }

  const auto obj_self = gdbpy_cuda_lane (self);
  if (obj_self == nullptr)
    return nullptr;

  const auto obj_other = gdbpy_cuda_lane (other);
  if (obj_other == nullptr)
    return nullptr;

  Py_RETURN_RICHCOMPARE ((obj_self->dev == obj_other->dev)
			     && (obj_self->sm == obj_other->sm)
			     && (obj_self->wp == obj_other->wp)
			     && (obj_self->ln == obj_other->ln),
			 true, opid);
}

static PyObject *
gdbpy_cuda_lane_method_repr (PyObject *self)
{
  gdbpy_cuda_lane_object *lane = (gdbpy_cuda_lane_object *)self;

  return PyUnicode_FromFormat ("<cuda.Lane dev%u.sm%u.wp%u.ln%u>", lane->dev,
			       lane->sm, lane->wp, lane->ln);
}

static PyObject *
gdbpy_cuda_lane_method_call_depth (PyObject *self, PyObject *args)
{
  return cuda_lane_get_attribute (
      self,
      [] (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln) -> PyObject * {
	return Py_BuildValue (
	    "I", cuda_state::lane_get_call_depth (dev, sm, wp, ln));
      });
}

static PyObject *
gdbpy_cuda_lane_method_read_generic_memory (PyObject *self, PyObject *args, PyObject *kwargs)
{
  auto lane = gdbpy_cuda_lane (self);
  if (!lane)
    return nullptr;

  uint64_t size = 0;
  uint64_t address = 0;
  static const char *kwlist[] = { "address", "size", nullptr };

  if (!PyArg_ParseTupleAndKeywords (args, kwargs, "KK", (char **)kwlist, &address, &size))
    return nullptr;

  try
    {
      gdbpy_ref<> buffer (PyBytes_FromStringAndSize (nullptr, size));
      if (buffer == nullptr)
	return nullptr;
      cuda_debugapi::read_generic_memory (
	  lane->dev, lane->sm, lane->wp, lane->ln, address,
	  PyBytes_AS_STRING (buffer.get ()), size);
      return buffer.release ();
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

static PyObject *
gdbpy_cuda_lane_method_read_local_memory (PyObject *self, PyObject *args, PyObject *kwargs)
{
  auto lane = gdbpy_cuda_lane (self);
  if (!lane)
    return nullptr;

  uint64_t size = 0;
  uint64_t address = 0;
  static const char *kwlist[] = { "address", "size", nullptr };

  if (!PyArg_ParseTupleAndKeywords (args, kwargs, "KK", (char **)kwlist, &address, &size))
    return nullptr;

  try
    {
      gdbpy_ref<> buffer (PyBytes_FromStringAndSize (nullptr, size));
      if (buffer == nullptr)
	return nullptr;

      cuda_debugapi::read_local_memory (
	  lane->dev, lane->sm, lane->wp, lane->ln, address,
	  PyBytes_AS_STRING (buffer.get ()), size);
      return buffer.release ();
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

static PyObject *
gdbpy_cuda_lane_method_logical (PyObject *self, PyObject *args)
{
  return cuda_lane_get_attribute (
      self,
      [] (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln) -> PyObject * {
	try
	  {
	    cuda_coords filter{ dev,
				sm,
				wp,
				ln,
				CUDA_WILDCARD,
				CUDA_WILDCARD,
				CUDA_WILDCARD_DIM,
				CUDA_WILDCARD_DIM,
				CUDA_WILDCARD_DIM,
				CUDA_WILDCARD_DIM };
	    cuda_coord_set<cuda_coord_set_type::threads,
			   select_valid | select_sngl>
		coord{ filter };
	    if (coord.size () == 0)
	      {
		PyErr_SetString (PyExc_RuntimeError, "Invalid coordinates");
		return nullptr;
	      }
	    return gdbpy_cuda_coords_logical_create (
		coord.begin ()->logical ());
	  }
	catch (const gdb_exception &e)
	  {
	    PyErr_SetString (PyExc_RuntimeError, e.what ());
	    return nullptr;
	  }
      });
}

static PyObject *
gdbpy_cuda_lane_method_physical (PyObject *self, PyObject *args)
{
  return cuda_lane_get_attribute (
      self,
      [] (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln) -> PyObject * {
	try
	  {
	    cuda_coords filter{ dev,
				sm,
				wp,
				ln,
				CUDA_WILDCARD,
				CUDA_WILDCARD,
				CUDA_WILDCARD_DIM,
				CUDA_WILDCARD_DIM,
				CUDA_WILDCARD_DIM,
				CUDA_WILDCARD_DIM };
	    cuda_coord_set<cuda_coord_set_type::threads,
			   select_valid | select_sngl>
		coord{ filter };
	    if (coord.size () == 0)
	      {
		PyErr_SetString (PyExc_RuntimeError, "Invalid coordinates");
		return nullptr;
	      }
	    return gdbpy_cuda_coords_physical_create (
		coord.begin ()->physical ());
	  }
	catch (const gdb_exception &e)
	  {
	    PyErr_SetString (PyExc_RuntimeError, e.what ());
	    return nullptr;
	  }
      });
}

static PyMethodDef gdbpy_cuda_lane_methods[] = {
  { "call_depth", gdbpy_cuda_lane_method_call_depth, METH_NOARGS,
    "Returns the call depth of the a cuda.Lane" },
  { "read_generic_memory", (PyCFunction) gdbpy_cuda_lane_method_read_generic_memory,
    METH_VARARGS | METH_KEYWORDS, "Reads from generic memory" },
  { "read_local_memory", (PyCFunction) gdbpy_cuda_lane_method_read_local_memory,
    METH_VARARGS | METH_KEYWORDS, "Reads from local memory" },
  { "logical", gdbpy_cuda_lane_method_logical, METH_NOARGS,
    "Returns the logical coordinates of the a cuda.Lane" },
  { "physical", gdbpy_cuda_lane_method_physical, METH_NOARGS,
    "Returns the physical coordinates of the a cuda.Lane" },
  { nullptr, nullptr, 0, nullptr },
};

static PyObject *
gdbpy_cuda_lane_create (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln)
{
  if (!gdbpy_check_lane (dev, sm, wp, ln))
    return nullptr;

  auto *self = PyObject_New (gdbpy_cuda_lane_object, &gdbpy_cuda_lane_type);
  if (self == nullptr)
    return nullptr;

  self->dev = dev;
  self->sm = sm;
  self->wp = wp;
  self->ln = ln;

  cuda_trace_domain (CUDA_TRACE_PYTHON,
		     "cuda_lane_create <dev%u.sm%u.wp%u.ln%u> @ %p", dev, sm,
		     wp, ln, self);

  return (PyObject *)self;
}

static void
gdbpy_cuda_lane_finalize (PyObject *self)
{
  auto obj = (gdbpy_cuda_lane_object *)self;
  cuda_trace_domain (CUDA_TRACE_PYTHON,
		     "cuda_lane_finalize <dev%u.sm%u.wp%u.ln%u> @ %p",
		     obj->dev, obj->sm, obj->wp, obj->ln, obj);
}

static bool
gdbpy_cuda_lane_type_init (PyObject *module)
{
  auto flags = Py_TPFLAGS_DEFAULT;

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 10
  // Don't allow users to create their own instances
  flags |= Py_TPFLAGS_DISALLOW_INSTANTIATION;
#endif

  gdbpy_cuda_lane_type.tp_name = "cuda.Lane";
  gdbpy_cuda_lane_type.tp_doc = PyDoc_STR ("CUDA Lane");
  gdbpy_cuda_lane_type.tp_basicsize = sizeof (gdbpy_cuda_lane_object);
  gdbpy_cuda_lane_type.tp_itemsize = 0;
  gdbpy_cuda_lane_type.tp_flags = flags;
  gdbpy_cuda_lane_type.tp_new = PyType_GenericNew;
  gdbpy_cuda_lane_type.tp_getset = gdbpy_cuda_lane_getset;
  gdbpy_cuda_lane_type.tp_repr = gdbpy_cuda_lane_method_repr;
  gdbpy_cuda_lane_type.tp_methods = gdbpy_cuda_lane_methods;
  gdbpy_cuda_lane_type.tp_finalize = gdbpy_cuda_lane_finalize;
  gdbpy_cuda_lane_type.tp_richcompare = gdbpy_cuda_lane_method_richcompare;

  return gdbpy_cuda_install_type_in_module (module, "Lane",
					    &gdbpy_cuda_lane_type);
}

/* gdb.cuda.CoordsPhysical */

typedef struct
{
  PyObject_HEAD uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
} gdbpy_cuda_coords_physical_object;

static PyObject *
gdbpy_cuda_coords_physical_get_dev (PyObject *self, void *closure)
{
  const auto coords = (gdbpy_cuda_coords_physical_object *)self;

  return PyLong_FromUnsignedLong (coords->dev);
}

static PyObject *
gdbpy_cuda_coords_physical_get_sm (PyObject *self, void *closure)
{
  const auto coords = (gdbpy_cuda_coords_physical_object *)self;

  return PyLong_FromUnsignedLong (coords->sm);
}

static PyObject *
gdbpy_cuda_coords_physical_get_wp (PyObject *self, void *closure)
{
  const auto coords = (gdbpy_cuda_coords_physical_object *)self;

  return PyLong_FromUnsignedLong (coords->wp);
}

static PyObject *
gdbpy_cuda_coords_physical_get_ln (PyObject *self, void *closure)
{
  const auto coords = (gdbpy_cuda_coords_physical_object *)self;

  return PyLong_FromUnsignedLong (coords->ln);
}

static gdb_PyGetSetDef gdbpy_cuda_coords_physical_getset[] = {
  { "device_id", gdbpy_cuda_coords_physical_get_dev, nullptr, "Device ID", nullptr },
  { "sm_id", gdbpy_cuda_coords_physical_get_sm, nullptr, "SM ID", nullptr },
  { "warp_id", gdbpy_cuda_coords_physical_get_wp, nullptr, "Warp ID", nullptr },
  { "lane_id", gdbpy_cuda_coords_physical_get_ln, nullptr, "Lane ID", nullptr },
  { nullptr },
};

static PyObject *
gdbpy_cuda_coords_physical_repr (PyObject *self)
{
  const auto coords = (gdbpy_cuda_coords_physical_object *)self;

  return PyUnicode_FromFormat ("<dev%u.sm%u.wp%u.ln%u>", coords->dev,
			       coords->sm, coords->wp, coords->ln);
}

static PyTypeObject gdbpy_cuda_coords_physical_type
    = { PyVarObject_HEAD_INIT (nullptr, 0) };

static PyObject *
gdbpy_cuda_coords_physical_richcompare (PyObject *self, PyObject *other,
					int opid)
{
  if (opid != Py_EQ && opid != Py_NE)
    {
      PyErr_SetString (PyExc_TypeError, "Invalid comparison");
      return nullptr;
    }
  if (other == Py_None)
    {
      if (opid == Py_EQ)
	Py_RETURN_FALSE;
      if (opid == Py_NE)
	Py_RETURN_TRUE;
    }
  if (!PyObject_TypeCheck (self, &gdbpy_cuda_coords_physical_type))
    {
      PyErr_SetString (PyExc_TypeError, "Invalid type");
      return nullptr;
    }
  if (!PyObject_TypeCheck (other, &gdbpy_cuda_coords_physical_type))
    {
      PyErr_SetString (PyExc_TypeError, "Invalid type");
      return nullptr;
    }

  auto a = (gdbpy_cuda_coords_physical_object *)self;
  auto b = (gdbpy_cuda_coords_physical_object *)other;
  Py_RETURN_RICHCOMPARE ((a->dev == b->dev) && (a->sm == b->sm)
			     && (a->wp == b->wp) && (a->ln == b->ln),
			 true, opid);
}

static int
gdbpy_cuda_coords_physical_init (PyObject *self, PyObject *args,
				 PyObject *kwds)
{
  static const char *kwlist[] = { "device_id", "sm_id", "warp_id", "lane_id", nullptr };

  // PyArg_ParseTupleAndKeywords does not modify fields corresponding
  // to missing arguments, so we need to initialize them to 0
  auto coords = (gdbpy_cuda_coords_physical_object *)self;
  coords->dev = 0;
  coords->sm = 0;
  coords->wp = 0;
  coords->ln = 0;

  if (!PyArg_ParseTupleAndKeywords (args, kwds, "|IIII", (char **)kwlist,
				    &coords->dev, &coords->sm, &coords->wp,
				    &coords->ln))
    return -1;
  return 0;
}

static PyObject *
gdbpy_cuda_coords_physical_create (const cuda_coords_physical &coords)
{
  gdbpy_ref<gdbpy_cuda_coords_physical_object> self (PyObject_New (gdbpy_cuda_coords_physical_object,
			    &gdbpy_cuda_coords_physical_type));
  if (self == nullptr)
    return nullptr;

  self->dev = coords.dev ();
  self->sm = coords.sm ();
  self->wp = coords.wp ();
  self->ln = coords.ln ();

  return (PyObject *)self.release ();
}

static bool
gdbpy_cuda_coords_physical_type_init (PyObject *module)
{
  gdbpy_cuda_coords_physical_type.tp_name = "cuda.CoordsPhysical";
  gdbpy_cuda_coords_physical_type.tp_doc
      = PyDoc_STR ("CUDA Physical Coordinates");
  gdbpy_cuda_coords_physical_type.tp_basicsize
      = sizeof (gdbpy_cuda_coords_physical_object);
  gdbpy_cuda_coords_physical_type.tp_itemsize = 0;
  gdbpy_cuda_coords_physical_type.tp_flags = Py_TPFLAGS_DEFAULT;
  gdbpy_cuda_coords_physical_type.tp_new = PyType_GenericNew;
  gdbpy_cuda_coords_physical_type.tp_getset
      = gdbpy_cuda_coords_physical_getset;
  gdbpy_cuda_coords_physical_type.tp_init = gdbpy_cuda_coords_physical_init;
  gdbpy_cuda_coords_physical_type.tp_repr = gdbpy_cuda_coords_physical_repr;
  gdbpy_cuda_coords_physical_type.tp_richcompare
      = gdbpy_cuda_coords_physical_richcompare;

  return gdbpy_cuda_install_type_in_module (module, "CoordsPhysical",
					    &gdbpy_cuda_coords_physical_type);
}

/* gdb.cuda.CoordsLogical */

typedef struct
{
  PyObject_HEAD
  uint64_t kernel_id;
  uint32_t dev_id;
  uint64_t grid_id;
  PyObject *cluster_idx;
  PyObject *cluster_dim;
  PyObject *block_idx;
  PyObject *thread_idx;
} gdbpy_cuda_coords_logical_object;

static PyObject *
gdbpy_cuda_coords_logical_get_kernel_id (PyObject *self, void *closure)
{
  const auto coords = (gdbpy_cuda_coords_logical_object *)self;

  return PyLong_FromUnsignedLong (coords->kernel_id);
}

static PyObject *
gdbpy_cuda_coords_logical_get_dev_id (PyObject *self, void *closure)
{
  const auto coords = (gdbpy_cuda_coords_logical_object *)self;

  return PyLong_FromUnsignedLong (coords->dev_id);
}

static PyObject *
gdbpy_cuda_coords_logical_get_grid_id (PyObject *self, void *closure)
{
  const auto coords = (gdbpy_cuda_coords_logical_object *)self;

  return PyLong_FromUnsignedLong (coords->grid_id);
}

static PyObject *
gdbpy_cuda_coords_logical_get_cluster_idx (PyObject *self, void *closure)
{
  const auto coords = (gdbpy_cuda_coords_logical_object *)self;

  Py_INCREF (coords->cluster_idx);
  return coords->cluster_idx;
}

static PyObject *
gdbpy_cuda_coords_logical_get_cluster_dim (PyObject *self, void *closure)
{
  const auto coords = (gdbpy_cuda_coords_logical_object *)self;

  Py_INCREF (coords->cluster_dim);
  return coords->cluster_dim;
}

static PyObject *
gdbpy_cuda_coords_logical_get_block_idx (PyObject *self, void *closure)
{
  const auto coords = (gdbpy_cuda_coords_logical_object *)self;

  Py_INCREF (coords->block_idx);
  return coords->block_idx;
}

static PyObject *
gdbpy_cuda_coords_logical_get_thread_idx (PyObject *self, void *closure)
{
  const auto coords = (gdbpy_cuda_coords_logical_object *)self;

  Py_INCREF (coords->thread_idx);
  return coords->thread_idx;
}

static gdb_PyGetSetDef gdbpy_cuda_coords_logical_getset[] = {
  { "block_idx", gdbpy_cuda_coords_logical_get_block_idx, nullptr,
    "Block Index", nullptr },
  { "cluster_idx", gdbpy_cuda_coords_logical_get_cluster_idx, nullptr,
    "Cluster Index", nullptr },
  { "cluster_dim", gdbpy_cuda_coords_logical_get_cluster_dim, nullptr,
    "Cluster Dimension", nullptr },
  { "dev_id", gdbpy_cuda_coords_logical_get_dev_id, nullptr, "Grid ID",
    nullptr },
  { "grid_id", gdbpy_cuda_coords_logical_get_grid_id, nullptr, "Grid ID",
    nullptr },
  { "kernel_id", gdbpy_cuda_coords_logical_get_kernel_id, nullptr, "Kernel ID",
    nullptr },
  { "thread_idx", gdbpy_cuda_coords_logical_get_thread_idx, nullptr,
    "Thread Index", nullptr },
  { nullptr },
};

static PyTypeObject gdbpy_cuda_coords_logical_type
    = { PyVarObject_HEAD_INIT (nullptr, 0) };

static PyObject *
gdbpy_cuda_coords_logical_richcompare (PyObject *self, PyObject *other,
				       int opid)
{
  if (opid != Py_EQ && opid != Py_NE)
    {
      PyErr_SetString (PyExc_TypeError, "Invalid comparison");
      return nullptr;
    }
  if (other == Py_None)
    {
      if (opid == Py_EQ)
	Py_RETURN_FALSE;
      if (opid == Py_NE)
	Py_RETURN_TRUE;
    }
  if (!PyObject_TypeCheck (self, &gdbpy_cuda_coords_logical_type))
    {
      PyErr_SetString (PyExc_TypeError, "Invalid type");
      return nullptr;
    }
  if (!PyObject_TypeCheck (other, &gdbpy_cuda_coords_logical_type))
    {
      PyErr_SetString (PyExc_TypeError, "Invalid type");
      return nullptr;
    }

  const auto a = (gdbpy_cuda_coords_logical_object *)self;
  const auto b = (gdbpy_cuda_coords_logical_object *)other;

  auto block_idx_p
      = gdbpy_cuda_cu_dim3_richcompare (a->block_idx, b->block_idx, opid);
  if ((block_idx_p == nullptr) || (block_idx_p == Py_False))
    return block_idx_p;

  auto thread_idx_p
      = gdbpy_cuda_cu_dim3_richcompare (a->thread_idx, b->thread_idx, opid);
  if ((thread_idx_p == nullptr) || (thread_idx_p == Py_False))
    return thread_idx_p;

  auto cluster_idx_p
      = gdbpy_cuda_cu_dim3_richcompare (a->cluster_idx, b->cluster_idx, opid);
  if ((cluster_idx_p == nullptr) || (cluster_idx_p == Py_False))
    return cluster_idx_p;

  auto cluster_dim_p
      = gdbpy_cuda_cu_dim3_richcompare (a->cluster_dim, b->cluster_dim, opid);
  if ((cluster_dim_p == nullptr) || (cluster_dim_p == Py_False))
    return cluster_dim_p;

  Py_RETURN_RICHCOMPARE ((a->kernel_id == b->kernel_id)
			     && (a->grid_id == b->grid_id),
			 true, opid);
}

static PyObject *
gdbpy_cuda_coords_logical_repr (PyObject *self)
{
  const auto lcoords = (gdbpy_cuda_coords_logical_object *)self;

  const auto cluster = (gdbpy_cuda_cu_dim3_object *)lcoords->cluster_idx;
  gdb_assert (cluster);
  const auto cluster_dim = (gdbpy_cuda_cu_dim3_object *)lcoords->cluster_dim;
  gdb_assert (cluster_dim);
  const auto block = (gdbpy_cuda_cu_dim3_object *)lcoords->block_idx;
  gdb_assert (block);
  const auto thread = (gdbpy_cuda_cu_dim3_object *)lcoords->thread_idx;
  gdb_assert (thread);

  cuda_coords coord{ CUDA_INVALID,
		     CUDA_INVALID,
		     CUDA_INVALID,
		     CUDA_INVALID,
		     lcoords->kernel_id,
		     lcoords->grid_id,
		     CuDim3{ cluster->x, cluster->y, cluster->z },
		     CuDim3{ cluster_dim->x, cluster_dim->y, cluster_dim->z },
		     CuDim3{ block->x, block->y, block->z },
		     CuDim3{ thread->x, thread->y, thread->z } };

  const auto repr = coord.to_string ();
  return PyUnicode_FromFormat ("<%s>", repr.c_str ());
}

static PyObject *
gdbpy_cuda_coords_logical_create (const cuda_coords_logical &coords)
{
  gdbpy_ref<gdbpy_cuda_coords_logical_object> self (PyObject_New (gdbpy_cuda_coords_logical_object,
			    &gdbpy_cuda_coords_logical_type));
  if (self == nullptr)
    return nullptr;

  const auto kernel_id = coords.kernelId ();
  const auto kernel = cuda_state::find_kernel_by_kernel_id (kernel_id);
  if (!kernel)
    {
      PyErr_Format (PyExc_RuntimeError, "Invalid kernel_id %lu", kernel_id);
      return nullptr;
    }

  self->kernel_id = kernel_id;
  self->dev_id = kernel->dev_id ();
  self->grid_id = kernel->grid_id ();
  self->cluster_idx = gdbpy_cuda_cu_dim3_create (coords.clusterIdx ());
  self->cluster_dim = gdbpy_cuda_cu_dim3_create (coords.clusterDim ());
  self->block_idx = gdbpy_cuda_cu_dim3_create (coords.blockIdx ());
  self->thread_idx = gdbpy_cuda_cu_dim3_create (coords.threadIdx ());

  return (PyObject *)self.release ();
}

static void
gdbpy_cuda_coords_logical_finalize (PyObject *self)
{
  PyObject *error_type, *error_value, *error_traceback;

  /* Save the current exception, if any. */
  PyErr_Fetch (&error_type, &error_value, &error_traceback);

  auto coords = (gdbpy_cuda_coords_logical_object *)self;
  Py_XDECREF (coords->thread_idx);
  Py_XDECREF (coords->block_idx);
  Py_XDECREF (coords->cluster_idx);
  Py_XDECREF (coords->cluster_dim);

  /* Restore the saved exception. */
  PyErr_Restore (error_type, error_value, error_traceback);
}

static bool
gdbpy_cuda_coords_logical_type_init (PyObject *module)
{
  auto flags = Py_TPFLAGS_DEFAULT;

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 10
  // Don't allow users to create their own instances
  flags |= Py_TPFLAGS_DISALLOW_INSTANTIATION;
#endif

  gdbpy_cuda_coords_logical_type.tp_name = "cuda.CoordsLogical";
  gdbpy_cuda_coords_logical_type.tp_doc
      = PyDoc_STR ("CUDA Logical Coordinates");
  gdbpy_cuda_coords_logical_type.tp_basicsize
      = sizeof (gdbpy_cuda_coords_logical_object);
  gdbpy_cuda_coords_logical_type.tp_itemsize = 0;
  gdbpy_cuda_coords_logical_type.tp_flags = flags;
  gdbpy_cuda_coords_logical_type.tp_new = PyType_GenericNew;
  gdbpy_cuda_coords_logical_type.tp_finalize
      = gdbpy_cuda_coords_logical_finalize;
  gdbpy_cuda_coords_logical_type.tp_getset = gdbpy_cuda_coords_logical_getset;
  gdbpy_cuda_coords_logical_type.tp_repr = gdbpy_cuda_coords_logical_repr;
  gdbpy_cuda_coords_logical_type.tp_richcompare
      = gdbpy_cuda_coords_logical_richcompare;
  return gdbpy_cuda_install_type_in_module (module, "CoordsLogical",
					    &gdbpy_cuda_coords_logical_type);
}

/* gdb.cuda methods */

static PyObject *
gdbpy_cuda_execute_internal_command (PyObject *self, PyObject *args)
{
  const char *command;
  unsigned long long buffer_size = DEFAULT_BUFFER_SIZE;

  if (!PyArg_ParseTuple (args, "s|K", &command, &buffer_size))
    return nullptr;

  try
    {
      gdb::unique_xmalloc_ptr<char> buffer ((char *)xmalloc (buffer_size));
      if (!cuda_debugapi::execute_internal_command (command, buffer.get (),
						    buffer_size))
	{
	  PyErr_SetString (PyExc_RuntimeError, "CUDA Debug API Error");
	  return nullptr;
	}

      return Py_BuildValue ("s", buffer.get ());
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

static PyObject *
gdbpy_cuda_get_focus_physical (PyObject *self, PyObject *args)
{
  const auto &focus = cuda_current_focus::get ();
  if (!focus.valid ())
    Py_RETURN_NONE;

  const auto &physical = focus.physical ();
  return gdbpy_cuda_coords_physical_create (physical);
}

static PyObject *
gdbpy_cuda_set_focus_physical (PyObject *self, PyObject *args)
{
  try
    {
      PyObject *coords;
      if (!PyArg_ParseTuple (args, "O", &coords))
	return nullptr;

      if (!PyObject_TypeCheck (coords, &gdbpy_cuda_coords_physical_type))
	{
	  PyErr_SetString (PyExc_TypeError,
			   "Invalid physical coordinate type");
	  return nullptr;
	}
      const auto pcoords = (gdbpy_cuda_coords_physical_object *)coords;
      cuda_coords filter{ pcoords->dev,	     pcoords->sm,
			  pcoords->wp,	     pcoords->ln,
			  CUDA_WILDCARD,     CUDA_WILDCARD,
			  CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM,
			  CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM };
      cuda_coord_set<cuda_coord_set_type::threads, select_valid | select_sngl>
	  coord{ filter };
      if (coord.size () == 0)
	{
	  PyErr_SetString (PyExc_RuntimeError, "Invalid coordinates");
	  return nullptr;
	}
      switch_to_cuda_thread (*coord.begin ());
      Py_RETURN_NONE;
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

static PyObject *
gdbpy_cuda_get_focus_logical (PyObject *self, PyObject *args)
{
  const auto &focus = cuda_current_focus::get ();
  if (!focus.valid ())
    Py_RETURN_NONE;

  const auto &logical = focus.logical ();
  return gdbpy_cuda_coords_logical_create (logical);
}

static PyObject *
gdbpy_cuda_set_focus_logical (PyObject *self, PyObject *args)
{
  try
    {
      PyObject *coords;
      if (!PyArg_ParseTuple (args, "O", &coords))
	return nullptr;

      if (!PyObject_TypeCheck (coords, &gdbpy_cuda_coords_logical_type))
	{
	  PyErr_SetString (PyExc_TypeError, "Invalid logical coordinate type");
	  return nullptr;
	}
      const auto lcoords = (gdbpy_cuda_coords_logical_object *)coords;

      const auto cluster = (gdbpy_cuda_cu_dim3_object *)lcoords->cluster_idx;
      gdb_assert (cluster);
      const auto cluster_dim = (gdbpy_cuda_cu_dim3_object *)lcoords->cluster_dim;
      gdb_assert (cluster_dim);
      const auto block = (gdbpy_cuda_cu_dim3_object *)lcoords->block_idx;
      gdb_assert (block);
      const auto thread = (gdbpy_cuda_cu_dim3_object *)lcoords->thread_idx;
      gdb_assert (thread);

      cuda_coords filter{ CUDA_WILDCARD,
			  CUDA_WILDCARD,
			  CUDA_WILDCARD,
			  CUDA_WILDCARD,
			  lcoords->kernel_id,
			  lcoords->grid_id,
			  CuDim3{ cluster->x, cluster->y, cluster->z },
			  CuDim3{ cluster_dim->x, cluster_dim->y, cluster_dim->z },
			  CuDim3{ block->x, block->y, block->z },
			  CuDim3{ thread->x, thread->y, thread->z } };
      cuda_coord_set<cuda_coord_set_type::threads, select_valid | select_sngl>
	  coord{ filter };
      if (coord.size () == 0)
	{
	  PyErr_SetString (PyExc_RuntimeError, "Invalid coordinates");
	  return nullptr;
	}
      switch_to_cuda_thread (*coord.begin ());
      Py_RETURN_NONE;
    }
  catch (const gdb_exception &e)
    {
      PyErr_SetString (PyExc_RuntimeError, e.what ());
      return nullptr;
    }
}

static PyObject *
gdbpy_cuda_get_devices (PyObject *self, PyObject *args)
{
  const uint32_t num_devices = cuda_state::get_num_devices ();

  gdbpy_ref<> list (PyList_New (num_devices));
  if (list == nullptr)
    return nullptr;

  for (auto i = 0; i < num_devices; i++)
    {
      auto device = gdbpy_cuda_device_create (i);
      if (device == nullptr)
	return nullptr;
      if (PyList_SetItem (list.get (), i, device) == -1)
	{
	  Py_DECREF (device);
	  return nullptr;
	}
    }

  return list.release ();
}

static PyObject *
gdbpy_cuda_get_device (PyObject *self, PyObject *args, PyObject *kwargs)
{
  uint32_t dev_id = 0;
  static const char *kwlist[] = { "device_id", nullptr };

  if (!PyArg_ParseTupleAndKeywords (args, kwargs, "I", (char **)kwlist,
				    &dev_id))
    return nullptr;

  return gdbpy_cuda_device_create (dev_id);
}

static PyObject *
gdbpy_cuda_get_sm (PyObject *self, PyObject *args, PyObject *kwargs)
{
  uint32_t dev_id = 0;
  uint32_t sm_id = 0;
  static const char *kwlist[] = { "device_id", "sm_id", nullptr };

  if (!PyArg_ParseTupleAndKeywords (args, kwargs, "II", (char **)kwlist,
				    &dev_id, &sm_id))
    return nullptr;

  return gdbpy_cuda_sm_create (dev_id, sm_id);
}

static PyObject *
gdbpy_cuda_get_warp (PyObject *self, PyObject *args, PyObject *kwargs)
{
  uint32_t dev_id = 0;
  uint32_t sm_id = 0;
  uint32_t wp_id = 0;
  static const char *kwlist[] = { "device_id", "sm_id", "warp_id", nullptr };

  if (!PyArg_ParseTupleAndKeywords (args, kwargs, "III", (char **)kwlist,
				    &dev_id, &sm_id, &wp_id))
    return nullptr;

  return gdbpy_cuda_warp_create (dev_id, sm_id, wp_id);
}

static PyObject *
gdbpy_cuda_get_lane (PyObject *self, PyObject *args, PyObject *kwargs)
{
  uint32_t dev_id = 0;
  uint32_t sm_id = 0;
  uint32_t wp_id = 0;
  uint32_t ln_id = 0;
  static const char *kwlist[] = { "device_id", "sm_id", "warp_id", "lane_id", nullptr };

  if (!PyArg_ParseTupleAndKeywords (args, kwargs, "IIII", (char **)kwlist,
				    &dev_id, &sm_id, &wp_id, &ln_id))
    return nullptr;

  return gdbpy_cuda_lane_create (dev_id, sm_id, wp_id, ln_id);
}

PyMODINIT_FUNC
gdbpy_cuda_init (void)
{
  auto module = PyModule_Create (&gdbpy_cuda_module);
  if (!module)
    return nullptr;

  if (!gdbpy_cuda_device_type_init (module)
      || !gdbpy_cuda_sm_type_init (module)
      || !gdbpy_cuda_warp_type_init (module)
      || !gdbpy_cuda_lane_type_init (module)
      || !gdbpy_cuda_cu_dim3_type_init (module)
      || !gdbpy_cuda_coords_physical_type_init (module)
      || !gdbpy_cuda_coords_logical_type_init (module))
    {
      Py_DECREF (module);
      return nullptr;
    }

  return module;
}

#endif
