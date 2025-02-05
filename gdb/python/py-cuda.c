/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2024 NVIDIA Corporation
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
#include "cuda/cuda-coords.h"
#include "cuda/cuda-state.h"
#include "gdbsupport/gdb_unique_ptr.h"

#ifdef HAVE_PYTHON

#include "structmember.h"

#define DEFAULT_BUFFER_SIZE 1024

/* gdb.cuda module components */

PyObject *gdbpy_cuda_execute_internal_command (PyObject *self, PyObject *args);
PyObject *gdbpy_cuda_get_focus_physical (PyObject *self, PyObject *args);
PyObject *gdbpy_cuda_get_focus_logical (PyObject *self, PyObject *args);
PyObject *gdbpy_cuda_get_devices (PyObject *self, PyObject *args);

static PyMethodDef gdbpy_cuda_methods[] = {
  {"execute_internal_command", gdbpy_cuda_execute_internal_command, METH_VARARGS,
   "execute internal command"},
  {"get_focus_physical", gdbpy_cuda_get_focus_physical, METH_NOARGS,
   "Returns the current focus as cuda.CoordsPhysical"},
  {"get_focus_logical", gdbpy_cuda_get_focus_logical, METH_NOARGS,
   "Returns the current focus as cuda.CoordsLogical"},
  {"get_devices", gdbpy_cuda_get_devices, METH_NOARGS,
   "Returns the list of devices as list of cuda.Device"},
  {nullptr, nullptr, 0, nullptr},
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

/* gdb.cuda.CuDim3 type */

typedef struct {
  PyObject_HEAD
  uint32_t x;
  uint32_t y;
  uint32_t z;
} gdbpy_cuda_cu_dim3_object;

static PyObject *gdbpy_cuda_cu_dim3_get_x (PyObject *self, void *closure)
{
  gdbpy_cuda_cu_dim3_object *dim3 = (gdbpy_cuda_cu_dim3_object *) self;

  return PyLong_FromUnsignedLong (dim3->x);
}

static PyObject *gdbpy_cuda_cu_dim3_get_y (PyObject *self, void *closure)
{
  gdbpy_cuda_cu_dim3_object *dim3 = (gdbpy_cuda_cu_dim3_object *) self;

  return PyLong_FromUnsignedLong (dim3->y);
}

static PyObject *gdbpy_cuda_cu_dim3_get_z (PyObject *self, void *closure)
{
  gdbpy_cuda_cu_dim3_object *dim3 = (gdbpy_cuda_cu_dim3_object *) self;

  return PyLong_FromUnsignedLong (dim3->z);
}

static gdb_PyGetSetDef gdbpy_cuda_cu_dim3_getset[] = {
  {"x", gdbpy_cuda_cu_dim3_get_x, nullptr, "x", nullptr},
  {"y", gdbpy_cuda_cu_dim3_get_y, nullptr, "y", nullptr},
  {"z", gdbpy_cuda_cu_dim3_get_z, nullptr, "z", nullptr},
  {nullptr},
};

static PyTypeObject gdbpy_cuda_cu_dim3_type = {
  PyVarObject_HEAD_INIT (nullptr, 0)
};

static PyObject *gdbpy_cuda_cu_dim3_init (const CuDim3 &dim3)
{
  gdbpy_cuda_cu_dim3_object *self = (gdbpy_cuda_cu_dim3_object *) PyObject_CallObject ((PyObject *) &gdbpy_cuda_cu_dim3_type, nullptr);

  if (self == nullptr)
    return nullptr;

  self->x = dim3.x;
  self->y = dim3.y;
  self->z = dim3.z;

  return (PyObject *) self;
}

static bool gdbpy_cuda_cu_dim3_type_init ()
{
  gdbpy_cuda_cu_dim3_type.tp_name = "cuda.CuDim3";
  gdbpy_cuda_cu_dim3_type.tp_doc = PyDoc_STR ("CUDA 3 dimensional coordinates");
  gdbpy_cuda_cu_dim3_type.tp_basicsize = sizeof (gdbpy_cuda_cu_dim3_object);
  gdbpy_cuda_cu_dim3_type.tp_itemsize = 0;
  gdbpy_cuda_cu_dim3_type.tp_flags = Py_TPFLAGS_DEFAULT;
  gdbpy_cuda_cu_dim3_type.tp_new = PyType_GenericNew;
  gdbpy_cuda_cu_dim3_type.tp_getset = gdbpy_cuda_cu_dim3_getset;

  return PyType_Ready (&gdbpy_cuda_cu_dim3_type) == 0;
}

/* gdb.cuda.Device type */

typedef struct {
  PyObject_HEAD
  uint32_t dev_id;
} gdbpy_cuda_device_object;

static PyObject *gdbpy_cuda_device_get_dev_id (PyObject *self, void *closure)
{
  gdbpy_cuda_device_object *device = (gdbpy_cuda_device_object *) self;

  return PyLong_FromUnsignedLong (device->dev_id);
}

static gdb_PyGetSetDef gdbpy_cuda_device_getset[] = {
  {"dev_id", gdbpy_cuda_device_get_dev_id, nullptr, "Device ID", nullptr},
  {nullptr},
};

static PyTypeObject gdbpy_cuda_device_type = {
  PyVarObject_HEAD_INIT (nullptr, 0)
};

static PyObject *gdbpy_cuda_device_init (uint32_t dev_id)
{
  gdbpy_cuda_device_object *self = (gdbpy_cuda_device_object *) PyObject_CallObject ((PyObject *) &gdbpy_cuda_device_type, nullptr);

  if (self == nullptr)
    return nullptr;

  self->dev_id = dev_id;

  return (PyObject *) self;
}

static bool gdbpy_cuda_device_type_init ()
{
  gdbpy_cuda_device_type.tp_name = "cuda.Device";
  gdbpy_cuda_device_type.tp_doc = PyDoc_STR ("CUDA Device");
  gdbpy_cuda_device_type.tp_basicsize = sizeof (gdbpy_cuda_device_object);
  gdbpy_cuda_device_type.tp_itemsize = 0;
  gdbpy_cuda_device_type.tp_flags = Py_TPFLAGS_DEFAULT;
  gdbpy_cuda_device_type.tp_new = PyType_GenericNew;
  gdbpy_cuda_device_type.tp_getset = gdbpy_cuda_device_getset;

  return PyType_Ready (&gdbpy_cuda_device_type) == 0;
}

/* gdb.cuda.CoordsPhysical */

typedef struct {
  PyObject_HEAD
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
} gdbpy_cuda_coords_physical_object;

static PyObject *gdbpy_cuda_coords_physical_get_dev (PyObject *self, void *closure)
{
  gdbpy_cuda_coords_physical_object *coords = (gdbpy_cuda_coords_physical_object *) self;

  return PyLong_FromUnsignedLong (coords->dev);
}

static PyObject *gdbpy_cuda_coords_physical_get_sm (PyObject *self, void *closure)
{
  gdbpy_cuda_coords_physical_object *coords = (gdbpy_cuda_coords_physical_object *) self;

  return PyLong_FromUnsignedLong (coords->sm);
}

static PyObject *gdbpy_cuda_coords_physical_get_wp (PyObject *self, void *closure)
{
  gdbpy_cuda_coords_physical_object *coords = (gdbpy_cuda_coords_physical_object *) self;

  return PyLong_FromUnsignedLong (coords->wp);
}

static PyObject *gdbpy_cuda_coords_physical_get_ln (PyObject *self, void *closure)
{
  gdbpy_cuda_coords_physical_object *coords = (gdbpy_cuda_coords_physical_object *) self;

  return PyLong_FromUnsignedLong (coords->ln);
}

static gdb_PyGetSetDef gdbpy_cuda_coords_physical_getset[] = {
  {"dev", gdbpy_cuda_coords_physical_get_dev, nullptr, "Device ID", nullptr},
  {"sm", gdbpy_cuda_coords_physical_get_sm, nullptr, "SM ID", nullptr},
  {"wp", gdbpy_cuda_coords_physical_get_wp, nullptr, "Warp ID", nullptr},
  {"ln", gdbpy_cuda_coords_physical_get_ln, nullptr, "Lane ID", nullptr},
  {nullptr},
};

static PyTypeObject gdbpy_cuda_coords_physical_type = {
  PyVarObject_HEAD_INIT (nullptr, 0)
};

static PyObject *gdbpy_cuda_coords_physical_init (const cuda_coords_physical &coords)
{
  gdbpy_cuda_coords_physical_object *self = (gdbpy_cuda_coords_physical_object *) PyObject_CallObject ((PyObject *) &gdbpy_cuda_coords_physical_type, nullptr);

  if (self == nullptr)
    return nullptr;

  self->dev = coords.dev ();
  self->sm = coords.sm ();
  self->wp = coords.wp ();
  self->ln = coords.ln ();

  return (PyObject *) self;
}

static bool gdbpy_cuda_coords_physical_type_init ()
{
  gdbpy_cuda_coords_physical_type.tp_name = "cuda.CoordsPhysical";
  gdbpy_cuda_coords_physical_type.tp_doc = PyDoc_STR ("CUDA Physical Coordinates");
  gdbpy_cuda_coords_physical_type.tp_basicsize = sizeof (gdbpy_cuda_coords_physical_object);
  gdbpy_cuda_coords_physical_type.tp_itemsize = 0;
  gdbpy_cuda_coords_physical_type.tp_flags = Py_TPFLAGS_DEFAULT;
  gdbpy_cuda_coords_physical_type.tp_new = PyType_GenericNew;
  gdbpy_cuda_coords_physical_type.tp_getset = gdbpy_cuda_coords_physical_getset;

  return PyType_Ready (&gdbpy_cuda_coords_physical_type) == 0;
}

/* gdb.cuda.CoordsLogical */

typedef struct {
  PyObject_HEAD
  uint32_t kernel_id;
  uint32_t grid_id;
  PyObject *cluster_idx;
  PyObject *block_idx;
  PyObject *thread_idx;
} gdbpy_cuda_coords_logical_object;

static PyObject *gdbpy_cuda_coords_logical_get_kernel_id (PyObject *self, void *closure)
{
  gdbpy_cuda_coords_logical_object *coords = (gdbpy_cuda_coords_logical_object *) self;

  return PyLong_FromUnsignedLong (coords->kernel_id);
}

static PyObject *gdbpy_cuda_coords_logical_get_grid_id (PyObject *self, void *closure)
{
  gdbpy_cuda_coords_logical_object *coords = (gdbpy_cuda_coords_logical_object *) self;

  return PyLong_FromUnsignedLong (coords->grid_id);
}

static PyObject *gdbpy_cuda_coords_logical_get_cluster_idx (PyObject *self, void *closure)
{
  gdbpy_cuda_coords_logical_object *coords = (gdbpy_cuda_coords_logical_object *) self;

  Py_INCREF (coords->cluster_idx);
  return coords->cluster_idx;
}

static PyObject *gdbpy_cuda_coords_logical_get_block_idx (PyObject *self, void *closure)
{
  gdbpy_cuda_coords_logical_object *coords = (gdbpy_cuda_coords_logical_object *) self;

  Py_INCREF (coords->block_idx);
  return coords->block_idx;
}

static PyObject *gdbpy_cuda_coords_logical_get_thread_idx (PyObject *self, void *closure)
{
  gdbpy_cuda_coords_logical_object *coords = (gdbpy_cuda_coords_logical_object *) self;

  Py_INCREF (coords->thread_idx);
  return coords->thread_idx;
}

static gdb_PyGetSetDef gdbpy_cuda_coords_logical_getset[] = {
  {"kernel_id", gdbpy_cuda_coords_logical_get_kernel_id, nullptr, "Kernel ID", nullptr},
  {"grid_id", gdbpy_cuda_coords_logical_get_grid_id, nullptr, "Grid ID", nullptr},
  {"cluster_idx", gdbpy_cuda_coords_logical_get_cluster_idx, nullptr, "Cluster Index", nullptr},
  {"block_idx", gdbpy_cuda_coords_logical_get_block_idx, nullptr, "Block Index", nullptr},
  {"thread_idx", gdbpy_cuda_coords_logical_get_thread_idx, nullptr, "Thread Index", nullptr},
  {nullptr},
};

static PyTypeObject gdbpy_cuda_coords_logical_type = {
  PyVarObject_HEAD_INIT (nullptr, 0)
};

static PyObject *gdbpy_cuda_coords_logical_init (const cuda_coords_logical &coords)
{
  gdbpy_cuda_coords_logical_object *self = (gdbpy_cuda_coords_logical_object *) PyObject_CallObject ((PyObject *) &gdbpy_cuda_coords_logical_type, nullptr);

  if (self == nullptr)
    return nullptr;

  self->kernel_id = coords.kernelId ();
  self->grid_id = coords.gridId ();
  self->cluster_idx = gdbpy_cuda_cu_dim3_init (coords.clusterIdx ());
  self->block_idx = gdbpy_cuda_cu_dim3_init (coords.blockIdx ());
  self->thread_idx = gdbpy_cuda_cu_dim3_init (coords.threadIdx ());

  return (PyObject *) self;
}

static bool gdbpy_cuda_coords_logical_type_init ()
{
  gdbpy_cuda_coords_logical_type.tp_name = "cuda.CoordsLogical";
  gdbpy_cuda_coords_logical_type.tp_doc = PyDoc_STR ("CUDA Logical Coordinates");
  gdbpy_cuda_coords_logical_type.tp_basicsize = sizeof (gdbpy_cuda_coords_logical_object);
  gdbpy_cuda_coords_logical_type.tp_itemsize = 0;
  gdbpy_cuda_coords_logical_type.tp_flags = Py_TPFLAGS_DEFAULT;
  gdbpy_cuda_coords_logical_type.tp_new = PyType_GenericNew;
  gdbpy_cuda_coords_logical_type.tp_getset = gdbpy_cuda_coords_logical_getset;

  return PyType_Ready (&gdbpy_cuda_coords_logical_type) == 0;
}

/* gdb.cuda methods */

PyObject *gdbpy_cuda_execute_internal_command (PyObject *self, PyObject *args)
{
  const char *command;
  unsigned long long buffer_size = DEFAULT_BUFFER_SIZE;
  
  if (!PyArg_ParseTuple (args, "s|K", &command, &buffer_size))
    return nullptr;

  gdb::unique_xmalloc_ptr<char> buffer((char *) xmalloc (buffer_size));

  if (!cuda_debugapi::execute_internal_command (command, buffer.get(), buffer_size))
    {
      PyErr_SetString (PyExc_RuntimeError, "CUDA Debug API Error");
      return nullptr;
    }

  return Py_BuildValue ("s", buffer.get());
}

PyObject *gdbpy_cuda_get_focus_physical (PyObject *self, PyObject *args)
{
  const auto& focus = cuda_current_focus::get ();

  if (!focus.valid ())
    {
      return Py_None;
    }

  const auto& physical = focus.physical ();

  return gdbpy_cuda_coords_physical_init (physical);
}

PyObject *gdbpy_cuda_get_focus_logical (PyObject *self, PyObject *args)
{
  const auto& focus = cuda_current_focus::get ();

  if (!focus.valid ())
    {
      return Py_None;
    }

  const auto& logical = focus.logical ();

  return gdbpy_cuda_coords_logical_init (logical);
}

PyObject *gdbpy_cuda_get_devices (PyObject *self, PyObject *args)
{
  const uint32_t num_devices = cuda_state::get_num_devices ();

  PyObject *devices = PyList_New (num_devices);

  for (uint32_t devId = 0; devId < num_devices; devId++)
    {
      PyObject *device = gdbpy_cuda_device_init (devId);
      if (device == nullptr)
        {
          Py_DECREF (devices);
          return nullptr;
        }

      PyList_SetItem (devices, devId, device);
    }

  return devices;
}

PyMODINIT_FUNC gdbpy_cuda_init(void)
{
  if (!gdbpy_cuda_device_type_init ()
      || !gdbpy_cuda_cu_dim3_type_init ()
      || !gdbpy_cuda_coords_physical_type_init ()
      || !gdbpy_cuda_coords_logical_type_init ())
    return nullptr;

  return PyModule_Create (&gdbpy_cuda_module);
}

#endif
