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
#include "gdbtypes.h"
#include "gdbarch.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "cuda-api.h"

#ifdef __APPLE__
#include <libproc.h>
#include <IOKit/IOKitLib.h>

/* Maximum length of IOKit registry objects list */
#define MAX_OBJLIST_LEN 128

/* Darwin process name length */
#define DARWIN_PROC_NAME_LEN 128

bool cuda_darwin_cuda_device_used_for_graphics (int dev_id);
bool cuda_darwin_is_launched_from_ssh_session (void);

static IOReturn
DarwinGetObjects(mach_port_t port, int *count, io_object_t *objects, const char *name)
{
  IOReturn rc;
  CFMutableDictionaryRef dict = NULL;
  io_iterator_t iterator = 0;
  io_object_t object = 0;

  dict = IOServiceMatching (name);
  if (!dict)
    return kIOReturnNoMemory;
  rc = IOServiceGetMatchingServices (port, dict, &iterator);
  if (rc != kIOReturnSuccess)
    {
      CFRelease (dict);
      return rc;
    }
  *count = 0;
  while ((object = IOIteratorNext (iterator)))
    {
      if (objects)
        objects[*count] = object;
      if ((*count)++ == MAX_OBJLIST_LEN) break;
    }
  return IOObjectRelease (iterator);
}

static IOReturn
DarwinGetParentOfType(io_object_t object, io_object_t *pParent, const char *type)
{
  IOReturn rc;
  io_object_t parent;
  io_name_t class_name;

  assert (pParent != NULL);

  while (object)
    {
      rc = IOObjectGetClass (object, class_name);
      if (rc != kIOReturnSuccess) return rc;
      if (!type || strcmp (class_name, type) == 0)
        {
          *pParent = object;
          return kIOReturnSuccess;
        }
      rc = IORegistryEntryGetParentEntry (object, kIOServicePlane, &parent);
      if ( rc != kIOReturnSuccess) return rc;
      object = parent;
    }
  return kIOReturnNoDevice;
}

static IOReturn
DarwinGetObjectChildsOfType(io_object_t obj, int *count, io_object_t *objects, const char *type)
{
  IOReturn rc;
  io_iterator_t iterator;
  io_object_t child;
  io_name_t class_name;

  rc = IORegistryEntryGetChildIterator (obj, kIOServicePlane, &iterator);
  if (rc != kIOReturnSuccess)
    return rc;

  *count = 0;
  while ((child = IOIteratorNext (iterator)))
    {
      rc = IOObjectGetClass (child, class_name);
      if (rc != kIOReturnSuccess) continue;
      if (type && strcmp (class_name, type)!=0) continue;
      if (objects)
        objects[*count] = child;
      if ((*count)++ == MAX_OBJLIST_LEN) break;
    }
  return IOObjectRelease (iterator);
}

static int
DarwinGetChildsOfTypeCount(io_object_t obj, const char *type)
{
  IOReturn rc;
  int count;

  rc = DarwinGetObjectChildsOfType (obj, &count, NULL, type);
  return rc != kIOReturnSuccess ? -1 : count;
}

static int
DarwinGetSiblingsOfTypeCount(io_object_t obj, const char *type)
{
  IOReturn rc;
  io_object_t parent;
  int count;

  rc = IORegistryEntryGetParentEntry (obj, kIOServicePlane, &parent);
  if ( rc != kIOReturnSuccess) return -1;
  return DarwinGetChildsOfTypeCount (parent, type);
}

static IOReturn
DarwinGetPCIBusInfo(io_object_t obj, uint32_t *pci_bus_id, uint32_t *pci_dev_id, uint32_t *pci_func_id)
{
  CFDataRef reg_ref;
  uint32_t pci_id;

  reg_ref = IORegistryEntryCreateCFProperty (obj, CFSTR("reg"), kCFAllocatorDefault, kNilOptions);

  if (!reg_ref) return kIOReturnNotFound;
  if (CFGetTypeID(reg_ref) != CFDataGetTypeID())
    {
      CFRelease (reg_ref);
      return kIOReturnNotFound;
    }
  pci_id = ((uint32_t *)CFDataGetBytePtr(reg_ref))[0];
  CFRelease (reg_ref);

  if (pci_bus_id)  *pci_bus_id  = (pci_id>>16)&0xff;
  if (pci_dev_id)  *pci_dev_id  = (pci_id>>11)&0x1f;
  if (pci_func_id) *pci_func_id = (pci_id>>8)&0x07;
  return kIOReturnSuccess;
}


/*
 * Tries to determine if GPU used for graphics is also used for CUDA
 * If any of the system calls fails, it assumes that's the case.
 * CUDA device is considered used for graphics if following conditions are met:
 * - Display is attached to given GPU
 * - At least one frame-buffer client is using this GPU (i.e. WindowServer is running)
 * - At least one compute client is using this GPU
 */

bool
cuda_darwin_cuda_device_used_for_graphics(int dev_id)
{
  uint32_t cuda_pci_bus_id, cuda_pci_dev_id;
  uint32_t pci_bus_id, pci_dev_id;
  int i, count;
  io_object_t displays[MAX_OBJLIST_LEN+1];
  io_object_t parent;
  IOReturn rc;

  memset (displays, 0, sizeof(displays));
  cuda_api_get_device_pci_bus_info (dev_id, &cuda_pci_bus_id, &cuda_pci_dev_id);


  /* Get IODisplayConnect objects */
  rc = DarwinGetObjects (kIOMasterPortDefault, &count, displays, "IODisplayConnect");
  if (rc != kIOReturnSuccess)
    return true;

  /* CUDA device is unlikely to be used for graphics in absence of displays */
  if (count == 0) return false;

  /* Iterate over IODisplayConnect objects*/
  for (i=0;i<count;i++)
    {
      /* Find PCI device display connected to */
      rc = DarwinGetParentOfType (displays[i], &parent, "IOPCIDevice");
      if (rc != kIOReturnSuccess || parent == 0)
        return true;
      /* Get devices bus/dev ids*/
      rc = DarwinGetPCIBusInfo (parent, &pci_bus_id, &pci_dev_id, NULL);
      if (rc != kIOReturnSuccess) return true;
      if (pci_bus_id != cuda_pci_bus_id || pci_dev_id != cuda_pci_dev_id) continue;
      /* Found CUDA device with display attached to it */
      /* Not safe to use unless we are in console mode */
      rc = DarwinGetSiblingsOfTypeCount (displays[i], "IOFramebufferUserClient");
      if (rc != 0) return true;
    }

  return false;
}

static int
darwin_find_session_leader (int *pPid, char *name)
{
  int pid, rc;
  struct proc_bsdinfo bsdinfo;

  pid = getpid();
  while (pid>0)
    {
      rc = proc_pidinfo (pid, PROC_PIDTBSDINFO, 0, &bsdinfo, sizeof(bsdinfo));
      if (rc != sizeof(bsdinfo)) return -1;
      if ((bsdinfo.pbi_flags & PROC_FLAG_SLEADER) == PROC_FLAG_SLEADER) 
        {
          *pPid = pid;
          rc = proc_name (pid, name, DARWIN_PROC_NAME_LEN);
          if (rc < 0) return rc;
          return 0;
        }
        pid = bsdinfo.pbi_ppid;
    }
  return -1;
}

bool
cuda_darwin_is_launched_from_ssh_session(void)
{
  int rc,pid;
  char name[DARWIN_PROC_NAME_LEN];

  rc =  darwin_find_session_leader (&pid, name);

  /* If application is launched from Terminal, its session leader is /sbin/login, which has suid bit*/
  if (rc != 0) return false;

  /* Session leader for anything launched from bash or ourselves*/
  return (strcmp (name, "bash")==0 || pid == getpid()) ? true : false;
}

#endif /*__APPLE__*/
