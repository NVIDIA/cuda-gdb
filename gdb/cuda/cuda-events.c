/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2024 NVIDIA Corporation
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
#include <signal.h>

#include "inferior.h"
#if defined(__linux__) && defined(GDB_NM_FILE)
#include "linux-nat.h"
#endif
#include "exec.h"
#include "observable.h"
#include "source.h"
#include "target.h"
#include "arch-utils.h"
#include "varobj.h"

#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-events.h"
#include "cuda-functions.h"
#include "cuda-kernel.h"
#include "cuda-modules.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-modules.h"
#include "cuda-elf-image.h"
#include "cuda-options.h"

static void
cuda_trace_event (const char *fmt, ...)
{

  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_EVENT, fmt, ap);
  va_end (ap);
}

static void
cuda_event_create_context (uint32_t dev_id, uint64_t context_id, uint32_t tid)
{
  contexts_t contexts;
  context_t  context;

  cuda_trace_event ("CUDBG_EVENT_CTX_CREATE dev_id=%u context=%llx tid=%u",
                    dev_id, (unsigned long long)context_id, tid);

  if (tid == ~0U)
    error (_("A CUDA event reported an invalid thread id."));

  contexts = cuda_state::device_get_contexts (dev_id);
  context  = context_new (context_id, dev_id);

  contexts_add_context (contexts, context);
  contexts_stack_context (contexts, context, tid);

  if (cuda_options_show_context_events ())
    printf_unfiltered (_("[Context Create of context 0x%llx on Device %u]\n"),
                       (unsigned long long)context_id, dev_id);
}

static void
cuda_event_destroy_context (uint32_t dev_id, uint64_t context_id, uint32_t tid)
{
  contexts_t contexts;
  context_t  context;

  cuda_trace_event ("CUDBG_EVENT_CTX_DESTROY dev_id=%u context=%llx tid=%u",
                    dev_id, (unsigned long long)context_id, tid);

  if (tid == ~0U)
    error (_("A CUDA event reported an invalid thread id."));

  contexts = cuda_state::device_get_contexts (dev_id);
  context  = contexts_find_context_by_id (contexts, context_id);

  if (contexts_get_active_context (contexts, tid) == context)
    context = contexts_unstack_context (contexts, tid);

  if (context == get_current_context ())
    set_current_context (NULL);

  contexts_remove_context (contexts, context);
  context_delete (context);

  if (cuda_options_show_context_events ())
    printf_unfiltered (_("[Context Destroy of context 0x%llx on Device %u]\n"),
                       (unsigned long long)context_id, dev_id);
}

static void
cuda_event_push_context (uint32_t dev_id, uint64_t context_id, uint32_t tid)
{
  contexts_t contexts;
  context_t  context;

  cuda_trace_event ("CUDBG_EVENT_CTX_PUSH dev_id=%u context=%llx tid=%u",
                    dev_id, (unsigned long long)context_id, tid);

  /* context push/pop events are ignored when attaching */
  if (cuda_debugapi::get_attach_state () != CUDA_ATTACH_STATE_NOT_STARTED)
      return;

  if (tid == ~0U)
    error (_("A CUDA event reported an invalid thread id."));

  contexts = cuda_state::device_get_contexts (dev_id);
  context  = contexts_find_context_by_id (contexts, context_id);

  contexts_stack_context (contexts, context, tid);

  if (cuda_options_show_context_events ())
    printf_unfiltered (_("[Context Push of context 0x%llx on Device %u]\n"),
                       (unsigned long long)context_id, dev_id);
}

static void
cuda_event_pop_context (uint32_t dev_id, uint64_t context_id, uint32_t tid)
{
  contexts_t contexts;
  context_t  context;

  cuda_trace_event ("CUDBG_EVENT_CTX_POP dev_id=%u context=%llx tid=%u",
                    dev_id, (unsigned long long)context_id, tid);

  /* context push/pop events are ignored when attaching */
  if (cuda_debugapi::get_attach_state () != CUDA_ATTACH_STATE_NOT_STARTED)
      return;

  if (tid == ~0U)
    error (_("A CUDA event reported an invalid thread id."));

  contexts = cuda_state::device_get_contexts (dev_id);
  context  = contexts_unstack_context (contexts, tid);

  gdb_assert (context_get_id (context) == context_id);

  if (cuda_options_show_context_events ())
    printf_unfiltered (_("[Context Pop of context 0x%llx on Device %u]\n"),
                       (unsigned long long)context_id, dev_id);
}

/* In native debugging, void *elf_image points to memory. In remote debugging, it
   points to a string which is ELF image file path in the temp folder. Both cases 
   will be handled by cuda_elf_image_new() differently. */
static void
cuda_event_load_elf_image (uint32_t dev_id, uint64_t context_id, uint64_t module_id,
                           void *elf_image_raw, uint64_t elf_image_size, uint32_t properties)
{
  context_t   context;
  modules_t   modules;
  module_t    module;
  elf_image_t elf_image;
  bool        is_system;

  cuda_trace_event ("CUDBG_EVENT_ELF_IMAGE_LOADED dev_id=%u context=%llx module=%llx",
                    dev_id, (unsigned long long)context_id, (unsigned long long)module_id);

  context = cuda_state::device_find_context_by_id (dev_id, context_id);
  modules = context_get_modules (context);
  module  = module_new (context, module_id, elf_image_raw, elf_image_size);
  modules_add (modules, module);

  is_system = properties & CUDBG_ELF_IMAGE_PROPERTIES_SYSTEM;

  elf_image = module_get_elf_image (module);
  cuda_elf_image_load (elf_image, is_system);

  auto objfile = cuda_elf_image_get_objfile (elf_image);
  cuda_trace_event ("    -> ELF image loaded %s %lld\n", objfile->original_name, cuda_elf_image_get_size (elf_image));
}

static void
cuda_event_unload_elf_image (uint32_t dev_id, uint64_t context_id, uint64_t module_id,
                             uint64_t handle)
{
  context_t   context;
  modules_t   modules;
  module_t    module;
  elf_image_t elf_image;

  cuda_trace_event ("CUDBG_EVENT_ELF_IMAGE_UNLOADED dev_id=%u context=%llx"
                    " module=%llx handle=%llx",
                    dev_id, (unsigned long long)context_id,
                    (unsigned long long)module_id, (unsigned long long)handle);

  context = cuda_state::device_find_context_by_id (dev_id, context_id);
  modules = context_get_modules (context);
  module = modules_find_module_by_id (modules, module_id);
  elf_image = module_get_elf_image (module);

  auto objfile = cuda_elf_image_get_objfile (elf_image);
  cuda_trace_event ("    -> ELF image unloaded %s %lld\n", objfile->original_name, cuda_elf_image_get_size (elf_image));

  gdb_assert (cuda_elf_image_is_loaded (elf_image));
  cuda_elf_image_unload (elf_image);

  modules_remove (modules, module);
}

static void
cuda_event_kernel_ready (uint32_t dev_id, uint64_t context_id, uint64_t module_id,
                         uint64_t grid_id, uint32_t tid, uint64_t virt_code_base,
                         CuDim3 grid_dim, CuDim3 block_dim, CUDBGKernelType type,
                         uint64_t parent_grid_id, CUDBGKernelOrigin origin)
{
  ptid_t           previous_ptid = inferior_ptid;
#if defined(__linux__) && defined(GDB_NM_FILE)
  struct lwp_info *lp            = NULL;
#endif

  cuda_trace_event ("CUDBG_EVENT_KERNEL_READY dev_id=%u context=%llx"
                    " module=%llx grid_id=%lld tid=%u type=%u"
                    " parent_grid_id=%lld\n",
                    dev_id, (unsigned long long)context_id,
                    (unsigned long long)module_id, (long long)grid_id,
                    tid, type, (long long)parent_grid_id);

  if (tid == ~0U)
    error (_("A CUDA event reported an invalid thread id."));

#if defined(__linux__) && defined(GDB_NM_FILE)
  lp = iterate_over_lwps (inferior_ptid, [=] (struct lwp_info *_lp)
		{
// Using TIDs on aarch64 was disabled due to DTCGDB-265
// Now on Linux aarch64 cuda_gdb_get_tid_or_pid returns a pid and the comparison works
#if !defined(__aarch64__)
		  gdb_assert (cuda_platform_supports_tid ());
#endif
		  return cuda_gdb_get_tid_or_pid (_lp->ptid) == tid;
		});

  if (lp)
    {
      previous_ptid = inferior_ptid;
      inferior_ptid = lp->ptid;
    }
#endif

  kernels_start_kernel (dev_id, grid_id, virt_code_base, context_id,
                        module_id, grid_dim, block_dim, type,
                        parent_grid_id, origin);

  // Add auto-breakpoints if necessary
  if (cuda_options_auto_breakpoints_needed ())
    {
      // Get the elf image of the kernel we just added
      elf_image_t elf_image = module_get_elf_image (kernel_get_module (kernels_get_first_kernel ()));
      // Add the auto breakpoint
      cuda_auto_breakpoints_event_add_break (elf_image, virt_code_base);
    }
#if defined(__linux__) && defined(GDB_NM_FILE)
  if (lp)
    inferior_ptid = previous_ptid;
#endif
}

static void
cuda_event_kernel_finished (uint32_t dev_id, uint64_t grid_id)
{
  kernel_t  kernel;

  cuda_trace_event ("CUDBG_EVENT_KERNEL_FINISHED dev_id=%u grid_id=%lld\n",
                    dev_id, (long long)grid_id);

  kernel = kernels_find_kernel_by_grid_id (dev_id, grid_id);
  kernels_terminate_kernel (kernel);

  clear_current_source_symtab_and_line ();
  clear_displays ();
}

static void
cuda_event_internal_error (CUDBGResult errorType)
{
  cuda_trace_event ("CUDBG_EVENT_INTERNAL_ERROR\n");

  // Stop cuda-gdb and show the error message.
  // We don't kill the app or do the cleanup here. That is done upon
  // exiting cuda-gdb.

  error (_("Error: Internal error reported by CUDA debugger API (error=%s(0x%x)). "
         "The application cannot be further debugged.\n"), 
         cudbgGetErrorString(errorType), errorType);
}

static void
cuda_event_timeout (void)
{
  cuda_trace_event ("CUDBG_EVENT_TIMEOUT\n");
}

#if CUDBG_API_VERSION_REVISION >= 132
static void
cuda_event_functions_loaded (uint32_t dev_id,
			     uint64_t context_id,
			     uint64_t module_id,
                             uint32_t count)
{
  cuda_trace_event ("CUDBG_FUNCTIONS_LOADED dev_id=%u context=0x%llx module=0x%llx count=%d",
                    dev_id,
                    (unsigned long long)context_id,
                    (unsigned long long)module_id,
                    count);

  auto context = cuda_state::device_find_context_by_id (dev_id, context_id);
  auto modules = context_get_modules (context);
  auto module = modules_find_module_by_id (modules, module_id);
  auto elf_image = module_get_elf_image (module);

  auto objfile_old = cuda_elf_image_get_objfile (elf_image);
  cuda_trace_event ("  -> old ELF image %p (%s)", elf_image, objfile_old->original_name);

  /* Record information about the current elf image */
  auto is_system = cuda_elf_image_is_system (elf_image);
  auto image_size = cuda_elf_image_get_size (elf_image);

  /* Save a copy of the ELF image for updating below */
  auto buffer = (void *)xmalloc (image_size);

  /* For debugging incremental vs. full reload */
  bool incremental_updates = true;
  if (incremental_updates)
    {
      cuda_trace_event ("%s: applying incremental updates", __FUNCTION__);

      /* Save a copy of the current cubin */
      cuda_elf_image_read (objfile_old->original_name, buffer, image_size);

      /* Unload the elf_image and objfile */
      cuda_elf_image_unload (elf_image);

      /* Fetch the incremental function load information */
      size_t len = count * sizeof(CUDBGLoadedFunctionInfo);
      auto info = (CUDBGLoadedFunctionInfo *)xmalloc (len);
      cuda_debugapi::get_loaded_function_info (dev_id, module_id, info, count);

      /* Apply the incremental relocations */
      cuda_apply_function_load_updates ((char *)buffer, info, count);
      xfree (info);

      /* For debugging, compare before and after incremental updates */
      bool compare_cubins = false;
      if (compare_cubins)
	{
	  cuda_trace_event ("%s: checking incremental updates", __FUNCTION__);

	  /* Save a copy of the ELF image for updating below */
	  auto original = (void *)xmalloc (image_size);

	  /* Re-fetch the whole cubin */
	  cuda_debugapi::get_elf_image (dev_id, module_id, true, original, image_size);

	  /* Compare */
	  if (memcmp (buffer, original, image_size))
	    error (_("%s: cubin buffers do not match after incremental update"), __FUNCTION__);

	  /* Done with the temp copy */
	  xfree (original);
	}
    }
  else
    {
      cuda_trace_event ("%s: skipping incremental updates", __FUNCTION__);

      /* Unload the elf_image and objfile */
      cuda_elf_image_unload (elf_image);

      /* Re-fetch the whole cubin */
      cuda_debugapi::get_elf_image (dev_id, module_id, true, buffer, image_size);
    }

  /* Write image back out to disk for cuda_elf_image_load() to create
     a BFD from. */
  cuda_elf_image_save (elf_image, buffer);
  xfree (buffer);

  /* Load the ELF image from the filesystem, creating a new objfile */
  cuda_elf_image_load (elf_image, is_system);

  // Flush the module disassembly caches as we've just updated the
  // ELF image
  auto disassembler = module->disassembler;
  if (disassembler)
    {
      disassembler->flush_elf_cache ();
      disassembler->flush_device_cache ();
    }
}
#endif

void
cuda_event_post_process (bool reset_bpt)
{

  /* Launch (kernel ready) events may require additional
     breakpoint handling (via remove/insert). */
  if (reset_bpt)
    breakpoint_re_set ();
}

void
cuda_process_events (CUDBGEvent *event, cuda_event_kind_t kind)
{
  bool reset_bpt = false;
  gdb_assert (event);

  /* Step 1:  Consume all events (synchronous and asynchronous).
     We must consume every event prior to any generic operations
     that will force a state collection across the device. */
  for (; event->kind != CUDBG_EVENT_INVALID;
       (kind == CUDA_EVENT_SYNC) ? cuda_debugapi::get_next_sync_event (event) :
                                   cuda_debugapi::get_next_async_event (event)) {
    cuda_process_event (event);
    if (event->kind == CUDBG_EVENT_KERNEL_READY)
        reset_bpt = true;
#if CUDBG_API_VERSION_REVISION >= 132
    if (event->kind == CUDBG_EVENT_FUNCTIONS_LOADED)
        reset_bpt = true;
#endif
  }

  /* Step 2:  Post-process events after they've all been consumed. */
  cuda_event_post_process (reset_bpt);
}

void
cuda_process_event (CUDBGEvent *event)
{
  uint32_t dev_id;
  uint64_t grid_id;
  uint32_t tid;
  uint64_t virt_code_base;
  uint64_t context_id;
  uint64_t module_id;
  uint64_t handle;
  uint32_t properties;
  uint64_t elf_image_size;
  uint64_t parent_grid_id;
  uint32_t count;
  void    *elf_image;
  CuDim3   grid_dim;
  CuDim3   block_dim;
  CUDBGKernelType type;
  CUDBGKernelOrigin origin;
  CUDBGResult errorType;

  gdb_assert (event);

      switch (event->kind)
        {
        case CUDBG_EVENT_ELF_IMAGE_LOADED:
          {
            dev_id         = event->cases.elfImageLoaded.dev;
            context_id     = event->cases.elfImageLoaded.context;
            module_id      = event->cases.elfImageLoaded.module;
            handle         = event->cases.elfImageLoaded.handle;
            properties     = event->cases.elfImageLoaded.properties;
            elf_image_size = event->cases.elfImageLoaded.size;
            elf_image      = malloc (elf_image_size);
            cuda_debugapi::get_elf_image (dev_id, handle, true, elf_image, elf_image_size);
            cuda_event_load_elf_image (dev_id, context_id, module_id,
                                       elf_image, elf_image_size, properties);
            free (elf_image);
            break;
          }
        case CUDBG_EVENT_KERNEL_READY:
          {
            dev_id         = event->cases.kernelReady.dev;
            context_id     = event->cases.kernelReady.context;
            module_id      = event->cases.kernelReady.module;
            grid_id        = event->cases.kernelReady.gridId;
            tid            = event->cases.kernelReady.tid;
            virt_code_base = event->cases.kernelReady.functionEntry;
            grid_dim       = event->cases.kernelReady.gridDim;
            block_dim      = event->cases.kernelReady.blockDim;
            type           = event->cases.kernelReady.type;
            parent_grid_id = event->cases.kernelReady.parentGridId;
            origin         = event->cases.kernelReady.origin;
            cuda_event_kernel_ready (dev_id, context_id, module_id, grid_id,
                                     tid, virt_code_base, grid_dim, block_dim,
                                     type, parent_grid_id, origin);
            break;
          }
        case CUDBG_EVENT_KERNEL_FINISHED:
          {
            dev_id  = event->cases.kernelFinished.dev;
            grid_id = event->cases.kernelFinished.gridId;
            cuda_event_kernel_finished (dev_id, grid_id);
            break;
          }
        case CUDBG_EVENT_CTX_PUSH:
          {
            dev_id     = event->cases.contextPush.dev;
            context_id = event->cases.contextPush.context;
            tid        = event->cases.contextPush.tid;
            cuda_event_push_context (dev_id, context_id, tid);
            break;
          }
        case CUDBG_EVENT_CTX_POP:
          {
            dev_id     = event->cases.contextPop.dev;
            context_id = event->cases.contextPop.context;
            tid        = event->cases.contextPop.tid;
            cuda_event_pop_context (dev_id, context_id, tid);
            break;
          }
        case CUDBG_EVENT_CTX_CREATE:
          {
            dev_id     = event->cases.contextCreate.dev;
            context_id = event->cases.contextCreate.context;
            tid        = event->cases.contextCreate.tid;
            cuda_event_create_context (dev_id, context_id, tid);
            break;
          }
        case CUDBG_EVENT_CTX_DESTROY:
          {
            dev_id     = event->cases.contextDestroy.dev;
            context_id = event->cases.contextDestroy.context;
            tid        = event->cases.contextDestroy.tid;
            cuda_event_destroy_context (dev_id, context_id, tid);
            break;
          }
        case CUDBG_EVENT_INTERNAL_ERROR:
          {
            errorType  = event->cases.internalError.errorType;
            cuda_event_internal_error (errorType);
            break;
          }
        case CUDBG_EVENT_TIMEOUT:
          {
            cuda_event_timeout ();
            break;
          }
        case CUDBG_EVENT_ATTACH_COMPLETE:
          {
            cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_APP_READY);
            break;
          }
        case CUDBG_EVENT_DETACH_COMPLETE:
          {
            cuda_debugapi::set_attach_state (CUDA_ATTACH_STATE_DETACH_COMPLETE);
            break;
          }
        case CUDBG_EVENT_ELF_IMAGE_UNLOADED:
          {
            dev_id         = event->cases.elfImageUnloaded.dev;
            context_id     = event->cases.elfImageUnloaded.context;
            module_id      = event->cases.elfImageUnloaded.module;
            handle         = event->cases.elfImageUnloaded.handle;
            cuda_event_unload_elf_image (dev_id, context_id, module_id,
                                         handle);
            break;
          }
#if CUDBG_API_VERSION_REVISION >= 132
        case CUDBG_EVENT_FUNCTIONS_LOADED:
          {
            dev_id         = event->cases.functionsLoaded.dev;
            context_id     = event->cases.functionsLoaded.context;
            module_id      = event->cases.functionsLoaded.module;
            count          = event->cases.functionsLoaded.count;
            cuda_event_functions_loaded (dev_id, context_id, module_id, count);
            break;
          }
#endif
        default:
          gdb_assert (0);
        }
}

