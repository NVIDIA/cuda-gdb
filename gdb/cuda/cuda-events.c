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
#include "cuda-kernel.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-modules.h"
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
cuda_event_create_context (uint32_t dev_id, uint64_t context_id, uint32_t thread_id)
{
  cuda_trace_event ("CUDBG_EVENT_CTX_CREATE dev_id=%u context=%0xllx thread_id=%u",
                    dev_id, (unsigned long long)context_id, thread_id);

  if (thread_id == ~0U)
    error (_("A CUDA context create event reported an invalid thread id."));

  cuda_state::create_context (dev_id, context_id, thread_id);

  if (cuda_options_show_context_events ())
    printf_unfiltered (_("[Context Create of context 0x%llx on Device %u]\n"),
                       (unsigned long long)context_id, dev_id);
}

static void
cuda_event_destroy_context (uint32_t dev_id, uint64_t context_id, uint32_t thread_id)
{
  cuda_trace_event ("CUDBG_EVENT_CTX_DESTROY dev_id=%u context=0x%llx thread_id=%u",
                    dev_id, (unsigned long long)context_id, thread_id);

  if (thread_id == ~0U)
    error (_("A CUDA context destroy event reported an invalid thread id."));

  cuda_state::destroy_context (context_id);

  if (cuda_options_show_context_events ())
    printf_unfiltered (_("[Context Destroy of context 0x%llx on Device %u]\n"),
                       (unsigned long long)context_id, dev_id);
}

static void
cuda_event_push_context (uint32_t dev_id, uint64_t context_id, uint32_t thread_id)
{
  cuda_trace_event ("CUDBG_EVENT_CTX_PUSH dev_id=%u context=0x%llx thread_id=%u",
                    dev_id, (unsigned long long)context_id, thread_id);

  /* context push/pop events are ignored when attaching */
  if (cuda_debugapi::get_attach_state () != CUDA_ATTACH_STATE_NOT_STARTED)
      return;

  if (thread_id == ~0U)
    error (_("A CUDA context push event reported an invalid thread id."));

  if (cuda_options_show_context_events ())
    printf_unfiltered (_("[Context Push of context 0x%llx on Device %u]\n"),
                       (unsigned long long)context_id, dev_id);
}

static void
cuda_event_pop_context (uint32_t dev_id, uint64_t context_id, uint32_t thread_id)
{
  cuda_trace_event ("CUDBG_EVENT_CTX_POP dev_id=%u context=0x%llx thread_id=%u",
                    dev_id, (unsigned long long)context_id, thread_id);

  /* context push/pop events are ignored when attaching */
  if (cuda_debugapi::get_attach_state () != CUDA_ATTACH_STATE_NOT_STARTED)
      return;

  if (thread_id == ~0U)
    error (_("A CUDA context pop event reported an invalid thread id."));

  if (cuda_options_show_context_events ())
    printf_unfiltered (_("[Context Pop of context 0x%llx on Device %u]\n"),
                       (unsigned long long)context_id, dev_id);
}

static void
cuda_event_load_elf_image (uint32_t dev_id, uint64_t context_id, uint64_t module_id,
                           uint64_t elf_image_size, uint32_t properties)
{
  auto module = cuda_state::create_module (module_id, (CUDBGElfImageProperties)properties,
					   context_id, elf_image_size);
  gdb_assert (module);

  cuda_trace_event ("CUDBG_EVENT_ELF_IMAGE_LOADED   %s module 0x%llx context 0x%llx size %7llu dev %u properties 0x%04x",
		    module->filename ().c_str (),
		    module_id,
		    context_id,
		    elf_image_size,
		    dev_id,
		    properties);
}

static void
cuda_event_unload_elf_image (uint32_t dev_id, uint64_t context_id, uint64_t module_id,
                             uint64_t handle)
{
  if (cuda_options_trace_domain_enabled (CUDA_TRACE_EVENT))
    {
      auto module = cuda_state::find_module_by_id (module_id);
      gdb_assert (module);

      cuda_trace_event ("CUDBG_EVENT_ELF_IMAGE_UNLOADED %s module 0x%llx context 0x%llx size %7llu",
			module->filename ().c_str (),
			module_id,
			context_id,
			module->size ());
    }
  cuda_state::destroy_module (module_id);
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
                    " parent_grid_id=%lld",
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
      // Get the cuda_module of the kernel we just added
      auto module = kernel_get_module (kernels_get_first_kernel ());
      cuda_auto_breakpoints_event_add_break (module, virt_code_base);
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

static void
cuda_event_functions_loaded (uint32_t dev_id,
			     uint64_t context_id,
			     uint64_t module_id,
                             uint32_t count)
{
  gdb_assert (module_id);

  auto module = cuda_state::find_module_by_id (module_id);
  gdb_assert (module);

  // Formatted to line up with ELF image LOAD / UNLOAD
  cuda_trace_event ("CUDBG_FUNCTIONS_LOADED         %s module 0x%llx (%llu + %llu)",
		    module->filename ().c_str (),
                    module_id,
		    module->functions_loaded (),
		    count - module->functions_loaded ());

  module->functions_loaded_event (count);
}

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
    if (event->kind == CUDBG_EVENT_FUNCTIONS_LOADED)
        reset_bpt = true;
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
            cuda_event_load_elf_image (dev_id, context_id, module_id,
                                       elf_image_size, properties);
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
        case CUDBG_EVENT_FUNCTIONS_LOADED:
          {
            dev_id         = event->cases.functionsLoaded.dev;
            context_id     = event->cases.functionsLoaded.context;
            module_id      = event->cases.functionsLoaded.module;
            count          = event->cases.functionsLoaded.count;
            cuda_event_functions_loaded (dev_id, context_id, module_id, count);
            break;
          }

        default:
          gdb_assert (0);
        }
}

