/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2025 NVIDIA Corporation
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
#include "arch-utils.h"
#include "exec.h"
#include "observable.h"
#include "source.h"
#include "target.h"
#include "varobj.h"

#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-events.h"
#include "cuda-kernel.h"
#include "cuda-modules.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#ifndef GDBSERVER
#include "cuda-utils.h"
#endif

static void
cuda_trace_event (const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  cuda_vtrace_domain (CUDA_TRACE_EVENT, fmt, ap);
  va_end (ap);
}

static void
cuda_event_create_context (const uint32_t &dev_id, const uint64_t &context_id,
			   const uint32_t &thread_id)
{
  cuda_trace_event (
      "CUDBG_EVENT_CTX_CREATE dev_id=%u context=0x%llx thread_id=%u", dev_id,
      (unsigned long long)context_id, thread_id);

  if (thread_id == static_cast<uint32_t> (~0U))
    error (_ ("A CUDA context create event reported an invalid thread id."));

  cuda_state::create_context (dev_id, context_id, thread_id);

  if (cuda_options_show_context_events ())
    printf_unfiltered (_ ("[Context Create of context 0x%llx on Device %u]\n"),
		       (unsigned long long)context_id, dev_id);
}

static void
cuda_event_destroy_context (const uint32_t &dev_id, const uint64_t &context_id,
			    const uint32_t &thread_id)
{
  cuda_trace_event (
      "CUDBG_EVENT_CTX_DESTROY dev_id=%u context=0x%llx thread_id=%u", dev_id,
      (unsigned long long)context_id, thread_id);

  if (thread_id == static_cast<uint32_t> (~0U))
    error (_ ("A CUDA context destroy event reported an invalid thread id."));

  cuda_state::destroy_context (context_id);

  if (cuda_options_show_context_events ())
    printf_unfiltered (
	_ ("[Context Destroy of context 0x%llx on Device %u]\n"),
	(unsigned long long)context_id, dev_id);
}

static void
cuda_event_push_context (const uint32_t &dev_id, const uint64_t &context_id,
			 const uint32_t &thread_id)
{
  cuda_trace_event (
      "CUDBG_EVENT_CTX_PUSH dev_id=%u context=0x%llx thread_id=%u", dev_id,
      (unsigned long long)context_id, thread_id);

  /* context push/pop events are ignored when attaching */
  if (cuda_debugapi::get_attach_state () != CUDA_ATTACH_STATE_NOT_STARTED)
    return;

  if (thread_id == static_cast<uint32_t> (~0U))
    error (_ ("A CUDA context push event reported an invalid thread id."));

  if (cuda_options_show_context_events ())
    printf_unfiltered (_ ("[Context Push of context 0x%llx on Device %u]\n"),
		       (unsigned long long)context_id, dev_id);
}

static void
cuda_event_pop_context (const uint32_t &dev_id, const uint64_t &context_id,
			const uint32_t &thread_id)
{
  cuda_trace_event (
      "CUDBG_EVENT_CTX_POP dev_id=%u context_id=0x%llx thread_id=%u", dev_id,
      (unsigned long long)context_id, thread_id);

  /* context push/pop events are ignored when attaching */
  if (cuda_debugapi::get_attach_state () != CUDA_ATTACH_STATE_NOT_STARTED)
    return;

  if (thread_id == static_cast<uint32_t> (~0U))
    error (_ ("A CUDA context pop event reported an invalid thread id."));

  if (cuda_options_show_context_events ())
    printf_unfiltered (_ ("[Context Pop of context 0x%llx on Device %u]\n"),
		       (unsigned long long)context_id, dev_id);
}

static void
cuda_event_load_elf_image (const uint32_t &dev_id, const uint64_t &context_id,
			   const uint64_t &module_id,
			   const uint64_t &elf_image_size,
			   const uint32_t &properties)
{
  cuda_trace_event ("CUDBG_EVENT_ELF_IMAGE_LOADED dev_id=%u context_id=0x%llx "
		    "module_id=0x%llx image_size=%7llu properties=0x%04x",
		    dev_id, context_id, module_id, elf_image_size, properties);

  auto module = cuda_state::create_module (module_id,
					   (CUDBGElfImageProperties)properties,
					   context_id, elf_image_size);

  gdb_assert (module);

  /* May continue GPU execution, ensure breakpoints are inserted. */
  insert_breakpoints ();

  cuda_trace_event ("CUDBG_EVENT_ELF_IMAGE_LOADED done loading %s",
		    module->filename ().c_str ());
}

static void
cuda_event_unload_elf_image (const uint32_t &dev_id,
			     const uint64_t &context_id,
			     const uint64_t &module_id, const uint64_t &handle)
{
  cuda_trace_event (
      "CUDBG_EVENT_ELF_IMAGE_UNLOADED dev_id=%u context_id=0x%llx "
      "module_id=0x%llx handle=0x%llx",
      dev_id, context_id, module_id, handle);

  if (cuda_options_trace_domain_enabled (CUDA_TRACE_EVENT))
    {
      auto module = cuda_state::find_module_by_id (module_id);
      gdb_assert (module);
      cuda_trace_event (
	"CUDBG_EVENT_ELF_IMAGE_UNLOADED unloading %s size=%7llu",
	module->filename ().c_str (), module->size ());
    }

  cuda_state::destroy_module (module_id);
}

static void
cuda_event_kernel_ready (const uint32_t &dev_id, const uint64_t &context_id,
			 const uint64_t &module_id, const uint64_t &grid_id,
			 const uint32_t &tid, const uint64_t &virt_code_base,
			 const CuDim3 &grid_dim, const CuDim3 &block_dim,
			 const CUDBGKernelType &type,
			 const uint64_t &parent_grid_id,
			 const CUDBGKernelOrigin &origin)
{
  cuda_trace_event ("CUDBG_EVENT_KERNEL_READY dev_id=%u context_id=0x%lx"
		    " module_id=0x%lx grid_id=%ld tid=%u type=%u"
		    " parent_grid_id=%ld",
		    dev_id, context_id, module_id, (int64_t)grid_id, tid, type,
		    (int64_t)parent_grid_id);

#if defined(__linux__) && defined(GDB_NM_FILE)
  ptid_t previous_ptid = inferior_ptid;
  struct lwp_info *lp = NULL;
#endif

  if (tid == static_cast<uint32_t> (~0U))
    error (_ ("A CUDA event reported an invalid thread id."));

#if defined(__linux__) && defined(GDB_NM_FILE)
  lp = iterate_over_lwps (inferior_ptid, [=] (struct lwp_info *_lp) {
  // Using TIDs on aarch64 was disabled due to DTCGDB-265
  // Now on Linux aarch64 cuda_gdb_get_tid_or_pid returns a pid and the
  // comparison works
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

  const auto kernel = cuda_state::create_kernel (
      dev_id, grid_id, virt_code_base, module_id, grid_dim, block_dim,
      CuDim3{ 0 }, CuDim3{ 0 }, type, origin, parent_grid_id);

  // Add auto-breakpoints if necessary
  if (cuda_options_auto_breakpoints_needed ())
    cuda_auto_breakpoints_event_add_break (kernel->module (), virt_code_base);

#if defined(__linux__) && defined(GDB_NM_FILE)
  if (lp)
    inferior_ptid = previous_ptid;
#endif

  /* May continue GPU execution, ensure breakpoints are inserted. */
  insert_breakpoints ();
}

static void
cuda_event_kernel_finished (const uint32_t &dev_id, const uint64_t &grid_id)
{
  cuda_trace_event ("CUDBG_EVENT_KERNEL_FINISHED dev_id=%u grid_id=%ld\n",
		    dev_id, (int64_t)grid_id);

  cuda_state::destroy_kernel (dev_id, grid_id);

  clear_current_source_symtab_and_line ();
  clear_displays ();
}

static void
cuda_event_internal_error (const CUDBGResult &errorType)
{
  cuda_trace_event ("CUDBG_EVENT_INTERNAL_ERROR\n");

  // Stop cuda-gdb and show the error message.
  // We don't kill the app or do the cleanup here. That is done upon
  // exiting cuda-gdb.

  error (_ ("Error: Internal error reported by CUDA debugger API "
	    "(error=%s(0x%x)). "
	    "The application cannot be further debugged.\n"),
	 cudbgGetErrorString (errorType), errorType);
}

static void
cuda_event_timeout (void)
{
  cuda_trace_event ("CUDBG_EVENT_TIMEOUT\n");
}

static void
cuda_event_functions_loaded (const uint32_t &dev_id,
			     const uint64_t &context_id,
			     const uint64_t &module_id, const uint32_t &count)
{
  cuda_trace_event ("CUDBG_EVENT_FUNCTIONS_LOADED dev_id=%u context_id=0x%llx "
		    "module_id=0x%llx count=%u",
		    dev_id, context_id, module_id, count);

  auto module = cuda_state::find_module_by_id (module_id);
  gdb_assert (module);

  module->functions_loaded_event (count);

  /* May continue GPU execution, ensure breakpoints are inserted. */
  insert_breakpoints ();

  cuda_trace_event ("CUDBG_EVENT_FUNCTIONS_LOADED done loading functions for "
		    "%s (total functions loaded:%llu remaining unloaded:%llu)",
		    module->filename ().c_str (), module->functions_loaded (),
		    count - module->functions_loaded ());
}

static void
cuda_event_cuda_logs_available (void)
{
  /* Logs available event is a notification that there are new logs 
   * to consume. We need to drain the queue by calling consumeCudaLogs
   * until no more logs are available.
   */
  cuda_trace_event ("CUDBG_EVENT_CUDA_LOGS_AVAILABLE");
  
  if (cuda_options_driver_logs_enabled())
    cuda_consume_and_print_driver_logs();
}

static void 
cuda_event_cuda_logs_threshold_reached (void)
{
  /* Threshold reached event indicates the log buffer is filling up
   * and we should drain it as soon as possible.
   */
  cuda_trace_event ("CUDBG_EVENT_CUDA_LOGS_THRESHOLD_REACHED");

  warning (_ ("CUDA driver log buffer is full. Some logs may be lost.\n"));
  
  if (cuda_options_driver_logs_enabled())
    cuda_consume_and_print_driver_logs();
}

static void
cuda_process_event (const CUDBGEvent &event)
{
  cuda_trace_event ("cuda_process_event: event=%u", event.kind);

  switch (event.kind)
    {
    case CUDBG_EVENT_ELF_IMAGE_LOADED:
      {
	const auto &dev_id = event.cases.elfImageLoaded.dev;
	const auto &context_id = event.cases.elfImageLoaded.context;
	const auto &module_id = event.cases.elfImageLoaded.module;
	const auto &properties = event.cases.elfImageLoaded.properties;
	const auto &elf_image_size = event.cases.elfImageLoaded.size;
	cuda_event_load_elf_image (dev_id, context_id, module_id,
				   elf_image_size, properties);
	break;
      }
    case CUDBG_EVENT_KERNEL_READY:
      {
	const auto &dev_id = event.cases.kernelReady.dev;
	const auto &context_id = event.cases.kernelReady.context;
	const auto &module_id = event.cases.kernelReady.module;
	const auto &grid_id = event.cases.kernelReady.gridId;
	const auto &tid = event.cases.kernelReady.tid;
	const auto &virt_code_base = event.cases.kernelReady.functionEntry;
	const auto &grid_dim = event.cases.kernelReady.gridDim;
	const auto &block_dim = event.cases.kernelReady.blockDim;
	const auto &type = event.cases.kernelReady.type;
	const auto &parent_grid_id = event.cases.kernelReady.parentGridId;
	const auto &origin = event.cases.kernelReady.origin;
	cuda_event_kernel_ready (dev_id, context_id, module_id, grid_id, tid,
				 virt_code_base, grid_dim, block_dim, type,
				 parent_grid_id, origin);
	break;
      }
    case CUDBG_EVENT_KERNEL_FINISHED:
      {
	const auto &dev_id = event.cases.kernelFinished.dev;
	const auto &grid_id = event.cases.kernelFinished.gridId;
	cuda_event_kernel_finished (dev_id, grid_id);
	break;
      }
    case CUDBG_EVENT_CTX_PUSH:
      {
	const auto &dev_id = event.cases.contextPush.dev;
	const auto &context_id = event.cases.contextPush.context;
	const auto &tid = event.cases.contextPush.tid;
	cuda_event_push_context (dev_id, context_id, tid);
	break;
      }
    case CUDBG_EVENT_CTX_POP:
      {
	const auto &dev_id = event.cases.contextPop.dev;
	const auto &context_id = event.cases.contextPop.context;
	const auto &tid = event.cases.contextPop.tid;
	cuda_event_pop_context (dev_id, context_id, tid);
	break;
      }
    case CUDBG_EVENT_CTX_CREATE:
      {
	const auto &dev_id = event.cases.contextCreate.dev;
	const auto &context_id = event.cases.contextCreate.context;
	const auto &tid = event.cases.contextCreate.tid;
	cuda_event_create_context (dev_id, context_id, tid);
	break;
      }
    case CUDBG_EVENT_CTX_DESTROY:
      {
	const auto &dev_id = event.cases.contextDestroy.dev;
	const auto &context_id = event.cases.contextDestroy.context;
	const auto &tid = event.cases.contextDestroy.tid;
	cuda_event_destroy_context (dev_id, context_id, tid);
	break;
      }
    case CUDBG_EVENT_INTERNAL_ERROR:
      {
	const auto &errorType = event.cases.internalError.errorType;
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
	const auto &dev_id = event.cases.elfImageUnloaded.dev;
	const auto &context_id = event.cases.elfImageUnloaded.context;
	const auto &module_id = event.cases.elfImageUnloaded.module;
	const auto &handle = event.cases.elfImageUnloaded.handle;
	cuda_event_unload_elf_image (dev_id, context_id, module_id, handle);
	break;
      }
    case CUDBG_EVENT_FUNCTIONS_LOADED:
      {
	const auto &dev_id = event.cases.functionsLoaded.dev;
	const auto &context_id = event.cases.functionsLoaded.context;
	const auto &module_id = event.cases.functionsLoaded.module;
	const auto &count = event.cases.functionsLoaded.count;
	cuda_event_functions_loaded (dev_id, context_id, module_id, count);
	break;
      }
    case CUDBG_EVENT_ALL_DEVICES_SUSPENDED:
      break;
    case CUDBG_EVENT_CUDA_LOGS_AVAILABLE:
      cuda_event_cuda_logs_available ();
      break;
    case CUDBG_EVENT_CUDA_LOGS_THRESHOLD_REACHED:
      cuda_event_cuda_logs_threshold_reached ();
      break;
    case CUDBG_EVENT_INVALID:
    default:
      gdb_assert (0);
    }

  cuda_trace_event ("cuda_process_event: event=%u done", event.kind);
}

bool
cuda_process_events (
    const cuda_event_kind_t kind,
    gdb::function_view<bool (const CUDBGEvent &)> custom_handler)
{
  bool handled_events = false;

  /* We must consume every event prior to any generic operations
     that will force a state collection across the device. */
  while (1)
    {
      CUDBGEvent event;

      switch (kind)
	{
	case CUDA_EVENT_SYNC:
	  cuda_debugapi::get_next_sync_event (&event);
	  break;
	case CUDA_EVENT_ASYNC:
	  cuda_debugapi::get_next_async_event (&event);
	  break;
	}

      if (event.kind == CUDBG_EVENT_INVALID)
	break;

      handled_events = true;

      /* Only process events if we don't have a custom handler or if we do and
       * it returns true */
      if (!custom_handler || custom_handler (event))
	cuda_process_event (event);
    }

  /* Acknowledge sync events if we handled any. */
  if ((kind == CUDA_EVENT_SYNC) && handled_events)
    cuda_debugapi::acknowledge_sync_events ();

  return handled_events;
}