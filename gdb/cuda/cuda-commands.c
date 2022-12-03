/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2021 NVIDIA Corporation
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
#include "gdbsupport/forward-scope-exit.h"
#include "cuda-tdep.h"
#include "cuda-options.h"
#include "cuda-parser.h"
#include "valprint.h"
#include "filenames.h"
#include "language.h"
#include "ui-out.h"
#include "ui-file.h"
#include "command.h"
#include "exceptions.h"
#include "gdbcmd.h"
#include "source.h"
#include "symtab.h"
#include "objfiles.h"
#include "breakpoint.h"
#include "cuda-context.h"
#include "cuda-iterator.h"
#include "cuda-kernel.h"
#include "cuda-state.h"
#include "cuda-convvars.h"
#include "cuda-exceptions.h"
#include "cuda-utils.h"
#include "arch-utils.h"
#include "block.h"
#include "cuda-commands.h"

#include "demangle.h"
#include "interps.h"

const char *status_string[] =
  { "Invalid", "Pending", "Active", "Sleeping", "Terminated", "Undetermined" };
const char *status_string_preempted = "Active (preempted)";

/* returned string must be freed */
static char *
get_filename (struct symtab *s)
{
  const char *full_path = NULL;
  char *real_path = NULL;

  if (!s)
    return NULL;

  /* in CLI mode, we only display the filename */
  if (!current_uiout->is_mi_like_p ())
    return xstrdup (s->filename);

  /* in MI mode, we display the canonicalized full path */
  full_path = symtab_to_fullname (s);
  if (!full_path)
    return NULL;

  real_path = realpath (full_path, NULL);
  return real_path;
}

typedef struct {
  /* cuda coordinates filter */
  cuda_coords_t coords;

  /* cuda breakpoints filter.
     Set to -1 to disable this filter.
     Set to 0 to check all cuda breakpoints.
     Set to a breakpoint number to check that breakpoint. */
  int bp_number;

  bool bp_number_p;
} cuda_filters_t;

#define CUDA_INVALID_FILTERS ((cuda_filters_t)              \
    { CUDA_INVALID_COORDS_INIT, 0, false})

#define CUDA_WILDCARD_FILTERS ((cuda_filters_t)             \
    { CUDA_WILDCARD_COORDS_INIT, 0, false })

static void
cuda_parser_request_to_coords (request_t *request, cuda_coords_t *coords)
{
  gdb_assert (request);
  gdb_assert (coords);

  switch (request->type)
    {
    case FILTER_TYPE_DEVICE : coords->dev       = request->value.scalar; break;
    case FILTER_TYPE_SM     : coords->sm        = request->value.scalar; break;
    case FILTER_TYPE_WARP   : coords->wp        = request->value.scalar; break;
    case FILTER_TYPE_LANE   : coords->ln        = request->value.scalar; break;
    case FILTER_TYPE_KERNEL : coords->kernelId  = request->value.scalar; break;
    case FILTER_TYPE_GRID   : coords->gridId    = request->value.scalar; break;
    case FILTER_TYPE_BLOCK  : coords->blockIdx  = request->value.cudim3; break;
    case FILTER_TYPE_THREAD : coords->threadIdx = request->value.cudim3; break;
    default                 : error (_("Unexpected request type."));
    }
}

static void
cuda_parser_result_to_coords (cuda_parser_result_t *result, cuda_coords_t *coords)
{
  uint32_t i;
  request_t *request;

  gdb_assert (result);
  gdb_assert (coords);

  for (i = 0, request = result->requests; i < result->num_requests; ++i, ++request)
    cuda_parser_request_to_coords (request, coords);
}

static void
cuda_parser_result_to_filters (cuda_parser_result_t *result, cuda_filters_t *filters)
{
  uint32_t i;
  request_t *request;

  gdb_assert (result);
  gdb_assert (filters);

  for (i = 0, request = result->requests; i < result->num_requests; ++i, ++request)
    if (request->type == FILTER_TYPE_BREAKPOINT)
      {
        filters->bp_number_p = true;
        filters->bp_number = request->value.scalar;
      }
    else
      cuda_parser_request_to_coords (request, &filters->coords);
}

static cuda_filters_t
cuda_build_filter (const char* filter_string, cuda_filters_t *default_filter, command_t command)
{
  cuda_filters_t filter;
  cuda_parser_result_t *result;

  if (filter_string && *filter_string != 0)
    {
      /* parse the filter string */
      cuda_parser (filter_string, command, &result, CUDA_WILDCARD);
      if (result->command != command)
        error (_("Incorrect filter: '%s'."), filter_string);

      /* build the filter object from the result of the parser */
      filter = CUDA_WILDCARD_FILTERS;
      cuda_parser_result_to_filters (result, &filter);
    }
  else if (default_filter)
    {
      /* expand the filter object */
      filter = *default_filter;
    }
  else
    {
      /* No filter means anything is acceptable */
      filter = CUDA_WILDCARD_FILTERS;
    }

  /* Evaluate the CUDA_CURRENT tokens */
  cuda_coords_evaluate_current (&filter.coords, false);

  /* sanity check */
  cuda_coords_check_fully_defined (&filter.coords, false, false, true);

  return filter;
}

typedef struct {
  bool        current;
  uint32_t    device;
  const char *name;
  const char *description;
  const char *sm_type;
  uint32_t    num_sms;
  uint32_t    num_warps;
  uint32_t    num_lanes;
  uint32_t    num_regs;
  uint32_t    active_sms_mask[(CUDBG_MAX_SMS + 31) / 32];
  uint32_t    pci_bus_id;
  uint32_t    pci_dev_id;
} cuda_info_device_t;

static void
cuda_info_devices (const char *filter_string, cuda_info_device_t **devices, uint32_t *num_devices)
{
  uint32_t num_elements;
  cuda_iterator iter;
  cuda_filters_t default_filter, filter;
  cuda_coords_t c;
  cuda_info_device_t *d;

  /* sanity checks */
  gdb_assert (devices);
  gdb_assert (num_devices);

  /* get the filter */
  default_filter = CUDA_WILDCARD_FILTERS;
  filter = cuda_build_filter (filter_string, &default_filter, CMD_FILTER);

  /* get the list of devices */
  iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_DEVICES, &filter.coords, CUDA_SELECT_ALL);
  num_elements = cuda_iterator_get_size (iter);
  *devices = (cuda_info_device_t *) xcalloc (num_elements, sizeof (**devices));
  *num_devices = 0;

  /* compile the needed info for each device */
  for (cuda_iterator_start (iter), d = *devices;
       !cuda_iterator_end (iter);
       cuda_iterator_next (iter), ++d)
    {
      c  = cuda_iterator_get_current (iter);

      device_get_active_sms_mask (c.dev, d->active_sms_mask);

      d->current         = cuda_coords_is_current (&c);
      d->device          = c.dev;
      d->name            = device_get_device_name (c.dev);
      d->description     = device_get_device_type (c.dev);
      d->sm_type         = device_get_sm_type (c.dev);
      d->num_sms         = device_get_num_sms (c.dev);
      d->num_warps       = device_get_num_warps (c.dev);
      d->num_lanes       = device_get_num_lanes (c.dev);
      d->num_regs        = device_get_num_registers (c.dev);
      d->pci_bus_id      = device_get_pci_bus_id (c.dev);
      d->pci_dev_id      = device_get_pci_dev_id (c.dev);

      ++*num_devices;
    }
}

static void
cuda_info_devices_destroy (cuda_info_device_t *devices)
{
  xfree (devices);
}

void
info_cuda_devices_command (const char *arg)
{
  struct ui_out *uiout = current_uiout;
  cuda_info_device_t *devices, *d;
  uint32_t i, j, num_devices;
  struct { size_t current, device, pci_bus, name, description, sm_type,
             num_sms, num_warps, num_lanes, num_regs, active_sms_mask; } width;

  /* column header */
  const char *header_current         = " ";
  const char *header_device          = "Dev";
  const char *header_pci_bus         = "PCI Bus/Dev ID";
  const char *header_name            = "Name";
  const char *header_description     = "Description";
  const char *header_sm_type         = "SM Type";
  const char *header_num_sms         = "SMs";
  const char *header_num_warps       = "Warps/SM";
  const char *header_num_lanes       = "Lanes/Warp";
  const char *header_num_regs        = "Max Regs/Lane";
  const char *header_active_sms_mask = "Active SMs Mask";

  /* get the information */
  cuda_info_devices (arg, &devices, &num_devices);

  /* output message if the list is empty */
  if (num_devices == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _("No CUDA devices.\n"));
      return;
    }

  /* column widths */
  width.current         = strlen (header_current);
  width.device          = strlen (header_device);
  width.pci_bus         = strlen (header_pci_bus);
  width.name            = strlen (header_name);
  width.description     = strlen (header_description);
  width.sm_type         = strlen (header_sm_type);
  width.num_sms         = strlen (header_num_sms);
  width.num_warps       = strlen (header_num_warps);
  width.num_lanes       = strlen (header_num_lanes);
  width.num_regs        = strlen (header_num_regs);
  width.active_sms_mask = strlen (header_active_sms_mask);

  for (d = devices, i = 0; i < num_devices; ++i, ++d)
    {
      width.name            = std::max (width.name, strlen (d->name));
      width.description     = std::max (width.description, strlen (d->description));
      width.sm_type         = std::max (width.sm_type,     strlen (d->sm_type));
      width.active_sms_mask = std::max (width.active_sms_mask, 2 + 2 * sizeof(d->active_sms_mask));
    }

  {
    /* print table header */
    ui_out_emit_table table_emitter (uiout, 11, num_devices, "InfoCudaDevicesTable");
    uiout->table_header (width.current         , ui_right, "current"         , header_current);
    uiout->table_header (width.device          , ui_right, "device"          , header_device);
    uiout->table_header (width.pci_bus         , ui_right, "pci_bus"         , header_pci_bus);
    uiout->table_header (width.name            , ui_right, "name"            , header_name);
    uiout->table_header (width.description     , ui_right, "description"     , header_description);
    uiout->table_header (width.sm_type         , ui_right, "sm_type"         , header_sm_type);
    uiout->table_header (width.num_sms         , ui_right, "num_sms"         , header_num_sms);
    uiout->table_header (width.num_warps       , ui_right, "num_warps"       , header_num_warps);
    uiout->table_header (width.num_lanes       , ui_right, "num_lanes"       , header_num_lanes);
    uiout->table_header (width.num_regs        , ui_right, "num_regs"        , header_num_regs);
    uiout->table_header (width.active_sms_mask , ui_right, "active_sms_mask" , header_active_sms_mask);
    uiout->table_body ();

    /* print table rows */
    for (d = devices, i = 0; i < num_devices; ++i, ++d)
      {
	char pci_bus[32];
	char active_sms_mask[3 + 2 * sizeof(d->active_sms_mask)] = { "0x" };

	for (j = 0; j < ARRAY_SIZE(d->active_sms_mask); ++j)
	  snprintf (active_sms_mask + 2 + j * 2 * sizeof(*d->active_sms_mask), 1 + 2 * sizeof(*d->active_sms_mask),
		    "%08x", d->active_sms_mask[ARRAY_SIZE(d->active_sms_mask) - j - 1]);
	snprintf (pci_bus, sizeof(pci_bus), "%02x:%02x.0", d->pci_bus_id, d->pci_dev_id);

	ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaDevicesRow");
	uiout->field_string ("current"        , d->current ? "*" : " ");
	uiout->field_signed    ("device"         , d->device);
	uiout->field_string ("pci_bus"        , pci_bus);
	uiout->field_string ("name"           , d->name);
	uiout->field_string ("description"    , d->description);
	uiout->field_string ("sm_type"        , d->sm_type);
	uiout->field_signed    ("num_sms"        , d->num_sms);
	uiout->field_signed	  ("num_warps"      , d->num_warps);
	uiout->field_signed	  ("num_lanes"      , d->num_lanes);
	uiout->field_signed	  ("num_regs"       , d->num_regs);
	uiout->field_string ("active_sms_mask", active_sms_mask);
	uiout->text         ("\n");
      }
  }
  
  gdb_flush (gdb_stdout);

  cuda_info_devices_destroy (devices);
}

typedef struct {
  bool     current;
  uint32_t device;
  uint32_t sm;
  cuda_api_warpmask active_warps_mask;
} cuda_info_sm_t;

static void
cuda_info_sms (const char *filter_string, cuda_info_sm_t **sms, uint32_t *num_sms)
{
  uint32_t num_elements;
  cuda_iterator iter;
  cuda_filters_t default_filter, filter;
  cuda_coords_t c;
  cuda_info_sm_t *s;

  /* sanity checks */
  gdb_assert (sms);
  gdb_assert (num_sms);

  /* set the filter */
  default_filter = CUDA_WILDCARD_FILTERS;
  default_filter.coords.dev = CUDA_CURRENT;
  filter = cuda_build_filter (filter_string, &default_filter, CMD_FILTER);

  /* get the list of sms */
  iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_SMS, &filter.coords, CUDA_SELECT_ALL);
  num_elements = cuda_iterator_get_size (iter);
  *sms = (cuda_info_sm_t *) xmalloc (num_elements * sizeof (**sms));
  *num_sms = 0;

  /* compile the needed info for each device */
  for (cuda_iterator_start (iter), s = *sms;
       !cuda_iterator_end (iter);
       cuda_iterator_next (iter), ++s)
    {
      c  = cuda_iterator_get_current (iter);

      s->current           = cuda_coords_is_current (&c);
      s->device            = c.dev;
      s->sm                = c.sm;
      cuda_api_cp_mask(&s->active_warps_mask, sm_get_valid_warps_mask(c.dev, c.sm));

      ++*num_sms;
    }

  cuda_iterator_destroy (iter);
}

static void
cuda_info_sms_destroy (cuda_info_sm_t *sms)
{
  xfree (sms);
}

void
info_cuda_sms_command (const char *arg)
{
  cuda_info_sm_t *sms, *s;
  uint32_t i, num_sms, current_device;
  struct { size_t current, sm, active_warps_mask; } width;

  /* column headers */
  const char *header_current            = " ";
  const char *header_sm                 = "SM";
  const char *header_active_warps_mask  = "Active Warps Mask";
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_info_sms (arg, &sms, &num_sms);

  /* output message if the list is empty */
  if (num_sms == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _("No CUDA SMs.\n"));
      return;
    }

  /* column widths */
  width.current            = strlen (header_current);
  width.sm                 = strlen (header_sm);
  width.active_warps_mask  = strlen (header_active_warps_mask);

  width.active_warps_mask  = std::max (width.active_warps_mask, (size_t)18);

  {
    ui_out_emit_table table_emitter (uiout, 3, num_sms, "InfoCudaSmsTable");

    /* print table header */
    uiout->table_header (width.current          , ui_right, "current"           , header_current);
    uiout->table_header (width.sm               , ui_right, "sm"                , header_sm);
    uiout->table_header (width.active_warps_mask, ui_right, "active_warps_mask" , header_active_warps_mask);
    uiout->table_body ();

    /* print table rows */
    for (s = sms, i = 0, current_device = -1; i < num_sms; ++i, ++s)
      {
	if (!uiout->is_mi_like_p () && s->device != current_device)
	  {
	    uiout->message ("Device %u\n", s->device);
	    current_device = s->device;
	  }

	ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaSmsRow");
	uiout->field_string ("current"          , s->current ? "*" : " ");
	uiout->field_signed    ("sm"               , s->sm);
	uiout->field_fmt    ("active_warps_mask", "%" WARP_MASK_FORMAT, cuda_api_mask_string(&s->active_warps_mask));
	uiout->text         ("\n");
      }
  }
  
  gdb_flush (gdb_stdout);

  cuda_info_sms_destroy (sms);
}

typedef struct {
  bool     current;
  uint32_t device;
  uint32_t sm;
  uint32_t wp;
  char     active_lanes_mask[11];
  char     divergent_lanes_mask[11];
  char     active_physical_pc[19];
  char     kernel_id[6];
  char     blockIdx[32];
  char     threadIdx[32];
} cuda_info_warp_t;

static void
cuda_info_warps (const char *filter_string, cuda_info_warp_t **warps, uint32_t *num_warps)
{
  uint32_t num_elements;
  cuda_iterator iter;
  cuda_filters_t default_filter, filter;
  cuda_coords_t c;
  cuda_info_warp_t *w;
  uint64_t active_physical_pc;
  uint32_t kernel_id, active_lanes_mask, divergent_lanes_mask;
  CuDim3 blockIdx;
  CuDim3 threadIdx;
  kernel_t kernel;

  /* sanity checks */
  gdb_assert (warps);
  gdb_assert (num_warps);

  /* set the filter */
  default_filter = CUDA_WILDCARD_FILTERS;
  default_filter.coords.dev = CUDA_CURRENT;
  default_filter.coords.sm  = CUDA_CURRENT;
  filter = cuda_build_filter (filter_string, &default_filter, CMD_FILTER);

  /* get the list of sms */
  iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_WARPS, &filter.coords, CUDA_SELECT_ALL);
  num_elements = cuda_iterator_get_size (iter);
  *warps = (cuda_info_warp_t *) xmalloc (num_elements * sizeof (**warps));
  *num_warps = 0;

  /* compile the needed info for each device */
  for (cuda_iterator_start (iter), w = *warps;
       !cuda_iterator_end (iter);
       cuda_iterator_next (iter), ++w)
    {
      c  = cuda_iterator_get_current (iter);

      w->current              = cuda_coords_is_current (&c);
      w->device               = c.dev;
      w->sm                   = c.sm;
      w->wp                   = c.wp;

      if (warp_is_valid (c.dev, c.sm, c.wp))
        {
          active_lanes_mask    = warp_get_active_lanes_mask (c.dev, c.sm, c.wp);
          divergent_lanes_mask = warp_get_divergent_lanes_mask (c.dev, c.sm, c.wp);
          kernel               = warp_get_kernel (c.dev, c.sm, c.wp);
          kernel_id            = kernel_get_id (kernel);
          blockIdx             = warp_get_block_idx (c.dev, c.sm, c.wp);
          active_physical_pc   = warp_get_active_pc (c.dev, c.sm, c.wp);
          threadIdx            = lane_get_thread_idx (c.dev, c.sm, c.wp, __builtin_ctz(active_lanes_mask));

          snprintf (w->active_lanes_mask    , sizeof (w->active_lanes_mask)    , "0x%08x"      , active_lanes_mask);
          snprintf (w->divergent_lanes_mask , sizeof (w->divergent_lanes_mask) , "0x%08x"      , divergent_lanes_mask);
          snprintf (w->kernel_id            , sizeof (w->kernel_id)            , "%u"          , kernel_id);
          snprintf (w->blockIdx             , sizeof (w->blockIdx)             , "(%u,%u,%u)"  , blockIdx.x, blockIdx.y, blockIdx.z);
          snprintf (w->threadIdx            , sizeof (w->threadIdx)            , "(%u,%u,%u)"  , threadIdx.x, threadIdx.y, threadIdx.z);
          snprintf (w->active_physical_pc   , sizeof (w->active_physical_pc)   , "0x%016llx"   , (unsigned long long)active_physical_pc);
        }
      else
        {
          snprintf (w->active_lanes_mask    , sizeof (w->active_lanes_mask)    , "0x%08x", 0);
          snprintf (w->divergent_lanes_mask , sizeof (w->divergent_lanes_mask) , "0x%08x", 0);
          snprintf (w->kernel_id            , sizeof (w->kernel_id)            , "n/a");
          snprintf (w->blockIdx             , sizeof (w->blockIdx)             , "n/a");
          snprintf (w->threadIdx            , sizeof (w->threadIdx)            , "n/a");
          snprintf (w->active_physical_pc   , sizeof (w->active_physical_pc   ), "n/a");
        }

      ++*num_warps;
    }

  cuda_iterator_destroy (iter);
}

static void
cuda_info_warps_destroy (cuda_info_warp_t *warps)
{
  xfree (warps);
}

void
info_cuda_warps_command (const char *arg)
{
  cuda_info_warp_t *warps, *w;
  uint32_t i, num_warps, current_device, current_sm;
  struct { size_t current, wp, active_lanes_mask, divergent_lanes_mask, active_physical_pc, kernel_id, blockIdx,threadIdx; } width;

  /* column headers */
  const char *header_current              = " ";
  const char *header_wp                   = "Wp";
  const char *header_active_lanes_mask    = "Active Lanes Mask";
  const char *header_divergent_lanes_mask = "Divergent Lanes Mask";
  const char *header_active_physical_pc   = "Active Physical PC";
  const char *header_kernel_id            = "Kernel";
  const char *header_blockIdx             = "BlockIdx";
  const char *header_threadIdx            = "First Active ThreadIdx";
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_info_warps (arg, &warps, &num_warps);

  /* output message if the list is empty */
  if (num_warps == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _("No CUDA Warps.\n"));
      return;
    }

  /* column widths */
  width.current              = strlen (header_current);
  width.wp                   = strlen (header_wp);
  width.active_lanes_mask    = strlen (header_active_lanes_mask);
  width.divergent_lanes_mask = strlen (header_divergent_lanes_mask);
  width.active_physical_pc   = strlen (header_active_physical_pc);
  width.kernel_id            = strlen (header_kernel_id);
  width.blockIdx             = strlen (header_blockIdx);
  width.threadIdx            = strlen (header_threadIdx);

  width.active_lanes_mask    = std::max (width.active_lanes_mask, (size_t)10);
  width.divergent_lanes_mask = std::max (width.divergent_lanes_mask, (size_t)10);
  width.active_physical_pc   = std::max (width.active_physical_pc, (size_t)18);
  for (w = warps, i = 0;i < num_warps; ++i, ++w)
    {
      width.blockIdx = std::max (width.blockIdx, strlen (w->blockIdx));
      width.threadIdx = std::max (width.threadIdx, strlen (w->threadIdx));
    }

  {
    /* print table header */
    ui_out_emit_table table_emitter (uiout, 8, num_warps, "InfoCudaWarpsTable");
    uiout->table_header (width.current             , ui_right, "current"             , header_current);
    uiout->table_header (width.wp                  , ui_right, "warp"                , header_wp);
    uiout->table_header (width.active_lanes_mask   , ui_right, "active_lanes_mask"   , header_active_lanes_mask);
    uiout->table_header (width.divergent_lanes_mask, ui_right, "divergent_lanes_mask", header_divergent_lanes_mask);
    uiout->table_header (width.active_physical_pc  , ui_right, "active_physical_pc"  , header_active_physical_pc);
    uiout->table_header (width.kernel_id           , ui_right, "kernel"              , header_kernel_id);
    uiout->table_header (width.blockIdx            , ui_right, "blockIdx"            , header_blockIdx);
    uiout->table_header (width.threadIdx           , ui_right, "threadIdx"           , header_threadIdx);
    uiout->table_body ();

    /* print table rows */
    for (w = warps, i = 0, current_device = -1, current_sm = -1; i < num_warps; ++i, ++w)
      {

	if (!uiout->is_mi_like_p () &&
	    (w->device != current_device || w->sm != current_sm))
	  {
	    uiout->message ("Device %u SM %u\n", w->device, w->sm);
	    current_device = w->device;
	    current_sm     = w->sm;
	  }

	ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaWarpsRow");
	uiout->field_string ("current"             , w->current ? "*" : " ");
	uiout->field_signed    ("warp"                , w->wp);
	uiout->field_string ("active_lanes_mask"   , w->active_lanes_mask);
	uiout->field_string ("divergent_lanes_mask", w->divergent_lanes_mask);
	uiout->field_string ("active_physical_pc"  , w->active_physical_pc);
	uiout->field_string ("kernel"              , w->kernel_id);
	uiout->field_string ("blockIdx"            , w->blockIdx);
	uiout->field_string ("threadIdx"           , w->threadIdx);
	uiout->text         ("\n");
      }
  }
  
  gdb_flush (gdb_stdout);

  cuda_info_warps_destroy (warps);
}

typedef struct {
  bool     current;
  char     state[20];
  uint32_t device;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  char     physical_pc[19];
  char     threadIdx[32];
  const char     *exception;
} cuda_info_lane_t;

static void
cuda_info_lanes (const char *filter_string, cuda_info_lane_t **lanes, uint32_t *num_lanes)
{
  CuDim3 threadIdx;
  uint32_t num_elements;
  cuda_iterator iter;
  cuda_filters_t default_filter, filter;
  cuda_coords_t c;
  cuda_info_lane_t *l;
  uint64_t physical_pc;
  bool active;
  CUDBGException_t exception;

  /* sanity checks */
  gdb_assert (lanes);
  gdb_assert (num_lanes);

  /* set the filter */
  default_filter = CUDA_WILDCARD_FILTERS;
  default_filter.coords.dev = CUDA_CURRENT;
  default_filter.coords.sm  = CUDA_CURRENT;
  default_filter.coords.wp  = CUDA_CURRENT;
  filter = cuda_build_filter (filter_string, &default_filter, CMD_FILTER);

  /* get the list of lanes */
  iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_LANES, &filter.coords, CUDA_SELECT_ALL);
  num_elements = cuda_iterator_get_size (iter);
  *lanes = (cuda_info_lane_t *) xmalloc (num_elements * sizeof (**lanes));
  *num_lanes = 0;

  /* compile the needed info for each device */
  for (cuda_iterator_start (iter), l = *lanes;
       !cuda_iterator_end (iter);
       cuda_iterator_next (iter), ++l)
    {
      c  = cuda_iterator_get_current (iter);

      l->current     = cuda_coords_is_current (&c);
      l->device      = c.dev;
      l->sm          = c.sm;
      l->wp          = c.wp;
      l->ln          = c.ln;

      if (lane_is_valid (c.dev, c.sm, c.wp, c.ln))
        {
          active      = lane_is_active (c.dev, c.sm, c.wp, c.ln);
          threadIdx   = lane_get_thread_idx (c.dev, c.sm, c.wp,c .ln);
          physical_pc = lane_get_pc (c.dev, c.sm, c.wp, c.ln);
          exception   = lane_get_exception (c.dev, c.sm, c.wp, c.ln);

          snprintf (l->state      , sizeof (l->state)      , "%s", active ? "active" : "divergent");
          snprintf (l->threadIdx  , sizeof (l->threadIdx)  , "(%u,%u,%u)", threadIdx.x, threadIdx.y, threadIdx.z);
          snprintf (l->physical_pc, sizeof (l->physical_pc), "0x%016llx", (unsigned long long)physical_pc);
          l->exception = exception == CUDBG_EXCEPTION_NONE ? "None" : cuda_exception_type_to_name (exception);
        }
      else
        {
          snprintf (l->state      , sizeof (l->state)      , "inactive");
          snprintf (l->threadIdx  , sizeof (l->threadIdx)  , "n/a");
          snprintf (l->physical_pc, sizeof (l->physical_pc), "n/a");
          l->exception = "n/a";
        }

      ++*num_lanes;
    }

  cuda_iterator_destroy (iter);
}

static void
cuda_info_lanes_destroy (cuda_info_lane_t *lanes)
{
  xfree (lanes);
}

void
info_cuda_lanes_command (const char *arg)
{
  cuda_info_lane_t *lanes, *l;
  uint32_t i, num_lanes, current_device, current_sm, current_wp;
  struct { size_t current, ln, state, physical_pc, thread_idx, exception; } width;

  /* column headers */
  const char *header_current     = " ";
  const char *header_ln          = "Ln";
  const char *header_state       = "State";
  const char *header_physical_pc = "Physical PC";
  const char *header_thread_idx  = "ThreadIdx";
  const char *header_exception   = "Exception";
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_info_lanes (arg, &lanes, &num_lanes);

  /* output message if the list is empty */
  if (num_lanes == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _("No CUDA Lanes.\n"));
      return;
    }

  /* column widths */
  width.current     = strlen (header_current);
  width.ln          = strlen (header_ln);
  width.state       = strlen (header_state);
  width.physical_pc = strlen (header_physical_pc);
  width.thread_idx  = strlen (header_thread_idx);
  width.exception  = strlen (header_exception);

  width.state       = std::max (width.state, strlen ("divergent"));
  width.physical_pc = std::max (width.physical_pc, (size_t)18);
  for (l = lanes, i = 0; i < num_lanes; ++i, ++l)
    {
      width.thread_idx = std::max (width.thread_idx, strlen (l->threadIdx));
      width.exception = std::max (width.exception, strlen (l->exception));
    }

  {
    /* print table header */
    ui_out_emit_table table_emitter (uiout, 6, num_lanes, "InfoCudaLanesTable");
    uiout->table_header (width.current     , ui_right, "current"     , header_current);
    uiout->table_header (width.ln          , ui_right, "lane"        , header_ln);
    uiout->table_header (width.state       , ui_center, "state"       , header_state);
    uiout->table_header (width.physical_pc , ui_center, "physical_pc" , header_physical_pc);
    uiout->table_header (width.thread_idx  , ui_right, "threadIdx"   , header_thread_idx);
    uiout->table_header (width.exception   , ui_center, "exception"   , header_exception);
    uiout->table_body ();

    /* print table rows */
    for (l = lanes, i = 0, current_device = -1, current_sm = -1, current_wp = -1; i < num_lanes; ++i, ++l)
      {
	if (!uiout->is_mi_like_p () &&
	    (l->device != current_device || l->sm != current_sm || l->wp != current_wp))
	  {
	    uiout->message ("Device %u SM %u Warp %u\n", l->device, l->sm, l->wp);
	    current_device = l->device;
	    current_sm     = l->sm;
	    current_wp     = l->wp;
	  }

	ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaLanesRow");
	uiout->field_string ("current"    , l->current ? "*" : " ");
	uiout->field_signed    ("lane"       , l->ln);
	uiout->field_string ("state"      , l->state);
	uiout->field_string ("physical_pc", l->physical_pc);
	uiout->field_string ("threadIdx"  , l->threadIdx);
	uiout->field_string ("exception"  , l->exception);
	uiout->text         ("\n");
      }
  }
  
  gdb_flush (gdb_stdout);

  cuda_info_lanes_destroy (lanes);
}

typedef struct {
  bool           current;
  kernel_t       kernel;
  uint32_t       kernel_id;
  uint32_t       device;
  long long      grid_id;
  char           parent[32];
  uint32_t       sms_mask;
  const char *   status;
  char           grid_dim[32];
  char           block_dim[32];
  char          *invocation;
} cuda_info_kernel_t;

static void
cuda_info_kernels_build (const char *filter_string, cuda_info_kernel_t **kernels, uint32_t *num_kernels)
{
  kernel_t kernel, parent_kernel;
  cuda_info_kernel_t *k;
  CuDim3 grid_dim;
  CuDim3 block_dim;
  const char *args;
  const char *name;
  int len;

  /* sanity checks */
  gdb_assert (kernels);
  gdb_assert (num_kernels);

  /* build the list of kernels */
  *num_kernels = cuda_system_get_num_present_kernels ();
  *kernels = (cuda_info_kernel_t *) xmalloc (*num_kernels * sizeof (**kernels));

  /* compile the needed info for each kernel */
  k = *kernels;
  for (kernel = kernels_get_first_kernel (); kernel; kernel = kernels_get_next_kernel (kernel))
    {
      CUDBGGridStatus status;

      if (!kernel_is_present (kernel))
        continue;

      grid_dim = kernel_get_grid_dim (kernel);
      block_dim = kernel_get_block_dim (kernel);
      parent_kernel = kernel_get_parent (kernel);
      name = kernel_get_name (kernel);
      args = kernel_get_args (kernel);

      if (!args)
        args = "n/a";

      k->kernel     = kernel;
      k->current    = kernel == cuda_current_kernel ();
      k->kernel_id  = kernel_get_id (kernel);
      k->device     = kernel_get_dev_id (kernel);
      k->grid_id    = kernel_get_grid_id (kernel);
      k->sms_mask   = kernel_compute_sms_mask (kernel);

      status = kernel_get_status (kernel);
      k->status = cuda_options_software_preemption () && status == CUDBG_GRID_STATUS_ACTIVE
                ? status_string_preempted
                : status_string[status];

      if (parent_kernel)
        snprintf (k->parent, sizeof (k->parent), "%llu",
                  (unsigned long long)kernel_get_id (parent_kernel));
      else
        snprintf (k->parent, sizeof (k->parent), "-");
      snprintf(k->grid_dim, sizeof (k->grid_dim), "(%u,%u,%u)",
               grid_dim.x, grid_dim.y, grid_dim.z);
      snprintf(k->block_dim, sizeof (k->block_dim), "(%u,%u,%u)",
               block_dim.x, block_dim.y, block_dim.z);

      len = strlen (name) + 2 + strlen (args) + 1;
      k->invocation = (char *) xmalloc (len);
      snprintf (k->invocation, len, "%s(%s)", name, args);

      ++k;
    }
}

static void
cuda_info_kernels_destroy (cuda_info_kernel_t *kernels)
{
  xfree (kernels);
}

void
info_cuda_kernels_command (const char *arg)
{
  cuda_info_kernel_t *kernels, *k;
  uint32_t i, num_kernels;
  struct { size_t current, kernel, device, grid, parent, status, sms_mask, grid_dim, block_dim, invocation; } width;

  /* column headers */
  const char *header_current   = " ";
  const char *header_kernel    = "Kernel";
  const char *header_device    = "Dev";
  const char *header_parent    = "Parent";
  const char *header_grid      = "Grid";
  const char *header_status    = "Status";
  const char *header_sms_mask  = "SMs Mask";
  const char *header_grid_dim  = "GridDim";
  const char *header_block_dim = "BlockDim";
  const char *header_invocation= "Invocation";
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_info_kernels_build (arg, &kernels, &num_kernels);

  /* output message if the list is empty */
  if (num_kernels == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _("No CUDA kernels.\n"));
      return;
    }

  /* column widths */
  width.current   = strlen (header_current);
  width.kernel    = strlen (header_kernel);
  width.device    = strlen (header_device);
  width.grid      = strlen (header_grid);
  width.status    = strlen (header_status);
  width.parent    = strlen (header_parent);
  width.sms_mask  = strlen (header_sms_mask);
  width.grid_dim  = strlen (header_grid_dim);
  width.block_dim = strlen (header_block_dim);
  width.invocation= strlen (header_invocation);

  for (k = kernels, i = 0; i < num_kernels; ++i, ++k)
    {
      width.status    = std::max (width.status, strlen (k->status));
      width.parent    = std::max (width.parent, strlen (k->parent));
      width.sms_mask  = std::max (width.sms_mask, (size_t)10);
      width.grid_dim  = std::max (width.grid_dim,  strlen (k->grid_dim));
      width.block_dim = std::max (width.block_dim, strlen (k->block_dim));
      width.invocation= std::max (width.invocation, strlen (header_invocation));
    }

  {
    /* print table header */
    ui_out_emit_table table_emitter (uiout, 10, num_kernels, "InfoCudaKernelsTable");
    uiout->table_header (width.current  , ui_right, "current"  , header_current);
    uiout->table_header (width.kernel   , ui_right, "kernel"   , header_kernel);
    uiout->table_header (width.parent   , ui_right, "parent"   , header_parent);
    uiout->table_header (width.device   , ui_right, "device"   , header_device);
    uiout->table_header (width.grid     , ui_right, "grid"     , header_grid);
    uiout->table_header (width.status   , ui_right, "status"   , header_status);
    uiout->table_header (width.sms_mask , ui_right, "sms_mask" , header_sms_mask);
    uiout->table_header (width.grid_dim , ui_right, "gridDim"  , header_grid_dim);
    uiout->table_header (width.block_dim, ui_right, "blockDim" , header_block_dim);
    uiout->table_header (width.invocation,ui_left , "invocation", header_invocation);
    uiout->table_body ();

    /* print table rows */
    for (k = kernels, i = 0; i < num_kernels; ++i, ++k)
      {
	ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaKernelsRow");
	uiout->field_string    ("current" , k->current ? "*" : " ");
	uiout->field_signed       ("kernel"  , k->kernel_id);
	uiout->field_string    ("parent"  , k->parent);
	uiout->field_signed       ("device"  , k->device);
	uiout->field_long_long ("grid"    , k->grid_id);
	uiout->field_string    ("status"  , k->status);
	uiout->field_fmt       ("sms_mask", "0x%08x", k->sms_mask);
	uiout->field_string    ("gridDim" , k->grid_dim);
	uiout->field_string    ("blockDim", k->block_dim);
	uiout->field_string    ("invocation", k->invocation);
	uiout->text            ("\n");
      }
  }
  
  gdb_flush (gdb_stdout);

  cuda_info_kernels_destroy (kernels);
}

typedef struct {
  bool           current;
  kernel_t       kernel;
  uint64_t       kernel_id;
  CuDim3         start_block_idx;
  CuDim3         end_block_idx;
  char           invocation[1024];
  const char    *kernel_dim;
  char           start_block_idx_string[32];
  char           end_block_idx_string[32];
  uint32_t       count;
  uint32_t       device;
  uint32_t       sm;
} cuda_info_block_t;

static void
cuda_info_blocks_build (const char *filter_string, cuda_info_block_t **blocks, uint32_t *num_blocks)
{
  uint32_t i, num_elements;
  cuda_iterator iter;
  cuda_filters_t default_filter, filter;
  cuda_coords_t c, expected;
  CuDim3 prev_block_idx = { CUDA_INVALID, CUDA_INVALID, CUDA_INVALID };
  kernel_t kernel;
  cuda_info_block_t *b;
  bool first_entry, break_of_contiguity;

  /* sanity checks */
  gdb_assert (blocks);
  gdb_assert (num_blocks);

  /* make valgrind not complain */
  expected = CUDA_INVALID_COORDS;

  /* get the filter */
  default_filter = CUDA_WILDCARD_FILTERS;
  default_filter.coords.kernelId = CUDA_CURRENT;
  filter = cuda_build_filter (filter_string, &default_filter, CMD_FILTER);

  /* get the list of blocks */
  iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_BLOCKS, &filter.coords, CUDA_SELECT_VALID);
  num_elements = cuda_iterator_get_size (iter);
  *blocks = (cuda_info_block_t *) xmalloc (num_elements * sizeof (**blocks));
  *num_blocks = 0;

  /* compile the needed info for each block */
  for (cuda_iterator_start (iter), first_entry = true, i = 0, b = *blocks;
       !cuda_iterator_end (iter);
       cuda_iterator_next (iter),  first_entry = false, ++i)
    {
      c  = cuda_iterator_get_current (iter);
      kernel = kernels_find_kernel_by_grid_id (c.dev, c.gridId);

      /* data for the current iteration */
      break_of_contiguity = cuda_coords_compare_logical (&expected, &c) != 0;

      /* close the current range */
      if (!first_entry && (break_of_contiguity || !cuda_options_coalescing ()))
        {
          b->end_block_idx = prev_block_idx;
          snprintf (b->end_block_idx_string, sizeof (b->end_block_idx_string),
                    "(%u,%u,%u)", prev_block_idx.x, prev_block_idx.y, prev_block_idx.z);
          ++b;
          ++*num_blocks;
        }

      /* start a new range */
      if (first_entry || break_of_contiguity || !cuda_options_coalescing ())
        {
          b->kernel          = kernel;
          b->current         = false;
          b->start_block_idx = c.blockIdx;
          b->count           = 0;
          b->kernel_id       = kernel_get_id (kernel);
          b->kernel_dim      = kernel_get_dimensions (kernel);
          b->device          = c.dev;
          b->sm              = c.sm;
          snprintf (b->start_block_idx_string, sizeof (b->start_block_idx_string),
                    "(%u,%u,%u)", c.blockIdx.x, c.blockIdx.y, c.blockIdx.z);
        }

      /* update the current range */
      b->current |= cuda_coords_is_current (&c);
      ++b->count;

      /* data for the next iteration */
      prev_block_idx = c.blockIdx;
      expected = CUDA_WILDCARD_COORDS;
      expected.kernelId = c.kernelId;
      expected.blockIdx = c.blockIdx;
      cuda_coords_increment_block (&expected, kernel_get_grid_dim (kernel));
    }

  /* close the last range */
  if (num_elements > 0)
    {
      b->end_block_idx = c.blockIdx;
      snprintf (b->end_block_idx_string, sizeof (b->end_block_idx_string),
                "(%u,%u,%u)", c.blockIdx.x, c.blockIdx.y, c.blockIdx.z);
      ++*num_blocks;
    }

  cuda_iterator_destroy (iter);
}

static void
cuda_info_blocks_destroy (cuda_info_block_t *blocks)
{
  xfree (blocks);
}

static void
info_cuda_blocks_print_uncoalesced (cuda_info_block_t *blocks, uint32_t num_blocks)
{
  cuda_info_block_t *b;
  uint32_t i, num_columns;
  uint64_t kernel_id;
  struct { size_t current, kernel, block_idx, state, device, sm; } width;

  /* column headers */
  const char *header_current   = " ";
  const char *header_kernel    = "Kernel";
  const char *header_block_idx = "BlockIdx";
  const char *header_state     = "State";
  const char *header_device    = "Dev";
  const char *header_sm        = "SM";
  struct ui_out *uiout = current_uiout;

  /* sanity checks */
  gdb_assert (!cuda_options_coalescing ());

  /* output message if the list is empty */
  if (num_blocks == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _("No CUDA blocks.\n"));
      return;
    }

  /* column widths */
  width.current   = strlen (header_current);
  width.kernel    = strlen (header_kernel);
  width.block_idx = strlen (header_block_idx);
  width.state     = strlen (header_state);
  width.device    = strlen (header_device);
  width.sm        = strlen (header_sm);

  width.state = std::max (width.state, sizeof ("running") - 1);
  for (b = blocks, i = 0; i < num_blocks; ++i, ++b)
    width.block_idx = std::max (width.block_idx, strlen (b->start_block_idx_string));

  /* print table header ('kernel' is only present in MI output) */
  num_columns = uiout->is_mi_like_p () ? 6 : 5;

  ui_out_emit_table table_emitter (uiout, num_columns, num_blocks, "UncoalescedInfoCudaBlocksTable");
  uiout->table_header (width.current  , ui_right, "current"  , header_current);
  if (uiout->is_mi_like_p ())
    uiout->table_header (width.kernel , ui_right, "kernel"   , header_kernel);
  uiout->table_header (width.block_idx, ui_right, "blockIdx" , header_block_idx);
  uiout->table_header (width.state    , ui_right, "state"    , header_state);
  uiout->table_header (width.device   , ui_right, "device"   , header_device);
  uiout->table_header (width.sm       , ui_right, "sm"       , header_sm);
  uiout->table_body ();

  /* print table rows */
  for (b = blocks, i = 0, kernel_id = ~0ULL; i < num_blocks; ++i, ++b)
    {
      if (!uiout->is_mi_like_p () && b->kernel_id != kernel_id)
        {
          /* row are grouped per kernel only in CLI output */
          uiout->message ("Kernel %llu\n", (unsigned long long)b->kernel_id),
          kernel_id = b->kernel_id;
        }

      ui_out_emit_tuple tuple_emitter (uiout, "UncoalescedInfoCudaBlocksRow");
      uiout->field_string ("current" , b->current ? "*" : " ");
      if (uiout->is_mi_like_p ())
        uiout->field_signed  ("kernel"  , b->kernel_id);
      uiout->field_string ("blockIdx", b->start_block_idx_string);
      uiout->field_string ("state"   , "running");
      uiout->field_signed	  ("device"  , b->device);
      uiout->field_signed	  ("sm"      , b->sm);
      uiout->text         ("\n");
    }
}

static void
info_cuda_blocks_print_coalesced (cuda_info_block_t *blocks, uint32_t num_blocks)
{
  cuda_info_block_t *b;
  uint32_t i, num_columns;
  uint64_t kernel_id;
  struct { size_t current, kernel, from, to, count, state; } width;

  /* column headers */
  const char *header_current   = " ";
  const char *header_kernel    = "Kernel";
  const char *header_from      = "BlockIdx";
  const char *header_to        = "To BlockIdx";
  const char *header_count     = "Count";
  const char *header_state     = "State";
  struct ui_out *uiout = current_uiout;

  /* sanity checks */
  gdb_assert (cuda_options_coalescing ());

  /* output message if the list is empty */
  if (num_blocks == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _("No CUDA blocks.\n"));
      return;
    }

  /* column widths */
  width.current   = strlen (header_current);
  width.kernel    = strlen (header_kernel);
  width.from      = strlen (header_from);
  width.to        = strlen (header_to);
  width.count     = strlen (header_count);
  width.state     = strlen (header_state);

  width.state = std::max (width.state, sizeof ("running") - 1);
  for (b = blocks, i = 0; i < num_blocks; ++i, ++b)
    {
      width.from = std::max (width.from, strlen (b->start_block_idx_string));
      width.to   = std::max (width.to  , strlen (b->end_block_idx_string));
    }

  /* print table header ('kernel' is only present in MI output) */
  num_columns = uiout->is_mi_like_p () ? 6 : 5;

  ui_out_emit_table table_emitter (uiout, num_columns, num_blocks, "CoalescedInfoCudaBlocksTable");
  uiout->table_header (width.current  , ui_right, "current"  , header_current);
  if (uiout->is_mi_like_p ())
    uiout->table_header (width.kernel , ui_right, "kernel"   , header_kernel);
  uiout->table_header (width.from     , ui_right, "from"     , header_from);
  uiout->table_header (width.to       , ui_right, "to"       , header_to);
  uiout->table_header (width.count    , ui_right, "count"    , header_count);
  uiout->table_header (width.state    , ui_right, "state"    , header_state);
  uiout->table_body   ();

  /* print table rows */
  for (b = blocks, i = 0, kernel_id = ~0ULL; i < num_blocks; ++i, ++b)
    {
      if (!uiout->is_mi_like_p () && b->kernel_id != kernel_id)
        {
          /* row are grouped per kernel only in CLI output */
          uiout->message ("Kernel %llu\n", (unsigned long long)b->kernel_id),
          kernel_id = b->kernel_id;
        }

      ui_out_emit_tuple tuple_emitter (uiout, "CoalescedInfoCudaBlocksRow");
      uiout->field_string ("current" , b->current ? "*" : " ");
      if (uiout->is_mi_like_p ())
        uiout->field_signed  ("kernel"  , b->kernel_id);
      uiout->field_string ("from"    , b->start_block_idx_string);
      uiout->field_string ("to"      , b->end_block_idx_string);
      uiout->field_signed	  ("count"   , b->count);
      uiout->field_string ("state"   , "running");
      uiout->text         ("\n");
    }
}

void
info_cuda_blocks_command (const char *arg)
{
  cuda_info_block_t *blocks;
  uint32_t num_blocks;

  cuda_info_blocks_build (arg, &blocks, &num_blocks);

  if (cuda_options_coalescing ())
    info_cuda_blocks_print_coalesced (blocks, num_blocks);
  else
    info_cuda_blocks_print_uncoalesced (blocks, num_blocks);

  gdb_flush (gdb_stdout);

  cuda_info_blocks_destroy (blocks);
}


typedef struct {
  bool           current;
  kernel_t       kernel;
  uint64_t       kernel_id;
  uint64_t       pc;
  char          *filename;
  uint32_t       line;
  CuDim3         start_block_idx;
  CuDim3         start_thread_idx;
  CuDim3         end_block_idx;
  CuDim3         end_thread_idx;
  uint32_t       count;
  const char    *kernel_dim;
  char           start_block_idx_string[32];
  char           start_thread_idx_string[32];
  char           end_block_idx_string[32];
  char           end_thread_idx_string[32];
  uint32_t       device;
  uint32_t       sm;
  uint32_t       wp;
  uint32_t       ln;
} cuda_info_thread_t;

static void
cuda_info_threads_build (const char *filter_string, cuda_info_thread_t **threads, uint32_t *num_threads)
{
  uint32_t i, num_elements;
  uint64_t pc = 0, prev_pc = 0;
  cuda_iterator iter;
  cuda_filters_t default_filter, filter;
  cuda_coords_t  c, expected;
  CuDim3 prev_block_idx = { CUDA_INVALID, CUDA_INVALID, CUDA_INVALID };
  CuDim3 prev_thread_idx = { CUDA_INVALID, CUDA_INVALID, CUDA_INVALID };
  kernel_t kernel;
  cuda_info_thread_t *t;
  struct symtab_and_line sal, prev_sal;
  bool first_entry, break_of_contiguity;
  struct value_print_options opts;

  /* sanity checks */
  gdb_assert (threads);
  gdb_assert (num_threads);
  *num_threads = 0;

  /* make valgrind not complain */
  expected = CUDA_INVALID_COORDS;

  /* get the print options */
  get_user_print_options (&opts);

  /* get the filter */
  default_filter = CUDA_WILDCARD_FILTERS;
  default_filter.coords.kernelId = CUDA_CURRENT;
  filter = cuda_build_filter (filter_string, &default_filter, CMD_FILTER);

  /* get the list of threads */
  iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_THREADS, &filter.coords, CUDA_SELECT_VALID);
  num_elements = cuda_iterator_get_size (iter);
  *threads = (cuda_info_thread_t *) xmalloc (num_elements * sizeof (**threads));

  /* compile the needed info for each block */
  for (cuda_iterator_start (iter), first_entry = true, i = 0, t = *threads, num_elements = 0;
       !cuda_iterator_end (iter);
       cuda_iterator_next (iter), ++i)
    {
      c  = cuda_iterator_get_current (iter);
      kernel = kernels_find_kernel_by_grid_id (c.dev, c.gridId);
      pc = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);

      if (filter.bp_number_p  && !cuda_eval_thread_at_breakpoint (pc, &c, filter.bp_number))
        continue;

      if (pc != prev_pc) /* optimization */
        sal = find_pc_line (pc, 0);

      /* data for the current iteration */
      break_of_contiguity =
        (cuda_coords_compare_logical (&expected, &c) != 0) ||
        (opts.addressprint && pc != prev_pc) ||
        (!opts.addressprint && sal.line != prev_sal.line);

      /* close the current range */
      if (!first_entry && (break_of_contiguity || !cuda_options_coalescing ()))
        {
          t->end_block_idx  = prev_block_idx;
          t->end_thread_idx = prev_thread_idx;
          snprintf (t->end_block_idx_string, sizeof (t->end_block_idx_string),
                    "(%u,%u,%u)", prev_block_idx.x, prev_block_idx.y, prev_block_idx.z);
          snprintf (t->end_thread_idx_string, sizeof (t->end_thread_idx_string),
                    "(%u,%u,%u)", prev_thread_idx.x, prev_thread_idx.y, prev_thread_idx.z);
          ++t;
          ++*num_threads;
        }

      /* start a new range */
      if (first_entry || break_of_contiguity || !cuda_options_coalescing ())
        {
          t->kernel           = kernel;
          t->current          = false;
          t->pc               = pc;
          t->line             = sal.line;
          t->start_block_idx  = c.blockIdx;
          t->start_thread_idx = c.threadIdx;
          t->count            = 0;
          t->kernel_id        = kernel_get_id (kernel);
          t->kernel_dim       = kernel_get_dimensions (kernel);
          t->device           = c.dev;
          t->sm               = c.sm;
          t->wp               = c.wp;
          t->ln               = c.ln;
          t->filename         = get_filename (sal.symtab);

          snprintf (t->start_block_idx_string, sizeof (t->start_block_idx_string),
                    "(%u,%u,%u)", c.blockIdx.x, c.blockIdx.y, c.blockIdx.z);
          snprintf (t->start_thread_idx_string, sizeof (t->start_thread_idx_string),
                    "(%u,%u,%u)", c.threadIdx.x, c.threadIdx.y, c.threadIdx.z);
        }

      /* update the current range */
      t->current |= cuda_coords_is_current (&c);
      ++t->count;
      first_entry = false;

      /* data for the next iteration */
      prev_pc  = pc;
      prev_sal = sal;
      prev_block_idx  = c.blockIdx;
      prev_thread_idx = c.threadIdx;
      expected = CUDA_WILDCARD_COORDS;
      expected.kernelId  = c.kernelId;
      expected.blockIdx  = c.blockIdx;
      expected.threadIdx = c.threadIdx;
      cuda_coords_increment_thread (&expected, kernel_get_grid_dim (kernel),
                                    kernel_get_block_dim (kernel));
      ++num_elements;
    }

  /* close the last range */
  if (num_elements > 0)
    {
      t->end_block_idx  = prev_block_idx;
      t->end_thread_idx = prev_thread_idx;
      snprintf (t->end_block_idx_string, sizeof (t->end_block_idx_string),
                "(%u,%u,%u)", prev_block_idx.x, prev_block_idx.y, prev_block_idx.z);
      snprintf (t->end_thread_idx_string, sizeof (t->end_thread_idx_string),
                "(%u,%u,%u)", prev_thread_idx.x, prev_thread_idx.y, prev_thread_idx.z);
      ++*num_threads;
    }

  cuda_iterator_destroy (iter);
}

static void
cuda_info_threads_destroy (cuda_info_thread_t *threads, uint32_t num_threads)
{
  uint32_t i = 0;
  cuda_info_thread_t *t = NULL;

  for (t = threads, i = 0; i < num_threads; ++i, ++t)
    xfree (t->filename);

  xfree (threads);
}

static void
info_cuda_threads_print_uncoalesced (cuda_info_thread_t *threads, uint32_t num_threads)
{
  cuda_info_thread_t *b;
  uint32_t i, num_columns;
  uint64_t kernel_id;
  struct { size_t current, kernel, block_idx, thread_idx, pc, device, sm, wp, ln, filename, line; } width;

  /* column headers */
  const char *header_current    = " ";
  const char *header_kernel     = "Kernel";
  const char *header_block_idx  = "BlockIdx";
  const char *header_thread_idx = "ThreadIdx";
  const char *header_pc         = "Virtual PC";
  const char *header_device     = "Dev";
  const char *header_sm         = "SM";
  const char *header_warp       = "Wp";
  const char *header_lane       = "Ln";
  const char *header_filename   = "Filename";
  const char *header_line       = "Line";
  struct ui_out *uiout = current_uiout;

  /* sanity checks */
  gdb_assert (!cuda_options_coalescing ());

  /* output message if the list is empty */
  if (num_threads == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _("No CUDA threads.\n"));
      return;
    }

  /* column widths */
  width.current    = strlen (header_current);
  width.kernel     = strlen (header_kernel);
  width.block_idx  = strlen (header_block_idx);
  width.thread_idx = strlen (header_thread_idx);
  width.pc         = strlen (header_pc);
  width.device     = strlen (header_device);
  width.sm         = strlen (header_sm);
  width.wp         = strlen (header_warp);
  width.ln         = strlen (header_lane);
  width.filename   = strlen (header_filename);
  width.line       = strlen (header_line);

  for (b = threads, i = 0; i < num_threads; ++i, ++b)
    {
      width.pc         = std::max (width.pc, (size_t)18);
      width.block_idx  = std::max (width.block_idx, strlen (b->start_block_idx_string));
      width.thread_idx = std::max (width.thread_idx, strlen (b->start_thread_idx_string));
      width.filename   = std::max (width.filename, b->filename ? strlen (b->filename): 0);
      width.line       = std::max (width.line, (size_t)5);
    }

  /* print table header ('kernel' is only present in MI output) */
  num_columns = uiout->is_mi_like_p () ? 11 : 10;
  ui_out_emit_table table_emitter (uiout, num_columns, num_threads, "UncoalescedInfoCudaThreadsTable");
  uiout->table_header (width.current   , ui_right, "current"   , header_current);
  if (uiout->is_mi_like_p ())
    uiout->table_header (width.kernel  , ui_right, "kernel"    , header_kernel);
  uiout->table_header (width.block_idx , ui_right, "blockIdx"  , header_block_idx);
  uiout->table_header (width.thread_idx, ui_right, "threadIdx" , header_thread_idx);
  uiout->table_header (width.pc        , ui_right, "virtual_pc", header_pc);
  uiout->table_header (width.device    , ui_right, "device"    , header_device);
  uiout->table_header (width.sm        , ui_right, "sm"        , header_sm);
  uiout->table_header (width.wp        , ui_right, "warp"      , header_warp);
  uiout->table_header (width.ln        , ui_right, "lane"      , header_lane);
  uiout->table_header (width.filename  , ui_right, "filename"  , header_filename);
  uiout->table_header (width.line      , ui_right, "line"      , header_line);
  uiout->table_body ();

  /* print table rows */
  for (b = threads, i = 0, kernel_id = ~0ULL; i < num_threads; ++i, ++b)
    {
      if (!uiout->is_mi_like_p () && b->kernel_id != kernel_id)
        {
          /* row are grouped per kernel only in CLI output */
          uiout->message ("Kernel %llu\n", (unsigned long long)b->kernel_id),
          kernel_id = b->kernel_id;
        }

      ui_out_emit_tuple tuple_emitter (uiout, "UncoalescedInfoCudaThreadsRow");
      uiout->field_string ("current"   , b->current ? "*" : " ");
      if (uiout->is_mi_like_p ())
        uiout->field_signed  ("kernel"    , b->kernel_id);
      uiout->field_string ("blockIdx"  , b->start_block_idx_string);
      uiout->field_string ("threadIdx" , b->start_thread_idx_string);
      uiout->field_fmt    ("virtual_pc", "0x%016llx", (unsigned long long)b->pc);
      uiout->field_signed    ("device"    , b->device);
      uiout->field_signed    ("sm"        , b->sm);
      uiout->field_signed    ("warp"      , b->wp);
      uiout->field_signed    ("lane"      , b->ln);
      uiout->field_string ("filename"  , b->filename ? b->filename : "n/a");
      uiout->field_signed    ("line"      , b->line);
      uiout->text         ("\n");
    }
}

static void
info_cuda_threads_print_coalesced (cuda_info_thread_t *threads, uint32_t num_threads)
{
  cuda_info_thread_t *b;
  uint32_t i, num_columns;
  uint64_t kernel_id;
  struct { size_t current, kernel, start_block_idx, start_thread_idx, end_block_idx,
      end_thread_idx, count, pc, filename, line; } width;

  /* column headers */
  const char *header_current          = " ";
  const char *header_kernel           = "Kernel";
  const char *header_start_block_idx  = "BlockIdx";
  const char *header_start_thread_idx = "ThreadIdx";
  const char *header_end_block_idx    = "To BlockIdx";
  const char *header_end_thread_idx   = "ThreadIdx";
  const char *header_count            = "Count";
  const char *header_pc               = "Virtual PC";
  const char *header_filename         = "Filename";
  const char *header_line             = "Line";
  struct ui_out *uiout = current_uiout;

  /* sanity checks */
  gdb_assert (cuda_options_coalescing ());

  /* output message if the list is empty */
  if (num_threads == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _("No CUDA threads.\n"));
      return;
    }

  /* column widths */
  width.current          = strlen (header_current);
  width.kernel           = strlen (header_kernel);
  width.start_block_idx  = strlen (header_start_block_idx);
  width.start_thread_idx = strlen (header_start_thread_idx);
  width.end_block_idx    = strlen (header_end_block_idx);
  width.end_thread_idx   = strlen (header_end_thread_idx);
  width.count            = strlen (header_count);
  width.pc               = strlen (header_pc);
  width.filename         = strlen (header_filename);
  width.line             = strlen (header_line);

  for (b = threads, i = 0; i < num_threads; ++i, ++b)
    {
      width.pc               = std::max (width.pc, (size_t)18);
      width.start_block_idx  = std::max (width.start_block_idx, strlen (b->start_block_idx_string));
      width.start_thread_idx = std::max (width.start_thread_idx, strlen (b->start_thread_idx_string));
      width.end_block_idx    = std::max (width.end_block_idx, strlen (b->end_block_idx_string));
      width.end_thread_idx   = std::max (width.end_thread_idx, strlen (b->end_thread_idx_string));
      width.filename         = std::max (width.filename, b->filename ? strlen (b->filename) : 0);
      width.line             = std::max (width.line, (size_t)5);
    }

  /* print table header ('kernel' is only present in MI output) */
  num_columns = uiout->is_mi_like_p () ? 10 : 9;
  ui_out_emit_table table_emitter (uiout, num_columns, num_threads, "CoalescedInfoCudaThreadsTable");
  uiout->table_header (width.current         , ui_right, "current"        , header_current);
  if (uiout->is_mi_like_p ())
    uiout->table_header (width.kernel        , ui_right, "kernel"         , header_kernel);
  uiout->table_header (width.start_block_idx , ui_right, "from_blockIdx"  , header_start_block_idx);
  uiout->table_header (width.start_thread_idx, ui_right, "from_threadIdx" , header_start_thread_idx);
  uiout->table_header (width.end_block_idx   , ui_right, "to_blockIdx"    , header_end_block_idx);
  uiout->table_header (width.end_thread_idx  , ui_right, "to_threadIdx"   , header_end_thread_idx);
  uiout->table_header (width.count           , ui_right, "count"          , header_count);
  uiout->table_header (width.pc              , ui_right, "virtual_pc"     , header_pc);
  uiout->table_header (width.filename        , ui_right, "filename"       , header_filename);
  uiout->table_header (width.line            , ui_right, "line"           , header_line);
  uiout->table_body ();

  /* print table rows */
  for (b = threads, i = 0, kernel_id = ~0ULL; i < num_threads; ++i, ++b)
    {
      if (!uiout->is_mi_like_p () && b->kernel_id != kernel_id)
        {
          /* row are grouped per kernel only in CLI output */
          uiout->message ("Kernel %llu\n", (unsigned long long)b->kernel_id),
          kernel_id = b->kernel_id;
        }

      ui_out_emit_tuple tuple_emitter (uiout, "CoalescedInfoCudaThreadsRow");
      uiout->field_string ("current"       , b->current ? "*" : " ");
      if (uiout->is_mi_like_p ())
        uiout->field_signed  ("kernel"        , b->kernel_id);
      uiout->field_string ("from_blockIdx" , b->start_block_idx_string);
      uiout->field_string ("from_threadIdx", b->start_thread_idx_string);
      uiout->field_string ("to_blockIdx"   , b->end_block_idx_string);
      uiout->field_string ("to_threadIdx"  , b->end_thread_idx_string);
      uiout->field_signed	  ("count"         , b->count);
      uiout->field_fmt    ("virtual_pc"    , "0x%016llx", (unsigned long long)b->pc);
      uiout->field_string ("filename"      , b->filename ? b->filename : "n/a");
      uiout->field_signed	  ("line"          , b->line);
      uiout->text         ("\n");
    }
}

void
info_cuda_threads_command (const char *arg)
{
  cuda_info_thread_t *threads;
  uint32_t num_threads;

  cuda_info_threads_build (arg, &threads, &num_threads);

  if (cuda_options_coalescing ())
    info_cuda_threads_print_coalesced (threads, num_threads);
  else
    info_cuda_threads_print_uncoalesced (threads, num_threads);

  gdb_flush (gdb_stdout);

  cuda_info_threads_destroy (threads, num_threads);
}

typedef struct {
  bool           current;
  kernel_t       kernel;
  uint32_t       level;
  uint32_t       kernel_id;
  uint32_t       device;
  long long      grid_id;
  const char*    status;
  char           grid_dim[32];
  char           block_dim[32];
  char          *invocation;
} cuda_info_launch_trace_t;

static void
cuda_info_launch_trace_build (const char *filter_string,
			      cuda_info_launch_trace_t **kernels,
			      uint32_t *num_kernels)
{
  kernel_t kernel = NULL;
  cuda_info_launch_trace_t *k = NULL;
  cuda_filters_t default_filter, filter;
  CuDim3 grid_dim;
  CuDim3 block_dim;
  uint32_t i = 0;
  const char *args;
  const char *name;
  int len;

  /* sanity checks */
  gdb_assert (kernels);
  gdb_assert (num_kernels);

  /* set the filter */
  default_filter = CUDA_WILDCARD_FILTERS;
  default_filter.coords.kernelId = CUDA_CURRENT;
  filter = cuda_build_filter (filter_string, &default_filter, CMD_FILTER_KERNEL);

  /* find the kernel to start the trace from */
  kernel = kernels_find_kernel_by_kernel_id (filter.coords.kernelId);
  if (!kernel)
    error ("Incorrect kernel specified or the focus is not set on a kernel");

  /* allocate the trace of kernels */
  *num_kernels = kernel_get_depth (kernel) + 1;
  *kernels = (cuda_info_launch_trace_t *) xmalloc (*num_kernels * sizeof (**kernels));

  /* populate the launch trace */
  for (i = 0; i < *num_kernels; ++i, kernel = kernel_get_parent (kernel))
    {
      grid_dim = kernel_get_grid_dim (kernel);
      block_dim = kernel_get_block_dim (kernel);
      name = kernel_get_name (kernel);
      args = kernel_get_args (kernel);

      if (!args)
        args = "n/a";

      k = &(*kernels)[i];
      k->current        = kernel == cuda_current_kernel ();
      k->kernel         = kernel;
      k->level          = i;
      k->kernel_id      = kernel_get_id (kernel);
      k->device         = kernel_get_dev_id (kernel);
      k->grid_id        = kernel_get_grid_id (kernel);
      k->status         = status_string[kernel_get_status (kernel)];

      snprintf (k->grid_dim, sizeof (k->grid_dim),
                "(%u,%u,%u)", grid_dim.x, grid_dim.y, grid_dim.z);
      snprintf (k->block_dim, sizeof (k->block_dim),
                "(%u,%u,%u)", block_dim.x, block_dim.y, block_dim.z);

      len = strlen (name) + 2 + strlen (args) + 1;
      k->invocation = (char *) xmalloc (len);
      snprintf (k->invocation, len, "%s(%s)", name, args);
    }
}

static void
cuda_info_launch_trace_destroy (cuda_info_launch_trace_t *kernels)
{
  xfree (kernels);
}

static void
cuda_info_launch_trace_print (cuda_info_launch_trace_t *kernels, uint32_t num_kernels)
{
  struct { size_t current, level, kernel, device, grid, status, invocation, grid_dim, block_dim; } width;
  uint32_t i = 0;
  cuda_info_launch_trace_t *k = NULL;

  /* column headers */
  const char *header_current     = " ";
  const char *header_level       = "Lvl";
  const char *header_kernel      = "Kernel";
  const char *header_device      = "Dev";
  const char *header_grid        = "Grid";
  const char *header_status      = "Status";
  const char *header_grid_dim    = "GridDim";
  const char *header_block_dim   = "BlockDim";
  const char *header_invocation  = "Invocation";
  struct ui_out *uiout = current_uiout;

  /* column widths */
  width.current     = strlen (header_current);
  width.level       = strlen (header_level);
  width.kernel      = strlen (header_kernel);
  width.device      = strlen (header_device);
  width.grid        = strlen (header_grid);
  width.status      = strlen (header_status);
  width.invocation  = strlen (header_invocation);
  width.grid_dim    = strlen (header_grid_dim);
  width.block_dim   = strlen (header_block_dim);

  for (k = kernels, i = 0; i < num_kernels; ++i, ++k)
    {
      width.status      = std::max (width.status,      strlen (k->status));
      width.invocation  = std::max (width.invocation,  strlen (k->invocation));
      width.grid_dim    = std::max (width.grid_dim,    strlen (k->grid_dim));
      width.block_dim   = std::max (width.block_dim,   strlen (k->block_dim));
    }

  /* print table header */
  ui_out_emit_table table_emitter (uiout, 9, num_kernels, "InfoCudaLaunchTraceTable");
  uiout->table_header (width.current    , ui_right, "current"   , header_current);
  uiout->table_header (width.level      , ui_left,  "level"     , header_level);
  uiout->table_header (width.kernel     , ui_right, "kernel"    , header_kernel);
  uiout->table_header (width.device     , ui_right, "device"    , header_device);
  uiout->table_header (width.grid       , ui_right, "grid"      , header_grid);
  uiout->table_header (width.status     , ui_right, "status"    , header_status);
  uiout->table_header (width.grid_dim   , ui_right, "gridDim"   , header_grid_dim);
  uiout->table_header (width.block_dim  , ui_right, "blockDim"  , header_block_dim);
  uiout->table_header (width.invocation , ui_left , "invocation", header_invocation);
  uiout->table_body ();

  /* print table rows */
  for (k = kernels, i = 0; i < num_kernels; ++i, ++k)
    {
      ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaLaunchTraceRow");
      uiout->field_string    ("current"   , k->current ? "*" : " ");
      uiout->text            ("#");
      uiout->field_signed       ("level"     , k->level);
      uiout->field_signed       ("kernel"    , k->kernel_id);
      uiout->field_signed       ("device"    , k->device);
      uiout->field_long_long ("grid"      , k->grid_id);
      uiout->field_string    ("status"    , k->status);
      uiout->field_string    ("gridDim"   , k->grid_dim);
      uiout->field_string    ("blockDim"  , k->block_dim);
      uiout->field_string    ("invocation", k->invocation);
      uiout->text            ("\n");
    }
}

void
info_cuda_launch_trace_command (const char *arg)
{
  cuda_info_launch_trace_t *kernels = NULL;
  uint32_t num_kernels = 0;

  cuda_info_launch_trace_build (arg, &kernels, &num_kernels);

  if (!current_uiout->is_mi_like_p () && num_kernels == ~0U)
    return;

  cuda_info_launch_trace_print (kernels, num_kernels);

  cuda_info_launch_trace_destroy (kernels);
}

typedef struct {
  bool           current;
  kernel_t       kernel;
  uint32_t       kernel_id;
  uint32_t       device;
  uint64_t       grid_id;
  char           grid_dim[32];
  char           block_dim[32];
  char          *invocation;
} cuda_info_launch_children_t;

static void
cuda_info_launch_children_build (const char *filter_string,
                                 cuda_info_launch_children_t **kernels,
                                 uint32_t *num_kernels)
{
  kernel_t kernel = NULL;
  cuda_info_launch_children_t *k = NULL;
  cuda_filters_t default_filter, filter;
  CuDim3 grid_dim;
  CuDim3 block_dim;
  uint32_t i = 0;
  const char *args;
  const char *name;
  int len;

  /* sanity checks */
  gdb_assert (kernels);
  gdb_assert (num_kernels);

  /* set the filter */
  default_filter = CUDA_WILDCARD_FILTERS;
  default_filter.coords.kernelId = CUDA_CURRENT;
  filter = cuda_build_filter (filter_string, &default_filter, CMD_FILTER_KERNEL);

  /* find the kernel to start the trace from */
  kernel = kernels_find_kernel_by_kernel_id (filter.coords.kernelId);
  if (!kernel)
    error ("Incorrect kernel specified or the focus is not set on a kernel");

  /* build the list of children */
  *num_kernels = kernel_get_num_children (kernel);
  *kernels = (cuda_info_launch_children_t *) xmalloc (*num_kernels * sizeof (**kernels));

  for (i = 0, kernel = kernel_get_children (kernel); kernel; ++i, kernel = kernel_get_sibling (kernel))
    {
      grid_dim = kernel_get_grid_dim (kernel);
      block_dim = kernel_get_block_dim (kernel);
      name = kernel_get_name (kernel);
      args = kernel_get_args (kernel);

      if (!args)
        args = "n/a";

      k = &(*kernels)[i];
      k->current        = kernel == cuda_current_kernel ();
      k->kernel         = kernel;
      k->kernel_id      = kernel_get_id (kernel);
      k->device         = kernel_get_dev_id (kernel);
      k->grid_id        = kernel_get_grid_id (kernel);

      snprintf (k->grid_dim, sizeof (k->grid_dim),
                "(%u,%u,%u)", grid_dim.x, grid_dim.y, grid_dim.z);
      snprintf (k->block_dim, sizeof (k->block_dim),
                "(%u,%u,%u)", block_dim.x, block_dim.y, block_dim.z);

      len = strlen (name) + 2 + strlen (args) + 1;
      k->invocation = (char *) xmalloc (len);
      snprintf (k->invocation, len, "%s(%s)", name, args);
    }
}

static void
cuda_info_launch_children_destroy (cuda_info_launch_children_t *kernels)
{
  xfree (kernels);
}

static void
cuda_info_launch_children_print (cuda_info_launch_children_t *kernels, uint32_t num_kernels)
{
  struct { size_t current, kernel, device, grid, invocation, grid_dim, block_dim; } width;
  uint32_t i = 0;
  cuda_info_launch_children_t *k = NULL;

  /* column headers */
  const char *header_current     = " ";
  const char *header_kernel      = "Kernel";
  const char *header_device      = "Dev";
  const char *header_grid        = "Grid";
  const char *header_invocation  = "Invocation";
  const char *header_grid_dim    = "GridDim";
  const char *header_block_dim   = "BlockDim";
  struct ui_out *uiout = current_uiout;

  /* column widths */
  width.current     = strlen (header_current);
  width.kernel      = strlen (header_kernel);
  width.device      = strlen (header_device);
  width.grid        = strlen (header_grid);
  width.invocation  = strlen (header_invocation);
  width.grid_dim    = strlen (header_grid_dim);
  width.block_dim   = strlen (header_block_dim);

  for (k = kernels, i = 0; i < num_kernels; ++i, ++k)
    {
      width.invocation  = std::max (width.invocation,  strlen (k->invocation));
      width.grid_dim    = std::max (width.grid_dim,    strlen (k->grid_dim));
      width.block_dim   = std::max (width.block_dim,   strlen (k->block_dim));
    }

  /* print table header */
  ui_out_emit_table table_emitter (uiout, 7, num_kernels, "InfoCudaLaunchChildrenTable");
  uiout->table_header (width.current    , ui_right, "current"   , header_current);
  uiout->table_header (width.kernel     , ui_right, "kernel"    , header_kernel);
  uiout->table_header (width.device     , ui_right, "device"    , header_device);
  uiout->table_header (width.grid       , ui_right, "grid"      , header_grid);
  uiout->table_header (width.grid_dim   , ui_right, "gridDim"   , header_grid_dim);
  uiout->table_header (width.block_dim  , ui_right, "blockDim"  , header_block_dim);
  uiout->table_header (width.invocation , ui_left , "invocation", header_invocation);
  uiout->table_body ();

  /* print table rows */
  for (k = kernels, i = 0; i < num_kernels; ++i, ++k)
    {
      ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaLaunchChildrenRow");
      uiout->field_string    ("current"   , k->current ? "*" : " ");
      uiout->field_signed       ("kernel"    , k->kernel_id);
      uiout->field_signed       ("device"    , k->device);
      uiout->field_long_long ("grid"      , k->grid_id);
      uiout->field_string    ("gridDim"   , k->grid_dim);
      uiout->field_string    ("blockDim"  , k->block_dim);
      uiout->field_string    ("invocation", k->invocation);
      uiout->text            ("\n");
    }
}

void
info_cuda_launch_children_command (const char *arg)
{
  cuda_info_launch_children_t *kernels = NULL;
  uint32_t num_kernels = 0;

  cuda_info_launch_children_build (arg, &kernels, &num_kernels);

  if (!current_uiout->is_mi_like_p () && num_kernels == ~0U)
    return;

  cuda_info_launch_children_print (kernels, num_kernels);

  cuda_info_launch_children_destroy (kernels);
}

typedef struct {
  bool           current;
  uint64_t       context_id;
  uint32_t       device;
  char           state[32];
} cuda_info_context_t;

static void
cuda_info_contexts_build (const char *filter_string, cuda_info_context_t **contexts, uint32_t *num_contexts)
{
  list_elt_t elt;
  cuda_info_context_t *c;
  cuda_filters_t default_filter, filter, invalid_filter;
  uint32_t num_dev;
  uint32_t num_elements = 0;
  uint32_t i;

  /* sanity checks */
  gdb_assert (contexts);
  gdb_assert (num_contexts);

  /* get the filter */
  default_filter = CUDA_WILDCARD_FILTERS;
  filter = cuda_build_filter (filter_string, &default_filter, CMD_FILTER);

  /* the only allowed filter is 'device' */
  invalid_filter = CUDA_INVALID_FILTERS;
  invalid_filter.coords.dev = CUDA_WILDCARD;
  if (!cuda_coords_equal (&filter.coords, &invalid_filter.coords))
    error ("Invalid filter. Only 'device' is supported'.");

  num_dev = cuda_system_get_num_devices ();
  for (i = 0; i < num_dev; i++)
   if (filter.coords.dev == CUDA_WILDCARD || filter.coords.dev == i)
     num_elements += contexts_get_list_size (device_get_contexts (i));

  *contexts = (cuda_info_context_t *) xmalloc (num_elements * sizeof (**contexts));
  c = *contexts;

  /* compile the needed info for each context */
  for (i = 0; i < num_dev; i++)
    {
      contexts_t ctxs;
      if (filter.coords.dev != CUDA_WILDCARD && filter.coords.dev != i)
        continue;

      /* find all active contexts of each device */
      ctxs = device_get_contexts (i);
      for (elt = ctxs->list; elt; elt = elt->next)
        {
           c->current    = elt->context == get_current_context () ? true : false;
           c->context_id = elt->context->context_id;
           c->device     = elt->context->dev_id;
           snprintf (c->state, sizeof (c->state), "%s",
                     device_is_active_context (i, elt->context) ? "active" : "inactive");
           ++*num_contexts;
           ++c;
        }
    }
}

static void
cuda_info_contexts_destroy (cuda_info_context_t *contexts)
{
  xfree (contexts);
}

void
info_cuda_contexts_command (const char *arg)
{
  cuda_info_context_t *contexts, *c;
  uint32_t i;
  uint32_t num_contexts = 0;
  struct { size_t current, context, device, state; } width;

  /* column header */
  const char *header_current         = " ";
  const char *header_context         = "Context";
  const char *header_device          = "Dev";
  const char *header_state           = "State";
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_info_contexts_build (arg, &contexts, &num_contexts);

  /* output message if the list is empty */
  if (num_contexts == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _("No CUDA contexts.\n"));
      return;
    }

  /* column widths */
  width.current   = strlen (header_current);
  width.context   = strlen (header_context);
  width.device    = strlen (header_device);
  width.state     = strlen (header_state);

  width.context = std::max (width.context, (size_t)10);
  width.state   = std::max (width.state, strlen ("inactive")); 

  /* print table header */
  ui_out_emit_table table_emitter (uiout, 4, num_contexts, "InfoCudaContextsTable");
  uiout->table_header (width.current  , ui_right, "current"  , header_current);
  uiout->table_header (width.context  , ui_right, "context"  , header_context);
  uiout->table_header (width.device   , ui_right, "device"   , header_device);
  uiout->table_header (width.state    , ui_right, "state"    , header_state);
  uiout->table_body ();

  /* print table rows */
  for (c = contexts, i = 0; i < num_contexts; ++i, ++c)
    {
      ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaContextsRow");
      uiout->field_string ("current" , c->current ? "*" : " ");
      uiout->field_fmt    ("context" , "0x%016llx", (unsigned long long)c->context_id);
      uiout->field_signed    ("device"  , c->device);
      uiout->field_string ("state"   , c->state);
      uiout->text         ("\n");
    }

  cuda_info_contexts_destroy (contexts);
}

static struct symbol *
msymbol_to_symbol (struct bound_minimal_symbol bmsym)
{
  struct obj_section *objsection  = MSYMBOL_OBJ_SECTION (bmsym.objfile, bmsym.minsym);
  struct objfile     *objfile = objsection ? objsection->objfile : NULL;
  const struct blockvector *bv;
  struct block       *block;
  struct symbol      *sym;

  if (!objfile)
    return NULL;

  for (compunit_symtab *cu : objfile->compunits ())
    {
      for (symtab *s : compunit_filetabs (cu))
      {
        bv = SYMTAB_BLOCKVECTOR (s);
        block = BLOCKVECTOR_BLOCK (bv, STATIC_BLOCK);
        sym = block_lookup_symbol (block, bmsym.minsym->print_name (),
            symbol_name_match_type::SEARCH_NAME, VAR_DOMAIN);
        if (sym)
          return fixup_symbol_section (sym, objfile);
      }
    }

  return NULL;
}

static void
print_managed_msymbol (struct ui_file *stb, struct bound_minimal_symbol bmsym)
{
  struct value_print_options opts;
  struct symbol *sym = msymbol_to_symbol (bmsym);
  struct value  *val = sym ? read_var_value (sym, NULL, NULL) : NULL;

  if (!val)
    {
      fprintf_filtered (stb, "%s\n", bmsym.minsym->print_name ());
      return;
    }
   if (!cuda_focus_is_device())
      val = value_ind (val);

   fprintf_filtered (stb, "%s = ", bmsym.minsym->print_name ());

  try
    {
      get_user_print_options (&opts);
      common_val_print (val, stb, 0, &opts, current_language);
    }
  catch (const gdb_exception_error &e)
    {
      fprintf_filtered (stb, "<value optimized out>\n");
      return;
    }

  fprintf_filtered (stb, "\n");
}

static void
info_cuda_managed_command (const char *arg)
{
  const struct bfd_arch_info *current_bfd_arch = gdbarch_bfd_arch_info (get_current_arch());

  if (cuda_focus_is_device())
    fprintf_filtered (gdb_stdout, "Static managed variables on device %d are:\n",
                      cuda_current_device());
  else
    fprintf_filtered (gdb_stdout, "Static managed variables on host are:\n");

  for (objfile *obj : current_program_space->objfiles ())
    {
      /* Skip objects which architecture different than current */
      if (gdbarch_bfd_arch_info (obj->arch ()) != current_bfd_arch)
        continue;

      for (minimal_symbol *msym : obj->msymbols ())
        {
	  struct bound_minimal_symbol minsym;

	  minsym = lookup_minimal_symbol (msym->linkage_name (), NULL, obj);

          if (!cuda_managed_msymbol_p (minsym))
            continue;
          print_managed_msymbol (gdb_stdout, minsym);
        }
    }
}

/*
  Very similar to exception info printing.
*/
static void
info_cuda_line_command (const char *args)
{
  if (args == NULL)
    error_no_arg (_("pc value"));

  CORE_ADDR pc = parse_and_eval_address (args);

  struct obj_section *section = find_pc_overlay (pc);

  struct cuda_debug_inline_info *inline_info = NULL;
  struct symtab_and_line sal = find_pc_sect_line (pc, section, 0, &inline_info);

  /* Possibly called from CLI mode, make sure we get the right uiout,
     not the current one. */
  struct ui_out *uiout = top_level_interpreter ()->interp_ui_out ();

  uiout->text ("PC ");
  uiout->field_fmt ("pc", "0x%lx", pc);

  const char *filename = sal.symtab ? symtab_to_filename_for_display (sal.symtab) : NULL;

  if (filename)
    {
      uiout->text         (" is at ");
      uiout->field_string ("filename", filename);
      uiout->text         (":");
      uiout->field_signed    ("line", sal.line);

      if (inline_info)
	{
	  char *demangled = language_demangle (language_def (current_language->la_language),
					       inline_info->function,
					       DMGL_ANSI);
	  uiout->text         (" inlined from ");
	  uiout->field_string ("inline_filename", lbasename (inline_info->filename));
	  uiout->text         (":");
	  uiout->field_signed    ("inline_line", inline_info->line);
	  uiout->text         (" by ");

	  const char *inline_function = demangled ? demangled : inline_info->function;
	  uiout->field_string ("inline_function", inline_function);

	  xfree (demangled);
	}
    }
  else
    {
      uiout->text (" has no linenumber information");
    }

  uiout->text ("\n");
}

using info_cuda_cleanup = FORWARD_SCOPE_EXIT (cuda_focus_restore);

void
run_info_cuda_command (void (*command)(const char *), const char *arg)
{
  cuda_focus_t focus;
  cuda_focus_init (&focus);

  /* Save the current focus and ELF image */
  cuda_focus_save (&focus);
  /* Create cleanup object to restore the original focus and ELF images */
  info_cuda_cleanup cleanups (&focus);

  /* Execute the proper info cuda command */
  command (arg);
}

static struct {
  const char *name;
  void (*func) (const char *);
  const char *help;
} cuda_info_subcommands[] =
{
  { "devices",          info_cuda_devices_command,
             "information about all the devices" },
  { "sms",              info_cuda_sms_command,
             "information about all the SMs in the current device" },
  { "warps",            info_cuda_warps_command,
             "information about all the warps in the current SM" },
  { "lanes",            info_cuda_lanes_command,
             "information about all the lanes in the current warp" },
  { "kernels",          info_cuda_kernels_command,
             "information about all the active kernels" },
  { "contexts",         info_cuda_contexts_command,
             "information about all the contexts" },
  { "blocks",           info_cuda_blocks_command,
             "information about all the active blocks in the current kernel" },
  { "threads",          info_cuda_threads_command,
             "information about all the active threads in the current kernel" },
  { "launch trace",     info_cuda_launch_trace_command,
             "information about the parent kernels of the kernel in focus" },
  { "launch children",  info_cuda_launch_children_command,
             "information about the kernels launched by the kernels in focus" },
  { "managed", info_cuda_managed_command,
             "information about global managed variables"},
  { "line", info_cuda_line_command,
             "information about the filename and linenumber for a given $pc" },
  { NULL, NULL, NULL},
};

static int
cuda_info_subcommands_max_name_length (void)
{
  int cnt;
  size_t rc;

  for (cnt=0,rc=0; cuda_info_subcommands[cnt].name; cnt++)
    rc = std::max (rc, strlen(cuda_info_subcommands[cnt].name));
  return (int)rc;
}

static void
info_cuda_command (const char *arg, int from_tty)
{
  int cnt;
  const char *argument;
  void (*command)(const char *) = NULL;

  if (!arg)
    error (_("Missing option."));

  /* Sanity check and save which command (with correct argument) to invoke. */
  for (cnt=0; cuda_info_subcommands[cnt].name; cnt++)
    if (strstr(arg, cuda_info_subcommands[cnt].name) == arg)
       {
         command = cuda_info_subcommands[cnt].func;
         argument = arg + strlen(cuda_info_subcommands[cnt].name);
         break;
       }

  if (!command)
    error (_("Unrecognized option: '%s'."), arg);

  run_info_cuda_command (command, argument);
}

struct cmd_list_element *cudalist;

void
cuda_command_switch (const char *switch_string)
{
  bool ignore_solution;
  uint32_t i;
  cuda_coords_kind_t ck;
  cuda_coords_t current, requested, processed, candidates[CK_MAX], *solution;
  request_t *request;
  cuda_parser_result_t *command;
  cuda_coords_special_value_t default_value;

  /* Read the current coordinates. */
  current = CUDA_INVALID_COORDS;
  cuda_coords_get_current (&current);

  /* Read the user request, including the uninitialized values. */
  requested = CUDA_INVALID_COORDS;
  default_value = cuda_focus_is_device () ? CUDA_CURRENT : CUDA_WILDCARD;
  cuda_parser (switch_string, CMD_SWITCH, &command, default_value);
  cuda_parser_result_to_coords (command, &requested);

  /* Replace the uninitialized user values with the current coordinates if
     any. Otherwise use wildcards. */
  processed = current.valid ? current : CUDA_WILDCARD_COORDS;
  cuda_parser_result_to_coords (command, &processed);
  cuda_coords_evaluate_current (&processed, true);
  cuda_coords_check_fully_defined (&processed, false, false, true);

  /* Physical or logical coordinates. Physical coordinates have priority. */
  ck = CK_CLOSEST_LOGICAL;
  for (i = 0, request = command->requests; i < command->num_requests; ++i, ++request)
    if (request->type == FILTER_TYPE_DEVICE || request->type == FILTER_TYPE_SM ||
        request->type == FILTER_TYPE_WARP   || request->type == FILTER_TYPE_LANE)
      ck = CK_CLOSEST_PHYSICAL;

  /* Find the closest match */
  cuda_coords_find_valid (processed, candidates);
  solution = &candidates[ck];

  /* Weed out the solution if the user request cannot be honored */
  ignore_solution = !cuda_coords_equal (&requested, solution);

  /* Do the actual switch if possible */
  if (!solution->valid)
    error (_("Invalid coordinates. CUDA focus unchanged."));
  else if (ignore_solution)
    error (_("Request cannot be satisfied. CUDA focus unchanged."));
  else if (current.valid && cuda_coords_equal (solution, &current))
    current_uiout->field_string (NULL, _("CUDA focus unchanged.\n"));
  else
    {
      cuda_coords_set_current (solution);
      cuda_update_convenience_variables ();
      cuda_update_cudart_symbols ();
      switch_to_cuda_thread (NULL);
      cuda_print_message_focus (true);
      print_stack_frame (get_selected_frame (NULL), 0, SRC_LINE, 1);
      do_displays ();
    }
}

void
cuda_command_query (const char *query_string)
{
  cuda_coords_t wished;
  const int string_size = 1000;
  char string[string_size];
  cuda_parser_result_t *command;

  /* Bail out if focus not set on a CUDA device */
  if (!cuda_focus_is_device ())
    {
      current_uiout->field_string (NULL, _("Focus is not set on any active CUDA kernel.\n"));
      return;
    }

  /* Build the coordinates based on the user request */
  wished = CUDA_INVALID_COORDS;
  cuda_parser (query_string, CMD_QUERY, &command, CUDA_CURRENT);
  cuda_parser_result_to_coords (command, &wished);
  cuda_coords_evaluate_current (&wished, false);
  cuda_coords_check_fully_defined (&wished, true, false, false);

  /* Print the current focus */
  cuda_coords_to_fancy_string (&wished, (char*)string, string_size);
  current_uiout->field_fmt (NULL, "%s\n", string);
  gdb_flush (gdb_stdout);
}

static void
cuda_command_all (const char *first_word, const char *args)
{
  char *input;
  uint32_t len1, len2;
  cuda_parser_result_t *result;

  /* Reassemble the whole command */
  len1 = first_word ? strlen (first_word) : 0;
  len2 = args ? strlen (args) : 0;
  input = (char *) xmalloc (len1 + 1 + len2 + 1);
  strncpy (input, first_word, len1);
  strncpy (input + len1, " ", 1);
  strncpy (input + len1 + 1, args, len2);
  input[len1 + 1 + len2] = 0;

 /* Dispatch to the right handler based on the command type */
  cuda_parser (input, (command_t)(CMD_QUERY | CMD_SWITCH), &result, CUDA_WILDCARD);
  switch (result->command)
    {
    case CMD_QUERY  : cuda_command_query (input); break;
    case CMD_SWITCH : cuda_command_switch (input); break;
    default: error (_("Unrecognized argument(s)."));
    }

  /* Clean up */
  xfree (input);
}

static void
cuda_device_command (const char *arg, int from_tty)
{
  cuda_command_all ("device", arg);
}

static void
cuda_sm_command (const char *arg, int from_tty)
{
  cuda_command_all ("sm", arg);
}

static void
cuda_warp_command (const char *arg, int from_tty)
{
  cuda_command_all ("warp", arg);
}

static void
cuda_lane_command (const char *arg, int from_tty)
{
  cuda_command_all ("lane", arg);
}

static void
cuda_kernel_command (const char *arg, int from_tty)
{
  cuda_command_all ("kernel", arg);
}

static void
cuda_grid_command (const char *arg, int from_tty)
{
  cuda_command_all ("grid", arg);
}

static void
cuda_block_command (const char *arg, int from_tty)
{
  cuda_command_all ("block", arg);
}

static void
cuda_thread_command (const char *arg, int from_tty)
{
  cuda_command_all ("thread", arg);
}

static void
cuda_command (const char *arg, int from_tty)
{
  if (!arg)
    error (_("Missing argument(s)."));
}

static char cuda_info_cmd_help_str[1024];

/* Prepare help for info cuda command */
static void
cuda_build_info_cuda_help_message (void)
{
  char *ptr = cuda_info_cmd_help_str;
  int size = sizeof(cuda_info_cmd_help_str);
  int rc, cnt;

  rc = snprintf (ptr, size,
    _("Print informations about the current CUDA activities. Available options:\n"));
  ptr += rc; size -= rc;
  for (cnt=0; cuda_info_subcommands[cnt].name; cnt++)
    {
       rc = snprintf (ptr, size, " %*s : %s\n",
          cuda_info_subcommands_max_name_length(),
          cuda_info_subcommands[cnt].name,
          _(cuda_info_subcommands[cnt].help) );
        if (rc <= 0) break;
        ptr += rc;
        size -= rc;
    }
}

static void
cuda_info_command_completer (struct cmd_list_element *ignore,
			     completion_tracker &tracker,
                             const char *text, const char *word)
{
  int cnt;
  long offset;
  const char *name;

  offset = (long)word-(long)text;

  for (cnt=0; cuda_info_subcommands[cnt].name; cnt++)
    {
       name = cuda_info_subcommands[cnt].name;
       if (offset >= strlen(name)) continue;
       if (strstr(name, text) != name) continue;
       gdb::unique_xmalloc_ptr<char> p_copy (xstrdup (name+offset));
       tracker.add_completion (std::move (p_copy));
    }
}

void
cuda_commands_initialize (void)
{
  struct cmd_list_element *cmd;

  add_prefix_cmd ("cuda", class_cuda, cuda_command,
                  _("Print or select the CUDA focus."),
                  &cudalist, "cuda ", 0, &cmdlist);

  add_cmd ("device", no_class, cuda_device_command,
           _("Print or select the current CUDA device."), &cudalist);

  add_cmd ("sm", no_class, cuda_sm_command,
           _("Print or select the current CUDA SM."), &cudalist);

  add_cmd ("warp", no_class, cuda_warp_command,
           _("Print or select the current CUDA warp."), &cudalist);

  add_cmd ("lane", no_class, cuda_lane_command,
           _("Print or select the current CUDA lane."), &cudalist);

  add_cmd ("kernel", no_class, cuda_kernel_command,
           _("Print or select the current CUDA kernel."), &cudalist);

  add_cmd ("grid", no_class, cuda_grid_command,
           _("Print or select the current CUDA grid."), &cudalist);

  add_cmd ("block", no_class, cuda_block_command,
           _("Print or select the current CUDA block."), &cudalist);

  add_cmd ("thread", no_class, cuda_thread_command,
           _("Print or select the current CUDA thread."), &cudalist);

  cuda_build_info_cuda_help_message ();
  cmd = add_info ("cuda", info_cuda_command, cuda_info_cmd_help_str);
  set_cmd_completer (cmd, cuda_info_command_completer);
}
