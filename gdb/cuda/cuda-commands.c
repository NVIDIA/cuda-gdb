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

#include "arch-utils.h"
#include "block.h"
#include "command.h"
#include "cuda-commands.h"
#include "cuda-context.h"
#include "cuda-convvars.h"
#include "cuda-coords.h"
#include "cuda-coord-set.h"
#include "cuda-exceptions.h"
#include "cuda-kernel.h"
#include "cuda-options.h"
#include "cuda-parser.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"
#include "exceptions.h"
#include "filenames.h"
#include "gdbcmd.h"
#include "gdbsupport/forward-scope-exit.h"
#include "gdbsupport/pathstuff.h"
#include "language.h"
#include "objfiles.h"
#include "source.h"
#include "symtab.h"
#include "ui-file.h"
#include "ui-out.h"
#include "valprint.h"

#include "demangle.h"
#include "interps.h"

#include <bitset>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>

// std::underlying_type_t was added in C++14
template< class T >
using underlying_type_t = typename std::underlying_type<T>::type;

class cuda_filters
{
private:
  /* cuda coordinates filter */
  cuda_coords m_coords;

public:
  /* Constructor/destructors */
  cuda_filters () : m_coords{} {}
  cuda_filters (const cuda_coords &c)
      : m_coords{ c }
  {
  }
  cuda_filters (const cuda_coords &initial, cuda_parser_result_t *result,
                bool rectify_wild = false)
      : m_coords{ initial }
  {
    gdb_assert (result);
    /* Iterate over each request */
    request_t *request = result->requests;
    for (auto i = 0; i < result->num_requests; ++i, ++request)
      processRequest (request);
    /* For focus commands, we want to limit the possibilities based on the
     * current focus. This means we should change wildcard coords to be the
     * current focus coords if we are searching for a lower ordered coordinate.
     */
    if (rectify_wild && !m_coords.isFullyDefined ()
        && cuda_current_focus::isDevice ())
      {
        cuda_coords cur = cuda_current_focus::get ();
        const auto &c_p = cur.physical ();
        const auto &c_l = cur.logical ();
        const auto &p = m_coords.physical ();
        const auto &l = m_coords.logical ();

        auto dev = p.dev ();
        auto sm = p.sm ();
        auto warp = p.wp ();
        auto lane = p.ln ();

        auto kernel = l.kernelId ();
        auto grid = l.gridId ();
        auto cluster = l.clusterIdx ();
        auto block = l.blockIdx ();
        auto thread = l.threadIdx ();

        /* Check dev and kernel */
        if (cuda_coord_is_special (p.dev ())
            && cuda_coord_is_special (l.kernelId ()))
          {
            /* Fix to current focus */
            dev = c_p.dev ();
            kernel = c_l.kernelId ();
            /* Check grid */
            if (cuda_coord_is_special (l.gridId ()))
              {
                /* Fix to current focus */
                grid = c_l.gridId ();
                /* Check sm and block */
                if (cuda_coord_is_special (p.sm ())
                    && cuda_coord_is_special (l.blockIdx ()))
                  {
                    /* Fix to current focus */
                    sm = c_p.sm ();
                    block = c_l.blockIdx ();
                    /* No need to check warp, lane, or thread. They are the
                     * lowest order coords. */
                  }
              }
          }
        /* Reset the coords */
        m_coords = cuda_coords{ dev,  sm,      warp,  lane,  kernel,
                                grid, cluster, block, thread };
      }
  }
  ~cuda_filters () = default;

  /* getters */
  const cuda_coords &
  coords () const
  {
    return m_coords;
  }

private:
  void
  processRequest (request_t *request)
  {
    gdb_assert (request);
    const auto &p = m_coords.physical ();
    const auto &l = m_coords.logical ();
    switch (request->type)
      {
      case FILTER_TYPE_DEVICE:
        m_coords = cuda_coords{
          request->value.scalar, p.sm (),     p.wp (),         p.ln (),
          l.kernelId (),         l.gridId (), l.clusterIdx (), l.blockIdx (),
          l.threadIdx ()
        };
        break;
      case FILTER_TYPE_SM:
        m_coords = cuda_coords{ p.dev (),        request->value.scalar,
                                p.wp (),         p.ln (),
                                l.kernelId (),   l.gridId (),
                                l.clusterIdx (), l.blockIdx (),
                                l.threadIdx () };
        break;
      case FILTER_TYPE_WARP:
        m_coords = cuda_coords{
          p.dev (),        p.sm (),       request->value.scalar,
          p.ln (),         l.kernelId (), l.gridId (),
          l.clusterIdx (), l.blockIdx (), l.threadIdx ()
        };
        break;
      case FILTER_TYPE_LANE:
        m_coords = cuda_coords{ p.dev (),        p.sm (),
                                p.wp (),         request->value.scalar,
                                l.kernelId (),   l.gridId (),
                                l.clusterIdx (), l.blockIdx (),
                                l.threadIdx () };
        break;
      case FILTER_TYPE_KERNEL:
        m_coords = cuda_coords{ p.dev (),
                                p.sm (),
                                p.wp (),
                                p.ln (),
                                request->value.scalar,
                                l.gridId (),
                                l.clusterIdx (),
                                l.blockIdx (),
                                l.threadIdx () };
        break;
      case FILTER_TYPE_GRID:
        m_coords = cuda_coords{ p.dev (),        p.sm (),
                                p.wp (),         p.ln (),
                                l.kernelId (),   request->value.scalar,
                                l.clusterIdx (), l.blockIdx (),
                                l.threadIdx () };
        break;
      case FILTER_TYPE_BLOCK:
        m_coords = cuda_coords{ p.dev (),        p.sm (),
                                p.wp (),         p.ln (),
                                l.kernelId (),   l.gridId (),
                                l.clusterIdx (), request->value.cudim3,
                                l.threadIdx () };
        break;
      case FILTER_TYPE_THREAD:
        m_coords = cuda_coords{
          p.dev (),        p.sm (),       p.wp (),
          p.ln (),         l.kernelId (), l.gridId (),
          l.clusterIdx (), l.blockIdx (), request->value.cudim3
        };
        break;
      default:
        error (_ ("Unexpected request type."));
      }
  }
};

template<typename T>
std::string to_hex (T&& val)
{
  std::stringstream ss;
  ss << "0x" << std::setfill ('0') << std::setw (sizeof(T)*2) << std::hex << val;
  return ss.str ();
}

static std::string
dim3_to_string (const CuDim3 &dim)
{
  std::stringstream ss;
  ss << "(" << dim.x << "," << dim.y << "," << dim.z << ")";
  return ss.str ();
}

static std::string
invocation_to_string (const char *name, const char *args)
{
  std::stringstream ss;
  ss << (name ? name : "??") << "(" << (args ? args : "") << ")";
  return ss.str ();
}

// Template class for holding info objects
template <typename T> class cuda_info
{
private:
  bool m_populated;

protected:
  std::vector<T> m_underlying;
  cuda_filters m_filter;

  /* Used to handle a coord that triggers an exception.
     By default: ignore the exception */
  virtual void
  handle_underlying_exception (const cuda_coords &coord)
  {
  }

  /* Used to do pre-processing on a coord and possibly ignore it prior to
     emplace. By default: Don't ignore. */
  virtual bool
  ignore_coord (const cuda_coords &coord)
  {
    return false;
  }

  /* Used to hanlde post processing right after we emplaced a coord.
     By default: Do nothing. */
  virtual void
  handle_emplaced_coord (const cuda_coords &coord)
  {
  }

  /* Used to handle post processing on populate after the last underlying type
     has been added.
     By default: Do nothing. */
  virtual void
  handle_post_populate ()
  {
  }

private:
  /* Populate the underlying type - we have to do this here to allow derived
     classes to override default behavior. */
  void
  populate ()
  {
    if (m_populated)
      return;

    /* Add each underlying type to the list */
    cuda_coord_set<T::iterator_type, T::iterator_mask, T::compare_type> coords{
      m_filter.coords ()
    };
    for (const auto &coord : coords)
      {
        /* Let the derived class do any necessary pre-processing */
        if (!ignore_coord (coord))
          {
            /* The constructor may throw if the object is invalid */
            try
              {
                m_underlying.emplace_back (coord);
                /* Let derived class do any post processing on coord. */
                handle_emplaced_coord (coord);
              }
            catch (const gdb_exception_error &except)
              {
                /* Let derived classes determine the behavior.
                 * By default, we ignore it. */
                handle_underlying_exception (coord);
              }
          }
      }
    /* Let the derived class do any necessary post-processing */
    handle_post_populate ();
    m_populated = true;
  }

public:
  explicit cuda_info (const char *filter_string) : m_populated{ false }
  {
    /* Build the filter - start by checking for user provided filter string */
    if (filter_string && *filter_string != 0)
      {
        /* Parse the user provided filter string */
        cuda_parser_result_t *result = nullptr;
        cuda_parser (filter_string, T::parser_command, &result, CUDA_WILDCARD);
        gdb_assert (result != nullptr);
        if (result->command != T::parser_command)
          error (_ ("Incorrect filter: '%s'."), filter_string);
        /* Build the filter object from the result of the parser */
        m_filter = cuda_filters{ cuda_coords::wild (), result };
      }
    /* Set to the provided default filter for this query */
    else
      {
        m_filter = T::default_filter ();
      }

    /* Check for invalid filters */
    T::check_invalid_filter (m_filter);

    /* We defer populating the coord set until populate is called.
     * This is necessary to allow derived classes to override the
     * default behavior. We cannot call virtual functions in a
     * destructor. TODO: Move to a functor impl instead.
     */
  }
  /* Init other destructors/constructors */
  cuda_info () = delete;
  cuda_info (const cuda_info &) = default;
  cuda_info (cuda_info &&) = default;
  cuda_info &operator= (const cuda_info &) = default;
  cuda_info &operator= (cuda_info &&) = default;
  virtual ~cuda_info () = default;

  /* Methods */

  /* Returns a CuDim3 representing a coord idx advanced to its nearest neighbor. */
  CuDim3
  incrementDim (const CuDim3 &idx, const CuDim3 &dim)
  {
    CuDim3 ret = idx;

    /* Increment the fastest varying dimension idx.x */
    if ((idx.x + 1) < dim.x)
      {
        ++ret.x;
      }
    /* Spill over to idx.y */
    else if ((idx.y + 1) < dim.y)
      {
        ret.x = 0;
        ++ret.y;
      }
    /* Spill over to idx.z */
    else if ((idx.z + 1) < dim.z)
      {
        ret.x = 0;
        ret.y = 0;
        ++ret.z;
      }
    /* Overflow! Return invalid coords. */
    else
      {
        ret.x = CUDA_INVALID;
        ret.y = CUDA_INVALID;
        ret.z = CUDA_INVALID;
      }

    return ret;
  }

  /* Returns number of devices. */
  std::size_t
  size ()
  {
    populate ();
    return m_underlying.size ();
  }

  /* Expose iterators */
  using iterator = typename std::vector<T>::iterator;
  iterator
  begin ()
  {
    populate ();
    return m_underlying.begin ();
  }
  iterator
  end ()
  {
    populate ();
    return m_underlying.end ();
  }
  using const_iterator = typename std::vector<T>::const_iterator;
  const_iterator
  cbegin ()
  {
    populate ();
    return m_underlying.cbegin ();
  }
  const_iterator
  cend ()
  {
    populate ();
    return m_underlying.cend ();
  }
};

const char *status_string[] = { "Invalid",  "Pending",    "Active",
                                "Sleeping", "Terminated", "Undetermined" };
const char *status_string_preempted = "Active (preempted)";

/* returned string must be freed */
static std::string
get_filename (struct symtab *s)
{
  if (!s)
    return "";

  /* in CLI mode, we only display the filename */
  if (!current_uiout->is_mi_like_p ())
    {
      if (s->filename)
        return s->filename;
      else
        return "";
    }

  /* in MI mode, we display the canonicalized full path */
  const char *full_path = symtab_to_fullname (s);
  if (!full_path)
    return "";

  auto real_path = gdb_realpath (full_path);
  return real_path.get ();
}

class cuda_device_info
{
private:
  bool m_current;
  uint32_t m_device;
  std::string m_name;
  std::string m_description;
  std::string m_sm_type;
  std::string m_active_sms;
  uint32_t m_num_sms;
  uint32_t m_num_warps;
  uint32_t m_num_lanes;
  uint32_t m_num_regs;
  std::string m_pci_bus;
  uint32_t m_pci_bus_id;
  uint32_t m_pci_dev_id;

public:
  /* Static members */
  constexpr static cuda_coord_set_type iterator_type
      = cuda_coord_set_type::devices;
  constexpr static underlying_type_t<cuda_coord_set_mask_t> iterator_mask
      = select_all;
  constexpr static cuda_coord_compare_type compare_type = cuda_coord_compare_type::physical;
  constexpr static command_t parser_command = CMD_FILTER;
  static cuda_filters
  default_filter ()
  {
    return cuda_filters{ cuda_coords::wild () };
  }
  static void
  check_invalid_filter (const cuda_filters &filter)
  {
  }

  /* Non-static members */
  explicit cuda_device_info (const cuda_coords &coords)
      : m_current{ coords == cuda_current_focus::get () },
        m_device{ coords.physical ().dev () },
        m_name{ cuda_state::device_get_device_name (
            coords.physical ().dev ()) },
        m_description{ cuda_state::device_get_device_type (
            coords.physical ().dev ()) },
        m_sm_type{ cuda_state::device_get_sm_type (
            coords.physical ().dev ()) },
        m_num_sms{ cuda_state::device_get_num_sms (
            coords.physical ().dev ()) },
        m_num_warps{ cuda_state::device_get_num_warps (
            coords.physical ().dev ()) },
        m_num_lanes{ cuda_state::device_get_num_lanes (
            coords.physical ().dev ()) },
        m_num_regs{ cuda_state::device_get_num_registers (
            coords.physical ().dev ()) },
        m_pci_bus_id{ cuda_state::device_get_pci_bus_id (
            coords.physical ().dev ()) },
        m_pci_dev_id{ cuda_state::device_get_pci_dev_id (
            coords.physical ().dev ()) }
  {
    // Create the active mask string
    m_active_sms = cuda_state::device_get_active_sms_mask (m_device).to_hex_string ();
    // Create the pci bus string
    std::stringstream pci_ss;
    pci_ss << std::setfill ('0') << std::setw (2) << std::hex << m_pci_bus_id
           << ":" << std::setfill ('0') << std::setw (2) << std::hex
           << m_pci_dev_id << ".0";
    m_pci_bus = pci_ss.str ();
  }

  /* Getters */
  bool
  current () const noexcept
  {
    return m_current;
  }
  uint32_t
  device () const noexcept
  {
    return m_device;
  }
  const std::string &
  name () const noexcept
  {
    return m_name;
  }
  const std::string &
  description () const noexcept
  {
    return m_description;
  }
  const std::string &
  sm_type () const noexcept
  {
    return m_sm_type;
  }
  const std::string &
  active_sms () const noexcept
  {
    return m_active_sms;
  }
  uint32_t
  num_sms () const noexcept
  {
    return m_num_sms;
  }
  uint32_t
  num_warps () const noexcept
  {
    return m_num_warps;
  }
  uint32_t
  num_lanes () const noexcept
  {
    return m_num_lanes;
  }
  uint32_t
  num_regs () const noexcept
  {
    return m_num_regs;
  }
  const std::string &
  pci_bus () const noexcept
  {
    return m_pci_bus;
  }

  /* Init other destructors/constructors */
  ~cuda_device_info () = default;
  cuda_device_info () = delete;
  cuda_device_info (const cuda_device_info &) = default;
  cuda_device_info (cuda_device_info &&) = default;
  cuda_device_info &operator= (const cuda_device_info &) = default;
  cuda_device_info &operator= (cuda_device_info &&) = default;
};

void
info_cuda_devices_command (const char *arg)
{
  struct ui_out *uiout = current_uiout;
  struct
  {
    size_t current, device, pci_bus, name, description, sm_type, num_sms,
        num_warps, num_lanes, num_regs, active_sms_mask;
  } width;

  /* column header */
  const std::string header_current{ " " };
  const std::string header_device{ "Dev" };
  const std::string header_pci_bus{ "PCI Bus/Dev ID" };
  const std::string header_name{ "Name" };
  const std::string header_description{ "Description" };
  const std::string header_sm_type{ "SM Type" };
  const std::string header_num_sms{ "SMs" };
  const std::string header_num_warps{ "Warps/SM" };
  const std::string header_num_lanes{ "Lanes/Warp" };
  const std::string header_num_regs{ "Max Regs/Lane" };
  const std::string header_active_sms_mask{ "Active SMs Mask" };

  /* get the information */
  cuda_info<cuda_device_info> devs{ arg };

  /* output message if the list is empty */
  if (devs.size () == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _ ("No CUDA devices.\n"));
      return;
    }

  /* column widths */
  width.current = header_current.length ();
  width.device = header_device.length ();
  width.pci_bus = header_pci_bus.length ();
  width.name = header_name.length ();
  width.description = header_description.length ();
  width.sm_type = header_sm_type.length ();
  width.num_sms = header_num_sms.length ();
  width.num_warps = header_num_warps.length ();
  width.num_lanes = header_num_lanes.length ();
  width.num_regs = header_num_regs.length ();
  width.active_sms_mask = header_active_sms_mask.length ();

  for (const auto &d : devs)
    {
      width.name = std::max (width.name, d.name ().length ());
      width.description
          = std::max (width.description, d.description ().length ());
      width.sm_type = std::max (width.sm_type, d.sm_type ().length ());
      width.active_sms_mask
          = std::max (width.active_sms_mask, d.active_sms ().length ());
    }

  /* print table header */
  ui_out_emit_table table_emitter (uiout, 11, devs.size (),
                                   "InfoCudaDevicesTable");
  uiout->table_header (width.current, ui_right, "current", header_current);
  uiout->table_header (width.device, ui_right, "device", header_device);
  uiout->table_header (width.pci_bus, ui_right, "pci_bus", header_pci_bus);
  uiout->table_header (width.name, ui_right, "name", header_name);
  uiout->table_header (width.description, ui_right, "description",
                       header_description);
  uiout->table_header (width.sm_type, ui_right, "sm_type", header_sm_type);
  uiout->table_header (width.num_sms, ui_right, "num_sms", header_num_sms);
  uiout->table_header (width.num_warps, ui_right, "num_warps",
                       header_num_warps);
  uiout->table_header (width.num_lanes, ui_right, "num_lanes",
                       header_num_lanes);
  uiout->table_header (width.num_regs, ui_right, "num_regs", header_num_regs);
  uiout->table_header (width.active_sms_mask, ui_right, "active_sms_mask",
                       header_active_sms_mask);
  uiout->table_body ();

  /* print table rows */
  for (const auto &d : devs)
    {
      ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaDevicesRow");
      uiout->field_string ("current", d.current () ? "*" : " ");
      uiout->field_signed ("device", d.device ());
      uiout->field_string ("pci_bus", d.pci_bus ());
      uiout->field_string ("name", d.name ());
      uiout->field_string ("description", d.description ());
      uiout->field_string ("sm_type", d.sm_type ());
      uiout->field_signed ("num_sms", d.num_sms ());
      uiout->field_signed ("num_warps", d.num_warps ());
      uiout->field_signed ("num_lanes", d.num_lanes ());
      uiout->field_signed ("num_regs", d.num_regs ());
      uiout->field_string ("active_sms_mask", d.active_sms ());
      uiout->text ("\n");
    }

  gdb_flush (gdb_stdout);
}

class cuda_sm_info
{
private:
  bool m_current;
  uint32_t m_device;
  uint32_t m_sm;
  cuda_api_warpmask m_active_warps_mask;
  std::string m_active_warps;

public:
  /* Static members */
  constexpr static cuda_coord_set_type iterator_type = cuda_coord_set_type::sms;
  constexpr static underlying_type_t<cuda_coord_set_mask_t> iterator_mask
      = select_all;
  constexpr static cuda_coord_compare_type compare_type = cuda_coord_compare_type::physical;
  constexpr static command_t parser_command = CMD_FILTER;
  static cuda_filters
  default_filter ()
  {
    /* Default to current device */
    cuda_coords c{ CUDA_CURRENT,      CUDA_WILDCARD,     CUDA_WILDCARD,
                   CUDA_WILDCARD,     CUDA_WILDCARD,     CUDA_WILDCARD,
                   CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM };
    return cuda_filters{ c };
  }
  static void
  check_invalid_filter (const cuda_filters &filter)
  {
  }

  /* Non-static members */
  explicit cuda_sm_info (const cuda_coords &coords)
      : m_current{ coords == cuda_current_focus::get () },
        m_device{ coords.physical ().dev () }, m_sm{ coords.physical ().sm () }
  {
    // Ignore invalid sms
    if (!cuda_state::sm_valid (m_device, m_sm))
      throw_error (NOT_AVAILABLE_ERROR, _ ("Ignoring invalid SM"));
    // Get the mask
    cuda_api_cp_mask (&m_active_warps_mask,
                      cuda_state::sm_get_valid_warps_mask (m_device, m_sm));
    // Ignore empty masks
    if (cuda_api_has_bit (&m_active_warps_mask))
      // Convert this to a string
      m_active_warps
          = std::string{ cuda_api_mask_string (&m_active_warps_mask) };
    else
      throw_error (NOT_AVAILABLE_ERROR, _ ("Ignoring SMs with empty masks"));
  }

  /* Getters */
  bool
  current () const noexcept
  {
    return m_current;
  }
  uint32_t
  device () const noexcept
  {
    return m_device;
  }
  uint32_t
  sm () const noexcept
  {
    return m_sm;
  }
  const std::string &
  active_warps () const noexcept
  {
    return m_active_warps;
  }

  /* Init other destructors/constructors */
  ~cuda_sm_info () = default;
  cuda_sm_info () = delete;
  cuda_sm_info (const cuda_sm_info &) = default;
  cuda_sm_info (cuda_sm_info &&) = default;
  cuda_sm_info &operator= (const cuda_sm_info &) = default;
  cuda_sm_info &operator= (cuda_sm_info &&) = default;
};

void
info_cuda_sms_command (const char *arg)
{
  struct
  {
    size_t current, sm, active_warps_mask;
  } width;

  /* column headers */
  const std::string header_current{ " " };
  const std::string header_sm{ "SM" };
  const std::string header_active_warps_mask{ "Active Warps Mask" };
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_info<cuda_sm_info> sms{ arg };

  /* output message if the list is empty */
  if (sms.size () == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _ ("No CUDA SMs.\n"));
      return;
    }

  /* column widths */
  width.current = header_current.length ();
  width.sm = header_sm.length ();
  width.active_warps_mask = header_active_warps_mask.length ();
  for (const auto &sm : sms)
    width.active_warps_mask
        = std::max (width.active_warps_mask, sm.active_warps ().length ());

  ui_out_emit_table table_emitter (uiout, 3, sms.size (), "InfoCudaSmsTable");

  /* print table header */
  uiout->table_header (width.current, ui_right, "current", header_current);
  uiout->table_header (width.sm, ui_right, "sm", header_sm);
  uiout->table_header (width.active_warps_mask, ui_right, "active_warps_mask",
                       header_active_warps_mask);
  uiout->table_body ();

  /* print table rows */
  bool first = true;
  uint32_t current_device = 0;
  for (const auto &sm : sms)
    {
      if (!uiout->is_mi_like_p ())
        {
          if (first || sm.device () != current_device)
            {
              first = false;
              current_device = sm.device ();
              uiout->message ("Device %u\n", sm.device ());
            }
        }
      ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaSmsRow");
      uiout->field_string ("current", sm.current () ? "*" : " ");
      uiout->field_signed ("sm", sm.sm ());
      uiout->field_string ("active_warps_mask", sm.active_warps ());
      uiout->text ("\n");
    }

  gdb_flush (gdb_stdout);
}

class cuda_warp_info
{
private:
  bool m_current;
  uint32_t m_device;
  uint32_t m_sm;
  uint32_t m_wp;
  std::string m_active_lanes;
  std::string m_divergent_lanes;
  std::string m_kernel_id;
  std::string m_blockIdx;
  std::string m_threadIdx;
  std::string m_pc;

public:
  /* Static members */
  constexpr static cuda_coord_set_type iterator_type
      = cuda_coord_set_type::warps;
  constexpr static underlying_type_t<cuda_coord_set_mask_t> iterator_mask
      = select_all;
  constexpr static cuda_coord_compare_type compare_type = cuda_coord_compare_type::physical;
  constexpr static command_t parser_command = CMD_FILTER;
  static cuda_filters
  default_filter ()
  {
    /* Default to current device and sm */
    cuda_coords c{ CUDA_CURRENT,      CUDA_CURRENT,      CUDA_WILDCARD,
                   CUDA_WILDCARD,     CUDA_WILDCARD,     CUDA_WILDCARD,
                   CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM };
    return cuda_filters{ c };
  }
  static void
  check_invalid_filter (const cuda_filters &filter)
  {
  }

  /* Non-static members */
  explicit cuda_warp_info (const cuda_coords &coords)
      : m_current{ coords == cuda_current_focus::get () },
        m_device{ coords.physical ().dev () },
        m_sm{ coords.physical ().sm () }, m_wp{ coords.physical ().wp () }
  {
    /* Ignore invalid warps */
    if (!cuda_state::sm_valid (m_device, m_sm))
	throw_error (NOT_AVAILABLE_ERROR, _ ("Ignoring invalid SM"));

    if (!cuda_state::warp_valid (m_device, m_sm, m_wp))
	throw_error (NOT_AVAILABLE_ERROR, _ ("Ignoring invalid warp"));

    auto active_lanes_mask
        = cuda_state::warp_get_active_lanes_mask (m_device, m_sm, m_wp);
    m_active_lanes = to_hex (active_lanes_mask);
    m_divergent_lanes = to_hex (
        cuda_state::warp_get_divergent_lanes_mask (m_device, m_sm, m_wp));
    auto kernel = cuda_state::warp_get_kernel (m_device, m_sm, m_wp);
    auto kernel_id = kernel_get_id (kernel);
    m_kernel_id = std::to_string (kernel_id);
    m_blockIdx = dim3_to_string (
        cuda_state::warp_get_block_idx (m_device, m_sm, m_wp));
    m_threadIdx = dim3_to_string (cuda_state::lane_get_thread_idx (
        m_device, m_sm, m_wp, __builtin_ctz (active_lanes_mask)));
    m_pc = to_hex (cuda_state::warp_get_active_pc (m_device, m_sm, m_wp));
  }

  /* Getters */
  bool
  current () const noexcept
  {
    return m_current;
  }
  uint32_t
  device () const noexcept
  {
    return m_device;
  }
  uint32_t
  sm () const noexcept
  {
    return m_sm;
  }
  uint32_t
  warp () const noexcept
  {
    return m_wp;
  }
  const std::string &
  active_lanes () const noexcept
  {
    return m_active_lanes;
  }
  const std::string &
  divergent_lanes () const noexcept
  {
    return m_divergent_lanes;
  }
  const std::string &
  kernel_id () const noexcept
  {
    return m_kernel_id;
  }
  const std::string &
  blockIdx () const noexcept
  {
    return m_blockIdx;
  }
  const std::string &
  threadIdx () const noexcept
  {
    return m_threadIdx;
  }
  const std::string &
  pc () const noexcept
  {
    return m_pc;
  }

  /* Init other destructors/constructors */
  ~cuda_warp_info () = default;
  cuda_warp_info () = delete;
  cuda_warp_info (const cuda_warp_info &) = default;
  cuda_warp_info (cuda_warp_info &&) = default;
  cuda_warp_info &operator= (const cuda_warp_info &) = default;
  cuda_warp_info &operator= (cuda_warp_info &&) = default;
};

void
info_cuda_warps_command (const char *arg)
{
  struct
  {
    size_t current, wp, active_lanes_mask, divergent_lanes_mask,
        active_pc, kernel_id, blockIdx, threadIdx;
  } width;

  /* column headers */
  const std::string header_current{ " " };
  const std::string header_wp{ "Wp" };
  const std::string header_active_lanes_mask{ "Active Lanes Mask" };
  const std::string header_divergent_lanes_mask{ "Divergent Lanes Mask" };
  const std::string header_active_pc{ "Active PC" };
  const std::string header_kernel_id{ "Kernel" };
  const std::string header_blockIdx{ "BlockIdx" };
  const std::string header_threadIdx{ "First Active ThreadIdx" };
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_info<cuda_warp_info> warps{ arg };

  /* output message if the list is empty */
  if (warps.size () == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _ ("No CUDA Warps.\n"));
      return;
    }

  /* column widths */
  width.current = header_current.length ();
  width.wp = header_wp.length ();
  width.active_lanes_mask = header_active_lanes_mask.length ();
  width.divergent_lanes_mask = header_divergent_lanes_mask.length ();
  width.active_pc = header_active_pc.length ();
  width.kernel_id = header_kernel_id.length ();
  width.blockIdx = header_blockIdx.length ();
  width.threadIdx = header_threadIdx.length ();

  for (const auto &wp : warps)
    {
      width.active_lanes_mask
          = std::max (width.active_lanes_mask, wp.active_lanes ().length ());
      width.divergent_lanes_mask = std::max (width.divergent_lanes_mask,
                                             wp.divergent_lanes ().length ());
      width.blockIdx = std::max (width.blockIdx, wp.blockIdx ().length ());
      width.threadIdx = std::max (width.threadIdx, wp.threadIdx ().length ());
      width.active_pc
          = std::max (width.active_pc, wp.pc ().length ());
    }

  /* print table header */
  ui_out_emit_table table_emitter (uiout, 8, warps.size (),
                                   "InfoCudaWarpsTable");
  uiout->table_header (width.current, ui_right, "current", header_current);
  uiout->table_header (width.wp, ui_right, "warp", header_wp);
  uiout->table_header (width.active_lanes_mask, ui_right, "active_lanes_mask",
                       header_active_lanes_mask);
  uiout->table_header (width.divergent_lanes_mask, ui_right,
                       "divergent_lanes_mask", header_divergent_lanes_mask);
  uiout->table_header (width.active_pc, ui_right,
                       "active_physical_pc", header_active_pc);
  uiout->table_header (width.kernel_id, ui_right, "kernel", header_kernel_id);
  uiout->table_header (width.blockIdx, ui_right, "blockIdx", header_blockIdx);
  uiout->table_header (width.threadIdx, ui_right, "threadIdx",
                       header_threadIdx);
  uiout->table_body ();

  /* print table rows */
  bool first = true;
  uint32_t current_device = 0;
  uint32_t current_sm = 0;
  for (const auto &wp : warps)
    {
      if (!uiout->is_mi_like_p ())
        {
          if (first || wp.device () != current_device
              || wp.sm () != current_sm)
            {
              uiout->message ("Device %u SM %u\n", wp.device (), wp.sm ());
              first = false;
              current_device = wp.device ();
              current_sm = wp.sm ();
            }
        }

      ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaWarpsRow");
      uiout->field_string ("current", wp.current () ? "*" : " ");
      uiout->field_signed ("warp", wp.warp ());
      uiout->field_string ("active_lanes_mask", wp.active_lanes ());
      uiout->field_string ("divergent_lanes_mask", wp.divergent_lanes ());
      uiout->field_string ("active_physical_pc", wp.pc ());
      uiout->field_string ("kernel", wp.kernel_id ());
      uiout->field_string ("blockIdx", wp.blockIdx ());
      uiout->field_string ("threadIdx", wp.threadIdx ());
      uiout->text ("\n");
    }

  gdb_flush (gdb_stdout);
}

class cuda_lane_info
{
private:
  bool m_current;
  uint32_t m_device;
  uint32_t m_sm;
  uint32_t m_wp;
  uint32_t m_ln;
  std::string m_threadIdx;
  std::string m_pc;
  std::string m_state;
  std::string m_exception;

public:
  /* Static members */
  constexpr static cuda_coord_set_type iterator_type
      = cuda_coord_set_type::lanes;
  constexpr static underlying_type_t<cuda_coord_set_mask_t> iterator_mask
      = select_all;
  constexpr static cuda_coord_compare_type compare_type = cuda_coord_compare_type::physical;
  constexpr static command_t parser_command = CMD_FILTER;
  static cuda_filters
  default_filter ()
  {
    /* Default to current device, sm, and wp */
    cuda_coords c{ CUDA_CURRENT,      CUDA_CURRENT,      CUDA_CURRENT,
                   CUDA_WILDCARD,     CUDA_WILDCARD,     CUDA_WILDCARD,
                   CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM };
    return cuda_filters{ c };
  }
  static void
  check_invalid_filter (const cuda_filters &filter)
  {
  }

  /* Non-static members */
  explicit cuda_lane_info (const cuda_coords &coords)
      : m_current{ coords == cuda_current_focus::get () },
        m_device{ coords.physical ().dev () },
        m_sm{ coords.physical ().sm () }, m_wp{ coords.physical ().wp () },
        m_ln{ coords.physical ().ln () }
  {
    if (!cuda_state::device_valid (m_device))
      throw_error (NOT_AVAILABLE_ERROR, _ ("Ignore invalid devices"));

    if (!cuda_state::sm_valid (m_device, m_sm))
      throw_error (NOT_AVAILABLE_ERROR, _ ("Ignore invalid SMs"));

    if (!cuda_state::warp_valid (m_device, m_sm, m_wp))
      throw_error (NOT_AVAILABLE_ERROR, _ ("Ignore invalid warps"));

    if (!cuda_state::lane_valid (m_device, m_sm, m_wp, m_ln))
      throw_error (NOT_AVAILABLE_ERROR, _ ("Ignore invalid lanes"));

    m_threadIdx = dim3_to_string (
        cuda_state::lane_get_thread_idx (m_device, m_sm, m_wp, m_ln));
    m_pc = to_hex (cuda_state::lane_get_pc (m_device, m_sm, m_wp, m_ln));
    auto active = cuda_state::lane_active (m_device, m_sm, m_wp, m_ln);
    m_state = active ? std::string{ "active" } : std::string{ "divergent" };
    auto exception
        = cuda_state::lane_get_exception (m_device, m_sm, m_wp, m_ln);
    m_exception
        = exception == CUDBG_EXCEPTION_NONE
              ? std::string{ "None" }
              : std::string{ cuda_exception::type_to_name (exception) };
  }

  /* Getters */
  bool
  current () const noexcept
  {
    return m_current;
  }
  uint32_t
  device () const noexcept
  {
    return m_device;
  }
  uint32_t
  sm () const noexcept
  {
    return m_sm;
  }
  uint32_t
  warp () const noexcept
  {
    return m_wp;
  }
  uint32_t
  lane () const noexcept
  {
    return m_ln;
  }
  const std::string &
  threadIdx () const noexcept
  {
    return m_threadIdx;
  }
  const std::string &
  pc () const noexcept
  {
    return m_pc;
  }
  const std::string &
  state () const noexcept
  {
    return m_state;
  }
  const std::string &
  exception () const noexcept
  {
    return m_exception;
  }

  /* Init other destructors/constructors */
  ~cuda_lane_info () = default;
  cuda_lane_info () = delete;
  cuda_lane_info (const cuda_lane_info &) = default;
  cuda_lane_info (cuda_lane_info &&) = default;
  cuda_lane_info &operator= (const cuda_lane_info &) = default;
  cuda_lane_info &operator= (cuda_lane_info &&) = default;
};

void
info_cuda_lanes_command (const char *arg)
{
  struct
  {
    size_t current, ln, state, pc, thread_idx, exception;
  } width;

  /* column headers */
  const std::string header_current{ " " };
  const std::string header_ln{ "Ln" };
  const std::string header_state{ "State" };
  const std::string header_pc{ "PC" };
  const std::string header_thread_idx{ "ThreadIdx" };
  const std::string header_exception{ "Exception" };
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_info<cuda_lane_info> lanes{ arg };

  /* output message if the list is empty */
  if (lanes.size () == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _ ("No CUDA Lanes.\n"));
      return;
    }

  /* column widths */
  width.current = header_current.length ();
  width.ln = header_ln.length ();
  width.state = header_state.length ();
  width.pc = header_pc.length ();
  width.thread_idx = header_thread_idx.length ();
  width.exception = header_exception.length ();

  for (const auto &ln : lanes)
    {
      width.thread_idx
          = std::max (width.thread_idx, ln.threadIdx ().length ());
      width.pc = std::max (width.pc, ln.pc ().length ());
      width.state = std::max (width.state, ln.state ().length ());
      width.exception = std::max (width.exception, ln.exception ().length ());
    }

  /* print table header */
  ui_out_emit_table table_emitter (uiout, 6, lanes.size (),
                                   "InfoCudaLanesTable");
  uiout->table_header (width.current, ui_right, "current", header_current);
  uiout->table_header (width.ln, ui_right, "lane", header_ln);
  uiout->table_header (width.state, ui_center, "state", header_state);
  uiout->table_header (width.pc, ui_center, "physical_pc",
                       header_pc);
  uiout->table_header (width.thread_idx, ui_right, "threadIdx",
                       header_thread_idx);
  uiout->table_header (width.exception, ui_center, "exception",
                       header_exception);
  uiout->table_body ();

  /* print table rows */
  bool first = true;
  uint32_t current_device = 0;
  uint32_t current_sm = 0;
  uint32_t current_wp = 0;
  for (const auto &ln : lanes)
    {
      if (!uiout->is_mi_like_p ())
        {
          if (first || ln.device () != current_device || ln.sm () != current_sm
              || ln.warp () != current_wp)
            {
              uiout->message ("Device %u SM %u Warp %u\n", ln.device (),
                              ln.sm (), ln.warp ());
              first = false;
              current_device = ln.device ();
              current_sm = ln.sm ();
              current_wp = ln.warp ();
            }
        }

      ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaLanesRow");
      uiout->field_string ("current", ln.current () ? "*" : " ");
      uiout->field_signed ("lane", ln.lane ());
      uiout->field_string ("state", ln.state ());
      uiout->field_string ("physical_pc", ln.pc ());
      uiout->field_string ("threadIdx", ln.threadIdx ());
      uiout->field_string ("exception", ln.exception ());
      uiout->text ("\n");
    }

  gdb_flush (gdb_stdout);
}

class cuda_kernel
{
private:
  bool m_current;
  uint64_t m_kernel_id;
  uint32_t m_device;
  int64_t m_grid_id;
  std::string m_sms_mask;
  std::string m_parent;
  std::string m_status;
  std::string m_grid_dim;
  std::string m_block_dim;
  std::string m_invocation;

public:
  /* Static members */
  constexpr static cuda_coord_set_type iterator_type
      = cuda_coord_set_type::kernels;
  constexpr static underlying_type_t<cuda_coord_set_mask_t> iterator_mask
      = select_all;
  constexpr static cuda_coord_compare_type compare_type = cuda_coord_compare_type::logical;
  constexpr static command_t parser_command = CMD_FILTER;
  static cuda_filters
  default_filter ()
  {
    return cuda_filters{ cuda_coords::wild () };
  }
  static void
  check_invalid_filter (const cuda_filters &filter)
  {
  }

  /* Non-static members */
  explicit cuda_kernel (const cuda_coords &coords)
      : m_kernel_id{ coords.logical ().kernelId () }
  {
    error (_ ("cuda_kernel cannot be constructed via coords!"));
  }

  explicit cuda_kernel (kernel_t kernel)
  {
    m_current = (kernel == cuda_current_focus::get ().logical ().kernel ());
    m_kernel_id = kernel_get_id (kernel);
    m_device = kernel_get_dev_id (kernel);
    m_grid_id = kernel_get_grid_id (kernel);

    // Get the number of sms for this device
    cuda_bitset active_sms_mask (cuda_state::device_get_num_sms (m_device));

    kernel_compute_sms_mask (kernel, active_sms_mask);
    m_sms_mask = active_sms_mask.to_hex_string ();

    auto parent_kernel = kernel_get_parent (kernel);
    if (parent_kernel)
      m_parent = std::to_string (kernel_get_id (parent_kernel));
    else
      m_parent = std::string{ "-" };

    auto status = kernel_get_status (kernel);
    m_status = (cuda_options_software_preemption ()
                && status == CUDBG_GRID_STATUS_ACTIVE)
                   ? std::string{ status_string_preempted }
                   : std::string{ status_string[status] };
    m_grid_dim = dim3_to_string (kernel_get_grid_dim (kernel));
    m_block_dim = dim3_to_string (kernel_get_block_dim (kernel));
    m_invocation = invocation_to_string (kernel_get_name (kernel),
                                         kernel_get_args (kernel));
  }

  /* Getters */
  bool
  current () const noexcept
  {
    return m_current;
  }
  uint64_t
  kernel_id () const noexcept
  {
    return m_kernel_id;
  }
  uint32_t
  device () const noexcept
  {
    return m_device;
  }
  int64_t
  grid_id () const noexcept
  {
    return m_grid_id;
  }
  const std::string &
  sms_mask () const noexcept
  {
    return m_sms_mask;
  }
  const std::string &
  parent () const noexcept
  {
    return m_parent;
  }
  const std::string &
  status () const noexcept
  {
    return m_status;
  }
  const std::string &
  grid_dim () const noexcept
  {
    return m_grid_dim;
  }
  const std::string &
  block_dim () const noexcept
  {
    return m_block_dim;
  }
  const std::string &
  invocation () const noexcept
  {
    return m_invocation;
  }

  /* Init other destructors/constructors */
  ~cuda_kernel () = default;
  cuda_kernel () = delete;
  cuda_kernel (const cuda_kernel &) = default;
  cuda_kernel (cuda_kernel &&) = default;
  cuda_kernel &operator= (const cuda_kernel &) = default;
  cuda_kernel &operator= (cuda_kernel &&) = default;
};

/* Special handling for lists of cuda_kernel */
class cuda_kernel_info final : public cuda_info<cuda_kernel>
{
public:
  explicit cuda_kernel_info (const char *filter_string)
      : cuda_info{ filter_string }
  {
  }

  /* FIXME: We don't have a proper cuda iterator for kernels... */
  /* This hacks things to create a list of kernel_t */
  bool
  ignore_coord (const cuda_coords &coords) override
  {
    /* Only do this once!! */
    if (!m_underlying.empty ())
      return true;

    /* Just ignore the filter - we want to print out every kernel. */
    for (auto kernel = kernels_get_first_kernel (); kernel;
         kernel = kernels_get_next_kernel (kernel))
      {
        if (!kernel_is_present (kernel))
          continue;

        m_underlying.emplace_back (kernel);
      }

    /* We fully constructed the list for every kernel in the chain at this
     * point. */
    return true;
  }
};

void
info_cuda_kernels_command (const char *arg)
{
  struct
  {
    size_t current, kernel, device, grid, parent, status, sms_mask, grid_dim,
        block_dim, invocation;
  } width;

  /* column headers */
  const std::string header_current{ " " };
  const std::string header_kernel{ "Kernel" };
  const std::string header_device{ "Dev" };
  const std::string header_parent{ "Parent" };
  const std::string header_grid{ "Grid" };
  const std::string header_status{ "Status" };
  const std::string header_sms_mask{ "SMs Mask" };
  const std::string header_grid_dim{ "GridDim" };
  const std::string header_block_dim{ "BlockDim" };
  const std::string header_invocation{ "Invocation" };
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_kernel_info kernels{ arg };

  /* output message if the list is empty */
  if (kernels.size () == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _ ("No CUDA kernels.\n"));
      return;
    }

  /* column widths */
  width.current = header_current.length ();
  width.kernel = header_kernel.length ();
  width.device = header_device.length ();
  width.grid = header_grid.length ();
  width.status = header_status.length ();
  width.parent = header_parent.length ();
  width.sms_mask = header_sms_mask.length ();
  width.grid_dim = header_grid_dim.length ();
  width.block_dim = header_block_dim.length ();
  width.invocation = header_invocation.length ();

  for (const auto &k : kernels)
    {
      width.status = std::max (width.status, k.status ().length ());
      width.parent = std::max (width.parent, k.parent ().length ());
      width.sms_mask = std::max (width.sms_mask, k.sms_mask ().length ());
      width.grid_dim = std::max (width.grid_dim, k.grid_dim ().length ());
      width.block_dim = std::max (width.block_dim, k.block_dim ().length ());
      width.invocation
          = std::max (width.invocation, k.invocation ().length ());
    }

  /* print table header */
  ui_out_emit_table table_emitter (uiout, 10, kernels.size (),
                                   "InfoCudaKernelsTable");
  uiout->table_header (width.current, ui_right, "current", header_current);
  uiout->table_header (width.kernel, ui_right, "kernel", header_kernel);
  uiout->table_header (width.parent, ui_right, "parent", header_parent);
  uiout->table_header (width.device, ui_right, "device", header_device);
  uiout->table_header (width.grid, ui_right, "grid", header_grid);
  uiout->table_header (width.status, ui_right, "status", header_status);
  uiout->table_header (width.sms_mask, ui_right, "sms_mask", header_sms_mask);
  uiout->table_header (width.grid_dim, ui_right, "gridDim", header_grid_dim);
  uiout->table_header (width.block_dim, ui_right, "blockDim",
                       header_block_dim);
  uiout->table_header (width.invocation, ui_left, "invocation",
                       header_invocation);
  uiout->table_body ();

  /* print table rows */
  for (const auto &k : kernels)
    {
      ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaKernelsRow");
      uiout->field_string ("current", k.current () ? "*" : " ");
      uiout->field_signed ("kernel", k.kernel_id ());
      uiout->field_string ("parent", k.parent ());
      uiout->field_signed ("device", k.device ());
      uiout->field_signed ("grid", k.grid_id ());
      uiout->field_string ("status", k.status ());
      uiout->field_string ("sms_mask", k.sms_mask ());
      uiout->field_string ("gridDim", k.grid_dim ());
      uiout->field_string ("blockDim", k.block_dim ());
      uiout->field_string ("invocation", k.invocation ());
      uiout->text ("\n");
    }

  gdb_flush (gdb_stdout);
}

class cuda_block
{
private:
  bool m_current;
  uint64_t m_kernel_id;
  std::string m_start_block_idx;
  std::string m_end_block_idx;
  uint32_t m_count;
  uint32_t m_device;
  uint32_t m_sm;

public:
  /* Static members */
  constexpr static cuda_coord_set_type iterator_type
      = cuda_coord_set_type::blocks;
  constexpr static underlying_type_t<cuda_coord_set_mask_t> iterator_mask
      = select_valid;
  constexpr static cuda_coord_compare_type compare_type = cuda_coord_compare_type::logical;
  constexpr static command_t parser_command = CMD_FILTER;
  static cuda_filters
  default_filter ()
  {
    /* Default to current kernel */
    cuda_coords c{ CUDA_WILDCARD,     CUDA_WILDCARD,     CUDA_WILDCARD,
                   CUDA_WILDCARD,     CUDA_CURRENT,      CUDA_WILDCARD,
                   CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM };
    return cuda_filters{ c };
  }
  static void
  check_invalid_filter (const cuda_filters &filter)
  {
  }

  /* Non-static members */
  explicit cuda_block (const cuda_coords &coords)
      : m_current{ coords == cuda_current_focus::get () },
        m_kernel_id{ coords.logical ().kernelId () }, m_count{ 1 },
        m_device{ coords.physical ().dev () }, m_sm{ coords.physical ().sm () }
  {
    /* Set the start index strings */
    m_start_block_idx = dim3_to_string (coords.logical ().blockIdx ());
  }

  /* Setters */
  void
  add_to_range (bool current)
  {
    ++m_count;
    m_current |= current;
  }
  /* Only call the following setters once! */
  void
  finalize_range (const CuDim3 &blockIdx)
  {
    gdb_assert (m_end_block_idx.length () == 0);
    m_end_block_idx = dim3_to_string (blockIdx);
  }

  /* Getters */
  bool
  current () const noexcept
  {
    return m_current;
  }
  uint64_t
  kernel_id () const noexcept
  {
    return m_kernel_id;
  }
  const std::string &
  start_block_idx () const noexcept
  {
    return m_start_block_idx;
  }
  const std::string &
  end_block_idx () const noexcept
  {
    return m_end_block_idx;
  }
  uint32_t
  count () const noexcept
  {
    return m_count;
  }
  uint32_t
  device () const noexcept
  {
    return m_device;
  }
  uint32_t
  sm () const noexcept
  {
    return m_sm;
  }

  /* Init other destructors/constructors */
  ~cuda_block () = default;
  cuda_block () = delete;
  cuda_block (const cuda_block &) = default;
  cuda_block (cuda_block &&) = default;
  cuda_block &operator= (const cuda_block &) = default;
  cuda_block &operator= (cuda_block &&) = default;
};

/* Special handling for lists of cuda_block */
class cuda_block_info final : public cuda_info<cuda_block>
{
private:
  cuda_coords m_expected;
  CuDim3 m_prev_block_idx;

  /* Returns a CuDim3 representing a block advanced to the nearest neighbor */
  CuDim3
  incrementBlock (const CuDim3 &blockIdx, const CuDim3 &gridDim)
  {
    return incrementDim (blockIdx, gridDim);
  }

public:
  explicit cuda_block_info (const char *filter_string)
      : cuda_info{ filter_string }, m_expected{ cuda_coords::wild () },
        m_prev_block_idx{ CUDA_INVALID, CUDA_INVALID, CUDA_INVALID }
  {
  }

  bool
  ignore_coord (const cuda_coords &coords) override
  {
    auto kernel = kernels_find_kernel_by_grid_id (coords.physical ().dev (),
                                                  coords.logical ().gridId ());

    /* See if this is a new range */
    bool break_of_contiguity = false;
    if (!m_underlying.empty ())
      {
        break_of_contiguity = (m_expected.logical () != coords.logical ());
      }

    /* close the current range */
    if (!m_underlying.empty ()
        && (break_of_contiguity || !cuda_options_coalescing ()))
      {
        m_underlying.back ().finalize_range (m_prev_block_idx);
      }

    /* data for the next iteration */
    m_prev_block_idx = coords.logical ().blockIdx ();
    /* Advance to the next block */
    auto nextBlockIdx
        = incrementBlock (m_prev_block_idx, kernel_get_grid_dim (kernel));
    m_expected = cuda_coords{ CUDA_WILDCARD,
                              CUDA_WILDCARD,
                              CUDA_WILDCARD,
                              CUDA_WILDCARD,
                              coords.logical ().kernelId (),
                              CUDA_WILDCARD,
                              CUDA_WILDCARD_DIM,
                              nextBlockIdx,
                              CUDA_WILDCARD_DIM };

    /* Check to see if we need to add this range. */
    if (m_underlying.empty () || break_of_contiguity
        || !cuda_options_coalescing ())
      {
        /* Add it */
        return false;
      }
    /* Don't add this range - return true to ignore */
    else
      {
        m_underlying.back ().add_to_range (coords
                                           == cuda_current_focus::get ());
        return true;
      }
  }

  void
  handle_post_populate () override
  {
    /* At this point, the list has been created. Let's finalize the last
     * element. */
    if (!m_underlying.empty ())
      m_underlying.back ().finalize_range (m_prev_block_idx);
  }
};

void
info_cuda_blocks_command (const char *arg)
{
  struct
  {
    size_t current, kernel, from, to, count, state, device, sm;
  } width;

  /* column headers */
  const std::string header_current{ " " };
  const std::string header_kernel{ "Kernel" };
  const std::string header_from{ "BlockIdx" };
  const std::string header_to{ "To BlockIdx" };
  const std::string header_count{ "Count" };
  const std::string header_state{ "State" };
  const std::string header_device{ "Dev" };
  const std::string header_sm{ "SM" };
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_block_info blocks{ arg };

  /* output message if the list is empty */
  if (blocks.size () == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _ ("No CUDA blocks.\n"));
      return;
    }

  std::string running{ "running" };

  /* column widths */
  width.current = header_current.length ();
  width.kernel = header_kernel.length ();
  width.from = header_from.length ();
  width.to = header_to.length ();
  width.count = header_count.length ();
  width.state = header_state.length ();
  width.device = header_device.length ();
  width.sm = header_sm.length ();

  width.state = std::max (width.state, running.length ());
  for (const auto &b : blocks)
    {
      width.from = std::max (width.from, b.start_block_idx ().length ());
      width.to = std::max (width.to, b.end_block_idx ().length ());
    }

  /* print table header ('kernel' is only present in MI output) */
  uint32_t num_columns = uiout->is_mi_like_p () ? 6 : 5;

  bool coalescing = cuda_options_coalescing ();

  /* Table/row names are based on if we're coalescing or not (for backward
   * compatibilty with Eclipse, etc) */
  const char *table_name = coalescing ? "CoalescedInfoCudaBlocksTable"
                                      : "UncoalescedInfoCudaBlocksTable";
  const char *row_name = coalescing ? "CoalescedInfoCudaBlocksRow"
                                    : "UncoalescedInfoCudaBlocksRow";

  ui_out_emit_table table_emitter (uiout, num_columns, blocks.size (),
                                   table_name);
  uiout->table_header (width.current, ui_right, "current", header_current);
  if (uiout->is_mi_like_p ())
    uiout->table_header (width.kernel, ui_right, "kernel", header_kernel);
  if (coalescing)
    {
      uiout->table_header (width.from, ui_right, "from", header_from);
      uiout->table_header (width.to, ui_right, "to", header_to);
      uiout->table_header (width.count, ui_right, "count", header_count);
    }
  else
    {
      uiout->table_header (width.from, ui_right, "blockIdx", header_from);
    }
  uiout->table_header (width.state, ui_right, "state", header_state);
  if (!coalescing)
    {
      uiout->table_header (width.device, ui_right, "device", header_device);
      uiout->table_header (width.sm, ui_right, "sm", header_sm);
    }
  uiout->table_body ();

  /* print table rows */
  uint64_t kernel_id = ~0ULL;
  for (const auto &b : blocks)
    {
      if (!uiout->is_mi_like_p () && b.kernel_id () != kernel_id)
        {
          /* row are grouped per kernel only in CLI output */
          uiout->message ("Kernel %llu\n", (unsigned long long)b.kernel_id ()),
              kernel_id = b.kernel_id ();
        }

      ui_out_emit_tuple tuple_emitter (uiout, row_name);
      uiout->field_string ("current", b.current () ? "*" : " ");
      if (uiout->is_mi_like_p ())
        uiout->field_signed ("kernel", b.kernel_id ());
      if (coalescing)
        {
          uiout->field_string ("from", b.start_block_idx ());
          uiout->field_string ("to", b.end_block_idx ());
          uiout->field_signed ("count", b.count ());
        }
      else
        {
          uiout->field_string ("blockIdx", b.start_block_idx ());
        }
      uiout->field_string ("state", running);
      if (!coalescing)
        {
          uiout->field_signed ("device", b.device ());
          uiout->field_signed ("sm", b.sm ());
        }
      uiout->text ("\n");
    }

  gdb_flush (gdb_stdout);
}

class cuda_thread
{
private:
  bool m_current;
  uint64_t m_kernel_id;
  std::string m_start_block_idx;
  std::string m_start_thread_idx;
  std::string m_end_block_idx;
  std::string m_end_thread_idx;
  uint32_t m_count;
  uint32_t m_device;
  uint32_t m_sm;
  uint32_t m_wp;
  uint32_t m_ln;
  std::string m_pc;
  std::string m_filename;
  uint32_t m_line;

public:
  /* Static members */
  constexpr static cuda_coord_set_type iterator_type
      = cuda_coord_set_type::threads;
  constexpr static underlying_type_t<cuda_coord_set_mask_t> iterator_mask
      = select_valid;
  constexpr static cuda_coord_compare_type compare_type = cuda_coord_compare_type::logical;
  constexpr static command_t parser_command = CMD_FILTER;
  static cuda_filters
  default_filter ()
  {
    /* Default to current kernel */
    cuda_coords c{ CUDA_WILDCARD,     CUDA_WILDCARD,     CUDA_WILDCARD,
                   CUDA_WILDCARD,     CUDA_CURRENT,      CUDA_WILDCARD,
                   CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM };
    return cuda_filters{ c };
  }
  static void
  check_invalid_filter (const cuda_filters &filter)
  {
  }

  /* Non-static members */
  explicit cuda_thread (const cuda_coords &coords)
      : m_current{ coords == cuda_current_focus::get () },
        m_kernel_id{ coords.logical ().kernelId () }, m_count{ 1 },
        m_device{ coords.physical ().dev () },
        m_sm{ coords.physical ().sm () }, m_wp{ coords.physical ().wp () },
        m_ln{ coords.physical ().ln () }, m_line{ 0 }
  {
    /* Set the start index strings */
    m_start_block_idx = dim3_to_string (coords.logical ().blockIdx ());
    m_start_thread_idx = dim3_to_string (coords.logical ().threadIdx ());
  }

  /* Setters */
  void
  add_to_range (bool current)
  {
    ++m_count;
    m_current |= current;
  }
  /* Only call the following setters once! */
  void
  finalize_range (const CuDim3 &blockIdx, const CuDim3 &threadIdx)
  {
    gdb_assert (m_end_block_idx.length () == 0);
    gdb_assert (m_end_thread_idx.length () == 0);
    m_end_block_idx = dim3_to_string (blockIdx);
    m_end_thread_idx = dim3_to_string (threadIdx);
  }
  void
  set_pc (uint64_t pc)
  {
    gdb_assert (m_pc.length () == 0);
    m_pc = to_hex (pc);
  }
  void
  set_filename (struct symtab *s)
  {
    gdb_assert (m_filename.length () == 0);
    m_filename = get_filename (s);
  }
  void
  set_line (uint32_t line)
  {
    gdb_assert (m_line == 0);
    m_line = line;
  }

  /* Getters */
  bool
  current () const noexcept
  {
    return m_current;
  }
  uint64_t
  kernel_id () const noexcept
  {
    return m_kernel_id;
  }
  const std::string &
  start_block_idx () const noexcept
  {
    return m_start_block_idx;
  }
  const std::string &
  start_thread_idx () const noexcept
  {
    return m_start_thread_idx;
  }
  const std::string &
  end_block_idx () const noexcept
  {
    return m_end_block_idx;
  }
  const std::string &
  end_thread_idx () const noexcept
  {
    return m_end_thread_idx;
  }
  uint32_t
  count () const noexcept
  {
    return m_count;
  }
  uint32_t
  device () const noexcept
  {
    return m_device;
  }
  uint32_t
  sm () const noexcept
  {
    return m_sm;
  }
  uint32_t
  warp () const noexcept
  {
    return m_wp;
  }
  uint32_t
  lane () const noexcept
  {
    return m_ln;
  }
  const std::string &
  pc () const noexcept
  {
    return m_pc;
  }
  const std::string &
  filename () const noexcept
  {
    return m_filename;
  }
  uint32_t
  line () const noexcept
  {
    return m_line;
  }

  /* Init other destructors/constructors */
  ~cuda_thread () = default;
  cuda_thread () = delete;
  cuda_thread (const cuda_thread &) = default;
  cuda_thread (cuda_thread &&) = default;
  cuda_thread &operator= (const cuda_thread &) = default;
  cuda_thread &operator= (cuda_thread &&) = default;
};

/* Special handling for lists of cuda_thread */
class cuda_thread_info final : public cuda_info<cuda_thread>
{
private:
  cuda_coords m_expected;
  uint64_t m_pc;
  struct symtab *m_symtab;
  uint32_t m_line;
  uint64_t m_prev_pc;
  uint32_t m_prev_line;
  CuDim3 m_prev_block_idx;
  CuDim3 m_prev_thread_idx;
  struct value_print_options m_opts;
  std::unordered_map<uint64_t, struct symtab_and_line> m_pc_to_sal;

  /* Returns a std::pair representing a block and thread advanced to the
   * nearest neighbor */
  std::pair<CuDim3, CuDim3>
  incrementThread (const CuDim3 &blockIdx, const CuDim3 &threadIdx,
                   const CuDim3 &gridDim, const CuDim3 &blockDim)
  {
    /* Initialize the returned block to blockIdx */
    CuDim3 ret_b = blockIdx;
    /* Try to increment threadIdx. */
    CuDim3 ret_t = incrementDim (threadIdx, blockDim);
    /* Check if we overflowed threadIdx. */
    if (cuda_coord_is_special (ret_t))
      {
        /* We did - increment blockIdx. */
        ret_b = incrementDim (blockIdx, gridDim);
        /* If we didn't overflow blockIdx, reset threadIdx. */
        if (!cuda_coord_is_special (ret_b))
          {
            ret_t.x = 0;
            ret_t.y = 0;
            ret_t.z = 0;
          }
      }

    return std::make_pair (ret_b, ret_t);
  }

public:
  explicit cuda_thread_info (const char *filter_string)
      : cuda_info{ filter_string }, m_expected{}, m_prev_pc{ 0 },
        m_prev_block_idx{ CUDA_INVALID, CUDA_INVALID, CUDA_INVALID },
        m_prev_thread_idx{ CUDA_INVALID, CUDA_INVALID, CUDA_INVALID }
  {
    get_user_print_options (&m_opts);
  }

  bool
  ignore_coord (const cuda_coords &coords) override
  {
    const auto &p = coords.physical ();
    const auto &l = coords.logical ();

    auto kernel = kernels_find_kernel_by_grid_id (p.dev (), l.gridId ());
    m_pc = cuda_state::lane_get_pc (p.dev (), p.sm (), p.wp (),
                                            p.ln ());

    /* Get the sal for this iteration. */
    auto sal_it = m_pc_to_sal.find (m_pc);
    /* Add this sal to the map if we haven't already done so */
    if (sal_it == m_pc_to_sal.end ())
      {
        auto this_sal = m_pc_to_sal.emplace (
            std::make_pair (m_pc, find_pc_line (m_pc, 0)));
        gdb_assert (this_sal.second);
        sal_it = this_sal.first;
      }
    /* Grab the stuff we need from the iterator */
    m_line = sal_it->second.line;
    m_symtab = sal_it->second.symtab;

    /* See if this is a new range */
    bool break_of_contiguity = false;
    if (!m_underlying.empty ())
      {
        break_of_contiguity = (l != m_expected.logical ());
        if (!break_of_contiguity)
          {
            if (m_opts.addressprint)
              {
                break_of_contiguity = (m_pc != m_prev_pc);
              }
            else
              {
                break_of_contiguity = (m_line != m_prev_line);
              }
          }
      }

    /* close the current range */
    if (!m_underlying.empty ()
        && (break_of_contiguity || !cuda_options_coalescing ()))
      {
        m_underlying.back ().finalize_range (m_prev_block_idx,
                                             m_prev_thread_idx);
      }

    /* data for the next iteration */
    m_prev_pc = m_pc;
    m_prev_block_idx = l.blockIdx ();
    m_prev_thread_idx = l.threadIdx ();
    /* Advance to the next thread */
    auto nextIdx = incrementThread (m_prev_block_idx, m_prev_thread_idx,
                                    kernel_get_grid_dim (kernel),
                                    kernel_get_block_dim (kernel));
    m_expected
        = cuda_coords{ CUDA_WILDCARD,     CUDA_WILDCARD, CUDA_WILDCARD,
                       CUDA_WILDCARD,     l.kernelId (), CUDA_WILDCARD,
                       CUDA_WILDCARD_DIM, nextIdx.first, nextIdx.second };

    /* Check to see if we need to add this range. */
    if (m_underlying.empty () || break_of_contiguity
        || !cuda_options_coalescing ())
      {
        /* Add it */
        return false;
      }
    /* Don't add this range - return true to ignore */
    else
      {
        m_underlying.back ().add_to_range (coords
                                           == cuda_current_focus::get ());
        return true;
      }
  }

  void
  handle_emplaced_coord (const cuda_coords &coords) override
  {
    /* The thread has been constructed and added at this point.
       Set the stuff we didn't have access to before and prepare for
       next iteration. */
    gdb_assert (!m_underlying.empty ());
    m_underlying.back ().set_pc (m_pc);
    m_underlying.back ().set_filename (m_symtab);
    m_underlying.back ().set_line (m_line);
  }

  void
  handle_post_populate () override
  {
    /* At this point, the list has been created. Let's finalize the last
     * elment. */
    if (!m_underlying.empty ())
      m_underlying.back ().finalize_range (m_prev_block_idx,
                                           m_prev_thread_idx);
  }
};

void
info_cuda_threads_command (const char *arg)
{
  struct
  {
    size_t current, kernel, start_block_idx, start_thread_idx, end_block_idx,
        end_thread_idx, count, pc, device, sm, wp, ln, filename, line;
  } width;

  /* column headers */
  const std::string header_current{ " " };
  const std::string header_kernel{ "Kernel" };
  const std::string header_start_block_idx{ "BlockIdx" };
  const std::string header_start_thread_idx{ "ThreadIdx" };
  const std::string header_end_block_idx{ "To BlockIdx" };
  const std::string header_end_thread_idx{ "To ThreadIdx" };
  const std::string header_count{ "Count" };
  const std::string header_pc{ "PC" };
  const std::string header_device{ "Dev" };
  const std::string header_sm{ "SM" };
  const std::string header_warp{ "Wp" };
  const std::string header_lane{ "Ln" };
  const std::string header_filename{ "Filename" };
  const std::string header_line{ "Line" };
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_thread_info threads{ arg };

  /* output message if the list is empty */
  if (threads.size () == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _ ("No CUDA threads.\n"));
      return;
    }

  /* column widths */
  width.current = header_current.length ();
  width.kernel = header_kernel.length ();
  width.start_block_idx = header_start_block_idx.length ();
  width.start_thread_idx = header_start_thread_idx.length ();
  width.end_block_idx = header_end_block_idx.length ();
  width.end_thread_idx = header_end_thread_idx.length ();
  width.count = header_count.length ();
  width.pc = header_pc.length ();
  width.device = header_device.length ();
  width.sm = header_sm.length ();
  width.wp = header_warp.length ();
  width.ln = header_lane.length ();
  width.filename = header_filename.length ();
  width.line = header_line.length ();

  width.line = std::max (width.line, (size_t)5);
  for (const auto &t : threads)
    {
      width.pc = std::max (width.pc, t.pc ().length ());
      width.start_block_idx
          = std::max (width.start_block_idx, t.start_block_idx ().length ());
      width.start_thread_idx
          = std::max (width.start_thread_idx, t.start_thread_idx ().length ());
      width.end_block_idx
          = std::max (width.end_block_idx, t.end_block_idx ().length ());
      width.end_thread_idx
          = std::max (width.end_thread_idx, t.end_thread_idx ().length ());
      width.filename = std::max (width.filename, t.filename ().length ());
    }

  /* print table header ('kernel' is only present in MI output) */
  size_t num_columns;

  bool coalescing = cuda_options_coalescing ();

  /* Table/row names are based on if we're coalescing or not (for backward
   * compatibilty with Eclipse, etc) */
  const char *table_name = coalescing ? "CoalescedInfoCudaThreadsTable"
                                      : "UncoalescedInfoCudaThreadsTable";
  const char *row_name = coalescing ? "CoalescedInfoCudaThreadsRow"
                                    : "UncoalescedInfoCudaThreadsRow";

  if (coalescing)
    num_columns = uiout->is_mi_like_p () ? 10 : 9;
  else
    num_columns = uiout->is_mi_like_p () ? 11 : 10;
  ui_out_emit_table table_emitter (uiout, num_columns, threads.size (),
                                   table_name);
  uiout->table_header (width.current, ui_right, "current", header_current);
  if (uiout->is_mi_like_p ())
    uiout->table_header (width.kernel, ui_right, "kernel", header_kernel);
  if (coalescing)
    {
      uiout->table_header (width.start_block_idx, ui_right, "from_blockIdx",
                           header_start_block_idx);
      uiout->table_header (width.start_thread_idx, ui_right, "from_threadIdx",
                           header_start_thread_idx);
      uiout->table_header (width.end_block_idx, ui_right, "to_blockIdx",
                           header_end_block_idx);
      uiout->table_header (width.end_thread_idx, ui_right, "to_threadIdx",
                           header_end_thread_idx);
      uiout->table_header (width.count, ui_right, "count", header_count);
    }
  else
    {
      uiout->table_header (width.start_block_idx, ui_right, "blockIdx",
                           header_start_block_idx);
      uiout->table_header (width.start_thread_idx, ui_right, "threadIdx",
                           header_start_thread_idx);
    }
  uiout->table_header (width.pc, ui_right, "virtual_pc", header_pc);
  if (!coalescing)
    {
      uiout->table_header (width.device, ui_right, "device", header_device);
      uiout->table_header (width.sm, ui_right, "sm", header_sm);
      uiout->table_header (width.wp, ui_right, "warp", header_warp);
      uiout->table_header (width.ln, ui_right, "lane", header_lane);
    }
  uiout->table_header (width.filename, ui_right, "filename", header_filename);
  uiout->table_header (width.line, ui_right, "line", header_line);
  uiout->table_body ();

  /* print table rows */
  uint64_t kernel_id = ~0ULL;
  for (const auto &t : threads)
    {
      if (!uiout->is_mi_like_p () && t.kernel_id () != kernel_id)
        {
          /* row are grouped per kernel only in CLI output */
          uiout->message ("Kernel %llu\n", (unsigned long long)t.kernel_id ()),
              kernel_id = t.kernel_id ();
        }

      ui_out_emit_tuple tuple_emitter (uiout, row_name);
      uiout->field_string ("current", t.current () ? "*" : " ");
      if (uiout->is_mi_like_p ())
        uiout->field_signed ("kernel", t.kernel_id ());
      if (coalescing)
        {
          uiout->field_string ("from_blockIdx", t.start_block_idx ());
          uiout->field_string ("from_threadIdx", t.start_thread_idx ());
          uiout->field_string ("to_blockIdx", t.end_block_idx ());
          uiout->field_string ("to_threadIdx", t.end_thread_idx ());
          uiout->field_signed ("count", t.count ());
        }
      else
        {
          uiout->field_string ("blockIdx", t.start_block_idx ());
          uiout->field_string ("threadIdx", t.start_thread_idx ());
        }
      uiout->field_string ("virtual_pc", t.pc ());
      if (!coalescing)
        {
          uiout->field_signed ("device", t.device ());
          uiout->field_signed ("sm", t.sm ());
          uiout->field_signed ("warp", t.warp ());
          uiout->field_signed ("lane", t.lane ());
        }
      uiout->field_string ("filename", t.filename ());
      uiout->field_signed ("line", t.line ());
      uiout->text ("\n");
    }

  gdb_flush (gdb_stdout);
}

class cuda_launch_trace
{
private:
  bool m_current;
  uint32_t m_level;
  uint64_t m_kernel_id;
  uint32_t m_device;
  int64_t m_grid_id;
  std::string m_status;
  std::string m_grid_dim;
  std::string m_block_dim;
  std::string m_invocation;

public:
  /* Static members */
  constexpr static cuda_coord_set_type iterator_type
      = cuda_coord_set_type::kernels;
  constexpr static underlying_type_t<cuda_coord_set_mask_t> iterator_mask
      = select_all;
  constexpr static cuda_coord_compare_type compare_type = cuda_coord_compare_type::logical;
  constexpr static command_t parser_command = CMD_FILTER_KERNEL;
  static cuda_filters
  default_filter ()
  {
    /* Default to current kernel */
    cuda_coords c{ CUDA_WILDCARD,     CUDA_WILDCARD,     CUDA_WILDCARD,
                   CUDA_WILDCARD,     CUDA_CURRENT,      CUDA_WILDCARD,
                   CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM };
    return cuda_filters{ c };
  }
  static void
  check_invalid_filter (const cuda_filters &filter)
  {
  }

  /* Non-static members */
  explicit cuda_launch_trace (const cuda_coords &coords)
  {
    error (_ ("cuda_launch_trace cannot be constructed via coords!"));
  }

  explicit cuda_launch_trace (kernel_t kernel, size_t level)
  {
    gdb_assert (kernel);
    m_current = (kernel == cuda_current_focus::get ().logical ().kernel ());
    m_level = level;
    m_kernel_id = kernel_get_id (kernel);
    m_device = kernel_get_dev_id (kernel);
    m_grid_id = kernel_get_grid_id (kernel);
    m_status = std::string{ status_string[kernel_get_status (kernel)] };
    m_grid_dim = dim3_to_string (kernel_get_grid_dim (kernel));
    m_block_dim = dim3_to_string (kernel_get_block_dim (kernel));
    m_invocation = invocation_to_string (kernel_get_name (kernel),
                                         kernel_get_args (kernel));
  }

  /* Getters */
  bool
  current () const noexcept
  {
    return m_current;
  }
  uint32_t
  level () const noexcept
  {
    return m_level;
  }
  uint64_t
  kernel () const noexcept
  {
    return m_kernel_id;
  }
  uint32_t
  device () const noexcept
  {
    return m_device;
  }
  int64_t
  grid () const noexcept
  {
    return m_grid_id;
  }
  const std::string &
  status () const noexcept
  {
    return m_status;
  }
  const std::string &
  grid_dim () const noexcept
  {
    return m_grid_dim;
  }
  const std::string &
  block_dim () const noexcept
  {
    return m_block_dim;
  }
  const std::string &
  invocation () const noexcept
  {
    return m_invocation;
  }

  /* Init other destructors/constructors */
  ~cuda_launch_trace () = default;
  cuda_launch_trace () = delete;
  cuda_launch_trace (const cuda_launch_trace &) = default;
  cuda_launch_trace (cuda_launch_trace &&) = default;
  cuda_launch_trace &operator= (const cuda_launch_trace &) = default;
  cuda_launch_trace &operator= (cuda_launch_trace &&) = default;
};

/* Special handling for lists of cuda_launch_trace */
class cuda_launch_trace_info final : public cuda_info<cuda_launch_trace>
{
public:
  explicit cuda_launch_trace_info (const char *filter_string)
      : cuda_info{ filter_string }
  {
  }

  /* FIXME: We don't have a proper cuda iterator for kernel launch tracing...
   */
  bool
  ignore_coord (const cuda_coords &coords) override
  {
    /* Add each kernel in the chain to the list */
    auto kernel
        = kernels_find_kernel_by_kernel_id (coords.logical ().kernelId ());
    if (!kernel)
      error ("Incorrect kernel specified or the focus is not set on a kernel");

    size_t level = 0;
    while (kernel)
      {
        m_underlying.emplace_back (kernel, level);
        kernel = kernel_get_parent (kernel);
        ++level;
      }

    /* We fully constructed the list for every kernel in the chain at this
     * point. */
    return true;
  }
};

void
info_cuda_launch_trace_command (const char *arg)
{
  struct
  {
    size_t current, level, kernel, device, grid, status, invocation, grid_dim,
        block_dim;
  } width;

  /* column headers */
  const std::string header_current{ " " };
  const std::string header_level{ "Lvl" };
  const std::string header_kernel{ "Kernel" };
  const std::string header_device{ "Dev" };
  const std::string header_grid{ "Grid" };
  const std::string header_status{ "Status" };
  const std::string header_grid_dim{ "GridDim" };
  const std::string header_block_dim{ "BlockDim" };
  const std::string header_invocation{ "Invocation" };
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_launch_trace_info kernels{ arg };

  /* output message if the list is empty */
  if (kernels.size () == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _ ("No CUDA kernels.\n"));
      return;
    }

  /* column widths */
  width.current = header_current.length ();
  width.level = header_level.length ();
  width.kernel = header_kernel.length ();
  width.device = header_device.length ();
  width.grid = header_grid.length ();
  width.status = header_status.length ();
  width.invocation = header_invocation.length ();
  width.grid_dim = header_grid_dim.length ();
  width.block_dim = header_block_dim.length ();

  for (const auto &k : kernels)
    {
      width.status = std::max (width.status, k.status ().length ());
      width.invocation
          = std::max (width.invocation, k.invocation ().length ());
      width.grid_dim = std::max (width.grid_dim, k.grid_dim ().length ());
      width.block_dim = std::max (width.block_dim, k.block_dim ().length ());
    }

  /* print table header */
  ui_out_emit_table table_emitter (uiout, 9, kernels.size (),
                                   "InfoCudaLaunchTraceTable");
  uiout->table_header (width.current, ui_right, "current", header_current);
  uiout->table_header (width.level, ui_left, "level", header_level);
  uiout->table_header (width.kernel, ui_right, "kernel", header_kernel);
  uiout->table_header (width.device, ui_right, "device", header_device);
  uiout->table_header (width.grid, ui_right, "grid", header_grid);
  uiout->table_header (width.status, ui_right, "status", header_status);
  uiout->table_header (width.grid_dim, ui_right, "gridDim", header_grid_dim);
  uiout->table_header (width.block_dim, ui_right, "blockDim",
                       header_block_dim);
  uiout->table_header (width.invocation, ui_left, "invocation",
                       header_invocation);
  uiout->table_body ();

  /* print table rows */
  for (const auto &k : kernels)
    {
      ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaLaunchTraceRow");
      uiout->field_string ("current", k.current () ? "*" : " ");
      uiout->text ("#");
      uiout->field_signed ("level", k.level ());
      uiout->field_signed ("kernel", k.kernel ());
      uiout->field_signed ("device", k.device ());
      uiout->field_signed ("grid", k.grid ());
      uiout->field_string ("status", k.status ());
      uiout->field_string ("gridDim", k.grid_dim ());
      uiout->field_string ("blockDim", k.block_dim ());
      uiout->field_string ("invocation", k.invocation ());
      uiout->text ("\n");
    }

  gdb_flush (gdb_stdout);
}

/* Special handling for lists of cuda_launch_trace for child tracing
   This is nearly identical to cuda_launch_trace, we just build for
   children instead of parents. */
class cuda_launch_children_info final : public cuda_info<cuda_launch_trace>
{
public:
  explicit cuda_launch_children_info (const char *filter_string)
      : cuda_info{ filter_string }
  {
  }

  /* FIXME: We don't have a proper cuda iterator for kernel launch tracing...
   */
  bool
  ignore_coord (const cuda_coords &coords) override
  {
    /* Add each kernel in the chain to the list */
    auto kernel
        = kernels_find_kernel_by_kernel_id (coords.logical ().kernelId ());
    if (!kernel)
      error ("Incorrect kernel specified or the focus is not set on a kernel");

    size_t level = 0;
    while (kernel)
      {
        m_underlying.emplace_back (kernel, level);
        kernel = kernel_get_sibling (kernel);
        ++level;
      }

    /* We fully constructed the list for every kernel in the chain at this
     * point. */
    return true;
  }
};

void
info_cuda_launch_children_command (const char *arg)
{
  struct
  {
    size_t current, kernel, device, grid, status, grid_dim, block_dim,
        invocation;
  } width;

  /* column headers */
  const std::string header_current{ " " };
  const std::string header_kernel{ "Kernel" };
  const std::string header_device{ "Dev" };
  const std::string header_grid{ "Grid" };
  const std::string header_status{ "Status" };
  const std::string header_grid_dim{ "GridDim" };
  const std::string header_block_dim{ "BlockDim" };
  const std::string header_invocation{ "Invocation" };
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_launch_children_info kernels{ arg };

  /* output message if the list is empty */
  if (kernels.size () == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _ ("No CUDA kernels.\n"));
      return;
    }

  /* column widths */
  width.current = header_current.length ();
  width.kernel = header_kernel.length ();
  width.device = header_device.length ();
  width.grid = header_grid.length ();
  width.status = header_status.length ();
  width.invocation = header_invocation.length ();
  width.grid_dim = header_grid_dim.length ();
  width.block_dim = header_block_dim.length ();

  for (const auto &k : kernels)
    {
      width.status = std::max (width.status, k.status ().length ());
      width.invocation
          = std::max (width.invocation, k.invocation ().length ());
      width.grid_dim = std::max (width.grid_dim, k.grid_dim ().length ());
      width.block_dim = std::max (width.block_dim, k.block_dim ().length ());
    }

  /* print table header */
  ui_out_emit_table table_emitter (uiout, 8, kernels.size (),
                                   "InfoCudaLaunchChildrenTable");
  uiout->table_header (width.current, ui_right, "current", header_current);
  uiout->table_header (width.kernel, ui_right, "kernel", header_kernel);
  uiout->table_header (width.device, ui_right, "device", header_device);
  uiout->table_header (width.grid, ui_right, "grid", header_grid);
  uiout->table_header (width.status, ui_right, "status", header_status);
  uiout->table_header (width.grid_dim, ui_right, "gridDim", header_grid_dim);
  uiout->table_header (width.block_dim, ui_right, "blockDim",
                       header_block_dim);
  uiout->table_header (width.invocation, ui_left, "invocation",
                       header_invocation);
  uiout->table_body ();

  /* print table rows */
  for (const auto &k : kernels)
    {
      ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaLaunchChildrenRow");
      uiout->field_string ("current", k.current () ? "*" : " ");
      uiout->field_signed ("kernel", k.kernel ());
      uiout->field_signed ("device", k.device ());
      uiout->field_signed ("grid", k.grid ());
      uiout->field_string ("status", k.status ());
      uiout->field_string ("gridDim", k.grid_dim ());
      uiout->field_string ("blockDim", k.block_dim ());
      uiout->field_string ("invocation", k.invocation ());
      uiout->text ("\n");
    }

  gdb_flush (gdb_stdout);
}

class cuda_context_obj
{
private:
  bool m_current;
  std::string m_context;
  uint32_t m_device;
  std::string m_state;

public:
  /* Static members */
  constexpr static cuda_coord_set_type iterator_type
      = cuda_coord_set_type::devices;
  constexpr static underlying_type_t<cuda_coord_set_mask_t> iterator_mask
      = select_all;
  constexpr static cuda_coord_compare_type compare_type = cuda_coord_compare_type::logical;
  constexpr static command_t parser_command = CMD_FILTER;
  static cuda_filters
  default_filter ()
  {
    return cuda_filters{ cuda_coords::wild () };
  }
  static void
  check_invalid_filter (const cuda_filters &filter)
  {
    /* Only filter allowed is device */
    /* FIXME: This wasn't doing anything. Previous implementation
     * would compare CUDA_INVALID and CUDA_WILDCARD as equal to any
     * coordinate! */
  }

  /* Non-static members */
  explicit cuda_context_obj (const cuda_coords &coords)
  {
    error (_ ("cuda_context_obj cannot be constructed via coords!"));
  }

  cuda_context_obj (cuda_context* context)
  {
    gdb_assert (context);
    m_current = (context == cuda_state::current_context ());
    m_context = to_hex (context->id ());
    m_device = context->dev_id ();
    m_state = cuda_state::is_context_active (context)
                  ? std::string{ "active" }
                  : std::string{ "inactive" };
  }

  /* Getters */
  bool
  current () const noexcept
  {
    return m_current;
  }
  const std::string &
  context () const noexcept
  {
    return m_context;
  }
  uint32_t
  device () const noexcept
  {
    return m_device;
  }
  const std::string &
  state () const noexcept
  {
    return m_state;
  }

  /* Init other destructors/constructors */
  ~cuda_context_obj () = default;
  cuda_context_obj () = delete;
  cuda_context_obj (const cuda_context_obj &) = default;
  cuda_context_obj (cuda_context_obj &&) = default;
  cuda_context_obj &operator= (const cuda_context_obj &) = default;
  cuda_context_obj &operator= (cuda_context_obj &&) = default;
};

/* Special handling for lists of cuda_context_obj */
class cuda_context_info final : public cuda_info<cuda_context_obj>
{
public:
  explicit cuda_context_info (const char *filter_string)
      : cuda_info{ filter_string }
  {
  }

  /* FIXME: We don't have a proper cuda iterator for contexts... */
  bool
  ignore_coord (const cuda_coords &coords) override
  {
    /* Check to see if we should ignore this device */
    if (!cuda_coord_equals (m_filter.coords ().physical ().dev (),
                            coords.physical ().dev ()))
      return true;

    /* Find all active contexts on this device */
    for (auto& iter : cuda_state::contexts ())
      if (iter.second->dev_id () == coords.physical ().dev ())
	m_underlying.emplace_back (cuda_context_obj (iter.second.get ()));

    /* We fully constructed the list for every context on this device at this
     * point. */
    return true;
  }
};

void
info_cuda_contexts_command (const char *arg)
{
  struct
  {
    size_t current, context, device, state;
  } width;

  /* column header */
  const std::string header_current{ " " };
  const std::string header_context{ "Context" };
  const std::string header_device{ "Dev" };
  const std::string header_state{ "State" };
  struct ui_out *uiout = current_uiout;

  /* get the information */
  cuda_context_info contexts{ arg };

  /* output message if the list is empty */
  if (contexts.size () == 0 && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL, _ ("No CUDA contexts.\n"));
      return;
    }

  /* column widths */
  width.current = header_current.length ();
  width.context = header_context.length ();
  width.device = header_device.length ();
  width.state = header_state.length ();

  for (const auto &c : contexts)
    {
      width.context = std::max (width.context, c.context ().length ());
      width.state = std::max (width.state, c.state ().length ());
    }

  /* print table header */
  ui_out_emit_table table_emitter (uiout, 4, contexts.size (),
                                   "InfoCudaContextsTable");
  uiout->table_header (width.current, ui_right, "current", header_current);
  uiout->table_header (width.context, ui_right, "context", header_context);
  uiout->table_header (width.device, ui_right, "device", header_device);
  uiout->table_header (width.state, ui_right, "state", header_state);
  uiout->table_body ();

  /* print table rows */
  for (const auto &c : contexts)
    {
      ui_out_emit_tuple tuple_emitter (uiout, "InfoCudaContextsRow");
      uiout->field_string ("current", c.current () ? "*" : " ");
      uiout->field_string ("context", c.context ());
      uiout->field_signed ("device", c.device ());
      uiout->field_string ("state", c.state ());
      uiout->text ("\n");
    }

  gdb_flush (gdb_stdout);
}

static struct symbol *
msymbol_to_symbol (struct bound_minimal_symbol &bmsym)
{
  struct obj_section *objsection = bmsym.obj_section ();
  struct objfile *objfile = objsection ? objsection->objfile : NULL;

  if (!objfile)
    return NULL;

  for (compunit_symtab *cu : objfile->compunits ())
    {
      for (symtab *s : cu->filetabs ())
        {
          const struct blockvector *bv = s->compunit ()->blockvector ();
          const struct block *block = bv->static_block ();
          struct symbol *sym = block_lookup_symbol (
              block, bmsym.minsym->print_name (),
              symbol_name_match_type::SEARCH_NAME, VAR_DOMAIN);
          if (sym)
            return fixup_symbol_section (sym, objfile);
        }
    }

  return NULL;
}

static void
print_managed_msymbol (struct ui_file *stb, struct bound_minimal_symbol &bmsym)
{
  struct value_print_options opts;
  struct symbol *sym = msymbol_to_symbol (bmsym);
  struct value *val = sym ? read_var_value (sym, NULL, NULL) : NULL;

  if (!val)
    {
      gdb_printf (stb, "%s\n", bmsym.minsym->print_name ());
      return;
    }
  if (!cuda_current_focus::isDevice ())
    val = value_ind (val);

  gdb_printf (stb, "%s = ", bmsym.minsym->print_name ());

  try
    {
      get_user_print_options (&opts);
      common_val_print (val, stb, 0, &opts, current_language);
    }
  catch (const gdb_exception_error &e)
    {
      gdb_printf (stb, "<value optimized out>\n");
      return;
    }

  gdb_printf (stb, "\n");
}

void
info_cuda_managed_command (const char *arg)
{
  const struct bfd_arch_info *current_bfd_arch
      = gdbarch_bfd_arch_info (get_current_arch ());

  if (cuda_current_focus::isDevice ())
    gdb_printf (gdb_stdout,
                      "Static managed variables on device %d are:\n",
                      cuda_current_focus::get ().physical ().dev ());
  else
    gdb_printf (gdb_stdout, "Static managed variables on host are:\n");

  for (objfile *obj : current_program_space->objfiles ())
    {
      /* Skip objects which architecture different than current */
      if (gdbarch_bfd_arch_info (obj->arch ()) != current_bfd_arch)
        continue;

      for (minimal_symbol *msym : obj->msymbols ())
        {
          /* Create a bound minsym based on the msym and the obj file. */
          struct bound_minimal_symbol minsym
          {
            msym, obj
          };
          if (!cuda_managed_msymbol_p (minsym))
            continue;
          print_managed_msymbol (gdb_stdout, minsym);
        }
    }
}

/*
  Very similar to exception info printing.
*/
void
info_cuda_line_command (const char *args)
{
  if (args == NULL)
    error_no_arg (_ ("pc value"));

  CORE_ADDR pc = parse_and_eval_address (args);

  struct obj_section *section = find_pc_overlay (pc);

  struct cuda_debug_inline_info *inline_info = NULL;
  struct symtab_and_line sal
      = find_pc_sect_line (pc, section, 0, &inline_info);

  /* Possibly called from CLI mode, make sure we get the right uiout,
     not the current one. */
  struct ui_out *uiout = top_level_interpreter ()->interp_ui_out ();

  uiout->text ("PC ");
  uiout->field_fmt ("pc", "0x%lx", pc);

  const char *filename
      = sal.symtab ? symtab_to_filename_for_display (sal.symtab) : NULL;

  if (filename)
    {
      uiout->text (" is at ");
      uiout->field_string ("filename", filename);
      uiout->text (":");
      uiout->field_signed ("line", sal.line);

      if (inline_info)
        {
          gdb::unique_xmalloc_ptr<char> demangled = language_demangle (
              language_def (current_language->la_language),
              inline_info->function, DMGL_ANSI);
          uiout->text (" inlined from ");
          uiout->field_string ("inline_filename",
                               lbasename (inline_info->filename));
          uiout->text (":");
          uiout->field_signed ("inline_line", inline_info->line);
          uiout->text (" by ");

          uiout->field_string ("inline_function", demangled.get ()
                                                      ? demangled.get ()
                                                      : inline_info->function);
        }
    }
  else
    {
      uiout->text (" has no linenumber information");
    }

  uiout->text ("\n");
}

void
run_info_cuda_command (void (*command) (const char *), const char *arg)
{
  /* Create cleanup object to restore the original focus and ELF images */
  cuda_focus_restore r;

  /* Execute the proper info cuda command */
  command (arg);
}

static struct
{
  const char *name;
  void (*func) (const char *);
  const char *help;
} cuda_info_subcommands[] = {
  { "devices", info_cuda_devices_command,
    "information about all the devices" },
  { "sms", info_cuda_sms_command,
    "information about all the SMs in the current device" },
  { "warps", info_cuda_warps_command,
    "information about all the warps in the current SM" },
  { "lanes", info_cuda_lanes_command,
    "information about all the lanes in the current warp" },
  { "kernels", info_cuda_kernels_command,
    "information about all the active kernels" },
  { "contexts", info_cuda_contexts_command,
    "information about all the contexts" },
  { "blocks", info_cuda_blocks_command,
    "information about all the active blocks in the current kernel" },
  { "threads", info_cuda_threads_command,
    "information about all the active threads in the current kernel" },
  { "launch trace", info_cuda_launch_trace_command,
    "information about the parent kernels of the kernel in focus" },
  { "launch children", info_cuda_launch_children_command,
    "information about the kernels launched by the kernels in focus" },
  { "managed", info_cuda_managed_command,
    "information about global managed variables" },
  { "line", info_cuda_line_command,
    "information about the filename and linenumber for a given $pc" },
  { NULL, NULL, NULL },
};

static int
cuda_info_subcommands_max_name_length (void)
{
  int cnt;
  size_t rc;

  for (cnt = 0, rc = 0; cuda_info_subcommands[cnt].name; cnt++)
    rc = std::max (rc, strlen (cuda_info_subcommands[cnt].name));
  return (int)rc;
}

static void
info_cuda_command (const char *arg, int from_tty)
{
  int cnt;
  const char *argument;
  void (*command) (const char *) = NULL;

  if (!arg)
    error (_ ("Missing option."));

  /* Sanity check and save which command (with correct argument) to invoke. */
  for (cnt = 0; cuda_info_subcommands[cnt].name; cnt++)
    if (strstr (arg, cuda_info_subcommands[cnt].name) == arg)
      {
        command = cuda_info_subcommands[cnt].func;
        argument = arg + strlen (cuda_info_subcommands[cnt].name);
        break;
      }

  if (!command)
    error (_ ("Unrecognized option: '%s'."), arg);

  run_info_cuda_command (command, argument);
}

struct cmd_list_element *cudalist;

void
cuda_command_switch (const char *switch_string)
{
  /* Build the parsed command - use either current or wildcard for unspecified
   * CuDim3 coords */
  cuda_parser_result_t *command;
  cuda_parser (switch_string, CMD_SWITCH, &command,
               cuda_current_focus::isDevice () ? CUDA_CURRENT : CUDA_WILDCARD);
  gdb_assert (command);

  /* Create a filter from this request. Use wildcard coords if unspecified. */
  cuda_filters filter{ cuda_coords::wild (), command, true };

  /* Determine the origin for the solution.
   * Either current focus or wildcard if there is none. */
  cuda_coords closest = cuda_current_focus::get ();
  if (!cuda_current_focus::isDevice ())
    closest = cuda_coords::wild ();

  /* Sort by physical or logical coordinates. Physical coordinates have
   * priority. */
  bool compare_logical = true;
  request_t *request = command->requests;
  for (auto i = 0; i < command->num_requests; ++i, ++request)
    {
      if (request->type == FILTER_TYPE_DEVICE
          || request->type == FILTER_TYPE_SM
          || request->type == FILTER_TYPE_WARP
          || request->type == FILTER_TYPE_LANE)
        {
          compare_logical = false;
          break;
        }
    }

  /* Find the closest match */
  cuda_coords res;
  if (compare_logical)
    {
      cuda_coord_set<cuda_coord_set_type::threads, select_valid,
                    cuda_coord_compare_type::logical>
          solution{ filter.coords (), closest };
      if (solution.size ())
        res = *solution.begin ();
    }
  else
    {
      cuda_coord_set<cuda_coord_set_type::threads, select_valid,
                    cuda_coord_compare_type::physical>
          solution{ filter.coords (), closest };
      if (solution.size ())
        res = *solution.begin ();
    }

  /* Do the actual switch if possible */
  if (!res.valid ())
    error (_ ("Invalid coordinates requested. CUDA focus unchanged."));
  else if (cuda_current_focus::isDevice ()
	   && (cuda_current_focus::get () == res))
    {
      if (current_uiout->is_mi_like_p ())
        cuda_current_focus::printFocus (false);
      else
	current_uiout->field_string ("CudaFocus",
				     _ ("CUDA focus unchanged.\n"));
    }
  else
    {
      cuda_current_focus::set (res);
      switch_to_cuda_thread (cuda_current_focus::get ());
      cuda_current_focus::printFocus (true);
      print_stack_frame (get_selected_frame (NULL), 0, SRC_LINE, 1);
      do_displays ();
    }
}

void
cuda_command_query (const char *query_string)
{
  /* Bail out if focus not set on a CUDA device */
  if (!cuda_current_focus::isDevice ())
    {
      if (!current_uiout->is_mi_like_p ())
	current_uiout->field_string (
	    "CudaFocus", _ ("Focus is not set on any active CUDA kernel.\n"));
      return;
    }

  /* Build the coordinates based on the user request */
  cuda_parser_result_t *result;
  cuda_parser (query_string, CMD_QUERY, &result, CUDA_CURRENT);
  cuda_coords wished;
  cuda_filters filter{ wished, result };

  /* Print the found coordinates */
  auto string = filter.coords ().to_string ();
  if (string.empty ())
    error (_ ("No CUDA coordinates found."));
  else
    {
      if (current_uiout->is_mi_like_p ())
	filter.coords ().emit_mi_output ();
      else
        current_uiout->field_fmt ("CudaFocusQuery", "%s\n", string.c_str ());
      gdb_flush (gdb_stdout);
    }
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
  input = (char *)xmalloc (len1 + 1 + len2 + 1);
  strncpy (input, first_word, len1);
  input[len1] = ' ';
  strncpy (input + len1 + 1, args, len2);
  input[len1 + 1 + len2] = '\0';

  /* Dispatch to the right handler based on the command type */
  cuda_parser (input, (command_t)(CMD_QUERY | CMD_SWITCH), &result,
               CUDA_WILDCARD);
  switch (result->command)
    {
    case CMD_QUERY:
      cuda_command_query (input);
      break;
    case CMD_SWITCH:
      cuda_command_switch (input);
      break;
    default:
      error (_ ("Unrecognized argument(s)."));
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
    error (_ ("Missing argument(s)."));
}

static char cuda_info_cmd_help_str[1024];

/* Prepare help for info cuda command */
static void
cuda_build_info_cuda_help_message (void)
{
  char *ptr = cuda_info_cmd_help_str;
  int size = sizeof (cuda_info_cmd_help_str);
  int rc, cnt;

  rc = snprintf (ptr, size,
                 _ ("Print informations about the current CUDA activities. "
                    "Available options:\n"));
  ptr += rc;
  size -= rc;
  for (cnt = 0; cuda_info_subcommands[cnt].name; cnt++)
    {
      rc = snprintf (ptr, size, " %*s : %s\n",
                     cuda_info_subcommands_max_name_length (),
                     cuda_info_subcommands[cnt].name,
                     _ (cuda_info_subcommands[cnt].help));
      if (rc <= 0)
        break;
      ptr += rc;
      size -= rc;
    }
}

static void
cuda_info_command_completer (struct cmd_list_element *ignore,
                             completion_tracker &tracker, const char *text,
                             const char *word)
{
  int cnt;
  long offset;
  const char *name;

  offset = (long)word - (long)text;

  for (cnt = 0; cuda_info_subcommands[cnt].name; cnt++)
    {
      name = cuda_info_subcommands[cnt].name;
      if (offset >= strlen (name))
        continue;
      if (strstr (name, text) != name)
        continue;
      gdb::unique_xmalloc_ptr<char> p_copy (xstrdup (name + offset));
      tracker.add_completion (std::move (p_copy));
    }
}

void _initialize_cuda_commands ();
void
_initialize_cuda_commands ()
{
  struct cmd_list_element *cmd;

  add_prefix_cmd ("cuda", class_cuda, cuda_command,
                  _ ("Print or select the CUDA focus."), &cudalist, 0,
                  &cmdlist);

  add_cmd ("device", no_class, cuda_device_command,
           _ ("Print or select the current CUDA device."), &cudalist);

  add_cmd ("sm", no_class, cuda_sm_command,
           _ ("Print or select the current CUDA SM."), &cudalist);

  add_cmd ("warp", no_class, cuda_warp_command,
           _ ("Print or select the current CUDA warp."), &cudalist);

  add_cmd ("lane", no_class, cuda_lane_command,
           _ ("Print or select the current CUDA lane."), &cudalist);

  add_cmd ("kernel", no_class, cuda_kernel_command,
           _ ("Print or select the current CUDA kernel."), &cudalist);

  add_cmd ("grid", no_class, cuda_grid_command,
           _ ("Print or select the current CUDA grid."), &cudalist);

  add_cmd ("block", no_class, cuda_block_command,
           _ ("Print or select the current CUDA block."), &cudalist);

  add_cmd ("thread", no_class, cuda_thread_command,
           _ ("Print or select the current CUDA thread."), &cudalist);

  cuda_build_info_cuda_help_message ();
  cmd = add_info ("cuda", info_cuda_command, cuda_info_cmd_help_str);
  set_cmd_completer (cmd, cuda_info_command_completer);
}
