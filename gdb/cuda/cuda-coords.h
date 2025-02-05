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

#ifndef _CUDA_COORDS_H
#define _CUDA_COORDS_H 1

#include "defs.h"

#include "cuda-defs.h"
#include "cudadebugger.h"

#include <string>

class cuda_coords_physical;
class cuda_coords_logical;
class cuda_coords;

typedef enum : uint32_t
{
  CUDA_INVALID = std::numeric_limits<uint32_t>::max (),
  CUDA_WILDCARD = CUDA_INVALID - 1,
  CUDA_CURRENT = CUDA_INVALID - 2,
  CUDA_IGNORE = CUDA_INVALID - 3,
} cuda_coords_special_value_t;

constexpr CuDim3 CUDA_WILDCARD_DIM{ CUDA_WILDCARD, CUDA_WILDCARD,
                                    CUDA_WILDCARD };
constexpr CuDim3 CUDA_INVALID_DIM{ CUDA_INVALID, CUDA_INVALID, CUDA_INVALID };
constexpr CuDim3 CUDA_CURRENT_DIM{ CUDA_CURRENT, CUDA_CURRENT, CUDA_CURRENT };
constexpr CuDim3 CUDA_IGNORE_DIM{ CUDA_IGNORE, CUDA_IGNORE, CUDA_IGNORE };

/* Check if coordinate is special */
template <typename T>
inline bool
cuda_coord_is_special (const T &c)
{
  return ((c == CUDA_INVALID) || (c == CUDA_WILDCARD) || (c == CUDA_CURRENT));
}
template <>
inline bool
cuda_coord_is_special<CuDim3> (const CuDim3 &c)
{
  if (cuda_coord_is_special (c.x))
    return true;
  if (cuda_coord_is_special (c.y))
    return true;
  if (cuda_coord_is_special (c.z))
    return true;
  return false;
}

/* Check if coordinate is ignored */
template <typename T>
inline bool
cuda_coord_is_ignored (const T &c)
{
  return ((c == CUDA_IGNORE) || (cuda_coord_is_special (c)));
}
template <>
inline bool
cuda_coord_is_ignored<CuDim3> (const CuDim3 &c)
{
  if (cuda_coord_is_ignored (c.x))
    return true;
  if (cuda_coord_is_ignored (c.y))
    return true;
  if (cuda_coord_is_ignored (c.z))
    return true;
  return false;
}

/* Replace CUDA_CURRENT with cuda_current_focus */
template <typename T>
inline void
cuda_evaluate_current (T &coord, const T &current)
{
  if (coord == CUDA_CURRENT)
    coord = current;
}
template <>
inline void
cuda_evaluate_current<CuDim3> (CuDim3 &coord, const CuDim3 &current)
{
  if (coord.x == CUDA_CURRENT)
    coord.x = current.x;
  if (coord.y == CUDA_CURRENT)
    coord.y = current.y;
  if (coord.z == CUDA_CURRENT)
    coord.z = current.z;
}

/* Comparison routines that accept wildcards */
template <typename T>
inline bool
cuda_coord_equals (const T &lhs, const T &rhs)
{
  return (((lhs == CUDA_WILDCARD) || (rhs == CUDA_WILDCARD)
           || (lhs == CUDA_IGNORE) || (rhs == CUDA_IGNORE))
          || (lhs == rhs));
}
template <>
inline bool
cuda_coord_equals<CuDim3> (const CuDim3 &lhs, const CuDim3 &rhs)
{
  return cuda_coord_equals (lhs.x, rhs.x) && cuda_coord_equals (lhs.y, rhs.y)
         && cuda_coord_equals (lhs.z, rhs.z);
}

/* Comparison operators for CuDim3 */
inline bool
operator== (const CuDim3 &lhs, const CuDim3 &rhs)
{
  return ((lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z));
}
inline bool
operator!= (const CuDim3 &lhs, const CuDim3 &rhs)
{
  return ((lhs.x != rhs.x) || (lhs.y != rhs.y) || (lhs.z != rhs.z));
}
inline bool
operator< (const CuDim3 &lhs, const CuDim3 &rhs)
{
  /* Want to compare slowest to fastest varying dimensions. */
  /* Check z dimension. */
  if (lhs.z != rhs.z)
    return (lhs.z < rhs.z);
  /* Check y dimension. */
  if (lhs.y != rhs.y)
    return (lhs.y < rhs.y);
  /* Check x dimension. */
  if (lhs.x != rhs.x)
    return (lhs.x < rhs.x);
  /* lhs == rhs case */
  return false;
}

// std::make_signed_t was added in C++14
template< class T >
using make_signed_t = typename std::make_signed<T>::type;

/* Used to compare the distance between two coordinates. If the distance is
 * equal it returns false. Otherwise it returns true and sets the lhs < rhs
 * distance result in res.*/
template <typename T>
inline bool
cuda_coord_distance (bool &res, const T &origin, const T &lhs, const T &rhs)
{
  // Ignore if origin is special or lhs == rhs
  if (cuda_coord_is_special (origin) || (lhs == rhs))
    return false;

  T lhs_dist
      = std::abs ((make_signed_t<T>)lhs - (make_signed_t<T>)origin);
  T rhs_dist
      = std::abs ((make_signed_t<T>)rhs - (make_signed_t<T>)origin);

  // Ignore if the calculated distances match
  if (lhs_dist == rhs_dist)
    return false;

  // We can apply less than operator on these distances
  res = (lhs_dist < rhs_dist);
  return true;
}
template <>
inline bool
cuda_coord_distance<CuDim3> (bool &res, const CuDim3 &origin,
			     const CuDim3 &lhs, const CuDim3 &rhs)
{
  /* Want to compare slowest to fastest varying dimension */
  /* Check z dimension. */
  if (cuda_coord_distance<decltype (origin.z)> (res, origin.z, lhs.z, rhs.z))
    return true;
  /* Check y dimension. */
  if (cuda_coord_distance<decltype (origin.y)> (res, origin.y, lhs.y, rhs.y))
    return true;
  /* Check x dimension. */
  if (cuda_coord_distance<decltype (origin.x)> (res, origin.x, lhs.x, rhs.x))
    return true;
  return false;
}

/*
 * FNV hash for CuDim3
 */
class cudim3_hash
{
private:
  template <class T>
  inline void
  hash_combine (std::size_t &res, const T &val) const
  {
    res = std::hash<T>{}(val) + 0x9e3779b9 + (res << 6) + (res >> 2);
  }

public:
  std::size_t
  operator() (const CuDim3 &dim) const
  {
    std::size_t res = 0;

    hash_combine<decltype (dim.x)> (res, dim.x);
    hash_combine<decltype (dim.y)> (res, dim.y);
    hash_combine<decltype (dim.z)> (res, dim.z);

    return res;
  }
};

/* physical coordinate specification */
class cuda_coords_physical final
{
private:
  uint32_t m_dev;
  uint32_t m_sm;
  uint32_t m_wp;
  uint32_t m_ln;

public:
  cuda_coords_physical ()
      : m_dev{ CUDA_INVALID }, m_sm{ CUDA_INVALID }, m_wp{ CUDA_INVALID },
        m_ln{ CUDA_INVALID }
  {
  }
  cuda_coords_physical (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln);
  cuda_coords_physical (const cuda_coords_physical &) = default;
  cuda_coords_physical (cuda_coords_physical &&) = default;
  ~cuda_coords_physical () = default;
  cuda_coords_physical &operator= (const cuda_coords_physical &) = default;
  cuda_coords_physical &operator= (cuda_coords_physical &&) = default;

  /* Getters */

  uint32_t
  dev () const
  {
    return m_dev;
  }

  uint32_t
  sm () const
  {
    return m_sm;
  }

  uint32_t
  wp () const
  {
    return m_wp;
  }

  uint32_t
  ln () const
  {
    return m_ln;
  }

  /* Methods */

  bool
  isFullyDefined () const
  {
    return !(cuda_coord_is_special (m_dev) || cuda_coord_is_special (m_sm)
             || cuda_coord_is_special (m_wp) || cuda_coord_is_special (m_ln));
  }

  bool isValidOnDevice (const cuda_coords_logical &expected) const;

  /* Operators */
  bool
  operator< (const cuda_coords_physical &coord) const
  {
    /* Check device */
    if (!cuda_coord_equals (m_dev, coord.m_dev))
      return m_dev < coord.m_dev;
    /* Check sm */
    if (!cuda_coord_equals (m_sm, coord.m_sm))
      return m_sm < coord.m_sm;
    /* Check warp */
    if (!cuda_coord_equals (m_wp, coord.m_wp))
      return m_wp < coord.m_wp;
    /* Check lane */
    if (!cuda_coord_equals (m_ln, coord.m_ln))
      return m_ln < coord.m_ln;
    /* a == b case */
    return false;
  }

  bool
  operator== (const cuda_coords_physical &coord) const
  {
    return (cuda_coord_equals (m_dev, coord.m_dev)
            && cuda_coord_equals (m_sm, coord.m_sm)
            && cuda_coord_equals (m_wp, coord.m_wp)
            && cuda_coord_equals (m_ln, coord.m_ln));
  }

  bool
  operator!= (const cuda_coords_physical &coord) const
  {
    return (!cuda_coord_equals (m_dev, coord.m_dev)
            || !cuda_coord_equals (m_sm, coord.m_sm)
            || !cuda_coord_equals (m_wp, coord.m_wp)
            || !cuda_coord_equals (m_ln, coord.m_ln));
  }
};

/* Logical coordinate specification */
class cuda_coords_logical final
{
private:
  uint64_t m_kernelId;
  uint64_t m_gridId;
  CuDim3 m_clusterIdx;
  CuDim3 m_blockIdx;
  CuDim3 m_threadIdx;

public:
  cuda_coords_logical ()
      : m_kernelId{ CUDA_INVALID }, m_gridId{ CUDA_INVALID },
        m_clusterIdx{ CUDA_INVALID_DIM }, m_blockIdx{ CUDA_INVALID_DIM },
        m_threadIdx{ CUDA_INVALID_DIM }
  {
  }
  cuda_coords_logical (uint64_t kernelId, uint64_t gridId, CuDim3 clusterIdx,
                       CuDim3 blockIdx, CuDim3 threadIdx);
  cuda_coords_logical (const cuda_coords_logical &) = default;
  cuda_coords_logical (cuda_coords_logical &&) = default;
  ~cuda_coords_logical () = default;
  cuda_coords_logical &operator= (const cuda_coords_logical &) = default;
  cuda_coords_logical &operator= (cuda_coords_logical &&) = default;

  /* Getters */

  uint64_t
  kernelId () const
  {
    return m_kernelId;
  }

  kernel_t kernel () const;

  uint64_t
  gridId () const
  {
    return m_gridId;
  }

  const CuDim3 &
  clusterIdx () const
  {
    return m_clusterIdx;
  }

  const CuDim3 &
  blockIdx () const
  {
    return m_blockIdx;
  }

  const CuDim3 &
  threadIdx () const
  {
    return m_threadIdx;
  }

  /* Methods */

  bool
  isFullyDefined () const
  {
    return !(cuda_coord_is_special (m_kernelId)
             || cuda_coord_is_special (m_gridId)
             || cuda_coord_is_special (m_clusterIdx)
             || cuda_coord_is_special (m_blockIdx)
             || cuda_coord_is_special (m_threadIdx));
  }

  /* Operators */

  bool
  operator< (const cuda_coords_logical &coord) const
  {
    /* Check kernel */
    if (!cuda_coord_equals (m_kernelId, coord.m_kernelId))
      return (m_kernelId < coord.m_kernelId);
    /* Check gridId */
    if (!cuda_coord_equals (m_gridId, coord.m_gridId))
      return (m_gridId < coord.m_gridId);
    /* Check blockIdx */
    if (!cuda_coord_equals (m_blockIdx, coord.m_blockIdx))
      return (m_blockIdx < coord.m_blockIdx);
    /* Check threadIdx */
    if (!cuda_coord_equals (m_threadIdx, coord.m_threadIdx))
      return (m_threadIdx < coord.m_threadIdx);
    /* a == b case */
    return false;
  }

  bool
  operator== (const cuda_coords_logical &coord) const
  {
    return (cuda_coord_equals (m_kernelId, coord.m_kernelId)
            && cuda_coord_equals (m_gridId, coord.m_gridId)
            && cuda_coord_equals (m_blockIdx, coord.m_blockIdx)
            && cuda_coord_equals (m_threadIdx, coord.m_threadIdx));
  }

  bool
  operator!= (const cuda_coords_logical &coord) const
  {
    return (!cuda_coord_equals (m_kernelId, coord.m_kernelId)
            || !cuda_coord_equals (m_gridId, coord.m_gridId)
            || !cuda_coord_equals (m_blockIdx, coord.m_blockIdx)
            || !cuda_coord_equals (m_threadIdx, coord.m_threadIdx));
  }
};

/* Fully qualified coordinate specification */
class cuda_coords final
{
private:
  bool m_valid;
  /* Order matters - sometimes we need physical coords in order to initialize
   * logical */
  cuda_coords_physical m_physical;
  cuda_coords_logical m_logical;

public:
  cuda_coords () : m_valid{ false }, m_physical{}, m_logical{} {}
  cuda_coords (uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln,
               uint64_t kernelId, uint64_t gridId, CuDim3 clusterIdx,
               CuDim3 blockIdx, CuDim3 threadIdx)
      : m_valid{ false }, m_physical{ dev, sm, wp, ln },
        m_logical{ kernelId, gridId, clusterIdx, blockIdx, threadIdx }
  {
    // Check validity
    isValidOnDevice ();
  }
  cuda_coords (const cuda_coords &) = default;
  cuda_coords (cuda_coords &&) = default;
  ~cuda_coords () = default;
  cuda_coords &operator= (const cuda_coords &) = default;
  cuda_coords &operator= (cuda_coords &&) = default;

  /* Getters */

  /*
   * We only want to present const access to the underlying coords
   * This allows us to avoid having to re-check for validity every
   * time a user could reset a coord value. Users can always construct
   * a new coord from an existing one using the full constructor.
   * This allows us to check for validity at construction once.
   * Callers can still set every coord to wild or every coord to invalid.
   * In that case we force the coord to become invalid.
   */
  const cuda_coords_physical &
  physical () const
  {
    return m_physical;
  }

  const cuda_coords_logical &
  logical () const
  {
    return m_logical;
  }

  bool
  valid () const
  {
    return m_valid;
  }

  /* Methods */

  static cuda_coords
  wild ()
  {
    return cuda_coords{
      CUDA_WILDCARD,     CUDA_WILDCARD,     CUDA_WILDCARD,
      CUDA_WILDCARD,     CUDA_WILDCARD,     CUDA_WILDCARD,
      CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM, CUDA_WILDCARD_DIM
    };
  }

  void
  invalidate ()
  {
    m_valid = false;
  }

  /*
   * Given the logical coordinates, try to update the physical coordinates as
   * they may have changed on the device. Needed for Maxwell with software
   * preemption.
   */
  void resetPhysical ();

  /* Ensure the physical coords are valid on the device and the logical coords
   * match the physical coords on the device. */
  bool
  isValidOnDevice ()
  {
    /* Reset valid */
    m_valid = false;
    /* Make sure there are not any special coordinates */
    if (!isFullyDefined ())
      return false;
    /* Make sure the physical coordinates are on the device and they match the
     * current logical coordinates */
    if (!m_physical.isValidOnDevice (m_logical))
      return false;
    /* Mark this coord as valid - we checked it */
    m_valid = true;
    return true;
  }

  /*
   * Force the coord to be valid. Do not use!
   * FIXME: This is a hack needed for cuda_current_focus::forceValid ()
   */
  void
  forceValid ()
  {
    m_valid = true;
  }

  /* Convert the coord to a std::string representation */
  const std::string to_string () const;
  /* Output mi_out_p like record for the coord to current_ui_out. */
  void emit_mi_output () const;

  bool
  isFullyDefined () const
  {
    return (m_physical.isFullyDefined () && m_logical.isFullyDefined ());
  }

  /* Operators */

  /* Ignore m_valid in comparison check. We want to allow wildcard coords to
   * compare to valid coords. */
  bool
  operator== (const cuda_coords &coord) const
  {
    return ((m_physical == coord.m_physical)
            && (m_logical == coord.m_logical));
  }

  bool
  operator!= (const cuda_coords &coord) const
  {
    return ((m_physical != coord.m_physical)
            || (m_logical != coord.m_logical));
  }
};

/* Singleton that represents the current cuda focus */
class cuda_current_focus final
{
private:
  cuda_current_focus () = default;
  ~cuda_current_focus () = default;

  static cuda_current_focus m_current_focus;
  cuda_coords m_coords;

public:
  /* Delete copy/move/assignment */
  cuda_current_focus (const cuda_current_focus &) = delete;
  cuda_current_focus (cuda_current_focus &&) = delete;
  cuda_current_focus &operator= (const cuda_current_focus &) = delete;
  cuda_current_focus &operator= (cuda_current_focus &&) = delete;

  /* Return a reference to the current coords */
  static const cuda_coords &
  get ()
  {
    return m_current_focus.m_coords;
  }

  /* Set the current focus */
  static void set (cuda_coords &coords);

  /* Return if current focus is on device */
  static bool
  isDevice ()
  {
    return m_current_focus.m_coords.valid ();
  }

  /* Make current focus invalid */
  static void invalidate ();

  /*
   * Force the current focus to be valid
   * FIXME: This is needed because sometimes we will switch away from
   * cuda-focus while the device is running. Setting focus back will require us
   * to validate the coordinates. We can't do this while the device is running.
   * To work around this issue, we just assume that the focus is valid.
   */
  static void
  forceValid ()
  {
    m_current_focus.m_coords.forceValid ();
  }

  /* Update the current coords */
  static void update ();

  /* Output a focus change message */
  static void printFocus (bool switching);
};

/*
 * Cleanup object used to save/restore current focus
 * Use this if you need to change current focus temporarily.
 */
class cuda_focus_restore final
{
private:
  bool m_restored;
  ptid_t m_ptid;
  cuda_coords m_coords;

  void do_restore ();

public:
  cuda_focus_restore ();
  ~cuda_focus_restore () { do_restore (); }
  DISABLE_COPY_AND_ASSIGN (cuda_focus_restore);

  void
  restore ()
  {
    do_restore ();
  }

  void
  dont_restore ()
  {
    m_restored = true;
  }
};
#endif
