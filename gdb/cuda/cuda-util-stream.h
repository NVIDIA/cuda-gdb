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
 *
 * This header defines file_stream: a light buffered reader class for streaming
 * a file.
 */

#ifndef _CUDA_UTIL_STREAM_H
#define _CUDA_UTIL_STREAM_H 1

#include "gdbsupport/common-exceptions.h"
#include "gdbsupport/common-utils.h"

#include <exception>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace util_stream
{
class istream
{
public:
  virtual ~istream () = default;
  virtual int getc () noexcept = 0;
  virtual int peekc () noexcept = 0;
  virtual size_t tell () const noexcept = 0;
};

/* This class is used only for testing. */
class string_stream : public istream
{
public:
  inline string_stream (std::string &&str) : m_str (std::move (str)), m_pos (0)
  {
  }

  int
  getc () noexcept override
  {
    return m_pos < m_str.size () ? (u_char)m_str[m_pos++] : EOF;
  }

  int
  peekc () noexcept override
  {
    return m_pos < m_str.size () ? (u_char)m_str[m_pos] : EOF;
  }

  size_t
  tell () const noexcept override
  {
    return m_pos;
  }

  virtual ~string_stream () = default;

private:
  std::string m_str;
  size_t m_pos;
};

/* Buffered stream from FD. */
class file_stream : public istream
{
  static constexpr size_t BUFFER_SIZE = BUFSIZ;

  u_char m_buffer[BUFFER_SIZE];
  size_t m_pos;
  size_t m_size;
  size_t m_offset;
  FILE *m_file;

public:
  inline file_stream (const std::string &filename)
      : m_buffer{}, m_pos (0), m_size (0), m_offset (0),
	m_file (fopen (filename.c_str (), "r"))
  {
    if (!m_file)
      throw_error (errors::GENERIC_ERROR, "Failed to open file %s: %s",
		   filename.c_str (), safe_strerror (errno));
  }

  inline file_stream (int fd)
      : m_buffer{}, m_pos (0), m_size (0), m_offset (0),
	m_file (fdopen (fd, "r"))
  {
    if (!m_file)
      throw_error (errors::GENERIC_ERROR,
		   "Failed to open file descriptor %d: %s", fd,
		   safe_strerror (errno));
  }

  /* Consume a character from the stream. Does not throw on errors, but
   * returns EOF. */
  int
  getc () noexcept override
  {
    int c = peekc ();
    if (c != EOF)
      m_pos++;
    return c;
  }

  /* Peek the current character from the stream. Does not throw on errors,
   * but returns EOF. */
  int
  peekc () noexcept override
  {
    if (m_pos >= m_size)
      {
	m_offset += m_size;
	m_size = fread (m_buffer, 1, BUFFER_SIZE, m_file);
	m_pos = 0;
      }
    return (m_pos < m_size) ? m_buffer[m_pos] : EOF;
  }

  /* Return the current position in the stream, as bytes consumed from the
   * start. */
  size_t
  tell () const noexcept override
  {
    return m_offset + m_pos;
  }

  inline
  operator bool () const
  {
    return m_file;
  }

  virtual ~file_stream ()
  {
    if (fclose (m_file) != 0)
      warning ("Failed to close file: %s", strerror (errno));
  }
};
} // namespace util_stream

#endif
