/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2021-2023 NVIDIA Corporation
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

#ifndef _CUDA_VERSION_H
#define _CUDA_VERSION_H 1

const int cuda_major_version (void);
const int cuda_minor_version (void);
const char *cuda_current_year (void);

class cuda_api_version
{
public:
    explicit cuda_api_version()
      : m_major {0}, m_minor {0}, m_revision {0} { }
    explicit cuda_api_version(uint32_t major, uint32_t minor, uint32_t revision)
      : m_major {major}, m_minor {minor}, m_revision {revision} { }
        
    uint32_t m_major;
    uint32_t m_minor;
    uint32_t m_revision;
};

#endif

