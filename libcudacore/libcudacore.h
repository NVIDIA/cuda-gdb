/*
 * Copyright (c) 2014-2025 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file libcudacore.h
 * \brief API for reading CUDA core files.
 *
 * This header file describes the library's interface for opening and
 * reading core files.
 */

#ifndef _LIBCUDACORE_H_
#define _LIBCUDACORE_H_

#include "cudadebugger.h"
#include "tls.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CudaCore_st CudaCore;

/**
 * \brief Get last error message.
 * \return String describing last error.
 *
 * This function returns a string which describes the last error condition
 * which occured (after a call to one of cuCore* functions).
 */
const char *cuCoreErrorMsg(void);

/**
 * \brief Open core file by name.
 * \param fileName Core file name, a string.
 * \return CudaCore object which should be used by subsequent calls
 *         to cuCore* functions. On error NULL is returned.
 * \sa cuCoreOpenInMemory(), cuCoreGetApi(), cuCoreFree()
 *
 * This function should be used to open a core file on the file system.
 * The CudaCore object returned from this function should be used by
 * subsequent calls to cuCore*() functions: cuCoreGetApi() and cuCoreFree().
 */
CudaCore *cuCoreOpenByName(const char *fileName);

/**
 * \brief Open core file already residing in memory.
 * \param buf Buffer filled with core file contents. Contents of the buffer
 *            should be valid and accessible until cuCoreFree() is called.
 * \param size Size of the core file.
 * \return CudaCore object which should be used by subsequent calls
 *         to cuCore*() functions. On error NULL is returned.
 * \sa cuCoreOpenByName(), cuCoreGetApi(), cuCoreFree()
 *
 * This function should be used to open a core file which is already residing
 * in memory.
 * The CudaCore object returned from this function should be used by
 * subsequent calls to cuCore*() functions: cuCoreGetApi() and cuCoreFree().
 */
CudaCore *cuCoreOpenInMemory(char *buf, size_t size);

/**
 * \brief Free (close) the core file.
 * \param cc CudaCore object returned by a previous call to one of
 *           cuCoreOpen*() functions.
 * \sa cuCoreOpenByName(), cuCoreOpenInMemory()
 *
 * This function should be used to close a core file and free all resources
 * associated with it.
 * The CudaCore object should not be used after calling this function.
 */
void cuCoreFree(CudaCore *cc);

/**
 * \brief Get CUDA debugger API.
 * \param cc CudaCore object returned by a previous call to one of
 *           cuCoreOpen*() functions.
 * \return CUDBGAPI structure pointer which provides the CUDA debugger API.
 * \sa cuCoreOpenByName(), cuCoreOpenInMemory()
 *
 * This function returns a CUDBGAPI function table for the CudaCore object,
 * and is valid until cuCoreFree() is called.
 */
CUDBGAPI cuCoreGetApi(CudaCore *cc);

#ifdef __cplusplus
}
#endif

#endif /* _LIBCUDACORE_H_ */
