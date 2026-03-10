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

#ifndef _ELFUTIL_H_
#define _ELFUTIL_H_

#include <stdlib.h>

#include "elfdefinitions.h"
#include "tls.h"

/* Android fix */
#ifndef SHN_XINDEX
#define SHN_XINDEX	0xFFFF
#endif

typedef struct Elf_st Elf;
typedef Elf64_Shdr Elf_Scn;

typedef struct {
	void *d_buf;
	uint64_t d_size;
	uint64_t d_entsize;
	uint32_t d_type;
} Elf_Data;

const char *elfErrorMsg(void);

Elf *elfOpenByName(const char *fileName, unsigned char *ident);
Elf *elfOpenInMemory(char *buf, size_t size, unsigned char *ident);

Elf64_Ehdr *elfGetHeader(Elf *e);
int elfGetSectionHeaderNum(Elf *e, size_t *shnum);
int elfGetSectionHeaderStrTblIdx(Elf *e, size_t *shstrndx);

Elf_Scn *elfGetSection(Elf *e, size_t secndx);
size_t elfGetSectionIndex(Elf *e, Elf_Scn *scn);
Elf_Scn *elfGetNextSection(Elf *e, Elf_Scn *scn);
Elf64_Shdr *elfGetSectionHeader(Elf_Scn *scn);
int elfGetSectionData(Elf *e, Elf_Scn *scn, Elf_Data *data);
const char *elfGetString(Elf *e, size_t secndx, size_t idx);

void elfFree(Elf *e);


#ifndef __arm__
#define readUint64(x) (*(x))
#define readUint32(x) (*(x))
#define readUint16(x) (*(x))
#else
static inline uint64_t readUint64(uint64_t *ptr)
{
	if (((long)ptr&7)!=0) {
		uint64_t local;
		memcpy ((void *)&local, (void *)ptr, sizeof(uint64_t));
		return local;
	}
	return *ptr;
}

static inline uint32_t readUint32(uint32_t *ptr)
{
	if (((long)ptr&3)!=0) {
		uint32_t local;
		memcpy ((void *)&local, (void *)ptr, sizeof(uint32_t));
		return local;
	}
	return *ptr;
}

static inline uint16_t readUint16(uint16_t *ptr)
{
	if (((long)ptr&1)!=0) {
		uint16_t local;
		memcpy ((void *)&local, (void *)ptr, sizeof(uint16_t));
		return local;
	}
	return *ptr;
}
#endif

#endif /* _ELFUTIL_H_ */
