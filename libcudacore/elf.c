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

#include "elfutil.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#else
#include <windows.h>
#endif

#ifndef _MSC_VER
#define VERIFY(val, errcode, fmt, args...)				\
	do {								\
		if (!(val)) {						\
			elfSetErrorMsg(fmt, ##args);			\
			return errcode;					\
		}							\
	} while (0)
#else
#define VERIFY(val, errcode, fmt, ...)					\
	do {								\
		if (!(val)) {						\
			elfSetErrorMsg(fmt, __VA_ARGS__);		\
			return errcode;					\
		}							\
	} while (0)
#endif

#define GET_HEADER(e, ehdr, errcode)					\
	do {								\
		(ehdr) = elfGetHeader(e);				\
		VERIFY((ehdr) != NULL, (errcode),			\
		       "Could not read ELF header");			\
	} while (0)

#define ERRMSG_LEN			256

static __THREAD char lastErrMsg[ERRMSG_LEN];

static void elfSetErrorMsg(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vsnprintf(lastErrMsg, ERRMSG_LEN, fmt, args);
}

const char *elfErrorMsg(void)
{
	return lastErrMsg;
}

struct Elf_st {
#ifndef _WIN32
	int fd;
#else
	HANDLE hFile;
	HANDLE hFileMappingObject;
#endif
	off_t size;
	void *mapped_addr;
};

int elfCheckHeader(Elf *e, unsigned char *ident)
{
	Elf64_Ehdr *ehdr;

	if (ident == NULL)
		return 0; /* Skip checking */

	GET_HEADER(e, ehdr, -1);

	VERIFY(memcmp(ehdr->e_ident, ident, EI_PAD) == 0, -1,
	       "ELF header 'ident' verification failed");

	return 0;
}

Elf *elfOpenByName(const char *fileName, unsigned char *ident)
{
	Elf *e;

	e = calloc(1, sizeof(*e));
	VERIFY(e != NULL, NULL, "Could not allocate memory");
#ifndef _WIN32
	e->mapped_addr = MAP_FAILED;

	e->fd = open(fileName, O_RDONLY);
	if (e->fd == -1) {
		elfSetErrorMsg("Could not open file '%s': %s", fileName,
			       strerror(errno));
		goto cleanup;
	}

	e->size = lseek(e->fd, 0, SEEK_END);
	if (e->size == (off_t)(-1)) {
		elfSetErrorMsg("lseek() failed: %s", strerror(errno));
		goto cleanup;
	}

	e->mapped_addr = mmap(NULL, e->size, PROT_READ, MAP_PRIVATE, e->fd, 0);
	if (e->mapped_addr == MAP_FAILED) {
		elfSetErrorMsg("mmap() failed: %s", strerror(errno));
		goto cleanup;
	}
#else
	e->mapped_addr = NULL;
	e->hFileMappingObject = NULL;

	e->hFile = CreateFile (fileName, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (e->hFile == INVALID_HANDLE_VALUE) {
		elfSetErrorMsg("Count not open file '%s': GetLastError()=0x%x:", fileName, (int)GetLastError());
		goto cleanup;
	}

	e->size = GetFileSize (e->hFile, NULL);

	e->hFileMappingObject = CreateFileMapping (e->hFile, NULL, PAGE_READONLY, 0, e->size, NULL);
	if (e->hFileMappingObject == NULL) {
		elfSetErrorMsg("CreateFileMapping() failed: GetLastError()=0x%x", (int)GetLastError());
		goto cleanup;
	}
	e->mapped_addr = MapViewOfFile (e->hFileMappingObject, FILE_MAP_READ, 0, 0, e->size);
	if (e->mapped_addr == NULL) {
		elfSetErrorMsg("MapViewOfFile() failed: GetLastError()=0x%x", (int)GetLastError());
		goto cleanup;
	}
#endif

	if (elfCheckHeader(e, ident) == -1)
		goto cleanup;

	return e;

cleanup:
	elfFree(e);
	return NULL;
}

Elf *elfOpenInMemory(char *buf, size_t size, unsigned char *ident)
{
	Elf *e;

	e = calloc(1, sizeof(*e));
	VERIFY(e != NULL, NULL, "Could not allocate memory");

#ifndef _WIN32
	e->fd = -1;
#else
	e->hFile = INVALID_HANDLE_VALUE;
	e->hFileMappingObject = NULL;
#endif
	e->size = size;
	e->mapped_addr = buf;

	if (elfCheckHeader(e, ident) == -1)
		goto cleanup;

	return e;

cleanup:
	elfFree(e);
	return NULL;
}

Elf64_Ehdr *elfGetHeader(Elf *e)
{
	return (Elf64_Ehdr *)e->mapped_addr;
}

int elfGetSectionHeaderNum(Elf *e, size_t *shnum)
{
	Elf64_Ehdr *ehdr;

	GET_HEADER(e, ehdr, -1);

	if (ehdr->e_shnum == 0) {
		Elf_Scn *scn;
		Elf64_Shdr *shdr;

		scn = elfGetSection(e, 0);
		if (scn == NULL)
			return -1;

		shdr = elfGetSectionHeader(scn);
		if (shdr == NULL)
			return -1;

		*shnum = shdr->sh_size;
	} else {
		*shnum = ehdr->e_shnum;
	}

	return 0;
}

int elfGetSectionHeaderStrTblIdx(Elf *e, size_t *shstrndx)
{
	Elf64_Ehdr *ehdr;

	GET_HEADER(e, ehdr, -1);

	if (ehdr->e_shstrndx == SHN_XINDEX) {
		Elf_Scn *scn;
		Elf64_Shdr *shdr;

		scn = elfGetSection(e, 0);
		if (scn == NULL)
			return -1;

		shdr = elfGetSectionHeader(scn);
		if (shdr == NULL)
			return -1;

		*shstrndx = shdr->sh_link;
	} else {
		*shstrndx = ehdr->e_shstrndx;
	}

	return 0;
}

Elf_Scn *elfGetSection(Elf *e, size_t secndx)
{
	Elf64_Ehdr *ehdr;
	intptr_t scn;

	GET_HEADER(e, ehdr, NULL);

	if (secndx != 0) {
		size_t shnum;

		VERIFY(elfGetSectionHeaderNum(e, &shnum) == 0, NULL,
		       "Could not read number of section headers");

		VERIFY(secndx < shnum, NULL, "Invalid section index");
	}

	VERIFY(readUint64(&ehdr->e_shoff) > 0, NULL, "Missing section header table");

	VERIFY(readUint16(&ehdr->e_shentsize) == sizeof(Elf64_Shdr), NULL,
	       "Section header size mismatch");

	/* Relative address in ELF image */
	scn = readUint64(&ehdr->e_shoff) + secndx * readUint16(&ehdr->e_shentsize);

	VERIFY(scn + readUint16(&ehdr->e_shentsize) <= e->size, NULL,
	       "Section offset out of ELF image bounds");

	/* Absolute address in memory */
	scn += (intptr_t)e->mapped_addr;

	return (Elf_Scn *)scn;
}

size_t elfGetSectionIndex(Elf *e, Elf_Scn *scn)
{
	Elf64_Ehdr *ehdr;
	intptr_t scnaddr = (intptr_t)scn;

	GET_HEADER(e, ehdr, SHN_UNDEF);

	/* Get relative address in ELF image */
	scnaddr -= (intptr_t)e->mapped_addr;

	return (scnaddr - readUint64(&ehdr->e_shoff)) / readUint16(&ehdr->e_shentsize);
}

Elf_Scn *elfGetNextSection(Elf *e, Elf_Scn *scn)
{
	size_t secndx;

	if (scn == NULL)
		return elfGetSection(e, 1);

	secndx = elfGetSectionIndex(e, scn);
	if (secndx == SHN_UNDEF)
		return NULL;

	return elfGetSection(e, secndx + 1);
}

Elf64_Shdr *elfGetSectionHeader(Elf_Scn *scn)
{
	return (Elf64_Shdr *)scn;
}

int elfGetSectionData(Elf *e, Elf_Scn *scn, Elf_Data *data)
{
	Elf64_Shdr *shdr;

	shdr = elfGetSectionHeader(scn);
	VERIFY(shdr != NULL, -1, "Could not get section header");

	VERIFY(readUint32(&shdr->sh_type) != SHT_NOBITS, -1, "No data for SHT_NOBITS");

	data->d_buf = (void *)((char *)e->mapped_addr + readUint64(&shdr->sh_offset));
	data->d_size = readUint64(&shdr->sh_size);
	data->d_type = readUint32(&shdr->sh_type);
	data->d_entsize = readUint64(&shdr->sh_entsize);

	return 0;
}

const char *elfGetString(Elf *e, size_t secndx, size_t idx)
{
	Elf_Scn *scn;
	Elf_Data data;

	scn = elfGetSection(e, secndx);
	if (scn == NULL)
		return NULL;

	if(elfGetSectionData(e, scn, &data) != 0)
		return NULL;

	VERIFY(idx < data.d_size, NULL, "String index out of bounds");

	return (const char *)data.d_buf + idx;
}

void elfFree(Elf *e)
{
	assert(e != NULL);
#ifndef _WIN32
	if (e->fd != -1) {
		if (e->mapped_addr != MAP_FAILED)
			munmap(e->mapped_addr, e->size);

		close(e->fd);
	}
#else
	if (e->hFile != INVALID_HANDLE_VALUE) {
		if (e->mapped_addr != NULL)
			UnmapViewOfFile (e->hFile);
		if (e->hFileMappingObject)
			CloseHandle (e->hFileMappingObject);
		CloseHandle (e->hFile);
	}
#endif

	free(e);
}
