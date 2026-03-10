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

#ifndef _COMMON_H_
#define _COMMON_H_

#include "elfutil.h"
#include "cudacoredump.h"
#include "cudadebugger.h"
#include "uthash.h"
#include "utarray.h"

/* Get bool type definitions */
#ifndef _MSC_VER
#include <stdbool.h>
#else
typedef unsigned char bool;
#endif

#ifndef _MSC_VER
#include <unistd.h>
#endif


/* ELF header information */
#define EM_CUDA			0xBE
#define ELFOSABI_CUDA		0x33
#define ELFOSABIV_LATEST	0x7

#define MAPIDENT_LEN		128
#define TMPBUF_LEN		256
#define DISASM_TMP_TEMPLATE	"/tmp/cudacore_disassembly_XXXXXX"

typedef struct {
	uint64_t address;
	uint64_t size;
	Elf *e;
	Elf_Scn *scn;
	Elf64_Shdr *shdr;
} MemorySeg;

/* Note: for now using string key (and fixed length) */
typedef struct {
	char ident[MAPIDENT_LEN];
	void *entryPtr;
	size_t size;
	bool needsFree;
	UT_hash_handle hh;
} MapEntry;

typedef struct CudaCoreEvent_st {
	CUDBGEvent event;
	struct CudaCoreEvent_st *next;
} CudaCoreEvent;

typedef struct CudaCoreELFImage_st {
	CudbgDeviceTableEntry *dte;
	Elf *e;
	Elf_Scn *scn;
	struct CudaCoreELFImage_st *next;
} CudaCoreELFImage;

struct CudaCore_st {
	Elf *e;				/* ELF handle for core dump */

	size_t shstrndx;		/* Section header string table index */
	size_t shnum;			/* Number of sections in core dump */
	size_t strndx;			/* String table section index */

	size_t numDevices;		/* Number of CUDA devices */
	MapEntry *tableEntriesMap;	/* Hash map with state information */
	UT_array *managedMemorySegs;	/* Sorted array of managed memory segments */
	UT_array *globalMemorySegs;	/* Sorted array of global memory segments */

	CudaCoreEvent *eventHead;	/* Single linked list of CUDA Events */
	CudaCoreELFImage *relocatedELFImageHead;
					/* Single linked list of CUDA ELF images */
};

#ifndef _MSC_VER

#define DPRINTF(level, fmt, args...)					\
	dbgprintf(level, "[%s:%d][%s] " fmt,				\
		  __FILE__, __LINE__, __FUNCTION__, ##args)

#define TRACE_FUNC(fmt, args...)					\
	DPRINTF(30, fmt "\n", ##args)

#define VERIFY(val, errcode, fmt, args...)				\
	do {								\
		if (!(val)) {						\
			DPRINTF(100, fmt "\n", ##args);			\
			cuCoreSetErrorMsg(fmt, ##args);			\
			return errcode;					\
		}							\
	} while (0)
#endif /*_MSC_VER*/

#define VERIFY_ARG(val)							\
	VERIFY((val) != NULL, CUDBG_ERROR_INVALID_ARGS,			\
	       "Invalid argument '" #val "'.")

#ifndef _MSC_VER
#define GET_TABLE_ENTRY(entry, size, errcode, key, args...)			\
	do {									\
		(entry) = cuCoreGetMapEntry(&curcc->tableEntriesMap, size,	\
					    key, ##args);			\
		if ((entry) == NULL)						\
			return errcode;						\
	} while (0)
#endif /*_MSC_VER*/

/**/
#ifndef _MSC_VER
#define _PRINTF_ARGS(fmt,var) __attribute__ ((format (printf, fmt, var)))
#define _INLINE inline
#define _SNPRINTF snprintf
#define _POPEN popen
#define _PCLOSE pclose
#else
#define _PRINTF_ARGS(fmt,var)
#define _INLINE __inline
#define _SNPRINTF _snprintf_c
#define _POPEN _popen
#define _PCLOSE _pclose
#endif

void dbgprintf(int level, const char *fmt, ...) _PRINTF_ARGS(2, 3);
int cuCoreSortMemorySegs(const void *a, const void *b);
void cuCoreSetErrorMsg(const char *fmt, ...) _PRINTF_ARGS(1, 2);
void *cuCoreGetMapEntry(MapEntry **map, size_t* size, const char *fmt, ...) _PRINTF_ARGS(3, 4);
size_t cuCoreGetNumDevices(CudaCore *cc);
const char *cuCoreGetStrTabByIndex(CudaCore *cc, size_t idx);
const CUDBGEvent *cuCoreGetEvent(CudaCore *cc);
int cuCoreDeleteEvent(CudaCore *cc);
int cuCoreReadSectionHeader(Elf_Scn *scn, Elf64_Shdr **shdr);
int cuCoreReadSectionData(Elf *e, Elf_Scn *scn, Elf_Data *data);

/* Inner ELF images */
typedef uint64_t cs_t;
void cuCoreExecuteCallStack(CudaCore *cc, cs_t *callStack);
int cuCoreIterateELFImages(CudaCore *cc, cs_t *callStack);
/* Callback type: ProcessELF */
int cuCoreIterateELFSections(CudaCore *cc, Elf *e, cs_t *callStack);
/* Callback type: ProcessSection */
int cuCoreIterateSymbolTable(CudaCore *cc, Elf *e, Elf_Scn *scn,
			     Elf64_Shdr *shdr, cs_t *callStack);
/* Callback type: ProcessSymbol */
int cuCoreFilterSymbolByName(CudaCore *cc, Elf *e, Elf_Scn *scn,
			     Elf64_Shdr *shdr, Elf64_Sym *sym,
			     cs_t *callStack);
int cuCoreFilterSymbolByAddress(CudaCore *cc, Elf *e, Elf_Scn *scn,
			        Elf64_Shdr *shdr, Elf64_Sym *sym,
			        cs_t *callStack);
int cuCoreFilterSymbolByType(CudaCore *cc, Elf *e, Elf_Scn *scn,
			     Elf64_Shdr *shdr, Elf64_Sym *sym,
			     cs_t *callStack);
int cuCoreReadSymbolAddress(CudaCore *cc, Elf *e, Elf_Scn *scn,
			    Elf64_Shdr *shdr, Elf64_Sym *sym,
			    cs_t *callStack);
int cuCoreReadSymbolSection(CudaCore *cc, Elf *e, Elf_Scn *scn,
			    Elf64_Shdr *shdr, Elf64_Sym *sym,
			    cs_t *callStack);
/* Callback type: ProcessSymbolSection */
int cuCoreReadSymbolData(CudaCore *cc, Elf *e, Elf_Scn *scn,
			 Elf64_Shdr *shdr, Elf64_Sym *sym,
			 Elf_Scn *symscn, Elf64_Shdr *symshdr,
			 cs_t *callStack);

/* Return 1 if bit is set, 0 otherwise */
static _INLINE int getBit(uint64_t val, unsigned bitNo) {
	return ((val >> bitNo) & 1);
}

#define CUDA_COREDUMP_SHARED_WINDOW_SIZE (16 * 1024 * 1024) // bytes
#define CUDA_COREDUMP_LOCAL_WINDOW_SIZE (16 * 1024 * 1024) // bytes


/* Microsoft Compiler specific defenitions of macros with variadic arguments */
#ifdef _MSC_VER
#define DPRINTF(level, fmt, ...)					\
	dbgprintf(level, "[%s:%d][%s] " fmt,				\
		  __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

#define TRACE_FUNC(fmt, ...)					\
	DPRINTF(30, fmt "\n", __VA_ARGS__)

#define VERIFY(val, errcode, fmt, ...)				\
	do {								\
		if (!(val)) {						\
			DPRINTF(100, fmt "\n", __VA_ARGS__);		\
			cuCoreSetErrorMsg(fmt, __VA_ARGS__);		\
			return errcode;					\
		}							\
	} while (0)

#define GET_TABLE_ENTRY(entry, size, errcode, key, ...)				\
	do {									\
		(entry) = cuCoreGetMapEntry(&curcc->tableEntriesMap, size,	\
					    key, __VA_ARGS__);			\
		if ((entry) == NULL)						\
			return errcode;						\
	} while (0)
#endif /* _MSC_VER */

#endif /* _COMMON_H_ */
