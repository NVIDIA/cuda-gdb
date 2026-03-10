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

#include "libcudacore.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "common.h"

#define ENV_VAR_DEBUG			"CUCORE_DEBUG"
#define GLOBAL_MEMORY_SEGMENTS_MIN	10
#define ERRMSG_LEN			256

static __THREAD char lastErrMsg[ERRMSG_LEN];
static int cuCoreAddMapEntry(MapEntry **, void *, size_t, bool, const char *, ...)
 _PRINTF_ARGS(5, 6);

void cuCoreSetErrorMsg(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vsnprintf(lastErrMsg, ERRMSG_LEN, fmt, args);
}

const char *cuCoreErrorMsg(void)
{
	return lastErrMsg;
}

/* Memory segment array descriptor */
static UT_icd memorySeg_icd = { sizeof(MemorySeg), NULL, NULL, NULL };

/* ELF core dump image identification signature */
static unsigned char cudaElfIdent[EI_PAD] = {
	ELFMAG0,
	ELFMAG1,
	ELFMAG2,
	ELFMAG3,
	ELFCLASS64,
	ELFDATA2LSB,
	EV_CURRENT,
	ELFOSABI_CUDA,
	ELFOSABIV_LATEST,
};

/* It is assumed that memory segments do not overlap when used for sorting.
 * This assumption allows to use this function also for searching a memory
 * segment by an address that it contains. */
int cuCoreSortMemorySegs(const void *a, const void *b)
{
	MemorySeg *memorySegA = (MemorySeg *)a;
	MemorySeg *memorySegB = (MemorySeg *)b;

	uint64_t startAddrA = memorySegA->address;
	uint64_t startAddrB = memorySegB->address;

	uint64_t endAddrA = memorySegA->address + memorySegA->size;
	uint64_t endAddrB = memorySegB->address + memorySegB->size;

	if ((startAddrA >= startAddrB && endAddrA <= endAddrB) ||
	    (startAddrB >= startAddrA && endAddrB <= endAddrA))
		return 0; /* A contains B or B contains A */

	return (memorySegA->address > memorySegB->address) ? 1 : -1;
}

static int debugLevel(void)
{
	static int __debugLevel = -1;
	char *env_var_debug;

	if (__debugLevel >= 0)
		return __debugLevel;

	env_var_debug = getenv(ENV_VAR_DEBUG);
	if (env_var_debug)
		__debugLevel = atoi(env_var_debug);

	if (__debugLevel < 0)
		__debugLevel = 0;

	return __debugLevel;
}

void dbgprintf(int level, const char *fmt, ...)
{
	va_list args;

	if (level > debugLevel())
		return;

	va_start(args, fmt);
	fprintf(stderr, "DBG[%02d]", level);
	vfprintf(stderr, fmt, args);
	va_end(args);
}

static int cuCoreInit(CudaCore *cc);
static int cuCoreAddEvent(CudaCore *cc, CUDBGEvent *evt);

CudaCore *cuCoreOpenByName(const char *fileName)
{
	CudaCore *cc;

	cc = calloc(1, sizeof(*cc));
	VERIFY(cc != NULL, NULL, "Could not allocate memory");

	cc->e = elfOpenByName(fileName, cudaElfIdent);
	if (cc->e == NULL) {
		cuCoreSetErrorMsg("elfOpenByName() failed: %s",
				  elfErrorMsg());
		goto cleanup;
	}

	if (cuCoreInit(cc))
		goto cleanup;

	return cc;

cleanup:
	cuCoreFree(cc);
	return NULL;
}

CudaCore *cuCoreOpenInMemory(char *buf, size_t size)
{
	CudaCore *cc;

	cc = calloc(1, sizeof(*cc));
	VERIFY(cc != NULL, NULL, "Could not allocate memory");

	cc->e = elfOpenInMemory(buf, size, cudaElfIdent);
	if (cc->e == NULL) {
		cuCoreSetErrorMsg("elfOpenInMemory() failed: %s",
				  elfErrorMsg());
		goto cleanup;
	}

	if (cuCoreInit(cc))
		goto cleanup;

	return cc;

cleanup:
	cuCoreFree(cc);
	return NULL;
}

int cuCoreReadSectionHeader(Elf_Scn *scn, Elf64_Shdr **shdr)
{
	*shdr = elfGetSectionHeader(scn);
	VERIFY(*shdr != NULL, -1, "elfGetSectionHeader() failed: %s",
	       elfErrorMsg());

	return 0;
}

int cuCoreReadSectionData(Elf *e, Elf_Scn *scn, Elf_Data *data)
{
	VERIFY(elfGetSectionData(e, scn, data) == 0, -1,
	       "Could not read section %llu data: %s",
		   (unsigned long long)elfGetSectionIndex(e, scn), elfErrorMsg());

	return 0;
}

static int cuCoreReadMemorySection(Elf *e, Elf_Scn *scn,
				   UT_array *memorySegs)
{
	MemorySeg memorySeg;

	if (cuCoreReadSectionHeader(scn, &memorySeg.shdr) != 0)
		return -1;

	memorySeg.e = e;
	memorySeg.scn = scn;
	memorySeg.address = readUint64(&memorySeg.shdr->sh_addr);
	memorySeg.size = readUint64(&memorySeg.shdr->sh_size);

	utarray_push_back(memorySegs, &memorySeg);

	DPRINTF(20, "Created memory segment 0x%llX 0x%llX\n",
		(long long unsigned int)memorySeg.address,
		(long long unsigned int)memorySeg.size);

	return 0;
}

static int cuCoreGenericReadTable(Elf *e, Elf_Scn *scn,
				  size_t *count, void **table,
				  size_t *shentsize,
				  size_t *parent, size_t *offset)
{
  Elf_Data data;
  Elf64_Shdr *shdr;

  if (cuCoreReadSectionData(e, scn, &data) != 0)
    return -1;

  if (cuCoreReadSectionHeader(scn, &shdr) != 0)
    return -1;

  if (parent)
    *parent = readUint32(&shdr->sh_link);
  if (offset)
    *offset = readUint32(&shdr->sh_info);
  *shentsize = readUint64(&shdr->sh_entsize);

  *count = data.d_size / readUint64(&shdr->sh_entsize);
  *table = data.d_buf;

  return 0;
}

static int cuCoreAddMapEntry(MapEntry **map, void *entryPtr, size_t size, bool needsFree,
			     const char *fmt, ...)
{
	MapEntry *mapEntry;
	va_list args;

	va_start(args, fmt);

	mapEntry = malloc(sizeof(*mapEntry));
	VERIFY(mapEntry != NULL, -1, "Could not allocate memory");

	vsnprintf(mapEntry->ident, MAPIDENT_LEN, fmt, args);
	mapEntry->entryPtr = entryPtr;
	mapEntry->size = size;
	mapEntry->needsFree = needsFree;

	DPRINTF(80, "Mapped '%s' to %p needsFree %d\n",
                mapEntry->ident, entryPtr, needsFree);

	HASH_ADD_STR(*map, ident, mapEntry);

	return 0;
}

void *cuCoreGetMapEntry(MapEntry **map, size_t* size, const char *fmt, ...)
{
	MapEntry *mapEntry;
	va_list args;
	char ident[MAPIDENT_LEN];

	va_start(args, fmt);
	vsnprintf(ident, MAPIDENT_LEN, fmt, args);

	HASH_FIND_STR(*map, ident, mapEntry);
	VERIFY(mapEntry != NULL, NULL, "Map entry '%s' not found", ident);

	DPRINTF(90, "Got map entry '%s' %p\n",
		mapEntry->ident, mapEntry->entryPtr);

	if (size)
	  {
		*size = mapEntry->size;
	  }

	return mapEntry->entryPtr;
}

static int cuCoreReadDeviceTable(CudaCore *cc, Elf_Scn *scn)
{
  uint8_t *dt;
  size_t dte_count;
  CudbgDeviceTableEntry *dte;
  size_t dteSz;
  size_t i;

  if (cuCoreGenericReadTable(cc->e, scn,
                             &dte_count,
                             (void **)&dt,
                             &dteSz, NULL, NULL) != 0)
    return -1;

  cc->numDevices = dte_count;

  for (i = 0; i < dte_count; ++i, dt += dteSz) {
    dte = (CudbgDeviceTableEntry *)dt;

    /* delete the dte if this entry is deleted */
    if (cuCoreAddMapEntry(&cc->tableEntriesMap, dte, dteSz, false,
                          "dev%u",
                          dte->devId))
      return -1;

    if (cuCoreAddMapEntry(&cc->tableEntriesMap, dte, dteSz, false,
                          "devtbl_offset%llu", (unsigned long long)i))
      return -1;
  }

  return 0;
}

static int cuCoreReadGridTable(CudaCore *cc, Elf_Scn *scn)
{
	size_t gte_count;
	uint8_t *gt;
	CudbgGridTableEntry *gte;
	CudbgDeviceTableEntry *dte;
	size_t dteSz, gteSz;
	size_t parent, offset;
	size_t i;

	if (cuCoreGenericReadTable(cc->e, scn,
				   &gte_count,
				   (void **)&gt,
				   &gteSz, &parent, &offset) != 0)
		return -1;

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&dteSz,
				"devtbl_offset%llu",
				(unsigned long long)offset);
	VERIFY(dte != NULL, -1, "Could not find Device table entry");

	for (i = 0; i < gte_count; ++i, gt += gteSz) {
		gte = (CudbgGridTableEntry *)gt;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, gte, gteSz, false,
				      "grid%llu_dev%u",
				      (unsigned long long)gte->gridId64,
				      dte->devId))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, gte, gteSz, false,
				      "grid_section%llu_offset%llu",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn),
					  (unsigned long long)i))
			return -1;

		/* Index dte by grid section and offset */
		if (cuCoreAddMapEntry(&cc->tableEntriesMap, dte, dteSz, false,
				      "grid_section%llu_offset%llu_dev",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn),
					  (unsigned long long)i))
			return -1;
	}

	return 0;
}

static int cuCoreReadSmTable(CudaCore *cc, Elf_Scn *scn)
{
	size_t ste_count;
	uint8_t *st;
	CudbgSmTableEntry *ste;
	CudbgDeviceTableEntry *dte;
	size_t dteSz, steSz;
	size_t parent, offset;
	size_t i;

	if (cuCoreGenericReadTable(cc->e, scn,
				   &ste_count,
				   (void **)&st,
				   &steSz, &parent, &offset) != 0)
		return -1;

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&dteSz,
				"devtbl_offset%llu",
				(unsigned long long)offset);
	VERIFY(dte != NULL, -1, "Could not find Device table entry");

	for (i = 0; i < ste_count; ++i, st += steSz) {
		ste = (CudbgSmTableEntry *)st;
		if (cuCoreAddMapEntry(&cc->tableEntriesMap, ste, steSz, false,
				      "sm%u_dev%u",
				      ste->smId,
				      dte->devId))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, ste, steSz, false,
				      "smtbl_section%llu_offset%llu",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;

		/* Link SM table entry to device */
		if (cuCoreAddMapEntry(&cc->tableEntriesMap, dte, dteSz, false,
				      "smtbl_section%llu_offset%llu_dev",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;
	}

	return 0;
}

static int cuCoreReadCTATable(CudaCore *cc, Elf_Scn *scn)
{
	size_t ctate_count;
	uint8_t *ctate;
	CudbgSmTableEntry *ste;
	CudbgDeviceTableEntry *dte;
	size_t dteSz, steSz, ctateSz;
	size_t parent, offset;
	size_t i;

	if (cuCoreGenericReadTable(cc->e, scn,
				   &ctate_count,
				   (void **)&ctate,
				   &ctateSz, &parent, &offset) != 0)
		return -1;

	ste = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&steSz,
				"smtbl_section%llu_offset%llu",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(ste != NULL, -1, "Could not find SM table entry");

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&dteSz,
				"smtbl_section%llu_offset%llu_dev",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(dte != NULL, -1, "Could not find Device table entry by SM");

	for (i = 0; i < ctate_count; ++i, ctate += ctateSz) {
		if (cuCoreAddMapEntry(&cc->tableEntriesMap, ctate, ctateSz, false,
				      "cta%u_sm%u_dev%u",
				      (uint32_t) i,
				      ste->smId,
				      dte->devId))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, ctate, ctateSz, false,
				      "ctatbl_section%llu_offset%llu",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, ste, steSz, false,
				      "ctatbl_section%llu_offset%llu_sm",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, dte, dteSz, false,
				      "ctatbl_section%llu_offset%llu_dev",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;
	}

	return 0;
}

static int cuCoreReadWarpTable(CudaCore *cc, Elf_Scn *scn)
{
	size_t wte_count;
	uint8_t *wt;
	CudbgWarpTableEntry *wte;
	CudbgCTATableEntry *ctate;
	CudbgSmTableEntry *ste;
	CudbgDeviceTableEntry *dte;
	size_t dteSz, steSz, ctateSz, wteSz;
	size_t parent, offset;
	size_t i;

	if (cuCoreGenericReadTable(cc->e, scn,
				   &wte_count,
				   (void **)&wt,
				   &wteSz, &parent, &offset) != 0)
		return -1;

	ctate = cuCoreGetMapEntry(&cc->tableEntriesMap,
				  &ctateSz,
				  "ctatbl_section%llu_offset%llu",
				  (unsigned long long)parent, (unsigned long long)offset);
	VERIFY(ctate != NULL, -1, "Could not find CTA table entry");

	ste = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&steSz,
				"ctatbl_section%llu_offset%llu_sm",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(ste != NULL, -1, "Could not find SM table entry by CTA");

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&dteSz,
				"ctatbl_section%llu_offset%llu_dev",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(dte != NULL, -1, "Could not find Device table entry by CTA");

	for (i = 0; i < wte_count; ++i, wt += wteSz) {
		wte = (CudbgWarpTableEntry *)wt;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, wte, wteSz, false,
				      "wp%u_sm%u_dev%u",
				      wte->warpId,
				      ste->smId,
				      dte->devId))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, ctate, ctateSz, false,
				      "wp%u_sm%u_dev%u_cta",
				      wte->warpId,
				      ste->smId,
				      dte->devId))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, wte, wteSz, false,
				      "wptbl_section%llu_offset%llu",
                                      (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, ste, steSz, false,
				      "wptbl_section%llu_offset%llu_sm",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, dte, dteSz, false,
				      "wptbl_section%llu_offset%llu_dev",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;
	}

	return 0;
}

static int cuCoreReadThreadTable(CudaCore *cc, Elf_Scn *scn)
{
	size_t tt_count;
	uint8_t *tt;
	CudbgThreadTableEntry *tte;
	CudbgWarpTableEntry *wte;
	CudbgSmTableEntry *ste;
	CudbgDeviceTableEntry *dte;
	size_t dteSz, steSz, wteSz, tteSz;
	size_t parent, offset;
	size_t i;

	cuCoreGenericReadTable(cc->e, scn, &tt_count, (void **)&tt,
			       &tteSz, &parent, &offset);

	wte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&wteSz,
				"wptbl_section%llu_offset%llu",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(wte != NULL, -1, "Could not find Warp table entry");

	ste = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&steSz,
				"wptbl_section%llu_offset%llu_sm",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(ste != NULL, -1, "Could not find SM table entry by Warp");

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&dteSz,
				"wptbl_section%llu_offset%llu_dev",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(dte != NULL, -1, "Could not find Device table entry by Warp");

	for (i = 0; i < tt_count; ++i, tt += tteSz) {
		tte = (CudbgThreadTableEntry *)tt;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, tte, tteSz, false,
				      "ln%u_wp%u_sm%u_dev%u",
				      tte->ln, wte->warpId,
				      ste->smId, dte->devId))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, tte, tteSz, false,
				      "lntbl_section%llu_offset%llu",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, wte, wteSz, false,
				      "lntbl_section%llu_offset%llu_wp",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, ste, steSz, false,
				      "lntbl_section%llu_offset%llu_sm",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, dte, dteSz, false,
				      "lntbl_section%llu_offset%llu_dev",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;
	}

	return 0;
}

static int cuCoreReadBacktraceTable(CudaCore *cc, Elf_Scn *scn)
{
	size_t bt_count;
	uint8_t *bt;
	CudbgBacktraceTableEntry *bte;
	CudbgThreadTableEntry *tte;
	CudbgWarpTableEntry *wte;
	CudbgSmTableEntry *ste;
	CudbgDeviceTableEntry *dte;
	size_t dteSz, steSz, wteSz, tteSz, bteSz;
	size_t parent, offset;
	size_t i;

	cuCoreGenericReadTable(cc->e, scn, &bt_count, (void **)&bt,
			       &bteSz, &parent, &offset);

	tte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&tteSz,
				"lntbl_section%llu_offset%llu",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(tte != NULL, -1, "Could not find Thread table entry");

	wte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&wteSz,
				"lntbl_section%llu_offset%llu_wp",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(wte != NULL, -1, "Could not find Warp table entry by Lane");

	ste = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&steSz,
				"lntbl_section%llu_offset%llu_sm",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(ste != NULL, -1, "Could not find SM table entry by Lane");

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&dteSz,
				"lntbl_section%llu_offset%llu_dev",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(dte != NULL, -1, "Could not find Device table entry by Lane");

	for (i = 0; i < bt_count; ++i, bt += bteSz) {
		bte = (CudbgBacktraceTableEntry *)bt;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, bte, bteSz, false,
				      "bt%u_ln%u_wp%u_sm%u_dev%u",
				      bte->level, tte->ln,
				      wte->warpId, ste->smId,
				      dte->devId))
			return -1;
	}

	return 0;
}

static int cuCoreReadContextTable(CudaCore *cc, Elf_Scn *scn)
{
	size_t cte_count;
	uint8_t *ct;
	CudbgContextTableEntry *cte;
	CudbgDeviceTableEntry *dte;
	size_t dteSz, cteSz;
	size_t parent, offset;
	size_t i;
	CUDBGEvent event;

	if (cuCoreGenericReadTable(cc->e, scn,
				   &cte_count,
				   (void **)&ct,
				   &cteSz, &parent, &offset) != 0)
		return -1;

	for (i = 0; i < cte_count; ++i, ct += cteSz) {
		cte = (CudbgContextTableEntry *)ct;

		dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
					&dteSz,
					"devtbl_offset%u",
					cte->deviceIdx);
		VERIFY(dte != NULL, -1, "Could not find Device table entry");

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, cte, cteSz, false,
				      "ctx%llu_dev%u",
				      (unsigned long long)cte->contextId,
				      dte->devId))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, cte, cteSz, false,
				      "ctxtbl_section%llu_offset%llu",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;

		/* Add context created event */
		event.kind = CUDBG_EVENT_CTX_CREATE;
		event.cases.contextCreate.dev = dte->devId;
		event.cases.contextCreate.tid = cte->tid;
		event.cases.contextCreate.context =  cte->contextId;
		if (cuCoreAddEvent(cc, &event) != 0)
			return -1;

		DPRINTF(20, "sharedWindowBase = 0x%llX "
			"localWindowBase = 0x%llX "
			"globalWindowBase = 0x%llX\n",
			(unsigned long long int)cte->sharedWindowBase,
			(unsigned long long int)cte->localWindowBase,
			(unsigned long long int)cte->globalWindowBase);
	}

	return 0;
}

static int cuCoreReadModuleTable(CudaCore *cc, Elf_Scn *scn)
{
	size_t mt_count;
	uint8_t *mte;
	CudbgContextTableEntry *cte;
	size_t cteSz, mteSz;
	size_t parent, offset;
	size_t i;

	cuCoreGenericReadTable(cc->e, scn, &mt_count, (void **)&mte,
			       &mteSz, &parent, &offset);

	cte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				&cteSz,
				"ctxtbl_section%llu_offset%llu",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(cte != NULL, -1, "Could not find Context table entry");

	for (i = 0; i < mt_count; ++i, mte += mteSz) {
		if (cuCoreAddMapEntry(&cc->tableEntriesMap, mte, mteSz, false,
				      "modtbl_section%llu_offset%llu",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;

		if (cuCoreAddMapEntry(&cc->tableEntriesMap, cte, cteSz, false,
				      "modtbl_section%llu_offset%llu_ctx",
					  (unsigned long long)elfGetSectionIndex(cc->e, scn), (unsigned long long)i))
			return -1;
	}
	return 0;
}

static int cuCoreReadConstBanksTable(CudaCore *cc, Elf_Scn *scn)
{
	CudbgGridTableEntry *gte;
	CudbgDeviceTableEntry *dte;
	CudbgConstBankTableEntry *cbte;
	size_t cbte_count;
	size_t cbte_size;
	size_t cbte_grid_idx;
	size_t cbte_gte_sh_off;

	if (cuCoreGenericReadTable(cc->e, scn, &cbte_count, (void**)&cbte,
				   &cbte_size, &cbte_gte_sh_off, &cbte_grid_idx))
		return -1;

	gte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"grid_section%llu_offset%llu",
				(unsigned long long)cbte_gte_sh_off,
				(unsigned long long)cbte_grid_idx);
	VERIFY(gte != NULL, -1, "Could not find Grid table entry");

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"grid_section%llu_offset%llu_dev",
				(unsigned long long)cbte_gte_sh_off,
				(unsigned long long)cbte_grid_idx);
	VERIFY(dte != NULL, -1, "Could not find Device table entry by Grid");

	for (; cbte_count-- > 0; cbte++)
	{
		if (cuCoreAddMapEntry(&cc->tableEntriesMap, cbte, cbte_size, false,
				      "grid%llu_dev%u_cbank%u",
				      (unsigned long long)gte->gridId64,
				      dte->devId, cbte->bankId))
			return -1;
	}

	return 0;
}

static int cuCoreReadSharedMemorySection(CudaCore *cc, Elf_Scn *scn)
{
	CudbgSmTableEntry *ste;
	CudbgDeviceTableEntry *dte;
	Elf64_Shdr *shdr;
	size_t parent, offset;

	if (cuCoreReadSectionHeader(scn, &shdr) != 0)
		return -1;

	parent = shdr->sh_link;
	offset = shdr->sh_info;

	ste = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"ctatbl_section%llu_offset%llu_sm",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(ste != NULL, -1, "Could not find SM table entry");

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"ctatbl_section%llu_offset%llu_dev",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(dte != NULL, -1, "Could not find Device table entry");

	if (cuCoreAddMapEntry(&cc->tableEntriesMap, scn, shdr->sh_size, false,
			      "cta%llu_sm%u_dev%u_shared",
			      (unsigned long long)offset,
			      ste->smId,
			      dte->devId))
		return -1;

	return 0;
}

static int cuCoreReadLocalMemorySection(CudaCore *cc, Elf_Scn *scn)
{
	CudbgThreadTableEntry *tte;
	CudbgWarpTableEntry *wte;
	CudbgSmTableEntry *ste;
	CudbgDeviceTableEntry *dte;
	Elf64_Shdr *shdr;
	size_t parent, offset;

	if (cuCoreReadSectionHeader(scn, &shdr) != 0)
		return -1;

	parent = shdr->sh_link;
	offset = shdr->sh_info;

	tte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"lntbl_section%llu_offset%llu",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(tte != NULL, -1, "Could not find Thread table entry");

	wte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"lntbl_section%llu_offset%llu_wp",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(wte != NULL, -1, "Could not find Warp table entry by Lane");

	ste = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"lntbl_section%llu_offset%llu_sm",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(ste != NULL, -1, "Could not find SM table entry by Lane");

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"lntbl_section%llu_offset%llu_dev",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(dte != NULL, -1, "Could not find Device table entry by Lane");

	if (cuCoreAddMapEntry(&cc->tableEntriesMap, scn, shdr->sh_size, false,
			      "ln%u_wp%u_sm%u_dev%u_local",
			      tte->ln,
			      wte->warpId,
			      ste->smId,
			      dte->devId))
		return -1;

	return 0;
}

static int cuCoreReadParamMemorySection(CudaCore *cc, Elf_Scn *scn)
{
	CudbgGridTableEntry *gte;
	CudbgDeviceTableEntry *dte;
	Elf64_Shdr *shdr;
	size_t parent, offset;

	if (cuCoreReadSectionHeader(scn, &shdr) != 0)
		return -1;

	parent = shdr->sh_link;
	offset = shdr->sh_info;

	gte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"grid_section%llu_offset%llu",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(gte != NULL, -1, "Could not find Grid table entry");

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"grid_section%llu_offset%llu_dev",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(dte != NULL, -1, "Could not find Device table entry by Grid");

	if (cuCoreAddMapEntry(&cc->tableEntriesMap, scn, shdr->sh_size, false,
			      "grid%llu_dev%u_param",
			      (unsigned long long)gte->gridId64,
			      dte->devId))
		return -1;

	return 0;
}

static int cuCoreAddELFImage(CudaCoreELFImage **elfImageHead,
			     CudbgDeviceTableEntry *dte,
			     Elf *e, Elf_Scn *scn)
{
	CudaCoreELFImage *elf;

	elf = malloc(sizeof(*elf));
	VERIFY(elf != NULL, -1, "Could not allocate memory");

	elf->dte = dte;
	elf->e = e;
	elf->scn = scn;
	elf->next = *elfImageHead;

	*elfImageHead = elf;

	return 0;
}

static void cuCoreRemoveELFImage(CudaCoreELFImage **elf)
{
	CudaCoreELFImage *tmp = *elf;
	assert(tmp != NULL);
	*elf = tmp->next;
	free(tmp);
}

static int cuCoreReadELFImage(CudaCore *cc, Elf_Scn *scn, bool reloc)
{
	Elf64_Shdr *hdr;
	CudbgModuleTableEntry *mte;
	CudbgContextTableEntry *cte;
	CudbgDeviceTableEntry *dte;
	CUDBGEvent event;

	if (cuCoreReadSectionHeader(scn, &hdr))
		return -1;

	mte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"modtbl_section%u_offset%u",
				hdr->sh_link, hdr->sh_info);
	VERIFY(mte != NULL, -1, "Could not find Module table entry");

	/* Hash the ELFs SCN by module handle */
	if (cuCoreAddMapEntry(&cc->tableEntriesMap, scn, hdr->sh_size, false,
			      "%celf_handle%llx",
			      reloc ? 'r' : 'u',
			      (unsigned long long)readUint64(&mte->moduleHandle)))
		return -1;

	if (!reloc)
		return 0;

	cte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"modtbl_section%u_offset%u_ctx",
				hdr->sh_link, hdr->sh_info);
	VERIFY(cte != NULL, -1, "Could not find Context table entry");

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"devtbl_offset%u",
				cte->deviceIdx);
	VERIFY(dte != NULL, -1, "Could not find Device table entry");

	/* Add module loaded event */
	event.kind = CUDBG_EVENT_ELF_IMAGE_LOADED;
	event.cases.elfImageLoaded.dev = dte->devId;
	event.cases.elfImageLoaded.context = cte->contextId;
	event.cases.elfImageLoaded.module = readUint64(&mte->moduleHandle);
	event.cases.elfImageLoaded.size = readUint64(&hdr->sh_size);
	event.cases.elfImageLoaded.handle = readUint64(&mte->moduleHandle);
	if (cuCoreAddEvent(cc, &event) != 0)
		return -1;

	/* Add ELF image to list */
	if (cuCoreAddELFImage(&cc->relocatedELFImageHead, dte, cc->e, scn) != 0)
		return -1;

	return 0;
}

static int cuCoreReadUnrelocatedELFImage(CudaCore *cc, Elf_Scn *scn)
{
	return cuCoreReadELFImage(cc, scn, false);
}

static int cuCoreReadRelocatedELFImage(CudaCore *cc, Elf_Scn *scn)
{
	return cuCoreReadELFImage(cc, scn, true);
}

static int cuCoreReadWarpInfo(const char *key, CudaCore *cc, Elf_Scn *scn)
{
	CudbgWarpTableEntry *wte;
	CudbgSmTableEntry *ste;
	CudbgDeviceTableEntry *dte;
	Elf64_Shdr *shdr;
	size_t parent, offset;

	if (cuCoreReadSectionHeader(scn, &shdr) != 0)
		return -1;

	parent = shdr->sh_link;
	offset = shdr->sh_info;

	wte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"wptbl_section%llu_offset%llu",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(wte != NULL, -1, "Could not find Warp table entry by Lane");

	ste = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"wptbl_section%llu_offset%llu_sm",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(ste != NULL, -1, "Could not find SM table entry by Lane");

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"wptbl_section%llu_offset%llu_dev",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(dte != NULL, -1, "Could not find Device table entry by Lane");

	/* Hash by dev, sm, wp and ln indexes */
	if (cuCoreAddMapEntry(&cc->tableEntriesMap, scn, shdr->sh_size, false,
			      "%s_dev%u_sm%u_wp%u", key,
			      dte->devId, ste->smId, wte->warpId))
		return -1;

	return 0;
}

static int cuCoreReadThreadInfo(const char *key, CudaCore *cc, Elf_Scn *scn)
{
	CudbgThreadTableEntry *tte;
	CudbgWarpTableEntry *wte;
	CudbgSmTableEntry *ste;
	CudbgDeviceTableEntry *dte;
	Elf64_Shdr *shdr;
	size_t parent, offset;

	if (cuCoreReadSectionHeader(scn, &shdr) != 0)
		return -1;

	parent = shdr->sh_link;
	offset = shdr->sh_info;

	tte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"lntbl_section%llu_offset%llu",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(tte != NULL, -1, "Could not find Thread table entry");

	wte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"lntbl_section%llu_offset%llu_wp",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(wte != NULL, -1, "Could not find Warp table entry by Lane");

	ste = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"lntbl_section%llu_offset%llu_sm",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(ste != NULL, -1, "Could not find SM table entry by Lane");

	dte = cuCoreGetMapEntry(&cc->tableEntriesMap,
				NULL,
				"lntbl_section%llu_offset%llu_dev",
				(unsigned long long)parent, (unsigned long long)offset);
	VERIFY(dte != NULL, -1, "Could not find Device table entry by Lane");

	/* Hash by dev, sm, wp and ln indexes */
	if (cuCoreAddMapEntry(&cc->tableEntriesMap, scn, shdr->sh_size, false,
			      "%s_dev%u_sm%u_wp%u_ln%u", key,
			      dte->devId, ste->smId, wte->warpId, tte->ln))
		return -1;

	return 0;
}

const char *cuCoreGetStrTabByIndex(CudaCore *cc, size_t idx)
{
	return elfGetString(cc->e, cc->strndx, idx);
}

static int cuCoreProcessSection(CudaCore *cc, bool *processed, Elf_Scn *scn)
{
	size_t ndxscn;
	Elf64_Shdr *shdr;
	const char *name;

	ndxscn = elfGetSectionIndex(cc->e, scn);
	VERIFY(ndxscn != SHN_UNDEF, -1, "elfGetSectionIndex() failed: %s",
	       elfErrorMsg());

	/* Avoid processing the same section twice */
	if (processed[ndxscn])
		return 0;

	shdr = elfGetSectionHeader(scn);
	VERIFY(shdr != NULL, -1, "elfGetSectionHeader() failed: %s",
	       elfErrorMsg());

	name = elfGetString(cc->e, cc->shstrndx, shdr->sh_name);
	VERIFY(name != NULL, -1, "elfGetString() failed: %s", elfErrorMsg());

	/* Process parent section first */
	if (shdr->sh_link != 0) {
		Elf_Scn *parent_scn;

		parent_scn = elfGetSection(cc->e, shdr->sh_link);
		VERIFY(parent_scn != NULL, -1,
		       "Could not find parent section '%u'", shdr->sh_link);

		cuCoreProcessSection(cc, processed, parent_scn);
	}

	DPRINTF(40, "Processing section %s (#%llu)\n", name, (unsigned long long)ndxscn);

	/* Mark section as processed */
	processed[ndxscn] = true;

	/* Handle string table special way */
	if (strcmp(name, ".strtab") == 0) {
		cc->strndx = ndxscn;
		return 0;
	}

	/* Ignore section header string table */
	if (strcmp(name, ".shstrtab") == 0)
		return 0;

	/* Handle Cudbg tables */
	switch (shdr->sh_type) {
	case CUDBG_SHT_MANAGED_MEM:
		return cuCoreReadMemorySection(cc->e, scn, cc->managedMemorySegs);
	case CUDBG_SHT_GLOBAL_MEM:
		return cuCoreReadMemorySection(cc->e, scn, cc->globalMemorySegs);
	case CUDBG_SHT_SHARED_MEM:
		return cuCoreReadSharedMemorySection(cc, scn);
	case CUDBG_SHT_LOCAL_MEM:
		return cuCoreReadLocalMemorySection(cc, scn);
	case CUDBG_SHT_PARAM_MEM:
		return cuCoreReadParamMemorySection(cc, scn);
	case CUDBG_SHT_DEV_TABLE:
		return cuCoreReadDeviceTable(cc, scn);
	case CUDBG_SHT_GRID_TABLE:
		return cuCoreReadGridTable(cc, scn);
	case CUDBG_SHT_WP_TABLE:
		return cuCoreReadWarpTable(cc, scn);
	case CUDBG_SHT_LN_TABLE:
		return cuCoreReadThreadTable(cc, scn);
	case CUDBG_SHT_MOD_TABLE:
		return cuCoreReadModuleTable(cc, scn);
	case CUDBG_SHT_ELF_IMG:
		return cuCoreReadUnrelocatedELFImage(cc, scn);
	case CUDBG_SHT_RELF_IMG:
		return cuCoreReadRelocatedELFImage(cc, scn);
	case CUDBG_SHT_BT:
		return cuCoreReadBacktraceTable(cc, scn);
	case CUDBG_SHT_CTX_TABLE:
		return cuCoreReadContextTable(cc, scn);
	case CUDBG_SHT_SM_TABLE:
		return cuCoreReadSmTable(cc, scn);
	case CUDBG_SHT_CTA_TABLE:
		return cuCoreReadCTATable(cc, scn);
	case CUDBG_SHT_DEV_REGS:
		return cuCoreReadThreadInfo("regs", cc, scn);
	case CUDBG_SHT_DEV_PRED:
		return cuCoreReadThreadInfo("pred", cc, scn);
	case CUDBG_SHT_DEV_UREGS:
		return cuCoreReadWarpInfo("uregs", cc, scn);
	case CUDBG_SHT_DEV_UPRED:
		return cuCoreReadWarpInfo("upred", cc, scn);
	case CUDBG_SHT_CBU_BAR:
		return cuCoreReadWarpInfo("cbu_bar", cc, scn);
	case CUDBG_SHT_CB_TABLE:
		return cuCoreReadConstBanksTable(cc, scn);
	default:
		DPRINTF(5, "Found section of unknown type (0x%x)\n",
			shdr->sh_type);
		break;
	}

	return 0;
}

static int cuCoreReadSections(CudaCore *cc)
{
	Elf_Scn *scn = NULL;
	bool *processed;
	int ret = 0;

	VERIFY(elfGetSectionHeaderStrTblIdx(cc->e, &cc->shstrndx) == 0, -1,
	       "elfGetSectionHeaderStrTblIdx() failed: %s", elfErrorMsg());

	VERIFY(elfGetSectionHeaderNum(cc->e, &cc->shnum) == 0, -1,
	       "elfGetSectionHeaderNum() failed: %s", elfErrorMsg());

	DPRINTF(10, "Found %llu sections.\n", (unsigned long long)cc->shnum);

	processed = calloc(cc->shnum, sizeof(*processed));
	VERIFY(processed != NULL, -1, "Could not allocate memory");

	while ((scn = elfGetNextSection(cc->e, scn)) != NULL) {
		if (cuCoreProcessSection(cc, processed, scn) != 0) {
			ret = -1;
			goto cleanup;
		}
	}

cleanup:
	free(processed);
	return ret;
}

static int cuCoreAddEvent(CudaCore *cc, CUDBGEvent *event)
{
	CudaCoreEvent *evt = NULL;
	CudaCoreEvent **pevt = &cc->eventHead;

	evt = malloc(sizeof(*evt));
	VERIFY(evt != NULL, -1, "Could not allocate memory");

	evt->event = *event;
	evt->next = NULL;

	/* Add even to the tail */
	while (*pevt != NULL)
		pevt = &(*pevt)->next;
	*pevt = evt;

	return 0;
}

const CUDBGEvent *cuCoreGetEvent(CudaCore *cc)
{
	if (cc->eventHead == NULL)
		return NULL;
	return &cc->eventHead->event;
}

int cuCoreDeleteEvent(CudaCore *cc)
{
	CudaCoreEvent *evt = cc->eventHead;
	if (evt == NULL)
		return -1;
	cc->eventHead = evt->next;
	free(evt);
	return 0;
}

size_t cuCoreGetNumDevices(CudaCore *cc)
{
	return cc->numDevices;
}

typedef int (*Cb)(CudaCore *cc, cs_t *callStack);
typedef int (*ProcessELF)(CudaCore *cc, Elf *e, cs_t *callStack);
typedef int (*ProcessSection)(CudaCore *cc, Elf *e, Elf_Scn *scn,
			      Elf64_Shdr *shdr, cs_t *callStack);
typedef int (*ProcessSymbol)(CudaCore *cc, Elf *e, Elf_Scn *scn,
			     Elf64_Shdr *shdr, Elf64_Sym *sym,
			     cs_t *callStack);
typedef int (*ProcessSymbolSection)(CudaCore *cc, Elf *e, Elf_Scn *scn,
				    Elf64_Shdr *shdr, Elf64_Sym *sym,
				    Elf_Scn *symscn, Elf64_Shdr *symshdr,
				    cs_t *callStack);

#define POP(callStack, type)		((type)(*((callStack)++)))
#define POP_PTR(callStack, type)	((type)(uintptr_t)(*((callStack)++)))

void cuCoreExecuteCallStack(CudaCore *cc, cs_t *callStack)
{
	Cb cb = POP_PTR(callStack, Cb);

	if (cb(cc, callStack) == -1) {
		cuCoreSetErrorMsg("Call stack execution failed: %s",
				  cuCoreErrorMsg());
	}
}

int cuCoreIterateELFImages(CudaCore *cc, cs_t *callStack)
{
	CudaCoreELFImage *elfImage;
	Elf *e;
	int ret = 0;

	bool relocated = POP(callStack, bool);
	ProcessELF processELF = POP_PTR(callStack, ProcessELF);

	for (elfImage = relocated ? cc->relocatedELFImageHead : NULL;
			elfImage != NULL && ret == 0;
			elfImage = elfImage->next) {
		Elf_Data data;

		if (cuCoreReadSectionData(elfImage->e, elfImage->scn, &data) != 0) {
			DPRINTF(1, "Skipping: %s\n", cuCoreErrorMsg());
			continue;
		}

		e = elfOpenInMemory(data.d_buf, data.d_size, NULL);
		if (e == NULL) {
			DPRINTF(1, "Skipping: Failed loading ELF from memory: %s\n",
				elfErrorMsg());
			continue;
		}

		ret = processELF(cc, e, callStack);

		elfFree(e);
	}

	return ret;
}

int cuCoreIterateELFSections(CudaCore *cc, Elf *e, cs_t *callStack)
{
	Elf_Scn *scn = NULL;
	int ret = 0;

	ProcessSection processSection = POP_PTR(callStack, ProcessSection);

	while ((scn = elfGetNextSection(e, scn)) != NULL && ret == 0) {
		Elf64_Shdr *shdr;

		shdr = elfGetSectionHeader(scn);
		if (shdr == NULL) {
			DPRINTF(1, "Skipping: Could not get section header: %s\n",
				elfErrorMsg());
			continue;
		}

		ret = processSection(cc, e, scn, shdr, callStack);
	}

	return ret;
}

int cuCoreIterateSymbolTable(CudaCore *cc, Elf *e, Elf_Scn *scn,
			     Elf64_Shdr *shdr, cs_t *callStack)
{
	Elf_Data symtab_data;
	int ret = 0;
	int idx, symsz = sizeof(Elf64_Sym);
	Elf64_Sym *sym;

	ProcessSymbol processSymbol = POP_PTR(callStack, ProcessSymbol);

	if (shdr->sh_type != SHT_SYMTAB)
		return 0;

	if (cuCoreReadSectionData(e, scn, &symtab_data) != 0)
		return -1;

	sym = symtab_data.d_buf;

	DPRINTF(40, "Found %llu symbols inside '%s' section\n",
		(unsigned long long)symtab_data.d_size / symsz,
		elfGetString(e, shdr->sh_link, shdr->sh_name));

	for (idx = 0; idx * symsz < symtab_data.d_size && ret == 0;
			++idx, ++sym)
		ret = processSymbol(cc, e, scn, shdr, sym, callStack);

	return ret;
}

int cuCoreFilterSymbolByName(CudaCore *cc, Elf *e, Elf_Scn *scn,
			     Elf64_Shdr *shdr, Elf64_Sym *sym,
			     cs_t *callStack)
{
	char *symName = POP_PTR(callStack, char *);
	bool *symFound = POP_PTR(callStack, bool *);
	ProcessSymbol processSymbol = POP_PTR(callStack, ProcessSymbol);

	if (strcmp(symName, elfGetString(e, shdr->sh_link, sym->st_name)) == 0) {
		if (symFound != NULL)
			*symFound = true;

		if (processSymbol != NULL)
			return processSymbol(cc, e, scn, shdr, sym, callStack);

		return 1;
	}

	return 0;
}

int cuCoreFilterSymbolByAddress(CudaCore *cc, Elf *e, Elf_Scn *scn,
			        Elf64_Shdr *shdr, Elf64_Sym *sym,
			        cs_t *callStack)
{
	uint64_t lbound, ubound;
	uint64_t addr = POP(callStack, uint64_t);
	uint32_t sz = POP(callStack, uint32_t);
	bool *symFound = POP_PTR(callStack, bool *);
	ProcessSymbol processSymbol = POP_PTR(callStack, ProcessSymbol);

	DPRINTF(60, "Found symbol '%s' "
		"type=%u value=0x%llx size=0x%llx\n",
		elfGetString(e, readUint32(&shdr->sh_link), readUint32(&sym->st_name)),
		ELF64_ST_TYPE(sym->st_info),
		(unsigned long long)readUint64(&sym->st_value),
		(unsigned long long)readUint64(&sym->st_size));

	lbound = readUint64(&sym->st_value);
	ubound = readUint64(&sym->st_value) + readUint64(&sym->st_size);

	if (addr >= lbound && addr + sz <= ubound) {
		if (symFound != NULL)
			*symFound = true;

		if (processSymbol != NULL)
			return processSymbol(cc, e, scn, shdr, sym, callStack);

		return 1;
	}

	return 0;
}

int cuCoreFilterSymbolByType(CudaCore *cc, Elf *e, Elf_Scn *scn,
			     Elf64_Shdr *shdr, Elf64_Sym *sym,
			     cs_t *callStack)
{
	unsigned char symType = POP(callStack, unsigned char);
	bool *symFound = POP_PTR(callStack, bool *);
	ProcessSymbol processSymbol = POP_PTR(callStack, ProcessSymbol);

	if (ELF64_ST_TYPE(sym->st_info) != symType)
		return 0;

	if (symFound != NULL)
		*symFound = true;

	if (processSymbol != NULL)
		return processSymbol(cc, e, scn, shdr, sym, callStack);

	return 1;
}

int cuCoreReadSymbolAddress(CudaCore *cc, Elf *e, Elf_Scn *scn,
			    Elf64_Shdr *shdr, Elf64_Sym *sym,
			    cs_t *callStack)
{
	uintptr_t *symAddr = POP_PTR(callStack, uintptr_t *);
	ProcessSymbol processSymbol = POP_PTR(callStack, ProcessSymbol);

	DPRINTF(60, "Reading symbol '%s' "
		"type=%u value=0x%llx size=0x%llx\n",
		elfGetString(e, shdr->sh_link, sym->st_name),
		ELF64_ST_TYPE(sym->st_info),
		(unsigned long long)sym->st_value,
		(unsigned long long)sym->st_size);

	if (symAddr != NULL)
		*symAddr = sym->st_value;

	if (processSymbol != NULL)
		return processSymbol(cc, e, scn, shdr, sym, callStack);

	return 1;
}

int cuCoreReadSymbolSection(CudaCore *cc, Elf *e, Elf_Scn *scn,
			    Elf64_Shdr *shdr, Elf64_Sym *sym,
			    cs_t *callStack)
{
	Elf_Scn *symscn;
	Elf64_Shdr *symshdr;

	Elf64_Shdr **symshdrRet = POP_PTR(callStack, Elf64_Shdr **);
	ProcessSymbolSection processSymbolSection = POP_PTR(callStack, ProcessSymbolSection);

	symscn = elfGetSection(e, sym->st_shndx);
	VERIFY(symscn != NULL, -1, "Could not find symbol section '%u'",
	       sym->st_shndx);

	symshdr = elfGetSectionHeader(symscn);
	VERIFY(symshdr != NULL, -1, "elfGetSectionHeader() failed: %s",
	       elfErrorMsg());

	if (symshdrRet != NULL)
		*symshdrRet = symshdr;

	if (processSymbolSection != NULL)
		return processSymbolSection(cc, e, scn, shdr, sym, symscn, symshdr, callStack);

	return 1;
}

int cuCoreReadSymbolData(CudaCore *cc, Elf *e, Elf_Scn *scn,
			 Elf64_Shdr *shdr, Elf64_Sym *sym,
			 Elf_Scn *symscn, Elf64_Shdr *symshdr,
			 cs_t *callStack)
{
	Elf_Data data;
	uint64_t offset;

	uint64_t addr = POP(callStack, uint64_t);
	void *buf = POP_PTR(callStack, void *);
	uint32_t sz = POP(callStack, uint32_t);
	CUDBGResult *result = POP_PTR(callStack, CUDBGResult *);
	ProcessSymbolSection processSymbolSection = POP_PTR(callStack, ProcessSymbolSection);

	if (cuCoreReadSectionData(e, symscn, &data) != 0)
		return -1;

	DPRINTF(60, "Reading symbol '%s' data "
		"type=%u value=0x%llx size=0x%llx\n",
		elfGetString(e, shdr->sh_link, sym->st_name),
		ELF64_ST_TYPE(sym->st_info),
		(unsigned long long)sym->st_value,
		(unsigned long long)sym->st_size);

	*result = CUDBG_ERROR_UNKNOWN;

	offset = addr - sym->st_value;

	if (offset + sz > data.d_size) {
		*result = CUDBG_ERROR_INVALID_MEMORY_ACCESS;
		return 1; /* Symbol was found, hence returning 1 */
	}

	memcpy(buf, (char *)data.d_buf + offset, sz);

	*result = CUDBG_SUCCESS;

	if (processSymbolSection != NULL)
		return processSymbolSection(cc, e, scn, shdr, sym, symscn, symshdr, callStack);

	return 1;
}

static int cuCoreInit(CudaCore *cc)
{
	Elf64_Ehdr *ehdr;

	/* Verify core dump ELF file */
	ehdr = elfGetHeader(cc->e);
	VERIFY(ehdr != NULL, -1, "elfGetHeader() failed: %s", elfErrorMsg());

	VERIFY(ehdr->e_type == ET_CORE, -1,
	       "Wrong ELF 'type': expected %d, got %d",
	       ET_CORE, ehdr->e_type);

	VERIFY(ehdr->e_machine == EM_CUDA, -1,
	       "Wrong ELF 'machine': expected %d, got %d",
	       EM_CUDA, ehdr->e_machine);

	utarray_new(cc->managedMemorySegs, &memorySeg_icd);
	utarray_new(cc->globalMemorySegs, &memorySeg_icd);
	utarray_reserve(cc->globalMemorySegs, GLOBAL_MEMORY_SEGMENTS_MIN);

	if (cuCoreReadSections(cc))
		return -1;

	/* Sort memory sections so that we can binary search by address */
	utarray_sort(cc->managedMemorySegs, cuCoreSortMemorySegs);
	utarray_sort(cc->globalMemorySegs, cuCoreSortMemorySegs);

	{ /* Get statistic on generic hash map */
		unsigned int entries = HASH_COUNT(cc->tableEntriesMap);
		DPRINTF(10, "Table entries map contains %d elements.\n",
			entries);
	}

	return 0;
}

void cuCoreFree(CudaCore *cc)
{
	assert(cc != NULL);

	/* Cleanup any left events */
	while (cc->eventHead != NULL)
		cuCoreDeleteEvent(cc);

	/* Cleanup ELF image list */
	while (cc->relocatedELFImageHead != NULL)
		cuCoreRemoveELFImage(&cc->relocatedELFImageHead);

	{ /* Cleanup table entries map */
		MapEntry *mapEntry, *tmp;
		HASH_ITER(hh, cc->tableEntriesMap, mapEntry, tmp) {
			HASH_DEL(cc->tableEntriesMap, mapEntry);
                        if (mapEntry->needsFree)
                            free(mapEntry->entryPtr);
			free(mapEntry);
		}
	}

	/* Cleanup memory segments arrays */
	if (cc->managedMemorySegs)
		utarray_free(cc->managedMemorySegs);
	if (cc->globalMemorySegs)
		utarray_free(cc->globalMemorySegs);

	if (cc->e != NULL)
		elfFree(cc->e);

	free(cc);
}
