/*
 * Copyright (c) 2014-2024 NVIDIA CORPORATION. All rights reserved.
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

#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "common.h"

#define DEF_API_CALL(name)	static CUDBGResult cuCoreApi_##name
#define API_CALL(name)		cuCoreApi_##name
#define PTR2CS(ptr)		((cs_t)(uintptr_t)(ptr))

#ifdef _WIN32
#include <fcntl.h>

static int mkstemp(char *name)
{
	if (_mktemp_s (name, TMPBUF_LEN))
		return -1;
	return open (name, O_RDWR);
}
#endif

static __THREAD CudaCore *curcc;

static CUDBGResult determineCTAId(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *cta)
{
	CudbgCTATableEntry *ctate, *ctate1;
	uint32_t ctateIdx = 0;

	GET_TABLE_ENTRY(ctate, NULL, CUDBG_ERROR_UNKNOWN,
			"wp%u_sm%u_dev%u_cta", wp, sm, dev);

	do {
		ctate1 = cuCoreGetMapEntry(&curcc->tableEntriesMap, NULL, "cta%u_sm%u_dev%u", ctateIdx++, sm, dev);
	} while (ctate1 != NULL && ctate1 != ctate);

	*cta = ctateIdx-1;
	return ctate1 == ctate ? CUDBG_SUCCESS: CUDBG_ERROR_INTERNAL;
}

static CUDBGResult getGridId(uint32_t dev, uint32_t sm, uint32_t wp, uint64_t *gridId)
{
	CudbgCTATableEntry *ctate;
	CudbgWarpTableEntry *wte;

	GET_TABLE_ENTRY(wte, NULL, CUDBG_ERROR_INVALID_WARP,
			"wp%u_sm%u_dev%u", wp, sm, dev);

	GET_TABLE_ENTRY(ctate, NULL, CUDBG_ERROR_UNKNOWN,
			"wp%u_sm%u_dev%u_cta", wp, sm, dev);

	*gridId = ctate->gridId64;
	return CUDBG_SUCCESS;
}

DEF_API_CALL(doNothing)()
{
	return CUDBG_SUCCESS;
}

DEF_API_CALL(notSupported)()
{
	return CUDBG_ERROR_NOT_SUPPORTED;
}

DEF_API_CALL(getElfImage32)(uint32_t dev, uint32_t sm, uint32_t wp,
			    bool relocated, void **elfImage, uint32_t *size)
{
	return API_CALL(notSupported)();
}

DEF_API_CALL(getElfImage)(uint32_t dev, uint32_t sm, uint32_t wp,
			  bool relocated, void **elfImage, uint64_t *size)
{
	CUDBGResult rc;
	uint64_t gridId;
	CudbgGridTableEntry *gte;
	Elf_Scn *scn;
	Elf_Data data;

	TRACE_FUNC("dev=%u sm=%u wp=%u relocated=%u elfImage=%p size=%p",
		   dev, sm, wp, relocated, elfImage, size);

	VERIFY_ARG(elfImage);
	VERIFY_ARG(size);

	if ((rc = getGridId(dev, sm, wp, &gridId)) != CUDBG_SUCCESS)
		return rc;

	GET_TABLE_ENTRY(gte, NULL, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", gridId, dev);

	GET_TABLE_ENTRY(scn, NULL, CUDBG_ERROR_INVALID_MODULE,
			"%celf_handle%llx",
			relocated ? 'r' : 'u', gte->moduleHandle);

	if (cuCoreReadSectionData(curcc->e, scn, &data) != 0)
		return CUDBG_ERROR_UNKNOWN;

	*size = data.d_size;
	*elfImage = data.d_buf;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getNumDevices)(uint32_t *numDev)
{
	TRACE_FUNC("numDev=%p", numDev);

	VERIFY_ARG(numDev);

	*numDev = cuCoreGetNumDevices(curcc);

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getNumSMs)(uint32_t dev, uint32_t *numSMs)
{
	CudbgDeviceTableEntry *dte;

	TRACE_FUNC("dev=%u numSMs=%p", dev, numSMs);

	VERIFY_ARG(numSMs);

	GET_TABLE_ENTRY(dte, NULL, CUDBG_ERROR_INVALID_DEVICE, "dev%u", dev);

	*numSMs = dte->numSMs;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getNumWarps)(uint32_t dev, uint32_t *numWarps)
{
	CudbgDeviceTableEntry *dte;

	TRACE_FUNC("dev=%u numWarps=%p", dev, numWarps);

	VERIFY_ARG(numWarps);

	GET_TABLE_ENTRY(dte, NULL, CUDBG_ERROR_INVALID_DEVICE, "dev%u", dev);

	*numWarps = dte->numWarpsPerSM;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getNumLanes)(uint32_t dev, uint32_t *numLanes)
{
	CudbgDeviceTableEntry *dte;

	TRACE_FUNC("dev=%u numLanes=%p", dev, numLanes);

	VERIFY_ARG(numLanes);

	GET_TABLE_ENTRY(dte, NULL, CUDBG_ERROR_INVALID_DEVICE, "dev%u", dev);

	*numLanes = dte->numLanesPerWarp;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getNumRegisters)(uint32_t dev, uint32_t *numRegs)
{
	CudbgDeviceTableEntry *dte;

	TRACE_FUNC("dev=%u numRegs=%p", dev, numRegs);

	VERIFY_ARG(numRegs);

	GET_TABLE_ENTRY(dte, NULL, CUDBG_ERROR_INVALID_DEVICE, "dev%u", dev);

	*numRegs = dte->numRegsPerLane;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(lookupDeviceCodeSymbol)(char *symName, bool *symFound,
				     uintptr_t *symAddr)
{
	/* FIXME take current context into account */

	cs_t callStack[] = {
		PTR2CS(cuCoreIterateELFImages), true,
		PTR2CS(cuCoreIterateELFSections),
		PTR2CS(cuCoreIterateSymbolTable),
		PTR2CS(cuCoreFilterSymbolByName), PTR2CS(symName), PTR2CS(symFound),
		PTR2CS(cuCoreReadSymbolAddress), PTR2CS(symAddr),
		PTR2CS(NULL),
	};

	TRACE_FUNC("symName='%s' symFound=%p symAddr=%p",
		   symName, symFound, symAddr);

	VERIFY_ARG(symName);
	VERIFY_ARG(symFound);
	VERIFY_ARG(symAddr);

	*symFound = false;
	*symAddr = 0x0;

	cuCoreExecuteCallStack(curcc, callStack);

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readCodeMemory)(uint32_t dev, uint64_t addr, void *buf,
			     uint32_t sz)
{
	CUDBGResult ret = CUDBG_ERROR_INVALID_MEMORY_ACCESS;

	cs_t callStack[] = {
		PTR2CS(cuCoreIterateELFImages), true,
		PTR2CS(cuCoreIterateELFSections),
		PTR2CS(cuCoreIterateSymbolTable),
		PTR2CS(cuCoreFilterSymbolByType), STT_FUNC, PTR2CS(NULL),
		PTR2CS(cuCoreFilterSymbolByAddress), addr, sz, PTR2CS(NULL),
		PTR2CS(cuCoreReadSymbolSection), PTR2CS(NULL),
		PTR2CS(cuCoreReadSymbolData), addr, PTR2CS(buf), sz, PTR2CS(&ret),
		PTR2CS(NULL),
	};

	TRACE_FUNC("dev=%u addr=0x%llx buf=%p sz=%u", dev, addr, buf, sz);

	VERIFY_ARG(buf);

	memset(buf, 0, sz);

	cuCoreExecuteCallStack(curcc, callStack);

	return ret;
}

DEF_API_CALL(readGlobalMemory)(uint64_t addr, void *buf, uint32_t sz)
{
	MemorySeg *memorySeg, memorySegToFind;
	Elf_Data data;

	TRACE_FUNC("addr=0x%llx buf=%p sz=%u", addr, buf, sz);

	VERIFY_ARG(buf);

	memorySegToFind.address = addr;
	memorySegToFind.size = sz;

	memorySeg = utarray_find(curcc->globalMemorySegs, &memorySegToFind,
				 cuCoreSortMemorySegs);
	if (memorySeg == NULL)
		return CUDBG_ERROR_MISSING_DATA;

	if (addr < memorySeg->address ||
			addr + sz > memorySeg->address + memorySeg->size)
		return CUDBG_ERROR_INVALID_MEMORY_ACCESS;

	if (cuCoreReadSectionData(memorySeg->e, memorySeg->scn, &data) != 0)
		return CUDBG_ERROR_UNKNOWN;

	memcpy(buf, (char *)data.d_buf + (addr - memorySeg->address), sz);

	return CUDBG_SUCCESS;

}

DEF_API_CALL(readSharedMemory)(uint32_t dev, uint32_t sm, uint32_t wp,
			       uint64_t addr, void *buf, uint32_t sz)
{
	uint32_t ctaId;
	Elf_Scn *scn;
	Elf_Data data;
	CUDBGResult rc;

	TRACE_FUNC("dev=%u sm=%u wp=%u addr=0x%llx buf=%p sz=%u",
		   dev, sm, wp, addr, buf, sz);

	VERIFY_ARG(buf);

	rc = determineCTAId (dev, sm, wp, &ctaId);
	if (rc != CUDBG_SUCCESS)
		return rc;

	GET_TABLE_ENTRY(scn, NULL, CUDBG_ERROR_MISSING_DATA,
			"cta%u_sm%u_dev%u_shared", ctaId, sm, dev);

	if (cuCoreReadSectionData(curcc->e, scn, &data) != 0)
		return CUDBG_ERROR_UNKNOWN;

	if (data.d_size < addr || sz > data.d_size - addr)
		return CUDBG_ERROR_INVALID_MEMORY_ACCESS;

	memcpy(buf, (char *)data.d_buf + addr, sz);

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readLocalMemory)(uint32_t dev, uint32_t sm, uint32_t wp,
			      uint32_t ln, uint64_t addr, void *buf,
			      uint32_t sz)
{
	Elf_Scn *scn;
	Elf64_Shdr *shdr;
	Elf_Data data;
	uint64_t offset;

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u addr=0x%llx buf=%p sz=%u",
		   dev, sm, wp, ln, addr, buf, sz);

	VERIFY_ARG(buf);

	GET_TABLE_ENTRY(scn, NULL, CUDBG_ERROR_MISSING_DATA,
			"ln%u_wp%u_sm%u_dev%u_local", ln, wp, sm, dev);

	if (cuCoreReadSectionData(curcc->e, scn, &data) != 0)
		return CUDBG_ERROR_UNKNOWN;

	if (cuCoreReadSectionHeader(scn, &shdr) != 0)
		return CUDBG_ERROR_UNKNOWN;

        TRACE_FUNC("shdr->sh_addr=0x%llx addr=0x%llx\n", shdr->sh_addr, addr);
	if (shdr->sh_addr > addr)
		return CUDBG_ERROR_INVALID_MEMORY_ACCESS;

	offset = addr - shdr->sh_addr;

	if (data.d_size < offset || sz > data.d_size - offset)
		return CUDBG_ERROR_INVALID_MEMORY_ACCESS;

	memcpy(buf, (char *)data.d_buf + offset, sz);

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readGenericMemory)(uint32_t dev, uint32_t sm, uint32_t wp,
				uint32_t ln, uint64_t addr, void *buf,
				uint32_t sz)
{
	CUDBGResult rc;
	uint64_t gridId;
	CudbgGridTableEntry *gte;
	CudbgContextTableEntry *cte;

	VERIFY_ARG(buf);

	if ((rc = getGridId(dev, sm, wp, &gridId)) != CUDBG_SUCCESS)
		return rc;

	GET_TABLE_ENTRY(gte, NULL, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", gridId, dev);

	GET_TABLE_ENTRY(cte, NULL, CUDBG_ERROR_INVALID_CONTEXT,
			"ctx%llu_dev%u", gte->contextId, dev);

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u addr=0x%llx buf=%p sz=%u lmemBase=0x%llx smemBase=0x%llx",
		   dev, sm, wp, ln, addr, buf, sz, cte->localWindowBase, cte->sharedWindowBase);

    // TODO: Fix me to use sharedWindow & localWindow sizes in the version 2 corefiles
    if (cte->sharedWindowBase <= addr && (addr-cte->sharedWindowBase) < CUDA_COREDUMP_SHARED_WINDOW_SIZE)
        return API_CALL(readSharedMemory)(dev, sm, wp,
                                          addr - cte->sharedWindowBase,
                                          buf, sz);
    else if (cte->localWindowBase <= addr && (addr-cte->localWindowBase) < CUDA_COREDUMP_LOCAL_WINDOW_SIZE)
        return API_CALL(readLocalMemory)(dev, sm, wp, ln,
                                         addr - cte->localWindowBase,
                                         buf, sz);
    else
        return API_CALL(readGlobalMemory)(addr, buf, sz);
}

DEF_API_CALL(readParamMemory)(uint32_t dev, uint32_t sm, uint32_t wp,
			      uint64_t addr, void *buf, uint32_t sz)
{
	CUDBGResult rc;
	uint64_t gridId;
	CudbgGridTableEntry *gte;
	Elf_Scn *scn;
	Elf_Data data;
	uint64_t offset;

	TRACE_FUNC("dev=%u sm=%u wp=%u addr=0x%llx buf=%p sz=%u",
		   dev, sm, wp, addr, buf, sz);

	VERIFY_ARG(buf);

	if ((rc = getGridId(dev, sm, wp, &gridId)) != CUDBG_SUCCESS)
		return rc;

	GET_TABLE_ENTRY(gte, NULL, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", gridId, dev);

	GET_TABLE_ENTRY(scn, NULL, CUDBG_ERROR_UNKNOWN,
			"grid%llu_dev%u_param", gridId, dev);

	if (cuCoreReadSectionData(curcc->e, scn, &data) != 0)
		return CUDBG_ERROR_UNKNOWN;

	if (gte->paramsOffset > addr)
		return CUDBG_ERROR_INVALID_MEMORY_ACCESS;

	offset = addr - gte->paramsOffset;

	if (data.d_size < offset || sz > data.d_size - offset)
		return CUDBG_ERROR_INVALID_MEMORY_ACCESS;

	memcpy(buf, (char *)data.d_buf + offset, sz);

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readConstMemory)(uint32_t dev, uint64_t addr, void *buf,
			      uint32_t sz)
{
	TRACE_FUNC("dev=%u addr=0x%llx buf=%p sz=%u", dev, addr, buf, sz);

	return API_CALL(readGenericMemory)(dev, 0, 0, 0, addr, buf, sz);
}

DEF_API_CALL(readWarpState)(uint32_t devId, uint32_t sm, uint32_t wp,
			    CUDBGWarpState *state)
{
	uint32_t numLanes;
	uint32_t ln;
	CudbgWarpTableEntry *wte;
	CudbgGridTableEntry *gte;
	CudbgCTATableEntry *ctate;
	CudbgThreadTableEntry *tte;
	CudbgSmTableEntry *smte;
	CUDBGResult rc;
	size_t ctateSize;
	size_t smteSize;

	TRACE_FUNC("devId=%u sm=%u wp=%u state=%p", devId, sm, wp, state);

	VERIFY_ARG(state);

	GET_TABLE_ENTRY(wte, NULL, CUDBG_ERROR_INVALID_WARP,
			"wp%u_sm%u_dev%u", wp, sm, devId);

	GET_TABLE_ENTRY(ctate, &ctateSize, CUDBG_ERROR_UNKNOWN,
			"wp%u_sm%u_dev%u_cta", wp, sm, devId);

	GET_TABLE_ENTRY(gte, NULL, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", ctate->gridId64, devId);

	GET_TABLE_ENTRY (smte, &smteSize, CUDBG_ERROR_INVALID_SM, "sm%u_dev%u",
			 sm, devId);

	memset(state, 0, sizeof(*state));
	state->gridId = gte->gridId64;
	state->errorPC = wte->errorPC;
	if (offsetof(CudbgCTATableEntry, clusterIdxZ) < ctateSize)
	{
		state->clusterIdx.x = ctate->clusterIdxX;
		state->clusterIdx.y = ctate->clusterIdxY;
		state->clusterIdx.z = ctate->clusterIdxZ;
	}
	if (offsetof(CudbgCTATableEntry, clusterDimZ) < ctateSize)
	{
		state->clusterDim.x = ctate->clusterDimX;
		state->clusterDim.y = ctate->clusterDimY;
		state->clusterDim.z = ctate->clusterDimZ;
	}

	if (offsetof (CudbgSmTableEntry, clusterExceptionTargetBlockIdxZ)
	    < smteSize)
	  {
	    state->clusterExceptionTargetBlockIdx.x
		= smte->clusterExceptionTargetBlockIdxX;
	    state->clusterExceptionTargetBlockIdx.y
		= smte->clusterExceptionTargetBlockIdxY;
	    state->clusterExceptionTargetBlockIdx.z
		= smte->clusterExceptionTargetBlockIdxZ;
	    state->clusterExceptionTargetBlockIdxValid
		= smte->clusterExceptionTargetBlockIdxValid;
	  }

	state->blockIdx.x = ctate->blockIdxX;
	state->blockIdx.y = ctate->blockIdxY;
	state->blockIdx.z = ctate->blockIdxZ;
	state->validLanes = wte->validLanesMask;
	state->activeLanes = wte->activeLanesMask;
	state->errorPCValid = wte->errorPCValid;

	/* Get number of lanes */
	rc = API_CALL(getNumLanes)(devId, &numLanes);
	if (rc != CUDBG_SUCCESS)
		return rc;

	/* Fill in the lane information using info from ThreadTableEntry */
	for (ln = 0; ln < numLanes; ++ln) {
		if (!getBit(state->validLanes,ln))
			continue;

		/* For every valid lane there must be a corresponding ThreadTableEntry */
		GET_TABLE_ENTRY(tte, NULL, CUDBG_ERROR_INTERNAL,
				"ln%u_wp%u_sm%u_dev%u", ln, wp, sm, devId);

		state->lane[ln].virtualPC = tte->virtualPC;
		state->lane[ln].threadIdx.x = tte->threadIdxX;
		state->lane[ln].threadIdx.y = tte->threadIdxY;
		state->lane[ln].threadIdx.z = tte->threadIdxZ;
		state->lane[ln].exception = tte->exception;
	}

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readThreadIdx)(uint32_t dev, uint32_t sm, uint32_t wp,
			    uint32_t ln, CuDim3 *threadIdx)
{
	CudbgThreadTableEntry *tte;

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u threadIdx=%p",
		   dev, sm, wp, ln, threadIdx);

	VERIFY_ARG(threadIdx);

	GET_TABLE_ENTRY(tte, NULL, CUDBG_ERROR_INVALID_LANE,
			"ln%u_wp%u_sm%u_dev%u", ln, wp, sm, dev);

	threadIdx->x = tte->threadIdxX;
	threadIdx->y = tte->threadIdxY;
	threadIdx->z = tte->threadIdxZ;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readVirtualPC)(uint32_t dev, uint32_t sm, uint32_t wp,
			    uint32_t ln, uint64_t *pc)
{
	CudbgThreadTableEntry *tte;

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u pc=%p", dev, sm, wp, ln, pc);

	VERIFY_ARG(pc);

	GET_TABLE_ENTRY(tte, NULL, CUDBG_ERROR_INVALID_LANE,
			"ln%u_wp%u_sm%u_dev%u", ln, wp, sm, dev);

	*pc = tte->virtualPC;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getTID)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t *tid)
{
	CUDBGResult rc;
	uint64_t gridId;
	CudbgGridTableEntry *gte;
	CudbgContextTableEntry *cte;

	TRACE_FUNC("dev=%u sm=%u wp=%u tid=%p", dev, sm, wp, tid);

	VERIFY_ARG(tid);

	if ((rc = getGridId(dev, sm, wp, &gridId)) != CUDBG_SUCCESS)
		return rc;

	GET_TABLE_ENTRY(gte, NULL, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", gridId, dev);

	GET_TABLE_ENTRY(cte, NULL, CUDBG_ERROR_INVALID_CONTEXT,
			"ctx%llu_dev%u", gte->contextId, dev);

	*tid = cte->tid;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readBlockIdx)(uint32_t dev, uint32_t sm, uint32_t wp,
			   CuDim3 *blockIdx)
{
	CudbgWarpTableEntry *wte;
	CudbgCTATableEntry *ctate;

	TRACE_FUNC("dev=%u sm=%u wp=%u blockIdx=%p", dev, sm, wp, blockIdx);

	VERIFY_ARG(blockIdx);

	GET_TABLE_ENTRY(wte, NULL, CUDBG_ERROR_INVALID_WARP,
			"wp%u_sm%u_dev%u", wp, sm, dev);

	GET_TABLE_ENTRY(ctate, NULL, CUDBG_ERROR_UNKNOWN,
			"wp%u_sm%u_dev%u_cta", wp, sm, dev);

	blockIdx->x = ctate->blockIdxX;
	blockIdx->y = ctate->blockIdxY;
	blockIdx->z = ctate->blockIdxZ;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getBlockDim)(uint32_t dev, uint32_t sm, uint32_t wp,
			  CuDim3 *blockDim)
{
	CUDBGResult rc;
	uint64_t gridId;
	CudbgGridTableEntry *gte;

	TRACE_FUNC("dev=%u sm=%u wp=%u blockDim=%p", dev, sm, wp, blockDim);

	VERIFY_ARG(blockDim);

	if ((rc = getGridId(dev, sm, wp, &gridId)) != CUDBG_SUCCESS)
		return rc;

	GET_TABLE_ENTRY(gte, NULL, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", gridId, dev);

	blockDim->x = gte->blockDimX;
	blockDim->y = gte->blockDimY;
	blockDim->z = gte->blockDimZ;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getGridDim)(uint32_t dev, uint32_t sm, uint32_t wp,
			 CuDim3 *gridDim)
{
	CUDBGResult rc;
	uint64_t gridId;
	CudbgGridTableEntry *gte;

	TRACE_FUNC("dev=%u sm=%u wp=%u gridDim=%p", dev, sm, wp, gridDim);

	VERIFY_ARG(gridDim);

	if ((rc = getGridId(dev, sm, wp, &gridId)) != CUDBG_SUCCESS)
		return rc;

	GET_TABLE_ENTRY(gte, NULL, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", gridId, dev);

	gridDim->x = gte->gridDimX;
	gridDim->y = gte->gridDimY;
	gridDim->z = gte->gridDimZ;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readGridId)(uint32_t dev, uint32_t sm, uint32_t wp,
			 uint64_t *gridId64)
{
	CUDBGResult rc;
	uint64_t gridId;
	CudbgGridTableEntry *gte;

	TRACE_FUNC("dev=%u sm=%u wp=%u gridId64=%p", dev, sm, wp, gridId64);

	VERIFY_ARG(gridId64);

	if ((rc = getGridId(dev, sm, wp, &gridId)) != CUDBG_SUCCESS)
		return rc;

	GET_TABLE_ENTRY(gte, NULL, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", gridId, dev);

	*gridId64 = gte->gridId64;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getGridInfo)(uint32_t devId, uint64_t gridId, CUDBGGridInfo *info)
{
	CudbgGridTableEntry *gte;
	CudbgContextTableEntry *cte;
	size_t gteSize;

	TRACE_FUNC("devId=%u gridId=%llu info=%p", devId, gridId, info);

	VERIFY_ARG(info);

	if (devId >= cuCoreGetNumDevices(curcc))
		return CUDBG_ERROR_INVALID_DEVICE;

	GET_TABLE_ENTRY(gte, &gteSize, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", gridId, devId);

	GET_TABLE_ENTRY(cte, NULL, CUDBG_ERROR_INVALID_CONTEXT,
			"ctx%llu_dev%u", gte->contextId, devId);

	assert(gridId == gte->gridId64);
	memset(info, 0, sizeof(*info));
	info->dev = devId;
	info->gridId64 = gridId;
	info->tid = cte->tid;
	info->context = cte->contextId;
	info->module = gte->moduleHandle;
	info->function = gte->function;
	info->functionEntry = gte->functionEntry;
	info->type = gte->kernelType;
	info->gridDim.x = gte->gridDimX;
	info->gridDim.y = gte->gridDimY;
	info->gridDim.z = gte->gridDimZ;
	info->blockDim.x = gte->blockDimX;
	info->blockDim.y = gte->blockDimY;
	info->blockDim.z = gte->blockDimZ;
	info->parentGridId = gte->parentGridId64;
	info->origin = gte->origin;
	if (offsetof(CudbgGridTableEntry, clusterDimZ) < gteSize)
	{
		info->clusterDim.x = gte->clusterDimX;
		info->clusterDim.y = gte->clusterDimY;
		info->clusterDim.z = gte->clusterDimZ;
	}
	if (offsetof(CudbgGridTableEntry, preferredClusterDimZ) < gteSize)
	{
		info->preferredClusterDim.x = gte->preferredClusterDimX;
		info->preferredClusterDim.y = gte->preferredClusterDimY;
		info->preferredClusterDim.z = gte->preferredClusterDimZ;
	}

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getGridStatus)(uint32_t devId, uint64_t gridId,
			    CUDBGGridStatus *status)
{
	CudbgGridTableEntry *gte;

	TRACE_FUNC("devId=%u gridId=%llu status=%p", devId, gridId, status);

	VERIFY_ARG(status);

	if (devId >= cuCoreGetNumDevices(curcc))
		return CUDBG_ERROR_INVALID_DEVICE;

	GET_TABLE_ENTRY(gte, NULL, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", gridId, devId);

	*status = gte->gridStatus;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readRegisterRange)(uint32_t devId, uint32_t sm, uint32_t wp,
				uint32_t ln, uint32_t index,
				uint32_t registers_size, uint32_t *registers)
{
	uint32_t max_registers;
	Elf_Scn *scn;
	Elf_Data data;
	unsigned offset;
	int size;
	CUDBGResult rc;

	TRACE_FUNC("devId=%u sm=%u wp=%u ln=%u index=%u "
		   "registers_size=%u registers=%p",
		   devId, sm, wp, ln, index, registers_size, registers);

	VERIFY_ARG(registers);

	/* Zero out the registers array */
	memset(registers, 0, registers_size * sizeof(*registers));

	/* Sanity checks */
	if (devId >= cuCoreGetNumDevices(curcc))
		return CUDBG_ERROR_INVALID_DEVICE;

	rc = API_CALL(getNumRegisters)(devId, &max_registers);
	if (rc != CUDBG_SUCCESS)
		return rc;

	if (index + registers_size > max_registers)
		return CUDBG_ERROR_INVALID_ARGS;

	GET_TABLE_ENTRY(scn, NULL, CUDBG_ERROR_INVALID_ARGS,
			"regs_dev%u_sm%u_wp%u_ln%u", devId, sm, wp, ln);

	if (cuCoreReadSectionData(curcc->e, scn, &data) != 0)
		return CUDBG_ERROR_UNKNOWN;

	/* Calculate offset and size in bytes withing the registers section */
	offset = index * sizeof(*registers);
	size = registers_size * sizeof(*registers);

	if (offset > data.d_size)
		return rc; /* Registers are not used by application */

	/* Make sure that memcpy is done within Registers section bounds */
	if (size > data.d_size - offset)
		size = data.d_size - offset;

	memcpy(registers, (char *)data.d_buf + offset, size);

	return rc;
}

DEF_API_CALL(readRegister)(uint32_t devId, uint32_t sm, uint32_t wp,
			   uint32_t ln, uint32_t index, uint32_t *pReg)
{
	TRACE_FUNC("devId=%u sm=%u wp=%u ln=%u index=%u pReg=%p",
		   devId, sm, wp, ln, index, pReg);

	return API_CALL(readRegisterRange)(devId, sm, wp, ln, index, 1, pReg);
}

DEF_API_CALL(readBrokenWarps)(uint32_t devId, uint32_t sm,
			      uint64_t *brokenWarpsMask)
{
	CudbgWarpTableEntry *wte;
	uint32_t wp, warpsPerSM;
	CUDBGResult rc;

	TRACE_FUNC("devId=%u sm=%u brokenWarpsMask=%p",
		   devId, sm, brokenWarpsMask);

	VERIFY_ARG(brokenWarpsMask);

	rc = API_CALL(getNumWarps)(devId, &warpsPerSM);
	if (rc != CUDBG_SUCCESS)
		return rc;

	*brokenWarpsMask = 0;
	for (wp = 0; wp < warpsPerSM; ++wp) {
		wte = cuCoreGetMapEntry(&curcc->tableEntriesMap, NULL,
					"wp%u_sm%u_dev%u", wp, sm, devId);
		if (wte && wte->isWarpBroken)
			*brokenWarpsMask |= 1ULL << wp;
	}

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readValidWarps)(uint32_t devId, uint32_t sm,
			     uint64_t *validWarpsMask)
{
	uint32_t wp, warpsPerSM;
	CUDBGResult rc;

	TRACE_FUNC("devId=%u sm=%u validWarpsMask=%p",
		   devId, sm, validWarpsMask);

	VERIFY_ARG(validWarpsMask);

	rc = API_CALL(getNumWarps)(devId, &warpsPerSM);
	if (rc != CUDBG_SUCCESS)
		return rc;

	*validWarpsMask = 0;
	for (wp = 0; wp < warpsPerSM; ++wp) {
		if (cuCoreGetMapEntry(&curcc->tableEntriesMap, NULL,
				      "wp%u_sm%u_dev%u", wp, sm, devId))
			*validWarpsMask |= 1ULL << wp;
	}

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readValidLanes)(uint32_t dev, uint32_t sm, uint32_t wp,
			     uint32_t *validLanesMask)
{
	CudbgWarpTableEntry *wte;

	TRACE_FUNC("dev=%u sm=%u wp=%u validLanesMask=%p",
		   dev, sm, wp, validLanesMask);

	VERIFY_ARG(validLanesMask);

	GET_TABLE_ENTRY(wte, NULL, CUDBG_ERROR_INVALID_WARP,
			"wp%u_sm%u_dev%u", wp, sm, dev);

	*validLanesMask = wte->validLanesMask;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readActiveLanes)(uint32_t dev, uint32_t sm, uint32_t wp,
			      uint32_t *activeLanesMask)
{
	CudbgWarpTableEntry *wte;

	TRACE_FUNC("dev=%u sm=%u wp=%u activeLanesMask=%p",
		   dev, sm, wp, activeLanesMask);

	VERIFY_ARG(activeLanesMask);

	GET_TABLE_ENTRY(wte, NULL, CUDBG_ERROR_INVALID_WARP,
			"wp%u_sm%u_dev%u", wp, sm, dev);

	*activeLanesMask = wte->activeLanesMask;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getSmType)(uint32_t devId, char *buf, uint32_t sz)
{
	CudbgDeviceTableEntry *dte;
	const char *smType = NULL;

	TRACE_FUNC("devId=%u buf=%p sz=%u", devId, buf, sz);

	VERIFY_ARG(buf);

	GET_TABLE_ENTRY(dte, NULL, CUDBG_ERROR_INVALID_DEVICE, "dev%u", devId);

	smType = cuCoreGetStrTabByIndex(curcc, dte->smType);
	strncpy(buf, smType, sz);

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getDeviceName)(uint32_t devId, char *buf, uint32_t sz)
{
	CudbgDeviceTableEntry *dte;
	const char *devName = NULL;

	TRACE_FUNC("devId=%u buf=%p sz=%u", devId, buf, sz);

	VERIFY_ARG(buf);

	GET_TABLE_ENTRY(dte, NULL, CUDBG_ERROR_INVALID_DEVICE, "dev%u", devId);

	devName = cuCoreGetStrTabByIndex(curcc, dte->devName);
	strncpy(buf, devName, sz);

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getDevicePCIBusInfo)(uint32_t devId, uint32_t *pciBusId,
				  uint32_t *pciDevId)
{
	CudbgDeviceTableEntry *dte;

	TRACE_FUNC("devId=%u pciBusId=%p pciDevId=%p",
		   devId, pciBusId, pciDevId);

	VERIFY_ARG(pciBusId);
	VERIFY_ARG(pciDevId);

	GET_TABLE_ENTRY(dte, NULL, CUDBG_ERROR_INVALID_DEVICE, "dev%u", devId);

	*pciDevId = dte->pciDevId;
	*pciBusId = dte->pciBusId;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getDeviceType)(uint32_t devId, char *buf, uint32_t sz)
{
	CudbgDeviceTableEntry *dte;
	const char *devType = NULL;

	TRACE_FUNC("devId=%u buf=%p sz=%u", devId, buf, sz);

	VERIFY_ARG(buf);

	GET_TABLE_ENTRY(dte, NULL, CUDBG_ERROR_INVALID_DEVICE, "dev%u", devId);

	devType = cuCoreGetStrTabByIndex(curcc, dte->devType);
	strncpy(buf, devType, sz);

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getNextEvent)(CUDBGEventQueueType type, CUDBGEvent *event)
{
	const CUDBGEvent *inputEvent;

	TRACE_FUNC("type=%u event=%p", type, event);

	VERIFY_ARG(event);

	inputEvent = cuCoreGetEvent(curcc);
	if (inputEvent == NULL || type == CUDBG_EVENT_QUEUE_TYPE_ASYNC)
		return CUDBG_ERROR_NO_EVENT_AVAILABLE;

	*event = *inputEvent;
	cuCoreDeleteEvent(curcc);

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readDeviceExceptionState)(uint32_t devId,
				       uint64_t *exceptionSMMask, uint32_t n)
{
	uint32_t numSMs, numWarps, numLanes;
	uint32_t sm, wp, ln;
	uint64_t validWarpsMask;
	CUDBGWarpState state;
	CUDBGResult rc;

	TRACE_FUNC("devId=%u exceptionSMMask=%p", devId, exceptionSMMask);

	VERIFY_ARG(exceptionSMMask);

	if (devId >= cuCoreGetNumDevices(curcc))
		return CUDBG_ERROR_INVALID_DEVICE;

	memset(exceptionSMMask, 0, n*sizeof(*exceptionSMMask));

	rc = API_CALL(getNumSMs)(devId, &numSMs);
	if (rc != CUDBG_SUCCESS)
		return rc;

	rc = API_CALL(getNumWarps)(devId, &numWarps);
	if (rc != CUDBG_SUCCESS)
		return rc;

	rc = API_CALL(getNumLanes)(devId, &numLanes);
	if (rc != CUDBG_SUCCESS)
		return rc;

	for (sm = 0; sm < numSMs; ++sm) {
		/* First check if the SM has a detected exception. */
		CudbgSmTableEntry *ste;
		GET_TABLE_ENTRY(ste, NULL, CUDBG_ERROR_INVALID_SM,
			"sm%u_dev%u", sm, devId);
		if (ste != NULL && ste->exception != CUDBG_EXCEPTION_NONE) {
			exceptionSMMask[sm/64] |= 1ULL << (sm % 64);
			continue;
		}
		rc = API_CALL(readValidWarps)(devId, sm, &validWarpsMask);
		if (rc != CUDBG_SUCCESS)
			return rc;
		if (validWarpsMask == 0)
			continue;
		for (wp = 0; wp < numWarps; ++wp) {
			if (!getBit(validWarpsMask, wp))
				continue;
			rc = API_CALL(readWarpState)(devId, sm, wp, &state);
			if (rc != CUDBG_SUCCESS)
				return rc;
			for (ln = 0; ln < numLanes; ++ln) {
				if (!getBit(state.activeLanes, ln))
					continue;
				if (state.lane[ln].exception != CUDBG_EXCEPTION_NONE)
					exceptionSMMask[sm/64] |= 1ULL << (sm % 64);
			}
		}
	}

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readErrorPC)(uint32_t devId, uint32_t sm, uint32_t wp,
			  uint64_t *errorPC, bool *errorPCValid)
{
	CudbgWarpTableEntry *wte;

	TRACE_FUNC("devId=%u sm=%u wp=%u errorPC=%p errorPCValid=%p",
		   devId, sm, wp, errorPC, errorPCValid);

	VERIFY_ARG(errorPC);
	VERIFY_ARG(errorPCValid);

	GET_TABLE_ENTRY(wte, NULL, CUDBG_ERROR_INVALID_WARP,
			"wp%u_sm%u_dev%u", wp, sm, devId);

	*errorPC = wte->errorPC;
	*errorPCValid = wte->errorPCValid;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readPC)(uint32_t devId, uint32_t sm, uint32_t wp, uint32_t ln,
		     uint64_t *pc)
{
	CudbgThreadTableEntry *tte;

	TRACE_FUNC("devId=%u sm=%u wp=%u ln=%u pc=%p", devId, sm, wp, ln, pc);

	VERIFY_ARG(pc);

	GET_TABLE_ENTRY(tte, NULL, CUDBG_ERROR_INVALID_LANE,
			"ln%u_wp%u_sm%u_dev%u", ln, wp, sm, devId);

	*pc = tte->physPC;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getElfImageByHandle)(uint32_t devId, uint64_t handle,
				  CUDBGElfImageType type,
				  void *elfImage, uint64_t size)
{
	Elf_Scn *scn;
	Elf_Data data;

	TRACE_FUNC("devId=%u handle=0x%llx type=%u elfImage=%p size=%llu",
		   devId, handle, type, elfImage, size);

	VERIFY_ARG(elfImage);

	GET_TABLE_ENTRY(scn, NULL, CUDBG_ERROR_INVALID_ARGS,
			"%celf_handle%llx",
			type ? 'r':'u', handle);

	if (cuCoreReadSectionData(curcc->e, scn, &data) != 0)
		return CUDBG_ERROR_UNKNOWN;

	if (size < data.d_size)
		return CUDBG_ERROR_INVALID_ARGS;

	memcpy(elfImage, data.d_buf, data.d_size);

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readLaneException)(uint32_t dev, uint32_t sm, uint32_t wp,
				uint32_t ln, CUDBGException_t *exception)
{
	CudbgThreadTableEntry *tte;

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u exception=%p",
		   dev, sm, wp, ln, exception);

	VERIFY_ARG(exception);

	GET_TABLE_ENTRY(tte, NULL, CUDBG_ERROR_INVALID_LANE,
			"ln%u_wp%u_sm%u_dev%u", ln, wp, sm, dev);

	*exception = (CUDBGException_t)tte->exception;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readLaneStatus)(uint32_t devId, uint32_t sm, uint32_t wp,
			     uint32_t ln, bool *error)
{
	CudbgThreadTableEntry *tte;

	TRACE_FUNC("devId=%u sm=%u wp=%u ln=%u error=%p",
		   devId, sm, wp, ln, error);

	VERIFY_ARG(error);

	GET_TABLE_ENTRY(tte, NULL, CUDBG_ERROR_INVALID_LANE,
			"ln%u_wp%u_sm%u_dev%u", ln, wp, sm, devId);

	*error = tte->exception != CUDBG_EXCEPTION_UNKNOWN;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readSyscallCallDepth)(uint32_t dev, uint32_t sm, uint32_t wp,
				   uint32_t ln, uint32_t *depth)
{
	CudbgThreadTableEntry *tte;

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u depth=%p",
		   dev, sm, wp, ln, depth);

	VERIFY_ARG(depth);

	GET_TABLE_ENTRY(tte, NULL, CUDBG_ERROR_INVALID_LANE,
			"ln%u_wp%u_sm%u_dev%u", ln, wp, sm, dev);

	*depth = tte->syscallCallDepth;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readCallDepth)(uint32_t dev, uint32_t sm, uint32_t wp,
			    uint32_t ln, uint32_t *depth)
{
	CudbgThreadTableEntry *tte;

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u depth=%p",
		   dev, sm, wp, ln, depth);

	VERIFY_ARG(depth);

	GET_TABLE_ENTRY(tte, NULL, CUDBG_ERROR_INVALID_LANE,
			"ln%u_wp%u_sm%u_dev%u", ln, wp, sm, dev);

	*depth = tte->callDepth;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readReturnAddress)(uint32_t dev, uint32_t sm, uint32_t wp,
				uint32_t ln, uint32_t level, uint64_t *ra)
{
	CudbgBacktraceTableEntry *bte;

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u level=%u ra=%p",
		   dev, sm, wp, ln, level, ra);

	VERIFY_ARG(ra);

	GET_TABLE_ENTRY(bte, NULL, CUDBG_ERROR_INVALID_CALL_LEVEL,
			"bt%u_ln%u_wp%u_sm%u_dev%u", level, ln, wp, sm, dev);

	*ra = bte->returnAddress;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readVirtualReturnAddress)(uint32_t dev, uint32_t sm, uint32_t wp,
				       uint32_t ln, uint32_t level,
				       uint64_t *ra)
{
	CudbgBacktraceTableEntry *bte;

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u level=%u ra=%p",
		   dev, sm, wp, ln, level, ra);

	VERIFY_ARG(ra);

	GET_TABLE_ENTRY(bte, NULL, CUDBG_ERROR_INVALID_CALL_LEVEL,
			"bt%u_ln%u_wp%u_sm%u_dev%u", level, ln, wp, sm, dev);

	*ra = bte->virtualReturnAddress;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(isDeviceCodeAddress)(uintptr_t addr, bool *isDeviceAddress)
{
	Elf64_Shdr *shdr;

	cs_t callStack[] = {
		PTR2CS(cuCoreIterateELFImages), true,
		PTR2CS(cuCoreIterateELFSections),
		PTR2CS(cuCoreIterateSymbolTable),
		PTR2CS(cuCoreFilterSymbolByAddress), addr, 0, PTR2CS(isDeviceAddress),
		PTR2CS(cuCoreReadSymbolSection), PTR2CS(&shdr),
		PTR2CS(NULL),
	};

	TRACE_FUNC("addr=0x%llx isDeviceAddress=%p", (unsigned long long)addr, isDeviceAddress);

	VERIFY_ARG(isDeviceAddress);

	*isDeviceAddress = false;

	cuCoreExecuteCallStack(curcc, callStack);

	/* If found check if section is allocated */
	if (*isDeviceAddress && !(shdr->sh_flags & SHF_ALLOC))
		*isDeviceAddress = false;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getAdjustedCodeAddress)(uint32_t devId, uint64_t address,
				     uint64_t *adjustedAddress,
				     CUDBGAdjAddrAction adjAction)
{
	TRACE_FUNC("devId=%u address=0x%llx adjustedAddress=%p adjAction=%u",
		   devId, address, adjustedAddress, adjAction);

	VERIFY_ARG(adjustedAddress);

	/* TODO? */
	*adjustedAddress = address;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getNumPredicates)(uint32_t dev, uint32_t *numPredicates)
{
	CudbgDeviceTableEntry *dte;

	TRACE_FUNC("dev=%u numPredicates=%p", dev, numPredicates);

	VERIFY_ARG(numPredicates);

	GET_TABLE_ENTRY(dte, NULL, CUDBG_ERROR_INVALID_DEVICE, "dev%u", dev);

	*numPredicates = dte->numPredicatesPrLane;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readPredicates)(uint32_t dev, uint32_t sm, uint32_t wp,
			     uint32_t ln, uint32_t predicates_size,
			     uint32_t *predicates)
{
	uint32_t num_predicates;
	Elf_Scn *scn;
	Elf_Data data;
	size_t size;
	CUDBGResult rc;

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u "
		   "predicates_size=%u predicates=%p",
		   dev, sm, wp, ln, predicates_size, predicates);

	VERIFY_ARG(predicates);

	/* Zero out the predicates array */
	memset(predicates, 0, predicates_size * sizeof(uint32_t));

	rc = API_CALL(getNumPredicates)(dev, &num_predicates);
	if (rc != CUDBG_SUCCESS)
		return rc;

	if (predicates_size > num_predicates)
		return CUDBG_ERROR_INVALID_ARGS;

	GET_TABLE_ENTRY(scn, NULL, CUDBG_ERROR_INVALID_ARGS,
			"pred_dev%u_sm%u_wp%u_ln%u", dev, sm, wp, ln);

	if (cuCoreReadSectionData(curcc->e, scn, &data) != 0)
		return CUDBG_ERROR_UNKNOWN;

	size = predicates_size * sizeof(uint32_t);

	/* Make sure that memcpy is done within Registers section bounds */
	if (size > data.d_size)
		size = data.d_size;

	memcpy(predicates, data.d_buf, size);

	return rc;
}

DEF_API_CALL(readCCRegister)(uint32_t dev, uint32_t sm, uint32_t wp,
			     uint32_t ln, uint32_t *val)
{
	CudbgThreadTableEntry *tte;

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u val=%p", dev, sm, wp, ln, val);

	VERIFY_ARG(val);

	GET_TABLE_ENTRY(tte, NULL, CUDBG_ERROR_INVALID_LANE,
			"ln%u_wp%u_sm%u_dev%u", ln, wp, sm, dev);

	*val = tte->ccRegister;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getGridAttributes)(uint32_t dev, uint32_t sm, uint32_t wp,
				CUDBGAttributeValuePair *pairs,
				uint32_t numPairs)
{
	CUDBGResult rc;
	uint64_t gridId;
	CudbgGridTableEntry *gte;
	CUDBGAttributeValuePair *pair;
	uint32_t pairId;

	TRACE_FUNC("dev=%u sm=%u wp=%u pairs=%p numPairs=%u",
		   dev, sm, wp, pairs, numPairs);

	VERIFY_ARG(pairs);

	if ((rc = getGridId(dev, sm, wp, &gridId)) != CUDBG_SUCCESS)
		return rc;

	GET_TABLE_ENTRY(gte, NULL, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", gridId, dev);

	for (pairId = 0; pairId < numPairs; ++pairId) {
		pair = &pairs[pairId];

		switch (pair->attribute) {
		case CUDBG_ATTR_GRID_LAUNCH_BLOCKING:
			pair->value = gte->attrLaunchBlocking;
			break;
		case CUDBG_ATTR_GRID_TID:
			pair->value = gte->attrHostTid;
			break;
		default:
			return CUDBG_ERROR_INVALID_ATTRIBUTE;
		}
	}

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getGridAttribute)(uint32_t dev, uint32_t sm, uint32_t wp,
			       CUDBGAttribute attr, uint64_t *value)
{
	CUDBGAttributeValuePair pair;
	CUDBGResult res;

	TRACE_FUNC("dev=%u sm=%u wp=%u attr=%u value=%p",
		   dev, sm, wp, attr, value);

	VERIFY_ARG(value);

	pair.attribute = attr;

	res = API_CALL(getGridAttributes)(dev, sm, wp, &pair, 1);
	if (res != CUDBG_SUCCESS)
		return res;

	*value = pair.value;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getManagedMemoryRegionInfo)(uint64_t startAddress,
					 CUDBGMemoryInfo *memoryInfo,
					 uint32_t memoryInfo_size,
					 uint32_t *numEntries)
{
	UT_array *managedSegs = curcc->managedMemorySegs;
	MemorySeg *memorySeg = (MemorySeg *)utarray_front(managedSegs);

	TRACE_FUNC("startAddress=0x%llx memoryInfo=%p "
		   "memoryInfo_size=%u numEntries=%p", 
		   startAddress, memoryInfo, memoryInfo_size, numEntries);

	VERIFY_ARG(memoryInfo);
	VERIFY_ARG(numEntries);

	/* Skip segments until startAddress is found */
	while (memorySeg && !(memorySeg->address <= startAddress &&
			memorySeg->address + memorySeg->size > startAddress))
		memorySeg = (MemorySeg *)utarray_next(managedSegs, memorySeg);

	*numEntries = 0;

	for (; memorySeg != NULL && *numEntries < memoryInfo_size;
			memorySeg = (MemorySeg *)utarray_next(managedSegs, memorySeg),
				(*numEntries)++, memoryInfo++) {
		memoryInfo->startAddress = memorySeg->address;
		memoryInfo->size = memorySeg->size;
	}

	return CUDBG_SUCCESS;
}

DEF_API_CALL(disassemble)(uint32_t dev, uint64_t addr, uint32_t *instSize,
			  char *buf, uint32_t sz)
{
	CudbgDeviceTableEntry *dte;

	TRACE_FUNC("dev=%u addr=0x%llx instSize=%p buf=%p sz=%u", dev, addr, instSize, buf, sz);

	VERIFY_ARG(instSize);

	if (buf)
		*buf = '\0';

	/* This assumes that the GPU architecture has a uniform instruction
	 * size, which is true on all GPU architectures except FERMI.
	 * Since cuda-gdb no longer supports FERMI as of 9.0 toolkit, this
	 * assumption is valid. */
	GET_TABLE_ENTRY(dte, NULL, CUDBG_ERROR_INVALID_DEVICE, "dev%u", dev);
	*instSize = dte->instructionSize;

	if (!sz)
		return CUDBG_SUCCESS;

	/* Verify a buf argument was passed after we verify sz != 0 */
	VERIFY_ARG(buf);
	
	/* Disassembling from device memory with a coredump is invalid */
	return CUDBG_ERROR_INVALID_DEVICE;
}

DEF_API_CALL(getNumUniformRegisters)(uint32_t dev, uint32_t *numRegs)
{
	CudbgDeviceTableEntry *dte;
	size_t dteSize;

	TRACE_FUNC("dev=%u numRegs=%p", dev, numRegs);

	VERIFY_ARG(numRegs);

	GET_TABLE_ENTRY(dte, &dteSize, CUDBG_ERROR_INVALID_DEVICE, "dev%u", dev);

	if (offsetof(CudbgDeviceTableEntry, numUniformRegsPrWarp) < dteSize)
		*numRegs = dte->numUniformRegsPrWarp;
	else
		*numRegs = 0;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readUniformRegisterRange)(uint32_t devId, uint32_t sm, uint32_t wp,
                                       uint32_t index,
                                       uint32_t registers_size, uint32_t *registers)
{
	uint32_t max_registers;
	Elf_Scn *scn;
	Elf_Data data;
	unsigned offset;
	CUDBGResult rc;
	uint32_t len;

	TRACE_FUNC("devId=%u sm=%u wp=%u index=%u "
		   "registers_size=%u registers=%p",
		   devId, sm, wp, index, registers_size, registers);

	VERIFY_ARG(registers);

	/* Zero out the registers array */
	memset(registers, 0, registers_size * sizeof(*registers));

	/* Sanity checks */
	if (devId >= cuCoreGetNumDevices(curcc))
		return CUDBG_ERROR_INVALID_DEVICE;

	GET_TABLE_ENTRY(scn, NULL, CUDBG_ERROR_MISSING_DATA,
			"uregs_dev%u_sm%u_wp%u", devId, sm, wp);

	rc = API_CALL(getNumUniformRegisters)(devId, &max_registers);
	if (rc != CUDBG_SUCCESS)
		return rc;

	if (index + registers_size > max_registers * sizeof(*registers))
		return CUDBG_ERROR_INVALID_ARGS;

	if (cuCoreReadSectionData(curcc->e, scn, &data) != 0)
		return CUDBG_ERROR_UNKNOWN;

	/* Calculate offset and size in bytes withing the registers section */
	offset = index * sizeof(*registers);
        len = registers_size * sizeof(uint32_t);

	if (offset > data.d_size)
		return rc; /* Registers are not used by application */

	/* Make sure that memcpy is done within Registers section bounds */
	if (len > data.d_size - offset)
            len = data.d_size - offset;

	memcpy(registers, (char *)data.d_buf + offset, len);

	return rc;
}

DEF_API_CALL(getNumUniformPredicates)(uint32_t dev, uint32_t *numPredicates)
{
	CudbgDeviceTableEntry *dte;
	size_t dteSize;

	TRACE_FUNC("dev=%u numPredicates=%p", dev, numPredicates);

	VERIFY_ARG(numPredicates);

	GET_TABLE_ENTRY(dte, &dteSize, CUDBG_ERROR_INVALID_DEVICE, "dev%u", dev);

	if (offsetof(CudbgDeviceTableEntry, numUniformPredicatesPrWarp) < dteSize)
		*numPredicates = dte->numUniformPredicatesPrWarp;
	else
		*numPredicates = 0;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readUniformPredicates)(uint32_t devId, uint32_t sm, uint32_t wp,
                                    uint32_t predicates_size, uint32_t *predicates)

{
	uint32_t max_predicates;
	Elf_Scn *scn;
	Elf_Data data;
	int size;
	CUDBGResult rc;

	TRACE_FUNC("devId=%u sm=%u wp=%u predicates_size=%u predicates=%p",
		   devId, sm, wp, predicates_size, predicates);

	VERIFY_ARG(predicates);

	/* Zero out the registers array */
	memset(predicates, 0, predicates_size * sizeof(*predicates));

	/* Sanity checks */
	if (devId >= cuCoreGetNumDevices(curcc))
		return CUDBG_ERROR_INVALID_DEVICE;

	GET_TABLE_ENTRY(scn, NULL, CUDBG_ERROR_MISSING_DATA,
			"upred_dev%u_sm%u_wp%u", devId, sm, wp);

	rc = API_CALL(getNumUniformPredicates)(devId, &max_predicates);
	if (rc != CUDBG_SUCCESS)
		return rc;

	if (predicates_size > max_predicates)
		return CUDBG_ERROR_INVALID_ARGS;

	if (cuCoreReadSectionData(curcc->e, scn, &data) != 0)
		return CUDBG_ERROR_UNKNOWN;

	/* Calculate size in bytes withing the registers section */
	size = predicates_size * sizeof(*predicates);

	/* Make sure that memcpy is done within Registers section bounds */
	if (size > data.d_size)
		size = data.d_size;

	memcpy(predicates, (char *)data.d_buf, size);

	return rc;
}

DEF_API_CALL(getClusterDim)(uint32_t devId, uint32_t sm, uint32_t wp, CuDim3 *clusterDim)
{
	CudbgGridTableEntry *gte;
	CudbgCTATableEntry *ctate;
	size_t gteSize;
	size_t ctateSize;

	TRACE_FUNC("devId=%u sm=%u wp=%u clusterDim=%p", devId, sm, wp, clusterDim);

	VERIFY_ARG(clusterDim);

	if (devId >= cuCoreGetNumDevices(curcc))
		return CUDBG_ERROR_INVALID_DEVICE;

	GET_TABLE_ENTRY(ctate, &ctateSize, CUDBG_ERROR_UNKNOWN,
			"wp%u_sm%u_dev%u_cta", wp, sm, devId);

	if (offsetof(CudbgCTATableEntry, clusterDimZ) < ctateSize)
	{
		clusterDim->x = ctate->clusterDimX;
		clusterDim->y = ctate->clusterDimY;
		clusterDim->z = ctate->clusterDimZ;
		return CUDBG_SUCCESS;
	}

	GET_TABLE_ENTRY(gte, &gteSize, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", ctate->gridId64, devId);

	assert(ctate->gridId64 == gte->gridId64);
	memset(clusterDim, 0, sizeof(*clusterDim));
	if (offsetof(CudbgGridTableEntry, clusterDimZ) < gteSize)
	{
		clusterDim->x = gte->clusterDimX;
		clusterDim->y = gte->clusterDimY;
		clusterDim->z = gte->clusterDimZ;
	}

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readClusterIdx)(uint32_t dev, uint32_t sm, uint32_t wp,
			     CuDim3 *clusterIdx)
{
	CudbgCTATableEntry *ctate;
	size_t ctateSize;

	TRACE_FUNC("dev=%u sm=%u wp=%u clusterIdx=%p", dev, sm, wp, clusterIdx);

	VERIFY_ARG(clusterIdx);

	GET_TABLE_ENTRY(ctate, &ctateSize, CUDBG_ERROR_UNKNOWN,
			"wp%u_sm%u_dev%u_cta", wp, sm, dev);

	memset(clusterIdx, 0, sizeof(*clusterIdx));
	if (offsetof(CudbgCTATableEntry, clusterIdxZ) < ctateSize)
	{
		clusterIdx->x = ctate->clusterIdxX;
		clusterIdx->y = ctate->clusterIdxY;
		clusterIdx->z = ctate->clusterIdxZ;
	}

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getErrorStringEx)(char *buf, uint32_t bufSz, uint32_t *msgSz)
{
	TRACE_FUNC("buf=%p bufSz=%u msgSz=%p", buf, bufSz, msgSz);

	if (buf && bufSz)
		buf[0] = 0;

	if (msgSz)
		*msgSz = 1;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getConstBankAddress123)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t bank, uint32_t offset, uint64_t* address)
{
	CUDBGResult rc;
	uint64_t gridId;
	CudbgGridTableEntry *gte;
	CudbgConstBankTableEntry *cbte;

	TRACE_FUNC("dev=%u sm=%u wp=%u bank=%u offset=%u",
		   dev, sm, wp, bank, offset);

	VERIFY_ARG(address);

	if ((rc = getGridId(dev, sm, wp, &gridId)) != CUDBG_SUCCESS)
		return rc;

	GET_TABLE_ENTRY(gte, NULL, CUDBG_ERROR_INVALID_GRID,
			"grid%llu_dev%u", gridId, dev);

	GET_TABLE_ENTRY(cbte, NULL, CUDBG_ERROR_MISSING_DATA,
			"grid%llu_dev%u_cbank%u", gte->gridId64, dev, bank);

	if (offset >= cbte->size)
		return CUDBG_ERROR_INVALID_MEMORY_ACCESS;

	*address = cbte->addr + offset;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(getConstBankAddress)(uint32_t dev, uint64_t gridId64, uint32_t bank, uint64_t* address, uint32_t* size)
{
	CUDBGResult rc;
	uint64_t gridId;
	CudbgGridTableEntry *gte;
	CudbgConstBankTableEntry *cbte;

	TRACE_FUNC("dev=%u gridId64=%llu bank=%u",
		   dev, gridId64, bank);

	VERIFY_ARG(address);
	VERIFY_ARG(size);

	GET_TABLE_ENTRY(cbte, NULL, CUDBG_ERROR_MISSING_DATA,
			"grid%llu_dev%u_cbank%u", gridId64, dev, bank);

	*address = cbte->addr;
	*size = cbte->size;

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readAllVirtualReturnAddresses)(uint32_t dev, uint32_t sm, uint32_t wp, uint32_t ln, uint64_t *addrs, uint32_t numAddrs, uint32_t *callDepth, uint32_t *syscallCallDepth)
{
	CudbgBacktraceTableEntry *bte;
	CudbgThreadTableEntry *tte;
	uint32_t level;

	TRACE_FUNC("dev=%u sm=%u wp=%u ln=%u addrs=%p numAddrs=%u "
			   "callDepth=%p syscallCallDepth=%p",
			   dev, sm, wp, ln, addrs, callDepth, syscallCallDepth);

	if (numAddrs == 0)
		VERIFY(!addrs, CUDBG_ERROR_INVALID_ARGS, "`addrs' must be null");
	else
		VERIFY_ARG(addrs);

	VERIFY_ARG(callDepth);
	VERIFY_ARG(syscallCallDepth);

	GET_TABLE_ENTRY(tte, NULL, CUDBG_ERROR_INVALID_LANE, "ln%u_wp%u_sm%u_dev%u",
					ln, wp, sm, dev);

	*callDepth = tte->callDepth;
	*syscallCallDepth = tte->syscallCallDepth;

	if (numAddrs > *callDepth)
		numAddrs = *callDepth;

	for (level = 0; level < numAddrs; level++) {
		GET_TABLE_ENTRY(bte, NULL, CUDBG_ERROR_INVALID_CALL_LEVEL,
						"bt%u_ln%u_wp%u_sm%u_dev%u", level, ln, wp, sm, dev);
		addrs[level] = bte->virtualReturnAddress;
	}

	return CUDBG_SUCCESS;
}

DEF_API_CALL(readSmException)(uint32_t dev, uint32_t sm,
			      CUDBGException_t *exception, uint64_t *errorPC, bool *errorPCValid)
{
	CudbgSmTableEntry *ste;

	TRACE_FUNC("dev=%u sm=%u exception=%p errorPC=%p errorPCValid=%p",
		   dev, sm, exception, errorPC, errorPCValid);

	VERIFY_ARG(exception);
	VERIFY_ARG(errorPC);
	VERIFY_ARG(errorPCValid);

	GET_TABLE_ENTRY(ste, NULL, CUDBG_ERROR_INVALID_SM,
			"sm%u_dev%u", sm, dev);
	if (ste != NULL) {
		*exception = (CUDBGException_t)ste->exception;
		*errorPC = ste->errorPC;
		*errorPCValid = ste->errorPCValid;
	}

	return CUDBG_SUCCESS;
}

DEF_API_CALL (getClusterExceptionTargetBlock)
(uint32_t dev, uint32_t sm, uint32_t wp, CuDim3 *blockIdx, bool *blockIdxValid)
{
  CudbgSmTableEntry *smte;
  size_t smteSize;

  TRACE_FUNC ("dev=%u sm=%u wp=%u blockIdx=%p blockIdxValid=%p", dev, sm, wp,
	      blockIdx, blockIdxValid);

  VERIFY_ARG (blockIdx);
  VERIFY_ARG (blockIdxValid);

  GET_TABLE_ENTRY (smte, &smteSize, CUDBG_ERROR_INVALID_SM, "sm%u_dev%u", sm,
		   dev);

  if (offsetof (CudbgSmTableEntry, clusterExceptionTargetBlockIdxZ) < smteSize)
    {
      blockIdx->x = smte->clusterExceptionTargetBlockIdxX;
      blockIdx->y = smte->clusterExceptionTargetBlockIdxY;
      blockIdx->z = smte->clusterExceptionTargetBlockIdxZ;
      *blockIdxValid = smte->clusterExceptionTargetBlockIdxValid;
    }
  else
    {
      memset (blockIdx, 0, sizeof (*blockIdx));
      *blockIdxValid = false;
    }

  return CUDBG_SUCCESS;
}

DEF_API_CALL(readWarpResources)(uint32_t dev, uint32_t sm, uint32_t wp,
				CUDBGWarpResources *resources)
{
  CudbgWarpTableEntry *wte;
  CUDBGResult rc;
  size_t wteSize;

  TRACE_FUNC("dev=%u sm=%u wp=%u resources=%p", dev, sm, wp, resources);
  
  VERIFY_ARG(resources);
  memset (resources, 0, sizeof (*resources));
  
  GET_TABLE_ENTRY(wte, NULL, CUDBG_ERROR_INVALID_WARP,
  		"wp%u_sm%u_dev%u", wp, sm, dev);
  
  if (offsetof (CudbgWarpTableEntry, numRegs) < wteSize)
    resources->numRegisters = wte->numRegs;
  if (offsetof (CudbgWarpTableEntry, sharedMemSize) < wteSize)
    resources->sharedMemSize = wte->sharedMemSize;
  
  return CUDBG_SUCCESS;
}

static const struct CUDBGAPI_st cudbgCoreApi = {
  /* Initialization */
  API_CALL (doNothing),
  API_CALL (doNothing),

  /* Device Execution Control */
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),

  /* Breakpoints */
  API_CALL (notSupported),
  API_CALL (notSupported),

  /* Device State Inspection */
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (readThreadIdx),
  API_CALL (readBrokenWarps),
  API_CALL (readValidWarps),
  API_CALL (readValidLanes),
  API_CALL (readActiveLanes),
  API_CALL (readCodeMemory),
  API_CALL (readConstMemory),
  API_CALL (notSupported),
  API_CALL (readParamMemory),
  API_CALL (readSharedMemory),
  API_CALL (readLocalMemory),
  API_CALL (readRegister),
  API_CALL (readPC),
  API_CALL (readVirtualPC),
  API_CALL (readLaneStatus),

  /* Device State Alteration */
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),

  /* Grid Properties */
  API_CALL (notSupported),
  API_CALL (getBlockDim),
  API_CALL (getTID),
  API_CALL (getElfImage32),

  /* Device Properties */
  API_CALL (getDeviceType),
  API_CALL (getSmType),
  API_CALL (getNumDevices),
  API_CALL (getNumSMs),
  API_CALL (getNumWarps),
  API_CALL (getNumLanes),
  API_CALL (getNumRegisters),

  /* DWARF-related routines */
  API_CALL (notSupported),
  API_CALL (disassemble),
  API_CALL (notSupported),
  API_CALL (lookupDeviceCodeSymbol),

  /* Events */
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),

  /* 3.1 Extensions */
  API_CALL (getGridAttribute),
  API_CALL (getGridAttributes),
  API_CALL (notSupported),
  API_CALL (readLaneException),
  API_CALL (notSupported),
  API_CALL (notSupported),

  /* 3.1 - ABI */
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),

  /* 3.2 Extensions */
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (readGlobalMemory),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),

  /* 4.0 Extensions */
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (readBlockIdx),
  API_CALL (getGridDim),
  API_CALL (readCallDepth),
  API_CALL (readReturnAddress),
  API_CALL (readVirtualReturnAddress),
  API_CALL (getElfImage),

  /* 4.1 Extensions */
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (readSyscallCallDepth),

  /* 4.2 Extensions */
  API_CALL (notSupported),

  /* 5.0 Extensions */
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),

  /* 5.5 Extensions */
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (readGridId),
  API_CALL (getGridStatus),
  API_CALL (notSupported),
  API_CALL (getDevicePCIBusInfo),
  API_CALL (notSupported),

  /* 6.0 Extensions */
  API_CALL (getAdjustedCodeAddress),
  API_CALL (readErrorPC),
  API_CALL (getNextEvent),
  API_CALL (getElfImageByHandle),
  API_CALL (notSupported),
  API_CALL (notSupported),
  API_CALL (readRegisterRange),
  API_CALL (readGenericMemory),
  API_CALL (notSupported),
  API_CALL (readGlobalMemory),
  API_CALL (notSupported),
  API_CALL (getManagedMemoryRegionInfo),
  API_CALL (isDeviceCodeAddress),
  API_CALL (notSupported),

  /* 6.5 Extensions */
  API_CALL (readPredicates),
  API_CALL (notSupported),
  API_CALL (getNumPredicates),
  API_CALL (readCCRegister),
  API_CALL (notSupported),

  API_CALL (getDeviceName),
  API_CALL (notSupported),

  /* 9.0 Extensions */
  API_CALL (readDeviceExceptionState),

  /* 10.0 Extensions */
  API_CALL (getNumUniformRegisters),
  API_CALL (readUniformRegisterRange),
  API_CALL (notSupported),
  API_CALL (getNumUniformPredicates),
  API_CALL (readUniformPredicates),
  API_CALL (notSupported),

  /* 11.8 Extensions */
  API_CALL (notSupported),

  /* 12.0 Extensions */
  API_CALL (notSupported), /* getGridInfo120 */
  API_CALL (notSupported), /* getClusterDim120 */
  API_CALL (notSupported), /* readWarpState120 */
  API_CALL (readClusterIdx),

  /* 12.2 Extensions */
  API_CALL (getErrorStringEx),

  /* 12.3 Extensions */
  API_CALL (notSupported), /* getLoadedFunctionInfo */
  API_CALL (notSupported), /* generateCoredump */
  API_CALL (getConstBankAddress123),

  /* 12.4 Extensions */
  API_CALL (notSupported), /* getDeviceInfoSizes */
  API_CALL (notSupported), /* getDeviceInfo */
  API_CALL (getConstBankAddress),
  API_CALL (notSupported), /* singleStepWarp */

  /* 12.5 Extensions */
  API_CALL (readAllVirtualReturnAddresses),
  API_CALL (notSupported), /* getSupportedDebuggerCapabilities */
  API_CALL (readSmException),

  /* 12.6 Extensions */
  API_CALL (notSupported), /* executeInternalCommand */

  /* 12.7 Extensions */
  API_CALL (getGridInfo),
  API_CALL (getClusterDim),
  API_CALL (readWarpState),
  API_CALL (getClusterExceptionTargetBlock),

  /* 12.8 Extensions */
  API_CALL (readWarpResources),
};

CUDBGAPI cuCoreGetApi(CudaCore *cc)
{
  curcc = cc;

  return &cudbgCoreApi;
}
