/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_memcopy_h__
#define __cuda_etbl_tools_memcopy_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


CU_DEFINE_UUID(CU_ETID_ToolsMemCopy,
    0x1b0f7735, 0x2e09, 0x4803, 0xa4, 0x8e, 0x5, 0x6f, 0xc4, 0x23, 0x96, 0x8d);

typedef struct CUetblToolsMemCopy_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief Inline memcpy from host --> device.
    /// This is useful for async-launch-debugging.  A tool can enqueue a
    /// block of host-memory to be copied prior to a launch, without
    /// requiring CPU/GPU synchronization.
    /// Call this on the CUcontext's thread to avoid threading issues.
    /// \param hMemObjDest corresponds to the destination memobj.
    /// \param pSource does not need to be pinned (pageable is OK)
    /// \param copySizeInBytes number of contiguous bytes to transfer
    CUresult (CUDAAPI *MemcpyInlineHtoD)(
        CUcontext ctx,
        CUtoolsStreamHandle stream,
        CUtoolsMemObjHandle hMemObjDest,
        size_t destOffset,
        const void *pSource,
        size_t copySizeInBytes);

    /// \brief memcpy from device --> host.
    CUresult (CUDAAPI *MemcpyDtoH)(
        CUcontext ctx,
        CUtoolsStreamHandle stream,
        CUtoolsMemObjHandle hMemObjSrc,
        uint64_t srcOffset,
        void *pDest,
        uint64_t copySizeInBytes);

    /// \brief Inline memcpy from host ---> device without any lock
    CUresult (CUDAAPI *MemcpyInlineHtoD1D)(
        CUcontext ctx,
        CUtoolsNvCurrent *pnvCurrent,
        size_t deviceAddr,
        const void *pSource,
        size_t copySizeInBytes,
        uint32_t membarType);

} CUetblToolsMemCopy;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
