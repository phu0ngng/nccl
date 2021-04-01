/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_runtime_h__
#define __cuda_etbl_tools_runtime_h__

#include "cuda.h"
#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"
#include "cuda_etbl/tools_callbacks.h"

#ifdef __cplusplus
extern "C" {
#endif

// -------------------- Hooks for runtime into driver callbacks ------------------------
CU_DEFINE_UUID(CU_ETID_ToolsRuntimeCallbackHooks,
    0x8c7994a0, 0x742e, 0x742e, 0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66);

typedef struct CUetblToolsRuntimeCallbackHooks_st {
    /// This export table supports versioning by adding to the end without changing
    /// the ETID.  The struct_size field will always be set to the size in bytes of
    /// the entire export table structure.
    size_t struct_size;

    // Issues callback from runtime api trace domain to tools api callback subscribers.
    CUresult (CUDAAPI *IssueCallback)(uint32_t cbid, const void *pParams);

    // Returns buffer to runtime which indicates which callbacks in the runtime api trace
    // domain are enabled/disabled. The number of entries in this buffer is also provided.
    void (CUDAAPI *GetEnableBuffer)(volatile uint32_t **rtCallbackEnable, size_t *bufferSize);

    // Returns the unique per context id for a driver stream
    CUresult (CUDAAPI *GetStreamId)(CUcontext context, CUstream stream, uint64_t *driverId);

    // Returns the unique per process id for a driver context 
    CUresult (CUDAAPI *GetContextId)(CUcontext context, uint64_t *contextId);

    // Issues callback from cuda runtime internal domain to tools api callback subscribers.
    CUresult (CUDAAPI *InternalIssueCallback)(uint32_t cbid, const void *pParams);

    // Returns buffer to runtime which indicates which callbacks in the cuda runtime internal
    // domain are enabled/disabled. The number of entries in this buffer is also provided.
    void (CUDAAPI *InternalGetEnableBuffer)(volatile uint32_t **rtCallbackEnable, size_t *bufferSize);
} CUetblToolsRuntimeCallbackHooks;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // file guard
