/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cross_proc_p2p_h__
#define __cross_proc_p2p_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
// Backdoor for cross-process shared memory (including P2P)
//------------------------------------------------------------------

CU_DEFINE_UUID(CU_ETID_CROSS_PROC_P2P, 
    0x5339b54e, 0x4b5e, 0x422c, 0x96, 0x22, 0xd1, 0x79, 0x1f, 0xbf, 0xee, 0xf1);

typedef struct {
    // RM client that allocate the handle
    unsigned int rmClient;    
    // RM handle of the allocation
    unsigned int rmHandle;

    // offset inside that allocation
    unsigned long long offset;
    // size of the allocation
    unsigned long long size;

    // Rm GPU ID of the device this was allocated on
    unsigned int rmGpuId;
    // SLI sub-device index
    unsigned int subDeviceIndex;
} cudaSharedMemKey_t;

typedef struct {
    // allocate size bytes, map it to device pointer ptr, and assign key to
    // refer to this allocation
    CUresult (CUDAAPI *cudaMallocShared)(void **ptr, cudaSharedMemKey_t *key, size_t size);

    // map key (from another process) into the current context
    CUresult (CUDAAPI *cudaMapShared)(void **ptr, cudaSharedMemKey_t key);
} CUetblCrossProcP2P;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard

