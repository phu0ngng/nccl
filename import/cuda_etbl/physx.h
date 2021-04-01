/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_physx_h__
#define __cuda_etbl_physx_h__


#include "cuda.h"
#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
// Cuda Private API Interfaces for PhysX
//------------------------------------------------------------------

/* This provides backdoor interfaces used by PhysX
 */

CU_DEFINE_UUID(CU_ETID_PhysXInterface,
    0x8c0ba50c, 0x0410, 0x9a92, 0x89, 0xa7, 0xd0, 0xdf, 0x10, 0xe7, 0x72, 0x86);

typedef struct CUetblPhysXInterface_st {
    /* Size of this structure */
    size_t size;
    
    /* Create a new CUDA context on Node3dLowLatency.
     * - will usually it will just call cuCtxCreateOnNode3DLowLatency.
     */
    CUresult (CUDAAPI *cuCtxCreateOnNode3DLowLatency)(
        CUcontext *pctx,
        unsigned int flags,
        CUdevice dev);

} CUetblPhysXInterface;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
