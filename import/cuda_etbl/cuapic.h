/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuapic_h__
#define __cuapic_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
//  Backdoor driver API for Cuda APIC replay
//------------------------------------------------------------------

CU_DEFINE_UUID(CU_ETID_CUAPIC, 
    0x44f453cc, 0xb112, 0x4a56, 0x9c, 0xa2, 0xee, 0x0, 0x65, 0xc5, 0xc9, 0x99);

typedef struct {
    size_t struct_size;

    CUresult (CUDAAPI *cuapicModuleLoadDataEx)(CUmodule *module, 
                                               const void *image, 
                                               unsigned int numOptions, 
                                               CUjit_option *options, 
                                               void **optionValues,
                                               unsigned int numSym,
                                               char** symNames,
                                               CUdeviceptr_v1 *symDptrs);

    CUresult (CUDAAPI *cuapicMemAlloc)(CUdeviceptr_v1 *dptr, unsigned int bytesize);
    CUresult (CUDAAPI *cuapicMemAllocPitch)(CUdeviceptr_v1 *dptr, 
                                            unsigned int *pPitch,
                                            unsigned int WidthInBytes, 
                                            unsigned int Height, 
                                            unsigned int ElementSizeBytes);
    CUresult (CUDAAPI *cuapicMemHostAlloc)(void **pp, size_t bytes, unsigned int Flags, 
                                           CUdeviceptr_v1 dptr);
} CUetblCUAPIC;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
