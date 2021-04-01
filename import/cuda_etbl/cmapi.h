/*
 * Copyright 1993-2012 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __CMAPI_H_
#define __CMAPI_H_

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CU_DEFINE_UUID(CU_ETID_CMAPI, // {B3F3ED02-C296-4e90-B637-A25AD1B1A7EE}
    0xb3f3ed02, 0xc296, 0x4e90, 0xb6, 0x37, 0xa2, 0x5a, 0xd1, 0xb1, 0xa7, 0xee);

// Default API flags
#define CU_MEM_ACQUIRE_FLAGS_NONE 0x0
#define CU_MEM_RELEASE_FLAGS_NONE 0x0

// Global flags
#define CU_CMAPI_FLAGS_NONE     0x0
#define CU_CMAPI_FLAGS_IMPLICIT 0x1

typedef struct CUetblCMAPI_st {
    size_t struct_size;
    CUresult (CUDAAPI *eticmInit)(unsigned int flags);

    CUresult (CUDAAPI *eticmMemAcquire)(void* dptr, size_t size, unsigned int flags);
    CUresult (CUDAAPI *eticmMemRelease)(void* dptr, unsigned int flags);
} CUetblCMAPI;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __CMAPI_H_
