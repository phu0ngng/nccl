/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __suballoc_h__
#define __suballoc_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
//  Backdoor driver API for Cuda memory suballocation testing
//------------------------------------------------------------------

CU_DEFINE_UUID(CU_ETID_suballoc, 
    0x88504084, 0x82bf, 0x4cc7, 0x9a, 0x60, 0xae, 0x1b, 0xa9, 0xb0, 0xfc, 0xa);

typedef struct CUetblSuballoc_st {
    size_t struct_size;

    CUresult (CUDAAPI *etiSuballocGetAllocatedUserBytes)(size_t* size);
    CUresult (CUDAAPI *etiSuballocGetGenericBlockSize)(size_t* size);
    CUresult (CUDAAPI *etiSuballocGetBinBytes)(size_t* size);
    CUresult (CUDAAPI *etiSuballocGetUserAllocatedBytes)(size_t* bytes);
} CUetblSuballoc;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
