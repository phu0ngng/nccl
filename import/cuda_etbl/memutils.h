/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_memutils_h__
#define __cuda_etbl_memutils_h__

#include "cuda.h"
#include "cuda_uuid.h"
#include "nvtypes.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CU_DEFINE_UUID(CU_ETID_Memutils, 
    0x46f3f6b7, 0x70f0, 0x43ba, 0xab, 0x76, 0x4d, 0x0, 0xa3, 0xb2, 0xeb, 0x4f);

typedef enum CUl2SectorPromotionGranularity_enum
{
    L2SectorPromotionDefault = 0x0,
    L2SectorPromotionOff = 0x1,
    L2SectorPromotion64B = 0x2,
    L2SectorPromotion128B = 0x3,
} CUl2SectorPromotionGranularity;

typedef struct CUetblMemory_st {
    // This export table supports versioning by adding to the end without
    // changing the ETID.  The struct_size field will always be set to the
    // size in bytes of the entire export table structure.
    size_t struct_size;

    // Set the prefered L2 sector promotion granularity for the specified context
    CUresult(CUDAAPI *SetL2SectorPromotion)(CUcontext context, CUl2SectorPromotionGranularity granularity);

} CUetblMemutils;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
