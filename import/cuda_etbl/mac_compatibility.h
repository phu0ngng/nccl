/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_mac_compatibility_h__
#define __cuda_etbl_mac_compatibility_h__

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


CU_DEFINE_UUID(CU_ETID_MacCompatibility,
    0x2813c2b1, 0x19d2, 0x4f47, 0xba, 0xeb, 0x93, 0x90, 0x69, 0x19, 0x80, 0x1d);

typedef struct CUetblMacCompatibility_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    CUresult (*GetAPILoaded)(void);

    CUresult (*GetRMVerOutOfRange)(void);

} CUetblMacCompatibility;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
