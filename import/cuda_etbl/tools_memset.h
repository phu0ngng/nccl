/*
 * Copyright 1993-2018 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_memset_h__
#define __cuda_etbl_tools_memset_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


CU_DEFINE_UUID(CU_ETID_ToolsMemset,
    0xeeb89aff, 0xd428, 0x4c0e, 0xb3, 0xb7, 0xfd, 0x72, 0x4a, 0x62, 0x77, 0xa8);

// enum for requesting a specific type of memset
typedef enum CUtoolsMemsetType_enum {
    CU_TOOLS_MEMSET_TYPE_DEFAULT   = 0,
    CU_TOOLS_MEMSET_TYPE_CE        = 1,
    CU_TOOLS_MEMSET_TYPE_KERNEL    = 2,
    CU_TOOLS_MEMSET_TYPE_INLINE    = 3,
    // add new values here
    CU_TOOLS_MEMSET_TYPE_FORCE_INT = 0x7fffffff
} CUtoolsMemsetType;

typedef struct CUtoolsMemsetDesc_st
{
    uint32_t struct_size;
    uint32_t reserved0;

    uint64_t dptr;        // destination device pointer

    uint32_t value;       // value to set
    uint32_t elementSize; // element size in bytes (1,2,4)

    uint64_t width;       // width in elements
    uint64_t pitch;       // pitch in bytes - use 0 for 1D memsets
    uint64_t height;      // height in rows - use 1 for 1D memsets

    uint32_t type;        // use values from CUtoolsMemsetType
    uint32_t reserved1;
} CUtoolsMemsetDesc;

typedef struct CUetblToolsMemset_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    CUresult (CUDAAPI *Memset)(
        CUcontext ctx,
        CUtoolsStreamHandle stream,
        CUtoolsMemsetDesc const *desc);

} CUetblToolsMemset;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
