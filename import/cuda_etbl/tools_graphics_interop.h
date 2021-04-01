/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_graphics_interop_h__
#define __cuda_etbl_tools_graphics_interop_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


/// Information for a mapped CUgraphicsResource. 
typedef struct CUtoolsGraphicsResourceMappingInfo_st
{
    size_t struct_size;

    void *devptr;
    size_t size;
    void *reserved1;
} CUtoolsGraphicsResourceMappingInfo;

// \brief ToolsGraphicsInterop deals with CUgraphicsResource 
CU_DEFINE_UUID(CU_ETID_ToolsGraphicsInterop,
    0x479ecf6a, 0x5e18, 0x4a41, 0x81, 0x97, 0xd8, 0x66, 0x3d, 0xd0, 0xce, 0xa5);

typedef struct CUetblToolsGraphicsInterop_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    CUresult (CUDAAPI *GraphicsResourceGetMappingInfo)(
        CUtoolsGraphicsResourceMappingInfo *pMappingInfo,
        CUgraphicsResource resource);

} CUetblToolsGraphicsInterop;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
