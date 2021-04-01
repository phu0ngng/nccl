/*
 * Copyright 1993-2019 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_nvscibuf_h__
#define __cuda_nvscibuf_h__

#include "cuda.h"
#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"
#include "nvscibuf.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
//  Backdoor driver API for importing NvSciBuf memory
//------------------------------------------------------------------

typedef struct CUetblNvSciBufHndl_st *CUetblNvSciBufHndl;

CU_DEFINE_UUID(CU_ETID_NvSciBuf,
        0xff7b0cf6, 0x124d, 0x11e9, 0xab, 0x14, 0xd6, 0x63, 0xbd, 0x87, 0x3d, 0x93);

typedef struct CUetblNvSciBuf_st {
    size_t struct_size;
    // Import NvSciBuf handle
    CUresult (CUDAAPI *NvSciBufImport)(
            CUetblNvSciBufHndl *bufHandle_out,
            NvSciBufObj handle,
            unsigned long long size,
            unsigned int flags);

    // Get Mapped Buffer from Imported Buffer handle
    CUresult (CUDAAPI *NvSciBufGetMappedBuffer)(
            CUdeviceptr *devPtr,
            CUetblNvSciBufHndl bufHandle,
            unsigned long long offset,
            unsigned long long size,
            unsigned int flags);

    // Get mapped Array from Imported Buffer handle
    CUresult (CUDAAPI *NvSciBufGetMappedArray)(
            CUarray *array,
            CUetblNvSciBufHndl bufHandle,
            const CUDA_ARRAY3D_DESCRIPTOR *arrayDesc,
            unsigned long long offset);

    // Destroy imported Buffer Handle
    CUresult (CUDAAPI *NvSciBufDestroy)(
            CUetblNvSciBufHndl bufHandle);
} CUetblNvSciBuf;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard

