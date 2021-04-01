/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_launch_h__
#define __cuda_etbl_tools_launch_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


CU_DEFINE_UUID(CU_ETID_ToolsLaunch,
    0x2753fdb9, 0x2806, 0x4fc2, 0x87, 0xa9, 0x98, 0x8c, 0xdd, 0xf7, 0x30, 0x89);

typedef struct CUetblToolsLaunch_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    CUresult (CUDAAPI *LaunchKernel)(
        CUfunction f,
        uint32_t gridDimX,
        uint32_t gridDimY,
        uint32_t gridDimZ,
        uint32_t blockDimX,
        uint32_t blockDimY,
        uint32_t blockDimZ,
        uint32_t sharedMemBytes,
        CUtoolsStreamHandle hStream,
        void **kernelParams,
        void **extra);

    CUresult (CUDAAPI *MemsetD)(
        CUcontext ctx,
        uint64_t dptrDst,
        uint32_t value,
        uint32_t elementSize,
        size_t width,
        size_t pitch,
        size_t height,
        CUtoolsStreamHandle hStream);

    CUresult (CUDAAPI *LaunchCooperativeKernel)(
        CUfunction f,
        uint32_t gridDimX,
        uint32_t gridDimY,
        uint32_t gridDimZ,
        uint32_t blockDimX,
        uint32_t blockDimY,
        uint32_t blockDimZ,
        uint32_t sharedMemBytes,
        CUtoolsStreamHandle hStream,
        void **kernelParams);

} CUetblToolsLaunch;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
