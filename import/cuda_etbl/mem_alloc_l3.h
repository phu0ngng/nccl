/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_mem_alloc_l3_h__
#define __cuda_etbl_mem_alloc_l3_h__

#include "cuda.h"
#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"
#include "nvtypes.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

///------------------------------------------------------------------------
///  CUDA L3 Cached Memory Allocation APIs Summary
///------------------------------------------------------------------------
///
/// These APIs allows programmer to allocate L3 Cached enabled CUDA Memory blocks.
///
///

CU_DEFINE_UUID(CU_ETID_MemAllocL3Cached,
    0x7310c5f2, 0x112f, 0x4730, 0xa9, 0xc5, 0x2a, 0x31, 0xca, 0xe6, 0x95, 0x41);

typedef struct CUetblMemAllocL3Cached_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    CUresult (CUDAAPI* memGetL3CacheSize) (
        size_t* pByteSize);

    CUresult (CUDAAPI* memAlloc) (
        CUdeviceptr* dptr,
        size_t byteSize);

    CUresult (CUDAAPI* memAllocPitch) (
        CUdeviceptr* dptr,
        size_t* pPitch,
        size_t widthInBytes,
        size_t height,
        unsigned int  elementSizeBytes);
        
    CUresult (CUDAAPI* memAlloc_v2) (
        CUdeviceptr* dptr,
        NvBool physContiguous,
        size_t byteSize);

    CUresult (CUDAAPI* memAllocPitch_v2) (
        CUdeviceptr* dptr,
        size_t* pPitch,
        size_t widthInBytes,
        size_t height,
        NvBool physContiguous,
        unsigned int  elementSizeBytes);

} CUetblMemAllocL3Cached;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
