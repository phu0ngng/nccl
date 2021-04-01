/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_APIC_h__
#define __cuda_etbl_tools_APIC_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


/// \brief ToolsAPIC provides functions to deal with allocating memory at specific VAs.
CU_DEFINE_UUID(CU_ETID_ToolsAPIC,
    0x2fd5b1e6, 0x84cc, 0x4377, 0x93, 0x41, 0xcd, 0xde, 0xe7, 0xef, 0x7c, 0xe1);

typedef struct CUetblToolsAPIC_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    CUresult (CUDAAPI *MemAllocFixedVA) (
        CUdeviceptr *ppDeviceMem,
        size_t sizeInBytes,
        CUdeviceptr FixedVA,
        CUcontext ctx);

    CUresult (CUDAAPI *MemAllocManagedFixedVA) (
        CUdeviceptr *dptr,
        size_t bytesize,
        unsigned int flags,
        CUdeviceptr FixedVA,
        CUcontext ctx);    

    CUresult (CUDAAPI *MemHostAllocFixedVA) (
        void **pp,
        size_t bytesize,
        unsigned int Flags,
        CUdeviceptr FixedVA,
        CUcontext ctx);

    CUresult (CUDAAPI *MemAllocPitchFixedVA) (
        CUdeviceptr *dptr,
        size_t *pPitch,
        size_t WidthInBytes,
        size_t Height,
        unsigned int ElementSizeBytes,
        CUdeviceptr FixedVA,
        CUcontext ctx);
    CUresult (CUDAAPI *MemFreeFixedVA) (
        CUdeviceptr p,
        CUcontext ctx);
} CUetblToolsAPIC;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
