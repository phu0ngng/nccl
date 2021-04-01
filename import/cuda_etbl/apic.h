/*
 * Copyright 1993-2019 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_APIC_h__
#define __cuda_etbl_tools_APIC_h__

#include "cuda.h"

#include "cuda_uuid.h"
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

    /// \brief Allocated GPU memory at \p FixedVA Virtual Address
    ///
    /// \param ppDeviceMem - Returned device pointer
    /// \param sizeInBytes - Requested allocation size in bytes
    /// \param FixedVA     - Requested allocation Virtual Address
    /// \param ctx         - Context on which to perform allocation
    ///
    /// \return
    /// ::CUDA_SUCCESS,
    /// ::CUDA_ERROR_NOT_SUPPORTED,
    /// ::CUDA_ERROR_INVALID_VALUE,
    /// ::CUDA_ERROR_ALREADY_MAPPED,
    /// ::CUDA_ERROR_INVALID_CONTEXT,
    /// ::CUDA_ERROR_OUT_OF_MEMORY
    CUresult (CUDAAPI *MemAllocFixedVA) (
        CUdeviceptr *ppDeviceMem,
        size_t sizeInBytes,
        CUdeviceptr FixedVA,
        CUcontext ctx);

    /// \brief Allocated Managed memory at \p FixedVA Virtual Address
    ///
    /// \param dptr     - Returned device pointer
    /// \param bytesize - Requested allocation size in bytes
    /// \param flags    - Must be one of ::CU_MEM_ATTACH_GLOBAL or ::CU_MEM_ATTACH_HOST
    /// \param FixedVA  - Requested allocation Virtual Address
    /// \param ctx      - Context on which to perform allocation
    ///
    /// \return
    /// ::CUDA_SUCCESS,
    /// ::CUDA_ERROR_NOT_SUPPORTED,
    /// ::CUDA_ERROR_INVALID_VALUE,
    /// ::CUDA_ERROR_ALREADY_MAPPED,
    /// ::CUDA_ERROR_INVALID_CONTEXT,
    /// ::CUDA_ERROR_OUT_OF_MEMORY
    CUresult (CUDAAPI *MemAllocManagedFixedVA) (
        CUdeviceptr *dptr,
        size_t bytesize,
        unsigned int flags,
        CUdeviceptr FixedVA,
        CUcontext ctx);    

    /// \brief Allocated Pinned Host memory at \p FixedVA Virtual Address
    ///
    /// \param pp       - Returned host pointer to page-locked memory 
    /// \param bytesize - Requested allocation size in bytes 
    /// \param Flags    - Flags for allocation request
    /// \param FixedVA  - Requested allocation Virtual Address
    /// \param ctx      - Context on which to perform allocation
    ///
    /// \return
    /// ::CUDA_SUCCESS,
    /// ::CUDA_ERROR_NOT_SUPPORTED,
    /// ::CUDA_ERROR_INVALID_VALUE,
    /// ::CUDA_ERROR_ALREADY_MAPPED,
    /// ::CUDA_ERROR_INVALID_CONTEXT,
    /// ::CUDA_ERROR_OUT_OF_MEMORY
    CUresult (CUDAAPI *MemHostAllocFixedVA) (
        void **pp,
        size_t bytesize,
        unsigned int Flags,
        CUdeviceptr FixedVA,
        CUcontext ctx);

    /// \brief Allocated Pitched Host memory at \p FixedVA Virtual Address
    ///
    /// \param dptr             - Returned device pointer 
    /// \param pPitch           - Returned pitch of allocation in bytes 
    /// \param WidthInBytes     - Requested allocation width in bytes 
    /// \param Height           - Requested allocation height in rows
    /// \param ElementSizeBytes - Size of largest reads/writes for range
    /// \param FixedVA          - Requested allocation Virtual Address
    /// \param ctx              - Context on which to perform allocation
    ///
    /// \return
    /// ::CUDA_SUCCESS,
    /// ::CUDA_ERROR_NOT_SUPPORTED,
    /// ::CUDA_ERROR_INVALID_VALUE,
    /// ::CUDA_ERROR_ALREADY_MAPPED,
    /// ::CUDA_ERROR_INVALID_CONTEXT,
    /// ::CUDA_ERROR_OUT_OF_MEMORY
    CUresult (CUDAAPI *MemAllocPitchFixedVA) (
        CUdeviceptr *dptr,
        size_t *pPitch,
        size_t WidthInBytes,
        size_t Height,
        unsigned int ElementSizeBytes,
        CUdeviceptr FixedVA,
        CUcontext ctx);

    /// \brief Free memory allocated via routines in this export table.
    ///
    /// \param dptr - Pointer to memory to free
    ///
    /// \return
    /// ::CUDA_SUCCESS,
    /// ::CUDA_ERROR_INVALID_VALUE,
    /// ::CUDA_ERROR_INVALID_CONTEXT
    CUresult (CUDAAPI *MemFreeFixedVA) (
        CUdeviceptr p,
        CUcontext ctx);
} CUetblToolsAPIC;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
