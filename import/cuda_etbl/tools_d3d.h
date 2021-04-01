/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_d3d_h__
#define __cuda_etbl_tools_d3d_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#include <Unknwn.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


CU_DEFINE_UUID(CU_ETID_ToolsD3DInterop,
    0xdc6e1b4c, 0x9074, 0x4ad6, 0xa8, 0xd5, 0x8e, 0x1a, 0xc9, 0x6b, 0xcf, 0x60);

typedef struct CUetblToolsD3DInterop_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \param pVersionMajorMinor is written with major version in high 16-bits, minor in low 16-bits
    /// \return Returns CUDA_ERROR_UNKNOWN if not a D3D context.
    CUresult (CUDAAPI *GetD3DVersion)(CUcontext ctx, uint32_t *pVersionMajorMinor);

    /// \brief Get the 'flags' parameter passed into cuD3D10RegisterResource()
    /// \param pResource should be a D3D Resource registered by the user program.
    /// \param pFlags will be written with the mapping type (device-pointer or array)
    CUresult (CUDAAPI *GetResourceMappingType)(
        CUcontext ctx, 
        IUnknown *pResource, 
        uint32_t *pFlags);

    /// \brief Get the number of DX subresources for pResource.
    /// This will be greater than 1 for textures with multiple mip-levels,
    /// or for 3D textures (each slice is a subresource).
    CUresult (CUDAAPI *GetSubResourceCount)(
        CUcontext ctx, 
        IUnknown *pResource, 
        uint32_t *pSubResourceCount);

    /// \brief If the resource is an array (texture type), the corresponding CUarray will be output. 
    /// \param pResource should be the ID3D10Resource registered by the user program.
    /// \param pArray will be written with the CUarray.
    /// \param pHandleForArray will be written with a DMAL handle.
    ///     Use the memory-related export-table to get further information.
    CUresult (CUDAAPI *GetArrayForSubResource)(
        CUcontext ctx, 
        IUnknown *pResource,
        uint32_t subResourceIndex,
        CUarray *pArray,
        CUtoolsMemObjHandle *pHandleForArray,
        size_t *pStartOffset);

    /// \brief If the resource is a device pointer, it will be output (as a void*).
    /// \param pResource should be the ID3D10Resource registered by the user program.
    /// \param pDevPtr will be written with the device pointer.
    /// \param pHandleForArray will be written with a DMAL handle.
    ///     Use the memory-related export-table to get further information.
    CUresult (CUDAAPI *GetPointerForSubResource)(
        CUcontext ctx, 
        IUnknown *pResource,
        uint32_t subResourceIndex,
        void **pDevPtr,
        CUtoolsMemObjHandle *pHandleForArray,
        size_t *pStartOffset);

} CUetblToolsD3DInterop;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
