/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_wddm_h__
#define __cuda_etbl_tools_wddm_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#include <Unknwn.h>
#include <d3dkmthk.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


/// The type of D3D interop.
/// The number appended to these constants is the first version of CUDA where
/// that type of interop was introduced.
typedef enum CUtools_d3d_interop_type_enum {
    CU_TOOLS_INTEROP_TYPE_INVALID   = 0,
    CU_TOOLS_INTEROP_TYPE_D3D9_1x   = 1, // interop with d3d9 started by cuD3DBegin on XP
    CU_TOOLS_INTEROP_TYPE_D3D9_20   = 2, // interop with d3d9 started by cuD3DCtxCreate on XP or LH
    CU_TOOLS_INTEROP_TYPE_D3D10_20  = 3, // interop with d3d10 started by cuD3D10CtxCreate on LH
    CU_TOOLS_INTEROP_TYPE_D3D11_30  = 4, // interop with d3d11 started by cuD3D11CtxCreate on LH
    // --- always add new constants to the end here ---
    CU_TOOLS_INTEROP_TYPE_SIZE,
    CU_TOOLS_INTEROP_TYPE_FORCE_INT = 0x7fffffff
} CUtools_d3d_interop_type;

/// For CUetblToolsWddm::GetContextHandles()
typedef struct CUtoolsContextHandlesWddm_st {
    /// The struct_size field will always be set to the size in bytes of
    /// the entire structure.
    uint32_t struct_size;
    uint32_t reserved0;

    /// WDDM hAdapter
    D3DKMT_HANDLE hAdapter;
    /// WDDM hDevice
    D3DKMT_HANDLE hDevice;

    /// pD3DDevice9, pD3D10Device, ID3D11Device, etc.
    IUnknown *pD3DDevice;
    /// The type of interop indicates how pD3DDevice should be interpreted.
    CUtools_d3d_interop_type D3DInteropType;
    CU_32_BIT_PAD_ON_64_BIT_BUILDS(reserved1)
} CUtoolsContextHandlesWddm;

//  For CUetblToolsWddm::GetMemObjHandles().
//  This structure is not version-safe. Please create a new tools API and corresponding structure
//  (preferably one which IS version-safe) instead of modifying this one.
typedef struct CUtoolsMemObjHandlesWddm_st {
    D3DKMT_HANDLE hResource;
    D3DKMT_HANDLE hAllocation;

    /// memBlockBaseAddress is the base-address of the WDDM hAllocation.
    /// memblock and hAllocation are 1:1.
    void *memBlockBaseAddress;
    size_t memBlockSizeInBytes;

    /// Virtual address of the underlying memobj.
    void *memObjBaseAddress;
    size_t memObjSizeInBytes;
} CUtoolsMemObjHandlesWddm;

/// For CUetblToolsWddm::GetContextRmHandles()
typedef struct CUtoolsContextRmHandlesWddm_st {
    /// The struct_size field will always be set to the size in bytes of
    /// the entire structure.
    uint32_t struct_size;
    uint32_t reserved0;
    
    /// Compute channel rmClient
    uint32_t computeRmClient;
    
    /// Compute channel rmChannel
    uint32_t computeRmChannel;
} CUtoolsContextRmHandlesWddm;

// NVL_ESC_ID_COMMON_GET_ALLOC_DEBUG_INFO
// Note: D3DKMT_HANDLE is a UINT, which is always 4 bytes.
// Both KMD handles must be specified if either is.
typedef struct CUtoolsWddmAllocHandles_v2_st {
    size_t struct_size;
    D3DKMT_HANDLE hAdapter;     // WDDM hAdapter
    D3DKMT_HANDLE hDevice;      // WDDM hDevice
    D3DKMT_HANDLE hAllocation;  // WDDM hAllocation
    uint32_t allocType;         // DXGK_HANDLE_TYPE, 1=allocation, 2=resource
    uint32_t deviceSpecific;
    CU_32_BIT_PAD_ON_64_BIT_BUILDS(reserved0)
    uint64_t hkmdAdapter;       // WDDM KMD adapter handle, overrides hAdapter
    uint64_t hkmdDevice;        // WDDM KMD device handle, overrides hDevice
} CUtoolsWddmAllocHandles_v2;

// Older version for backwards compatibility
typedef struct CUtoolsWddmAllocHandles_v1_st {
    size_t struct_size;
    D3DKMT_HANDLE hAdapter;     // WDDM hAdapter
    D3DKMT_HANDLE hDevice;      // WDDM hDevice
    D3DKMT_HANDLE hAllocation;  // WDDM hAllocation
    uint32_t allocType;         // DXGK_HANDLE_TYPE, 1=allocation, 2=resource
    uint32_t deviceSpecific;
} CUtoolsWddmAllocHandles_v1;

typedef CUtoolsWddmAllocHandles_v2 CUtoolsWddmAllocHandles;


// NVL_ESC_ID_COMMON_GET_ALLOC_DEBUG_INFO
typedef struct CUtoolsWddmAllocDebugInfo_st {
    size_t struct_size;
    uint32_t rmClient;          // RM hClient
    uint32_t rmDevice;          // RM hDevice
    // Note: Added padding after shipping this structure to
    // explicitly specify padding that was previously implicit.
    CU_32_BIT_PAD_ON_32_BIT_BUILDS(reserved0)
    uint64_t allocPAddr;        // GPU Physical Address
    uint64_t allocVAddr;        // GPU Virtual Address
    uint64_t allocSize;         // size of allocation (bytes)
    uint32_t allocSegmentID;
    uint32_t allocHintHandle;
} CUtoolsWddmAllocDebugInfo;

// NVL_ESC_ID_COMMON_GET_TDRCOUNT
typedef struct CUtoolsWddmTdrCount_st {
    size_t struct_size;
    uint32_t tdrCount;
    uint32_t rcCount;
    uint32_t maxConsecutiveRC;
    CU_32_BIT_PAD_ON_64_BIT_BUILDS(reserved1)
} CUtoolsWddmTdrCount;

/// This table allows tools like debuggers to override WDDM behavior.
CU_DEFINE_UUID(CU_ETID_ToolsWddm,
    0xefcbab78, 0xa43c, 0x4ec0, 0x89, 0x0, 0x1e, 0x1e, 0xc2, 0x8f, 0xbf, 0x44);

typedef struct CUetblToolsWddm_st {
    // This export table supports versioning by adding to the end without
    // changing the ETID.  The struct_size field will always be set to the
    // size in bytes of the entire export table structure.
    size_t struct_size;

    /// Debuggers should enable this to prevent deadlock issue between
    /// D3DKMTRender, D3DKMTEscape, and GPU stalling @ breakpoint.  See
    /// Bug #590780 for more detail.
    /// Threading: Not thread-safe by design.  Only call this at startup.
    void (CUDAAPI *ForceFlushAllRenderCalls)(uint32_t bEnabled);

    /// Typically a debugger will set this so that all WDDM hAllocations
    /// (corresponding to vidmem allocations) can be accessed on the CPU
    /// via D3DKMTLock().
    /// Threading: Not thread-safe by design.  Only call this at startup.
    void (CUDAAPI *ForceAllAllocationsLockable)(uint32_t bEnabled);

    /// Get WDDM handles for the CUcontext.
    CUresult (CUDAAPI *GetContextHandles)(
        CUtoolsContextHandlesWddm *pContextHandlesWddm,
        CUcontext ctx);
    
    /// Get WDDM hContext that is owner of the Node3D (where the compute engine is bound).
    /// On Tesla and Fermi, the hContext has a single underlying GPU channel
    /// where all compute work is pushed; there is only one such hContext per
    /// CUcontext on these GPUs.
    void (CUDAAPI *GetComputeNodeWddmContext)(
        D3DKMT_HANDLE *phContext,
        CUcontext ctx);

    /// Get WDDM allocation handles and offset information for a memobj.
    CUresult (CUDAAPI *MemObjGetWddmHandles)(
        CUtoolsMemObjHandlesWddm *pMemObjHandlesWddm,
        CUcontext ctx,
        CUtoolsMemObjHandle hMemObj);

    /// Get RM client and channels for a WDDM CUcontext.
    CUresult (CUDAAPI *GetContextRmHandles)(
        CUtoolsContextRmHandlesWddm *pContextRmHandlesWddm,
        CUcontext ctx);

    /// NVL_ESC_ID_COMMON_CHECK_NV for hAdapter.
    /// Returns CUDA_SUCCESS if NVIDIA adapter; otherwise returns an error code.
    CUresult (CUDAAPI *IsNvidiaAdapter)(D3DKMT_HANDLE hAdapter);

    /// NVL_ESC_ID_COMMON_GET_ALLOC_DEBUG_INFO
    /// Request RM handles for a WDDM allocation.
    CUresult (CUDAAPI *GetAllocDebugInfo)(
        const CUtoolsWddmAllocHandles *pAllocHandles,
        CUtoolsWddmAllocDebugInfo *pInfo);

    /// NVL_PRIV_QUERYADAPTERINFO/NV_SYSTEM_INFO
    /// Request underlying RM device/subdevice instance ordinals for a WDDM hAdapter.
    /// Pre-conditions:
    ///     pSubDeviceInstances points at an array of *pSubDeviceCount elements.
    /// Post-conditions:
    ///     *pSubDeviceCount is updated with the true number of subDevice instances.
    CUresult (CUDAAPI *AdapterGetRmSubDeviceInstances)(
        D3DKMT_HANDLE hAdapter,
        uint32_t *pDeviceInstance,
        uint32_t *pSubDeviceInstances,
        size_t *pSubDeviceCount);

    /// Request underlying RM client/device/subdevice handles for a WDDM hAdapter.
    CUresult (CUDAAPI *AdapterGetRmHandles)(
        D3DKMT_HANDLE hAdapter,
        uint32_t *pRmClient,
        uint32_t *pRmDevice,
        uint32_t *pRmSubDevices,
        size_t *pRmSubDeviceCount);

    CUresult (CUDAAPI *AdapterGetTdrCount)(
        D3DKMT_HANDLE hAdapter,
        CUtoolsWddmTdrCount *pTdrCount);

    /// Should only be used for chip bringup or other unsupported scenario.
    CUresult (CUDAAPI *MapBar0ToUser)(
        D3DKMT_HANDLE hAdapter,
        void **ppBar0);

    /// Should only be used for chip bringup or other unsupported scenario.
    CUresult (CUDAAPI *UnmapBar0FromUser)(
        D3DKMT_HANDLE hAdapter,
        void *pBar0);

    /// Returns the highest supported GR engine class.
    CUresult (CUDAAPI *AdapterGetGrEngineClass)(
        D3DKMT_HANDLE hAdapter,
        CUtools_gr_engine_type engineType,
        CUtools_gr_engine_class *pEngineClass);

    /// Returns the default handle value used for binding a
    /// graphics engine class instance.  For example, NV50_COMPUTE
    /// or FERMI_A or FERMI_COMPUTE_B all get bound to this.
    CUresult (CUDAAPI *GetDefaultRmHandleForGrEngineClass)(
        CUtools_gr_engine_class engineClass,
        uint32_t *pRmEngineClass);

    /// Translates user-mode D3DKMT_HANDLE objects into KMD handles.
    CUresult (CUDAAPI *TranslateWddmHandles)(
        D3DKMT_HANDLE hAdapter,
        D3DKMT_HANDLE hDevice,
        D3DKMT_HANDLE hContext,
        uint64_t *phkmdAdapter,
        uint64_t *phkmdDevice,
        uint64_t *phkmdContext);

    /// AdapterGetRmHandles reimplementation using KMD adapter handles.
    CUresult (CUDAAPI *KmdAdapterGetRmHandles)(
        uint64_t hkmdAdapter,
        uint32_t *pRmClient,
        uint32_t *pRmDevice,
        uint32_t *pRmSubDevices,
        size_t *pRmSubDeviceCount);

    /// AdapterGetTdrCount reimplementation using KMD adapter handles.
    CUresult (CUDAAPI *KmdAdapterGetTdrCount)(
        uint64_t hkmdAdapter,
        CUtoolsWddmTdrCount *pTdrCount);

    /// MapBar0ToUser reimplementation using KMD adapter handles.
    CUresult (CUDAAPI *KmdMapBar0ToUser)(
        uint64_t hkmdAdapter,
        void **ppBar0);

    /// UnmapBar0FromUser reimplementation using KMD adapter handles.
    CUresult (CUDAAPI *KmdUnmapBar0FromUser)(
        uint64_t hkmdAdapter,
        void *pBar0);

} CUetblToolsWddm;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard


