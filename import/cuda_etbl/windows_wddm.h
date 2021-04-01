/*
* Copyright 1993-2017 by NVIDIA Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to NVIDIA
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of NVIDIA Corporation is prohibited.
*/

#ifndef __cuda_etbl_windows_wddm_h__
#define __cuda_etbl_windows_wddm_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CU_DEFINE_UUID(CU_ETID_WindowsWddm,
    0xd3c821bd, 0x3c92, 0x470d, 0x92, 0x7f, 0x43, 0xb8, 0x81, 0xc9, 0x2, 0x9e);


#define CUDA_WDDM_MAX_NODE_COUNT   20

typedef enum CUwddmNodeType_enum
{
    WddmNode3D = 0x0,
    WddmNode3DLowLatency = 0x1,
    WddmNodeDmaCopy0 = 0x10,
    WddmNodeDmaCopy1 = 0x11,
    WddmNodeDmaCopy2 = 0x12,
    WddmNodeDmaCopy3 = 0x13,
    WddmNodeDmaCopy4 = 0x14,
    WddmNodeDmaCopy5 = 0x15,

    WddmNodeUnknown = 0x0
} CUwddmNodeType;

typedef struct CUwddmNodeInfo_st
{
    CUwddmNodeType type;
    uint32_t       hwScheduling;
} CUwddmNodeInfo;

typedef struct CUwddmAdapterInfo_st
{
    uint32_t nodeCount;
    CUwddmNodeInfo nodeInfo[CUDA_WDDM_MAX_NODE_COUNT];
    uint32_t cilpSupported;
    uint64_t cilpCount;
    uint32_t tdrEnabled;
} CUwddmAdapterInfo;

typedef struct CUetblWindowsWddm_st {
    // This export table supports versioning by adding to the end without
    // changing the ETID.  The struct_size field will always be set to the
    // size in bytes of the entire export table structure.
    size_t struct_size;

    /// Return the WDDM adapter info for a CUDA device.
    CUresult (CUDAAPI *GetAdapterInfoFromCudaDevice)(CUdevice device, uint32_t adapterInfoSize, CUwddmAdapterInfo *pAdapterInfo);

    /// Put the WDDM Device associated to the CUDA context in error mode
    CUresult(CUDAAPI *MarkDeviceAsErrorForCudaContext)(CUcontext context);
} CUetblWindowsWddm;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
