/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_video_h__
#define __cuda_etbl_video_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

///------------------------------------------------------------------------
///  CUDA Video interop Summary
///------------------------------------------------------------------------
///
/// Low-level interop between the cuda video decode api and cuda
/// Provides the following functionality:
///
/// - Get OS-specific handles for the cuda device and cuda allocations
/// - Create video-engine-compatible 2D arrays (BL 64x16)
///

typedef enum CUvideo_interop_device_type_enum
{
    CU_VIDEO_INTEROP_DEVICE_TYPE_INVALID                = 0,
    CU_VIDEO_INTEROP_DEVICE_TYPE_RMAPI                  = 1,
    CU_VIDEO_INTEROP_DEVICE_TYPE_WDDM                   = 2,
    CU_VIDEO_INTEROP_DEVICE_TYPE_D3D_NVAPI              = 3,
    // --- always add new constants to the end here ---
    CU_VIDEO_INTEROP_DEVICE_TYPE_SIZE,
    CU_VIDEO_INTEROP_DEVICE_TYPE_FORCE_INT              = 0x7fffffff
} CUvideo_interop_device_type;

// Structure used to return OS-specific device handles
typedef struct CUvideo_interop_device_st {
    unsigned int device_type;
    union {
        struct {
            uint32_t hClient;
            uint32_t hDevice;
            uint32_t hSubDevice;
            uint32_t devInstance;
            uint32_t subdevInstance;
        } rmapi;
        struct {
            uint64_t hAdapter;
            uint32_t hDevice;
            uint32_t EngineOrdinal;
        } wddm;
        struct {
            void *pD3DDevice;
            uint32_t hAdapter;
            uint32_t hDevice;
        } d3d_nvapi;
    } handles;
} CUvideo_interop_device;

// Structure used to return allocation handles
typedef struct CUvideo_interop_allocation_st {
    union {
        struct {
            uint32_t hMemObj;
            uint32_t hCtxDma;
        } rmapi;
        struct {
            uint64_t hResource;
            uint64_t hAllocation;
        } wddm;
        struct {
            void *memptr; // Host memory that Amodel uses as simulated-GPU memory.  See IDirectAmodelAllocSurface()
        } amod;
    } handles;
    uint64_t AllocationOffset;
    uint64_t SizeInBytes;
    uint64_t SubAllocationOffset;
} CUvideo_interop_allocation;


CU_DEFINE_UUID(CU_ETID_VideoInterop,
    0x7b70f09a, 0x2d8e, 0x4cd8, 0x8e, 0x4e, 0xb9, 0x94, 0xc8, 0x2d, 0xdc, 0x35);

// XXX - egregious hack until video builds pick up new headers...
#if !defined(__CUDA_VIDEO_TYPE_HACK) && !defined(__CUDA_API_VERSION_INTERNAL)
#define __CUDA_VIDEO_TYPE_HACK
typedef CUdeviceptr           CUdeviceptr_v1;
typedef CUDA_ARRAY_DESCRIPTOR CUDA_ARRAY_DESCRIPTOR_v1;
#endif

typedef struct CUetblVideoInterop_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    CUresult (CUDAAPI *GetDeviceHandles)(
        CUvideo_interop_device *deviceHandles);

    CUresult (CUDAAPI *GetDevicePointerHandles)(
        CUvideo_interop_allocation *allocHandles,
        void *devptr);

    CUresult (CUDAAPI *GetDeviceArrayHandles)(
        CUvideo_interop_allocation *allocHandles,
        CUarray arr,
        int version);

    CUresult (CUDAAPI *CreateVideoArray)(
        CUarray *pHandle,
        const CUDA_ARRAY_DESCRIPTOR_v1 *pAllocateArray);

    CUresult (CUDAAPI *StreamReleaseSemaphore)(
        CUstream hStream, 
        CUdeviceptr dptr,
        uint32_t valueToRelease);

    CUresult (CUDAAPI *StreamAcquireSemaphore)(
        CUstream hStream, 
        CUdeviceptr dptr,
        uint32_t valueToAcquire);

    CUresult (CUDAAPI *StreamSignalSyncObject)(
        CUstream hStream, 
        uint64_t hSyncObject);

    CUresult (CUDAAPI *StreamWaitSyncObject)(
        CUstream hStream, 
        uint64_t hSyncObject);

    CUresult (CUDAAPI *CreateVideoArrayEx)(
        CUarray *pHandle,
        const CUDA_ARRAY_DESCRIPTOR_v1 *pAllocateArray, uint32_t flags);

} CUetblVideoInterop;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
