/*
 * Copyright 1993-2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _p2p_loopback_h___
#define _p2p_loopback_h___

#include "cuda.h"
#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"
#include "nvtypes.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


CU_DEFINE_UUID(CU_ETID_P2PLoopback,
    0x1b90a00a, 0x228b, 0x43d6, 0xa1, 0x71, 0xe3, 0x98, 0x98, 0x70, 0x1c, 0x9a);


typedef struct CUetblP2PLoopback_st {
    size_t struct_size;

    // Enable loopback on device
    // Return value: CUDA_SUCCESS, CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
    //               CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_TOO_MANY_PEERS, CUDA_ERROR_MAP_FAILED,
    //               CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED 
    CUresult (CUDAAPI *DeviceEnableLoopback)(
        CUdevice device);

    // Disable loopback on device
    // Return value: CUDA_SUCCESS, CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
    CUresult (CUDAAPI *DeviceDisableLoopback)(
        CUdevice device);

    // Map inDevPtr as loopback, and store the loopback VA in outPtr.
    // Return value: CUDA_SUCCESS, CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,CUDA_ERROR_ALREADY_MAPPED,
    //               CUDA_ERROR_INVALID_VALUE, etc.
    CUresult (CUDAAPI *MemMapLoopback)(
        CUdeviceptr inDevPtr,
        CUdeviceptr *outPtr);

    // Unmap loopbackPtr.
    // Return value: CUDA_SUCCESS, CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, CUDA_ERROR_INVALID_VALUE, etc.
    CUresult (CUDAAPI *MemUnmapLoopback)(
        CUdeviceptr loopbackPtr);
} CUetblP2PLoopback;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard

