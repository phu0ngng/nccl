/*
 * Copyright 1993-2019 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
 
#ifndef __cuda_nvscisync_h__
#define __cuda_nvscisync_h__
 
#include "cuda.h"
#include "cuda_uuid.h"
#include "nvscisync.h"
#include "nvscierror.h"
 
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
 
CU_DEFINE_UUID(CU_ETID_CudaNvSciSync,
    0x751e0e34, 0x2c95, 0x42fb, 0xab, 0xf1, 0xcd, 0x2b, 0xd4, 0xf5, 0x44, 0xd1);

//------------------------------------------------------------------
//  Backdoor driver API for signaling and waiting for NvSciSync
//------------------------------------------------------------------
 
typedef struct CUetblCudaNvSciSync_st {
    size_t struct_size;
    /**
     * API to inform NvSciSync the capabilities of cuda device
     */ 
    CUresult (*cueDeviceGetNvSciSyncAttributes)(NvSciSyncAttrList attr, CUdevice dev, int flags);
    /**
     * Allows cuda stream to signal an NvSciSyncFence, this fence can be waited upon
     * by a different thread, process, SOC-engine, CPU etc.
     * This API remains agnostic as to who would potentially use the fence downstream.
     */ 
    CUresult (*cueStreamSignalNvSciSync) (CUstream hStream, NvSciSyncObj nvSciSyncObj, NvSciSyncFence* nvSciSyncFence);
    CUresult (*cueStreamSignalNvSciSync_ptsz) (CUstream hStream, NvSciSyncObj nvSciSyncObj, NvSciSyncFence* nvSciSyncFence);
    /**
     * Allows cuda stream to wait for an NvSciSyncFence, this has similar effect as cuStreamWaitEvent.
     * The fence could be have sent by a different thread, process, SOC-engine, CPU etc
     * This API remains agnostic as to which upstream entity sent the fence
     */ 
    CUresult (*cueStreamWaitNvSciSync) (CUstream hStream, NvSciSyncObj nvSciSyncObj, NvSciSyncFence* nvSciSyncFence);
    CUresult (*cueStreamWaitNvSciSync_ptsz) (CUstream hStream, NvSciSyncObj nvSciSyncObj, NvSciSyncFence* nvSciSyncFence);
} CUetblCudaNvSciSync;
 
#if defined(__CUDA_API_PER_THREAD_DEFAULT_STREAM)
    #define cueStreamSignalNvSciSync          __CUDA_API_PTSZ(cueStreamSignalNvSciSync)
    #define cueStreamWaitNvSciSync            __CUDA_API_PTSZ(cueStreamWaitNvSciSync)
#endif
#ifdef __cplusplus
}
#endif // __cplusplus
 
#endif // file guard
