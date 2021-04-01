/*
 * Copyright 2012-2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_mobile_h__
#define __cuda_etbl_mobile_h__

#include "cuda.h"
#include "cuda_uuid.h"
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
// Backdoor driver API for various Mobile only interfaces
//------------------------------------------------------------------
// This etbl contains functions meant for the NV internal users on the mobile platforms.
// These are used to perform some low-level operations needed for performance or for
// programming model support (e.g. RenderScript programming model).

CU_DEFINE_UUID(CU_ETID_MOBILE,
    0x2cd10c39, 0x745e, 0x4447, 0xb4, 0xa8, 0x56, 0xf9, 0xa1, 0x5b, 0x5e, 0x3d);

typedef enum CUmobilePerfHint_enum
{
    CU_MOBILE_PERF_HINT_DISABLE = 0,
    CU_MOBILE_PERF_HINT_ENABLE  = 1
} CUmobilePerfHint;

typedef struct CUetblMobile_t {
    size_t struct_size;
    // This is used to be able to access some managed buffer on CPU when a kernel is running.
    // (e.g. for RenderScript metadata)
    CUresult        (*etiMobileDisableMProtectOnAndroidOnThisMemobj)(CUcontext, void*, int);
    // This is used to enable/disable the perf hints we send on mobile to clock up the gpu.
    // Disabling the hints can reduce power consumption in some cases where less than max perf is
    // acceptable (e.g. Video decode)
    CUresult        (*etiMobilePerfHintOnStream)(CUstream, CUmobilePerfHint);
    // This is used to skip the UVM operations at sync. To perform the UVM operations a default
    // cuda*Synchrnoize API can be used.
    CUresult        (*etiMobileStreamSyncNoUvmNotify)(CUstream);
    // This is used to perform low-level CPU cache operations.
    CUresult        (*etiMobileDoCpuCacheOpForPtr)(CUcontext ctx, CUdeviceptr devPtr, size_t size, int cacheOp);
    // This is used to indicate that CUDA context is created and used by RenderScript.
    CUresult        (*etiMobileNotifyContextIsRenderScript)(CUcontext);
    // This is used to perform low-level GPU cache operations.
    CUresult        (*etiMobileGpuL2Sync)(CUcontext ctx);
    // This is similar to cuMemAllocManaged except that host mapping uses WriteCombine if available
    // and uncached otherwise
    CUresult        (*etiMobileMemAllocManagedHostCacheWC)(CUdeviceptr *devPtr, size_t size, int flags);
    // This creates a non-coherent zero-copy allocation that is cached on both Host and Device, used
    // for SOL perf evaluation testing by doing explicit cacheops via etbls.
    CUresult        (*etiMobileMemAllocHostDeviceIncoherent)(CUdeviceptr *devPtr, size_t size);
    // This is used to perform low-level CPU cache operations for a list of ranges
    CUresult        (*etiMobileDoCpuCacheOpForListOfRanges)(CUcontext ctx, CUdeviceptr *devPtrs, size_t *sizes, unsigned int count, int cacheOp);
    // This is used to toggle between Mprotect or Reserve implementation, synchronizes context internally, but avoid launching anything else on the same context while this is running.
    CUresult        (*etiMobileToggleReserveOrMprotectThreadUnsafe)(CUcontext ctx, unsigned int useMprotect);
    // This is similar to cuMemAlloc except that mapping to device is done on first access using
    // replayable page faults.
    // Coherence with host is guaranteed only if GPU cacheops are done.
    CUresult        (*etiMobileMemAllocUnmappedPte)(CUdeviceptr *devPtr, size_t size, int mapOnHost);
} CUetblMobile;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __cuda_etbl_mobile_h__
