/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_pushbuffer_h__
#define __cuda_etbl_tools_pushbuffer_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum CUtools_tools_semaphore_acquire_flags_enum
{
    CU_TOOLS_SEMAPHORE_ACQUIRE_FLAGS_NONE = 0,
    CU_TOOLS_SEMAPHORE_ACQUIRE_FLAGS_GEQ = 1,
    CU_TOOLS_SEMAPHORE_ACQUIRE_FLAGS_EQUAL = 2,
    CU_TOOLS_SEMAPHORE_ACQUIRE_FLAGS_AND = 3,
    CU_TOOLS_SEMAPHORE_ACQUIRE_FLAGS_NOR = 4,
    CU_TOOLS_SEMAPHORE_ACQUIRE_FLAGS_OP_MASK = 7,

    CU_TOOLS_SEMAPHORE_ACQUIRE_FLAGS_NOSWITCH = 8,

    // --- always add new constants to the end here ---
    CU_TOOLS_SEMAPHORE_ACQUIRE_FLAGS_SIZE,
    CU_TOOLS_SEMAPHORE_ACQUIRE_FLAGS_INT = 0x7fffffff
} CUtools_tools_semaphore_acquire_flags;

typedef enum CUtools_tools_semaphore_release_flags_enum
{
    CU_TOOLS_SEMAPHORE_RELEASE_FLAGS_NONE = 0,
    CU_TOOLS_SEMAPHORE_RELEASE_FLAGS_TIMESTAMP_DISABLE = 1,
    CU_TOOLS_SEMAPHORE_RELEASE_FLAGS_WFI_DISABLE = 2,

    // --- always add new constants to the end here ---
    CU_TOOLS_SEMAPHORE_RELEASE_FLAGS_SIZE,
    CU_TOOLS_SEMAPHORE_RELEASE_FLAGS_INT = 0x7fffffff
} CUtools_tools_semaphore_release_flags;

typedef enum CUtools_tools_semaphore_reduction_flags_enum
{
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_MIN_SIGNED   = 0,
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_MAX_SIGNED   = 1,
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_MIN_UNSIGNED = 2,
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_MAX_UNSIGNED = 3,
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_XOR          = 4,
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_AND          = 5,
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_OR           = 6,
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_ADD_SIGNED   = 7,
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_ADD_UNSIGNED = 8,
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_INC          = 9,
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_DEC          = 10,

    // --- always add new constants to the end here ---
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_SIZE,
    CU_TOOLS_SEMAPHORE_REDUCTION_FLAGS_INT = 0x7fffffff
} CUtools_tools_semaphore_reduction;

// Tools can assume this is the largest size that can be submitted in
// one pushbuffer.  It may be possible to submit larger sizes, but if
// attempting to submit more than CU_TOOLS_PUSHBUFFER_MAX_WORDS, tools
// must be tolerant of error returns and retry with smaller buffers.
#define CU_TOOLS_PUSHBUFFER_MAX_WORDS (2048)  // Each word is a uint32_t

/// PushBufferHal provides functions for push-buffer manipulation.
CU_DEFINE_UUID(CU_ETID_ToolsPushBufferHal,
    0x99ffb1a6, 0xc4ec, 0x4fc9, 0x92, 0xf9, 0x19, 0x28, 0x66, 0x3d, 0x55, 0x85);

typedef struct CUetblToolsPushBufferHal_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// Use this function to emit GPU commands into the pushbuffer
    /// to set a new CTA entry point.  NOTE:  Only implemented for
    /// Tesla and Fermi architectures.  For QMD-based architectures,
    /// this must be done by modifying fields in the QMD.
    CUresult (CUDAAPI *SetEntryPoint)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        uint64_t entryPointPc);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to set SM_TIMEOUT_INTERVAL. NOTE:  Only implemented for
    /// Tesla and Fermi architectures.  This function is not used
    /// for recent debuggers, because channel resets are a better
    /// way to stop the GPU.
    CUresult (CUDAAPI *SetSmTimeoutInterval)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        uint32_t smTimeoutIntervalLog2);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to invalidate all GPU caches (instructions, data, texture, constant).
    CUresult (CUDAAPI *InvalidateAllCaches)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to invalidate GPU instruction caches without wait-for-idle.
    CUresult (CUDAAPI *InvalidateInstructionCaches)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx);

    CUresult (CUDAAPI *WaitForIdle)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx);

    CUresult (CUDAAPI *PmTriggerBegin)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx);

    CUresult (CUDAAPI *PmTriggerEnd)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to set a new CTA entry point.  NOTE:  Only implemented for
    /// Tesla and Fermi architectures.  For QMD-based architectures,
    /// this must be done by modifying fields in the QMD.
    CUresult (CUDAAPI *SetSharedMemorySize)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        uint32_t size);

    /// The following functions are used for externally-generated pushbuffer methods
    CUresult (CUDAAPI *PushMethods)(
        CUtoolsNvCurrent *pnvCurrent,
        const uint32_t *pBuffer,
        uint32_t numWords);

    /// Note: this function is only useful on Tesla and Fermi.
    /// Starting with Kepler, it changed from dynamic to static,
    /// so tools can use the set-in-stone values defined in the
    /// class headers, e.g. NVA06F_SUBCHANNEL_COMPUTE.
    CUresult (CUDAAPI *GetComputeSubchannelId)(
        uint32_t *pSubChannelId);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to invalidate all TLBs (HUB clients as well as GPC clients)
    CUresult (CUDAAPI *InvalidateAllTLB)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to flush and invalidate GPU L2 cache.  It is not safe to
    /// invalidate L2 without flushing first.
    CUresult (CUDAAPI *FlushAndInvalidateL2Cache)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to flush L2 GPU cache.
    CUresult (CUDAAPI *FlushL2Cache)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx);

    /// Use this function to emit GPU commands using a new pushbuffer,
    /// under the ctx lock.
    CUresult (CUDAAPI *SubmitComputePushbuffer)(
        CUcontext ctx,
        CUtoolsStreamHandle stream,    // Use barrier stream if null is passed
        const uint32_t *pBuffer,
        uint32_t numWords);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to invalidate GPU constant cache without wait-for-idle
    CUresult (CUDAAPI *InvalidateConstantCache)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to invalidate GPU texture cache with or without wait-for-idle
    CUresult (CUDAAPI *InvalidateTextureCache)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        uint32_t waitForIdle);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to memcpy from src to dst
    /// membarType is an enum value from CUtoolsMembarType
    CUresult (CUDAAPI *Memcpy)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        CUdeviceptr dst,
        CUdeviceptr src,
        size_t copySizeInBytes,
        uint32_t membarType);

    /// Use this function to emit GPU commands using a new pushbuffer
    /// under the ctx lock in a channel
    /// channelUse is an enum value from CUtools_channel_use_type
    CUresult (CUDAAPI *SubmitPushbuffer)(
        CUcontext ctx,
        uint32_t channelUse,
        CUtoolsStreamHandle stream,    // Use barrier stream if null is passed
        const uint32_t *pBuffer,
        uint32_t numWords);

    /// Use this function GPU commands into the pushbuffer
    /// to issue a Host WFI
    /// Note: this also involves a sysmembar on Fermi and gk10x
    CUresult(CUDAAPI *HostWfi)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to issue a Semaphore acquire targeting the host unit
    /// The payload will be a 32 bits payload
    CUresult (CUDAAPI *SemaphoreAcquireHost)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        CUdeviceptr semaphorePtr,
        uint32_t payload,
        uint32_t flags);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to issue a Semaphore acquire targeting the host unit
    /// The payload will be a 64 bits payload
    /// NOTE: 64 Bits payloads require Volta or higher
    CUresult (CUDAAPI *SemaphoreAcquireHost64)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        CUdeviceptr semaphorePtr,
        uint64_t payload,
        uint32_t flags);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to issue a Semaphore realease targeting the host unit
    /// The payload will be a 32 bits payload
    CUresult (CUDAAPI *SemaphoreReleaseHost)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        CUdeviceptr semaphorePtr,
        uint32_t payload,
        uint32_t flags);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to issue a Semaphore realease targeting the host unit
    /// The payload will be a 64 bits payload
    /// NOTE: 64 Bits payloads require Volta or higher
    CUresult (CUDAAPI *SemaphoreReleaseHost64)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        CUdeviceptr semaphorePtr,
        uint64_t payload,
        uint32_t flags);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to issue a Semaphore reduction targeting the host unit
    /// The payload will be a 32 bits payload
    CUresult (CUDAAPI *SemaphoreReductionHost)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        uint64_t semaphorePtr,
        uint32_t payload,
        uint32_t flags);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to issue a Semaphore reduction targeting the host unit
    /// The payload will be a 64 bits payload
    /// NOTE: 64 Bits payloads require Volta or higher
    CUresult (CUDAAPI *SemaphoreReductionHost64)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        uint64_t semaphorePtr,
        uint64_t payload,
        uint32_t flags);

    /// Use this function to emit GPU commands into the pushbuffer
    /// to inline memcpy from src to dst
    /// membarType is an enum value from CUtoolsMembarType
    CUresult (CUDAAPI *MemcpyI2M)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        CUdeviceptr dst,
        const void *src,
        size_t copySizeInBytes,
        uint32_t membarType);
} CUetblToolsPushBufferHal;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
