/*
 * Copyright 2016 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_performance_h__
#define __cuda_etbl_performance_h__

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


CU_DEFINE_UUID(CU_ETID_Performance,
    0x6d0173b5, 0x226b, 0x40cb, 0x81, 0x12, 0x4d, 0xc6, 0x13, 0xab, 0x61, 0x4e);

typedef struct CUperformanceEntry_st {
    uint64_t nanoseconds;
    const char *name;
} CUperformanceEntry;

#define CUDA_PERF_GROUP_ALL ~(uint64_t)0
#define CUDA_PERF_GROUP_DEFAULT (1 << 0)
#define CUDA_PERF_GROUP_INIT (1 << 1)
#define CUDA_PERF_GROUP_KERNEL_LAUNCH (1 << 2)
#define CUDA_PERF_GROUP_CTX_CREATE (1 << 3)
#define CUDA_PERF_GROUP_APPLICATION (1 << 4)
#define CUDA_PERF_GROUP_SYSCALL (1 << 5)

#define CUDA_PERF_SUBGROUP_ALL ~(uint64_t)0
#define CUDA_PERF_SUBGROUP_DEFAULT (1 << 0)

// subgroups for CUDA_PERF_GROUP_SYSCALL
#define CUDA_PERF_SUBGROUP_RM (1 << 1)
#define CUDA_PERF_SUBGROUP_UVM (1 << 2)
#define CUDA_PERF_SUBGROUP_CUOS (1 << 3)

typedef struct CUetblPerformance_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;
    CUresult (CUDAAPI *Init)(void);
    CUresult (CUDAAPI *Enable)(CUperformanceEntry *buffer, size_t elementCount, uint64_t groupMask, uint64_t subgroupMask);
    CUresult (CUDAAPI *DisableNoName)(size_t *eventCount);
    void (CUDAAPI *Begin)(const char *name, uint64_t group, uint64_t subgroup);
    void (CUDAAPI *End)(const char *name, uint64_t group, uint64_t subgroup);
    CUresult (CUDAAPI *Disable)(const char *name, size_t *eventCount);

    // Thread affinity API exposed for testing
    void (CUDAAPI *GetThreadAffinity)(void *thread, size_t *mask);
    void (CUDAAPI *SetThreadAffinity)(void *thread, const size_t *mask);
    uint32_t (CUDAAPI *GetProcessorCount)(void);
    uint32_t (CUDAAPI *GetCurrentProcessor)(void);
    size_t (CUDAAPI *ProcessorMaskSize)(void); 
    void (CUDAAPI *ProcessorMaskZero)(size_t *mask);
    void (CUDAAPI *ProcessorMaskSet)(size_t *mask, uint32_t proc);
    void (CUDAAPI *ProcessorMaskClear)(size_t *mask, uint32_t proc);
    int (CUDAAPI *ProcessorMaskEqual)(const size_t *mask1, const size_t *mask2);
    int (CUDAAPI *ThreadCreate)(void **thread, int (*fn)(void *), void *data);
    void (CUDAAPI *ThreadJoin)(void *thread, int *status);
} CUetblPerformance;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
