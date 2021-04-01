/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __unittest_h__
#define __unittest_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
// Backdoor driver API for driver unit tests which can't (or are
// hard to) test via the public API
//------------------------------------------------------------------

CU_DEFINE_UUID(CU_ETID_UNIT_TEST, 
    0xd14617a3, 0x7554, 0x4eef, 0xae, 0xb6, 0xdf, 0xb1, 0x47, 0x6a, 0xdd, 0x6c);

#define CU_ETID_UnitTest CU_ETID_UNIT_TEST

typedef enum {
    UVM_EMU_ADDR_MODE_X86 = 0,
    UVM_EMU_ADDR_MODE_ARM = 1,
    UVM_EMU_ADDR_MODE_POWER_STRICT = 2,
    UVM_EMU_ADDR_MODE_POWER = 3,
    UVM_EMU_ADDR_MODE_PERMISSIVE = 4,
    UVM_EMU_NUM_ADDR_MODES = 5
} CUuvmEmuAddressMode;

typedef enum {
    UNITTEST_CHANNEL_USE_COMPUTE = 0,
    UNITTEST_CHANNEL_USE_HTOD = 1,
    UNITTEST_CHANNEL_USE_DTOH = 2,
    UNITTEST_CHANNEL_USE_COUNT = 3,
} UnittestChannelUse;

typedef enum {
    UVM_TARGET_VA_MODE_ALL      = 0,
    UVM_TARGET_VA_MODE_TARGETED = 1,
} CUuvmTargetVaMode;

typedef enum {
    UVM_TLB_INVALIDATE_MEMBAR_NONE  = 0,
    UVM_TLB_INVALIDATE_MEMBAR_SYS   = 1,
    UVM_TLB_INVALIDATE_MEMBAR_LOCAL = 2,
} CUuvmTlbInvalidateMembarType;

typedef enum {
    UVM_INVALIDATE_TLB_LEVEL_ALL = 0,
    UVM_INVALIDATE_TLB_LEVEL_PTE = 1,
    UVM_INVALIDATE_TLB_LEVEL_PL0 = 2,
    UVM_INVALIDATE_TLB_LEVEL_PL1 = 3,
    UVM_INVALIDATE_TLB_LEVEL_PL2 = 4,
    UVM_INVALIDATE_TLB_LEVEL_PL3 = 5,
} CUuvmInvalidateTlbLevel;

typedef enum {
    UVM_TARGET_PDB_MODE_ALL = 0,
    UVM_TARGET_PDB_MODE_ONE = 1,
} CUuvmTargetPdbMode;

#define UVM_SET_FAULT_POLICY_FLAG_READ_ONLY      0x1
#define UVM_SET_FAULT_POLICY_FLAG_DISABLE_ATOMIC 0x2
#define UVM_SET_FAULT_POLICY_FLAG_CANCEL         0x4
#define UVM_SET_FAULT_POLICY_FLAG_SYSTEM_ONLY    0x8

typedef struct {
    CUresult (CUDAAPI *setAddressMode)(CUcontext ctx, CUuvmEmuAddressMode mode);
    CUresult (CUDAAPI *getAndResetWasLastIllegalAddrOor)(int *wasOor_out);
    CUresult (CUDAAPI *invalidateCache)(CUcontext ctx);
    CUresult (CUDAAPI *memAllocWithHostMapping)(CUcontext ctx, CUdeviceptr *dptr, void **hptr, size_t bytesize);
    CUresult (CUDAAPI *uvmUnmap)(CUcontext ctx, CUdeviceptr dptr, size_t bytesize, int skipTlbInvalidate);
    CUresult (CUDAAPI *uvmSetFaultPolicy)(CUcontext ctx, CUdeviceptr dptr, size_t bytesize, int flags);
    CUresult (CUDAAPI *uvmTlbInvalidate)(CUcontext ctx, CUdeviceptr dptr, CUuvmTargetVaMode vaMode,
                                         CUuvmTlbInvalidateMembarType membarType, CUuvmInvalidateTlbLevel tlbLevel,
                                         CUuvmTargetPdbMode targetPdb, int invalidateGpc);
} CUetblUnitTestUvmPascalEmu;

typedef struct CUetblUnitTest_st {
    size_t struct_size;
    CUresult (CUDAAPI *CtxSetBlockingSyncTimeoutMsec)(unsigned int msec);

    // Enable fault injection
    // - fail on the specified call (1 means next call,
    //   2 means call after that, 0 means don't fail -- 
    //   just update the fault-able call count)
    CUresult (CUDAAPI *FaultInjectionEnable)(unsigned int failCallIndex);    
    // Return the number of fault-able calls which
    // have been called sinced fault injection was enabled
    CUresult (CUDAAPI *FaultInjectionCallCount)(unsigned int *outCallCount);
    // Disable fault injection
    CUresult (CUDAAPI *FaultInjectionDisable)(void);

    // Internal control global control to enable/disable assert 
    // killing the context. Assert will invoke rc recovery 
    // by default.
    CUresult (CUDAAPI *AssertRcRecoveryEnable)(CUcontext ctx, unsigned int enable);
    // Internal control global control enable/disable initialization
    // of the printf flusher thread. Auto flushing is on by default.
    CUresult (CUDAAPI *PrintfAutoFlushEnable)(CUcontext ctx, unsigned int enable);

    // cuiHash functions
    // The typedefs aren't available here...
    struct cuiStringHash_st *(*cuiStringHashCreate)(unsigned int nbins);
    void                     (*cuiStringHashDestroy)(struct cuiStringHash_st *hash);
    CUresult                 (*cuiStringHashAddEntry)(struct cuiStringHash_st *hash, const char *key, void *value);
    void                    *(*cuiStringHashGetEntry)(struct cuiStringHash_st *hash, const char *key);
    void                    *(*cuiStringHashDeleteEntry)(struct cuiStringHash_st *hash, const char *key);
    struct cuiNumHash_st    *(*cuiNumHashCreate)(unsigned int nbins);
    void                     (*cuiNumHashDestroy)(struct cuiNumHash_st *hash);
    CUresult                 (*cuiNumHashAddEntry)(struct cuiNumHash_st *hash, uint64_t key, void *value);
    void                    *(*cuiNumHashGetEntry)(struct cuiNumHash_st *hash, uint64_t key);
    void                    *(*cuiNumHashDeleteEntry)(struct cuiNumHash_st *hash, uint64_t key);
    void                     (*cuiStringHashApplyFunctor)(struct cuiStringHash_st *hash, void (*functor)(const char *key, void *value, void *data), void *data);
    void                     (*cuiNumHashApplyFunctor)(struct cuiNumHash_st *hash, void (*functor)(uint64_t key, void *value, void *data), void *data);
    void                     (*cuiStringHashClear)(struct cuiStringHash_st *hash);
    void                     (*cuiNumHashClear)(struct cuiNumHash_st *hash);

    // Unit test to inject ECC exceptions intended to test the fast ECC path on RM
    CUresult (CUDAAPI *InjectEccException)(CUcontext ctx, void *injectEccParams);

    // Returns CUDA_ERROR_NOT_FOUND if there is no matching entry.
    CUresult (CUDAAPI *DeletePtxJitCacheEntry)(CUcontext ctx, void *ptx, unsigned numOptions, CUjit_option *options, void **optionVals, int relocatable);

    // Indicates the value of cuosBuildIsPublicRelease() for this driver.
    int      (CUDAAPI *BuildIsPublicRelease)(void);

    // Set *res to 1 if cnp is initialized, 0 otherwise.
    CUresult (CUDAAPI *CnpIsInitialized)(int *res);

    uint64_t (CUDAAPI *PerThreadStreamGetUsageCount)(void);

    void (CUDAAPI *PerThreadStreamEnableCnpUsageCounter)(void);

    const CUetblUnitTestUvmPascalEmu *uvmPascalEmu;

    // Interrupt unit test helpers
    int (CUDAAPI *InterruptNonStallingSupported)(void);
    void (CUDAAPI *InterruptForceStalling)(int force);
    CUresult (CUDAAPI *InterruptPush)(UnittestChannelUse unittestChannelUse);
    uint64_t (CUDAAPI *InterruptGetCount)(UnittestChannelUse unittestChannelUse);

    // Force user constant bank to be in managed memory
    CUresult (CUDAAPI *ForceManagedConstMemory)(int force);

    // Make memcpy SYSMEMBAR go through Host
    CUresult (CUDAAPI *ForceMemcpySysMembarThroughHost)(int force);

    // Query CILP support
    int (CUDAAPI *CilpEnabled)(void);

    // Manipulating the global error. Once its set, you can't reset the context/device to try again,
    // so tests would need to do a lot of fork()s to test just basic error setting.
    void (CUDAAPI *ToggleProcessErrorMode)(int enabled);
    void (CUDAAPI *SetProcessError)(CUresult error);
    void (CUDAAPI *ClearProcessError)(void);
    CUresult (CUDAAPI *GetProcessError)(void);

    // Returns total number of compute channels associated with the CTX.
    CUresult (CUDAAPI *GetComputeEngineChannelCount)(CUcontext ctx, unsigned *channelCount);

    // TODO: Remove this define once this header has propagated to chips_a
#define CU_ETBL_UNIT_TEST_GET_CHANNEL_HANDLES_HAS_TSG 1

    // Returns all the compute channel RM handles associated with the CTX.
    CUresult (CUDAAPI *GetComputeEngineChannelHandles)(CUcontext ctx,
                                                       unsigned *rmClient,
                                                       unsigned *rmChannelGroup,
                                                       unsigned *rmChannels,
                                                       unsigned channelCount);

    // Returns CUDA's RM device, RM vaSpace (UVM managed) and client handles.
    CUresult (CUDAAPI *GetCudaDeviceAndVaSpaceHandles)(CUdevice device, unsigned *rmClient, unsigned *rmDevice, unsigned *rmVaSpace);

    // Allocates an empty handle from the CUDA's handle pool.
    CUresult (CUDAAPI *AllocHandleFromCudaPool)(unsigned *rmHandle);

    // Frees a handle to the CUDA's handle pool.
    CUresult (CUDAAPI *FreeHandleToCudaPool)(unsigned rmHandle);

    // Get vidmem usage for internal allocations associated with the CTX
    CUresult (CUDAAPI *GetVidmemUsageForCtxInternalAllocs)(CUcontext ctx, uint64_t *vidmemUsage);

    // Assign a callback function for testing asynchronous printf
    CUresult (CUDAAPI *AssignCallbackForAsyncPrintfTest)(void (*callbackFunction)(void));

    // Query/change compute modes
    CUresult (CUDAAPI *GetComputeMode)(int *mode, int device);
    CUresult (CUDAAPI *SetComputeMode)(int mode, int device);

    // Query UVA virtual heap statistics
    CUresult (CUDAAPI *UvaVirtMemGetInfo)(uint64_t *free, uint64_t *total);

    // Find a virtual alias of addr in the range [aliasMin, aliasMax] and return in alias_out
    // if found.  Otherwise, alias_out is set to NULL.
    CUresult (CUDAAPI *FindVirtualAlias)(CUdeviceptr addr, CUdeviceptr aliasMin, CUdeviceptr aliasMax, CUdeviceptr *alias_out);

    // Check if it is MPS client
    int (CUDAAPI *IsMpsClient)(void);

    // Enable stream callbacks if otherwise disabled
    void (CUDAAPI *AllowStreamCallbacks)(void);

    /// Get attribute of a memory range - unlike the CUDA API function, this allows querying
    /// ranges that aren't currently allocated - such as managed memory that was previously freed
    CUresult (CUDAAPI *MemRangeGetAttribute)(
        int *data,
        size_t numDataElements,
        CUmem_range_attribute attribute,
        CUdeviceptr vaddr,
        size_t size);

    // The printf function used by the CUDA driver
    //
    // Notably it will be different if the application uses a different C lib
    // than the CUDA driver, which is common on Windows.
    int (CUDAAPI *driverPrintf)(const char *format, ...);

    // Forces driver to mimic the behaviour when ptxjitcompiler is not present
    void (CUDAAPI *DeletePtxJitCompiler)(void);
    // Undo DeletePtxJitCompiler
    void (CUDAAPI *UndeletePtxJitCompiler)(void);

    // Sets the three parameters for eL1/Shared Memory carveout (Volta+)
    CUresult (CUDAAPI *SetMinTargetMaxShMem)(void (*setShMemConfiguration)(uint32_t occupancy, uint32_t *min, uint32_t *target, uint32_t *max));

    // Set the number of TPCs used by any context created in this
    // process
    CUresult (CUDAAPI *SetNumTpcLimit)(
        int numTpc);

    // Get the supported subcontext count
    CUresult (CUDAAPI *GetMaxSubcontextCount)(
        int *maxSubcontext);

    // Get the allocated VEID for this context
    CUresult (CUDAAPI *GetVEIDForCtx)(CUcontext ctx, unsigned int *subctxId);

    // Request a specific VEID for next subcontext allocation
    CUresult (CUDAAPI *SetVEIDForNextCtxAlloc)(unsigned int subctxId);

    // Set globals flag to disable CILP.
    void (CUDAAPI *DisableCilp)(void);

    // Set globals flag to retain the primary context of the device and make future
    // non-primary contexts as subcontexts of the primary context's TSG.
    void (CUDAAPI *NonPrimaryCtxAreSubcontexts)(void);

    // Disable the device watchdog.
    CUresult (CUDAAPI *DisableWatchdog)(int device);

    // Get VA to shared preemption barrier, used for unit testing the shared physical
    // barrier address mapping across all subctx clients.
    CUresult (CUDAAPI *GetPreemptionBarrierDevPtr)(CUcontext ctx, unsigned int **barrierDevPtr);

    // This function disables the HMM support in CUDA for the calling process.
    // It must be called prior to calling cuInit to have an effect.
    void (CUDAAPI *DisableHMM)(void);

    // Allow/disallow usage of the vidmem barrier when synchronizing threads across a multi-gpu
    // cooperative launch. By default, vidmem barriers are allowed.
    void (CUDAAPI *ToggleMultiGpuCoopLaunchVidmemBarrier)(int allowVidmemBarrier);

    // Unconditionally enable stream memory operations (MemOps) on supported 
    // platforms, even when disabled by RM
    CUresult (CUDAAPI *EnableStreamMemOps)(void);

    // This function causes the CUDA driver to pass the
    // UVM_INIT_FLAGS_DISABLE_TEARDOWN_ON_PROCESS_EXIT flag to UvmInitialize.
    // See uvm.h for a description of what that flag does. This function must be
    // called prior to calling cuInit to have an effect.
    void (CUDAAPI *UvmDisableTeardownOnProcessExit)(void);
    // Return if compression is supported by the driver on this device
    int (CUDAAPI *CompressionSupported)(CUcontext ctx);

    // Get the compression mode for an allocation
    CUresult (CUDAAPI *GetCompressionMode)(CUcontext ctx, CUdeviceptr vaddr, uint32_t *mode);

    // Create a new allocation of the given size that has the specified compression mode set
    CUresult (CUDAAPI *MemAllocCompressed)(CUdeviceptr* dptr, uint64_t size, uint32_t mode);

    // Indicates the value of cuosBuildIsRelease() for this driver.
    int      (CUDAAPI *BuildIsRelease)(void);

    // Get the drop-to-idle timeout.
    uint32_t (CUDAAPI *GetD2ITimeout)(int device);

    // Set the drop-to-idle timeout.
    void (CUDAAPI *SetD2ITimeout)(int device, uint32_t timeout);

    // Check if the GPU is boosted or not.
    uint32_t (CUDAAPI *IsGPUBoosted)(CUcontext ctx);

    // Force the VA to be 40-Bit
    void (CUDAAPI *Force40BitVA)(void);

    // Returns the rmVersion and the driver build version
    CUresult (CUDAAPI *GetDriverVersions)(
        uint32_t *rmVersion, uint32_t *driverBuildVersion);

    // Returns if device will be sync on free of allocations that are mapped there
    CUresult (CUDAAPI *WillSyncCtxOnFree)(int device, int *willSync);

    // Get lmem size associated with the CTX
    CUresult (CUDAAPI *GetCtxLmemSize)(CUcontext ctx, uint64_t *lmemSize);

    // Returns if DX11-CUDA interop is supported by the driver
    uint32_t (CUDAAPI *IsDX11InteropSupported)(void);
} CUetblUnitTest;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard

