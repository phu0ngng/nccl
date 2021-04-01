/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_context_h__
#define __cuda_etbl_tools_context_h__

#include "cuda.h"

// Include experimental graph API. Once this moves from toolsAPI to a
// public header, this can be removed and should be replaced with the
// official header - in case this is not yet covered by the existing
// includes above anyway.
#include "cuda_etbl/cuda_graphs.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


typedef enum CUtools_api_enum {
    CU_TOOLS_API_INVALID         = 0,
    CU_TOOLS_API_CUDA            = 1,
    CU_TOOLS_API_OPENCL_NV       = 2,
    CU_TOOLS_API_OPENCL_APPLE    = 3,
    // --- always add new constants to the end here ---
    CU_TOOLS_API_SIZE,
    CU_TOOLS_API_FORCE_INT       = 0x7fffffff
} CUtools_api;

typedef enum CUtools_power_target_enum {
    CU_TOOLS_POWER_TARGET_INVALID         = 0,
    CU_TOOLS_POWER_TARGET_GR_ENGINE       = 1,
    // --- always add new constants to the end here ---
    CU_TOOLS_POWER_TARGET_SIZE,
    CU_TOOLS_POWER_TARGET_FORCE_INT       = 0x7fffffff
} CUtools_power_target;

typedef enum CUtools_power_level_enum {
    CU_TOOLS_POWER_LEVEL_INVALID         = 0,
    CU_TOOLS_POWER_LEVEL_FULL            = 1,
    CU_TOOLS_POWER_LEVEL_AUTOMATIC       = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_POWER_LEVEL_SIZE,
    CU_TOOLS_POWER_LEVEL_FORCE_INT       = 0x7fffffff
} CUtools_power_level;

typedef enum CUtoolsPreemptionMode_enum {
    CU_TOOLS_PREEMPTION_MODE_WFI       = 0,
    CU_TOOLS_PREEMPTION_MODE_CTA       = 1,
    CU_TOOLS_PREEMPTION_MODE_CILP      = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_PREEMPTION_MODE_SIZE,
    CU_TOOLS_PREEMPTION_MODE_FORCE_INT = 0x7fffffff
} CUtoolsPreemptionMode;

typedef struct CUtoolsLMemLayout_st {
    uint32_t struct_size;

    /// A range of LMemHi within the debugger region will be free for external tools.
    /// On GK110 this range should be [0x00fffe00, 0x00fffea0)
    uint32_t freeLMemHiAddr;
    uint32_t freeLMemHiSize;

    /// LMem address where syscall functions save the user stack-pointer (R1)
    /// before switching stacks.
    uint32_t syscallR1Addr;
    /// LMem address where blockIdx is saved for CNP continuations.
    uint32_t cnpBlockIdxAddr;

    /// LMem address reserved for cudaGetLastError() on device.
    uint32_t cnpLastErrorAddr;

    /// Register spill addresses for the membar WAR.
    uint32_t membarWarSpace_R0;
    uint32_t membarWarSpace_R2;
    uint32_t membarWarSpace_R3;
    uint32_t membarWarSpace_PR;
} CUtoolsLMemLayout;

typedef struct CUtoolsEnumStreamCallbackData_st {
    uint32_t struct_size;
    uint32_t reserved0;
    CUtoolsStreamHandle stream;
    void* reserved1;
} CUtoolsEnumStreamCallbackData;

typedef struct CUtoolsEnumModuleCallbackData_st {
    uint32_t struct_size;
    uint32_t moduleOwner;
    CUmodule mod;
    const void *elf;
    size_t elfSize;
    void* reserved1;
} CUtoolsEnumModuleCallbackData;

/// \brief Callback signature for EnumerateStreams()
typedef void (CUDAAPI *CUtoolsEnumerateStreamsCallback)(
    void* pUserData,
    CUtoolsEnumStreamCallbackData *streamData);

/// \brief Callback signature for EnumerateModules()
typedef void (CUDAAPI *CUtoolsEnumerateModulesCallback)(
    void* pUserData,
    CUtoolsEnumModuleCallbackData *moduleData);

typedef struct CUtoolsEnumStreamData_st {
    uint32_t struct_size;
    uint32_t reserved0;
    CUcontext ctx;
    CUtoolsEnumerateStreamsCallback pfnCallback;
    void *pUserData;
    void* reserved1;
} CUtoolsEnumStreamData;

typedef struct CUtoolsEnumModuleData_st {
    uint32_t struct_size;
    uint32_t reserved0;
    CUcontext ctx;
    CUtoolsEnumerateModulesCallback pfnCallback;
    void *pUserData;
    void* reserved1;
} CUtoolsEnumModuleData;

typedef struct CUtoolsMmuFaultInfoTpc_st {
    uint8_t hasFatalFaults;

    // PC of the instruction that triggered the fault
    uint64_t faultPc;

    // VA that triggered the fault
    uint64_t faultVa;

    // There are some other pending faults
    uint8_t multipleFaults;

} CUtoolsMmuFaultInfoTpc;

typedef struct CUtoolsMmuFaultInfo_st {
    uint32_t numTpc;

    CUtoolsMmuFaultInfoTpc *tpcInfo;
} CUtoolsMmuFaultInfo;

typedef struct CUtoolsContextCreateFlags_st {
    uint32_t struct_size;

    uint8_t forceEnableCILP;
    uint8_t forceDisableISR;

    uint16_t reserved0;
} CUtoolsContextCreateFlags;

/// \brief ApiContext contains functions for controlling a CUcontext
CU_DEFINE_UUID(CU_ETID_ToolsContext,
    0x608c3121, 0x1497, 0x4832, 0x8c, 0xa6, 0x41, 0xff, 0x73, 0x24, 0xc8, 0xf2);

typedef struct CUetblToolsContext_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief Wait for a stream to be idle.
    /// \param ctx The context for this stream.
    /// \param stream The stream to wait on.
    /// \return CUDA_ERROR_UNKNOWN if the wait failed (due to HW notifier or other reason).
    ///     Otherwise, returns CUDA_SUCCESS.
    /// IMPLEMENTATION : wrapper of cuiStreamSynchronize()
    CUresult (CUDAAPI *StreamSynchronize)(
        CUcontext ctx,
        CUtoolsStreamHandle stream);

    /// \brief returns the instantaneous idle-status of the stream.
    /// \param ctx The context for this stream.
    /// \param stream The stream to query.
    /// \return CUDA_SUCCESS if idle.
    /// Tests all semaphores associated with the context (once), and return
    /// the instantaneous idle-state.
    /// Useful when a piece of code needs to block on multiple events,
    /// and one of the events is GPU-idleness.
    /// IMPLEMENTATION : wrapper for cuiStreamQuery()
    CUresult (CUDAAPI *StreamQuery)(
        CUcontext ctx,
        CUtoolsStreamHandle stream);

    /// Get CUdevice index for the device bound to a context
    CUresult (CUDAAPI *CtxGetDevice)(
        CUcontext ctx,
        uint32_t* deviceIndex);

    /// \brief Gets the unique ID for the context.
    /// Context IDs are linearly increasing.  Unique IDs should be used in
    /// place of the CUcontext address as the address can be re-used.
    CUresult (CUDAAPI *CtxGetId)(
        CUcontext ctx,
        uint64_t* pContextId);

    /// DEPRECATED in CUDA 6.0. Use StreamGetIdEx instead.
    /// \brief Gets the unique ID for the stream.
    /// Stream IDs are unique numbers within a context for the lifetime of
    /// of the context.  This is convenient for post-mortem tools to
    /// distinguish streams in the case where stream handles get re-used.
    CUresult (CUDAAPI *StreamGetId)(
        CUcontext ctx,
        CUtoolsStreamHandle stream,
        uint64_t* pStreamId);

    /// \brief Force all work of this context to be synchronizes.
    /// \param ctx The context to synchronize.
    /// \return CUDA_ERROR_UNKNOWN if the wait failed.
    ///     Otherwise, returns CUDA_SUCCESS.
    /// IMPLEMENTATION : Basically a wrapper of cuiCtxSynchronize();
    CUresult (CUDAAPI *CtxSynchronize)(
        CUcontext ctx);

    /// Set default flags for any future CUcontext creation.
    /// Flags will be computed according to the expression:
    ///     flags = (UserFlags & ~forceDisableCtxCreateFlags) | forceEnableCtxCreateFlags;
    /// @param forceDisableCtxCreateFlags Specifies which flag bits to clear.
    /// @param forceEnableCtxCreateFlags Specifies which flag bits to set.
    /// Restrictions:
    ///     CU_CTX_SCHED_BLOCKING_SYNC : no issue
    ///     CU_CTX_MAP_HOST            : has the potential to break an application that
    ///         uses a lot of vidmem by taking some of the address space away from vidmem
    ///     CU_CTX_LMEM_RESIZE_TO_MAX  : has the potential to break an application if the user
    ///         allocates memory after running the last instance of a kernel that uses a lot of lmem.
    void (CUDAAPI *SetDefaultCtxCreateFlags)(
        uint32_t forceDisableCtxCreateFlags,
        uint32_t forceEnableCtxCreateFlags);

    void (CUDAAPI *CtxGetApi)(
        CUcontext ctx,
        CUtools_api* pApi);

    /// Analogous to cuCtxSetLimit().
    CUresult (CUDAAPI *CtxSetLimit)(
        CUcontext ctx,
        CUlimit limit,
        size_t value);

    /// Analogous to cuCtxGetLimit().
    CUresult (CUDAAPI *CtxGetLimit)(
        CUcontext ctx,
        CUlimit limit,
        size_t *pValue);

    /// Get the null stream for a given context
    CUresult (CUDAAPI *CtxGetNullStream)(
        CUcontext ctx,
        CUtoolsStreamHandle *stream);

    /// Check if a given stream is the null stream for its context
    CUresult (CUDAAPI *StreamIsNullStream)(
        CUcontext ctx,
        CUtoolsStreamHandle stream,
        uint8_t *isNullStream);

    /// Get which version of the CUDA API this context supports
    CUresult (CUDAAPI *CtxGetApiVersion)(
        CUcontext ctx,
        uint32_t *pApiVersion);

    CUresult(CUDAAPI *CtxGetLocalWindowBase)(
        CUcontext ctx,
        uint64_t *pvalue);

    CUresult(CUDAAPI *CtxGetSharedWindowBase)(
        CUcontext ctx,
        uint64_t *pvalue);

    /// Get public stream handle from an internal stream handle
    /// \param stream must not be NULL.
    CUresult (CUDAAPI *StreamGetPublicHandle)(
        CUtoolsStreamHandle stream,
        CUstream *hStream);

    /// DEPRECATED in CUDA 8.0. Use StreamGetToolsHandleEx instead.
    /// Get internal stream handle from a public stream handle
    CUresult (CUDAAPI *StreamGetToolsHandle)(
        CUcontext ctx,
        CUstream hStream,
        CUtoolsStreamHandle *stream);

    /// DEPRECATED. Use StreamCreateEx instead.
    /// Create a stream (useful for allowing a tool to launch
    /// overlapping work without hijacking the app's streams)
    CUresult (CUDAAPI *StreamCreate)(
        CUcontext ctx,
        CUtoolsStreamHandle *stream);

    /// Destroy a stream.  Tools should only destroy streams
    /// they created with StreamCreate.
    CUresult (CUDAAPI *StreamDestroy)(
        CUcontext ctx,
        CUtoolsStreamHandle stream);

    /// Get context for a stream
    CUresult (CUDAAPI *StreamGetCtx)(
        CUtoolsStreamHandle stream,
        CUcontext* ctx);

    /// Check if context supports CNP
    CUresult (CUDAAPI *CtxSupportsCnp)(
        CUcontext ctx,
        CUtools_cnp_support *flag);

    /// Change the power state for the context
    //.Current options allow to disable the power gating and clock gating both
    CUresult (CUDAAPI *CtxChangePowerState)(
        CUcontext ctx,
        CUtools_power_target target,
        CUtools_power_level level);

    /// Get LMem layout information for ranges not defined by the ABI nor DCI.
    CUresult (CUDAAPI *GetLMemLayout)(
        CUcontext ctx,
        CUtoolsLMemLayout *layout);

    /// For a given context (which is only used to query the appropriate
    /// device), retrieve the number of a const bank used by tools to
    /// store data to be accessed quickly from device code patches or
    /// callbacks.  Also retrieve the offset and size within the const
    /// bank.  If offset is zero and size is 64K, that indicates the
    /// entire const bank is reserved for tools usage.  Note that this
    /// describes only a contract -- the driver will not transfer any
    /// data or enable/redirect this const bank during launches.  It is
    /// the tool's responsibility to direct each launch to use a memobj
    /// containing the intended data.
    CUresult (CUDAAPI *CtxGetToolsConstBankInfo)(
        CUcontext ctx,
        uint32_t *pBankNumber,
        uint32_t *pOffset,
        uint32_t *pSize);

    /// For a given context (which is only used to query the appropriate
    /// device), retrieve the number of a const bank used by tools to
    /// store data to be accessed quickly from device code patches or
    /// callbacks.  Also retrieve the offset and size within the const
    /// bank.  If offset is zero and size is 64K, that indicates the
    /// entire const bank is reserved for tools usage.  This describes
    /// a contract for tools ownership of that region.  The driver will
    /// automatically upload data configured with CUetblToolsModule's
    /// FunctionSetDebuggerParams.  Tools may also upload the data using
    /// the other APIs below and MemCpy.
    CUresult (CUDAAPI *CtxGetToolsPerLaunchConstBankInfo)(
        CUcontext ctx,
        uint32_t *pBankNumber,
        uint32_t *pOffset,
        uint32_t *pSize);

    /// For a given context (which is only used to query the appropriate
    /// device), retrieve the number of const banks.
    CUresult (CUDAAPI *CtxGetConstBankCount)(
        CUcontext ctx,
        uint32_t *pBankCount);

    /// For a given stream, retrieve a handle for the memobj (or NULL if
    /// there isn't one) for a given const bank.  Valid bankNumber values
    /// are 0 to bankCount - 1, where bankCount is the value returned by
    /// CtxGetToolsConstBankCount.
    /// For bank 0, the memobj is potentially shared between multiple streams
    /// Use StreamGetConstBankMemobjOffset to get the offset for a specific stream
    CUresult (CUDAAPI *StreamGetConstBankMemObj)(
        CUtoolsStreamHandle stream,
        uint32_t bankNumber,
        CUtoolsMemObjHandle *phMemObj);

    /// For a given stream, assign a memobj (or NULL if there isn't one)
    /// for a given const bank.  This allows tools to allocate the tools
    /// memobj for streams, since the driver does not guarantee that it
    /// will be allocated automatically during stream creation.  Valid
    /// bankNumber values are 0 to bankCount - 1, where bankCount is the
    /// value returned by CtxGetToolsConstBankCount.  This is safe to
    /// call in the STREAM_CREATED and STREAM_DESTROY_STARTING callbacks.
    CUresult (CUDAAPI *StreamSetConstBankMemObj)(
        CUtoolsStreamHandle stream,
        uint32_t bankNumber,
        CUtoolsMemObjHandle hMemObj);

    /// For a given context, retrieve memobj/offset to the scheduler QMD.
    CUresult (CUDAAPI *CtxGetSchedulerQmd)(
        CUcontext ctx,
        CUtoolsMemObjHandle *phMemObjQmd,
        uint64_t *pOffsetQmd);

    /// Get the barrier stream for a given context
    CUresult (CUDAAPI *CtxGetBarrierStream)(
        CUcontext ctx,
        CUtoolsStreamHandle *stream);

    /// Check if a given stream is the barrier stream for its context
    CUresult (CUDAAPI *StreamIsBarrierStream)(
        CUcontext ctx,
        CUtoolsStreamHandle stream,
        uint8_t *isBarrierStream);

    /// Check if a given stream synchronizes with the null stream
    CUresult (CUDAAPI *StreamIsNonBlocking)(
        CUtoolsStreamHandle stream,
        uint8_t *isNonBlocking);

    /// \brief adds a callback to be called after all currently enqueued items in the stream have completed.
    /// \param ctx The context for this stream.
    /// \param stream The stream to add callback to.
    /// \param callback The function to call once preceding stream operations are complete.
    /// \param userData User specified data to be passed to the callback function.
    /// \param flags Reserved for future use, must be 0.
    /// \return CUDA_SUCCESS if added successfully.
    /// IMPLEMENTATION : wrapper for cuiStreamAddCallback()
    CUresult (CUDAAPI *StreamAddCallback)(
        CUcontext ctx,
        CUtoolsStreamHandle stream,
        CUstreamCallback callback,
        void* userData,
        unsigned int flags);

    //The expected usage of CtxEnumerateStreams and CtxEnumerateModules is to only
    //enumerate streams/modules from inside a callback under EnumerateContexts 
    //which holds the ctxLock 

    //CtxEnumerateStreams function should be called from within a ctxLock to ensure
    //there are no streams added or deleted while enumeration is in progress.
    /// \brief Enumerates streams for given context.
    /// \param ctx The context for which streams are to be enumerated
    /// \param pfnCallback Callback function that will be called for each stream.
    CUresult (CUDAAPI *CtxEnumerateStreams)(
        CUtoolsEnumStreamData *enumStreamData);

    //CtxEnumerateModules function should be called from within a ctxLock to ensure
    //there are no modules are added or deleted while enumeration is in progress.
    /// \brief Enumerates modules for given context.
    /// \param ctx The context for which modules are to be enumerated
    /// \param pfnCallback Callback function that will be called for each module.
    CUresult (CUDAAPI *CtxEnumerateModules)(
        CUtoolsEnumModuleData *enumModuleData);

    /// \brief Return the bounds at which stream priorities are clamped for this ctx.
    /// \param ctx The context of interest
    /// \param leastPriority (out, optional) The lower bound (which is the HIGHEST integer value)
    /// \param highestPriority (out, optional) The upper bound (which is the LOWEST integer value)
    CUresult (CUDAAPI *CtxGetStreamPriorityRange)(
        CUcontext ctx,
        int32_t *leastPriority,
        int32_t *highestPriority);

    /// \brief Return the priority a user specified for when creating a stream
    /// \param stream The stream of interest
    /// \param requestedPriority (out, optional)  Stream priority value requested by user
    /// \param clampedPriority (out, optional)  Stream priority value permitted by context
    /// \param internalPriority (out, optional)  Stream priority value passed to hardware
    CUresult (CUDAAPI *StreamGetPriority)(
        CUtoolsStreamHandle stream,
        int32_t *requestedPriority,
        int32_t *clampedPriority,
        int32_t *internalPriority);

    /// \brief Create a stream with CU_STREAM_* flags and priority
    /// \param ctx The context on which the stream will be created
    /// \param flags The CU_STREAM_* flags
    /// \param priority The requested stream priority
    /// \param stream Return the created stream handle
    CUresult (CUDAAPI *StreamCreateEx)(
        CUcontext ctx,
        uint32_t flags,
        int32_t priority,
        CUtoolsStreamHandle *stream);

    /// \brief Gets the unique ID for the stream.
    /// Stream IDs are unique numbers per process for the lifetime of
    /// the process.
    CUresult (CUDAAPI *StreamGetIdEx)(
        CUcontext ctx,
        CUtoolsStreamHandle stream,
        uint64_t* pStreamId);

    /// \brief Gets the parent context if it's a dummy context.
    /// Returns NULL if it's a normal context.
    CUresult (CUDAAPI *CtxGetParentCtx)(
        CUcontext ctx,
        CUcontext *parentCtx);

    /// \brief Enable or disable SMPC context switching via the RM.
    /// This function modifies channels' ctxsw header, and must be done
    /// while the channels are idle.  To maximize the likelihood of
    /// success, call this after a CtxSynchronize.
    CUresult (CUDAAPI *SetSMPCCtxswMode)(
        CUcontext ctx,
        uint32_t enabled);

    /// \brief Gets the unique ID for the event.
    /// Event IDs are unique numbers per process for the lifetime of
    /// the process.
    CUresult (CUDAAPI *EventGetId)(
        CUevent event,
        uint64_t* pEventId);

    /// Get context for a event
    CUresult (CUDAAPI *EventGetCtx)(
        CUevent event,
        CUcontext* ctx);

    /// Enable or disable auto boost
    CUresult (CUDAAPI *SetAutoBoostState)(
        CUcontext ctx,
        uint32_t enable);

    /// Query auto boost state
    /// Returns CUDA_SUCCESS if the auto boost state can be queried
    /// \param enabled will be set to 0 in case auto boost is disabled
    /// and set to 1 in case auto boost is enabled.
    CUresult (CUDAAPI *GetAutoBoostState)(
        CUcontext ctx,
        uint32_t *enabled);

    /// Query list of client pid that have requested the current auto boost state.
    /// Returns CUDA_SUCCESS if the user has permission to query the process ids.
    /// \param count Number of process ids requested. Returns Min of (Number of
    /// processes currently requested the auto boost state,
    /// NV2080_CTRL_PERF_MAX_LOCKED_CLOCKS_CLIENT_PID). Count will be zero in
    /// case no process has toggled the auto boost state.
    /// \param pid Array of process ids.
    CUresult (CUDAAPI *GetAutoBoostClientPid)(
        CUcontext ctx,
        uint32_t *count,
        uint32_t *pid);

    /// Invoke \param func under the context lock.
    /// A large number of CUDA Tools API ETBL functions may only be called under
    /// the context lock (like from callbacks).  This function allows arbitrary code
    /// to call ETBl functions.
    CUresult(CUDAAPI *InvokeUnderContextLock)(
        CUcontext ctx,
        void (CUDAAPI *func)(void* userdata),
        void* userdata);

    /// By default CUDA clears the ESR information on kernel launch failure check.
    /// This function allows to modify that behavior.
    ///
    /// clear = 0: do not clear ESR information
    /// clear = 1: clear ESR information
    CUresult (CUDAAPI *CtxSetClearLaunchErrorOnCheck)(
        CUcontext ctx,
        uint8_t clear);

    /// Obtain the MMU fault information for Pascal and later GPUs. MMU debug mode
    /// needs to be enabled first for the information to be generated.
    CUresult (CUDAAPI *CtxGetKernelLaunchMmuFaultInfo)(
        CUcontext ctx,
        CUtoolsMmuFaultInfo *faultInfo);

    /// Enable/Disable MMU debug mode
    CUresult (CUDAAPI *CtxSetMmuDebugMode)(
        CUcontext ctx,
        uint8_t enable);

    /// Get internal stream handle from a public stream handle
    /// \param bPerThreadStream (in) indicates if stream is per-thread-stream
    CUresult (CUDAAPI *StreamGetToolsHandleEx)(
        CUcontext ctx,
        CUstream hStream,
        CUtoolsStreamHandle *stream,
        uint8_t bPerThreadStream);

    /// Get preemption mode for this context
    CUresult (CUDAAPI *CtxGetPreemptionMode)(
        CUcontext ctx,
        CUtoolsPreemptionMode *preemptionMode);

    /// \brief Queries capture graph of the stream.
    /// Return NULL for the graph, if the stream is not in an active capture.
    /// Otherwise return the handle to the graph used for capture.
    CUresult (CUDAAPI *StreamGetCapturingGraph)(
        CUtoolsStreamHandle stream,
        CUgraph* pGraph);

    /// Get the pipeline depth of const bank sets which is used to overlap
    /// const bank programming with the previous launch
    CUresult (CUDAAPI *StreamGetConstBankPipelineDepth)(
        CUtoolsStreamHandle stream,
        uint32_t *pBankPipelineDepth);

    /// Get the index, within the pipeline, of the current const bank set
    CUresult (CUDAAPI *StreamGeConstBankPipelineIndex)(
        CUtoolsStreamHandle stream,
        uint32_t *pCurBankPipelineIndex);

    /// Get the memobj of a const bank
    /// For bank 0, the memobj is potentially shared between multiple streams
    /// Use StreamGetConstBankMemobjOffset to get the offset for a specific stream
    CUresult (CUDAAPI *StreamGetConstBankPipelineMemObj)(
        CUtoolsStreamHandle stream,
        uint32_t pipelineIndex,
        uint32_t bankNumber,
        CUtoolsMemObjHandle *phMemObj);

    /// Set the memobj of a const bank
    CUresult (CUDAAPI *StreamSetConstBankPipelineMemObj)(
        CUtoolsStreamHandle stream,
        uint32_t pipelineIndex,
        uint32_t bankNumber,
        CUtoolsMemObjHandle hMemObj);

    /// Get the devvaddr and size of a const bank
    CUresult (CUDAAPI *StreamGetConstBankPipelineAddrAndSize)(
        CUtoolsStreamHandle stream,
        uint32_t pipelineIndex,
        uint32_t bankNumber,
        uint64_t *devvaddr,
        uint64_t *size);

    /// Get the offset in bytes in the memobj of a const bank
    CUresult (CUDAAPI *StreamGetConstBankMemobjOffset)(
        CUtoolsStreamHandle stream,
        uint32_t bankNumber,
        uint64_t *offset);

    /// Set the context creation flags pre-init
    CUresult (CUDAAPI *CtxSetToolsCreationFlags)(
        CUcontext ctx,
        CUtoolsContextCreateFlags *pFlags);
} CUetblToolsContext;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
