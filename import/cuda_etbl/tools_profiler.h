/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_profiler_h__
#define __cuda_etbl_tools_profiler_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_stdint.h"
#include "cuprofiler.h"

#include "cuda_etbl/tools_common_types.h"

#include "stdio.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// \brief Profiler provides functions for 3rd party tools to access the hardware counters.
// {9B3FB67C-A49C-428e-A484-BEB66E6AED7F}
CU_DEFINE_UUID(CU_ETID_ToolsProfiler,
    0x9b3fb67c, 0xa49c, 0x428e, 0xa4, 0x84, 0xbe, 0xb6, 0x6e, 0x6a, 0xed, 0x7f);

typedef enum CUetblToolsProfRegOpType_enum {
    CU_TOOLS_PROF_REG_OP_TYPE_CONTEXT   = 0x1,
    CU_TOOLS_PROF_REG_OP_TYPE_GLOBAL,
    CU_TOOLS_PROF_REG_OP_TYPE_CONTEXT_QUAD,
    // --- always add new RegOps to the end here ---
    CU_TOOLS_PROF_REG_OP_TYPE_INVALID,
    CU_TOOLS_PROF_REG_OP_TYPE_FORCE_INT = 0x7fffffff,
}CUetblToolsProfRegOpType;

typedef enum CUetblToolsProfRegOp_enum {
    CU_TOOLS_PROF_REG_OP_WRITE_32        = 0x1,
    CU_TOOLS_PROF_REG_OP_WRITE_MASKED_32,
    CU_TOOLS_PROF_REG_OP_READ_32,
    // --- always add new RegOps to the end here ---
    CU_TOOLS_PROF_REG_OP_INVALID,
    CU_TOOLS_PROF_REG_OP_FORCE_INT = 0x7fffffff,
}CUetblToolsProfRegOp;

typedef enum CUetblToolsProfilerClassScopeType_enum{
    CU_TOOLS_PROF_CLASS_SCOPE_INVALID                                                   =   0,
    CU_TOOLS_PROF_CLASS_SCOPE_GLOBAL                                                    =   1,
    CU_TOOLS_PROF_CLASS_SCOPE_CONTEXT                                                   =   2,
    CU_TOOLS_PROF_CLASS_SCOPE_FORCE_INT                                                 =   0x7fffffff,
}CUetblToolsProfilerClassScopeType;

#define CU_TOOLS_PROF_CLASS_RESERVE                              (0x00000001)
#define CU_TOOLS_PROF_CLASS_RELEASE                              (0x00000002)


#define CU_TOOLS_PROF_CLASS_CONTROL_IGNORE                    (0x00000000)
#define CU_TOOLS_PROF_CLASS_CONTROL_DISABLE                   (0x00000001)
#define CU_TOOLS_PROF_CLASS_CONTROL_ENABLE                    (0x00000002)
#define CU_TOOLS_PROF_CLASS_CONTROL_RELEASE                   (0x00000003)

#define CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FULFILLED         (0x00000000)
#define CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_REJECTED          (0x00000001)
#define CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_NOT_SUPPORTED     (0x00000002)
#define CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FAILED            (0x00000003)

#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_ELCG                 1:0
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_ELCG_IGNORE                     \
    CU_TOOLS_PROF_CLASS_CONTROL_IGNORE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_ELCG_DISABLE                    \
    CU_TOOLS_PROF_CLASS_CONTROL_DISABLE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_ELCG_ENABLE                     \
    CU_TOOLS_PROF_CLASS_CONTROL_ENABLE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_ELCG_RELEASE                    \
    CU_TOOLS_PROF_CLASS_CONTROL_RELEASE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_ELCG_REQUEST_FULFILLED          \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FULFILLED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_ELCG_REQUEST_REJECTED           \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_REJECTED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_ELCG_REQUEST_NOT_SUPPORTED      \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_NOT_SUPPORTED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_ELCG_REQUEST_FAILED             \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FAILED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_BLCG                 3:2
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_BLCG_IGNORE                     \
    CU_TOOLS_PROF_CLASS_CONTROL_IGNORE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_BLCG_DISABLE                    \
    CU_TOOLS_PROF_CLASS_CONTROL_DISABLE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_BLCG_ENABLE                     \
    CU_TOOLS_PROF_CLASS_CONTROL_ENABLE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_BLCG_RELEASE                    \
    CU_TOOLS_PROF_CLASS_CONTROL_RELEASE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_BLCG_REQUEST_FULFILLED          \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FULFILLED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_BLCG_REQUEST_REJECTED           \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_REJECTED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_BLCG_REQUEST_NOT_SUPPORTED      \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_NOT_SUPPORTED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_BLCG_REQUEST_FAILED             \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FAILED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_SLCG                 5:4
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_SLCG_IGNORE                     \
    CU_TOOLS_PROF_CLASS_CONTROL_IGNORE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_SLCG_DISABLE                    \
    CU_TOOLS_PROF_CLASS_CONTROL_DISABLE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_SLCG_ENABLE                     \
    CU_TOOLS_PROF_CLASS_CONTROL_ENABLE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_SLCG_RELEASE                    \
    CU_TOOLS_PROF_CLASS_CONTROL_RELEASE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_SLCG_REQUEST_FULFILLED          \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FULFILLED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_SLCG_REQUEST_REJECTED           \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_REJECTED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_SLCG_REQUEST_NOT_SUPPORTED      \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_NOT_SUPPORTED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_SLCG_REQUEST_FAILED             \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FAILED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_VAT                  11:10
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_VAT_IGNORE                      \
    CU_TOOLS_PROF_CLASS_CONTROL_IGNORE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_VAT_DISABLE                     \
    CU_TOOLS_PROF_CLASS_CONTROL_DISABLE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_VAT_ENABLE                      \
    CU_TOOLS_PROF_CLASS_CONTROL_ENABLE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_VAT_RELEASE                     \
    CU_TOOLS_PROF_CLASS_CONTROL_RELEASE
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_VAT_REQUEST_FULFILLED           \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FULFILLED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_VAT_REQUEST_REJECTED            \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_REJECTED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_VAT_REQUEST_NOT_SUPPORTED       \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_NOT_SUPPORTED
#define CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_VAT_REQUEST_FAILED              \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FAILED

#define CU_TOOLS_PROF_CLASS_PG_CONTROL_MASK_ELPG                 1:0
#define CU_TOOLS_PROF_CLASS_PG_CONTROL_MASK_ELPG_IGNORE                     \
    CU_TOOLS_PROF_CLASS_CONTROL_IGNORE
#define CU_TOOLS_PROF_CLASS_PG_CONTROL_MASK_ELPG_DISABLE                    \
    CU_TOOLS_PROF_CLASS_CONTROL_DISABLE
#define CU_TOOLS_PROF_CLASS_PG_CONTROL_MASK_ELPG_ENABLE                     \
    CU_TOOLS_PROF_CLASS_CONTROL_ENABLE
#define CU_TOOLS_PROF_CLASS_PG_CONTROL_MASK_ELPG_RELEASE                    \
    CU_TOOLS_PROF_CLASS_CONTROL_RELEASE
#define CU_TOOLS_PROF_CLASS_PG_CONTROL_MASK_ELPG_REQUEST_FULFILLED          \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FULFILLED
#define CU_TOOLS_PROF_CLASS_PG_CONTROL_MASK_ELPG_REQUEST_REJECTED           \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_REJECTED
#define CU_TOOLS_PROF_CLASS_PG_CONTROL_MASK_ELPG_REQUEST_NOT_SUPPORTED      \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_NOT_SUPPORTED
#define CU_TOOLS_PROF_CLASS_PG_CONTROL_MASK_ELPG_REQUEST_FAILED             \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FAILED

#define CU_TOOLS_PROF_CLASS_THERMAL_CONTROL_MASK_IDLE_SLOWDOWN       1:0
#define CU_TOOLS_PROF_CLASS_THERMAL_CONTROL_MASK_IDLE_SLOWDOWN_IGNORE            \
    CU_TOOLS_PROF_CLASS_CONTROL_IGNORE
#define CU_TOOLS_PROF_CLASS_THERMAL_CONTROL_MASK_IDLE_SLOWDOWN_DISABLE           \
    CU_TOOLS_PROF_CLASS_CONTROL_DISABLE
#define CU_TOOLS_PROF_CLASS_THERMAL_CONTROL_MASK_IDLE_SLOWDOWN_ENABLE            \
    CU_TOOLS_PROF_CLASS_CONTROL_ENABLE
#define CU_TOOLS_PROF_CLASS_THERMAL_CONTROL_MASK_IDLE_SLOWDOWN_RELEASE           \
    CU_TOOLS_PROF_CLASS_CONTROL_RELEASE
#define CU_TOOLS_PROF_CLASS_THERMAL_CONTROL_MASK_IDLE_SLOWDOWN_REQUEST_FULFILLED \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FULFILLED
#define CU_TOOLS_PROF_CLASS_THERMAL_CONTROL_MASK_IDLE_SLOWDOWN_REQUEST_REJECTED  \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_REJECTED
#define CU_TOOLS_PROF_CLASS_THERMAL_CONTROL_MASK_IDLE_SLOWDOWN_REQUEST_NOT_SUPPORTED \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_NOT_SUPPORTED
#define CU_TOOLS_PROF_CLASS_THERMAL_CONTROL_MASK_IDLE_SLOWDOWN_REQUEST_FAILED    \
    CU_TOOLS_PROF_CLASS_CONTROL_REQUEST_FAILED

typedef struct CUetblToolsProfRegInfo_st {
    /// Number of registers to be verified
    NvU32 count;

    /// RegOp to be used while reading/writing the registers
    CUetblToolsProfRegOpType regOpType;

    /// Offsets of registers
    NvU32 *offset;

    /// Masks to be used while writing
    NvU32 *mask;

    /// Value which is to be written to the registers
    NvU32 *value;

    /// Quad number in case if the regOp is CU_TOOLS_PRI_REG_CONTEXT_QUAD
    NvU8 *quads;

}CUetblToolsProfRegInfo;

typedef struct CUtoolsPCSamplingConfig_st {
    uint32_t struct_size;
    //Sampling period in terms of levels
    CUprofSamplingPeriod period;
    uint32_t period2;
    uint64_t numBlocks;
} CUtoolsPCSamplingConfig;

//smId in virtual PRI order for a vsmId
typedef struct CUtoolsProfVsmMapping_st {
    uint32_t smId;
    uint32_t reserved;
} CUtoolsProfVsmMapping;

typedef struct CUtoolsProfVsmMappings_st  {
    uint32_t struct_size;   // sizeof(CUtoolsProfVsmMappings)  size of this struct
    uint32_t entry_size;    // sizeof(CUtoolsProfVsmMapping)  size of each entry
    uint32_t numSm;         // [out]
    uint32_t entryCount;    // [in] count of CUtoolsVsmMapping's allocated
    CUtoolsProfVsmMapping* entries;
    void *rsvd1;
} CUtoolsProfVsmMappings;

typedef struct CUetblToolsProfiler_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;


    CUprofResult (CUDAAPI *etiDeviceGetAttribute)(
        CUdevice device,
        CUprof_DeviceAttribute attrib,
        uint64_t *value);

    CUprofResult (CUDAAPI *etiDeviceGetTimestamp)(
        CUcontext context,
        uint64_t *timestamp);

    CUprofResult (CUDAAPI *etiDeviceGetNumEventDomains)(
        CUdevice device,
        uint32_t *numdomains);

    CUprofResult (CUDAAPI *etiDeviceEnumEventDomains)(
        CUdevice device,
        size_t *arraySizeBytes,
        CUprof_EventDomainID *domainArray);

    CUprofResult (CUDAAPI *etiEventDomainGetAttribute)(
    CUdevice device,
        CUprof_EventDomainID eventDomain,
        CUprof_EventDomainAttribute attrib,
        size_t *attribSize,
        void *value);

    CUprofResult (CUDAAPI *etiEventDomainGetNumEvents)(
        CUdevice device,
        CUprof_EventDomainID eventDomain,
        uint32_t *numevents);

    CUprofResult (CUDAAPI *etiEventDomainEnumEvents)(
        CUdevice device,
        CUprof_EventDomainID eventDomain,
        size_t *arraySizeBytes,
        CUprof_EventID *eventarray);

    CUprofResult (CUDAAPI *etiEventGetAttribute)(
        CUdevice device,
        CUprof_EventID eventID,
        CUprof_EventAttribute attrib,
        size_t *attribSize,
        void *value);

    CUprofResult (CUDAAPI *etiEventGetIdFromName)(
        CUdevice device,
        const char *eventName,
        CUprof_EventID *eventID);

    CUprofResult (CUDAAPI *etiEventGroupCreate)(
        CUcontext ctx,
        CUprof_EventGroup *eventGroup,
        uint32_t flags);

    CUprofResult (CUDAAPI *etiEventGroupDestroy)(
        CUprof_EventGroup eventGroup);

    CUprofResult (CUDAAPI *etiEventGroupGetAttribute)(
        CUprof_EventGroup eventGroup,
        CUprof_EventGroupAttribute attrib,
        uint64_t *value);

    CUprofResult (CUDAAPI *etiEventGroupSetAttribute)(
        CUprof_EventGroup eventGroup,
        CUprof_EventGroupAttribute attrib,
        uint64_t value);

    CUprofResult (CUDAAPI *etiEventGroupAddEvent)(
        CUprof_EventGroup eventGroup,
        CUprof_EventID eventID);

    CUprofResult (CUDAAPI *etiEventGroupRemoveEvent)(
        CUprof_EventGroup eventGroup,
        CUprof_EventID eventID);

    CUprofResult (CUDAAPI *etiEventGroupRemoveAllEvents)(
        CUprof_EventGroup eventGroup);

    CUprofResult (CUDAAPI *etiEventGroupResetAllEvents)(
        CUprof_EventGroup eventGroup);

    CUprofResult (CUDAAPI *etiEventGroupEnable)(
        CUprof_EventGroup eventGroup);

    CUprofResult (CUDAAPI *etiEventGroupDisable)(
        CUprof_EventGroup eventGroup);

    CUprofResult (CUDAAPI *etiEventGroupReadEvent)(
        CUprof_EventGroup       eventGroup,
        CUprof_ReadEventFlags   flags,
        CUprof_EventID          eventID,
        size_t                 *bufferSizeBytes,
        uint64_t               *counterData);

    CUprofResult (CUDAAPI *etiEventGroupReadAllEvents)(
        CUprof_EventGroup       eventGroup,
        CUprof_ReadEventFlags   flags,
        size_t                 *bufferSizeBytes,
        uint64_t               *counterDataBuffer,
        size_t                 *arraySizeBytes,
        CUprof_EventID         *eventIDArray,
        size_t                 *numCountersRead);

    CUprofResult (CUDAAPI *etiDeviceGetEventDomainAttribute)(
        CUdevice device,
        CUprof_EventDomainID eventDomain,
        CUprof_EventDomainAttribute attrib,
        size_t *attribSize,
        void *value);

    CUprofResult (CUDAAPI *etiSetEventCollectionMode)(
        CUcontext context,
        CUprof_EventCollectionMode value);

    CUprofResult (CUDAAPI *etiDomainCheckCompatible)(
        CUcontext ctx,
        CUprof_EventDomainID domainId1,
        CUprof_EventDomainID domainId2,
        uint32_t *isCompatible);

    CUprofResult (CUDAAPI *etiDeviceGetAttribute_v41)(
        CUdevice device,
        CUprof_DeviceAttribute attrib,
        size_t *attribSize,
        void *value);

    CUprofResult (CUDAAPI *etiEventDomainGetAttribute_v41)(
        CUprof_EventDomainID eventDomain,
        CUprof_EventDomainAttribute attrib,
        size_t *attribSize,
        void *value);

    CUprofResult (CUDAAPI *etiEventDomainGetNumEvents_v41)(
        CUprof_EventDomainID eventDomain,
        uint32_t *numevents);

    CUprofResult (CUDAAPI *etiEventDomainEnumEvents_v41)(
        CUprof_EventDomainID eventDomain,
        size_t *arraySizeBytes,
        CUprof_EventID *eventarray);

    CUprofResult (CUDAAPI *etiEventGetAttribute_v41)(
        CUprof_EventID eventID,
        CUprof_EventAttribute attrib,
        size_t *attribSize,
        void *value);

    CUprofResult (CUDAAPI *etiEventGroupGetAttribute_v41)(
        CUprof_EventGroup eventGroup,
        CUprof_EventGroupAttribute attrib,
        size_t *attribSize,
        void *value);

    CUprofResult (CUDAAPI *etiEventGroupSetAttribute_v41)(
        CUprof_EventGroup eventGroup,
        CUprof_EventGroupAttribute attrib,
        size_t attribSize,
        void *value);

    CUprofResult (CUDAAPI *etiProfilerInit)(void);

    CUprofResult (CUDAAPI *etiEventGroupGetContext)(CUprof_EventGroup eventGroup, CUcontext *ctx);

    CUprofResult (CUDAAPI *etiProfilerDeInit)(void);

    CUprofResult (CUDAAPI *etiRegisterReadWrite)(CUcontext ctx, CUetblToolsProfRegOp operation, CUetblToolsProfRegInfo *param);

    // The below functions are deprecated but are retained to maintain backward compatibility
    // The size of CUetblToolsProfiler is not allowed to shrink. It can only grow.

    CUprofResult (CUDAAPI *etiProfilingModeCompatibleDeprecated)(void);

    CUprofResult (CUDAAPI *etiSetProfilingModeDeprecated)(void);

    // Allocate profiler class. If ctx handle is null or scope is CU_TOOLS_PROF_CLASS_SCOPE_GLOBAL
    // a profiler object is allocated for global use. If ctx!=NULL && scope is CU_TOOLS_PROF_CLASS_SCOPE_CONTEXT
    // then the profiler class is allocated per context use.
    // Return values: CUPROF_ERROR_INVALID_DEVICE, CUPROF_ERROR_INVALID_CONTEXT,
    //                CUPROF_ERROR_INVALID_PARAMETER, CUPROF_ERROR_IN_USE, CUPROF_SUCCESS
    CUprofResult (CUDAAPI *etiProfilerClassAllocate)(const CUdevice device, CUcontext ctx,
                                                     uint32_t *profilerObjHandle, CUetblToolsProfilerClassScopeType scope);

    // Free profiler class. Ctx is not used inside the function
    // Return values: CUPROF_SUCCESS, CUPROF_ERROR_INVALID_DEVICE,
    //                CUPROF_ERROR_INVALID_PARAMETER,
    CUprofResult (CUDAAPI *etiProfilerClassFree)(const CUdevice device, CUcontext ctx,
                                                     uint32_t *profilerObjHandle);

    // Reserve or release perfmon.
    // If profiler class is allocated for global use and if one process
    // acquires perfmon then other process cannot acquire it unless first process releases it.
    // If profiler object is allocated for context use then multiple contexts across multiple processes
    // can acquire perfmon.
    // Inside the process client is supposed to refcount and do only one time access to this function
    // in both cases.
    // Ctx is not used inside the function, scope is defined by the flag used while creating
    // profiler object.
    // Return values: CUPROF_SUCCESS, CUPROF_ERROR_INVALID_DEVICE,
    //                CUPROF_ERROR_INVALID_PARAMETER, CUPROF_ERROR_INVALID_REQUEST
    CUprofResult (CUDAAPI *etiProfilerClassControlPerfmon)(const CUdevice device, CUcontext ctx,
                                                     uint32_t profilerObjHandle,
                                                     uint32_t flag);

    // Reserve or release clock gating controls.
    // ELCG, BLCG and SLCG can be disabled independently using controlmask that is
    // set using the CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_* flags above.
    // Status of each request is returned using CU_TOOLS_PROF_CLASS_CG_CONTROL_REQUEST_FULFILLED/
    // REJECTED/NOT_SUPPORTED in the same locations.
    // Scope is defined by the flag used while creating profiler object.
    // Return values: CUPROF_SUCCESS, CUPROF_ERROR_INVALID_DEVICE,
    //                CUPROF_ERROR_INVALID_PARAMETER
    CUprofResult (CUDAAPI *etiProfilerClassControlClockGating)(const CUdevice device, CUcontext ctx,
                                                     uint32_t profilerObjHandle,
                                                     uint32_t controlMask,
                                                     uint32_t *statusMask,
                                                     uint32_t flag);

    // Reserve or release power gating controls.
    // ELPG can be disabled independently using controlmask that is
    // set using the CU_TOOLS_PROF_CLASS_PG_CONTROL_MASK_* flags above.
    // Status of each request is returned using CU_TOOLS_PROF_CLASS_PG_CONTROL_REQUEST_FULFILLED/
    // REJECTED/NOT_SUPPORTED in the same locations.
    // Still to be tested for different scopes.
    // Scope is defined by the flag used while creating profiler object.
    // Return values: CUPROF_SUCCESS, CUPROF_ERROR_INVALID_DEVICE,
    //                CUPROF_ERROR_INVALID_PARAMETER
    CUprofResult (CUDAAPI *etiProfilerClassControlPowerGating)(const CUdevice device, CUcontext ctx,
                                                     uint32_t profilerObjHandle,
                                                     uint32_t controlMask,
                                                     uint32_t *statusMask,
                                                     uint32_t flag);


    // Reserve or release thermal controls.
    // Thernal controls can be disabled independently using controlmask that is
    // set using the CU_TOOLS_PROF_CLASS_THERMAL_CONTROL_MASK_* flags above.
    // Status of each request is returned using CU_TOOLS_PROF_CLASS_THERMAL_CONTROL_REQUEST_FULFILLED/
    // REJECTED/NOT_SUPPORTED in the same locations.
    // Still to be tested for different scopes.
    // Scope is defined by the flag used while creating profiler object.
    // Return values: CUPROF_SUCCESS, CUPROF_ERROR_INVALID_DEVICE,
    //                CUPROF_ERROR_INVALID_PARAMETER
    CUprofResult (CUDAAPI *etiProfilerClassControlThermalSettings)(const CUdevice device, CUcontext ctx,
                                                                   uint32_t profilerObjHandle,
                                                                   uint32_t controlMask,
                                                                   uint32_t *statusMask,
                                                                   uint32_t flag);

    // Device level PRI access:
    CUprofResult (CUDAAPI *etiDevRegisterReadWrite)(const CUdevice device, CUetblToolsProfRegOp operation, CUetblToolsProfRegInfo *param);

    // Set PM ctxsw bit:
    CUresult (CUDAAPI *etiSetPmCtxswMode)(CUcontext ctx, uint32_t bEnable);

    // Enable/Disable PC Sampling mode:
    CUprofResult (CUDAAPI *etiPCSamplingSetMode)(CUcontext ctx, uint32_t bEnable);

    // Register the callback to read and dump the data into required format:
    CUprofResult (CUDAAPI *etiPCSamplingRegisterReadCallback)(CUcontext ctx,
                                                                  CUprofPCSamplingReadDataCallback pfnCallback,
                                                                  void* pUserData);

    /* Set the sampling period:
     * Deprecated from CUDA 9.0
     */
    CUprofResult (CUDAAPI *etiPCSamplingSetConfig)(CUcontext ctx, CUtoolsPCSamplingConfig *config);

    CUprofResult (CUDAAPI *etiPCSamplingGetPeriodInCycles)(CUcontext ctx, uint64_t *cycles);

    CUprofResult (CUDAAPI *etiIsEventGroupEnabled)(CUprof_EventGroup eventGroup, uint32_t *bEnable);

    // Reserve or release Voltage assisted throttle (VAT).
    // VAT controls can be disabled independently using controlmask that is
    // set using the CU_TOOLS_PROF_CLASS_CG_CONTROL_MASK_* flags above.
    // Status of each request is returned using CU_TOOLS_PROF_CLASS_CG_CONTROL_REQUEST_FULFILLED/
    // REJECTED/NOT_SUPPORTED in the same locations.
    // Still to be tested for different scopes.
    // Scope is defined by the flag used while creating profiler object.
    // Return values: CUPROF_SUCCESS, CUPROF_ERROR_INVALID_DEVICE,
    //                CUPROF_ERROR_INVALID_PARAMETER
    CUprofResult (CUDAAPI *etiProfilerClassControlVat)(const CUdevice device, CUcontext ctx,
                                                       uint32_t profilerObjHandle,
                                                       uint32_t controlMask,
                                                       uint32_t *statusMask,
                                                       uint32_t flag);

    // Get/Set CILP TimeSlice
    CUresult (CUDAAPI *etiGetTimeSlice)(const CUcontext ctx, NvU64 *timeSlice);
    CUresult (CUDAAPI *etiSetTimeSlice)(const CUcontext ctx, const NvU64 timeSlice);
    CUresult (CUDAAPI *etiGetDPrecisionCapabilities)( const CUdevice device, NvBool *isDpEnabled);

    // Check if PC Sampling is supported
    CUresult (CUDAAPI *etiIsPCSamplingSupported)(const CUdevice device, NvBool *bSupport);

    // Enable Channel
    CUresult (CUDAAPI *etiChannelEnable)(const CUcontext ctx, NvBool bEnable);

    // Get/Set 32 Bit Timeslice
    // __TEMP_WAR__ for http://nvbugs/200289939
    CUresult (CUDAAPI *etiGet32BitTimeSlice)(const CUcontext ctx, NvU32 *timeSlice);
    CUresult (CUDAAPI *etiSet32BitTimeSlice)(const CUcontext ctx, const NvU32 timeSlice);

    // Set the sampling period:
    CUprofResult (CUDAAPI *etiPCSamplingSetConfig_v2)(CUcontext ctx, CUtoolsPCSamplingConfig *config);

    //Get the mapping between SM PRI virtual order to vsmid (read from SR_VirtId) order
    CUresult(CUDAAPI *etiDeviceGetSmToVsmMap)(const CUdevice device, CUtoolsProfVsmMappings *params);

    // Begin PC Sampling mode:
    CUprofResult (CUDAAPI *etiBeginPCSampling)(CUcontext ctx);
    // End PC Sampling mode:
    CUprofResult (CUDAAPI *etiEndPCSampling)(CUcontext ctx);
    // Allocate PMA buffer
    CUprofResult (CUDAAPI *etiAllocatePMABuffer)(CUcontext ctx);
    // Free PMA buffer
    CUprofResult (CUDAAPI *etiFreePMABuffer)(CUcontext ctx);
    // Read PMA buffer
    CUprofResult (CUDAAPI *etiConsumePMABuffer)(CUcontext ctx);
    // createWorkerThread
    CUprofResult (CUDAAPI *etiReadIntermdiateBuffer)(CUcontext ctx);
} CUetblToolsProfiler;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
