/*
* Copyright 2009-2012 by NVIDIA Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to NVIDIA
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of NVIDIA Corporation is prohibited.
*/

/******************************************************************************
*
*   Module: cuprofiler.h
*
*   Description:
*       enums to be exposed.
*
******************************************************************************/

#ifndef __CUPROFILER_H__
#define __CUPROFILER_H__

#include "cuda_stdint.h"
#include "nvtypes.h"

typedef enum CUprofResult_enum {
    CUPROF_SUCCESS                                       = 0,
    CUPROF_ERROR_INVALID_PARAMETER                       = 1,
    CUPROF_ERROR_INVALID_DEVICE                          = 2,
    CUPROF_ERROR_INVALID_CONTEXT                         = 3,
    CUPROF_ERROR_INVALID_EVENT_DOMAIN_ID                 = 4,
    CUPROF_ERROR_INVALID_EVENT_ID                        = 5,
    CUPROF_ERROR_INVALID_EVENT_NAME                      = 6,
    CUPROF_ERROR_INVALID_OPERATION                       = 7,
    CUPROF_ERROR_OUT_OF_MEMORY                           = 8,
    CUPROF_ERROR_HARDWARE                                = 9,
    CUPROF_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT           = 10,
    CUPROF_ERROR_API_NOT_IMPLEMENTED                     = 11,
    CUPROF_ERROR_MAX_LIMIT_REACHED                       = 12,
    CUPROF_ERROR_NOT_READY                               = 13,
    CUPROF_ERROR_NOT_COMPATIBLE                          = 14,
    CUPROF_ERROR_NOT_INITIALIZED                         = 15,
    CUPROF_ERROR_DEVICE_MEMORY                           = 16,
    CUPROF_ERROR_IN_USE                                  = 17,
    CUPROF_ERROR_NOT_SUPPORTED                           = 18,
    CUPROF_ERROR_PROFILER_DISABLED                       = 19,
    CUPROF_ERROR_DISABLED                                = 100,
    CUPROF_ERROR_UNKNOWN                                 = 999,
    CUPROF_ERROR_FORCE_INT                               = 0x7fffffff,
} CUprofResult;

// Attributes queryable for a device
typedef enum CUprofDeviceAttribute_enum {
    CUPROF_DEVICE_ATTR_MAX_EVENT_ID        = 1, // The maximum event ID which could be returned
    CUPROF_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID = 2, // The maximum domain ID which could be returned
    CUPROF_DEVICE_ATTR_FORCE_INT           = 0x7fffffff,
} CUprof_DeviceAttribute;

// Attributes queryable for an event domain
typedef enum CUprofEventDomainAttribute_enum {
    CUPROF_EVENT_DOMAIN_ATTR_NAME                   = 0, // value is a pointer to a null terminated const c-string
    CUPROF_EVENT_DOMAIN_ATTR_INSTANCE_COUNT         = 1, // value is an integer count of the number of instances of the domain for which event counts can be collected.
    CUPROF_EVENT_DOMAIN_MAX_EVENTS                  = 2, // value is an integer count of the number of max events available in this domain
    CUPROF_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT   = 3, // value is an integer count of the total number of instances of the domain, including instances that cannot be profiled.
    CUPROF_EVENT_DOMAIN_ATTR_COLLECTION_METHOD      = 4, // value is of type CUprof_EventCollectionMethod.
    CUPROF_EVENT_DOMAIN_ATTR_FORCE_INT              = 0x7fffffff,
} CUprof_EventDomainAttribute;

// Attributes queryable for an event
typedef enum CUprofEventAttribute_enum {
    CUPROF_EVENT_ATTR_NAME              = 0, // value is a pointer to a null terminated const c-string
    CUPROF_EVENT_ATTR_SHORT_DESCRIPTION = 1, // value is a pointer to a null terminated const c-string
    CUPROF_EVENT_ATTR_LONG_DESCRIPTION  = 2, // value is a pointer to a null terminated const c-string
    CUPROF_EVENT_ATTR_CATEGORY          = 3, // value is a pointer to a null terminated const c-string
    CUPROF_EVENT_ATTR_DOMAIN_ID         = 4, // value is of type CUprof_EventDomainID. This attribute is only for internal use.
    CUPROF_EVENT_ATTR_SCOPE             = 5, // value is of type CUprof_EventCollectionScope.
    CUPROF_EVENT_ATTR_FORCE_INT         = 0x7fffffff,
} CUprof_EventAttribute;

// Attributes for an event group some are read-only, others are read/write where noted.
typedef enum CUprofEventGroupAttribute_enum {
    CUPROF_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID                = 0, /// [rw] the domain to which the event group is bound, set by the first event added. 
                                                                ///      May be set on a new event group prior to adding any events to limit which events may be added.
    CUPROF_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES   = 1, /// [rw] Profile all the instances of the current eventgroup domain. 
    CUPROF_EVENT_GROUP_ATTR_USER_DATA                      = 2, /// [rw] opaque user data
    CUPROF_EVENT_GROUP_ATTR_NUM_EVENTS                     = 3, /// [ro] Number of events present in the group
    CUPROF_EVENT_GROUP_ATTR_EVENTS                         = 4, /// [ro] Array of events added in the group
    CUPROF_EVENT_GROUP_ATTR_INSTANCE_COUNT                 = 5, /// [ro] Number of instances of the domain bound to this event group that will be counted
    CUPROF_EVENT_GROUP_ATTR_SCOPE                          = 6, /// [ro] Profiling scope of the eventGroup.
    CUPROF_EVENT_GROUP_ATTR_FORCE_INT                      = 0x7fffffff,
} CUprof_EventGroupAttribute;

typedef enum CUprofEventCollectionMode_enum {
    CUPROF_EVENT_COLLECTION_MODE_CONTINUOUS                = 0,   // sampling mode (default)
    CUPROF_EVENT_COLLECTION_MODE_KERNEL                    = 1,   // kernel mode - for tight profiling of kernel
    CUPROF_EVENT_COLLECTION_MODE_FORCE_INT                 = 0x7fffffff
} CUprof_EventCollectionMode;

typedef enum CUprofEventCategory_enum {
    CUPROF_EVENT_CATEGORY_INSTRUCTION                                = 0,
    CUPROF_EVENT_CATEGORY_MEMORY                                     = 1,
    CUPROF_EVENT_CATEGORY_CACHE                                      = 2,
    CUPROF_EVENT_CATEGORY_PROF_TRIG                                  = 3,
    CUPROF_EVENT_CATEGORY_SYS                                        = 4,
    CUPROF_EVENT_CATEGORY_FORCE_INT                                  = 0x7fffffff
} CUprof_EventCategory;

typedef enum CUprofEventCollectionMethod_enum {
    CUPROF_EVENT_COLLECTION_METHOD_HWPM                 = 0,
    CUPROF_EVENT_COLLECTION_METHOD_SMPERF               = 1,
    CUPROF_EVENT_COLLECTION_METHOD_SW                   = 2,
    CUPROF_EVENT_COLLECTION_METHOD_HWPMMUX              = 3,
    // keep internal only attributes at the end
    CUPROF_EVENT_COLLECTION_METHOD_NOPTRIG_SW           = 100,
    CUPROF_EVENT_COLLECTION_METHOD_PPA_NOPTRIG_SW       = 101,
    CUPROF_EVENT_COLLECTION_METHOD_NVLINK_TC            = 102,
    CUPROF_EVENT_COLLECTION_METHOD_CUPTI_SW             = 103,
    CUPROF_EVENT_COLLECTION_METHOD_INVALID              = 999,
    CUPROF_EVENT_COLLECTION_METHOD_FORCE_INT            = 0x7fffffff
} CUprof_EventCollectionMethod;

typedef enum CUprofNvlinkDataTransferDirection_enum {
    CUPROF_NVLINK_DATA_INVALID                   = 0,
    CUPROF_NVLINK_DATA_TRANSMIT                  = 1,
    CUPROF_NVLINK_DATA_RECIEVE                   = 2,
    CUPROF_NVLINK_DATA_FORCE_INT                 = 0x7fffffff,
}CUprofNvlinkDataTransferDirection;

typedef enum CUprofNvlinkNumber_enum {
    CUPROF_NVLINK_0                              = 0,
    CUPROF_NVLINK_1                              = 1,
    CUPROF_NVLINK_2                              = 2,
    CUPROF_NVLINK_3                              = 3,
    CUPROF_NVLINK_INVALID                        = 999,
    CUPROF_NVLINK_FORCE_INT                      = 0x7fffffff,
}CUprofNvlinkNumber;

// Flags for counter reading APIs
typedef enum CUprofReadEventFlags_enum {
    CUPROF_READ_EVENT_FLAG_NONE          = 0,
    CUPROF_READ_EVENT_FLAG_FORCE_INT     = 0x7fffffff,
} CUprof_ReadEventFlags;

// cant_issue_reasons exposed by toolkit:
typedef enum CUprofPCSamplingStallReasons_enum {
    CUPROF_PC_SAMPLING_STALL_GROUP_INVALID                        = 0,
    CUPROF_PC_SAMPLING_STALL_GROUP_NONE                           = 1,
    CUPROF_PC_SAMPLING_STALL_GROUP_INST_FETCH                     = 2,
    CUPROF_PC_SAMPLING_STALL_GROUP_EXEC_DEPENDENCY                = 3,
    CUPROF_PC_SAMPLING_STALL_GROUP_MEMORY_DEPENDENCY              = 4,
    CUPROF_PC_SAMPLING_STALL_GROUP_TEXTURE                        = 5,
    CUPROF_PC_SAMPLING_STALL_GROUP_SYNC                           = 6,
    CUPROF_PC_SAMPLING_STALL_GROUP_CONSTANT_MEMORY_DEPENDENCY     = 7,
    CUPROF_PC_SAMPLING_STALL_GROUP_PIPE_BUSY                      = 8,
    CUPROF_PC_SAMPLING_STALL_GROUP_MEMORY_THROTTLE                = 9,
    CUPROF_PC_SAMPLING_STALL_GROUP_NOT_SELECTED                   = 10,
    CUPROF_PC_SAMPLING_STALL_GROUP_OTHER                          = 11,
    CUPROF_PC_SAMPLING_STALL_GROUP_BRANCH_RESOLVING               = 12,
    CUPROF_PC_SAMPLING_STALL_GROUP_SLEEPING                       = 13,
    CUPROF_PC_SAMPLING_STALL_GROUP_MAX_CIR                        = 14,
    CUPROF_PC_SAMPLING_STALL_GROUP_FORCE_INT                      = 0x7fffffff
}CUprofPCSamplingStallReasons;

// cant_issue_reasons extracted from h/w:
typedef enum CUprofWarpCantIssueReasons_enum {
    CUPROF_WARP_CANT_ISSUE_INVALID                         = 0,
    CUPROF_WARP_CANT_ISSUE_DRAIN                           = 1,
    CUPROF_WARP_CANT_ISSUE_ICACHE_MISS                     = 2,
    CUPROF_WARP_CANT_ISSUE_IMC_MISS                        = 3,
    CUPROF_WARP_CANT_ISSUE_LONG_SCOREBOARD                 = 4,
    CUPROF_WARP_CANT_ISSUE_BARRIER                         = 5,
    CUPROF_WARP_CANT_ISSUE_MEMBAR                          = 6,
    CUPROF_WARP_CANT_ISSUE_OFF_DECK_SHORT_SCOREBOARD       = 7,
    CUPROF_WARP_CANT_ISSUE_TILE_ALLOCATION_STALL           = 8,
    CUPROF_WARP_CANT_ISSUE_ALLOCATION_STALL_NOT_ELIGIBLE   = 9,
    CUPROF_WARP_CANT_ISSUE_ALLOCATION_STALL_NO_SPACE       = 10,
    CUPROF_WARP_CANT_ISSUE_WAIT                            = 11,
    CUPROF_WARP_CANT_ISSUE_ON_DECK_LONG_SCOREBOARD         = 12,
    CUPROF_WARP_CANT_ISSUE_ON_DECK_SHORT_SCOREBOARD        = 13,
    CUPROF_WARP_CANT_ISSUE_NO_INSTRUCTIONS                 = 14,
    CUPROF_WARP_CANT_ISSUE_SHADOW_PIPE_THROTTLE            = 15,
    CUPROF_WARP_CANT_ISSUE_TEX_THROTTLE                    = 16,
    CUPROF_WARP_CANT_ISSUE_MIO_THROTTLE                    = 17,
    CUPROF_WARP_CANT_ISSUE_DISPATCH_STALL                  = 18,
    CUPROF_WARP_CANT_ISSUE_NOT_SELECTED                    = 19,
    CUPROF_WARP_CANT_ISSUE_SELECTED                        = 20,
    CUPROF_WARP_CANT_ISSUE_MISC                            = 21,
    CUPROF_WARP_CANT_ISSUE_ON_DECK_MISC                    = 22,
    CUPROF_WARP_CANT_ISSUE_BRANCH_RESOLVING                = 23,
    CUPROF_WARP_CANT_ISSUE_SLEEPING                        = 24,
    CUPROF_WARP_CANT_ISSUE_LG_THROTTLE                     = 25,
    CUPROF_WARP_CANT_ISSUE_MAX_CIR                         = 26,
    CUPROF_WARP_CANT_ISSUE_FORCE_INT                       = 0x7fffffff,
}CUprofWarpCantIssueReasons;

typedef enum CUprofSamplingPeriod_enum {
    CUPROF_SAMPLING_PERIOD_INVALID = 0,
    CUPROF_SAMPLING_PERIOD_MIN = 1,
    CUPROF_SAMPLING_PERIOD_LOW = 2,
    CUPROF_SAMPLING_PERIOD_MID = 3,
    CUPROF_SAMPLING_PERIOD_HIGH = 4,
    CUPROF_SAMPLING_PERIOD_MAX = 5,
    CUPROF_SAMPLING_PERIOD_FORCE_INT = 0x7fffffff
} CUprofSamplingPeriod;

typedef enum CUprofEventCollectionscope_enum {
    CUPROF_EVENT_COLLECTION_SCOPE_CONTEXT               = 0,
    CUPROF_EVENT_COLLECTION_SCOPE_DEVICE                = 1,
    CUPROF_EVENT_COLLECTION_SCOPE_BOTH                  = 2,
    CUPROF_EVENT_COLLECTION_SCOPE_FORCE_INT             = 0x7fffffff
} CUprof_EventCollectionScope;

#define CUPROF_EVENT_DOMAIN_DEFAULT 0

// Adding a mechanism to simulate PC sampling
#define SIMULATE_PC_SAMPLING 0

typedef uint32_t CUprof_EventDomainID;
typedef uint32_t CUprof_EventID;
typedef void * CUprof_EventGroup;

/// \brief Callback signature for readSamplingData()
typedef void (CUDAAPI *CUprofPCSamplingReadDataCallback)(uint32_t* pBuffer, size_t bufferSize, void* pUserData);

typedef struct CUprofSSYInstrTable_st {
    NvU32 destOffset;
    NvU32 codeOffset;
}CUprofSSYInstrTable;

typedef struct {
  CUprofSamplingPeriod period;
} CUprofSamplingConfig;

// Structure to store function state
typedef struct profFuncState_st {
    CUprofSSYInstrTable *ssyInstrTable;
    NvU32 numSsyInsts;
    NvU32 originalCodeSize;
    NvU32 originalLmemSize;
    NvU32 originalRegCount;
    NvU32 pcOffset;
    NvU32 patchStubPCOffset;
    unsigned char *patchedCode;
}profFuncState_st;

#endif /* __CUPROFILER_H__ */
