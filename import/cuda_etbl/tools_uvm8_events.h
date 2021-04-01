/*
 * Copyright 1993-2016 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_uvm8_events_h__
#define __cuda_etbl_tools_uvm8_events_h__

#include "cuda_uuid.h"
#include "cuda_stdint.h"
#include "cuda_etbl/tools_common_types.h"
#include "cuda_etbl/tools_uvm.h"
#if defined(_WIN32) || defined (_WIN64) // _WIN32 would be sufficient in MSVC
#include <windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// UVM event buffer must be page-aligned
#if defined(__powerpc64__)
#define CU_TOOLS_UVM8_EVENT_BUFFER_ALIGNMENT (64*1024)
#else
#define CU_TOOLS_UVM8_EVENT_BUFFER_ALIGNMENT 4096
#endif 

// The UVM 8 session object
typedef struct CUtoolsUvm8Session_st* CUtoolsUvm8Session;

// The UVM 8 event queue object
typedef struct CUtoolsUvm8EventQueue_st* CUtoolsUvm8EventQueue;

#if defined(_WIN32) || defined (_WIN64) // _WIN32 would be sufficient in MSVC
    typedef HANDLE CUtoolsOsHandle;
#else
    typedef int CUtoolsOsHandle;
#endif

typedef CUtoolsOsHandle CUtoolsUvm8FileDescriptor;
typedef CUtoolsOsHandle CUtoolsUvm8EventQueueNotificationHandle;

typedef enum CUtools_uvm8_event_memory_access_type_enum 
{
    CU_TOOLS_UVM8_EVENT_MEMORY_ACCESS_TYPE_INVALID   = 0,
    CU_TOOLS_UVM8_EVENT_MEMORY_ACCESS_TYPE_READ      = 1,
    CU_TOOLS_UVM8_EVENT_MEMORY_ACCESS_TYPE_WRITE     = 2,
    CU_TOOLS_UVM8_EVENT_MEMORY_ACCESS_TYPE_ATOMIC    = 3,
    CU_TOOLS_UVM8_EVENT_MEMORY_ACCESS_TYPE_PREFETCH  = 4,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM8_EVENT_MEMORY_ACCESS_TYPE_SIZE,
    CU_TOOLS_UVM8_EVENT_MEMORY_ACCESS_TYPE_FORCE_INT = 0x7fffffff
} CUtools_uvm8_event_memory_access_type;

typedef enum CUtools_uvm8_event_type_enum
{
    CU_TOOLS_UVM8_EVENT_TYPE_INVALID                   = 0,
    CU_TOOLS_UVM8_EVENT_TYPE_CPU_FAULT                 = 1,
    // TODO: remove when tools change to CPU_FAULT
    CU_TOOLS_UVM8_EVENT_TYPE_MEMORY_VIOLATION          = CU_TOOLS_UVM8_EVENT_TYPE_CPU_FAULT,
    CU_TOOLS_UVM8_EVENT_TYPE_MIGRATION                 = 2,
    CU_TOOLS_UVM8_EVENT_TYPE_GPU_FAULT                 = 3,
    CU_TOOLS_UVM8_EVENT_TYPE_GPU_FAULT_REPLAY          = 4,
    CU_TOOLS_UVM8_EVENT_TYPE_FAULT_BUFFER_OVERFLOW     = 5,
    CU_TOOLS_UVM8_EVENT_TYPE_FATAL_FAULT               = 6,
    CU_TOOLS_UVM8_EVENT_TYPE_READ_DUPLICATE            = 7,
    CU_TOOLS_UVM8_EVENT_TYPE_READ_DUPLICATE_INVALIDATE = 8,
    CU_TOOLS_UVM8_EVENT_TYPE_PAGE_SIZE_CHANGE          = 9,
    CU_TOOLS_UVM8_EVENT_TYPE_THRASHING_DETECTED        = 10,
    CU_TOOLS_UVM8_EVENT_TYPE_THROTTLING_START          = 11,
    CU_TOOLS_UVM8_EVENT_TYPE_THROTTLING_END            = 12,
    CU_TOOLS_UVM8_EVENT_TYPE_MAP_REMOTE                = 13,
    CU_TOOLS_UVM8_EVENT_TYPE_EVICTION                  = 14,
    CU_TOOLS_UVM8_NUM_EVENT_TYPES                      = 15,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM8_EVENT_TYPE_SIZE,
    CU_TOOLS_UVM8_EVENT_TYPE_FORCE_INT           = 0x7fffffff
} CUtools_uvm8_event_type;

typedef enum CUtools_uvm8_event_migration_cause_enum
{
    CU_TOOLS_UVM8_EVENT_MIGRATION_CAUSE_INVALID         = 0,
    CU_TOOLS_UVM8_EVENT_MIGRATION_CAUSE_USER            = 1,
    CU_TOOLS_UVM8_EVENT_MIGRATION_CAUSE_COHERENCE       = 2,
    CU_TOOLS_UVM8_EVENT_MIGRATION_CAUSE_PREFETCH        = 3,
    CU_TOOLS_UVM8_EVENT_MIGRATION_CAUSE_EVICTION        = 4,
    CU_TOOLS_UVM8_EVENT_MIGRATION_CAUSE_ACCESS_COUNTERS = 5,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM8_EVENT_MIGRATION_CAUSE_SIZE,
    CU_TOOLS_UVM8_EVENT_MIGRATION_CAUSE_FORCE_INT  = 0x7fffffff
} CUtools_uvm8_event_migration_cause;

typedef enum CUtools_uvm8_event_map_remote_cause_enum
{
    CU_TOOLS_UVM8_EVENT_MAP_REMOTE_CAUSE_INVALID       = 0,
    CU_TOOLS_UVM8_EVENT_MAP_REMOTE_CAUSE_COHERENCE     = 1,
    CU_TOOLS_UVM8_EVENT_MAP_REMOTE_CAUSE_THRASHING     = 2,
    CU_TOOLS_UVM8_EVENT_MAP_REMOTE_CAUSE_POLICY        = 3,
    CU_TOOLS_UVM8_EVENT_MAP_REMOTE_CAUSE_OUT_OF_MEMORY = 4,
    CU_TOOLS_UVM8_EVENT_MAP_REMOTE_CAUSE_EVICTION      = 5,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM8_EVENT_MAP_REMOTE_CAUSE_SIZE,
    CU_TOOLS_UVM8_EVENT_MAP_REMOTE_CAUSE_FORCE_INT  = 0x7fffffff
} CUtools_uvm8_event_map_remote_cause;

typedef enum CUtools_uvm8_event_timestamp_type_enum
{
    CU_TOOLS_UVM8_EVENT_TIMESTAMP_TYPE_INVALID               = 0,
    CU_TOOLS_UVM8_EVENT_TIMESTAMP_TYPE_WIN32_QPC             = 1,
    CU_TOOLS_UVM8_EVENT_TIMESTAMP_TYPE_POSIX_CLOCK_GET_TIME  = 2,
    CU_TOOLS_UVM8_EVENT_TIMESTAMP_TYPE_AUTO                  = 3,
    CU_TOOLS_UVM8_EVENT_TIMESTAMP_TYPE_MACH_ABSOLUTE_TIME    = 4,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM8_EVENT_TIMESTAMP_TYPE_SIZE,
    CU_TOOLS_UVM8_EVENT_TIMESTAMP_TYPE_FORCE_INT             = 0x7fffffff
} CUtools_uvm8_event_timestamp_type;

typedef enum CUtools_uvm8_fault_event_type_enum
{
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_INVALID                  = 0,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_INVALID_PDE              = 1,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_INVALID_PTE              = 2,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_WRITE                    = 3,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_ATOMIC                   = 4,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_INVALID_PDE_SIZE         = 5,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_LIMIT_VIOLATION          = 6,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_UNBOUND_INST_BLOCK       = 7,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_PRIV_VIOLATION           = 8,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_PITCH_MASK_VIOLATION     = 9,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_WORK_CREATION            = 10,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_UNSUPPORTED_APERTURE     = 11,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_COMPRESSION_FAILURE      = 12,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_UNSUPPORTED_KIND         = 13,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_REGION_VIOLATION         = 14,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_POISON                   = 15,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_SIZE,
    CU_TOOLS_UVM8_FAULT_EVENT_TYPE_FORCE_INT                = 0x7fffffff
} CUtools_uvm8_fault_event_type;

typedef enum CUtools_uvm8_event_fatal_reason_type_enum
{
    CU_TOOLS_UVM8_EVENT_FATAL_REASON_TYPE_INVALID             = 0,
    CU_TOOLS_UVM8_EVENT_FATAL_REASON_TYPE_INVALID_ADDRESS     = 1,
    CU_TOOLS_UVM8_EVENT_FATAL_REASON_TYPE_INVALID_PERMISSIONS = 2,
    CU_TOOLS_UVM8_EVENT_FATAL_REASON_TYPE_INVALID_FAULT_TYPE  = 3,
    CU_TOOLS_UVM8_EVENT_FATAL_REASON_TYPE_OUT_OF_MEMORY       = 4,
    CU_TOOLS_UVM8_EVENT_FATAL_REASON_TYPE_INTERNAL_ERROR      = 5,
    CU_TOOLS_UVM8_EVENT_FATAL_REASON_TYPE_INVALID_OPERATION   = 6,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM8_EVENT_FATAL_REASON_TYPE_SIZE,
    CU_TOOLS_UVM8_EVENT_FATAL_REASON_TYPE_FORCE_INT           = 0x7fffffff
} CUtools_uvm8_event_fatal_reason_type;

typedef enum
{
    CU_TOOLS_UVM8_FAULT_CLIENT_TYPE_INVALID     = 0,
    CU_TOOLS_UVM8_FAULT_CLIENT_TYPE_GPC         = 1,
    CU_TOOLS_UVM8_FAULT_CLIENT_TYPE_HUB         = 2,

    // ---- Add new values above this line
    CU_TOOLS_UVM8_FAULT_CLIENT_TYPE_SIZE,
    CU_TOOLS_UVM8_FAULT_CLIENT_TYPE_FORCE_INT   = 0x7fffffff
} CUtools_uvm8_event_fault_client_type;

// Information associated with a memory violation event
typedef struct CUtoolsUvm8EventCpuFaultInfo_st
{
    // eventType has to be the 1st argument of this structure.
    uint8_t eventType;
    uint8_t accessType;          // read/write violation (CUtools_uvm_event_memory_access_type)
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack 
    // or malign-double will have no effect on the field offsets.
    uint16_t padding16Bits;
    uint32_t padding32Bits;
    uint64_t address;            // faulting address
    uint64_t timeStamp;          // time when the fault occurred
    uint32_t pid;                // process id causing the fault
    uint32_t threadId;           // thread id causing the fault
    uint64_t pc;                 // address of the instruction causing the fault
} CUtoolsUvm8EventCpuFaultInfo;

// TODO: Bug 1773738: Switch tools to use the proper type name and remove this WAR
typedef CUtoolsUvm8EventCpuFaultInfo CUtoolsUvm8EventMemoryViolationInfo;

// Information associated with a migration event
typedef struct CUtoolsUvm8EventMigrationInfo_st
{
    // eventType has to be the 1st argument of this structure.
    uint8_t eventType;
    uint8_t migrationCause;                    // Cause that triggered the migration
    // Indices are used for the source and destination of migration instead of
    // using gpu uuid/cpu id. This reduces the size of each event.
    // gpuIndex to gpuUuid relation can be obtained from UvmEventGetGpuUuidTable.
    // Currently we do not distinguish between CPUs so they all use index 0.
    uint8_t srcIndex;                          // source CPU/GPU index
    uint8_t dstIndex;                          // destination CPU/GPU index
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack 
    // or malign-double will have no effect on the field offsets
    uint32_t padding32Bits;
    uint64_t address;                          // virtual addr used for migration
    uint64_t migratedBytes;                    // number of bytes migrated
    uint64_t beginTimeStamp;                   // cpu time stamp when the memory transfer
                                               // was queued on the gpu
    uint64_t endTimeStamp;                     // cpu time stamp when the memory transfer
                                               // finalization was communicated to the cpu
                                               // For asynchronous operations this field
                                               // will be zero
    uint64_t streamId;                         // stream tied with this migration
    uint64_t beginTimeStampGpu;                // time stamp when the migration started
                                               // on the gpu
    uint64_t endTimeStampGpu;                  // time stamp when the migration finished
                                               // on the gpu
} CUtoolsUvm8EventMigrationInfo;

// Information associated with a GPU page fault event
typedef struct CUtoolsUvm8EventGpuFaultInfo_st
{
    // eventType has to be the 1st argument of this structure
    uint8_t eventType;
    uint8_t faultType;        // type of fault (CUtools_uvm_fault_event_type)
    uint8_t accessType;       // memory access type (CUtools_uvm_event_memory_access_type)
    uint8_t gpuIndex;
    union
    {
        uint16_t gpcId;      // If this is a replayable fault, this field contains
                             // the physical GPC index where the fault was
        uint16_t channelId;  // triggered. Otherwise, it contains the id of the
                             // channel that launched the operation that caused the
                             // fault.
    };
    uint16_t clientId;       // Id of the MMU client that triggered the fault. This
                             // is the value provided by HW and is architecture-
                             // specific. There are separate client ids for
                             // different client types (See dev_fault.h).
    uint64_t address;        // virtual address at which gpu faulted
    union
    {
    // TODO: Bug 1773738: Switch tools to use the proper field and remove this WAR
    uint64_t timestampCpu;
    uint64_t timeStamp;       // time stamp when the cpu started processing the
                              // fault
    };
    uint64_t timeStampGpu;   // gpu time stamp when the fault entry was written
                             // in the fault buffer
    uint32_t batchId;        // Per-GPU unique id to identify the faults serviced
                             // in batch before:
                             // - Issuing a replay for replayable faults
                             // - Re-scheduling the channel for non-replayable
                             //   faults.
    uint8_t clientType;      // Volta+ GPUs can fault on clients other than GR.
                             // CU_TOOLS_UVM8_FAULT_CLIENT_TYPE_GPC indicates replayable
                             // fault, while CU_TOOLS_UVM8_FAULT_CLIENT_TYPE_HUB indicates
                             // non-replayable fault.

    //
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack
    // or malign-double will have no effect on the field offsets
    //
    uint8_t padding8Bits;
    uint16_t padding16Bits;
} CUtoolsUvm8EventGpuFaultInfo;

// TODO: Bug 1773738: Switch tools to use the proper type name and remove this WAR
typedef CUtoolsUvm8EventGpuFaultInfo CUtoolsUvm8EventGpuPageFaultInfo;

// Information associated with a GPU page fault replay event
typedef struct CUtoolsUvm8EventGpuFaultReplayInfo_st
{
    // eventType has to be the 1st argument of this structure
    uint8_t eventType;
    uint8_t gpuIndex;
    uint8_t clientType;        // See clientType in CUtoolsUvm8EventGpuFaultInfo
    //
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack
    // or malign-double will have no effect on the field offsets
    //
    uint8_t padding8bits;
    uint32_t batchId;          // Per-GPU unique id to identify the faults that have
                            // been serviced in batch
    uint64_t timeStamp;        // cpu time when the replay of the faulting memory
                            // accesses is queued on the gpu
    uint64_t timeStampGpu;     // gpu time stamp when the replay operation finished
                            // executing on the gpu
} CUtoolsUvm8EventGpuFaultReplayInfo;

// TODO: Bug 1773738: Switch tools to use the proper type name and remove this WAR
typedef CUtoolsUvm8EventGpuFaultReplayInfo CUtoolsUvm8EventGpuPageFaultReplayInfo;

typedef struct CUtoolsUvm8EventFatalFaultInfo_st
{
    // eventType has to be the 1st argument of this structure.
    uint8_t eventType;
    uint8_t faultType;       // type of gpu fault, refer UvmEventFaultType. Only valid
                             // if processorIndex is a GPU
    uint8_t accessType;      // memory access type, refer UvmEventMemoryAccessType
    uint8_t processorIndex;  // processor that experienced the fault
    uint8_t reason;          // reason why the fault is fatal, refer UvmEventFatalReason
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack
    // or malign-double will have no effect on the field offsets
    uint8_t padding8bits;
    uint16_t padding16bits;
    uint64_t address;        // virtual address at which the processor faulted
    uint64_t timeStamp;      // CPU time when the fault is detected to be fatal
} CUtoolsUvm8EventFatalFaultInfo;

typedef struct CUtoolsUvm8EventReadDuplicateInfo_st
{
    // eventType has to be the 1st argument of this structure
    uint8_t eventType;
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack
    // or malign-double will have no effect on the field offsets
    uint8_t padding8bits;
    uint16_t padding16bits;
    uint32_t padding32bits;
    uint64_t processors;    // mask that specifies in which processors this
                            // memory region is read-duplicated
    uint64_t address;       // virtual address of the memory region that is
                            // read-duplicated
    uint64_t size;          // size in bytes of the memory region that is
                            // read-duplicated
    uint64_t timeStamp;     // cpu time stamp when the memory region becomes
                            // read-duplicate. Since many processors can
                            // participate in read-duplicate this is time stamp
                            // when all the operations have been pushed to all
                            // the processors.
} CUtoolsUvm8EventReadDuplicateInfo;

typedef struct CUtoolsUvm8EventReadDuplicateInvalidateInfo_st
{
    // eventType has to be the 1st argument of this structure
    uint8_t eventType;
    uint8_t residentIndex;  // index of the cpu/gpu that now contains the only
                            // valid copy of the memory region
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack
    // or malign-double will have no effect on the field offsets
    uint16_t padding16bits;
    uint32_t padding32bits;
    uint64_t address;       // virtual address of the memory region that is
                            // read-duplicated
    uint64_t size;          // size of the memory region that is
                            // read-duplicated
    uint64_t timeStamp;     // cpu time stamp when the memory region is no
                            // longer read-duplicate. Since many processors can
                            // participate in read-duplicate this is time stamp
                            // when all the operations have been pushed to all
                            // the processors.
} CUtoolsUvm8EventReadDuplicateInvalidateInfo;

typedef struct CUtoolsUvm8EventPageSizeChangeInfo_st
{
    // eventType has to be the 1st argument of this structure
    uint8_t eventType;
    uint8_t processorIndex; // cpu/gpu processor index for which the page size
                            // changed
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack
    // or malign-double will have no effect on the field offsets
    uint16_t padding16bits;
    uint32_t size;          // new page size
    uint64_t address;       // virtual address of the page whose size has
                            // changed
    uint64_t timeStamp;     // cpu time stamp when the new page size is
                            // queued on the gpu
} CUtoolsUvm8EventPageSizeChangeInfo;

typedef struct CUtoolsUvm8EventThrashingDetectedInfo_st
{
    // eventType has to be the 1st argument of this structure.
    uint8_t eventType;
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack
    // or malign-double will have no effect on the field offsets
    uint8_t padding8bits;
    uint16_t padding16bits;
    uint32_t padding32bits;
    uint64_t processors;    // mask that specifies which processors are
                            // fighting for this memory region
    uint64_t address;       // virtual address of the memory region that is
                            // thrashing
    uint64_t size;          // size of the memory region that is thrashing
    uint64_t timeStamp;     // cpu time stamp when thrashing is detected
} CUtoolsUvm8EventThrashingDetectedInfo;

typedef struct CUtoolsUvm8EventThrottlingStartInfo_st
{
    // eventType has to be the 1st argument of this structure.
    uint8_t eventType;
    uint8_t processorIndex;    // index of the cpu/gpu that was throttled
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack
    // or malign-double will have no effect on the field offsets
    uint16_t padding16bits;
    uint32_t padding32bits;
    uint64_t address;          // address of the page whose servicing is being
                               // throttled
    uint64_t timeStamp;        // cpu start time stamp for the throttling operation
} CUtoolsUvm8EventThrottlingStartInfo;

typedef struct CUtoolsUvm8EventThrottlingEndInfo_st
{
    // eventType has to be the 1st argument of this structure.
    uint8_t eventType;
    //
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack
    // or malign-double will have no effect on the field offsets
    //
    uint8_t processorIndex;    // index of the cpu/gpu that was throttled
    uint16_t padding16bits;
    uint32_t padding32bits;
    uint64_t address;          // address of the page whose servicing was
                               // throttled
    uint64_t timeStamp;        // cpu end time stamp for the throttling operation
} CUtoolsUvm8EventThrottlingEndInfo;

typedef struct CUtoolsUvm8EventMapRemoteInfo_st
{
    // eventType has to be the 1st argument of this structure.
    uint8_t eventType;
    uint8_t srcIndex;       // index of the cpu/gpu being remapped
    uint8_t dstIndex;       // index of the cpu/gpu memory that contains the
                            // valid copy of data
    uint8_t mapRemoteCause; // field to type CUtools_uvm8_event_map_remote_cause
                            // that tells the cause for the page to be mapped
                            // remotely
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack
    // or malign-double will have no effect on the field offsets
    uint32_t padding32bits;
    uint64_t address;       // virtual address of the memory region that is
                            // mapped remotely
    uint64_t size;          // size of the memory region that is mapped remotely
    uint64_t timeStamp;     // cpu time stamp when all the required operations
                            // have been pushed to the processor
    uint64_t timeStampGpu;  // time stamp when the new mapping is effective in
                            // the processor specified by srcIndex. If srcIndex
                            // is a cpu, this field will be zero.
} CUtoolsUvm8EventMapRemoteInfo;

typedef struct CUtoolsUvm8EventEvictionInfo_st
{
    // eventType has to be the 1st argument of this structure.
    uint8_t eventType;
    uint8_t srcIndex;       // index of the cpu/gpu from which data is being
                            // evicted
    uint8_t dstIndex;       // index of the cpu/gpu memory to which data is
                            // going to be stored
    // This structure is shared between UVM kernel and tools.
    // Manually padding the structure so that compiler options like pragma pack
    // or malign-double will have no effect on the field offsets
    uint8_t padding8bits;
    uint32_t padding32bits;
    uint64_t addressOut;    // virtual address of the memory region that is
                            // being evicted
    uint64_t addressIn;     // virtual address that caused the eviction
    uint64_t size;          // size of the memory region that being evicted
    uint64_t timeStamp;     // cpu time stamp when eviction starts on the cpu
} CUtoolsUvm8EventEvictionInfo;

typedef union CUtoolsUvm8EventInfo_union
{
    uint8_t eventType;
    CUtoolsUvm8EventCpuFaultInfo cpuFault;
    CUtoolsUvm8EventMigrationInfo migration;
    CUtoolsUvm8EventGpuFaultInfo gpuFault;
    CUtoolsUvm8EventGpuFaultReplayInfo gpuFaultReplay;
    CUtoolsUvm8EventFatalFaultInfo fatalFault;
    CUtoolsUvm8EventReadDuplicateInfo readDuplicate;
    CUtoolsUvm8EventReadDuplicateInvalidateInfo readDuplicateInvalidate;
    CUtoolsUvm8EventPageSizeChangeInfo pageSizeChange;
    CUtoolsUvm8EventThrashingDetectedInfo thrashing;
    CUtoolsUvm8EventThrottlingStartInfo throttlingStart;
    CUtoolsUvm8EventThrottlingEndInfo throttlingEnd;
    CUtoolsUvm8EventMapRemoteInfo mapRemote;
    CUtoolsUvm8EventEvictionInfo eviction;
} CUtoolsUvm8EventInfo;

// The UVM 8 event control data
typedef struct CUtoolsUvm8EventControlData_st {
   // entries between get_ahead and get_behind are currently being read
    volatile uint32_t get_ahead;
    volatile uint32_t get_behind;
    // entries between put_ahead and put_behind are currently being written
    volatile uint32_t put_ahead;
    volatile uint32_t put_behind;

    // counter of dropped events
    uint64_t dropped[CU_TOOLS_UVM8_NUM_EVENT_TYPES];
} CUtoolsUvm8EventControlData;

/// \brief interface for UVM counters/events
CU_DEFINE_UUID(CU_ETID_ToolsUvm8Events,
    0x8d9cddc, 0x6689, 0x4500, 0xbb, 0x1f, 0x77, 0x59, 0x8a, 0xca, 0x11, 0x33);

typedef struct CUetblToolsUvm8Events_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief Each migration event entry contains the gpu index to/from where data is
    /// migrated. This index maps to a corresponding gpu/cpu uuid in the processorUuidTable.
    /// Using indices saves on the size of each event entry. This API provides the
    /// gpuIndex to gpuUuid relation to the user.
    /// \param session (in) The UVM8 session
    /// \param gpuUuidTable (out) The return value is an array of uuids.
    /// The array index is the corresponding gpuIndex. There can be at max 32 gpus and
    /// 1 CPU associated with UVM, so array size is 33.
    /// \param validCount (out) The system doesn't normally contain 32 GPUs.
    // This field gives the count of entries that are valid in the returned gpuUuidTable.
    CUresult (CUDAAPI *EventGetProcessorUuidTable)(
        CUtoolsUvm8Session session,
        CUuuid *gpuUuidTable,
        uint32_t *validCount);

    /// \brief Check if device supports UVM events
    CUresult (CUDAAPI *DeviceSupportsEvents)(
        CUdevice dev,
        CUtools_uvm_event_support *flag);

    /// \brief Check if UVM 8 is supported by the driver or not
    /// \param supported (out) Returns whether UVM 8 is supported
    /// or not
    CUresult (CUDAAPI *IsUvm8Supported)(
        uint32_t *supported);

    /// \brief Get the file descriptor 
    /// \param fd (out) Returns the file descriptor for UVM 8
    /// or not
    CUresult (CUDAAPI *GetFileDescriptor)(
        CUtoolsUvm8FileDescriptor *fd);

    /// \brief Create a session for UVM 8
    /// \param fd (in) The file descriptor
    /// \param session (out) Returns the handle for created session
    CUresult (CUDAAPI *CreateSession)(
        CUtoolsUvm8FileDescriptor fd,
        CUtoolsUvm8Session *session);

    /// \brief Destroye a session for UVM 8
    /// \param session (in) The session handle
    CUresult (CUDAAPI *DestroySession)(
        CUtoolsUvm8Session session);

    /// \brief Create an event queue for UVM 8
    /// \param sessionHandle (in) The session handle
    /// \param queueHandle (out) Returns the handle of the created queue
    /// \param event_buffer (in) User allocated buffer, must be page-aligned.
    /// Must be large enough to hold at least 'event_buffer_size' events. Gets
    /// pinned until queue is destroyed.
    /// \param event_buffer_size (in) The number of events the buffer can hold
    /// \param event_control (in) User allocated buffer. Must be page-aligned.
    /// Must be large enough to hold 'CUtoolsUvm8EventControlData'. One
    /// should call Uvm8GetEventControlDataSize() to get correct size of 
    /// 'CUtoolsUvm8EventControlData'
    CUresult (CUDAAPI *EventQueueCreate)(
        CUtoolsUvm8Session sessionHandle,
        CUtoolsUvm8EventQueue *queueHandle,
        void *event_buffer,
        size_t event_buffer_size,
        void *event_control);

    /// \brief Set notification threshold in number of events for a given queue for UVM 8
    /// \param queue (in) The queue handle
    /// \param notification_threshold (in) Threshold, in number of events to be set for the queue
    CUresult (CUDAAPI *SetNotificationThreshold)(
        CUtoolsUvm8EventQueue queue,
        size_t notification_threshold);

    /// \brief Get event queue notification handle for UVM 8
    /// \param queue (in) The queue handle
    /// \param handle (out) The event queue notification handle
    CUresult (CUDAAPI *GetEventQueueNotificationHandle)(
        CUtoolsUvm8EventQueue queue,
        CUtoolsUvm8EventQueueNotificationHandle *handle);

    /// \brief Get the size of 'CUtoolsUvm8EventControlData' for UVM 8
    /// \param eventControlDataSize (out) returns the size
    CUresult (CUDAAPI *GetEventControlDataSize)(
        size_t *eventControlDataSize);

    /// \brief Get the size of 'CUtoolsUvmEventInfo' for UVM 8
    /// \param eventEntrySize (out) returns the size
    CUresult (CUDAAPI *GetEventEntrySize)(
        size_t *eventEntrySize);

    /// \brief Destroy the event queue for UVM 8
    /// \param queueHandle (in) the event queue
    CUresult (CUDAAPI *EventQueueDestroy)(
        CUtoolsUvm8EventQueue queueHandle);

    /// \brief Enable the events for the queue for UVM 8
    /// \param queueHandle (in) the event queue
    /// \param events (in) array of event types to enable
    /// \param count (in) the number of elements in events
    CUresult (CUDAAPI *EventQueueEnableEvents)(
        CUtoolsUvm8EventQueue queueHandle,
        CUtools_uvm8_event_type *events,
        size_t count);

    /// \brief Disable the events for the queue for UVM 8
    /// \param queueHandle (in) the event queue
    /// \param events (in) array of event types to enable
    /// \param count (in) the number of elements in events
    CUresult (CUDAAPI *EventQueueDisableEvents)(
        CUtoolsUvm8EventQueue queueHandle,
        CUtools_uvm8_event_type *events,
        size_t count);

    /// \brief Flush completed events to queue for UVM 8
    /// \param sessionHandle (in) the session handle
    CUresult (CUDAAPI *FlushEvents)(
        CUtoolsUvm8Session sessionHandle);

} CUetblToolsUvm8Events;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
