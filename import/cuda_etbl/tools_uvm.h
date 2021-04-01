/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_uvm_h__
#define __cuda_etbl_tools_uvm_h__

#include "cuda_uuid.h"
#include "cuda_stdint.h"
#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// UVM event buffer must be 64-bit aligned
#define CU_TOOLS_UVM_EVENT_BUFFER_ALIGNMENT 8

typedef struct CUtoolsUvmSession_st* CUtoolsUvmSession;
typedef void* CUtoolsUvmUPtr;
typedef uint64_t CUtoolsUvmQueue;
typedef uint64_t CUtoolsUvmNotificationEventHandle;
typedef uint8_t CUtoolsUvmEventType;

typedef enum CUtools_uvm_counter_scope_enum
{
    CU_TOOLS_UVM_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE   = 1,
    CU_TOOLS_UVM_COUNTER_SCOPE_PROCESS_ALL_DEVICES     = 2,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM_COUNTER_SCOPE_SIZE,
    CU_TOOLS_UVM_COUNTER_SCOPE_FORCE_INT               = 0x7fffffff
} CUtools_uvm_counter_scope;

typedef enum CUtools_uvm_counter_name_enum
{
    CU_TOOLS_UVM_COUNTER_NAME_BYTES_TRANSFER_HTOD   = 1,
    CU_TOOLS_UVM_COUNTER_NAME_BYTES_TRANSFER_DTOH   = 2,
    CU_TOOLS_UVM_COUNTER_NAME_CPU_PAGE_FAULT_COUNT  = 3,
    CU_TOOLS_UVM_COUNTER_NAME_PREFETCH_BYTES_TRANSFER_HTOD = 4,
    CU_TOOLS_UVM_COUNTER_NAME_PREFETCH_BYTES_TRANSFER_DTOH = 5,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM_COUNTER_NAME_SIZE,
    CU_TOOLS_UVM_COUNTER_NAME_FORCE_INT             = 0x7fffffff
} CUtools_uvm_counter_name;

typedef enum CUtools_uvm_event_memory_access_type_enum 
{
    CU_TOOLS_UVM_EVENT_MEMORY_ACCESS_TYPE_INVALID   = 0,
    CU_TOOLS_UVM_EVENT_MEMORY_ACCESS_TYPE_READ      = 1,
    CU_TOOLS_UVM_EVENT_MEMORY_ACCESS_TYPE_WRITE     = 2,
    CU_TOOLS_UVM_EVENT_MEMORY_ACCESS_TYPE_ATOMIC    = 3,
    CU_TOOLS_UVM_EVENT_MEMORY_ACCESS_TYPE_PREFETCH  = 4,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM_EVENT_MEMORY_ACCESS_TYPE_SIZE,
    CU_TOOLS_UVM_EVENT_MEMORY_ACCESS_TYPE_FORCE_INT = 0x7fffffff
} CUtools_uvm_event_memory_access_type;

typedef enum CUtools_uvm_event_type_enum
{
    CU_TOOLS_UVM_EVENT_TYPE_INVALID             = 0,
    CU_TOOLS_UVM_EVENT_TYPE_MEMORY_VIOLATION    = 1,
    CU_TOOLS_UVM_EVENT_TYPE_MIGRATION           = 2,
    CU_TOOLS_UVM_EVENT_TYPE_GPU_ALLOC           = 3,
    CU_TOOLS_UVM_EVENT_TYPE_CPU_ALLOC           = 4,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM_EVENT_TYPE_SIZE,
    CU_TOOLS_UVM_EVENT_TYPE_FORCE_INT           = 0x7fffffff
} CUtools_uvm_event_type;

typedef enum CUtools_uvm_event_migration_direction_enum
{
    CU_TOOLS_UVM_EVENT_MIGRATION_DIRECTION_INVALID    = 0,
    CU_TOOLS_UVM_EVENT_MIGRATION_DIRECTION_CPU_TO_GPU = 1,
    CU_TOOLS_UVM_EVENT_MIGRATION_DIRECTION_GPU_TO_CPU = 2,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM_EVENT_MIGRATION_DIRECTION_SIZE,
    CU_TOOLS_UVM_EVENT_MIGRATION_DIRECTION_FORCE_INT  = 0x7fffffff
} CUtools_uvm_event_migration_direction;

typedef enum CUtools_uvm_event_timestamp_type_enum
{
    CU_TOOLS_UVM_EVENT_TIMESTAMP_TYPE_INVALID               = 0,
    CU_TOOLS_UVM_EVENT_TIMESTAMP_TYPE_WIN32_QPC             = 1,
    CU_TOOLS_UVM_EVENT_TIMESTAMP_TYPE_POSIX_CLOCK_GET_TIME  = 2,
    CU_TOOLS_UVM_EVENT_TIMESTAMP_TYPE_AUTO                  = 3,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM_EVENT_TIMESTAMP_TYPE_SIZE,
    CU_TOOLS_UVM_EVENT_TIMESTAMP_TYPE_FORCE_INT             = 0x7fffffff
} CUtools_uvm_event_timestamp_type;

typedef enum CUtools_uvm_event_support_enum
{
    CU_TOOLS_UVM_EVENT_NOT_SUPPORTED                 = 0,
    CU_TOOLS_UVM_EVENT_NOT_SUPPORTED_ON_DEVICE       = 1,
    CU_TOOLS_UVM_EVENT_MANAGED_FORCED_TO_ZERO_COPY   = 2,
    CU_TOOLS_UVM_EVENT_SUPPORTED                     = 3,
    // --- always add new constants above this line ---
    CU_TOOLS_UVM_EVENT_SIZE,
    CU_TOOLS_UVM_EVENT_FORCE_INT                     = 0x7fffffff
} CUtools_uvm_event_support;

typedef struct CUtoolsUvmCounterConfig_st
{
    uint32_t struct_size;
    uint32_t reserved0;
    uint32_t scope;         // Use values from CUtools_uvm_counter_scope
    uint32_t name;          // Use values from CUtools_uvm_counter_name
    CUdevice dev;           // size == int32_t
    uint32_t state;
} CUtoolsUvmCounterConfig;

#define CU_TOOLS_UVM_EVENT_FETCH_TIMEOUT_IMMEDIATE 0

typedef struct CUtoolsUvmUserMemObjOwnership_st {
    uint32_t struct_size;
    uint32_t ownerType;             // one of CUtoolsUvmLiteOwnerType
    CUtoolsStreamHandle stream;     // stream, if ownerType == CU_TOOLS_UVM_LITE_OWNER_TYPE_ONE_STREAM
    void* rsvd0;
} CUtoolsUvmMemObjOwnership;

// Information associated with a memory violation event
typedef struct CUtoolsUvmEventMemoryViolationInfo_st
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
    uint64_t timeStamp;          // time when violation occurred
    uint32_t pid;                // process id causing violation
    uint32_t threadId;           // thread id causing violation
} CUtoolsUvmEventMemoryViolationInfo;

// Information associated with a migration event
typedef struct CUtoolsUvmEventMigrationInfo_st
{
    // eventType has to be the 1st argument of this structure.
    uint8_t eventType;
    uint8_t direction;    // direction of migration (CUtools_uvm_event_migration_direction)
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
    uint64_t beginTimeStamp;                   // migration start time stamp
    uint64_t endTimeStamp;                     // migration end time stamp
    uint64_t streamId;                         // stream causing the migration
} CUtoolsUvmEventMigrationInfo;

typedef union CUtoolsUvmEventInfo_union
{
    uint8_t eventType;
    CUtoolsUvmEventMemoryViolationInfo memoryViolation;
    CUtoolsUvmEventMigrationInfo migration;
} CUtoolsUvmEventInfo;

/// \brief interface for UVM counters/events
CU_DEFINE_UUID(CU_ETID_ToolsUvm,
    0x423f6832, 0x96be, 0x4552, 0x99, 0x50, 0x83, 0x8b, 0x1c, 0x3d, 0xe1, 0x55);

typedef struct CUetblToolsUvm_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief Creates a handle for the session
    CUresult (CUDAAPI *CreateSession)(
        uint32_t pid,
        CUtoolsUvmSession *session);

    /// \brief Destroys the session
    CUresult (CUDAAPI *DestroySession)(
        CUtoolsUvmSession session);

    /// \brief Returns handle for the given scope/counter combination
    CUresult (CUDAAPI *GetCountersHandle)(
        CUtoolsUvmSession session,
        CUtools_uvm_counter_scope scope,
        CUtools_uvm_counter_name counterName,
        CUdevice dev,
        CUtoolsUvmUPtr *pCounterHandle);

    /// \brief Returns the counter value
    CUresult (CUDAAPI *GetCounterVal)(
        CUtoolsUvmSession session,
        const CUtoolsUvmUPtr *counterHandleArray,
        size_t handleCount,
        uint64_t *pCounterVal);

    /// \brief Enables/disables the counters
    CUresult (CUDAAPI *CountersEnable)(
        CUtoolsUvmSession session,
        const CUtoolsUvmCounterConfig *config,
        size_t count);

    /// \brief Check if device supports UVM counters
    CUresult (CUDAAPI *DeviceSupportsCounter)(
        uint32_t *pFlag,
        CUdevice dev);

    /// \brief Query memobj residency from the current usermode driver state.
    /// Can be used by the debugger to determine how to access the memobj (host ptr or BAR1) while at a GPU breakpoint.
    CUresult (CUDAAPI *QueryMemObjOwnership)(
        CUtoolsMemObjHandle hMemObj,
        CUtoolsUvmMemObjOwnership *ownership);

    /// \brief Create an event queue of the given size.
    /// \param queueHandle (out) Handle to the created queue
    /// \param queueSize Size of the event queue buffer in units of
    /// sizeof of biggest event.
    /// \param notificationCount Number of entries after which the user
    /// should be notified that there are events to fetch. User is notified
    /// when queueEntries >= notification count.
    CUresult (CUDAAPI *EventQueueCreate)(
        CUtoolsUvmSession sessionHandle,
        CUtoolsUvmQueue *queueHandle,
        uint64_t queueSize,
        uint64_t notificationCount,
        CUtools_uvm_event_timestamp_type timeStampType);

    /// \brief Free all the resources associated with the queue.
    CUresult (CUDAAPI *EventQueueDestroy)(
        CUtoolsUvmSession sessionHandle,
        CUtoolsUvmQueue queueHandle);

    /// \brief Enable a list of events in the event queue.
    /// All events are disabled by default when a queue is created.
    /// \param events (in) Array of events to be enabled
    /// \param count Number of events in the array
    CUresult (CUDAAPI *EventEnable)(
        CUtoolsUvmSession sessionHandle,
        CUtoolsUvmQueue queueHandle,
        CUtools_uvm_event_type *events,
        size_t count);

    /// \brief Disable a list of events in the queue.
    /// \param events (in) Array of events to be disabled
    /// \param count Number of events in the array
    CUresult (CUDAAPI *EventDisable)(
        CUtoolsUvmSession sessionHandle,
        CUtoolsUvmQueue queueHandle,
        CUtools_uvm_event_type *events,
        size_t count);

    /// \brief User is notified when queueEntries >= notification count.
    /// This call does a blocking wait for this notification. It returns when
    /// at least one of the queue handles has events to be fetched or if it timeouts.
    /// \param queueHandleArray (in) Array of queue handles
    /// \param arraySize (in) Number of handles in array
    /// \param timeout Timeout in msec
    /// \param pNotificationFlags (out) If a particular queue handle in the input array
    /// is notified then the respective bit flag is set in pNotificationFlags.
    CUresult (CUDAAPI *EventWaitOnQueueHandles)(
        CUtoolsUvmQueue *queueHandleArray,
        uint32_t arraySize,
        uint64_t timeout,
        uint32_t *pNotificationFlags);

    /// \brief User is notified when queueEntries >= notification count.
    /// The user can directly get the queue notification handles rather than using
    /// a UVM API to wait on queue handles. This helps the user to wait on other
    /// objects (apart from queue notification) along with queue notification
    /// handles in the same thread. The user can safely use this call along with the
    /// library supported wait call EventWaitOnQueueHandles.
    /// \param queueHandleArray (in) Array of queue handles
    /// \param arraySize (in) Number of handles in the array
    /// \param notificationHandleArray (out)
    ///     Windows: Output of this call contains an array of 'windows event
    ///              handles' corresponding to the queue handles passes as input.
    ///     Linux: All queues belonging to the same process share the same
    ///            file descriptor(fd) for notification. If the user chooses to use
    ///            EventGetNotificationHandles then he should check all queues
    ///            for new events (by calling EventFetch) when notified on the fd.
    CUresult (CUDAAPI *EventGetNotificationHandles)(
        CUtoolsUvmQueue *queueHandleArray,
        uint32_t arraySize,
        CUtoolsUvmNotificationEventHandle **notificationHandleArray);

    /// \brief Fetch the queue entries in a user buffer
    /// \param pBuffer (out) Pointer to the buffer where the API will copy the events.
    /// User shall ensure the size is enough and buffer is 64 bit aligned.
    /// \param nEntries (in/out) It provides the maximum number of entries that will
    /// be fetched from the queue. If this number is larger than the size of the queue
    /// it will be internally capped to that value. As output it returns the actual
    /// number of entries copies to the buffer.
    CUresult (CUDAAPI *EventFetch)(
        CUtoolsUvmSession sessionHandle,
        CUtoolsUvmQueue queueHandle,
        void *pBuffer,
        uint64_t *nEntries);

    /// \brief Drop all event entries from the queue.
    CUresult (CUDAAPI *EventSkipAll)(
        CUtoolsUvmSession sessionHandle,
        CUtoolsUvmQueue queueHandle);

    /// \brief Return the type of timestamp used in an event entry for a given queue.
    /// \param timeStampType (out) type of time stamp used in event entry.
    CUresult (CUDAAPI *EventQueryTimeStampType)(
        CUtoolsUvmSession sessionHandle,
        CUtoolsUvmQueue queueHandle,
        CUtools_uvm_event_timestamp_type *timeStampType);

    /// \brief Each migration event entry contains the gpu index to/from where data is
    /// migrated. This index maps to a corresponding gpu uuid in the gpuUuidTable.
    /// Using indices saves on the size of each event entry. This API provides the
    /// gpuIndex to gpuUuid relation to the user.
    /// \param gpuUuidTable (out) The return value is an array of uuid’s.
    /// The array index is the corresponding gpuIndex. There can be at max 32 gpus
    /// associated with UVM, so array size is 32.
    /// \param validCount (out) The system doesn't normally contain 32 GPUs.
    // This field gives the count of entries that are valid in the returned gpuUuidTable.
    CUresult (CUDAAPI *EventGetGpuUuidTable)(
        CUuuid *gpuUuidTable,
        uint32_t *validCount);

    /// \brief Check if device supports UVM events
    CUresult (CUDAAPI *DeviceSupportsEvent)(
        CUdevice dev,
        CUtools_uvm_event_support *flag);

    /// Prevent a zero-copy memobj in sysmem from getting unmapped
    CUresult (CUDAAPI *MemobjForceAlwaysHostAccessible)(void);

    /// \brief Return UVM event maximum size. This is to be used prior to
    /// EventFetchUnpacked by the caller to find out the buffer size needed.
    CUresult (CUDAAPI *EventGetMaximumSize)(uint32_t *pSize);

    /// \brief Return an event size for the current driver. This is to be used by
    /// tools to ensure they're compatible with this driver version. If the type
    /// doesn't have a UVM event associated, this function returns CUDA_SUCCESS
    /// with an output size of 0. If eventType is unknown by the driver, this
    /// function returns CUDA_ERROR_INVALID_VALUE and the output size is unchanged.
    CUresult (CUDAAPI *EventTypeGetSize)(uint8_t eventType, uint32_t *pSize);

    /// \brief Fetch the queue entries in a user buffer
    /// \param pBuffer (out) Pointer to the buffer where the API will copy the events.
    /// User shall ensure the size is enough (by a prior call to EventGetMaximumSize)
    /// and buffer is 64 bit aligned. The returned value is not necessarily an array
    /// of structs with sizeof(CUtoolsUvmEventInfo) since tools and driver may have different
    /// versions of that union, resulting in different sizes. Tools have to ensure that
    /// sizeof(CUtoolsUvmEventInfo) is the same as the value returned by EventGetMaximumSize
    /// if they want to use pBuffer as a C array.
    /// \param nEntries (in/out) It provides the maximum number of entries that will
    /// be fetched from the queue. If this number is larger than the size of the queue
    /// it will be internally capped to that value. As output it returns the actual
    /// number of entries copies to the buffer.
    CUresult (CUDAAPI *EventFetchUnpacked)(
        CUtoolsUvmSession sessionHandle,
        CUtoolsUvmQueue queueHandle,
        CUtoolsUvmEventInfo *pBuffer,
        uint64_t *nEntries);

} CUetblToolsUvm;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
