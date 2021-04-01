/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_task_h__
#define __cuda_etbl_tools_task_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// \brief Task provides functions for internal tracking of tasks.
CU_DEFINE_UUID(CU_ETID_ToolsTask,
    0x31487d12, 0x33ee, 0x4457, 0x97, 0x42, 0xb8, 0x44, 0x7b, 0x7a, 0x23, 0x64);

typedef enum CUtools_task_status_enum {
    CU_TOOLS_TASK_STATUS_INVALID  = 0,
    CU_TOOLS_TASK_STATUS_QUEUED   = 1,
    CU_TOOLS_TASK_STATUS_RUNNING  = 2,
    CU_TOOLS_TASK_STATUS_COMPLETE = 3,
    // --- always add new constants to the end here ---
    CU_TOOLS_TASK_STATUS_SIZE,
    CU_TOOLS_TASK_STATUS_FORCE_INT                     = 0x7fffffff
} CUtools_task_status;

typedef enum CUtools_task_profile_info_enum {
    CU_TOOLS_TASK_PROFILE_INFO_INVALID  = 0,
    CU_TOOLS_TASK_PROFILE_INFO_ENTRY    = 1,
    CU_TOOLS_TASK_PROFILE_INFO_START    = 2,
    CU_TOOLS_TASK_PROFILE_INFO_END      = 3,
    CU_TOOLS_TASK_PROFILE_INFO_SUBMIT   = 4,
    // --- always add new constants to the end here ---
    CU_TOOLS_TASK_PROFILE_INFO_SIZE,
    CU_TOOLS_TASK_PROFILE_INFO_FORCE_INT               = 0x7fffffff
} CUtools_task_profile_info;

typedef enum CUtools_task_create_flags_enum {
    CU_TOOLS_TASK_CREATE_FLAGS_NONE    = 0,
    CU_TOOLS_TASK_CREATE_FLAGS_PROFILE = 1,
    CU_TOOLS_TASK_CREATE_FLAGS_SUBMIT_TIME_NOT_REQD = 1<<1,
    CU_TOOLS_TASK_CREATE_FLAGS_MARKER_STATUS_CHECK_NOT_REQD = 1<<2,
    // --- always add new constants to the end here ---
    CU_TOOLS_TASK_CREATE_FLAGS_LAST = 1<<3,
    CU_TOOLS_TASK_CREATE_FLAGS_SIZE = CU_TOOLS_TASK_CREATE_FLAGS_LAST,
    CU_TOOLS_TASK_CREATE_FLAGS_FORCE_INT               = 0x7fffffff
} CUtools_task_create_flags;

typedef struct CUetblTask_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief Create a task for the context provided.
    CUresult (CUDAAPI *Create)(
        CUtoolsTaskHandle *phTask,
        CUcontext ctx,
        uint32_t flags);

    /// \brief Destroy a task.
    CUresult (CUDAAPI *Destroy)(
        CUtoolsTaskHandle hTask);

    /// \brief Retrieve the status of the task provided.
    /// This function may only be called on the CUcontext's thread.
    CUresult (CUDAAPI *GetStatus)(
        CUtoolsTaskHandle hTask,
        CUtools_task_status *taskStatus);

    /// \brief Query the profile information associated with a task.
    CUresult (CUDAAPI *GetProfileInfo)(
        CUtoolsTaskHandle hTask,
        CUtools_task_profile_info info,
        uint64_t *data);

    /// \brief Push the start of the task to channel.
    CUresult (CUDAAPI *PushStart)(
        CUtoolsTaskHandle hTask,
        CUtools_engine_type engineType,
        CUtoolsNvCurrent *nvCurrent,
        CUtoolsChannelHandle hChannel);

    /// \brief Push the end of the task to channel.
    CUresult (CUDAAPI *PushEnd)(
        CUtoolsTaskHandle hTask,
        CUtools_engine_type engineType,
        CUtoolsNvCurrent *nvCurrent);

    /// \brief Get the execution status of the task based on directly
    /// reading the task's semaphore status.  This function is free-threaded.
    CUresult (CUDAAPI *GetSemaphoreStatus)(
        CUtoolsTaskHandle hTask,
        CUtools_task_status *taskStatus);

    /// \brief Set the marker for the task
    CUresult (CUDAAPI *SetMarker)(
        CUtoolsTaskHandle hTask);

} CUetblToolsTask;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
