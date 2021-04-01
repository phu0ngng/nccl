/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_driver_h__
#define __cuda_etbl_tools_driver_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum CUtools_driver_state_enum {
    CU_TOOLS_DRIVER_STATE_INITIALIZED       = 0,
    CU_TOOLS_DRIVER_STATE_DESTROYED         = 1,
    CU_TOOLS_DRIVER_STATE_NOT_INITIALIZED   = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_DRIVER_STATE_SIZE,
    CU_TOOLS_DRIVER_STATE_FORCE_INT         = 0x7fffffff
} CUtools_driver_state;

typedef struct CUtoolsEnumContextCallbackData_st {
    uint32_t struct_size;
    uint32_t reserved0;
    CUcontext ctx;
    void* reserved1;
} CUtoolsEnumContextCallbackData;

/// \brief Callback signature for EnumerateContext()
typedef void (CUDAAPI *CUtoolsEnumerateContextsCallback)(
    void* pUserData,
    CUtoolsEnumContextCallbackData *contextData);

typedef struct CUtoolsEnumContext_st {
    uint32_t struct_size;
    uint32_t reserved0;
    CUtoolsEnumerateContextsCallback pfnCallback;
    void *pUserData;
} CUtoolsEnumContextData;

/// \brief The driver table provides utility functions that do not belong to specific tables
///f88abdd7-ace7-4aca-bdbc-fb1f1b3fcf33
CU_DEFINE_UUID(CU_ETID_ToolsDriver,
    0xf88abdd7, 0xace7, 0x4aca, 0xbd, 0xbc, 0xfb, 0x1f, 0x1b, 0x3f, 0xcf, 0x33);

typedef struct CUetblToolsDriver_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// Returns branch version, major and minor version, should be used only for mac
    CUresult (CUDAAPI *GetMacBranchVersion)(
        int *branchVersionMajor,
        int *branchVersionMinor);

    /// Returns driver version. This is a wrapper of cuiDriverGetVersion
    CUresult (CUDAAPI *GetDriverVersion)(
        int *driverVersion);

    /// Returns the state of the driver (initialized/destroyed/not initialized)
    CUresult (CUDAAPI *GetDriverState)(
        uint32_t *state);

    //It is *not safe* to call EnumerateContexts from any tools callback 
    //that is issued under a ctxLock. It can be called from the API enter/exit
    //callbacks as they never occur under ctxLock.
    /// Creates contexts already created in the driver
    CUresult (CUDAAPI *EnumerateContexts)(
        CUtoolsEnumContextData *enumCtxData);
} CUetblToolsDriver;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
