/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_perfmon_h__
#define __cuda_etbl_tools_perfmon_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#include "stdio.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// \brief Perfmon provides functions for internal tracking of the PMHW and its counters.
CU_DEFINE_UUID(CU_ETID_ToolsPerfmon,
    0x3aba02f8, 0xd4c9, 0x49c9, 0xba, 0x41, 0x2f, 0x80, 0x82, 0xaf, 0x15, 0xac);

typedef struct CUtoolsPerfmon_st *CUtoolsPerfmonHandle;
typedef uint32_t CUtools_perfmon_counter_id;

typedef struct CUpmCounter_st
{
    // Name of the counter
    const char* name;
    
    void *reserved0;
    // TODO: shall we expose additional information???
    //       For example: unit, group, ...
} CUpmCounter;

typedef struct CUetblPerfmon_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief (Experimental) Reserve the HWPM unit from a context.
    CUresult (CUDAAPI *ReservePmHw)(
        CUcontext ctx);
        
    /// \brief (Experimental) Release the HWPM unit from a context.
    CUresult (CUDAAPI *ReleasePmHw)(
        CUcontext ctx);
        
    /// \brief (Experimental) Query the available counters from a context.
    CUresult (CUDAAPI *QueryPmCountersFromContext)(
        CUcontext ctx,
        uint32_t *numCounters,
        struct CUpmCounter_st *counters);
    
    /// \brief (Experimental) Query the available counters from a device.
    CUresult (CUDAAPI *QueryPmCounters)(
        CUdevice dev,
        uint32_t *numCounters,
        struct CUpmCounter_st *counters);
    
    /// \brief (Experimental) Create PM stucture.
    CUresult (CUDAAPI *CreatePmStructure)(
        CUcontext ctx,
        CUtoolsPerfmonHandle *pmHandle,
        uint32_t numCounters,
        struct CUpmCounter_st *counters);

    /// \brief (Experimental) Free PM stucture.
    CUresult (CUDAAPI *FreePmStructure)(
        CUtoolsPerfmonHandle *pmHandle);

    /// \brief (Experimental) Update the device to use the PM stucture.
    CUresult (CUDAAPI *UsePmStructure)(
        CUcontext ctx,
        CUtoolsPerfmonHandle pmHandle);

    /// \brief (Experimental) Update the device to use the PM stucture.
    CUresult (CUDAAPI *ResetCounters)(
        CUcontext ctx,
        CUtoolsPerfmonHandle pmHandle);

    /// \brief (Experimental) Sample counters.
    CUresult (CUDAAPI *SampleCounters)(
        CUcontext ctx,
        CUtoolsPerfmonHandle pmHandle,
        uint32_t *values);

} CUetblToolsPerfmon;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
