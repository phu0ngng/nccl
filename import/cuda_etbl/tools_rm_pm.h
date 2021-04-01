/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_rm_pm_h__
#define __cuda_etbl_tools_rm_pm_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// \brief EXPERIMENTAL interface for acquiring the RM's PM reservation
CU_DEFINE_UUID(CU_ETID_ToolsRmPm,
    0x15212f7c, 0x6376, 0x4422, 0x9c, 0x3, 0xc6, 0x29, 0xe4, 0x75, 0x93, 0xd3);

typedef struct CUetblToolsRmPm_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief (Experimental) Acquire the RM's PM reservation
    CUresult (CUDAAPI *Reserve)(
        uint32_t rmGpuId,
        uint32_t rmClient,
        uint32_t rmSubDevice);

    /// \brief (Experimental) Release the RM's PM reservation
    CUresult (CUDAAPI *Release)(
        uint32_t rmGpuId,
        uint32_t rmClient,
        uint32_t rmSubDevice);

} CUetblToolsRmPm;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
