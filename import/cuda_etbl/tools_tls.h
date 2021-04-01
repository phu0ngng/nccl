/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_tls_h__
#define __cuda_etbl_tools_tls_h__

#include "cuda.h"

#include "cuda_uuid.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


/// PushBufferHal provides functions for push-buffer manipulation.
CU_DEFINE_UUID(CU_ETID_ToolsTls,
    0x815ad842, 0xf623, 0x47cb, 0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc);

typedef struct CUetblToolsTls_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief Equivalent of cuCtxSetCurrent
    /// Should only be used in the following pattern:
    /// CtxGetCurrent(&oldCtx)
    /// CtxSetCurrent(newCtx)
    /// ...do stuff on newCtx
    /// CtxSetCurrent(oldCtx)
    CUresult (CUDAAPI *CtxSetCurrent)(
        CUcontext ctx);

    /// \brief Equivalent of cuCtxGetCurrent
    /// Should only be used in the following pattern:
    /// CtxGetCurrent(&oldCtx)
    /// CtxSetCurrent(newCtx)
    /// ...do stuff on newCtx
    /// CtxSetCurrent(oldCtx)
    CUresult (CUDAAPI *CtxGetCurrent)(
        CUcontext *pCtx);

} CUetblToolsTls;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
