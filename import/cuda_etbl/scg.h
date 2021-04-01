/*
* Copyright 1993-2016 by NVIDIA Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to NVIDIA
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of NVIDIA Corporation is prohibited.
*/

#ifndef __cuda_etbl_scg_h__
#define __cuda_etbl_scg_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


    CU_DEFINE_UUID(CU_ETID_Scg,
        0x40f58300, 0xe34f, 0x4684, 0x8e, 0xca, 0x1f, 0x2c, 0x33, 0x41, 0x1b, 0x30);

    typedef enum CUscgContextType_enum
    {
        UNKNOWN      = 0x0,
        D3D_DEVICE   = 0x1,
    } CUscgContextType;

    typedef struct CUscgContextCreationInfo_st
    {
        void*              pMasterContext;
        CUscgContextType   masterContextType;
        uint64_t           subcontextId;
    } CUscgContextCreationInfo;


    typedef struct CUetblScg_st {
        // This export table supports versioning by adding to the end without changing
        // the ETID.  The struct_size field will always be set to the size in bytes of
        // the entire export table structure.
        size_t struct_size;

        /// \brief Create a subcontext suitable for SCG uses. This Subcontext will use
        ///        the device at the same time as the graphics API linked to it.
        ///        Depending on the architecture some CUDA features might not be supported
        ///        in this mode.
        /// \param contextCreationInfo contains information on how to create the subcontex.
        /// \param contextCreationInfoSize must be sizeof(CUscgContextCreationInfo).
        /// \param pCtx The context created by this call
        CUresult(CUDAAPI *ScgCreateSubContext)(
            CUscgContextCreationInfo* contextCreationInfo,
            uint32_t contextCreationInfoSize,
            CUcontext* pCtx);

    } CUetblScg;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
