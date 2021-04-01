/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_third_party_p2p_h__
#define __cuda_etbl_third_party_p2p_h__

#include "cuda.h"

#include "cuda_uuid.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
//  ThirdPartyP2P - Enable third party device and software access to GPU memory
//------------------------------------------------------------------

CU_DEFINE_UUID(CU_ETID_ThirdPartyP2P, /* 11f70e38-dc8a-46d2-a28f-fffdef89e013 */
               0x11f70e38, 0xdc8a, 0x46d2, 0xa2, 0x8f, 0xff, 0xfd, 0xef, 0x89, 0xe0, 0x13);

typedef struct {
    /**
     * \brief Export a context's device virtual address space for third party access
     *
     * This function gets a pair of tokens (\p p2pToken and \p vaSpaceToken) to be used
     * with the nv-p2p.h Linux kernel interface. \p p2pToken is guaranteed to be consistent
     * between contexts on the same device. \p vaSpaceToken is only valid for the
     * supplied context although several contexts may share a vaSpaceToken.
     *
     * Previously allocated device memory in this context will be exported along with 
     * all future allocations in this context. Only memory allocated through ::cuMemAlloc 
     * will be visible through this interface.
     *
     * \param p2pToken     - Token for use with nv-p2p.h Linux kernel interface
     * \param vaSpaceToken - Token for use with nv-p2p.h Linux kernel interface
     * \param ctx          - Context get tokens for
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_INVALID_CONTEXT,
     * ::CUDA_ERROR_UNKNOWN
     *
     * \sa 
     * ::cuThirdPartyP2PExportCtxNVP2P,
     * ::cuThirdPartyP2PUnexportCtxBAR1,
     * ::cuThirdPartyP2PUnexportCtxNVP2P
     */
    CUresult (CUDAAPI *cuThirdPartyP2PExportCtxBAR1)(unsigned long long *p2pToken, unsigned int *vaSpaceToken, CUcontext ctx);
    CUresult (CUDAAPI *cuThirdPartyP2PExportCtxNVP2P)(unsigned long long *p2pToken, unsigned int *vaSpaceToken, CUcontext ctx);

    /**
     * \brief Unexport a context's device virtual address space for third party access
     * 
     * This function will unexport all memory exported by a previous call to 
     * ::cuThirdPartyP2PExportCtxBAR1. Device resources are refcounted internally so
     * resources associated with p2pToken will only be freed after all exported contexts
     * sharing a device have called ::cuThirdPartyP2PUnexportCtxBAR1.
     *
     * Destroying a context will safely unexport a context's device allocations making this
     * call optional.
     *
     * \param ctx - Context to unexport
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_INVALID_CONTEXT
     *
     * \sa
     * ::cuThirdPartyP2PExportCtxBAR1,
     * ::cuThirdPartyP2PExportCtxNVP2P,
     * ::cuThirdPartyP2PUnexportCtxNVP2P
     * 
     */
    CUresult (CUDAAPI *cuThirdPartyP2PUnexportCtxBAR1)(CUcontext ctx);
    CUresult (CUDAAPI *cuThirdPartyP2PUnexportCtxNVP2P)(CUcontext ctx);
} CUetblThirdPartyP2P;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
