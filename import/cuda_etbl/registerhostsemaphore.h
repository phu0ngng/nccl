/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_register_host_samphore_h__
#define __cuda_etbl_register_host_samphore_h__

#include "cuda_uuid.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


//---------------------------------------------------------------------
//  RegisterHostSemaphore, allow sharing of a external sysmem samphore
//     see //sw/gpgpu/doc/feature_planning/CUDA_4.1/External_syncs.txt
//     for more details.
//---------------------------------------------------------------------

CU_DEFINE_UUID(CU_ETID_RegisterHostSemaphore, /* a95f46a4-36cc-40d4-845c-5afb4bada743 */
    0xa95f46a4, 0x36cc, 0x40d4, 0x84, 0x5c, 0x5a, 0xfb, 0x4b, 0xad, 0xa7, 0x43);

typedef struct {
    /// \param stream is the stream that will do the wait
    /// \param hptr is a pointer that was allocated using cuMemHostAlloc/cuMemAllocHost 
    ///        or registered using cuMemHostRegister
    /// \param payload is the payload to wait for
    CUresult (CUDAAPI *cuStreamAcquireSemaphore)(CUstream stream, void *hptr, unsigned int payload);
    /// \param stream is the stream that will do the release
    /// \param hptr is a pointer that was allocated using cuMemHostAlloc/cuMemAllocHost 
    ///        or registered using cuMemHostRegister
    /// \param payload is the payload to release
    CUresult (CUDAAPI *cuStreamReleaseSemaphore)(CUstream stream, void *hptr, unsigned int payload);
} CUetblRegisterHostSemaphore;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
