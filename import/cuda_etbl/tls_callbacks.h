/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tls_callbacks_h__
#define __cuda_etbl_tls_callbacks_h__

#include "cuda_uuid.h"
#include "cuda_stdint.h"

#if defined(_WIN32) && !defined(NV_MODS)
typedef struct CUtlsCallbackHandle_st CUtlsCallbackHandle;

enum CUthreadReason_enum {
    CU_THREAD_FINISHED,
};

typedef enum CUthreadReason_enum CUthreadReason;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CU_DEFINE_UUID(CU_ETID_TlsCallbackInterface,
    0xf4cb5b19, 0x7dd6, 0x4a02, 0xac, 0xc5, 0x1d, 0x29, 0xce, 0xa6, 0x31, 0xae);

typedef struct CUetblTlsCallback_st {
    size_t struct_size;

    CUresult (CUDAAPI *RegisterCallback)(
        CUtlsCallbackHandle **handle,
        void (*callback)(CUthreadReason, void *),
        void *user_data);

    CUresult (CUDAAPI *UnregisterCallback)(
        CUtlsCallbackHandle *handle,
        void **user_data);
} CUetblTlsCallback;

#ifdef __cplusplus
}
#endif
#endif
#endif
