/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_rcuda_h__
#define __cuda_etbl_rcuda_h__

#include "cuda_uuid.h"
#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
// Remote Cuda API Interfaces
//------------------------------------------------------------------

CU_DEFINE_UUID(CU_ETID_RCudaInterface,
    0xebcbd296, 0xbe2a, 0x11e0, 0xb7, 0x04, 0x5a, 0xf5, 0x48, 0x24, 0x01, 0x9b);

/**
 * enumeration that capures all the routines that can be overwritten
 * by a client. The type the function required is in comments. When calling
 * RCudaSetIOCallBack, the the type of the 'fun' argument must match this type.
 * __cdecl is required on windows, as the standard library functions use
 * the __cdecl calling convention
 */
typedef enum {
    CUOS_IO_SET_STDOUT,  /* FILE *(__cdecl*)(void); */
    CUOS_IO_SET_STDERR,  /* FILE *(__cdecl*)(void); */
    CUOS_IO_SET_PRINTF,  /* int (__cdecl*)(const char*, ...); */
    CUOS_IO_SET_FFLUSH,  /* int (__cdecl*)(FILE*); */
    CUOS_IO_SET_FPRINTF  /* int (__cdecl*)(FILE*, const char*, ...); */
} cuosIoId_t;

typedef struct CUetblRCudaInterface_st {
    /* Size of this structure */
    size_t size;
    /* Set the callback function identified by 'id' to 'fun'
     * - returns CUDA_ERROR_INVALID_VALUE for invalid parameters
     */
    CUresult (*RCudaSetIOCallBack)(cuosIoId_t id, void *fun);
} CUetblRCudaInterface;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif //__cuda_etbl_rcuda_h__
