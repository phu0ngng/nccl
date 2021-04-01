/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_mem_host_register_h__
#define __cuda_etbl_mem_host_register_h__

#include "cuda_uuid.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


//------------------------------------------------------------------
//  MemHostRegister, register VA range for access by GPU
//------------------------------------------------------------------

CU_DEFINE_UUID(CU_ETID_MemHostRegister, /* a95f46a4-36cc-40d4-845c-5afb4bada742 */
    0xa95f46a4, 0x36cc, 0x40d4, 0x84, 0x5c, 0x5a, 0xfb, 0x4b, 0xad, 0xa7, 0x42);

typedef struct {
    /// \param p base pointer of address range
    /// \param bytes number of bytes in address range
    /// \param Flags flags word
    /// \return CUDA_SUCCESS if
    CUresult (CUDAAPI *cuMemHostRegister)(void *p, size_t bytes, unsigned int Flags);
    CUresult (CUDAAPI *cuMemHostUnregister)(void *p);
} CUetblMemHostRegister;



#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
