/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _API_VISIBLE_TEST_H_
#define _API_VISIBLE_TEST_H_

#include "cuda.h"
#include "cuda_uuid.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

// Possible permissions an api user can have for memory.
// These can be OR'd together to create a full permissions flag.
#define GET_ADDRESS_RANGE 0x01
#define FREE              0x02
#define MEMCPY            0x04

// {4B6734F1-1A0C-4eef-AC01-9E9B7D1A2F6E}
CU_DEFINE_UUID(CU_ETID_apiVisibleTest, 
0x4b6734f1, 0x1a0c, 0x4eef, 0xac, 0x1, 0x9e, 0x9b, 0x7d, 0x1a, 0x2f, 0x6e);

// A structure that contains device pointers. This can be filled with functions like getFunctionDevicePtrInfo
// and getStreamDevicePtrInfo, in whch case [ptrs] contains [numPtrs] device pointers. The expected permissions
// for each pointer are stored in the corresponding element of [memPermissions]. The caller is responsible
// for calling freeDevicePtrInfo to free all used memory.
typedef struct {

    unsigned int numPtrs;
    unsigned int capacity;
    uintptr_t *ptrs;
    int *permissions;

} CUdevicePtrInfo;

// The export table containing functions needed for the api visibility test.
typedef struct {

    CUresult (*getFunctionDevicePtrInfo) (CUfunction fn,   CUdevicePtrInfo* info);
    CUresult (*getStreamDevicePtrInfo)   (CUstream stream, CUdevicePtrInfo* info);
    CUresult (*getModuleDevicePtrInfo)   (CUmodule mod,    CUdevicePtrInfo* info);
    CUresult (*freeDevicePtrInfo)        (CUdevicePtrInfo* info);

} CUetblApiVisibleTest;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _API_VISIBLE_TEST_H_
