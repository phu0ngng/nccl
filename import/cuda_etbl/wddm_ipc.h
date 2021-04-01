/*
* Copyright 1993-2017 by NVIDIA Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to NVIDIA
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of NVIDIA Corporation is prohibited.
*/

#ifndef __cuda_etbl_windows_ipc_h__
#define __cuda_etbl_windows_ipc_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct CUIpcAllocInfoWin32 {
    void  *winObjAttributes; //OBJECT_ATTRIBUTES structure
} CUIpcAllocInfoWin32;

// {38d7f13e-3605-40f6-9128-6034574dfbf2}
CU_DEFINE_UUID(CU_ETID_WddmIpc,
    0x38d7f13e, 0x3605, 0x40f6, 0x91, 0x28, 0x60, 0x34, 0x57, 0x4d, 0xfb, 0xf2);

typedef struct CUetblWddmIpc_st {
    // This export table supports versioning by adding to the end without
    // changing the ETID.  The struct_size field will always be set to the
    // size in bytes of the entire export table structure.
    size_t struct_size;

    // Allocate memory for IPC through WDDM and return an exportable handle
    CUresult (CUDAAPI *cuIpcAllocMemHandle)(size_t allocSize, CUipcMemHandle *ipcHandle, CUdeviceptr *dptr, CUIpcAllocInfoWin32 *pObjAttributes);

    // Using an exported IPC handle open previously allocated memory
    CUresult (CUDAAPI *cuIpcOpenMemHandle)(CUdeviceptr *dptr, const CUipcMemHandle *ipcHandle);

    // Close an allocated handle
    CUresult (CUDAAPI *cuIpcCloseMemhandle)(CUdeviceptr dptr);
} CUetblWddmIpc;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
