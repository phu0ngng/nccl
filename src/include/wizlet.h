/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _WIZLET_H_
#define _WIZLET_H_

#include <cuda.h>

#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)

/**
* Fabric handle
* \sa MemExportToShareableHandle, MemImportFromShareableHandle
*/
typedef struct CUmemFabricHandle_st {
    uint64_t data[128];
} CUmemFabricHandle;


#endif // _WIZLET_H_
