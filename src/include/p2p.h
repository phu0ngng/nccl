/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

#include <cuda.h>

#ifdef MNNVL_SUPPORT
#if CUDART_VERSION < 12030
#include "wizlet.h"
#endif
#endif

typedef union {
  uint64_t data; // Needs to hold a CUmemGenericAllocationHandle for UDS fd support
  CUmemGenericAllocationHandle handle;
} ncclCuDesc;

typedef union {
  // Legacy CUDA IPC
  cudaIpcMemHandle_t devIpc;
  // cuMem API support
  ncclCuDesc cuDesc;
} ncclIpcDesc;

ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, ncclIpcDesc *ipcDesc, void **ptr);
ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc);
ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm *comm, int tpPeer, size_t size, ncclIpcDesc *ipcDesc, void **devMemPtr);

#if CUDART_VERSION >= 11030

#include <cuda.h>
#include "cudawrap.h"

static ncclResult_t ncclP2pHandleType(CUmemAllocationHandleType *type) {
#if CUDART_VERSION >= 12030
  int cudaDev;
  int flag = 0;
  CUdevice currentDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  // Ignore error if CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED is not supported
  (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));;
  *type = flag ? CU_MEM_HANDLE_TYPE_FABRIC : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  return ncclSuccess;
#else
  *type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  return ncclSuccess;
#endif // CUDART_VERSION >= 12030
}
#endif // CUDART_VERSION >= 11030

#endif
