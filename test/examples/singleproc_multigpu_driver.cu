/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
//#include "cuda_runtime.h"
#include "nccl.h"

#define CUDACHECK(cmd) do {                         \
  CUresult e = cmd;                                 \
  if( e != CUDA_SUCCESS ) {                         \
    printf("Failed: Cuda error %s:%d %d\n",         \
           __FILE__,__LINE__, e);                   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[])
{
  ncclComm_t comms[4];

  //managing 4 devices
  int nDev = 2;
  int size = 32*1024*1024;
  int devs[4] = { 0, 1, 2, 3 };

  //allocating and initializing device buffers
  CUdeviceptr* sendbuff = (CUdeviceptr*)malloc(nDev * sizeof(CUdeviceptr));
  CUdeviceptr* recvbuff = (CUdeviceptr*)malloc(nDev * sizeof(CUdeviceptr));
  CUstream* s = (CUstream*)malloc(sizeof(CUstream)*nDev);
  CUcontext* context = (CUcontext *)malloc(sizeof(CUcontext)*nDev);

  CUresult err = cuInit(0);

  for (int i = 0; i < nDev; ++i) {
    CUdevice device;

    CUDACHECK(cuDeviceGet(&device, i));
    #if CUDART_VERSION >= 13000
    CUDACHECK(cuCtxCreate(context+i, NULL, 0, device));
    #else
    CUDACHECK(cuCtxCreate(context+i, 0, device));
    #endif
    CUDACHECK(cuCtxSetCurrent(context[i]));
    CUDACHECK(cuMemAlloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cuMemAlloc(recvbuff + i, size * sizeof(float)));
    printf("%d sendbuf %llx recvbuff %llx\n", i, sendbuff[i], recvbuff[i]);
    CUDACHECK(cuMemsetD8(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cuMemsetD8(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cuStreamCreate(s+i, CU_STREAM_NON_BLOCKING));
//    s[i] = NULL;
  }

  CUDACHECK(cuCtxSetCurrent(context[0]));
  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], 128*1024, ncclFloat, ncclSum,
                            comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());

  //synchronize on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cuStreamSynchronize(s[i]));
  }

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cuMemFree(sendbuff[i]));
    CUDACHECK(cuMemFree(recvbuff[i]));
  }

  //destroy NCCL communicator
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);

  printf("Success \n");
  return 0;
}
