/*************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "common_coll.h"

ncclResult_t ncclLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *paramsList, int* cudaDevs, int numDevices, int cgMode) {
#if __CUDACC_VER_MAJOR__ >= 9
  if (cgMode & 0x01) {
    CUDACHECK(cudaLaunchCooperativeKernelMultiDevice(paramsList, numDevices,
          // These flags are to reduce the latency of using this API
          cudaCooperativeLaunchMultiDeviceNoPreSync|cudaCooperativeLaunchMultiDeviceNoPostSync));
    return ncclSuccess;
  }
#endif
  int savedDev;
  CUDACHECK(cudaGetDevice(&savedDev));
  for (int i = 0; i < numDevices; i++) {
    struct cudaLaunchParams* params = paramsList+i;
    CUDACHECK(cudaSetDevice(cudaDevs[i]));
    CUDACHECK(cudaLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, params->stream));
  }
  CUDACHECK(cudaSetDevice(savedDev));
  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierCheckin(ncclComm_t comm) {
  if (comm->nRanks == 1) return ncclSuccess;
  /* Setup launch params */
  struct cudaLaunchParams* params = comm->intraParams+comm->intraRank;
  switch (comm->collNThreads) {
    case 64 :
    case 65 :
     params->func = (void*)ncclKernel64; break;
    case 129 :
     params->func = (void*)ncclKernel128; break;
    case 257 :
     params->func = (void*)ncclKernel256; break;
    case 513 :
     params->func = (void*)ncclKernel512; break;
    default:
     WARN("Invalid nthread count %d", comm->collNThreads);
     return ncclInternalError;
  }
  params->blockDim.x = comm->collNThreads; params->blockDim.y = params->blockDim.z = 1;
  params->gridDim.x = comm->collNBlocks; params->gridDim.y = params->gridDim.z = 1;
  params->args = &comm->argsptr;
  params->sharedMem = sizeof(struct ncclComm);
  params->stream = comm->ncclStream;

  if (comm->launchMode == ncclComm::GROUP) {
    // Enqueue stream dependency
    CUDACHECK(cudaEventRecord(comm->doneEvent, comm->userStream));
    CUDACHECK(cudaStreamWaitEvent(comm->ncclStream, comm->doneEvent, 0));
  } else {
    if (comm->userStream != comm->ncclStream) {
      CUDACHECK(cudaStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
    }
  }
  // Notify I'm ready
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  int val = *ptr;
  bool done = false;
  while (done == false) {
    if (val >= comm->intraRanks) {
      WARN("Trying to launch too many collectives");
      return ncclInvalidUsage;
    }
    if (val+1 == comm->intraRanks) {
      if (comm->launchMode == ncclComm::GROUP) {
        // I'm the last. Launch all operations.
        NCCLCHECK(ncclLaunchCooperativeKernelMultiDevice(comm->intraParams, comm->intraCudaDevs, comm->intraRanks, *comm->intraCGMode));
      }
      // Reset the barrier.
      comm->intraBarrier[comm->intraPhase^1] = 0;
    }
    done = __sync_bool_compare_and_swap(ptr, val, val+1);
    val++;
  }
  return ncclSuccess;
}
ncclResult_t ncclCpuBarrierWait(ncclComm_t comm) {
  if (comm->nRanks == 1) return ncclSuccess;
  // We can't print the CG mode before the first barrier happened.
  if (comm->rank == 0 && *comm->intraCGMode & 0x10) {
    *comm->intraCGMode ^= 0x10;
    INFO("Launch mode %s%s", comm->launchMode == ncclComm::GROUP ? "Group" : "Parallel", *comm->intraCGMode ? "/CGMD" : "" );
  }
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  while (*ptr < comm->intraRanks) pthread_yield();
  comm->intraPhase ^= 1;
  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(cudaEventRecord(comm->doneEvent, comm->ncclStream));
    CUDACHECK(cudaStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
  } else {
    struct cudaLaunchParams *params = comm->intraParams+comm->intraRank;
    CUDACHECK(cudaLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, comm->userStream));
    CUDACHECK(cudaEventRecord(comm->doneEvent, comm->userStream));
    comm->ncclStream = comm->userStream;
  }
  comm->args.nColls = 0;
  comm->args.startColl = comm->collFifoTail;
  NCCLCHECK(transportStartProxies(comm));
  return ncclSuccess;
}

ncclResult_t ncclEnqueueCheck(ncclFunc_t func, const char* primName, const void* sendbuff, 
    void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root,
    ncclComm_t comm, cudaStream_t stream) {
  // Launch asynchronously if needed
  if (ncclAsyncMode()) {
    if (ncclChecks) {
      int savedDev;
      CUDACHECK(cudaGetDevice(&savedDev));
      CUDACHECK(cudaSetDevice(comm->cudaDev));
      // Check arguments
      ncclResult_t ret = ArgsCheck(sendbuff, recvbuff, count, type, op, root, comm, primName);
      NCCLCHECK(ncclAsyncErrCheck(ret));
      CUDACHECK(cudaSetDevice(savedDev));
    }
    NCCLCHECK(func(sendbuff, recvbuff, count, type, op, root, comm, stream));
    NCCLCHECK(ncclAsyncColl(comm));
    return ncclSuccess;
  } else {
    if (ncclChecks) NCCLCHECK(ArgsCheck(sendbuff, recvbuff, count, type, op, root, comm, primName));
    NCCLCHECK(func(sendbuff, recvbuff, count, type, op, root, comm, stream));
    NCCLCHECK(ncclCpuBarrierCheckin(comm));
    NCCLCHECK(ncclCpuBarrierWait(comm));
    return ncclSuccess;
  }
}
