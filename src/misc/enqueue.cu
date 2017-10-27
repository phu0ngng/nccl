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

ncclResult_t setupLaunch(struct ncclComm* comm, struct cudaLaunchParams* params) {
  // Set active = 2 for last operation
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclRing* ring = comm->rings+r;
    ring->collectives[(ring->collStart+ring->collCount-1)%NCCL_MAX_OPS].active = 2;
  }
  // Pass the first operation as argument to reduce latency
  memcpy(&comm->args, comm->rings[0].collectives+comm->rings[0].collStart, sizeof(struct ncclColl));
  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierCheckin(struct ncclComm* comm) {
  if (comm->nRanks == 1) return ncclSuccess;
  struct cudaLaunchParams* params = comm->myParams;

  NCCLCHECK(setupLaunch(comm, params));

  if (comm->launchMode == ncclComm::GROUP) {
    // Enqueue stream dependency
    CUDACHECK(cudaEventRecord(comm->doneEvent, comm->userStream));
    CUDACHECK(cudaStreamWaitEvent(params->stream, comm->doneEvent, 0));
  } else {
    if (comm->userStream != params->stream) {
      CUDACHECK(cudaStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
    }
    params->stream = comm->userStream;
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
  struct cudaLaunchParams *params = comm->myParams;
  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(cudaEventRecord(comm->doneEvent, params->stream));
    CUDACHECK(cudaStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
  } else {
    CUDACHECK(cudaLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, params->stream));
    CUDACHECK(cudaEventRecord(comm->doneEvent, comm->userStream));
  }
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclRing* ring = comm->rings+r;
    ring->collStart = ring->collFifoTail;
    ring->collCount = 0;
  }
  params->gridDim.x = params->blockDim.x = 0;
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
