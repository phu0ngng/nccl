/*************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "common_coll.h"

#include "collectives/collectives.h"

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(nthreads, coll, op) \
  (void*)NCCL_KERN_NAME(coll, op,  i8, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op,  u8, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op, i32, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op, u32, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op, i64, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op, u64, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op, f16, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op, f32, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op, f64, nthreads)
#define NCCL_FUNCS3B(nthreads, coll, op) \
  (void*)NCCL_KERN_NAME(coll, op,  i8, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op,  i8, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op,  i8, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op,  i8, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op,  i8, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op,  i8, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op,  i8, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op,  i8, nthreads), \
  (void*)NCCL_KERN_NAME(coll, op,  i8, nthreads)

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(nthreads, coll) \
  NCCL_FUNCS3A(nthreads, coll, sum ), \
  NCCL_FUNCS3A(nthreads, coll, prod), \
  NCCL_FUNCS3A(nthreads, coll, max ), \
  NCCL_FUNCS3A(nthreads, coll, min )
#define NCCL_FUNCS2B(nthreads, coll) \
  NCCL_FUNCS3B(nthreads, coll, copy), \
  NCCL_FUNCS3B(nthreads, coll, copy), \
  NCCL_FUNCS3B(nthreads, coll, copy), \
  NCCL_FUNCS3B(nthreads, coll, copy)

// Must be consistent with ncclColl_t
#define NCCL_FUNCS(nthreads) { \
  NCCL_FUNCS2B(nthreads, ncclBcast), \
  NCCL_FUNCS2A(nthreads, ncclReduce), \
  NCCL_FUNCS2B(nthreads, ncclAllGather), \
  NCCL_FUNCS2A(nthreads, ncclReduceScatter), \
  NCCL_FUNCS2A(nthreads, ncclAllReduce) }

// Must be consistent with the ncclFuncSet enum
static void* const ncclLLKerns[ncclCollCount*ncclNumOps*ncclNumTypes] = {
    NCCL_FUNCS2B(LL_NTHREADS, ncclBcastLL),
    NCCL_FUNCS2A(LL_NTHREADS, ncclReduceLL),
    NCCL_FUNCS2B(LL_NTHREADS, ncclAllGatherLL),
    NCCL_FUNCS2A(LL_NTHREADS, ncclReduceScatterLL),
    NCCL_FUNCS2A(LL_NTHREADS, ncclAllReduceLL)
};

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
  params->gridDim.x = min(params->gridDim.x, comm->nRings);

  int totalOps = 0;
  for (int r=0; r<params->gridDim.x; r++) totalOps += comm->rings[r].collCount;

  struct ncclColl* coll = comm->rings[0].collectives+comm->rings[0].collStart;
  memcpy(&comm->args, coll, sizeof(struct ncclColl));

  // One operation
  if (totalOps == 1 && coll->ll) {
    coll->active = 0;
    params->func = ncclLLKerns[coll->funcIndex];
    return ncclSuccess;
  }

  // Aggregated operations
  params->func = (void*)ncclMultiOpKernel;
  // Set active = 2 for the last operation
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclRing* ring = comm->rings+r;
    ring->collectives[(ring->collStart+ring->collCount-1)%NCCL_MAX_OPS].active = 2;
  }
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
  if (comm->launchMode == ncclComm::PARALLEL) {
    CUDACHECK(cudaLaunchKernel(params->func, params->gridDim, params->blockDim, params->args, params->sharedMem, comm->userStream));
  }
  // Start the network proxies as soon as the kernel has been launched. We can't
  // perform any CUDA call between the two or having a cudaFree between the CUDA
  // launch and the transportStartProxies call could cause a deadlock.
  // Also, starting the proxies after the CUDA launch seems to be better for
  // performance (latency).
  for (int r=0; r<params->gridDim.x; r++) {
    struct ncclRing* ring = comm->rings+r;
    ring->collStart = ring->collFifoTail;
    ring->collCount = 0;
  }
  params->gridDim.x = params->blockDim.x = 0;
  NCCLCHECK(transportStartProxies(comm));

  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(cudaEventRecord(comm->doneEvent, params->stream));
    CUDACHECK(cudaStreamWaitEvent(comm->userStream, comm->doneEvent, 0));
  } else {
    CUDACHECK(cudaEventRecord(comm->doneEvent, comm->userStream));
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueCheck(ncclFunc_t func, const char* primName, const void* sendbuff, 
    void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root,
    ncclComm_t comm, cudaStream_t stream) {
  if (comm == NULL) return ncclInvalidArgument;
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
