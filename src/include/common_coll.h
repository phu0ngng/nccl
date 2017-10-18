/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef COMMON_COLL_H_
#define COMMON_COLL_H_

#include "core.h"

static ncclResult_t PointerCheck(const void* pointer, struct ncclComm* comm, const char* ptrname, const char* opname) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, pointer);
  if (err != cudaSuccess || attr.devicePointer == NULL) {
    WARN("%s : %s is not a valid pointer", opname, ptrname);
    return ncclInvalidArgument;
  }
  if (attr.memoryType == cudaMemoryTypeDevice && attr.device != comm->cudaDev) {
    WARN("%s : %s allocated on device %d mismatchs with NCCL device %d", opname, ptrname, attr.device, comm->cudaDev);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

static ncclResult_t PtrCheck(void* ptr, const char* opname, const char* ptrname) {
  if (ptr == NULL) {
    WARN("%s : %s argument is NULL", opname, ptrname);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

static ncclResult_t ArgsCheck(const void* sendbuff, const void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, struct ncclComm* comm, const char* opname) {
  NCCLCHECK(PtrCheck(comm, opname, "comm"));
  // First, the easy ones
  if (root < 0 || root >= comm->nRanks) {
    WARN("%s : invalid root %d (root should be in the 0..%d range)", opname, root, comm->nRanks);
    return ncclInvalidArgument;
  }
  if (type < 0 || type >= ncclNumTypes) {
    WARN("%s : invalid type %d", opname, type);
    return ncclInvalidArgument;
  }
  if (op < 0 || op >= ncclNumOps) {
    WARN("%s : invalid reduction operation %d", opname, op);
    return ncclInvalidArgument;
  }

  // Check pointers
  NCCLCHECK(PointerCheck(sendbuff, comm, "sendbuff", opname))
  if (strcmp(opname, "Reduce") == 0 && comm->rank != root) {
    // No need to check recvbuff pointer for non-root reduce
    return ncclSuccess;
  }
  NCCLCHECK(PointerCheck(recvbuff, comm, "recvbuff", opname))
  return ncclSuccess;
}

template<typename T>
ncclResult_t ArgsSetup(const T* sendbuff, T* recvbuff,
		const int root, const size_t count, ncclComm *comm) {
  if (comm->args.nColls == NCCL_MAX_OPS) {
    WARN("Too many aggregated operations (%d max)", NCCL_MAX_OPS);
    return ncclInvalidUsage;
  }
  struct ncclColl* coll = comm->collectives+comm->collFifoTail;
  volatile int* functionPtr = (volatile int*)&coll->function;
  while (functionPtr[0] != 0) sched_yield();
  struct CollectiveArgs* args = &coll->args;
  args->root = root;
  args->N = count;
  args->ThisInput = sendbuff;
  args->ThisOutput = recvbuff;
  args->comm = comm->devComm;
  args->opCount = comm->opCount;
  return ncclSuccess;
}

static __inline__ int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return -1;
  }
}

static void saveKernel(int coll, ncclRedOp_t op, ncclDataType_t dtype, int nbytes, struct ncclComm* comm, cudaStream_t stream, int ll) {
  struct ncclColl* collective = comm->collectives+comm->collFifoTail;
  collective->function = NCCL_FUNCTION(coll, op, dtype, ll, 1);
  collective->nBlocks = ll ? 1 : LIMIT_NRINGS(nbytes, comm->nRings);
  collective->nThreads = ll ? LL_NTHREADS : comm->nThreads+1;
  comm->collNBlocks = max(comm->collNBlocks, collective->nBlocks);
  comm->collNThreads = max(comm->collNThreads, collective->nThreads);
  comm->userStream = stream;
  comm->args.nColls++;
  comm->collFifoTail = (comm->collFifoTail+1)%NCCL_MAX_OPS;
}

extern __global__ void ncclKernel64(struct KernelArgs args);
extern __global__ void ncclKernel128(struct KernelArgs args);
extern __global__ void ncclKernel256(struct KernelArgs args);
extern __global__ void ncclKernel512(struct KernelArgs args);

#endif
