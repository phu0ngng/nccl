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

static ncclResult_t saveKernel(int coll, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t dtype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int nbytes, int ll) {
  int nBlocks = ll ? 1 : LIMIT_NRINGS(nbytes, comm->nRings);
  comm->myParams->gridDim.x = max(comm->myParams->gridDim.x, nBlocks);
  int nThreads = ll ? LL_NTHREADS : comm->nThreads+1;
  comm->myParams->blockDim.x = max(comm->myParams->blockDim.x, nThreads);
  comm->userStream = stream;
  for (int r=0; r<nBlocks; r++) {
    struct ncclRing* ring = comm->rings+r;
    if (ring->collCount == NCCL_MAX_OPS) {
      WARN("Too many aggregated operations (%d max)", NCCL_MAX_OPS);
      return ncclInvalidUsage;
    }
    struct ncclColl* c = ring->collectives+ring->collFifoTail;
    volatile uint8_t* activePtr = (volatile uint8_t*)&c->active;
    while (activePtr[0] != 0) sched_yield();

    struct CollectiveArgs* args = &c->args;
    args->root = root;
    args->N = count;
    args->ThisInput = sendbuff;
    args->ThisOutput = recvbuff;
    args->comm = comm->devComm;
    args->opCount = comm->opCount;
    args->nRings = nBlocks;

    c->nThreads = nThreads;
    c->coll = coll; c->redop = op; c->dtype = dtype; c->ll = ll;
    c->active = 1;
    ring->collFifoTail = (ring->collFifoTail+1)%NCCL_MAX_OPS;
    ring->collCount ++;
  }
  return ncclSuccess;
}

extern __global__ void ncclKernel64 (struct ncclColl firstColl);
extern __global__ void ncclKernel128(struct ncclColl firstColl);
extern __global__ void ncclKernel256(struct ncclColl firstColl);
extern __global__ void ncclKernel512(struct ncclColl firstColl);

#endif
