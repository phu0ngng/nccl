/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllGatherKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  __shared__ T* sharedNextOutput;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;
  struct ncclConnInfo* send = &ring->send.conn;
  struct ncclConnInfo* recv = &ring->recv.conn;
  int prevdirect = recv->direct;
  int nextdirect = send->direct;

  ncclPrimitives<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLREDUCE_SLICESTEPS, T>
    prims(tid, nthreads, recv, send, args->comm->abortFlag);

  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int buffSize = ring->buffSize / sizeof(T);
  const int stepSize = buffSize / NCCL_STEPS;
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nRings*(ssize_t)chunkSize;

  int noffset = (prims.getSendStep()%NCCL_STEPS)*stepSize;
  int poffset = (prims.getRecvStep()%NCCL_STEPS)*stepSize;

  if (tid == 0) {
    if (prevdirect) {
      *ring->recv.conn.ptrExchange = args->ThisOutput;
    }
    if (nextdirect) {
      void* volatile* ptr = &(ring->devMemSend->ptrExchange);
      while (*ptr == nullptr);
      sharedNextOutput = (T*)*ptr;
      *ptr = nullptr;
    }
  }
  __syncthreads();

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ prevInput = (T*)recv->buff;
  T * __restrict__ nextOutput = (T*)send->buff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,args->nRings));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*realChunkSize;

    /////////////// begin AllGather steps ///////////////
    ssize_t offset;
    int maxOffset = min(realChunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    T* output = nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset);
    if (thisInput + chunkOffset == thisOutput + offset) { // In place
      prims.send(thisInput+chunkOffset, output, chunkSize, maxOffset);
    } else {
      prims.copySend(thisInput+chunkOffset, thisOutput+offset, output, chunkSize, maxOffset);
    }
    if ((noffset += chunkSize) == buffSize) noffset = 0;

    // k-2 steps: copy to next GPU
    if (prevdirect) {
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ring->devUserRanks[nranks-j];
        offset = chunkOffset + rankDest * size;

        T* output = nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset);
        prims.recvSend(thisOutput+offset, output, chunkSize, maxOffset);
        if ((poffset += chunkSize) == buffSize) poffset = 0;
        if ((noffset += chunkSize) == buffSize) noffset = 0;
      }
      prims.recv(NULL, NULL, 0, 0);
    } else {
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ring->devUserRanks[nranks-j];
        offset = chunkOffset + rankDest * size;

        T* output = nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset);
        prims.recvCopySend(prevInput+poffset, thisOutput+offset, output, chunkSize, maxOffset);
        if ((poffset += chunkSize) == buffSize) poffset = 0;
        if ((noffset += chunkSize) == buffSize) noffset = 0;
      }

      // Make final copy from buffer to dest.
      rankDest = ring->devUserRanks[1];
      offset = chunkOffset + rankDest * size;

      // Here we need to copy from buffer to this output.
      prims.recv(prevInput+poffset, thisOutput+offset, chunkSize, maxOffset);
      if ((poffset += chunkSize) == buffSize) poffset = 0;
    }
  }
}

#include "ll_kernel.h"

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllGatherLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;

  ncclLLPrimitives<T, FUNC> LLprims(tid, nthreads, &ring->recv.conn, &ring->send.conn, comm->abortFlag);

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nRings*chunkSize;


  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    if (size-gridOffset < loopSize) {
      chunkSize = args->lastChunkSize;
    }
    ssize_t chunkOffset = gridOffset + bid*chunkSize;

    /////////////// begin AllGather steps ///////////////
    ssize_t offset;
    int maxOffset = min(chunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    if (thisInput + chunkOffset == thisOutput + offset) { // In place
      LLprims.send(thisInput+chunkOffset, maxOffset);
    } else {
      LLprims.copySend(thisInput+chunkOffset, thisOutput+offset, maxOffset);
    }

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      LLprims.recvCopySend(thisOutput+offset, maxOffset);
    }

    // step k-1: final store
    rankDest = ring->devUserRanks[1];
    offset = chunkOffset + rankDest * size;

    LLprims.recv(thisOutput+offset, maxOffset);
  }
}
