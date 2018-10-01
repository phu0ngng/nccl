/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceKernel(struct CollectiveArgs* args) {
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

  ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, FUNC>
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

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nRings));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int maxOffset;
    int slice;

    // step 0: push data to next GPU
    slice = ring->devUserRanks[nranks-1];
    offset = chunkOffset + slice * realChunkSize;
    maxOffset = min(realChunkSize, size-offset);

    prims.send(thisInput+offset, nextOutput+noffset, chunkSize, maxOffset);
    if ((noffset += chunkSize) == buffSize) noffset = 0;

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      maxOffset = min(realChunkSize, size-offset);

      prims.recvReduceSend(prevInput+poffset, thisInput+offset, nextOutput+noffset, chunkSize, maxOffset);
      if ((poffset += chunkSize) == buffSize) poffset = 0;
      if ((noffset += chunkSize) == buffSize) noffset = 0;
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * realChunkSize;
    maxOffset = min(realChunkSize, size-offset);

    T* output = nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset);
    prims.recvReduceCopySend(prevInput+poffset, thisInput+offset, output, thisOutput+offset, chunkSize, maxOffset);
    if ((poffset += chunkSize) == buffSize) poffset = 0;
    if ((noffset += chunkSize) == buffSize) noffset = 0;

    // k-2 steps: copy to next GPU
    if (prevdirect) {
      for (int j=1; j<nranks-1; ++j) {
        slice = ring->devUserRanks[nranks - j];
        offset = chunkOffset + slice * realChunkSize;
        maxOffset = min(realChunkSize, size-offset);

        T* output = nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset);
        prims.recvSend(thisOutput+offset, output, chunkSize, maxOffset);
        if ((poffset += chunkSize) == buffSize) poffset = 0;
        if ((noffset += chunkSize) == buffSize) noffset = 0;
      }
      prims.recv(NULL, NULL, 0, 0);
      if ((poffset += chunkSize) == buffSize) poffset = 0;
    } else {
      for (int j=1; j<nranks-1; ++j) {
        slice = ring->devUserRanks[nranks - j];
        offset = chunkOffset + slice * realChunkSize;
        maxOffset = min(realChunkSize, size-offset);

        T* output = nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset);
        prims.recvCopySend(prevInput+poffset, thisOutput+offset, output, chunkSize, maxOffset);
        if ((poffset += chunkSize) == buffSize) poffset = 0;
        if ((noffset += chunkSize) == buffSize) noffset = 0;
      }

      // Make final copy from buffer to dest.
      slice = ring->devUserRanks[1];
      offset = chunkOffset + slice * realChunkSize;
      maxOffset = min(realChunkSize, size-offset);

      // Here we need to copy from buffer to this output.
      prims.recv(prevInput+poffset, thisOutput+offset, chunkSize, maxOffset);
      if ((poffset += chunkSize) == buffSize) poffset = 0;
    }
  }
}

#include "ll_kernel.h"

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;

  ncclLLPrimitives<T, FUNC> LLprims(tid, nthreads, &ring->recv.conn, &ring->send.conn, comm->abortFlag);

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  const int nringsXnranks = args->nRings * nranks;
  ssize_t sliceSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = nringsXnranks*sliceSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  int chunkSize = sliceSize;
  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    ssize_t rest = size - gridOffset;
    if (rest < loopSize) {
      chunkSize = min(sliceSize, DIVUP(rest, nringsXnranks));
      ALIGN_SIZE(chunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    }
    ssize_t chunkOffset = gridOffset + bid*nranks*chunkSize;

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int maxOffset;
    int slice;

    // step 0: push data to next GPU
    slice = ring->devUserRanks[nranks-1];
    offset = chunkOffset + slice * chunkSize;
    maxOffset = min(chunkSize, size-offset);

    LLprims.send(thisInput+offset, maxOffset);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * chunkSize;
      maxOffset = min(chunkSize, size-offset);

      LLprims.recvReduceSend(thisInput+offset, maxOffset);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * chunkSize;
    maxOffset = min(chunkSize, size-offset);

    LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, maxOffset);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * chunkSize;
      maxOffset = min(chunkSize, size-offset);

      LLprims.recvCopySend(thisOutput+offset, maxOffset);
    }

    // Make final copy from buffer to dest.
    slice = ring->devUserRanks[1];
    offset = chunkOffset + slice * chunkSize;
    maxOffset = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(thisOutput+offset, maxOffset);
  }
}
