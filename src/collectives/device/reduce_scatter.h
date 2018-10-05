/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclReduceScatterKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  struct ncclComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  struct ncclConnInfo* recv = &channel->devPeers[ring->prev].recv.conn;
  struct ncclConnInfo* send = &channel->devPeers[ring->next].send.conn;

  ncclPrimitives<UNROLL, REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, T, FUNC>
    prims(tid, nthreads, recv, send, args->comm->abortFlag);

  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int buffSize = channel->buffSize / sizeof(T);
  const int stepSize = buffSize / NCCL_STEPS;
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;

  int noffset = (prims.getSendStep()%NCCL_STEPS)*stepSize;
  int poffset = (prims.getRecvStep()%NCCL_STEPS)*stepSize;
  // Need all threads to read this before thread 0 might increment it
  __syncthreads();

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ prevInput = (T*)recv->buff;
  T * __restrict__ nextOutput = (T*)send->buff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,args->nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*realChunkSize;

    /////////////// begin ReduceScatter steps ///////////////
    ssize_t offset;
    int maxOffset = min(realChunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[nranks-1];
    offset = chunkOffset + rankDest * size;

    prims.send(thisInput+offset, nextOutput+noffset, chunkSize, maxOffset);
    if ((noffset += chunkSize) == buffSize) noffset = 0;

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      prims.recvReduceSend(prevInput+poffset, thisInput+offset, nextOutput+noffset, chunkSize, maxOffset);
      if ((poffset += chunkSize) == buffSize) poffset = 0;
      if ((noffset += chunkSize) == buffSize) noffset = 0;
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    prims.recvReduce(prevInput+poffset, thisInput+offset, thisOutput+chunkOffset, chunkSize, maxOffset);
    if ((poffset += chunkSize) == buffSize) poffset = 0;
  }
}

#include "ll_kernel.h"

template<int UNUSED, class FUNC, typename T>
__device__ void ncclReduceScatterLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  struct ncclConnInfo* recv = &channel->devPeers[ring->prev].recv.conn;
  struct ncclConnInfo* send = &channel->devPeers[ring->next].send.conn;

  ncclLLPrimitives<T, FUNC> LLprims(tid, nthreads, recv, send, comm->abortFlag);

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    if (size-gridOffset < loopSize) {
      chunkSize = args->lastChunkSize;
    }
    ssize_t chunkOffset = gridOffset + bid*chunkSize;

    /////////////// begin ReduceScatter steps ///////////////
    ssize_t offset;
    int maxOffset = min(chunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[nranks-1];
    offset = chunkOffset + rankDest * size;

    LLprims.send(thisInput+offset, maxOffset);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      LLprims.recvReduceSend(thisInput+offset, maxOffset);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    LLprims.recvReduce(thisInput+offset, thisOutput+chunkOffset, maxOffset);
  }
}
