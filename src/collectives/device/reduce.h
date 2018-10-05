/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclReduceKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  struct ncclComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  struct ncclConnInfo* recv = &channel->devPeers[ring->prev].recv.conn;
  struct ncclConnInfo* send = &channel->devPeers[ring->next].send.conn;

  ncclPrimitives<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, T, FUNC>
    prims(tid, nthreads, recv, send, args->comm->abortFlag);

  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int buffSize = channel->buffSize / sizeof(T);
  const int stepSize = buffSize / NCCL_STEPS;
  const int chunkSize = stepSize * REDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;
  const int rank = ring->devUserRanks[0];
  const int prevRank = ring->devUserRanks[nranks-1];
  const int root = args->root;

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
    ssize_t offset = gridOffset + bid*realChunkSize;
    int maxOffset = min(realChunkSize, size-offset);
    if (prevRank == root) {
      prims.send(thisInput+offset, nextOutput+noffset, chunkSize, maxOffset);
    } else if (rank == root) {
      prims.recvReduce(prevInput+poffset, thisInput+offset, thisOutput+offset, chunkSize, maxOffset);
    } else {
      prims.recvReduceSend(prevInput+poffset, thisInput+offset, nextOutput+noffset, chunkSize, maxOffset);
    }
    if ((poffset += chunkSize) == buffSize) poffset = 0;
    if ((noffset += chunkSize) == buffSize) noffset = 0;
  }
}

#include "ll_kernel.h"

template<int UNUSED, class FUNC, typename T>
__device__ void ncclReduceLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  struct ncclConnInfo* recv = &channel->devPeers[ring->prev].recv.conn;
  struct ncclConnInfo* send = &channel->devPeers[ring->next].send.conn;

  ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &recv, &send, comm->abortFlag);

  const ssize_t size = args->N;
  const int rank = comm->rank;
  const int nranks = comm->nRanks;
  const int prevRank = ring->devUserRanks[nranks-1];
  const int root = args->root;

  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    if (size-gridOffset < loopSize) {
      chunkSize = args->lastChunkSize;
    }
    ssize_t offset = gridOffset + bid*chunkSize;

    int maxOffset = min(chunkSize, size-offset);
    if (prevRank == root) {
      LLprims.send(thisInput+offset, maxOffset);
    } else if (rank == root) {
      LLprims.recvReduce(thisInput+offset, thisOutput+offset, maxOffset);
    } else {
      LLprims.recvReduceSend(thisInput+offset, maxOffset);
    }
  }
}
