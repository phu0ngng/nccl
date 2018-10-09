/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclBroadcastRingKernel(struct CollectiveArgs* args) {
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  struct ncclComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  struct ncclConnInfo* recv = &channel->devPeers[ring->prev].recv.conn;
  struct ncclConnInfo* send = &channel->devPeers[ring->next].send.conn;

  ncclPrimitives<UNROLL, BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS, T>
    prims(threadIdx.x, nthreads, recv, send, args->comm->abortFlag);

  const ssize_t size = args->N;
  const int buffSize = channel->buffSize / sizeof(T);
  const int stepSize = buffSize / NCCL_STEPS;
  const int chunkSize = stepSize * BROADCAST_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;
  const int rank = ring->devUserRanks[0];
  const int nextRank = ring->devUserRanks[1];
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

    if (rank == root) {
      if (thisInput == thisOutput) {
        prims.send(thisInput+offset, nextOutput+noffset, chunkSize, maxOffset);
      } else {
        prims.copySend(thisInput+offset, thisOutput+offset, nextOutput+noffset, chunkSize, maxOffset);
      }
    } else if (nextRank == root) {
      prims.recv(prevInput+poffset, thisOutput+offset, chunkSize, maxOffset);
    } else {
      prims.recvCopySend(prevInput+poffset, thisOutput+offset, nextOutput+noffset, chunkSize, maxOffset);
    }
    if ((poffset += chunkSize) == buffSize) poffset = 0;
    if ((noffset += chunkSize) == buffSize) noffset = 0;
  }
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclBroadcastTreeKernel(struct CollectiveArgs* args) { }

#include "ll_kernel.h"

template<int UNUSED, class FUNC, typename T>
__device__ void ncclBroadcastLLRingKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  struct ncclConnInfo* recv = &channel->devPeers[ring->prev].recv.conn;
  struct ncclConnInfo* send = &channel->devPeers[ring->next].send.conn;

  ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, 1, &recv, 1, &send, comm->abortFlag);

  const ssize_t size = args->N;
  const int rank = comm->rank;
  const int nextRank = ring->devUserRanks[1];
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
    if (rank == root) {
      if (thisInput == thisOutput) {
        LLprims.send(thisInput+offset, maxOffset);
      } else {
        LLprims.copySend(thisInput + offset, thisOutput + offset, maxOffset);
      }
    } else if (nextRank == root) {
      LLprims.recv(thisOutput + offset, maxOffset);
    } else {
      LLprims.recvCopySend(thisOutput + offset, maxOffset);
    }
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclBroadcastLLTreeKernel(struct CollectiveArgs* args) { }
