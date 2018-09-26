/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclBroadcastKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;
  struct ncclConnInfo* send = &ring->send.conn;
  struct ncclConnInfo* recv = &ring->recv.conn;

  ncclPrimitives<UNROLL, BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS, T>
    prims(tid, nthreads, recv, send, args->comm->abortFlag);

  const ssize_t size = args->N;
  const int buffSize = ring->buffSize / sizeof(T);
  const int stepSize = buffSize / NCCL_STEPS;
  const int chunkSize = stepSize * BROADCAST_CHUNKSTEPS;
  const ssize_t loopSize = args->nRings*(ssize_t)chunkSize;
  const int rank = ring->devUserRanks[0];
  const int nextRank = ring->devUserRanks[1];
  const int root = args->root;

  int noffset = (send->waitStep%NCCL_STEPS)*stepSize;
  int poffset = (recv->waitStep%NCCL_STEPS)*stepSize;
  // Need all threads to read this before thread 0 might increment it
  __syncthreads();

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ prevInput = (T*)recv->buff;
  T * __restrict__ nextOutput = (T*)send->buff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,args->nRings));
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

#include "ll_kernel.h"

#define NEXT_STEP_LL \
  boffset += NCCL_LL_SLICE_LINES; \
  if (boffset == NCCL_LL_BUFF_LINES) boffset = 0; \
  flag++; \
  step++;

template<int UNUSED, class FUNC, typename T>
__device__ void ncclBroadcastLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;
  volatile uint64_t * recvHeadPtr = ring->recv.conn.llHead;
  volatile uint64_t * sendHeadPtr = ring->send.conn.llHead;
  volatile int * sizesFifo = ring->send.conn.llFifo;
  uint64_t sendHead = sendHeadPtr[0];
  const int rank = comm->rank;
  const int nextRank = ring->devUserRanks[1];
  const int root = args->root;

  typedef LLPrimitives<T, FUNC> LL;

  const ssize_t size = args->N;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nRings*chunkSize;

  uint64_t step = ring->send.conn.llStep;
  uint32_t flag = step + 1;
  int boffset = NCCL_LL_SLICE_LINES * STEP_TO_SLOT(step);

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  union ncclLLFifoLine * prevInput = (union ncclLLFifoLine *)ring->recv.conn.llBuff;
  union ncclLLFifoLine * nextOutput = (union ncclLLFifoLine *)ring->send.conn.llBuff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    if (size-gridOffset < loopSize) {
      chunkSize = args->lastChunkSize;
    }
    ssize_t offset = gridOffset + bid*chunkSize;

    int maxOffset = min(chunkSize, size-offset);
    if (rank == root) {
      WAIT_NEXT;
      if (thisInput == thisOutput) {
        LL::ReduceCopy(
            args->comm->abortFlag,
            thisInput + offset,
            nextOutput + boffset,
            maxOffset, flag,
            tid, nthreads);
      } else {
        LL::ReduceCopy(
            args->comm->abortFlag,
            thisInput + offset,
            thisOutput + offset,
            nextOutput + boffset,
            maxOffset, flag,
            tid, nthreads);
      }
      POST_SIZE;
      NEXT_STEP_LL;
    } else if (nextRank == root) {
      LL::ReduceCopy(
          args->comm->abortFlag,
          prevInput + boffset,
          thisOutput + offset,
          maxOffset, flag,
          tid, nthreads);
      NEXT_STEP_LL;
      ACK_PREV;
    } else {
      WAIT_NEXT;
      LL::ReduceCopy(
          args->comm->abortFlag,
          prevInput + boffset,
          thisOutput + offset,
          nextOutput + boffset,
          maxOffset, flag, flag,
          tid, nthreads);
      POST_SIZE;
      NEXT_STEP_LL;
      ACK_PREV;
    }
  }

  // We need everyone to acknowledge data even if they didn't receive anything
  // so that the next collective can start right away.
  ACK_PREV;

  FIFO_CLEANING_AND_SAVE_STEP(flag);
}
