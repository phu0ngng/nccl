/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

// Increase Step and boffset for buffer sync
#define NEXT_STEP \
  step += BROADCAST_CHUNKSTEPS; \
  boffset += chunkSize; \
  if (boffset == buffSize) boffset = 0;

template<int UNROLL, class FUNC, typename T>
__device__ void ncclBroadcastKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;

  WaitFlag waitDoneFromNext(ring->send.conn.head, NCCL_STEPS-BROADCAST_CHUNKSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, 0);
  PostFlag postDoneToPrev(ring->recv.conn.head, 0, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, ring->send.conn.fifo, NCCL_STEPS);

  typedef Primitives<UNROLL, BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS, T> Prims;

  const ssize_t size = args->N;
  const int buffSize = ring->buffSize / sizeof(T);
  const int stepSize = buffSize / NCCL_STEPS;
  const int chunkSize = stepSize * BROADCAST_CHUNKSTEPS;
  const ssize_t loopSize = args->nRings*(ssize_t)chunkSize;
  const int rank = ring->devUserRanks[0];
  const int nextRank = ring->devUserRanks[1];
  const int root = args->root;

  uint64_t step = ring->send.conn.step;
  step = ROUNDUP(step, BROADCAST_CHUNKSTEPS);
  int boffset = (step%NCCL_STEPS)*stepSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,args->nRings));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t offset = gridOffset + bid*realChunkSize;
    int maxOffset = min(realChunkSize, size-offset);

    if (rank == root) {
      if (thisInput == thisOutput) {
        Prims::Copy(tid, nthreads,
            thisInput  + offset,
            nextOutput + boffset,
            chunkSize, maxOffset,
            step,
            waitDoneFromNext,
            postReadyToNext);
      } else {
        Prims::DoubleCopy(tid, nthreads,
            thisInput  + offset,
            thisOutput + offset,
            nextOutput + boffset,
            chunkSize, maxOffset,
            step,
            waitDoneFromNext,
            postReadyToNext);
      }
    } else if (nextRank == root) {
      Prims::Copy(tid, nthreads,
          prevInput  + boffset,
          thisOutput + offset,
          chunkSize, maxOffset,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    } else {
      Prims::DoubleCopy(tid, nthreads,
          prevInput + boffset,
          thisOutput + offset,
          nextOutput + boffset,
          chunkSize, maxOffset,
          step,
          waitDoneFromNext, waitReadyFromPrev,
          postReadyToNext, postDoneToPrev);
    }
    NEXT_STEP; // Increases step, boffset
  }

  // Save step counter for next op
  if (tid == 0) {
    ring->send.conn.step = step;
    // Make sure root update prev's head otherwise it will be blocked
    // on the next operation
    if (rank == root)
      *ring->recv.conn.head = step;
  }
  __syncthreads();
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
  const int ll_nthreads = args->nThreads;
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
            thisInput + offset,
            nextOutput + boffset,
            maxOffset, flag, ll_nthreads);
      } else {
        LL::ReduceCopy(
            thisInput + offset,
            thisOutput + offset,
            nextOutput + boffset,
            maxOffset, flag, ll_nthreads);
      }
      POST_SIZE;
      NEXT_STEP_LL;
    } else if (nextRank == root) {
      LL::ReduceCopy(
          prevInput + boffset,
          thisOutput + offset,
          maxOffset, flag, ll_nthreads);
      NEXT_STEP_LL;
      ACK_PREV;
    } else {
      WAIT_NEXT;
      LL::ReduceCopy(
          prevInput + boffset,
          thisOutput + offset,
          nextOutput + boffset,
          maxOffset, flag, flag, ll_nthreads);
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
