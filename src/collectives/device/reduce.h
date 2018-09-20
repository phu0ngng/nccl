/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

// Increase Step and boffset for buffer sync
#define NEXT_STEP \
  step += REDUCE_CHUNKSTEPS; \
  boffset += chunkSize; \
  if (boffset == buffSize) boffset = 0;

template<int UNROLL, class FUNC, typename T>
__device__ void ncclReduceKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  struct ncclComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  struct ncclConnector* recv = &channel->devPeers[ring->prev].recv;
  struct ncclConnector* send = &channel->devPeers[ring->next].send;

  WaitFlag waitDoneFromNext(send->conn.head, NCCL_STEPS-REDUCE_CHUNKSTEPS);
  WaitFlag waitReadyFromPrev(recv->conn.tail, 0);
  PostFlag postDoneToPrev(recv->conn.head, 0, NULL, 0);
  PostFlag postReadyToNext(send->conn.tail, 0, send->conn.fifo, NCCL_STEPS);

  typedef Primitives<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, T, FUNC> Prims;

  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int buffSize = channel->buffSize / sizeof(T);
  const int stepSize = buffSize / NCCL_STEPS;
  const int chunkSize = stepSize * REDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;
  const int rank = ring->devUserRanks[0];
  const int prevRank = ring->devUserRanks[nranks-1];
  const int root = args->root;

  uint64_t step = send->conn.step;
  step = ROUNDUP(step, REDUCE_CHUNKSTEPS);
  int boffset = (step%NCCL_STEPS)*stepSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ prevInput = (T*)recv->conn.buff;
  T * __restrict__ nextOutput = (T*)send->conn.buff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,args->nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t offset = gridOffset + bid*realChunkSize;
    int maxOffset = min(realChunkSize, size-offset);
    if (prevRank == root) {
      Prims::Copy(tid, nthreads,
          thisInput + offset,
          nextOutput + boffset,
          chunkSize, maxOffset,
          step,
          waitDoneFromNext,
          postReadyToNext);
    } else if (rank == root) {
      Prims::Reduce(tid, nthreads,
          prevInput  + boffset,
          thisInput + offset,
          thisOutput + offset,
          chunkSize, maxOffset,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    } else {
      Prims::Reduce(tid, nthreads,
          prevInput + boffset,
          thisInput + offset,
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
    send->conn.step = step;
    // Make sure last rank updates root's head otherwise it will be blocked
    // on the next operation
    if (prevRank == root)
      *recv->conn.head = step;
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
__device__ void ncclReduceLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  struct ncclConnector* recv = &channel->devPeers[ring->prev].recv;
  struct ncclConnector* send = &channel->devPeers[ring->next].send;
  volatile uint64_t * recvHeadPtr = recv->conn.llHead;
  volatile uint64_t * sendHeadPtr = send->conn.llHead;
  volatile int * sizesFifo = send->conn.llFifo;
  uint64_t sendHead = sendHeadPtr[0];
  const int nranks = comm->nRanks;
  const int rank = comm->rank;
  const int prevRank = ring->devUserRanks[nranks-1];
  const int root = args->root;

  typedef LLPrimitives<T, FUNC> LL;

  const ssize_t size = args->N;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;

  uint64_t step = send->conn.llStep;
  uint32_t flag = step + 1;
  int boffset = NCCL_LL_SLICE_LINES * STEP_TO_SLOT(step);

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  union ncclLLFifoLine * prevInput = (union ncclLLFifoLine *)recv->conn.llBuff;
  union ncclLLFifoLine * nextOutput = (union ncclLLFifoLine *)send->conn.llBuff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    if (size-gridOffset < loopSize) {
      chunkSize = args->lastChunkSize;
    }
    ssize_t offset = gridOffset + bid*chunkSize;

    int maxOffset = min(chunkSize, size-offset);
    if (prevRank == root) {
      WAIT_NEXT;
      LL::ReduceCopy(
          thisInput + offset,
          nextOutput + boffset,
          maxOffset, flag,
          tid, nthreads);
      POST_SIZE;
      NEXT_STEP_LL;
    } else if (rank == root) {
      LL::ReduceCopy(
          thisInput + offset,
          prevInput  + boffset,
          thisOutput + offset,
          maxOffset, flag,
          tid, nthreads);
      NEXT_STEP_LL;
      ACK_PREV;
    } else {
      WAIT_NEXT;
      LL::ReduceCopy(
          thisInput + offset,
          prevInput + boffset,
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
