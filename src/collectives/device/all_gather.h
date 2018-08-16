/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

// Increase Step and poffset/noffset for buffer sync
#define NEXT_STEP \
  step += ALLGATHER_CHUNKSTEPS; \
  poffset = noffset; \
  noffset += chunkSize; \
  if (noffset == buffSize) noffset = 0;

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllGatherKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  __shared__ T* sharedNextOutput;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;
  int prevdirect = ring->recv.conn.direct;
  int nextdirect = ring->send.conn.direct;

  WaitFlag waitDoneFromNext(args->comm->abortFlag, ring->send.conn.head, NCCL_STEPS);
  WaitFlag waitReadyFromPrev(args->comm->abortFlag, ring->recv.conn.tail, ALLGATHER_CHUNKSTEPS);
  PostFlag postDoneToPrev(ring->recv.conn.head, ALLGATHER_CHUNKSTEPS, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, ring->send.conn.fifo, NCCL_STEPS);

  typedef Primitives<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLREDUCE_SLICESTEPS, T> Prims;

  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int buffSize = ring->buffSize / sizeof(T);
  const int stepSize = buffSize / NCCL_STEPS;
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nRings*(ssize_t)chunkSize;

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

  uint64_t step = ring->send.conn.step;
  step = ROUNDUP(step, ALLGATHER_CHUNKSTEPS);
  int poffset, noffset = (step%NCCL_STEPS)*stepSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

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

    if (thisInput + chunkOffset == thisOutput + offset) { // In place
      Prims::Copy(tid, nthreads,
          thisInput  + chunkOffset,
          nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset),
          chunkSize, maxOffset,
          step,
          waitDoneFromNext,
          postReadyToNext);
    } else {
      Prims::DoubleCopy(tid, nthreads,
          thisInput  + chunkOffset,
          thisOutput + offset,
	  nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset),
          chunkSize, maxOffset,
          step,
          waitDoneFromNext,
          postReadyToNext);
    }

    NEXT_STEP; // Increases step, poffset, noffset

    // k-2 steps: copy to next GPU
    if (prevdirect) {
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ring->devUserRanks[nranks-j];
        offset = chunkOffset + rankDest * size;

        Prims::Copy(tid, nthreads,
            thisOutput + offset,
	    nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset),
            chunkSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);

        NEXT_STEP;
      }
      Prims::Copy(tid, nthreads,
          NULL,
          NULL,
          0, 0,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    } else {
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ring->devUserRanks[nranks-j];
        offset = chunkOffset + rankDest * size;

        Prims::DoubleCopy(tid, nthreads,
            prevInput + poffset,
            thisOutput + offset,
	    nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset),
            chunkSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);

        NEXT_STEP;
      }

      // Make final copy from buffer to dest.
      rankDest = ring->devUserRanks[1];
      offset = chunkOffset + rankDest * size;

      // Here we need to copy from buffer to this output.
      Prims::Copy(tid, nthreads,
          prevInput + poffset,
          thisOutput + offset,
          chunkSize, maxOffset,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    }
  }

  // Save step counter for next op
  if (tid == 0) ring->send.conn.step = step;
  __syncthreads();
}

#include "ll_kernel.h"

#define NEXT_STEP_LL \
  poffset = noffset; \
  pflag = nflag; \
  noffset += NCCL_LL_SLICE_LINES; \
  if (noffset == NCCL_LL_BUFF_LINES) { noffset = 0; } \
  nflag++; \
  step++;

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllGatherLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = comm->rings+blockIdx.x;
  volatile uint64_t * recvHeadPtr = ring->recv.conn.llHead;
  volatile uint64_t * sendHeadPtr = ring->send.conn.llHead;
  volatile int * sizesFifo = ring->send.conn.llFifo;
  uint64_t sendHead = sendHeadPtr[0];

  typedef LLPrimitives<T, FUNC> LL;

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nRings*chunkSize;

  uint64_t step = ring->send.conn.llStep;
  uint32_t pflag, nflag = step + 1;
  int poffset, noffset = NCCL_LL_SLICE_LINES * STEP_TO_SLOT(step);

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  union ncclLLFifoLine * prevInput = (union ncclLLFifoLine *)ring->recv.conn.llBuff;
  union ncclLLFifoLine * nextOutput = (union ncclLLFifoLine *)ring->send.conn.llBuff;

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

    WAIT_NEXT;
    if (thisInput + chunkOffset == thisOutput + offset) { // In place
      LL::ReduceCopy(
          args->comm->abortFlag,
          thisInput  + chunkOffset,
          nextOutput + noffset,
          maxOffset, nflag,
          tid, nthreads);
    } else {
      LL::ReduceCopy(
          args->comm->abortFlag,
          thisInput  + chunkOffset,
          thisOutput + offset,
          nextOutput + noffset,
          maxOffset, nflag,
          tid, nthreads);
    }
    POST_SIZE;

    NEXT_STEP_LL;

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      WAIT_NEXT;
      LL::ReduceCopy(
          args->comm->abortFlag,
          prevInput  + poffset,
          thisOutput + offset,
          nextOutput + noffset,
          maxOffset, pflag, nflag,
          tid, nthreads);
      POST_SIZE;
      ACK_PREV;

      NEXT_STEP_LL;
    }

    // step k-1: final store
    rankDest = ring->devUserRanks[1];
    offset = chunkOffset + rankDest * size;

    LL::ReduceCopy(
        args->comm->abortFlag,
        prevInput  + poffset,
        thisOutput + offset,
        maxOffset, pflag,
        tid, nthreads);
    ACK_PREV;
  }

  FIFO_CLEANING_AND_SAVE_STEP(nflag);
}
