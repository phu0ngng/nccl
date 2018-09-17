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
  step += ALLREDUCE_CHUNKSTEPS; \
  poffset = noffset; \
  noffset += chunkSize; \
  if (noffset == buffSize) noffset = 0;

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  __shared__ T* sharedNextOutput;
  struct ncclComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  int prevdirect = ring->recv.conn.direct;
  int nextdirect = ring->send.conn.direct;

  WaitFlag waitDoneFromNext(ring->send.conn.head, NCCL_STEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, ALLREDUCE_CHUNKSTEPS);
  PostFlag postDoneToPrev(ring->recv.conn.head, ALLREDUCE_CHUNKSTEPS, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, ring->send.conn.fifo, NCCL_STEPS);

  typedef Primitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, FUNC> Prims;

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  const int buffSize = channel->buffSize / sizeof(T);
  const int stepSize = buffSize / NCCL_STEPS;
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;

  if (tid == 0) {
    if (prevdirect) {
      *ring->recv.conn.ptrExchange = args->ThisOutput;
    }
    if (nextdirect) {
      void* volatile* ptr = ring->send.conn.ptrExchange;
      while (*ptr == nullptr);
      sharedNextOutput = (T*)*ptr;
      *ptr = nullptr;
    }
  }
  __syncthreads();

  uint64_t step = ring->send.conn.step;
  step = ROUNDUP(step, ALLREDUCE_CHUNKSTEPS);
  int poffset, noffset = (step%NCCL_STEPS)*stepSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
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

    Prims::Copy(tid, nthreads,
        thisInput  + offset,
        nextOutput + noffset,
        chunkSize, maxOffset,
        step,
        waitDoneFromNext,
        postReadyToNext);

    NEXT_STEP; // Increases step, poffset, noffset

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      maxOffset = min(realChunkSize, size-offset);

      Prims::Reduce(tid, nthreads,
          prevInput  + poffset,
          thisInput  + offset,
          nextOutput + noffset,
          chunkSize, maxOffset,
          step,
          waitDoneFromNext, waitReadyFromPrev,
          postReadyToNext, postDoneToPrev);

      NEXT_STEP;
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * realChunkSize;
    maxOffset = min(realChunkSize, size-offset);

    Prims::ReduceCopy(tid, nthreads,
        prevInput  + poffset,
        thisInput  + offset,
        nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset),
        thisOutput + offset,
        chunkSize, maxOffset,
        step,
        waitDoneFromNext, waitReadyFromPrev,
        postReadyToNext, postDoneToPrev);

    NEXT_STEP;

    // k-2 steps: copy to next GPU
    if (prevdirect) {
      for (int j=1; j<nranks-1; ++j) {
        slice = ring->devUserRanks[nranks - j];
        offset = chunkOffset + slice * realChunkSize;
        maxOffset = min(realChunkSize, size-offset);

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
        slice = ring->devUserRanks[nranks - j];
        offset = chunkOffset + slice * realChunkSize;
        maxOffset = min(realChunkSize, size-offset);

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
      slice = ring->devUserRanks[1];
      offset = chunkOffset + slice * realChunkSize;
      maxOffset = min(realChunkSize, size-offset);

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
__device__ void ncclAllReduceLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclRing* ring = &comm->channels[blockIdx.x].ring;
  volatile uint64_t * recvHeadPtr = ring->recv.conn.llHead;
  volatile uint64_t * sendHeadPtr = ring->send.conn.llHead;
  volatile int * sizesFifo = ring->send.conn.llFifo;
  uint64_t sendHead = sendHeadPtr[0];

  typedef LLPrimitives<T, FUNC> LL;

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  const int nringsXnranks = args->nChannels * nranks;
  ssize_t sliceSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = nringsXnranks*sliceSize;

  uint64_t step = ring->send.conn.llStep;
  uint32_t pflag, nflag = step + 1;
  int poffset, noffset = NCCL_LL_SLICE_LINES * STEP_TO_SLOT(step);

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  union ncclLLFifoLine * prevInput = (union ncclLLFifoLine *)ring->recv.conn.llBuff;
  union ncclLLFifoLine * nextOutput = (union ncclLLFifoLine *)ring->send.conn.llBuff;

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

    WAIT_NEXT;
    LL::ReduceCopy(
        thisInput  + offset,
        nextOutput + noffset,
        maxOffset, nflag,
        tid, nthreads);
    POST_SIZE;

    NEXT_STEP_LL;

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * chunkSize;
      maxOffset = min(chunkSize, size-offset);

      WAIT_NEXT;
      LL::ReduceCopy(
          thisInput  + offset,
          prevInput  + poffset,
          nextOutput + noffset,
          maxOffset, pflag, nflag,
          tid, nthreads);
      POST_SIZE;
      ACK_PREV;

      NEXT_STEP_LL;
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * chunkSize;
    maxOffset = min(chunkSize, size-offset);

    WAIT_NEXT;
    LL::ReduceCopy(
        thisInput  + offset,
        prevInput  + poffset,
        thisOutput + offset,
        nextOutput + noffset,
        maxOffset, pflag, nflag,
        tid, nthreads);
    POST_SIZE;
    ACK_PREV;

    NEXT_STEP_LL;

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      slice = ring->devUserRanks[nranks - j];
      offset = chunkOffset + slice * chunkSize;
      maxOffset = min(chunkSize, size-offset);

      WAIT_NEXT;
      LL::ReduceCopy(
          prevInput + poffset,
          thisOutput + offset,
          nextOutput + noffset,
          maxOffset, pflag, nflag,
          tid, nthreads);
      POST_SIZE;
      ACK_PREV;

      NEXT_STEP_LL;
    }

    // Make final copy from buffer to dest.
    slice = ring->devUserRanks[1];
    offset = chunkOffset + slice * chunkSize;
    maxOffset = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LL::ReduceCopy(
        prevInput + poffset,
        thisOutput + offset,
        maxOffset, pflag,
        tid, nthreads);
    ACK_PREV;
  }

  FIFO_CLEANING_AND_SAVE_STEP(nflag);
}
