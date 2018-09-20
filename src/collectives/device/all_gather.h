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
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  struct ncclConnector* recv = &channel->devPeers[ring->prev].recv;
  struct ncclConnector* send = &channel->devPeers[ring->next].send;
  int prevdirect = recv->conn.direct;
  int nextdirect = send->conn.direct;

  WaitFlag waitDoneFromNext(send->conn.head, NCCL_STEPS);
  WaitFlag waitReadyFromPrev(recv->conn.tail, ALLGATHER_CHUNKSTEPS);
  PostFlag postDoneToPrev(recv->conn.head, ALLGATHER_CHUNKSTEPS, NULL, 0);
  PostFlag postReadyToNext(send->conn.tail, 0, send->conn.fifo, NCCL_STEPS);

  typedef Primitives<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLREDUCE_SLICESTEPS, T> Prims;

  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int buffSize = channel->buffSize / sizeof(T);
  const int stepSize = buffSize / NCCL_STEPS;
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;

  if (tid == 0) {
    if (prevdirect) {
      *recv->conn.ptrExchange = args->ThisOutput;
    }
    if (nextdirect) {
      void* volatile* ptr = send->conn.ptrExchange;
      while (*ptr == nullptr);
      sharedNextOutput = (T*)*ptr;
      *ptr = nullptr;
    }
  }
  __syncthreads();

  uint64_t step = send->conn.step;
  step = ROUNDUP(step, ALLGATHER_CHUNKSTEPS);
  int poffset, noffset = (step%NCCL_STEPS)*stepSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  T * __restrict__ prevInput = (T*)recv->conn.buff;
  T * __restrict__ nextOutput = (T*)send->conn.buff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,args->nChannels));
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
  if (tid == 0) send->conn.step = step;
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
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  struct ncclConnector* recv = &channel->devPeers[ring->prev].recv;
  struct ncclConnector* send = &channel->devPeers[ring->next].send;
  volatile uint64_t * recvHeadPtr = recv->conn.llHead;
  volatile uint64_t * sendHeadPtr = send->conn.llHead;
  volatile int * sizesFifo = send->conn.llFifo;
  uint64_t sendHead = sendHeadPtr[0];

  typedef LLPrimitives<T, FUNC> LL;

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;

  uint64_t step = send->conn.llStep;
  uint32_t pflag, nflag = step + 1;
  int poffset, noffset = NCCL_LL_SLICE_LINES * STEP_TO_SLOT(step);

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  union ncclLLFifoLine * prevInput = (union ncclLLFifoLine *)recv->conn.llBuff;
  union ncclLLFifoLine * nextOutput = (union ncclLLFifoLine *)send->conn.llBuff;

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
          thisInput  + chunkOffset,
          nextOutput + noffset,
          maxOffset, nflag,
          tid, nthreads);
    } else {
      LL::ReduceCopy(
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
        prevInput  + poffset,
        thisOutput + offset,
        maxOffset, pflag,
        tid, nthreads);
    ACK_PREV;
  }

  FIFO_CLEANING_AND_SAVE_STEP(nflag);
}
