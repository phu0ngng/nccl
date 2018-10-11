/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceRingKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x - 1;
  const int bid = args->bid;
  struct ncclComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  struct ncclConnInfo* recv = &channel->devPeers[ring->prev].recv.conn;
  struct ncclConnInfo* send = &channel->devPeers[ring->next].send.conn;
  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, FUNC>
    prims(tid, nthreads, stepSize, recv, send, thisOutput, args->comm->abortFlag);

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int slice;

    // step 0: push data to next GPU
    slice = ring->devUserRanks[nranks-1];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    prims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.directRecvCopySend(thisOutput+offset, offset, nelem);
    }

    // Make final copy from buffer to dest.
    slice = ring->devUserRanks[1];
    offset = chunkOffset + slice * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    // Final wait/copy.
    prims.directRecv(thisOutput+offset, offset, nelem);
  }
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceTreeKernel(struct CollectiveArgs* args) {
}

#include "ll_kernel.h"

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceLLRingKernel(struct CollectiveArgs* args) {
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
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  const int nringsXnranks = args->nChannels * nranks;
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
    int nelem;
    int slice;

    // step 0: push data to next GPU
    slice = ring->devUserRanks[nranks-1];
    offset = chunkOffset + slice * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring->devUserRanks[0];
    offset = chunkOffset + slice * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      slice = ring->devUserRanks[nranks-j];
      offset = chunkOffset + slice * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // Make final copy from buffer to dest.
    slice = ring->devUserRanks[1];
    offset = chunkOffset + slice * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(thisOutput+offset, nelem);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceLLTreeKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclTree* tree = &channel->tree;

  // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
  // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
  struct ncclConnInfo* upSend = tree->nUp ? &channel->devPeers[tree->up].send.conn : NULL;
  struct ncclConnInfo* dnRecv = tree->nUp ? &channel->devPeers[tree->up].recv.conn : NULL;
  struct ncclConnInfo* upRecvs[3];
  struct ncclConnInfo* dnSends[3];
  for (int i=0; i<tree->nDown; i++) {
    upRecvs[i] = &channel->devPeers[tree->down[i]].recv.conn;
    dnSends[i] = &channel->devPeers[tree->down[i]].send.conn;
  }

  const ssize_t size = args->N;

  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  do {
    ncclLLPrimitives<T, FUNC, 3, 1> LLprims(tid, nthreads, tree->nDown, upRecvs, tree->nUp, &upSend, comm->abortFlag);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->nUp == 0) {
        LLprims.recvReduce(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->nDown) {
        LLprims.recvReduceSend(thisInput+offset, nelem);
      } else {
        LLprims.send(thisInput + offset, nelem);
      }
    }
  } while(0);

  do {
    ncclLLPrimitives<T, FUNC, 1, 3> LLprims(tid, nthreads, tree->nUp, &dnRecv, tree->nDown, dnSends, comm->abortFlag);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->nUp == 0) {
        LLprims.send(thisOutput+offset, nelem);
      } else if (tree->nDown) {
        LLprims.recvCopySend(thisOutput + offset, nelem);
      } else {
        LLprims.recv(thisOutput + offset, nelem);
      }
    }
  } while(0);
}
