/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run() {
      ncclWorkElem *args = &ncclShmem.work.elems[0];
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads-WARP_SIZE;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclRing* ring = &ncclShmem.channel->ring;
      const int stepSize = ncclShmem.comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * BROADCAST_CHUNKSTEPS;
      const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
      const ssize_t size = args->coll.count;
      const int rank = ring->devUserRanks[0];
      const int nextRank = ring->devUserRanks[1];
      const int root = args->coll.root;

      T *inputBuf = (T*)args->sendbuff;
      T *outputBuf = (T*)args->recvbuff;
      ncclPrimitives<UNROLL, BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS, T, 1, 1, 0, FUNC>
        prims(tid, nthreads, &ring->prev, &ring->next, stepSize, inputBuf, outputBuf);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t offset = gridOffset + bid*realChunkSize;
        int nelem = min(realChunkSize, size-offset);

        if (rank == root) {
          if (inputBuf == outputBuf) {
            prims.send(offset, nelem);
          } else {
            prims.copySend(offset, offset, nelem);
          }
        } else if (nextRank == root) {
          prims.recv(offset, nelem);
        } else {
          prims.recvCopySend(offset, nelem);
        }
      }
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run() {
      ncclWorkElem *args = &ncclShmem.work.elems[0];
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclRing* ring = &ncclShmem.channel->ring;
      const int stepLines = ncclShmem.comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;
      const int rank = ring->devUserRanks[0];
      const int nextRank = ring->devUserRanks[1];
      const int root = args->coll.root;

      T *inputBuf = (T*)args->sendbuff;
      T *outputBuf = (T*)args->recvbuff;
      ncclLLPrimitives<T, FUNC, 1, 1> LLprims(
        tid, nthreads, &ring->prev, &ring->next, stepLines, inputBuf, outputBuf
      );

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        if (size-gridOffset < loopSize) {
          chunkSize = args->coll.lastChunkSize;
        }
        ssize_t offset = gridOffset + bid*chunkSize;

        int nelem = min(chunkSize, size-offset);
        if (rank == root) {
          if (inputBuf == outputBuf) {
            LLprims.send(offset, nelem);
          } else {
            LLprims.copySend(offset, offset, nelem);
          }
        } else if (nextRank == root) {
          LLprims.recv(offset, nelem);
        } else {
          LLprims.recvCopySend(offset, nelem);
        }
      }
    }
};

#include "prims_ll128.h"
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run() {
      ncclWorkElem *args = &ncclShmem.work.elems[0];
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclRing* ring = &ncclShmem.channel->ring;
      const int stepSize = ncclShmem.comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;
      const int rank = ring->devUserRanks[0];
      const int nextRank = ring->devUserRanks[1];
      const int root = args->coll.root;

      T *inputBuf = (T*)args->sendbuff;
      T *outputBuf = (T*)args->recvbuff;
      ncclLL128Primitives<T, FUNC, 1, 1> LLprims(
        tid, nthreads, &ring->prev, &ring->next, stepSize, inputBuf, outputBuf
      );

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        chunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);
        ssize_t offset = gridOffset + bid*chunkSize;

        int nelem = min(chunkSize, size-offset);
        if (rank == root) {
          if (inputBuf == outputBuf) {
            LLprims.send(offset, nelem);
          } else {
            LLprims.copySend(offset, offset, nelem);
          }
        } else if (nextRank == root) {
          LLprims.recv(offset, nelem);
        } else {
          LLprims.recvCopySend(offset, nelem);
        }
      }
    }
};

template<int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction<ncclFuncBroadcast, NCCL_ALGO_TREE, PROTO, REDOP, T, UNROLL> {
  public:
    __device__ void run() {}
};

template<int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction<ncclFuncBroadcast, NCCL_ALGO_COLLNET, PROTO, REDOP, T, UNROLL> {
  public:
    __device__ void run() {}
};
