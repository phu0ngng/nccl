/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads-WARP_SIZE;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclRing* ring = &ncclShmem.channel->ring;
      int const *ringRanks = ring->devUserRanks;
      const int stepSize = ncclShmem.comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * REDUCESCATTER_CHUNKSTEPS;
      const int nranks = ncclShmem.comm->nRanks;
      const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
      const ssize_t size = args->coll.count;

      ncclPrimitives<UNROLL, REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, T, 1, 1, 0, FUNC>
        prims(tid, nthreads, &ring->prev, &ring->next, stepSize, args->sendbuff, args->recvbuff);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*realChunkSize;

        /////////////// begin ReduceScatter steps ///////////////
        ssize_t offset;
        int nelem = min(realChunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ringRanks[nranks-1];
        offset = chunkOffset + rankDest * size;

        prims.send(offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j=2; j<nranks; ++j) {
          rankDest = ringRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          prims.recvReduceSend(offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final result
        rankDest = ringRanks[0];
        offset = chunkOffset + rankDest * size;

        prims.recvReduceCopy(offset, chunkOffset, nelem, /*postOp=*/true);
      }
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclRing* ring = &ncclShmem.channel->ring;
      int const *ringRanks = ring->devUserRanks;
      const int stepLines = ncclShmem.comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
      const int nranks = ncclShmem.comm->nRanks;
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;

      ncclLLPrimitives<T, FUNC, 1, 1> LLprims(
        tid, nthreads, &ring->prev, &ring->next, stepLines, args->sendbuff, args->recvbuff
      );

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        if (size-gridOffset < loopSize) {
          chunkSize = args->coll.lastChunkSize;
        }
        ssize_t chunkOffset = gridOffset + bid*chunkSize;

        /////////////// begin ReduceScatter steps ///////////////
        ssize_t offset;
        int nelem = min(chunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ringRanks[nranks-1];
        offset = chunkOffset + rankDest * size;

        LLprims.send(offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j=2; j<nranks; ++j) {
          rankDest = ringRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          LLprims.recvReduceSend(offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final
        // result that we store in this data
        rankDest = ringRanks[0];
        offset = chunkOffset + rankDest * size;

        LLprims.recvReduceCopy(offset, chunkOffset, nelem, /*postOp=*/true);
      }
    }
};

#include "prims_ll128.h"
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclRing* ring = &ncclShmem.channel->ring;
      int const *ringRanks = ring->devUserRanks;
      const int stepSize = ncclShmem.comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
      const int nranks = ncclShmem.comm->nRanks;
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;

      ncclLL128Primitives<T, FUNC, 1, 1> LLprims(
        tid, nthreads, &ring->prev, &ring->next, stepSize, args->sendbuff, args->recvbuff
      );

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        chunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);

        ssize_t chunkOffset = gridOffset + bid*chunkSize;

        /////////////// begin ReduceScatter steps ///////////////
        ssize_t offset;
        int nelem = min(chunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ringRanks[nranks-1];
        offset = chunkOffset + rankDest * size;

        LLprims.send(offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j=2; j<nranks; ++j) {
          rankDest = ringRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          LLprims.recvReduceSend(offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final
        // result that we store in this data
        rankDest = ringRanks[0];
        offset = chunkOffset + rankDest * size;

        LLprims.recvReduceCopy(offset, chunkOffset, nelem, /*postOp=*/true);
      }
    }
};

template<int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_TREE, PROTO, REDOP, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

template<int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_COLLNET, PROTO, REDOP, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};
