/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"
#include "prims_ll.h"
#include "prims_ll128.h"

template<class FUNC, typename T, int UNROLL>
struct ncclFunctionWorkElem<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
    __device__ void run(ncclWorkElem *args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads-WARP_SIZE;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclRing* ring = &ncclShmem.channel->ring;
      const int stepSize = ncclShmem.comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * ALLGATHER_CHUNKSTEPS;
      const int nranks = ncclShmem.comm->nRanks;
      const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
      const ssize_t size = args->coll.count;

      T *inputBuf = (T*)args->sendbuff;
      T *outputBuf = (T*)args->recvbuff;
      ncclPrimitives<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, T, 1, 1, 1, FUNC>
        prims(tid, nthreads, &ring->prev, &ring->next, stepSize, inputBuf, outputBuf);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*realChunkSize;

        /////////////// begin AllGather steps ///////////////
        ssize_t offset;
        int nelem = min(realChunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        if (inputBuf + chunkOffset == outputBuf + offset) { // In place
          prims.directSend(chunkOffset, offset, nelem);
        } else {
          prims.directCopySend(chunkOffset, offset, offset, nelem);
        }

        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          prims.directRecvCopySend(offset, offset, nelem);
        }

        // Make final copy from buffer to dest.
        rankDest = ring->devUserRanks[1];
        offset = chunkOffset + rankDest * size;

        // Final wait/copy.
        prims.directRecv(offset, nelem);
      }
    }
};

template<class FUNC, typename T, int UNROLL>
struct ncclFunctionWorkElem<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
    __device__ void run(ncclWorkElem *args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclRing* ring = &ncclShmem.channel->ring;
      const int stepLines = ncclShmem.comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
      const int nranks = ncclShmem.comm->nRanks;
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;

      T *inputBuf = (T*)args->sendbuff;
      T *outputBuf = (T*)args->recvbuff;
      ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepLines, inputBuf, outputBuf);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        if (size-gridOffset < loopSize) {
          chunkSize = args->coll.lastChunkSize;
        }
        ssize_t chunkOffset = gridOffset + bid*chunkSize;

        /////////////// begin AllGather steps ///////////////
        ssize_t offset;
        int nelem = min(chunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        if (inputBuf + chunkOffset == outputBuf + offset) { // In place
          LLprims.send(chunkOffset, nelem);
        } else {
          LLprims.copySend(chunkOffset, offset, nelem);
        }

        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          LLprims.recvCopySend(offset, nelem);
        }

        // step k-1: final store
        rankDest = ring->devUserRanks[1];
        offset = chunkOffset + rankDest * size;

        LLprims.recv(offset, nelem);
      }
    }
};

template<class FUNC, typename T, int UNROLL>
struct ncclFunctionWorkElem<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
    __device__ void run(ncclWorkElem *args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclRing* ring = &ncclShmem.channel->ring;
      const int stepSize = ncclShmem.comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
      const int nranks = ncclShmem.comm->nRanks;
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;

      T *inputBuf = (T*)args->sendbuff;
      T *outputBuf = (T*)args->recvbuff;
      ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepSize, inputBuf, outputBuf);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        chunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);

        ssize_t chunkOffset = gridOffset + bid*chunkSize;

        /////////////// begin AllGather steps ///////////////
        ssize_t offset;
        int nelem = min(chunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        if (inputBuf + chunkOffset == outputBuf + offset) { // In place
          LLprims.send(chunkOffset, nelem);
        } else {
          LLprims.copySend(chunkOffset, offset, nelem);
        }

        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          LLprims.recvCopySend(offset, nelem);
        }

        // step k-1: final store
        rankDest = ring->devUserRanks[1];
        offset = chunkOffset + rankDest * size;

        LLprims.recv(offset, nelem);
      }
    }
};
