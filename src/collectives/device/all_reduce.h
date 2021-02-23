/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include <cassert>

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads-WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclRing* ring = &ncclShmem.channel->ring;
    int *ringUserRanks = ring->devUserRanks;
    const int stepSize = ncclShmem.comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
    const int nranks = ncclShmem.comm->nRanks;
    const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
    const ssize_t size = args->coll.count;

    ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
      prims(tid, nthreads, &ring->prev, &ring->next, stepSize, args->sendbuff, args->recvbuff);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
      ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ringUserRanks[nranks-1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.send(offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ringUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.recvReduceSend(offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringUserRanks[0];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.directRecvReduceCopySend(offset, offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ringUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.directRecvCopySend(offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ringUserRanks[1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      // Final wait/copy.
      prims.directRecv(offset, nelem);
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads-2*WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    const int stepSize = ncclShmem.comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    int chunkSize = args->coll.lastChunkSize;
    const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

#if 1
    if (tid < nthreads+WARP_SIZE) {
      ncclTree *tree = &ncclShmem.channel->tree;
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_DEV_ARITY, 1, 0, FUNC>
        prims(tid, nthreads, tree->down, &tree->up, stepSize, args->sendbuff, args->recvbuff);

      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceSend(offset, nelem);
        }
      }
    }


    if (tid < nthreads+WARP_SIZE) {
      ncclTree *tree = &ncclShmem.channel->tree;
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclPrimitives<UNROLL, 1, 1, T, 1, NCCL_MAX_DEV_ARITY, 1, FUNC>
        prims(tid, nthreads, &tree->up, tree->down, stepSize, args->sendbuff, args->recvbuff);

      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.directSendFromOutput(offset, offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.directRecvCopySend(offset, offset, nelem);
        }
      }
    }
#else
    int nthreadsSplit = nthreads/2;
    if (nthreadsSplit == 256) nthreadsSplit += 64;
    if (tree->up == -1) {
      if (tid < nthreads+WARP_SIZE) {
        // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
        ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_DEV_ARITY, NCCL_MAX_DEV_ARITY, 1, FUNC>
          prims(tid, nthreads, tree->down, tree->down, stepSize, args->sendbuff, args->recvbuff);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.directRecvReduceCopySend<Input, Output, /*DoPost=*/true>(offset, offset, offset, nelem);
        }
      }
    } else {
      if (tid < nthreadsSplit + WARP_SIZE) {
        // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
        ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_DEV_ARITY, 1, 0, FUNC>
          prims(tid, nthreadsSplit, tree->down, &tree->up, stepSize, args->sendbuff, args->recvbuff, 0);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Up
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.send(offset, nelem);
          } else {
            prims.recvReduceSend(offset, nelem);
          }
        }
      } else {
        // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
        ncclPrimitives<UNROLL, 1, 1, T, 1, NCCL_MAX_DEV_ARITY, 1, FUNC>
          prims(tid-nthreadsSplit-WARP_SIZE, nthreads-nthreadsSplit, &tree->up, tree->down, stepSize, args->sendbuff, args->recvbuff, 2);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Down
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.directRecv(offset, nelem);
          } else {
            prims.directRecvCopySend(offset, offset, nelem);
          }
        }
      }
    }
#endif
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads-WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    const int nRanks = ncclShmem.comm->nRanks;
    struct ncclTree* tree = &ncclShmem.channel->collTree;
    const int stepSize = ncclShmem.comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    int chunkSize = args->coll.lastChunkSize;
    const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    if (blockIdx.x < nChannels) { // first half of the channels do reduce
      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC>
        prims(tid, nthreads, tree->down, &tree->up, stepSize, args->sendbuff, args->recvbuff);
      if (tree->up == -1) {
        assert(0);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceCopy(offset, offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceSend(offset, nelem);
        }
      }
    }

    if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC>
        prims(tid, nthreads, &tree->up, tree->down, stepSize, args->sendbuff, args->recvbuff);

      // Compared to previous version of the code, these conditionals have been
      // manually hoisted outside of the loop. The extra insns should be negligible
      // while the benefits are reduced register pressure and fewer branches.
      if (tree->up == -1) {
        assert(0);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.sendFromOutput(offset, nelem); // should postOp=true???
        }
      }
      else if (tree->up == nRanks) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recvCopySend(offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recv(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.recvCopySend(offset, nelem);
        }
      }
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclRing* ring = &ncclShmem.channel->ring;
    int *ringUserRanks = ring->devUserRanks;
    const int stepLines = ncclShmem.comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);
    const int nranks = ncclShmem.comm->nRanks;
    const ssize_t loopSize = nChannels*nranks*chunkSize;
    const ssize_t size = args->coll.count;

    ncclLLPrimitives<T, FUNC, 1, 1> LLprims(
      tid, nthreads, &ring->prev, &ring->next, stepLines, args->sendbuff, args->recvbuff
    );

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ringUserRanks[nranks-1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.send(offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ringUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvReduceSend(offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringUserRanks[0];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceCopySend(offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ringUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvCopySend(offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ringUserRanks[1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      // Here we need to copy from buffer to this output.
      LLprims.recv(offset, nelem);
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclTree* tree = &ncclShmem.channel->tree;
    const int stepLines = ncclShmem.comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    do {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclLLPrimitives<T, FUNC, NCCL_MAX_DEV_ARITY, 1> LLprims
        (tid, nthreads, tree->down, &tree->up, stepLines, args->sendbuff, args->recvbuff);
      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.send(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.recvReduceSend(offset, nelem);
        }
      }
    } while(0);

    do {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclLLPrimitives<T, FUNC, 1, NCCL_MAX_DEV_ARITY> LLprims
        (tid, nthreads, &tree->up, tree->down, stepLines, args->sendbuff, args->recvbuff);
      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.sendFromOutput(offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.recv(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.recvCopySend(offset, nelem);
        }
      }
    } while(0);
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    const int nRanks = ncclShmem.comm->nRanks;
    struct ncclTree* tree = &ncclShmem.channel->collTree;
    const int stepLines = ncclShmem.comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    if (blockIdx.x < nChannels) { // first half of the channels do reduce
      ncclLLPrimitives<T, FUNC, 1, 1> LLprims
        (tid, nthreads, tree->down, &tree->up, stepLines, args->sendbuff, args->recvbuff);
      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.recvReduceCopy(offset, offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.send(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.recvReduceSend(offset, nelem);
        }
      }
    }

    if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
      ncclLLPrimitives<T, FUNC, 1, 1> LLprims
        (tid, nthreads, &tree->up, tree->down, stepLines, args->sendbuff, args->recvbuff);
      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.sendFromOutput(offset, nelem);
        }
      }
      else if (tree->up == nRanks) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.recvCopySend(offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.recv(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          LLprims.recvCopySend(offset, nelem);
        }
      }
    }
  }
};

#include "prims_ll128.h"
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclRing* ring = &ncclShmem.channel->ring;
    int *ringUserRanks = ring->devUserRanks;
    const int stepSize = ncclShmem.comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
    ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
    // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
    const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
    const int nranks = ncclShmem.comm->nRanks;
    const ssize_t loopSize = nChannels*nranks*chunkSize;
    const ssize_t size = args->coll.count;

    ncclLL128Primitives<T, FUNC, 1, 1> LLprims(
      tid, nthreads, &ring->prev, &ring->next, stepSize, args->sendbuff, args->recvbuff
    );

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ringUserRanks[nranks-1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.send(offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ringUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvReduceSend(offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringUserRanks[0];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceCopySend(offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ringUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvCopySend(offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ringUserRanks[1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      // Here we need to copy from buffer to this output.
      LLprims.recv(offset, nelem);
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclTree* tree = &ncclShmem.channel->tree;
    const int stepSize = ncclShmem.comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
    ssize_t chunkSize = args->coll.lastChunkSize;
    const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/8;
    const ssize_t loopSize = nChannels*chunkSize;
    int nthreadsSplit = NCCL_LL128_SPLIT(nthreads);
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    if (tree->up == -1) {
      // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
      ncclLL128Primitives<T, FUNC, NCCL_MAX_DEV_ARITY, NCCL_MAX_DEV_ARITY> LLprims
        (tid, nthreads, tree->down, tree->down, stepSize, args->sendbuff, args->recvbuff);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        LLprims.recvReduceCopySend(offset, offset, nelem, /*postOp=*/true);
      }
    } else {
      if (tid < nthreadsSplit) {
        // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
        ncclLL128Primitives<T, FUNC, NCCL_MAX_DEV_ARITY, 1> LLprims
          (tid, nthreadsSplit, tree->down, &tree->up, stepSize, args->sendbuff, args->recvbuff);
        if (tree->down[0] == -1) {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*chunkSize;
            int nelem = min(chunkSize, size-offset);
            LLprims.send(offset, nelem);
          }
        }
        else {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*chunkSize;
            int nelem = min(chunkSize, size-offset);
            LLprims.recvReduceSend(offset, nelem);
          }
        }
      } else {
        // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
        ncclLL128Primitives<T, FUNC, 1, NCCL_MAX_DEV_ARITY> LLprims
          (tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, stepSize, args->sendbuff, args->recvbuff);
        if (tree->down[0] == -1) {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*chunkSize;
            int nelem = min(chunkSize, size-offset);
            LLprims.recv(offset, nelem);
          }
        }
        else {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*chunkSize;
            int nelem = min(chunkSize, size-offset);
            LLprims.recvCopySend(offset, nelem);
          }
        }
      }
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
public:
  __device__ void run(struct ncclWorkElem* args) {}
};
