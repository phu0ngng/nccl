/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    ncclRing *ring = &ncclShmem.channel.ring;
    int ringIx = ring->index;
    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t loopSize = nChannels*nranks*chunkSize;
    const ssize_t size = args->coll.count;

    int minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL)
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    if (Proto::Id == NCCL_PROTO_LL128) {
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2;
    }

    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto> prims
      (tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*nranks));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      else
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize);
      realChunkSize = int(realChunkSize);

      auto calcOffset = [&]__device__(int chunk)->ssize_t {
        if (Proto::Id == NCCL_PROTO_SIMPLE)
          return gridOffset + bid*nranks*realChunkSize + chunk*realChunkSize;
        else
          return gridOffset + (chunk*nChannels + bid)*realChunkSize;
      };
      auto modRanks = [&]__device__(int r)->int {
        return r - (r >= nranks ? nranks : 0);
      };

      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = modRanks(ringIx + nranks-1);
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);
      prims.send(offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = modRanks(ringIx + nranks-j);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size-offset);
        prims.recvReduceSend(offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringIx + 0;
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);
      prims.directRecvReduceCopySend(offset, offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = modRanks(ringIx + nranks-j);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size-offset);
        prims.directRecvCopySend(offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = modRanks(ringIx + 1);
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);
      prims.directRecv(offset, nelem);
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ void runTreeUpDown(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    ncclTree *tree = &ncclShmem.channel.tree;
    ssize_t chunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? args->coll.lastChunkSize
                   /* LL & LL128 */  : Proto::calcBytePerStep()/sizeof(T));
    const ssize_t minChunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? (nthreads - WARP_SIZE)*8*(sizeof(uint64_t)/sizeof(T))
                   /* LL & LL128 */  : nthreads*(Proto::calcBytePerGrain()/sizeof(T)));
    const ssize_t loopSize = int(nChannels*chunkSize);
    const ssize_t size = args->coll.count;

    if (loopSize > size)
      chunkSize = divUp((int)size, int(nChannels*minChunkSize))*int(minChunkSize);

    { // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, 0, Proto> prims
        (tid, nthreads, tree->down, &tree->up, args->sendbuff, args->recvbuff);
      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceSend(offset, nelem);
        }
      }
    }

    { // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, 1, Proto> prims
        (tid, nthreads, &tree->up, tree->down, args->sendbuff, args->recvbuff);
      if (tree->up == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directSendFromOutput(offset, offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecvCopySend(offset, offset, nelem);
        }
      }
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ void runTreeSplit(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    ncclTree *tree = &ncclShmem.channel.tree;
    ssize_t chunkSize = int(
      Proto::Id != NCCL_PROTO_LL ? args->coll.lastChunkSize
                                 : Proto::calcBytePerStep()/sizeof(T));
    const ssize_t minChunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? (nthreads - 2*WARP_SIZE)*8*(sizeof(uint64_t)/sizeof(T)) :
      Proto::Id == NCCL_PROTO_LL     ? nthreads*(Proto::calcBytePerGrain()/sizeof(T))
                   /* LL128 */       : nthreads*(Proto::calcBytePerGrain()/sizeof(T))/8);
    const ssize_t loopSize = int(nChannels*chunkSize);
    const ssize_t size = args->coll.count;

    int nthreadsSplit;
    if (Proto::Id == NCCL_PROTO_SIMPLE)
      nthreadsSplit = nthreads/2;
    else { // LL & LL128
      // Receiving from up to 3 sources is more compute intensive than sending
      // to 3 dests. Use 70% for reduce and 30% for bcast.
      nthreadsSplit = (nthreads*7/(10*32))*32;
    }

    if (loopSize > size)
      chunkSize = divUp((int)size, nChannels*int(minChunkSize))*int(minChunkSize);

    if (tree->up == -1) {
      // Reduce and broadcast. Max number of recv is 3, max number of send is 3
      Primitives<T, RedOp, FanSymmetric<NCCL_MAX_DEV_ARITY>, 1, Proto>
        prims(tid, nthreads, tree->down, tree->down, args->sendbuff, args->recvbuff);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*int(chunkSize);
        int nelem = min(chunkSize, size-offset);
        prims.directRecvReduceCopySend(offset, offset, offset, nelem, /*doPost=*/true);
      }
    }
    else if (tid < nthreadsSplit) {
      // Reduce up. Max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, 0, Proto>
        prims(tid, nthreadsSplit, tree->down, &tree->up, args->sendbuff, args->recvbuff, 0*Proto::MaxGroupWidth);
      if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.send(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.recvReduceSend(offset, nelem);
        }
      }
    }
    else {
      // Broadcast down. Max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, 1, Proto>
        prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, args->sendbuff, args->recvbuff, 1*Proto::MaxGroupWidth);
      if (tree->down[0] == -1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*int(chunkSize);
          int nelem = min(chunkSize, size-offset);
          prims.directRecvCopySend(offset, offset, nelem);
        }
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE> {
  __device__ void run(ncclWorkElem *args) {
    runTreeSplit<T, RedOp, ProtoSimple<1, 1>>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE> {
  __device__ void run(ncclWorkElem *args) {
    static constexpr int COLLNET_COPY_THREADS = 96;
    const int tid = threadIdx.x;
    //const int nthreads = args->nThreads-3*WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    //const int nRanks = ncclShmem.comm->nRanks;
    struct ncclDirect* tree = &ncclShmem.channel.collTree;
    const ssize_t chunkSize = int(args->coll.lastChunkSize);
    const ssize_t size = args->coll.count;
    const ssize_t loopSize = nChannels*tree->nHeads*chunkSize;

    const int hasUp = (tree->up[0] >= 0) ? 1 : 0;
    const int hasDn = (tree->down[0] >= 0) ? 1 : 0;
    const int nThreadsScatter = (hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 3*COLLNET_COPY_THREADS : 0;
    const int nThreadsGather  = (hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 2*COLLNET_COPY_THREADS : 0;
    const int nThreadsBcast   = (hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 0 : 2*COLLNET_COPY_THREADS;
    // Gather does not need sync threads, sparing one more warp for reduce
    const int nThreadsReduce = NCCL_SIMPLE_MAX_NTHREADS + WARP_SIZE - nThreadsScatter - nThreadsGather - nThreadsBcast;
    const int tidStartBcast = nThreadsGather;
    const int tidStartScatter = tidStartBcast + nThreadsBcast + WARP_SIZE;
    const int tidStartReduce = tidStartScatter + nThreadsScatter + WARP_SIZE;

    using Proto = ProtoSimple<1, 1>;

    if (tid >= tidStartScatter && tid < tidStartReduce && hasUp) {
      // Scatter
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, 0, Proto>
        prims(tid-tidStartScatter, nThreadsScatter+WARP_SIZE, NULL, tree->up, args->sendbuff, args->recvbuff, 2*Proto::MaxGroupWidth);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*tree->nHeads*chunkSize;
        int nelem = min(tree->nHeads*chunkSize, size-offset);
        prims.scatter(offset, nelem, chunkSize, tree->headRank, tree->shift);
      }
    } else if (tid >= tidStartReduce && tree->out != -1) {
      // Reduce, send to network
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, 0, Proto>
        prims(tid-tidStartReduce, nThreadsReduce+WARP_SIZE, tree->down, &tree->out, args->sendbuff, args->recvbuff, 3*Proto::MaxGroupWidth);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + (bid*tree->nHeads+tree->headRank)*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (hasDn) {
          prims.recvReduceSend(offset, nelem);
        } else {
          prims.send(offset, nelem);
        }
      }
    } else if (tid < tidStartBcast && hasUp) {
      // Gather
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, 0, Proto>
        prims(tid, nThreadsGather+WARP_SIZE, tree->up, NULL, args->sendbuff, args->recvbuff, 0*Proto::MaxGroupWidth);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*tree->nHeads*chunkSize;
        int nelem = min(tree->nHeads*chunkSize, size-offset);
        prims.gather(offset, nelem, chunkSize, tree->headRank, tree->shift);
      }
    } else if (tid >= tidStartBcast && tid < tidStartScatter && tree->out != -1) {
      // Recv from network, broadcast
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, 0, Proto>
        prims(tid-tidStartBcast, nThreadsBcast+WARP_SIZE, &tree->out, tree->down, args->sendbuff, args->recvbuff, 1*Proto::MaxGroupWidth);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + (bid*tree->nHeads+tree->headRank)*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (hasDn) {
          prims.recvCopySend(offset, nelem, /*postOp=*/true);
        } else {
          prims.recv(offset, nelem, /*postOp=*/true);
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL> {
  __device__ void run(ncclWorkElem *args) {
    runTreeSplit<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL128>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL128> {
  __device__ void run(ncclWorkElem *args) {
    runTreeSplit<T, RedOp, ProtoLL128>(args);
  }
};
