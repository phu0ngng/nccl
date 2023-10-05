/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclRing *ring = &ncclShmem.channel.ring;
    int const *ringRanks = ring->userRanks;
    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? REDUCESCATTER_CHUNKSTEPS : 1));
    // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
    const ssize_t minChunkSizeLL128 = int(nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2);
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->count;

    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, args->redOpArg);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      else if (Proto::Id == NCCL_PROTO_LL)
        realChunkSize = size-gridOffset < loopSize ? args->lastChunkSize : chunkSize;
      else if (Proto::Id == NCCL_PROTO_LL128)
        realChunkSize = min(divUp(size-gridOffset, nChannels*minChunkSizeLL128)*minChunkSizeLL128, chunkSize);
      realChunkSize = int(realChunkSize);

      ssize_t chunkOffset = gridOffset + bid*int(realChunkSize);

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
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL128>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const ssize_t chunkSize = int(args->lastChunkSize);
    const ssize_t size = args->count;
    const ssize_t loopSize = nChannels*chunkSize;
    const int rank = ncclShmem.comm.rank;
    const int nranks = ncclShmem.comm.nRanks;

    /* if we are direct NVLS, we only need to allocate 1 warp to scatter for sync; 
     * if not, based on #ranks, we allocate 7 or 5 warps to reduce to saturate bandwidth
     * and the rest are allocated to scatter. */
    const int nThreadsReduce = args->regUsed ? (NCCL_MAX_NTHREADS - WARP_SIZE) : (nranks <= 6 ? 7 * WARP_SIZE : 5 * WARP_SIZE);
    const int nThreadsScatter = args->regUsed ? WARP_SIZE : (NCCL_MAX_NTHREADS - nThreadsReduce);
    const int tidEndScatter = nThreadsScatter;
    const int tidEndReduce = tidEndScatter + nThreadsReduce;

    if (!args->regUsed) {
      if (tid < tidEndScatter) {
        // Scatter
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsScatter, NULL, nvls->up, args->sendbuff, NULL,
            args->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.scatter(offset, nvls->nHeads * size, nelem, size, -1, 0);
        }
      } else if (tid < tidEndReduce) {
        // Reduce through NVLS
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
        Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndScatter, nThreadsReduce, &nvls->down, NULL, NULL, args->recvbuff,
            args->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.recv(offset, nelem);
        }
      }
    } else {
      if (tid < tidEndScatter) {
        // Scatter
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanSymmetric<NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsScatter, nvls->up, nvls->up, NULL, NULL,
            args->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          prims.scatter(0, 0, 0, 0, -1, 0);
        }

        /* gather used as sync */
        prims.gather(0, 0, 0, 0, -1, 0);
      } else if (tid < tidEndReduce) {
        // Reduce through NVLS
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndScatter, nThreadsReduce, &nvls->down, &nvls->down, NULL, args->recvbuff,
            args->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0, args);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t outOffset = gridOffset + bid * chunkSize;
          ssize_t inpOffset = outOffset + rank * size;
          int nelem = min(chunkSize, size - outOffset);
          prims.directRecvCopy(inpOffset, outOffset, nelem);
        }

        /* send for sync */
        prims.send(0, 0);
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  template<bool ReduceSendNotRecv>
  struct Scatterer {
    struct ncclWorkElem* args;
    int chunkSize;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes
      ) {
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = direct->nHeads;
      int bid = args->bid;
      void* inbuf = (void*)args->sendbuff;
      ssize_t sizePerRank = args->count;

      ssize_t railAllBeg = min(railGridOffset + bid*chunkSize, nNodes*sizePerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes*sizePerRank);
      int railAllSize = railAllEnd - railAllBeg;
      if (tid < nDsts) dstSizes[tid] = railAllSize;

      int dst = 0;
      int rail;
      if (!ReduceSendNotRecv) {
        rail = direct->headRank;
      } else {
        rail = direct->headRank+1;
        if (rail == nRails) rail = 0;
      }
      do {
        int node = railAllBeg/sizePerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node*sizePerRank;
          ssize_t railOneEnd = railOneBeg + sizePerRank;
          ssize_t railOneOffset = (railAllBeg+railAllOffset) - railOneBeg;
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg+railAllOffset);
          int rank = ncclShmem.comm.collNetDenseToUserRank[node*nRails + rail];
          ssize_t userOneBeg = rank*sizePerRank + railOneOffset;
          reduceCopy<ncclCollUnroll(), RedOp, T,
                     /*MultimemSrcs=*/0, 1+MinSrcs, 1+MaxSrcs,
                     /*MultimemDsts,MinDsts,MaxDsts=*/0,1,1,
                     /*PreOpSrcs=*/1>
            (tid, tn, args->redOpArg, &args->redOpArg, false,
             /*nSrcs=*/1+nSrcs, [=]__device__(int s) {
               return s==0 ? (T*)inbuf + userOneBeg
                           : (T*)srcPtrs[s-1] + railAllOffset;
             },
             /*nDsts=*/1, [=]__device__(int d/*==0*/) {
               return (T*)dstPtrs[dst] + railAllOffset;
             },
             delta);
          railAllOffset += delta;
          node += 1;
        }
        dst += 1;
        rail += 1;
        if (rail == nRails) rail = 0;
      } while (ReduceSendNotRecv && dst < nRails-1);
    }
  };

  __device__ __forceinline__ void run(ncclWorkElem *args) {
    int tid = threadIdx.x;
    const int nChannels = args->nChannels;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    int const &nNodes = ncclShmem.comm.nNodes;
    ssize_t chunkSize = int(args->lastChunkSize);
    ssize_t sizePerRank = args->count;

    if (direct->out == -1) __trap();
    bool isMultiRail = (direct->nHeads > 1);
    int nWarps1 = (isMultiRail ? 2 : 0);
    int nWarps2 = (isMultiRail ? 2 : 1);
    int nWarps3 = 1;
    float denom = float(args->nWarps)/float(nWarps1+nWarps2+nWarps3);
    nWarps3 = int(denom*nWarps3);
    nWarps2 = int(denom*nWarps2);
    nWarps1 = args->nWarps - (nWarps2+nWarps3);

    using Proto = ProtoSimple<1, 1>;

    int tn = nWarps1*WARP_SIZE;
    if (tid < tn) {
      // Phase 1: Scatter inputs to peers
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid, tn, nullptr, direct->heads+1, nullptr, nullptr,
              /*redOpArg=*/0, 0*Proto::MaxGroupWidth, 1, 1);
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*sizePerRank; railGridOffset += nChannels*chunkSize) {
        Scatterer</*ReduceSendNotRecv=*/true> scat;
        scat.args = args;
        scat.chunkSize = chunkSize;
        scat.railGridOffset = railGridOffset;
        prims.process</*Recv=*/0, /*Send=*/1>(scat);
      }
      return;
    }
    tid -= tn;

    tn = nWarps2*WARP_SIZE;
    if (tid < tn) {
      // Phase 2: Reduce from peers + local input -> send to network
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0>
        prims(tid, tn, direct->heads+1, &direct->out, nullptr, nullptr,
              /*redOpArg=*/0, 1*Proto::MaxGroupWidth, 1, 1);
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*sizePerRank; railGridOffset += nChannels*chunkSize) {
        Scatterer</*ReduceSendNotRecv=*/false> scat;
        scat.args = args;
        scat.chunkSize = chunkSize;
        scat.railGridOffset = railGridOffset;
        prims.process</*Recv=*/1, /*Send=*/1>(scat);
      }
      return;
    }
    tid -= tn;

    tn = nWarps3*WARP_SIZE;
    if (tid < tn) {
      // Phase 3: recv from network
      Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
        prims(tid, tn, &direct->out, nullptr, nullptr, args->recvbuff,
              args->redOpArg, 2*Proto::MaxGroupWidth, 0, 0);
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*sizePerRank; railGridOffset += nChannels*chunkSize) {
        ssize_t railAllBeg = railGridOffset + args->bid*chunkSize;
        ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes*sizePerRank);
        ssize_t railOneBeg = ncclShmem.comm.node*sizePerRank;
        ssize_t railOneEnd = railOneBeg + sizePerRank;
        ssize_t beg = max(railAllBeg, railOneBeg);
        ssize_t end = min(railAllEnd, railOneEnd);
        prims.recv(beg-railOneBeg, max(ssize_t(0), end-beg), /*postOp=*/true);
      }
      return;
    }
  }
};
