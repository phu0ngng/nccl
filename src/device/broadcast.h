/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
#ifdef ALLGATHERV_IMPL
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void setDataPtrsHelper(Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0, 0>& prims,
                                                    void const* srcBuf, void* dstBuf, uint64_t redOpArg) {
    prims.setDataPtrs(srcBuf, dstBuf);
  }

  template<typename T, typename RedOp>
  __device__ __forceinline__ void setDataPtrsHelper(Primitives<T, RedOp, FanSymmetric<1>, 0, ProtoSimple<1,1>, 0, 0>& prims,
                                                    void const* srcBuf, void* dstBuf, uint64_t redOpArg) {
    prims.setDataPtrs(srcBuf, dstBuf, redOpArg, nullptr, 0, 0);
  }
#endif
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    const int rank = ring->userRanks[0];
    const int nextRank = ring->userRanks[1];
    const int root = work->root;
    ssize_t chunkCount;
    ssize_t channelCount;
    ssize_t gridOffset;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;
    int workNthreads;
    bool isNetOffload = work->isOneRPN && work->netRegUsed;

    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;
    workNthreads = isNetOffload ? WARP_SIZE : nthreads;

    if (tid < workNthreads) {
      // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
      // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0>
        prims(tid, workNthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, 0, 0, work);

      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);

        if (rank == root) {
          if (inputBuf == outputBuf || isNetOffload) {
            prims.directSend(offset, offset, nelem);
          } else {
            prims.directCopySend(offset, offset, nelem);
          }
        } else if (nextRank == root) {
          prims.directRecv(offset, nelem);
        } else {
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
      }
    } else if (inputBuf != outputBuf && rank == root) {
      inputBuf = inputBuf + gridOffset;
      outputBuf = outputBuf + gridOffset;
      reduceCopy<COLL_UNROLL, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>
        (tid - workNthreads, nthreads - workNthreads, work->redOpArg, &work->redOpArg, false, 1, (void**)&inputBuf, 1, (void**)&outputBuf, channelCount);
    }
    if (isNetOffload) barrier_sync(14, nthreads);
  }
}

#ifdef ALLGATHERV_IMPL
template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBcast() {
  int tid = threadIdx.x;
  int tn = blockDim.x;
  ncclRing* ring = &ncclShmem.channel.ring;


  ncclDevWorkBcast *works = (ncclDevWorkBcast*)ncclShmem.workStorage;
  Primitives<int8_t, FuncSum<int8_t>, FanSymmetric<1>, /*Direct=*/0, Proto, 0>
    prims(tid, tn, &ring->prev, &ring->next, nullptr, nullptr, /*redOpArg=*/0);
  int w = 0; // `works[]` index of the current broadcaster.
  int wPrev = -1; // Value of `w` for previous loop iteration.
  bool wPrevIsEmpty = false; // Local cache of `works[wPrev].bytes==0`

  while (true) {
    int nWorks = ncclShmem.nWorks;
    int nRanks = ncclShmem.comm.nRanks;
    size_t bytes = works[w].bytes;
    int ringDepth = works[w].ringDepth; // How far down the ring am I from this broadcaster
    void* srcBuf = ringDepth==0 ? works[w].coll.sendbuff : nullptr;   // related to channel ID
    void* dstBuf = works[w].coll.recvbuff;
    bool inPlace = srcBuf == dstBuf;
    setDataPtrsHelper(prims, (void const*)srcBuf, (void *)dstBuf, works[w].coll.redOpArg);

    int wNext = (w+1 == nWorks) ? 0 : w+1;
    while ((wNext==wPrev) ? wPrevIsEmpty : (works[wNext].bytes == 0)) {
      wNext = (wNext+1 == nWorks) ? 0 : wNext+1;
    }

    int ringGap = INT_MAX;
    if (w != wNext) {
      ringGap = works[wNext].ringDepth - ringDepth;
      if (ringGap <= 0) ringGap += nRanks;
    }
    size_t offset = 0;
    // sync-process offset from 0 -> works[w].bytes, each send/recv = chunkBytes
    do {
      int chunkBytes = works[w].chunksize;
      int delta = min(bytes, (size_t)chunkBytes);   // how many elements (uint8) we are gonna to process. less than chunkBytes

      if (ringDepth == 0) { // This rank is broadcast root
        if (inPlace) {

          prims.send(offset, delta); // barrier
        } else {
          prims.copySend(offset, offset, delta); // barrier
        }
      } else if (ringDepth == nRanks-1) { // Downstream neighbor is broadcast root
        prims.recv(offset, delta); // barrier
      } else {
        prims.recvCopySend(offset, delta); // barrier
      }
      bytes -= delta;
      offset += delta;
      ringGap -= 1;
    } while (ringGap != 0 && bytes != 0);
    // ...a thread barrier has happened.

    // If the next broadcaster is the current one, then we just completed all
    // work since ringGap was INT_MAX.
    if (wNext == w) {
      break;
    }

    if (tid == 0) {
      // Since there was a thread barrier in the primitive operation above, we
      // know that these writes to shmem are safe since all threads have moved
      // on to reading from works[wNext], and we know wNext cannot be equal to w.
      works[w].bytes = bytes;
      works[w].coll.recvbuff = (char*)works[w].coll.recvbuff + offset;
      if (ringDepth == 0) {
        works[w].coll.sendbuff = (char*)works[w].coll.sendbuff + offset;
      }
    }
    // The `tid!=0` threads in the next loop iteration must avoid racing with
    // `tid==0` above since it is updating `works[wPrev]`. From the perspective
    // of the next iteration, the threads are interested in `works[wPrev].bytes==0`,
    // so we compute it in a local boolean here to be carried forward.
    wPrevIsEmpty = (bytes == 0);
    wPrev = w;
    w = wNext;
  }
}

// Specialized for broadcast
template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run() {
    using Proto = ProtoSimple<1,1>;
    runBcast<T, RedOp, Proto>();  // TODO: think if we need to specialize proto
  }
};
template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run() {
    runBcast<T, RedOp, ProtoLL>();  // TODO: think if we need to specialize proto
  }
};
template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run() {
    runBcast<T, RedOp, ProtoLL128>();  // TODO: think if we need to specialize proto
  }
};
#endif


template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
