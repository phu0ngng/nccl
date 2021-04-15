/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <cassert>
#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common.h"

#define NCCL_SPINS_BEFORE_CHECK_ABORT 1000000

// Unroll unconditionally the first send/recv since nsend/nrecv should be at
// least 1 if SEND/RECV is set.
#define FOR_SEND(func, ...) do { \
  if (SEND) { \
    /* Send to far first, then close */ \
    for (int i=1; i<NSEND && i<nsend; i++) func(i, ##__VA_ARGS__); \
    func(0, ##__VA_ARGS__); \
  } \
} while (0)

#define FOR_RECV(func, ...) do { \
  if (RECV) { \
    /* Recv from close first, then far */ \
    func(0, ##__VA_ARGS__); \
    for (int i=1; i<NRECV && i<nrecv; i++) func(i, ##__VA_ARGS__); \
  } \
} while (0)

#define ROLE_INPUT     0x01
#define ROLE_OUTPUT    0x02
#define ROLE_WAIT_RECV 0x04
#define ROLE_WAIT_SEND 0x08
#define ROLE_POST_SEND 0x10
#define ROLE_POST_RECV 0x20

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, int DIRECT, class FUNC>
class ncclPrimitives {
 private:
  static constexpr int Input=0, Output=1;
  const int tid;
  int nthreads;
  int nworkers;
  const int stepSize;
  int nrecv, nsend;
  FUNC const fn;
  volatile int* connSizesFifoPtr = nullptr;
  void** connPtrsFifoPtr = nullptr;
  union {
    volatile uint64_t* connHeadPtr;
    volatile uint64_t* connTailPtr;
  };
  union {
    uint64_t connTailCache; // Cache last seen value
    uint64_t connHeadCache; // Cache last seen value
  };

  int index; // Peer index I'm responsible for
  int role = 0;
  int group;
  ncclShmemGroup *shmem;
  uint64_t step;
  T* direct = nullptr;
  T* buff;

  // Don't use barrier 0 as it's used by the final sync
  inline __device__ void barrier() {
    if (nthreads == WARP_SIZE) __syncwarp();
    else asm volatile ("bar.sync %0, %1;" :: "r"(group+1), "r"(nthreads));
  }
  inline __device__ void subBarrier() {
    if (nworkers == nthreads) barrier();
    else asm volatile ("bar.sync %0, %1;" :: "r"(group+2), "r"(nworkers));
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort() {
    spins++;
    if (abort == 0 && spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
      //printf("r=%d b=%d t=%d SPUN OUT\n", ncclShmem.comm->rank, blockIdx.x, threadIdx.x);
      abort = *(ncclShmem.comm->abortFlag);
      spins = 0;
    }
    return abort;
  }

  template <int DIRECTPTR>
  inline __device__ T* directPtr(ssize_t directOffset) {
    return DIRECTPTR && direct ? direct+directOffset : buff+(step%NCCL_STEPS)*stepSize;
  }

  template <int DST, int DIRECTSEND>
  inline __device__ void waitSend(ssize_t directOffset, int nbytes) {
    spins = 0;
    while (connHeadCache + NCCL_STEPS < step + SLICESTEPS) {
      connHeadCache = *connHeadPtr;
      if (checkAbort()) break;
    }
    if (connSizesFifoPtr) {
      connSizesFifoPtr[step%NCCL_STEPS] = nbytes;
    }

    if (connPtrsFifoPtr) loadPtr(connPtrsFifoPtr+step%NCCL_STEPS, shmem->dsts[DST+index]);
    else shmem->dsts[DST+index] = directPtr<DIRECTSEND>(directOffset);
    step += SLICESTEPS;
  }

  template <int SRC, int DIRECTRECV>
  inline __device__ void waitRecv(ssize_t directOffset) {
    spins = 0;
    while (connTailCache < step + SLICESTEPS) {
      connTailCache = *connTailPtr;
      if (checkAbort()) break;
    }
    if (connPtrsFifoPtr) loadPtr(connPtrsFifoPtr+step%NCCL_STEPS, shmem->srcs[SRC+index]);
    else shmem->srcs[SRC+index] = directPtr<DIRECTRECV>(directOffset);
    step += SLICESTEPS;
  }

  inline __device__ void postRecv() {
    *connHeadPtr = step += SLICESTEPS;
  }

  inline __device__ void postSend() {
    *connTailPtr = step += SLICESTEPS;
  }

  template <int DIRECTRECV1, int DIRECTSEND1, int RECV, int SEND, int SRCBUF, int DSTBUF>
  inline __device__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, intptr_t remoteOutIx, int nelem, bool postOp
    ) {
    constexpr int DIRECTRECV = 1 && DIRECTRECV1;
    constexpr int DIRECTSEND = 1 && DIRECTSEND1;
    constexpr int SRC = SRCBUF != -1;
    constexpr int DST = DSTBUF != -1;
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (tid < nworkers) {
        if (SRC && (role & (SRCBUF==Input ? ROLE_INPUT : ROLE_OUTPUT)))
          shmem->srcs[0] = buff + srcIx + offset;
        if (DST && (role & (DSTBUF==Input ? ROLE_INPUT : ROLE_OUTPUT)))
          shmem->dsts[0] = buff + dstIx + offset;
        if (RECV && (role & ROLE_WAIT_RECV)) waitRecv<SRC, DIRECTRECV>(dstIx+offset);
        if (SEND && (role & ROLE_WAIT_SEND)) waitSend<DST, DIRECTSEND>(remoteOutIx+offset, realSize*sizeof(T));
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && shmem->srcs[0] == shmem->dsts[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              // (1-SEND) is only there to avoid compilation errors in case NSEND=0 (and SEND=0).
              ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, (1-SEND)+NSEND>(tid, nworkers, fn, false, false, 1, (T const**)shmem->srcs, nsend, (T**)shmem->dsts+1, realSize);
            }
          } else {
            ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nworkers, fn, SRCBUF==Input, postOp, RECV*nrecv+SRC, (T const**)shmem->srcs, SEND*nsend+DST, (T**)shmem->dsts, realSize);
          }
        }
      }
      barrier();
      if (SEND && (role & ROLE_POST_SEND) && realSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      if (SEND && (role & ROLE_POST_SEND)) postSend();
      if (RECV && (role & ROLE_POST_RECV)) postRecv();
      offset += realSize;
    }
  }

  // Scatter and gather do not support DIRECT
  template <int RECV, int SEND>
  inline __device__ void
  ScatterGatherOp(intptr_t inpIx, intptr_t outIx, int totalElem, int peerElem, int skip, int shift, bool postOp) {
    int offset = 0; // slice offset
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(peerElem, 16*SLICESPERCHUNK)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, peerElem-offset));
      if (tid < nworkers) {
        if (SEND && (role & ROLE_INPUT)) shmem->srcs[0] = buff + inpIx + offset;
        if (RECV && (role & ROLE_OUTPUT)) shmem->dsts[0] = buff + outIx + offset;
        if (RECV && (role & ROLE_WAIT_RECV)) waitRecv<0, 0>(0);
        // realSize is not accurate here; but intra-node does not rely on sizes FIFO
        if (SEND && (role & ROLE_WAIT_SEND)) waitSend<0, 0>(0, realSize*sizeof(T));
        subBarrier();
        if (SEND) {
          #pragma unroll
          for (int j=0; j<nsend; j++) {
            int i = (j+shift)%nsend;
            int peerOffset = i*peerElem;
            if (skip >= 0 && i >= skip) peerOffset += peerElem;
            const T* src0 = (T*)shmem->srcs[0] + peerOffset;
            int realPeerSize = min(realSize, totalElem-peerOffset);
            if (realPeerSize > 0) ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nworkers, fn, true, false, 1, &src0, 1, (T**)shmem->dsts+i, realPeerSize);
          }
        } else if (RECV) {
          #pragma unroll
          for (int j=0; j<nrecv; j++) {
            int i = (j+shift)%nrecv;
            int peerOffset = i*peerElem;
            if (skip >= 0 && i >= skip) peerOffset += peerElem;
            T* dst0 = (T*)shmem->dsts[0] + peerOffset;
            int realPeerSize = min(realSize, totalElem-peerOffset);
            if (realPeerSize > 0) ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nworkers, fn, false, postOp, 1, (T const**)shmem->srcs+i, 1, &dst0, realPeerSize);
          }
        }
      }
      barrier();
      if (SEND && (role & ROLE_POST_SEND) && realSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      if (SEND && (role & ROLE_POST_SEND)) postSend();
      if (RECV && (role & ROLE_POST_RECV)) postRecv();
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclPeer *peer, T* directBuff) {
    if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)) {
      // For oneshot: groups 0,2 use conn 0, groups 4,6 use conn 1
      const int connIndex = (NSEND == NCCL_MAX_DIRECT_ARITY || NRECV == NCCL_MAX_DIRECT_ARITY) ? group/4 : 0;
      auto *conn = &peer->recv[connIndex].conn;
      step = conn->step;
      step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
      if (role & ROLE_POST_RECV) {
        shmem->recvConns[index] = conn; // POST role saves since that's who needs it in saveSync
        connHeadPtr = conn->head;
        // Return credits in case we rounded up.
        *connHeadPtr = step;
      }
      if (role & ROLE_WAIT_RECV) {
        buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
        if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
          void* volatile* slot = conn->ptrExchange;
          while (*slot != nullptr);
          direct = directBuff;
          *slot = directBuff;
        }
        connTailPtr = conn->tail;
        connTailCache = *connTailPtr;
        connPtrsFifoPtr = conn->ptrsFifo;
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclPeer *peer) {
    if (role & (ROLE_WAIT_SEND|ROLE_POST_SEND)) {
      // For oneshot: groups 0,2 use conn 0, groups 4,6 use conn 1
      const int connIndex = (NSEND == NCCL_MAX_DIRECT_ARITY || NRECV == NCCL_MAX_DIRECT_ARITY) ? group/4 : 0;
      auto *conn = &peer->send[connIndex].conn;
      step = conn->step;
      step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
      if (role & ROLE_POST_SEND) {
        shmem->sendConns[index] = conn; // POST role saves since that's who needs it in saveSync
        connTailPtr = conn->tail;
      }
      if (role & ROLE_WAIT_SEND) {
        buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
        if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
          void* volatile* ptr = conn->ptrExchange;
          while ((direct = (T*)(*ptr)) == NULL) { if (checkAbort()) break; }
          *ptr = NULL;
        }
        connHeadPtr = conn->head;
        connHeadCache = *connHeadPtr;
        connSizesFifoPtr = conn->sizesFifo;
        connPtrsFifoPtr = conn->ptrsFifo;
      }
    }
  }

  __device__ __forceinline__ void saveSync() {
    if (role & (ROLE_POST_SEND|ROLE_POST_RECV)) {
      auto *conns = (role & ROLE_POST_SEND) ? shmem->sendConns : shmem->recvConns;
      conns[index]->step = step;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nworkers, int* recvPeers, int* sendPeers, int stepSize, void const *inputBuf, void *outputBuf, int group=0)
    : tid(tid), nworkers(nworkers), stepSize(stepSize), fn(FuncTraits<FUNC>::make(ncclShmem.comm->nRanks)), group(group), shmem(&ncclShmem.groups[group]) {
    nthreads = nworkers;
    // For send operations, we need an extra warp to overlap the threadfence and the copy
    int postThreads = NSEND && nworkers >= 64 ? WARP_SIZE : 0;
    nthreads += postThreads;

    for (nrecv=0; nrecv < NRECV && recvPeers[nrecv] != -1; nrecv++);
    for (nsend=0; nsend < NSEND && sendPeers[nsend] != -1; nsend++);

    // Make sure no threads in previous class instances are looking at shared state (shmem).
    // Also make sure step is updated before we read it.
    barrier();

    #define SYNC_GROUP 8
    static_assert(NSEND < SYNC_GROUP && NRECV < SYNC_GROUP, "Not enough threads to cover all peers");

    int g = tid / SYNC_GROUP;
    int ng = nthreads / SYNC_GROUP;
    index = tid % SYNC_GROUP;

    if (g == 0) {
      if (index < nrecv) role |= ROLE_WAIT_RECV;
      if (index == nrecv) role |= ROLE_INPUT;
    } else if (g == 1) {
      if (index < nsend) role |= ROLE_WAIT_SEND;
      if (index == nsend) role |= ROLE_OUTPUT;
    } else if (g == ng - 2) {
      if (index < nrecv) role |= ROLE_POST_RECV;
    } else if (g == ng - 1) {
      if (index < nsend) role |= ROLE_POST_SEND;
    }

    int peer = 0;
    if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)) peer = recvPeers[index];
    if (role & (ROLE_WAIT_SEND|ROLE_POST_SEND)) peer = sendPeers[index];

    if (role & ROLE_INPUT) buff = (T*)inputBuf;
    if (role & ROLE_OUTPUT) buff = (T*)outputBuf;

    loadRecvConn(ncclShmem.channel->devPeers + peer, (T*)outputBuf);
    loadSendConn(ncclShmem.channel->devPeers + peer);
  }

  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, -1, eltN, false);
  }
  __device__ __forceinline__ void sendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, -1, eltN, false);
  }
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t remoteOutIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, -1, remoteOutIx, eltN, false);
  }
  __device__ __forceinline__ void directSendFromOutput(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    genericOp<0, 1, 0, 1, Output, -1>(outIx, -1, remoteOutIx, eltN, false);
  }

  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, -1, Output>(-1, outIx, -1, eltN, postOp);
  }

  __device__ __forceinline__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 0, 1, Input, Output>(inpIx, outIx, remoteOutIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopySend(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    genericOp<1, 1, 1, 1, -1, Output>(-1, outIx, remoteOutIx, eltN, false);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, -1, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    genericOp<0, 1, 1, 1, Input, Output>(inpIx, outIx, remoteOutIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  scatter(intptr_t inpIx, int totalElem, int peerElem, int skip, int shift) {
    ScatterGatherOp<0, 1>(inpIx, -1, totalElem, peerElem, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, int totalElem, int peerElem, int skip, int shift, bool postOp=false) {
    ScatterGatherOp<1, 0>(-1, outIx, totalElem, peerElem, skip, shift, postOp);
  }

  __device__ __forceinline__ ~ncclPrimitives() {
    // Save steps for the next operation
    saveSync();
  }
};

#include "prims_ll.h"
//#include "prims_ll128.h"

#endif
