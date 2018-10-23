/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#define SPINS_BEFORE_CHECK_ABORT 100000

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, class FUNC>
class ncclPrimitives {
 private:
  const int tid;
  const int nthreads;
  const int nrecv;
  const int nsend;
  const int stepSize;
  struct ncclConnInfo* recvConn[NRECV];
  struct ncclConnInfo* sendConn[NSEND];
  volatile uint64_t* waitPtr;
  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  uint64_t sendConnHead[NSEND];
  const T* recvDirectBuff[NRECV];
  T* sendDirectBuff[NSEND];
  const T* recvBuff[NRECV];
  T* sendBuff[NSEND];

  volatile uint32_t* abortFlagPtr = NULL;
  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ const T* recvPtr(int i) { return ((const T*)recvBuff[i])+recvOffset(i); }
  inline __device__ T* sendPtr(int i) { return ((T*)sendBuff[i])+sendOffset(i); }

  // Each thread sets a predicate to true if val == 1
  // all CTA's threads enter the barrier and do a popc on their predicates being True
  // If any of the thread's predicate was True, all the threads call exit()
  inline __device__ void exitIfAbortBarrier() {
    uint32_t popc;
    asm ("{");
    asm volatile ("   .reg .pred barr_pred;");
    asm volatile ("   setp.eq.u32 barr_pred,%0,1;" :: "r"(abort));
    asm volatile ("   bar.red.popc.u32 %0, 14, barr_pred;" : "=r"(popc));
    asm ("}");
    if (popc) { asm volatile ("exit;"); }
  }

  inline __device__ void barrier() {
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
  }

  inline __device__ int checkAbort() {
    spins++;
    if (spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = *abortFlagPtr;
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitRecv(int i) {
    spins = 0;
    recvStep[i] += SLICESTEPS;
    if (tid == i) {
      while (*(waitPtr) < recvStep[i]) {
        if (checkAbort()) break;
      }
    }
  }

  inline __device__ void waitSend(int i) {
    spins = 0;
    sendStep[i] += SLICESTEPS;
    if (tid == WARP_SIZE+i) {
      while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {
        sendConnHead[i] = *waitPtr;
        if (checkAbort()) break;
      }
    }
  }

  inline __device__ void postRecv(int i) {
    *(recvConn[i]->head) = recvStep[i] += SLICESTEPS;
  }

  inline __device__ void postSend(int i) {
    *(sendConn[i]->tail) = sendStep[i] += SLICESTEPS;
  }

  inline __device__ void postSendSize(int i, int size) {
    if (sendConn[i]->fifo) sendConn[i]->fifo[sendStep[i]%NCCL_STEPS] = size;
  }

  template <int DIRECTRECV>
  inline __device__ const T* directRecvPtr(int i, int directOffset) {
    return DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i]+directOffset : recvPtr(i);
  }

  template <int DIRECTSEND>
  inline __device__ T* directSendPtr(int i, int directOffset) {
    return DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i]+directOffset : sendPtr(i);
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, int nelem, int directOffset) {
    int offset = 0;
    int sliceSize = stepSize * SLICESTEPS;

    const T* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? srcPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    T* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? dstPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);
    }

    #pragma unroll 1
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(sliceSize, nelem-offset));
      if (tid < nthreads) {
        if (SEND) { waitSend(0); for (int i=1; i<NSEND && i<nsend; i++) waitSend(i); }
        if (RECV) { waitRecv(0); for (int i=1; i<NRECV && i<nrecv; i++) waitRecv(i); }
        if (SEND || RECV) barrier();

        if (realSize > 0) {
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              //ReduceOrCopy<UNROLL, FUNC, T, false, false>(tid, nthreads, dsts[1], NULL, srcs[0], NULL, realSize);
              ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, NSEND>(tid, nthreads, 1, srcs, nsend, dsts+1, realSize);
            }
          } else {
            //ReduceOrCopy<UNROLL, FUNC, T, SEND*NSEND+DST == 2, RECV*NRECV+SRC == 2>(tid, nthreads, dsts[0], dsts[1], srcs[0], srcs[1], realSize);
            ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nthreads, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
          }
        }

        exitIfAbortBarrier();
      } else {
        exitIfAbortBarrier();
        if (SEND) { postSendSize(0, realSize*sizeof(T)); for (int i=1; i<NSEND && i<nsend; i++) postSendSize(i, realSize*sizeof(T)); }
        if (SEND) __threadfence_system();
        if (SEND) { postSend(0); for (int i=1; i<NSEND && i<nsend; i++) postSend(i); }
        if (RECV) { postRecv(0); for (int i=1; i<NRECV && i<nrecv; i++) postRecv(i); }
      }
      for (int i=0; i<RECV*NRECV+SRC; i++) srcs[i] += sliceSize;
      for (int i=0; i<SEND*NSEND+DST; i++) dsts[i] += sliceSize;
      offset += sliceSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    recvConn[i] = conn;
    recvBuff[i] = (const T*)recvConn[i]->buff;
    recvStep[i] = recvConn[i]->step;
    recvStep[i] = ROUNDUP(recvStep[i], SLICESPERCHUNK*SLICESTEPS);
    // Return credits in case we rounded up.
    if (tid == nthreads) *recvConn[i]->head = recvStep[i];
    if (tid == i) waitPtr = recvConn[i]->tail;
    recvDirectBuff[i] = NULL;
    if (directBuff && recvConn[i]->direct) {
      recvDirectBuff[i] = directBuff;
      if (tid == 0) *recvConn[i]->ptrExchange = directBuff;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    sendConn[i] = conn;
    sendBuff[i] = (T*)sendConn[i]->buff;
    sendStep[i] = sendConn[i]->step;
    sendStep[i] = ROUNDUP(sendStep[i], SLICESPERCHUNK*SLICESTEPS);
    if (tid == WARP_SIZE+i) waitPtr = sendConn[i]->head;
    if (tid == WARP_SIZE+i) sendConnHead[i] = *waitPtr;
    sendDirectBuff[i] = NULL;
    if (directBuff && sendConn[i]->direct) {
      void* volatile* ptr = sendConn[i]->ptrExchange;
      while ((sendDirectBuff[i] = (T*)(*ptr)) == NULL);
    }
  }

  __device__ __forceinline__ void saveRecvConn(int i) {
    if (tid == i) recvConn[i]->step = recvStep[i];
  }

  __device__ __forceinline__ void saveSendConn(int i) {
    if (tid == WARP_SIZE+i) sendConn[i]->step = sendStep[i];
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, const int nrecv, int* recvPeers, const int nsend, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, volatile uint32_t* abortFlagPtr)
    : abortFlagPtr(abortFlagPtr), tid(tid), nthreads(nthreads), nsend(nsend), nrecv(nrecv), stepSize(stepSize) {
    // Make sure step is updated before we read it
    __syncthreads();

    loadRecvConn(&channel->devPeers[recvPeers[0]].recv.conn, 0, directBuff);
    for (int i=1; i<NRECV && i<nrecv; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, directBuff);
    loadSendConn(&channel->devPeers[sendPeers[0]].send.conn, 0, directBuff);
    for (int i=1; i<NSEND && i<nsend; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i, directBuff);
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
  }
  __device__ __forceinline__ void
  directSend(const T* src, int directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0>(src, NULL, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecv(T* dst, int directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directCopySend(const T* src, T* dst, int directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvCopySend(T* dst, int directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1>(src, dst, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvReduceCopySend(const T* src, T* dst, int directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ ~ncclPrimitives() {
    // Save steps for next collective. Have thread 0 do it to be compatible
    // with the way LL works.
    saveRecvConn(0);
    for (int i=1; i<NRECV && i<nrecv; i++) saveRecvConn(i);
    saveSendConn(0);
    for (int i=1; i<NSEND && i<nsend; i++) saveSendConn(i);
  }
};

#endif
