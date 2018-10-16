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
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, typename REDOP=FuncSum<T>>
class ncclPrimitives {
 private:
  const int tid;
  const int nthreads;
  const int nrecv;
  const int nsend;
  const int stepSize;
  struct ncclConnInfo* recvConn[NRECV];
  struct ncclConnInfo* sendConn[NSEND];
  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  uint64_t sendConnHead[NSEND];
  const T* recvDirectBuff[NRECV];
  T* sendDirectBuff[NSEND];

  volatile uint32_t* abortFlagPtr = NULL;
  uint64_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ const T* recvPtr(int i) { return ((const T*)recvConn[i]->buff)+recvOffset(i); }
  inline __device__ T* sendPtr(int i) { return ((T*)sendConn[i]->buff)+sendOffset(i); }

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
    if (tid == 0) {
      volatile uint64_t* ptr = recvConn[i]->tail;
      while (*(ptr) < recvStep[i]) {
        if (checkAbort()) break;
      }
    }
  }

  inline __device__ void waitSend(int i) {
    spins = 0;
    sendStep[i] += SLICESTEPS;
    if (tid == 0) {
      while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {
        volatile uint64_t* ptr = sendConn[i]->head;
        sendConnHead[i] = *ptr;
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

    const T* srcs[NRECV+1];
    srcs[0] = SRC ? srcPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) for (int i=1-SRC; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);

    T* dsts[NSEND+1];
    dsts[0] = DST ? dstPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) for (int i=1-DST; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);

    #pragma unroll 1
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(sliceSize, nelem-offset));
      if (tid < nthreads) {
        if (SEND) for (int i=0; i<NSEND && i<nsend; i++) waitSend(i);
        if (RECV) for (int i=0; i<NRECV && i<nrecv; i++) waitRecv(i);
        if (SEND || RECV) asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));

        if (realSize > 0) {
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              //ReduceOrCopy<UNROLL, REDOP, T, false, false>(tid, nthreads, dsts[0], NULL, srcs[0], NULL, realSize);
              ReduceOrCopyMulti<UNROLL, REDOP, T, 1, NSEND>(tid, nthreads, 1, srcs, nsend, dsts, realSize);
            }
          } else {
            //ReduceOrCopy<UNROLL, REDOP, T, SEND*NSEND+DST == 2, RECV*NRECV+SRC == 2>(tid, nthreads, dsts[0], dsts[1], srcs[0], srcs[1], realSize);
            ReduceOrCopyMulti<UNROLL, REDOP, T, RECV*NRECV+SRC, SEND*NSEND+DST>(tid, nthreads, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
          }
        }

        exitIfAbortBarrier();
      } else {
        exitIfAbortBarrier();
        if (SEND) for (int i=0; i<NSEND && i<nsend; i++) postSendSize(i, realSize*sizeof(T));
        if (SEND || RECV) __threadfence_system();
        if (SEND) for (int i=0; i<NSEND && i<nsend; i++) postSend(i);
        if (RECV) for (int i=0; i<NRECV && i<nrecv; i++) postRecv(i);
      }
      for (int i=0; i<RECV*NRECV+SRC; i++) srcs[i] += sliceSize;
      for (int i=0; i<SEND*NSEND+DST; i++) dsts[i] += sliceSize;
      offset += sliceSize;
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, const int nrecv, int* recvPeers, const int nsend, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, volatile uint32_t* abortFlagPtr)
    : abortFlagPtr(abortFlagPtr), tid(tid), nthreads(nthreads), nsend(nsend), nrecv(nrecv), stepSize(stepSize)
  {
    // Make sure step is updated before we read it
    __syncthreads();
    for (int i=0; i<NRECV && i<nrecv; i++) {
      recvConn[i] = &channel->devPeers[recvPeers[i]].recv.conn;
      recvStep[i] = ROUNDUP(recvConn[i]->step, SLICESPERCHUNK*SLICESTEPS);
      recvDirectBuff[i] = NULL;
      if (directBuff && recvConn[i]->direct) {
        recvDirectBuff[i] = directBuff;
        if (tid == 0) *recvConn[i]->ptrExchange = directBuff;
      }
    }
    // Skip some slots if needed to make sure we can get aligned buffers chunks
    for (int i=0; i<NSEND && i<nsend; i++) {
      sendConn[i] = &channel->devPeers[sendPeers[i]].send.conn;
      sendStep[i] = ROUNDUP(sendConn[i]->step, SLICESPERCHUNK*SLICESTEPS);
      sendConnHead[i] = *(sendConn[i]->head);
      sendDirectBuff[i] = NULL;
      if (directBuff && sendConn[i]->direct) {
        void* volatile* ptr = sendConn[i]->ptrExchange;
        while ((sendDirectBuff[i] = (T*)(*ptr)) == NULL);
      }
    }
    __syncthreads();
    for (int i=0; i<NSEND && i<nsend; i++) {
      if (directBuff && sendConn[i]->direct && tid == 0) {
        *sendConn[i]->ptrExchange = NULL;
      }
    }
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
    if (tid == 0) {
      for (int i=0; i<NRECV && i<nrecv; i++) recvConn[i]->step = recvStep[i];
      for (int i=0; i<NSEND && i<nsend; i++) sendConn[i]->step = sendStep[i];
      __threadfence();
    }
  }
};

#endif
