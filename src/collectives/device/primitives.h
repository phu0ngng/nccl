/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs


#define SPINS_BEFORE_CHECK_ABORT 100000

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, typename REDOP=FuncSum<T> >
class ncclPrimitives {
 private:
  const int tid;
  const int nthreads;
  const int stepSize;
  struct ncclConnInfo* recvConn;
  struct ncclConnInfo* sendConn;
  uint64_t recvStep;
  uint64_t sendStep;
  uint64_t sendConnHead;
  T* recvDirectBuff = NULL;
  T* sendDirectBuff = NULL;

  volatile uint32_t* abortFlagPtr = NULL;
  uint64_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int recvOffset() { return (recvStep%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset() { return (sendStep%NCCL_STEPS)*stepSize; }
  inline __device__ const T* recvPtr() { return ((const T*)recvConn->buff)+recvOffset(); }
  inline __device__ T* sendPtr() { return ((T*)sendConn->buff)+sendOffset(); }

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

  inline __device__ void waitRecv() {
    spins = 0;
    recvStep += SLICESTEPS;
    volatile uint64_t* ptr = recvConn->tail;
    while (*(ptr) < recvStep) {
      if (checkAbort()) return;
    }
  }

  inline __device__ void waitSend() {
    spins = 0;
    sendStep += SLICESTEPS;
    while (sendConnHead + NCCL_STEPS < sendStep) {
      volatile uint64_t* ptr = sendConn->head;
      sendConnHead = *ptr;
      if (checkAbort()) return;
    }
  }

  inline __device__ void postRecv() {
    *(recvConn->head) = recvStep += SLICESTEPS;
  }

  inline __device__ void postSend() {
    *(sendConn->tail) = sendStep += SLICESTEPS;
  }

  inline __device__ void postSendSize(int size) {
    if (sendConn->fifo) sendConn->fifo[sendStep%NCCL_STEPS] = size;
  }

  template <bool directrecv, bool directsend, bool recv, bool send, bool src, bool dst>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, int nelem, int directOffset) {
    int offset = 0;
    int sliceSize = stepSize * SLICESTEPS;
    T* dst1 = send ?
      (directsend && sendDirectBuff ? sendDirectBuff+directOffset : sendPtr())
      : dstPtr;
    const T* src1 = recv ?
      (directrecv && recvDirectBuff ? recvDirectBuff+directOffset : recvPtr())
      : srcPtr;

    #pragma unroll 1
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(sliceSize, nelem-offset));
      if (tid < nthreads) {
        if (send) waitSend();
        if (recv) waitRecv();

        if (realSize > 0) {
          if (directrecv && recvDirectBuff) {
            // Since src1 == dstPtr+offset, skip one copy
            if (send) {
              ReduceOrCopy<UNROLL, REDOP, T, false, false>(tid, nthreads, dst1+offset, NULL, src1+offset, NULL, realSize);
            }
          } else {
            if (tid < realSize)
            ReduceOrCopy<UNROLL, REDOP, T, dst&&send, src&&recv>(tid, nthreads, dst1+offset, dstPtr+offset, src1+offset, srcPtr+offset, realSize);
          }
        }

        exitIfAbortBarrier();
      } else {
        exitIfAbortBarrier();
        if (send) postSendSize(realSize*sizeof(T));
        if (send || recv) __threadfence_system();
        if (send) postSend();
        if (recv) postRecv();
      }
      offset += sliceSize;
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, int stepSize, struct ncclConnInfo* recv, struct ncclConnInfo* send, T* directBuff, volatile uint32_t* abortFlagPtr)
    : abortFlagPtr(abortFlagPtr), tid(tid), nthreads(nthreads), stepSize(stepSize), recvConn(recv), sendConn(send)
  {
    // Make sure step is updated before we read it
    __syncthreads();
    // Skip some slots if needed to make sure we can get aligned buffers chunks
    sendStep = ROUNDUP(sendConn->step, SLICESPERCHUNK*SLICESTEPS);
    recvStep = ROUNDUP(recvConn->step, SLICESPERCHUNK*SLICESTEPS);
    sendConnHead = *(sendConn->head);

    if (directBuff) {
      if (recvConn->direct) {
        recvDirectBuff = directBuff;
        if (tid == 0) *recvConn->ptrExchange = directBuff;
      }
      if (sendConn->direct) {
        void* volatile* ptr = sendConn->ptrExchange;
        while ((sendDirectBuff = (T*)(*ptr)) == NULL);
      }
    }
    __syncthreads();
    if (directBuff && sendConn->direct && tid == 0) {
      *sendConn->ptrExchange = NULL;
      // We should issue a threadfence_system here, but there will be communication
      // after that which will ensure it is reset.
    }
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOp<false, false, false, true, true, false>(src, NULL, nelem, 0);
  }
  __device__ __forceinline__ void
  directSend(const T* src, int directOffset, int nelem) {
    GenericOp<false, true, false, true, true, false>(src, NULL, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOp<false, false, true, false, false, true>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecv(T* dst, int directOffset, int nelem) {
    GenericOp<true, false, true, false, false, true>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst, int nelem) {
    GenericOp<false, false, false, true, true, true>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directCopySend(const T* src, T* dst, int directOffset, int nelem) {
    GenericOp<false, true, false, true, true, true>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp<false, false, true, true, false, true>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvCopySend(T* dst, int directOffset, int nelem) {
    GenericOp<true, true, true, true, false, true>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp<false, false, true, false, true, true>(src, dst, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src, int nelem) {
    GenericOp<false, false, true, true, true, false>(src, NULL, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp<false, false, true, true, true, true>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvReduceCopySend(const T* src, T* dst, int directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<false, true, true, true, true, true>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ ~ncclPrimitives() {
    // Save steps for next collective. Have thread 0 do it to be compatible
    // with the way LL works.
    if (tid == 0) {
      recvConn->step = recvStep;
      sendConn->step = sendStep;
      __threadfence();
    }
  }
};

#endif // end include guard
