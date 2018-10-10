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

  inline __device__ void
  GenericOp( const T* src1, const T* src2, T* dst1, T* dst2, int nelem, bool r, bool s) {
    int offset = 0;
    int sliceSize = stepSize * SLICESTEPS;

    #pragma unroll 1
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(sliceSize, nelem-offset));
      if (tid < nthreads) {
        if (s) waitSend();
        if (r) waitRecv();
  //      if (r || s) asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
        if (src2) {
          if (dst2) {
            ReduceOrCopy<UNROLL, REDOP, T, true,  true >(tid, nthreads, dst1+offset, dst2+offset, src1+offset, src2+offset, realSize);
          } else {
            ReduceOrCopy<UNROLL, REDOP, T, false, true >(tid, nthreads, dst1+offset, NULL,        src1+offset, src2+offset, realSize);
          }
        } else { 
          if (dst2) {
            ReduceOrCopy<UNROLL, REDOP, T, true,  false>(tid, nthreads, dst1+offset, dst2+offset, src1+offset, NULL,        realSize);
          } else {
            ReduceOrCopy<UNROLL, REDOP, T, false, false>(tid, nthreads, dst1+offset, NULL,        src1+offset, NULL,        realSize);
          }
        }
        exitIfAbortBarrier();
      } else {
        exitIfAbortBarrier();
        if (s) postSendSize(realSize*sizeof(T));
        if (s || r) __threadfence_system();
        if (s) postSend();
        if (r) postRecv();
      }
      offset += sliceSize;
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, int stepSize, struct ncclConnInfo* recv, struct ncclConnInfo* send, volatile uint32_t* abortFlagPtr)
    : abortFlagPtr(abortFlagPtr), tid(tid), nthreads(nthreads), stepSize(stepSize), recvConn(recv), sendConn(send)
  {
    // Make sure step is updated before we read it
    __syncthreads();
    // Skip some slots if needed to make sure we can get aligned buffers chunks
    sendStep = ROUNDUP(sendConn->step, SLICESPERCHUNK*SLICESTEPS);
    recvStep = ROUNDUP(recvConn->step, SLICESPERCHUNK*SLICESTEPS);
    sendConnHead = *(sendConn->head);
    // No need to sync the other way as there will be a syncthreads later on
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOp(src, NULL, sendPtr(), NULL, nelem, false, true);
  }
  __device__ __forceinline__ void
  sendDirect(const T* src, T* dst, int nelem) {
    GenericOp(src, NULL, dst, NULL, nelem, false, true);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOp(recvPtr(), NULL, dst, NULL, nelem, true, false);
  }
  __device__ __forceinline__ void
  recvDirect(const T* src, int nelem) {
    // We just need to check data has been written but no copy is necessary
    GenericOp(src, NULL, NULL, NULL, 0, true, false);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst1, int nelem) {
    GenericOp(src, NULL, dst1, sendPtr(), nelem, false, true);
  }
  __device__ __forceinline__ void
  copySendDirect(const T* src, T* dst1, T* dst2, int nelem) {
    GenericOp(src, NULL, dst1, dst2, nelem, false, true);
  }

  __device__ __forceinline__ void
  recvDirectSend(const T* src, int nelem) {
    GenericOp(src, NULL, sendPtr(), NULL, nelem, true, true);
  }
  __device__ __forceinline__ void
  recvDirectSendDirect(const T* src, T* dst, int nelem) {
    GenericOp(src, NULL, dst, NULL, nelem, true, true);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp(recvPtr(), NULL, dst, sendPtr(), nelem, true, true);
  }
  __device__ __forceinline__ void
  recvCopySendDirect(T* dst1, T* dst2, int nelem) {
    GenericOp(recvPtr(), NULL, dst1, dst2, nelem, true, true);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp(recvPtr(), src, dst, NULL, nelem, true, false);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src, int nelem) {
    GenericOp(recvPtr(), src, sendPtr(), NULL, nelem, true, true);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp(recvPtr(), src, dst, sendPtr(), nelem, true, true);
  }
  __device__ __forceinline__ void
  recvReduceCopySendDirect(const T* src, T* dst1, T* dst2, int nelem) {
    GenericOp(recvPtr(), src, dst1, dst2, nelem, true, true);
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
