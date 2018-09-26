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
  struct ncclConnInfo* recvConn;
  struct ncclConnInfo* sendConn;

  volatile uint32_t* abortFlagPtr = NULL;
  uint64_t spins = 0;
  uint32_t abort = 0;

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
    recvConn->waitStep += SLICESTEPS;
    volatile uint64_t* ptr = recvConn->tail;
    while (*(ptr) < recvConn->waitStep) {
      if (checkAbort()) return;
    }
  }
    
 inline __device__ void waitSend() {
    spins = 0;
    sendConn->waitStep += SLICESTEPS;
    volatile uint64_t* ptr = sendConn->head;
    while (*(ptr) + NCCL_STEPS < sendConn->waitStep) {
      if (checkAbort()) return;
    }
  }

  inline __device__ void postRecv() {
    *(recvConn->head) = recvConn->postStep += SLICESTEPS;
  }
    
  inline __device__ void postSend() {
    *(sendConn->tail) = sendConn->postStep += SLICESTEPS;
  }
    
  inline __device__ void postSendSize(int size) {
    if (sendConn->fifo) sendConn->fifo[sendConn->postStep%NCCL_STEPS] = size;
  }

  inline __device__ void
  GenericOp( const T* src1, const T* src2, T* dst1, T* dst2, int chunkSize, int size, bool r, bool s) {
    int sliceSize = chunkSize / SLICESPERCHUNK;
    int offset = 0;

    #pragma unroll 1
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(sliceSize, size-offset));
      if (tid < nthreads) {
        if (s && tid == 0) waitSend();
        if (r && tid == 0) waitRecv();
        if (r || s) asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
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
  ncclPrimitives(const int tid, const int nthreads, struct ncclConnInfo* recv, struct ncclConnInfo* send, volatile uint32_t* abortFlagPtr)
    : abortFlagPtr(abortFlagPtr), tid(tid), nthreads(nthreads), recvConn(recv), sendConn(send)
  {
    // Skip some slots if needed to make sure we can get aligned buffers chunks
    if (tid == 0) {
      sendConn->waitStep = ROUNDUP(sendConn->waitStep, SLICESPERCHUNK*SLICESTEPS);
      recvConn->waitStep = ROUNDUP(recvConn->waitStep, SLICESPERCHUNK*SLICESTEPS);
    }
    if (tid == nthreads) {
      sendConn->postStep = ROUNDUP(sendConn->postStep, SLICESPERCHUNK*SLICESTEPS);
      recvConn->postStep = ROUNDUP(recvConn->postStep, SLICESPERCHUNK*SLICESTEPS);
    }  
  }

  __device__ __forceinline__ void
  send(const T* src, T* dst, int len, int maxOffset) {
    GenericOp(src, NULL, dst, NULL, len, maxOffset, false, true);
  }

  __device__ __forceinline__ void
  recv(const T* src, T* dst, int len, int maxOffset) {
    GenericOp(src, NULL, dst, NULL, len, maxOffset, true, false);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst1, T* dst2, int len, int maxOffset) {
    GenericOp(src, NULL, dst1, dst2, len, maxOffset, false, true);
  }

  __device__ __forceinline__ void
  recvSend(const T* src, T* dst1, int len, int maxOffset) {
    GenericOp(src, NULL, dst1, NULL, len, maxOffset, true, true);
  }

  __device__ __forceinline__ void
  recvCopySend(const T* src, T* dst1, T* dst2, int len, int maxOffset) {
    GenericOp(src, NULL, dst1, dst2, len, maxOffset, true, true);
  }

  __device__ __forceinline__ void
  recvReduce(const T* src1, const T* src2, T* dst, int len, int maxOffset) {
    GenericOp(src1, src2, dst, NULL, len, maxOffset, true, false);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src1, const T* src2, T* dst, int len, int maxOffset) {
    GenericOp(src1, src2, dst, NULL, len, maxOffset, true, true);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src1, const T* src2, T* dst1, T* dst2, int len, int maxOffset) {
    GenericOp(src1, src2, dst1, dst2, len, maxOffset, true, true);
  }
};

#endif // end include guard
