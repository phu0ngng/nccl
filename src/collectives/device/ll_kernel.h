/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_LL_KERNEL_H_
#define NCCL_LL_KERNEL_H_

#define LL_SPINS_BEFORE_CHECK_ABORT 100000

template <typename T, class FUNC>
class ncclLLPrimitives {
 private:
  const int tid;
  const int nthreads;
  struct ncclConnInfo* recvConn;
  struct ncclConnInfo* sendConn;
  uint64_t recvStep;
  uint64_t sendStep;
  uint64_t sendConnHead;

  volatile uint32_t* abortFlagPtr = NULL;
  uint64_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int recvOffset() { return (recvStep%NCCL_LL_STEPS)*NCCL_LL_SLICE_LINES; }
  inline __device__ int sendOffset() { return (sendStep%NCCL_LL_STEPS)*NCCL_LL_SLICE_LINES; }
  inline __device__ uint32_t recvFlag() { return recvStep+1; }
  inline __device__ uint32_t sendFlag() { return sendStep+1; }

  // Each thread sets a predicate to true if val == 1
  // all CTA's threads enter the barrier and do a popc on their predicates being True
  // If any of the thread's predicate was True, all the threads call exit()
  inline __device__ void exitIfAbortBarrier() {
    uint32_t popc;
    asm ("{");
    asm volatile ("   .reg .pred barr_pred;");
    asm volatile ("   setp.eq.u32 barr_pred,%0,1;" :: "r"(abort));
    asm volatile ("   bar.red.popc.u32 %0, 14, %1, barr_pred;" : "=r"(popc) : "r"(nthreads));
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

  inline __device__ void waitSend() {
    spins = 0;
    while (sendConnHead + NCCL_LL_STEPS < sendStep + 1) {
      volatile uint64_t* ptr = sendConn->llHead;
      sendConnHead = *ptr;
      if (checkAbort()) break;
    }
  }

  inline __device__ void postRecv() {
    recvStep++;
    if (tid == 0) {
      volatile uint64_t* ptr = recvConn->llHead;
      *ptr = recvStep;
    }
  }

  inline __device__ void postSend(int size) {
    if (tid == 0 && sendConn->llFifo) sendConn->llFifo[sendStep%NCCL_LL_STEPS] = size;
    sendStep++;
  }

  __device__ uint64_t readLL(union ncclLLFifoLine* src, uint32_t flag) {
    uint32_t data1, flag1, data2, flag2;
    spins = 0;
    do {
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2) : "l"(&src->i4));
      if (checkAbort()) break;
    } while ((flag1 != flag) || (flag2 != flag));
    uint64_t val64 = data1 + (((uint64_t)data2) << 32);
    return val64;
  }

  __device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag), "r"((uint32_t)(val >> 32)), "r"(flag));
  }

  // Using memcpy handles misaligned pointers.
  __device__ uint64_t readAL(uint64_t* src) {
    uint64_t val;
    memcpy((char*)&val, (char*)src, sizeof(uint64_t));
    return val;
  }

  __device__ void storeAL(uint64_t* dst, uint64_t val) {
    memcpy((char*)dst, (char*)&val, sizeof(uint64_t));
  }

  __device__ void LLGenericOp(const T* src, T* dst, int size, bool r, bool s) {
    if (size <= 0) return;
    if (s) waitSend();
    size_t size64 = size * sizeof(T) / sizeof(uint64_t);
    uint64_t* srcA = (uint64_t*)src;
    uint64_t* dstA = (uint64_t*)dst;
    // Do multiples of 64 bits
#pragma unroll 1
    for (int offset = tid; offset < size64; offset += nthreads) {
      uint64_t val;
      if (src) {
        val = readAL(srcA+offset);
        if (r) val = MULTI<FUNC, T>()(readLL(recvConn->llBuff+recvOffset()+offset, recvFlag()), val);
      } else if (r) {
        val = readLL(recvConn->llBuff+recvOffset()+offset, recvFlag());
      }
      if (dst) storeAL(dstA+offset, val);
      if (s) storeLL(sendConn->llBuff+sendOffset()+offset, val, sendFlag());
    }
    // Finish last 64-bits word
    int sizeDone = size64 * (sizeof(uint64_t)/sizeof(T));
    int sizeRem = size - sizeDone;
    if (tid == 0 && sizeRem) {
      const T* src1B = src + sizeDone;
      T* dstB = dst + sizeDone;

      uint64_t lastVal;
      T* vals = (T*)&lastVal;

      if (r) {
        uint64_t lastVal2 = readLL(recvConn->llBuff+recvOffset()+size64, recvFlag());
        T* src2B = (T*)&lastVal2;
        for (int offset = 0; offset < sizeRem; offset++) {
          vals[offset] = src ? FUNC()(src2B[offset], src1B[offset]) : src2B[offset];
        }
      } else if (src) {
        for (int offset = 0; offset < sizeRem; offset++) {
          vals[offset] = src1B[offset];
        }
      }
      if (s) storeLL(sendConn->llBuff+sendOffset()+size64, lastVal, sendFlag());
      if (dst) {
        for (int offset = 0; offset < sizeRem; offset++) {
          dstB[offset] = vals[offset];
        }
      }
    }
    if (s) postSend(size*2*(int)sizeof(T));
    exitIfAbortBarrier();
    if (r) postRecv();
  }

  public:
  __device__ __forceinline__
  ncclLLPrimitives(const int tid, const int nthreads, struct ncclConnInfo* recv, struct ncclConnInfo* send, volatile uint32_t* abortFlagPtr)
    : abortFlagPtr(abortFlagPtr), tid(tid), nthreads(nthreads), recvConn(recv), sendConn(send) {
    recvStep = recvConn->llStep;
    sendStep = sendConn->llStep;
    sendConnHead = *(sendConn->llHead);
    // Make sure all threads start with the same steps before moving on
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
  }

  __device__ void send(const T* src, int size) {
    return LLGenericOp(src, NULL, size, false, true);
  }

  __device__ void recv(T* dst, int size) {
    return LLGenericOp(NULL, dst, size, true, false);
  }

  __device__ void recvReduceSend(const T* src, int size) {
    return LLGenericOp(src, NULL, size, true, true);
  }

  __device__ void recvReduce(const T* src, T* dst, int size) {
    return LLGenericOp(src, dst, size, true, false);
  }

  __device__ void copySend(const T* src, T* dst, int size) {
    return LLGenericOp(src, dst, size, false, true);
  }

  __device__ void recvCopySend(T* dst, int size) {
    return LLGenericOp(NULL, dst, size, true, true);
  }

  __device__ void recvReduceCopySend(const T* src, T* dst, int size) {
    return LLGenericOp(src, dst, size, true, true);
  }

  __device__ __forceinline__ ~ncclLLPrimitives() {
    if (sendStep > sendConn->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
      /* Reset all flags */
      static_assert((NCCL_LL_BUFF_SIZE % NCCL_LL_MAX_NTHREADS) == 0, "NCCL_LL_BUFF_SIZE must be a multiple of THREADS");
      static_assert(NCCL_LL_BUFF_SIZE/(sizeof(union ncclLLFifoLine)*NCCL_LL_MAX_NTHREADS) > 0, "NCCL_LL_BUFF_SIZE is less than 16 bytes*THREADS");
      for (int s=0; s<NCCL_LL_STEPS; s++) {
        waitSend();
        for (int o=tid; o<NCCL_LL_SLICE_LINES; o+=nthreads) {
          const union ncclLLFifoLine resetLine = { 0, sendFlag(), 0, sendFlag() };
          sendConn->llBuff[sendOffset()+o].i4 = resetLine.i4;
        }
        postSend(0);
      }
      if (tid == 0) sendConn->llLastCleaning = sendStep;
    }
    if (recvStep > recvConn->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
      recvStep += NCCL_LL_STEPS;
      if (tid == 0) recvConn->llLastCleaning = recvStep;
    }
    // Save llStep for the next operation
    if (tid == 0) {
      recvConn->llStep = recvStep;
      sendConn->llStep = sendStep;
      __threadfence();
    }
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
  }
};

#endif
