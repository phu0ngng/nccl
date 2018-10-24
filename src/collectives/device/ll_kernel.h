/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_LL_KERNEL_H_
#define NCCL_LL_KERNEL_H_

#define SPINS_BEFORE_CHECK_ABORT 1000000

template <typename T, class FUNC, int NRECV, int NSEND>
class ncclLLPrimitives {
 private:
  const int tid;
  const int nthreads;
  const int nrecv;
  const int nsend;
  struct ncclConnInfo* recvConn[NRECV];
  struct ncclConnInfo* sendConn[NSEND];
  volatile uint64_t* waitPtr;
  volatile uint64_t* postPtr;
  volatile int* fifoPtr;
  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  uint64_t sendConnHead;
  union ncclLLFifoLine* recvBuff[NRECV];
  union ncclLLFifoLine* sendBuff[NSEND];

  volatile uint32_t* abortFlagPtr = NULL;
  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*NCCL_LL_SLICE_LINES; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*NCCL_LL_SLICE_LINES; }
  inline __device__ union ncclLLFifoLine* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ union ncclLLFifoLine* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint32_t recvFlag(int i) { return recvStep[i]+1; }
  inline __device__ uint32_t sendFlag(int i) { return sendStep[i]+1; }

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

  inline __device__ void waitSend(int i) {
    spins = 0;
    if (tid == WARP_SIZE+i) {
      while (sendConnHead + NCCL_STEPS < sendStep[i] + 1) {
        sendConnHead = *waitPtr;
        if (checkAbort()) break;
      }
    }
  }

  inline __device__ void postRecv(int i) {
    recvStep[i]++;
    if (tid == i) *postPtr = recvStep[i];
  }

  inline __device__ void postSend(int i, int nbytes) {
    if (tid == WARP_SIZE+i && fifoPtr) fifoPtr[sendStep[i]%NCCL_STEPS] = nbytes;
    sendStep[i]++;
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

  __device__ void storeAL(uint64_t* dst, uint64_t val, uint32_t nbytes) {
    memcpy((char*)dst, (char*)&val, nbytes);
  }

  template <int RECV, int SEND, int SRC, int DST>
  __device__ void LLGenericOp(const T* srcPtr, T* dstPtr, int nelem) {
    uint32_t nbytes = nelem < 0 ? 0 : nelem*sizeof(T);
    if (SEND) { waitSend(0); for(int i=1; i<NSEND && i<nsend; i++) waitSend(i); barrier(); }
    uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));
    uint64_t* srcPack = (uint64_t*)srcPtr;
    uint64_t* dstPack = (uint64_t*)dstPtr;
    // Do multiples of 64 bits
    #pragma unroll 2
    for (int line=0, offset=tid; line<NUM_LINES_PER_THREAD && offset<npack; line++, offset+=nthreads) {
      uint64_t val = SRC ? readAL(srcPack+offset) : readLL(recvPtr(0)+offset, recvFlag(0));
      if (RECV) {
        if (SRC) val = MULTI<FUNC, T>()(readLL(recvPtr(0)+offset, recvFlag(0)), val);
        for (int i= 1; i<NRECV && i<nrecv; i++) {
          val = MULTI<FUNC, T>()(readLL(recvPtr(i)+offset, recvFlag(i)), val);
        }
      }
      if (SEND) {
        storeLL(sendPtr(0)+offset, val, sendFlag(0));
        for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, val, sendFlag(i));
      }
      if (DST) {
        if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {
          // Last incomplete word
          storeAL(dstPack+offset, val, nbytes & 0x7);
        } else {
          storeAL(dstPack+offset, val, sizeof(uint64_t));
        }
      }
    }
    if (SEND) { postSend(0, nbytes*2); for(int i=1; i<NSEND && i<nsend; i++) postSend(i, nbytes*2); }
    exitIfAbortBarrier();
    if (RECV) { postRecv(0); for(int i=1; i<NRECV && i<nrecv; i++) postRecv(i); }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvConn[i] = conn;
    recvBuff[i] = recvConn[i]->llBuff;
    recvStep[i] = recvConn[i]->step;
    if (tid == i) {
      postPtr = recvConn[i]->head;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendConn[i] = conn;
    sendBuff[i] = sendConn[i]->llBuff;
    sendStep[i] = sendConn[i]->step;
    if (tid == WARP_SIZE+i) {
      waitPtr = sendConn[i]->head;
      fifoPtr = sendConn[i]->fifo;
      sendConnHead = *waitPtr;
    }
  }

  __device__ __forceinline__ void saveRecvConn(int i) {
    if (tid == i) {
      recvConn[i]->step = recvStep[i];
      __threadfence_block();
    }
  }

  __device__ __forceinline__ void saveSendConn(int i) {
    if (tid == WARP_SIZE+i) {
      sendConn[i]->step = sendStep[i];
      __threadfence_block();
    }
  }

  __device__ __forceinline__ void llSendCleaning(int i) {
    if (sendStep[i] > sendConn[i]->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
      /* Reset all flags */
      static_assert((NCCL_LL_BUFF_SIZE % NCCL_LL_MAX_NTHREADS) == 0, "NCCL_LL_BUFF_SIZE must be a multiple of THREADS");
      static_assert(NCCL_LL_BUFF_SIZE/(sizeof(union ncclLLFifoLine)*NCCL_LL_MAX_NTHREADS) > 0, "NCCL_LL_BUFF_SIZE is less than 16 bytes*THREADS");
      for (int s=0; s<NCCL_STEPS; s++) {
        waitSend(i);
        for (int o=tid; o<NCCL_LL_SLICE_LINES; o+=nthreads) {
          const union ncclLLFifoLine resetLine = { 0, sendFlag(i), 0, sendFlag(i) };
          sendPtr(i)[o].i4 = resetLine.i4;
        }
        postSend(i, 0);
      }
      if (tid == 0) sendConn[i]->llLastCleaning = sendStep[i];
    }
  }

  __device__ __forceinline__ void llRecvCleaning(int i) {
    if (recvStep[i] > recvConn[i]->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
      recvStep[i] += NCCL_STEPS;
      if (tid == 0) recvConn[i]->llLastCleaning = recvStep[i];
    }
  }

 public:
  __device__ __forceinline__
  ncclLLPrimitives(const int tid, const int nthreads, const int nrecv, int* recvPeers, const int nsend, int* sendPeers, struct ncclChannel* channel, volatile uint32_t* abortFlagPtr)
    : abortFlagPtr(abortFlagPtr), tid(tid), nthreads(nthreads), nrecv(nrecv), nsend(nsend) {
    // Make sure step is updated before we read it.
    barrier();

    loadRecvConn(&channel->devPeers[recvPeers[0]].recv.conn, 0);
    for (int i=1; i<NRECV && i<nrecv; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i);
    loadSendConn(&channel->devPeers[sendPeers[0]].send.conn, 0);
    for (int i=1; i<NSEND && i<nsend; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
  }

  __device__ void send(const T* src, int nelem) {
    return LLGenericOp<0, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recv(T* dst, int nelem) {
    return LLGenericOp<1, 0, 0, 1>(NULL, dst, nelem);
  }

  __device__ void recvReduceSend(const T* src, int nelem) {
    return LLGenericOp<1, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recvReduceCopy(const T* src, T* dst, int nelem) {
    return LLGenericOp<1, 0, 1, 1>(src, dst, nelem);
  }

  __device__ void copySend(const T* src, T* dst, int nelem) {
    return LLGenericOp<0, 1, 1, 1>(src, dst, nelem);
  }

  __device__ void recvCopySend(T* dst, int nelem) {
    return LLGenericOp<1, 1, 0, 1>(NULL, dst, nelem);
  }

  __device__ void recvReduceCopySend(const T* src, T* dst, int nelem) {
    return LLGenericOp<1, 1, 1, 1>(src, dst, nelem);
  }

  __device__ __forceinline__ ~ncclLLPrimitives() {
    llSendCleaning(0);
    for (int i=1; i<NSEND && i<nsend; i++) llSendCleaning(i);
    llRecvCleaning(0);
    for (int i=1; i<NRECV && i<nrecv; i++) llRecvCleaning(i);
    // Save steps for the next operation
    saveRecvConn(0);
    for (int i=1; i<NRECV && i<nrecv; i++) saveRecvConn(i);
    saveSendConn(0);
    for (int i=1; i<NSEND && i<nsend; i++) saveSendConn(i);
  }
};

#endif
