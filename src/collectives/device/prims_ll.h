/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

template <typename T, class FUNC, int NRECV, int NSEND>
class ncclLLPrimitives {
private:
  static constexpr int Input=0, Output=1;
  FUNC fn;
  const int tid;
  const int nthreads;
  const int wid;
  const int stepLines;
  int nrecv = 0;
  int nsend = 0;
  T *userBufs[2];
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  union ncclLLFifoLine* recvBuff[NRECV];
  union ncclLLFifoLine* sendBuff[NSEND];

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepLines; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepLines; }
  inline __device__ union ncclLLFifoLine* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ union ncclLLFifoLine* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint32_t recvFlag(int i) { return NCCL_LL_FLAG(recvStep[i]+1); }
  inline __device__ uint32_t sendFlag(int i) { return NCCL_LL_FLAG(sendStep[i]+1); }

  inline __device__ void barrier() {
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
  }

  uint32_t abort = 0;

  inline __device__ int checkAbort(int &spins, int i, int send) {
    spins++;
    if (abort == 0 && spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
      abort = *(ncclShmem.comm->abortFlag);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
    if (sendConnHeadPtr) {
      int spins = 0;
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
        sendConnHeadCache = *sendConnHeadPtr;
        if (checkAbort(spins, wid, 1)) break;
      }
      if (sendConnFifoPtr) {
        int size = ((sendConnHead & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) ? stepLines*sizeof(union ncclLLFifoLine) : nbytes;
        sendConnFifoPtr[sendConnHead%NCCL_STEPS] = size;
      }
      sendConnHead += 1;
    }
    barrier();
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += 1;
  }
  inline __device__ void postRecv() {
    barrier();
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
  }

  inline __device__ void incSend(int i, int offset) {
    // LL Cleanup : write all flags in the slice to make sure we don't have
    // data corruption when flag loops over.
    if ((sendStep[i] & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) {
      for (int o = offset; o<stepLines; o+=nthreads) storeLL(sendPtr(i)+o, 0, sendFlag(i));
    }
    sendStep[i]++;
  }

  __device__ uint64_t readLL(int i, int offset) {
    union ncclLLFifoLine* src = recvPtr(i) + offset;
    uint32_t flag = recvFlag(i);
    uint32_t data1, flag1, data2, flag2;
    int spins = 0;
    do {
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2) : "l"(&src->i4));
      if (checkAbort(spins, i, 0)) break;
    } while ((flag1 != flag) || (flag2 != flag));
    uint64_t val64 = data1 + (((uint64_t)data2) << 32);
    return val64;
  }

  __device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag), "r"((uint32_t)(val >> 32)), "r"(flag));
  }

  static constexpr int EltPerPack = sizeof(uint64_t)/sizeof(T);
  union EltPack {
    uint64_t word;
    T elt[EltPerPack];
  };

  // Using memcpy handles misaligned pointers.
  __device__ uint64_t readAL(T *src) {
    EltPack pack;
    #pragma unroll EltPerPack
    for(int i=0; i < EltPerPack; i++)
      pack.elt[i] = src[i];
    return pack.word;
  }

  __device__ void storeAL(T *dst, uint64_t val, int nelem) {
    EltPack pack;
    pack.word = val;
    dst[0] = pack.elt[0];
    nelem -= 1;
    #pragma unroll
    for(int i=1; i < EltPerPack; i++) {
      if (nelem > 0)
        dst[i] = pack.elt[i];
      nelem -= 1;
    }
  }

  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ void LLGenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    nelem = nelem < 0 ? 0 : nelem;
    int npack = DIVUP(nelem, EltPerPack);
    int offset = tid;
    T *srcElts = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx + offset*EltPerPack;
    T *dstElts = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx + offset*EltPerPack;

    // Always waitSend in case of cleanup
    if (SEND) waitSend(npack*sizeof(union ncclLLFifoLine));

    // Do multiples of 64 bits
    #pragma unroll 2
    for (; offset<npack; offset+=nthreads) {
      // Recv : local, then intra-node, then inter-node
      uint64_t val;
      if (SRC) {
        val = readAL(srcElts);
        srcElts += nthreads*EltPerPack;
        if (SrcBuf == Input)
          val = MULTI<FUNC, T>().preOp(fn, val);
      }
      else
        val = readLL(0, offset);

      if (RECV) {
        for (int i=1-SRC; i<NRECV && i<nrecv; i++) {
          val = MULTI<FUNC, T>()(fn, readLL(i, offset), val);
        }
      }

      if (postOp) val = MULTI<FUNC, T>().postOp(fn, val);

      // Send : inter-node, then intra-node, then local
      if (SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, val, sendFlag(i));
        storeLL(sendPtr(0)+offset, val, sendFlag(0));
      }
      if (DST) {
        storeAL(dstElts, val, nelem - offset*EltPerPack);
        dstElts += nthreads*EltPerPack;
      }
    }
    FOR_RECV(incRecv); if (RECV) postRecv();
    FOR_SEND(incSend, offset);
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (union ncclLLFifoLine*)conn->buffs[NCCL_PROTO_LL];
    recvStep[i] = conn->step;
    if (wid == i) recvConn = conn;
    nrecv++;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConnHeadPtr = recvConn->head;
      recvConnHead = recvConn->step;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (union ncclLLFifoLine*)conn->buffs[NCCL_PROTO_LL];
    sendStep[i] = conn->step;
    if (wid == i) sendConn = conn;
    nsend++;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnHead = sendConn->step;
      sendConnFifoPtr = sendConn->sizesFifo;
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConn->step = recvConnHead;
      __threadfence_block();
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid < nsend) {
      sendConn->step = sendConnHead;
      __threadfence_block();
    }
  }

 public:
  __device__  ncclLLPrimitives(
      const int tid, const int nthreads, int const *recvPeers, int const *sendPeers,
      int stepLines, void const *inputBuf, void *outputBuf
    ):
    fn(FuncTraits<FUNC>().make(ncclShmem.comm->nRanks)),
    tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepLines(stepLines) {

    userBufs[Input] = (T*)inputBuf;
    userBufs[Output] = (T*)outputBuf;
    // Make sure step is updated before we read it.
    barrier();

    auto *channel = ncclShmem.channel;
    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
    loadRecvSync();
    loadSendSync();
  }

  __device__ void send(intptr_t inpIx, int eltN) {
    return LLGenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void sendFromOutput(intptr_t outIx, int eltN) {
    return LLGenericOp<0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    return LLGenericOp<1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceSend(intptr_t inpIx, int eltN) {
    return LLGenericOp<1, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return LLGenericOp<1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return LLGenericOp<0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    return LLGenericOp<1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return LLGenericOp<1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ ~ncclLLPrimitives() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};
