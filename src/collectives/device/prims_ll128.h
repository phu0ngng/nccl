/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "op128.h"

#define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS-1)

template <typename T, class FUNC, int NRECV, int NSEND>
class ncclLL128Primitives {
private:
  static constexpr int Input=0, Output=1;
  FUNC fn;
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  const int warp;
  const bool flagThread;
  int nrecv = 0;
  int nsend = 0;
  T *userBufs[2];
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  uint64_t* recvBuff[NRECV];
  uint64_t* sendBuff[NSEND];

  volatile uint64_t* shmem;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ uint64_t* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ uint64_t* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint64_t recvFlag(int i) { return recvStep[i]+1; }
  inline __device__ uint64_t sendFlag(int i) { return sendStep[i]+1; }

  inline __device__ void barrier() {
    if (NSEND>NRECV) {
      asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
    } else {
      asm volatile ("bar.sync 2, %0;" :: "r"(nthreads));
    }
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
        sendConnFifoPtr[sendStep[wid]%NCCL_STEPS] = nbytes;
      }
      sendConnHead += 1;
    }
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += 1;
  }
  inline __device__ void postRecv() {
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
  }

  inline __device__ void incSend(int i) {
    sendStep[i] += 1;
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) { __threadfence(); *sendConnTailPtr = sendConnTail += 1; }
  }

  template <int ELEMS_PER_THREAD>
  inline __device__ void loadSrcToShmem128(int maxOffset, const uint64_t* src64Ptr) {
#if 0
    uint64_t v[ELEMS_PER_THREAD];
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      if (u*WARP_SIZE < maxOffset) load128(src64Ptr+u*WARP_SIZE, v[u], v[u+1]);
    }
    uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      storeShmem128(shmemAsmPtr+u*WARP_SIZE, v[u], v[u+1]);
    }
#else
    uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      if (u*WARP_SIZE < maxOffset) {
        uint64_t v0, v1;
        load128(src64Ptr+u*WARP_SIZE, v0, v1);
        storeShmem128(shmemAsmPtr+u*WARP_SIZE, v0, v1);
      }
    }
#endif
  }

  inline __device__ void loadSrcToShmem(int start, int end, const T* srcPtr) {
    T* shmemPtr = (T*)(shmem-2*wid);
    for (int offset = start+wid; offset < end; offset += WARP_SIZE) {
      shmemPtr[offset] = srcPtr[offset];
    }
  }

  template <int ELEMS_PER_THREAD>
  inline __device__ void storeShmemToDst128(int maxOffset, uint64_t* dst64Ptr) {
    uint64_t v[ELEMS_PER_THREAD];
    uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      loadShmem128(shmemAsmPtr+u*WARP_SIZE, v[u], v[u+1]);
    }
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      if (u*WARP_SIZE < maxOffset) store128(dst64Ptr+u*WARP_SIZE, v[u], v[u+1]);
    }
  }

  inline __device__ void storeShmemToDst(int start, int end, T* dstPtr) {
    T* shmemPtr = (T*)(shmem-2*wid);
    for (int offset = start+wid; offset < end; offset += WARP_SIZE) {
      dstPtr[offset] = shmemPtr[offset];
    }
  }

  #define WARP_MASK 0xffffffff

  template <int ELEMS_PER_THREAD, int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void recvReduceSendCopy(int ll128Offset, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    uint64_t v[ELEMS_PER_THREAD];

    /************* Data Loading : SHMEM -> REG **************/
    if (SRC) {
      volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        v[u] = shmem64Ptr[u*(WARP_SIZE-2)];
        if (SrcBuf == Input)
          v[u] = MULTI<FUNC, T>().preOp(fn, v[u]);
        if (!flagThread) {
          v[u+1] = shmem64Ptr[u*(WARP_SIZE-2)+1];
          if (SrcBuf == Input)
            v[u+1] = MULTI<FUNC, T>().preOp(fn, v[u+1]);
        }
      }
    }
    /*********** End Data Loading : SHMEM -> REG ************/

    /************************ Recv **************************/
    if (RECV) {
      uint64_t flag = recvFlag(0);
      uint64_t* ptr = recvPtr(0)+ll128Offset;
      bool needReload;
      uint64_t v0, v1;
      int spins = 0;
      do {
        needReload = false;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          load128(ptr+u*WARP_SIZE, v0, v1);
          needReload |= flagThread && (v1 != flag);
        }
      } while (__any_sync(WARP_MASK, needReload) && checkAbort(spins, 0, 0) == 0);

      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        load128(ptr+u*WARP_SIZE, v0, v1);
        v[u] = SRC ? MULTI<FUNC, T>()(fn, v0, v[u]) : v0;
        v[u+1] = SRC ? MULTI<FUNC, T>()(fn, v1, v[u+1]) : v1;
      }
      for (int i=1; i<NRECV && i<nrecv; i++) {
        uint64_t flag = recvFlag(i);
        uint64_t* ptr = recvPtr(i)+ll128Offset;
        uint64_t v0, v1;
        spins = 0;
        do {
          needReload = false;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, v0, v1);
            needReload |= flagThread && (v1 != flag);
          }
        } while (__any_sync(WARP_MASK, needReload) && checkAbort(spins, i, 0) == 0);

        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          load128(ptr+u*WARP_SIZE, v0, v1);
          v[u] = MULTI<FUNC, T>()(fn, v0, v[u]);
          v[u+1] = MULTI<FUNC, T>()(fn, v1, v[u+1]);
        }
      }
    }
    /********************** End Recv ************************/

    if (postOp && !FuncTraits<FUNC>::IsPostOpIdentity) {
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        v[u]   = MULTI<FUNC, T>().postOp(fn, v[u]);
        v[u+1] = MULTI<FUNC, T>().postOp(fn, v[u+1]);
      }
    }

    /************************ Send **************************/
    if (SEND) {
      for (int i=1; i<NSEND && i<nsend; i++) {
        uint64_t flag = sendFlag(i);
        uint64_t* ptr = sendPtr(i)+ll128Offset;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
        }
      }
      uint64_t flag = sendFlag(0);
      uint64_t* ptr = sendPtr(0)+ll128Offset;
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
      }
    }
    /********************** End Send ************************/

    /************* Data Storing : REG -> SHMEM **************/
    if (DST) {
      volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        shmem64Ptr[u*(WARP_SIZE-2)] = v[u];
        if (!flagThread) shmem64Ptr[u*(WARP_SIZE-2)+1] = v[u+1];
      }
    }
    /*********** End data Storing : REG -> SHMEM ************/
  }

  #define LL128INC (WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD)
  #define ELEMINC (LL128INC-(LL128INC/NCCL_LL128_LINEELEMS))

  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ void GenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    #if 0
    static_assert(-1<=SrcBuf && SrcBuf < 2, "Uhoh");
    static_assert(-1<=DstBuf && DstBuf < 2, "Uhoh");
    static_assert(DstBuf!=Input, "Mistake?");
    assert((SrcBuf==-1) == (srcIx==-1));
    assert((DstBuf==-1) == (dstIx==-1));
    #endif

    if (nelem <= 0) {
      // Don't move any data but still increase steps and sync with prev/next
      if (SEND) waitSend(0);
      FOR_SEND(incSend); if (SEND) postSend();
      FOR_RECV(incRecv); if (RECV) postRecv();
      return;
    }
    const int nelem64 = ((nelem*sizeof(T))/(2*sizeof(uint64_t)))*2;
    T const *srcPtr = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx;
    T       *dstPtr = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx;
    uint64_t const *src64Ptr = (uint64_t*)srcPtr;
    uint64_t       *dst64Ptr = (uint64_t*)dstPtr;

    int ll128Offset = LL128INC*warp+2*wid;
    int elemOffset = ELEMINC*warp;
    const int nwarps = nthreads/WARP_SIZE;

    if (SEND) waitSend(DIVUP(nelem*sizeof(T), ELEMINC*sizeof(uint64_t))*LL128INC*sizeof(uint64_t));
    barrier();

    while (elemOffset*(sizeof(uint64_t)/sizeof(T)) < nelem) {
      const int maxOffset128 = min(nelem64-elemOffset, (int)ELEMINC);
      const int maxOffset = min(nelem-(elemOffset*((int)(sizeof(uint64_t)/sizeof(T)))), (int)(ELEMINC*(sizeof(uint64_t)/sizeof(T))));
      if (SRC) {
        int done = 0;
        if ((reinterpret_cast<uintptr_t>(srcPtr)&0xf) == 0) {
          loadSrcToShmem128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, src64Ptr+elemOffset+2*wid);
          done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
        }
        loadSrcToShmem(done, maxOffset, (T*)(src64Ptr+elemOffset));
      }
      __syncwarp();
      recvReduceSendCopy<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SrcBuf, DstBuf>(ll128Offset, postOp);
      __syncwarp();
      if (DST) {
        int done = 0;
        if ((reinterpret_cast<uintptr_t>(dstPtr)&0xf) == 0) {
          storeShmemToDst128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, dst64Ptr+elemOffset+2*wid);
          done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
        }
        storeShmemToDst(done, maxOffset, (T*)(dst64Ptr+elemOffset));
      }
      __syncwarp();
      ll128Offset += LL128INC*nwarps;
      elemOffset += ELEMINC*nwarps;
    }

    barrier();
    FOR_SEND(incSend); if (SEND) postSend();
    FOR_RECV(incRecv); if (RECV) postRecv();
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
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
    sendBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
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
    if (tid >= nthreads-WARP_SIZE && wid<nsend) {
      if (sendConn->sizesFifo) {
        sendConnTailPtr = sendConn->tail;
        sendConnTail = sendConn->step;
      }
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
  __device__ ncclLL128Primitives(
      const int tid, const int nthreads, int const *recvPeers, int const *sendPeers,
      int stepSize, void const *inputBuf, void *outputBuf
    ):
    fn(FuncTraits<FUNC>().make(ncclShmem.comm->nRanks)),
    tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), warp(tid/WARP_SIZE),
    flagThread((tid%8)==7), stepSize(stepSize),
    shmem(ncclShmem.data + (threadIdx.x/WARP_SIZE)*NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE+2*wid) {

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
    return GenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void sendFromOutput(intptr_t outIx, int eltN) {
    return GenericOp<0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceSend(intptr_t inpIx, int eltN) {
    return GenericOp<1, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ ~ncclLL128Primitives() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};
