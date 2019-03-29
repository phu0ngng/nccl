/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common.h"

#define SPINS_BEFORE_CHECK_ABORT 1000000

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

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, class FUNC>
class ncclPrimitives {
 private:
  const int tid;
  const int nthreads;
  int nrecv = 0;
  int nsend = 0;
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
  struct ncclDevComm* comm;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ const T* recvPtr(int i) { return ((const T*)recvBuff[i])+recvOffset(i); }
  inline __device__ T* sendPtr(int i) { return ((T*)sendBuff[i])+sendOffset(i); }

  inline __device__ void barrier() {
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
  }

  uint32_t mismatch = 0;
  const uint64_t opCount;

  inline __device__ void checkMismatch(volatile uint64_t* remoteOpCount) {
    if (mismatch) {
      // In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch
      *(comm->fatalDevError) = ncclDevAssertedMismatch;
    } else if (remoteOpCount && *remoteOpCount > opCount) {
      mismatch += 1;
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort(volatile uint64_t* remoteOpCount) {
    spins++;
    if (spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = *(comm->abortFlag);
      checkMismatch(remoteOpCount);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitRecv(int i) {
    spins = 0;
    mismatch = 0;
    recvStep[i] += SLICESTEPS;
    if (tid == i) {
      while (*(waitPtr) < recvStep[i]) {
        if (checkAbort(recvConn[i]->opCountRem)) break;
      }
    }
  }

  inline __device__ void waitSend(int i) {
    spins = 0;
    mismatch = 0;
    sendStep[i] += SLICESTEPS;
    if (tid == WARP_SIZE+i) {
      while (sendConnHead[i] + NCCL_STEPS < sendStep[i]) {
        sendConnHead[i] = *waitPtr;
        if (checkAbort(sendConn[i]->opCountRem)) break;
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
        FOR_SEND(waitSend);
        FOR_RECV(waitRecv);
        if (realSize > 0) {
          barrier();
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, NSEND>(tid, nthreads, 1, srcs, nsend, dsts+1, realSize);
            }
          } else {
            ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nthreads, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
          }
        }
        exitIfAbortBarrier(abort);
      } else {
        exitIfAbortBarrier(abort);
        FOR_SEND(postSendSize, realSize*sizeof(T));
        if (SEND) __threadfence_system();
        FOR_SEND(postSend);
        FOR_RECV(postRecv);
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
    if (tid == i) {
      waitPtr = recvConn[i]->tail;
      *(recvConn[i]->opCountLoc) = opCount;
    }
    recvDirectBuff[i] = NULL;
    if (directBuff && recvConn[i]->direct) {
      recvDirectBuff[i] = directBuff;
      if (tid == 0) *recvConn[i]->ptrExchange = directBuff;
    }
    nrecv++;
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    sendConn[i] = conn;
    sendBuff[i] = (T*)sendConn[i]->buff;
    sendStep[i] = sendConn[i]->step;
    sendStep[i] = ROUNDUP(sendStep[i], SLICESPERCHUNK*SLICESTEPS);
    if (tid == WARP_SIZE+i) {
      waitPtr = sendConn[i]->head;
      sendConnHead[i] = *waitPtr;
      *(sendConn[i]->opCountLoc) = opCount;
    }
    sendDirectBuff[i] = NULL;
    if (directBuff && sendConn[i]->direct) {
      void* volatile* ptr = sendConn[i]->ptrExchange;
      while ((sendDirectBuff[i] = (T*)(*ptr)) == NULL);
      __syncthreads();
      if (tid == 0) *ptr = NULL;
    }
    nsend++;
  }

  __device__ __forceinline__ void saveRecvConn(int i) {
    if (tid == i) {
      recvConn[i]->step = recvStep[i];
      __threadfence_system();
      *(recvConn[i]->opCountLoc) += 1;
    }
  }

  __device__ __forceinline__ void saveSendConn(int i) {
    if (tid == WARP_SIZE+i) {
      sendConn[i]->step = sendStep[i];
      __threadfence_system();
      *(sendConn[i]->opCountLoc) += 1;
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount)
    : comm(comm), tid(tid), nthreads(nthreads), stepSize(stepSize), opCount(opCount) {
    // Make sure step is updated before we read it
    __syncthreads();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, directBuff);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i, directBuff);
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
    for (int i=0; i<NRECV && i<nrecv; i++) saveRecvConn(i);
    for (int i=0; i<NSEND && i<nsend; i++) saveSendConn(i);
  }
};

#include "prims_ll.h"
//#include "prims_ll128.h"

#endif
