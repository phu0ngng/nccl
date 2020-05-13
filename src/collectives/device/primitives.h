/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
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

#define ROLE_RECV 0x01
#define ROLE_SEND 0x02
#define ROLE_SRC  0x04
#define ROLE_DST  0x08
#define ROLE_SYNC 0x10

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, int DIRECT, class FUNC, int GROUP>
class ncclPrimitives {
 private:
  const int tid;
  const int nthreads;
  const int stepSize;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* conn = NULL;
  volatile int* connSizesFifoPtr = NULL;
  volatile void** connPtrsFifoPtr = NULL;
  volatile uint64_t* connHeadPtr = NULL;
  volatile uint64_t* connTailPtr = NULL;
  uint64_t connTailCache; // Cache last seen value
  uint64_t connHeadCache; // Cache last seen value

  int index; // Peer index I'm responsible for
  int peer = -1;
  int role = 0;
  uint64_t step;
  T* direct = NULL;
  T* buff;
  struct ncclDevComm* comm;

  const T** srcs;
  T** dsts;

  inline __device__ void barrier() {
    asm volatile ("bar.sync %0, %1;" :: "r"(GROUP), "r"(nthreads+WARP_SIZE));
  }
  inline __device__ void subBarrier() {
    asm volatile ("bar.sync %0, %1;" :: "r"(GROUP+8), "r"(nthreads));
  }

  uint32_t mismatch = 0;
  const uint64_t opCount;

  inline __device__ void checkMismatch() {
    if (mismatch) {
      // In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch
      *(comm->fatalDevError) = ncclDevAssertedMismatch;
    } else if (conn && *conn->opCountRem > opCount) {
      mismatch += 1;
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort() {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = *(comm->abortFlag);
      checkMismatch();
      spins = 0;
    }
    return abort;
  }

  template <int DIRECTPTR>
  inline __device__ T* directPtr(ssize_t directOffset) {
    return DIRECTPTR && direct ? direct+directOffset : buff+(step%NCCL_STEPS)*stepSize;
  }

  template <int DST, int DIRECTSEND>
  inline __device__ void waitSend(ssize_t directOffset, int nbytes) {
    spins = 0;
    mismatch = 0;
    while (connHeadCache + NCCL_STEPS < step + SLICESTEPS) {
      connHeadCache = *connHeadPtr;
      if (checkAbort()) break;
    }
    if (connSizesFifoPtr) {
      connSizesFifoPtr[step%NCCL_STEPS] = nbytes;
    }
    dsts[DST+index] =
      connPtrsFifoPtr ?
      (T*)connPtrsFifoPtr[step%NCCL_STEPS] :
      directPtr<DIRECTSEND>(directOffset);
    step += SLICESTEPS;
  }

  template <int SRC, int DIRECTRECV>
  inline __device__ void waitRecv(ssize_t directOffset) {
    spins = 0;
    mismatch = 0;
    while (connTailCache < step + SLICESTEPS) {
      connTailCache = *connTailPtr;
      if (checkAbort()) break;
    }
    srcs[SRC+index] =
      connPtrsFifoPtr ?
      (const T*)connPtrsFifoPtr[step%NCCL_STEPS] :
      directPtr<DIRECTRECV>(directOffset);
    step += SLICESTEPS;
  }

  inline __device__ void postRecv() {
    *connHeadPtr = step += SLICESTEPS;
  }

  inline __device__ void postSend() {
    *connTailPtr = step += SLICESTEPS;
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, int nelem, ssize_t directOffset) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if ((role & ROLE_SYNC) == 0) {
        if (SRC && (role & ROLE_SRC)) srcs[0] = srcPtr+offset;
        if (RECV && (role & ROLE_RECV)) waitRecv<SRC, DIRECTRECV>(directOffset+offset);
        if (DST && (role & ROLE_DST)) dsts[0] = dstPtr+offset;
        if (SEND && (role & ROLE_SEND)) waitSend<DST, DIRECTSEND>(directOffset+offset, realSize*sizeof(T));
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && srcs[0] == dsts[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, NSEND>(tid, nthreads, 1, srcs, nsend, dsts+1, realSize);
            }
          } else {
            ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nthreads, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
          }
        }
      }
      barrier();
      if (role & ROLE_SYNC) {
        if (SEND && (role & ROLE_SEND) && realSize > 0 && tid == nthreads) __threadfence_system();
       __syncwarp();
       if (SEND && (role & ROLE_SEND)) postSend();
        if (RECV && (role & ROLE_RECV)) postRecv();
      }
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclChannel* channel, T* directBuff) {
    if (role & ROLE_RECV) {
      conn = &channel->devPeers[peer].recv.conn;
      step = conn->step;
      step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
      if (role & ROLE_SYNC) {
        connHeadPtr = conn->head;
        // Return credits in case we rounded up.
        *connHeadPtr = step;
        // Update opCount in case we skipped some operations
        *(conn->opCountLoc) = opCount;
      } else {
        buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
        if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
          direct = directBuff;
          *conn->ptrExchange = directBuff;
         }
        connTailPtr = conn->tail;
        connTailCache = *connTailPtr;
        connPtrsFifoPtr = (volatile void**)conn->ptrsFifo;
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclChannel* channel) {
    if (role & ROLE_SEND) {
      conn = &channel->devPeers[peer].send.conn;
      step = conn->step;
      step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
      if (role & ROLE_SYNC) {
        connTailPtr = conn->tail;
        // Update opCount in case we skipped some operations
        *(conn->opCountLoc) = opCount;
      } else {
        buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
        if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
          void* volatile* ptr = conn->ptrExchange;
          while ((direct = (T*)(*ptr)) == NULL);
          *ptr = NULL;
        }
        connHeadPtr = conn->head;
        connHeadCache = *connHeadPtr;
        connSizesFifoPtr = conn->sizesFifo;
        connPtrsFifoPtr = (volatile void**)conn->ptrsFifo;
      }
    }
  }

  __device__ __forceinline__ void saveSync() {
    if ((role & ROLE_SYNC) && conn) {
      conn->step = step;
      *(conn->opCountLoc) = opCount+1;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount, struct ncclShmemPtrs* ptrs)
    : comm(comm), tid(tid), nthreads(nthreads), stepSize(stepSize), opCount(opCount), srcs((const T**)ptrs->srcs), dsts((T**)ptrs->dsts) {
    // Make sure step is updated before we read it.
    barrier();

    if (tid >= nthreads) role |= ROLE_SYNC;

    for (int i=0; i<NRECV; i++) if (recvPeers[i] != -1) nrecv++;
    for (int i=0; i<NSEND; i++) if (sendPeers[i] != -1) nsend++;
    if (role & ROLE_SYNC) {
      index = tid-nthreads;
      if (index < NSEND) peer = sendPeers[index];
      if (peer != -1) {
        role |= ROLE_SEND;
      } else {
        index -= WARP_SIZE/2;
        if (index >= 0 && index < NRECV) peer = recvPeers[index];
        if (peer != -1) role |= ROLE_RECV;
      }
    } else {
      index = tid;
      if (index == NSEND) role |= ROLE_DST;
      if (index < NSEND) peer = sendPeers[index];
      if (peer != -1) {
        role |= ROLE_SEND;
      } else {
        if (nthreads > WARP_SIZE) index -= WARP_SIZE; else index -= WARP_SIZE/2;
        if (index == NRECV) role |= ROLE_SRC;
        if (index >= 0 && index < NRECV) peer = recvPeers[index];
        if (peer != -1) role |= ROLE_RECV;
      }
    }
    loadRecvConn(channel, directBuff);
    loadSendConn(channel);
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
  }
  __device__ __forceinline__ void
  directSend(const T* src, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0>(src, NULL, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecv(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvCopySend(T* dst, ssize_t directOffset, int nelem) {
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
  directRecvReduceCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ ~ncclPrimitives() {
    // Save steps for the next operation
    saveSync();
  }
};

#include "prims_ll.h"
//#include "prims_ll128.h"

#endif
