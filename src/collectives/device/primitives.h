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

#define NCCL_SPINS_BEFORE_CHECK_ABORT 1000000

// Unroll unconditionally the first send/recv since nsend/nrecv should be at
// least 1 if Send/Recv is set.
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
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, int DIRECT, class FUNC>
class ncclPrimitives {
 private:
  static constexpr int Input=0, Output=1;
  static constexpr int RoleInput = 0x01,
                       RoleOutput = 0x02,
                       RoleWaitRecv = 0x04,
                       RoleWaitSend = 0x08,
                       RolePostSend = 0x10,
                       RolePostRecv = 0x20,
                       Aborted = 0x40,
                       PtrsFifoEnabled = 0x80,
                       SizesFifoEnabled = 0x100;
  const int tid;
  int nthreads;
  int nworkers;
  const int stepSize;
  int nrecv, nsend;
  FUNC const fn;
  int index; // Peer index I'm responsible for
  int flags;
  int group;
  uint64_t step;
  union {
    void **connPtrsFifoPtr; // (flags & PtrsFifoEnabled)
    T *userBuff;            // (flags & (RoleInput|RoleOutput))
    T *connEltsFifo;        // !(flags & (PtrsFifoEnabled|RoleInput|RoleOutput))
  };
  union {
    int volatile *connSizesFifoPtr; //  (flags & SizesFifoEnabled)
    T *directBuff;                  // !(flags & SizesFifoEnabled)
  };
  uint64_t volatile *connStepPtr;
  uint64_t connStepCache; // Cache last seen value of (*connStepPtr)

  // Don't use barrier 0 as it's used by the final sync
  inline __device__ void barrier() {
    if (nthreads == WARP_SIZE) __syncwarp();
    else asm volatile ("bar.sync %0, %1;" :: "r"(group+1), "r"(nthreads));
  }
  inline __device__ void subBarrier() {
    if (nworkers == nthreads) barrier();
    else asm volatile ("bar.sync %0, %1;" :: "r"(group+2), "r"(nworkers));
  }

  inline __device__ bool checkAbort(int &spins) {
    spins++;
    if (!(flags & Aborted) && spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
      //printf("r=%d b=%d t=%d SPUN OUT\n", ncclShmem.comm->rank, blockIdx.x, threadIdx.x);
      flags |= *(ncclShmem.comm->abortFlag) ? Aborted : 0;
      spins = 0;
    }
    return flags & Aborted;
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  inline __device__ void waitPeer(intptr_t dstIx, intptr_t remoteOutIx, int offset, int nelts) {
    if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
      bool const isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
      int spins = 0;
      while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + SLICESTEPS) {
        connStepCache = *connStepPtr;
        if (checkAbort(spins)) break;
      }
      if (isSendNotRecv && (flags & SizesFifoEnabled)) {
        connSizesFifoPtr[step%NCCL_STEPS] = nelts*sizeof(T);
      }
      void **ptrs = isSendNotRecv ? (ncclShmem.groups[group].dsts + Dst)
                                  : (ncclShmem.groups[group].srcs + Src);
      if (flags & PtrsFifoEnabled)
        loadPtr(connPtrsFifoPtr + step%NCCL_STEPS, ptrs[index]);
      else if ((isSendNotRecv ? DirectSend : DirectRecv) && !(flags & SizesFifoEnabled) && directBuff)
        ptrs[index] = directBuff + (isSendNotRecv ? remoteOutIx : dstIx) + offset;
      else
        ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
      step += SLICESTEPS;
    }
  }

  template<int Recv, int Send>
  inline __device__ void postPeer() {
    if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
      step += SLICESTEPS;
      *connStepPtr = step;
    }
  }

  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  inline __device__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, intptr_t remoteOutIx, int nelem, bool postOp
    ) {
    constexpr int DirectRecv = 1 && DirectRecv1;
    constexpr int DirectSend = 1 && DirectSend1;
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize*SLICESTEPS;
    sliceSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);
    int slice = 0;
    int offset = 0;

    if (tid < nworkers && offset < nelem) {
      // Worker-only loop for non-empty slices. Non-workers and empty slices are
      // processed in the loop following this if block. The benefit of splitting
      // the loop like this is we pull two branches out of the critical path.
      // Using "number of branch insns (taken or not) encountered dynamically"
      // as the performance metric, then:
      //   perf_orig = 2*numslices
      //   perf_new = 2+numslices
      // So the new code and old code behave the same for numslices=2, and for
      // numslices>2 the new code is superior.
      //
      // ORIGINAL CODE:
      //   unrolled for(slices) {
      //     if(worker) { // This branch removed
      //       wait();
      //       subBarrier();
      //       if(slice not empty) // This branch removed
      //         ReduceCopyMulti();
      //     }
      //     barrier();
      //     post();
      //   } // Since we no longer unroll, new branch added here
      #pragma unroll 1
      do {
        sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
        if (Src && (flags & (SrcBuf==Input ? RoleInput : RoleOutput)))
          ncclShmem.groups[group].srcs[0] = userBuff + srcIx + offset;
        if (Dst && (flags & (DstBuf==Input ? RoleInput : RoleOutput)))
          ncclShmem.groups[group].dsts[0] = userBuff + dstIx + offset;
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(dstIx, remoteOutIx, offset, sliceSize);
        subBarrier();
        if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]) {
          // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
          if (Send) {
            // (1-Send) is only there to avoid compilation errors in case NSEND=0 (and Send=0).
            ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, (1-Send)+NSEND>
              (tid, nworkers, fn, false, false,
               1, (T const**)ncclShmem.groups[group].srcs,
               nsend, (T**)ncclShmem.groups[group].dsts+1,
               sliceSize);
          }
        } else {
          ReduceOrCopyMulti<UNROLL, FUNC, T, Recv+Src, Recv*NRECV+Src, Send+Dst, Send*NSEND+Dst>
            (tid, nworkers, fn, SrcBuf==Input, postOp,
             Recv*nrecv+Src, (T const**)ncclShmem.groups[group].srcs,
             Send*nsend+Dst, (T**)ncclShmem.groups[group].dsts,
             sliceSize);
        }
        barrier(); // This barrier has a counterpart in following loop
        if (Send && (flags & RolePostSend) && index == 0) __threadfence_system();
        __syncwarp();
        postPeer<Recv, Send>();
        offset += sliceSize;
        slice += 1;
      } while (offset < nelem && slice < SLICESPERCHUNK);
    }

    // Non-workers come straight here. Workers too but only once the remaining
    // slices are all empty. Since empty slices are the uncommon case, and
    // worker perf is the limiter, perf-wise this loop is effectively unentered,
    // hence just a single branch insn.
    #pragma unroll 1
    while (offset < nelem || slice < SLICESPERCHUNK) {
      sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
      { // Only workers could have Wait roles so we know the slice must be empty
        // since we've exited the loop above.
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(0, 0, 0, 0);
      }
      barrier(); // Has couterpart in preceding worker-only loop.
      if (Send && (flags & RolePostSend) && sliceSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      postPeer<Recv, Send>();
      offset += sliceSize;
      slice += 1;
    }
  }

  // Scatter and gather do not support DIRECT
  template <int Recv, int Send>
  inline __device__ void
  ScatterGatherOp(intptr_t inpIx, intptr_t outIx, int totalElem, int peerElem, int skip, int shift, bool postOp) {
    int offset = 0; // slice offset
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(peerElem, 16*SLICESPERCHUNK)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, peerElem-offset));
      if (tid < nworkers) {
        if (Send && (flags & RoleInput)) ncclShmem.groups[group].srcs[0] = userBuff + inpIx + offset;
        if (Recv && (flags & RoleOutput)) ncclShmem.groups[group].dsts[0] = userBuff + outIx + offset;
        // realSize is not accurate here; but intra-node does not rely on sizes FIFO
        waitPeer<0, 0, Recv, Send, 0, 0>(0, 0, 0, realSize);
        subBarrier();
        if (Send) {
          #pragma unroll
          for (int j=0; j<nsend; j++) {
            int i = (j+shift)%nsend;
            int peerOffset = i*peerElem;
            if (skip >= 0 && i >= skip) peerOffset += peerElem;
            const T* src0 = (T*)ncclShmem.groups[group].srcs[0] + peerOffset;
            int realPeerSize = min(realSize, totalElem-peerOffset);
            if (realPeerSize > 0) ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nworkers, fn, true, false, 1, &src0, 1, (T**)ncclShmem.groups[group].dsts+i, realPeerSize);
          }
        } else if (Recv) {
          #pragma unroll
          for (int j=0; j<nrecv; j++) {
            int i = (j+shift)%nrecv;
            int peerOffset = i*peerElem;
            if (skip >= 0 && i >= skip) peerOffset += peerElem;
            T* dst0 = (T*)ncclShmem.groups[group].dsts[0] + peerOffset;
            int realPeerSize = min(realSize, totalElem-peerOffset);
            if (realPeerSize > 0) ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nworkers, fn, false, postOp, 1, (T const**)ncclShmem.groups[group].srcs+i, 1, &dst0, realPeerSize);
          }
        }
      }
      barrier();
      if (Send && (flags & RolePostSend) && realSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      postPeer<Recv, Send>();
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclPeer *peer, T *outputBuf) {
    if (flags & (RoleWaitRecv|RolePostRecv)) {
      // For oneshot: groups 0,2 use conn 0, groups 4,6 use conn 1
      const int connIndex = (NSEND == NCCL_MAX_DIRECT_ARITY || NRECV == NCCL_MAX_DIRECT_ARITY) ? group/4 : 0;
      auto *conn = &peer->recv[connIndex].conn;
      step = conn->step;
      step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
      if (flags & RolePostRecv) {
        ncclShmem.groups[group].recvConns[index] = conn; // Post role saves since that's who needs it in saveSync
        connStepPtr = conn->head;
        // Return credits in case we rounded up.
        *connStepPtr = step;
      }
      if (flags & RoleWaitRecv) {
        connStepPtr = conn->tail;
        connStepCache = *connStepPtr;
        flags |= (conn->ptrsFifo != nullptr) ? PtrsFifoEnabled : 0;
        if (flags & PtrsFifoEnabled)
          connPtrsFifoPtr = conn->ptrsFifo;
        else
          connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];

        if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
          void *volatile *slot = conn->ptrExchange;
          while (*slot != nullptr);
          directBuff = outputBuf;
          *slot = outputBuf;
        }
        else
          directBuff = nullptr;
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclPeer *peer) {
    if (flags & (RoleWaitSend|RolePostSend)) {
      // For oneshot: groups 0,2 use conn 0, groups 4,6 use conn 1
      const int connIndex = (NSEND == NCCL_MAX_DIRECT_ARITY || NRECV == NCCL_MAX_DIRECT_ARITY) ? group/4 : 0;
      auto *conn = &peer->send[connIndex].conn;
      step = conn->step;
      step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
      if (flags & RolePostSend) {
        ncclShmem.groups[group].sendConns[index] = conn; // Post role saves since that's who needs it in saveSync
        connStepPtr = conn->tail;
      }
      if (flags & RoleWaitSend) {
        connStepPtr = conn->head;
        connStepCache = *connStepPtr;
        flags |= (conn->ptrsFifo != nullptr) ? PtrsFifoEnabled : 0;
        if (flags & PtrsFifoEnabled)
          connPtrsFifoPtr = conn->ptrsFifo;
        else
          connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];

        flags |= (conn->sizesFifo != nullptr) ? SizesFifoEnabled : 0;
        if (flags & SizesFifoEnabled)
          connSizesFifoPtr = conn->sizesFifo;
        else if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
          int spins = 0;
          void *volatile *slot = conn->ptrExchange;
          void *ptr;
          do ptr = *slot;
          while (ptr == nullptr && !checkAbort(spins));
          directBuff = (T*)ptr;
          *slot = nullptr;
        }
        else
          directBuff = nullptr;
      }
    }
  }

  __device__ __forceinline__ void saveSync() {
    if (flags & (RolePostSend|RolePostRecv)) {
      auto *conns = (flags & RolePostSend) ? ncclShmem.groups[group].sendConns : ncclShmem.groups[group].recvConns;
      conns[index]->step = step;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nworkers, int* recvPeers, int* sendPeers, int stepSize, void const *inputBuf, void *outputBuf, int group=0)
    : tid(tid), nworkers(nworkers), stepSize(stepSize), fn(FuncTraits<FUNC>::make(ncclShmem.comm->nRanks)), group(group) {
    nthreads = nworkers;
    // For send operations, we need an extra warp to overlap the threadfence and the copy
    int postThreads = NSEND && nworkers >= 64 ? WARP_SIZE : 0;
    nthreads += postThreads;

    for (nrecv=0; nrecv < NRECV && recvPeers[nrecv] != -1; nrecv++);
    for (nsend=0; nsend < NSEND && sendPeers[nsend] != -1; nsend++);

    // Make sure no threads in previous class instances are looking at shared state (ncclShmem.groups[group]).
    // Also make sure step is updated before we read it.
    barrier();

    #define SYNC_GROUP 8
    static_assert(NSEND < SYNC_GROUP && NRECV < SYNC_GROUP, "Not enough threads to cover all peers");

    int g = tid / SYNC_GROUP;
    int ng = nthreads / SYNC_GROUP;
    index = tid % SYNC_GROUP;
    flags = 0;
    if (g == 0) {
      if (index < nrecv) flags |= RoleWaitRecv;
      if (index == nrecv) flags |= RoleInput;
    } else if (g == 1) {
      if (index < nsend) flags |= RoleWaitSend;
      if (index == nsend) flags |= RoleOutput;
    } else if (g == ng - 2) {
      if (index < nrecv) flags |= RolePostRecv;
    } else if (g == ng - 1) {
      if (index < nsend) flags |= RolePostSend;
    }

    int peer = 0;
    if (flags & (RoleWaitRecv|RolePostRecv)) peer = recvPeers[index];
    if (flags & (RoleWaitSend|RolePostSend)) peer = sendPeers[index];
    if (flags & RoleInput) userBuff = (T*)inputBuf;
    if (flags & RoleOutput) userBuff = (T*)outputBuf;

    loadRecvConn(ncclShmem.channel->devPeers + peer, (T*)outputBuf);
    loadSendConn(ncclShmem.channel->devPeers + peer);
  }

  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, -1, eltN, false);
  }
  __device__ __forceinline__ void sendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, -1, eltN, false);
  }
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t remoteOutIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, -1, remoteOutIx, eltN, false);
  }
  __device__ __forceinline__ void directSendFromOutput(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    genericOp<0, 1, 0, 1, Output, -1>(outIx, -1, remoteOutIx, eltN, false);
  }

  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, -1, Output>(-1, outIx, -1, eltN, postOp);
  }

  __device__ __forceinline__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 0, 1, Input, Output>(inpIx, outIx, remoteOutIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopySend(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    genericOp<1, 1, 1, 1, -1, Output>(-1, outIx, remoteOutIx, eltN, false);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, -1, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    genericOp<0, 1, 1, 1, Input, Output>(inpIx, outIx, remoteOutIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  scatter(intptr_t inpIx, int totalElem, int peerElem, int skip, int shift) {
    ScatterGatherOp<0, 1>(inpIx, -1, totalElem, peerElem, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, int totalElem, int peerElem, int skip, int shift, bool postOp=false) {
    ScatterGatherOp<1, 0>(-1, outIx, totalElem, peerElem, skip, shift, postOp);
  }

  __device__ __forceinline__ ~ncclPrimitives() {
    // Save steps for the next operation
    saveSync();
  }
};

#include "prims_ll.h"
//#include "prims_ll128.h"

#endif
