/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

template<typename T, typename RedOp, typename Fan, int Direct,
         int SlicePerChunk, int StepPerSlice, int Unroll>
class Primitives<
    T, RedOp, Fan, Direct, ProtoSimple<SlicePerChunk, StepPerSlice, Unroll>
  > {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1;
  static constexpr int RoleInput = 0x01,
                       RoleOutput = 0x02,
                       RoleWaitRecv = 0x04,
                       RoleWaitSend = 0x08,
                       RolePostSend = 0x10,
                       RolePostRecv = 0x20,
                       Aborted = 0x40,
                       PtrsFifoEnabled = 0x80,
                       SizesFifoEnabled = 0x100,
                       DirectWrite = 0x200,
                       DirectRead = 0x400,
                       ThreadsSynced = 0x800;
  const int tid;
  int nthreads;
  int nworkers;
  const int stepSize;
  Fan fan;
  RedOp const redOp;
  int index; // Peer index I'm responsible for
  int flags;
  int group;
  int connIndex;
  uint64_t step;
  union {
    void **connPtrsFifoPtr; // (flags & PtrsFifoEnabled)
    T *userBuff;            // (flags & (RoleInput|RoleOutput))
    T *connEltsFifo;        // !(flags & (PtrsFifoEnabled|RoleInput|RoleOutput))
  };
  int volatile *connSizesFifoPtr; //  (flags & SizesFifoEnabled)
  T *directBuff;                  // !(flags & SizesFifoEnabled)
  uint64_t volatile *connStepPtr;
  uint64_t connStepCache; // Cache last seen value of (*connStepPtr)

  // Don't use barrier 0 as it's used by the final sync
  inline __device__ void barrier() {
    if (nthreads == WARP_SIZE)
      __syncwarp();
    else
      asm volatile("bar.sync %0, %1;" :: "r"(group+1), "r"(nthreads));
    flags |= ThreadsSynced;
  }
  inline __device__ void subBarrier() {
    if (nworkers == nthreads)
      barrier();
    else
      asm volatile("bar.sync %0, %1;" :: "r"(group+2), "r"(nworkers));
  }

  inline __device__ bool checkAbort(int &spins) {
    spins++;
    if (!(flags & Aborted) && spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
      flags |= *ncclShmem.comm.abortFlag ? Aborted : 0;
      spins = 0;
    }
    return flags & Aborted;
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  inline __device__ void waitPeer(intptr_t dstIx, intptr_t remoteIx, int offset, int nelts) {
    bool const isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
    if (((flags & (Recv*RoleWaitRecv)) && !(DirectRecv && Src && (flags & DirectRead))) || // no wait when directly reading from remote input
        ((flags & (Send*RoleWaitSend)) && !DirectSend)) { // no wait in empty send (e.g. directScatter) or direct write
      int spins = 0;
      while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
        connStepCache = *connStepPtr;
        if (checkAbort(spins)) break;
        //if (spins == 0) printf("r=%d b=%d t=%d SPUN OUT got=%d want=%d\n", ncclShmem.comm.rank, blockIdx.x, threadIdx.x, int(connStepCache + (isSendNotRecv ? NCCL_STEPS : 0)), int(step+StepPerSlice));
      }
      //printf("Rank %d group %d connIndex %d index %d wait for %s spins %d\n", ncclShmem.comm.rank, group, connIndex, index, isSendNotRecv ? "send" : "recv", spins);
    }

    if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
      if (isSendNotRecv && (flags & SizesFifoEnabled))
        connSizesFifoPtr[step%NCCL_STEPS] = nelts*sizeof(T);

      void **ptrs = isSendNotRecv ? (ncclShmem.groups[group].dsts + Dst)
                                  : (ncclShmem.groups[group].srcs + Src);
      if (flags & PtrsFifoEnabled)
        loadPtr(connPtrsFifoPtr + step%NCCL_STEPS, ptrs[index]);
      else if (isSendNotRecv && DirectSend) {
        if (flags & DirectWrite) {
          ptrs[index] = directBuff + remoteIx + offset;
          //printf("Rank %d group %d connIndex %d index %d direct send %p\n", ncclShmem.comm.rank, group, connIndex, index, ptrs[index]);
        } else {
          ptrs[index] = nullptr;
        }
      } else if ((!isSendNotRecv) && DirectRecv) {
        if (flags & DirectRead) {
          ptrs[index] = directBuff + remoteIx + offset;
          //printf("Rank %d group %d connIndex %d index %d direct recv %p\n", ncclShmem.comm.rank, group, connIndex, index, ptrs[index]);
        } else {
          ptrs[index] = nullptr;
        }
      }
      else {
        ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
        //printf("Rank %d group %d connIndex %d index %d %s %p\n", ncclShmem.comm.rank, group, connIndex, index, isSendNotRecv ? "send" : "recv", ptrs[index]);
      }
      step += StepPerSlice;
    }
  }

  template<int Recv, int Send>
  inline __device__ void postPeer() {
    if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
      step += StepPerSlice;
      *connStepPtr = step;
    }
  }

  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  inline __device__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, intptr_t remoteIx, int nelem, bool postOp
    ) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
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
      // numslices>2 the new code is superior. And note that in the case
      // numslices=1, the loop is trivially unrollable (single iteration) so we
      // don't incur that that tail branch and we still have perf_new=2.
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
      #if __CUDA_ARCH__ < 700
        // Yeah, so all that above don't matter a lick on older hardware.
        #pragma unroll SlicePerChunk
      #else
        #pragma unroll 1
      #endif
      do {
        sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
        if (Src && (flags & (SrcBuf==Input ? RoleInput : RoleOutput)))
          ncclShmem.groups[group].srcs[0] = userBuff + srcIx + offset;
        if (Dst && (flags & (DstBuf==Input ? RoleInput : RoleOutput)))
          ncclShmem.groups[group].dsts[0] = userBuff + dstIx + offset;
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(dstIx, remoteIx, offset, sliceSize);
        subBarrier();
        if (DirectRecv && DirectSend/*ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]*/) {
          // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
          // (1-Send) is only there to avoid compilation errors in case MaxSend=0 (and Send=0).
          ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, (1-Send)+MaxSend>
            (tid, nworkers, redOp, false, false,
             1, (T const**)ncclShmem.groups[group].srcs,
             fan.nsend(), (T**)ncclShmem.groups[group].dsts+1,
             sliceSize);
          //if (tid == 0) printf("Rank %d group %d connIndex %d tid %d in path 1\n", ncclShmem.comm.rank, group, connIndex, tid);
        } else if (DirectRecv && Send) {
          // For reducer in CollNet to do direct fetch
          ReduceOrCopyMulti<Unroll, RedOp, T, 1+Src, MaxRecv+Src, 1+Dst, MaxSend+Dst>
            (tid, nworkers, redOp, SrcBuf==Input, postOp,
             fan.nrecv()+Src, (T const**)ncclShmem.groups[group].srcs,
             fan.nsend()+Dst, (T**)ncclShmem.groups[group].dsts,
             sliceSize);
          //if (tid == 0) printf("Rank %d group %d connIndex %d tid %d in path 3\n", ncclShmem.comm.rank, group, connIndex, tid);
        } else if (DirectSend && !DirectRecv && SrcBuf != Input && ncclShmem.groups[group].dsts[Dst] == nullptr) {
          // For broadcast in CollNet to do empty send
          ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1>
            (tid, nworkers, redOp, false, postOp,
             Recv, (T const**)ncclShmem.groups[group].srcs,
             Dst, (T**)ncclShmem.groups[group].dsts,
             sliceSize);
        } else {
          ReduceOrCopyMulti<Unroll, RedOp, T, Recv+Src, Recv*MaxRecv+Src, Send+Dst, Send*MaxSend+Dst>
            (tid, nworkers, redOp, SrcBuf==Input, postOp,
             Recv*fan.nrecv()+Src, (T const**)ncclShmem.groups[group].srcs,
             Send*fan.nsend()+Dst, (T**)ncclShmem.groups[group].dsts,
             sliceSize);
        }
        barrier(); // This barrier has a counterpart in following loop
        if (Send && (flags & RolePostSend) && index == 0) __threadfence_system();
        __syncwarp();
        postPeer<Recv, Send>();
        offset += sliceSize;
        slice += 1;
      } while (slice < SlicePerChunk && offset < nelem);
    }

    // Non-workers come straight here. Workers too but only once the remaining
    // slices are all empty. Since empty slices are the uncommon case, and
    // worker perf is the limiter, perf-wise this loop is effectively unentered,
    // hence just a single branch insn.
    #pragma unroll 1
    while (slice < SlicePerChunk) {
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

  // Scatter do not support Direct
  template <int DirectRecv1, int DirectSend1, int Recv, int Send>
  inline __device__ void
  ScatterGatherOp(intptr_t inpIx, intptr_t outIx, int totalElem, int peerElem, int skip, int shift, bool postOp) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    int offset = 0; // slice offset
    int sliceSize = stepSize*StepPerSlice;
    int dataSize = max(DIVUP(peerElem, 16*SlicePerChunk)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SlicePerChunk; ++slice) {
      int realSize = max(0, min(dataSize, peerElem-offset));
      if (tid < nworkers) {
        if (Send && (flags & RoleInput)) ncclShmem.groups[group].srcs[0] = userBuff + inpIx + offset;
        if (Recv && (flags & RoleOutput)) ncclShmem.groups[group].dsts[0] = userBuff + outIx + offset;
        // realSize is not accurate here; but intra-node does not rely on sizes FIFO
        waitPeer<DirectRecv, DirectSend, Recv, Send, Send, Recv>(0, Send ? inpIx : outIx, offset, realSize);
        subBarrier();
        if (DirectSend && ncclShmem.groups[group].dsts[0] == nullptr) {
          // Do nothing
          //if (tid == 0) printf("Rank %d group %d connIndex %d tid %d in empty scatter\n", ncclShmem.comm.rank, group, connIndex, tid);
        } else if (Send) {
          #pragma unroll
          for (int j=0; j<fan.nsend(); j++) {
            int i = (j+shift)%fan.nsend();
            int peerOffset = i*peerElem;
            if (skip >= 0 && i >= skip) peerOffset += peerElem;
            const T* src0 = (T*)ncclShmem.groups[group].srcs[0] + peerOffset;
            int realPeerSize = min(realSize, totalElem-peerOffset);
            if (realPeerSize > 0) ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1>(tid, nworkers, redOp, true, false, 1, &src0, 1, (T**)ncclShmem.groups[group].dsts+i, realPeerSize);
          }
          //if (tid == 0) printf("Rank %d group %d connIndex %d tid %d in solid scatter\n", ncclShmem.comm.rank, group, connIndex, tid);
        } else if (DirectRecv && ncclShmem.groups[group].srcs[0] == nullptr) {
          // Do nothing
          //if (tid == 0) printf("Rank %d group %d connIndex %d tid %d in empty gather\n", ncclShmem.comm.rank, group, connIndex, tid);
        } else if (Recv) {
          #pragma unroll
          for (int j=0; j<fan.nrecv(); j++) {
            int i = (j+shift)%fan.nrecv();
            int peerOffset = i*peerElem;
            if (skip >= 0 && i >= skip) peerOffset += peerElem;
            T* dst0 = (T*)ncclShmem.groups[group].dsts[0] + peerOffset;
            const T* src = (const T*)ncclShmem.groups[group].srcs[i] + peerOffset;
            int realPeerSize = min(realSize, totalElem-peerOffset);
            if (realPeerSize > 0) ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1>(tid, nworkers, redOp, false, postOp, 1, &src, 1, &dst0, realPeerSize);
          }
        }
      }
      barrier();
      if (Send && (flags & RolePostSend) && realSize > 0 && index == 0 &&
          !(DirectSend && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0])) {
        __threadfence_system();
        __syncwarp();
      }
      postPeer<Recv, Send>();
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclPeer *peer) {
    if (flags & (RoleWaitRecv|RolePostRecv)) {
      auto *conn = &peer->recv[connIndex%2].conn;
      step = conn->step;
      step = roundUp(step, SlicePerChunk*StepPerSlice);
      if (flags & RolePostRecv) {
        connStepPtr = conn->head;
        *connStepPtr = step; // Return credits in case we rounded up.
      }
      if (flags & RoleWaitRecv) {
        ncclShmem.groups[group].recvConns[index] = conn; // WaitRecv role saves since that's who needs it in setDataPtrs()
        connStepPtr = conn->tail;
        connStepCache = *connStepPtr;
        flags |= (conn->ptrsFifo != nullptr) ? PtrsFifoEnabled : 0;
        if (Direct) {
          flags |= (conn->direct & NCCL_DIRECT_WRITE) ? DirectWrite :
                   (conn->direct & NCCL_DIRECT_READ)  ? DirectRead  : 0;
          // Flip direct read/write if connIndex >= 2
          if (connIndex >= 2 && (flags & (DirectWrite|DirectRead)))
            flags ^= DirectWrite|DirectRead;
        }
        if (flags & PtrsFifoEnabled)
          connPtrsFifoPtr = conn->ptrsFifo;
        else
          connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclPeer *peer) {
    if (flags & (RoleWaitSend|RolePostSend)) {
      auto *conn = &peer->send[connIndex%2].conn;
      step = conn->step;
      step = roundUp(step, SlicePerChunk*StepPerSlice);
      if (flags & RolePostSend) {
        connStepPtr = conn->tail;
      }
      if (flags & RoleWaitSend) {
        ncclShmem.groups[group].sendConns[index] = conn; // WaitSend role saves since that's who needs it in setDataPtrs()
        connStepPtr = conn->head;
        connStepCache = *connStepPtr;
        flags |= (conn->ptrsFifo != nullptr) ? PtrsFifoEnabled : 0;
        if (flags & PtrsFifoEnabled)
          connPtrsFifoPtr = conn->ptrsFifo;
        else
          connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];

        if (conn->sizesFifo != nullptr) {
          flags |= SizesFifoEnabled;
          connSizesFifoPtr = conn->sizesFifo;
        } else if (Direct) {
          flags |= (conn->direct & NCCL_DIRECT_WRITE) ? DirectWrite :
                   (conn->direct & NCCL_DIRECT_READ)  ? DirectRead  : 0;
          // Flip direct read/write if connIndex >= 2
          if (connIndex >= 2 && (flags & (DirectWrite|DirectRead)))
            flags ^= DirectWrite|DirectRead;
        }
      }
    }
  }

 public:
  __device__ Primitives(
      int tid, int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, int group=0, int connIndex=0
    ):
    tid(tid),
    stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T)),
    redOp(FuncTraits<RedOp>::make(ncclShmem.comm.nRanks)) {

    // For send operations, we need an extra warp to overlap the threadfence and the copy
    this->nthreads = nthreads;
    this->nworkers = nthreads - (MaxSend > 0 && nthreads-WARP_SIZE >= 64 ? WARP_SIZE : 0);
    this->group = group;
    this->connIndex = connIndex;

    int nrecv=0, nsend=0;
    while (nrecv < MaxRecv && recvPeers[nrecv] != -1) nrecv++;
    while (nsend < MaxSend && sendPeers[nsend] != -1) nsend++;
    this->fan = Fan(nrecv, nsend);

    constexpr int ThreadPerSync = 8;
    static_assert(MaxSend < ThreadPerSync && MaxRecv < ThreadPerSync, "Not enough threads to cover all peers");

    int g = tid / ThreadPerSync;
    int ng = nthreads / ThreadPerSync;
    index = tid % ThreadPerSync;
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

    loadRecvConn(&ncclShmem.channel.devPeers[peer]);
    loadSendConn(&ncclShmem.channel.devPeers[peer]);

    setDataPtrs(inputBuf, outputBuf);
  }

  __device__ ~Primitives() {
    // Ensure ncclShmem.groups[].send/recvConns are available
    if (!(flags & ThreadsSynced))
      barrier();
    // Save steps for the next operation
    if (flags & (RolePostSend|RolePostRecv)) {
      auto *conns = (flags & RolePostSend) ? ncclShmem.groups[group].sendConns : ncclShmem.groups[group].recvConns;
      conns[index]->step = step;
    }
    // Make sure all threads are done writing back conn->step and done using
    // ncclShmem.groups[group]
    barrier();
  }

  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf) {
    if (flags & RoleInput) userBuff = (T*)inputBuf;
    if (flags & RoleOutput) userBuff = (T*)outputBuf;
    bool recvProvider = flags == (flags|RoleWaitRecv|DirectWrite);
    bool sendAcceptor = flags == (flags|RoleWaitSend|DirectWrite);
    bool sendProvider = flags == (flags|RoleWaitSend|DirectRead); // sender provides direct buffer (to be fetched)
    bool recvAcceptor = flags == (flags|RoleWaitRecv|DirectRead); // receiver accepts direct buffer
    if (Direct && recvProvider) {
      int spins = 0;
      void *volatile *slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
      // Wait for consumer to consume previous value before trampling it.
      while (*slot != nullptr && !checkAbort(spins)) {
      }
      directBuff = (T*)outputBuf;
      //printf("Rank %d group %d connIndex %d tid %d spins %d recv provide slot %p %p\n", ncclShmem.comm.rank, group, connIndex, tid, spins, slot, directBuff);
      // Encode pointer by XOR'ing against some address they definitely wouldn't send
      // since we want to allow them sending us nullptr while not colliding with
      // the empty slot value.
      *slot = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(directBuff) ^ reinterpret_cast<uintptr_t>(slot));
    }
    if (Direct && sendAcceptor) {
      int spins = 0;
      void *volatile *slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
      void *ptr;
      while (true) {
        ptr = *slot;
        if (ptr != nullptr || checkAbort(spins)) break;
      }
      directBuff = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(ptr) ^ reinterpret_cast<uintptr_t>(slot));
      *slot = nullptr;
      //printf("Rank %d group %d connIndex %d tid %d spins %d send accept slot %p %p\n", ncclShmem.comm.rank, group, connIndex, tid, spins, slot, directBuff);
    }
    if (Direct && sendProvider) {
      int spins = 0;
      void *volatile *slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
      // Wait for consumer to consume previous value before trampling it.
      while (*slot != nullptr && !checkAbort(spins)) {
        //if (spins % 0x10000 == 0) printf("Rank %d group %d connIndex %d tid %d spins %d sendProvider waiting for slot %p %p\n", ncclShmem.comm.rank, group, connIndex, tid, spins, slot, *slot);
      }
      // If there is no recv, then we are directly pulling from input buffer (e.g. directScatter)
      // Otherwise, we are pulling from output buffer (e.g. recvCopyDirectSend)
      directBuff = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
      // Encode pointer by XOR'ing against some address they definitely wouldn't send
      // since we want to allow them sending us nullptr while not colliding with
      // the empty slot value.
      *slot = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(directBuff) ^ reinterpret_cast<uintptr_t>(slot));
    }
    if (Direct && recvAcceptor) {
      int spins = 0;
      void *volatile *slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
      void *ptr;
      while (true) {
        ptr = *slot;
        if (ptr != nullptr || checkAbort(spins)) break;
        //if (spins % 0x10000 == 0) printf("Rank %d group %d connIndex %d tid %d spins %d recvAcceptor waiting for slot %p %p\n", ncclShmem.comm.rank, group, connIndex, tid, spins, slot, *slot);
      }
      directBuff = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(ptr) ^ reinterpret_cast<uintptr_t>(slot));
      *slot = nullptr;
    }
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    if (flags & (RoleInput|RoleOutput))
      userBuff += delta;
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
  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN) {
    genericOp<1, 0, 1, 0, -1, Output>(-1, outIx, -1, eltN, /*postOp=*/false);
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
  __device__ __forceinline__ void recvCopyDirectSend(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    genericOp<0, 1, 1, 1, -1, Output>(-1, outIx, remoteOutIx, eltN, false);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceSend(intptr_t inpIx, intptr_t remoteInpIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, Input, -1>(inpIx, -1, remoteInpIx, eltN, postOp); //FIXME: Recv = 1
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
    ScatterGatherOp<0, 0, 0, 1>(inpIx, -1, totalElem, peerElem, skip, shift, /*postOp=*/false);
  }
  __device__ __forceinline__ void
  directScatter(intptr_t inpIx, int totalElem, int peerElem, int skip, int shift) {
    ScatterGatherOp<0, 1, 0, 1>(inpIx, -1, totalElem, peerElem, skip, shift, /*postOp=*/false); //FIXME: Send = 1
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, int totalElem, int peerElem, int skip, int shift, bool postOp=false) {
    ScatterGatherOp<0, 0, 1, 0>(-1, outIx, totalElem, peerElem, skip, shift, postOp);
  }
  __device__ __forceinline__ void
  directGather(intptr_t outIx, int totalElem, int peerElem, int skip, int shift) {
    ScatterGatherOp<1, 0, 1, 0>(-1, outIx, totalElem, peerElem, skip, shift, /*postOp=*/false);
  }
};
