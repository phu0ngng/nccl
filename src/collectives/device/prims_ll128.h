#define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS-1)

template <typename T, class FUNC, int NRECV, int NSEND>
class ncclLL128Primitives {
 private:
  const int tid;
  const int wid;
  const int wnb;
  const int nthreads;
  int nrecv = 0;
  int nsend = 0;

  struct ncclConnInfo* recvConn[NRECV];
  volatile uint64_t* recvConnHeadPtr;

  struct ncclConnInfo* sendConn[NSEND];
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  uint64_t* recvBuff[NRECV];
  uint64_t* sendBuff[NSEND];
  struct ncclDevComm* comm;

  volatile uint64_t* shmem;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*NCCL_LL128_SLICE_ELEMS; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*NCCL_LL128_SLICE_ELEMS; }
  inline __device__ uint64_t* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ uint64_t* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint64_t recvFlag(int i) { return recvStep[i]+1; }
  inline __device__ uint64_t sendFlag(int i) { return sendStep[i]+1; }

  // Exit If Abort Barrier : make sure all threads exit consistently
  // Each thread sets a predicate to true if val == 1
  // all CTA's threads enter the barrier and do a popc on their predicates being True
  // If any of the thread's predicate was True, all the threads call exit()
  inline __device__ void exitIfAbortLocalBarrier() {
    uint32_t popc;
    asm ("{");
    asm volatile ("   .reg .pred barr_pred;");
    asm volatile ("   setp.eq.u32 barr_pred,%0,1;" :: "r"(abort));
    asm volatile ("   bar.red.popc.u32 %0, 14, %1, barr_pred;" : "=r"(popc) : "r"(nthreads));
    asm ("}");
    if (popc) {
      // Make sure threads not participating in the operation get the abort and all threads exit
      exitIfAbortBarrier(1);
    }
  }

  inline __device__ void barrier() {
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
  }

  uint32_t mismatch = 0;
  const uint64_t opCount;

  inline __device__ void checkMismatch(volatile uint64_t* remoteOpCount) {
    if (mismatch > 20) {
      // We have seen that the peer advanced opcount so many times yet we are still waiting for credit of current op, so it is _most likely_ a mismatch
      // Note that we are not using _threadfence_system in LL so the error cannot be asserted
      *(comm->fatalDevError) = ncclDevSuspectedMismatch;
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

  inline __device__ void waitSend(int i, int nbytes) {
    spins = 0;
    mismatch = 0;
    if (tid == i) {
      while (sendConnHead + NCCL_STEPS < sendStep[i] + 1) {
        sendConnHead = *sendConnHeadPtr;
        if (checkAbort(sendConn[i]->opCountRem)) break;
      }
      if (sendConnFifoPtr) sendConnFifoPtr[sendStep[i]%NCCL_STEPS] = nbytes;
    }
  }

  inline __device__ void postRecv(int i) {
    recvStep[i]++;
    if (tid == i) *recvConnHeadPtr = recvStep[i];
  }

  inline __device__ void postSend(int i, int nbytes) {
    sendStep[i]++;
    if (tid == i && sendConnTailPtr) *sendConnTailPtr = sendStep[i];
  }

  #define WARP_MASK 0xffffffff
  /* TODO : manage < 64bits sizes */
  template <int RECV, int SEND, int SRC, int DST>
  __device__ void SendRecvReduce(uint64_t* src, uint64_t* dst, int& offset, int&ll128offset, int nelems) {
    uint64_t v;
    
    if (SRC && wid != NCCL_LL128_FLAGTHREAD && offset < nelems) {
      v = src[offset];
    }

    if (RECV) {
      volatile uint64_t* ptr = recvPtr(0)+ll128offset;
      uint64_t flag = recvFlag(0);
      uint64_t val;
      do val = *ptr; while (__any_sync(WARP_MASK, (wid == NCCL_LL128_FLAGTHREAD) && (val != flag)));
      val = *ptr;
      v = SRC ? MULTI<FUNC, T>()(val, v) : val;

      for (int i=1; i<NRECV && i<nrecv; i++) {
        volatile uint64_t* ptr = recvPtr(i)+ll128offset;
        uint64_t flag = recvFlag(i);
        uint64_t val;
        do val = *ptr; while (__any_sync(WARP_MASK, (wid == NCCL_LL128_FLAGTHREAD) && (val != flag)));
        val = *ptr;
        v = MULTI<FUNC, T>()(val, v);
      }
    }

    if (SEND) {
      for (int i=1; i<NSEND && i<nsend; i++) {
        volatile uint64_t* ptr = sendPtr(i) + ll128offset;
        int flag = sendFlag(i);
        *ptr = wid == NCCL_LL128_FLAGTHREAD ? flag : v;
      }
      volatile uint64_t* ptr = sendPtr(0) + ll128offset;
      int flag = sendFlag(0);
      *ptr = wid == NCCL_LL128_FLAGTHREAD ? flag : v;
    }

    if (DST && wid != NCCL_LL128_FLAGTHREAD && offset < nelems) {
      dst[offset] = v;
    }

    offset += NCCL_LL128_DATAELEMS;
    ll128offset += NCCL_LL128_LINEELEMS;
  }

  __device__ void load16(uint64_t* ptr, uint64_t& v0, uint64_t &v1) {
    asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v0), "=l"(v1) : "l"(ptr));
    shmem[2*wid] = v0;
    shmem[2*wid+1] = v1;
    __syncwarp();
    v0 = shmem[wid];
    v1 = shmem[NCCL_LL128_DATAELEMS+wid];
    __syncwarp();
  }

  __device__ void store16(uint64_t* ptr, uint64_t& v0, uint64_t &v1) {
    shmem[wid] = v0;
    shmem[wid+NCCL_LL128_DATAELEMS] = v1;
    __syncwarp();
    v0 = shmem[2*wid];
    v1 = shmem[2*wid+1];
    __syncwarp();
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(ptr), "l"(v0), "l"(v1));
  }

  __device__ void ll128recv16(uint64_t* ptr, uint64_t flag, uint64_t& v0, uint64_t &v1) {
    do {
      asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v0), "=l"(v1) : "l"(ptr));
      shmem[2*wid] = v0;
      shmem[2*wid+1] = v1;
      __syncwarp();
    } while (__any_sync(WARP_MASK, (wid == NCCL_LL128_FLAGTHREAD) && (shmem[wid] != flag || shmem[NCCL_LL128_LINEELEMS+wid] != flag)));
    asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v0), "=l"(v1) : "l"(ptr));
    shmem[2*wid] = v0;
    shmem[2*wid+1] = v1;
    __syncwarp();
    v0 = shmem[wid];
    v1 = shmem[wid+NCCL_LL128_LINEELEMS];
    __syncwarp();
  }

  __device__ void ll128send16(uint64_t* ptr, uint64_t& v0, uint64_t &v1) {
    shmem[wid] = v0;
    shmem[wid+NCCL_LL128_LINEELEMS] = v1;
    __syncwarp();
    v0 = shmem[2*wid];
    v1 = shmem[2*wid+1];
    __syncwarp();
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(ptr), "l"(v0), "l"(v1));
  }

  /* Note : UNROLL has to be a multiple of 2 */
  template <int UNROLL, int RECV, int SEND, int SRC, int DST>
  __device__ void SendRecvReduceUnroll(uint64_t* src, uint64_t* dst, int& offset, int&ll128offset, int nelems) {
    uint64_t v[UNROLL];
    uint64_t v0, v1;
    
    if (SRC && wid != NCCL_LL128_FLAGTHREAD) {
      uint64_t* ptr = src+offset+wid;
      for (int u=0; u<UNROLL; u+=2) {
        load16(ptr+u*NCCL_LL128_DATAELEMS, v0, v1);
        v[u] = v0;
        v[u+1] = v1;
      }
    }

    if (RECV) {
      uint64_t flag = recvFlag(0);
      uint64_t* ptr = recvPtr(0)+ll128offset+wid;
      #pragma unroll
      for (int u=0; u<UNROLL; u+=2) {
        ll128recv16(ptr+u*NCCL_LL128_LINEELEMS, flag, v0, v1);
        if (wid != NCCL_LL128_FLAGTHREAD) {
          v[u] = SRC ? MULTI<FUNC, T>()(v0, v[u]) : v0;
          v[u+1] = SRC ? MULTI<FUNC, T>()(v1, v[u+1]) : v1;
        }
      }

      for (int i=1; i<NRECV && i<nrecv; i++) {
        uint64_t flag = recvFlag(i);
        uint64_t* ptr = recvPtr(i)+ll128offset+wid;
        #pragma unroll
        for (int u=0; u<UNROLL; u+=2) {
          ll128recv16(ptr+u*NCCL_LL128_LINEELEMS, flag, v0, v1);
          if (wid != NCCL_LL128_FLAGTHREAD) {
            v[u] = MULTI<FUNC, T>()(v0, v[u]);
            v[u+1] = MULTI<FUNC, T>()(v1, v[u+1]);
          }
        }
      }
    }

    if (SEND) {
      for (int i=1; i<NSEND && i<nsend; i++) {
        int flag = sendFlag(i);
        uint64_t* ptr = sendPtr(i)+ll128offset+wid;
        #pragma unroll
        for (int u=0; u<UNROLL; u+=2) {
          v0 = wid == NCCL_LL128_FLAGTHREAD ? flag : v[u];
          v1 = wid == NCCL_LL128_FLAGTHREAD ? flag : v[u+1];
          ll128send16(ptr+u*NCCL_LL128_LINEELEMS, v0, v1);
        }
      }
      int flag = sendFlag(0);
      uint64_t* ptr = sendPtr(0)+ll128offset+wid;
      #pragma unroll
      for (int u=0; u<UNROLL; u+=2) {
        v0 = wid == NCCL_LL128_FLAGTHREAD ? flag : v[u];
        v1 = wid == NCCL_LL128_FLAGTHREAD ? flag : v[u+1];
        ll128send16(ptr+u*NCCL_LL128_LINEELEMS, v0, v1);
      }
    }

    if (DST && wid != NCCL_LL128_FLAGTHREAD) {
      uint64_t* ptr = dst+offset+wid;
      #pragma unroll
      for (int u=0; u<UNROLL; u+=2) {
        v0 = v[u];
        v1 = v[u+1];
        store16(ptr+u*NCCL_LL128_DATAELEMS, v0, v1);
      }
    }

    offset += NCCL_LL128_DATAELEMS*UNROLL;
    ll128offset += NCCL_LL128_LINEELEMS*UNROLL;
  }

  template <int RECV, int SEND, int SRC, int DST>
  __device__ void GenericOp(const T* srcPtr, T* dstPtr, int nelem) {
    uint32_t nbytes = nelem < 0 ? 0 : nelem*sizeof(T);
    uint64_t* src64Ptr = (uint64_t*)srcPtr;
    uint64_t* dst64Ptr = (uint64_t*)dstPtr;
    // TODO : properly handle small datatypes stopping on in the middle of a uint64
    int nelem64 = DIVUP(nbytes,sizeof(uint64_t));

    int elemsPerIter = (nthreads/NCCL_LL128_LINEELEMS)*NCCL_LL128_DATAELEMS;
    int iters = DIVUP(nelem64,elemsPerIter);
    int fifoNbytes = iters*nthreads*sizeof(uint64_t);

    FOR_SEND(waitSend, fifoNbytes);
    barrier();

    int offset = wnb*iters*NCCL_LL128_DATAELEMS + wid;
    int ll128offset = wnb*iters*NCCL_LL128_LINEELEMS + wid;
    int i = 0;
    // Unroll is ptrs are aligned enough
    if ((((uint64_t)src64Ptr | (uint64_t)dst64Ptr) & 0xf) == 0) {
      #pragma unroll 1
      while (i< iters-7) {
        SendRecvReduceUnroll<8, RECV, SEND, SRC, DST>(src64Ptr, dst64Ptr, offset, ll128offset, nelem64);
        i+=8;
      }
    }
    #pragma unroll 1
    while (i< iters) {
      SendRecvReduce<RECV, SEND, SRC, DST>(src64Ptr, dst64Ptr, offset, ll128offset, nelem64);
      i++;
    }

    exitIfAbortLocalBarrier();
    FOR_RECV(postRecv);
    FOR_SEND(postSend, fifoNbytes);
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvConn[i] = conn;
    recvBuff[i] = recvConn[i]->ll128Buff;
    recvStep[i] = recvConn[i]->step;
    if (tid == i) {
      recvConnHeadPtr = recvConn[i]->head;
      *(recvConn[i]->opCountLoc) = opCount;
    }
    nrecv++;
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendConn[i] = conn;
    sendBuff[i] = sendConn[i]->ll128Buff;
    sendStep[i] = sendConn[i]->step;
    if (tid == i) {
      sendConnFifoPtr = sendConn[i]->fifo;
      if (sendConnFifoPtr) sendConnTailPtr = sendConn[i]->tail;
      sendConnHeadPtr = sendConn[i]->head;
      sendConnHead = *sendConnHeadPtr;
      *(sendConn[i]->opCountLoc) = opCount;
    }
    nsend++;
  }

  __device__ __forceinline__ void saveRecvConn(int i) {
    if (tid == i) {
      recvConn[i]->step = recvStep[i];
      *(recvConn[i]->opCountLoc) += 1;
      __threadfence_block();
    }
  }

  __device__ __forceinline__ void saveSendConn(int i) {
    if (tid == i) {
      sendConn[i]->step = sendStep[i];
      *(sendConn[i]->opCountLoc) += 1;
      __threadfence_block();
    }
  }

 public:
  __device__ __forceinline__
  ncclLL128Primitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount)
    : comm(comm), tid(tid), wid(tid%NCCL_LL128_LINEELEMS), wnb(tid/NCCL_LL128_LINEELEMS), nthreads(nthreads), opCount(opCount), shmem(((volatile uint64_t*)ncclShmem)+wnb*NCCL_LL128_LINEELEMS*2) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
  }

  __device__ void send(const T* src, int nelem) {
    return GenericOp<0, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recv(T* dst, int nelem) {
    return GenericOp<1, 0, 0, 1>(NULL, dst, nelem);
  }

  __device__ void recvReduceSend(const T* src, int nelem) {
    return GenericOp<1, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recvReduceCopy(const T* src, T* dst, int nelem) {
    return GenericOp<1, 0, 1, 1>(src, dst, nelem);
  }

  __device__ void copySend(const T* src, T* dst, int nelem) {
    return GenericOp<0, 1, 1, 1>(src, dst, nelem);
  }

  __device__ void recvCopySend(T* dst, int nelem) {
    return GenericOp<1, 1, 0, 1>(NULL, dst, nelem);
  }

  __device__ void recvReduceCopySend(const T* src, T* dst, int nelem) {
    return GenericOp<1, 1, 1, 1>(src, dst, nelem);
  }

  __device__ __forceinline__ ~ncclLL128Primitives() {
    // Save steps for the next operation
    for (int i=0; i<NRECV && i<nrecv; i++) saveRecvConn(i);
    for (int i=0; i<NSEND && i<nsend; i++) saveSendConn(i);
  }
};
