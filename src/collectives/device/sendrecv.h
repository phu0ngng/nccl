/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct CollectiveArgs* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->p2p.nThreads-2*WARP_SIZE;

      // Compute pointers
      const T* sendbuff = (const T*)args->p2p.sendbuff[0];
      T* recvbuff = (T*)args->p2p.recvbuff[0];
      const ssize_t sendCount = args->p2p.sendCount[0];
      const ssize_t recvCount = args->p2p.recvCount[0];
      const int delta = args->p2p.delta[0];

      if (delta < 0 ) return; // No-op

      if (delta == 0) {
        if (tid < nthreads && sendbuff != recvbuff) {
          // local copy : ReduceOrCopyMulti takes an int as number of elements,
          // so we split it in blocks of 1G elements.
          int blockSize = 1<<30;
          for (size_t offset=0; offset<sendCount; offset += blockSize) {
            size_t remaining = sendCount - offset;
            if (remaining < blockSize) blockSize = remaining;
            ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nthreads, 1, &sendbuff, 1, &recvbuff, blockSize);
            sendbuff += blockSize; recvbuff += blockSize;
          }
        }
        return;
      }

      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels+blockIdx.x;

      const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/(sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize/SENDRECV_SLICEFACTOR;

      int nthreadsSplit = nthreads/2;

      if (tid < nthreadsSplit + WARP_SIZE) {
        if (sendCount < 0) return;

        int peer = (comm->rank+delta)%comm->nRanks;
        ncclPrimitives<UNROLL, 1, 1, T, 0, 1, 1, FUNC, 0>
          prims(tid, nthreadsSplit, NULL, &peer, recvbuff, stepSize, channel, comm, ncclShmem->ptrs);

        if (sendCount == 0) {
          prims.send(sendbuff, 0);
        } else for (ssize_t offset = 0; offset < sendCount; offset += chunkSize) {
          int realChunkSize = min(chunkSize, sendCount-offset);
          ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
          int nelem = min(realChunkSize, sendCount-offset);
          prims.directSend(sendbuff+offset, offset, nelem);
        }
      } else {
        if (recvCount < 0) return;

        int peer = (comm->rank-delta+comm->nRanks)%comm->nRanks;
        ncclPrimitives<UNROLL, 1, 1, T, 1, 0, 1, FUNC, 1>
          prims(tid-nthreadsSplit-WARP_SIZE, nthreads-nthreadsSplit, &peer, NULL, recvbuff, stepSize, channel, comm, ncclShmem->ptrs+1);

        if (recvCount == 0) {
          prims.recv(recvbuff, 0);
        } else for (ssize_t offset = 0; offset < recvCount; offset += chunkSize) {
          int realChunkSize = min(chunkSize, recvCount-offset);
          ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
          int nelem = min(realChunkSize, recvCount-offset);
          prims.directRecv(recvbuff+offset, offset, nelem);
        }
      }
    }
};
