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
      int tid = threadIdx.x;
      const int nThreadsTotal = args->p2p.nThreads;

      for (int s=0; s<NCCL_MAX_SEGMENTS; s++) {
        const int delta = args->p2p.delta[s];
        if (delta < 0 ) return; // No-op

        int nThreadsSegment = args->p2p.nThreadsPerOp[s];
        if (tid < nThreadsSegment) {
          const int nThreads = nThreadsSegment > 128 ? nThreadsSegment-WARP_SIZE : nThreadsSegment;

          // Compute pointers
          const T* sendbuff = (const T*)args->p2p.sendbuff[s];
          T* recvbuff = (T*)args->p2p.recvbuff[s];
          const ssize_t sendCount = args->p2p.sendCount[s];
          const ssize_t recvCount = args->p2p.recvCount[s];

          if (delta == 0) {
            if (tid < nThreads && sendbuff != recvbuff) {
              // local copy : ReduceOrCopyMulti takes an int as number of elements,
              // so we split it in blocks of 1G elements.
              int blockSize = 1<<30;
              for (size_t offset=0; offset<sendCount; offset += blockSize) {
                size_t remaining = sendCount - offset;
                if (remaining < blockSize) blockSize = remaining;
                ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nThreads, 1, &sendbuff, 1, &recvbuff, blockSize);
                sendbuff += blockSize; recvbuff += blockSize;
              }
            }
          } else {
            struct ncclDevComm* comm = args->comm;
            struct ncclChannel* channel = comm->channels+blockIdx.x;

            const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/(sizeof(T)*NCCL_STEPS);
            const int chunkSize = stepSize/SENDRECV_SLICEFACTOR;

            int nThreadsSplit = nThreads/2;
            if ((tid < nThreadsSplit) && recvCount >= 0) {
              int peer = (comm->rank-delta+comm->nRanks)%comm->nRanks;
              int nt = nThreadsSplit;
              //if (comm->rank == 0 && tid == 0) printf("[%d-%d-%d] Recv %d / %d / %d\n", s, delta, peer, nThreadsSegment, nThreadsSplit, nt);
              ncclPrimitives<UNROLL, 1, 1, T, 1, 0, 1, FUNC>
                prims(tid, nt, &peer, NULL, recvbuff, stepSize, channel, comm, ncclShmem->ptrs, 2*s);

              if (recvCount == 0) {
                prims.recv(recvbuff, 0);
              } else for (ssize_t offset = 0; offset < recvCount; offset += chunkSize) {
                int realChunkSize = min(chunkSize, recvCount-offset);
                ALIGN_SIZE(realChunkSize, nt*sizeof(uint64_t)/sizeof(T));
                int nelem = min(realChunkSize, recvCount-offset);
                prims.directRecv(recvbuff+offset, offset, nelem);
              }
            } else if ((tid >= nThreadsSplit) && sendCount >= 0) {
              int peer = (comm->rank+delta)%comm->nRanks;
              int nt = nThreads-nThreadsSplit;
              //if (comm->rank == 0 && tid == 0) printf("[%d-%d-%d] Send %d / %d / %d\n", s, delta, peer, nThreadsSegment, nThreadsSplit, nt);
              ncclPrimitives<UNROLL, 1, 1, T, 0, 1, 1, FUNC>
                prims(tid-nThreadsSplit, nt, NULL, &peer, recvbuff, stepSize, channel, comm, ncclShmem->ptrs, 2*s+1);

              if (sendCount == 0) {
                prims.send(sendbuff, 0);
              } else for (ssize_t offset = 0; offset < sendCount; offset += chunkSize) {
                int realChunkSize = min(chunkSize, sendCount-offset);
                ALIGN_SIZE(realChunkSize, nt*sizeof(uint64_t)/sizeof(T));
                int nelem = min(realChunkSize, sendCount-offset);
                prims.directSend(sendbuff+offset, offset, nelem);
              }
            }
          }
        }
        tid = (tid+nThreadsTotal-nThreadsSegment) % nThreadsTotal;
      }
    }
};
