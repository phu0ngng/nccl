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
      const int nthreads = args->p2p.nThreads-WARP_SIZE;

      // Compute pointers
      const T* sendbuff = (const T*)args->sendbuff;
      T* recvbuff = (T*)args->recvbuff;

      if (args->p2p.delta < 0 ) return; // No-op

      if (args->p2p.delta == 0) {
        if (tid < nthreads && sendbuff != recvbuff) {
          // local copy : ReduceOrCopyMulti takes an int as number of elements,
          // so we split it in blocks of 1G elements.
          int blockSize = 1<<30;
          for (size_t offset=0; offset<args->p2p.sendCount; offset += blockSize) {
            size_t remaining = args->p2p.sendCount - offset;
            if (remaining < blockSize) blockSize = remaining;
            ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, 1>(tid, nthreads, 1, &sendbuff, 1, &recvbuff, blockSize);
            sendbuff += blockSize; recvbuff += blockSize;
          }
        }
        return;
      }

      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels+blockIdx.x;

      const ssize_t sendSize = args->p2p.sendCount;
      const ssize_t recvSize = args->p2p.recvCount;
      const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * SENDRECV_CHUNKSTEPS;
      int peerRecv = recvSize >= 0 ? (comm->rank-(int)args->p2p.delta+comm->nRanks)%comm->nRanks : -1;
      int peerSend = sendSize >= 0 ? (comm->rank+(int)args->p2p.delta)%comm->nRanks : -1;

      ncclPrimitives<UNROLL, SENDRECV_CHUNKSTEPS/SENDRECV_SLICESTEPS, SENDRECV_SLICESTEPS, T, 1, 1, 1, FUNC>
        prims(tid, nthreads, &peerRecv, &peerSend, recvbuff, stepSize, channel, comm, args->opCount);

      int maxSize = sendSize-chunkSize>recvSize ? sendSize-chunkSize : recvSize;

      if (sendSize >= 0) {
        int realChunkSize = min(chunkSize, sendSize);
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        int nelem = min(realChunkSize, sendSize);
        prims.directSend(sendbuff, 0, nelem);
      }

      for (ssize_t gridOffset = 0; gridOffset < maxSize; gridOffset += chunkSize) {
        if (gridOffset+chunkSize < sendSize) {
          int realChunkSize = min(chunkSize, sendSize-gridOffset-chunkSize);
          ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
          ssize_t offset = gridOffset + chunkSize;
          int nelem = min(realChunkSize, sendSize-offset);
          prims.directSend(sendbuff+offset, offset, nelem);
        }
        if (gridOffset < recvSize) {
          int realChunkSize = min(chunkSize, recvSize-gridOffset);
          ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
          ssize_t offset = gridOffset;
          int nelem = min(realChunkSize, recvSize-offset);
          prims.directRecv(recvbuff+offset, offset, nelem);
        }
      }

      if (recvSize == 0) prims.recv(recvbuff, 0);
    }
};
