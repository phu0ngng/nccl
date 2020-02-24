/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclSendRecvKernel(struct CollectiveArgs* args) {
  if(args->rankDelta==0) return; //NOOP
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads-WARP_SIZE;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;

  const ssize_t sendSize = args->p2p.sendCount;
  const ssize_t recvSize = args->p2p.recvCount;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * SENDRECV_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;
  int peerRecv = recvSize>0?(int)(((int)comm->rank+(int)comm->nRanks-(int)args->rankDelta)%comm->nRanks):-1;
  int peerSend = sendSize>0?(int)((comm->rank+args->rankDelta)%comm->nRanks):-1;
  // Compute pointers
  const T * __restrict__ sendbuff = (const T*)args->sendbuff;
  T * __restrict__ recvbuff = (T*)args->recvbuff;
  //if(tid==0) printf("rank %d[ch%d] delta %d sendrecv kernel  %ld send to %d  %ld recv peer %d\n",comm->rank,blockIdx.x,args->rankDelta,sendSize,peerSend,recvSize,peerRecv);
  
  ncclPrimitives<UNROLL, SENDRECV_CHUNKSTEPS/SENDRECV_SLICESTEPS, SENDRECV_SLICESTEPS, T, 1, 1, FUNC>
    prims(tid, args->nThreads, &peerRecv, &peerSend, NULL, stepSize, channel, comm, args->opCount);

  int maxSize = sendSize>recvSize ? sendSize : recvSize;

  for (ssize_t gridOffset = 0; gridOffset < maxSize; gridOffset += loopSize) {
    if(gridOffset<sendSize) {
      int realChunkSize = min(chunkSize, DIVUP(sendSize-gridOffset,args->nChannels));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      ssize_t offset = gridOffset + bid*realChunkSize;
      int nelem = min(realChunkSize, sendSize-offset);
      prims.send(sendbuff+offset, nelem);
    }
    if(gridOffset<recvSize){
      int realChunkSize = min(chunkSize, DIVUP(recvSize-gridOffset,args->nChannels));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      ssize_t offset = gridOffset + bid*realChunkSize;
      int nelem = min(realChunkSize, recvSize-offset);  
      prims.recv(recvbuff+offset, nelem);
    }
  }

}
