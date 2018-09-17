/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel.h"
#include "param.h"

NCCL_PARAM(Buffsize, "BUFFSIZE", DEFAULT_BUFFER_SIZE_BYTES);

ncclResult_t initChannel(struct ncclComm* comm, int channelid) {
  struct ncclChannel* channel = comm->channels+channelid;
  channel->id = channelid;

  // Setup intermediate buffering
  channel->buffSize = ncclParamBuffsize();

  // Pre-configure send/recv pointers. Those are the default, they may change later.
  //channel->ring.recv.conn.buff = recvMem->buff;
  //channel->ring.recv.conn.llBuff = recvMem->llBuff;
  //channel->ring.recv.conn.tail = &recvMem->tail;
  //channel->ring.recv.conn.opCount = &recvMem->opCount;
  //channel->ring.send.conn.head = &sendMem->head;
  //channel->ring.send.conn.llHead = &sendMem->llHead;
  //channel->ring.recv.conn.direct = 0;
  //channel->ring.send.conn.direct = 0;
  //channel->ring.send.conn.llStep = 0;
  //channel->ring.send.conn.llLastCleaning = 0;

  // Ring index to user rank table.
  NCCLCHECK(ncclCudaCalloc(&channel->ring.devUserRanks, comm->nRanks));
  NCCLCHECK(ncclCalloc(&channel->ring.userRanks, comm->nRanks));

  // Per-channel operation list.
  NCCLCHECK(ncclCudaHostAlloc((void**)&channel->collectives, (void**)&channel->devCollectives, sizeof(struct ncclColl)*NCCL_MAX_OPS));
  return ncclSuccess;
}

ncclResult_t freeChannel(struct ncclChannel* channel) {
  // Operation list
  NCCLCHECK(ncclCudaHostFree(channel->collectives));

  // Free Ring
  struct ncclRing* ring = &channel->ring;
  // Index to rank table
  free(ring->userRanks);
  CUDACHECK(cudaFree(ring->devUserRanks));

  // Free transport proxy resources
  if (ring->send.transportResources) NCCLCHECK(ring->send.transport->send.free(ring->send.transportResources));
  NCCLCHECK(transportDestroyProxy(&ring->send));
  if (ring->recv.transportResources) NCCLCHECK(ring->recv.transport->recv.free(ring->recv.transportResources));
  NCCLCHECK(transportDestroyProxy(&ring->recv));
  return ncclSuccess;
}
