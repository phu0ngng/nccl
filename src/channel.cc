/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel.h"
#include "param.h"
#include "gdrwrap.h"

ncclResult_t initChannel(struct ncclComm* comm, int channelId) {
  struct ncclChannel* channel = &comm->channels[channelId];
  if (channel->id != -1) return ncclSuccess;

  int nRanks = comm->nRanks;
  int nPeers = nRanks + 1 /* Collnet */ + comm->localRanks /* NVLS */;
  channel->id = channelId;
  channel->workFifoSent = 0;

  struct ncclSharedResources* sharedRes = comm->sharedRes;

  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&sharedRes->deviceStream));

  if (channel->peers == NULL) {
    // The extra on nRanks+1 is for collnet root (i.e. network)
    // Allocate everything related to sharedRes with ncclCalloc as this can be
    // shared between communicators hence should not be tied to comm.
    if (sharedRes->peers[channelId] == NULL) {
      NCCLCHECK(ncclCalloc(sharedRes->peers + channelId, sharedRes->tpNRanks));
    }
    channel->peers = ncclMemoryStackAlloc<struct ncclChannelPeer*>(&comm->memPermanent, nPeers);
    channel->collNetPeers = ncclMemoryStackAlloc<struct ncclChannelPeer>(&comm->memPermanent, 1);
    channel->nvlsPeers = ncclMemoryStackAlloc<struct ncclChannelPeer>(&comm->memPermanent, comm->localRanks);
    for (int r = 0; r < nPeers; r++) {
      if (r < nRanks) {
        channel->peers[r] = comm->sharedRes->peers[channelId] + comm->topParentRanks[r];
      } else if (r == nRanks) {
        channel->peers[r] = channel->collNetPeers;
      } else {
        channel->peers[r] = &channel->nvlsPeers[r - nRanks - 1];
      }
      __atomic_add_fetch(&channel->peers[r]->refCount, 1, __ATOMIC_RELAXED);
    }
  }

  if (channel->devPeers == NULL) {
    if (sharedRes->devPeers[channelId] == NULL) {
      NCCLCHECK(ncclCudaCallocAsync(sharedRes->devPeers + channelId, sharedRes->tpNRanks, sharedRes->deviceStream.cudaStream));
    }
    /* channel->devPeers is not shared, so just free it when calling commFree() */
    NCCLCHECK(ncclCudaCallocAsync(&channel->devPeers, nPeers, sharedRes->deviceStream.cudaStream));
    NCCLCHECK(ncclCudaCallocAsync(&channel->collNetDevPeers, 1, sharedRes->deviceStream.cudaStream));
    NCCLCHECK(ncclCudaCallocAsync(&channel->nvlsDevPeers, comm->localRanks, sharedRes->deviceStream.cudaStream));
    ncclCommPushCudaFree(comm, channel->devPeers);
    ncclCommPushCudaFree(comm, channel->collNetDevPeers);
    ncclCommPushCudaFree(comm, channel->nvlsDevPeers);
    for (int r = 0; r < nPeers; r++) {
      if (r < nRanks) {
        uintptr_t addr = (uintptr_t)(comm->sharedRes->devPeers[channelId] + comm->topParentRanks[r]);
        NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
      } else if (r == nRanks) {
        NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&channel->collNetDevPeers, 1, sharedRes->deviceStream.cudaStream));
      } else {
        uintptr_t addr = (uintptr_t)(channel->nvlsDevPeers + (r - nRanks - 1));
        NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
      }
    }
  }

  channel->ring.userRanks = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
  NCCLCHECK(ncclCudaCallocAsync(&channel->devRingUserRanks, nRanks, sharedRes->deviceStream.cudaStream));
  ncclCommPushCudaFree(comm, channel->devRingUserRanks);

  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream));

  return ncclSuccess;
}

ncclResult_t freeChannel(struct ncclChannel* channel, int nPeers) {
  /* channel peers are only valid when async init thread completes commAlloc() and
   * the channel is intialized with initChannel(); if either is not done, this channel
   * should never be free. */
  if (channel->id == -1 || channel->peers == NULL) return ncclSuccess;

  // Free transport proxy resources
  // Note: free all send resources first due to CollNet arrangement
  for (int r = 0; r < nPeers; r++) {
    struct ncclChannelPeer* peer = channel->peers[r];
    __atomic_sub_fetch(&peer->refCount, 1, __ATOMIC_RELAXED);
    for (int b=0; b<NCCL_MAX_CONNS; b++) {
      if ((peer->refCount == 0) && peer->send[b].transportComm) NCCLCHECK(peer->send[b].transportComm->free(peer->send+b));
    }
  }
  for (int r = 0; r < nPeers; r++) {
    struct ncclChannelPeer* peer = channel->peers[r];
    for (int b=0; b<NCCL_MAX_CONNS; b++) {
      if ((peer->refCount == 0) && peer->recv[b].transportComm) NCCLCHECK(peer->recv[b].transportComm->free(peer->recv+b));
    }
  }
  return ncclSuccess;
}
