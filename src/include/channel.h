/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CHANNEL_H_
#define NCCL_CHANNEL_H_
#include "comm.h"

ncclResult_t initChannel(struct ncclComm* comm, int channelid);
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks);
static ncclResult_t ncclChannelCompute(struct ncclComm* comm, int peer, int channelInc, int coll, int*channelId) {
  int p2pGroupSize = NCCL_MAX_WORK_ELEMENTS_P2P/2;
  int peerNode = comm->rankToNode[peer];
  int peerIndex = comm->rankToLocalRank[peer];
  int nsteps = comm->maxLocalRanks;
  int rankIndex = comm->rankToLocalRank[comm->rank];
  int step, delta;
  if (coll == ncclFuncSend) {
    step = (nsteps + peerIndex - rankIndex)%nsteps;
    delta = (comm->nNodes + peerNode - comm->node) % comm->nNodes;
    if (comm->nNodes == 1) delta = (comm->nRanks + peer - comm->rank) % comm->nRanks;
  } else if (coll == ncclFuncRecv) {
    if (comm->nNodes == 1) delta = (comm->nRanks - peer + comm->rank) % comm->nRanks;
    step = (nsteps + rankIndex - peerIndex)%nsteps;
    delta = (comm->nNodes + comm->node - peerNode) % comm->nNodes;
  } else {
    return ncclInternalError;
  }
  int shuffle = comm->nNodes > 1 ? delta+(step/p2pGroupSize) : step;
  *channelId = (shuffle+comm->p2pChannels[channelInc % comm->p2pnChannelsPerPeer]) % comm->p2pnChannels;
  return ncclSuccess;
}

#endif
