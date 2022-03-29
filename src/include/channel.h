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
static ncclResult_t getChannel(struct ncclComm* comm, int peer, int channel, int coll) {
  int p2pGroupSize = NCCL_MAX_WORK_ELEMENTS_P2P/2;
  int peerNode = comm->rankToNode[peer];
  int peerIndex = comm->rankToLocalRank[peer];
  int nsteps = comm->maxLocalRanks;
  int rankIndex = comm->rankToLocalRank[comm->rank];
  if (coll == ncclFuncSend) {
      int step = (nsteps + peerIndex - rankIndex)%nsteps;
      int delta = (comm->nNodes + peerNode - comm->node) % comm->nNodes;
      if (comm->nNodes == 1) delta = (comm->nRanks + peer - comm->rank) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int shuffle = comm->nNodes > 1 ? delta+(step/p2pGroupSize) : step;
        int channelId = (shuffle+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].send[1].connected == 0) {
          comm->connectSend[peer] |= (1<<channelId);
        }
      }
  } else if (coll == ncclFuncRecv) {
      int step = (nsteps + rankIndex - peerIndex)%nsteps;
      delta = (comm->nNodes + comm->node - peerNode) % comm->nNodes;
      if (comm->nNodes == 1) delta = (comm->nRanks - peer + comm->rank) % comm->nRanks;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        int shuffle = comm->nNodes > 1 ? delta+(step/p2pGroupSize) : step;
        int channelId = (shuffle+comm->p2pChannels[c]) % comm->p2pnChannels;
        if (comm->channels[channelId].peers[peer].recv[1].connected == 0) {
          comm->connectRecv[peer] |= (1<<channelId);
        }
      }

}

#endif
