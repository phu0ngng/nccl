/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "utils.h"
#include "trees.h"
#include "rings.h"

/******************************************************************/
/********************* Internode connection ***********************/
/******************************************************************/

ncclResult_t ncclTopoPreset(struct ncclComm* comm, int* firstRanks,
    struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph,
    struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->ring.prev = channel->ring.next = -1;
    channel->treeUp.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->treeUp.down[i] = -1;
    channel->treeDn.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->treeDn.down[i] = -1;

    for (int i=0; i<comm->localRanks; i++) {
      if (ringGraph->intra[i] == rank) {
        topoRanks->ringRecv[c] = ringGraph->intra[0];
        topoRanks->ringSend[c] = ringGraph->intra[comm->localRanks-1];
        channel->ring.prev = (i == 0) ? -1 : ringGraph->intra[i-1];
        channel->ring.next = (i == comm->localRanks-1) ? -1 : ringGraph->intra[i+1];
      }
      if (treeGraph->intra[i] == rank) {
        int recvIndex = 0, sendIndex = treeGraph->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
        topoRanks->treeDnRecv[c] = treeGraph->intra[recvIndex];
        topoRanks->treeDnSend[c] = treeGraph->intra[sendIndex];
        channel->treeDn.up = (i == recvIndex) ? -1 : treeGraph->intra[i-1];
        channel->treeDn.down[0] = (i == comm->localRanks-1) ? -1 : treeGraph->intra[i+1];
        if (treeGraph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) {
          // Tree loop always flows in the same direction
          topoRanks->treeUpRecv[c] = treeGraph->intra[recvIndex];
          topoRanks->treeUpSend[c] = treeGraph->intra[sendIndex];
          channel->treeUp.up = (i == sendIndex) ? -1 : treeGraph->intra[(i+nranks+1)%nranks];
          channel->treeUp.down[0] = (i == sendIndex+1) ? -1 : treeGraph->intra[(i+nranks-1)%nranks];
        } else {
          // Other trees are symmetric and flow in opposite directions for up/down
          topoRanks->treeUpRecv[c] = topoRanks->treeDnSend[c]; 
          topoRanks->treeUpSend[c] = topoRanks->treeDnRecv[c];
          channel->treeUp.up = channel->treeDn.up;
          channel->treeUp.down[0] = channel->treeDn.down[0];
        }
      }
    }
    topoRanks->ringPrev[c] = channel->ring.prev;
    topoRanks->ringNext[c] = channel->ring.next;
  }
  // Duplicate channels rings/trees
  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel0 = comm->channels+c;
    struct ncclChannel* channel1 = channel0+nChannels;
    memcpy(channel1, channel0, sizeof(struct ncclChannel));
  }
  return ncclSuccess;
}

static ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks) {
  int nChannels = comm->nChannels;
  int nNodes = comm->nNodes;
  for (int c=0; c<nChannels; c++) {
    int* recv = ringRecv+c*comm->nRanks;
    int* send = ringSend+c*comm->nRanks;
    int* prev = ringPrev+c*comm->nRanks;
    int* next = ringNext+c*comm->nRanks;
    struct ncclChannel* channel0 = comm->channels+c;
    struct ncclChannel* channel1 = channel0+nChannels;
    for (int n=0; n<nNodes; n++) {
      int recvRank = recv[firstRanks[n]];
      int prevSendRank = send[firstRanks[(n-1+nNodes)%nNodes]];
      prev[c*comm->nRanks+recvRank] = prevSendRank;
      if (comm->rank == recvRank) {
        channel0->ring.prev = prevSendRank;
        channel1->ring.prev = prevSendRank;
      }
      int sendRank = send[firstRanks[n]];
      int nextRecvRank = recv[firstRanks[(n+1)%nNodes]];
      next[c*comm->nRanks+sendRank] = nextRecvRank;
      if (comm->rank == sendRank) {
        channel0->ring.next = nextRecvRank;
        channel1->ring.next = nextRecvRank;
      }
    }
    TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c, channel0->ring.prev, comm->rank, channel0->ring.next);
    TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c+nChannels, channel1->ring.prev, comm->rank, channel1->ring.next);
  }
  return ncclSuccess;
}

static ncclResult_t getIndexes(int* ranks, int* indexes, int nNodes, int* firstRanks) {
 for (int n=0; n<nNodes; n++) indexes[n] = ranks[firstRanks[n]];
 return ncclSuccess;
}

static ncclResult_t setTreeUp(struct ncclTree* tree0, struct ncclTree* tree1, int* indexes, int u0, int u1) {
  if (u0 != -1) tree0->up = indexes[u0];
  if (u1 != -1) tree1->up = indexes[u1];
  return ncclSuccess;
}

static ncclResult_t addRanksDown(int* down, int* indexes, int r0, int r1) {
  int x = 0;
  if (down[x] >= 0) x++;
  if (down[x] >= 0) {
    WARN("Internal error : tree already has more than one child (%d %d %d)\n", down[0], down[1], down[2]);
    return ncclInternalError;
  }
  if (r0 != -1) down[x++] = indexes[r0];
  if (r1 != -1) down[x++] = indexes[r1];
  return ncclSuccess;
}

static ncclResult_t setTreeDown(struct ncclTree* tree0, struct ncclTree* tree1, int* indexes, int d0_0, int d0_1, int d1_0, int d1_1) {
  NCCLCHECK(addRanksDown(tree0->down, indexes, d0_0, d0_1));
  NCCLCHECK(addRanksDown(tree1->down, indexes, d1_0, d1_1));
  return ncclSuccess;
}

static ncclResult_t connectTrees(struct ncclComm* comm, int* treeUpRecv, int* treeUpSend, int* treeDnRecv, int* treeDnSend, int* firstRanks) {
  const int nChannels = comm->nChannels, nNodes = comm->nNodes, node = comm->node;
  int* indexes;
  NCCLCHECK(ncclCalloc(&indexes, nNodes));

  // Compute tree depth. Not an exact value but a good approximation in most
  int depth = comm->nRanks/nNodes + log2(nNodes);

  int u0, d0_0, d0_1, u1, d1_0, d1_1;
  NCCLCHECK(ncclGetDtree(nNodes, node, &u0, &d0_0, &d0_1, &u1, &d1_0, &d1_1));
  for (int c=0; c<nChannels; c++) {
     struct ncclChannel* channel0 = comm->channels+c;
     struct ncclChannel* channel1 = channel0+nChannels;
     NCCLCHECK(getIndexes(treeUpSend+c*comm->nRanks, indexes, nNodes, firstRanks));
     if (indexes[node] == comm->rank) NCCLCHECK(setTreeUp(&channel0->treeUp, &channel1->treeUp, indexes, u0, u1));
     NCCLCHECK(getIndexes(treeUpRecv+c*comm->nRanks, indexes, nNodes, firstRanks));
     if (indexes[node] == comm->rank) NCCLCHECK(setTreeDown(&channel0->treeUp, &channel1->treeUp, indexes, d0_0, d0_1, d1_0, d1_1));
     NCCLCHECK(getIndexes(treeDnRecv+c*comm->nRanks, indexes, nNodes, firstRanks));
     if (indexes[node] == comm->rank) NCCLCHECK(setTreeUp(&channel0->treeDn, &channel1->treeDn, indexes, u0, u1));
     NCCLCHECK(getIndexes(treeDnSend+c*comm->nRanks, indexes, nNodes, firstRanks));
     if (indexes[node] == comm->rank) NCCLCHECK(setTreeDown(&channel0->treeDn, &channel1->treeDn, indexes, d0_0, d0_1, d1_0, d1_1));
     TRACE(NCCL_GRAPH, "TreeUp %d : %d -> %d/%d/%d", c,           channel0->treeUp.up, channel0->treeUp.down[0], channel0->treeUp.down[1], channel0->treeUp.down[2]);
     TRACE(NCCL_GRAPH, "TreeUp %d : %d -> %d/%d/%d", c+nChannels, channel1->treeUp.up, channel1->treeUp.down[0], channel1->treeUp.down[1], channel1->treeUp.down[2]);
     TRACE(NCCL_GRAPH, "TreeDn %d : %d -> %d/%d/%d", c,           channel0->treeDn.up, channel0->treeDn.down[0], channel0->treeDn.down[1], channel0->treeDn.down[2]);
     TRACE(NCCL_GRAPH, "TreeDn %d : %d -> %d/%d/%d", c+nChannels, channel1->treeDn.up, channel1->treeDn.down[0], channel1->treeDn.down[1], channel1->treeDn.down[2]);
     channel0->treeUp.depth = channel1->treeUp.depth = depth;
  }
  free(indexes);
  return ncclSuccess;
}

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, struct ncclTopoRanks** allTopoRanks, int* rings) {
  // Gather data from all ranks
  int *ringRecv, *ringSend, *ringPrev, *ringNext, *treeUpRecv, *treeUpSend, *treeDnRecv,*treeDnSend;
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;
  NCCLCHECK(ncclCalloc(&ringRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringSend, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringPrev, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringNext, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeUpRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeUpSend, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeDnRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeDnSend, nranks*MAXCHANNELS));
  for (int i=0; i<nranks; i++) {
    for (int c=0; c<nChannels;c++) {
      ringRecv[c*nranks+i] = allTopoRanks[i]->ringRecv[c];
      ringSend[c*nranks+i] = allTopoRanks[i]->ringSend[c];
      ringPrev[c*nranks+i] = allTopoRanks[i]->ringPrev[c];
      ringNext[c*nranks+i] = allTopoRanks[i]->ringNext[c];
      treeUpRecv[c*nranks+i] = allTopoRanks[i]->treeUpRecv[c];
      treeUpSend[c*nranks+i] = allTopoRanks[i]->treeUpSend[c];
      treeDnRecv[c*nranks+i] = allTopoRanks[i]->treeDnRecv[c];
      treeDnSend[c*nranks+i] = allTopoRanks[i]->treeDnSend[c];
    }
  }

  NCCLCHECK(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext, firstRanks));
  NCCLCHECK(connectTrees(comm, treeUpRecv, treeUpSend, treeDnRecv, treeDnSend, firstRanks));

  NCCLCHECK(ncclBuildRings(comm->nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext));

  free(ringRecv);
  free(ringSend);
  free(ringPrev);
  free(ringNext);
  free(treeUpRecv);
  free(treeUpSend);
  free(treeDnRecv);
  free(treeDnSend);

  return ncclSuccess;
}
