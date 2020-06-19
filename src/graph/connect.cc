/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "trees.h"
#include "rings.h"

/******************************************************************/
/********************* Internode connection ***********************/
/******************************************************************/

ncclResult_t ncclTopoPreset(struct ncclComm* comm,
    struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph, struct ncclTopoGraph* collNetGraph,
    struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->ring.prev = channel->ring.next = -1;
    channel->tree.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->tree.down[i] = -1;
    channel->collTree.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->collTree.down[i] = -1;

    int* ringIntra = ringGraph->intra+c*localRanks;
    int* treeIntra = treeGraph->intra+c*localRanks;
    int* collNetIntra = collNetGraph->intra+c*localRanks;

    for (int i=0; i<localRanks; i++) {
      if (ringIntra[i] == rank) {
        topoRanks->ringRecv[c] = ringIntra[0];
        topoRanks->ringSend[c] = ringIntra[localRanks-1];
        channel->ring.prev = (i == 0) ? -1 : ringIntra[i-1];
        channel->ring.next = (i == localRanks-1) ? -1 : ringIntra[i+1];
      }
      if (treeIntra[i] == rank) {
        int parentIndex = 0;
        int child0Index = treeGraph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ? 1 : 0;
        int child1Index = treeGraph->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;

        topoRanks->treeParent[c] = treeIntra[parentIndex];
        topoRanks->treeChild0[c] = treeIntra[child0Index];
        topoRanks->treeChild1[c] = treeIntra[child1Index];
        channel->tree.up         = i == 0 ? -1 : treeIntra[i-1];
        channel->tree.down[0]    = i == localRanks-1 ? -1 : treeIntra[i+1];
      }
      if (collNetIntra[i] == rank) {
        int prev = (i-1+localRanks)%localRanks, next = (i+1)%localRanks;

        channel->collTree.up      = collNetIntra[prev];
        channel->collTree.down[0] = collNetIntra[next];
      }
    }
    topoRanks->ringPrev[c] = channel->ring.prev;
    topoRanks->ringNext[c] = channel->ring.next;
  }
  // Duplicate channels rings/trees
  struct ncclChannel* channel0 = comm->channels;
  struct ncclChannel* channel1 = channel0+nChannels;
  memcpy(channel1, channel0, nChannels*sizeof(struct ncclChannel));
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
      prev[recvRank] = prevSendRank;
      if (comm->rank == recvRank) {
        channel0->ring.prev = prevSendRank;
        channel1->ring.prev = prevSendRank;
      }
      int sendRank = send[firstRanks[n]];
      int nextRecvRank = recv[firstRanks[(n+1)%nNodes]];
      next[sendRank] = nextRecvRank;
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

static ncclResult_t setTreeUp(struct ncclTree* tree, int* indexes, int u) {
  if (u == -1) return ncclSuccess;
  tree->up = indexes[u];
  return ncclSuccess;
}

static ncclResult_t setTreeDown(struct ncclTree* tree, int* indexes, int d) {
  if (d == -1) return ncclSuccess;
  int x = 0;
  while (x < NCCL_MAX_TREE_ARITY && tree->down[x] >= 0) x++;
  if (x == NCCL_MAX_TREE_ARITY) {
    WARN("Internal error : tree already has %d children (%d %d %d)\n", x, tree->down[0], tree->down[1], tree->down[2]);
    return ncclInternalError;
  }
  tree->down[x] = indexes[d];
  return ncclSuccess;
}

static ncclResult_t connectTrees(struct ncclComm* comm, int* treeParent, int* treeChild0, int* treeChild1, int* firstRanks, int* treePatterns) {
  const int nChannels = comm->nChannels, nNodes = comm->nNodes, node = comm->node;
  int* indexesParent, *indexesChild0, *indexesChild1;
  NCCLCHECK(ncclCalloc(&indexesParent, nNodes));
  NCCLCHECK(ncclCalloc(&indexesChild0, nNodes));
  NCCLCHECK(ncclCalloc(&indexesChild1, nNodes));

  // Compute tree depth. Not an exact value but a good approximation in most
  // cases
  int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);

  int u0, d0_0, d0_1, childType0, u1, d1_0, d1_1, childType1;
  NCCLCHECK(ncclGetDtree(nNodes, node, &u0, &d0_0, &d0_1, &childType0, &u1, &d1_0, &d1_1, &childType1));
  for (int c=0; c<nChannels; c++) {
     struct ncclChannel* channel0 = comm->channels+c;
     struct ncclChannel* channel1 = channel0+nChannels;
     NCCLCHECK(getIndexes(treeParent+c*comm->nRanks, indexesParent, nNodes, firstRanks));
     NCCLCHECK(getIndexes(treeChild0+c*comm->nRanks, indexesChild0, nNodes, firstRanks));
     NCCLCHECK(getIndexes(treeChild1+c*comm->nRanks, indexesChild1, nNodes, firstRanks));
     if (comm->rank == indexesParent[node]) {
       NCCLCHECK(setTreeUp(&channel0->tree, childType0 == 0 ? indexesChild0 : indexesChild1, u0));
       NCCLCHECK(setTreeUp(&channel1->tree, childType1 == 0 ? indexesChild0 : indexesChild1, u1));
     }
     if (comm->rank == indexesChild0[node]) {
       NCCLCHECK(setTreeDown(&channel0->tree, indexesParent, d0_0));
       NCCLCHECK(setTreeDown(&channel1->tree, indexesParent, d1_0));
     }
     if (comm->rank == indexesChild1[node]) {
       NCCLCHECK(setTreeDown(&channel0->tree, indexesParent, d0_1));
       NCCLCHECK(setTreeDown(&channel1->tree, indexesParent, d1_1));
     }
     if (comm->rank == indexesParent[node] ||
         comm->rank == indexesChild0[node] ||
         comm->rank == indexesChild1[node]) {
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c,           channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c+nChannels, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
     }
     channel0->tree.depth = channel1->tree.depth = depth;
  }
  free(indexesParent);
  free(indexesChild0);
  free(indexesChild1);
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, int rank) {
  int nranks = comm->nRanks;
  int depth = nranks/comm->nNodes;
  int sendIndex = collNetGraph->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;  // send GPU index depends on topo pattern
  int sendEndIndex = (sendIndex+comm->localRanks-1)%comm->localRanks;
  for (int c=0; c<comm->nChannels/2; c++) {
    struct ncclChannel* channel = comm->channels+c;
    // Set root of collTree to id nranks
    if (rank == collNetGraph->intra[sendIndex+c*comm->localRanks]) { // is master
      channel->collTree.up = nranks;
    }
    if (rank == collNetGraph->intra[sendEndIndex+c*comm->localRanks]) { // is bottom of intra-node chain
      channel->collTree.down[0] = -1;
    }
    channel->collTree.depth = depth;
    INFO(NCCL_GRAPH, "CollNet Channel %d rank %d up %d down %d", c, rank, channel->collTree.up, channel->collTree.down[0]);
  }
  int recvIndex = 0;  // recv GPU index is always 0
  int recvEndIndex = (recvIndex+comm->localRanks-1)%comm->localRanks;
  for (int c=0; c<comm->nChannels/2; c++) {
    struct ncclChannel* channel = comm->channels+comm->nChannels/2+c;
    // Set root of collTree to id nranks
    if (rank == collNetGraph->intra[recvIndex+c*comm->localRanks]) { // is master
      channel->collTree.up = nranks;
    }
    if (rank == collNetGraph->intra[recvEndIndex+c*comm->localRanks]) { // is bottom of intra-node chain
      channel->collTree.down[0] = -1;
    }
    channel->collTree.depth = depth;
    INFO(NCCL_GRAPH, "CollNet Channel %d rank %d up %d down %d", comm->nChannels/2+c, rank, channel->collTree.up, channel->collTree.down[0]);
  }
  return ncclSuccess;
}

// Legacy naming
NCCL_PARAM(MinNrings, "MIN_NRINGS", -2);
NCCL_PARAM(MaxNrings, "MAX_NRINGS", -2);
// New naming
NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2);
NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2);

int ncclMinNchannels() {
  int minNchannels = 0;
  if (ncclParamMinNrings() != -2) minNchannels = ncclParamMinNrings();
  if (ncclParamMinNchannels() != -2) minNchannels = ncclParamMinNchannels();
  if (minNchannels > MAXCHANNELS) {
    WARN("User asked for a minimum of %d channels, limiting to %d\n", minNchannels, MAXCHANNELS);
    minNchannels = MAXCHANNELS;
  }
  if (minNchannels < 0) minNchannels = 0;
  return minNchannels;
}
int ncclMaxNchannels() {
  int maxNchannels = MAXCHANNELS;
  if (ncclParamMaxNrings() != -2) maxNchannels = ncclParamMaxNrings();
  if (ncclParamMaxNchannels() != -2) maxNchannels = ncclParamMaxNchannels();
  if (maxNchannels > MAXCHANNELS) maxNchannels = MAXCHANNELS;
  if (maxNchannels < 1) {
    WARN("User asked for a maximum of %d channels, setting it to 1\n", maxNchannels);
    maxNchannels = 1;
  }
  return maxNchannels;
}

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns, struct ncclTopoRanks** allTopoRanks, int* rings) {
  // Gather data from all ranks
  int *ringRecv, *ringSend, *ringPrev, *ringNext, *treeParent, *treeChild0, *treeChild1;
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;
  NCCLCHECK(ncclCalloc(&ringRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringSend, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringPrev, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringNext, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeParent, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeChild0, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeChild1, nranks*MAXCHANNELS));
  for (int i=0; i<nranks; i++) {
    for (int c=0; c<nChannels;c++) {
      ringRecv[c*nranks+i] = allTopoRanks[i]->ringRecv[c];
      ringSend[c*nranks+i] = allTopoRanks[i]->ringSend[c];
      ringPrev[c*nranks+i] = allTopoRanks[i]->ringPrev[c];
      ringNext[c*nranks+i] = allTopoRanks[i]->ringNext[c];
      treeParent[c*nranks+i] = allTopoRanks[i]->treeParent[c];
      treeChild0[c*nranks+i] = allTopoRanks[i]->treeChild0[c];
      treeChild1[c*nranks+i] = allTopoRanks[i]->treeChild1[c];
    }
  }

  // Connect rings and trees. This should also duplicate the channels.
  NCCLCHECK(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext, firstRanks));
  NCCLCHECK(connectTrees(comm, treeParent, treeChild0, treeChild1, firstRanks, treePatterns));

  // Duplicate ringPrev/ringNext for ncclBuildRing
  memcpy(ringPrev+nChannels*nranks, ringPrev, nChannels*nranks*sizeof(int));
  memcpy(ringNext+nChannels*nranks, ringNext, nChannels*nranks*sizeof(int));

  // Duplication should be complete now
  nChannels = comm->nChannels = std::min(MAXCHANNELS,nChannels*2);

  // Honor NCCL_MIN_NRINGS/NCCL_MAX_NRINGS.
  // We permit combining max, then min, to only use the first channels, then duplicate them.
  nChannels = comm->nChannels = std::min((int)ncclMaxNchannels(), nChannels);
  int c;
  for (c=nChannels; c<ncclMinNchannels(); c++) {
    memcpy(ringPrev+c*nranks, ringPrev+(c-nChannels)*nranks, nranks*sizeof(int));
    memcpy(ringNext+c*nranks, ringNext+(c-nChannels)*nranks, nranks*sizeof(int));
    memcpy(comm->channels+c, comm->channels+c-nChannels, sizeof(struct ncclChannel));
  }
  nChannels = comm->nChannels = c;

  // Create rings array and check all is fine
  NCCLCHECK(ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext));

  free(ringRecv);
  free(ringSend);
  free(ringPrev);
  free(ringNext);
  free(treeParent);
  free(treeChild0);
  free(treeChild1);

  return ncclSuccess;
}
