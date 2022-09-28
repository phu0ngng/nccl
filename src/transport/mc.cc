/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"

/* Determine if two peers can communicate through mc */
ncclResult_t mcCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // This transport cannot be used for p2p
  *ret = 0;
  return ncclSuccess;
}

ncclResult_t mcSendFree(struct ncclConnector* send) {
  return ncclSuccess;
}

ncclResult_t mcRecvFree(struct ncclConnector* recv) {
  return ncclSuccess;
}

struct ncclTransport mcTransport = {
  "MC",
  mcCanConnect,
  { NULL, NULL, mcSendFree, NULL, NULL, NULL, NULL, NULL },
  { NULL, NULL, mcRecvFree, NULL, NULL, NULL, NULL, NULL }
};

#define MC_HANDLE_SIZE 64

typedef void* mcGroup_t; //TODO

struct mcResources {
  mcGroup_t* mcGroups;
  char** mcMems;
};

ncclResult_t mcGroupCreate(char* handle, mcGroup_t* group) {
  // TODO: Create an MC group
  return ncclSuccess;
}

ncclResult_t mcGroupDestroy(mcGroup_t group) {
  // TODO: Create an MC group
  return ncclSuccess;
}

ncclResult_t mcGroupConnect(char* handle, mcGroup_t* group) {
  // TODO: Connect to an MC group created by another rank
  return ncclSuccess;
}

ncclResult_t mcGroupDisconnect(mcGroup_t group) {
  // TODO: Connect to an MC group created by another rank
  return ncclSuccess;
}

ncclResult_t mcGroupBindMem(mcGroup_t group, char** mem, size_t size) {
  // TODO: Alloc `mem` of size `size` and bind it to the mem handle
  return ncclSuccess;
}

ncclResult_t mcGroupUnbindMem(mcGroup_t group, char* mem) {
  // TODO: Alloc `mem` of size `size` and bind it to the mem handle
  return ncclSuccess;
}

#include "bootstrap.h"

NCCL_PARAM(McBuffSize, "MC_BUFFSIZE", (1UL<<22));
#define MC_MEM_ALIGN_SIZE (1 << 21)

ncclResult_t ncclMcSetup(struct ncclComm* comm) {
  ncclResult_t res = ncclSuccess;
  struct mcResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  comm->mcResources = resources;

  int buffSize = ncclParamMcBuffSize();
  int memSize = 2*sizeof(uint64_t);
  ALIGN_SIZE(buffSize, MC_MEM_ALIGN_SIZE);
  ALIGN_SIZE(memSize, MC_MEM_ALIGN_SIZE);
  int mcTotalSize = comm->nChannels*(buffSize+memSize);

  char* mcHandles = NULL;
  NCCLCHECKGOTO(ncclCalloc(&mcHandles, comm->localRanks*MC_HANDLE_SIZE), res, cleanup);
  NCCLCHECKGOTO(ncclCalloc(&resources->mcGroups, comm->localRanks), res, cleanup);
  NCCLCHECKGOTO(ncclCalloc(&resources->mcMems, comm->localRanks), res, cleanup);
  NCCLCHECKGOTO(mcGroupCreate(mcHandles+comm->localRank*MC_HANDLE_SIZE, resources->mcGroups+comm->localRank), res, cleanup);
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, mcHandles, MC_HANDLE_SIZE), res, cleanup);
  for (int r=0; r<comm->localRanks; r++) {
    if (r != comm->localRank) NCCLCHECKGOTO(mcGroupConnect(mcHandles+r*MC_HANDLE_SIZE, resources->mcGroups+r), res, cleanup);
    NCCLCHECKGOTO(mcGroupBindMem(resources->mcGroups[r], resources->mcMems+r, mcTotalSize), res, cleanup);
    for (int c=0; c<comm->nChannels; c++) {
      char* mem = resources->mcMems[r] + c*(buffSize+memSize);
      struct ncclChannelPeer* peer = comm->channels[c].peers+comm->nRanks+1+r;

      peer->send->transportComm = &mcTransport.send;
      peer->send->conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->send->conn.head = (uint64_t*)(mem+buffSize);
      peer->send->conn.tail = (uint64_t*)(mem+buffSize+sizeof(uint64_t));

      peer->recv->transportComm = &mcTransport.recv;
      peer->recv->conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->recv->conn.head = (uint64_t*)(mem+buffSize);
      peer->recv->conn.tail = (uint64_t*)(mem+buffSize+sizeof(uint64_t));
    }
  }
cleanup:
  free(mcHandles);
  return res;
}

ncclResult_t ncclMcFree(struct ncclComm* comm) {
  struct mcResources* resources = (struct mcResources*)comm->mcResources;
  if (resources == NULL) return ncclSuccess;
  for (int r=0; r<comm->localRanks; r++) {
    if (resources->mcGroups[r] == NULL) continue;
    if (resources->mcMems[r]) NCCLCHECK(mcGroupUnbindMem(resources->mcGroups[r], resources->mcMems[r]));
    NCCLCHECK(mcGroupDisconnect(resources->mcGroups[r]));
  }
  NCCLCHECK(mcGroupDestroy(resources->mcGroups[comm->localRank]));
  free(resources->mcGroups);
  free(resources->mcMems);
  free(resources);
  comm->mcResources = NULL;
  return ncclSuccess;
}
