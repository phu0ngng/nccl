/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "coll_net.h"
#include "graph.h"
#include "proxy.h"

struct collNetRecvConnectInfo {
  int rank;
  int nranks;
  collNetHandle_t collNetHandle;
};

struct collNetSendConnectInfo {
  void* mhandles[NCCL_NUM_PROTOCOLS];
  void* reqFifo;
};

#define COLLNET_GROUP_NSUBS 8
#define COLLNET_MAX_GROUPS (NCCL_PROXY_MAX_SUBS/COLLNET_GROUP_NSUBS)

struct connectMap {
  int sameProcess;
  int shared;
  int useGdc;
  struct {
    char* gpuPtr;
    char* cpuPtr;
    char shmPath[PATH_MAX];
    int shmSize;
  } hostMem;
  struct {
    char* gpuPtr;
    char* cpuPtr;
    cudaIpcMemHandle_t ipc;
  } devMem;
  // Offsets. Positive values are for hostMem, negative values are for devMem, 0 means NULL;
  struct {
    int sendMem;
    int recvMem;
    int buffs[NCCL_NUM_PROTOCOLS];
  } offsets;
};

#define NCCL_NET_MAP_GET_POINTER(mapStruct, cpuOrGpu, offsetName) \
  ((mapStruct)->offsets.offsetName  == 0 ? NULL : \
   (mapStruct)->offsets.offsetName > 0 ? \
   (mapStruct)->hostMem.cpuOrGpu##Ptr + (mapStruct)->offsets.offsetName-1 : \
   (mapStruct)->devMem.cpuOrGpu##Ptr - (mapStruct)->offsets.offsetName+1)

#define NCCL_NET_MAP_DEV_MEM(mapStruct, offsetName) \
  ((mapStruct)->offsets.offsetName<0 ? 1 : 0)

#define NCCL_NET_MAP_ADD_POINTER(mapStruct, dev, size, offsetName) do { \
    if (dev) { \
      (mapStruct)->offsets.offsetName = - devMemSize - 1; \
      devMemSize += size; \
    } else { \
      (mapStruct)->offsets.offsetName = hostMemSize + 1; \
      hostMemSize += size; \
    } \
} while (0);

struct reqSlot {
  volatile void* recvBuff;
  volatile int size;
};

struct sendResources {
  struct connectMap map;
  void* collNetComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int rank;
  int nranks;
  int netDev;
  int useGdr;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  void* sendMhandles[NCCL_NUM_PROTOCOLS];
  void* recvMhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
  struct reqSlot (*reqFifo)[NCCL_STEPS];
};

struct recvResources {
  struct connectMap map;
  void* collNetComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int rank;
  int nranks;
  int netDev;
  int useGdr;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
  struct reqSlot reqFifo[COLLNET_MAX_GROUPS][NCCL_STEPS];
};

struct collNetSharedResources {
  void* collNetListenComms[MAXCHANNELS];
  void* collNetComms[MAXCHANNELS];
  int collNetCommRefCount[MAXCHANNELS];
};

/* Determine if we can communicate with the peer */
static ncclResult_t canConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

struct setupReq {
  int netDev;
  int useGdr;
};

/* Setup send connector, and return connect information for others in the coll
 * communicator to connect to me */
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req;

  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, -1, &req.netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 1, &req.useGdr));

  NCCLCHECK(ncclTopoGetLocalRank(comm->topo, myInfo->rank, &send->proxyConn.localRank));
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_COLLNET, 1, myInfo->rank, &send->proxyConn));
  NCCLCHECK(ncclProxyCall(&send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  INFO(NCCL_INIT|NCCL_NET,"CollNet %02d : %d [send] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), req.netDev,
      req.useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

static ncclResult_t recvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct setupReq req;

  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, -1, &req.netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 0, &req.useGdr));

  NCCLCHECK(ncclTopoGetLocalRank(comm->topo, myInfo->rank, &recv->proxyConn.localRank));
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_COLLNET, 0, myInfo->rank, &recv->proxyConn));
  struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*) connectInfo;
  NCCLCHECK(ncclProxyCall(&recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), &info->collNetHandle, sizeof(collNetHandle_t)));

  INFO(NCCL_INIT|NCCL_NET,"CollNet %02d : %d [receive] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), req.netDev,
      req.useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

static ncclResult_t sendConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct connectMap map;
  NCCLCHECK(ncclProxyCall(&send->proxyConn, ncclProxyMsgConnect, connectInfos, sizeof(collNetHandle_t), &map, sizeof(struct connectMap)));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(&map, gpu, sendMem);
  send->conn.head = &sendMem->head;

  // Don't give credits yet in shared mode.
  sendMem->head = -NCCL_STEPS;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(&map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;
  send->conn.sizesFifo = recvMem->sizesFifo;
  for (int i=0; i<NCCL_STEPS; i++) send->conn.sizesFifo[i] = -1;
  send->conn.offsFifo = recvMem->offsFifo;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(&map, gpu, buffs[p]);
  return ncclSuccess;
}

static ncclResult_t recvConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* recv) {
  struct connectMap map;
  NCCLCHECK(ncclProxyCall(&recv->proxyConn, ncclProxyMsgConnect, NULL, 0, &map, sizeof(struct connectMap)));

  return ncclSuccess;
}

static ncclResult_t sendFree(struct ncclConnector* send) {
  return ncclSuccess;
}

static ncclResult_t recvFree(struct ncclConnector* recv) {
  return ncclSuccess;
}

static ncclResult_t sendProxySetup(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct setupReq req;
  NCCLCHECK(ncclSocketRecv(connection->sock, &req, sizeof(req)));

  struct sendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->netDev = req.netDev;
  resources->useGdr = req.useGdr;
  return ncclSuccess;
}

static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct setupReq req;
  NCCLCHECK(ncclSocketRecv(connection->sock, &req, sizeof(req)));

  struct recvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->netDev = req.netDev;
  connection->shared = 1;

  collNetHandle_t netHandle;
#if 0 // TODO shared listen
  struct collNetSharedResources* resources = (struct collNetSharedResources*)comm->proxyState.progressState.collNet.resources;
  if (resources == NULL) {
    NCCLCHECK(ncclCalloc(&resources, 1));
    comm->proxyState.progressState.collNet.resources = resources;
  }
  if (resources->collNetComms[req.netDev] == NULL)
    NCCLCHECK(collNetListen(req.netDev, &netHandle, resources->collNetListenComms+req.netDev));
#endif
  NCCLCHECK(ncclSocketSend(connection->sock, &netHandle, sizeof(collNetHandle_t)));
  return ncclSuccess;
}

static ncclResult_t sendProxyConnect(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);
  collNetHandle_t netHandle;
  NCCLCHECK(ncclSocketRecv(connection->sock, &netHandle, sizeof(collNetHandle_t)));
  return ncclSuccess;
}

static ncclResult_t recvProxyConnect(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  return ncclSuccess;
}

static ncclResult_t sendProxyFree(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  return ncclSuccess;
}

static ncclResult_t recvProxyFree(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  return ncclSuccess;
}


static ncclResult_t collNetSharedConnect(struct ncclComm* comm, int netDev, struct ncclConnect* connectInfos, int nranks, int rank, void** collNetComm) {
  struct collNetSharedResources* resources = (struct collNetSharedResources*)comm->proxyState.progressState.collNet.resources;
  if (resources->collNetComms[netDev] == NULL) {
    // Connect to coll comm
    collNetHandle_t** handlePtrs = NULL;
    NCCLCHECK(ncclCalloc(&handlePtrs, nranks));
    for (int i = 0; i < nranks; i++) {
      struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*)(connectInfos+i);
      handlePtrs[i] = &(info->collNetHandle);
    }
    ncclResult_t ret = collNetConnect((void**)handlePtrs, nranks, rank,
          resources->collNetListenComms[netDev],
          resources->collNetComms+netDev);
    free(handlePtrs);
    NCCLCHECK(ret);
    // Close listen comm
    NCCLCHECK(collNetCloseListen(resources->collNetListenComms[netDev]));
  }
  *collNetComm = resources->collNetComms[netDev];
  resources->collNetCommRefCount[netDev]++;
  return ncclSuccess;
}

static ncclResult_t collNetSharedFree(struct ncclComm* comm, int netDev) {
  struct collNetSharedResources* resources = (struct collNetSharedResources*)comm->proxyState.progressState.collNet.resources;
  resources->collNetCommRefCount[netDev]--;
  if (resources->collNetCommRefCount[netDev] == 0) {
    NCCLCHECK(collNetCloseColl(resources->collNetComms[netDev]));
  }
  for (int c=0; c<MAXCHANNELS; c++) if (resources->collNetCommRefCount[c]) return ncclSuccess;
  comm->proxyState.progressState.collNet.resources = NULL;
  free(resources);
  return ncclSuccess;
}

#define LAST_OF_GROUP(s) \
  (s % COLLNET_GROUP_NSUBS == COLLNET_GROUP_NSUBS-1 || s == args->nsubs-1)

static ncclResult_t sendProxyProgress(struct ncclComm* comm, struct ncclProxyArgs* args) {
  if (args->protocol == NCCL_PROTO_LL128) {
    WARN("CollNet does not support LL128");
    return ncclInternalError;
  }
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendResources* resources = (struct sendResources*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->received = sub->transmitted = sub->done = 0;
      resources->step = sub->base + sub->nsteps;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int nGroups = DIVUP(args->nsubs, COLLNET_GROUP_NSUBS);
    int perGroupSteps = NCCL_STEPS / nGroups;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendResources* resources = (struct sendResources*) (sub->connection->transportResources);
      void* sendMhandle = resources->sendMhandles[p];
      void* recvMhandle = resources->recvMhandles[p];
      int stepSize = comm->buffSizes[p] / NCCL_STEPS;
      auto reqFifo = resources->reqFifo;
      if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        if (p == NCCL_PROTO_SIMPLE) {
          int sharedBuffSlot = sub->posted%NCCL_STEPS;
          int offset;
          NCCLCHECK(ncclProxySharedBuffersGetCollNet(comm, 0, sharedBuffSlot, 0, &offset));
          resources->recvMem->offsFifo[buffSlot] = offset + s*args->chunkSize;
          __sync_synchronize();
        }
        volatile uint64_t* sendHead = &resources->sendMem->head;
        sub->posted += args->sliceSteps;
        *sendHead = sub->base + sub->posted - NCCL_STEPS;
      }
      // Enforce sync between operations of the same group.
      bool groupSync = (((s == 0) && ((sub+args->nsubs-1)->received == sub->received)) || (s && (sub-1)->received > sub->received));
      if (groupSync && sub->received < sub->posted && sub->received < sub->done + perGroupSteps) {
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;
        int sharedBuffSlot = sub->received%NCCL_STEPS;
        volatile int* sizesFifo = resources->recvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, gpu, buffs[p]);
        if (sizesFifo[buffSlot] != -1 && ((*recvTail > (sub->base+sub->received)) || p == NCCL_PROTO_LL)) {
          // We have something to receive, let's check whether data is ready.
          int size = sizesFifo[buffSlot];
          int ready = 1;
          if (s == 0) {
            int offset;
            NCCLCHECK(ncclProxySharedBuffersGetCollNet(comm, 0, sharedBuffSlot, 0, &offset));
            args->sharedBuff[sharedBuffSlot] = localBuff + offset;
            args->sharedSize[sharedBuffSlot] = p == NCCL_PROTO_SIMPLE ? args->chunkSize : size/2;
          }
          if (p == NCCL_PROTO_LL) {
            uint32_t flag = NCCL_LL_FLAG(sub->base + sub->received + 1);
            int nFifoLines = size / sizeof(union ncclLLFifoLine);
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+buffSlot*stepSize);
            // Pack data into the shared buffer
            uint32_t* sendBuff = (uint32_t*)(args->sharedBuff[sharedBuffSlot]+args->sharedSize[sharedBuffSlot]*s);
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *d1 = &lines[i].data1;
              volatile uint32_t *f2 = &lines[i].flag2;
              volatile uint32_t *d2 = &lines[i].data2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
              sendBuff[2*i] = d1[0];
              sendBuff[2*i+1] = d2[0];
            }
          }
          if (ready) {
            sizesFifo[buffSlot] = -1;
            sub->received += args->sliceSteps;
            args->idle = 0;
            //continue;
          }
        }
      }
      if (LAST_OF_GROUP(s) && (sub->transmitted < sub->received)) {
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        int sharedBuffSlot = sub->transmitted%NCCL_STEPS;
        if (reqFifo[group][buffSlot].recvBuff != NULL) {
          int totalSize = (s-group*COLLNET_GROUP_NSUBS+1) * args->sharedSize[sharedBuffSlot];
          int count = totalSize / ncclTypeSize(args->dtype);
          reqFifo[group][buffSlot].size = args->sharedSize[sharedBuffSlot];
          char* sendAddress = (char*)args->sharedBuff[sharedBuffSlot] + group*COLLNET_GROUP_NSUBS*args->sharedSize[sharedBuffSlot];
          NCCLCHECK(collNetIallreduce(resources->collNetComm, sendAddress, (void*)(reqFifo[group][buffSlot].recvBuff), count, args->dtype, args->redOp, sendMhandle, recvMhandle, sub->requests+buffSlot));
          if (sub->requests[buffSlot] == NULL) continue;

          TRACE(NCCL_NET, "sendProxy [%d/%d/%d] Iallreduce posted, size %d req %p", sub->transmitted, group, buffSlot, totalSize, sub->requests[buffSlot]);
          // Make sure size is reset to zero before we update the head.
          __sync_synchronize();
          sub->transmitted += args->sliceSteps;
          args->idle = 0;
          continue;
        }
      }
      // Check whether the network has completed some send operations.
      if (LAST_OF_GROUP(s) && sub->done < sub->transmitted) {
        int done, size;
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        NCCLCHECK(collNetTest((void*)(sub->requests[buffSlot]), &done, &size));
        if (done) {
          TRACE(NCCL_NET, "sendProxy [%d/%d/%d] request %p done, size %d", sub->done, group, buffSlot, sub->requests[buffSlot], size);
          // Make sure size is updated before we set recvBuff to NULL (from the view of recv proxy, concerning the flush)
          // (reordered store after store is possible on POWER, though not on x86)
          __sync_synchronize();
          reqFifo[group][buffSlot].recvBuff = NULL; // Notify recvProxy
          for (int i=group*COLLNET_GROUP_NSUBS; i<=s; i++) args->subs[i].done += args->sliceSteps;
          args->idle = 0;
          int allDone = 1;
          for (int i=0; i<args->nsubs; i++) {
            if (args->subs[i].done < args->subs[i].nsteps) { allDone = 0; break; }
          }
          if (allDone) {
            args->state = ncclProxyOpNone;
            TRACE(NCCL_NET, "sendProxy [%d/%d] stopped", sub->done, s);
          }
        }
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t recvProxyProgress(struct ncclComm* comm, struct ncclProxyArgs* args) {
  if (args->protocol == NCCL_PROTO_LL128) {
    WARN("CollNet does not support LL128");
    return ncclInternalError;
  }
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->received = sub->flushed = sub->transmitted = sub->done = 0;
      resources->step = sub->base + sub->nsteps;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int nGroups = DIVUP(args->nsubs, COLLNET_GROUP_NSUBS);
    int perGroupSteps = NCCL_STEPS / nGroups;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
      void* mhandle = resources->mhandles[p];
      int stepSize = comm->buffSizes[p] / NCCL_STEPS;
      auto reqFifo = resources->reqFifo;
      char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
      
      // Enforce sync between operations of the same group.
      if (LAST_OF_GROUP(s) && (sub->posted < sub->done + perGroupSteps) && (sub->posted < sub->nsteps)) {
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        int sharedBuffSlot = sub->posted%NCCL_STEPS;
        int startChannel = group*COLLNET_GROUP_NSUBS;
        int offset;
        NCCLCHECK(ncclProxySharedBuffersGetCollNet(comm, 1, sharedBuffSlot, startChannel, &offset));
        reqFifo[group][buffSlot].recvBuff = localBuff + offset;
        TRACE(NCCL_NET, "recvProxy [%d/%d/%d] posted buffer %p", sub->posted, group, buffSlot, reqFifo[group][buffSlot].recvBuff);
        sub->posted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      if (LAST_OF_GROUP(s) && (sub->posted > sub->received)) {
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;
        int sharedBuffSlot = sub->received%NCCL_STEPS;
        if (reqFifo[group][buffSlot].recvBuff == NULL) { // Buffer is cleared : coll is complete
          args->sharedSize[sharedBuffSlot] = reqFifo[group][buffSlot].size;
          int totalSize = args->sharedSize[sharedBuffSlot]*(s-group*COLLNET_GROUP_NSUBS+1);
          TRACE(NCCL_NET, "recvProxy [%d/%d/%d] received, size %d", sub->received, group, buffSlot, totalSize);
          sub->received += args->sliceSteps;
          if (reqFifo[group][buffSlot].size > 0 && p == NCCL_PROTO_SIMPLE && resources->useGdr) {
            int startChannel = group*COLLNET_GROUP_NSUBS;
            int offset;
            NCCLCHECK(ncclProxySharedBuffersGetCollNet(comm, 1, sharedBuffSlot, startChannel, &offset));
            NCCLCHECK(collNetIflush(resources->collNetComm, localBuff + offset, totalSize, mhandle, sub->requests+buffSlot));
          } else {
            for (int i=group*COLLNET_GROUP_NSUBS; i<=s; i++) args->subs[i].flushed += args->sliceSteps;
          }
          args->idle = 0;
          continue;
        }
      }
      if (LAST_OF_GROUP(s) && (sub->received > sub->flushed)) {
        // Progress flush operations
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base + sub->flushed)%NCCL_STEPS;
        int done = 1;
        if (sub->requests[buffSlot]) NCCLCHECK(collNetTest(sub->requests[buffSlot], &done, NULL));
        if (done) {
          TRACE(NCCL_NET, "recvProxy [%d/%d/%d] flushed", sub->flushed, group, buffSlot);
          for (int i=group*COLLNET_GROUP_NSUBS; i<=s; i++) args->subs[i].flushed += args->sliceSteps;
          args->idle = 0;
          //continue;
        }
      }
      if (sub->flushed > sub->transmitted) {
        int group = s / COLLNET_GROUP_NSUBS;
        int buffSlot = (sub->base + sub->transmitted)%NCCL_STEPS;
        int sharedBuffSlot = sub->transmitted%NCCL_STEPS;
        int startChannel = group*COLLNET_GROUP_NSUBS;
        int offset;
        NCCLCHECK(ncclProxySharedBuffersGetCollNet(comm, 1, sharedBuffSlot, startChannel, &offset));
        char* ptr = localBuff + offset + (s%COLLNET_GROUP_NSUBS)*args->sharedSize[sharedBuffSlot];
        if (p == NCCL_PROTO_SIMPLE) {
          volatile int* offsFifo = (volatile int*)resources->recvMem->offsFifo;
          offsFifo[buffSlot] = offset;
          __sync_synchronize();
          resources->recvMem->tail = sub->base + sub->flushed;
        }
        if (p == NCCL_PROTO_LL) { // ll
          // re-attach flag
          uint32_t flag = NCCL_LL_FLAG(sub->base + sub->transmitted + 1);
          union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+buffSlot*stepSize);
          uint32_t* recvData = (uint32_t*)ptr;
          int nFifoLines = DIVUP(args->sharedSize[sharedBuffSlot], 2*sizeof(uint32_t));
          for (int i=0; i<nFifoLines; i++) {
            lines[i].v[0] = ((uint64_t)flag << 32) + recvData[2*i];
            lines[i].v[1] = ((uint64_t)flag << 32) + recvData[2*i+1];
          }
        }
        sub->transmitted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      // Enforce sync here to make sure the last sub doesn't increase "done" before all others in the group have
      // reached the same point, otherwise we would start posting buffers to the send proxy before we're done
      // processing all the shared buffer.
      bool groupSync = (((s == 0) && ((sub+args->nsubs-1)->done == sub->done)) || (s && (sub-1)->done > sub->done));
      volatile uint64_t* sendHead = &resources->sendMem->head;
      if (groupSync && sub->done < sub->transmitted && (sub->base+sub->done) < *sendHead) {
        sub->done += args->sliceSteps;
        args->idle = 0;
        if (sub->done == sub->nsteps && s == args->nsubs-1) {
          args->state = ncclProxyOpNone;
          TRACE(NCCL_NET, "recvProxy [%d/%d] stopped", sub->done, s);
        }
      }
    }
  }
  return ncclSuccess;
}

struct ncclTransport collNetTransport = {
  "COL",
  canConnect,
  { sendSetup, sendConnect, sendFree, sendProxySetup, sendProxyConnect, sendProxyFree, sendProxyProgress },
  { recvSetup, recvConnect, recvFree, recvProxySetup, recvProxyConnect, recvProxyFree, recvProxyProgress }
};
