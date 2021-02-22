/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "coll_net.h"
#include "graph.h"
#include <assert.h>

struct collNetRecvConnectInfo {
  collNetHandle_t collNetHandle;
};

struct collNetSendConnectInfo {
  void* mhandles[NCCL_NUM_PROTOCOLS];
  struct reqSlot* reqFifo;
};

struct reqSlot {
  volatile void* recvBuff;
  volatile int size;
};

struct collNetSendResources {
  struct ncclComm* comm;
  void* collNetComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  uint32_t* llData;
  int netDev;
  int useGdr;
  void* sendMhandles[NCCL_NUM_PROTOCOLS];
  void* recvMhandles[NCCL_NUM_PROTOCOLS];
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
  struct reqSlot* reqFifo;
  int collNetRank;
};

struct collNetRecvResources {
  struct ncclComm* comm;
  void* collNetComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  uint32_t* llData;
  int netDev;
  int useGdr;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
  struct reqSlot* reqFifo;
  int collNetRank;
};

struct collNetSharedResources {
  void* collNetListenComms[MAXCHANNELS];
  void* collNetComms[MAXCHANNELS];
  int collNetCommRefCount[MAXCHANNELS];
};

/* Determine if we can communicate with the peer */
ncclResult_t collNetCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

ncclResult_t collNetSharedListen(struct ncclComm* comm, int netDev, void* collNetHandle) {
  struct collNetSharedResources* resources = (struct collNetSharedResources*)comm->proxyState.sharedBuffs.collNetResources;
  if (resources == NULL) {
    NCCLCHECK(ncclCalloc(&resources, 1));
    comm->proxyState.sharedBuffs.collNetResources = resources;
  }
  if (resources->collNetComms[netDev] == NULL)
    NCCLCHECK(collNetListen(netDev, collNetHandle, resources->collNetListenComms+netDev));
  return ncclSuccess;
}

/* Setup send connector, and return connect information for others in the coll communicator to connect to me */
ncclResult_t collNetSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct collNetSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  send->conn.shared = 1;
  resources->comm = comm;

  NCCLCHECK(ncclTopoGetNetDev(comm->topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, resources->netDev, 1, &resources->useGdr));

  send->proxyAppendPtr = comm->proxyState.sharedBuffs.proxyAppendCollNet+2*resources->netDev+1;

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));

  int recvSize = offsetof(struct ncclRecvMem, buff);
  // Simple uses shared buffers and we don't support LL128 
  recvSize += send->comm->buffSizes[NCCL_PROTO_LL];

  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostCalloc((char**)&resources->recvMem, recvSize));
  NCCLCHECK(ncclIbMalloc((void**)&(resources->llData), send->comm->buffSizes[NCCL_PROTO_LL]/2));

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [send] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

/* Setup recv connector */
ncclResult_t collNetRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct collNetRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  recv->conn.shared = 1;
  resources->comm = comm;

  NCCLCHECK(ncclTopoGetNetDev(comm->topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, resources->netDev, 0, &resources->useGdr));

  recv->proxyAppendPtr = comm->proxyState.sharedBuffs.proxyAppendCollNet+2*resources->netDev;

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));

  int recvSize = offsetof(struct ncclRecvMem, buff);
  // Simple uses shared buffers and we don't support LL128 
  recvSize += recv->comm->buffSizes[NCCL_PROTO_LL];

  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostCalloc((char**)&resources->recvMem, recvSize));

  NCCLCHECK(ncclIbMalloc((void**)&(resources->llData), recv->comm->buffSizes[NCCL_PROTO_LL]/2));

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [receive] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*) connectInfo;

  NCCLCHECK(collNetSharedListen(comm, resources->netDev, &info->collNetHandle));
  return ncclSuccess;
}

ncclResult_t collNetSharedConnect(struct ncclComm* comm, int netDev, struct ncclConnect* connectInfos, int nranks, int rank, void** collNetComm) {
  struct collNetSharedResources* resources = (struct collNetSharedResources*)comm->proxyState.sharedBuffs.collNetResources;
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

ncclResult_t collNetSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct collNetSendResources* resources = (struct collNetSendResources*)send->transportResources;
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(connectInfos+rank);

  // Intermediate buffering on GPU for GPU Direct RDMA, but LL buffer is always on host
  send->conn.buffs[NCCL_PROTO_LL] = resources->recvMem->buff;
  send->conn.buffs[NCCL_PROTO_LL128] = send->conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
  send->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;

  // Head/Tail/Opcount/Fifos are always on host
  send->conn.tail = &resources->recvMem->tail;
  send->conn.sizesFifo = resources->recvMem->sizesFifo;
  send->conn.ptrsFifo = resources->recvMem->ptrsFifo;
  send->conn.head = &resources->sendMem->head;
  resources->sendMem->head = -NCCL_STEPS; // Don't give any credit yet when sharing buffers
  for (int i=0; i<NCCL_STEPS; i++) send->conn.sizesFifo[i] = -1;

  // Get info from recv side
  resources->collNetRank = rank;
  resources->reqFifo = info->reqFifo;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    resources->recvMhandles[p] = info->mhandles[p];

  NCCLCHECK(collNetSharedConnect(comm, resources->netDev, connectInfos, nranks, rank, &resources->collNetComm));

  int size;
  char* ptr;
  NCCLCHECK(ncclProxySharedBuffersInit(send->comm, resources->useGdr, &size, &ptr));

  // Register buffers
  NCCLCHECK(collNetRegMr(resources->collNetComm,
        resources->useGdr ? send->comm->proxyState.sharedBuffs.cudaBuff : send->comm->proxyState.sharedBuffs.hostBuff,
        size,
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
        &resources->sendMhandles[NCCL_PROTO_SIMPLE]));
  NCCLCHECK(collNetRegMr(resources->collNetComm, resources->llData, send->comm->buffSizes[NCCL_PROTO_LL]/2,
        NCCL_PTR_HOST, &resources->sendMhandles[NCCL_PROTO_LL]));
  return ncclSuccess;
}

ncclResult_t collNetRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct collNetRecvResources* resources = (struct collNetRecvResources*)recv->transportResources;
  struct collNetSendConnectInfo* info = (struct collNetSendConnectInfo*)(connectInfos+rank);
  resources->collNetRank = rank;

  // Intermediate buffering on GPU for GPU Direct RDMA
  struct ncclRecvMem* recvMem = resources->useGdr ? resources->devRecvMem : resources->recvMem;
  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = (p == NCCL_PROTO_LL ? resources->recvMem->buff : recvMem->buff) + offset;
    offset += recv->comm->buffSizes[p];
  }
  recv->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;

  // Head/Tail/Opcount are always on host
  recv->conn.tail = &resources->recvMem->tail;
  recv->conn.ptrsFifo = resources->recvMem->ptrsFifo;
  recv->conn.head = &resources->sendMem->head;

  NCCLCHECK(collNetSharedConnect(comm, resources->netDev, connectInfos, nranks, rank, &resources->collNetComm));
  int size;
  char* ptr;
  NCCLCHECK(ncclProxySharedBuffersInit(recv->comm, resources->useGdr, &size, &ptr));

  // Register buffers
  NCCLCHECK(collNetRegMr(resources->collNetComm,
        resources->useGdr ? recv->comm->proxyState.sharedBuffs.cudaBuff : recv->comm->proxyState.sharedBuffs.hostBuff,
        size,
        resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST,
        &resources->mhandles[NCCL_PROTO_SIMPLE]));
  NCCLCHECK(collNetRegMr(resources->collNetComm, resources->llData, recv->comm->buffSizes[NCCL_PROTO_LL]/2,
        NCCL_PTR_HOST, &resources->mhandles[NCCL_PROTO_LL]));

  // Create shared info between send and recv proxies
  NCCLCHECK(ncclCalloc(&(resources->reqFifo), NCCL_STEPS));

  // Pass info to send side
  info->reqFifo = resources->reqFifo;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    info->mhandles[p] = resources->mhandles[p];

  return ncclSuccess;
}

ncclResult_t collNetSharedFree(struct ncclComm* comm, int netDev) {
  struct collNetSharedResources* resources = (struct collNetSharedResources*)comm->proxyState.sharedBuffs.collNetResources;
  resources->collNetCommRefCount[netDev]--;
  if (resources->collNetCommRefCount[netDev] == 0) {
    NCCLCHECK(collNetCloseColl(resources->collNetComms[netDev]));
  }
  for (int c=0; c<MAXCHANNELS; c++) if (resources->collNetCommRefCount[c]) return ncclSuccess;
  comm->proxyState.sharedBuffs.collNetResources = NULL;
  free(resources);
  return ncclSuccess;
}

ncclResult_t collNetSendFree(void* sendTransportResources) {
  struct collNetSendResources* resources = (struct collNetSendResources*)sendTransportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  if (resources->collNetComm) {
    NCCLCHECK(collNetDeregMr(resources->collNetComm, resources->sendMhandles[NCCL_PROTO_LL]));
    NCCLCHECK(collNetDeregMr(resources->collNetComm, resources->sendMhandles[NCCL_PROTO_SIMPLE]));
  }
  if (resources->useGdr)
    CUDACHECK(cudaFree(resources->devRecvMem));
  free(resources->llData);

  NCCLCHECK(collNetSharedFree(resources->comm, resources->netDev));
  free(resources);
  return ncclSuccess;
}

ncclResult_t collNetRecvFree(void* recvTransportResources) {
  struct collNetRecvResources* resources = (struct collNetRecvResources*)recvTransportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  if (resources->collNetComm) {
    NCCLCHECK(collNetDeregMr(resources->collNetComm, resources->mhandles[NCCL_PROTO_LL]));
    NCCLCHECK(collNetDeregMr(resources->collNetComm, resources->mhandles[NCCL_PROTO_SIMPLE]));
  }
  if (resources->useGdr)
    CUDACHECK(cudaFree(resources->devRecvMem));
  free(resources->llData);
  free(resources->reqFifo);

  NCCLCHECK(collNetSharedFree(resources->comm, resources->netDev));
  free(resources);
  return ncclSuccess;
}

ncclResult_t collNetSendProxy(struct ncclProxyArgs* args) {
  if (args->protocol == NCCL_PROTO_LL128) {
    WARN("CollNet does not support LL128");
    return ncclInternalError;
  }
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct collNetSendResources* resources = (struct collNetSendResources*) (sub->connector->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct collNetSendResources* resources = (struct collNetSendResources*) (sub->connector->transportResources);
      void* sendMhandle = resources->sendMhandles[p];
      void* recvMhandle = resources->recvMhandles[p];
      int stepSize = sub->connector->comm->buffSizes[p] / NCCL_STEPS;
      char* localBuff = sub->connector->conn.buffs[p];
      struct reqSlot* reqFifo = resources->reqFifo;
      if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        if (p == NCCL_PROTO_SIMPLE) {
          char* ptr;
          int sharedBuffSlot = sub->posted%NCCL_STEPS;
          NCCLCHECK(ncclProxySharedBuffersGetCollNet(sub->connector->comm, resources->useGdr, 0, sub->channel->id, sharedBuffSlot, s, &ptr));
          resources->recvMem->ptrsFifo[buffSlot] = ptr;
          __sync_synchronize();
        }
        volatile uint64_t* sendHead = &resources->sendMem->head;
        sub->posted += args->sliceSteps;
        *sendHead = sub->base + sub->posted - NCCL_STEPS;
      }
      // Enforce sync between operations of the same group.
      bool groupSync = (((s == 0) && ((sub+args->nsubs-1)->transmitted == sub->transmitted)) || (s && (sub-1)->transmitted > sub->transmitted));
      int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
      if (groupSync && sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS
        && reqFifo[buffSlot].recvBuff != NULL) {
        volatile int* sizesFifo = resources->recvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        if (sizesFifo[buffSlot] != -1 && ((*recvTail > (sub->base+sub->transmitted)) || p == NCCL_PROTO_LL)) {
          // We have something to receive, let's check if it's completely ready.
          int size = sizesFifo[buffSlot];
          char* buff = p == NCCL_PROTO_LL ? localBuff+buffSlot*stepSize : (char*)resources->recvMem->ptrsFifo[buffSlot];
          int ready = 1;
          if (p == NCCL_PROTO_LL) {
            uint32_t flag = NCCL_LL_FLAG(sub->base + sub->transmitted + 1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)buff;
            // Pack data into another buffer
            int stepLines = stepSize / sizeof(union ncclLLFifoLine);
            uint32_t* sendBuff = resources->llData+buffSlot*2*stepLines;  // each line has two data elements
            buff = (char*)sendBuff;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *d1 = &lines[i].data1;
              volatile uint32_t *f2 = &lines[i].flag2;
              volatile uint32_t *d2 = &lines[i].data2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
              sendBuff[2*i] = d1[0];
              sendBuff[2*i+1] = d2[0];
            }
            size = nFifoLines*2*sizeof(uint32_t);
          }
          if (ready) {
            // Data is ready, try to send.
            int count = size/ncclTypeSize(args->dtype);
            NCCLCHECK(collNetIallreduce(resources->collNetComm, (void*) buff, (void*)(reqFifo[buffSlot].recvBuff), count, args->dtype, args->redOp, sendMhandle, recvMhandle, sub->requests+buffSlot));
            if (sub->requests[buffSlot] != NULL) {
              TRACE(NCCL_NET, "sendProxy [%d/%d] Iallreduce posted, req %p", sub->transmitted, buffSlot, sub->requests[buffSlot]);
              sizesFifo[buffSlot] = -1;
              // Make sure size is reset to zero before we update the head.
              __sync_synchronize();
              sub->transmitted += args->sliceSteps;
              args->idle = 0;
              continue;
            }
          }
        }
      }
      // Check whether the network has completed some send operations.
      if (sub->done < sub->transmitted) {
        int done, size;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        NCCLCHECK(collNetTest((void*)(sub->requests[buffSlot]), &done, &size));
        if (done) {
          TRACE(NCCL_NET, "sendProxy [%d/%d] request %p done, size %d", sub->done, buffSlot, sub->requests[buffSlot], size);
          reqFifo[buffSlot].size = size;
          // Make sure size is updated before we set recvBuff to NULL (from the view of recv proxy, concerning the flush)
          // (reordered store after store is possible on POWER, though not on x86)
          __sync_synchronize();
          reqFifo[buffSlot].recvBuff = NULL; // Notify recvProxy
          sub->done += args->sliceSteps;
          args->idle = 0;
          if (sub->done == sub->nsteps) {
            resources->step = sub->base + sub->nsteps;
            args->done++;
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

ncclResult_t collNetRecvProxy(struct ncclProxyArgs* args) {
  if (args->protocol == NCCL_PROTO_LL128) {
    WARN("CollNet does not support LL128");
    return ncclInternalError;
  }
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct collNetRecvResources* resources = (struct collNetRecvResources*) (sub->connector->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->received = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct collNetRecvResources* resources = (struct collNetRecvResources*) (sub->connector->transportResources);
      void* mhandle = resources->mhandles[p];
      int stepSize = sub->connector->comm->buffSizes[p] / NCCL_STEPS;
      char* localBuff = sub->connector->conn.buffs[p];
      struct reqSlot* reqFifo = resources->reqFifo;
      if ((sub->posted < sub->done + NCCL_STEPS) && (sub->posted < sub->nsteps)) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        int recvStepSize = p == NCCL_PROTO_LL ? stepSize/2 : stepSize;
        char* ptr = ((char*)resources->llData) + buffSlot*recvStepSize;
        if (p == NCCL_PROTO_SIMPLE) {
          int sharedBuffSlot = sub->posted%NCCL_STEPS;
          NCCLCHECK(ncclProxySharedBuffersGetCollNet(sub->connector->comm, resources->useGdr, 1, sub->channel->id, sharedBuffSlot, s, &ptr));
          volatile void** ptrsFifo = (volatile void**)resources->recvMem->ptrsFifo;
          ptrsFifo[buffSlot] = ptr;
        }
        reqFifo[buffSlot].recvBuff = ptr;
        TRACE(NCCL_NET, "recvProxy [%d/%d] posted buffer %p", sub->posted, buffSlot, reqFifo[buffSlot].recvBuff);
        sub->posted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      if (sub->posted > sub->received) {
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;
        if (reqFifo[buffSlot].recvBuff == NULL) { // Buffer is cleared : coll is complete
          TRACE(NCCL_NET, "recvProxy [%d/%d] done, size %d", sub->received, buffSlot, reqFifo[buffSlot].size);
          if (p == NCCL_PROTO_LL) { // ll
            // re-attach flag
            uint32_t flag = NCCL_LL_FLAG(sub->base + sub->received + 1);
            int stepLines = stepSize / sizeof(union ncclLLFifoLine);
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+buffSlot*stepSize);
            uint32_t* recvData = resources->llData+buffSlot*2*stepLines;
            int nFifoLines = DIVUP(reqFifo[buffSlot].size, 2*sizeof(uint32_t));
            for (int i=0; i<nFifoLines; i++) {
              lines[i].v[0] = ((uint64_t)flag << 32) + recvData[2*i];
              lines[i].v[1] = ((uint64_t)flag << 32) + recvData[2*i+1];
            }
          }
          sub->received += args->sliceSteps;
          if (reqFifo[buffSlot].size > 0 && p == NCCL_PROTO_SIMPLE && resources->useGdr) {
            volatile void** ptrsFifo = (volatile void**)resources->recvMem->ptrsFifo;
            NCCLCHECK(collNetIflush(resources->collNetComm, (char*)ptrsFifo[buffSlot], reqFifo[buffSlot].size, mhandle, sub->requests+buffSlot));
          } else {
            sub->requests[buffSlot] = NULL;
          }
          args->idle = 0;
          continue;
        }
      }
      if (sub->received > sub->transmitted) {
        // Progress flush operations
        int buffSlot = (sub->base + sub->transmitted)%NCCL_STEPS;
        int done = 1;
        if (sub->requests[buffSlot]) NCCLCHECK(collNetTest(sub->requests[buffSlot], &done, NULL));
        if (done) {
          sub->transmitted += args->sliceSteps;
          __sync_synchronize();
          resources->recvMem->tail = sub->base + sub->transmitted;
          args->idle = 0;
          continue;
        }
      }
      if (sub->transmitted > sub->done) {
        volatile uint64_t* sendHead = &resources->sendMem->head;
        uint64_t done = *sendHead;
        while (done > (sub->base + sub->done) &&
            // LL and LL128 can acknowledge 0-bytes send before they even happen. Don't go past what we transmitted.
            sub->transmitted > sub->done) {
          sub->done += args->sliceSteps;
          args->idle = 0;
          if (sub->done == sub->nsteps) {
            resources->step = sub->base + sub->nsteps;
            args->done++;
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

struct ncclTransport collNetTransport = {
  "COL",
  collNetCanConnect,
  { collNetSendSetup, collNetSendConnect, collNetSendFree, collNetSendProxy },
  { collNetRecvSetup, collNetRecvConnect, collNetRecvFree, collNetRecvProxy }
};
