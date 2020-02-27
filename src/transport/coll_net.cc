/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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
  void* collNetComm;
  void* mhandle;
  void* llMhandle;
  struct reqSlot* reqFifo;
};

struct reqSlot {
  volatile void* recvBuff;
  volatile int size;
};

struct collNetSendResources {
  void* collNetSendComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  uint32_t* llData;
  int netDev;
  int useGdr;
  void* sendMhandle;
  void* llSendMhandle;
  void* recvMhandle;
  void* llRecvMhandle;
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
  struct reqSlot* reqFifo;
  int collNetRank;
};

struct collNetRecvResources {
  void* netListenComm;
  void* collNetRecvComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  uint32_t* llData;
  int netDev;
  int useGdr;
  void* mhandle;
  void* llMhandle;
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
  struct reqSlot* reqFifo;
  int collNetRank;
};

/* Determine if we can communicate with the peer */
ncclResult_t collNetCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

/* Setup send connector, and return connect information for others in the coll communicator to connect to me */
ncclResult_t collNetSendSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId) {
  struct collNetSendResources* sendResources;
  NCCLCHECK(ncclCalloc(&sendResources, 1));
  send->transportResources = sendResources;

  NCCLCHECK(ncclTopoGetNetDev(graph, 1, channelId, &sendResources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(topo, myInfo->busId, sendResources->netDev, 1, &sendResources->useGdr));

  int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaHostAlloc((void**)&sendResources->hostSendMem, (void**)&sendResources->devHostSendMem, sendSize));

  int recvSize = offsetof(struct ncclRecvMem, buff) +
    send->comm->llBuffSize +
    send->comm->ll128BuffSize +
    send->comm->buffSize;
  if (sendResources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&sendResources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&sendResources->hostRecvMem, (void**)&sendResources->devHostRecvMem, recvSize));
  NCCLCHECK(ncclIbMalloc((void**)&(sendResources->llData), send->comm->llBuffSize/2));

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [send] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), sendResources->netDev,
      sendResources->useGdr ? "/GDRDMA" : "");

  return ncclSuccess;
}

/* Setup recv connector */
ncclResult_t collNetRecvSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId) {
  struct collNetRecvResources* recvResources;
  NCCLCHECK(ncclCalloc(&recvResources, 1));
  recv->transportResources = recvResources;

  NCCLCHECK(ncclTopoGetNetDev(graph, 0, channelId, &recvResources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(topo, myInfo->busId, recvResources->netDev, 0, &recvResources->useGdr));

  int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaHostAlloc((void**)&recvResources->hostSendMem, (void**)&recvResources->devHostSendMem, sendSize));

  int recvSize = offsetof(struct ncclRecvMem, buff) +
    recv->comm->llBuffSize +
    recv->comm->ll128BuffSize +
    recv->comm->buffSize;
  if (recvResources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&recvResources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&recvResources->hostRecvMem, (void**)&recvResources->devHostRecvMem, recvSize));
  NCCLCHECK(ncclIbMalloc((void**)&(recvResources->llData), recv->comm->llBuffSize/2));

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [receive] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), recvResources->netDev,
      recvResources->useGdr ? "/GDRDMA" : "");

  struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*) connectInfo;
  NCCLCHECK(collNetListen(recvResources->netDev, &info->collNetHandle, &recvResources->netListenComm));

  return ncclSuccess;
}

ncclResult_t collNetSendConnect(struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct collNetSendResources* sendResources = (struct collNetSendResources*)send->transportResources;
  sendResources->collNetRank = rank;

  // Get info from recv side
  struct collNetSendConnectInfo* sInfo = (struct collNetSendConnectInfo*)(connectInfos+rank);
  sendResources->reqFifo = sInfo->reqFifo;
  sendResources->collNetSendComm = sInfo->collNetComm;
  sendResources->recvMhandle = sInfo->mhandle;
  sendResources->llRecvMhandle = sInfo->llMhandle;

  // Intermediate buffering on GPU for GPU Direct RDMA, but LL buffer is always on host
  struct ncclRecvMem* sRecvMem = sendResources->useGdr ? sendResources->devRecvMem : sendResources->devHostRecvMem;
  // Register buffers
  NCCLCHECK(collNetRegMr(sendResources->collNetSendComm, sRecvMem->buff, send->comm->buffSize,
        sendResources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &sendResources->sendMhandle));
  NCCLCHECK(collNetRegMr(sendResources->collNetSendComm, sendResources->llData, send->comm->llBuffSize/2,
        NCCL_PTR_HOST, &sendResources->llSendMhandle));

  send->conn.buff = sRecvMem->buff+send->comm->llBuffSize+send->comm->ll128BuffSize;
  send->conn.llBuff = (union ncclLLFifoLine*)sendResources->devHostRecvMem->buff;
  send->conn.direct |= sendResources->useGdr ? NCCL_DIRECT_NIC : 0;

  // Head/Tail/Opcount/Fifos are always on host
  send->conn.tail = &sendResources->devHostRecvMem->tail;
  send->conn.opCountRem = &sendResources->devHostRecvMem->opCount;
  send->conn.fifo = sendResources->devHostRecvMem->sizesFifo;
  send->conn.head = &sendResources->devHostSendMem->head;
  send->conn.opCountLoc = &sendResources->devHostSendMem->opCount;
  for (int i=0; i<NCCL_STEPS; i++) send->conn.fifo[i] = -1;

  return ncclSuccess;
}

ncclResult_t collNetRecvConnect(struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct collNetRecvResources* recvResources = (struct collNetRecvResources*)recv->transportResources;
  struct collNetSendConnectInfo* sInfo = (struct collNetSendConnectInfo*)(connectInfos+rank);
  recvResources->collNetRank = rank;

  // Intermediate buffering on GPU for GPU Direct RDMA
  struct ncclRecvMem* rRecvMem = recvResources->useGdr ? recvResources->devRecvMem : recvResources->devHostRecvMem;
  recv->conn.buff = rRecvMem->buff+recv->comm->llBuffSize+recv->comm->ll128BuffSize;
  recv->conn.llBuff = (union ncclLLFifoLine*)recvResources->devHostRecvMem->buff;  // recv LL buff always on host
  recv->conn.direct |= recvResources->useGdr ? NCCL_DIRECT_NIC : 0;

  // Head/Tail/Opcount are always on host
  recv->conn.tail = &recvResources->devHostRecvMem->tail;
  recv->conn.opCountLoc = &recvResources->devHostRecvMem->opCount;
  recv->conn.head = &recvResources->devHostSendMem->head;
  recv->conn.opCountRem = &recvResources->devHostSendMem->opCount;

  // Connect to coll comm
  collNetHandle_t** handlePtrs = NULL;
  NCCLCHECK(ncclCalloc(&handlePtrs, nranks));
  for (int i = 0; i < nranks; i++) {
    struct collNetRecvConnectInfo* info = (struct collNetRecvConnectInfo*)(connectInfos+i);
    handlePtrs[i] = &(info->collNetHandle);
  }
  ncclResult_t res;
  NCCLCHECKGOTO(collNetConnect((void**)handlePtrs, nranks, rank, recvResources->netListenComm, &recvResources->collNetRecvComm), res, cleanup);

  // Register buffers
  NCCLCHECK(collNetRegMr(recvResources->collNetRecvComm, rRecvMem->buff, recv->comm->buffSize,
        recvResources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &recvResources->mhandle));
  NCCLCHECK(collNetRegMr(recvResources->collNetRecvComm, recvResources->llData, recv->comm->llBuffSize/2,
        NCCL_PTR_HOST, &recvResources->llMhandle));

  // Create shared info between send and recv proxies
  NCCLCHECK(ncclCalloc(&(recvResources->reqFifo), NCCL_STEPS));

  // Pass info to send side
  sInfo->reqFifo = recvResources->reqFifo;
  sInfo->collNetComm = recvResources->collNetRecvComm;
  sInfo->mhandle = recvResources->mhandle;
  sInfo->llMhandle = recvResources->llMhandle;

cleanup:
  if (handlePtrs != NULL) free(handlePtrs);
  // Close listen comm
  NCCLCHECK(collNetCloseListen(recvResources->netListenComm));

  return res;
}

ncclResult_t collNetSendFree(void* sendTransportResources) {
  struct collNetSendResources* sendResources = (struct collNetSendResources*)sendTransportResources;
  NCCLCHECK(ncclCudaHostFree(sendResources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(sendResources->hostRecvMem));
  if (sendResources->collNetSendComm) {
    NCCLCHECK(collNetDeregMr(sendResources->collNetSendComm, sendResources->sendMhandle));
    NCCLCHECK(collNetDeregMr(sendResources->collNetSendComm, sendResources->llSendMhandle));
  }
  if (sendResources->useGdr)
    CUDACHECK(cudaFree(sendResources->devRecvMem));
  free(sendResources->llData);
  free(sendResources);
  return ncclSuccess;
}

ncclResult_t collNetRecvFree(void* recvTransportResources) {
  struct collNetRecvResources* recvResources = (struct collNetRecvResources*)recvTransportResources;
  NCCLCHECK(ncclCudaHostFree(recvResources->hostSendMem));
  if (recvResources->collNetRecvComm) {
    NCCLCHECK(collNetDeregMr(recvResources->collNetRecvComm, recvResources->mhandle));
    NCCLCHECK(collNetDeregMr(recvResources->collNetRecvComm, recvResources->llMhandle));
  }
  NCCLCHECK(ncclCudaHostFree(recvResources->hostRecvMem));
  if (recvResources->useGdr)
    CUDACHECK(cudaFree(recvResources->devRecvMem));
  free(recvResources->llData);
  free(recvResources->reqFifo);

  // Make sure SendFree is called before RecvFree
  if (recvResources->collNetRecvComm) {
    NCCLCHECK(collNetCloseColl(recvResources->collNetRecvComm));
  }
  free(recvResources);
  return ncclSuccess;
}

ncclResult_t collNetSendProxy(struct ncclProxyArgs* args) {
  if (args->protocol == NCCL_PROTO_LL128) {
    WARN("CollNet does not support LL128");
    return ncclInternalError;
  }
  struct collNetSendResources* resources = (struct collNetSendResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Update opCount
    resources->hostRecvMem->opCount = args->opCount;

    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->head = resources->step;
    args->tail = resources->step;
    args->end = args->head + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    args->idle = 1;
    struct reqSlot* reqFifo = resources->reqFifo;
    if (args->head < args->end) {
      int buffSlot = args->tail%NCCL_STEPS;
      if (args->tail < args->end && args->tail < args->head + NCCL_STEPS
          && reqFifo[buffSlot].recvBuff != NULL) {
        volatile int* sizesFifo = resources->hostRecvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->hostRecvMem->tail;
        if (args->protocol == NCCL_PROTO_LL) {
          int stepLines = args->connector->comm->llBuffSize/(NCCL_STEPS*sizeof(union ncclLLFifoLine));
          int size = sizesFifo[buffSlot];
          if (size != -1) {
            uint32_t flag = NCCL_LL_FLAG(args->tail + 1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            union ncclLLFifoLine* lines = args->connector->conn.llBuff+buffSlot*stepLines;
            int ready = 1;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
            }
            if (ready) {
              int stepLines = args->connector->comm->llBuffSize/(NCCL_STEPS*sizeof(union ncclLLFifoLine));
              //separate data from flag
              uint32_t* sendBuff = resources->llData+buffSlot*2*stepLines;
              for (int i=0; i<nFifoLines; i++) {
                volatile uint32_t *d1 = &lines[i].data1;
                volatile uint32_t *d2 = &lines[i].data2;
                sendBuff[2*i] = d1[0];
                sendBuff[2*i+1] = d2[0];
              }
              int count = nFifoLines*2*sizeof(uint32_t) / ncclTypeSize(args->dtype);
              NCCLCHECK(collNetIallreduce(resources->collNetSendComm, (void*)sendBuff, (void*)(reqFifo[buffSlot].recvBuff), count, args->dtype, args->redOp, resources->llSendMhandle, resources->llRecvMhandle, args->requests+buffSlot));
              if (args->requests[buffSlot] != NULL) {
                TRACE(NCCL_NET, "sendProxy [%d/%d] Iallreduce (LL) posted, req %p", args->head, buffSlot, args->requests[buffSlot]);
                sizesFifo[buffSlot] = -1;
                // Make sure size is reset to zero before we update the head.
                __sync_synchronize();
                args->tail += args->sliceSteps;
                args->idle = 0;
              }
            }
          }
        } else if (args->tail < *recvTail) {
          int stepSize = args->connector->comm->buffSize/NCCL_STEPS;
          struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
          // Send through network
          if (sizesFifo[buffSlot] != -1) {
            int count = sizesFifo[buffSlot]/ncclTypeSize(args->dtype);
            NCCLCHECK(collNetIallreduce(resources->collNetSendComm, localMem->buff+buffSlot*stepSize, (void*)(reqFifo[buffSlot].recvBuff), count, args->dtype, args->redOp, resources->sendMhandle, resources->recvMhandle, args->requests+buffSlot));
            if (args->requests[buffSlot] != NULL) {
              TRACE(NCCL_NET, "sendProxy [%d/%d] Iallreduce posted, req %p count %d", args->head, buffSlot, args->requests[buffSlot], count);
              sizesFifo[buffSlot] = -1;
              // Make sure size is reset to zero before we update the head.
              __sync_synchronize();
              args->tail += args->sliceSteps;
              args->idle = 0;
            }
          }
        }
      }
      if (args->head < args->tail) {
        int done, size;
        int buffSlot = args->head%NCCL_STEPS;
        NCCLCHECK(collNetTest((void*)(args->requests[buffSlot]), &done, &size));
        if (done) {
          TRACE(NCCL_NET, "sendProxy [%d/%d] request %p done, size %d", args->head, buffSlot, args->requests[buffSlot], size);
          reqFifo[buffSlot].size = size;
          // Make sure size is updated before we set recvBuff to NULL (from the view of recv proxy, concerning the flush)
          // (reordered store after store is possible on POWER, though not on x86)
          __sync_synchronize();
          reqFifo[buffSlot].recvBuff = NULL; // Notify recvProxy
          args->head += args->sliceSteps;
          resources->hostSendMem->head = args->head;
          args->idle = 0;
        }
      }
    }
    if (args->head == args->end) {
      resources->step = args->end;
      args->idle = 0;
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
  struct collNetRecvResources* resources = (struct collNetRecvResources*) (args->connector->transportResources);
  if (args->state == ncclProxyOpReady) {
    // Update opCount
    resources->hostSendMem->opCount = args->opCount;

    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    args->head = resources->step;
    args->tail = resources->step;
    args->end = args->head + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    args->idle = 1;
    int stepSize = ( args->protocol == NCCL_PROTO_LL ? args->connector->comm->llBuffSize : args->protocol == NCCL_PROTO_LL128 ? args->connector->comm->ll128BuffSize : args->connector->comm->buffSize ) / NCCL_STEPS;
    struct reqSlot* reqFifo = resources->reqFifo;
    if (args->head < args->end) {
      struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
      char* localBuff = args->protocol == NCCL_PROTO_LL ? (char*)resources->llData : localMem->buff;
      void* mhandle = args->protocol == NCCL_PROTO_LL ? resources->llMhandle : resources->mhandle;
      if ((args->tail < args->head + NCCL_STEPS) && (args->tail < (resources->hostSendMem->head) + NCCL_STEPS) && (args->tail < args->end)) {
        int buffSlot = args->tail%NCCL_STEPS;
        reqFifo[buffSlot].recvBuff = localBuff+buffSlot*stepSize;
        TRACE(NCCL_NET, "recvProxy [%d/%d] posted buffer %p", args->tail, buffSlot, localBuff+buffSlot*stepSize);
        args->tail += args->sliceSteps;
        args->idle = 0;
      }
      if (args->tail > args->head) {
        int buffSlot = args->head%NCCL_STEPS;
        if (reqFifo[buffSlot].recvBuff == NULL) { // Buffer is cleared : coll is complete
          TRACE(NCCL_NET, "recvProxy [%d/%d] done, size %d", args->head, buffSlot, reqFifo[buffSlot].size);
          args->head += args->sliceSteps;
          if (args->protocol == NCCL_PROTO_LL) { // ll
            // re-attach flag
            uint32_t flag = args->head;
            int stepLines = stepSize / sizeof(union ncclLLFifoLine);
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(args->connector->conn.llBuff)+buffSlot*stepLines;
            uint32_t* recvData = resources->llData+buffSlot*2*stepLines;
            int nFifoLines = DIVUP(reqFifo[buffSlot].size, 2*sizeof(uint32_t));
            for (int i=0; i<nFifoLines; i++) {
              lines[i].v[0] = ((uint64_t)flag << 32) + recvData[2*i];
              lines[i].v[1] = ((uint64_t)flag << 32) + recvData[2*i+1];
            }
          } else if (args->protocol == NCCL_PROTO_SIMPLE) {
            if (resources->useGdr) collNetFlush(resources->collNetRecvComm, localBuff+buffSlot*stepSize, reqFifo[buffSlot].size, mhandle);
            resources->hostRecvMem->tail = args->head;
          }
          args->idle = 0;
        }
      }
    }
    if (args->head == args->end) {
      resources->step = args->end;
      args->idle = 0;
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
