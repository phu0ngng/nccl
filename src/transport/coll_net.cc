/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "transport.h"
#include "nvmlwrap.h"
#include "coll_net.h"
#include "param.h"
#include "nvlink.h"
#include "net_common.h"
#include <cuda_runtime.h>
#include <assert.h>

static uint64_t collNetScores[NET_MAX_GPUS] = { NET_SCORES_UNSET };
static int collNetNDev;

struct collNetConnectInfo {
  collNetHandle_t collNetHandle;
};

struct ncclLLDataLine {
  uint32_t data1;
  uint32_t data2;
};
static_assert(sizeof(struct ncclLLDataLine) == sizeof(union ncclLLFifoLine)>>1, "ncclLLDataLine is not half size of ncclLLFifoLine");

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
  struct ncclLLDataLine* llData;
  int netDev;
  int useGdr;
  int buffSize;
  void* sendMhandle;
  void* llSendMhandle;
  void* recvMhandle;
  void* llRecvMhandle;
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llStep;
  uint64_t llLastCleaning;
  struct reqSlot* reqFifo;
};

struct collNetRecvResources {
  void* netListenComm;
  void* collNetRecvComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  struct ncclLLDataLine* llData;
  int netDev;
  int useGdr;
  int buffSize;
  void* mhandle;
  void* llMhandle;
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llStep;
  uint64_t llLastCleaning;
  struct reqSlot* reqFifo;
};

struct netInfoFuncs collNetInfoFuncs = {
  &collNetName,
  &collNetDevices,
  &collNetPciPath,
  &collNetPtrSupport
};

/* Determine if we can communicate with the peer */
ncclResult_t collNetCanConnect(int* ret, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo) {
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  ret[0] = collNetScores[cudaDev];
  if (ret[0] == NET_SCORES_UNSET) {
    if (cudaDev >= NET_MAX_GPUS) {
      WARN("CUDA device %d >= MAX %d\n", cudaDev, NET_MAX_GPUS);
      return ncclInternalError;
    }
    int nDev;
    short* distances;
    NCCLCHECK(netDevices(&nDev, &distances, &collNetInfoFuncs));
    collNetScores[cudaDev] = ret[0] = getTvalue(distances, nDev);
    collNetNDev = nDev;
    free(distances);
  }
  return ncclSuccess;
}

/* Setup send connector, and return connect information for others in the coll communicator to connect to me */
ncclResult_t collNetSendSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int buffSize, int channelId) {
  int sendSize = sizeof(struct ncclSendMem);
  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  int netDev = getDev(channelId/2, collNetScores[cudaDev], collNetNDev);

  // send side
  struct collNetSendResources* sendResources;
  NCCLCHECK(ncclCalloc(&sendResources, 1));
  send->transportResources = sendResources;

  sendResources->netDev = netDev;
  NCCLCHECK(netGetGdrSupport(sendResources->netDev, 1, &sendResources->useGdr, &collNetInfoFuncs));

  NCCLCHECK(ncclCudaHostAlloc((void**)&sendResources->hostSendMem, (void**)&sendResources->devHostSendMem, sendSize));

  if (sendResources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&sendResources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&sendResources->hostRecvMem, (void**)&sendResources->devHostRecvMem, recvSize));
  NCCLCHECK(ncclIbMalloc((void**)&(sendResources->llData), NCCL_LL_BUFF_LINES*sizeof(struct ncclLLDataLine)));
  sendResources->buffSize = buffSize;

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [send] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), sendResources->netDev,
      sendResources->useGdr ? "/GDRDMA" : "");

  // create shared info between send and recv proxies
  NCCLCHECK(ncclCalloc(&(sendResources->reqFifo), NCCL_STEPS));

  return ncclSuccess;
}

/* Setup recv connector */
ncclResult_t collNetRecvSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int buffSize, int channelId) {
  int sendSize = sizeof(struct ncclSendMem);
  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  int netDev = getDev(channelId/2, collNetScores[cudaDev], collNetNDev);

  // recv side
  struct collNetRecvResources* recvResources;
  NCCLCHECK(ncclCalloc(&recvResources, 1));
  recv->transportResources = recvResources;

  recvResources->netDev = netDev;
  NCCLCHECK(netGetGdrSupport(recvResources->netDev, 0, &recvResources->useGdr, &collNetInfoFuncs));

  NCCLCHECK(ncclCudaHostAlloc((void**)&recvResources->hostSendMem, (void**)&recvResources->devHostSendMem, sendSize));

  if (recvResources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&recvResources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&recvResources->hostRecvMem, (void**)&recvResources->devHostRecvMem, recvSize));
  NCCLCHECK(ncclIbMalloc((void**)&(recvResources->llData), NCCL_LL_BUFF_LINES*sizeof(struct ncclLLDataLine)));
  recvResources->buffSize = buffSize;

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [receive] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), recvResources->netDev,
      recvResources->useGdr ? "/GDRDMA" : "");

  struct collNetConnectInfo* info = (struct collNetConnectInfo*) connectInfo;
  NCCLCHECK(collNetListen(recvResources->netDev, &info->collNetHandle, &recvResources->netListenComm));

  return ncclSuccess;
}

ncclResult_t collNetSendConnect(struct ncclConnect* connectInfos, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct collNetSendResources* sendResources = (struct collNetSendResources*)send->transportResources;

  // Intermediate buffering on GPU for GPU Direct RDMA, but LL buffer is always on host
  struct ncclRecvMem* sRecvMem = sendResources->useGdr ? sendResources->devRecvMem : sendResources->devHostRecvMem;
  send->conn.buff = sRecvMem->buff;
  send->conn.llBuff = sendResources->devHostRecvMem->llBuff;

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

  // Intermediate buffering on GPU for GPU Direct RDMA
  struct ncclRecvMem* rRecvMem = recvResources->useGdr ? recvResources->devRecvMem : recvResources->devHostRecvMem;
  recv->conn.buff = rRecvMem->buff;
  recv->conn.llBuff = recvResources->devHostRecvMem->llBuff;  // recv LL buff always on host

  // Head/Tail/Opcount are always on host
  recv->conn.tail = &recvResources->devHostRecvMem->tail;
  recv->conn.opCountLoc = &recvResources->devHostRecvMem->opCount;
  recv->conn.head = &recvResources->devHostSendMem->head;
  recv->conn.opCountRem = &recvResources->devHostSendMem->opCount;

  // Connect to coll comm
  collNetHandle_t** handlePtrs = NULL;
  NCCLCHECK(ncclCalloc(&handlePtrs, nranks));
  for (int i = 0; i < nranks; i++) {
    struct collNetConnectInfo* info = (struct collNetConnectInfo*)(connectInfos+i);
    handlePtrs[i] = &(info->collNetHandle);
  }
  ncclResult_t res;
  NCCLCHECKGOTO(collNetConnect((void**)handlePtrs, nranks, rank, recvResources->netListenComm, &recvResources->collNetRecvComm), res, cleanup);

cleanup:
  if (handlePtrs != NULL) free(handlePtrs);
  // Close listen comm
  NCCLCHECK(collNetCloseListen(recvResources->netListenComm));

  return res;
}

ncclResult_t collNetConnectSendRecv(ncclConnector* send, ncclConnector* recv) {
  struct collNetSendResources* sendResources = (struct collNetSendResources*)send->transportResources;
  struct collNetRecvResources* recvResources = (struct collNetRecvResources*)recv->transportResources;
  recvResources->reqFifo = sendResources->reqFifo;
  sendResources->collNetSendComm = recvResources->collNetRecvComm;
  // Register buffer
  // send side
  struct ncclRecvMem* sRecvMem = sendResources->useGdr ? sendResources->devRecvMem : sendResources->devHostRecvMem;
  NCCLCHECK(collNetRegMr(sendResources->collNetSendComm, sRecvMem->buff, sendResources->buffSize,
        sendResources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &sendResources->sendMhandle));
  NCCLCHECK(collNetRegMr(sendResources->collNetSendComm, sendResources->llData,
        NCCL_LL_BUFF_LINES*sizeof(struct ncclLLDataLine), NCCL_PTR_HOST, &sendResources->llSendMhandle));
  // recv side
  struct ncclRecvMem* rRecvMem = recvResources->useGdr ? recvResources->devRecvMem : recvResources->devHostRecvMem;
  NCCLCHECK(collNetRegMr(recvResources->collNetRecvComm, rRecvMem->buff, recvResources->buffSize,
        recvResources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &recvResources->mhandle));
  NCCLCHECK(collNetRegMr(recvResources->collNetRecvComm, recvResources->llData,
        NCCL_LL_BUFF_LINES*sizeof(struct ncclLLDataLine), NCCL_PTR_HOST, &recvResources->llMhandle));
  // Share with send side as well (since iallreduce will need it)
  sendResources->recvMhandle = recvResources->mhandle;
  sendResources->llRecvMhandle = recvResources->llMhandle;

  return ncclSuccess;
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
  free(sendResources->reqFifo);
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

  // Make sure SendFree is called before RecvFree
  if (recvResources->collNetRecvComm) {
    NCCLCHECK(collNetCloseColl(recvResources->collNetRecvComm));
  }
  free(recvResources);
  return ncclSuccess;
}

ncclResult_t collNetSendProxy(struct ncclProxyArgs* args) {
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
        if (args->llMode) {
          int size = sizesFifo[buffSlot];
          if (size != -1) {
            uint32_t flag = NCCL_LL_FLAG(args->tail + 1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            union ncclLLFifoLine* lines = resources->hostRecvMem->llBuff+buffSlot*NCCL_LL_SLICE_LINES;
            int ready = 1;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
            }
            if (ready) {
              //separate data from flag
              struct ncclLLDataLine* sendBuff = resources->llData+buffSlot*NCCL_LL_SLICE_LINES;
              for (int i=0; i<nFifoLines; i++) {
                volatile uint32_t *d1 = &lines[i].data1;
                volatile uint32_t *d2 = &lines[i].data2;
                sendBuff[i].data1 = d1[0];
                sendBuff[i].data2 = d2[0];
              }
              int count = nFifoLines*sizeof(struct ncclLLDataLine) / ncclTypeSize(args->dtype);
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
          int stepSize = args->channel->buffSize/NCCL_STEPS;
          struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
          // Send through network
          int count = sizesFifo[buffSlot]/ncclTypeSize(args->dtype);
          NCCLCHECK(collNetIallreduce(resources->collNetSendComm, localMem->buff+buffSlot*stepSize, (void*)(reqFifo[buffSlot].recvBuff), count, args->dtype, args->redOp, resources->sendMhandle, resources->recvMhandle, args->requests+buffSlot));
          if (args->requests[buffSlot] != NULL) {
            TRACE(NCCL_NET, "sendProxy [%d/%d] Iallreduce posted, req %p", args->head, buffSlot, args->requests[buffSlot]);
            sizesFifo[buffSlot] = -1;
            // Make sure size is reset to zero before we update the head.
            __sync_synchronize();
            args->tail += args->sliceSteps;
            args->idle = 0;
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
    int stepSize = ( args->llMode ? NCCL_LL_BUFF_LINES*sizeof(struct ncclLLDataLine) : args->channel->buffSize ) / NCCL_STEPS;
    struct reqSlot* reqFifo = resources->reqFifo;
    if (args->head < args->end) {
      struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
      char* localBuff = args->llMode ? (char*)resources->llData : localMem->buff;  // LL always on host mem
      void* mhandle = args->llMode ? resources->llMhandle : resources->mhandle;
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
          if (args->llMode == 0) {
            if (resources->useGdr) collNetFlush(resources->collNetRecvComm, localBuff+buffSlot*stepSize, reqFifo[buffSlot].size, mhandle);
            resources->hostRecvMem->tail = args->head;
          } else { // ll
            // re-attach flag
            uint32_t flag = args->head;
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(resources->hostRecvMem->llBuff)+buffSlot*NCCL_LL_SLICE_LINES;
            struct ncclLLDataLine* recvData = resources->llData+buffSlot*NCCL_LL_SLICE_LINES;
            int nFifoLines = DIVUP(reqFifo[buffSlot].size, sizeof(struct ncclLLDataLine));
            for (int i=0; i<nFifoLines; i++) {
              lines[i].v[0] = ((uint64_t)flag << 32) + recvData[i].data1;
              lines[i].v[1] = ((uint64_t)flag << 32) + recvData[i].data2;
            }
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

struct ncclCollTransport collNetTransport = {
  "COL",
  collNetCanConnect,
  collNetConnectSendRecv,
  { collNetSendSetup, collNetSendConnect, collNetSendFree, collNetSendProxy },
  { collNetRecvSetup, collNetRecvConnect, collNetRecvFree, collNetRecvProxy }
};
