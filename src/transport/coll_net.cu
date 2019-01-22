/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
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

static ncclTvalue_t collNetTvalues[NET_MAX_GPUS] = { NET_TVALUE_UNKNOWN };
static int collNetNDev;

struct collNetConnectInfo {
  collNetHandle_t collNetHandle;
};

/* State type */
typedef enum { collReqNone        =  0,
               collReqPosted      =  1,
               collReqDone        =  2,
               collReqNumStates   =  3 } collReqState_t;

// TODO: find correct way to share things between two proxies
#define SHARED_REQ_Q

#ifdef SHARED_REQ_Q
struct reqState {
  volatile void* intmBuff;
  volatile collReqState_t state;
  volatile void* request;
};
#endif

struct collNetSendResources {
  void* collNetSendComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  int netDev;
  int useGdr;
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llStep;
  uint64_t llLastCleaning;
#ifdef SHARED_REQ_Q
  struct reqState* reqFifo;
#endif
};

struct collNetRecvResources {
  void* netListenComm;
  void* collNetRecvComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  int netDev;
  int useGdr;
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llStep;
  uint64_t llLastCleaning;
#ifdef SHARED_REQ_Q
  struct reqState* reqFifo;
#endif
  uint64_t reqFifoTail;
};

struct netInfoFuncs collNetInfoFuncs = {
  &collNetName,
  &collNetDevices,
  &collNetPciPath,
  &collNetPtrSupport
};
  
/* Determine if we can communicate with the peer */
ncclResult_t collNetCanConnect(ncclTvalue_t* ret, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo) {
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  ret[0] = collNetTvalues[cudaDev];
  if (ret[0] == NET_TVALUE_UNKNOWN) {
    if (cudaDev >= NET_MAX_GPUS) {
      WARN("CUDA device %d >= MAX %d\n", cudaDev, NET_MAX_GPUS);
      return ncclInternalError;
    }
    int nDev;
    short* distances;
    NCCLCHECK(netDevices(&nDev, &distances, &collNetInfoFuncs));
    collNetTvalues[cudaDev] = ret[0] = getTvalue(distances, nDev);
    collNetNDev = nDev;
    free(distances);
  }
  return ncclSuccess;
}

/* Setup send connector and recv connector, and return connect information for others in the coll communicator to connect to me */
ncclResult_t collNetSetup(struct ncclPeerInfo* myInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, struct ncclConnector* recv, int buffSize, int channelId) {
  int sendSize = sizeof(struct ncclSendMem);
  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  int netDev = getDev(channelId, collNetTvalues[cudaDev], collNetNDev);

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

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [send] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), sendResources->netDev,
      sendResources->useGdr ? "/GDRDMA" : "");

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

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [receive] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), recvResources->netDev,
      recvResources->useGdr ? "/GDRDMA" : "");

  struct collNetConnectInfo* info = (struct collNetConnectInfo*) connectInfo;
  NCCLCHECK(collNetListen(recvResources->netDev, &info->collNetHandle, &recvResources->netListenComm));

#ifdef SHARED_REQ_Q
  // create shared info between send and recv proxies
  NCCLCHECK(ncclCalloc(&(sendResources->reqFifo), NCCL_STEPS));
  recvResources->reqFifo = sendResources->reqFifo;
#endif

  return ncclSuccess;
}

ncclResult_t collNetConnect(struct ncclConnect* connectInfos, int nranks, struct ncclConnector* send, struct ncclConnector* recv) {
  // send side
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

  // recv side
  // Setup device pointers
  struct collNetRecvResources* recvResources = (struct collNetRecvResources*)recv->transportResources;

  // Intermediate buffering on GPU for GPU Direct RDMA
  struct ncclRecvMem* rRecvMem = recvResources->useGdr ? recvResources->devRecvMem : recvResources->devHostRecvMem;
  recv->conn.buff = rRecvMem->buff;
  recv->conn.llBuff = rRecvMem->llBuff;

  // Head/Tail/Opcount are always on host
  recv->conn.tail = &recvResources->devHostRecvMem->tail;
  recv->conn.opCountLoc = &recvResources->devHostRecvMem->opCount;
  recv->conn.head = &recvResources->devHostSendMem->head;
  recv->conn.opCountRem = &recvResources->devHostSendMem->opCount;

  // Connect to coll comm
  struct collNetConnectInfo* infos = (struct collNetConnectInfo*)connectInfos;
  collNetHandle_t* handlePtrs[nranks];
  for (int i = 0; i < nranks; i++) {
    handlePtrs[i] = &(infos[i].collNetHandle);
  }
  NCCLCHECK(collNetConnect((void**)handlePtrs, nranks, recvResources->netListenComm, &sendResources->collNetSendComm));
  recvResources->collNetRecvComm = sendResources->collNetSendComm;

  // Close listen comm
  NCCLCHECK(collNetCloseListen(recvResources->netListenComm));

  return ncclSuccess;
}

ncclResult_t collNetFree(void* sendTransportResources, void* recvTransportResources) {
  // send side
  struct collNetSendResources* sendResources = (struct collNetSendResources*)sendTransportResources;
  NCCLCHECK(ncclCudaHostFree(sendResources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(sendResources->hostRecvMem));
  if (sendResources->useGdr)
    CUDACHECK(cudaFree(sendResources->devRecvMem));
  NCCLCHECK(collNetCloseColl(sendResources->collNetSendComm));
  free(sendResources->reqFifo);
  free(sendResources);

  // recv side
  struct collNetRecvResources* recvResources = (struct collNetRecvResources*)recvTransportResources;
  NCCLCHECK(ncclCudaHostFree(recvResources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(recvResources->hostRecvMem));
  if (recvResources->useGdr)
    CUDACHECK(cudaFree(recvResources->devRecvMem));
  free(recvResources);
  return ncclSuccess;
}

ncclResult_t collNetSendProxy(struct ncclProxyArgs* args) {
  struct collNetSendResources* resources = (struct collNetSendResources*) (args->connector->transportResources);
  int supported;
  NCCLCHECK(collNetReduceSupport(args->dtype, args->redOp, &supported));
  if (supported != 1) return ncclInternalError;

  ///////////////////// start //////////////////
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
    volatile uint64_t* prevTail = &resources->hostRecvMem->tail;
    struct ncclSendMem* prevMem = resources->hostSendMem;
    uint64_t* prevHead = &prevMem->head;
    struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
    union ncclLLFifoLine* llBuff = resources->hostRecvMem->llBuff;
    int ptrType = resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
    volatile int* sizesFifo = resources->hostRecvMem->sizesFifo;
    int stepSize = args->channel->buffSize/NCCL_STEPS;
#ifdef SHARED_REQ_Q
    struct reqState* reqFifo = resources->reqFifo;
#endif

    INFO(NCCL_INIT,"Send proxy : opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d ==> Start", args->opCount, args->head, args->tail, args->end, args->nsteps, args->llMode);
    TRACE(NET,"opCount %lx stepSize %d stepSize %d ptrType %d", args->opCount, stepSize, stepSize, ptrType);

    if (args->head < args->end) {
      if (args->tail < args->end && args->tail < args->head + NCCL_STEPS) {
        if (args->llMode) {
          int buffSlot = args->tail%NCCL_STEPS;
          int size = sizesFifo[buffSlot];
          if (size != -1) {
            uint32_t flag = args->tail + 1;
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            size = nFifoLines * sizeof(union ncclLLFifoLine);
            union ncclLLFifoLine* lines = llBuff+buffSlot*NCCL_LL_SLICE_LINES;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              while (f1[0] != flag || f2[0] != flag);
            }
            int count = size / ncclTypeSize(args->dtype);
            NCCLCHECK(collNetIallreduce(resources->collNetSendComm, lines, (void*)(reqFifo[buffSlot].intmBuff), count, args->dtype, args->redOp, ptrType, args->requests+buffSlot));
            if (args->requests[buffSlot] != NULL) {
              sizesFifo[buffSlot] = -1;
              // Make sure size is reset to zero before we update the head.
              __sync_synchronize();
              args->tail += args->sliceSteps;
              args->idle = 0;
            }
          }
        } else if (args->tail < *prevTail) {
          // Send through network
          int buffSlot = args->tail%NCCL_STEPS;
          int count = sizesFifo[buffSlot]/ncclTypeSize(args->dtype);
#ifdef SHARED_REQ_Q
          if (reqFifo[buffSlot].state != collReqNone || reqFifo[buffSlot].intmBuff == NULL) {
            goto end;
          }
#endif
          NCCLCHECK(collNetIallreduce(resources->collNetSendComm, localMem->buff+buffSlot*stepSize, (void*)(reqFifo[buffSlot].intmBuff), count, args->dtype, args->redOp, ptrType, args->requests+buffSlot));
          INFO(NCCL_INIT,"Send proxy : opCount %lx head %lx tail %lx prevTail %p prevTail %lx end %lx nsteps %d llMode %d count %d request %p ==> Posted", args->opCount, args->head, args->tail, prevTail, *prevTail, args->end, args->nsteps, args->llMode, count, args->requests[buffSlot]);
          if (args->requests[buffSlot] != NULL) {
            sizesFifo[buffSlot] = -1;
#ifdef SHARED_REQ_Q
            reqFifo[buffSlot].state = collReqPosted;
            reqFifo[buffSlot].request = args->requests[buffSlot];
#endif
            // Make sure size is reset to zero before we update the head.
            __sync_synchronize();
            args->tail += args->sliceSteps;
            args->idle = 0;
          }
end:
        }
      }
      if (args->head < args->tail) {
        int buffSlot = args->head%NCCL_STEPS;
        if (reqFifo[buffSlot].state == collReqDone) {
          INFO(NCCL_INIT,"Send proxy : opCount %lx head %lx tail %lx prevTail %p prevTail %lx end %lx nsteps %d llMode %d request %p ==> Done", args->opCount, args->head, args->tail, prevTail, *prevTail, args->end, args->nsteps, args->llMode, reqFifo[buffSlot].request);
#ifdef SHARED_REQ_Q
          reqFifo[buffSlot].state = collReqNone;
          reqFifo[buffSlot].intmBuff = NULL;
#endif
          args->head += args->sliceSteps;
          *prevHead = args->head;
          args->idle = 0;
        }
      }
      if (args->head == args->end) {
        resources->step = args->end;
        args->state = ncclProxyOpDone;
      }
    }
  }
  if (args->state == ncclProxyOpDone) {
    union ncclLLFifoLine* llBuff = resources->hostRecvMem->llBuff;
    struct ncclSendMem* prevMem = resources->hostSendMem;
    uint64_t* prevHead = &prevMem->head;
    INFO(NCCL_INIT,"Send proxy : opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d ==> Cleaning", args->opCount, args->head, args->tail, args->end, args->nsteps, args->llMode);
    if (args->llMode && resources->step > resources->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
      for (int i=0; i< NCCL_LL_BUFF_LINES; i++) llBuff[i].flag1 = llBuff[i].flag2 = resources->step;
      resources->step += NCCL_STEPS;
      *prevHead = resources->step;
      resources->llLastCleaning = resources->step;
    }
    args->state = ncclProxyOpNone;
  }
  return ncclSuccess;
}

ncclResult_t collNetRecvProxy(struct ncclProxyArgs* args) {
  struct collNetRecvResources* resources = (struct collNetRecvResources*) (args->connector->transportResources);
  int supported;
  NCCLCHECK(collNetReduceSupport(args->dtype, args->redOp, &supported));
  if (supported != 1) return ncclInternalError;

  ///////// START /////////
  if (args->state == ncclProxyOpReady) {
    // Update opCount
    resources->hostSendMem->opCount = args->opCount;

    // Round to next multiple of sliceSteps
    resources->step = ROUNDUP(resources->step, args->chunkSteps);
    resources->reqFifoTail = resources->step;
    args->head = resources->step;
    args->tail = resources->step;
    args->end = args->head + args->nsteps;
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    volatile uint64_t* nextHead = &resources->hostSendMem->head;
    struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
    char* localBuff = args->llMode ? (char*)localMem->llBuff : localMem->buff;
    int ptrType = resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
    uint64_t* nextTail = &resources->hostRecvMem->tail;

    int stepSize = ( args->llMode ? NCCL_LL_BUFF_SIZE : args->channel->buffSize ) / NCCL_STEPS;

#ifdef SHARED_REQ_Q
    struct reqState* reqFifo = resources->reqFifo;
#endif
    uint64_t* reqFifoTail = &resources->reqFifoTail;

    INFO(NCCL_INIT,"Recv proxy : opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d ==> Start", args->opCount, args->head, args->tail, args->end, args->nsteps, args->llMode);
    TRACE(NET,"opCount %lx buffSize %d stepSize %d ptrType %d", args->opCount, args->channel->buffSize, stepSize, ptrType);

    if (args->head < args->end) {
#ifdef SHARED_REQ_Q
      // enqueue an intermediate buff address
      if (*reqFifoTail < args->tail + NCCL_STEPS && *reqFifoTail < args->end) {
        int readyTail = *reqFifoTail%NCCL_STEPS;
        if (reqFifo[readyTail].intmBuff == NULL) {
          reqFifo[readyTail].intmBuff = localBuff+readyTail*stepSize;
          *reqFifoTail += args->sliceSteps;
        }
      }
#endif
      if ((args->tail < args->head + NCCL_STEPS) && (args->tail < (*nextHead) + NCCL_STEPS) && (args->tail < args->end)) {
        int buffSlot = args->tail%NCCL_STEPS;
#ifdef SHARED_REQ_Q
        // test if send request is posted
        if (reqFifo[buffSlot].state == collReqNone) {
          goto quit;
        }
#endif
        INFO(NCCL_INIT,"Recv proxy : opCount %lx head %lx tail %lx nextTail %p nextTail %lx end %lx nsteps %d llMode %d ==> Posted", args->opCount, args->head, args->tail, nextTail, *nextTail, args->end, args->nsteps, args->llMode);
        args->tail += args->sliceSteps;
        args->idle = 0;
quit:
      }
      if (args->tail > args->head) {
        int done, size;
        int buffSlot = args->head%NCCL_STEPS;
        if (reqFifo[buffSlot].request != NULL) NCCLCHECK(collNetTest((void*)(reqFifo[buffSlot].request), &done, &size));
        if (done) {
          INFO(NCCL_INIT,"Recv proxy : opCount %lx head %lx tail %lx nextTail %p nextTail %lx end %lx nsteps %d llMode %d size %d => Done", args->opCount, args->head, args->tail, nextTail, *nextTail, args->end, args->nsteps, args->llMode, size);
        // cleaning
#ifdef SHARED_REQ_Q
          reqFifo[buffSlot].state = collReqDone;
#endif
          args->head += args->sliceSteps;
          if (args->llMode == 0) {
            if (ptrType == NCCL_PTR_CUDA) collNetFlush(resources->collNetRecvComm, localBuff+buffSlot*stepSize, size);
            *nextTail = args->head;
          }
          args->idle = 0;
        }
      }
      if (args->head == args->end) {
        resources->step = args->end;
        args->state = ncclProxyOpDone;
      }
    }
  }

  if (args->state == ncclProxyOpDone) {
    volatile uint64_t* nextHead = &resources->hostSendMem->head;
    INFO(NCCL_INIT,"Recv proxy : opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d ==> Cleaning", args->opCount, args->head, args->tail, args->end, args->nsteps, args->llMode);
    if (args->llMode && resources->step > resources->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
      resources->step += NCCL_STEPS;
      while (*nextHead < resources->step);
      resources->llLastCleaning = resources->step;
    }
    args->state = ncclProxyOpNone;
  }
  return ncclSuccess;
}

struct ncclCollTransport collNetTransport = {
  "COL",
  collNetCanConnect,
  { collNetSetup, collNetConnect, collNetFree, collNetSendProxy, collNetRecvProxy }
};
