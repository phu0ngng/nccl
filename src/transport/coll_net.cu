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
#include <cuda_runtime.h>
#include <assert.h>

#define COLL_NET_MAX_IFS 16

// We encode 3 bits of distance per interface into a ncclTvalue_t (64-bit)
#define NET_BITS_PER_IF 3
#define NET_BITS_PER_IF_MASK ((1<<NET_BITS_PER_IF)-1)
static_assert(sizeof(ncclTvalue_t)*8 >= COLL_NET_MAX_IFS*NET_BITS_PER_IF, "COLL_NET_MAX_IFS*NET_BITS_PER_IF must fit in a ncclTvalue_t");

struct collNetConnectInfo {
  collNetHandle_t collNetHandle;
};

// TODO: find correct way to share things between two proxies
#define SHARED_REQ_Q

#ifdef SHARED_REQ_Q
struct reqState {
  volatile void* intmBuff;
  volatile short sendReady;
};
#endif

struct collNetSendResources {
  void* collNetSendComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  int netDev;
  bool cudaSupport;
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
  bool cudaSupport;
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llStep;
  uint64_t llLastCleaning;
#ifdef SHARED_REQ_Q
  struct reqState* reqFifo;
#endif
  uint64_t reqFifoTail;
};

static ncclResult_t collNetDistance(int cudaDev, int dev, short* distance) {
  char* cudaPath = NULL;
  char* nicPath = NULL;
  NCCLCHECK(getCudaPath(cudaDev, &cudaPath));
  NCCLCHECK(collNetPciPath(dev, &nicPath));
  *distance = (nicPath == NULL || cudaPath == NULL) ? PATH_SOC : pciDistance(nicPath, cudaPath);
  if (nicPath) free(nicPath);
  if (cudaPath) free(cudaPath);
  return ncclSuccess;
}

static ncclResult_t collNetDevices(int* ndev, short** distances) {
  NCCLCHECK(collNetDevices(ndev));
  if (*ndev == 0) {
    WARN("Error : Network returned 0 device");
    return ncclSystemError;
  }
  if (*ndev > COLL_NET_MAX_IFS) *ndev = COLL_NET_MAX_IFS;

  *distances = (short*)malloc(*ndev*sizeof(short));
  if (*distances == NULL) return ncclSystemError;

  // Find distance with current GPU
  int cudaDev;
  cudaGetDevice(&cudaDev);
  char line[1024];
  sprintf(line, "CUDA Dev %d, %s NIC distance : ", cudaDev, collNetName());
  for (int d=0; d<*ndev; d++) {
    NCCLCHECK(collNetDistance(cudaDev, d, *distances+d));
    sprintf(line+strlen(line), " %s", pathDists[(*distances)[d]]);
  }
  INFO(NCCL_INIT|NCCL_NET, "%s", line);
  return ncclSuccess;
}

extern ncclTvalue_t getTvalue(short* distances, int ndev);

/* Determine if we can communicate with the peer */
ncclResult_t collNetCanConnect(ncclTvalue_t* ret, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo) {
#if 1
  ret[0] = 1;
#else
  int nDev;
  short* distances;
  NCCLCHECK(collNetDevices(&nDev, &distances));
  ret[0] = getTvalue(distances, nDev);
  free(distances);
#endif
  return ncclSuccess;
}

int getCollNetDev(int ringId) {
  int nDev;
  short* distances;
  NCCLCHECK(collNetDevices(&nDev, &distances));

  int dev = 0;
  int minDistance = PATH_SOC;
  for (int d=0; d<nDev; d++) if (distances[d] < minDistance) minDistance = distances[d];
  int skip = ringId+1;
  while (skip) {
    for (int d=0; d<nDev; d++) {
      if (distances[d] == minDistance) {
        skip--;
        if (skip == 0) { dev = d; goto end; }
      }
    }
  }
end:
  free(distances);
  return dev;
}

extern int64_t ncclParamNetGdrRead();

// Enable GDR read by default when:
// 1) user sets it, or
// 2) we are on a NVSwitch platform (i.e. no P2P traffic over PCI-E switch) AND the GPU is Volta
ncclResult_t collNetUseGdrForReads(int* useGdr) {
  // Get user's GDR READ setting
  int gdrReadParam = ncclParamNetGdrRead();
  if (gdrReadParam >= 0) {
    *useGdr = gdrReadParam;
    return ncclSuccess;
  }

  // Determine whether the GPU has NVLink
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  CUDACHECK(cudaDeviceGetPCIBusId(busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, cudaDev));
  int nvlinks = getNvlinkGpu(busId, NULL);
  *useGdr = nvlinks >= CONNECT_NVSWITCH && ncclCudaCompCap() > 6 ? 1 : 0;
  return ncclSuccess;
}

/* Setup send connector and recv connector, and return connect information for others in the coll communicator to connect to me */
ncclResult_t collNetSetup(struct ncclPeerInfo* myInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, struct ncclConnector* recv, int buffSize, int channelId) {
  int sendSize = sizeof(struct ncclSendMem);
  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  int netDev = getCollNetDev(channelId);

  // send side
  struct collNetSendResources* sendResources;
  NCCLCHECK(ncclCalloc(&sendResources, 1));
  send->transportResources = sendResources;

  sendResources->netDev = netDev;

  int sendFlags, useGdrForReads;
  NCCLCHECK(collNetPtrSupport(sendResources->netDev, &sendFlags));
  NCCLCHECK(collNetUseGdrForReads(&useGdrForReads));
  sendResources->cudaSupport = (sendFlags & NCCL_PTR_CUDA) && useGdrForReads ? true : false;

  NCCLCHECK(ncclCudaHostAlloc((void**)&sendResources->hostSendMem, (void**)&sendResources->devHostSendMem, sendSize));

  if (sendResources->cudaSupport) {
    NCCLCHECK(ncclCudaCalloc((char**)(&sendResources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&sendResources->hostRecvMem, (void**)&sendResources->devHostRecvMem, recvSize));

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [send] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), sendResources->netDev,
      sendResources->cudaSupport ? "/GDRDMA" : "");

  // recv side
  struct collNetRecvResources* recvResources;
  NCCLCHECK(ncclCalloc(&recvResources, 1));
  recv->transportResources = recvResources;

  recvResources->netDev = netDev;

  int recvFlags;
  NCCLCHECK(collNetPtrSupport(recvResources->netDev, &recvFlags));
  recvResources->cudaSupport = (recvFlags & NCCL_PTR_CUDA) ? true : false;

  NCCLCHECK(ncclCudaHostAlloc((void**)&recvResources->hostSendMem, (void**)&recvResources->devHostSendMem, sendSize));

  if (recvResources->cudaSupport) {
    NCCLCHECK(ncclCudaCalloc((char**)(&recvResources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&recvResources->hostRecvMem, (void**)&recvResources->devHostRecvMem, recvSize));

  INFO(NCCL_INIT|NCCL_NET,"Coll %02d : %d [receive] via COLLNET/%s/%d%s", channelId, myInfo->rank, collNetName(), recvResources->netDev,
      recvResources->cudaSupport ? "/GDRDMA" : "");

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
  struct ncclRecvMem* sRecvMem = sendResources->cudaSupport ? sendResources->devRecvMem : sendResources->devHostRecvMem;
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
  struct ncclRecvMem* rRecvMem = recvResources->cudaSupport ? recvResources->devRecvMem : recvResources->devHostRecvMem;
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
  if (sendResources->cudaSupport)
    CUDACHECK(cudaFree(sendResources->devRecvMem));
  NCCLCHECK(collNetCloseColl(sendResources->collNetSendComm));
  free(sendResources->reqFifo);
  free(sendResources);

  // recv side
  struct collNetRecvResources* recvResources = (struct collNetRecvResources*)recvTransportResources;
  NCCLCHECK(ncclCudaHostFree(recvResources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(recvResources->hostRecvMem));
  if (recvResources->cudaSupport)
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
    struct ncclRecvMem* localMem = resources->cudaSupport ? resources->devRecvMem : resources->hostRecvMem;
    union ncclLLFifoLine* llBuff = resources->hostRecvMem->llBuff;
    int ptrType = resources->cudaSupport ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
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
#ifdef SHARED_REQ_Q
          int readySlot = args->tail%NCCL_STEPS;
#endif
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
            NCCLCHECK(collNetIallreduce(resources->collNetSendComm, lines, (void*)(reqFifo[readySlot].intmBuff), count, args->dtype, args->redOp, ptrType, args->requests+buffSlot));
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
          int readySlot = args->tail%NCCL_STEPS;
          if (reqFifo[readySlot].sendReady != 0 || reqFifo[readySlot].intmBuff == NULL) {
            goto end;
          }
#endif
          NCCLCHECK(collNetIallreduce(resources->collNetSendComm, localMem->buff+buffSlot*stepSize, (void*)(reqFifo[readySlot].intmBuff), count, args->dtype, args->redOp, ptrType, args->requests+buffSlot));
          INFO(NCCL_INIT,"Send proxy : opCount %lx head %lx tail %lx prevTail %p prevTail %lx end %lx nsteps %d llMode %d count %d request %p ==> Posted", args->opCount, args->head, args->tail, prevTail, *prevTail, args->end, args->nsteps, args->llMode, count, args->requests[buffSlot]);
          if (args->requests[buffSlot] != NULL) {
            sizesFifo[buffSlot] = -1;
            // Make sure size is reset to zero before we update the head.
            __sync_synchronize();
            args->tail += args->sliceSteps;
            args->idle = 0;
          }
end:
        }
      }
      if (args->head < args->tail) {
        int done, size;
        int buffSlot = args->head%NCCL_STEPS;
        NCCLCHECK(collNetTest(args->requests[buffSlot], &done, &size));
        if (done) {
          INFO(NCCL_INIT,"Send proxy : opCount %lx head %lx tail %lx prevTail %p prevTail %lx end %lx nsteps %d llMode %d size %d request %p ==> Done", args->opCount, args->head, args->tail, prevTail, *prevTail, args->end, args->nsteps, args->llMode, size, args->requests[buffSlot]);
#ifdef SHARED_REQ_Q
          int readySlot = args->head%NCCL_STEPS;
          reqFifo[readySlot].sendReady = 1;
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
    struct ncclRecvMem* localMem = resources->cudaSupport ? resources->devRecvMem : resources->hostRecvMem;
    char* localBuff = args->llMode ? (char*)localMem->llBuff : localMem->buff;
    int ptrType = resources->cudaSupport ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
    uint64_t* nextTail = &resources->hostRecvMem->tail;

    int stepSize = ( args->llMode ? NCCL_LL_BUFF_SIZE : args->channel->buffSize ) / NCCL_STEPS;

#ifdef SHARED_REQ_Q
    struct reqState* reqFifo = resources->reqFifo;
#endif
    uint64_t* reqFifoHead = &args->tail;
    uint64_t* reqFifoTail = &resources->reqFifoTail;

    INFO(NCCL_INIT,"Recv proxy : opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d ==> Start", args->opCount, args->head, args->tail, args->end, args->nsteps, args->llMode);
    TRACE(NET,"opCount %lx buffSize %d stepSize %d ptrType %d", args->opCount, args->channel->buffSize, stepSize, ptrType);

    if (args->head < args->end) {
#ifdef SHARED_REQ_Q
      // enqueue an intermediate buff address
      if (*reqFifoTail < *reqFifoHead + NCCL_STEPS && *reqFifoTail < args->end) {
        int buffSlot = *reqFifoTail%NCCL_STEPS;
        int readyTail = *reqFifoTail%NCCL_STEPS;
        if (reqFifo[readyTail].intmBuff == NULL) {
          reqFifo[readyTail].intmBuff = localBuff+buffSlot*stepSize;
          *reqFifoTail += args->sliceSteps;
        }
      }
#endif
      if ((args->tail < args->head + NCCL_STEPS) && (args->tail < (*nextHead) + NCCL_STEPS) && (args->tail < args->end)) {
        int buffSlot = args->tail%NCCL_STEPS;
#ifdef SHARED_REQ_Q
        int readyHead = *reqFifoHead%NCCL_STEPS;
        // test if send request is complete
        if (reqFifo[readyHead].sendReady == 0) {
          goto quit;
        }
        INFO(NCCL_INIT,"Recv proxy : send request %lx ==> Ready", buffSlot);
#endif
        *(args->requests+buffSlot) = (void*)0xdeadbeef;  //TODO
        INFO(NCCL_INIT,"Recv proxy : opCount %lx head %lx tail %lx nextTail %p nextTail %lx end %lx nsteps %d llMode %d ==> Posted", args->opCount, args->head, args->tail, nextTail, *nextTail, args->end, args->nsteps, args->llMode);
        // cleaning
#ifdef SHARED_REQ_Q
        reqFifo[readyHead].sendReady = 0;
        reqFifo[readyHead].intmBuff = NULL;
#endif
        if (args->requests[buffSlot] != NULL) {
          args->tail += args->sliceSteps;
          args->idle = 0;
        }
quit:
      }
      if (args->tail > args->head) {
        int done;
        int buffSlot = args->head%NCCL_STEPS;
        int size;
        NCCLCHECK(collNetTest(args->requests[buffSlot], &done, &size));
        if (done) {
          INFO(NCCL_INIT,"Recv proxy : opCount %lx head %lx tail %lx nextTail %p nextTail %lx end %lx nsteps %d llMode %d ==> Done", args->opCount, args->head, args->tail, nextTail, *nextTail, args->end, args->nsteps, args->llMode);
          *(args->requests+buffSlot) = NULL;  //TODO
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
