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
};

static ncclResult_t netDevices(int* ndev, int** scores) {
  NCCLCHECK(collNetDevices(ndev));
  if (*ndev == 0) {
    WARN("Error : Network returned 0 device");
    return ncclSystemError;
  }
  if (*ndev > COLL_NET_MAX_IFS) *ndev = COLL_NET_MAX_IFS;
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t collNetCanConnect(ncclTvalue_t* ret, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo) {
  int nDev;
  int* scores;
  NCCLCHECK(netDevices(&nDev, &scores));
  ret[0] = (nDev > 0) ? 1 : 0; //TODO: figure correct value
  free(scores);
  return ncclSuccess;
}

static inline int groupBestStart(int nranks, int* groups, int group, ncclTvalue_t* values, int card, int minScore) {
  int bestRank = -1;
  int bestScore = 0;
  for (int rank=0; rank<nranks; rank++) {
    if (groups[rank] != group) continue;
    for (int i=0; i<nranks; i++) {
      ncclTvalue_t netValue = values[rank*nranks+i];
      if (netValue != 0) {
        ncclTvalue_t score = (netValue>>(NET_BITS_PER_IF*card)) & NET_BITS_PER_IF_MASK;
        if (score >= minScore && score > bestScore) {
          bestScore = score;
          bestRank = rank;
        }
        // All other values should be the same, stop here for this rank
        break;
      }
    }
  }
  return bestRank;
}
static inline int groupBestEnd(int nranks, int* groups, int group, int* subgroups, int startSubGroup, int startRank, ncclTvalue_t* values, int card, int minScore) {
  // For the last rank, we don't need the absolute best score, just to be within minScore.
  for (int rank=nranks-1; rank>=0; rank--) {
    if (groups[rank] != group) continue;
    if (startSubGroup != -1 && startSubGroup == subgroups[rank]) continue;
    if (startRank == rank) continue;
    for (int i=0; i<nranks; i++) {
      ncclTvalue_t netValue = values[rank*nranks+i];
      if (netValue != 0) {
        ncclTvalue_t score = (netValue>>(NET_BITS_PER_IF*card)) & NET_BITS_PER_IF_MASK;
        if (score >= minScore) {
          return rank;
        }
        // All other values should be the same, stop here for this rank
        break;
      }
    }
  }
  return -1;
}


ncclResult_t collNetGetRings(int nranks, int* groups, int* subgroups, ncclTvalue_t* values, int* nringsRet, int* prev, int* next, int minScore, int* nthreads) {
  int nGroups = groups[nranks-1] + 1;
  int cardUsed[COLL_NET_MAX_IFS*nGroups];
  for (int c=0; c<COLL_NET_MAX_IFS*nGroups; c++) cardUsed[c] = 0;

  for (int ring = 0; ring<*nringsRet; ring++) {
    int starts[nGroups];
    int ends[nGroups];
    for (int group = 0; group<nGroups; group++) {
      int nranksInGroup = 0;
      int nsubGroups = 0;
      for (int rank=0; rank<nranks; rank++) if (groups[rank] == group) {
          nranksInGroup++;
          nsubGroups = std::max(subgroups[rank], nsubGroups);
        }
      starts[group] = ends[group] = -1;
      // Receive on the rank closest to the NIC
      for (int card=0; card<COLL_NET_MAX_IFS; card++) {
        if (cardUsed[group*COLL_NET_MAX_IFS+card] == 1) continue;
        int start = groupBestStart(nranks, groups, group, values, card, minScore);
        // Send from any rank, but best on a different subgroup and close to the NIC also.
        int end = (nranksInGroup == 1) ? start
            : groupBestEnd(nranks, groups, group, subgroups, nsubGroups ? subgroups[start] : -1, start, values, card, minScore);
        //printf("Ring %d, Minscore %d, Card %d, group %d, start = %d, end = %d\n", ring, minScore, card, group, start, end);
        if (start != -1 && end != -1) {
          cardUsed[group*COLL_NET_MAX_IFS+card] = 1;
          starts[group] = start;
          ends[group] = end;
          break;
        }
      }
      if (starts[group] == -1 || ends[group] == -1) {
        *nringsRet = ring;
        return ncclSuccess;
      }
    }
    // Link groups together
    for (int group = 0; group<nGroups; group++) {
      int nextGroup = (group+1)%nGroups;
      next[ring*nranks+ends[group]] = starts[nextGroup];
      prev[ring*nranks+starts[nextGroup]] = ends[group];
    }
  }
  return ncclSuccess;
}

int getCollDev(int ringId) {
  int nDev;
  int* scores;
  NCCLCHECK(netDevices(&nDev, &scores));

  int dev = 0;
  int maxScore = 0;
  for (int d=0; d<nDev; d++) if (scores[d] > maxScore) maxScore = scores[d];
  int skip = ringId+1;
  while (skip) {
    for (int d=0; d<nDev; d++) {
      if (scores[d] == maxScore) {
        skip--;
        if (skip == 0) { dev = d; goto end; }
      }
    }
  }
end:
  free(scores);
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

  // send side
  struct collNetSendResources* sendResources;
  NCCLCHECK(ncclCalloc(&sendResources, 1));
  send->transportResources = sendResources;

  sendResources->netDev = getCollDev(channelId);

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

  recvResources->netDev = getCollDev(channelId);

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
  NCCLCHECK(collNetConnect(sendResources->netDev, (void**)handlePtrs, nranks, recvResources->netListenComm, &sendResources->collNetSendComm));

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
  const int llMode = args->llMode;

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

  // Round to next multiple of sliceSteps
  resources->step = ROUNDUP(resources->step, args->chunkSteps);

  uint64_t head = resources->step;
  uint64_t tail = resources->step;
  uint64_t end = head + args->nsteps;

  int idle = 0;
  void* requests[NCCL_STEPS];

  INFO(NCCL_INIT,"Send proxy : opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d ==> Start", args->opCount, head, tail, end, args->nsteps, llMode);
  TRACE(NET,"opCount %lx stepSize %d stepSize %d ptrType %d", args->opCount, stepSize, stepSize, ptrType);

  while (head < end) {
    idle++;
    if (tail < end && tail < head + NCCL_STEPS) {
      if (llMode) {
        int buffSlot = tail%NCCL_STEPS;
#ifdef SHARED_REQ_Q
        int readySlot = tail%NCCL_STEPS;
#endif
        int size = sizesFifo[buffSlot];
        if (size != -1) {
          uint32_t flag = tail + 1;
          int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
          size = nFifoLines * sizeof(union ncclLLFifoLine);
          union ncclLLFifoLine* lines = llBuff+buffSlot*NCCL_LL_SLICE_LINES;
          for (int i=0; i<nFifoLines; i++) {
            volatile uint32_t *f1 = &lines[i].flag1;
            volatile uint32_t *f2 = &lines[i].flag2;
            while (f1[0] != flag || f2[0] != flag);
          }
          // Some reduce / all-reduce call here
          NCCLCHECK(collNetIsend(resources->collNetSendComm, lines, (void*)(reqFifo[readySlot].intmBuff), size, ptrType, requests+buffSlot));
          if (requests[buffSlot] != NULL) {
            sizesFifo[buffSlot] = -1;
            // Make sure size is reset to zero before we update the head.
            __sync_synchronize();
            tail += args->sliceSteps;
            idle = 0;
          }
        }
      } else if (tail < *prevTail) {
        // Send through network
        int buffSlot = tail%NCCL_STEPS;
#ifdef SHARED_REQ_Q
        int readySlot = tail%NCCL_STEPS;
        // TODO: currently we just wait until the recv is done
        while(reqFifo[readySlot].sendReady != 0 || reqFifo[readySlot].intmBuff == NULL);
#endif
        // Some reduce / all-reduce call here
        NCCLCHECK(collNetIsend(resources->collNetSendComm, localMem->buff+buffSlot*stepSize, (void*)(reqFifo[readySlot].intmBuff), sizesFifo[buffSlot], ptrType, requests+buffSlot));
        INFO(NCCL_INIT,"Send proxy : opCount %lx head %lx tail %lx prevTail %p prevTail %lx end %lx nsteps %d llMode %d ==> Posted", args->opCount, head, tail, prevTail, *prevTail, end, args->nsteps, llMode);
        if (requests[buffSlot] != NULL) {
          sizesFifo[buffSlot] = -1;
          // Make sure size is reset to zero before we update the head.
          __sync_synchronize();
          tail += args->sliceSteps;
          idle = 0;
        }
      }
    }
    if (head < tail) {
      int done;
      int buffSlot = head%NCCL_STEPS;
      NCCLCHECK(collNetTest(requests[buffSlot], &done, NULL));
      if (done) {
#ifdef SHARED_REQ_Q
        int readySlot = head%NCCL_STEPS;
        reqFifo[readySlot].sendReady = 1;
#endif
        head += args->sliceSteps;
        *prevHead = head;
        idle = 0;
        INFO(NCCL_INIT,"Send proxy : opCount %lx head %lx tail %lx prevTail %p prevTail %lx end %lx nsteps %d llMode %d ==> Done", args->opCount, head, tail, prevTail, *prevTail, end, args->nsteps, llMode);
      }
    }
    if (idle) transportProxyIdle(idle);
  }
  resources->step = end;

  INFO(NCCL_INIT,"Send proxy : opCount %lx head %lx tail %lx prevTail %p prevTail %lx end %lx nsteps %d llMode %d ==> Cleaning", args->opCount, head, tail, prevTail, *prevTail, end, args->nsteps, llMode);
  if (llMode && resources->step > resources->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
    for (int i=0; i< NCCL_LL_BUFF_LINES; i++) llBuff[i].flag1 = llBuff[i].flag2 = resources->step;
    resources->step += NCCL_STEPS;
    *prevHead = resources->step;
    resources->llLastCleaning = resources->step;
  }
  return ncclSuccess;
}

ncclResult_t collNetRecvProxy(struct ncclProxyArgs* args) {
  struct collNetRecvResources* resources = (struct collNetRecvResources*) (args->connector->transportResources);
  int llMode = args->llMode;

  volatile uint64_t* nextHead = &resources->hostSendMem->head;
  struct ncclRecvMem* localMem = resources->cudaSupport ? resources->devRecvMem : resources->hostRecvMem;
  char* localBuff = llMode ? (char*)localMem->llBuff : localMem->buff;
  int ptrType = resources->cudaSupport ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
  uint64_t* nextTail = &resources->hostRecvMem->tail;

  int stepSize = ( llMode ? NCCL_LL_BUFF_SIZE : args->channel->buffSize ) / NCCL_STEPS;
  int sliceSize = stepSize * args->sliceSteps;

#ifdef SHARED_REQ_Q
  struct reqState* reqFifo = resources->reqFifo;
#endif

  // Round to next multiple of sliceSteps
  resources->step = ROUNDUP(resources->step, args->chunkSteps);

  uint64_t head = resources->step;
  uint64_t tail = resources->step;
  uint64_t* reqFifoHead = &tail;
  uint64_t reqFifoTail = resources->step;
  uint64_t end = head + args->nsteps;

  int idle = 0;
  void* requests[NCCL_STEPS];

  INFO(NCCL_INIT,"Recv proxy : opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d ==> Start", args->opCount, head, tail, end, args->nsteps, llMode);
  TRACE(NET,"opCount %lx buffSize %d stepSize %d ptrType %d", args->opCount, args->channel->buffSize, stepSize, ptrType);

  while (head < end) {
    idle++;
#ifdef SHARED_REQ_Q
    // enqueue an intermediate buff address
    if (reqFifoTail < *reqFifoHead + NCCL_STEPS && reqFifoTail < end) {
      int buffSlot = reqFifoTail%NCCL_STEPS;
      int readyTail = reqFifoTail%NCCL_STEPS;
      reqFifo[readyTail].intmBuff = localBuff+buffSlot*stepSize;
      reqFifoTail += args->sliceSteps;
    }
#endif
    if ((tail < head + NCCL_STEPS) && (tail < (*nextHead) + NCCL_STEPS) && (tail < end)) {
      int buffSlot = tail%NCCL_STEPS;
#ifdef SHARED_REQ_Q
      int readyHead = *reqFifoHead%NCCL_STEPS;
      // test if send request is complete
      while(reqFifo[readyHead].sendReady == 0);
      INFO(NCCL_INIT,"Recv proxy : send request %lx ==> Ready", buffSlot);
#endif
      // broadcast or wait for all-reduce to complete
      NCCLCHECK(collNetIrecv(resources->collNetRecvComm, /*localBuff+buffSlot*stepSize*/ (void*)(reqFifo[readyHead].intmBuff), sliceSize, ptrType, requests+buffSlot));
      INFO(NCCL_INIT,"Recv proxy : opCount %lx head %lx tail %lx nextTail %p nextTail %lx end %lx nsteps %d llMode %d ==> Posted", args->opCount, head, tail, nextTail, *nextTail, end, args->nsteps, llMode);
      // cleaning
#ifdef SHARED_REQ_Q
      reqFifo[readyHead].sendReady = 0;
      reqFifo[readyHead].intmBuff = NULL;
#endif
      if (requests[buffSlot] != NULL) {
        tail += args->sliceSteps;
        idle = 0;
      }
    }
    if (tail > head) {
      int done;
      int buffSlot = head%NCCL_STEPS;
      int size;
      NCCLCHECK(collNetTest(requests[buffSlot], &done, &size));
      if (done) {
        INFO(NCCL_INIT,"Recv proxy : opCount %lx head %lx tail %lx nextTail %p nextTail %lx end %lx nsteps %d llMode %d ==> Done", args->opCount, head, tail, nextTail, *nextTail, end, args->nsteps, llMode);
        head += args->sliceSteps;
        if (llMode == 0) {
          if (ptrType == NCCL_PTR_CUDA) collNetFlush(resources->collNetRecvComm, localBuff+buffSlot*stepSize, size);
          *nextTail = head;
        }
        idle = 0;
      }
    }
    if (idle) transportProxyIdle(idle);
  }
  resources->step = end;

  INFO(NCCL_INIT,"Recv proxy : opCount %lx head %lx tail %lx nextTail %p nextTail %lx end %lx nsteps %d llMode %d ==> Cleaning", args->opCount, head, tail, nextTail, *nextTail, end, args->nsteps, llMode);
  if (llMode && resources->step > resources->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
    resources->step += NCCL_STEPS;
    while (*nextHead < resources->step);
    resources->llLastCleaning = resources->step;
  }
  return ncclSuccess;
}

struct ncclCollTransport collNetTransport = {
  "COL",
  collNetCanConnect,
  { collNetSetup, collNetConnect, collNetFree, collNetSendProxy, collNetRecvProxy }
};
