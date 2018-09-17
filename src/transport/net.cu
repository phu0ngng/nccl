/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "transport.h"
#include "nvmlwrap.h"
#include "net.h"
#include "param.h"
#include "nvlink.h"
#include <cuda_runtime.h>
#include <assert.h>

#define NET_MAX_IFS 16

// We encode 3 bits of distance per interface into a ncclTvalue_t (64-bit)
#define NET_BITS_PER_IF 3
#define NET_BITS_PER_IF_MASK ((1<<NET_BITS_PER_IF)-1)
static_assert(sizeof(ncclTvalue_t)*8 >= NET_MAX_IFS*NET_BITS_PER_IF, "NET_MAX_IFS*NET_BITS_PER_IF must fit in a ncclTvalue_t");

struct netInfo {
  int rank;
  int ndev;
  short scores[NET_MAX_IFS];
};

struct netConnectInfo {
  ncclNetHandle_t netHandle;
};

struct netSendResources {
  void* netSendComm;
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
};

struct netRecvResources {
  void* netListenComm;
  void* netRecvComm;
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
};

/* Fill information necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t netFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct netInfo* info = (struct netInfo*)opaqueInfo;
  static_assert(sizeof(struct netInfo) <= sizeof(ncclTinfo_t), "NET Info too large");
  info->rank = rank;
  int *scores;
  NCCLCHECK(ncclNetDevices(&info->ndev, &scores));
  if (info->ndev == 0) {
    WARN("Error : Network returned 0 device");
    return ncclSystemError;
  }
  if (info->ndev > NET_MAX_IFS) info->ndev = NET_MAX_IFS;
  for (int d=0; d<info->ndev; d++) info->scores[d] = scores[d];
  free(scores);
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t netCanConnect(ncclTvalue_t* ret, ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo) {
  ret[0] = 0;
  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  for (int d=0; d<myInfo->ndev; d++) {
    // Keep 3 bits of score info per dev
    ret[0] |= ((myInfo->scores[d] & NET_BITS_PER_IF_MASK)<<(NET_BITS_PER_IF*d));
  }
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


ncclResult_t netGetRings(int nranks, int* groups, int* subgroups, ncclTvalue_t* values, int* nringsRet, int* prev, int* next, int minScore, int* nthreads) {
  int nGroups = groups[nranks-1] + 1;
  int cardUsed[NET_MAX_IFS*nGroups];
  for (int c=0; c<NET_MAX_IFS*nGroups; c++) cardUsed[c] = 0;

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
      for (int card=0; card<NET_MAX_IFS; card++) {
        if (cardUsed[group*NET_MAX_IFS+card] == 1) continue;
        int start = groupBestStart(nranks, groups, group, values, card, minScore);
        // Send from any rank, but best on a different subgroup and close to the NIC also.
        int end = (nranksInGroup == 1) ? start
            : groupBestEnd(nranks, groups, group, subgroups, nsubGroups ? subgroups[start] : -1, start, values, card, minScore);
        //printf("Ring %d, Minscore %d, Card %d, group %d, start = %d, end = %d\n", ring, minScore, card, group, start, end);
        if (start != -1 && end != -1) {
          cardUsed[group*NET_MAX_IFS+card] = 1;
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

int getDev(int ringId, int nDev, short* scores) {
  int maxScore = 0;
  for (int d=0; d<nDev; d++) if (scores[d] > maxScore) maxScore = scores[d];
  int skip = ringId+1;
  while (skip) {
    for (int d=0; d<nDev; d++) {
      if (scores[d] == maxScore) {
        skip--;
        if (skip == 0) return d;
      }
    }
  }
  return 0;
}

NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);

// Enable GDR read by default when:
// 1) user sets it, or
// 2) we are on a NVSwitch platform (i.e. no P2P traffic over PCI-E switch) AND the GPU is Volta
ncclResult_t netUseGdrForReads(int* useGdr) {
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
  int nvlinks = getNumNvlinks(busId);
  *useGdr = nvlinks >= CONNECT_NVSWITCH && ncclCudaCompCap() > 6 ? 1 : 0;
  return ncclSuccess;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t netSendSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int buffSize, int channelId) {
  struct netSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  resources->netDev = getDev(channelId, myInfo->ndev, myInfo->scores);
  int flags, usePtrForReads;
  NCCLCHECK(ncclNetPtrSupport(resources->netDev, &flags));
  NCCLCHECK(netUseGdrForReads(&usePtrForReads));
  resources->cudaSupport = (flags & NCCL_PTR_CUDA) && usePtrForReads ? true : false;

  int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, sendSize));

  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  if (resources->cudaSupport) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, recvSize));

  struct netInfo* peerInfo = (struct netInfo*)peerOpaqueInfo;
  INFO(INIT|NET,"Ring %02d : %d -> %d [send] via NET/%s/%d%s", channelId, myInfo->rank, peerInfo->rank, ncclNetName(), resources->netDev,
      resources->cudaSupport ? "/GDRDMA" : "");
  return ncclSuccess;
}

ncclResult_t netRecvSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int buffSize, int channelId) {
  struct netRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  resources->netDev = getDev(channelId, myInfo->ndev, myInfo->scores);
  int flags;
  NCCLCHECK(ncclNetPtrSupport(resources->netDev, &flags));
  resources->cudaSupport = (flags & NCCL_PTR_CUDA) ? true : false;

  int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, sendSize));

  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  if (resources->cudaSupport) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, recvSize));

  struct netInfo* peerInfo = (struct netInfo*)peerOpaqueInfo;
  INFO(INIT|NET,"Ring %02d : %d -> %d [receive] via NET/%s/%d%s", channelId, peerInfo->rank, myInfo->rank, ncclNetName(), resources->netDev,
      resources->cudaSupport ? "/GDRDMA" : "");
  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(ncclNetListen(resources->netDev, &info->netHandle, &resources->netListenComm));
  return ncclSuccess;
}

ncclResult_t netSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct netSendResources* resources = (struct netSendResources*)send->transportResources;

  // Intermediate buffering on GPU for GPU Direct RDMA, but LL buffer is always on host
  struct ncclRecvMem* recvMem = resources->cudaSupport ? resources->devRecvMem : resources->devHostRecvMem;
  send->conn.buff = recvMem->buff;
  send->conn.llBuff = resources->devHostRecvMem->llBuff;

  // Head/Tail/Opcount/Fifos are always on host
  send->conn.tail = &resources->devHostRecvMem->tail;
  send->conn.opCount = &resources->devHostRecvMem->opCount;
  send->conn.fifo = resources->devHostRecvMem->sizesFifo;
  send->conn.llFifo = resources->devHostRecvMem->llSizesFifo;
  send->conn.head = &resources->devHostSendMem->head;
  send->conn.llHead = &resources->devHostSendMem->llHead;

  // Connect to remote peer
  struct netConnectInfo* info = (struct netConnectInfo*)connectInfo;
  NCCLCHECK(ncclNetConnect(resources->netDev, info->netHandle, &resources->netSendComm));

  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t netRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  // Setup device pointers
  struct netRecvResources* resources = (struct netRecvResources*)recv->transportResources;

  // Intermediate buffering on GPU for GPU Direct RDMA
  struct ncclRecvMem* recvMem = resources->cudaSupport ? resources->devRecvMem : resources->devHostRecvMem;
  recv->conn.buff = recvMem->buff;
  recv->conn.llBuff = recvMem->llBuff;

  // Head/Tail/Opcount are always on host
  recv->conn.tail = &resources->devHostRecvMem->tail;
  recv->conn.opCount = &resources->devHostRecvMem->opCount;
  recv->conn.head = &resources->devHostSendMem->head;
  recv->conn.llHead = &resources->devHostSendMem->llHead;

  // Finish connection establishment from remote peer
  NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
  NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

  return ncclSuccess;
}

ncclResult_t netSendFree(void* transportResources) {
  struct netSendResources* resources = (struct netSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(resources->hostRecvMem));
  if (resources->cudaSupport)
    CUDACHECK(cudaFree(resources->devRecvMem));
  NCCLCHECK(ncclNetCloseSend(resources->netSendComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netRecvFree(void* transportResources) {
  struct netRecvResources* resources = (struct netRecvResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(resources->hostRecvMem));
  if (resources->cudaSupport)
    CUDACHECK(cudaFree(resources->devRecvMem));
  NCCLCHECK(ncclNetCloseRecv(resources->netRecvComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  struct ncclChannel* channel = args->channel;
  struct netSendResources* resources = (struct netSendResources*) (channel->ring.send.transportResources);
  const int llMode = args->llMode;

  volatile uint64_t* prevTail = &resources->hostRecvMem->tail;
  struct ncclSendMem* prevMem = resources->hostSendMem;
  uint64_t* prevHead = llMode ? &prevMem->llHead : &prevMem->head;
  struct ncclRecvMem* localMem = resources->cudaSupport ? resources->devRecvMem : resources->hostRecvMem;
  char* localBuff = llMode ? resources->hostRecvMem->llBuff : localMem->buff;
  int ptrType = resources->cudaSupport ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
  volatile int* sizesFifo = llMode ? resources->hostRecvMem->llSizesFifo : resources->hostRecvMem->sizesFifo;
  int stepSize = llMode ? NCCL_LL_BUFF_SIZE/NCCL_LL_CHUNKS : channel->buffSize/NCCL_STEPS;
  int buffSteps = llMode ? NCCL_LL_CHUNKS : NCCL_STEPS;

  // Round to next multiple of sliceSteps
  resources->step = ROUNDUP(resources->step, args->chunkSteps);

  uint64_t head = llMode ? resources->llStep : resources->step;
  uint64_t tail = llMode ? resources->llStep : resources->step;
  uint64_t end = head + args->nsteps;

  int idle = 0;
  void* requests[buffSteps];

  if (!args->needProxy) goto nextColl;

  TRACE(NET,"opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d", args->opCount, head, tail, end, args->nsteps, llMode);
  TRACE(NET,"opCount %lx stepSize %d stepSize %d ptrType %d", args->opCount, stepSize, stepSize, ptrType);

  while (head < end) {
    idle++;
    if (tail < end && tail < head + buffSteps) {
      if (llMode) {
        int buffSlot = tail%buffSteps;
        int size = sizesFifo[buffSlot];
        if (size != 0) {
          if (size == -1) size = 0;
          uint32_t flag = tail + 1;
          int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
          size = nFifoLines * sizeof(union ncclLLFifoLine);
          union ncclLLFifoLine* lines = (union ncclLLFifoLine*)(localBuff+buffSlot*stepSize);
          for (int i=0; i<nFifoLines; i++) {
            volatile uint32_t *f1 = &lines[i].flag1;
            volatile uint32_t *f2 = &lines[i].flag2;
            while (f1[0] != flag || f2[0] != flag);
          }
          NCCLCHECK(ncclNetIsend(resources->netSendComm, lines, size, ptrType, requests+buffSlot));
          sizesFifo[buffSlot] = 0;
          tail += args->sliceSteps;
          idle = 0;
          // Make sure size is reset to zero before we update the head.
          __sync_synchronize();
        }
      } else if (tail < *prevTail) {
        // Send through network
        int buffSlot = tail%buffSteps;
        NCCLCHECK(ncclNetIsend(resources->netSendComm, localBuff+buffSlot*stepSize, sizesFifo[buffSlot], ptrType, requests+buffSlot));
        tail += args->sliceSteps;
        idle = 0;
      }
    }
    if (head < tail) {
      int done;
      int buffSlot = head%buffSteps;
      NCCLCHECK(ncclNetTest(requests[buffSlot], &done, NULL));
      if (done) {
        head += args->sliceSteps;
        *prevHead = head;
        idle = 0;
      }
    }
    if (idle) transportProxyIdle(idle);
  }

nextColl:
  if (llMode) {
    resources->llStep += args->nsteps;
    // Update in case we didn't need a proxy and jumped directly here
    *prevHead = resources->llStep;
    if (resources->llStep > resources->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
      memset(localBuff, 0, NCCL_LL_BUFF_SIZE);
      resources->llStep += NCCL_LL_CHUNKS;
      *prevHead = resources->llStep;
      resources->llLastCleaning = resources->llStep;
    }
  } else {
    resources->step += args->nsteps;
    // Update in case we didn't need a proxy and jumped directly here
    *prevHead = resources->step;
  }
  return ncclSuccess;
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  struct ncclChannel* channel = args->channel;
  struct netRecvResources* resources = (struct netRecvResources*) (channel->ring.recv.transportResources);
  int llMode = args->llMode;

  volatile uint64_t* nextHead = llMode ? &resources->hostSendMem->llHead : &resources->hostSendMem->head;
  struct ncclRecvMem* localMem = resources->cudaSupport ? resources->devRecvMem : resources->hostRecvMem;
  char* localBuff = llMode ? localMem->llBuff : localMem->buff;
  int ptrType = resources->cudaSupport ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
  uint64_t* nextTail = &resources->hostRecvMem->tail;

  int stepSize = llMode ? NCCL_LL_BUFF_SIZE/NCCL_LL_CHUNKS : channel->buffSize/NCCL_STEPS;
  int sliceSize = stepSize * args->sliceSteps;
  int buffSteps = llMode ? NCCL_LL_CHUNKS : NCCL_STEPS;

  // Round to next multiple of sliceSteps
  resources->step = ROUNDUP(resources->step, args->chunkSteps);

  uint64_t head = llMode ? resources->llStep : resources->step;
  uint64_t tail = llMode ? resources->llStep : resources->step;
  uint64_t end = head + args->nsteps;
//  printf("[R#%ld/%d] %ld / %ld -> %ld | %d / %d\n", args->opCount, channel->id, head, tail, end, buffSteps, args->sliceSteps);

  int idle = 0;
  void* requests[buffSteps];

  if (!args->needProxy) goto nextColl;

  TRACE(NET,"opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d", args->opCount, head, tail, end, args->nsteps, llMode);
  TRACE(NET,"opCount %lx buffSize %d stepSize %d ptrType %d", args->opCount, channel->buffSize, stepSize, ptrType);

  while (head < end) {
    idle++;
    if ((tail < head + buffSteps) && (tail < (*nextHead) + buffSteps) && (tail < end)) {
      int buffSlot = tail%buffSteps;
      NCCLCHECK(ncclNetIrecv(resources->netRecvComm, localBuff+buffSlot*stepSize, sliceSize, ptrType, requests+buffSlot));
      tail += args->sliceSteps;
      idle = 0;
    }
    if (tail > head) {
      int done;
      int buffSlot = head%buffSteps;
      int size;
      NCCLCHECK(ncclNetTest(requests[buffSlot], &done, &size));
      if (done) {
        head += args->sliceSteps;
        if (llMode == 0) {
          if (ptrType == NCCL_PTR_CUDA) ncclNetFlush(resources->netRecvComm, localBuff+buffSlot*stepSize, size);
          *nextTail = head;
        }
        idle = 0;
      }
    }
    if (idle) transportProxyIdle(idle);
  }

nextColl:
  if (llMode) {
    resources->llStep += args->nsteps;
    if (resources->llStep > resources->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
      resources->llStep += NCCL_LL_CHUNKS;
      while (*nextHead < resources->llStep);
      resources->llLastCleaning = resources->llStep;
    }
  } else {
    resources->step += args->nsteps;
  }
  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  netFillInfo,
  netCanConnect,
  netGetRings,
  { netSendSetup, netSendConnect, netSendFree, netSendProxy },
  { netRecvSetup, netRecvConnect, netRecvFree, netRecvProxy }
};
