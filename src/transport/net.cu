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
static ncclTvalue_t getTvalue(short* distances, int ndev) {
  ncclTvalue_t tvalue = 0;
  for (int d=0; d<ndev; d++) {
    int score = 1 + PATH_SOC - distances[d];
    // Keep 3 bits of score info per dev
    tvalue |= ((score & NET_BITS_PER_IF_MASK)<<(NET_BITS_PER_IF*d));
  }
  return tvalue;
}

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
  int useGdr;
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
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
  int useGdr;
  struct ncclRecvMem* devRecvMem;
  uint64_t step;
  uint64_t llLastCleaning;
};

static ncclResult_t netDistance(int cudaDev, int dev, short* distance) {
  char* cudaPath;
  char* nicPath;
  NCCLCHECK(getCudaPath(cudaDev, &cudaPath));
  NCCLCHECK(ncclNetPciPath(dev, &nicPath));
  *distance = (nicPath == NULL || cudaPath == NULL) ? PATH_SOC : pciDistance(nicPath, cudaPath);
  free(nicPath);
  free(cudaPath);
  return ncclSuccess;
}

static ncclResult_t netDevices(int* ndev, short** distances) {
  NCCLCHECK(ncclNetDevices(ndev));
  if (*ndev == 0) {
    WARN("Error : Network returned 0 device");
    return ncclSystemError;
  }
  if (*ndev > NET_MAX_IFS) *ndev = NET_MAX_IFS;

  *distances = (short*)malloc(*ndev*sizeof(short));
  // Find distance with current GPU
  int cudaDev;
  cudaGetDevice(&cudaDev);
  char line[1024];
  sprintf(line, "CUDA Dev %d, %s NIC distance : ", cudaDev, ncclNetName());
  for (int d=0; d<*ndev; d++) {
    NCCLCHECK(netDistance(cudaDev, d, *distances+d));
    sprintf(line+strlen(line), " %s", pathDists[*distances[d]]);
  }
  INFO(NCCL_INIT|NCCL_NET, "%s", line);
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t netCanConnect(ncclTvalue_t* ret, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo) {
  int nDev;
  short* distances;
  NCCLCHECK(netDevices(&nDev, &distances));
  ret[0] = getTvalue(distances, nDev);
  free(distances);
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

int getDev(int ringId) {
  int nDev;
  short* distances;
  NCCLCHECK(netDevices(&nDev, &distances));

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

NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);
NCCL_PARAM(NetGdrLevel, "NET_GDR_LEVEL", PATH_PHB);

static ncclResult_t netGetGdrSupport(int dev, int read, int* useGdr) {
  *useGdr = 0;

  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (read) { // For reads (sends) only enable under certain conditions
    int gdrReadParam = ncclParamNetGdrRead();
    if (gdrReadParam == 0) return ncclSuccess;
    else if (gdrReadParam < 0) { // default : enable only on DGX2
      char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      CUDACHECK(cudaDeviceGetPCIBusId(busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, cudaDev));
      int nvlinks = getNvlinkGpu(busId, NULL);
      if (nvlinks < CONNECT_NVSWITCH || ncclCudaCompCap() < 7) return ncclSuccess;
    }
  }

  // Check if we are close enough that it makes sense to enable GDR
  int netGdrLevel = ncclParamNetGdrLevel();
  short distance;
  NCCLCHECK(netDistance(cudaDev, dev, &distance));
  if (distance >= netGdrLevel) {
    INFO(NCCL_INIT|NCCL_NET,"NET/%s : GPU Direct RDMA Disabled for GPU %d / HCA %d (distance %d >= %d)", ncclNetName(), cudaDev, dev, distance, netGdrLevel);
    return ncclSuccess;
  }

  // Finally, check if the NIC supports it
  int flags;
  NCCLCHECK(ncclNetPtrSupport(dev, &flags));
  if (flags & NCCL_PTR_CUDA == 0) return ncclSuccess;
  *useGdr = 1;
  INFO(NCCL_INIT|NCCL_NET,"NET/%s : GPU Direct RDMA Enabled for GPU %d / HCA %d (distance %d >= %d), read %d", ncclNetName(), cudaDev, dev, distance, netGdrLevel, read);
  return ncclSuccess;
}

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t netSendSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int buffSize, int channelId) {
  struct netSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  resources->netDev = getDev(channelId);
  NCCLCHECK(netGetGdrSupport(resources->netDev, 1, &resources->useGdr));

  int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, sendSize));

  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, recvSize));

  INFO(NCCL_INIT|NCCL_NET,"Ring %02d : %d -> %d [send] via NET/%s/%d%s", channelId, myInfo->rank, peerInfo->rank, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  return ncclSuccess;
}

ncclResult_t netRecvSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int buffSize, int channelId) {
  struct netRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  resources->netDev = getDev(channelId);
  NCCLCHECK(netGetGdrSupport(resources->netDev, 0, &resources->useGdr));

  int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, sendSize));

  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  if (resources->useGdr) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devRecvMem), recvSize));
  }
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, recvSize));

  INFO(NCCL_INIT|NCCL_NET,"Ring %02d : %d -> %d [receive] via NET/%s/%d%s", channelId, peerInfo->rank, myInfo->rank, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "");
  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(ncclNetListen(resources->netDev, &info->netHandle, &resources->netListenComm));
  return ncclSuccess;
}

ncclResult_t netSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct netSendResources* resources = (struct netSendResources*)send->transportResources;

  // Intermediate buffering on GPU for GPU Direct RDMA, but LL buffer is always on host
  struct ncclRecvMem* recvMem = resources->useGdr ? resources->devRecvMem : resources->devHostRecvMem;
  send->conn.buff = recvMem->buff;
  send->conn.llBuff = resources->devHostRecvMem->llBuff;

  // Head/Tail/Opcount/Fifos are always on host
  send->conn.tail = &resources->devHostRecvMem->tail;
  send->conn.opCount = &resources->devHostRecvMem->opCount;
  send->conn.fifo = resources->devHostRecvMem->sizesFifo;
  send->conn.head = &resources->devHostSendMem->head;
  for (int i=0; i<NCCL_STEPS; i++) send->conn.fifo[i] = -1;

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
  struct ncclRecvMem* recvMem = resources->useGdr ? resources->devRecvMem : resources->devHostRecvMem;
  recv->conn.buff = recvMem->buff;
  recv->conn.llBuff = recvMem->llBuff;

  // Head/Tail/Opcount are always on host
  recv->conn.tail = &resources->devHostRecvMem->tail;
  recv->conn.opCount = &resources->devHostRecvMem->opCount;
  recv->conn.head = &resources->devHostSendMem->head;

  // Finish connection establishment from remote peer
  NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
  NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

  return ncclSuccess;
}

ncclResult_t netSendFree(void* transportResources) {
  struct netSendResources* resources = (struct netSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(resources->hostRecvMem));
  if (resources->useGdr)
    CUDACHECK(cudaFree(resources->devRecvMem));
  NCCLCHECK(ncclNetCloseSend(resources->netSendComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netRecvFree(void* transportResources) {
  struct netRecvResources* resources = (struct netRecvResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->hostSendMem));
  NCCLCHECK(ncclCudaHostFree(resources->hostRecvMem));
  if (resources->useGdr)
    CUDACHECK(cudaFree(resources->devRecvMem));
  NCCLCHECK(ncclNetCloseRecv(resources->netRecvComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  struct netSendResources* resources = (struct netSendResources*) (args->connector->transportResources);
  const int llMode = args->llMode;

  volatile uint64_t* prevTail = &resources->hostRecvMem->tail;
  struct ncclSendMem* prevMem = resources->hostSendMem;
  uint64_t* prevHead = &prevMem->head;
  struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
  union ncclLLFifoLine* llBuff = resources->hostRecvMem->llBuff;
  int ptrType = resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
  volatile int* sizesFifo = resources->hostRecvMem->sizesFifo;
  int stepSize = args->channel->buffSize/NCCL_STEPS;

  // Round to next multiple of sliceSteps
  resources->step = ROUNDUP(resources->step, args->chunkSteps);

  uint64_t head = resources->step;
  uint64_t tail = resources->step;
  uint64_t end = head + args->nsteps;

  int idle = 0;
  void* requests[NCCL_STEPS];

  TRACE(NCCL_NET,"opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d", args->opCount, head, tail, end, args->nsteps, llMode);
  TRACE(NCCL_NET,"opCount %lx stepSize %d stepSize %d ptrType %d", args->opCount, stepSize, stepSize, ptrType);

  while (head < end) {
    idle++;
    if (tail < end && tail < head + NCCL_STEPS) {
      if (llMode) {
        int buffSlot = tail%NCCL_STEPS;
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
          NCCLCHECK(ncclNetIsend(resources->netSendComm, lines, size, ptrType, requests+buffSlot));
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
        NCCLCHECK(ncclNetIsend(resources->netSendComm, localMem->buff+buffSlot*stepSize, sizesFifo[buffSlot], ptrType, requests+buffSlot));
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
      NCCLCHECK(ncclNetTest(requests[buffSlot], &done, NULL));
      if (done) {
        head += args->sliceSteps;
        *prevHead = head;
        idle = 0;
      }
    }
    if (idle) transportProxyIdle(idle);
  }
  resources->step = end;

  if (llMode && resources->step > resources->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
    for (int i=0; i< NCCL_LL_BUFF_LINES; i++) llBuff[i].flag1 = llBuff[i].flag2 = resources->step;
    resources->step += NCCL_STEPS;
    *prevHead = resources->step;
    resources->llLastCleaning = resources->step;
  }
  return ncclSuccess;
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  struct netRecvResources* resources = (struct netRecvResources*) (args->connector->transportResources);
  int llMode = args->llMode;

  volatile uint64_t* nextHead = &resources->hostSendMem->head;
  struct ncclRecvMem* localMem = resources->useGdr ? resources->devRecvMem : resources->hostRecvMem;
  char* localBuff = llMode ? (char*)localMem->llBuff : localMem->buff;
  int ptrType = resources->useGdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
  uint64_t* nextTail = &resources->hostRecvMem->tail;

  int stepSize = ( llMode ? NCCL_LL_BUFF_SIZE : args->channel->buffSize ) / NCCL_STEPS;
  int sliceSize = stepSize * args->sliceSteps;

  // Round to next multiple of sliceSteps
  resources->step = ROUNDUP(resources->step, args->chunkSteps);

  uint64_t head = resources->step;
  uint64_t tail = resources->step;
  uint64_t end = head + args->nsteps;

  int idle = 0;
  void* requests[NCCL_STEPS];

  TRACE(NCCL_NET,"opCount %lx head %lx tail %lx end %lx nsteps %d llMode %d", args->opCount, head, tail, end, args->nsteps, llMode);
  TRACE(NCCL_NET,"opCount %lx buffSize %d stepSize %d ptrType %d", args->opCount, args->channel->buffSize, stepSize, ptrType);

  while (head < end) {
    idle++;
    if ((tail < head + NCCL_STEPS) && (tail < (*nextHead) + NCCL_STEPS) && (tail < end)) {
      int buffSlot = tail%NCCL_STEPS;
      NCCLCHECK(ncclNetIrecv(resources->netRecvComm, localBuff+buffSlot*stepSize, sliceSize, ptrType, requests+buffSlot));
      if (requests[buffSlot] != NULL) {
        tail += args->sliceSteps;
        idle = 0;
      }
    }
    if (tail > head) {
      int done;
      int buffSlot = head%NCCL_STEPS;
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
  resources->step = end;

  if (llMode && resources->step > resources->llLastCleaning + NCCL_LL_CLEAN_FREQ) {
    resources->step += NCCL_STEPS;
    while (*nextHead < resources->step);
    resources->llLastCleaning = resources->step;
  }
  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  netCanConnect,
  netGetRings,
  { netSendSetup, netSendConnect, netSendFree, netSendProxy },
  { netRecvSetup, netRecvConnect, netRecvFree, netRecvProxy }
};
