/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "net.h"
#include "graph.h"
#include "collectives.h"
#include "gdrwrap.h"
#include "bootstrap.h"

struct netConnectInfo {
  ncclNetHandle_t netHandle;
};

#define LOC_HOSTMEM 0
#define LOC_DEVMEM  1
#define LOC_COUNT   2

struct netConnectMap {
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

struct netSendResources {
  struct netConnectMap map;
  void* netSendComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  // GDRCOPY support
  void* gdrDesc;

  int rank;
  int netDev;
  int useGdr;
  int useGdc;
  int shared;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
};

struct netRecvResources {
  struct netConnectMap map;
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  // GDRCOPY support
  void* gdrDesc;

  int rank;
  int netDev;
  int useGdr;
  int useGdc;
  int shared;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
};

/* Determine if two peers can communicate with NET */
static ncclResult_t netCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

NCCL_PARAM(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);

struct netSetupReq {
  int rank;
  int shared;
  int netDev;
  int useGdr;
};

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
static ncclResult_t netSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct netSetupReq req;

  send->conn.shared = req.shared = ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : graph ? 0 : 1;
  send->proxyAppendPtr = send->conn.shared ? comm->proxyState.sharedBuffs.proxyAppend+2*channelId+1 : &send->proxyAppend;

  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &req.netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 1, &req.useGdr));
  req.rank = myInfo->rank;

  send->conn.direct |= req.useGdr ? NCCL_DIRECT_NIC : 0;

  NCCLCHECK(bootstrapProxyConnect(comm->bootstrap, TRANSPORT_NET, 1, myInfo->rank, &send->fd));
  NCCLCHECK(socketSend(send->fd, &req, sizeof(req)));
  printf("Send Setup / Send req\n");

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d%s%s", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, ncclNetName(), req.netDev,
      req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  return ncclSuccess;
}

// GDRCOPY support: TAIL_ENABLE When enabled locates the RX proxy tail in CUDA memory
NCCL_PARAM(GdrCopySyncEnable, "GDRCOPY_SYNC_ENABLE", 1);
// GDRCOPY support: FLUSH_ENABLE When enabled uses a PCI-E read to flush GDRDMA buffers
NCCL_PARAM(GdrCopyFlushEnable, "GDRCOPY_FLUSH_ENABLE", 0);

static ncclResult_t netRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct netSetupReq req;

  recv->conn.shared = req.shared = ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : graph ? 0 : 1;
  recv->proxyAppendPtr = recv->conn.shared ? comm->proxyState.sharedBuffs.proxyAppend+2*channelId : &recv->proxyAppend;

  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &req.netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 0, &req.useGdr));
  req.rank = myInfo->rank;

  NCCLCHECK(bootstrapProxyConnect(comm->bootstrap, TRANSPORT_NET, 0, myInfo->rank, &recv->fd));
  NCCLCHECK(socketSend(recv->fd, &req, sizeof(req)));

  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(socketRecv(recv->fd, &info->netHandle, sizeof(ncclNetHandle_t)));

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [receive] via NET/%s/%d%s%s", channelId, peerInfo->rank, peerInfo->busId, myInfo->rank, myInfo->busId, ncclNetName(), req.netDev,
      req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  return ncclSuccess;
}

static ncclResult_t netMapShm(char** cpuPtr, char** gpuPtr, char* shmPath, int shmSize) {
  int fd = open(shmPath, O_RDWR);
  *cpuPtr = (char*)mmap(NULL, shmSize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  CUDACHECK(cudaHostRegister(*cpuPtr, shmSize, cudaHostRegisterMapped));
  CUDACHECK(cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0));
  unlink(shmPath);
  close(fd);
  return ncclSuccess;
}
static ncclResult_t netCreateShm(int shmSize, char** cpuPtr, char* shmPath) {
  sprintf(shmPath, "/dev/shm/nccl-XXXXXX");
  int fd = mkstemp(shmPath);
  if (fd == -1) {
    WARN("Error: could not create shared memory in /dev/shm");
    return ncclSystemError;
  }
  if (ftruncate(fd, shmSize) != 0) {
    WARN("Error: failed to extend %s to %d bytes", shmPath, shmSize);
    close(fd);
    return ncclSystemError;
  }
  *cpuPtr = (char*)mmap(NULL, shmSize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  return ncclSuccess;
}

static ncclResult_t netSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct netConnectInfo* info = (struct netConnectInfo*)connectInfo;
  NCCLCHECK(socketSend(send->fd, &info->netHandle, sizeof(ncclNetHandle_t)));
  printf("Send connect / Sent net handle\n");

  struct netConnectMap map;
  NCCLCHECK(socketRecv(send->fd, &map, sizeof(struct netConnectMap)));
  printf("Send Connect / Received map\n");

  if (map.sameProcess == 0) {
    NCCLCHECK(netMapShm(&map.hostMem.cpuPtr, &map.hostMem.gpuPtr, map.hostMem.shmPath, map.hostMem.shmSize));
    if (map.devMem.cpuPtr) {
      CUDACHECK(cudaIpcOpenMemHandle((void**)&map.devMem.gpuPtr, map.devMem.ipc, cudaIpcMemLazyEnablePeerAccess));
    }
  }

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(&map, gpu, sendMem);
  send->conn.head = &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(&map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;
  send->conn.sizesFifo = recvMem->sizesFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  send->conn.ptrsFifo = map.shared ? recvMem->ptrsFifo : NULL;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(&map, gpu, buffs[p]);
  return ncclSuccess;
}

/* Connect to this peer */
static ncclResult_t netRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  NCCLCHECK(socketSend(recv->fd, &rank, sizeof(rank)));

  struct netConnectMap map;
  NCCLCHECK(socketRecv(recv->fd, &map, sizeof(struct netConnectMap)));

  if (map.sameProcess == 0) {
    NCCLCHECK(netMapShm(&map.hostMem.cpuPtr, &map.hostMem.gpuPtr, map.hostMem.shmPath, map.hostMem.shmSize));
    if (map.devMem.cpuPtr) {
      CUDACHECK(cudaIpcOpenMemHandle((void**)&map.devMem.gpuPtr, map.devMem.ipc, cudaIpcMemLazyEnablePeerAccess));
    }
  }

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(&map, gpu, sendMem);
  recv->conn.head = &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(&map, gpu, recvMem);
  recv->conn.tail = &recvMem->tail;
  recv->conn.sizesFifo = recvMem->sizesFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  recv->conn.ptrsFifo = map.shared ? recvMem->ptrsFifo : NULL;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    recv->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(&map, gpu, buffs[p]);
  return ncclSuccess;
}

static ncclResult_t netSendFree(struct ncclConnector* send) {
  close(send->fd);
  return ncclSuccess;
}

static ncclResult_t netRecvFree(struct ncclConnector* recv) {
  close(recv->fd);
  return ncclSuccess;
}

static ncclResult_t netSendProxyCall(int fd, void** state, struct ncclComm* comm) {
  if (*state == NULL) { // Setup
    struct netSetupReq req;
    NCCLCHECK(socketRecv(fd, &req, sizeof(req)));

    struct netSendResources* resources;
    NCCLCHECK(ncclCalloc(&resources, 1));
    *state = resources;

    resources->rank = req.rank;
    resources->netDev = req.netDev;
    resources->shared = req.shared;
    resources->useGdr = req.useGdr;
  } else { // Connect
    struct netSendResources* resources = (struct netSendResources*)(*state);
    ncclNetHandle_t netHandle;
    NCCLCHECK(socketRecv(fd, &netHandle, sizeof(ncclNetHandle_t)));

    // Connect to remote peer
    NCCLCHECK(ncclNetConnect(resources->netDev, netHandle, &resources->netSendComm));

    // Create structures
    struct netConnectMap* map = &resources->map;
    map->sameProcess =
        comm->peerInfo[resources->rank].pidHash == comm->peerInfo[comm->rank].pidHash ? 1 : 0;
    map->useGdc = map->sameProcess && ncclGdrCopy != NULL && ncclParamGdrCopySyncEnable();
    map->shared = resources->shared;

    int hostMemSize = 0, devMemSize = 0;

    if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        NCCL_NET_MAP_ADD_POINTER(map, p!= NCCL_PROTO_LL && resources->useGdr, comm->buffSizes[p], buffs[p]);
        resources->buffSizes[p] = comm->buffSizes[p];
      }
    } else {
      // Get shared buffers
      NCCLCHECK(ncclProxySharedBuffersInitP2p(comm, resources->useGdr, resources->netDev, resources->buffSizes+NCCL_PROTO_SIMPLE, resources->buffers+NCCL_PROTO_SIMPLE));
    }

    NCCL_NET_MAP_ADD_POINTER(map, map->useGdc, sizeof(struct ncclSendMem), sendMem);
    NCCL_NET_MAP_ADD_POINTER(map, 0, sizeof(struct ncclRecvMem), recvMem);

    if (devMemSize) {
      if (!map->sameProcess) {
        ALIGN_SIZE(devMemSize, CUDA_IPC_MIN);
      }
      if (map->useGdc) {
        NCCLCHECK(ncclGdrCudaCalloc(&map->devMem.cpuPtr, &map->devMem.gpuPtr, devMemSize, &resources->gdrDesc));
      } else {
        NCCLCHECK(ncclCudaCalloc(&map->devMem.gpuPtr, devMemSize));
      }
      if (!map->sameProcess) {
        CUDACHECK(cudaIpcGetMemHandle(&map->devMem.ipc, &map->devMem.gpuPtr));
      }
    }

    if (map->sameProcess) {
      NCCLCHECK(ncclCudaHostCalloc(&map->hostMem.cpuPtr, hostMemSize));
      map->hostMem.gpuPtr = map->hostMem.cpuPtr;
    } else {
      NCCLCHECK(netCreateShm(hostMemSize, &map->hostMem.cpuPtr, map->hostMem.shmPath));
      map->hostMem.shmSize = hostMemSize;
    }

    resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
    resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);
      if (resources->buffers[p])
        NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
    }

    NCCLCHECK(socketSend(fd, map, sizeof(struct netConnectMap)));
    printf("Send proxy / Sent map\n");
  }
  return ncclSuccess;
}

static ncclResult_t netRecvProxyCall(int fd, void** state, struct ncclComm* comm) {
  if (*state == NULL) { // Setup
    struct netSetupReq req;
    NCCLCHECK(socketRecv(fd, &req, sizeof(req)));

    struct netRecvResources* resources;
    NCCLCHECK(ncclCalloc(&resources, 1));
    *state = resources;

    resources->rank = req.rank;
    resources->netDev = req.netDev;
    resources->shared = req.shared;
    resources->useGdr = req.useGdr;

    ncclNetHandle_t netHandle;
    NCCLCHECK(ncclNetListen(req.netDev, &netHandle, &resources->netListenComm));
    NCCLCHECK(socketSend(fd, &netHandle, sizeof(ncclNetHandle_t)));
  } else { // Connect
    struct netRecvResources* resources = (struct netRecvResources*)(*state);

    int rank;
    NCCLCHECK(socketRecv(fd, &rank, sizeof(rank)));

    // Finish connection establishment from remote peer
    NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
    NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

    // Create structures
    struct netConnectMap* map = &resources->map;
    map->sameProcess =
        comm->peerInfo[resources->rank].pidHash == comm->peerInfo[comm->rank].pidHash ? 1 : 0;
    map->useGdc = map->sameProcess && ncclGdrCopy != NULL && ncclParamGdrCopySyncEnable();
    map->shared = resources->shared;

    int hostMemSize = 0, devMemSize = 0;
    
    if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        NCCL_NET_MAP_ADD_POINTER(map, resources->useGdr, comm->buffSizes[p], buffs[p]);
        resources->buffSizes[p] = comm->buffSizes[p];
      }
    } else {
      // Get shared buffers
      NCCLCHECK(ncclProxySharedBuffersInitP2p(comm, resources->useGdr, resources->netDev, resources->buffSizes+NCCL_PROTO_SIMPLE, resources->buffers+NCCL_PROTO_SIMPLE));
    }

    NCCL_NET_MAP_ADD_POINTER(map, 0, sizeof(struct ncclSendMem), sendMem);
    NCCL_NET_MAP_ADD_POINTER(map, map->useGdc, sizeof(struct ncclRecvMem), recvMem);

    if (devMemSize) {
      if (!map->sameProcess) {
        ALIGN_SIZE(devMemSize, CUDA_IPC_MIN);
      }
      if (map->useGdc) {
        NCCLCHECK(ncclGdrCudaCalloc(&map->devMem.cpuPtr, &map->devMem.gpuPtr, devMemSize, &resources->gdrDesc));
      } else {
        NCCLCHECK(ncclCudaCalloc(&map->devMem.gpuPtr, devMemSize));
      }
      if (!map->sameProcess) {
        CUDACHECK(cudaIpcGetMemHandle(&map->devMem.ipc, &map->devMem.gpuPtr));
      }
    }

    if (map->sameProcess) {
      NCCLCHECK(ncclCudaHostCalloc(&map->hostMem.cpuPtr, hostMemSize));
      map->hostMem.gpuPtr = map->hostMem.cpuPtr;
    } else {
      NCCLCHECK(netCreateShm(hostMemSize, &map->hostMem.cpuPtr, map->hostMem.shmPath));
      map->hostMem.shmSize = hostMemSize;
    }

    resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
    resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);
      if (resources->buffers[p])
        NCCLCHECK(ncclNetRegMr(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
    }

    NCCLCHECK(socketSend(fd, map, sizeof(struct netConnectMap)));
  }
  return ncclSuccess;
}

static ncclResult_t netSendProxyFree(void* state, struct ncclComm* comm) {
  return ncclSuccess;
}

static ncclResult_t netRecvProxyFree(void* state, struct ncclComm* comm) {
  return ncclSuccess;
}

static_assert(NCCL_STEPS <= NCCL_NET_MAX_REQUESTS, "Not enough net requests to cover for steps");

static ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct netSendResources* resources = (struct netSendResources*) (sub->connector->transportResources);
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
      if (sub->done == sub->nsteps) continue;
      struct netSendResources* resources = (struct netSendResources*) (sub->connector->transportResources);
      void* mhandle = resources->mhandles[p];
      int stepSize = sub->connector->comm->buffSizes[p] / NCCL_STEPS;
      char* localBuff = sub->connector->conn.buffs[p];
      int buffSize = stepSize*args->sliceSteps;
      if (resources->shared) buffSize /= SENDRECV_SLICEFACTOR;
      if (sub->sendbytes < buffSize) buffSize = sub->sendbytes;
      // Post buffers to the GPU
      if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        if (resources->shared) {
          char* ptr;
          int sharedBuffSlot = sub->posted%NCCL_STEPS;
          NCCLCHECK(ncclProxySharedBuffersGetP2p(sub->connector->comm, resources->useGdr, resources->netDev, 0, sub->channel->id, sharedBuffSlot, s, &ptr));
          resources->recvMem->ptrsFifo[buffSlot] = ptr;
          __sync_synchronize();
          volatile uint64_t* sendHead = &resources->sendMem->head;
          sub->posted += args->sliceSteps;
          *sendHead = sub->base + sub->posted - NCCL_STEPS;
        } else sub->posted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      // Check whether we received data from the GPU and send it to the network
      if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile int* sizesFifo = resources->recvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        if (sizesFifo[buffSlot] != -1 && ((*recvTail > (sub->base+sub->transmitted)) || p == NCCL_PROTO_LL)) {
          // We have something to receive, let's check if it's completely ready.
          int size = sizesFifo[buffSlot];
          char* buff = resources->shared ? (char*)resources->recvMem->ptrsFifo[buffSlot] : localBuff+buffSlot*stepSize;
          int ready = 1;
          if (p == NCCL_PROTO_LL128) {
            ready = resources->useGdr;
            if (!ready) {
              // When data is in sysmem, we need to wait until all flags are correct since the GPU only
              // called threadfence()
              uint64_t flag = sub->base+sub->transmitted+1;
              int nFifoLines = DIVUP(sizesFifo[buffSlot], sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
              volatile uint64_t* lines = (volatile uint64_t*)buff;
              ready = 1;
              for (int i=0; i<nFifoLines; i++) {
                if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
              }
            }
          } else if (p == NCCL_PROTO_LL) {
            uint32_t flag = NCCL_LL_FLAG(sub->base+sub->transmitted+1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)buff;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
            }
          }
          if (ready) {
            // Data is ready, try to send.
            NCCLCHECK(ncclNetIsend(resources->netSendComm, buff, size, mhandle, sub->requests+buffSlot));
            if (sub->requests[buffSlot] != NULL) {
              TRACE(NCCL_NET, "sendProxy [%ld/%d] Isend (LL) posted, req %p", sub->transmitted, buffSlot, sub->requests[buffSlot]);
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
        int done;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        NCCLCHECK(ncclNetTest(sub->requests[buffSlot], &done, NULL));
        if (done) {
          TRACE(NCCL_NET, "sendProxy [%ld/%d] request %p done", sub->done, buffSlot, sub->requests[buffSlot]);
          sub->done += args->sliceSteps;

          if (resources->shared == 0) {
            resources->sendMem->head = sub->base + sub->done;
          }
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

static ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct netRecvResources* resources = (struct netRecvResources*) (sub->connector->transportResources);
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
      if (sub->done == sub->nsteps) continue;
      struct netRecvResources* resources = (struct netRecvResources*) (sub->connector->transportResources);
      void* mhandle = resources->mhandles[p];
      int stepSize = sub->connector->comm->buffSizes[p] / NCCL_STEPS;
      char* localBuff = sub->connector->conn.buffs[p];
      int buffSize = stepSize*args->sliceSteps;
      if (resources->shared) buffSize /= SENDRECV_SLICEFACTOR;
      if (sub->recvbytes < buffSize) buffSize = sub->recvbytes;

      if ((sub->posted < sub->done + NCCL_STEPS) && (sub->posted < sub->nsteps)) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        char* ptr;
        if (resources->shared) {
          int sharedBuffSlot = sub->posted%NCCL_STEPS;
          NCCLCHECK(ncclProxySharedBuffersGetP2p(sub->connector->comm, resources->useGdr, resources->netDev, 1, sub->channel->id, sharedBuffSlot, s, &ptr));
          volatile void** ptrsFifo = (volatile void**)resources->recvMem->ptrsFifo;
          ptrsFifo[buffSlot] = ptr;
        } else {
          ptr = localBuff+buffSlot*stepSize;
        }
        NCCLCHECK(ncclNetIrecv(resources->netRecvComm, ptr, buffSize, mhandle, sub->requests+buffSlot));
        if (sub->requests[buffSlot] != NULL) {
          TRACE(NCCL_NET, "recvProxy [%ld/%d] posted recv request %p", sub->posted, buffSlot, sub->requests[buffSlot]);
          sub->posted += args->sliceSteps;
          args->idle = 0;
          continue;
        }
      }
      if (sub->posted > sub->received) {
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;
        int done, size;
        NCCLCHECK(ncclNetTest(sub->requests[buffSlot], &done, &size));
        if (done) {
          sub->received += args->sliceSteps;
          if (size > 0 && p == NCCL_PROTO_SIMPLE && resources->useGdr) {
            // Don't pass data to the GPU yet, flush first.

            // GDRCOPY support
            if (resources->useGdc && ncclParamGdrCopyFlushEnable()) {
#if defined (__x86_64__)
              // Force a PCI-E read from GPU memory
              asm volatile ("mov (%0), %%eax" :: "l"(&resources->recvMem->flush) : "%eax");
#else
              WARN("NET: GDR Flush only supported on x86_64");
              return ncclInternalError;
#endif
              sub->requests[buffSlot] = NULL;
            } else {
              volatile void** ptrsFifo = (volatile void**)resources->recvMem->ptrsFifo;
              char* ptr = resources->shared ? (char*)(ptrsFifo[buffSlot]) : localBuff+buffSlot*stepSize;
              NCCLCHECK(ncclNetIflush(resources->netRecvComm, ptr, size, mhandle, sub->requests+buffSlot));
            }
          } else {
            sub->requests[buffSlot] = NULL;
          }
          args->idle = 0;
          continue;
        }
      }
      if (sub->received > sub->transmitted) {
        // Progress flush operations
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        int done = 1;
        if (sub->requests[buffSlot]) NCCLCHECK(ncclNetTest(sub->requests[buffSlot], &done, NULL));
        if (done) {
          sub->transmitted += args->sliceSteps;
          __sync_synchronize();
          resources->recvMem->tail = sub->base + sub->transmitted;
          if (resources->useGdc) {
            // GDRCOPY support: Write updated tail directly to the device memory
            wc_store_fence(); // Flush out WC write
          }
          args->idle = 0;
          continue;
        }
      }
      if (sub->transmitted > sub->done) {
        volatile uint64_t* sendHead = &resources->sendMem->head;
        uint64_t done = *sendHead;
        while (done > sub->base + sub->done &&
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

struct ncclTransport netTransport = {
  "NET",
  netCanConnect,
  { netSendSetup, netSendConnect, netSendFree, netSendProxyCall, netSendProxyFree, netSendProxy },
  { netRecvSetup, netRecvConnect, netRecvFree, netRecvProxyCall, netRecvProxyFree, netRecvProxy }
};
