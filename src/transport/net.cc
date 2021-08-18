/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "net.h"
#include "graph.h"
#include "proxy.h"
#include "collectives.h"
#include "gdrwrap.h"

struct netConnectInfo {
  ncclNetHandle_t netHandle;
};

#define LOC_HOSTMEM 0
#define LOC_DEVMEM  1
#define LOC_COUNT   2

struct connectMapMem{
  char* gpuPtr;
  char* cpuPtr;
  int size;
  union {
    char shmPath[PATH_MAX];
    cudaIpcMemHandle_t ipc;
  };
};

struct connectMap {
  int sameProcess;
  int shared;
  int useGdc;
  // Mem banks. 001 is host mem, 011 is dev mem, 101 is shared host mem and 111 is shared dev mem.
  struct connectMapMem mems[4];
  // Offsets. 3 MSBs indicate mem bank, 111 indicates NULL.
  struct {
    uint32_t sendMem;
    uint32_t recvMem;
    uint32_t buffs[NCCL_NUM_PROTOCOLS];
  } offsets;
};

#define NCCL_NET_MAP_HOSTMEM 0
#define NCCL_NET_MAP_DEVMEM 1
#define NCCL_NET_MAP_SHARED_HOSTMEM 2
#define NCCL_NET_MAP_SHARED_DEVMEM 3
#define NCCL_NET_MAP_MASK_DEVMEM 0x40000000
#define NCCL_NET_MAP_MASK_SHARED 0x80000000
#define NCCL_NET_MAP_MASK_USED   0x20000000
#define NCCL_NET_MAP_MASK_OFFSET 0x1fffffff

#define NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName) \
  ((mapStruct)->offsets.offsetName >> 30)

#define NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName >> 29) == 0)

#define NCCL_NET_MAP_GET_POINTER(mapStruct, cpuOrGpu, offsetName) \
  (NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) ? NULL : \
   (mapStruct)->mems[NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName)].cpuOrGpu##Ptr + ((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_OFFSET))

#define NCCL_NET_MAP_DEV_MEM(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_DEVMEM) != 0)

#define NCCL_NET_MAP_ADD_POINTER(mapStruct, shared, dev, memSize, offsetName) do { \
    int bank = NCCL_NET_MAP_MASK_USED + (dev)*NCCL_NET_MAP_MASK_DEVMEM + (shared)*NCCL_NET_MAP_MASK_SHARED; \
    if ((shared) == 0) { \
      if (dev) { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size += memSize; \
      } else { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size += memSize; \
      } \
    } else { \
      (mapStruct)->offsets.offsetName = bank; \
    } \
} while (0);

struct sendResources {
  struct connectMap map;
  void* netSendComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  // GDRCOPY support
  void* gdrDesc;

  int rank;
  int localRank;
  int remoteRank;
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

struct recvResources {
  struct connectMap map;
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  // GDRCOPY support
  void* gdrDesc;

  int rank;
  int localRank;
  int remoteRank;
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
static ncclResult_t canConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

NCCL_PARAM(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);

struct setupReq {
  int rank;
  int localRank;
  int remoteRank;
  int shared;
  int netDev;
  int useGdr;
};

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req;

  send->conn.shared = req.shared = ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : graph ? 0 : 1;

  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &req.netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 1, &req.useGdr));
  send->conn.direct |= req.useGdr ? NCCL_DIRECT_NIC : 0;

  int proxyRank;
  NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, myInfo->rank, req.netDev, &proxyRank));
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, proxyRank, &send->proxyConn));
  req.rank = myInfo->rank;
  req.localRank = send->proxyConn.localRank;
  req.remoteRank = peerInfo->rank;
  NCCLCHECK(ncclProxyCall(&send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  if (proxyRank == myInfo->rank) {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d%s%s", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, ncclNetName(), req.netDev,
        req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  } else {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d(%d)%s%s", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, ncclNetName(), req.netDev,
        proxyRank, req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  }
  return ncclSuccess;
}

// GDRCOPY support: TAIL_ENABLE When enabled locates the RX proxy tail in CUDA memory
NCCL_PARAM(GdrCopySyncEnable, "GDRCOPY_SYNC_ENABLE", 1);
// GDRCOPY support: FLUSH_ENABLE When enabled uses a PCI-E read to flush GDRDMA buffers
NCCL_PARAM(GdrCopyFlushEnable, "GDRCOPY_FLUSH_ENABLE", 0);

/* Setup recv connector */
static ncclResult_t recvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct setupReq req;

  recv->conn.shared = req.shared = ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : graph ? 0 : 1;

  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &req.netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, req.netDev, 0, &req.useGdr));

  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 0, myInfo->rank, &recv->proxyConn));

  req.rank = myInfo->rank;
  req.localRank = recv->proxyConn.localRank;
  req.remoteRank = peerInfo->rank;
  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(ncclProxyCall(&recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), &info->netHandle, sizeof(ncclNetHandle_t)));

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [receive] via NET/%s/%d%s%s", channelId, peerInfo->rank, peerInfo->busId, myInfo->rank, myInfo->busId, ncclNetName(), req.netDev,
      req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  return ncclSuccess;
}

static ncclResult_t netMapShm(struct connectMapMem* mem) {
  int fd = open(mem->shmPath, O_RDWR);
  mem->cpuPtr = (char*)mmap(NULL, mem->size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  CUDACHECK(cudaHostRegister(mem->cpuPtr, mem->size, cudaHostRegisterMapped));
  CUDACHECK(cudaHostGetDevicePointer(&mem->gpuPtr, mem->cpuPtr, 0));
  unlink(mem->shmPath);
  close(fd);
  return ncclSuccess;
}
static ncclResult_t netCreateShm(struct connectMapMem* mem) {
  sprintf(mem->shmPath, "/dev/shm/nccl-XXXXXX");
  int fd = mkstemp(mem->shmPath);
  if (fd == -1) {
    WARN("Error: could not create shared memory in /dev/shm");
    return ncclSystemError;
  }
  if (ftruncate(fd, mem->size) != 0) {
    WARN("Error: failed to extend %s to %d bytes", mem->shmPath, mem->size);
    close(fd);
    return ncclSystemError;
  }
  mem->cpuPtr = (char*)mmap(NULL, mem->size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  return ncclSuccess;
}

static ncclResult_t netDumpMap(struct connectMap* map) {
  printf("Dump map same process %d shared %d useGdc %d\n", map->sameProcess, map->shared, map->useGdc);
  struct connectMapMem *mem = map->mems+NCCL_NET_MAP_HOSTMEM;
  printf("Mem 0: Host mem %s (%x B) CPU %p GPU %p\n", mem->shmPath, mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_DEVMEM;
  printf("Mem 1: Vid  mem CPU (%x B) %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_HOSTMEM;
  printf("Mem 2: Shared Host mem %s (%x B) CPU %p GPU %p\n", mem->shmPath, mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_DEVMEM;
  printf("Mem 3: Shared Vid  (%x B) mem CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  printf("SendMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.sendMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, sendMem), map->offsets.sendMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem), NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem));
  printf("RecvMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.recvMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, recvMem), map->offsets.recvMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem), NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem));
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    printf("Proto %d -> Used %d Bank %d Offset %x, cpu %p, gpu %p\n", p,
        map->offsets.buffs[p] & NCCL_NET_MAP_MASK_USED ? 1 : 0,
        NCCL_NET_MAP_OFFSET_BANK(map, buffs[p]), map->offsets.buffs[p] & NCCL_NET_MAP_MASK_OFFSET,
        NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]), NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]));
  }
  printf("End of dump\n");
  return ncclSuccess;
}

static ncclResult_t sendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct netConnectInfo* info = (struct netConnectInfo*)connectInfo;
  struct connectMap map;
  NCCLCHECK(ncclProxyCall(&send->proxyConn, ncclProxyMsgConnect, &info->netHandle, sizeof(ncclNetHandle_t), &map, sizeof(struct connectMap)));

  if (map.sameProcess == 0) {
    NCCLCHECK(netMapShm(map.mems+NCCL_NET_MAP_HOSTMEM));
    if (map.mems[NCCL_NET_MAP_DEVMEM].size) {
      CUDACHECK(cudaIpcOpenMemHandle((void**)&map.mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map.mems[NCCL_NET_MAP_DEVMEM].ipc, cudaIpcMemLazyEnablePeerAccess));
      map.mems[NCCL_NET_MAP_DEVMEM].cpuPtr = NULL;
    }
  }
  NCCLCHECK(netDumpMap(&map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(&map, gpu, sendMem);
  send->conn.head = &sendMem->head;

  // Don't give credits yet in shared mode.
  sendMem->head = map.shared ? -NCCL_STEPS : 0;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(&map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;
  send->conn.sizesFifo = recvMem->sizesFifo;
  for (int i=0; i<NCCL_STEPS; i++) send->conn.sizesFifo[i] = -1;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  send->conn.offsFifo = map.shared ? recvMem->offsFifo : NULL;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(&map, gpu, buffs[p]);
  return ncclSuccess;
}

/* Connect to this peer */
static ncclResult_t recvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct connectMap map;
  NCCLCHECK(ncclProxyCall(&recv->proxyConn, ncclProxyMsgConnect, NULL, 0, &map, sizeof(struct connectMap)));

  if (map.sameProcess == 0) {
    NCCLCHECK(netMapShm(map.mems+NCCL_NET_MAP_HOSTMEM));
    if (map.mems[NCCL_NET_MAP_DEVMEM].size) {
      CUDACHECK(cudaIpcOpenMemHandle((void**)&map.mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map.mems[NCCL_NET_MAP_DEVMEM].ipc, cudaIpcMemLazyEnablePeerAccess));
      map.mems[NCCL_NET_MAP_DEVMEM].cpuPtr = NULL;
    }
  }
  NCCLCHECK(netDumpMap(&map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(&map, gpu, sendMem);
  recv->conn.head = &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(&map, gpu, recvMem);
  recv->conn.tail = &recvMem->tail;
  recv->conn.sizesFifo = recvMem->sizesFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  recv->conn.offsFifo = map.shared ? recvMem->offsFifo : NULL;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    recv->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(&map, gpu, buffs[p]);
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

  resources->rank = req.rank;
  resources->localRank = req.localRank;
  resources->remoteRank = req.remoteRank;
  resources->netDev = req.netDev;
  resources->shared = connection->shared = req.shared;
  resources->useGdr = req.useGdr;
  return ncclSuccess;
}

static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct setupReq req;
  NCCLCHECK(ncclSocketRecv(connection->sock, &req, sizeof(req)));

  struct recvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->rank = req.rank;
  resources->localRank = req.localRank;
  resources->remoteRank = req.remoteRank;
  resources->netDev = req.netDev;
  resources->shared = connection->shared = req.shared;
  resources->useGdr = req.useGdr;

  ncclNetHandle_t netHandle;
  NCCLCHECK(ncclNetListen(req.netDev, &netHandle, &resources->netListenComm));
  NCCLCHECK(ncclSocketSend(connection->sock, &netHandle, sizeof(ncclNetHandle_t)));
  return ncclSuccess;
}

static ncclResult_t sendProxyConnect(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);
  ncclNetHandle_t netHandle;
  NCCLCHECK(ncclSocketRecv(connection->sock, &netHandle, sizeof(ncclNetHandle_t)));

  if (resources->shared) {
    // Shared connection
    struct ncclProxyProgressState* progressState = &comm->proxyState.progressState;
    if (progressState->localPeers == NULL) {
      NCCLCHECK(ncclCalloc(&progressState->localPeers, comm->localRanks));
    }
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    if (localPeers[resources->localRank] == NULL) {
      NCCLCHECK(ncclCalloc(localPeers+resources->localRank, 1));
    }
    struct ncclProxySharedP2p* p2p = &localPeers[resources->localRank]->send;
    // Connect or reuse connection for a netdev/remote rank.
    if (p2p->transportResources[resources->netDev] == NULL) {
      NCCLCHECK(ncclCalloc(p2p->transportResources+resources->netDev, comm->nRanks));
    }
    if (p2p->transportResources[resources->netDev][resources->remoteRank] == NULL) {
      NCCLCHECK(ncclNetConnect(resources->netDev, netHandle, &resources->netSendComm));
      p2p->transportResources[resources->netDev][resources->remoteRank] = resources->netSendComm;
    } else {
      resources->netSendComm = p2p->transportResources[resources->netDev][resources->remoteRank];
    }
    connection->proxyAppendPtr = &p2p->proxyAppend;
  } else {
    // Connect to remote peer
    NCCLCHECK(ncclNetConnect(resources->netDev, netHandle, &resources->netSendComm));
    connection->proxyAppendPtr = &connection->proxyAppend;
  }

  // Create structures
  struct connectMap* map = &resources->map;
  map->sameProcess =
    comm->peerInfo[resources->rank].pidHash == comm->peerInfo[comm->rank].pidHash ? 1 : 0;
  map->shared = resources->shared;
  map->useGdc = map->shared == 0 && map->sameProcess && ncclGdrCopy != NULL && ncclParamGdrCopySyncEnable();

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, p!= NCCL_PROTO_LL && resources->useGdr, comm->buffSizes[p], buffs[p]);
      resources->buffSizes[p] = comm->buffSizes[p];
    }
  } else {
    // Get shared buffers
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    struct connectMapMem* mapMem = map->mems+bank;
    NCCLCHECK(ncclProxySharedBuffersInitP2p(
          comm, resources->useGdr, resources->localRank, 0, 1,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, NULL));
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;
    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  NCCL_NET_MAP_ADD_POINTER(map, 0, map->useGdc, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    if (resources->shared == 0) {
      if (!map->sameProcess) {
        ALIGN_SIZE(map->mems[NCCL_NET_MAP_DEVMEM].size, CUDA_IPC_MIN);
      }
      if (map->useGdc) {
        NCCLCHECK(ncclGdrCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr, &map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size, &resources->gdrDesc));
      } else {
        NCCLCHECK(ncclCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size));
        map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
      }
    }
    if (!map->sameProcess) {
      CUDACHECK(cudaIpcGetMemHandle(&map->mems[NCCL_NET_MAP_DEVMEM].ipc, map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
    }
  }

  if (map->sameProcess) {
    NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
    map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  } else {
    NCCLCHECK(netCreateShm(map->mems+NCCL_NET_MAP_HOSTMEM));
  }

  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    if (resources->buffers[p]) {
      NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
    }
  }

  //NCCLCHECK(netDumpMap(map));
  NCCLCHECK(ncclSocketSend(connection->sock, map, sizeof(struct connectMap)));
  return ncclSuccess;
}

static ncclResult_t recvProxyConnect(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);

  // Finish connection establishment from remote peer
  if (resources->shared) {
    // Shared connection
    struct ncclProxyProgressState* progressState = &comm->proxyState.progressState;
    if (progressState->localPeers == NULL) {
      NCCLCHECK(ncclCalloc(&progressState->localPeers, comm->localRanks));
    }
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    if (localPeers[resources->localRank] == NULL) {
      NCCLCHECK(ncclCalloc(localPeers+resources->localRank, 1));
    }
    struct ncclProxySharedP2p* p2p = &localPeers[resources->localRank]->recv;
    // Connect or reuse connection for a netdev/remote rank.
    if (p2p->transportResources[resources->netDev] == NULL) {
      NCCLCHECK(ncclCalloc(p2p->transportResources+resources->netDev, comm->nRanks));
    }
    if (p2p->transportResources[resources->netDev][resources->remoteRank] == NULL) {
      NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
      p2p->transportResources[resources->netDev][resources->remoteRank] = resources->netRecvComm;
    } else {
      resources->netRecvComm = p2p->transportResources[resources->netDev][resources->remoteRank];
    }
    connection->proxyAppendPtr = &p2p->proxyAppend;
  } else {
    // Connect to remote peer
    NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
    connection->proxyAppendPtr = &connection->proxyAppend;
  }
  NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

  // Create structures
  struct connectMap* map = &resources->map;
  map->sameProcess =
    comm->peerInfo[resources->rank].pidHash == comm->peerInfo[comm->rank].pidHash ? 1 : 0;
  map->shared = resources->shared;
  map->useGdc = map->shared == 0 && map->sameProcess && ncclGdrCopy != NULL && ncclParamGdrCopySyncEnable();

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, resources->useGdr, comm->buffSizes[p], buffs[p]);
      resources->buffSizes[p] = comm->buffSizes[p];
    }
  } else {
    // Get shared buffers
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    struct connectMapMem* mapMem = map->mems+bank;
    NCCLCHECK(ncclProxySharedBuffersInitP2p(
          comm, resources->useGdr, resources->localRank, 1, map->sameProcess,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, &mapMem->ipc));
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;
    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, map->useGdc, sizeof(struct ncclRecvMem), recvMem);

  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    if (resources->shared == 0) {
      if (!map->sameProcess) {
        ALIGN_SIZE(map->mems[NCCL_NET_MAP_DEVMEM].size, CUDA_IPC_MIN);
      }
      if (map->useGdc) {
        NCCLCHECK(ncclGdrCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr, &map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size, &resources->gdrDesc));
      } else {
        NCCLCHECK(ncclCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size));
        map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
      }
    }
    if (!map->sameProcess) {
      CUDACHECK(cudaIpcGetMemHandle(&map->mems[NCCL_NET_MAP_DEVMEM].ipc, map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
    }
  }

  if (map->sameProcess) {
    NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
    map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  } else {
    NCCLCHECK(netCreateShm(map->mems+NCCL_NET_MAP_HOSTMEM));
  }

  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    if (resources->buffers[p]) {
      NCCLCHECK(ncclNetRegMr(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
    }
  }

  //NCCLCHECK(netDumpMap(map));
  NCCLCHECK(ncclSocketSend(connection->sock, map, sizeof(struct connectMap)));
  return ncclSuccess;
}

static ncclResult_t sendProxyFree(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);
  NCCLCHECK(ncclProxySharedBuffersDestroyP2p(comm, resources->localRank, 0));
  // TODO : free shared peers
  free(connection->transportResources);
  return ncclSuccess;
}

static ncclResult_t recvProxyFree(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);
  NCCLCHECK(ncclProxySharedBuffersDestroyP2p(comm, resources->localRank, 1));
  // TODO : free shared peers
  free(connection->transportResources);
  return ncclSuccess;
}

static_assert(NCCL_STEPS <= NCCL_NET_MAX_REQUESTS, "Not enough net requests to cover for steps");

static ncclResult_t sendProxyProgress(struct ncclComm* comm, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendResources* resources = (struct sendResources*) (sub->connection->transportResources);
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
      struct sendResources* resources = (struct sendResources*) (sub->connection->transportResources);
      void* mhandle = resources->mhandles[p];
      int stepSize = comm->buffSizes[p] / NCCL_STEPS;
      char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
      int buffSize = stepSize*args->sliceSteps;
      if (resources->shared) buffSize /= SENDRECV_SLICEFACTOR;
      if (sub->sendbytes < buffSize) buffSize = sub->sendbytes;
      // Post buffers to the GPU
      if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        if (resources->shared) {
          int sharedBuffSlot = sub->posted%NCCL_STEPS;
          int offset;
          NCCLCHECK(ncclProxySharedBuffersGetP2p(comm, sub->channel->id, sharedBuffSlot, s, &offset));
          resources->recvMem->offsFifo[buffSlot] = offset;
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
          char* buff = resources->shared ? localBuff+resources->recvMem->offsFifo[buffSlot] : localBuff+buffSlot*stepSize;
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
              TRACE(NCCL_NET, "sendProxy [%ld/%d] Isend posted, req %p", sub->transmitted, buffSlot, sub->requests[buffSlot]);
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

static ncclResult_t recvProxyProgress(struct ncclComm* comm, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
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
      struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
      void* mhandle = resources->mhandles[p];
      int stepSize = comm->buffSizes[p] / NCCL_STEPS;
      char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
      int buffSize = stepSize*args->sliceSteps;
      if (resources->shared) buffSize /= SENDRECV_SLICEFACTOR;
      if (sub->recvbytes < buffSize) buffSize = sub->recvbytes;

      if ((sub->posted < sub->done + NCCL_STEPS) && (sub->posted < sub->nsteps)) {
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        char* ptr;
        if (resources->shared) {
          int sharedBuffSlot = sub->posted%NCCL_STEPS;
          int offset;
          NCCLCHECK(ncclProxySharedBuffersGetP2p(comm, sub->channel->id, sharedBuffSlot, s, &offset));
          volatile int* offsFifo = (volatile int*)resources->recvMem->offsFifo;
          offsFifo[buffSlot] = offset;
          ptr = localBuff+offset;
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
              char* ptr = resources->shared ? localBuff+resources->recvMem->offsFifo[buffSlot] : localBuff+buffSlot*stepSize;
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
  canConnect,
  { sendSetup, sendConnect, sendFree, sendProxySetup, sendProxyConnect, sendProxyFree, sendProxyProgress },
  { recvSetup, recvConnect, recvFree, recvProxySetup, recvProxyConnect, recvProxyFree, recvProxyProgress }
};
