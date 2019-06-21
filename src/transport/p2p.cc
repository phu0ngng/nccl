/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "utils.h"
#include "graph.h"
#include "transport.h"
#include "param.h"
#include <unistd.h>
#include <cuda_runtime.h>
#include <ctype.h>
#include "nvlink.h"

struct p2pConnectInfo {
  int direct;
  union {
    void* directPtr;
    cudaIpcMemHandle_t devIpc;
  };
};

struct p2pSendResources {
  struct ncclSendMem* devMem;
  void* ipcPtr;
};

struct p2pRecvResources {
  struct ncclRecvMem* devMem;
  void* ipcPtr;
};

#include <sys/types.h>

NCCL_PARAM(P2pLevel, "P2P_LEVEL", -2);
NCCL_PARAM(P2pDisable, "P2P_DISABLE", -2);

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
static int busIdToCudaDev(const char* busId) {
  int ndev;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess)
    return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (cudaDeviceGetPCIBusId(devBusId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i) != cudaSuccess)
      return -1;
    if (strcmp(busId, devBusId) == 0) {
      return i;
    }
  }
  // BusId was not found in our locally visible CUDA devices
  return -1;
}

/* Determine if we can communicate with the peer through p2p */
ncclResult_t p2pCanConnect(int* ret, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo) {
  // Do not use P2P across root complexes by default (provided CUDA permits it)
  int p2pLevel = PATH_NODE;
  if (ncclParamP2pDisable() == 1) p2pLevel = 0;
  if (ncclParamP2pLevel() != -2) p2pLevel = ncclParamP2pLevel();

  *ret = 0;

  if (p2pLevel == 0) return ncclSuccess;

  // Rule out different nodes
  if (myInfo->hostHash != peerInfo->hostHash) return ncclSuccess;

  // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
  int peerCudaDev = busIdToCudaDev(peerInfo->busId);
  if (peerCudaDev == -1) {
    // Peer's CUDA device is not visible in this process
#if CUDART_VERSION >= 10010
    // But in CUDA 10.1 we can still communicate with 'invisible' devices
    TRACE(NCCL_INIT|NCCL_P2P, "Checking P2P connection between %d(%s) and %d(%s)", myInfo->nvmlDev, myInfo->busId, peerInfo->nvmlDev, peerInfo->busId);
    // Check for NVLink/NVswitch including P2P access
    int nvlinkp2p = getNvlinkGpu(myInfo->busId, peerInfo->busId);
    if (nvlinkp2p > 0) {
      *ret = 1;
      return ncclSuccess;
    }
#endif
    return ncclSuccess;
  }

  TRACE(NCCL_INIT|NCCL_P2P, "Checking P2P connection between [%d=%d] and [%d=%d]", myInfo->cudaDev, myInfo->nvmlDev, peerCudaDev, peerInfo->nvmlDev);

  // Do not detect topology if we're on the same GPU. Note this is not really supported.
  if (myInfo->cudaDev == peerCudaDev) {
    *ret = 1;
    return ncclSuccess;
  }

  // See if CUDA can do P2P
  int p2p;
  if (cudaDeviceCanAccessPeer(&p2p, myInfo->cudaDev, peerCudaDev) != cudaSuccess) {
    INFO(NCCL_INIT|NCCL_P2P,"peer query failed between dev %d(=%d) and dev %d(=%d)",
         myInfo->cudaDev, myInfo->nvmlDev, peerCudaDev, peerInfo->nvmlDev);
    return ncclSuccess;
  }
  if (p2p == 0) return ncclSuccess;

  // Check for NVLink/NVswitch
  int nvlinkp2p = getNvlinkGpu(myInfo->busId, peerInfo->busId);
  if (nvlinkp2p > 0) {
    *ret = 1;
    return ncclSuccess;
  }

  // Finally compute the PCI distance and compare with the p2pLevel.
  char* myPath;
  char* peerPath;
  ncclResult_t err1 = getCudaPath(myInfo->cudaDev, &myPath);
  ncclResult_t err2 = getCudaPath(peerCudaDev, &peerPath);
  if (err1 == ncclSuccess && err2 == ncclSuccess) {
    int distance = pciDistance(myPath, peerPath);
    if (distance < p2pLevel) {
      *ret = 1;
    }
  }
  if (err1 == ncclSuccess) free(myPath);
  if (err2 == ncclSuccess) free(peerPath);
  return ncclSuccess;
}

#define TRACE_DUMP_IPC(DEVIPC)                                                             \
  do {                                                                                     \
    unsigned long *devIpc = (unsigned long *) (DEVIPC);                                    \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

/* Send: Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSendSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int buffSize, int channelId) {

  struct p2pSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  const int sendSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclCudaCalloc((char**)&resources->devMem, sendSize));

  struct p2pConnectInfo info;
  if (myInfo->pidHash == peerInfo->pidHash) {
    info.direct = 1;
    info.directPtr = resources->devMem;
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d -> %d via P2P/common device", channelId, myInfo->rank, peerInfo->rank);
    } else {
      // Enable P2P access
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d(=%d): %d %s",
             peerInfo->cudaDev, peerInfo->nvmlDev, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
      INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] -> %d[%d] via P2P/direct pointer",
          channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
    }
  } else {
    // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
    int peerCudaDev = busIdToCudaDev(peerInfo->busId);
    info.direct = 0;
    // Map IPC and enable P2P access
    cudaError_t err = cudaIpcGetMemHandle(&info.devIpc, (void*)resources->devMem);
    if (err != cudaSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d(=%d) : %d %s",
           myInfo->rank, peerCudaDev, peerInfo->nvmlDev, err, cudaGetErrorString(err));
      return ncclInternalError;
    }
    INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] -> %d[%d] via P2P/IPC",
        channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
    //TRACE_DUMP_IPC(&info.devIpc);
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pRecvSetup(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int buffSize, int channelId) {

  struct p2pRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  const int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  NCCLCHECK(ncclCudaCalloc((char**)&resources->devMem, recvSize));

  struct p2pConnectInfo info;
  if (myInfo->pidHash == peerInfo->pidHash) {
    info.direct = 1;
    info.directPtr = resources->devMem;
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      TRACE(NCCL_INIT|NCCL_P2P,"%d <- %d via P2P/common device", myInfo->rank, peerInfo->rank);
    } else {
      // Enable P2P access
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d(=%d): %d %s",
             peerInfo->cudaDev, peerInfo->nvmlDev, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
      TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] <- %d[%d] via P2P/direct pointer", channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
    }
  } else {
    // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
    int peerCudaDev = busIdToCudaDev(peerInfo->busId);
    info.direct = 0;
    // Map IPC and enable P2P access
    cudaError_t err = cudaIpcGetMemHandle(&info.devIpc, (void*)resources->devMem);
    if (err != cudaSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d(=%d) : %d %s",
           myInfo->rank, peerCudaDev, peerInfo->nvmlDev, err, cudaGetErrorString(err));
      return ncclInternalError;
    }
    TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] <- %d[%d] via P2P/IPC", channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
    //TRACE_DUMP_IPC(&info.devIpc);
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Connect/Send to this peer */
static ncclResult_t p2pSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  struct p2pSendResources* resources = (struct p2pSendResources*)send->transportResources;
  struct ncclRecvMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  if (info->direct) {
    remDevMem = (struct ncclRecvMem*)(info->directPtr);
    send->conn.direct = 1;
  } else {
    //TRACE_DUMP_IPC(&info->devIpc);
    cudaError_t err = cudaIpcOpenMemHandle(&resources->ipcPtr, info->devIpc, cudaIpcMemLazyEnablePeerAccess);
    remDevMem = (struct ncclRecvMem*)resources->ipcPtr;
    if (err != cudaSuccess) {
      WARN("failed to open CUDA IPC handle : %d %s",
          err, cudaGetErrorString(err));
      return ncclUnhandledCudaError;
    }
  }

  send->conn.buff = remDevMem->buff;
  send->conn.llBuff = remDevMem->llBuff;
  send->conn.ll128Buff = remDevMem->ll128Buff;
  send->conn.tail = &remDevMem->tail;
  send->conn.opCountRem = &remDevMem->opCount;
  send->conn.head = &resources->devMem->head;
  send->conn.ptrExchange = &resources->devMem->ptrExchange;
  send->conn.opCountLoc = &resources->devMem->opCount;
  return ncclSuccess;
}

/* Connect/Recv from this peer */
ncclResult_t p2pRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  struct p2pRecvResources* resources = (struct p2pRecvResources*)recv->transportResources;
  struct ncclSendMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  if (info->direct) {
    remDevMem = (struct ncclSendMem*)(info->directPtr);
    recv->conn.direct = 1;
    recv->conn.ptrExchange = &remDevMem->ptrExchange;
  } else {
    //TRACE_DUMP_IPC(&info->devIpc);
    cudaError_t err = cudaIpcOpenMemHandle(&resources->ipcPtr, info->devIpc, cudaIpcMemLazyEnablePeerAccess);
    remDevMem = (struct ncclSendMem*)resources->ipcPtr;
    if (err != cudaSuccess) {
      WARN("failed to open CUDA IPC handle : %d %s",
          err, cudaGetErrorString(err));
      return ncclUnhandledCudaError;
    }
  }

  recv->conn.buff = resources->devMem->buff;
  recv->conn.llBuff = resources->devMem->llBuff;
  recv->conn.ll128Buff = resources->devMem->ll128Buff;
  recv->conn.tail = &resources->devMem->tail;
  recv->conn.opCountLoc = &resources->devMem->opCount;
  recv->conn.head = &remDevMem->head;
  recv->conn.opCountRem = &remDevMem->opCount;
  return ncclSuccess;
}

ncclResult_t p2pSendFree(void* resources) {
  struct p2pSendResources* sendRes = (struct p2pSendResources*)resources;
  if (sendRes->ipcPtr)
    CUDACHECK(cudaIpcCloseMemHandle(sendRes->ipcPtr));
  CUDACHECK(cudaFree(sendRes->devMem));
  free(sendRes);
  return ncclSuccess;
}

ncclResult_t p2pRecvFree(void* resources) {
  struct p2pRecvResources* recvRes = (struct p2pRecvResources*)resources;
  if (recvRes->ipcPtr)
    CUDACHECK(cudaIpcCloseMemHandle(recvRes->ipcPtr));
  CUDACHECK(cudaFree(recvRes->devMem));
  free(recvRes);
  return ncclSuccess;
}

struct ncclTransport p2pTransport = {
  "P2P",
  p2pCanConnect,
  { p2pSendSetup, p2pSendConnect, p2pSendFree, NULL },
  { p2pRecvSetup, p2pRecvConnect, p2pRecvFree, NULL }
};
