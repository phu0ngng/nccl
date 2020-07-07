#define MNNVL_SUPPORT 1

/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"

#ifdef MNNVL_SUPPORT
#include <cuda_device_runtime_api.h>
#include "wizlet.h"
#endif

enum p2pType { P2P_DIRECT, P2P_IPC, P2P_MULTINODE };

struct p2pConnectInfo {
  enum p2pType type;
  int read;
  union {
    void* directPtr;
#ifdef MNNVL_SUPPORT
    CUmemFabricHandle desc; // 1KiB
    size_t size;
#endif
    cudaIpcMemHandle_t devIpc;
  };
};

struct p2pSendResources {
  enum p2pType type;
  struct ncclSendMem* devMem;
#ifdef MNNVL_SUPPORT
  CUmemGenericAllocationHandle importHandle;
  CUmemGenericAllocationHandle exportHandle;
  size_t importSize, exportSize;
#endif
  void* remotePtr;
};

struct p2pRecvResources {
  enum p2pType type;
  struct ncclRecvMem* devMem;
#ifdef MNNVL_SUPPORT
  CUmemGenericAllocationHandle importHandle;
  CUmemGenericAllocationHandle exportHandle;
  size_t importSize, exportSize;
#endif
  void* remotePtr;
};

#include <sys/types.h>

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
static int busIdToCudaDev(int64_t busId) {
  int ndev;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess)
    return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (cudaDeviceGetPCIBusId(devBusIdStr, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i) != cudaSuccess)
      return -1;
    int64_t devBusId;
    NCCLCHECK(busIdToInt64(devBusIdStr, &devBusId));
    if (busId == devBusId) return i;
  }
  // BusId was not found in our locally visible CUDA devices
  return -1;
}

/* Determine if two peers can communicate through p2p */
ncclResult_t p2pCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
#ifndef MNNVL_SUPPORT
  // Rule out different nodes
  if (info1->hostHash != info2->hostHash) {
    *ret = 0;
    return ncclSuccess;
  }
#endif

  // Check topology / p2p level.
  int read;
  NCCLCHECK(ncclTopoCheckP2p(topo, info1->busId, info2->busId, ret, &read));
  if (*ret == 0) return ncclSuccess;

  // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
  int cudaDev1 = busIdToCudaDev(info1->busId);
  int cudaDev2 = busIdToCudaDev(info2->busId);
  if (cudaDev1 == -1 || cudaDev2 == -1) {
#if CUDART_VERSION >= 10010
    // CUDA 10.1 and later can use P2P with invisible devices.
    return ncclSuccess;
#else
    // Peer's CUDA device is not visible in this process : we can't communicate with it.
    *ret = 0;
    return ncclSuccess;
#endif
  }

  // Check that CUDA can do P2P
  if (info1->busId != info2->busId) {
    int p2p;
    if (cudaDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2) != cudaSuccess) {
      INFO(NCCL_INIT|NCCL_P2P,"peer query failed between dev %d(=%lx) and dev %d(=%lx)",
           cudaDev1, info1->busId, cudaDev2, info2->busId);
      *ret = 0;
      return ncclSuccess;
    }
    if (p2p == 0) {
      INFO(NCCL_INIT|NCCL_P2P,"Could not enable P2P between dev %d(=%lx) and dev %d(=%lx)",
           cudaDev1, info1->busId, cudaDev2, info2->busId);
      *ret = 0;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

#define TRACE_DUMP_IPC(DEVIPC)                                                             \
  do {                                                                                     \
    unsigned long *devIpc = (unsigned long *) (DEVIPC);                                    \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

#ifdef MNNVL_SUPPORT
// MNNVL: Multi-node NVLink
static ncclResult_t allocateShareableBuffer(int device, size_t size,
                                            CUmemFabricHandle *desc, CUmemGenericAllocationHandle *handle, void **devMemPtr) {
  CUmemAllocationProp prop;
  CUmemAccessDesc accessDesc;

  init_etbl();

  INFO(NCCL_P2P, "Allocating shareable buffer device %d size %zi handle %p", device, size, handle);

  // Allocation properties
  memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  prop.location.id = device;

  // Allocate and export
  CUDACHECK_DEV(cuExperimentalMemCreate(handle, size, &prop, /* alloc_flags */ 0));

  TRACE(NCCL_INIT|NCCL_P2P, "Allocated shareable buffer device %d size %zi handle 0x%x", device, size, *handle);

  CUDACHECK_DEV(cuExperimentalMemExportToShareableHandle(desc, *handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));

  TRACE(NCCL_INIT|NCCL_P2P, "Exported shareable buffer device %d size %zi handle 0x%x to desc %p", device, size, *handle, desc);

  CUdeviceptr dptr = 0;

  // In addition to allocating for export, also map for access by the local GPU
  CUDACHECK_DEV(cuExperimentalMemAddressReserve(&dptr, size, /* alignment */ 0, /* addr */ 0, /* flags */ 0));
  CUDACHECK_DEV(cuExperimentalMemMap(dptr, size, /*offset*/ 0, *handle, /* flags */ 0));
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUDACHECK_DEV(cuExperimentalMemSetAccess(dptr, size, &accessDesc, 1));

  TRACE(NCCL_INIT|NCCL_P2P, "Mapped shareable buffer device %d size %zi handle 0x%x dptr %p", device, size, *handle, dptr);

  *devMemPtr = (void *)dptr;

  return ncclSuccess;
}

static ncclResult_t freeShareableBuffer(void *buff, size_t size, CUmemGenericAllocationHandle handle) {
  CUdeviceptr dptr = (CUdeviceptr) buff;

  INFO(NCCL_P2P, "Free shareable buffer %p size %zi handle 0x%x", buff, size, handle);

  // Take care of the local GPU mappings we made
  CUDACHECK_DEV(cuExperimentalMemUnmap(dptr, size));
  CUDACHECK_DEV(cuExperimentalMemAddressFree(dptr, size));

  // Release the allocation
  CUDACHECK_DEV(cuExperimentalMemRelease(handle));

  return ncclSuccess;
}

#endif /* MNNVL_SUPPORT */

// Setting this to non zero causes P2P to use Reads rather than Writes
NCCL_PARAM(P2pReadEnable, "P2P_READ_ENABLE", -2);

static int p2pUseRead(struct ncclTopoSystem* topo, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  int readEnable = ncclParamP2pReadEnable();
  if (readEnable != -2) return readEnable;

  int p2p, read;
  // Queries the topology to see if the GPUs are Ampere and
  // connected via NVLink, if so we enable P2P Read by default
  NCCLCHECK(ncclTopoCheckP2p(topo, info1->busId, info2->busId, &p2p, &read));

  return read;
}

/* Send: Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSendSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId) {

  struct p2pSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  int useRead = p2pUseRead(topo, myInfo, peerInfo);
  int sendSize = sizeof(struct ncclSendMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  if (useRead) sendSize += send->comm->buffSizes[NCCL_PROTO_SIMPLE];
  ALIGN_SIZE(sendSize, CUDA_IPC_MIN);

  struct p2pConnectInfo info;
  memset(&info, 0, sizeof(info));
  info.read = useRead;
  const char* useReadStr = info.read ? "/read" : "";

  // MNNVL: Multi-node NVLink
  if (myInfo->hostHash != peerInfo->hostHash) {
    // Different hosts, so assume multi-node NVLink
    info.size = resources->exportSize = sendSize;
    info.type = P2P_MULTINODE;
    NCCLCHECK(allocateShareableBuffer(myInfo->cudaDev, sendSize, &info.desc, &resources->exportHandle, (void **)&resources->devMem));

    INFO(NCCL_INIT|NCCL_P2P,"Channel %02d : %d[%d] -> %d[%d] via P2P/MNNVL",
         channelId, myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev);
  } else {
    NCCLCHECK(ncclCudaCalloc((char**)&resources->devMem, sendSize));
    if (myInfo->pidHash == peerInfo->pidHash) {
      info.type = P2P_DIRECT;
      info.directPtr = resources->devMem;
      if (myInfo->cudaDev == peerInfo->cudaDev) {
        INFO(NCCL_INIT|NCCL_P2P,"Channel %02d : %d[%d] -> %d[%d] via P2P/common device%s",
             channelId, myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev, useReadStr);
        return ncclInternalError;
      } else {
        // Enable P2P access
        cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled) {
          cudaGetLastError();
        } else if (err != cudaSuccess) {
          WARN("failed to peer with device %d(=%lx): %d %s",
               peerInfo->cudaDev, peerInfo->busId, err, cudaGetErrorString(err));
          return ncclInternalError;
        }
        INFO(NCCL_INIT|NCCL_P2P,"Channel %02d : %d[%lx] -> %d[%lx] via P2P/direct pointer%s",
             channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, useReadStr);
      }
    } else {
      // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
      int peerCudaDev = busIdToCudaDev(peerInfo->busId);
      info.type = P2P_IPC;
      // Map IPC and enable P2P access
      cudaError_t err = cudaIpcGetMemHandle(&info.devIpc, (void*)resources->devMem);
      if (err != cudaSuccess) {
        WARN("rank %d failed to get CUDA IPC handle to device %d(=%lx) : %d %s",
             myInfo->rank, peerCudaDev, peerInfo->busId, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
      INFO(NCCL_INIT|NCCL_P2P,"Channel %02d : %d[%lx] -> %d[%lx] via P2P/IPC%s",
           channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, useReadStr);
      //TRACE_DUMP_IPC(&info.devIpc);
    }
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pRecvSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int channelId) {

  struct p2pRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  int useRead = p2pUseRead(topo, myInfo, peerInfo);
  int recvSize = offsetof(struct ncclRecvMem, buff);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) if (!(useRead && p == NCCL_PROTO_SIMPLE)) recvSize += recv->comm->buffSizes[p];
  ALIGN_SIZE(recvSize, CUDA_IPC_MIN);

  struct p2pConnectInfo info;
  memset(&info, 0, sizeof(info));
  info.read = useRead;

  // MNNVL: Multi-node NVLink support
  if (myInfo->hostHash != peerInfo->hostHash) {
    // Different hosts, so assume multi-node NVLink
    info.size = resources->exportSize = recvSize;
    info.type = P2P_MULTINODE;

    NCCLCHECK(allocateShareableBuffer(myInfo->cudaDev, recvSize, &info.desc, &resources->exportHandle, (void **)&resources->devMem));
    TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] <- %d[%d] via P2P/MNNVL", channelId, myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev);
  } else {
    NCCLCHECK(ncclCudaCalloc((char**)&resources->devMem, recvSize));
    if (myInfo->pidHash == peerInfo->pidHash) {
      info.type = P2P_DIRECT;
      info.directPtr = resources->devMem;
      if (myInfo->cudaDev == peerInfo->cudaDev) {
        TRACE(NCCL_INIT|NCCL_P2P,"%d <- %d via P2P/common device", myInfo->rank, peerInfo->rank);
      } else {
        // Enable P2P access
        cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled) {
          cudaGetLastError();
        } else if (err != cudaSuccess) {
          WARN("failed to peer with device %d(=%lx): %d %s",
               peerInfo->cudaDev, peerInfo->busId, err, cudaGetErrorString(err));
          return ncclInternalError;
        }
        TRACE(NCCL_INIT|NCCL_P2P,"Channel %02d : %d[%lx] <- %d[%lx] via P2P/direct pointer", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
      }
    } else {
      // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
      int peerCudaDev = busIdToCudaDev(peerInfo->busId);
      info.type = P2P_IPC;
      // Map IPC and enable P2P access
      cudaError_t err = cudaIpcGetMemHandle(&info.devIpc, (void*)resources->devMem);
      if (err != cudaSuccess) {
        WARN("rank %d failed to get CUDA IPC handle to device %d(=%lx) : %d %s",
               myInfo->rank, peerCudaDev, peerInfo->busId, err, cudaGetErrorString(err));
        return ncclInternalError;
        }
      TRACE(NCCL_INIT|NCCL_P2P,"Channel %02d : %d[%lx] <- %d[%lx] via P2P/IPC", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
      //TRACE_DUMP_IPC(&info.devIpc);
    }
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

#ifdef MNNVL_SUPPORT
// MNNVL: Multi-node NVLink support
static ncclResult_t importShareableBuffer(int device, size_t size,
                                          CUmemFabricHandle *desc, CUmemGenericAllocationHandle *handle, void **devMemPtr) {
  CUmemAccessDesc accessDesc;
  CUdeviceptr dptr = 0;

  init_etbl();

  INFO(NCCL_P2P, "Importing shareable buffer device %d size %zi", device, size);

  // Import and map the remote memory descriptor to the local GPU
  CUDACHECK_DEV(cuExperimentalMemImportFromShareableHandle(handle, desc, CU_MEM_HANDLE_TYPE_FABRIC));
  CUDACHECK_DEV(cuExperimentalMemAddressReserve(&dptr, size, /* alignment */ 0, /* addr */ 0, /* flags */ 0));
  CUDACHECK_DEV(cuExperimentalMemMap(dptr, size, /* offset */ 0, *handle, /* flags */ 0));
  INFO(NCCL_P2P, "Imported shareable buffer device %d size %zi handle 0x%x dptr %p", device, size, *handle, dptr);

  // Allow access by the local GPU
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUDACHECK_DEV(cuExperimentalMemSetAccess(dptr, size, &accessDesc, 1));

  *devMemPtr = (void *)dptr;

  return ncclSuccess;
}

static ncclResult_t unimportShareableBuffer(void *buff, size_t size, CUmemGenericAllocationHandle handle) {
  CUdeviceptr dptr = (CUdeviceptr) buff;

  INFO(NCCL_P2P, "Unimport shareable buffer %p size %zi handle 0x%x", buff, size, handle);

  CUDACHECK_DEV(cuExperimentalMemUnmap(dptr, size));
  CUDACHECK_DEV(cuExperimentalMemAddressFree(dptr, size));
  CUDACHECK_DEV(cuExperimentalMemRelease(handle));

  return ncclSuccess;
}
#endif

/* Connect/Send to this peer */
static ncclResult_t p2pSendConnect(struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct p2pSendResources* resources = (struct p2pSendResources*)send->transportResources;
  struct ncclRecvMem* remDevMem = NULL;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  resources->type = info->type;
  if (info->type == P2P_DIRECT) {
    remDevMem = (struct ncclRecvMem*)(info->directPtr);
    if (info->read == 0) send->conn.direct |= NCCL_DIRECT_GPU;
  } else if (info->type == P2P_IPC) {
    //TRACE_DUMP_IPC(&info->devIpc);
    cudaError_t err = cudaIpcOpenMemHandle(&resources->remotePtr, info->devIpc, cudaIpcMemLazyEnablePeerAccess);
    remDevMem = (struct ncclRecvMem*)resources->remotePtr;
    if (err != cudaSuccess) {
      WARN("failed to open CUDA IPC handle : %d %s",
          err, cudaGetErrorString(err));
      return ncclUnhandledCudaError;
    }
  } else if (info->type == P2P_MULTINODE) {
    // MNNVL: multi-node NVLink
    NCCLCHECK(importShareableBuffer(send->comm->cudaDev, info->size, &info->desc, &resources->importHandle, &resources->remotePtr));
    remDevMem = (struct ncclRecvMem*)resources->remotePtr;
    resources->importSize = info->size;
  }

  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      /* For P2P Read the SIMPLE buffer is local (ncclSendMem) */
      send->conn.buffs[p] = resources->devMem->buff;
    } else {
      send->conn.buffs[p] = remDevMem->buff + offset;
      offset += send->comm->buffSizes[p];
    }
  }
  send->conn.tail = &remDevMem->tail;
  send->conn.opCountRem = &remDevMem->opCount;
  send->conn.head = &resources->devMem->head;
  send->conn.ptrExchange = &resources->devMem->ptrExchange;
  send->conn.opCountLoc = &resources->devMem->opCount;
  return ncclSuccess;
}

/* Connect/Recv from this peer */
ncclResult_t p2pRecvConnect(struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct p2pRecvResources* resources = (struct p2pRecvResources*)recv->transportResources;
  struct ncclSendMem* remDevMem = NULL;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  resources->type = info->type;
  if (info->type == P2P_DIRECT) {
    remDevMem = (struct ncclSendMem*)(info->directPtr);
    if (info->read == 0) {
      recv->conn.direct |= NCCL_DIRECT_GPU;
      recv->conn.ptrExchange = &remDevMem->ptrExchange;
    }
  } else if (info->type == P2P_IPC) {
    //TRACE_DUMP_IPC(&info->devIpc);
    cudaError_t err = cudaIpcOpenMemHandle(&resources->remotePtr, info->devIpc, cudaIpcMemLazyEnablePeerAccess);
    remDevMem = (struct ncclSendMem*)resources->remotePtr;
    if (err != cudaSuccess) {
      WARN("failed to open CUDA IPC handle : %d %s",
          err, cudaGetErrorString(err));
      return ncclUnhandledCudaError;
    }
  } else if (info->type == P2P_MULTINODE) {
    NCCLCHECK(importShareableBuffer(recv->comm->cudaDev, info->size, &info->desc, &resources->importHandle, &resources->remotePtr));
    remDevMem = (struct ncclSendMem*)resources->remotePtr;
    resources->importSize = info->size;
  }

  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      /* For P2P Read the SIMPLE buffer is remote (ncclSendMem) */
      recv->conn.buffs[p] = remDevMem->buff;
    } else {
      recv->conn.buffs[p] = resources->devMem->buff + offset;
      offset += recv->comm->buffSizes[p];
    }
  }
  recv->conn.tail = &resources->devMem->tail;
  recv->conn.opCountLoc = &resources->devMem->opCount;
  recv->conn.head = &remDevMem->head;
  recv->conn.opCountRem = &remDevMem->opCount;
  return ncclSuccess;
}

ncclResult_t p2pSendFree(void* resources) {
  struct p2pSendResources* sendRes = (struct p2pSendResources*)resources;
  if (sendRes->type == P2P_MULTINODE) {
    // Multi-node NVLink
    NCCLCHECK(unimportShareableBuffer(sendRes->remotePtr, sendRes->importSize, sendRes->importHandle));
    NCCLCHECK(freeShareableBuffer(sendRes->devMem, sendRes->exportSize, sendRes->exportHandle));
  }
  else {
    if (sendRes->type == P2P_IPC)
      CUDACHECK(cudaIpcCloseMemHandle(sendRes->remotePtr));
    CUDACHECK(cudaFree(sendRes->devMem));
  }
  free(sendRes);
  return ncclSuccess;
}

ncclResult_t p2pRecvFree(void* resources) {
  struct p2pRecvResources* recvRes = (struct p2pRecvResources*)resources;
  if (recvRes->type == P2P_MULTINODE) {
    // Multi-node NVLink
    NCCLCHECK(unimportShareableBuffer(recvRes->remotePtr, recvRes->importSize, recvRes->importHandle));
    NCCLCHECK(freeShareableBuffer(recvRes->devMem, recvRes->exportSize, recvRes->exportHandle));
  }
  else {
    if (recvRes->type == P2P_IPC)
      CUDACHECK(cudaIpcCloseMemHandle(recvRes->remotePtr));
    CUDACHECK(cudaFree(recvRes->devMem));
  }
  free(recvRes);
  return ncclSuccess;
}

struct ncclTransport p2pTransport = {
  "P2P",
  p2pCanConnect,
  { p2pSendSetup, p2pSendConnect, p2pSendFree, NULL },
  { p2pRecvSetup, p2pRecvConnect, p2pRecvFree, NULL }
};
