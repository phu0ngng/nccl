/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "utils.h"
#include "topo.h"
#include "transport.h"
#include <unistd.h>
#include <cuda_runtime.h>
#include "nvmlwrap.h"
#include <ctype.h>
#include "nvlink.h"

#define MAXNVLINKS 8

struct p2pInfo {
  int rank;
  int cudaDev;
  int pid;
  uint64_t hostHash;
  int hostNumber;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
};

struct p2pConnectInfo {
  int direct;
  union {
    struct ncclSendRecvMem* directPtr;
    cudaIpcMemHandle_t devIpc;
  };
};

#include <sys/types.h>

/* Fill information necessary to exchange between ranks to choose whether or not
 * to use this transport */
ncclResult_t p2pFillInfo(ncclTinfo_t* opaqueInfo, int rank) {
  struct p2pInfo* info = (struct p2pInfo*)opaqueInfo;
  static_assert(sizeof(struct p2pInfo) <= sizeof(ncclTinfo_t), "p2p Info too large");
  info->rank = rank;
  CUDACHECK(cudaGetDevice(&info->cudaDev));
  info->pid = getpid();
  char hostname[1024];
  getHostName(hostname, 1024);
  info->hostHash=getHostHash(hostname);
  info->hostNumber=getHostNumber(hostname);
  CUDACHECK(cudaDeviceGetPCIBusId(info->busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, info->cudaDev));
  for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
    if (info->busId[c] == 0) break;
    info->busId[c] = tolower(info->busId[c]);
  }
  return ncclSuccess;
}

/* Determine if we can communicate with the peer */
ncclResult_t p2pCanConnect(int* ret, ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo) {
  struct p2pInfo* myInfo = (struct p2pInfo*)myOpaqueInfo;
  struct p2pInfo* peerInfo = (struct p2pInfo*)peerOpaqueInfo;
  int p2p = 0;
  if (myInfo->hostHash == peerInfo->hostHash) {
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      p2p = 1;
    } else {
      if (cudaDeviceCanAccessPeer(&p2p, myInfo->cudaDev, peerInfo->cudaDev) != cudaSuccess) {
        INFO("peer query failed between dev %d and dev %d",
          myInfo->cudaDev, peerInfo->cudaDev);
        p2p = 0;
      }
      if (p2p == 1) {
        int nlinks = getNvlinkGpu(myInfo->busId, peerInfo->busId);
        if (nlinks > 0) {
          p2p = nlinks*CONNECT_NVLINK;
        } else if (nlinks < 0) {
          p2p = (-nlinks)*CONNECT_NVSWITCH;
        } else {
          char* myPath;
          char* peerPath;
          ncclResult_t err1 = getCudaPath(myInfo->cudaDev, &myPath);
          ncclResult_t err2 = getCudaPath(peerInfo->cudaDev, &peerPath);
          if (err1 == ncclSuccess && err2 == ncclSuccess) {
            p2p += PATH_SOC - pciDistance(myPath, peerPath);
            if (err1 == ncclSuccess) free(myPath);
            if (err2 == ncclSuccess) free(peerPath);
          }
        }
      }
    }
  }
  *ret = p2p;
  return ncclSuccess;
}

static int computeRingsRec(int* matrix, int n, int *rings, int currentRing, int nRingsMax, int* inTheRing, int current, int remaining, int connect) {
  int nrings = 0;
  int* line = matrix+current*n;
  inTheRing[current] = 1;
  int currentStep = (currentRing+1)*n-remaining;
  rings[currentStep-1] = current;
  if (remaining == 0) {
    int looprank = rings[currentRing*n];
    if (line[looprank] > 0) {
      if (currentRing+1 == nRingsMax) {
        nrings = 1;
      } else {
	line[looprank]--;
	for (int i=0; i<n; i++) inTheRing[i] = 0;
        if (connect) {
          // First two slots are already set and we need to respect those constraints
          inTheRing[rings[currentStep]] = 1;
	  nrings = 1 + computeRingsRec(matrix, n, rings, currentRing+1, nRingsMax, inTheRing, rings[currentStep+1], n-2, connect);
        } else {
          rings[(currentRing+1)*n] = 0;
	  nrings = 1 + computeRingsRec(matrix, n, rings, currentRing+1, nRingsMax, inTheRing, 0, n-1, connect);
        }
	line[looprank]++;
	for (int i=0; i<n; i++) inTheRing[i] = 1;
      }
    }
  } else {
    int ringsSave[nRingsMax*n];
    int maxStep = 0;
    for (int i=0; i<n; i++) {
      if (inTheRing[i] == 0 && line[i] > 0) {
        line[i]--;
        int nr = computeRingsRec(matrix, n, rings, currentRing, nRingsMax, inTheRing, i, remaining-1, connect);
        if (nr > nrings) {
          nrings = nr;
          maxStep = (nr+currentRing)*n;
          ringsSave[currentStep] = i;
          // Save the rest of the rings
          for (int r=currentStep+1; r<maxStep; r++) {
            ringsSave[r] = rings[r];
          }
          if (nrings + currentRing == nRingsMax) {
            // We found an optimal solution. Let's stop there.
            break;
          }
        }
        line[i]++;
      }
    }
    for (int r=currentStep; r<maxStep; r++) {
      rings[r] = ringsSave[r];
    }
  }
  inTheRing[current] = 0;
  return nrings;
}

int p2pComputeRingsNvlink(int* matrix, int nranks, int *rings, int nringsMax, int connect) {
  int* inTheRing = (int*)malloc(sizeof(int)*nranks);
  for (int i=0; i<nranks; i++) inTheRing[i] = 0;
  int nrings;
  if (connect) {
    inTheRing[rings[0]] = 1;
    nrings = computeRingsRec(matrix, nranks, rings, 0, nringsMax, inTheRing, rings[1], nranks-2, connect);
  } else {
    rings[0] = 0;
    nrings = computeRingsRec(matrix, nranks, rings, 0, nringsMax, inTheRing, 0, nranks-1, connect);
  }
  free(inTheRing);
  return nrings;
}

static inline int findConnect(int nranks, int* ranks) {
  for (int i = 0; i<nranks; i++) {
    if (ranks[i] != -1) return i;
  }
  return -1;
}

int p2pComputeRingsNvlink(int* values, int nranks, int* rings, int nrings, int* prev, int* next, int oversubscribe, int* nthreads) {
  if (nrings == 0) return 0;
  if (nrings > MAXRINGS) {
    WARN("Max rings reached, limiting to %d\n", MAXRINGS);
    nrings = MAXRINGS;
  }
  // Find existing constraints / connections
  int connect = 0;
  for (int r=0; r<nrings; r++) {
    int start = findConnect(nranks, prev+r*nranks);
    int end = findConnect(nranks, next+r*nranks);
    if (start != -1 && end != -1) {
      rings[r*nranks] = end;
      rings[r*nranks+1] = start;
      connect = 1;
    }
  }
  // Compute rings
  int matrix[nranks*nranks];
  for (int i=0; i<nranks; i++) for (int j=0; j<nranks; j++)
    matrix[i*nranks+j] = oversubscribe ? values[i*nranks+j]/CONNECT_NVLINK*2 : values[i*nranks+j]/CONNECT_NVLINK ;

  int compNrings = p2pComputeRingsNvlink(matrix, nranks, rings, nrings, connect);
  if (connect == 0) {
    if (oversubscribe == 0 && compNrings && compNrings < nrings && nranks <= 4) {
      // Try to oversubscribe to get a better result
      int rings2[MAXRINGS*nranks];
      for (int i=0; i<MAXRINGS*nranks; i++) rings2[i] = -1;
      int compNrings2 = p2pComputeRingsNvlink(values, nranks, rings2, nrings*2, prev, next, 1, nthreads);
      if (compNrings2 > compNrings*2) {
        // Oversubscription worked.
        for (int i=0; i<compNrings2*nranks; i++) rings[i] = rings2[i];
        INFO("Oversubscribing, original nrings = %d, new nrings = %d", compNrings, compNrings2);
        return compNrings2;
      }
    }
    // Duplicate the rings for NVLink alone
    for (int r=0; r<compNrings; r++) {
      for (int i=0; i<nranks; i++) rings[(r+compNrings)*nranks+i] = rings[r*nranks+i];
    }
    compNrings *= 2;
    *nthreads = *nthreads >> 1;
    INFO("Doubling rings to %d, halving threads to %d", compNrings, *nthreads);
  }
  return compNrings;
}

int p2pComputeRingsSeqConnect(int* values, int nranks, int* rings, int nringsStart, int* prev, int* next, int minScore, int* nthreads) {
  int nrings = nringsStart;
  int connect = 0;
  for (int r=0; r<nrings; r++) {
    int start = findConnect(nranks, prev+r*nranks);
    int end = findConnect(nranks, next+r*nranks);
    if (start != -1 && end != -1) {
      rings[r*nranks] = end;
      rings[r*nranks+1] = start;
      int cur = start;
      for (int i=2; i<nranks; i++) {
        int next = (cur+1) % nranks;
        while (next == end || next == start) next = (next+1) % nranks;
        if (values[cur*nranks+next] < minScore) {
          return 0;
        }
        rings[r*nranks+i] = next;
        cur = next;
      }
      connect = 1;
    } else {
      if (connect == 1 && r > 0) {
        WARN("Connecting rings but did not find start/end for ring %d. Disabling other rings.", r);
        return r;
      } else {
        return 0;
      }
    }
  }
  return nrings;
}

int p2pComputeRingsSeqNew(int* values, int nranks, int* rings, int nringsStart, int* prev, int* next, int minScore, int* nthreads) {
  for (int r=0; r<nringsStart; r++) {
    for (int i=0; i<nranks; i++) {
      if (r % 2 == 0)
        rings[r*nranks+i] = i;
      else
        rings[r*nranks+i] = nranks-1-i;
    }
  }
  return nringsStart;
}

ncclResult_t p2pGetRings(int nranks, int* groups, int* subgroups, int* values, int* nringsRet, int* prev, int* next, int minScore, int* nthreads) {
  if (*nringsRet == 0) return ncclSuccess;
  int rings[MAXRINGS*nranks];
  for (int i=0; i<MAXRINGS*nranks; i++) rings[i] = -1;
  int nrings = *nringsRet;

  // NVswitch
  for (int rank=0; rank<nranks; rank++) {
    int links = 0;
    for (int j=1; j<nranks; j++) {
      int i = (rank + j) % nranks;
      int nvswitch_links = values[rank*nranks+i]/CONNECT_NVSWITCH;
      if (j>1 && links != nvswitch_links) {
        WARN("Internal error : NVswitch links mismatch");
        return ncclInternalError;
      }
      links = nvswitch_links;
    }
    nrings = min(nrings, links);
    if (nrings > 0) {
      int nringsConnect = p2pComputeRingsSeqConnect(values, nranks, rings, nrings, prev, next, minScore, nthreads);
      if (nringsConnect > 0) {
        nrings = nringsConnect;
      } else {
        nrings = p2pComputeRingsSeqNew(values, nranks, rings, nrings, prev, next, minScore, nthreads);;
      }
    }
  }

  if (nrings == 0) {
    nrings = *nringsRet;
    // point-to-point NVLink
    for (int rank=0; rank<nranks; rank++) {
      int nr = 0;
      for (int i=0; i<nranks; i++) {
        int val = values[rank*nranks+i];
        if (val >= CONNECT_NVSWITCH) continue;
        nr += val/CONNECT_NVLINK;
      }
      nrings = min(nrings, nr);
    }
    if (nrings > 0) nrings = p2pComputeRingsNvlink(values, nranks, rings, nrings, prev, next, 0, nthreads);
  }
 
  if (nrings == 0) {
    nrings = *nringsRet;
    // PCI or QPI
    int nringsConnect = p2pComputeRingsSeqConnect(values, nranks, rings, nrings, prev, next, minScore, nthreads);
    if (nringsConnect > 0) {
      nrings = nringsConnect;
    } else {
      nrings = p2pComputeRingsSeqNew(values, nranks, rings, 1, prev, next, minScore, nthreads);;
    }
  }

  // Duplicate the rings
  char* str = getenv("NCCL_DUP_RINGS");
  if (str && strlen(str) > 0) {
    int dup = atoi(str);
    INFO("Duplicating rings by %d", dup);
    for (int d=1; d<dup; d++) {
      for (int r=0; r<nrings; r++) {
        for (int i=0; i<nranks; i++) rings[(r+d*nrings)*nranks+i] = rings[r*nranks+i];
      }
    }
    nrings *= dup;
  }

  *nringsRet = nrings;
  for (int ring = 0; ring<nrings; ring++) {
    for (int index=0; index<nranks; index++) {
      int prevIndex = (index - 1 + nranks) % nranks;
      int nextIndex = (index + 1) % nranks;
      int curRank = rings[ring*nranks+index];
      int prevRank = rings[ring*nranks+prevIndex];
      int nextRank = rings[ring*nranks+nextIndex];
      if (prev[ring*nranks+curRank] == -1) prev[ring*nranks+curRank] = prevRank;
      if (next[ring*nranks+curRank] == -1) next[ring*nranks+curRank] = nextRank;
    }
  }
  
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct p2pInfo* myInfo = (struct p2pInfo*)myOpaqueInfo;
  struct p2pInfo* peerInfo = (struct p2pInfo*)peerOpaqueInfo;
  struct p2pConnectInfo info;
  if (myInfo->pid == peerInfo->pid) {
    info.direct = 1;
    info.directPtr = ring->devMem;
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      INFO("%d -> %d via P2P/common device", myInfo->rank, peerInfo->rank);
    } else {
      // Enable P2P access
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d: %s",
            peerInfo->cudaDev, cudaGetErrorString(err));
        return ncclInternalError;
      }
      INFO("%d -> %d via P2P/direct pointer", myInfo->rank, peerInfo->rank);
    }
  } else {
    info.direct = 0;
    // Map IPC and enable P2P access
    if (cudaIpcGetMemHandle(&info.devIpc, (void*)ring->devMem) != cudaSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d", myInfo->rank, peerInfo->cudaDev);
      return ncclInternalError;
    }
    INFO("%d -> %d via P2P/IPC", myInfo->rank, peerInfo->rank);
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

static ncclResult_t p2pConnect(struct ncclConnect* connectInfo, struct ncclConnector* connector, struct ncclSendRecvMem** remDevMem) {
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  if (info->direct) {
    *remDevMem = info->directPtr;
    connector->conn.direct = 1;
    connector->conn.ptrExchange = &((*remDevMem)->ptrExchange);
  } else {
    cudaError_t err = cudaIpcOpenMemHandle((void**)remDevMem,
          info->devIpc, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
      WARN("failed to open CUDA IPC handle : %s",
          cudaGetErrorString(err));
      return ncclUnhandledCudaError;
    }
  }
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t p2pSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  struct ncclSendRecvMem* remDevMem;
  NCCLCHECK(p2pConnect(connectInfo, send, &remDevMem));
  send->conn.buff = remDevMem->buff;
  send->conn.tail = &remDevMem->tail;
  send->conn.opCount = &remDevMem->opCount;
  // send->conn->head should have been set to devMem already
  return ncclSuccess;
}

ncclResult_t p2pRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  struct ncclSendRecvMem* remDevMem;
  NCCLCHECK(p2pConnect(connectInfo, recv, &remDevMem));
  // recv->conn->buff should have been set to devMem already
  // recv->conn->tail should have been set to devMem already
  // recv->conn->opCount should have been set to devMem already
  recv->conn.head = &remDevMem->head;
  return ncclSuccess;
}

ncclResult_t p2pFree(void* resources) {
  return ncclSuccess;
}

struct ncclTransport p2pTransport = {
  "P2P",
  p2pFillInfo,
  p2pCanConnect,
  p2pGetRings,
  { p2pSetup, p2pSendConnect, p2pFree, NULL },
  { p2pSetup, p2pRecvConnect, p2pFree, NULL }
};


