/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ENQUEUE_H_
#define NCCL_ENQUEUE_H_

#include "comm.h"
#include "group.h"
#include "collectives.h"

#define NCCL_MIN_CHANNEL_SIZE (NCCL_LL_THREAD_THRESHOLD*64)
#define NCCL_AGG_CHANNEL_SIZE (1LL << 21) /* 2 MiB, ideal per-channel size to fully utilize bandwidth */

size_t ncclKernMaxLocalSize();
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ncclCpuBarrierIn(struct ncclComm* comm, int* isLast);
ncclResult_t ncclCpuBarrierLast(struct ncclComm* comm);
ncclResult_t ncclCpuBarrierOut(struct ncclComm* comm);
ncclResult_t ncclLaunchBarrier(struct ncclComm* comm);
ncclResult_t ncclLaunchKernel(ncclComm_t comm);
ncclResult_t ncclRecordEvents(struct ncclComm* comm);
ncclResult_t ncclLaunchReset(ncclComm_t comm);
ncclResult_t ncclSetupP2pKernel(struct ncclInfo* info);
ncclResult_t ncclSetupAsyncKernels(struct ncclComm* comm);
template<int USING_CUDA_GRAPH>
void CUDART_CB ncclEnqueueHostSetup(void* arg);
ncclResult_t ncclGetCudaGraph(ncclComm_t comm, cudaGraph_t* graph);
ncclResult_t ncclCudaGraphHostSetup(ncclComm_t comm, cudaGraph_t graph);

struct ncclBuffRegInfo {
  void* sendbuffsBase[NCCL_MAX_INTRA_RANKS];
  void* recvbuffsBase[NCCL_MAX_INTRA_RANKS];
  void* sendbuffs[NCCL_MAX_INTRA_RANKS];
  void* recvbuffs[NCCL_MAX_INTRA_RANKS];
  int nBuffs;
};

// Enqueue information (for kernel and proxy) for each operation
struct ncclQueueElem {
  struct ncclWorkElem work;
  struct ncclProxyArgs proxyArgs;
  struct ncclBuffRegInfo buffRegInfo;
};

typedef ncclRecyclableList<struct ncclQueueElem> ncclQueueElemList;

// Structure passed to CUDA graph
struct ncclQueueInfo {
  ncclComm_t comm;
  int maxChannels;    // Dynamic version of gridDim
  ncclResult_t ret;   // Return value of host setup call
  ncclQueueElemList* elemList;
};

static ncclResult_t ncclCreateQueueInfo(struct ncclQueueInfo** eqInfo, ncclComm_t comm) {
  NCCLCHECK(ncclCalloc(eqInfo, 1));
  (*eqInfo)->comm = comm;
  (*eqInfo)->elemList = new ncclQueueElemList();
  return ncclSuccess;
}

// Reset element queue
static ncclResult_t ncclResetQueueInfo(struct ncclQueueInfo* eqInfo) {
  if (eqInfo == NULL) return ncclInternalError;
  eqInfo->maxChannels = 0;
  eqInfo->ret = ncclSuccess;
  eqInfo->elemList->recycle();
  return ncclSuccess;
}

// Destroy enqueue info space
// used by both CUDA graph and non CUDA graph
static void ncclDestroyQueueInfo(void* ptr) {
  if (ptr == NULL) return;
  struct ncclQueueInfo* eqInfo = (struct ncclQueueInfo*)ptr;
  struct ncclComm* comm = eqInfo->comm;
  // Close IPC mem handles for registered buffers
  struct ncclQueueElem* eqElem = eqInfo->elemList->begin();
#if 0
  // Ideally, the deregistration should happen here
  // but currently the destroy function of CUDA objects does not allow CUDA API calls
  while (eqElem != NULL) {
    for (int i=0; i<eqElem->buffRegInfo.nBuffs; i++) {
      if (i == eqInfo->comm->intraNodeRank) continue;
      CUDACHECKIGNORE(cudaIpcCloseMemHandle(eqElem->buffRegInfo.sendbuffsBase[i]));
      CUDACHECKIGNORE(cudaIpcCloseMemHandle(eqElem->buffRegInfo.recvbuffsBase[i]));
    }
    eqElem = eqInfo->elemList->getNext();
  }
#else
  // Instead, we push these pointers to a pool owned by ncclComm
  // and asks a helper thread to close mem handles
  struct ncclGraphHelperResources* res = comm->graphHelperResources;
  volatile int* ipcCount = &res->ipcCount;
  pthread_mutex_lock(&res->threadLock);
  while (eqElem != NULL) {
    if (eqElem->buffRegInfo.nBuffs > 0) {
      memcpy(res->ipcBases+(*ipcCount), eqElem->buffRegInfo.sendbuffsBase,
          eqElem->buffRegInfo.nBuffs*sizeof(void*));
      (*ipcCount) += eqElem->buffRegInfo.nBuffs;
      memcpy(res->ipcBases+(*ipcCount), eqElem->buffRegInfo.recvbuffsBase,
          eqElem->buffRegInfo.nBuffs*sizeof(void*));
      (*ipcCount) += eqElem->buffRegInfo.nBuffs;
    }
    eqElem = eqInfo->elemList->getNext();
  }
  if (*ipcCount > 0) {
    res->threadState = ThreadStart;
    INFO(NCCL_COLL, "CUDA Graph destroy function signaling helper thread with %d IPC handles", *ipcCount);
    pthread_cond_signal(&res->threadCond);
  }
  pthread_mutex_unlock(&res->threadLock);
#endif
  delete eqInfo->elemList;
  free(eqInfo);
}
#endif // End include guard
