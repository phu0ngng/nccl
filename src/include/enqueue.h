/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ENQUEUE_H_
#define NCCL_ENQUEUE_H_

#include "comm.h"
#include "group.h"
#include "collectives.h"

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ncclCpuBarrierIn(struct ncclComm* comm, int* isLast);
ncclResult_t ncclCpuBarrierLast(struct ncclComm* comm);
ncclResult_t ncclCpuBarrierOut(struct ncclComm* comm);
ncclResult_t ncclLaunchBarrier(struct ncclComm* comm);
ncclResult_t ncclLaunch(ncclComm_t comm);
ncclResult_t ncclRecordEvents(struct ncclComm* comm);
ncclResult_t ncclLaunchReset(ncclComm_t comm, int destroyInfo);
ncclResult_t ncclSetupP2pKernel(struct ncclInfo* info);
ncclResult_t ncclSetupAsyncKernels(struct ncclComm* comm);
template<int USING_CUDA_GRAPH>
void CUDART_CB ncclEnqueueHostSetup(void* arg);
ncclResult_t ncclGetCudaGraph(ncclComm_t comm, cudaGraph_t* graph);
ncclResult_t ncclCudaGraphHostSetup(ncclComm_t comm, cudaGraph_t graph);

struct ncclCudaGraphElem {
  struct ncclWorkElem work;
  struct ncclProxyArgs proxyArgs;
  struct ncclCudaGraphElem* next;
};

struct ncclCgElemList {
  struct ncclCudaGraphElem* head;
  struct ncclCudaGraphElem* tail;
};

struct ncclCudaGraphInfo {
  ncclComm_t comm;
  int maxChannels;
  ncclResult_t ret;
  struct ncclCgElemList cgElemList;
};

static ncclResult_t getNewCudaGraphElem(struct ncclCudaGraphInfo* cgInfo, struct ncclCudaGraphElem** elemOut) {
  if (cgInfo == NULL) return ncclInternalError;
  struct ncclCgElemList* list = &cgInfo->cgElemList;
  struct ncclCudaGraphElem* next;
  NCCLCHECK(ncclCalloc(&next, 1));
  *elemOut = next;
  if (list->tail != NULL) list->tail->next = next;
  list->tail = next;
  if (list->head == NULL) list->head = next;
  return ncclSuccess;
}

static void destroyCudaGraphInfo(void* ptr) {
  if (ptr == NULL) return;
  struct ncclCudaGraphInfo* cgInfo = (struct ncclCudaGraphInfo*)ptr;
  struct ncclCudaGraphElem* head = cgInfo->cgElemList.head;
  while (head != NULL) {
    struct ncclCudaGraphElem* temp = head;
    head = head->next;
    free(temp);
  }
  free(cgInfo);
}
#endif // End include guard
