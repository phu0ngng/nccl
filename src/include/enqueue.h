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
ncclResult_t ncclGetCudaGraph(ncclComm_t comm, cudaGraph_t* graph, int* usingCudaGraph);
ncclResult_t ncclCudaGraphHostSetup(ncclComm_t comm, cudaGraph_t graph);

// Enqueue information (for kernel and proxy) for each operation
struct ncclQueueElem {
  struct ncclWorkElem work;
  struct ncclProxyArgs proxyArgs;
  struct ncclQueueElem* next;
};

// Store enqueue elements in a list
struct ncclQueueElemList {
  struct ncclQueueElem* head;
  struct ncclQueueElem* tail;
};

// Structure passed to CUDA graph
struct ncclQueueInfo {
  ncclComm_t comm;
  int maxChannels;    // Dynamic version of gridDim
  ncclResult_t ret;   // Return value of host setup call
  struct ncclQueueElemList elemList;
};

// Get next element from enqueue list
static ncclResult_t ncclAddQueueElem(struct ncclQueueInfo* eqInfo, struct ncclQueueElem** elemOut) {
  if (eqInfo == NULL) return ncclInternalError;
  struct ncclQueueElemList* list = &eqInfo->elemList;
  struct ncclQueueElem* next;
  NCCLCHECK(ncclCalloc(&next, 1));
  *elemOut = next;
  if (list->tail != NULL) list->tail->next = next;
  list->tail = next;
  if (list->head == NULL) list->head = next;
  return ncclSuccess;
}

// Destroy enqueue info space
// used by both CUDA graph and non CUDA graph
static void ncclDestroyQueueInfo(void* ptr) {
  if (ptr == NULL) return;
  struct ncclQueueInfo* eqInfo = (struct ncclQueueInfo*)ptr;
  struct ncclQueueElem* head = eqInfo->elemList.head;
  while (head != NULL) {
    struct ncclQueueElem* temp = head;
    head = head->next;
    free(temp);
  }
  free(eqInfo);
}
#endif // End include guard
