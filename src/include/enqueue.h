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
struct ncclEnqueueElem {
  struct ncclWorkElem work;
  struct ncclProxyArgs proxyArgs;
  struct ncclEnqueueElem* next;
};

// Store enqueue elements in a list
struct ncclEnqueueElemList {
  struct ncclEnqueueElem* head;
  struct ncclEnqueueElem* tail;
};

// Structure passed to CUDA graph
struct ncclEnqueueInfo {
  ncclComm_t comm;
  int maxChannels;    // Dynamic version of gridDim
  ncclResult_t ret;   // Return value of host setup call
  struct ncclEnqueueElemList eqElemList;
};

// Get next element from enqueue list
static ncclResult_t getNewEnqueueElem(struct ncclEnqueueInfo* eqInfo, struct ncclEnqueueElem** elemOut) {
  if (eqInfo == NULL) return ncclInternalError;
  struct ncclEnqueueElemList* list = &eqInfo->eqElemList;
  struct ncclEnqueueElem* next;
  NCCLCHECK(ncclCalloc(&next, 1));
  *elemOut = next;
  if (list->tail != NULL) list->tail->next = next;
  list->tail = next;
  if (list->head == NULL) list->head = next;
  return ncclSuccess;
}

// Destroy enqueue info space
// used by both CUDA graph and non CUDA graph
static void destroyEnqueueInfo(void* ptr) {
  if (ptr == NULL) return;
  struct ncclEnqueueInfo* eqInfo = (struct ncclEnqueueInfo*)ptr;
  struct ncclEnqueueElem* head = eqInfo->eqElemList.head;
  while (head != NULL) {
    struct ncclEnqueueElem* temp = head;
    head = head->next;
    free(temp);
  }
  free(eqInfo);
}
#endif // End include guard
