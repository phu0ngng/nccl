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
ncclResult_t ncclLaunchReset(ncclComm_t comm);
ncclResult_t ncclSetupP2pKernel(struct ncclInfo* info);
ncclResult_t ncclSetupAsyncKernels(struct ncclComm* comm);
template<int USING_CUDA_GRAPH>
void CUDART_CB ncclEnqueueHostSetup(void* arg);
ncclResult_t ncclGetCudaGraph(ncclComm_t comm, cudaGraph_t* graph);
ncclResult_t ncclCudaGraphHostSetup(ncclComm_t comm, cudaGraph_t graph);

#endif // End include guard
