/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "collectives.h"
#include "reduce.h"

ncclResult_t ncclReduceFunc(const void* sendbuff, void* recvbuff, const size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int nbytes = count*ncclTypeSize(datatype);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, nbytes, cudaMemcpyDeviceToDevice, stream));
  } else {
    ArgsSetup(sendbuff, recvbuff, root, count, comm);
    if (nbytes <= comm->llThreshold) {
      NCCLCHECK(transportSaveProxies(1, NUM_LL_CHUNKS, 1, 1, 2*nbytes, proxyPatternTo(root), comm, 1));
      saveKernel(ncclCollReduce, op, datatype, nbytes, comm, stream, 1);
    } else {
      NCCLCHECK(transportSaveProxies(NUM_SUBSTEPS, NUM_BUFCHUNKS, 1, 1, nbytes, proxyPatternTo(root), comm, 0));
      saveKernel(ncclCollReduce, op, datatype, nbytes, comm, stream, 0);
      comm->opCount++;
    }
  }

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclReduceFunc, "Reduce", sendbuff, recvbuff, count, datatype,
      op, root, comm, stream);
}
