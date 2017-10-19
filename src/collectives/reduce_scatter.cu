/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "collectives.h"
#include "reduce_scatter.h"

ncclResult_t ncclReduceScatterFunc(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int nbytes = count*ncclTypeSize(datatype);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, nbytes, cudaMemcpyDeviceToDevice, stream));
  } else {
    if (nbytes*comm->nRanks <= comm->llThreshold) {
      NCCLCHECK(transportSaveProxies(1, NUM_LL_CHUNKS, comm->nRanks-1, 1, 2*nbytes, proxyPatternRing, comm, 1));
      NCCLCHECK(saveKernel(ncclCollReduceScatter, sendbuff, recvbuff, count, datatype, op, root, comm, stream, nbytes*comm->nRanks, 1));
    } else {
      NCCLCHECK(transportSaveProxies(NUM_SUBSTEPS, NUM_BUFCHUNKS, comm->nRanks-1, 1, nbytes, proxyPatternRing, comm, 0));
      NCCLCHECK(saveKernel(ncclCollReduceScatter, sendbuff, recvbuff, count, datatype, op, root, comm, stream, nbytes*comm->nRanks, 0));
      comm->opCount++;
    }
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclReduceScatterFunc, "ReduceScatter", sendbuff, recvbuff, recvcount, datatype,
      op, 0, comm, stream);
}
