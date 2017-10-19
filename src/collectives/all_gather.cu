/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "collectives.h"
#include "all_gather.h"

ncclResult_t ncclAllGatherFunc(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int nbytes = count*ncclTypeSize(datatype);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, nbytes, cudaMemcpyDeviceToDevice, stream));
  } else {
    if (nbytes*comm->nRanks <= comm->llThreshold) {
      NCCLCHECK(transportSaveProxies(1, NUM_LL_CHUNKS, comm->nRanks-1, 1, 2*nbytes, proxyPatternRing, comm, 1));
      NCCLCHECK(saveKernel(ncclCollAllGather, sendbuff, recvbuff, count, datatype, op, root, comm, stream, nbytes*comm->nRanks, 1));
    } else {
      NCCLCHECK(transportSaveProxies(NUM_SUBSTEPS, NUM_BUFCHUNKS, comm->nRanks-1, 1, nbytes, proxyPatternRing, comm, 0));
      NCCLCHECK(saveKernel(ncclCollAllGather, sendbuff, recvbuff, count, datatype, op, root, comm, stream, nbytes*comm->nRanks, 0));
      comm->opCount++;
    }
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclAllGatherFunc, "AllGather", sendbuff, recvbuff, sendcount, datatype, 
      ncclSum, 0, comm, stream);
}
