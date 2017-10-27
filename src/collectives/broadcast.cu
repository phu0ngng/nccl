/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "collectives.h"

ncclResult_t ncclBroadcastFunc(const void* sendbuff, void* recvbuff, const size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int nbytes = count*ncclTypeSize(datatype);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, nbytes, cudaMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(transportSaveProxies(BROADCAST_SUBSTEPS, BROADCAST_BUFCHUNKS, 1, 1, nbytes, proxyPatternFrom(root), comm));
    NCCLCHECK(saveKernel(ncclCollBcast, sendbuff, recvbuff, count, datatype, op, root, comm, stream, nbytes));
  }

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclBroadcastFunc, "Bcast", buff, buff, count, datatype,
     ncclSum, root, comm, stream);
}
