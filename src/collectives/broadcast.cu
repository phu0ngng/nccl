/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "collectives.h"
#include "broadcast.h"

ncclResult_t ncclBroadcastFunc(const void* sendbuff, void* recvbuff, const size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int nbytes = count*ncclTypeSize(datatype);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, nbytes, cudaMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(ArgsSetup(sendbuff, recvbuff, root, count, comm));
    if (nbytes <= comm->llThreshold) {
      NCCLCHECK(transportSaveProxies(1, NUM_LL_CHUNKS, 1, 1, 2*nbytes, proxyPatternFrom(root), comm, 1));
      saveKernel(ncclCollBcast, op, datatype, nbytes, comm, stream, 1);
    } else {
      NCCLCHECK(transportSaveProxies(NUM_SUBSTEPS, NUM_BUFCHUNKS, 1, 1, nbytes, proxyPatternFrom(root), comm, 0));
      saveKernel(ncclCollBcast, op, datatype, nbytes, comm, stream, 0);
      comm->opCount++;
    }
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
