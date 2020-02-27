/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollSendRecv, "SendRecv",
    sendbuff, NULL, count*ncclTypeSize(datatype), ncclInt8, ncclSum, peer, comm, stream, /* Args */
    SENDRECV_CHUNKSTEPS, SENDRECV_SLICESTEPS };
  info.sendbytes = count*ncclTypeSize(datatype);
  info.recvbytes = -1;
  info.delta = (comm->nRanks - (comm->rank-peer)) % comm->nRanks;
  return ncclEnqueueCheck(&info);
}
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollSendRecv, "SendRecv",
    NULL, recvbuff, count*ncclTypeSize(datatype), ncclInt8, ncclSum, peer, comm, stream, /* Args */
    SENDRECV_CHUNKSTEPS, SENDRECV_SLICESTEPS };
  info.sendbytes = -1;
  info.recvbytes = count*ncclTypeSize(datatype);
  info.delta = (comm->nRanks+(comm->rank-peer)) % comm->nRanks;
  return ncclEnqueueCheck(&info);
}
