/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ring.h"

ncclResult_t initRing(struct ncclComm* comm, int ringid) {
  struct ncclRing* ring = comm->rings+ringid;
  ring->id = ringid;

  // Setup intermediate buffering
  const char* str = getenv("NCCL_BUFFSIZE");
  int buffSize;
  if (str != NULL) {
    errno = 0;
    buffSize = strtol(str, NULL, 10);
    if (errno == ERANGE || buffSize == 0) {
      INFO("invalid NCCL_BUFFSIZE: %s, using default %lu",
          str, DEFAULT_BUFFER_SIZE_BYTES);
      buffSize = DEFAULT_BUFFER_SIZE_BYTES;
    }
  } else {
    buffSize = DEFAULT_BUFFER_SIZE_BYTES;
  }
  ring->buffSize = buffSize;
  const int size = ring->devMemSize = offsetof(struct ncclSendRecvMem, buff)+buffSize;
  struct ncclSendRecvMem* mem;
  CUDACHECK(cudaMalloc(&mem, size));
  CUDACHECK(cudaMemset(mem, 0, size));
  ring->devMem = mem;

  // Pre-configure send/recv pointers. Those are the default, they may change later.
  ring->recv.conn.buff = mem->buff;
  ring->recv.conn.llBuff = mem->llBuff;
  ring->recv.conn.tail = &mem->tail;
  ring->recv.conn.opCount = &mem->opCount;
  ring->recv.conn.direct = 0;
  ring->send.conn.head = &mem->head;
  ring->send.conn.llHead = &mem->llHead;
  ring->send.conn.direct = 0;
  ring->send.conn.llStep = 0;
  ring->send.conn.llLastCleaning = 0;

  // Ring index to user rank table.
  CUDACHECK(cudaMalloc(&ring->devUserRanks, comm->nRanks*sizeof(int)));
  ring->userRanks = (int*)malloc(comm->nRanks*sizeof(int));
  
  // Per-ring operation list.
  static_assert(sizeof(struct ncclColl) == 64, "ncclColl should be 64 bytes");
  ring->collectives = (struct ncclColl*)malloc(sizeof(struct ncclColl)*NCCL_MAX_OPS);
  memset(ring->collectives, 0, sizeof(struct ncclColl)*NCCL_MAX_OPS);
  CUDACHECK(cudaHostRegister(ring->collectives, sizeof(struct ncclColl)*NCCL_MAX_OPS, cudaHostRegisterMapped));
  CUDACHECK(cudaHostGetDevicePointer(&ring->devCollectives, ring->collectives, 0));
  return ncclSuccess;
}

ncclResult_t freeRing(struct ncclRing* ring) {
  // Intermediate buffering
  CUDACHECK(cudaFree(ring->devMem));

  // Index to rank table
  free(ring->userRanks);
  CUDACHECK(cudaFree(ring->devUserRanks));

  // Operation list
  CUDACHECK(cudaHostUnregister(ring->collectives));
  free(ring->collectives);

  // Free transport proxy resources
  NCCLCHECK(ring->send.transport->send.free(ring->send.transportResources));
  NCCLCHECK(transportDestroyProxy(&ring->send));
  NCCLCHECK(ring->recv.transport->recv.free(ring->recv.transportResources));
  NCCLCHECK(transportDestroyProxy(&ring->recv));
  return ncclSuccess;
}
