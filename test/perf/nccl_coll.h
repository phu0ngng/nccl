/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __NCCL_COLL_H__
#define __NCCL_COLL_H__

#include "nccl.h"

struct ncclColl_t {
  const char name[20];

  void (*getCollByteCount)(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t *procSharedCount, int *sameExpected, size_t count, int nranks);

  void (*InitRecvResult)(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first);

  void (*GetBw)(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks);

  void (*RunColl)(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

};

#endif
