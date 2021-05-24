/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"
#include "comm.h"
#include "socket.h"

ncclResult_t bootstrapNetInit();
ncclResult_t bootstrapCreateRoot(ncclUniqueId* commId, bool idFromEnv);
ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out);
ncclResult_t bootstrapInit(ncclUniqueId* id, struct ncclComm* comm);
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size);
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size);
ncclResult_t bootstrapProxyConnect(void* commState, int transport, int send, int rank, int* fd);
ncclResult_t bootstrapProxyClose(int fd);
ncclResult_t bootstrapClose(void* commState);
ncclResult_t bootstrapAbort(void* commState);
#endif
