/*************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

typedef enum { ncclCollBcast, ncclCollReduce, ncclCollAllGather, ncclCollReduceScatter, ncclCollAllReduce, ncclCollNcolls } ncclColl_t;

#define ALLREDUCE_SUBSTEPS 2
#define ALLREDUCE_BUFCHUNKS 2
#define ALLGATHER_SUBSTEPS 4
#define ALLGATHER_BUFCHUNKS 2
#define REDUCESCATTER_SUBSTEPS 4
#define REDUCESCATTER_BUFCHUNKS 2
#define BROADCAST_SUBSTEPS 4
#define BROADCAST_BUFCHUNKS 2
#define REDUCE_SUBSTEPS 4
#define REDUCE_BUFCHUNKS 2

#endif
