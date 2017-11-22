/*************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

typedef enum { ncclCollBcast, ncclCollReduce, ncclCollAllGather, ncclCollReduceScatter, ncclCollAllReduce, ncclCollCount } ncclColl_t;

#define FUNC_INDEX(coll, redop, dtype) (((coll*ncclNumOps + redop)*ncclNumTypes) + dtype)

#define NCCL_COLL_NAME(coll, op, dtype, nthreads) \
  coll##_##op##_##dtype##_##nthreads

#define NCCL_KERN_NAME(coll, op, dtype, nthreads) \
  coll##Kernel_##op##_##dtype##_##nthreads

/* Declare all collective operations */
#define DECL_COLL4(coll, op, dtype, nthreads) \
  extern __device__ void NCCL_COLL_NAME(coll, op, dtype, nthreads)(struct CollectiveArgs* args); \
  extern __global__ void NCCL_KERN_NAME(coll, op, dtype, nthreads)(struct ncclColl coll); \

#define DECL_COLL3(coll, op, dtype) \
  DECL_COLL4(coll##LL, op, dtype, LL_NTHREADS) \
  DECL_COLL4(coll, op, dtype, 128) \
  DECL_COLL4(coll, op, dtype, 256) \
  DECL_COLL4(coll, op, dtype, 512)

#define DECL_COLL2(coll, op) \
  DECL_COLL3(coll, op, i8) \
  DECL_COLL3(coll, op, u8) \
  DECL_COLL3(coll, op, i32) \
  DECL_COLL3(coll, op, u32) \
  DECL_COLL3(coll, op, i64) \
  DECL_COLL3(coll, op, u64) \
  DECL_COLL3(coll, op, f16) \
  DECL_COLL3(coll, op, f32) \
  DECL_COLL3(coll, op, f64)

#define DECL_COLL(coll) \
  DECL_COLL2(coll, sum) \
  DECL_COLL2(coll, prod) \
  DECL_COLL2(coll, min) \
  DECL_COLL2(coll, max)

#define DECL_ALL_COLLS \
  DECL_COLL2(ncclBcast, copy) \
  DECL_COLL(ncclReduce) \
  DECL_COLL2(ncclAllGather, copy) \
  DECL_COLL(ncclReduceScatter) \
  DECL_COLL(ncclAllReduce) \

DECL_ALL_COLLS

enum { ncclFuncSetLL = 0, ncclFuncSet128 = 1, ncclFuncSet256 = 2, ncclFuncSet512 = 3, ncclFuncSetNotFound = 4 };
#define FUNC_SET(ll, nthreads) \
        (ll       == 1   ? ncclFuncSetLL  : \
        (nthreads == 129 ? ncclFuncSet128 : \
        (nthreads == 257 ? ncclFuncSet256 : \
        (nthreads == 513 ? ncclFuncSet512 : \
        ncclFuncSetNotFound))))


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
