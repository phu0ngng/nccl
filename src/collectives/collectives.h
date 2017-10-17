/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"

typedef enum { ncclCollBcast, ncclCollReduce, ncclCollAllGather, ncclCollReduceScatter, ncclCollAllReduce, ncclCollNcolls } ncclColl_t;

#define NCCL_COLL_NAME(coll, op, dtype, nthreads) \
  coll##_##op##_##dtype##_##nthreads

/* Declare all collective operations */
#define DECL_COLL4(coll, op, dtype, nthreads) \
  extern __device__ void NCCL_COLL_NAME(coll, op, dtype, nthreads)(struct CollectiveArgs* args);

#define DECL_COLL3(coll, op, dtype) \
  DECL_COLL4(coll##LL, op, dtype, 64) \
  DECL_COLL4(coll, op, dtype, 64) \
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

/* Implement collectives with templates */
#define IMPL_COLL4(coll, op, ncclFunc, dtype, ctype, nthreads) \
  __device__ void NCCL_COLL_NAME(coll, op, dtype, nthreads)(struct CollectiveArgs* args) { \
  coll##Kernel<nthreads, UNROLL, ncclFunc<ctype>, ctype>(args); \
}

#define IMPL_COLL3(coll, op, ncclFunc, dtype, ctype) \
  IMPL_COLL4(coll##LL, op, ncclFunc, dtype, ctype, 64) \
  IMPL_COLL4(coll, op, ncclFunc, dtype, ctype, 64) \
  IMPL_COLL4(coll, op, ncclFunc, dtype, ctype, 128) \
  IMPL_COLL4(coll, op, ncclFunc, dtype, ctype, 256) \
  IMPL_COLL4(coll, op, ncclFunc, dtype, ctype, 512)

#define IMPL_COLL2(coll, op, ncclFunc) \
  IMPL_COLL3(coll, op, ncclFunc, i8, int8_t) \
  IMPL_COLL3(coll, op, ncclFunc, u8, uint8_t) \
  IMPL_COLL3(coll, op, ncclFunc, i32, int32_t) \
  IMPL_COLL3(coll, op, ncclFunc, u32, uint32_t) \
  IMPL_COLL3(coll, op, ncclFunc, i64, int64_t) \
  IMPL_COLL3(coll, op, ncclFunc, u64, uint64_t) \
  IMPL_COLL3(coll, op, ncclFunc, f16, half) \
  IMPL_COLL3(coll, op, ncclFunc, f32, float) \
  IMPL_COLL3(coll, op, ncclFunc, f64, double)

/* not used to improve compilation time */
#define IMPL_COLL(coll) \
  IMPL_COLL2(coll, sum, FuncSum) \
  IMPL_COLL2(coll, prod, FuncProd) \
  IMPL_COLL2(coll, min, FuncMin) \
  IMPL_COLL2(coll, max, FuncMax)

