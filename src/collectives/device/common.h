/*************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "../collectives.h"

/* Implement collectives with templates */
#define IMPL_COLL4(coll, op, ncclFunc, dtype, ctype, nthreads) \
__device__ void NCCL_COLL_NAME(coll, op, dtype, nthreads)(struct CollectiveArgs* args) { \
  coll##Kernel<nthreads, UNROLL, ncclFunc<ctype>, ctype>(args); \
} \
__global__ void NCCL_KERN_NAME(coll, op, dtype, nthreads)(struct ncclColl coll) { \
  coll##Kernel<nthreads, UNROLL, ncclFunc<ctype>, ctype>(&coll.args); \
}

#define IMPL_COLL3(coll, op, ncclFunc, dtype, ctype) \
  IMPL_COLL4(coll##LL, op, ncclFunc, dtype, ctype, LL_NTHREADS) \
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

#endif
