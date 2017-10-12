/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "collectives.h"

typedef void(*ncclKern_t)(struct CollectiveArgs* args);

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS3(nthreads, coll, dtype) { \
  NCCL_COLL_NAME(coll, sum, dtype, nthreads), \
  NCCL_COLL_NAME(coll, prod, dtype, nthreads), \
  NCCL_COLL_NAME(coll, min, dtype, nthreads), \
  NCCL_COLL_NAME(coll, max, dtype, nthreads) }

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS2(nthreads, coll) { \
  NCCL_FUNCS3(nthreads, coll, i8), \
  NCCL_FUNCS3(nthreads, coll, u8), \
  NCCL_FUNCS3(nthreads, coll, i32), \
  NCCL_FUNCS3(nthreads, coll, u32), \
  NCCL_FUNCS3(nthreads, coll, i64), \
  NCCL_FUNCS3(nthreads, coll, u64), \
  NCCL_FUNCS3(nthreads, coll, f16), \
  NCCL_FUNCS3(nthreads, coll, f32), \
  NCCL_FUNCS3(nthreads, coll, f64) }

// Must be consistent with ncclColl::coll
#define NCCL_FUNCS(nthreads) { \
  NCCL_FUNCS2(nthreads, ncclBcast), \
  NCCL_FUNCS2(nthreads, ncclReduce), \
  NCCL_FUNCS2(nthreads, ncclAllGather), \
  NCCL_FUNCS2(nthreads, ncclReduceScatter), \
  NCCL_FUNCS2(nthreads, ncclAllReduce) }

static __device__ ncclKern_t ncclFuncs[][ncclCollNcolls][ncclNumTypes][ncclNumOps] = {
  NCCL_FUNCS(64),
  NCCL_FUNCS(128),
  NCCL_FUNCS(256),
  NCCL_FUNCS(512)
};

static __device__ ncclKern_t ncclFuncsLL[ncclCollNcolls][ncclNumTypes][ncclNumOps] = {
  NCCL_FUNCS2(64, ncclBcastLL), \
  NCCL_FUNCS2(64, ncclReduceLL), \
  NCCL_FUNCS2(64, ncclAllGatherLL), \
  NCCL_FUNCS2(64, ncclReduceScatterLL), \
  NCCL_FUNCS2(64, ncclAllReduceLL)
};

template <int NTHREADS_SET>
static __device__ void ncclKernel(void* _args) {
  struct KernelArgs* args = (struct KernelArgs*)_args;
  struct ncclColl* collectives = args->colls;
  for (int c=0; c<args->nColls; c++) {
    struct ncclColl* coll = collectives+c;
    ncclKern_t func;
    if (coll->ll) func = ncclFuncsLL[coll->coll][coll->dtype][coll->op];
    else func = ncclFuncs[NTHREADS_SET][coll->coll][coll->dtype][coll->op];
    func(&coll->args);
  }
}

__global__ void ncclKernel64(void* args) { ncclKernel<0>(args); }
__global__ void ncclKernel128(void* args) { ncclKernel<1>(args); }
__global__ void ncclKernel256(void* args) { ncclKernel<2>(args); }
__global__ void ncclKernel512(void* args) { ncclKernel<3>(args); }

