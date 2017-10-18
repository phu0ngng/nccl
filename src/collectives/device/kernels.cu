/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "collectives.h"

typedef void(*ncclKern_t)(struct CollectiveArgs* args);

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3(nthreads, coll, op) { \
  NCCL_COLL_NAME(coll, op,  i8, nthreads), \
  NCCL_COLL_NAME(coll, op,  u8, nthreads), \
  NCCL_COLL_NAME(coll, op, i32, nthreads), \
  NCCL_COLL_NAME(coll, op, u32, nthreads), \
  NCCL_COLL_NAME(coll, op, i64, nthreads), \
  NCCL_COLL_NAME(coll, op, u64, nthreads), \
  NCCL_COLL_NAME(coll, op, f16, nthreads), \
  NCCL_COLL_NAME(coll, op, f32, nthreads), \
  NCCL_COLL_NAME(coll, op, f64, nthreads) }

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(nthreads, coll) { \
  NCCL_FUNCS3(nthreads, coll, sum ), \
  NCCL_FUNCS3(nthreads, coll, prod), \
  NCCL_FUNCS3(nthreads, coll, min ), \
  NCCL_FUNCS3(nthreads, coll, max ) }
#define NCCL_FUNCS2B(nthreads, coll) { \
  NCCL_FUNCS3(nthreads, coll, copy), \
  NCCL_FUNCS3(nthreads, coll, copy), \
  NCCL_FUNCS3(nthreads, coll, copy), \
  NCCL_FUNCS3(nthreads, coll, copy) }

// Must be consistent with ncclColl_t
#define NCCL_FUNCS(nthreads) { \
  NCCL_FUNCS2B(nthreads, ncclBcast), \
  NCCL_FUNCS2A(nthreads, ncclReduce), \
  NCCL_FUNCS2B(nthreads, ncclAllGather), \
  NCCL_FUNCS2A(nthreads, ncclReduceScatter), \
  NCCL_FUNCS2A(nthreads, ncclAllReduce) }

static __device__ ncclKern_t ncclFuncs[][ncclCollNcolls][ncclNumOps][ncclNumTypes] = {
  NCCL_FUNCS(64),
  NCCL_FUNCS(128),
  NCCL_FUNCS(256),
  NCCL_FUNCS(512)
};

static __device__ ncclKern_t ncclFuncsLL[ncclCollNcolls][ncclNumOps][ncclNumTypes] = {
  NCCL_FUNCS2B(64, ncclBcastLL),
  NCCL_FUNCS2A(64, ncclReduceLL),
  NCCL_FUNCS2B(64, ncclAllGatherLL),
  NCCL_FUNCS2A(64, ncclReduceScatterLL),
  NCCL_FUNCS2A(64, ncclAllReduceLL)
};

static __device__ void load_coll(void* dst, void* src, size_t size, int tid) {
  int* d = (int*)dst;
  int* s = (int*)src;
  for (int o = tid; o < (size/sizeof(int)); o += 64) d[o] = s[o];
  __syncthreads();
}

template <int NTHREADS_SET>
static __device__ void ncclKernel(struct KernelArgs args) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ struct ncclColl localColl;

  struct ncclColl* collectives = args.colls;
  for (int c=0; c<args.nColls; c++) {
    struct ncclColl* collPtr = collectives+((args.startColl+c)%NCCL_MAX_OPS);
    load_coll(&localColl, collPtr, sizeof(struct ncclColl), tid);
    struct ncclColl* coll = &localColl;

    if (bid >= coll->nBlocks || tid >= coll->nThreads) continue;

    uint32_t function = coll->function;
    ncclKern_t func = NCCL_FUNCTION_LL(function) ?
        ncclFuncsLL            [NCCL_FUNCTION_COLL(function)][NCCL_FUNCTION_REDOP(function)][NCCL_FUNCTION_DTYPE(function)]:
        ncclFuncs[NTHREADS_SET][NCCL_FUNCTION_COLL(function)][NCCL_FUNCTION_REDOP(function)][NCCL_FUNCTION_DTYPE(function)];

    func(&coll->args);

    // Ack the completion of the function
    if (tid == 0) collPtr->function = 0;
  }
}

__global__ void ncclKernel64 (struct KernelArgs args) { ncclKernel<0>(args); }
__global__ void ncclKernel128(struct KernelArgs args) { ncclKernel<1>(args); }
__global__ void ncclKernel256(struct KernelArgs args) { ncclKernel<2>(args); }
__global__ void ncclKernel512(struct KernelArgs args) { ncclKernel<3>(args); }

