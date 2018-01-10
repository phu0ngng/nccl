/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "collectives.h"
#include "common.h"

typedef void(*ncclKern_t)(struct CollectiveArgs* args);

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(nthreads, coll, op) \
  NCCL_COLL_NAME(coll, op,  i8, nthreads), \
  NCCL_COLL_NAME(coll, op,  u8, nthreads), \
  NCCL_COLL_NAME(coll, op, i32, nthreads), \
  NCCL_COLL_NAME(coll, op, u32, nthreads), \
  NCCL_COLL_NAME(coll, op, i64, nthreads), \
  NCCL_COLL_NAME(coll, op, u64, nthreads), \
  NCCL_COLL_NAME(coll, op, f16, nthreads), \
  NCCL_COLL_NAME(coll, op, f32, nthreads), \
  NCCL_COLL_NAME(coll, op, f64, nthreads)
#define NCCL_FUNCS3B(nthreads, coll, op) \
  NCCL_COLL_NAME(coll, op,  i8, nthreads), \
  NCCL_COLL_NAME(coll, op,  i8, nthreads), \
  NCCL_COLL_NAME(coll, op,  i8, nthreads), \
  NCCL_COLL_NAME(coll, op,  i8, nthreads), \
  NCCL_COLL_NAME(coll, op,  i8, nthreads), \
  NCCL_COLL_NAME(coll, op,  i8, nthreads), \
  NCCL_COLL_NAME(coll, op,  i8, nthreads), \
  NCCL_COLL_NAME(coll, op,  i8, nthreads), \
  NCCL_COLL_NAME(coll, op,  i8, nthreads)

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(nthreads, coll) \
  NCCL_FUNCS3A(nthreads, coll, sum ), \
  NCCL_FUNCS3A(nthreads, coll, prod), \
  NCCL_FUNCS3A(nthreads, coll, max ), \
  NCCL_FUNCS3A(nthreads, coll, min )
#define NCCL_FUNCS2B(nthreads, coll) \
  NCCL_FUNCS3B(nthreads, coll, copy), \
  NCCL_FUNCS3B(nthreads, coll, copy), \
  NCCL_FUNCS3B(nthreads, coll, copy), \
  NCCL_FUNCS3B(nthreads, coll, copy)

// Must be consistent with ncclColl_t
#define NCCL_FUNCS(nthreads) { \
  NCCL_FUNCS2B(nthreads, ncclBcast), \
  NCCL_FUNCS2A(nthreads, ncclReduce), \
  NCCL_FUNCS2B(nthreads, ncclAllGather), \
  NCCL_FUNCS2A(nthreads, ncclReduceScatter), \
  NCCL_FUNCS2A(nthreads, ncclAllReduce) }

// Must be consistent with the ncclFuncSet enum
static __device__ ncclKern_t const ncclFuncs[][ncclCollCount*ncclNumOps*ncclNumTypes] = {
  {
    NCCL_FUNCS2B(LL_NTHREADS, ncclBcastLL),
    NCCL_FUNCS2A(LL_NTHREADS, ncclReduceLL),
    NCCL_FUNCS2B(LL_NTHREADS, ncclAllGatherLL),
    NCCL_FUNCS2A(LL_NTHREADS, ncclReduceScatterLL),
    NCCL_FUNCS2A(LL_NTHREADS, ncclAllReduceLL) },
  NCCL_FUNCS(128),
  NCCL_FUNCS(256),
  NCCL_FUNCS(512)
};

static __device__ void load_coll(void* dst, void* src, size_t size, int tid) {
  int* d = (int*)dst;
  int* s = (int*)src;
  __syncthreads();
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
  __syncthreads();
}

__global__ void ncclMultiOpKernel(struct ncclColl firstColl) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ struct ncclColl localColl[MAXRINGS];

  struct ncclComm* comm = firstColl.args.comm;
  struct ncclRing* ring = comm->rings+bid;
  int index = ring->collFifoHead;
  // To optimize for latency, (only the) first operation is passed as argument.
  struct ncclColl* coll = &firstColl;
  if (bid != 0) {
    coll = localColl+bid;
    load_coll(coll, ring->devCollectives+index, sizeof(struct ncclColl), tid);
  }
  int c = 0;
  ncclKern_t func;
  while (1) {
    if (tid < coll->nThreads) {
      // Ack the coll has been loaded and can be reused.
      if (tid == 0) ring->devCollectives[index].active = 0;

      int funcSet = FUNC_SET(coll->ll, coll->nThreads);

      if (funcSet == ncclFuncSetNotFound) {
        if (tid == 0) printf("NCCL Kernel internal error : invalid thread count %d", coll->nThreads);
        return;
      }

      func = ncclFuncs[funcSet][coll->funcIndex];
      func(&coll->args);
    }
    index = (index + 1) % NCCL_MAX_OPS;

    if (coll->active == 2) {
      if (tid == 0) ring->collFifoHead = index;
      return;
    }

    // Load next collective operation
    struct ncclColl* nextColl = ring->devCollectives+index;
    coll = localColl+bid;
    load_coll(coll, nextColl, sizeof(struct ncclColl), tid);
    c++;
  }
}
