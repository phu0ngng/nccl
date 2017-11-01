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
  NCCL_FUNCS3(nthreads, coll, max ), \
  NCCL_FUNCS3(nthreads, coll, min ) }
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

/* Always make sure this enum is consistent with the order of the functions in the ncclFuncs array */
enum { ncclFuncSetLL = 0, ncclFuncSet64 = 1, ncclFuncSet128 = 2, ncclFuncSet256 = 3, ncclFuncSet512 = 4, ncclFuncSetNotFound = 5 };
static __device__ ncclKern_t ncclFuncs[][ncclCollNcolls][ncclNumOps][ncclNumTypes] = {
  {
    NCCL_FUNCS2B(64, ncclBcastLL),
    NCCL_FUNCS2A(64, ncclReduceLL),
    NCCL_FUNCS2B(64, ncclAllGatherLL),
    NCCL_FUNCS2A(64, ncclReduceScatterLL),
    NCCL_FUNCS2A(64, ncclAllReduceLL)
  },
  NCCL_FUNCS(64),
  NCCL_FUNCS(128),
  NCCL_FUNCS(256),
  NCCL_FUNCS(512)
};

static __device__ void load_coll(void* dst, void* src, size_t size, int tid) {
  int* d = (int*)dst;
  int* s = (int*)src;
  __syncthreads();
  for (int o = tid; o < (size/sizeof(int)); o += 64) d[o] = s[o];
  __syncthreads();
}

__global__ void ncclKernel(struct ncclColl firstColl) {
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

      int funcSet =
         coll->ll       == 1   ? ncclFuncSetLL  :
        (coll->nThreads == 65  ? ncclFuncSet64  :
        (coll->nThreads == 129 ? ncclFuncSet128 :
        (coll->nThreads == 257 ? ncclFuncSet256 :
        (coll->nThreads == 513 ? ncclFuncSet512 :
        ncclFuncSetNotFound))));

      if (funcSet == ncclFuncSetNotFound) {
        if (tid == 0) printf("NCCL Kernel internal error : invalid thread count %d", coll->nThreads);
        return;
      }

      func = ncclFuncs[funcSet][coll->coll][coll->redop][coll->dtype];
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
