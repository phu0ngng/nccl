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

template <int NTHREADS_SET>
static __device__ void ncclKernel(struct KernelArgs args) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int rank = args.colls->args.comm->rank;
  static int dump = 1;
  if (dump == 0) {
    if (tid == 0 && rank == 0) for (int c=0; c<ncclCollNcolls; c++) {
      printf(" *** Coll %d ***\n", c);
      for (int t=0; t<ncclNumTypes; t++) printf("%16d ", t);
      printf("\n");
      for (int o=0; o<ncclNumOps; o++) {
        printf("%d :", o);
        for (int t=0; t<ncclNumTypes; t++) {
          printf("%16p ", ncclFuncs[2][c][o][t]);
        }
        printf("\n");
      }
    }
    dump = 1;
  }

  //if (tid == 0) printf("Starting %d collectives at %p\n", args.nColls, args.colls);

  struct ncclColl* collectives = args.colls;
  for (int c=0; c<args.nColls; c++) {
    struct ncclColl* coll = collectives+((args.startColl+c)%NCCL_MAX_OPS);

    if (bid >= coll->nBlocks || tid >= coll->nThreads) continue;

    ncclKern_t func;
    if (coll->ll) func = ncclFuncsLL[coll->coll][coll->op][coll->dtype];
    else func = ncclFuncs[NTHREADS_SET][coll->coll][coll->op][coll->dtype];

    //if (tid == 0) printf("[%d] %d: func %d, dtype %d, op %d, ll %d -> %p. %d/%d | %p : in %p, out %p, size %ld, root %d, comm %p, opCount %d\n", rank, c, coll->coll, coll->dtype, coll->op, coll->ll, func, coll->nThreads, coll->nBlocks, &coll->args, coll->args.ThisInput, coll->args.ThisOutput, coll->args.N, coll->args.root, coll->args.comm, coll->args.opCount);
    func(&coll->args);
    //if (tid == 0) printf("[%d] done\n", c);

    if (tid == 0) args.colls->args.comm->devCollFifoHead[0] = (args.colls->args.comm->devCollFifoHead[0]+1) % NCCL_MAX_OPS;
    /* Collectives may not flush some operations considering they are the last, let's play safe. */
    if (c < args.nColls-1) __threadfence_system();
  }
}

__global__ void ncclKernel64(struct KernelArgs args) { ncclKernel<0>(args); }
__global__ void ncclKernel128(struct KernelArgs args) { ncclKernel<1>(args); }
__global__ void ncclKernel256(struct KernelArgs args) { ncclKernel<2>(args); }
__global__ void ncclKernel512(struct KernelArgs args) { ncclKernel<3>(args); }

