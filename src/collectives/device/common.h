/*************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"
#include <cstdio>

#if __CUDA_ARCH__ >= 800
#define COLL_UNROLL 8
#define NCCL_DEV_TREE_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree
#else
#define COLL_UNROLL 4
#define NCCL_DEV_TREE_ARITY NCCL_MAX_TREE_ARITY
#endif

__device__ inline bool barrierReduceAny(int bit) {
  uint32_t popc;
  asm ("{"
    ".reg .pred barr_pred;"
    "setp.eq.u32 barr_pred, %1, 1;"
    "bar.red.popc.u32 %0, 0, barr_pred;"
  "}" : "=r"(popc) : "r"(bit));
  return popc != 0;
}

typedef void(*ncclKern_t)();
extern __device__ ncclKern_t ncclFuncs[];

template<typename T>
__device__ void copyToShmem(T *dst, T const *src, int n) {
  static_assert(sizeof(uint32_t) <= alignof(T), "Uhoh");
  uint32_t *d = reinterpret_cast<uint32_t*>(dst);
  uint32_t const *s = reinterpret_cast<uint32_t const*>(src);
  n *= sizeof(T)/sizeof(uint32_t);
  n -= threadIdx.x;
  d += threadIdx.x;
  s += threadIdx.x;
  while (n > 0) {
    *d = *s;
    d += blockDim.x;
    s += blockDim.x;
    n -= blockDim.x;
  }
}

template<ncclFunc_t Fn, int Algo, int Proto, typename Op, typename T, int Unroll>
struct ncclFunctionWorkElem {
  __device__ void run(ncclWorkElem*) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

template<ncclFunc_t Fn, int Algo, int Proto, typename Op, typename T, int Unroll>
struct ncclFunctionWork {
  __device__ void run(ncclWork *w) {
    for(int e=0; e < NCCL_MAX_WORK_ELEMENTS; e++) {
      //if(w->elems[e].active == 0)
      //  break;
      ncclFunctionWorkElem<Fn,Algo,Proto,Op,T,Unroll>().run(&w->elems[e]);
      break; // Until we have aggregation there is but one.
    }
  }
};

struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_DIRECT_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_DIRECT_ARITY];
  void* srcs[NCCL_MAX_DIRECT_ARITY+1];
  void* dsts[NCCL_MAX_DIRECT_ARITY+1];
};

struct ncclShmemData {
  union {
    uint64_t ll128warp[NCCL_LL128_MAX_NTHREADS/WARP_SIZE][NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE];
    struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  };
  ncclDevComm *comm;
  ncclChannel *channel;
  ncclWork work;
};

extern __shared__ ncclShmemData ncclShmem;

template<ncclFunc_t Fn, int Algo, int Proto, typename Op, typename T, int Unroll, int FnIndex>
__device__ void ncclKernel(ncclWorkElem first)  {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  ncclDevComm *comm = first.comm;
  ncclChannel *channel = &comm->channels[bid];
  ncclWork *workFifoHost = channel->workFifo;
  ncclWork *workFifoDev = channel->workFifoDev;
  int workFifoIx = channel->index;

  if (tid == 0) {
    ncclShmem.comm = comm;
    ncclShmem.channel = channel;
    // Still needs a barrier to publish.
  }

  /* To optimize for latency, (only) the first operation is passed as argument.*/
  if (bid == 0 && first.funcIndex != FUNC_INDEX_P2P) {
    copyToShmem(&ncclShmem.work.elems[0], &first, 1);
    __syncthreads();
    goto SkipLoadWork;
  }

  do {
    copyToShmem(&ncclShmem.work, &workFifoDev[workFifoIx], 1);
    { // Check whether the last operation was aborted and make sure all threads exit
      int aborted = tid == 0 ? *(comm->abortFlag) : 0;
      if (barrierReduceAny(aborted))
        break;
      if (tid == 0)
        workFifoHost[workFifoIx].elems[0].active = 0;
    }
    workFifoIx += 1;
    if (workFifoIx == NCCL_MAX_OPS)
      workFifoIx = 0;
    if (tid == 0)
      channel->index = workFifoIx;

  SkipLoadWork:
    if (tid < ncclShmem.work.elems[0].nThreads) {
      if (ncclShmem.work.elems[0].funcIndex == FnIndex)
        ncclFunctionWork<Fn, Algo, Proto, Op, T, Unroll>().run(&ncclShmem.work);
      else
        ncclFuncs[ncclShmem.work.elems[0].funcIndex]();
    }
  } while(ncclShmem.work.elems[0].active != 2);
}

// Only generate kernels for SUM
#if NCCL_OP == 0
#define IMPL_COLL_KERN(func, algo, proto, redop, type, fIndex) \
__global__ void NCCL_KERN_NAME(func, algo, proto, redop, type)(ncclWorkElem first) { \
  ncclKernel<ncclFunc##func, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##redop<type>, type, COLL_UNROLL, fIndex>(first); \
}
#else
#define IMPL_COLL_KERN(func, algo, proto, redop, type, fInded)
#endif

// Examples :     AllReduce, RING, LL,    Sum,   uint8
#define IMPL_COLL_FUNC(func, algo, proto, redop, type) \
__device__ void NCCL_FUNC_NAME(func, algo, proto, redop, type)() { \
  ncclFunctionWork<ncclFunc##func, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##redop<type>, type, COLL_UNROLL>().run(&ncclShmem.work); \
}

// Only generate inline kernels for LL
#define IMPL_COLL4(func, algo, redop, type, ncclType) \
  IMPL_COLL_FUNC(func, algo, LL,     redop, type) \
  IMPL_COLL_FUNC(func, algo, LL128,  redop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, redop, type) \
  IMPL_COLL_KERN(func, algo, LL,     redop, type, FUNC_INDEX(ncclFunc##func, nccl##redop, ncclType, NCCL_ALGO_##algo, NCCL_PROTO_LL)) \

#define IMPL_COLL3(func, redop, type, ncclType) \
  IMPL_COLL4(func, TREE,    redop, type, ncclType) \
  IMPL_COLL4(func, RING,    redop, type, ncclType) \
  IMPL_COLL4(func, COLLNET, redop, type, ncclType)

#if NCCL_TYPE == 0
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, int8_t,   ncclInt8)
#elif NCCL_TYPE == 1
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, uint8_t,  ncclUint8)
#elif NCCL_TYPE == 2
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, int32_t,  ncclInt32)
#elif NCCL_TYPE == 3
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, uint32_t, ncclUint32)
#elif NCCL_TYPE == 4
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, int64_t,  ncclInt64)
#elif NCCL_TYPE == 5
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, uint64_t, ncclUint64)
#elif NCCL_TYPE == 6
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, half,     ncclFloat16)
#elif NCCL_TYPE == 7
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, float,    ncclFloat32)
#elif NCCL_TYPE == 8
#define IMPL_COLL2(func, redop) IMPL_COLL3(func, redop, double,   ncclFloat64)
#endif

// Reduction define all functions
#if NCCL_OP == 0
#define IMPL_COLL_R(func) IMPL_COLL2(func, Sum);
#elif NCCL_OP == 1
#define IMPL_COLL_R(func) IMPL_COLL2(func, Prod);
#elif NCCL_OP == 2
#define IMPL_COLL_R(func) IMPL_COLL2(func, Min);
#elif NCCL_OP == 3
#define IMPL_COLL_R(func) IMPL_COLL2(func, Max);
#elif NCCL_OP == 4
#define IMPL_COLL_R(func) IMPL_COLL2(func, Avg);
#endif

#if NCCL_OP == 0 && NCCL_TYPE == 0
// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t, ncclInt8);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE, Sum, int8_t, 0);
#else
#define IMPL_COLL_C(func)
#define IMPL_COLL_P(func)
#endif

#endif
