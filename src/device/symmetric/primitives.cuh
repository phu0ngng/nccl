#ifndef NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_
#define NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_

#include "sym_kernels.h"
#include "bitops.h"
#include "collectives.h"
#include "../op128.h"
#include "../reduce_kernel.h"

#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif

// flattenIx(pos0, dim0, pos1, dim1, pos2, dim2, ...)
// Given a position vector `pos` in a rectangular index space with lengths in the `dim`
// vector, flatten that down to a linear index. The fastest moving dimension is given first.
__device__ __forceinline__ int flattenIx() { return 0; }

template<typename Int0, typename Int1, typename ...Ints>
static __device__ Int0 flattenIx(Int0 pos, Int1 size, Ints ...more) {
  return pos + size*flattenIx(more...);
}

namespace {
struct ncclSymkKernelStuff {
  ncclSymComm const& comm;
  int nBlocks;
  uint32_t nRanks_rcp32;
  uint32_t nBlocks_rcp32;
  uint32_t nBlocks_nWarps_rcp32;
  uint32_t nRanks_nBlocks_rcp32;

  __device__ ncclSymkKernelStuff(ncclSymkDevArgs const* args):
    comm(args->comm) {
    nBlocks = gridDim.x;
    nRanks_rcp32 = args->comm.nRanks_rcp32;
    nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);
    using nccl::utility::imulRcp32;
    nRanks_nBlocks_rcp32 = imulRcp32(comm.nRanks, nRanks_rcp32, nBlocks, nBlocks_rcp32);
  }
};
}

template<template<typename> typename Red, typename T, bool nvls>
struct ncclSymkAccumType { using Type = T; };

// Only Red's whose opArg is invariant w.r.t. the datatype can have a different
// accumulator type. At the moment this excludes integer min/max, sumpostdiv,
// and premulsum.
template<> struct ncclSymkAccumType<FuncSum, __half, false> { using Type = float; };
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<> struct ncclSymkAccumType<FuncSum, __nv_bfloat16, false> { using Type = float; };
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template<> struct ncclSymkAccumType<FuncSum, __nv_fp8_e4m3, false> { using Type = float; };
template<> struct ncclSymkAccumType<FuncSum, __nv_fp8_e5m2, false> { using Type = float; };
#endif
#endif
