/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_KERNELS_CUH_
#define _REDUCE_COPY_TEST_KERNELS_CUH_

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>  // For printf in device code
#include <cassert>  // For assert
#include <cstdint>
#include <type_traits>
#include "config.h"
#include "api_function_traits.h"
#include "support.h"

#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif

// Traits for multimem/ReduceSum/etc. come from api_function_traits.h (single table, host+device).

template <typename T, ApiFunctionId FuncId>
__host__ __device__ constexpr bool multimemAccessSupportedForFunc() {
  return (!hasMultimemSourceConstexpr(FuncId) || TypeSupportMatrix::isMultimemSupported<T>()) &&
       (!hasMultimemDestinationConstexpr(FuncId) || TypeSupportMatrix::isMultimemDestinationSupported<T>());
}

// C++14 multimem dispatch helpers.
// In C++14, dead branches of `if` in device template functions are still type-checked
// and instantiated.  The multimem functions have static_asserts that fire for
// unsupported types (int8_t, uint8_t) even inside a dead `if NCCL_IF_CONSTEXPR (false)`
// branch.  These tag-dispatch free functions avoid the issue: the Unsupported overload
// has an empty body and does not reference any multimem function, so only the Supported
// overload is instantiated when MultimemSrcTag<T,FuncId> resolves to MultimemSrcSupported.
struct MultimemSrcSupported {};
struct MultimemSrcUnsupported {};
template<typename T, ApiFunctionId FuncId>
using MultimemSrcTag = typename std::conditional<
  multimemAccessSupportedForFunc<T, FuncId>(),
  MultimemSrcSupported, MultimemSrcUnsupported>::type;

// ncclMultimemReduceSumCopy – Windows / SymPtr / RawPtr
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSumCopy(MultimemSrcUnsupported, Coop&,
  ncclWindow_t, size_t, ncclMultimemHandle, ncclWindow_t, size_t, ncclMultimemHandle, size_t) {}
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSumCopy(MultimemSrcSupported, Coop& coop,
  ncclWindow_t sw, size_t so, ncclMultimemHandle smh,
  ncclWindow_t rw, size_t ro, ncclMultimemHandle dmh, size_t n) {
  ncclMultimemReduceSumCopy<T, Coop, size_t, UNROLL>(coop, sw, so, smh, rw, ro, dmh, n); }
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSumCopy(MultimemSrcUnsupported, Coop&,
  ncclSymPtr<T>, ncclMultimemHandle, ncclSymPtr<T>, ncclMultimemHandle, size_t) {}
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSumCopy(MultimemSrcSupported, Coop& coop,
  ncclSymPtr<T> sp, ncclMultimemHandle smh, ncclSymPtr<T> dp, ncclMultimemHandle dmh, size_t n) {
  ncclMultimemReduceSumCopy<T, Coop, size_t, UNROLL>(coop, sp, smh, dp, dmh, n); }
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSumCopy(MultimemSrcUnsupported, Coop&, T*, T*, size_t) {}
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSumCopy(MultimemSrcSupported, Coop& coop,
  T* sp, T* dp, size_t n) {
  ncclMultimemReduceSumCopy<T, Coop, size_t, UNROLL>(coop, sp, dp, n); }

// ncclMultimemReduceSumLsaCopy – Lambda / SymPtr / RawPtr
template<typename T, typename Coop, int UNROLL, typename SrcL, typename DstL>
__device__ inline void callMultimemReduceSumLsaCopyLambda(MultimemSrcUnsupported, Coop&,
  SrcL, int, DstL, int, size_t) {}
template<typename T, typename Coop, int UNROLL, typename SrcL, typename DstL>
__device__ inline void callMultimemReduceSumLsaCopyLambda(MultimemSrcSupported, Coop& coop,
  SrcL sl, int ns, DstL dl, int nd, size_t n) {
  ncclMultimemReduceSumLsaCopy<T, Coop, SrcL, DstL, size_t, UNROLL>(coop, sl, ns, dl, nd, n); }
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSumLsaCopy(MultimemSrcUnsupported, Coop&,
  ncclSymPtr<T>, ncclMultimemHandle, ncclSymPtr<T>, ncclTeam, size_t) {}
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSumLsaCopy(MultimemSrcSupported, Coop& coop,
  ncclSymPtr<T> sp, ncclMultimemHandle smh, ncclSymPtr<T> dp, ncclTeam team, size_t n) {
  ncclMultimemReduceSumLsaCopy<T, Coop, size_t, UNROLL>(coop, sp, smh, dp, team, n); }
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSumLsaCopy(MultimemSrcUnsupported, Coop&,
  T*, ncclSymPtr<T>, ncclTeam, size_t) {}
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSumLsaCopy(MultimemSrcSupported, Coop& coop,
  T* sp, ncclSymPtr<T> dp, ncclTeam team, size_t n) {
  ncclMultimemReduceSumLsaCopy<T, Coop, size_t, UNROLL>(coop, sp, dp, team, n); }

// ncclMultimemReduceSumMultimemCopy – Lambda
template<typename T, typename Coop, int UNROLL, typename SrcL, typename DstL>
__device__ inline void callMultimemReduceSumMultimemCopyLambda(MultimemSrcUnsupported, Coop&,
  SrcL, int, DstL, int, size_t) {}
template<typename T, typename Coop, int UNROLL, typename SrcL, typename DstL>
__device__ inline void callMultimemReduceSumMultimemCopyLambda(MultimemSrcSupported, Coop& coop,
  SrcL sl, int ns, DstL dl, int nd, size_t n) {
  ncclMultimemReduceSumMultimemCopy<T, Coop, SrcL, DstL, size_t, UNROLL>(coop, sl, ns, dl, nd, n); }

// ncclMultimemReduceSum – SymPtr/Window / RawPtr
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSum(MultimemSrcUnsupported, Coop&,
  ncclSymPtr<T>, T*, size_t, ncclMultimemHandle) {}
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSum(MultimemSrcSupported, Coop& coop,
  ncclSymPtr<T> sp, T* dp, size_t n, ncclMultimemHandle h) {
  ncclMultimemReduceSum<T, Coop, size_t, UNROLL>(coop, sp, dp, n, h); }
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSum(MultimemSrcUnsupported, Coop&, T*, T*, size_t) {}
template<typename T, typename Coop, int UNROLL>
__device__ inline void callMultimemReduceSum(MultimemSrcSupported, Coop& coop,
  T* sp, T* dp, size_t n) {
  ncclMultimemReduceSum<T, Coop, size_t, UNROLL>(coop, sp, dp, n); }

template <typename T>
__device__ __forceinline__ size_t lambdaOffsetBytes(size_t offsetElts) {
  return offsetElts * sizeof(T);
}

template <typename T>
__device__ __forceinline__ size_t resolveLambdaSendOffsetBytes(
  size_t sendoffset, int peer) {
  return (sendoffset != 0)
    ? lambdaOffsetBytes<T>(lambdaSendOffsetElts(peer))
    : 0;
}

template <typename T>
__device__ __forceinline__ size_t resolveLambdaRecvOffsetBytes(
  size_t recvoffset, int peer) {
  return (recvoffset != 0)
    ? lambdaOffsetBytes<T>(lambdaRecvOffsetElts(peer))
    : 0;
}


// Custom sum operator for generic ReduceCopy tests
template <typename T>
struct testOpSum {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

// Match NCCL AccumulateType<OpSum<half>> = float: accumulate in float, then round to half
template <>
struct testOpSum<half> {
  __device__ __forceinline__ half operator()(const half& a, const half& b) const {
    return __float2half_rn(__half2float(a) + __half2float(b));
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
// Match NCCL AccumulateType<OpSum<__nv_bfloat16>> = float: accumulate in float, then round
template <>
struct testOpSum<__nv_bfloat16> {
  __device__ __forceinline__ __nv_bfloat16 operator()(const __nv_bfloat16& a, const __nv_bfloat16& b) const {
    return __float2bfloat16_rn(__bfloat162float(a) + __bfloat162float(b));
  }
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
template <>
struct testOpSum<__nv_fp8_e4m3> {
  __device__ __forceinline__ __nv_fp8_e4m3 operator()(const __nv_fp8_e4m3& a, const __nv_fp8_e4m3& b) const {
    #if __CUDA_ARCH__ >= 800
      return __nv_fp8_e4m3(__hadd(__half(a), __half(b)));
    #else
      return __nv_fp8_e4m3(float(a) + float(b));
    #endif
  }
};

template <>
struct testOpSum<__nv_fp8_e5m2> {
  __device__ __forceinline__ __nv_fp8_e5m2 operator()(const __nv_fp8_e5m2& a, const __nv_fp8_e5m2& b) const {
    #if __CUDA_ARCH__ >= 800
      return __nv_fp8_e5m2(__hadd(__half(a), __half(b)));
    #else
      return __nv_fp8_e5m2(float(a) + float(b));
    #endif
  }
};
#endif

// Custom multiplication operator (similar to OpSum)
template <typename T>
struct OpMul {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a * b;
  }
};

template <>
struct OpMul<half> {
  __device__ __forceinline__ half operator()(const half& a, const half& b) const {
    #if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
      return __hmul(a, b);
    #else
      return __float2half(__half2float(a) * __half2float(b));
    #endif
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct OpMul<__nv_bfloat16> {
  __device__ __forceinline__ __nv_bfloat16 operator()(const __nv_bfloat16& a, const __nv_bfloat16& b) const {
    #if __CUDA_ARCH__ >= 800
      return __hmul(a, b);
    #else
      return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
    #endif
  }
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
template <>
struct OpMul<__nv_fp8_e4m3> {
  __device__ __forceinline__ __nv_fp8_e4m3 operator()(const __nv_fp8_e4m3& a, const __nv_fp8_e4m3& b) const {
    #if __CUDA_ARCH__ >= 800
      return __nv_fp8_e4m3(__hmul(__half(a), __half(b)));
    #else
      return __nv_fp8_e4m3(float(a) * float(b));
    #endif
  }
};

template <>
struct OpMul<__nv_fp8_e5m2> {
  __device__ __forceinline__ __nv_fp8_e5m2 operator()(const __nv_fp8_e5m2& a, const __nv_fp8_e5m2& b) const {
    #if __CUDA_ARCH__ >= 800
      return __nv_fp8_e5m2(__hmul(__half(a), __half(b)));
    #else
      return __nv_fp8_e5m2(float(a) * float(b));
    #endif
  }
};
#endif

// Ensure NCCL reduce copy uses correct accumulation type for testOpSum (match OpSum: FP8->half, half/bf16->float).
// Without these, AccumulateType<testOpSum<T>> would use the default Red<T>::Type = T and skip upcast.
namespace nccl { namespace utility {
template<> struct AccumulateType<testOpSum<half>> { using Type = float; };
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<> struct AccumulateType<testOpSum<__nv_bfloat16>> { using Type = float; };
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template<> struct AccumulateType<testOpSum<__nv_fp8_e4m3>> { using Type = half; };
template<> struct AccumulateType<testOpSum<__nv_fp8_e5m2>> { using Type = half; };
#endif
}}  // namespace nccl::utility

// Map CooperationLevel to the corresponding coop type (use CoopTypeOf<L>::Type directly)
template<CooperationLevel L> struct CoopTypeOf;
template<> struct CoopTypeOf<CooperationLevel::Thread> { using Type = ncclCoopThread; };
template<> struct CoopTypeOf<CooperationLevel::Warp>   { using Type = ncclCoopWarp; };
template<> struct CoopTypeOf<CooperationLevel::Cta>    { using Type = ncclCoopCta; };

// Device function to compute work distribution based on cooperation level
// Returns (startOffset, chunkSize) for the current thread/warp/block
template<CooperationLevel Level>
__device__ void computeWorkDistribution(
  size_t count,
  int rank, int nRanks,
  int blockIdx, int gridDim,
  int threadIdx, int blockDim,
  size_t& startOffset,
  size_t& chunkSize
) {
  if NCCL_IF_CONSTEXPR (Level == CooperationLevel::Thread) {
    // Thread-level: Each thread gets a contiguous chunk (similar to warp/CTA)
    const int globalTid = threadIdx + blockDim * (rank + blockIdx * nRanks);
    const int globalNumThreads = blockDim * gridDim * nRanks;

    const size_t elemsPerThread = count / globalNumThreads;
    const size_t remainder = count % globalNumThreads;

    startOffset = globalTid * elemsPerThread + min((size_t)globalTid, remainder);
    chunkSize = elemsPerThread + (globalTid < remainder ? 1 : 0);
  } else if NCCL_IF_CONSTEXPR (Level == CooperationLevel::Warp) {
    // Warp-level: Each warp gets a contiguous chunk
    const int warpId = threadIdx / 32;
    const int numWarpsPerBlock = (blockDim + 31) / 32;
    const int globalWarpIdx = warpId + numWarpsPerBlock * (rank + blockIdx * nRanks);
    const int globalNumWarps = numWarpsPerBlock * gridDim * nRanks;

    const size_t elemsPerWarp = count / globalNumWarps;
    const size_t remainder = count % globalNumWarps;

    startOffset = globalWarpIdx * elemsPerWarp + min((size_t)globalWarpIdx, remainder);
    chunkSize = elemsPerWarp + (globalWarpIdx < remainder ? 1 : 0);
  } else if NCCL_IF_CONSTEXPR (Level == CooperationLevel::Cta) {
    // CTA-level: Each block gets a contiguous chunk (like kernel 2)
    const int globalBlockIdx = rank + blockIdx * nRanks;
    const int globalNumBlocks = gridDim * nRanks;

    const size_t elemsPerBlock = count / globalNumBlocks;
    const size_t remainder = count % globalNumBlocks;

    startOffset = globalBlockIdx * elemsPerBlock + min((size_t)globalBlockIdx, remainder);
    chunkSize = elemsPerBlock + (globalBlockIdx < remainder ? 1 : 0);
  }
}

// Device function to compute work distribution for AllGather based on cooperation level
// For AllGather, work is distributed per-rank (not global across ranks)
// Each rank processes its own chunk, divided among blocks/threads/warps
// Returns (startOffset, chunkSize) for the current thread/warp/block
template<CooperationLevel Level, typename T>
__device__ void computeAllGatherWorkDistribution(
  size_t count,
  int blockIdx, int gridDim,
  int threadIdx, int blockDim,
  bool isMultimem,
  size_t& startOffset,
  size_t& chunkSize
) {
  if NCCL_IF_CONSTEXPR (Level == CooperationLevel::Cta) {
    // CTA-level: Each block gets a chunk of this rank's count
    (void)isMultimem;
    const size_t elemsPerBlock = count / gridDim;
    const size_t remainder = count % gridDim;

    startOffset = blockIdx * elemsPerBlock + min((size_t)blockIdx, remainder);
    chunkSize = elemsPerBlock + (blockIdx < remainder ? 1 : 0);
  } else if NCCL_IF_CONSTEXPR (Level == CooperationLevel::Thread) {
    // Thread-level: Each thread gets a chunk of this rank's count
    const int globalTid = threadIdx + blockDim * blockIdx;
    const int globalNumThreads = blockDim * gridDim;

    const size_t elemsPerThread = count / globalNumThreads;
    const size_t remainder = count % globalNumThreads;

    startOffset = globalTid * elemsPerThread + min((size_t)globalTid, remainder);
    chunkSize = elemsPerThread + (globalTid < remainder ? 1 : 0);
  } else if NCCL_IF_CONSTEXPR (Level == CooperationLevel::Warp) {
    // Warp-level: Each warp gets a chunk of this rank's count
    const int warpId = threadIdx / 32;
    const int numWarpsPerBlock = (blockDim + 31) / 32;
    const int globalWarpIdx = warpId + numWarpsPerBlock * blockIdx;
    const int globalNumWarps = numWarpsPerBlock * gridDim;

    const size_t elemsPerWarp = count / globalNumWarps;
    const size_t remainder = count % globalNumWarps;

    startOffset = globalWarpIdx * elemsPerWarp + min((size_t)globalWarpIdx, remainder);
    chunkSize = elemsPerWarp + (globalWarpIdx < remainder ? 1 : 0);
  }
}

template <typename T>
__device__ __forceinline__ size_t storageCountForElements(size_t count) {
  constexpr int bits = bitSizeOfElement<T>();
  constexpr size_t bytesPerElement = (bits + 7) / 8;
  return ((count * static_cast<size_t>(bits) + 7) / 8) / bytesPerElement;
}

template <typename T>
__device__ __forceinline__ size_t storageOffsetForElements(size_t offsetElts) {
  constexpr int bits = bitSizeOfElement<T>();
  constexpr size_t bytesPerElement = (bits + 7) / 8;
  return ((offsetElts * static_cast<size_t>(bits)) / 8) / bytesPerElement;
}

template <typename T>
__device__ __forceinline__ constexpr size_t elementsPerStorage() {
  constexpr int bits = bitSizeOfElement<T>();
  constexpr size_t bytesPerStorage = (bits + 7) / 8;
  return (bytesPerStorage * 8) / bits;
}

template <typename T>
__device__ __forceinline__ size_t elementCountForStorageRange(
  size_t storageStart, size_t storageCount, size_t totalElements) {
  const size_t eltsPer = elementsPerStorage<T>();
  const size_t elementStart = storageStart * eltsPer;
  size_t elementEnd = elementStart + storageCount * eltsPer;
  if (elementEnd > totalElements) {
    elementEnd = totalElements;
  }
  return (elementEnd > elementStart) ? (elementEnd - elementStart) : 0;
}

template<typename T, CooperationLevel coopLevel, int UNROLL>
__global__ void allreduceKernel(
  ApiFunctionId functionID,
  ncclWindow_t sendwin, size_t sendoffset,
  ncclWindow_t recvwin, size_t recvoffset,
  int nSrc, int nDst, int nDstStart,
  size_t count,
  ncclDevComm devComm
) {

  // Create cooperative group based on template parameter (for reduce/copy operation)
  typename CoopTypeOf<coopLevel>::Type coop;

  // Create barrier session (always CTA-level for block synchronization)
  // Always use non-multimem barrier path (multimem is an optimization for copy operations, not barriers)
  ncclCoopCta ctaCoop;
  const int rank = devComm.rank, nRanks = devComm.nRanks;
  ncclTeam team = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar{ctaCoop, devComm, team, devComm.lsaBarrier, blockIdx.x};
  bar.sync(ctaCoop, cuda::memory_order_relaxed);

  const size_t countUnits = storageCountForElements<T>(count);

  // Compute work distribution based on cooperation level
  size_t startOffset, chunkSize;
  computeWorkDistribution<coopLevel>(
    countUnits,
    rank, nRanks,
    blockIdx.x, gridDim.x,
    threadIdx.x, blockDim.x,
    startOffset, chunkSize
  );
  const size_t chunkCountElts = elementCountForStorageRange<T>(startOffset, chunkSize, count);

  // Process this thread/warp/block's assigned chunk
  if (chunkSize > 0) {
    size_t baseSendOffset = (sendoffset != 0) ? 0 : sendoffset;
    size_t baseRecvOffset = (recvoffset != 0) ? 0 : recvoffset;
    size_t srcOffset = baseSendOffset + startOffset * bytesPerElement<T>();
    size_t dstOffset = baseRecvOffset + startOffset * bytesPerElement<T>();

    switch (functionID) {
      case ApiFunctionId::LsaReduceSumCopy_Windows:
        // LSA ReduceSumCopy with windows
        ncclLsaReduceSumCopy<T, decltype(coop), size_t, UNROLL>(
          coop,
          sendwin, srcOffset,
          recvwin, dstOffset,
          chunkCountElts,
          devComm
        );
        break;

      case ApiFunctionId::LsaReduceSumCopy_DevComm:
        // LSA ReduceSumCopy with SymPtr and devComm
        {
          ncclSymPtr<T> srcSymPtr{sendwin, srcOffset};
          ncclSymPtr<T> dstSymPtr{recvwin, dstOffset};
          ncclLsaReduceSumCopy<T, decltype(coop), size_t, UNROLL>(
            coop,
            srcSymPtr, dstSymPtr,
            chunkCountElts,
            devComm
          );
        }
        break;

      case ApiFunctionId::LsaReduceSumCopy_SameTeam:
        // LSA ReduceSumCopy with SymPtr and team
        {
          ncclTeam team = ncclTeamLsa(devComm);
          ncclSymPtr<T> srcSymPtr{sendwin, srcOffset};
          ncclSymPtr<T> dstSymPtr{recvwin, dstOffset};
          ncclLsaReduceSumCopy<T, decltype(coop), size_t, UNROLL>(
            coop,
            srcSymPtr, dstSymPtr,
            chunkCountElts,
            team
          );
        }
        break;

      case ApiFunctionId::MultimemReduceSumCopy_Windows:
        // Multimem ReduceSumCopy with windows
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemReduceSumCopy_Windows>()) {
            ncclMultimemHandle multimemHandle = devComm.lsaMultimem;
            callMultimemReduceSumCopy<T, decltype(coop), UNROLL>(
              MultimemSrcTag<T, ApiFunctionId::MultimemReduceSumCopy_Windows>{},
              coop, sendwin, srcOffset, multimemHandle,
              recvwin, dstOffset, multimemHandle, chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allreduceKernel: MultimemReduceSumCopy_Windows not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::MultimemReduceSumCopy_SymPtr:
        // Multimem ReduceSumCopy with SymPtr
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemReduceSumCopy_SymPtr>()) {
            ncclMultimemHandle multimemHandle = devComm.lsaMultimem;
            ncclSymPtr<T> srcSymPtr{sendwin, srcOffset};
            ncclSymPtr<T> dstSymPtr{recvwin, dstOffset};
            callMultimemReduceSumCopy<T, decltype(coop), UNROLL>(
              MultimemSrcTag<T, ApiFunctionId::MultimemReduceSumCopy_SymPtr>{},
              coop, srcSymPtr, multimemHandle, dstSymPtr, multimemHandle, chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allreduceKernel: MultimemReduceSumCopy_SymPtr not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::MultimemReduceSumCopy_RawPtr:
        // Multimem ReduceSumCopy with raw pointers
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemReduceSumCopy_RawPtr>()) {
            T* mcSrcPtr = (T*)ncclGetLsaMultimemPointer(sendwin, srcOffset, devComm);
            T* mcDstPtr = (T*)ncclGetLsaMultimemPointer(recvwin, dstOffset, devComm);
            callMultimemReduceSumCopy<T, decltype(coop), UNROLL>(
              MultimemSrcTag<T, ApiFunctionId::MultimemReduceSumCopy_RawPtr>{},
              coop, mcSrcPtr, mcDstPtr, chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allreduceKernel: MultimemReduceSumCopy_RawPtr not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::LsaReduceSumLsaCopy:
        // Lambda-based LSA ReduceSumCopy (LSA sources -> LSA destinations).
        // NM mode (nDstStart>0): srcRanks={0..nSrc-1}, dstRanks={nDstStart..nDstStart+nDst-1}.
        // NxN mode (nDstStart==0): all ranks are both sources and destinations.
        if (nDstStart > 0) {
          auto srcLambda = [=] __device__ (int i) -> T* {
            size_t baseOffset = resolveLambdaSendOffsetBytes<T>(sendoffset, i);
            size_t peerSrcOffset = baseOffset + startOffset * bytesPerElement<T>();
            return (T*)ncclGetLsaPointer(sendwin, peerSrcOffset, i);
          };
          auto dstLambda = [=] __device__ (int i) -> T* {
            // NM: destination index i refers to rank (nDstStart + i)
            size_t baseOffset = resolveLambdaRecvOffsetBytes<T>(recvoffset, nDstStart + i);
            size_t peerDstOffset = baseOffset + startOffset * bytesPerElement<T>();
            return (T*)ncclGetLsaPointer(recvwin, peerDstOffset, nDstStart + i);
          };
          ncclLsaReduceSumLsaCopy<T, decltype(coop), decltype(srcLambda), decltype(dstLambda), size_t, UNROLL>(
            coop,
            srcLambda, nSrc,
            dstLambda, nDst,
            chunkCountElts
          );
        } else {
          // NxN mode: all ranks as both sources and destinations
          auto srcLambda = [=] __device__ (int i) -> T* {
            size_t baseOffset = resolveLambdaSendOffsetBytes<T>(sendoffset, i);
            size_t peerSrcOffset = baseOffset + startOffset * bytesPerElement<T>();
            return (T*)ncclGetLsaPointer(sendwin, peerSrcOffset, i);
          };
          auto dstLambda = [=] __device__ (int i) -> T* {
            size_t baseOffset = resolveLambdaRecvOffsetBytes<T>(recvoffset, i);
            size_t peerDstOffset = baseOffset + startOffset * bytesPerElement<T>();
            return (T*)ncclGetLsaPointer(recvwin, peerDstOffset, i);
          };
          ncclLsaReduceSumLsaCopy<T, decltype(coop), decltype(srcLambda), decltype(dstLambda), size_t, UNROLL>(
            coop,
            srcLambda, nRanks,
            dstLambda, nRanks,
            chunkCountElts
          );
        }
        break;

      case ApiFunctionId::LsaReduceSumMultimemCopy:
        // Lambda-based LSA -> Multimem ReduceSumCopy
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::LsaReduceSumMultimemCopy>()) {
            ncclTeam team = ncclTeamLsa(devComm);
            (void)team;
            // LSA sources: one pointer per peer
            auto srcLambda = [=] __device__ (int i) -> T* {
              size_t baseOffset = resolveLambdaSendOffsetBytes<T>(sendoffset, i);
              size_t peerSrcOffset = baseOffset + startOffset * bytesPerElement<T>();
              return (T*)ncclGetLsaPointer(sendwin, peerSrcOffset, i);
            };
            // Multimem destination: single shared pointer (use multimem pointer access)
            auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
              return (T*)ncclGetLsaMultimemPointer(recvwin, dstOffset, devComm);
            };
            ncclLsaReduceSumMultimemCopy<T, decltype(coop), decltype(srcLambda), decltype(dstLambda), size_t, UNROLL>(
              coop,
              srcLambda, nRanks,
              dstLambda, 1,
              chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allreduceKernel: LsaReduceSumMultimemCopy not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::MultimemReduceSumLsaCopy:
        // Lambda-based Multimem -> LSA ReduceSumCopy
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemReduceSumLsaCopy>()) {
            // Multimem source: single shared pointer (use multimem pointer access)
            auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
              return (T*)ncclGetLsaMultimemPointer(sendwin, srcOffset, devComm);
            };
            // LSA destinations: one pointer per peer
            auto dstLambda = [=] __device__ (int i) -> T* {
              size_t baseOffset = resolveLambdaRecvOffsetBytes<T>(recvoffset, i);
              size_t peerDstOffset = baseOffset + startOffset * bytesPerElement<T>();
              return (T*)ncclGetLsaPointer(recvwin, peerDstOffset, i);
            };
            callMultimemReduceSumLsaCopyLambda<T, decltype(coop), UNROLL>(
              MultimemSrcTag<T, ApiFunctionId::MultimemReduceSumLsaCopy>{},
              coop, srcLambda, 1, dstLambda, nRanks, chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allreduceKernel: MultimemReduceSumLsaCopy not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::MultimemReduceSumMultimemCopy:
        // Lambda-based Multimem -> Multimem ReduceSumCopy
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemReduceSumMultimemCopy>()) {
            // Multimem source: single shared pointer (use multimem pointer access)
            auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
              return (T*)ncclGetLsaMultimemPointer(sendwin, srcOffset, devComm);
            };
            // Multimem destination: single shared pointer (use multimem pointer access)
            auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
              return (T*)ncclGetLsaMultimemPointer(recvwin, dstOffset, devComm);
            };
            callMultimemReduceSumMultimemCopyLambda<T, decltype(coop), UNROLL>(
              MultimemSrcTag<T, ApiFunctionId::MultimemReduceSumMultimemCopy>{},
              coop, srcLambda, 1, dstLambda, 1, chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allreduceKernel: MultimemReduceSumMultimemCopy not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::LsaReduceLsaCopy_Generic:
        // Generic LSA ReduceCopy with RedOp (LSA sources -> LSA destinations) - Sum variant
        {
          testOpSum<T> sumOp{};
          // Create lambdas that return pointers for each peer
          auto srcLambda = [=] __device__ (int i) -> T* {
            size_t baseOffset = resolveLambdaSendOffsetBytes<T>(sendoffset, i);
            size_t peerSrcOffset = baseOffset + startOffset * bytesPerElement<T>();
            return (T*)ncclGetLsaPointer(sendwin, peerSrcOffset, i);
          };
          auto dstLambda = [=] __device__ (int i) -> T* {
            size_t baseOffset = resolveLambdaRecvOffsetBytes<T>(recvoffset, i);
            size_t peerDstOffset = baseOffset + startOffset * bytesPerElement<T>();
            return (T*)ncclGetLsaPointer(recvwin, peerDstOffset, i);
          };
          ncclLsaReduceLsaCopy<T, decltype(coop),
            decltype(srcLambda), decltype(dstLambda), testOpSum<T>, size_t, UNROLL>(
            coop,
            srcLambda, nRanks,
            dstLambda, nRanks,
            sumOp,
            chunkCountElts
          );
        }
        break;

      case ApiFunctionId::LocalReduceSumCopy_Strided:
        // Local ReduceSumCopy with strided sources/destinations (single-device)
        {
          T* srcBasePtr = (T*)ncclGetLocalPointer(sendwin, srcOffset);
          T* dstBasePtr = (T*)ncclGetLocalPointer(recvwin, dstOffset);
          size_t displ = countUnits;
          ncclLocalReduceSumCopy<T, decltype(coop), size_t, UNROLL>(
            coop,
            nSrc, srcBasePtr, displ,
            nDst, dstBasePtr, displ,
            chunkCountElts
          );
        }
        break;

      case ApiFunctionId::LsaReduceLsaCopy_Generic_Mul:
        // Generic LSA ReduceCopy with multiplication Op (LSA sources -> LSA destinations)
        // Check if multiplication is supported for this type
        {
          // Use central function to check support at compile time
          constexpr bool mulSupported = TypeSupportMatrix::isMulSupported<T>();

          if NCCL_IF_CONSTEXPR (mulSupported) {
            // Multiplication is supported - proceed with multiplication variant
            OpMul<T> mulOp{};
            // Create lambdas that return pointers for each peer
            auto srcLambda = [=] __device__ (int i) -> T* {
              size_t baseOffset = resolveLambdaSendOffsetBytes<T>(sendoffset, i);
              size_t peerSrcOffset = baseOffset + startOffset * bytesPerElement<T>();
              return (T*)ncclGetLsaPointer(sendwin, peerSrcOffset, i);
            };
            auto dstLambda = [=] __device__ (int i) -> T* {
              size_t baseOffset = resolveLambdaRecvOffsetBytes<T>(recvoffset, i);
              size_t peerDstOffset = baseOffset + startOffset * bytesPerElement<T>();
              return (T*)ncclGetLsaPointer(recvwin, peerDstOffset, i);
            };
            ncclLsaReduceLsaCopy<T, decltype(coop), decltype(srcLambda), decltype(dstLambda), OpMul<T>, size_t, UNROLL>(
              coop,
              srcLambda, nRanks,
              dstLambda, nRanks,
              mulOp,
              chunkCountElts
            );
          } else {
            // Multiplication not supported - this should not happen if tests are properly filtered
            // Use runtime assert as a safety check
            assert(false && "Multiplication variant not supported for this type "
                           "(FP8 types not supported due to NCCL's type promotion)");
            if (threadIdx.x == 0 && blockIdx.x == 0) {
              printf("ERROR: allreduceKernel: Multiplication variant not supported for this type (functionID=%d).\n",
                   static_cast<int>(functionID));
            }
          }
        }
        break;

      case ApiFunctionId::LsaReduceMultimemCopy_Generic:
        // Generic LSA -> Multimem ReduceCopy with RedOp
        {
          testOpSum<T> sumOp{};
          // LSA sources: one pointer per peer
          auto srcLambda = [=] __device__ (int i) -> T* {
            size_t baseOffset = resolveLambdaSendOffsetBytes<T>(sendoffset, i);
            size_t peerSrcOffset = baseOffset + startOffset * bytesPerElement<T>();
            return (T*)ncclGetLsaPointer(sendwin, peerSrcOffset, i);
          };
          // Multimem destination: single shared pointer
          auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
            return (T*)ncclGetLsaMultimemPointer(recvwin, dstOffset, devComm);
          };
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::LsaReduceMultimemCopy_Generic>()) {
            ncclLsaReduceMultimemCopy<T, decltype(coop),
              decltype(srcLambda), decltype(dstLambda), testOpSum<T>, size_t, UNROLL>(
              coop,
              srcLambda, nRanks,
              dstLambda, 1,
              sumOp,
              chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allreduceKernel: LsaReduceMultimemCopy_Generic not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::LsaReduceSumCopy_DifferentTeams:
        // LSA ReduceSumCopy with different teams.
        // NM mode (nDstStart>0): srcTeam={nSrc ranks from 0}, dstTeam={nDst ranks from nDstStart}.
        // NxN mode (nDstStart==0): both teams are the full LSA team.
        {
          ncclSymPtr<T> srcSymPtr{sendwin, srcOffset};
          ncclSymPtr<T> dstSymPtr{recvwin, dstOffset};
          if (nDstStart > 0) {
            // NM split: construct separate src and dst teams using peerPtr formula.
            // For any calling rank r (lsaRank), the peerPtr formula resolves to:
            //   srcTeam({nSrc, r, 1}): peerPtr(i) = r + (i - r)*1 = i  → rank i's memory
            //   dstTeam({nDst, r-nDstStart, 1}): peerPtr(i) = r + (i-(r-nDstStart))*1 = i+nDstStart
            ncclTeam srcTeam = {nSrc, devComm.lsaRank, 1};
            ncclTeam dstTeam = {nDst, devComm.lsaRank - nDstStart, 1};
            ncclLsaReduceSumCopy<T, decltype(coop), size_t, UNROLL>(
              coop,
              srcSymPtr, srcTeam,
              dstSymPtr, dstTeam,
              chunkCountElts
            );
          } else {
            // NxN mode: use the full LSA team for both src and dst
            ncclTeam team = ncclTeamLsa(devComm);
            ncclLsaReduceSumCopy<T, decltype(coop), size_t, UNROLL>(
              coop,
              srcSymPtr, team,
              dstSymPtr, team,
              chunkCountElts
            );
          }
        }
        break;

      case ApiFunctionId::LsaReduceSumMultimemCopy_SymPtr:
        // LSA -> Multimem ReduceSumCopy with SymPtr
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::LsaReduceSumMultimemCopy_SymPtr>()) {
            ncclTeam team = ncclTeamLsa(devComm);
            ncclMultimemHandle multimemHandle = devComm.lsaMultimem;
            ncclSymPtr<T> srcSymPtr{sendwin, srcOffset};
            ncclSymPtr<T> dstSymPtr{recvwin, dstOffset};
            ncclLsaReduceSumMultimemCopy<T, decltype(coop), size_t, UNROLL>(
              coop,
              srcSymPtr, team,
              dstSymPtr, multimemHandle,
              chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allreduceKernel: LsaReduceSumMultimemCopy_SymPtr not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::LsaReduceSumMultimemCopy_RawPtr:
        // LSA -> Multimem ReduceSumCopy with raw destination pointer
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::LsaReduceSumMultimemCopy_RawPtr>()) {
            ncclTeam team = ncclTeamLsa(devComm);
            ncclSymPtr<T> srcSymPtr{sendwin, srcOffset};
            T* mcDstPtr = (T*)ncclGetLsaMultimemPointer(recvwin, dstOffset, devComm);
            ncclLsaReduceSumMultimemCopy<T, decltype(coop), size_t, UNROLL>(
              coop,
              srcSymPtr, team,
              mcDstPtr,
              chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allreduceKernel: LsaReduceSumMultimemCopy_RawPtr not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::MultimemReduceSumLsaCopy_SymPtr:
        // Multimem -> LSA ReduceSumCopy with SymPtr
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemReduceSumLsaCopy_SymPtr>()) {
            ncclTeam team = ncclTeamLsa(devComm);
            ncclMultimemHandle multimemHandle = devComm.lsaMultimem;
            ncclSymPtr<T> srcSymPtr{sendwin, srcOffset};
            ncclSymPtr<T> dstSymPtr{recvwin, dstOffset};
            callMultimemReduceSumLsaCopy<T, decltype(coop), UNROLL>(
              MultimemSrcTag<T, ApiFunctionId::MultimemReduceSumLsaCopy_SymPtr>{},
              coop, srcSymPtr, multimemHandle, dstSymPtr, team, chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allreduceKernel: MultimemReduceSumLsaCopy_SymPtr not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::MultimemReduceSumLsaCopy_RawPtr:
        // Multimem -> LSA ReduceSumCopy with raw source pointer
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemReduceSumLsaCopy_RawPtr>()) {
            ncclTeam team = ncclTeamLsa(devComm);
            T* mcSrcPtr = (T*)ncclGetLsaMultimemPointer(sendwin, srcOffset, devComm);
            ncclSymPtr<T> dstSymPtr{recvwin, dstOffset};
            callMultimemReduceSumLsaCopy<T, decltype(coop), UNROLL>(
              MultimemSrcTag<T, ApiFunctionId::MultimemReduceSumLsaCopy_RawPtr>{},
              coop, mcSrcPtr, dstSymPtr, team, chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allreduceKernel: MultimemReduceSumLsaCopy_RawPtr not supported for this type.\n");
          }
        }
        break;

      default:
        // Unsupported function ID
        if (threadIdx.x == 0 && blockIdx.x == 0) {
          printf("ERROR: allreduceKernel: Unsupported functionID=%d. Only ReduceSumCopy variants are supported.\n",
               static_cast<int>(functionID));
        }
        break;

    }
  }

  bar.sync(ctaCoop, cuda::memory_order_release);
}

template<typename T, CooperationLevel coopLevel, int UNROLL>
__global__ void reduceSumKernel(
  ApiFunctionId functionID,
  ncclWindow_t sendwin, size_t sendoffset,
  ncclWindow_t recvwin, size_t recvoffset,
  int nSrc, int nDst,
  size_t count,
  ncclDevComm devComm
) {
  // Create cooperative group based on template parameter
  typename CoopTypeOf<coopLevel>::Type coop;

  // Create barrier session (always CTA-level for block synchronization)
  ncclCoopCta ctaCoop;
  const int rank = devComm.rank;
  const int nRanks = devComm.nRanks;
  (void)nDst;
  ncclTeam team = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar{ctaCoop, devComm, team, devComm.lsaBarrier, blockIdx.x};
  bar.sync(ctaCoop, cuda::memory_order_relaxed);

  const size_t countUnits = storageCountForElements<T>(count);

  // Work distribution is per-rank (ReduceScatter semantics)
  size_t chunkStart, chunkCount;
  computeAllGatherWorkDistribution<coopLevel, T>(
    countUnits,
    blockIdx.x, gridDim.x,
    threadIdx.x, blockDim.x,
    false,
    chunkStart, chunkCount
  );
  const size_t chunkCountElts = elementCountForStorageRange<T>(chunkStart, chunkCount, count);

  if (chunkCount > 0) {
    size_t baseSendOffset = (sendoffset != 0)
      ? lambdaOffsetBytes<T>(lambdaSendOffsetElts(rank))
      : sendoffset;
    size_t baseRecvOffset = (recvoffset != 0)
      ? lambdaOffsetBytes<T>(lambdaRecvOffsetElts(rank))
      : recvoffset;
    size_t srcOffset = baseSendOffset + (rank * countUnits + chunkStart) * bytesPerElement<T>();
    char* dstBase = static_cast<char*>(ncclGetLocalPointer(recvwin, baseRecvOffset));
    T* dstPtr = reinterpret_cast<T*>(dstBase + chunkStart * bytesPerElement<T>());

    switch (functionID) {
      case ApiFunctionId::LsaReduceSum_Lambda:
        // LSA ReduceSum with lambda
        {
          auto srcLambda = [=] __device__ (int i) -> T* {
            size_t baseOffset = resolveLambdaSendOffsetBytes<T>(sendoffset, i);
            size_t peerSrcOffset = baseOffset +
                         (rank * countUnits + chunkStart) * bytesPerElement<T>();
            return (T*)ncclGetLsaPointer(sendwin, peerSrcOffset, i);
          };
          ncclLsaReduceSum<T, decltype(coop), decltype(srcLambda), size_t, UNROLL>(
            coop,
            srcLambda, nRanks,
            dstPtr,
            chunkCountElts
          );
        }
        break;

      case ApiFunctionId::LsaReduceSum_SymPtr_Team:
        {
          ncclSymPtr<T> srcSymPtr{sendwin, srcOffset};
          ncclLsaReduceSum<T, decltype(coop), size_t, UNROLL>(
            coop,
            srcSymPtr, dstPtr,
            chunkCountElts,
            team
          );
        }
        break;

      case ApiFunctionId::LsaReduceSum_SymPtr_DevComm:
        {
          ncclSymPtr<T> srcSymPtr{sendwin, srcOffset};
          ncclLsaReduceSum<T, decltype(coop), size_t, UNROLL>(
            coop,
            srcSymPtr, dstPtr,
            chunkCountElts,
            devComm
          );
        }
        break;

      case ApiFunctionId::LsaReduceSum_Window_Team:
        ncclLsaReduceSum<T, decltype(coop), size_t, UNROLL>(
          coop,
          sendwin, srcOffset,
          dstPtr,
          chunkCountElts,
          team
        );
        break;

      case ApiFunctionId::LsaReduceSum_Window_DevComm:
        ncclLsaReduceSum<T, decltype(coop), size_t, UNROLL>(
          coop,
          sendwin, srcOffset,
          dstPtr,
          chunkCountElts,
          devComm
        );
        break;

      case ApiFunctionId::MultimemReduceSum_SymPtr:
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemReduceSum_SymPtr>()) {
            ncclMultimemHandle multimemHandle = devComm.lsaMultimem;
            ncclSymPtr<T> srcSymPtr{sendwin, srcOffset};
            callMultimemReduceSum<T, decltype(coop), UNROLL>(
              MultimemSrcTag<T, ApiFunctionId::MultimemReduceSum_SymPtr>{},
              coop, srcSymPtr, dstPtr, chunkCountElts, multimemHandle
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: reduceSumKernel: MultimemReduceSum_SymPtr not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::MultimemReduceSum_RawPtr:
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemReduceSum_RawPtr>()) {
            T* mcSrcPtr = (T*)ncclGetLsaMultimemPointer(sendwin, srcOffset, devComm);
            callMultimemReduceSum<T, decltype(coop), UNROLL>(
              MultimemSrcTag<T, ApiFunctionId::MultimemReduceSum_RawPtr>{},
              coop, mcSrcPtr, dstPtr, chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: reduceSumKernel: MultimemReduceSum_RawPtr not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::MultimemReduceSum_Window:
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemReduceSum_Window>()) {
            ncclMultimemHandle multimemHandle = devComm.lsaMultimem;
            ncclSymPtr<T> srcSymPtr{sendwin, srcOffset};
            callMultimemReduceSum<T, decltype(coop), UNROLL>(
              MultimemSrcTag<T, ApiFunctionId::MultimemReduceSum_Window>{},
              coop, srcSymPtr, dstPtr, chunkCountElts, multimemHandle
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: reduceSumKernel: MultimemReduceSum_Window not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::LocalReduceSum_Lambda:
        {
          auto srcLambda = [=] __device__ (int i) -> T* {
            size_t localOffset = sendoffset + (i * countUnits + chunkStart) * bytesPerElement<T>();
            return (T*)ncclGetLocalPointer(sendwin, localOffset);
          };
          ncclLocalReduceSum<T, decltype(coop), decltype(srcLambda), size_t, UNROLL>(
            coop,
            srcLambda, nSrc,
            dstPtr,
            chunkCountElts
          );
        }
        break;

      case ApiFunctionId::LocalReduceSum_Strided:
        {
          char* basePtrBytes = static_cast<char*>(ncclGetLocalPointer(sendwin, sendoffset));
          T* basePtr = reinterpret_cast<T*>(basePtrBytes + chunkStart * bytesPerElement<T>());
          size_t displ = countUnits;
          ncclLocalReduceSum<T, decltype(coop), size_t, UNROLL>(
            coop,
            nSrc, basePtr, displ,
            dstPtr,
            chunkCountElts
          );
        }
        break;

      default:
        if (threadIdx.x == 0 && blockIdx.x == 0) {
          printf("ERROR: reduceSumKernel: Unsupported functionID=%d. Only ReduceSum variants are supported.\n",
               static_cast<int>(functionID));
        }
        break;
    }
  }

  bar.sync(ctaCoop, cuda::memory_order_release);
}

// Single launcher function for ReduceSumCopy variants
// Template parameters: T, UNROLL, CoopLevel
// Automatically instantiates the kernel when called
template<typename T, int UNROLL, CooperationLevel CoopLevel>
void launchAllReduceKernel(
  ApiFunctionId functionID,
  ncclWindow_t sendwin, size_t sendoffset,
  ncclWindow_t recvwin, size_t recvoffset,
  int nSrc, int nDst, int nDstStart,
  size_t count,
  ncclDevComm devComm,
  int gridSize, int blockSize,
  cudaStream_t stream
) {
  // Verify this is a ReduceSumCopy variant
  if (!isReduceSumCopyVariantConstexpr(functionID)) {
    printf("ERROR: launchAllReduceKernel called with non-ReduceSumCopy functionID=%d\n",
         static_cast<int>(functionID));
    return;
  }

  // Launch the kernel - this will automatically instantiate the template
  allreduceKernel<T, CoopLevel, UNROLL><<<gridSize, blockSize, 0, stream>>>(
    functionID,
    sendwin, sendoffset,
    recvwin, recvoffset,
    nSrc, nDst, nDstStart,
    count,
    devComm
  );
}

// Single launcher function for ReduceSum variants
// Template parameters: T, UNROLL, CoopLevel
// Automatically instantiates the kernel when called
template<typename T, int UNROLL, CooperationLevel CoopLevel>
void launchReduceSumKernel(
  ApiFunctionId functionID,
  ncclWindow_t sendwin, size_t sendoffset,
  ncclWindow_t recvwin, size_t recvoffset,
  int nSrc, int nDst,
  size_t count,
  ncclDevComm devComm,
  int gridSize, int blockSize,
  cudaStream_t stream
) {
  // Launch the kernel - this will automatically instantiate the template
  reduceSumKernel<T, CoopLevel, UNROLL><<<gridSize, blockSize, 0, stream>>>(
    functionID,
    sendwin, sendoffset,
    recvwin, recvoffset,
    nSrc, nDst,
    count,
    devComm
  );
}



template<typename T, CooperationLevel coopLevel, int UNROLL>
__global__ void allGatherKernel(
  ApiFunctionId functionID,
  ncclWindow_t sendwin, size_t sendoffset,
  ncclWindow_t recvwin, size_t recvoffset,
  int nSrc, int nDst,
  size_t count,
  ncclDevComm devComm
) {
  // Create cooperative group based on template parameter
  typename CoopTypeOf<coopLevel>::Type coop;

  // Create barrier session (always CTA-level for block synchronization)
  // Always use non-multimem barrier path (multimem is an optimization for copy operations, not barriers)
  ncclCoopCta ctaCoop;

  // Determine if this is a multimem variant for work distribution (needed for CTA-level multimem divisibility)
  bool isMultimem = isMultimemVariantConstexpr(functionID);

  (void)nSrc;

  const int rank = devComm.rank;
  const int nRanks = devComm.nRanks;
  ncclTeam team = ncclTeamLsa(devComm);
  ncclLsaBarrierSession<ncclCoopCta> bar{ctaCoop, devComm, team, devComm.lsaBarrier, blockIdx.x};
  bar.sync(ctaCoop, cuda::memory_order_relaxed);

  const size_t countUnits = storageCountForElements<T>(count);

  // For allGather, work distribution is per-rank (not global across ranks)
  // Each thread/warp/block processes a chunk of this rank's send buffer
  size_t chunkStart, chunkCount;
  computeAllGatherWorkDistribution<coopLevel, T>(
    countUnits,
    blockIdx.x, gridDim.x,
    threadIdx.x, blockDim.x,
    isMultimem,
    chunkStart, chunkCount
  );
  const size_t chunkCountElts = elementCountForStorageRange<T>(chunkStart, chunkCount, count);

  // Process this thread/warp/block's assigned chunk
  if (chunkCount > 0) {
    // Get pointer to this rank's send buffer (source)
    size_t baseSendOffset = (sendoffset != 0)
      ? lambdaOffsetBytes<T>(lambdaSendOffsetElts(rank))
      : sendoffset;
    size_t baseRecvOffset = (recvoffset != 0)
      ? lambdaOffsetBytes<T>(lambdaRecvOffsetElts(rank))
      : recvoffset;
    char* mySendBase = static_cast<char*>(ncclGetLocalPointer(sendwin, baseSendOffset));
    T* srcPtr = reinterpret_cast<T*>(mySendBase + chunkStart * bytesPerElement<T>());

    // Calculate destination offset for this chunk's portion
    // Points to (rank * count + chunkStart) in each peer's receive buffer
    size_t dstOffset = baseRecvOffset + (rank * countUnits + chunkStart) * bytesPerElement<T>();

    // Use the appropriate copy function based on functionID
    switch (functionID) {
      case ApiFunctionId::LsaCopy_Lambda:
        // LSA Copy with lambda (1->N, but for AllGather: each rank contributes)
        {
          // Use team computed at the beginning
          // Create lambda that returns pointers for each destination rank
          auto dstLambda = [=] __device__ (int i) -> T* {
            size_t peerBase = resolveLambdaRecvOffsetBytes<T>(recvoffset, i);
            size_t peerDstOffset = peerBase + (rank * countUnits + chunkStart) * bytesPerElement<T>();
            return (T*)ncclGetLsaPointer(recvwin, peerDstOffset, i);
          };
          ncclLsaCopy<T, decltype(coop), decltype(dstLambda), size_t, UNROLL>(
            coop,
            srcPtr, dstLambda, nRanks,
            chunkCountElts
          );
        }
        break;

      case ApiFunctionId::LsaCopy_SymPtr_Team:
        // LSA Copy with SymPtr and team
        {
          // Use team computed at the beginning
          ncclSymPtr<T> dstSymPtr{recvwin, dstOffset};
          ncclLsaCopy<T, decltype(coop), size_t, UNROLL>(
            coop,
            srcPtr, dstSymPtr,
            chunkCountElts, team
          );
        }
        break;

      case ApiFunctionId::LsaCopy_SymPtr_DevComm:
        // LSA Copy with SymPtr and devComm
        {
          ncclSymPtr<T> dstSymPtr{recvwin, dstOffset};
          ncclLsaCopy<T, decltype(coop), size_t, UNROLL>(
            coop,
            srcPtr, dstSymPtr,
            chunkCountElts, devComm
          );
        }
        break;

      case ApiFunctionId::LsaCopy_Window_Team:
        // LSA Copy with window and team
        {
          // Use team computed at the beginning
          ncclLsaCopy<T, decltype(coop), size_t, UNROLL>(
            coop,
            srcPtr, recvwin, dstOffset,
            chunkCountElts, team
          );
        }
        break;

      case ApiFunctionId::LsaCopy_Window_DevComm:
        // LSA Copy with window and devComm
        {
          ncclLsaCopy<T, decltype(coop), size_t, UNROLL>(
            coop,
            srcPtr, recvwin, dstOffset,
            chunkCountElts, devComm
          );
        }
        break;

      case ApiFunctionId::MultimemCopy_SymPtr:
        // Multimem Copy with SymPtr
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemCopy_SymPtr>()) {
            ncclMultimemHandle multimemHandle = devComm.lsaMultimem;
            ncclSymPtr<T> dstSymPtr{recvwin, dstOffset};
            ncclMultimemCopy<T, decltype(coop), size_t, UNROLL>(
              coop,
              srcPtr, dstSymPtr,
              chunkCountElts, multimemHandle
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allGatherKernel: MultimemCopy_SymPtr not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::MultimemCopy_RawPtr:
        // Multimem Copy with raw pointer
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemCopy_RawPtr>()) {
            // For AllGather, we need to copy to all ranks' multimem destinations
            // Get the multimem pointer for this rank's portion
            T* mcDstPtr = (T*)ncclGetLsaMultimemPointer(recvwin, dstOffset, devComm);
            ncclMultimemCopy<T, decltype(coop), size_t, UNROLL>(
              coop,
              srcPtr, mcDstPtr,
              chunkCountElts
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allGatherKernel: MultimemCopy_RawPtr not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::MultimemCopy_Window:
        // Multimem Copy with windows
        {
          if NCCL_IF_CONSTEXPR (multimemAccessSupportedForFunc<T, ApiFunctionId::MultimemCopy_Window>()) {
            ncclMultimemHandle multimemHandle = devComm.lsaMultimem;
            ncclMultimemCopy<T, decltype(coop), size_t, UNROLL>(
              coop,
              srcPtr, recvwin, dstOffset,
              chunkCountElts, multimemHandle
            );
          } else if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("ERROR: allGatherKernel: MultimemCopy_Window not supported for this type.\n");
          }
        }
        break;

      case ApiFunctionId::LocalCopy_Lambda:
        // Local Copy with lambda (for single-device tests)
        {
          // Create lambda that returns pointers for each destination
          auto dstLambda = [=] __device__ (int i) -> T* {
            size_t peerDstOffset = recvoffset + (i * countUnits + chunkStart) * bytesPerElement<T>();
            return (T*)ncclGetLocalPointer(recvwin, peerDstOffset);
          };
          ncclLocalCopy<T, decltype(coop), decltype(dstLambda), size_t, UNROLL>(
            coop,
            srcPtr, dstLambda, nDst,
            chunkCountElts
          );
        }
        break;

      case ApiFunctionId::LocalCopy_Strided:
        // Local Copy with strided access (for single-device tests)
        {
          // For AllGather, copy from srcPtr to all ranks' destinations
          // Each rank's data goes to offset (rank * count + chunkStart) in the receive buffer
          char* basePtrBytes = static_cast<char*>(ncclGetLocalPointer(recvwin, recvoffset));
          T* basePtr = reinterpret_cast<T*>(basePtrBytes + chunkStart * bytesPerElement<T>());
          size_t displ = countUnits;  // Stride between destinations (elements)
          ncclLocalCopy<T, decltype(coop), size_t, UNROLL>(
            coop,
            srcPtr, nDst, basePtr, displ,
            chunkCountElts
          );
        }
        break;

      default:
        // Unsupported function ID
        if (threadIdx.x == 0 && blockIdx.x == 0) {
          printf("ERROR: allGatherKernel: Unsupported functionID=%d. Only Copy variants are supported.\n",
               static_cast<int>(functionID));
        }
        break;
    }
  }

  bar.sync(ctaCoop, cuda::memory_order_release);
}

// Single launcher function for AllGather variants
// Template parameters: T, UNROLL, CoopLevel
// Automatically instantiates the kernel when called
template<typename T, int UNROLL, CooperationLevel CoopLevel>
void launchAllGatherKernel(
  ApiFunctionId functionID,
  ncclWindow_t sendwin, size_t sendoffset,
  ncclWindow_t recvwin, size_t recvoffset,
  int nSrc, int nDst,
  size_t count,
  ncclDevComm devComm,
  int gridSize, int blockSize,
  cudaStream_t stream
) {
  // Launch the kernel - this will automatically instantiate the template
  allGatherKernel<T, CoopLevel, UNROLL><<<gridSize, blockSize, 0, stream>>>(
    functionID,
    sendwin, sendoffset,
    recvwin, recvoffset,
    nSrc, nDst,
    count,
    devComm
  );
}

#define NCCL_REDUCE_COPY_DECLARE_LAUNCHERS_FOR_UNROLL(T, UNROLL_EXPR) \
  extern template void launchReduceSumKernel<T, UNROLL_EXPR, CooperationLevel::Thread>( \
    ApiFunctionId functionID, \
    ncclWindow_t sendwin, size_t sendoffset, \
    ncclWindow_t recvwin, size_t recvoffset, \
    int nSrc, int nDst, \
    size_t count, \
    ncclDevComm devComm, \
    int gridSize, int blockSize, \
    cudaStream_t stream); \
  extern template void launchReduceSumKernel<T, UNROLL_EXPR, CooperationLevel::Warp>( \
    ApiFunctionId functionID, \
    ncclWindow_t sendwin, size_t sendoffset, \
    ncclWindow_t recvwin, size_t recvoffset, \
    int nSrc, int nDst, \
    size_t count, \
    ncclDevComm devComm, \
    int gridSize, int blockSize, \
    cudaStream_t stream); \
  extern template void launchReduceSumKernel<T, UNROLL_EXPR, CooperationLevel::Cta>( \
    ApiFunctionId functionID, \
    ncclWindow_t sendwin, size_t sendoffset, \
    ncclWindow_t recvwin, size_t recvoffset, \
    int nSrc, int nDst, \
    size_t count, \
    ncclDevComm devComm, \
    int gridSize, int blockSize, \
    cudaStream_t stream); \
  extern template void launchAllReduceKernel<T, UNROLL_EXPR, CooperationLevel::Thread>( \
    ApiFunctionId functionID, \
    ncclWindow_t sendwin, size_t sendoffset, \
    ncclWindow_t recvwin, size_t recvoffset, \
    int nSrc, int nDst, int nDstStart, \
    size_t count, \
    ncclDevComm devComm, \
    int gridSize, int blockSize, \
    cudaStream_t stream); \
  extern template void launchAllReduceKernel<T, UNROLL_EXPR, CooperationLevel::Warp>( \
    ApiFunctionId functionID, \
    ncclWindow_t sendwin, size_t sendoffset, \
    ncclWindow_t recvwin, size_t recvoffset, \
    int nSrc, int nDst, int nDstStart, \
    size_t count, \
    ncclDevComm devComm, \
    int gridSize, int blockSize, \
    cudaStream_t stream); \
  extern template void launchAllReduceKernel<T, UNROLL_EXPR, CooperationLevel::Cta>( \
    ApiFunctionId functionID, \
    ncclWindow_t sendwin, size_t sendoffset, \
    ncclWindow_t recvwin, size_t recvoffset, \
    int nSrc, int nDst, int nDstStart, \
    size_t count, \
    ncclDevComm devComm, \
    int gridSize, int blockSize, \
    cudaStream_t stream);

#define NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(T) \
  NCCL_REDUCE_COPY_DECLARE_LAUNCHERS_FOR_UNROLL(T, 1) \
  NCCL_REDUCE_COPY_DECLARE_LAUNCHERS_FOR_UNROLL(T, getDefaultUnroll<T>())

// Separate macro for AllGather launcher extern templates: only declared for copyOnly types
// (those with inst_gather_* instantiation files).  Non-copyOnly types omit these
// extern template declarations so the compiler inlines from the template definition.
#define NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS_FOR_UNROLL(T, UNROLL_EXPR) \
  extern template void launchAllGatherKernel<T, UNROLL_EXPR, CooperationLevel::Thread>( \
    ApiFunctionId functionID, \
    ncclWindow_t sendwin, size_t sendoffset, \
    ncclWindow_t recvwin, size_t recvoffset, \
    int nSrc, int nDst, \
    size_t count, \
    ncclDevComm devComm, \
    int gridSize, int blockSize, \
    cudaStream_t stream); \
  extern template void launchAllGatherKernel<T, UNROLL_EXPR, CooperationLevel::Warp>( \
    ApiFunctionId functionID, \
    ncclWindow_t sendwin, size_t sendoffset, \
    ncclWindow_t recvwin, size_t recvoffset, \
    int nSrc, int nDst, \
    size_t count, \
    ncclDevComm devComm, \
    int gridSize, int blockSize, \
    cudaStream_t stream); \
  extern template void launchAllGatherKernel<T, UNROLL_EXPR, CooperationLevel::Cta>( \
    ApiFunctionId functionID, \
    ncclWindow_t sendwin, size_t sendoffset, \
    ncclWindow_t recvwin, size_t recvoffset, \
    int nSrc, int nDst, \
    size_t count, \
    ncclDevComm devComm, \
    int gridSize, int blockSize, \
    cudaStream_t stream);

#define NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS(T) \
  NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS_FOR_UNROLL(T, 1) \
  NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS_FOR_UNROLL(T, getDefaultUnroll<T>())

NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(float)
NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(double)
NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(int)
NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(unsigned int)
NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(int8_t)
NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(uint8_t)
NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(long long)
NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(unsigned long long)
NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(half)
#if defined(__CUDA_BF16_TYPES_EXIST__)
NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(__nv_bfloat16)
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(__nv_fp8_e4m3)
NCCL_REDUCE_COPY_DECLARE_LAUNCHERS(__nv_fp8_e5m2)
#endif

// AllGather extern templates: only for copyOnly types (int8_t, float, double, bf16, fp8).
// Matches the set of types for which inst_gather_* instantiation files are generated.
NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS(int8_t)
NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS(float)
NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS(double)
#if defined(__CUDA_BF16_TYPES_EXIST__)
NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS(__nv_bfloat16)
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS(__nv_fp8_e4m3)
NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS(__nv_fp8_e5m2)
#endif

#undef NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS
#undef NCCL_REDUCE_COPY_DECLARE_ALLGATHER_LAUNCHERS_FOR_UNROLL
#undef NCCL_REDUCE_COPY_DECLARE_LAUNCHERS
#undef NCCL_REDUCE_COPY_DECLARE_LAUNCHERS_FOR_UNROLL

#endif // _REDUCE_COPY_TEST_KERNELS_CUH_

