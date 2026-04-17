/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_CONFIG_H_
#define _REDUCE_COPY_TEST_CONFIG_H_

#include <cstddef>
#include <cstdint>
#include "api_function_traits.h"

// C++14/17 compatibility: mirrors NCCL_IF_CONSTEXPR in src/include/nccl_device/utility.h.
// Use as:  if NCCL_IF_CONSTEXPR (cond) { ... }
// On C++17: becomes 'if constexpr'; on C++14: becomes plain 'if'.
// All if-NCCL_IF_CONSTEXPR bodies must be type-safe under plain 'if' (no
// type-specific intrinsics in dead branches).  Use explicit template
// specialisations instead for bodies that call type-specific intrinsics.
#if !defined(NCCL_IF_CONSTEXPR)
  #if defined(__cpp_if_constexpr) && __cpp_if_constexpr >= 201606
  #define NCCL_IF_CONSTEXPR constexpr
  #else
  #define NCCL_IF_CONSTEXPR
  #endif
#endif

#if defined(__CUDACC__)
#define NCCL_TEST_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define NCCL_TEST_HOST_DEVICE inline
#endif

template <typename T>
NCCL_TEST_HOST_DEVICE constexpr int bitSizeOfElement() { return 8 * (int)sizeof(T); }

template <typename T>
NCCL_TEST_HOST_DEVICE constexpr size_t bytesPerElement() { return sizeof(T); }

#undef NCCL_TEST_HOST_DEVICE

// Single source of truth for test count values (shared by C++ and python generator)
#define NCCL_REDUCE_COPY_COUNTS_LIST 0, 1, 2, 997, 32768

// Lambda offset helpers (full elements)
constexpr size_t kMaxLambdaOffsetElts = 7;
constexpr size_t kLambdaSendOffsetStride = 3;
constexpr size_t kLambdaRecvOffsetStride = 5;
constexpr size_t kLambdaRecvOffsetBias = 1;

__host__ __device__ constexpr size_t lambdaSendOffsetElts(int rank) {
  return (static_cast<size_t>(rank) * kLambdaSendOffsetStride) % (kMaxLambdaOffsetElts + 1);
}

__host__ __device__ constexpr size_t lambdaRecvOffsetElts(int rank) {
  return (static_cast<size_t>(rank) * kLambdaRecvOffsetStride + kLambdaRecvOffsetBias) %
       (kMaxLambdaOffsetElts + 1);
}

// Cooperation level for kernel execution
enum class CooperationLevel {
  Thread,  // ncclCoopThread - single thread
  Warp,    // ncclCoopWarp - warp-level (32 threads)
  Cta      // ncclCoopCta - CTA/block-level
};

// Test configuration structure
struct TestConfig {
  ApiFunctionId functionId;
  size_t count;              // Number of elements per source/destination
  int nSrc;                  // Number of sources (ranks for inter-rank, chunks for local)
  int nDst;                  // Number of destinations (ranks for inter-rank, chunks for local)
  CooperationLevel coopLevel; // Cooperation level (Thread, Warp, Cta)
  int unroll;                // UNROLL template parameter value
  int gridSize;              // Number of blocks per grid
  int blockSize;             // Number of threads per block
};

// Helper functions for UNROLL calculation
template<typename T>
constexpr int getDefaultUnroll() {
  return (4 * 16 * 8) / bitSizeOfElement<T>();  // 64 bytes worth of bits
}

#endif // _REDUCE_COPY_TEST_CONFIG_H_

