/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_DATA_H_
#define _REDUCE_COPY_TEST_DATA_H_

#include <vector>
#include <cstddef>
#include <cstdint>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <type_traits>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "util.h"

// CPU-side test data initialization utilities
// Simple deterministic pattern for testing (can be enhanced later with verifiable system)

template<typename T>
class TestDataInitializer {
public:
  // Initialize test data with a deterministic pattern
  // For inter-rank: each rank gets different data
  // For local: each chunk gets different data
  // useLargerVariance: when true (custom-op sum path), scale up slightly so real errors are easier to spot
  // useMulVariant: when true (OpMul path), scale down by 1/(nSrc^2) so product of nSrc values stays in range
  static void initializeTestData(
    std::vector<T>& data,
    int sourceIndex,  // Rank index or chunk index
    size_t count,
    size_t elementCount,
    bool useLargerVariance = false,
    bool useMulVariant = false,
    int nSrc = 1
  ) {
    data.resize(count);

    if NCCL_IF_CONSTEXPR (!std::is_integral<T>::value && !std::is_floating_point<T>::value) {
      // Low-precision float: one generator per rank, N(0,1) scaled by type
      const uint64_t seed = getReduceCopyTestSeed();
      std::mt19937_64 gen(seed + static_cast<uint64_t>(sourceIndex) * 0x10001ULL);
      std::normal_distribution<double> dist(0.0, 1.0);
      double scale = 23.0;  // half / bfloat16: ~±100
      #if defined(__CUDA_FP8_TYPES_EXIST__)
      if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_fp8_e4m3>::value) scale = 12.0;   // e4m3 max 448, use more range
      else if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_fp8_e5m2>::value) scale = 100.0;  // e5m2 max 57344
      #endif
      if (useMulVariant && nSrc > 1) {
        scale /= (static_cast<double>(nSrc) * nSrc);
      } else if (useLargerVariance) {
        scale *= 1.25;
      }

      // Mul variant: hard clamp so product of nSrc values cannot overflow (limit^nSrc in range)
      constexpr double kSafety = 0.5;
      auto mulClampLimit = [nSrc]() -> double {
        if NCCL_IF_CONSTEXPR (std::is_same<T, half>::value) {
          constexpr double kHalfMax = 65504.0;
          return kSafety * std::pow(kHalfMax, 1.0 / static_cast<double>(nSrc));
        }
        #if defined(__CUDA_BF16_TYPES_EXIST__)
        else if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_bfloat16>::value) {
          // bf16: 1 sign, 8 exponent, 7 mantissa → max = (2 - 2^-7) * 2^127
          constexpr double kBf16Max = (2.0 - 1.0 / 128.0) * 0x1.0p127;
          return kSafety * std::pow(kBf16Max, 1.0 / static_cast<double>(nSrc));
        }
        #endif
        #if defined(__CUDA_FP8_TYPES_EXIST__)
        else if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_fp8_e4m3>::value) {
          constexpr double kFp8E4m3Max = 448.0;
          return kSafety * std::pow(kFp8E4m3Max, 1.0 / static_cast<double>(nSrc));
        }
        else if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_fp8_e5m2>::value) {
          constexpr double kFp8E5m2Max = 57344.0;
          return kSafety * std::pow(kFp8E5m2Max, 1.0 / static_cast<double>(nSrc));
        }
        #endif
        else return 1.0;
      };
      const double clampLimit = (useMulVariant && nSrc > 1) ? mulClampLimit() : 0.0;  // 0 = no clamp

      {
        for (size_t i = 0; i < count; ++i) {
          double v = dist(gen) * scale;
          if (clampLimit > 0) {
            v = std::max(-clampLimit, std::min(clampLimit, v));
          }
          data[i] = static_cast<T>(static_cast<float>(v));
        }
      }
    } else if NCCL_IF_CONSTEXPR (std::is_floating_point<T>::value) {
      // Native float/double: N(0,1) with scaling; for mul variant clamp using
      // nSrc+2 as exponent divisor (extra margin vs nSrc) so product of nSrc
      // values stays well below overflow even with 8 GPUs.
      const uint64_t seed = getReduceCopyTestSeed();
      std::mt19937_64 gen(seed + static_cast<uint64_t>(sourceIndex) * 0x10001ULL);
      std::normal_distribution<double> dist(0.0, 1.0);
      double scale = 1.0;
      if (useMulVariant && nSrc > 1) {
        scale /= (static_cast<double>(nSrc) * nSrc);
      } else if (useLargerVariance) {
        scale *= 1.25;
      }
      constexpr double kSafety = 0.5;
      const double kMax = static_cast<double>(std::numeric_limits<T>::max());
      const double clampLimit = (useMulVariant && nSrc > 1)
        ? kSafety * std::pow(kMax, 1.0 / static_cast<double>(nSrc + 2))
        : 0.0;
      for (size_t i = 0; i < count; ++i) {
        double v = dist(gen) * scale;
        if (clampLimit > 0.0) {
          v = std::max(-clampLimit, std::min(clampLimit, v));
        }
        data[i] = static_cast<T>(v);
      }
    } else {
      // Integral types
      for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(sourceIndex * static_cast<int>(count) + static_cast<int>(i));
      }
    }
  }

  static void initializeTestData(
    std::vector<T>& data,
    int sourceIndex,  // Rank index or chunk index
    size_t count
  ) {
    initializeTestData(data, sourceIndex, count, count, false, false, 1);
  }

  static void initializeTestData(
    std::vector<T>& data,
    int sourceIndex,
    size_t count,
    bool useLargerVariance,
    bool useMulVariant = false,
    int nSrc = 1
  ) {
    initializeTestData(data, sourceIndex, count, count, useLargerVariance, useMulVariant, nSrc);
  }

};

#endif // _REDUCE_COPY_TEST_DATA_H_

