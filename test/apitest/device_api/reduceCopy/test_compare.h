/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_REDUCE_COPY_TEST_COMPARE_H
#define NCCL_REDUCE_COPY_TEST_COMPARE_H

#include <cmath>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif
#include "acc_type_trait.cuh"

// C++14 replacement for std::optional<double>
struct OptDouble {
  OptDouble() : hasVal(false), val(0.0) {}
  explicit OptDouble(double v) : hasVal(true), val(v) {}
  OptDouble& operator=(double v) { hasVal = true; val = v; return *this; }
  bool has_value() const { return hasVal; }
  double operator*() const { return val; }
private:
  bool hasVal;
  double val;
};

// Output type mantissa bits (for ULP-based sum comparison tolerance).
template<typename T> struct OutputMantissaBits { static constexpr int value = 23; };
template<> struct OutputMantissaBits<half> { static constexpr int value = 10; };
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<> struct OutputMantissaBits<__nv_bfloat16> { static constexpr int value = 7; };
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template<> struct OutputMantissaBits<__nv_fp8_e4m3> { static constexpr int value = 3; };
template<> struct OutputMantissaBits<__nv_fp8_e5m2> { static constexpr int value = 2; };
#endif
template<> struct OutputMantissaBits<float> { static constexpr int value = 23; };
template<> struct OutputMantissaBits<double> { static constexpr int value = 52; };

// Returns true if |v1 - v2| is within the type-dependent ULP tolerance used for sum comparisons (nSrc >= 2).
// relaxedSumUlp: when true (Generic/custom-op path), allow more ULPs for bf16/half rounding spread.
// maxAbsSource: when provided, scale M = max(*maxAbsSource, |v1|, |v2|) so tolerance scales with input magnitude.
inline bool withinSumUlpTolerance(double v1, double v2, int mantissaBits, bool relaxedSumUlp = false,
                 OptDouble maxAbsSource = OptDouble()) {
  int ulp_floor;
  switch (mantissaBits) {
    case 7:   ulp_floor = relaxedSumUlp ? 32 : 16; break;  // bf16: relaxed for Generic
    case 10:  ulp_floor = relaxedSumUlp ? 16 : 12; break;   // half: relaxed for Generic
    case 2: case 3:  ulp_floor = 16; break;                // fp8
    case 23: case 52:
    default:  ulp_floor = 8; break;                       // float, double
  }
  double M = std::max(std::max(std::fabs(v1), std::fabs(v2)), 1e-10);
  if (maxAbsSource.has_value()) {
    M = std::max(M, *maxAbsSource);
  }
  const double ulp_tol = static_cast<double>(ulp_floor) * M * std::ldexp(1.0, -mantissaBits);
  return std::fabs(v1 - v2) <= ulp_tol;
}

// Helper for the native float/double branch of valuesEqual (float, double).
// Isolated in a struct to avoid C++14 dead-branch type-checking errors:
// in C++14, `if NCCL_IF_CONSTEXPR` does not suppress instantiation of dead branches,
// so std::isnan / std::abs / constexpr T(...) for bfloat16/fp8 would fail without this.
template<typename T, bool IsNativeFloat>
struct NativeFloatValuesEqual {
  // Unused no-op for non-native-float types (e.g. bfloat16, fp8) in dead branches.
  static bool compare(T, T, int, bool, OptDouble) { return false; }
};
template<typename T>
struct NativeFloatValuesEqual<T, true> {
  static bool compare(T expected, T actual, int nSrc, bool strictLowPrecisionTolerance,
            OptDouble maxAbsSource) {
    if (std::isnan(expected) || std::isnan(actual) || std::isinf(expected) || std::isinf(actual)) {
      return false;
    }
    if (nSrc >= 2 && withinSumUlpTolerance(static_cast<double>(expected), static_cast<double>(actual),
                         OutputMantissaBits<T>::value, !strictLowPrecisionTolerance,
                         maxAbsSource))
      return true;
    // Legacy: relative tolerance
    T tolerance = std::is_same<T, float>::value ? T(1e-5) : T(1e-10);
    T diff = std::abs(expected - actual);
    T maxVal = std::max(std::abs(expected), std::abs(actual));
    if (maxVal < tolerance) return diff < tolerance;
    return diff <= maxVal * tolerance;
  }
};

// Helper for the integral else-branch of valuesEqual (int, uint, int8_t, etc.).
// Isolated to prevent C++14 dead-branch instantiation of operator== for fp8 types,
// which have no operator== and would cause a compile error on GCC 9.
template<typename T, bool IsIntegral>
struct IntegralEqualityCheck {
  // Stub for non-integral types (fp8) — never reached at runtime.
  static bool check(T, T) { return false; }
};
template<typename T>
struct IntegralEqualityCheck<T, true> {
  static bool check(T a, T b) { return a == b; }
};

// When nSrc >= 2, use ULP-based tolerance for sum comparisons (type-dependent ULP count).
// Otherwise legacy. Pass nSrc=0 to skip.
// maxAbsSource: when provided, ULP scale uses max of source element magnitudes (single authority for sum tolerance).
template<typename T>
bool valuesEqual(T expected, T actual, bool strictLowPrecisionTolerance,
         int nSrc, OptDouble maxAbsSource = OptDouble()) {
  // Low-precision types: convert to AccType, then to float; do normal ops (compare) in float.
  constexpr bool useAccTypeCompare = std::is_same<T, half>::value
    #if defined(__CUDA_BF16_TYPES_EXIST__)
    || std::is_same<T, __nv_bfloat16>::value
    #endif
    #if defined(__CUDA_FP8_TYPES_EXIST__)
    || std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value
    #endif
    ;
  if NCCL_IF_CONSTEXPR (useAccTypeCompare) {
    using AccType = typename TestAccTypeMap<T>::AccType;
    AccType exp_acc = toAccType(expected);
    AccType act_acc = toAccType(actual);
    float exp_f = accTypeToFloat(exp_acc);
    float act_f = accTypeToFloat(act_acc);
    if (std::isnan(exp_f) || std::isnan(act_f) || std::isinf(exp_f) || std::isinf(act_f)) {
      return false;
    }
    float diff = std::abs(exp_f - act_f);
    if (nSrc >= 2 && withinSumUlpTolerance(static_cast<double>(exp_f), static_cast<double>(act_f),
          OutputMantissaBits<T>::value, !strictLowPrecisionTolerance, maxAbsSource))
      return true;
    float maxVal = std::max(std::abs(exp_f), std::abs(act_f));
    // Legacy tolerance (when model not used)
    if NCCL_IF_CONSTEXPR (std::is_same<AccType, float>::value && std::is_same<T, half>::value) {
      if (strictLowPrecisionTolerance) {
        constexpr float halfMinNorm = 1.0f / (1 << 14);
        float scale = (maxVal >= halfMinNorm) ? std::scalbn(1.0f, std::ilogb(maxVal)) : halfMinNorm;
        float halfUlp = scale / 1024.0f;
        // ULP-based with minimum 0.1: library reduce_copy matches reference semantics; codegen can differ.
        const int ulpSlack = 64;
        const float ulpTolerance = static_cast<float>(ulpSlack) * halfUlp;
        constexpr float minHalfTolerance = 0.1f;
        return diff <= std::max(ulpTolerance, minHalfTolerance);
      }
      // Custom op (Generic): round after each step → use relative tolerance (tree order can diverge more)
      const float tolerance = 0.35f;
      if (maxVal < tolerance) return diff < tolerance;
      return diff <= maxVal * tolerance;
    } else if NCCL_IF_CONSTEXPR (std::is_same<AccType, float>::value) {
      const float tolerance = strictLowPrecisionTolerance ? 0.02f : 0.05f;
      if (maxVal < tolerance) return diff < tolerance;
      return diff <= maxVal * tolerance;
    } else {
      // AccType is half (fp8)
      const float tolerance = strictLowPrecisionTolerance
        ? (std::is_same<T, __nv_fp8_e4m3>::value ? 0.2f : 0.3f)
        : (std::is_same<T, __nv_fp8_e4m3>::value ? 0.4f : 0.6f);
      if (maxVal < tolerance) return diff < tolerance;
      return diff <= maxVal * tolerance;
    }
  }
  else if NCCL_IF_CONSTEXPR (std::is_floating_point<T>::value) {
    return NativeFloatValuesEqual<T, std::is_floating_point<T>::value>::compare(
      expected, actual, nSrc, strictLowPrecisionTolerance, maxAbsSource);
  } else {
    // For integral types, use exact equality.
    // IntegralEqualityCheck guards against C++14 dead-branch instantiation of
    // operator== for fp8 types (no operator== on GCC 9).
    return IntegralEqualityCheck<T, std::is_integral<T>::value>::check(expected, actual);
  }
}

// Helper function to convert to float for display; uses TestAccTypeMap (T -> AccType -> float).
template<typename T>
float toFloatForDisplay(T value) {
  constexpr bool useAcc = std::is_same<T, half>::value
    #if defined(__CUDA_BF16_TYPES_EXIST__)
    || std::is_same<T, __nv_bfloat16>::value
    #endif
    #if defined(__CUDA_FP8_TYPES_EXIST__)
    || std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value
    #endif
    ;
  if NCCL_IF_CONSTEXPR (useAcc) {
    return accTypeToFloat(toAccType(value));
  }
  else {
    return static_cast<float>(value);
  }
}

// Helper function to check if a value is zero (handles low-precision types)
template<typename T>
bool isZero(T value) {
  if NCCL_IF_CONSTEXPR (std::is_same<T, half>::value ||
          #if defined(__CUDA_BF16_TYPES_EXIST__)
          std::is_same<T, __nv_bfloat16>::value ||
          #endif
          #if defined(__CUDA_FP8_TYPES_EXIST__)
          std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value ||
          #endif
          false) {
    return toFloatForDisplay(value) == 0.0f;
  } else {
    return value == T(0);
  }
}

// strictLowPrecisionTolerance: true for built-in reduceCopySum (strict); false for custom op (testOpSum/OpMul).
// nSrc: when nSrc >= 2 use ULP-based tolerance for sum comparisons; else legacy. Pass 0 when not comparing sums.
// maxAbsSource: when provided, ULP scale uses max of source element magnitudes (single authority for sum tolerance).
template<typename T>
void expectValuesEqual(T expected, T actual, const char* errorMsg, bool strictLowPrecisionTolerance,
             int nSrc, OptDouble maxAbsSource = OptDouble()) {
  if NCCL_IF_CONSTEXPR (std::is_same<T, half>::value ||
          #if defined(__CUDA_BF16_TYPES_EXIST__)
          std::is_same<T, __nv_bfloat16>::value ||
          #endif
          #if defined(__CUDA_FP8_TYPES_EXIST__)
          std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value ||
          #endif
          false) {
    float exp_f = toFloatForDisplay(expected);
    float act_f = toFloatForDisplay(actual);
    EXPECT_TRUE(valuesEqual(expected, actual, strictLowPrecisionTolerance, nSrc, maxAbsSource)) << errorMsg
      << " (expected=" << exp_f << ", actual=" << act_f << ", diff=" << std::abs(exp_f - act_f) << ")";
  } else if NCCL_IF_CONSTEXPR (std::is_floating_point<T>::value) {
    float exp_f = toFloatForDisplay(expected);
    float act_f = toFloatForDisplay(actual);
    EXPECT_TRUE(valuesEqual(expected, actual, true, nSrc, maxAbsSource)) << errorMsg
      << " (expected=" << exp_f << ", actual=" << act_f << ", diff=" << std::abs(exp_f - act_f) << ")";
  } else {
    // Use toFloatForDisplay to avoid operator<< issues for half/bfloat16/fp8 in
    // C++14 dead branches on GCC 9 (ambiguous or missing operator<<).
    EXPECT_TRUE(valuesEqual(expected, actual, true, nSrc, maxAbsSource)) << errorMsg
      << " (expected=" << toFloatForDisplay(expected) << ", actual=" << toFloatForDisplay(actual) << ")";
  }
}

// Helper macro for type-dependent assertion (calls template function).
// Pass nSrc (use 0 when not comparing sums).
#define EXPECT_VALUES_EQUAL(expected, actual, errorMsg, strictTolerance, nSrc) \
  expectValuesEqual(expected, actual, errorMsg, strictTolerance, nSrc)

#endif // NCCL_REDUCE_COPY_TEST_COMPARE_H
