/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_SUPPORT_H_
#define _REDUCE_COPY_TEST_SUPPORT_H_

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <type_traits>
#include "nccl.h"
#include "nccl_device.h"
#include "config.h"
#include "api_function_traits.h"

// Check compile-time macros to determine which fp8 multimem architectures are available
// These macros are set by the build system (Makefile/CMake) based on NVCC_GENCODE
// Priority matches multimem__funcs.h: sm_100f > sm_100a > sm_101f > sm_101a > sm_120a > sm_121a
#if defined(NCCL_BUILD_SM100F) && NCCL_BUILD_SM100F == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM100F 1
#elif defined(NCCL_BUILD_SM100A) && NCCL_BUILD_SM100A == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM100A 1
#elif defined(NCCL_BUILD_SM101F) && NCCL_BUILD_SM101F == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM101F 1
#elif defined(NCCL_BUILD_SM101A) && NCCL_BUILD_SM101A == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM101A 1
#elif defined(NCCL_BUILD_SM120A) && NCCL_BUILD_SM120A == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM120A 1
#elif defined(NCCL_BUILD_SM121A) && NCCL_BUILD_SM121A == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM121A 1
#else
  #define FP8_MULTIMEM_ARCH_AVAILABLE 0
#endif

// Central support matrix for type and multimem support
// This matrix determines which types support which features
class TypeSupportMatrix {
public:
  // Check if a type is supported at all (for any operation)
  template<typename T>
  static constexpr bool isTypeSupported() {
    // Standard types are always supported
    if NCCL_IF_CONSTEXPR (std::is_same<T, float>::value ||
            std::is_same<T, double>::value ||
            std::is_same<T, int>::value ||
            std::is_same<T, unsigned int>::value ||
            std::is_same<T, int8_t>::value ||
            std::is_same<T, uint8_t>::value ||
            std::is_same<T, long long>::value ||
            std::is_same<T, unsigned long long>::value) {
      return true;
    }

    // Check for half (always available in CUDA)
    if NCCL_IF_CONSTEXPR (std::is_same<T, half>::value) {
      return true;
    }

    // Check for bfloat16 (requires CUDA 11.0+)
    #ifdef __CUDA_BF16_TYPES_EXIST__
    if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_bfloat16>::value) {
      return true;
    }
    #endif

    // Check for FP8 (requires sm_90+)
    #ifdef __CUDA_FP8_TYPES_EXIST__
    if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_fp8_e4m3>::value ||
            std::is_same<T, __nv_fp8_e5m2>::value) {
      return true;  // Runtime check done separately
    }
    #endif

    return false;
  }

  // Check if multiplication operation is supported for a type
  // FP8 types are not supported because NCCL promotes them to half for vectorization,
  // but OpMul only accepts the exact type, causing type mismatches
  // This function must be __host__ __device__ to be callable from device code
  template<typename T>
  __host__ __device__ static constexpr bool isMulSupported() {
    // Standard types support multiplication
    if NCCL_IF_CONSTEXPR (std::is_same<T, float>::value ||
            std::is_same<T, double>::value ||
            std::is_same<T, int>::value ||
            std::is_same<T, unsigned int>::value ||
            std::is_same<T, int8_t>::value ||
            std::is_same<T, uint8_t>::value ||
            std::is_same<T, long long>::value ||
            std::is_same<T, unsigned long long>::value) {
      return true;
    }

    // Half supports multiplication
    if NCCL_IF_CONSTEXPR (std::is_same<T, half>::value) {
      return true;
    }

    // bfloat16 supports multiplication
    #ifdef __CUDA_BF16_TYPES_EXIST__
    if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_bfloat16>::value) {
      return true;
    }
    #endif

    // FP8 types do NOT support multiplication variant
    // NCCL promotes FP8 to half for vectorization, but OpMul only accepts FP8
    #ifdef __CUDA_FP8_TYPES_EXIST__
    if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_fp8_e4m3>::value ||
            std::is_same<T, __nv_fp8_e5m2>::value) {
      return false;
    }
    #endif

    return false;
  }

  // Check if a type supports multimem source operations
  // Based on multimem__funcs.h: multimemLoadSum/multimemStore specializations
  // Supported types: float, double, half, __nv_bfloat16, __nv_fp8_e4m3, __nv_fp8_e5m2,
  //                  int32_t (int), uint32_t (unsigned int), int64_t, uint64_t
  // NOT supported: int8_t, uint8_t, long long, unsigned long long
  // Note: int64_t/uint64_t are supported, but long long/unsigned long long may not match
  //       the template specializations exactly, so we exclude them from multimem tests
  template<typename T>
  __host__ __device__ static constexpr bool isMultimemSupported() {
    // Floating point types that support multimem
    if NCCL_IF_CONSTEXPR (std::is_same<T, float>::value ||
            std::is_same<T, double>::value ||
            std::is_same<T, half>::value) {
      return true;
    }

    // bfloat16 supports multimem
    #ifdef __CUDA_BF16_TYPES_EXIST__
    if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_bfloat16>::value) {
      return true;
    }
    #endif

    // FP8 types support multimem
    #ifdef __CUDA_FP8_TYPES_EXIST__
    if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_fp8_e4m3>::value ||
            std::is_same<T, __nv_fp8_e5m2>::value) {
      return true;
    }
    #endif

    // Integer types that support multimem
    // Note: int32_t (int) and uint32_t (unsigned int) are supported
    //       int64_t and uint64_t are supported, but long long/unsigned long long
    //       may not match the template specializations, so we exclude them
    if NCCL_IF_CONSTEXPR (std::is_same<T, int>::value ||
            std::is_same<T, unsigned int>::value) {
      return true;
    }

    // long long and unsigned long long do NOT support multimem
    // (template specializations are for int64_t/uint64_t, not long long/unsigned long long)
    return false;
  }

  // Check if a type supports multimem destination operations
  template<typename T>
  __host__ __device__ static constexpr bool isMultimemDestinationSupported() {
    return true;
  }
};

// Helper class to check if features are supported
class TestSupportChecker {
public:
  // Check if a type is supported on the current architecture
  template<typename T>
  static bool isTypeSupported() {
    if (!TypeSupportMatrix::isTypeSupported<T>()) {
      return false;
    }

    // Runtime checks for types that require specific compute capability
    #ifdef __CUDA_FP8_TYPES_EXIST__
    if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_fp8_e4m3>::value ||
            std::is_same<T, __nv_fp8_e5m2>::value) {
      int major, minor;
      if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess) {
        return false;
      }
      if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess) {
        return false;
      }
      // FP8 requires sm_90+ (compute capability 9.0+)
      return major >= 9;
    }
    #endif

    return true;
  }

  // Check if a type supports multimem source operations
  template<typename T>
  static bool isMultimemTypeSupported() {
    // First check compile-time support
    if (!TypeSupportMatrix::isMultimemSupported<T>()) {
      return false;
    }

    // Runtime checks for types that require specific compute capability
    #ifdef __CUDA_FP8_TYPES_EXIST__
    if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_fp8_e4m3>::value ||
            std::is_same<T, __nv_fp8_e5m2>::value) {
      if (!FP8_MULTIMEM_ARCH_AVAILABLE) {
        return false;
      }
      int major, minor;
      if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess) {
        return false;
      }
      if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess) {
        return false;
      }
      // FP8 requires sm_90+ (compute capability 9.0+)
      return major >= 9;
    }
    #endif

    #ifdef __CUDA_BF16_TYPES_EXIST__
    if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_bfloat16>::value) {
      // bfloat16 multimem requires sm_90+
      int major, minor;
      if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess) {
        return false;
      }
      if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess) {
        return false;
      }
      return major >= 9;
    }
    #endif

    return true;
  }

  // Check if a type supports multimem destination operations
  template<typename T>
  static bool isMultimemDestinationTypeSupported() {
    return TypeSupportMatrix::isMultimemDestinationSupported<T>();
  }

  // Check if multimem is supported (NCCL ncclCommQueryProperties.multimemSupport)
  static bool isMultimemSupported(ncclComm_t comm) {
    if (comm == nullptr) {
      return false;
    }
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    if (ncclCommQueryProperties(comm, &props) != ncclSuccess) {
      return false;
    }
    return props.multimemSupport != 0;
  }

  // Check if device API is supported (NCCL ncclCommQueryProperties)
  static bool isDeviceApiSupported(ncclComm_t comm) {
    if (comm == nullptr) {
      return false;
    }
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    if (ncclCommQueryProperties(comm, &props) != ncclSuccess) {
      return false;
    }
    return props.deviceApiSupport != 0;
  }

  // Check if all ranks have P2P connectivity (same LSA team). Required for device API
  // LSA/ReduceCopy tests; see test/perf/all_reduce.cu AllReduceGetDevCommRequirements.
  static bool isP2pConnected(ncclComm_t comm) {
    if (comm == nullptr) {
      return false;
    }
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    if (ncclCommQueryProperties(comm, &props) != ncclSuccess) {
      return false;
    }
    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    return props.nRanks == lsaTeam.nRanks;
  }

  // Check if a function ID requires multimem support (has multimem source or destination)
  static bool isMultimemVariant(ApiFunctionId funcId) {
    return hasMultimemSource(funcId) || hasMultimemDestination(funcId);
  }

  // Check if a function ID has multimem sources (shared across all ranks)
  static bool hasMultimemSource(ApiFunctionId funcId) {
    return apiTraitsHasMultimemSource(funcId);
  }

  // Check if a function ID has multimem destinations (shared across all ranks)
  static bool hasMultimemDestination(ApiFunctionId funcId) {
    return apiTraitsHasMultimemDestination(funcId);
  }

  // Check if a function ID is a local variant
  static bool isLocalVariant(ApiFunctionId funcId) {
    return apiTraitsIsLocal(funcId);
  }
};

#endif // _REDUCE_COPY_TEST_SUPPORT_H_

