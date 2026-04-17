/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_ACC_TYPE_TRAIT_CUH_
#define _REDUCE_COPY_TEST_ACC_TYPE_TRAIT_CUH_

#include <type_traits>
#include <cuda_fp16.h>
#if defined(__CUDA_BF16_TYPES_EXIST__)
#include <cuda_bf16.h>
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif
// Accumulator type for ReduceSum (matches NCCL library AccumulateType<OpSum<T>>):
//   default T -> T;  half, bf16 -> float;  fp8 -> half.
template<typename T>
struct TestAccTypeMap {
  using AccType = T;
};

template<>
struct TestAccTypeMap<half> {
  using AccType = float;
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
struct TestAccTypeMap<__nv_bfloat16> {
  using AccType = float;
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
template<>
struct TestAccTypeMap<__nv_fp8_e4m3> {
  using AccType = half;
};

template<>
struct TestAccTypeMap<__nv_fp8_e5m2> {
  using AccType = half;
};
#endif

// Convert T to AccType for comparison / accumulation. Use normal ops on AccType.
// General template: identity-cast (covers int, uint, long long, float, double, …)
template<typename T>
__host__ __device__ inline typename TestAccTypeMap<T>::AccType toAccType(T value) {
  return static_cast<typename TestAccTypeMap<T>::AccType>(value);
}
template<>
__host__ __device__ inline float toAccType<half>(half value) {
  return __half2float(value);
}
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
__host__ __device__ inline float toAccType<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template<>
__host__ __device__ inline half toAccType<__nv_fp8_e4m3>(__nv_fp8_e4m3 value) {
  __nv_fp8_storage_t f8 = *reinterpret_cast<const __nv_fp8_storage_t*>(&value);
  __half_raw h_raw = __nv_cvt_fp8_to_halfraw(f8, __NV_E4M3);
  return half(h_raw);
}
template<>
__host__ __device__ inline half toAccType<__nv_fp8_e5m2>(__nv_fp8_e5m2 value) {
  __nv_fp8_storage_t f8 = *reinterpret_cast<const __nv_fp8_storage_t*>(&value);
  __half_raw h_raw = __nv_cvt_fp8_to_halfraw(f8, __NV_E5M2);
  return half(h_raw);
}
#endif

// Convert AccType back to T (for reference accumulation: sum in AccType, single cast at end).
// General template: identity-cast
template<typename T>
__host__ __device__ inline T fromAccType(typename TestAccTypeMap<T>::AccType value) {
  return static_cast<T>(value);
}
template<>
__host__ __device__ inline half fromAccType<half>(float value) {
  return __float2half_rn(value);
}
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
__host__ __device__ inline __nv_bfloat16 fromAccType<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template<>
__host__ __device__ inline __nv_fp8_e4m3 fromAccType<__nv_fp8_e4m3>(half value) {
  __nv_fp8_storage_t f8 = __nv_cvt_halfraw_to_fp8(
    *reinterpret_cast<const __half_raw*>(&value), __NV_SATFINITE, __NV_E4M3);
  return *reinterpret_cast<const __nv_fp8_e4m3*>(&f8);
}
template<>
__host__ __device__ inline __nv_fp8_e5m2 fromAccType<__nv_fp8_e5m2>(half value) {
  __nv_fp8_storage_t f8 = __nv_cvt_halfraw_to_fp8(
    *reinterpret_cast<const __half_raw*>(&value), __NV_SATFINITE, __NV_E5M2);
  return *reinterpret_cast<const __nv_fp8_e5m2*>(&f8);
}
#endif

// Promote AccType to float for comparison (normal ops in float).
inline __host__ __device__ float accTypeToFloat(float v) { return v; }
inline __host__ __device__ float accTypeToFloat(double v) { return static_cast<float>(v); }
inline __host__ __device__ float accTypeToFloat(half v) { return __half2float(v); }
inline __host__ __device__ float accTypeToFloat(int v) { return static_cast<float>(v); }
inline __host__ __device__ float accTypeToFloat(unsigned int v) { return static_cast<float>(v); }
inline __host__ __device__ float accTypeToFloat(int8_t v) { return static_cast<float>(v); }
inline __host__ __device__ float accTypeToFloat(uint8_t v) { return static_cast<float>(v); }
inline __host__ __device__ float accTypeToFloat(long long v) { return static_cast<float>(v); }
inline __host__ __device__ float accTypeToFloat(unsigned long long v) { return static_cast<float>(v); }

#endif // _REDUCE_COPY_TEST_ACC_TYPE_TRAIT_CUH_
