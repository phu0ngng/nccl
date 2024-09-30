#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif


template<typename T>
struct FloatLayout { static constexpr bool is_floating_point = false; };
template<>
struct FloatLayout<float> {
  static constexpr bool is_floating_point = true;
  static constexpr int exponent_bits = 8, mantissa_bits = 23;
  static constexpr int exponent_bias = (1<<(exponent_bits-1))-1;
};
template<>
struct FloatLayout<double> {
  static constexpr bool is_floating_point = true;
  static constexpr int exponent_bits = 11, mantissa_bits = 52;
  static constexpr int exponent_bias = (1<<(exponent_bits-1))-1;
};
template<>
struct FloatLayout<half> {
  static constexpr bool is_floating_point = true;
  static constexpr int exponent_bits = 5, mantissa_bits = 10;
  static constexpr int exponent_bias = (1<<(exponent_bits-1))-1;
};
#ifdef __CUDA_BF16_TYPES_EXIST__
template<>
struct FloatLayout<__nv_bfloat16> {
  static constexpr bool is_floating_point = true;
  static constexpr int exponent_bits = 8, mantissa_bits = 7;
  static constexpr int exponent_bias = (1<<(exponent_bits-1))-1;
};
#endif
#ifdef __CUDA_FP8_TYPES_EXIST__
template<>
struct FloatLayout<__nv_fp8_e4m3> {
  static constexpr bool is_floating_point = true;
  static constexpr int exponent_bits = 4, mantissa_bits = 3;
  static constexpr int exponent_bias = (1<<(exponent_bits-1))-1;
};
template<>
struct FloatLayout<__nv_fp8_e5m2> {
  static constexpr bool is_floating_point = true;
  static constexpr int exponent_bits = 5, mantissa_bits = 2;
  static constexpr int exponent_bias = (1<<(exponent_bits-1))-1;
};
#endif

template<typename T>
__device__ uint32_t delta(T a, T b) {
  union { T at; uint32_t au; }; at = a;
  union { T bt; uint32_t bu; }; bt = b;
  return au < bu ? bu-au : au-bu;
}

template<typename T>
__global__ void kernel_ring() {
  uint64_t rng = threadIdx.x;
  T acc;
  double accd;
  for (int j=0; j < 8; j++) {
    for (int i=0; i < 8; i++) {
      union {
        uint64_t bits;
        T val;
      };
      bits = 0;
      for (int k=0; k < 2; k++) {
        bits += rng & (1ul<<FloatLayout<T>::mantissa_bits)-1;
        rng += 0xb81b99d5cb268a4a;
        rng *= 0x4a8aba95dbaf42ef;
        rng ^= rng>>32;
      }
      uint32_t expo = FloatLayout<T>::exponent_bias - 2;
      bits += expo<<FloatLayout<T>::mantissa_bits;
      //if(threadIdx.x==0)printf("val=%f\n", float(val));
      acc = T(float(acc) + float(val));
      accd += double(val);
    }
    uint32_t del = delta(acc, T(accd));
    uint32_t maxdel = __reduce_max_sync(~0u, del);
    uint32_t mindel = __reduce_min_sync(~0u, del);
    if (threadIdx.x==0) printf("  nranks=%d ulps max=%d min=%d lossy=%f truth=%f\n", 8*j+8, maxdel, mindel, float(acc), float(T(accd)));
  }
}

template<typename T>
__global__ void kernel_tree() {
  uint64_t rng = threadIdx.x;

  T accs[64];
  double accds[64];
  for (int i=0; i < 64; i++) {
    union { uint64_t bits; T val; };
    bits = 0;
    for (int k=0; k < 2; k++) {
      bits += rng & (1ul<<FloatLayout<T>::mantissa_bits)-1;
      rng += 0xb81b99d5cb268a4a;
      rng *= 0x4a8aba95dbaf42ef;
      rng ^= rng>>32;
    }
    uint32_t expo = FloatLayout<T>::exponent_bias - 2;
    bits += expo<<FloatLayout<T>::mantissa_bits;
    accs[i] = val;
    accds[i] = double(val);
  }

  for (int i=0; i < 6; i++) {
    int d = 1<<(5-i);
    for (int j=0; j < d; j++) {
      accs[j] = T(float(accs[j]) + float(accs[j+d]));
      accds[j] = accds[j] + accds[j+d];
    }
    uint32_t del = delta(accs[0], T(accds[0]));
    uint32_t maxdel = __reduce_max_sync(~0u, del);
    uint32_t mindel = __reduce_min_sync(~0u, del);
    if (threadIdx.x==0) printf("  nranks=%d ulps max=%d min=%d lossy=%f truth=%f\n", 2<<i, maxdel, mindel, float(accs[0]), accds[0]);
  }
}

int main() {
  cudaSetDevice(0);

  printf("ring fp16\n");
  kernel_ring<__half><<<1,32>>>();
  cudaDeviceSynchronize();
  printf("tree fp16\n");
  kernel_tree<__half><<<1,32>>>();
  cudaDeviceSynchronize();

  printf("ring __nv_fp8_e4m3\n");
  kernel_ring<__nv_fp8_e4m3><<<1,32>>>();
  cudaDeviceSynchronize();
  printf("tree __nv_fp8_e4m3\n");
  kernel_tree<__nv_fp8_e4m3><<<1,32>>>();
  cudaDeviceSynchronize();

  printf("ring __nv_fp8_e5m2\n");
  kernel_ring<__nv_fp8_e5m2><<<1,32>>>();
  cudaDeviceSynchronize();
  printf("tree __nv_fp8_e5m2\n");
  kernel_tree<__nv_fp8_e5m2><<<1,32>>>();
  cudaDeviceSynchronize();
  return 0;
}
