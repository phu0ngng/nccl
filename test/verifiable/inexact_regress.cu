/* Generate parameters for our error bound model of floating point average
 * (sum of scaled values) by sampling sums of random sequences for each
 * floating point type.
 *
 * The model has parameters "coef" and "power", where for two floats a & b,
 * they are close enough if and only if:
 *   abs(intBits(a) - intBits(b)) <= 1 + coef*pow(rank_n, power);
 *
 * Where intBits(x) is the reinterpretation of the float bitpattern as an integer.
 *
 * Compile with:
 *   nvcc -gencode=arch=compute_80,code=sm_80
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using std::uint64_t;
using std::uint32_t;
using bfloat16 = __nv_bfloat16;

template<typename T>
struct float_traits;

template<>
struct float_traits<float> {
  static constexpr int mantissa_bits = 23;
  static constexpr float loss_power = .5f;
  using uint_t = uint32_t;
  __device__ static float make(double x) { return (float)x; }
  __device__ static float make(uint64_t x) { return (float)x; }
  __device__ static double todouble(float x) { return x; }
  __device__ static float add(float a, float b) { return a+b; }
  __device__ static float mul(float a, float b) { return a*b; }
};
template<>
struct float_traits<double> {
  static constexpr int mantissa_bits = 52;
  static constexpr float loss_power = .5f;
  using uint_t = uint64_t;
  __device__ static double make(double x) { return x; }
  __device__ static double make(uint64_t x) { return (double)x; }
  __device__ static double todouble(double x) { return x; }
  __device__ static double add(double a, double b) { return a+b; }
  __device__ static double mul(double a, double b) { return a*b; }
};
template<>
struct float_traits<half> {
  static constexpr int mantissa_bits = 11;
  static constexpr float loss_power = .7f;
  using uint_t = uint16_t;
  __device__ static half make(double x) { return __double2half(x); }
  __device__ static half make(uint64_t x) { return __int2half_rn(x); }
  __device__ static double todouble(half x) { return __half2float(x); }
  __device__ static half add(half a, half b) { return __hadd(a, b); }
  __device__ static half mul(half a, half b) { return __hmul(a, b); }
};
template<>
struct float_traits<bfloat16> {
  static constexpr int mantissa_bits = 7;
  using uint_t = uint16_t;
  static constexpr float loss_power = .7f;
  __device__ static bfloat16 make(double x) { return __double2bfloat16(x); }
  __device__ static bfloat16 make(uint64_t x) { return __int2bfloat16_rn(x); }
  __device__ static double todouble(bfloat16 x) { return __bfloat162float(x); }
  __device__ static bfloat16 add(bfloat16 a, bfloat16 b) { return __hadd(a, b); }
  __device__ static bfloat16 mul(bfloat16 a, bfloat16 b) { return __hmul(a, b); }
};

template<typename F>
__device__ int compare(F a, F b) {
  union { typename float_traits<F>::uint_t ua; F fa; };
  union { typename float_traits<F>::uint_t ub; F fb; };
  ua=0; ub=0;
  fa=a; fb=b;
  //std::printf("bits(%1.10f)=%x bits(%1.10f)=%x\n", fa, ua, fb, ub);
  return ua < ub ? ub-ua : ua-ub;
}

struct xoshiro256ss {
	uint64_t s[4];
  __device__ xoshiro256ss(int seed) {
    constexpr uint64_t src[4] = {0xbb99e851d1f545cc, 0xbfc4022389ca40cb, 0xe84aff5cb1914af5, 0x845999858284de77};
    for(int i=0; i < 4; i++)
      s[i] = src[i] + (seed + i)*0xb45de8a52fdb65d3;
  }
  __device__ uint64_t operator()() {
    auto rol64 = [](uint64_t x, int k) {
      return (x << k) | (x >> (64 - k));
    };
    uint64_t const result = rol64(s[1] * 5, 7) * 9;
    uint64_t const t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rol64(s[3], 45);
    return result;
  }
};

template<typename F>
__global__ void kernel() {
  using traits = float_traits<F>;
  constexpr int samps = 4<<10;
  __shared__ F accf[samps];
  __shared__ double accd[samps];

  for(int i=threadIdx.x; i < samps; i += blockDim.x) {
    accf[i] = 0;
    accd[i] = 0;
  }
  __syncthreads();

  int rprev=1;
  F scalar = traits::make(1.0/11000.0);
  int maxerr = 0;
  float coef = 0;
  double avgpow = 0;
  int avgpow_n = 0;
  int preverr = 1;
  xoshiro256ss rng(threadIdx.x);

  for(int r=2; r <= 16<<10; r++) {
    int err = 0;
    for(int i=threadIdx.x; i < samps; i+=blockDim.x) {
      constexpr uint64_t m = (1ll<<traits::mantissa_bits)-1;
      F f = traits::make(0.5 + double(rng() & m)/(m+1));
      accf[i] = traits::add(accf[i], traits::mul(scalar, f));
      accd[i] += traits::todouble(f);
      int e = compare(accf[i], traits::mul(scalar, traits::make(accd[i])));
      err = err > e ? err : e;
    }
    err = __reduce_max_sync(-1u, err);
    if(threadIdx.x == 0) {
      // err = 1 + S*sqrt(r)
      float c = float(err-1)/powf(float(r), traits::loss_power);
      coef = coef > c ? coef : c;
      maxerr = maxerr > err ? maxerr : err;
      double pow = log2f(2+maxerr)/log2f(r);
      avgpow += pow;
      avgpow_n++;
      if(float(r) > rprev*1.25 && maxerr > preverr) {
        std::printf("err=%d coef=%1.9f, pow=%1.6f up to ranks=%d\n", maxerr, coef, avgpow/avgpow_n, r);
        preverr = maxerr;
        rprev = r;
      }
    }
  }
  if(threadIdx.x == 0)
    std::printf("FINAL coef=%1.10f pow=%1.10f\n", coef, avgpow/avgpow_n);
}

int main() {
  std::printf("type=float:\n");
  kernel<float><<<1,32>>>();
  cudaDeviceSynchronize();

  std::printf("\ntype=half:\n");
  kernel<half><<<1,32>>>();
  cudaDeviceSynchronize();

  std::printf("\ntype=bfloat16:\n");
  kernel<bfloat16><<<1,32>>>();
  cudaDeviceSynchronize();
  return 0;
}
