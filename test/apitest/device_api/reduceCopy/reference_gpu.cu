/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <type_traits>
#include "reference_gpu.cuh"
#include "acc_type_trait.cuh"
#if defined(__CUDA_BF16_TYPES_EXIST__)
#include <cuda_bf16.h>
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif
// Use TestAccTypeMap: accumulate in AccType, single cast to T at end.
template<typename T>
__global__ void gpuReferenceReduceSumKernel(
  const T* const* sources,
  int nSrc,
  T* destination,
  size_t count
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  using AccType = typename TestAccTypeMap<T>::AccType;
  AccType sum = AccType(0);
  for (int i = 0; i < nSrc; ++i) {
    sum += toAccType(sources[i][idx]);
  }
  destination[idx] = fromAccType<T>(sum);
}

// GPU kernel to compute reference Copy
template<typename T>
__global__ void gpuReferenceCopyKernel(
  const T* source,
  T* const* destinations,  // Array of pointers to destination arrays
  int nDst,
  size_t count
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  // Simple copy to all destinations
  for (int i = 0; i < nDst; ++i) {
    destinations[i][idx] = source[idx];
  }
}

// GPU kernel to compute reference ReduceSumCopy (same AccumulateType semantics as ReduceSum kernel)
template<typename T>
__global__ void gpuReferenceReduceSumCopyKernel(
  const T* const* sources,
  int nSrc,
  T* const* destinations,
  int nDst,
  size_t count
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  using AccType = typename TestAccTypeMap<T>::AccType;
  AccType sum = AccType(0);
  for (int i = 0; i < nSrc; ++i) {
    sum += toAccType(sources[i][idx]);
  }
  T reduced = fromAccType<T>(sum);

  // Then copy to all destinations
  for (int i = 0; i < nDst; ++i) {
    destinations[i][idx] = reduced;
  }
}

// GPU kernel: ReduceSumCopy with round-after-each-step (matches LsaReduceLsaCopy_Generic testOpSum semantics).
template<typename T>
__global__ void gpuReferenceReduceSumCopyRoundEachStepKernel(
  const T* const* sources,
  int nSrc,
  T* const* destinations,
  int nDst,
  size_t count
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  T acc = sources[0][idx];
  for (int i = 1; i < nSrc; ++i) {
    acc = fromAccType<T>(toAccType(acc) + toAccType(sources[i][idx]));
  }

  for (int i = 0; i < nDst; ++i) {
    destinations[i][idx] = acc;
  }
}

// GPU kernel to compute reference ReduceScatter (float accumulate, single round for half/bf16)
template<typename T>
__global__ void gpuReferenceReduceScatterKernel(
  const T* const* sources,
  int nSrc,
  T* const* destinations,
  int nDst,
  size_t countPerRank
) {
  const size_t totalElts = static_cast<size_t>(nDst) * countPerRank;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalElts) return;

  const size_t dst = idx / countPerRank;
  const size_t localIdx = idx % countPerRank;
  const size_t srcIdx = dst * countPerRank + localIdx;

  using AccType = typename TestAccTypeMap<T>::AccType;
  AccType sum = AccType(0);
  for (int src = 0; src < nSrc; ++src) {
    sum += toAccType(sources[src][srcIdx]);
  }
  destinations[dst][localIdx] = fromAccType<T>(sum);
}

// Host wrapper for ReduceScatter
template<typename T>
void gpuReferenceReduceScatter(
  const T* const* d_sources,
  int nSrc,
  T* const* d_destinations,
  int nDst,
  size_t countPerRank,
  cudaStream_t stream
) {
  const size_t totalElts = static_cast<size_t>(nDst) * countPerRank;
  const int blockSize = 256;
  const int gridSize = (totalElts + blockSize - 1) / blockSize;
  gpuReferenceReduceScatterKernel<T><<<gridSize, blockSize, 0, stream>>>(
    d_sources, nSrc, d_destinations, nDst, countPerRank
  );
}

// Host wrapper functions
template<typename T>
void gpuReferenceReduceSum(
  const T* const* d_sources,
  int nSrc,
  T* d_destination,
  size_t count,
  cudaStream_t stream
) {
  const int blockSize = 256;
  const int gridSize = (count + blockSize - 1) / blockSize;
  gpuReferenceReduceSumKernel<T><<<gridSize, blockSize, 0, stream>>>(
    d_sources, nSrc, d_destination, count
  );
}

template<typename T>
void gpuReferenceCopy(
  const T* d_source,
  T* const* d_destinations,
  int nDst,
  size_t count,
  cudaStream_t stream
) {
  const int blockSize = 256;
  const int gridSize = (count + blockSize - 1) / blockSize;
  gpuReferenceCopyKernel<T><<<gridSize, blockSize, 0, stream>>>(
    d_source, d_destinations, nDst, count
  );
}

template<typename T>
void gpuReferenceReduceSumCopy(
  const T* const* d_sources,
  int nSrc,
  T* const* d_destinations,
  int nDst,
  size_t count,
  cudaStream_t stream
) {
  const int blockSize = 256;
  const int gridSize = (count + blockSize - 1) / blockSize;
  gpuReferenceReduceSumCopyKernel<T><<<gridSize, blockSize, 0, stream>>>(
    d_sources, nSrc, d_destinations, nDst, count
  );
}

template<typename T>
void gpuReferenceReduceSumCopyRoundEachStep(
  const T* const* d_sources,
  int nSrc,
  T* const* d_destinations,
  int nDst,
  size_t count,
  cudaStream_t stream
) {
  const int blockSize = 256;
  const int gridSize = (count + blockSize - 1) / blockSize;
  gpuReferenceReduceSumCopyRoundEachStepKernel<T><<<gridSize, blockSize, 0, stream>>>(
    d_sources, nSrc, d_destinations, nDst, count
  );
}

// Explicit instantiations for common types
template void gpuReferenceReduceSum<float>(const float* const*, int, float*, size_t, cudaStream_t);
template void gpuReferenceCopy<float>(const float*, float* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceSumCopy<float>(const float* const*, int, float* const*, int, size_t, cudaStream_t);

template void gpuReferenceReduceSum<double>(const double* const*, int, double*, size_t, cudaStream_t);
template void gpuReferenceCopy<double>(const double*, double* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceSumCopy<double>(const double* const*, int, double* const*, int, size_t, cudaStream_t);

template void gpuReferenceReduceSum<int>(const int* const*, int, int*, size_t, cudaStream_t);
template void gpuReferenceCopy<int>(const int*, int* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceSumCopy<int>(const int* const*, int, int* const*, int, size_t, cudaStream_t);

template void gpuReferenceReduceSum<long long>(const long long* const*, int, long long*, size_t, cudaStream_t);
template void gpuReferenceCopy<long long>(const long long*, long long* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceSumCopy<long long>(
  const long long* const*, int, long long* const*, int, size_t, cudaStream_t);

template void gpuReferenceReduceSum<unsigned long long>(
  const unsigned long long* const*, int, unsigned long long*, size_t, cudaStream_t);
template void gpuReferenceCopy<unsigned long long>(
  const unsigned long long*, unsigned long long* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceSumCopy<unsigned long long>(
  const unsigned long long* const*, int, unsigned long long* const*, int, size_t, cudaStream_t);

template void gpuReferenceReduceSum<half>(const half* const*, int, half*, size_t, cudaStream_t);
template void gpuReferenceCopy<half>(const half*, half* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceSumCopy<half>(const half* const*, int, half* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceSumCopyRoundEachStep<half>(
  const half* const*, int, half* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceScatter<half>(const half* const*, int, half* const*, int, size_t, cudaStream_t);

#if defined(__CUDA_BF16_TYPES_EXIST__)
template void gpuReferenceReduceSum<__nv_bfloat16>(
  const __nv_bfloat16* const*, int, __nv_bfloat16*, size_t, cudaStream_t);
template void gpuReferenceCopy<__nv_bfloat16>(
  const __nv_bfloat16*, __nv_bfloat16* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceSumCopy<__nv_bfloat16>(
  const __nv_bfloat16* const*, int, __nv_bfloat16* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceSumCopyRoundEachStep<__nv_bfloat16>(
  const __nv_bfloat16* const*, int, __nv_bfloat16* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceScatter<__nv_bfloat16>(
  const __nv_bfloat16* const*, int, __nv_bfloat16* const*, int, size_t, cudaStream_t);
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
template void gpuReferenceReduceSum<__nv_fp8_e4m3>(
  const __nv_fp8_e4m3* const*, int, __nv_fp8_e4m3*, size_t, cudaStream_t);
template void gpuReferenceCopy<__nv_fp8_e4m3>(
  const __nv_fp8_e4m3*, __nv_fp8_e4m3* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceSumCopy<__nv_fp8_e4m3>(
  const __nv_fp8_e4m3* const*, int, __nv_fp8_e4m3* const*, int, size_t, cudaStream_t);

template void gpuReferenceReduceSum<__nv_fp8_e5m2>(
  const __nv_fp8_e5m2* const*, int, __nv_fp8_e5m2*, size_t, cudaStream_t);
template void gpuReferenceCopy<__nv_fp8_e5m2>(
  const __nv_fp8_e5m2*, __nv_fp8_e5m2* const*, int, size_t, cudaStream_t);
template void gpuReferenceReduceSumCopy<__nv_fp8_e5m2>(
  const __nv_fp8_e5m2* const*, int, __nv_fp8_e5m2* const*, int, size_t, cudaStream_t);
#endif


