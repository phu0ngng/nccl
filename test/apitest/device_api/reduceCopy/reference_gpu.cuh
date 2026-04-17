/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_REFERENCE_GPU_CUH_
#define _REDUCE_COPY_TEST_REFERENCE_GPU_CUH_

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declarations for GPU reference kernels
template<typename T>
void gpuReferenceReduceSum(
  const T* const* d_sources,
  int nSrc,
  T* d_destination,
  size_t count,
  cudaStream_t stream
);

template<typename T>
void gpuReferenceCopy(
  const T* d_source,
  T* const* d_destinations,
  int nDst,
  size_t count,
  cudaStream_t stream
);

template<typename T>
void gpuReferenceReduceSumCopy(
  const T* const* d_sources,
  int nSrc,
  T* const* d_destinations,
  int nDst,
  size_t count,
  cudaStream_t stream
);

// Round-after-each-step semantics (matches LsaReduceLsaCopy_Generic with testOpSum: accumulate in T, round each add).
template<typename T>
void gpuReferenceReduceSumCopyRoundEachStep(
  const T* const* d_sources,
  int nSrc,
  T* const* d_destinations,
  int nDst,
  size_t count,
  cudaStream_t stream
);

template<typename T>
void gpuReferenceReduceScatter(
  const T* const* d_sources,
  int nSrc,
  T* const* d_destinations,
  int nDst,
  size_t countPerRank,
  cudaStream_t stream
);

#endif // _REDUCE_COPY_TEST_REFERENCE_GPU_CUH_

