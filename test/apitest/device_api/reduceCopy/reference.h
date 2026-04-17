/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_REFERENCE_H_
#define _REDUCE_COPY_TEST_REFERENCE_H_

#include <vector>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif
#include "reference_gpu.cuh"

namespace detail {
template <typename T>
static void freeDevicePointers(std::vector<T*>& ptrs) {
  for (auto* ptr : ptrs) {
    if (ptr != NULL) cudaFree(ptr);
  }
  ptrs.clear();
}

template <typename T>
static T** makeDevicePointerArray(const std::vector<T*>& ptrs) {
  if (ptrs.empty()) return nullptr;
  T** d_ptrs = nullptr;
  cudaMalloc(&d_ptrs, ptrs.size() * sizeof(T*));
  cudaMemcpy(d_ptrs, ptrs.data(), ptrs.size() * sizeof(T*), cudaMemcpyHostToDevice);
  return d_ptrs;
}

template <typename T>
static std::vector<T*> copySourcesToDevice(
  const std::vector<std::vector<T>>& sources,
  size_t count
) {
  std::vector<T*> d_sources(sources.size(), nullptr);
  for (size_t i = 0; i < sources.size(); ++i) {
    cudaMalloc(&d_sources[i], count * sizeof(T));
    cudaMemcpy(d_sources[i], sources[i].data(),
           count * sizeof(T), cudaMemcpyHostToDevice);
  }
  return d_sources;
}

template <typename T>
static void runGpuReferenceReduceSum(
  const std::vector<std::vector<T>>& sources,
  std::vector<T>& destination
) {
  if (sources.empty() || sources[0].empty()) {
    destination.clear();
    return;
  }

  size_t count = sources[0].size();
  destination.resize(count);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<T*> d_sources = copySourcesToDevice<T>(sources, count);
  T** d_sources_array = makeDevicePointerArray(d_sources);

  T* d_destination = nullptr;
  cudaMalloc(&d_destination, count * sizeof(T));

  ::gpuReferenceReduceSum<T>(d_sources_array, sources.size(), d_destination, count, stream);

  cudaMemcpy(destination.data(), d_destination,
         count * sizeof(T), cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(stream);

  cudaFree(d_destination);
  cudaFree(d_sources_array);
  freeDevicePointers(d_sources);
  cudaStreamDestroy(stream);
}

template <typename T>
static void runGpuReferenceCopy(
  const std::vector<T>& source,
  std::vector<std::vector<T>>& destinations,
  int nDst
) {
  if (source.empty()) {
    destinations.clear();
    return;
  }

  size_t count = source.size();
  int numDst = nDst < 0 ? (destinations.empty() ? 1 : destinations.size()) : nDst;
  destinations.resize(numDst);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  T* d_source = nullptr;
  cudaMalloc(&d_source, count * sizeof(T));
  cudaMemcpy(d_source, source.data(),
         count * sizeof(T), cudaMemcpyHostToDevice);

  std::vector<T*> d_destinations(numDst, nullptr);
  for (int i = 0; i < numDst; ++i) {
    cudaMalloc(&d_destinations[i], count * sizeof(T));
  }
  T** d_destinations_array = makeDevicePointerArray(d_destinations);

  ::gpuReferenceCopy<T>(d_source, d_destinations_array, numDst, count, stream);

  for (int i = 0; i < numDst; ++i) {
    destinations[i].resize(count);
    cudaMemcpy(destinations[i].data(), d_destinations[i],
           count * sizeof(T), cudaMemcpyDeviceToHost);
  }
  cudaStreamSynchronize(stream);

  cudaFree(d_destinations_array);
  freeDevicePointers(d_destinations);
  cudaFree(d_source);
  cudaStreamDestroy(stream);
}

template <typename T>
static void runGpuReferenceReduceSumCopy(
  const std::vector<std::vector<T>>& sources,
  std::vector<std::vector<T>>& destinations,
  int nDst
) {
  if (sources.empty() || sources[0].empty()) {
    destinations.clear();
    return;
  }

  size_t count = sources[0].size();
  int numDst = nDst < 0 ? (destinations.empty() ? sources.size() : destinations.size()) : nDst;
  destinations.resize(numDst);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<T*> d_sources = copySourcesToDevice<T>(sources, count);
  T** d_sources_array = makeDevicePointerArray(d_sources);

  std::vector<T*> d_destinations(numDst, nullptr);
  for (int i = 0; i < numDst; ++i) {
    cudaMalloc(&d_destinations[i], count * sizeof(T));
  }
  T** d_destinations_array = makeDevicePointerArray(d_destinations);

  ::gpuReferenceReduceSumCopy<T>(d_sources_array, sources.size(),
                   d_destinations_array, numDst, count, stream);

  for (int i = 0; i < numDst; ++i) {
    destinations[i].resize(count);
    cudaMemcpy(destinations[i].data(), d_destinations[i],
           count * sizeof(T), cudaMemcpyDeviceToHost);
  }
  cudaStreamSynchronize(stream);

  cudaFree(d_destinations_array);
  freeDevicePointers(d_destinations);
  cudaFree(d_sources_array);
  freeDevicePointers(d_sources);
  cudaStreamDestroy(stream);
}

// Round-after-each-step semantics (matches LsaReduceLsaCopy_Generic testOpSum).
template <typename T>
static void runGpuReferenceReduceSumCopyRoundEachStep(
  const std::vector<std::vector<T>>& sources,
  std::vector<std::vector<T>>& destinations,
  int nDst
) {
  if (sources.empty() || sources[0].empty()) {
    destinations.clear();
    return;
  }

  size_t count = sources[0].size();
  int numDst = nDst < 0 ? (destinations.empty() ? sources.size() : destinations.size()) : nDst;
  destinations.resize(numDst);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<T*> d_sources = copySourcesToDevice<T>(sources, count);
  T** d_sources_array = makeDevicePointerArray(d_sources);

  std::vector<T*> d_destinations(numDst, nullptr);
  for (int i = 0; i < numDst; ++i) {
    cudaMalloc(&d_destinations[i], count * sizeof(T));
  }
  T** d_destinations_array = makeDevicePointerArray(d_destinations);

  ::gpuReferenceReduceSumCopyRoundEachStep<T>(d_sources_array, sources.size(),
                   d_destinations_array, numDst, count, stream);

  for (int i = 0; i < numDst; ++i) {
    destinations[i].resize(count);
    cudaMemcpy(destinations[i].data(), d_destinations[i],
           count * sizeof(T), cudaMemcpyDeviceToHost);
  }
  cudaStreamSynchronize(stream);

  cudaFree(d_destinations_array);
  freeDevicePointers(d_destinations);
  cudaFree(d_sources_array);
  freeDevicePointers(d_sources);
  cudaStreamDestroy(stream);
}

template <typename T>
static void runGpuReferenceReduceScatter(
  const std::vector<std::vector<T>>& sources,
  std::vector<std::vector<T>>& destinations,
  int nDst,
  size_t countPerRank
) {
  if (sources.empty() || sources[0].empty() || nDst <= 0) {
    destinations.clear();
    return;
  }

  const size_t sourceCount = static_cast<size_t>(nDst) * countPerRank;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<T*> d_sources = copySourcesToDevice<T>(sources, sourceCount);
  T** d_sources_array = makeDevicePointerArray(d_sources);

  std::vector<T*> d_destinations(static_cast<size_t>(nDst), nullptr);
  for (int i = 0; i < nDst; ++i) {
    cudaMalloc(&d_destinations[i], countPerRank * sizeof(T));
  }
  T** d_destinations_array = makeDevicePointerArray(d_destinations);

  ::gpuReferenceReduceScatter<T>(d_sources_array, static_cast<int>(sources.size()),
                   d_destinations_array, nDst, countPerRank, stream);

  destinations.resize(static_cast<size_t>(nDst));
  for (int i = 0; i < nDst; ++i) {
    destinations[i].resize(countPerRank);
    cudaMemcpy(destinations[i].data(), d_destinations[i],
           countPerRank * sizeof(T), cudaMemcpyDeviceToHost);
  }
  cudaStreamSynchronize(stream);

  cudaFree(d_destinations_array);
  freeDevicePointers(d_destinations);
  cudaFree(d_sources_array);
  freeDevicePointers(d_sources);
  cudaStreamDestroy(stream);
}
} // namespace detail

// Overloaded helpers for mulValues — C++14 dead-branch workaround:
// using overload resolution instead of if/else avoids instantiating the wrong
// conversion (__half2float on bfloat16 or vice versa) in dead branches.
inline half mulValuesHelper(const half& a, const half& b) {
  return __float2half(__half2float(a) * __half2float(b));
}
#if defined(__CUDA_BF16_TYPES_EXIST__)
inline __nv_bfloat16 mulValuesHelper(const __nv_bfloat16& a, const __nv_bfloat16& b) {
  return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
}
#endif
template<typename T>
inline T mulValuesHelper(const T& a, const T& b) { return a * b; }

// CPU reference implementation for ReduceCopy operations
// CRITICAL: These functions are reused by ALL variants - do NOT duplicate
template<typename T>
class CpuReference {
public:
  // ReduceSum: N sources -> 1 destination
  // This single function is reused by ALL ReduceSum variants (Series 3.x)
  static void reduceSum(
    const std::vector<std::vector<T>>& sources,
    std::vector<T>& destination
  ) {
    if (sources.empty() || sources[0].empty()) {
      destination.clear();
      return;
    }

    size_t count = sources[0].size();
    destination.resize(count);

    // Initialize destination with first source
    std::copy(sources[0].begin(), sources[0].end(), destination.begin());

    // Sum remaining sources
    for (size_t i = 1; i < sources.size(); ++i) {
      for (size_t j = 0; j < count; ++j) {
        destination[j] += sources[i][j];
      }
    }
  }

  // Copy: 1 source -> N destinations
  // This single function is reused by ALL Copy variants (Series 4.x)
  // Note: If destinations is empty, it will be resized to 1 destination
  //       Otherwise, it will use the existing size
  static void copy(
    const std::vector<T>& source,
    std::vector<std::vector<T>>& destinations,
    int nDst = -1  // -1 means use destinations.size() if non-empty, else 1
  ) {
    if (source.empty()) {
      destinations.clear();
      return;
    }

    size_t count = source.size();

    // Determine number of destinations
    int numDst = nDst;
    if (numDst < 0) {
      numDst = destinations.empty() ? 1 : destinations.size();
    }

    destinations.resize(numDst);

    // Copy source to each destination
    for (auto& dst : destinations) {
      dst.resize(count);
      std::copy(source.begin(), source.end(), dst.begin());
    }
  }

  // ReduceSumCopy: N sources -> M destinations. Uses AccumulateType
  // (default T->T, half/bf16->float, fp8->half) then single cast.
  static void reduceSumCopy(
    const std::vector<std::vector<T>>& sources,
    std::vector<std::vector<T>>& destinations,
    int nDst = -1  // -1 means use destinations.size() if non-empty, else use nSrc
  ) {
    if (sources.empty() || sources[0].empty()) {
      destinations.clear();
      return;
    }

    size_t count = sources[0].size();

    int numDst = nDst;
    if (numDst < 0) {
      numDst = destinations.empty() ? sources.size() : destinations.size();
    }

    // For half/bf16 use GPU reference so AccumulateType (float) accumulation matches library
    if NCCL_IF_CONSTEXPR (std::is_same<T, half>::value
      #if defined(__CUDA_BF16_TYPES_EXIST__)
      || std::is_same<T, __nv_bfloat16>::value
      #endif
    ) {
      detail::runGpuReferenceReduceSumCopy<T>(sources, destinations, numDst);
      return;
    }

    std::vector<T> reduced;
    reduceSum(sources, reduced);
    copy(reduced, destinations, numDst);
  }

  // ReduceSumCopy with round-after-each-step (matches LsaReduceLsaCopy_Generic testOpSum).
  // For half/bf16 uses GPU kernel; for other types same as reduceSumCopy.
  static void reduceSumCopyRoundEachStep(
    const std::vector<std::vector<T>>& sources,
    std::vector<std::vector<T>>& destinations,
    int nDst = -1
  ) {
    if (sources.empty() || sources[0].empty()) {
      destinations.clear();
      return;
    }

    size_t count = sources[0].size();
    int numDst = nDst < 0 ? (destinations.empty() ? sources.size() : destinations.size()) : nDst;
    destinations.resize(numDst);

    if NCCL_IF_CONSTEXPR (std::is_same<T, half>::value
      #if defined(__CUDA_BF16_TYPES_EXIST__)
      || std::is_same<T, __nv_bfloat16>::value
      #endif
    ) {
      detail::runGpuReferenceReduceSumCopyRoundEachStep<T>(sources, destinations, numDst);
      return;
    }

    std::vector<T> reduced;
    reduceSum(sources, reduced);
    copy(reduced, destinations, numDst);
  }

  // ReduceMul: N sources -> 1 destination (element-wise multiplication)
  static void reduceMul(
    const std::vector<std::vector<T>>& sources,
    std::vector<T>& destination
  ) {
    if (sources.empty() || sources[0].empty()) {
      destination.clear();
      return;
    }

    size_t count = sources[0].size();
    destination.resize(count);

    for (size_t j = 0; j < count; ++j) {
      T acc = sources[0][j];
      for (size_t i = 1; i < sources.size(); ++i) {
        acc = mulValues(acc, sources[i][j]);
      }
      destination[j] = acc;
    }
  }

  // ReduceMulCopy: N sources -> M destinations (element-wise multiplication)
  static void reduceMulCopy(
    const std::vector<std::vector<T>>& sources,
    std::vector<std::vector<T>>& destinations,
    int nDst = -1  // -1 means use destinations.size() if non-empty, else use nSrc
  ) {
    if (sources.empty() || sources[0].empty()) {
      destinations.clear();
      return;
    }

    int numDst = nDst;
    if (numDst < 0) {
      numDst = destinations.empty() ? sources.size() : destinations.size();
    }

    std::vector<T> reduced;
    reduceMul(sources, reduced);
    copy(reduced, destinations, numDst);
  }

  // ReduceScatter: N sources -> N destinations (each rank receives its chunk)
  // Each source provides nDst contiguous chunks of size countPerRank.
  static void reduceScatter(
    const std::vector<std::vector<T>>& sources,
    std::vector<std::vector<T>>& destinations,
    int nDst,
    size_t countPerRank
  ) {
    if (sources.empty() || sources[0].empty() || nDst <= 0) {
      destinations.clear();
      return;
    }

    // For half/bf16 use GPU reference so AccumulateType (float) accumulation matches library
    if NCCL_IF_CONSTEXPR (std::is_same<T, half>::value
      #if defined(__CUDA_BF16_TYPES_EXIST__)
      || std::is_same<T, __nv_bfloat16>::value
      #endif
    ) {
      detail::runGpuReferenceReduceScatter<T>(sources, destinations, nDst, countPerRank);
      return;
    }

    destinations.resize(nDst);
    for (int dst = 0; dst < nDst; ++dst) {
      destinations[dst].resize(countPerRank);
      for (size_t i = 0; i < countPerRank; ++i) {
        T acc = T(0);
        for (size_t src = 0; src < sources.size(); ++src) {
          size_t idx = static_cast<size_t>(dst) * countPerRank + i;
          if (idx < sources[src].size()) {
            acc = (src == 0) ? sources[src][idx] : (acc + sources[src][idx]);
          }
        }
        destinations[dst][i] = acc;
      }
    }
  }

  // AllGather: N sources -> N destinations (each destination receives all sources concatenated)
  // For AllGather, each rank contributes its chunk and all ranks receive all chunks
  static void allGather(
    const std::vector<std::vector<T>>& sources,
    std::vector<std::vector<T>>& destinations,
    int numDst
  ) {
    if (sources.empty() || sources[0].empty()) {
      destinations.clear();
      return;
    }

    size_t countPerRank = sources[0].size();
    size_t totalCount = countPerRank * sources.size();

    destinations.resize(numDst);
    for (int dst = 0; dst < numDst; ++dst) {
      destinations[dst].resize(totalCount);
      // Concatenate all sources: [source0, source1, ..., sourceN-1]
      for (size_t src = 0; src < sources.size(); ++src) {
        size_t dstOffset = src * countPerRank;
        if (src < sources.size() && sources[src].size() == countPerRank) {
          std::copy(sources[src].begin(), sources[src].end(),
               destinations[dst].begin() + dstOffset);
        }
      }
    }
  }

  // Helper for local strided variants: gather strided data into contiguous vectors
  // This allows reusing the same core computation functions
  static void gatherStridedSources(
    const T* basePtr,
    size_t displ,
    int nSrc,
    size_t count,
    std::vector<std::vector<T>>& sources
  ) {
    sources.resize(nSrc);
    for (int i = 0; i < nSrc; ++i) {
      sources[i].resize(count);
      const T* srcPtr = basePtr + i * displ;
      memcpy(sources[i].data(), srcPtr, count * sizeof(T));
    }
  }

  // Helper for local strided variants: scatter results to strided destinations
  static void scatterStridedDestinations(
    const std::vector<std::vector<T>>& destinations,
    T* basePtr,
    size_t displ,
    int nDst,
    size_t count
  ) {
    for (int i = 0; i < nDst && i < static_cast<int>(destinations.size()); ++i) {
      T* dstPtr = basePtr + i * displ;
      if (i < static_cast<int>(destinations.size()) && destinations[i].size() == count) {
        memcpy(dstPtr, destinations[i].data(), count * sizeof(T));
      }
    }
  }

  // Helper for element-wise multiplication across sources
  static T mulValues(const T& a, const T& b) {
    return mulValuesHelper(a, b);
  }
};

// Specializations for low-precision types that don't support += on CPU
// These use GPU reference instead
#if defined(__CUDA_FP8_TYPES_EXIST__)
template<>
class CpuReference<__nv_fp8_e4m3> {
public:
  static void reduceSum(
    const std::vector<std::vector<__nv_fp8_e4m3>>& sources,
    std::vector<__nv_fp8_e4m3>& destination
  ) {
    detail::runGpuReferenceReduceSum<__nv_fp8_e4m3>(sources, destination);
  }

  static void copy(
    const std::vector<__nv_fp8_e4m3>& source,
    std::vector<std::vector<__nv_fp8_e4m3>>& destinations,
    int nDst = -1
  ) {
    detail::runGpuReferenceCopy<__nv_fp8_e4m3>(source, destinations, nDst);
  }

  static void reduceSumCopy(
    const std::vector<std::vector<__nv_fp8_e4m3>>& sources,
    std::vector<std::vector<__nv_fp8_e4m3>>& destinations,
    int nDst = -1
  ) {
    detail::runGpuReferenceReduceSumCopy<__nv_fp8_e4m3>(sources, destinations, nDst);
  }

  // No round-each-step GPU reference for E4M3; use same as reduceSumCopy.
  static void reduceSumCopyRoundEachStep(
    const std::vector<std::vector<__nv_fp8_e4m3>>& sources,
    std::vector<std::vector<__nv_fp8_e4m3>>& destinations,
    int nDst = -1
  ) {
    reduceSumCopy(sources, destinations, nDst);
  }

  static void reduceMul(
    const std::vector<std::vector<__nv_fp8_e4m3>>& sources,
    std::vector<__nv_fp8_e4m3>& destination
  ) {
    if (sources.empty() || sources[0].empty()) {
      destination.clear();
      return;
    }

    size_t count = sources[0].size();
    destination.resize(count);
    for (size_t j = 0; j < count; ++j) {
      float acc = toFloat(sources[0][j]);
      for (size_t i = 1; i < sources.size(); ++i) {
        acc *= toFloat(sources[i][j]);
      }
      destination[j] = fromFloat(acc);
    }
  }

  static void reduceMulCopy(
    const std::vector<std::vector<__nv_fp8_e4m3>>& sources,
    std::vector<std::vector<__nv_fp8_e4m3>>& destinations,
    int nDst = -1
  ) {
    if (sources.empty() || sources[0].empty()) {
      destinations.clear();
      return;
    }

    int numDst = nDst;
    if (numDst < 0) {
      numDst = destinations.empty() ? sources.size() : destinations.size();
    }

    std::vector<__nv_fp8_e4m3> reduced;
    reduceMul(sources, reduced);
    copy(reduced, destinations, numDst);
  }

  static void reduceScatter(
    const std::vector<std::vector<__nv_fp8_e4m3>>& sources,
    std::vector<std::vector<__nv_fp8_e4m3>>& destinations,
    int nDst,
    size_t countPerRank
  ) {
    if (sources.empty() || sources[0].empty() || nDst <= 0) {
      destinations.clear();
      return;
    }

    destinations.resize(nDst);
    for (int dst = 0; dst < nDst; ++dst) {
      std::vector<std::vector<__nv_fp8_e4m3>> chunkSources(sources.size());
      for (size_t src = 0; src < sources.size(); ++src) {
        size_t start = static_cast<size_t>(dst) * countPerRank;
        size_t end = start + countPerRank;
        if (end > sources[src].size()) {
          end = sources[src].size();
        }
        chunkSources[src].assign(sources[src].begin() + start, sources[src].begin() + end);
      }
      std::vector<__nv_fp8_e4m3> reduced;
      reduceSum(chunkSources, reduced);
      destinations[dst] = reduced;
    }
  }

  static void allGather(
    const std::vector<std::vector<__nv_fp8_e4m3>>& sources,
    std::vector<std::vector<__nv_fp8_e4m3>>& destinations,
    int nDst = -1
  ) {
    if (sources.empty() || sources[0].empty()) {
      destinations.clear();
      return;
    }

    size_t countPerRank = sources[0].size();
    size_t totalCount = countPerRank * sources.size();

    int numDst = nDst;
    if (numDst < 0) {
      numDst = destinations.empty() ? sources.size() : destinations.size();
    }

    destinations.resize(numDst);
    for (int dst = 0; dst < numDst; ++dst) {
      destinations[dst].resize(totalCount);
      // Concatenate all sources: [source0, source1, ..., sourceN-1]
      for (size_t src = 0; src < sources.size(); ++src) {
        size_t dstOffset = src * countPerRank;
        if (src < sources.size() && sources[src].size() == countPerRank) {
          memcpy(destinations[dst].data() + dstOffset,
               sources[src].data(),
               countPerRank * sizeof(__nv_fp8_e4m3));
        }
      }
    }
  }

private:
  static float toFloat(__nv_fp8_e4m3 value) {
    __nv_fp8_storage_t storage = *reinterpret_cast<const __nv_fp8_storage_t*>(&value);
    __half_raw h_raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
    return __half2float(half(h_raw));
  }

  static __nv_fp8_e4m3 fromFloat(float value) {
    return __nv_fp8_e4m3(value);
  }
};

template<>
class CpuReference<__nv_fp8_e5m2> {
public:
  static void reduceSum(
    const std::vector<std::vector<__nv_fp8_e5m2>>& sources,
    std::vector<__nv_fp8_e5m2>& destination
  ) {
    detail::runGpuReferenceReduceSum<__nv_fp8_e5m2>(sources, destination);
  }

  static void copy(
    const std::vector<__nv_fp8_e5m2>& source,
    std::vector<std::vector<__nv_fp8_e5m2>>& destinations,
    int nDst = -1
  ) {
    detail::runGpuReferenceCopy<__nv_fp8_e5m2>(source, destinations, nDst);
  }

  static void reduceSumCopy(
    const std::vector<std::vector<__nv_fp8_e5m2>>& sources,
    std::vector<std::vector<__nv_fp8_e5m2>>& destinations,
    int nDst = -1
  ) {
    detail::runGpuReferenceReduceSumCopy<__nv_fp8_e5m2>(sources, destinations, nDst);
  }

  // No round-each-step GPU reference for E5M2; use same as reduceSumCopy.
  static void reduceSumCopyRoundEachStep(
    const std::vector<std::vector<__nv_fp8_e5m2>>& sources,
    std::vector<std::vector<__nv_fp8_e5m2>>& destinations,
    int nDst = -1
  ) {
    reduceSumCopy(sources, destinations, nDst);
  }

  static void reduceMul(
    const std::vector<std::vector<__nv_fp8_e5m2>>& sources,
    std::vector<__nv_fp8_e5m2>& destination
  ) {
    if (sources.empty() || sources[0].empty()) {
      destination.clear();
      return;
    }

    size_t count = sources[0].size();
    destination.resize(count);
    for (size_t j = 0; j < count; ++j) {
      float acc = toFloat(sources[0][j]);
      for (size_t i = 1; i < sources.size(); ++i) {
        acc *= toFloat(sources[i][j]);
      }
      destination[j] = fromFloat(acc);
    }
  }

  static void reduceMulCopy(
    const std::vector<std::vector<__nv_fp8_e5m2>>& sources,
    std::vector<std::vector<__nv_fp8_e5m2>>& destinations,
    int nDst = -1
  ) {
    if (sources.empty() || sources[0].empty()) {
      destinations.clear();
      return;
    }

    int numDst = nDst;
    if (numDst < 0) {
      numDst = destinations.empty() ? sources.size() : destinations.size();
    }

    std::vector<__nv_fp8_e5m2> reduced;
    reduceMul(sources, reduced);
    copy(reduced, destinations, numDst);
  }

  static void reduceScatter(
    const std::vector<std::vector<__nv_fp8_e5m2>>& sources,
    std::vector<std::vector<__nv_fp8_e5m2>>& destinations,
    int nDst,
    size_t countPerRank
  ) {
    if (sources.empty() || sources[0].empty() || nDst <= 0) {
      destinations.clear();
      return;
    }

    destinations.resize(nDst);
    for (int dst = 0; dst < nDst; ++dst) {
      std::vector<std::vector<__nv_fp8_e5m2>> chunkSources(sources.size());
      for (size_t src = 0; src < sources.size(); ++src) {
        size_t start = static_cast<size_t>(dst) * countPerRank;
        size_t end = start + countPerRank;
        if (end > sources[src].size()) {
          end = sources[src].size();
        }
        chunkSources[src].assign(sources[src].begin() + start, sources[src].begin() + end);
      }
      std::vector<__nv_fp8_e5m2> reduced;
      reduceSum(chunkSources, reduced);
      destinations[dst] = reduced;
    }
  }

  static void allGather(
    const std::vector<std::vector<__nv_fp8_e5m2>>& sources,
    std::vector<std::vector<__nv_fp8_e5m2>>& destinations,
    int nDst = -1
  ) {
    if (sources.empty() || sources[0].empty()) {
      destinations.clear();
      return;
    }

    size_t countPerRank = sources[0].size();
    size_t totalCount = countPerRank * sources.size();

    int numDst = nDst;
    if (numDst < 0) {
      numDst = destinations.empty() ? sources.size() : destinations.size();
    }

    destinations.resize(numDst);
    for (int dst = 0; dst < numDst; ++dst) {
      destinations[dst].resize(totalCount);
      // Concatenate all sources: [source0, source1, ..., sourceN-1]
      for (size_t src = 0; src < sources.size(); ++src) {
        size_t dstOffset = src * countPerRank;
        if (src < sources.size() && sources[src].size() == countPerRank) {
          memcpy(destinations[dst].data() + dstOffset,
               sources[src].data(),
               countPerRank * sizeof(__nv_fp8_e5m2));
        }
      }
    }
  }

private:
  static float toFloat(__nv_fp8_e5m2 value) {
    __nv_fp8_storage_t storage = *reinterpret_cast<const __nv_fp8_storage_t*>(&value);
    __half_raw h_raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E5M2);
    return __half2float(half(h_raw));
  }

  static __nv_fp8_e5m2 fromFloat(float value) {
    return __nv_fp8_e5m2(value);
  }
};
#endif // __CUDA_FP8_TYPES_EXIST__

#endif // _REDUCE_COPY_TEST_REFERENCE_H_

