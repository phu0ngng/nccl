/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_REDUCE_COPY_TEST_DISPATCH_H
#define NCCL_REDUCE_COPY_TEST_DISPATCH_H

#include <cstdio>
#include "config.h"
#include "test_functions.h"
#include "kernels.cuh"

// First layer: Dispatch based on API function ID
template<typename T, int UNROLL, CooperationLevel coopLevel>
void apiFunctionDispatch(
  ApiFunctionId funcId,
  ncclWindow_t sendwin, size_t sendoffset,
  ncclWindow_t recvwin, size_t recvoffset,
  int nSrc, int nDst, int nDstStart,
  size_t count,  // Total count (needed for ReduceSumCopy variants)
  ncclDevComm devComm,
  int gridSize, int blockSize, cudaStream_t stream
) {
  if (isReduceSumVariant(funcId)) {
    launchReduceSumKernel<T, UNROLL, coopLevel>(
      funcId, sendwin, sendoffset, recvwin, recvoffset,
      nSrc, nDst, count, devComm, gridSize, blockSize, stream);
  } else if (isReduceSumCopyVariant(funcId)) {
    launchAllReduceKernel<T, UNROLL, coopLevel>(
      funcId, sendwin, sendoffset, recvwin, recvoffset,
      nSrc, nDst, nDstStart, count, devComm, gridSize, blockSize, stream);
  } else if (isAllGatherVariant(funcId)) {
    launchAllGatherKernel<T, UNROLL, coopLevel>(
      funcId, sendwin, sendoffset, recvwin, recvoffset,
      nSrc, nDst, count, devComm, gridSize, blockSize, stream);
  } else {
    printf("[TEST] WARNING: Unhandled API function in apiFunctionDispatch (funcId=%d)\n",
         static_cast<int>(funcId));
  }
}

// Second layer: Dispatch cooperation level
template<typename T, int UNROLL>
void coopDispatch(
  CooperationLevel coopLevel,
  ApiFunctionId funcId,
  ncclWindow_t sendwin, size_t sendoffset,
  ncclWindow_t recvwin, size_t recvoffset,
  int nSrc, int nDst, int nDstStart,
  size_t count,  // Total count (needed for ReduceSumCopy variants)
  ncclDevComm devComm,
  int gridSize, int blockSize, cudaStream_t stream
) {
  switch (coopLevel) {
  case CooperationLevel::Thread:
    apiFunctionDispatch<T, UNROLL, CooperationLevel::Thread>(
      funcId, sendwin, sendoffset, recvwin, recvoffset,
      nSrc, nDst, nDstStart,
      count, devComm, gridSize, blockSize, stream);
    break;
  case CooperationLevel::Warp:
    apiFunctionDispatch<T, UNROLL, CooperationLevel::Warp>(
      funcId, sendwin, sendoffset, recvwin, recvoffset,
      nSrc, nDst, nDstStart,
      count, devComm, gridSize, blockSize, stream);
    break;
  case CooperationLevel::Cta:
    apiFunctionDispatch<T, UNROLL, CooperationLevel::Cta>(
      funcId, sendwin, sendoffset, recvwin, recvoffset,
      nSrc, nDst, nDstStart,
      count, devComm, gridSize, blockSize, stream);
    break;
  default:
    printf("[TEST] WARNING: Unhandled cooperation level in coopDispatch (coopLevel=%d)\n", static_cast<int>(coopLevel));
    break;
  }
}

// Third layer: Dispatch UNROLL value
template<typename T>
void unrollDispatch(
  int unroll,
  CooperationLevel coopLevel,
  ApiFunctionId funcId,
  ncclWindow_t sendwin, size_t sendoffset,
  ncclWindow_t recvwin, size_t recvoffset,
  int nSrc, int nDst, int nDstStart,
  size_t count,  // Total count (needed for ReduceSumCopy variants)
  ncclDevComm devComm,
  int gridSize, int blockSize, cudaStream_t stream
) {
  constexpr int unroll1 = 1;
  constexpr int unrollDefault = getDefaultUnroll<T>();

  if (unroll == unroll1) {
    coopDispatch<T, unroll1>(coopLevel, funcId, sendwin, sendoffset, recvwin, recvoffset,
                  nSrc, nDst, nDstStart, count, devComm, gridSize, blockSize, stream);
  } else if (unroll == unrollDefault) {
    coopDispatch<T, unrollDefault>(coopLevel, funcId, sendwin, sendoffset, recvwin, recvoffset,
                     nSrc, nDst, nDstStart, count, devComm, gridSize, blockSize, stream);
  } else {
    printf("[TEST] WARNING: Unhandled UNROLL value in unrollDispatch (unroll=%d, expected 1 or %d)\n",
         unroll, unrollDefault);
  }
}

#endif // NCCL_REDUCE_COPY_TEST_DISPATCH_H
