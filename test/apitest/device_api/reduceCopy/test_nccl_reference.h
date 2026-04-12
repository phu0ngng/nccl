/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_REDUCE_COPY_TEST_NCCL_REFERENCE_H
#define NCCL_REDUCE_COPY_TEST_NCCL_REFERENCE_H

#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif
#include "checks.h"
#include "config.h"
#include "nccl.h"
#include "test_functions.h"

// For custom-op path (Generic/Generic_Mul), expected must match device (left-to-right, round each step).
// Skip NCCL reference for half/bfloat16 so GPU reference is used instead.

template<typename T>
class InputFactoryBase;

// Map C++ types to NCCL data types for official reference
template<typename T>
struct NcclTypeMap {
  static constexpr bool supported = false;
  static ncclDataType_t type() { return ncclFloat; }
};

template<>
struct NcclTypeMap<float> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclFloat; }
};

template<>
struct NcclTypeMap<double> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclDouble; }
};

template<>
struct NcclTypeMap<int> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclInt; }
};

template<>
struct NcclTypeMap<unsigned int> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclUint32; }
};

template<>
struct NcclTypeMap<int8_t> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclInt8; }
};

template<>
struct NcclTypeMap<uint8_t> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclUint8; }
};

template<>
struct NcclTypeMap<long long> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclInt64; }
};

template<>
struct NcclTypeMap<unsigned long long> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclUint64; }
};

template<>
struct NcclTypeMap<half> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclHalf; }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
struct NcclTypeMap<__nv_bfloat16> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclBfloat16; }
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
template<>
struct NcclTypeMap<__nv_fp8_e4m3> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclFloat8e4m3; }
};

template<>
struct NcclTypeMap<__nv_fp8_e5m2> {
  static constexpr bool supported = true;
  static ncclDataType_t type() { return ncclFloat8e5m2; }
};
#endif

template<typename T>
bool computeExpectedDataWithNccl(
  ApiFunctionId funcId,
  InputFactoryBase<T>* factory,
  ncclComm_t* comms,
  cudaStream_t* streams,
  int nRanks,
  size_t count,
  int nDst,
  std::vector<std::vector<T>>& expectedData,
  const std::vector<size_t>* sendOffsetsBytes = nullptr
) {
  // For custom-op path (testOpSum/OpMul), use GPU reference so expected matches device (round each step).
  if (usesCustomReduceOp(funcId) && (std::is_same<T, half>::value
    #if defined(__CUDA_BF16_TYPES_EXIST__)
    || std::is_same<T, __nv_bfloat16>::value
    #endif
    )) {
    return false;
  }
  // For half/bf16 ReduceSumCopy, use GPU reference so expected matches device API (float accumulate, single round).
  // NCCL AllReduce can use different reduction order or bf16 intermediates and produce different rounding.
  if (isReduceSumCopyVariant(funcId) && (std::is_same<T, half>::value
    #if defined(__CUDA_BF16_TYPES_EXIST__)
    || std::is_same<T, __nv_bfloat16>::value
    #endif
    )) {
    return false;
  }
  // For half/bf16 ReduceSum (ReduceScatter), use GPU reference for the same reason.
  if (isReduceSumVariant(funcId) && (std::is_same<T, half>::value
    #if defined(__CUDA_BF16_TYPES_EXIST__)
    || std::is_same<T, __nv_bfloat16>::value
    #endif
    )) {
    return false;
  }
  // For low-precision FP8 types, prefer the GPU reference path which
  // uses explicit saturation and matches device behavior more closely.
  #if defined(__CUDA_FP8_TYPES_EXIST__)
  if NCCL_IF_CONSTEXPR (std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value) {
    return false;
  }
  #endif
  if (!NcclTypeMap<T>::supported) {
    return false;
  }
  if (!factory || nRanks <= 0) {
    return false;
  }

  const bool isAllGather = isAllGatherVariant(funcId);
  const bool isReduceSum = isReduceSumVariant(funcId);
  const bool isReduceSumCopy = isReduceSumCopyVariant(funcId);
    const bool useMulVariant = isMulVariant(funcId);
  if (!isAllGather && !isReduceSum && !isReduceSumCopy && !useMulVariant) {
    return false;
  }
  // When AllGather uses per-rank send offsets (lambda offsets), skip NCCL reference and use
  // CPU allGather so expected matches the device kernel (which reads from sendBase+offset per rank).
  if (isAllGather && sendOffsetsBytes != nullptr && !sendOffsetsBytes->empty()) {
    return false;
  }

  ncclDataType_t dtype = NcclTypeMap<T>::type();
  size_t recvCount = isAllGather ? (count * nRanks) : count;

  std::vector<T*> refRecv(nRanks, nullptr);
  for (int i = 0; i < nRanks; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&refRecv[i], recvCount * sizeof(T)));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nRanks; ++i) {
    CUDACHECK(cudaSetDevice(i));
    void* sendBase = factory->getSendBuffer(i);
    if (!sendBase) {
      NCCLCHECK(ncclGroupEnd());
      for (int j = 0; j < nRanks; ++j) {
        CUDACHECK(cudaSetDevice(j));
        if (refRecv[j]) {
          CUDACHECK(cudaFree(refRecv[j]));
        }
      }
      return false;
    }
    if (sendOffsetsBytes && i < static_cast<int>(sendOffsetsBytes->size())) {
      sendBase = static_cast<char*>(sendBase) + (*sendOffsetsBytes)[i];
    }
    if (isAllGather) {
      NCCLCHECK(ncclAllGather(sendBase, refRecv[i], count, dtype, comms[i], streams[i]));
    } else if (isReduceSumCopy || useMulVariant) {
      ncclRedOp_t op = useMulVariant ? ncclProd : ncclSum;
      NCCLCHECK(ncclAllReduce(sendBase, refRecv[i], count, dtype, op, comms[i], streams[i]));
    } else if (isReduceSum) {
      NCCLCHECK(ncclReduceScatter(sendBase, refRecv[i], count, dtype, ncclSum, comms[i], streams[i]));
    } else {
      NCCLCHECK(ncclReduce(sendBase, refRecv[i], count, dtype, ncclSum, 0, comms[i], streams[i]));
    }
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nRanks);

  for (int i = 0; i < nRanks; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
  }

  expectedData.clear();
  expectedData.resize(nDst);
  if (isAllGather) {
    for (int dst = 0; dst < nDst && dst < nRanks; ++dst) {
      expectedData[dst].resize(recvCount);
      CUDACHECK(cudaMemcpy(expectedData[dst].data(), refRecv[dst],
                 recvCount * sizeof(T), cudaMemcpyDeviceToHost));
    }
  } else if (isReduceSum) {
    for (int dst = 0; dst < nDst && dst < nRanks; ++dst) {
      expectedData[dst].resize(recvCount);
      CUDACHECK(cudaMemcpy(expectedData[dst].data(), refRecv[dst],
                 recvCount * sizeof(T), cudaMemcpyDeviceToHost));
    }
  } else if (isReduceSumCopy || useMulVariant) {
    int refRank = 0;
    for (int dst = 0; dst < nDst; ++dst) {
      expectedData[dst].resize(recvCount);
      CUDACHECK(cudaMemcpy(expectedData[dst].data(), refRecv[refRank],
                 recvCount * sizeof(T), cudaMemcpyDeviceToHost));
    }
  } else {
    int refRank = 0;
    for (int dst = 0; dst < nDst; ++dst) {
      expectedData[dst].resize(recvCount);
      CUDACHECK(cudaMemcpy(expectedData[dst].data(), refRecv[refRank],
                 recvCount * sizeof(T), cudaMemcpyDeviceToHost));
    }
  }

  for (int i = 0; i < nRanks; ++i) {
    CUDACHECK(cudaSetDevice(i));
    if (refRecv[i]) {
      CUDACHECK(cudaFree(refRecv[i]));
    }
  }

  return true;
}

#endif // NCCL_REDUCE_COPY_TEST_NCCL_REFERENCE_H
