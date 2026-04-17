/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_REDUCE_COPY_TEST_REDUCE_COPY_H
#define NCCL_REDUCE_COPY_TEST_REDUCE_COPY_H

#include "common.cuh"
#include "factories.h"
#include "reference.h"
#include "data.h"
#include "test_matrix.h"
// Include kernel declarations and launcher functions
#include "kernels.cuh"
#include <vector>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <type_traits>
#include <algorithm>
#include <sstream>
#include <string>
#include "util.h"
#include <gtest/gtest.h>

#include "test_types.h"
#include "test_comms.h"
#include "test_compare.h"
#include "test_functions.h"
#include "api_function_traits.h"
#include "test_nccl_reference.h"
#include "test_dispatch.h"
#include "test_params.h"

template <typename T>
constexpr size_t storageCountFromElementCount(size_t count) {
  return count;
}

template <typename T>
constexpr size_t lambdaOffsetBytesFromElements(size_t elementOffset) {
  return elementOffset * sizeof(T);
}

// Base factory interface type erasure helper
template<typename T>
class InputFactoryBase {
public:
  virtual ~InputFactoryBase() = default;
  virtual void prepareData(int nSrc, int nDst, size_t count) = 0;
  virtual KernelParams getKernelParams(int deviceId) const = 0;
  virtual void cleanup() = 0;
  virtual int getNSrc() const = 0;
  virtual int getNDst() const = 0;
  virtual void* getSendBuffer(int deviceId) const = 0;
  virtual void* getRecvBuffer(int deviceId) const = 0;
  virtual bool isLambdaVariant() const = 0;
};

// Wrapper to erase template parameter from InputFactory
template<typename T, ApiFunctionId FuncId>
class InputFactoryWrapper : public InputFactoryBase<T> {
  std::unique_ptr<InputFactory<T, FuncId>> factory_;

public:
  InputFactoryWrapper(std::unique_ptr<InputFactory<T, FuncId>> factory)
    : factory_(std::move(factory)) {}

  void prepareData(int nSrc, int nDst, size_t count) override {
    factory_->prepareData(nSrc, nDst, count);
  }

  KernelParams getKernelParams(int deviceId) const override {
    return factory_->getKernelParams(deviceId);
  }

  void cleanup() override {
    factory_->cleanup();
  }

  int getNSrc() const override {
    return factory_->getNSrc();
  }

  int getNDst() const override {
    return factory_->getNDst();
  }

  void* getSendBuffer(int deviceId) const override {
    // Try to cast to WindowInputFactory to get buffer
    WindowInputFactory<T>* winFactory = dynamic_cast<WindowInputFactory<T>*>(factory_.get());
    if (winFactory) {
      return winFactory->getSendBuffer(deviceId);
    }
    ReduceSumInputFactory<T>* reduceFactory = dynamic_cast<ReduceSumInputFactory<T>*>(factory_.get());
    if (reduceFactory) {
      return reduceFactory->getSendBuffer(deviceId);
    }
    LambdaInputFactory<T>* lambdaFactory = dynamic_cast<LambdaInputFactory<T>*>(factory_.get());
    if (lambdaFactory) {
      return lambdaFactory->getSendBuffer(deviceId);
    }
    LocalStridedInputFactory<T>* localFactory = dynamic_cast<LocalStridedInputFactory<T>*>(factory_.get());
    if (localFactory) {
      return localFactory->getSendBuffer(deviceId);
    }
    return nullptr;
  }

  void* getRecvBuffer(int deviceId) const override {
    // Try to cast to WindowInputFactory to get buffer
    WindowInputFactory<T>* winFactory = dynamic_cast<WindowInputFactory<T>*>(factory_.get());
    if (winFactory) {
      return winFactory->getRecvBuffer(deviceId);
    }
    ReduceSumInputFactory<T>* reduceFactory = dynamic_cast<ReduceSumInputFactory<T>*>(factory_.get());
    if (reduceFactory) {
      return reduceFactory->getRecvBuffer(deviceId);
    }
    LambdaInputFactory<T>* lambdaFactory = dynamic_cast<LambdaInputFactory<T>*>(factory_.get());
    if (lambdaFactory) {
      return lambdaFactory->getRecvBuffer(deviceId);
    }
    LocalStridedInputFactory<T>* localFactory = dynamic_cast<LocalStridedInputFactory<T>*>(factory_.get());
    if (localFactory) {
      return localFactory->getRecvBuffer(deviceId);
    }
    return nullptr;
  }

  bool isLambdaVariant() const override {
    return dynamic_cast<LambdaInputFactory<T>*>(factory_.get()) != nullptr;
  }
};

// Factory function to create appropriate input factory (uses central traits only).
template<typename T>
std::unique_ptr<InputFactoryBase<T>>
createInputFactory(ncclComm_t* comms, int nDevices, ApiFunctionId funcId,
           bool enableLambdaOffsets = false) {
  if (apiTraitsIsLocal(funcId)) {
    auto factory = std::make_unique<LocalStridedInputFactory<T>>(comms, nDevices);
    return std::make_unique<InputFactoryWrapper<T, ApiFunctionId::LocalReduceSumCopy_Strided>>(std::move(factory));
  }
  if (apiTraitsCategory(funcId) == ApiFunctionCategory::ReduceSum) {
    auto factory = std::make_unique<ReduceSumInputFactory<T>>(comms, nDevices);
    return std::make_unique<InputFactoryWrapper<T, ApiFunctionId::LsaReduceSum_Window_DevComm>>(std::move(factory));
  }
  ApiFunctionCategory c = apiTraitsCategory(funcId);
  if (apiTraitsIsLambda(funcId) && (c == ApiFunctionCategory::GenericReduceCopy ||
      c == ApiFunctionCategory::ReduceSumCopy || c == ApiFunctionCategory::AllGather)) {
    auto factory = std::make_unique<LambdaInputFactory<T>>(comms, nDevices, enableLambdaOffsets);
    return std::make_unique<InputFactoryWrapper<T, ApiFunctionId::LsaReduceSum_Lambda>>(std::move(factory));
  }
  // Window-based (AllGather non-lambda, ReduceSumCopy N->M)
  auto factory = std::make_unique<WindowInputFactory<T>>(comms, nDevices);
  return std::make_unique<InputFactoryWrapper<T, ApiFunctionId::LsaReduceSumCopy_Windows>>(std::move(factory));
}

struct RunFullTestOptions {
  bool skipUnimplemented = false;
};

template<typename T>
void runFullTestImpl(const TestParams& params,
          ncclComm_t* comms, int nVis,
          cudaStream_t* streams,
          RunFullTestOptions options = {});

template<typename T>
void runFullTestImpl(const TestParams& params,
          ncclComm_t* comms, int nVis,
          cudaStream_t* streams,
          RunFullTestOptions options) {
  if (options.skipUnimplemented &&
    !isReduceSumVariant(params.funcId) &&
    !isReduceSumCopyVariant(params.funcId) &&
    !isAllGatherVariant(params.funcId)) {
    fprintf(stderr, "[SKIP] funcId=%d: skipUnimplemented\n", static_cast<int>(params.funcId));
    return;
  }
  if (!TestSupportChecker::isTypeSupported<T>()) {
    fprintf(stderr, "[SKIP] funcId=%d: typeNotSupported\n", static_cast<int>(params.funcId));
    return;
  }
  if (isMulVariant(params.funcId) &&
    !TypeSupportMatrix::isMulSupported<T>()) {
    fprintf(stderr, "[SKIP] funcId=%d: mulNotSupported\n", static_cast<int>(params.funcId));
    return;
  }
  if (!TestSupportChecker::isDeviceApiSupported(comms[0])) {
    fprintf(stderr, "[SKIP] funcId=%d: deviceApiNotSupported\n", static_cast<int>(params.funcId));
    return;
  }
  bool isLocalVariant = TestSupportChecker::isLocalVariant(params.funcId);
  if (!isLocalVariant && !TestSupportChecker::isP2pConnected(comms[0])) {
    fprintf(stderr, "[SKIP] funcId=%d: p2pNotConnected nRanks=%d lsaNRanks=%d\n",
        static_cast<int>(params.funcId),
        [&]{ ncclCommProperties_t p = NCCL_COMM_PROPERTIES_INITIALIZER;
             ncclCommQueryProperties(comms[0], &p); return p.nRanks; }(),
        static_cast<int>(ncclTeamLsa(comms[0]).nRanks));
    return;
  }
  bool requiresMultimem = TestSupportChecker::isMultimemVariant(params.funcId);
  bool hasMultimemSource = TestSupportChecker::hasMultimemSource(params.funcId);
  bool hasMultimemDestination = TestSupportChecker::hasMultimemDestination(params.funcId);
  if (!isLocalVariant && nVis < 2) {
    fprintf(stderr, "[SKIP] funcId=%d: nVis=%d < 2\n", static_cast<int>(params.funcId), nVis);
    return;
  }
  if (requiresMultimem) {
    if (!TestSupportChecker::isMultimemSupported(comms[0])) {
        fprintf(stderr, "[SKIP] funcId=%d: multimemNotSupported\n", static_cast<int>(params.funcId));
        return; // Skip: multimem not supported (ncclCommQueryProperties.multimemSupport)
      }
      if (hasMultimemSource && !TestSupportChecker::isMultimemTypeSupported<T>()) {
        fprintf(stderr, "[SKIP] funcId=%d: multimemSrcTypeNotSupported\n", static_cast<int>(params.funcId));
        return; // Skip test - type doesn't support multimem source operations
      }
      if (hasMultimemDestination && !TestSupportChecker::isMultimemDestinationTypeSupported<T>()) {
        fprintf(stderr, "[SKIP] funcId=%d: multimemDstTypeNotSupported\n", static_cast<int>(params.funcId));
        return; // Skip test - type doesn't support multimem destination operations
      }
    }

  const bool enableLambdaOffsets =
    isLambdaVariantFunction(params.funcId) && params.enableLambdaOffsets;
  int maxTestGpus = 1;
  if (isLocalVariant) {
    if (params.maxTestGpus > 1) {
      fprintf(stderr, "[SKIP] funcId=%d: localVariant maxTestGpus=%d > 1\n",
          static_cast<int>(params.funcId), params.maxTestGpus);
      return;
    }
  } else {
    maxTestGpus = (params.maxTestGpus > 0 ? params.maxTestGpus : nVis);
    if (maxTestGpus > nVis) {
      fprintf(stderr, "[SKIP] funcId=%d: maxTestGpus=%d > nVis=%d\n",
          static_cast<int>(params.funcId), maxTestGpus, nVis);
      return;
    }
  }
  ActiveComms activeComms = createActiveComms(maxTestGpus, comms, nVis);
  int numDevices = activeComms.n;
  std::vector<ncclDevComm_t> devComms(nVis, ncclDevComm_t{});

  // Determine if this is an NM split scenario: explicit nDstStart with a function that
  // supports independent src/dst teams (DifferentTeams) or lambdas (LsaReduceSumLsaCopy).
  const bool isNmSplit = (params.nDstStart > 0);

  if (isNmSplit) {
    if (numDevices < 4) {
      fprintf(stderr, "[SKIP] funcId=%d: NM split requires nVis>=4 (nVis=%d)\n",
          static_cast<int>(params.funcId), numDevices);
      destroyActiveComms(&activeComms);
      return;
    }
    if (numDevices < params.nDstStart + 1) {
      fprintf(stderr, "[SKIP] funcId=%d: nVis=%d < nDstStart+1=%d\n",
          static_cast<int>(params.funcId), numDevices, params.nDstStart + 1);
      destroyActiveComms(&activeComms);
      return;
    }
    if (numDevices < params.nSrc) {
      fprintf(stderr, "[SKIP] funcId=%d: nVis=%d < nSrc=%d\n",
          static_cast<int>(params.funcId), numDevices, params.nSrc);
      destroyActiveComms(&activeComms);
      return;
    }
  }

  TestParams adaptedParams = params;
  if (!isLocalVariant) {
    if (isNmSplit) {
      // NM split: nSrc from params (fixed), nDst = nVis - nDstStart
      adaptedParams.nSrc = params.nSrc;
      adaptedParams.nDst = numDevices - params.nDstStart;
    } else if (isReduceSumVariant(params.funcId)) {
        // ReduceSum: use ReduceScatter semantics (each rank receives its chunk)
        adaptedParams.nSrc = numDevices;
        adaptedParams.nDst = numDevices;
      } else if (isAllGatherVariant(params.funcId)) {
        // AllGather: N sources -> N destinations (each rank contributes and receives all)
        adaptedParams.nSrc = numDevices;
        adaptedParams.nDst = numDevices;
      } else { // ReduceSumCopy
        // Multimem sources/destinations are shared across all ranks
        adaptedParams.nSrc = TestSupportChecker::hasMultimemSource(params.funcId) ? 1 : numDevices;
        adaptedParams.nDst = TestSupportChecker::hasMultimemDestination(params.funcId) ? 1 : numDevices;
        // LSA sources -> multimem destination (e.g. LsaReduceMultimemCopy_Generic)
        if (apiTraitsCategory(params.funcId) == ApiFunctionCategory::GenericReduceCopy &&
          apiTraitsHasMultimemDestination(params.funcId)) {
          adaptedParams.nSrc = numDevices;
        }
      }
    }
    // For local variants, use params as-is (nSrc/nDst represent local chunks, not ranks)
    if (isLocalVariant && isReduceSumVariant(params.funcId)) {
      adaptedParams.nDst = 1;
    }

    // Create device communicators for this test
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = adaptedParams.gridSize; // Need one barrier per block
    if (requiresMultimem) {
      reqs.lsaMultimem = true;
    }

    // Create device communicators for each device (mirror perf/common.cu).
    ncclResult_t result = ncclSuccess;
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < numDevices; ++i) {
      CUDACHECK(cudaSetDevice(i));
      int commCount = 0;
      NCCLCHECK(ncclCommCount(activeComms.comms[i], &commCount));
      VERBOSE_PRINTF("[TEST] runFullTest: comm %d has %d ranks\n", i, commCount);
    result = ncclDevCommCreate(activeComms.comms[i], &reqs, &devComms[i]);
      if (result != ncclSuccess) {
        NCCLCHECK(ncclGroupEnd());
        NCCLCHECK(result);
      }
    }
    NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), activeComms.comms, numDevices);

    // Create input factory
    auto factory = createInputFactory<T>(
      activeComms.comms, numDevices, adaptedParams.funcId,
      enableLambdaOffsets
    );
    if (!factory) {
      // Factory not implemented for this variant yet
    ADD_FAILURE() << "Input factory creation failed for funcId="
            << static_cast<int>(adaptedParams.funcId);
    for (int i = 0; i < numDevices; ++i) {
      CUDACHECK(cudaSetDevice(i));
      NCCLCHECK(ncclDevCommDestroy(activeComms.comms[i], &devComms[i]));
    }
    destroyActiveComms(&activeComms);
    return;
  }

  const size_t elementCount = adaptedParams.count;
    const size_t storageCountPerRank = storageCountFromElementCount<T>(elementCount);
    size_t launchCountPerRank = elementCount;
    factory->prepareData(adaptedParams.nSrc, adaptedParams.nDst, storageCountPerRank);

    // NM split: overwrite ALL recv buffers with sentinel so src-only ranks' buffers
    // remain untouched after the kernel (verified below).
    static constexpr float kNMSentinel = -999.0f;
    if (isNmSplit && elementCount > 0) {
      const T sentinelVal = static_cast<T>(kNMSentinel);
      std::vector<T> sentinelBuf(storageCountPerRank, sentinelVal);
      for (int i = 0; i < numDevices; ++i) {
        CUDACHECK(cudaSetDevice(i));
        void* recvBase = factory->getRecvBuffer(i);
        if (recvBase) {
          CUDACHECK(cudaMemcpy(recvBase, sentinelBuf.data(),
            storageCountPerRank * sizeof(T), cudaMemcpyHostToDevice));
        }
      }
    }

    bool isLambdaVariant = factory->isLambdaVariant();
    const bool useLambdaOffsets = isLambdaVariant && enableLambdaOffsets;
    auto getLambdaOffsets = [&](int deviceId, int rank) -> std::pair<size_t, size_t> {
      if (!useLambdaOffsets) {
        return {0, 0};
      }
      return {
        lambdaOffsetBytesFromElements<T>(lambdaSendOffsetElts(rank)),
        lambdaOffsetBytesFromElements<T>(lambdaRecvOffsetElts(rank))
      };
    };

    // Prepare source data on CPU
    std::vector<std::vector<T>> sourceData(adaptedParams.nSrc);
    size_t sourceCount = storageCountPerRank;
    size_t sourceElementCount = elementCount;
    if (isReduceSumVariant(adaptedParams.funcId) && !isLocalVariant) {
      sourceCount = storageCountPerRank * static_cast<size_t>(adaptedParams.nDst);
      sourceElementCount = elementCount * static_cast<size_t>(adaptedParams.nDst);
    } else if (useLambdaOffsets && !isLocalVariant &&
               !isAllGatherVariant(adaptedParams.funcId) && !isReduceSumVariant(adaptedParams.funcId)) {
      // ReduceSumCopy with lambda offsets: kernel reads from [sendOffset(r)+j], need extra space
      sourceCount = storageCountPerRank + kMaxLambdaOffsetElts;
    }
    // Initialize source data on CPU. Mul variant: scale down by 1/(nSrc^2) so product stays in range.
    const bool useMulVariant = isMulVariant(adaptedParams.funcId);
    const bool useLargerVariance = usesCustomReduceOp(adaptedParams.funcId) && !useMulVariant;
    const int nSrc = adaptedParams.nSrc;
    for (int i = 0; i < adaptedParams.nSrc; ++i) {
      sourceData[i].resize(sourceCount);
      TestDataInitializer<T>::initializeTestData(sourceData[i], i, sourceCount, useLargerVariance, useMulVariant, nSrc);
    }

    // Copy source data to device
    // When useLambdaOffsets, use per-rank lambdaSendOffsetElts (via getLambdaOffsets) so kernel
    // and host agree on buffer placement; kernel uses resolveLambdaSendOffsetBytes(sendoffset, rank).
    for (int i = 0; i < numDevices; ++i) {
      CUDACHECK(cudaSetDevice(i));
      void* sendBase = factory->getSendBuffer(i);
      if (sendBase) {
        int rank = isLocalVariant ? 0 : i;
        std::pair<size_t,size_t> lambdaOff0 = getLambdaOffsets(i, rank);
        size_t sendOffsetBytes = lambdaOff0.first;
        (void)lambdaOff0.second;
        char* sendBaseBytes = static_cast<char*>(sendBase) + sendOffsetBytes;
        if (isLocalVariant) {
          // Local variant: copy strided sources for ReduceSum/ReduceSumCopy,
          // or single source for LocalCopy variants.
          if (adaptedParams.nSrc > 0) {
            T* sendPtr = reinterpret_cast<T*>(sendBaseBytes);
            if (isAllGatherVariant(adaptedParams.funcId)) {
              CUDACHECK(cudaMemcpy(sendPtr, sourceData[0].data(),
                storageCountPerRank * sizeof(T), cudaMemcpyHostToDevice));
            } else {
              for (int src = 0; src < adaptedParams.nSrc; ++src) {
                T* dstPtr = sendPtr + src * storageCountPerRank;
                CUDACHECK(cudaMemcpy(dstPtr, sourceData[src].data(),
                  storageCountPerRank * sizeof(T), cudaMemcpyHostToDevice));
              }
            }
          }
        } else {
          // Inter-rank: copy source data based on operation type
          bool isAllGather = isAllGatherVariant(adaptedParams.funcId);
          T* sendPtr = reinterpret_cast<T*>(sendBaseBytes);
          if (isAllGather) {
            if (i < adaptedParams.nSrc) {
              CUDACHECK(cudaMemcpy(sendPtr, sourceData[i].data(),
                storageCountPerRank * sizeof(T), cudaMemcpyHostToDevice));
            }
          } else if (isReduceSumVariant(adaptedParams.funcId)) {
            if (i < adaptedParams.nSrc) {
              CUDACHECK(cudaMemcpy(sendPtr, sourceData[i].data(),
                sourceData[i].size() * sizeof(T), cudaMemcpyHostToDevice));
            }
          } else {
            // ReduceSumCopy: each rank's data at per-rank offset; kernel reads from same base
            if (i < adaptedParams.nSrc) {
              CUDACHECK(cudaMemcpy(sendPtr, sourceData[i].data(),
                sourceData[i].size() * sizeof(T), cudaMemcpyHostToDevice));
            }
          }
        }
      }
    }

    // Compute reference (prefer official NCCL collectives)
    bool isAllGather = isAllGatherVariant(adaptedParams.funcId);
    const bool isMulVariantRef = isMulVariant(adaptedParams.funcId);
    std::vector<std::vector<T>> expectedData;
    bool usedNcclReference = false;
    std::vector<size_t> sendOffsetsBytes;
    if (useLambdaOffsets && !isLocalVariant) {
      sendOffsetsBytes.resize(numDevices, 0);
      for (int i = 0; i < numDevices; ++i) {
        sendOffsetsBytes[i] = lambdaOffsetBytesFromElements<T>(lambdaSendOffsetElts(i));
      }
    }
  if (!isLocalVariant) {
    usedNcclReference = computeExpectedDataWithNccl<T>(
      adaptedParams.funcId,
      factory.get(),
      activeComms.comms,
      streams,
      numDevices,
        adaptedParams.count,
        adaptedParams.nDst,
        expectedData,
        useLambdaOffsets ? &sendOffsetsBytes : nullptr
      );
    }

    if (!usedNcclReference) {
      if (isMulVariantRef) {
        expectedData.resize(adaptedParams.nDst);
        CpuReference<T>::reduceMulCopy(sourceData, expectedData, adaptedParams.nDst);
      } else if (isReduceSumVariant(adaptedParams.funcId)) {
        if (isLocalVariant) {
          std::vector<T> expectedSingle;
          CpuReference<T>::reduceSum(sourceData, expectedSingle);
          expectedData.resize(1);
          expectedData[0] = expectedSingle;
        } else {
          expectedData.resize(adaptedParams.nDst);
          size_t countPerRankForRef = static_cast<size_t>(adaptedParams.count);
          CpuReference<T>::reduceScatter(sourceData, expectedData,
                          adaptedParams.nDst, countPerRankForRef);
        }
      } else if (isAllGather) {
        // AllGather: each rank receives all ranks' data concatenated
        expectedData.resize(adaptedParams.nDst);
        if (isLocalVariant) {
          CpuReference<T>::copy(sourceData[0], expectedData, adaptedParams.nDst);
        } else {
          CpuReference<T>::allGather(sourceData, expectedData, adaptedParams.nDst);
        }
      } else { // ReduceSumCopy
        // Per-rank offset is for buffer placement only; kernel reads from start of placed buffer.
        // Use only first count elements per source so reference matches test count.
        expectedData.resize(adaptedParams.nDst);
        const size_t count = static_cast<size_t>(adaptedParams.count);
        std::vector<std::vector<T>> sourceForRef(adaptedParams.nSrc);
        for (int r = 0; r < adaptedParams.nSrc; ++r) {
          size_t n = std::min(count, sourceData[r].size());
          sourceForRef[r].assign(sourceData[r].begin(), sourceData[r].begin() + n);
        }
        if (usesCustomReduceOp(adaptedParams.funcId) &&
          (std::is_same<T, half>::value
           #if defined(__CUDA_BF16_TYPES_EXIST__)
           || std::is_same<T, __nv_bfloat16>::value
           #endif
           )) {
          CpuReference<T>::reduceSumCopyRoundEachStep(sourceForRef, expectedData, adaptedParams.nDst);
        } else {
          CpuReference<T>::reduceSumCopy(sourceForRef, expectedData, adaptedParams.nDst);
        }
      }
    }

    // Compute work distribution and launch kernels

    // Use ncclGroupStart/End when launching kernels on multiple devices
    if (numDevices > 1) {
      NCCLCHECK(ncclGroupStart());
    }

    // Progress output for debugging
    if (numDevices > 1) {
      VERBOSE_PRINTF("[TEST] Launching kernels on %d devices with ncclGroupStart/End\n", numDevices);

    }

    // For AllGather (N->N), all ranks launch kernels (each rank contributes its chunk)
    // For ReduceSum (N->1) and ReduceSumCopy (N->M), all ranks launch kernels
    // isAllGather already declared above
    int kernelLaunchCount = numDevices;

    for (int i = 0; i < kernelLaunchCount; ++i) {
      int deviceId = isLocalVariant ? 0 : i;

      CUDACHECK(cudaSetDevice(deviceId));

      // Get kernel parameters for this specific device with rank-based offsets
      KernelParams kernelParams = factory->getKernelParams(deviceId);

      // Use layered dispatch: unroll -> coop -> apiFunctionId
      // Type T is already known from template context
      // This automatically instantiates templates as we go down the layers
      // Work distribution is now handled inside the kernel
    unrollDispatch<T>(
      adaptedParams.unroll,
      adaptedParams.coopLevel,
      adaptedParams.funcId,
      kernelParams.sendwin, kernelParams.sendoffset,
      kernelParams.recvwin, kernelParams.recvoffset,
      kernelParams.nSrc, kernelParams.nDst, adaptedParams.nDstStart,
      launchCountPerRank, devComms[deviceId],
      adaptedParams.gridSize, adaptedParams.blockSize, streams[deviceId]
    );
  }
  if (numDevices > 1) {
    NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), activeComms.comms, numDevices);
  }
  int syncCount = numDevices;
  for (int i = 0; i < syncCount; ++i) {
    int deviceId = isLocalVariant ? 0 : i;
    CUDACHECK(cudaSetDevice(deviceId));
    CUDACHECK(cudaStreamSynchronize(streams[deviceId]));
  }
  for (int i = 0; i < syncCount; ++i) {
    int deviceId = isLocalVariant ? 0 : i;
    CUDACHECK(cudaSetDevice(deviceId));
    CUDACHECK(cudaDeviceSynchronize());
  }

    // Copy results back and validate
    // isAllGather already declared above
    const bool hasMultimemDst = TestSupportChecker::hasMultimemDestination(adaptedParams.funcId);
    // For AllGather, each rank receives all ranks' data concatenated (count * nRanks elements)
    // For other operations, each rank receives count elements
    size_t recvCount = isAllGather ? (storageCountPerRank * numDevices) : storageCountPerRank;
    // Strict tolerance for built-in reduceCopySum (float/higher-precision acc); relaxed for custom op (testOpSum/OpMul)
    const bool strictLowPrecisionTolerance = !usesCustomReduceOp(adaptedParams.funcId);

    if (!isLocalVariant && hasMultimemDst) {
      // Multimem destination is shared across ranks and visible on each device.
      // Validate that every device sees the same expected result.
      //
      // Lambda recv-offset note: for multimem destinations, the kernel computes
      // dstOffset as baseRecvOffset + startOffset, where baseRecvOffset is always 0
      // regardless of the recvoffset flag (see allreduceKernel: baseRecvOffset =
      // (recvoffset != 0) ? 0 : recvoffset, and dstLambda ignores the per-rank
      // resolveLambdaRecvOffsetBytes path). The multimem buffer is a single shared
      // physical allocation — per-rank recv offsets do not apply. Read from recvBase
      // directly (no getLambdaOffsets). Do NOT add an offset here; doing so would
      // mismatch what the kernel wrote.
      for (int deviceId = 0; deviceId < numDevices; ++deviceId) {
        CUDACHECK(cudaSetDevice(deviceId));
        void* recvBase = factory->getRecvBuffer(deviceId);
        if (!recvBase || expectedData.empty()) continue;
        char* recvBaseBytes = static_cast<char*>(recvBase);
        std::vector<T> actual(recvCount);
        T* dstPtr = reinterpret_cast<T*>(recvBaseBytes);
        CUDACHECK(cudaMemcpy(
          actual.data(),
          dstPtr,
          recvCount * sizeof(T),
          cudaMemcpyDeviceToHost
        ));
        const bool useRoundingModelMultimem = !sourceData.empty() && adaptedParams.nSrc >= 2 && sourceData.size() >= 2u;
        const int nSrcMultimem = useRoundingModelMultimem ? adaptedParams.nSrc : 0;
        for (size_t j = 0; j < recvCount && j < expectedData[0].size(); ++j) {
          OptDouble maxAbsSource;
          if (useRoundingModelMultimem) {
            double maxAbs = 0;
            for (int src = 0; src < adaptedParams.nSrc && src < static_cast<int>(sourceData.size()); ++src)
              if (j < sourceData[src].size())
                maxAbs = std::max(maxAbs, std::abs(static_cast<double>(toFloatForDisplay(sourceData[src][j]))));
            maxAbsSource = maxAbs;
          }
          std::ostringstream oss;
          oss << "Mismatch at multimem destination on device " << deviceId
            << ", element " << j;
          expectValuesEqual<T>(expectedData[0][j], actual[j], oss.str().c_str(),
              strictLowPrecisionTolerance, nSrcMultimem, maxAbsSource);
        }
      }
    } else {
      std::vector<std::vector<T>> actualData(adaptedParams.nDst);
      for (int i = 0; i < adaptedParams.nDst; ++i) {
        actualData[i].resize(recvCount);
      }

      for (int i = 0; i < numDevices; ++i) {
        int deviceId = isLocalVariant ? 0 : i;
        int rank = isLocalVariant ? 0 : deviceId;
        CUDACHECK(cudaSetDevice(deviceId));

        void* recvBase = factory->getRecvBuffer(deviceId);
        if (recvBase) {
          std::pair<size_t,size_t> lambdaOff1 = getLambdaOffsets(deviceId, rank);
          size_t recvOffsetBytes = lambdaOff1.second;
          (void)lambdaOff1.first;
          char* recvBaseBytes = static_cast<char*>(recvBase) + recvOffsetBytes;
          if (isLocalVariant) {
            T* recvPtr = reinterpret_cast<T*>(recvBaseBytes);
            for (int dstIdx = 0; dstIdx < adaptedParams.nDst; ++dstIdx) {
              T* dstPtr = recvPtr + dstIdx * storageCountPerRank;
              CUDACHECK(cudaMemcpy(
                actualData[dstIdx].data(),
                dstPtr,
                recvCount * sizeof(T),
                cudaMemcpyDeviceToHost
              ));
            }
          } else if (isNmSplit) {
            // NM: dst ranks are nDstStart..nDstStart+nDst-1; map to actualData[0..nDst-1]
            const int nDstStart = adaptedParams.nDstStart;
            if (deviceId >= nDstStart && deviceId < nDstStart + adaptedParams.nDst) {
              int dstIdx = deviceId - nDstStart;
              T* dstPtr = reinterpret_cast<T*>(recvBaseBytes);
              CUDACHECK(cudaMemcpy(
                actualData[dstIdx].data(),
                dstPtr,
                recvCount * sizeof(T),
                cudaMemcpyDeviceToHost
              ));
            }
          } else if (deviceId == 0 || !isLocalVariant) {
            int dstIdx = deviceId;
            if (dstIdx < adaptedParams.nDst) {
              T* dstPtr = reinterpret_cast<T*>(recvBaseBytes);
              CUDACHECK(cudaMemcpy(
                actualData[dstIdx].data(),
                dstPtr,
                recvCount * sizeof(T),
                cudaMemcpyDeviceToHost
              ));
            }
          }
        }
      }

      // Validate results
      // For NxN ReduceSumCopy, the API may only write to rank 0, so compare all expectedData[i]
      // against actualData[0].  For NM, each dst rank has its own independently-written result.
      const bool useSingleActualForReduceSumCopy =
        !isLocalVariant && !isAllGather && !hasMultimemDst && !isNmSplit &&
        !isReduceSumVariant(adaptedParams.funcId);  // i.e. NxN ReduceSumCopy
      const int actualIdx = (useSingleActualForReduceSumCopy && adaptedParams.nDst > 0) ? 0 : -1;

      bool printedMismatchDetails = false;
      for (int i = 0; i < adaptedParams.nDst; ++i) {
        int compareActualIdx = (actualIdx >= 0) ? actualIdx : i;
        if (compareActualIdx >= static_cast<int>(actualData.size())) continue;
        if (i < static_cast<int>(expectedData.size())) {
          // For AllGather, use recvCount; for others, use packed storage count
          size_t validateCount = isAllGather ? recvCount : storageCountPerRank;
          for (size_t j = 0; j < validateCount && j < expectedData[i].size() &&
               j < actualData[compareActualIdx].size(); ++j) {
            T expectedVal = expectedData[i][j];
            T actualVal = actualData[compareActualIdx][j];
            // Use rounding-probability model for sum comparisons when we have multiple sources.
            // This applies to both ReduceSumCopy and ReduceSum variants; the ULP tolerance
            // scales with source magnitude (maxAbsSource) so it handles near-cancellation correctly.
            const bool useRoundingModel = !sourceData.empty() &&
                            static_cast<size_t>(adaptedParams.nSrc) >= 2u;
            const int nSrcForModel = useRoundingModel ? adaptedParams.nSrc : 0;
            OptDouble maxAbsSourceForJ;
            if (useRoundingModel) {
              double maxAbs = 0;
              for (int src = 0; src < adaptedParams.nSrc && src < static_cast<int>(sourceData.size()); ++src)
                if (j < sourceData[src].size())
                  maxAbs = std::max(maxAbs, std::abs(static_cast<double>(toFloatForDisplay(sourceData[src][j]))));
              maxAbsSourceForJ = maxAbs;
            }
            if (!printedMismatchDetails &&
                !valuesEqual(expectedVal, actualVal, strictLowPrecisionTolerance, nSrcForModel, maxAbsSourceForJ) &&
                !sourceData.empty() && useSingleActualForReduceSumCopy) {
              VERBOSE_PRINTF("[TEST] Reduce mismatch at destination %d, element %zu:\n", i, static_cast<size_t>(j));
              VERBOSE_PRINTF("[TEST]   Per-rank values (as stored, shown in float):\n");
              for (int src = 0; src < adaptedParams.nSrc && src < static_cast<int>(sourceData.size()); ++src) {
                if (j < sourceData[src].size()) {
                  VERBOSE_PRINTF("[TEST]     Src %d: %.6e\n", src,
                      static_cast<double>(toFloatForDisplay(sourceData[src][j])));
                }
              }
              VERBOSE_PRINTF("[TEST]   expected=%.6e, actual=%.6e\n",
                   static_cast<double>(toFloatForDisplay(expectedVal)),
                   static_cast<double>(toFloatForDisplay(actualVal)));
              printedMismatchDetails = true;
            }
            std::ostringstream oss;
            oss << "Mismatch at destination " << i << ", element " << j;
            expectValuesEqual<T>(expectedVal, actualVal, oss.str().c_str(),
                strictLowPrecisionTolerance, nSrcForModel, maxAbsSourceForJ);
          }
        }
      }
    }

    // NM split: verify that src-only ranks (0..nDstStart-1) have the sentinel value in
    // their recv buffers — the kernel must not have touched them.
    if (isNmSplit && elementCount > 0) {
      const float sentinelFloat = kNMSentinel;
      const T sentinelVal = static_cast<T>(kNMSentinel);
      for (int i = 0; i < adaptedParams.nDstStart; ++i) {
        CUDACHECK(cudaSetDevice(i));
        void* recvBase = factory->getRecvBuffer(i);
        if (!recvBase) continue;
        std::vector<T> recvBuf(storageCountPerRank);
        CUDACHECK(cudaMemcpy(recvBuf.data(), recvBase,
          storageCountPerRank * sizeof(T), cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < elementCount; ++j) {
          std::ostringstream oss;
          oss << "NM src-only rank " << i << " element " << j
              << " was modified (expected sentinel " << sentinelFloat << ")";
          EXPECT_NEAR(static_cast<float>(recvBuf[j]), sentinelFloat, 1e-3f)
            << oss.str();
        }
      }
    }

    // Mark this test combination as executed only if validation succeeded
    if (!::testing::Test::HasFailure()) {
      // Use params.unroll which should match the expected UNROLL values in the test matrix
      MARK_TEST_EXECUTED(T, params.funcId, params.coopLevel, params.unroll, params.count,
                 params.maxTestGpus, params.enableLambdaOffsets);
    }

  for (int i = 0; i < numDevices; ++i) {
    CUDACHECK(cudaSetDevice(i));
    NCCLCHECK(ncclDevCommDestroy(activeComms.comms[i], &devComms[i]));
  }
  factory->cleanup();
  destroyActiveComms(&activeComms);
}

// Parameterized test suite for window-based variants. Single inheritance from
// TestWithParam to avoid ambiguous Test base; per-test SetUp/TearDown (stream
// sync) are called explicitly via ReduceCopyTestBase<T>::PerTest*.
//
// Note: inheriting from ncclDevApiCommon_test here would cause an ambiguous
// ::testing::Test base because ncclCommon_test uses non-virtual ::testing::Test
// inheritance, which conflicts with TestWithParam's own ::testing::Test base.
template<typename T>
class ReduceCopyWindowTestImpl : public ::testing::TestWithParam<TestParams> {
public:
  using Base = ReduceCopyTestBase<T>;

  static void SetUpTestCase() {
    ReduceCopyTestBase<T>::SetUpTestCase();
  }

  static void TearDownTestCase() {
    ReduceCopyTestBase<T>::TearDownTestCase();
  }

protected:
  void SetUp() override { ReduceCopyTestBase<T>::PerTestSetUp(); }
  void TearDown() override { ReduceCopyTestBase<T>::PerTestTearDown(); }

  void runFullTest(const TestParams& params) {
    runTestWithAbortHandling([&]() {
      runFullTestImpl<T>(params, Base::comms.data(), Base::nVis,
          Base::streams.data(), RunFullTestOptions{.skipUnimplemented = true});
    });
  }
};

#endif // NCCL_REDUCE_COPY_TEST_REDUCE_COPY_H
