/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_REDUCE_COPY_TEST_PARAMS_H
#define NCCL_REDUCE_COPY_TEST_PARAMS_H

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include "config.h"
#include "api_function_traits.h"
#include "test_functions.h"
#include "test_matrix.h"

// Test parameter structure for parameterized tests
struct TestParams {
  ApiFunctionId funcId;
  size_t count;
  int nSrc;
  int nDst;
  CooperationLevel coopLevel;
  int unroll;
  int gridSize;
  int blockSize;
  int maxTestGpus;
  bool enableLambdaOffsets;
  // NM split: when >0, dstRanks start at this index (srcRanks = 0..nSrc-1,
  // dstRanks = nDstStart..nVis-1).  0 = standard NxN mode.
  int nDstStart = 0;
};

// Unroll variants used by generated test instantiations.
enum class UnrollKind {
  One,
  Default
};

template<typename T>
inline int getUnrollForKind(UnrollKind kind) {
  switch (kind) {
    case UnrollKind::One: return 1;
    case UnrollKind::Default: return getDefaultUnroll<T>();
    default: return getDefaultUnroll<T>();
  }
}

template<typename T>
inline TestParams makeWindowTestParams(ApiFunctionId funcId,
                     CooperationLevel coopLevel,
                     size_t count,
                     UnrollKind unrollKind) {
  TestParams p;
  p.funcId = funcId;
  p.count = count;
  p.nSrc = 2;   // Placeholder - will be set to nVis for inter-rank tests
  p.nDst = 2;   // Placeholder - will be set based on operation type
  p.coopLevel = coopLevel;
  p.unroll = getUnrollForKind<T>(unrollKind);
  p.gridSize = 1;
  p.blockSize = 256;
  p.maxTestGpus = 0;
  p.enableLambdaOffsets = false;
  return p;
}

template<typename T>
inline TestParams makeWindowTestParamsWithRunMode(ApiFunctionId funcId,
                          CooperationLevel coopLevel,
                          size_t count,
                          UnrollKind unrollKind,
                          int maxTestGpus,
                          bool enableLambdaOffsets) {
  TestParams p = makeWindowTestParams<T>(funcId, coopLevel, count, unrollKind);
  p.maxTestGpus = maxTestGpus;
  p.enableLambdaOffsets = enableLambdaOffsets;
  return p;
}

// NM split scenario parameters: srcRanks = {0..nSrc-1}, dstRanks = {nDstStart..nVis-1}.
// nDst is left 0 and computed at runtime as nVis - nDstStart.
template<typename T>
inline TestParams makeNMTestParams(ApiFunctionId funcId,
                   CooperationLevel coopLevel,
                   size_t count,
                   UnrollKind unrollKind,
                   int nSrc,
                   int nDstStart) {
  TestParams p = makeWindowTestParams<T>(funcId, coopLevel, count, unrollKind);
  p.nSrc = nSrc;
  p.nDst = 0;  // Computed as nVis - nDstStart in runFullTestImpl
  p.nDstStart = nDstStart;
  return p;
}

template<typename T>
std::vector<TestParams> getWindowTestParamsForType() {
  std::vector<TestParams> params;

  // Base parameters (same for all tests)
  TestParams base;
  base.nSrc = 2;  // Placeholder (will be set to nVis in test execution)
  base.nDst = 2;  // Placeholder
  base.gridSize = 1;
  base.blockSize = 256;
  base.maxTestGpus = 0;
  base.enableLambdaOffsets = false;

  // Parameter lists
  std::vector<CooperationLevel> coopLevels = {
    CooperationLevel::Thread,
    CooperationLevel::Warp,
    CooperationLevel::Cta
  };

  std::vector<int> unrollValues = {
    1,
    getDefaultUnroll<T>()
  };

  std::vector<size_t> counts = {NCCL_REDUCE_COPY_COUNTS_LIST};

  auto pushParam = [&](const TestParams& p) {
    params.push_back(p);
  };

  // Single loop over all API functions; nSrc/nDst from central traits.
  for (ApiFunctionId funcId : getAllApiFunctionIds()) {
    TestParams baseForFunc = base;
    baseForFunc.nSrc = apiTraitsNSrcDefault(funcId);
    baseForFunc.nDst = apiTraitsNDstDefault(funcId);
    for (CooperationLevel coopLevel : coopLevels) {
      for (size_t count : counts) {
        for (int unroll : unrollValues) {
          TestParams p = baseForFunc;
          p.funcId = funcId;
          p.coopLevel = coopLevel;
          p.count = count;
          p.unroll = unroll;
          pushParam(p);
        }
      }
    }
  }

  return params;
}

// Filtered test parameters for a single API function.
template<typename T>
std::vector<TestParams> getWindowTestParamsForTypeAndFunc(ApiFunctionId funcId) {
  std::vector<TestParams> params = getWindowTestParamsForType<T>();
  params.erase(
    std::remove_if(params.begin(), params.end(),
             [funcId](const TestParams& p) { return p.funcId != funcId; }),
    params.end());
  return params;
}

// Custom test name generator. Param name omits func+coop (they appear in the test case name).
struct TestParamsNameGenerator {
  template <class ParamType>
  std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const {
    const TestParams& params = info.param;
    std::ostringstream oss;
    oss << "UNROLL" << params.unroll << "_count" << params.count;
    if (params.maxTestGpus > 0) {
      oss << "_gpus" << params.maxTestGpus;
    }
    if (params.enableLambdaOffsets) {
      oss << "_offsets";
    }
    if (params.nDstStart > 0) {
      oss << "_nSrc" << params.nSrc << "_nDstStart" << params.nDstStart;
    }
    return oss.str();
  }
};

#endif // NCCL_REDUCE_COPY_TEST_PARAMS_H
