/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_COMMON_CUH_
#define _REDUCE_COPY_TEST_COMMON_CUH_

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include "nccl.h"
#include "nccl_device.h"
#include "checks.h"
#include "config.h"
#include "support.h"
#include "test_matrix.h"

// Base test fixture for ReduceCopy tests (device API test suite).
// Use virtual inheritance to support diamond inheritance when combined with TestWithParam.
// Each test skips via TestSupportChecker::isDeviceApiSupported() when device API is not supported.
//
// Note: ncclDevApiCommon_test cannot be used as a base here because ncclCommon_test<char>
// uses non-virtual inheritance from ::testing::Test, which conflicts with TestWithParam's
// ::testing::Test base and produces an ambiguous base class error.
template<typename T>
class ReduceCopyTestBase : public virtual ::testing::Test {
public:
  static int nVis;              // Number of visible GPUs
  static std::vector<ncclComm_t> comms; // One per device [nVis]
  static std::vector<cudaStream_t> streams;  // One per device [nVis]

  static void SetUpTestCase() {
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
    ASSERT_GT(nVis, 0) << "No GPUs available";
    TestMatrix::getInstance().setVisibleGpus(nVis);

    // Clear any lingering CUDA errors before the suite begins.
    for (int i = 0; i < nVis; ++i) {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
      cudaGetLastError();
    }

    comms.assign(nVis, ncclComm_t{});
    NCCLCHECK(ncclCommInitAll(comms.data(), nVis, NULL));

    streams.assign(nVis, nullptr);
    for (int i = 0; i < nVis; ++i) {
      EXPECT_EQ(cudaSuccess, cudaSetDevice(i));
      EXPECT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
    }
  }

  static void TearDownTestCase() {
    for (int i = 0; i < nVis; ++i) {
      CUDACHECK_NO_THROW(cudaSetDevice(i));
      if (!streams.empty() && i < static_cast<int>(streams.size()) && streams[i]) {
        CUDACHECK_NO_THROW(cudaStreamSynchronize(streams[i]));
        CUDACHECK_NO_THROW(cudaStreamDestroy(streams[i]));
      }
    }
    streams.clear();

    for (int i = 0; i < nVis; ++i) {
      if (i < static_cast<int>(comms.size()) && comms[i]) {
        NCCLCHECK_NO_THROW(ncclCommDestroy(comms[i]));
      }
    }
    comms.clear();
  }

  // Per-test setup/teardown (stream sync). Public so parameterized fixtures
  // that cannot inherit from this class can call them directly.
  static void PerTestSetUp() {
    if (nVis > 0 && !streams.empty()) {
      for (int i = 0; i < nVis; ++i) {
        if (i < static_cast<int>(streams.size()) && streams[i]) {
          CUDACHECK_NO_THROW(cudaSetDevice(i));
          CUDACHECK_NO_THROW(cudaStreamSynchronize(streams[i]));
        }
      }
    }
  }

  static void PerTestTearDown() {
    if (nVis > 0 && !streams.empty()) {
      for (int i = 0; i < nVis; ++i) {
        if (i < static_cast<int>(streams.size()) && streams[i]) {
          CUDACHECK_NO_THROW(cudaSetDevice(i));
          CUDACHECK_NO_THROW(cudaStreamSynchronize(streams[i]));
        }
      }
      for (int i = 0; i < nVis; ++i) {
        CUDACHECK_NO_THROW(cudaSetDevice(i));
        CUDACHECK_NO_THROW(cudaDeviceSynchronize());
      }
    }
  }

protected:
  void SetUp() override { PerTestSetUp(); }
  void TearDown() override { PerTestTearDown(); }
};

// Static member definitions
template<typename T>
int ReduceCopyTestBase<T>::nVis = -1;

template<typename T>
std::vector<ncclComm_t> ReduceCopyTestBase<T>::comms;

template<typename T>
std::vector<cudaStream_t> ReduceCopyTestBase<T>::streams;

#endif // _REDUCE_COPY_TEST_COMMON_CUH_
