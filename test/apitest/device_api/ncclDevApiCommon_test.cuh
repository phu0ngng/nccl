#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <nccl_device.h>

#include <map>
#include <string>
#include <vector>
#include <utility>

#include "../ncclCommon_test.cuh"

// Prints expects and actual if not equal.
#define KERNEL_ASSERT_EQ(expected, actual, msg) do { \
  if ((expected) != (actual)) { \
    printf("%s: expected %llu but got %llu\n", msg, (unsigned long long)(expected), (unsigned long long)(actual)); \
    assert((expected) == (actual) && msg); \
  } \
} while(0)

enum class TestResult_t {
  testSuccess = 0,
  testSkipped = 1,
  testError = 2
};

// Macro to check the status and skip/fail as appropriate
#define TESTCHECK(status) do { \
  TestResult_t _status = (status); \
  if (_status == TestResult_t::testSkipped) { \
    return; \
  } else if (_status == TestResult_t::testError) { \
    FAIL(); \
  } \
} while(0)

// ncclDevApiCommon_test is a base class for all Device API tests.
// Comms are created once at the beginning of the test suite and destroyed at the end.
// DevComms are not automatically created - each test must call createDevComms with the appropriate requirements.
class ncclDevApiCommon_test : public ncclCommon_test<char> {
public:
  // Per-test state. Each test case must create its own devComms.
  std::vector<ncclDevComm> devComms;

  // Called once before all tests in the suite
  static void SetUpTestCase() {
    // Call parent's SetUpTestCase to initialize comms and streams
    ncclCommon_test<char>::SetUpTestCase();
  }

  // Called once after all tests in the suite
  static void TearDownTestCase() {
    // Call parent's TearDownTestCase to destroy comms
    ncclCommon_test<char>::TearDownTestCase();
  }

protected:

  // Called before each test.
  void SetUp() override {
    ncclCommon_test<char>::SetUp();
    cudaGetLastError();  // Clear any stale errors. Ignore value.
  }

  // Called after each test - destroys devComms if created
  void TearDown() override {
    syncAllDevices();
    if (!devComms.empty()) {
      for (int i = 0; i < nVis; i++) {
        cudaSetDevice(i);
        if (comms && comms[i]) {
          ncclDevCommDestroy(comms[i], &devComms[i]);
        }
      }
      devComms.clear();
    }

    ncclCommon_test<char>::TearDown();
  }

  // call cudaStreamSynchronize on all streams
  void syncAllDevices() {
    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
    }
  }

  // Helper to create devComms with specific requirements
  // Returns TestResult_t: testSuccess, testSkipped (if not supported), or testError
  // Tests should use TESTCHECK(createDevComms(reqs)) to handle the result
  TestResult_t createDevComms(const ncclDevCommRequirements& reqs) {
    if (devComms.size() != 0) { // Something has gone wrong if we already have devComms
      return TestResult_t::testError;
    }
    devComms.resize(nVis);
    
    // First, query properties and check if supported
    for (int i = 0; i < nVis; i++) {
      ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
      ncclResult_t res = ncclCommQueryProperties(comms[i], &props);
      if (res != ncclSuccess) {
        return TestResult_t::testError;
      }
      
      if (!props.deviceApiSupport) {
        return TestResult_t::testSkipped;
      }
      bool ginRequested = reqs.ginForceEnable || reqs.ginConnectionType != NCCL_GIN_CONNECTION_NONE;
      if (ginRequested && props.ginType == NCCL_GIN_TYPE_NONE) {
        return TestResult_t::testSkipped;
      }
    }
    
    // Now create the devComms
    ncclResult_t res = ncclGroupStart();
    if (res != ncclSuccess) return TestResult_t::testError;
    
    for (int i = 0; i < nVis; i++) {
      cudaError_t cudaErr = cudaSetDevice(i);
      if (cudaErr != cudaSuccess) {
        ncclGroupEnd();
        return TestResult_t::testError;
      }
      res = ncclDevCommCreate(comms[i], &reqs, &devComms[i]);
      if (res != ncclSuccess) {
        ncclGroupEnd();
        return TestResult_t::testError;
      }
    }
    
    res = ncclGroupEnd();
    if (res != ncclSuccess) return TestResult_t::testError;
    
    cudaGetLastError();  // Clear any stale errors
    return TestResult_t::testSuccess;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Helper functions for window management
////////////////////////////////////////////////////////////////////////////////

// Helper function to allocate and register windows for all devices
inline void allocateAndRegisterWindows(int nVis, ncclComm_t* comms, size_t size, 
                                       std::vector<void*>& ptrs, std::vector<ncclWindow_t>& wins) {
  ptrs.resize(nVis);
  wins.resize(nVis);
  
  ncclResult_t res = ncclGroupStart();
  ASSERT_EQ(ncclSuccess, res);
  
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(ncclSuccess, ncclMemAlloc(&ptrs[i], size));
    ASSERT_NE(nullptr, ptrs[i]);
    
    // Initialize to zero
    ASSERT_EQ(cudaSuccess, cudaMemset(ptrs[i], 0, size));
    
    // Register window using public API
    ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], ptrs[i], size,
                                                   &wins[i], NCCL_WIN_COLL_SYMMETRIC));
  }
  
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
}

// Helper function to deregister and free windows for all devices
inline void deregisterAndFreeWindows(int nVis, ncclComm_t* comms,
                                     std::vector<void*>& ptrs, std::vector<ncclWindow_t>& wins) {
  for (int i = 0; i < nVis; i++) {
    if (wins[i] != nullptr) {
      ncclCommWindowDeregister(comms[i], wins[i]);
    }
    if (ptrs[i] != nullptr) {
      ncclMemFree(ptrs[i]);
    }
  }
}
