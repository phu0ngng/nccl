// GIN Signal tests (indexed signals)
#include "ncclDevApiCommon_test.cuh"
#include <cassert>

////////////////////////////////////////////////////////////////////////////////
// Test kernels for indexed signals
////////////////////////////////////////////////////////////////////////////////

__global__ void signalRingKernel(ncclDevComm comm, int contextIdx, int signalIdx) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, contextIdx);

  int nextRank = (world.rank + 1) % world.nRanks;

  // Signal the next rank
  gin.signal(world, nextRank, ncclGin_SignalInc{(ncclGinSignal_t)signalIdx});

  // Wait for signal from previous rank
  gin.waitSignal(ncclCoopCta(), signalIdx, 1);
#endif
}

__global__ void signalResetKernel(ncclDevComm comm, int contextIdx, int signalIdx) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, contextIdx);

  gin.signal(world, (world.rank + 1) % world.nRanks, ncclGin_SignalInc{(ncclGinSignal_t)signalIdx});
  gin.waitSignal(ncclCoopCta(), signalIdx, 1);
  gin.resetSignal(signalIdx);
  uint64_t value = gin.readSignal(signalIdx);
  KERNEL_ASSERT_EQ(0, value, "Signal should be 0 after reset");
#endif
}

__global__ void signalBasicKernel(ncclDevComm comm, int contextIdx, ncclGinSignal_t signalIdx, bool isAdd) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, contextIdx);
  
  if (world.nRanks < 2) {
    return;
  }
  
  if (world.rank == 0) {
    if (isAdd) {
      gin.signal(world, 1, ncclGin_SignalAdd{signalIdx, 1});
    } else {
      gin.signal(world, 1, ncclGin_SignalInc{signalIdx});
    }
  }

  if (world.rank == 1) {
    gin.waitSignal(ncclCoopCta(), signalIdx, 1);
    uint64_t value = gin.readSignal(signalIdx);
    KERNEL_ASSERT_EQ(1, value, "Signal should be 1 after wait signal");
  }
#endif
}

// Tests that signals for different contexts are independent.
__global__ void signalMultipleContextsKernel(ncclDevComm comm, int maxContexts, int signalIdx) {
  ncclTeam world = ncclTeamWorld(comm);
  for (int contextIdx = 0; contextIdx < maxContexts; contextIdx++) {
    ncclGin gin(comm, contextIdx);
    gin.signal(world, (world.rank + 1) % world.nRanks, ncclGin_SignalInc{(ncclGinSignal_t)signalIdx});
  }

  for (int contextIdx = 0; contextIdx < maxContexts; contextIdx++) {
    ncclGin gin(comm, contextIdx);
    gin.waitSignal(ncclCoopCta(), signalIdx, 1);
    uint64_t value = gin.readSignal(signalIdx);
    KERNEL_ASSERT_EQ(1, value, "Signal should be 1 after wait signal");
  }
}

// Simple kernel to verify all signals and counters are zero
__global__ void verifyAllZeroKernel(ncclDevComm comm, int contextIdx, 
                                    int signalCount, int counterCount) {
#if __CUDA_ARCH__ >= 700
  if (threadIdx.x != 0 || blockIdx.x != 0) return;  // Only thread 0 does work

  ncclGin gin(comm, contextIdx);

  // Check all signals
  for (int i = 0; i < signalCount; i++) {
    uint64_t value = gin.readSignal(i);
    KERNEL_ASSERT_EQ(0, value, "Signal value is not 0");
  }

  // Check all counters
  for (int i = 0; i < counterCount; i++) {
    uint64_t value = gin.readCounter(i);
    KERNEL_ASSERT_EQ(0, value, "Counter value is not 0");
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Test parameters
////////////////////////////////////////////////////////////////////////////////

struct GinSignalParams {
  int signalIdx;
  int contextIdx;
  
  std::string toString() const {
    return std::string("signal_") + std::to_string(signalIdx) + 
           "_context_" + std::to_string(contextIdx);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Test class for indexed signals
////////////////////////////////////////////////////////////////////////////////

class GinSignal_test : public ncclDevApiCommon_test,
                      public ::testing::WithParamInterface<GinSignalParams> {};

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

TEST_P(GinSignal_test, ring) {
  const GinSignalParams& params = GetParam();
  
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = params.signalIdx + 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  TESTCHECK(createDevComms(reqs));
  
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalRingKernel<<<1, 1, 0, streams[i]>>>(
      devComms[i], params.contextIdx, params.signalIdx);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinSignal_test, reset) {
  const GinSignalParams& params = GetParam();
  
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = params.signalIdx + 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  TESTCHECK(createDevComms(reqs));
  
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalResetKernel<<<1, 1, 0, streams[i]>>>(
      devComms[i], params.contextIdx, params.signalIdx);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinSignal_test, basic_add) {
  const GinSignalParams& params = GetParam();
  
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = params.signalIdx + 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalBasicKernel<<<1, 1, 0, streams[i]>>>(
      devComms[i], params.contextIdx, (ncclGinSignal_t)params.signalIdx, true);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinSignal_test, basic_inc) {
  const GinSignalParams& params = GetParam();
  
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = params.signalIdx + 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalBasicKernel<<<1, 1, 0, streams[i]>>>(
      devComms[i], params.contextIdx, (ncclGinSignal_t)params.signalIdx, false);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinSignal_test, independent_contexts) {
  const GinSignalParams& params = GetParam();
  
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = params.signalIdx + 1;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalMultipleContextsKernel<<<1, 1, 0, streams[i]>>>(
      devComms[i], params.contextIdx + 1, params.signalIdx);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinSignal_test, signal_counter_init_zero) {
  const GinSignalParams& params = GetParam();

  const int signalCount = 5;
  const int counterCount = 5;

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = signalCount;
  reqs.ginCounterCount = counterCount;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));

    signalRingKernel<<<1, 1, 0, streams[i]>>>(devComms[i], params.contextIdx, params.signalIdx);
  }
  syncAllDevices();

  // Step 2: Destroy devComms (this should trigger the reset kernel)
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(ncclSuccess, ncclDevCommDestroy(comms[i], &devComms[i]));
  }
  devComms.clear();

  // Step 3: Recreate devComms to access the same memory
  TESTCHECK(createDevComms(reqs));

  // Step 4: Verify all signals and counters are zero
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));

    // Launch verification kernel
    verifyAllZeroKernel<<<1, 1, 0, streams[i]>>>(
      devComms[i], params.contextIdx, signalCount, counterCount);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));

    // Check for kernel assertion failures
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(cudaSuccess, err) << "Device " << i << ": " << cudaGetErrorString(err);
  }

  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Test failed: " << cudaGetErrorString(err);
}

////////////////////////////////////////////////////////////////////////////////
// Test instantiation
////////////////////////////////////////////////////////////////////////////////

std::string GinSignalParamsName(const ::testing::TestParamInfo<GinSignalParams>& info) {
  return info.param.toString();
}

INSTANTIATE_TEST_CASE_P(
  Test,
  GinSignal_test,
  ::testing::Values(
    GinSignalParams{0, 0},  // Signal 0, Context 0
    GinSignalParams{0, 1},  // Signal 0, Context 1
    GinSignalParams{1, 0},  // Signal 1, Context 0
    GinSignalParams{1, 1}   // Signal 1, Context 1
  ),
  GinSignalParamsName
);
