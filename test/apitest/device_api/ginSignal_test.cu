// GIN Signal tests (indexed signals)
#include "ncclDevApiCommon_test.cuh"
#include <cassert>

template <typename Coop>
__device__ void ginSignalInc(ncclGin& gin, Coop coop, int destRank, ncclGinSignal_t signalIdx, GinSignalType signalType) {
  switch (signalType) {
    case GinSignalType::Strong: gin.signal(coop, destRank, ncclGin_StrongSignalInc{signalIdx}); break;
    case GinSignalType::Weak:   gin.signal(coop, destRank, ncclGin_WeakSignalInc{signalIdx});   break;
    case GinSignalType::Legacy: gin.signal(coop, destRank, ncclGin_SignalInc{signalIdx});       break;
  }
}

template <typename Coop>
__device__ void ginSignalAdd(ncclGin& gin, Coop coop, int destRank, ncclGinSignal_t signalIdx, uint64_t value, GinSignalType signalType) {
  switch (signalType) {
    case GinSignalType::Strong: gin.signal(coop, destRank, ncclGin_StrongSignalAdd{signalIdx, value}); break;
    case GinSignalType::Weak:   gin.signal(coop, destRank, ncclGin_WeakSignalAdd{signalIdx, value});   break;
    case GinSignalType::Legacy: gin.signal(coop, destRank, ncclGin_SignalAdd{signalIdx, value});       break;
  }
}

__global__ void signalRingKernel(ncclDevComm comm, int contextIdx, int signalIdx, GinSignalType signalType) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, contextIdx);

  int nextRank = (world.rank + 1) % world.nRanks;

  ginSignalInc(gin, world, nextRank, (ncclGinSignal_t)signalIdx, signalType);

  // Wait for signal from previous rank
  gin.waitSignal(ncclCoopCta(), signalIdx, 1);
#endif
}

__global__ void signalResetKernel(ncclDevComm comm, int contextIdx, int signalIdx, GinSignalType signalType) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, contextIdx);

  ginSignalInc(gin, world, (world.rank + 1) % world.nRanks, (ncclGinSignal_t)signalIdx, signalType);
  gin.waitSignal(ncclCoopCta(), signalIdx, 1);
  gin.resetSignal(signalIdx);
  uint64_t value = gin.readSignal(signalIdx);
  KERNEL_ASSERT_EQ(0, value, "Signal should be 0 after reset");
#endif
}

__global__ void signalBasicKernel(ncclDevComm comm, int contextIdx, ncclGinSignal_t signalIdx, bool isAdd, GinSignalType signalType) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, contextIdx);

  if (world.nRanks < 2) {
    return;
  }

  if (world.rank == 0) {
    if (isAdd) {
      ginSignalAdd(gin, world, 1, signalIdx, 1, signalType);
    } else {
      ginSignalInc(gin, world, 1, signalIdx, signalType);
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
__global__ void signalMultipleContextsKernel(ncclDevComm comm, int maxContexts, int signalIdx, GinSignalType signalType) {
  ncclTeam world = ncclTeamWorld(comm);
  for (int contextIdx = 0; contextIdx < maxContexts; contextIdx++) {
    ncclGin gin(comm, contextIdx);
    ginSignalInc(gin, world, (world.rank + 1) % world.nRanks, (ncclGinSignal_t)signalIdx, signalType);
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
  GinSignalType signalType;

  std::string toString() const {
    const char* typeName = (signalType == GinSignalType::Strong) ? "strong"
                         : (signalType == GinSignalType::Weak)   ? "weak"
                                                                  : "legacy";
    return std::string("signal_") + std::to_string(signalIdx) +
           "_context_" + std::to_string(contextIdx) +
           "_" + typeName;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Test class for indexed signals
////////////////////////////////////////////////////////////////////////////////

class GinSignal_test : public ncclDevApiCommon_test,
                      public ::testing::WithParamInterface<GinSignalParams> {};

// Non-parameterized test class for misc signal tests (weak signals and init-to-zero).
class GinSignalMisc_test : public ncclDevApiCommon_test {};

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
      devComms[i], params.contextIdx, params.signalIdx, params.signalType);
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
      devComms[i], params.contextIdx, params.signalIdx, params.signalType);
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
      devComms[i], params.contextIdx, (ncclGinSignal_t)params.signalIdx, true, params.signalType);
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
      devComms[i], params.contextIdx, (ncclGinSignal_t)params.signalIdx, false, params.signalType);
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
      devComms[i], params.contextIdx + 1, params.signalIdx, params.signalType);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

////////////////////////////////////////////////////////////////////////////////
// Non-parameterized tests: weak signals and signal/counter init-to-zero
////////////////////////////////////////////////////////////////////////////////

TEST_F(GinSignalMisc_test, signal_counter_init_zero) {
  const int signalIdx = 0;
  const int contextIdx = 0;
  const int signalCount = 5;
  const int counterCount = 5;

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = signalCount;
  reqs.ginCounterCount = counterCount;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalRingKernel<<<1, 1, 0, streams[i]>>>(devComms[i], contextIdx, signalIdx, GinSignalType::Strong);
  }
  syncAllDevices();

  destroyDevComms();
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    verifyAllZeroKernel<<<1, 1, 0, streams[i]>>>(devComms[i], contextIdx, signalCount, counterCount);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(cudaSuccess, err) << "Device " << i << ": " << cudaGetErrorString(err);
  }

  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Test failed: " << cudaGetErrorString(err);
}

TEST_F(GinSignalMisc_test, weak_basic_add) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginStrongSignalsRequired = false;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalBasicKernel<<<1, 1, 0, streams[i]>>>(devComms[i], 0, (ncclGinSignal_t)0, true, GinSignalType::Weak);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(GinSignalMisc_test, weak_basic_inc) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginStrongSignalsRequired = false;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalBasicKernel<<<1, 1, 0, streams[i]>>>(devComms[i], 0, (ncclGinSignal_t)0, false, GinSignalType::Weak);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

////////////////////////////////////////////////////////////////////////////////
// Parameterized test instantiation
////////////////////////////////////////////////////////////////////////////////

std::string GinSignalParamsName(const ::testing::TestParamInfo<GinSignalParams>& info) {
  return info.param.toString();
}

INSTANTIATE_TEST_CASE_P(
  Test,
  GinSignal_test,
  ::testing::Values(
    GinSignalParams{0, 0, GinSignalType::Strong},
    GinSignalParams{0, 1, GinSignalType::Legacy},
    GinSignalParams{1, 0, GinSignalType::Legacy},
    GinSignalParams{1, 1, GinSignalType::Strong}
  ),
  GinSignalParamsName
);
