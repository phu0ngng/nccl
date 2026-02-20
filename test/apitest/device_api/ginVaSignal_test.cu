// GIN VA Signal tests
#include "ncclDevApiCommon_test.cuh"
#include <cassert>

////////////////////////////////////////////////////////////////////////////////
// Cooperation level enum and helper
////////////////////////////////////////////////////////////////////////////////

enum class CoopLevel {
  Thread = 0,
  Cta = 2,
  Warp = 1
};

inline int getNumThreadsForOneCoop(CoopLevel level) {
  switch (level) {
    case CoopLevel::Thread:
      return 1;
    case CoopLevel::Warp:
      return 32;
    case CoopLevel::Cta:
      return 64;
  }
  return 1; // Default
}

__device__ inline ncclCoopAny getCoopFromLevel(CoopLevel level) {
  if (level == CoopLevel::Thread) {
    return ncclCoopThread();
  } else if (level == CoopLevel::Warp) {
    return ncclCoopWarp();
  } else {
    return ncclCoopCta();
  }
}

// This function only exists due to https://nvbugspro.nvidia.com/bug/5870672.
// There is a suspected compiler bug when you call waitSignal using a coop variable that has type ncclCoopAny.
// This function avoids the bug by switching based on the coop level.
__device__ inline void waitSignal(ncclGin gin, CoopLevel level, ncclWindow_t window, size_t offset) {
  switch (level) {
    case CoopLevel::Thread:
      gin.waitSignal(ncclCoopThread(), window, offset, 1);
      break;
    case CoopLevel::Warp:
      gin.waitSignal(ncclCoopWarp(), window, offset, ncclCoopWarp().size());
      break;
    case CoopLevel::Cta:
      gin.waitSignal(ncclCoopCta(), window, offset, ncclCoopCta().size());
      break;
    default:
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Test kernels for VA signals
////////////////////////////////////////////////////////////////////////////////

__global__ void signalVaRingKernel(ncclDevComm comm, int contextIdx,
                                    ncclWindow_t window, size_t offset, CoopLevel coopLevel) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, contextIdx);
  ncclCoopAny coop = getCoopFromLevel(coopLevel);
  KERNEL_ASSERT_EQ(coop.size(), ncclCoopCta().size(), "This test assumes exactly one coop is launched");

  int nextRank = (world.rank + 1) % world.nRanks;

  // Signal the next rank
  gin.signal(world, nextRank, ncclGin_VASignalInc{window, offset});

  // Wait for signal from previous rank
  waitSignal(gin, coopLevel, window, offset);
  // TODO: call waitSignal directly once https://nvbugspro.nvidia.com/bug/5870672 is fixed
  // gin.waitSignal(coop, window, offset, coop.size());
#endif
}

__global__ void signalVaResetKernel(ncclDevComm comm, int contextIdx,
                                     ncclWindow_t window, size_t offset, CoopLevel coopLevel) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, contextIdx);
  ncclCoopAny coop = getCoopFromLevel(coopLevel);
  KERNEL_ASSERT_EQ(coop.size(), ncclCoopCta().size(), "This test assumes exactly one coop is launched");


  gin.signal(world, (world.rank + 1) % world.nRanks, ncclGin_VASignalInc{window, offset});

  // Wait for signal from previous rank
  waitSignal(gin, coopLevel, window, offset);
  // TODO: call waitSignal directly once https://nvbugspro.nvidia.com/bug/5870672 is fixed
  // gin.waitSignal(coop, window, offset, coop.size());

  gin.resetSignal(window, offset);
  uint64_t value = gin.readSignal(window, offset);
  KERNEL_ASSERT_EQ(0, value, "Signal should be 0 after reset");
#endif
}

__global__ void signalVaBasicKernel(ncclDevComm comm, int contextIdx, bool isAdd,
                                     ncclWindow_t window, size_t offset, CoopLevel coopLevel) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, contextIdx);
  ncclCoopAny coop = getCoopFromLevel(coopLevel);
  KERNEL_ASSERT_EQ(coop.size(), ncclCoopCta().size(), "This test assumes exactly one coop is launched");

  if (world.nRanks < 2) {
    return;
  }

  if (world.rank == 0) {
    if (isAdd) {
      gin.signal(world, 1, ncclGin_VASignalAdd{window, offset, 1});
    } else {
      gin.signal(world, 1, ncclGin_VASignalInc{window, offset});
    }
  }

  if (world.rank == 1) {
    // Wait for signal
    waitSignal(gin, coopLevel, window, offset);
    // TODO: call waitSignal directly once https://nvbugspro.nvidia.com/bug/5870672 is fixed
    // gin.waitSignal(coop, window, offset, coop.size());

    uint64_t value = gin.readSignal(window, offset);
    KERNEL_ASSERT_EQ(coop.size(), value, "Signal should be coop.size() after wait signal");
    uint64_t* signalPtr = (uint64_t*)ncclGetLocalPointer(window, offset);
    KERNEL_ASSERT_EQ(coop.size(), *signalPtr, "Signal should be coop.size() after wait signal");
  }
#endif
  }

////////////////////////////////////////////////////////////////////////////////
// Test parameters
////////////////////////////////////////////////////////////////////////////////

struct GinVaSignalParams {
  int contextIdx;
  size_t offset;
  CoopLevel coop;

  std::string toString() const {
    std::string coopStr = (coop == CoopLevel::Thread ? "thread" :
                           coop == CoopLevel::Warp ? "warp" : "cta");
    return std::string("context_") + std::to_string(contextIdx) +
           "_offset_" + std::to_string(offset) +
           "_coop_" + coopStr;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Test class for VA signals
////////////////////////////////////////////////////////////////////////////////

class GinVaSignal_test : public ncclDevApiCommon_test,
                        public ::testing::WithParamInterface<GinVaSignalParams> {
protected:
  std::vector<void*> signalBuffers;
  std::vector<ncclWindow_t> windows;
  size_t signalBufferSize;


public:
  void SetUp() override {
    ncclDevApiCommon_test::SetUp();

    // Calculate buffer size based on maximum offset in test parameters
    // Test params use offsets: 0, sizeof(uint64_t)
    // Need room for signal at max offset plus the signal itself
    size_t maxOffset = sizeof(uint64_t);
    signalBufferSize = maxOffset + sizeof(uint64_t);

    allocateAndRegisterWindows(nVis, comms, signalBufferSize, signalBuffers, windows);
  }

  void TearDown() override {
    deregisterAndFreeWindows(nVis, comms, signalBuffers, windows);
    ncclDevApiCommon_test::TearDown();
  }
};

////////////////////////////////////////////////////////////////////////////////
// VA Signal Tests
////////////////////////////////////////////////////////////////////////////////

TEST_P(GinVaSignal_test, ring) {
  const GinVaSignalParams& params = GetParam();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));

  int numThreads = getNumThreadsForOneCoop(params.coop);
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalVaRingKernel<<<1, numThreads, 0, streams[i]>>>(
      devComms[i], params.contextIdx, windows[i], params.offset, params.coop);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinVaSignal_test, reset) {
  const GinVaSignalParams& params = GetParam();

  int numThreads = getNumThreadsForOneCoop(params.coop);
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalVaResetKernel<<<1, numThreads, 0, streams[i]>>>(
      devComms[i], params.contextIdx, windows[i], params.offset, params.coop);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinVaSignal_test, basic_add) {
  const GinVaSignalParams& params = GetParam();

  int numThreads = getNumThreadsForOneCoop(params.coop);
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalVaBasicKernel<<<1, numThreads, 0, streams[i]>>>(
      devComms[i], params.contextIdx, true, windows[i], params.offset, params.coop);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinVaSignal_test, basic_inc) {
  const GinVaSignalParams& params = GetParam();
  int numThreads = getNumThreadsForOneCoop(params.coop);

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalVaBasicKernel<<<1, numThreads, 0, streams[i]>>>(
      devComms[i], params.contextIdx, false, windows[i], params.offset, params.coop);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

////////////////////////////////////////////////////////////////////////////////
// Test instantiation
////////////////////////////////////////////////////////////////////////////////

// Custom name generator for readable test names
std::string GinVaSignalParamsName(const ::testing::TestParamInfo<GinVaSignalParams>& info) {
  return info.param.toString();
}

INSTANTIATE_TEST_CASE_P(
  Test,
  GinVaSignal_test,
  ::testing::Values(
    GinVaSignalParams{0, 0, CoopLevel::Thread},                  // Context 0, Offset 0, Thread
    GinVaSignalParams{0, 0, CoopLevel::Warp},                    // Context 0, Offset 0, Warp
    GinVaSignalParams{0, 0, CoopLevel::Cta},                     // Context 0, Offset 0, Cta
    GinVaSignalParams{0, sizeof(uint64_t), CoopLevel::Cta},      // Context 0, Offset 8, Cta
    GinVaSignalParams{1, 0, CoopLevel::Warp}                     // Context 1, Offset 0, Warp
  ),
  GinVaSignalParamsName
);
