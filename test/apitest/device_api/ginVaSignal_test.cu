// GIN VA Signal tests
#include "ncclDevApiCommon_test.cuh"
#include <cassert>

////////////////////////////////////////////////////////////////////////////////
// VA signal dispatch helpers
////////////////////////////////////////////////////////////////////////////////

template <typename Coop>
__device__ void ginVaSignalInc(ncclGin& gin, Coop coop, int destRank, ncclWindow_t window, size_t offset, GinSignalType signalType) {
  switch (signalType) {
    case GinSignalType::Strong: gin.signal(coop, destRank, ncclGin_StrongVASignalInc{window, offset}); break;
    case GinSignalType::Weak:   gin.signal(coop, destRank, ncclGin_WeakVASignalInc{window, offset});   break;
    case GinSignalType::Legacy: gin.signal(coop, destRank, ncclGin_VASignalInc{window, offset});       break;
  }
}

template <typename Coop>
__device__ void ginVaSignalAdd(ncclGin& gin, Coop coop, int destRank, ncclWindow_t window, size_t offset, uint64_t value, GinSignalType signalType) {
  switch (signalType) {
    case GinSignalType::Strong: gin.signal(coop, destRank, ncclGin_StrongVASignalAdd{window, offset, value}); break;
    case GinSignalType::Weak:   gin.signal(coop, destRank, ncclGin_WeakVASignalAdd{window, offset, value});   break;
    case GinSignalType::Legacy: gin.signal(coop, destRank, ncclGin_VASignalAdd{window, offset, value});       break;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Test kernels for VA signals
////////////////////////////////////////////////////////////////////////////////

__global__ void signalVaRingKernel(ncclDevComm comm, int contextIdx,
                                    ncclWindow_t window, size_t offset, CoopLevel coopLevel, GinSignalType signalType) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, contextIdx);
  ncclCoopAny coop = getCoopFromLevel(coopLevel);
  KERNEL_ASSERT_EQ(coop.size(), ncclCoopCta().size(), "This test assumes exactly one coop is launched");

  int nextRank = (world.rank + 1) % world.nRanks;

  ginVaSignalInc(gin, world, nextRank, window, offset, signalType);

  // Wait for signal from previous rank
  waitSignal(gin, coopLevel, window, offset);
  // TODO: call waitSignal directly once https://nvbugspro.nvidia.com/bug/5870672 is fixed
  // gin.waitSignal(coop, window, offset, coop.size());
#endif
}

__global__ void signalVaResetKernel(ncclDevComm comm, int contextIdx,
                                     ncclWindow_t window, size_t offset, CoopLevel coopLevel, GinSignalType signalType) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, contextIdx);
  ncclCoopAny coop = getCoopFromLevel(coopLevel);
  KERNEL_ASSERT_EQ(coop.size(), ncclCoopCta().size(), "This test assumes exactly one coop is launched");

  ginVaSignalInc(gin, world, (world.rank + 1) % world.nRanks, window, offset, signalType);

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
                                     ncclWindow_t window, size_t offset, CoopLevel coopLevel, GinSignalType signalType) {
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
      ginVaSignalAdd(gin, world, 1, window, offset, 1, signalType);
    } else {
      ginVaSignalInc(gin, world, 1, window, offset, signalType);
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
  GinSignalType signalType;

  std::string toString() const {
    std::string coopStr = (coop == CoopLevel::Thread ? "thread" :
                           coop == CoopLevel::Warp ? "warp" : "cta");
    const char* typeName = (signalType == GinSignalType::Strong) ? "strong"
                         : (signalType == GinSignalType::Weak)   ? "weak"
                                                                  : "legacy";
    return std::string("context_") + std::to_string(contextIdx) +
           "_offset_" + std::to_string(offset) +
           "_coop_" + coopStr +
           "_" + typeName;
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
      devComms[i], params.contextIdx, windows[i], params.offset, params.coop, params.signalType);
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
      devComms[i], params.contextIdx, windows[i], params.offset, params.coop, params.signalType);
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
      devComms[i], params.contextIdx, true, windows[i], params.offset, params.coop, params.signalType);
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
      devComms[i], params.contextIdx, false, windows[i], params.offset, params.coop, params.signalType);
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
    GinVaSignalParams{0, 0, CoopLevel::Thread, GinSignalType::Strong},
    GinVaSignalParams{0, 0, CoopLevel::Cta,    GinSignalType::Legacy},
    GinVaSignalParams{0, 0, CoopLevel::Warp,   GinSignalType::Legacy},
    GinVaSignalParams{0, sizeof(uint64_t), CoopLevel::Cta, GinSignalType::Strong}
  ),
  GinVaSignalParamsName
);

////////////////////////////////////////////////////////////////////////////////
// Non-parameterized weak VA signal tests
////////////////////////////////////////////////////////////////////////////////

class GinVaSignalMisc_test : public ncclDevApiCommon_test {
protected:
  std::vector<void*> signalBuffers;
  std::vector<ncclWindow_t> windows;

public:
  void SetUp() override {
    ncclDevApiCommon_test::SetUp();
    allocateAndRegisterWindows(nVis, comms, 2 * sizeof(uint64_t), signalBuffers, windows);
  }

  void TearDown() override {
    deregisterAndFreeWindows(nVis, comms, signalBuffers, windows);
    ncclDevApiCommon_test::TearDown();
  }
};

TEST_F(GinVaSignalMisc_test, weak_basic_add) {
  int numThreads = getNumThreadsForOneCoop(CoopLevel::Cta);
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalVaBasicKernel<<<1, numThreads, 0, streams[i]>>>(
      devComms[i], 0, true, windows[i], 0, CoopLevel::Cta, GinSignalType::Weak);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(GinVaSignalMisc_test, weak_basic_inc) {
  int numThreads = getNumThreadsForOneCoop(CoopLevel::Cta);
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalVaBasicKernel<<<1, numThreads, 0, streams[i]>>>(
      devComms[i], 0, false, windows[i], 0, CoopLevel::Cta, GinSignalType::Weak);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}
