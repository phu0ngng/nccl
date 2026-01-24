// GIN Signal tests
#include "ncclDevApiCommon_test.cuh"
#include <cassert>

////////////////////////////////////////////////////////////////////////////////
// Test kernels
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

  uint64_t startValue = gin.readSignal(signalIdx);
  KERNEL_ASSERT_EQ(0, startValue, "Signal should be 0 before signal");

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

////////////////////////////////////////////////////////////////////////////////
// Test parameters
////////////////////////////////////////////////////////////////////////////////

struct GinSignalParams {
  int signalIdx;
  int contextIdx;
  
  std::string toString() const {
    return std::string("Signal") + std::to_string(signalIdx) + 
           "_Context" + std::to_string(contextIdx);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Test class
////////////////////////////////////////////////////////////////////////////////

class GinSignalTest : public ncclDevApiCommon_test,
                      public ::testing::WithParamInterface<GinSignalParams> {};

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

TEST_P(GinSignalTest, SignalRing) {
  const GinSignalParams& params = GetParam();
  
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = params.signalIdx + 1;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));
  
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalRingKernel<<<1, 1, 0, streams[i]>>>(devComms[i], params.contextIdx, params.signalIdx);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinSignalTest, SignalReset) {
  const GinSignalParams& params = GetParam();
  
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = params.signalIdx + 1;
  reqs.ginForceEnable = true;
  createDevComms(reqs);
  
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalResetKernel<<<1, 1, 0, streams[i]>>>(devComms[i], params.contextIdx, params.signalIdx);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinSignalTest, SignalBasicAdd) {
  const GinSignalParams& params = GetParam();
  
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = params.signalIdx + 1;
  reqs.ginForceEnable = true;
  createDevComms(reqs);

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalBasicKernel<<<1, 1, 0, streams[i]>>>(devComms[i], params.contextIdx, (ncclGinSignal_t)params.signalIdx, true);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinSignalTest, SignalBasicInc) {
  const GinSignalParams& params = GetParam();
  
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = params.signalIdx + 1;
  reqs.ginForceEnable = true;
  createDevComms(reqs);

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    signalBasicKernel<<<1, 1, 0, streams[i]>>>(devComms[i], params.contextIdx, (ncclGinSignal_t)params.signalIdx, false);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

////////////////////////////////////////////////////////////////////////////////
// Test instantiations
////////////////////////////////////////////////////////////////////////////////

INSTANTIATE_TEST_CASE_P(
  SignalAndContext,
  GinSignalTest,
  ::testing::Values(
    GinSignalParams{0, 0},  // Signal 0, Context 0
    GinSignalParams{0, 1},  // Signal 0, Context 1
    GinSignalParams{1, 0},  // Signal 1, Context 0
    GinSignalParams{1, 1}   // Signal 1, Context 1
  )
);