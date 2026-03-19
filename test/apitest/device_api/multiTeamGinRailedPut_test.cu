#include "nccl_device.h"
#include "ncclMultiTeamCommon_test.cuh"

constexpr uint64_t NCCL_PUT_VALUE_BASE = 28;  // atomic number of nickel
constexpr int SRC_RANK = 0;
constexpr int DST_RANK = 1;

__device__ uint64_t getPutValue(ncclDevComm comm) {
  int myRailTeam = ncclTeamLsa(comm).rank;
  return NCCL_PUT_VALUE_BASE + myRailTeam;
}

__global__ void putKernel(ncclDevComm comm, ncclWindow_t window, size_t offset) {
#if __CUDA_ARCH__ >= 700
  ncclTeam railTeam = ncclTeamRail(comm);
  ncclGin gin(comm, 0);
  ncclGinSignal_t signalIdx = 0;

  if (railTeam.nRanks < 2) return;

  uint64_t putValue = getPutValue(comm);
  if (railTeam.rank == SRC_RANK) {
    uint64_t* putSrcPtr = (uint64_t*)ncclGetLocalPointer(window, offset);
    *putSrcPtr = putValue;
    gin.put(railTeam, DST_RANK,
            window, offset,
            window, offset,
            sizeof(uint64_t), ncclGin_SignalInc{signalIdx});
  }

  if (railTeam.rank == DST_RANK) {
    volatile uint64_t* putPtr = (uint64_t*)ncclGetLocalPointer(window, offset);
    gin.waitSignal(ncclCoopThread(), signalIdx, 1);
    assert(*putPtr == putValue && "Put value mismatch");
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Test class
////////////////////////////////////////////////////////////////////////////////

class MultiTeamGinRailedPut_test : public ncclMultiTeamCommon_test {
protected:
  std::vector<void*> putBuffers;
  std::vector<ncclWindow_t> putWindows;
  static constexpr size_t bufferSize = sizeof(uint64_t);

  void SetUp() override {
    ncclMultiTeamCommon_test::SetUp();
    allocateAndRegisterWindows(nVis, ncclMultiTeamCommon_test::multiTeamComms, bufferSize, putBuffers, putWindows);
  }

  void TearDown() override {
    deregisterAndFreeWindows(nVis, ncclMultiTeamCommon_test::multiTeamComms, putBuffers, putWindows);
    ncclMultiTeamCommon_test::TearDown();
  }
};

////////////////////////////////////////////////////////////////////////////////
// Test case
////////////////////////////////////////////////////////////////////////////////

TEST_F(MultiTeamGinRailedPut_test, multi_railed_put_signal) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  reqs.ginSignalCount = 1;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    putKernel<<<1, 1, 0, streams[i]>>>(devComms[i], putWindows[i], 0);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}
