#include <string>
#include "nccl_device.h"
#include "ncclDevApiCommon_test.cuh"

constexpr int LOCAL_RANK = 0;
constexpr int REMOTE_RANK = 1;

constexpr uint8_t NCCL_BYTE_VALUE = 28; // atomic number of nickel

__device__ bool verifyBytesPattern(ncclWindow_t window, size_t offset, size_t nBytes) {
  volatile uint8_t* p = (volatile uint8_t*)ncclGetLocalPointer(window, offset);
  for (size_t i = threadIdx.x; i < nBytes; i += blockDim.x) {
    uint8_t expected = NCCL_BYTE_VALUE + i;
    if (p[i] != expected) return false;
  }
  return true;
}

__device__ void setBytesPattern(ncclWindow_t window, size_t offset, size_t nBytes) {
  uint8_t* p = (uint8_t*)ncclGetLocalPointer(window, offset);
  for (int i = threadIdx.x; i < nBytes; i += blockDim.x) {
    p[i] = NCCL_BYTE_VALUE + i;
  }
}

__global__ void getKernel(ncclDevComm comm, ncclWindow_t window, size_t offset, size_t getSize,
                          ncclGinSignal_t signalIdx, bool useFlush) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, 0);

  if (world.nRanks < 2) {
    return;
  }

  if (world.rank == REMOTE_RANK) {
    setBytesPattern(window, offset, getSize);
    gin.signal(world, LOCAL_RANK, ncclGin_SignalInc{signalIdx}, ncclCoopCta()); // signal to local rank that the value is ready
  }

  if (world.rank == LOCAL_RANK) {
    gin.waitSignal(ncclCoopCta(), signalIdx, 1);
    if (useFlush) {
      gin.get(world, REMOTE_RANK, window, offset, window, offset, getSize, ncclCoopCta());
      gin.flush(ncclCoopCta());
      KERNEL_ASSERT_EQ(verifyBytesPattern(window, offset, getSize), true, "getPtr should have Bytes pattern after get");
    } else {
      if (ncclCoopCta().thread_rank() == 0) {
        gin.get(world, REMOTE_RANK, window, offset, window, offset, getSize);
      }
      while (!verifyBytesPattern(window, offset, getSize)) {
        continue;
      }
    }
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Test class
////////////////////////////////////////////////////////////////////////////////

class GinGet_test : public ncclDevApiCommon_test, public ::testing::WithParamInterface<size_t> {
public:
protected:
  std::vector<void*> getBuffers;
  std::vector<ncclWindow_t> getWindows;
  size_t bufferSize;

  void SetUp() override {
    ncclDevApiCommon_test::SetUp();
    bufferSize = 2ULL << 30;
    allocateAndRegisterWindows(nVis, comms, bufferSize, getBuffers, getWindows);
  }

  void TearDown() override {
    deregisterAndFreeWindows(nVis, comms, getBuffers, getWindows);
    ncclDevApiCommon_test::TearDown();
  }
};

////////////////////////////////////////////////////////////////////////////////
// Test cases
////////////////////////////////////////////////////////////////////////////////

TEST_P(GinGet_test, get) {
  size_t getSize = GetParam();
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginSignalCount = 1;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    getKernel<<<1, 512, 0, streams[i]>>>(devComms[i], getWindows[i], 0, getSize, /*signalIdx*/ 0, /*useFlush*/ false);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_P(GinGet_test, get_with_flush) {
  size_t getSize = GetParam();
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginSignalCount = 1;
  TESTCHECK(createDevComms(reqs));
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    getKernel<<<1, 512, 0, streams[i]>>>(devComms[i], getWindows[i], 0, getSize, /*signalIdx*/ 0, /*useFlush*/ true);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

std::string GinGetParamsName(const ::testing::TestParamInfo<size_t>& info) {
  if (info.param == 8) return std::string("8B");
  if (info.param == 1024) return std::string("1kB");
  if (info.param == (1ULL << 30)) return std::string("1GB");
  if (info.param == (2ULL << 30)) return std::string("2GB");
  return std::to_string(info.param);
}

INSTANTIATE_TEST_CASE_P(Test, GinGet_test,
                        ::testing::Values(8, 1024, 1ULL << 30, 2ULL << 30),
                        GinGetParamsName);
