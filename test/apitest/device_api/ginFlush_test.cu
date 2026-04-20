#include <string>
#include "nccl_device.h"
#include "ncclDevApiCommon_test.cuh"

constexpr int LOCAL_RANK = 0;
constexpr int REMOTE_RANK = 1;

constexpr uint8_t NCCL_BYTE_VALUE = 28;

static __device__ bool verifyBytesPattern(ncclWindow_t window, size_t offset, size_t nBytes) {
  volatile uint8_t* p = (volatile uint8_t*)ncclGetLocalPointer(window, offset);
  for (size_t i = threadIdx.x; i < nBytes; i += blockDim.x) {
    uint8_t expected = NCCL_BYTE_VALUE + i;
    if (p[i] != expected) return false;
  }
  return true;
}

static __device__ void setBytesPattern(ncclWindow_t window, size_t offset, size_t nBytes) {
  uint8_t* p = (uint8_t*)ncclGetLocalPointer(window, offset);
  for (int i = threadIdx.x; i < nBytes; i += blockDim.x) {
    p[i] = NCCL_BYTE_VALUE + i;
  }
}

__global__ void flushAsyncWithSmemKernel(ncclDevComm comm, ncclWindow_t window, size_t offset, size_t getSize,
                                         ncclGinSignal_t signalIdx) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, 0);

  if (world.nRanks < 2) {
    return;
  }

  __shared__ ncclGinDescriptorSmem desc;

  if (world.rank == REMOTE_RANK) {
    setBytesPattern(window, offset, getSize);
    gin.signal(world, LOCAL_RANK, ncclGin_SignalInc{signalIdx}, ncclCoopCta());
  }

  if (world.rank == LOCAL_RANK) {
    gin.waitSignal(ncclCoopCta(), signalIdx, 1);
    ncclGinRequest_t ticket;
    gin.get(world, REMOTE_RANK, window, offset, window, offset, getSize, ncclCoopCta());
    gin.flushAsync(world, REMOTE_RANK, &ticket, ncclCoopCta(), ncclGinOptFlagsDefault, ncclGin_DescriptorSmem{&desc});
    gin.wait(ticket, ncclCoopCta());
    KERNEL_ASSERT_EQ(verifyBytesPattern(window, offset, getSize), true,
                     "window should have Bytes pattern after flushAsync-with-smem + wait");
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Test class
////////////////////////////////////////////////////////////////////////////////

class GinFlush_test : public ncclDevApiCommon_test {
public:
protected:
  std::vector<void*> buffers;
  std::vector<ncclWindow_t> windows;
  size_t bufferSize;

  void SetUp() override {
    ncclDevApiCommon_test::SetUp();
    bufferSize = 1ULL << 20;
    allocateAndRegisterWindows(nVis, comms, bufferSize, buffers, windows);
  }

  void TearDown() override {
    deregisterAndFreeWindows(nVis, comms, buffers, windows);
    ncclDevApiCommon_test::TearDown();
  }
};

////////////////////////////////////////////////////////////////////////////////
// Test cases
////////////////////////////////////////////////////////////////////////////////

TEST_F(GinFlush_test, flushAsync_with_smem) {
  size_t getSize = 1024;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginSignalCount = 1;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    flushAsyncWithSmemKernel<<<1, 512, 0, streams[i]>>>(devComms[i], windows[i], 0, getSize, /*signalIdx*/ 0);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}
