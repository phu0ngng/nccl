#include "nccl_device.h"
#include "ncclDevApiCommon_test.cuh"

__global__ void putKernel(ncclDevComm comm, ncclWindow_t window, size_t offset, bool useSignal, ncclWindow_t signalWindow, size_t signalOffset) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, 0);

  if (world.nRanks < 2) {
    return;
  }

  if (world.rank == 0) {
    uint64_t* putPtr = (uint64_t*)ncclGetLocalPointer(window, offset);
    *putPtr = 1;
    if (useSignal) {
      gin.put(world, 1, window, offset, window, offset, sizeof(uint64_t), ncclGin_VASignalInc{signalWindow, signalOffset});
    } else {
      gin.put(world, 1, window, offset, window, offset, sizeof(uint64_t));
    }
  }

  if (world.rank == 1) {
    uint64_t* putPtr = (uint64_t*)ncclGetLocalPointer(window, offset);
    if (useSignal) {
      gin.waitSignal(ncclCoopCta(), signalWindow, signalOffset, 1);
      KERNEL_ASSERT_EQ(*putPtr, 1, "Ptr should be 1 after putSignal");
    } else {
      while (*putPtr != 1) {
        continue;
      }
    }
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Test class
////////////////////////////////////////////////////////////////////////////////

class GinPut_test : public ncclDevApiCommon_test {
public:
protected:
  std::vector<void*> signalBuffers;
  std::vector<ncclWindow_t> signalWindows;
  std::vector<void*> putBuffers;
  std::vector<ncclWindow_t> putWindows;
  size_t bufferSize;

  void SetUp() override {
    ncclDevApiCommon_test::SetUp();
    // Allocate buffers: 8 bytes for signal (1 uint64_t), 8 bytes for put (1 uint64_t)
    bufferSize = sizeof(uint64_t);

    allocateAndRegisterWindows(nVis, comms, bufferSize, signalBuffers, signalWindows);
    allocateAndRegisterWindows(nVis, comms, bufferSize, putBuffers, putWindows);
  }

  void TearDown() override {
    deregisterAndFreeWindows(nVis, comms, signalBuffers, signalWindows);
    deregisterAndFreeWindows(nVis, comms, putBuffers, putWindows);
    ncclDevApiCommon_test::TearDown();
  }
};

////////////////////////////////////////////////////////////////////////////////
// Test cases
////////////////////////////////////////////////////////////////////////////////

TEST_F(GinPut_test, put_no_signal) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    putKernel<<<1, 1, 0, streams[i]>>>(devComms[i], putWindows[i], 0, false, signalWindows[i], 0);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(GinPut_test, put_signal) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    putKernel<<<1, 1, 0, streams[i]>>>(devComms[i], putWindows[i], 0, true, signalWindows[i], 0);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}