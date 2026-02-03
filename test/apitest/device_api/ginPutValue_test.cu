#include "nccl_device.h"
#include "ncclDevApiCommon_test.cuh"

const uint64_t NCCL_PUT_VALUE = 28; // atomic number of nickel
const int SRC_RANK = 0;
const int DST_RANK = 1;

__global__ void putValueKernel(ncclDevComm comm, ncclWindow_t window, size_t offset, bool useSignal, ncclWindow_t signalWindow, size_t signalOffset) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, 0);

  if (world.nRanks < 2) {
    return;
  }

  if (world.rank == SRC_RANK) {
    if (useSignal) {
      gin.putValue(world, DST_RANK, window, offset, NCCL_PUT_VALUE, ncclGin_VASignalInc{signalWindow, signalOffset});
    } else {
      gin.putValue(world, DST_RANK, window, offset, NCCL_PUT_VALUE);
    }
  }

  if (world.rank == DST_RANK) {
    uint64_t* putPtr = (uint64_t*)ncclGetLocalPointer(window, offset);

    // Check omitted due to a race condition with the putValue.
    // The buffer is guaranteed to be initialized to 0 by the test framework.
    // KERNEL_ASSERT_EQ(*putPtr, 0, "putPtr should be 0 before putValue");

    if (useSignal) {
      gin.waitSignal(ncclCoopCta(), signalWindow, signalOffset, 1);
      KERNEL_ASSERT_EQ(*putPtr, NCCL_PUT_VALUE, "Ptr should be NCCL_PUT_VALUE after putSignal");
    } else {
      while (*putPtr != NCCL_PUT_VALUE) {
        continue;
      }
    }
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Test class
////////////////////////////////////////////////////////////////////////////////

class GinPutValue_test : public ncclDevApiCommon_test {
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

TEST_F(GinPutValue_test, put_value_no_signal) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    putValueKernel<<<1, 1, 0, streams[i]>>>(devComms[i], putWindows[i], 0, false, signalWindows[i], 0);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}

TEST_F(GinPutValue_test, put_value_signal) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = true;
  TESTCHECK(createDevComms(reqs));

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    putValueKernel<<<1, 1, 0, streams[i]>>>(devComms[i], putWindows[i], 0, true, signalWindows[i], 0);
  }
  syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
}