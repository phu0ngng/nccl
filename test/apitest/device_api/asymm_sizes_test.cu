#include "nccl_device.h"
#include "ncclDevApiCommon_test.cuh"

#include <stdint.h>

namespace {

constexpr uint64_t kPutValue = 28; // atomic number of nickel
constexpr int kSrcRank = 0;
constexpr int kDstRank = 1;
constexpr size_t kSrcOffset = 0;
constexpr size_t kDstOffset = kSrcOffset + sizeof(uint64_t);
constexpr size_t kSrcWindowSize = sizeof(uint64_t);
constexpr size_t kDstWindowSize = 2 * sizeof(uint64_t);

__global__ void putAsymmetricSizesKernel(ncclDevComm comm, ncclWindow_t putWindow,
                                         bool useSignal, ncclWindow_t signalWindow,
                                         size_t signalOffset) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, 0);

  if (world.nRanks < 2) {
    return;
  }

  if (world.rank == kSrcRank) {
    uint64_t* putSrcPtr = (uint64_t*)ncclGetLocalPointer(putWindow, kSrcOffset);

    // Rank 1 should be 0 as well. We do not explicitly check it due to a race condition with the put.
    KERNEL_ASSERT_EQ(*putSrcPtr, 0, "putSrcPtr should be 0 before put");

    *putSrcPtr = kPutValue;
    if (useSignal) {
      gin.put(world, kDstRank, putWindow, kDstOffset, putWindow, kSrcOffset,
              sizeof(kPutValue), ncclGin_VASignalInc{signalWindow, signalOffset});
    } else {
      gin.put(world, kDstRank, putWindow, kDstOffset, putWindow, kSrcOffset,
              sizeof(kPutValue));
    }
  }

  if (world.rank == kDstRank) {
    volatile uint64_t* putDstPtr = (uint64_t*)ncclGetLocalPointer(putWindow, kDstOffset);
    if (useSignal) {
      gin.waitSignal(ncclCoopCta(), signalWindow, signalOffset, 1);
      KERNEL_ASSERT_EQ(*putDstPtr, kPutValue, "Ptr should be kPutValue after putSignal");
    } else {
      while (*putDstPtr != kPutValue) {
        continue;
      }
    }
  }
#endif
}

__global__ void lsaStoreAsymmetricSizesKernel(ncclDevComm comm, ncclWindow_t putWindow) {
    ncclTeam lsa = ncclTeamLsa(comm);
    if (lsa.nRanks < 2) {
      return;
    }

#if __CUDA_ARCH__ >= 700
  if (comm.rank == kSrcRank) {
    uint64_t* putSrcPtr = (uint64_t*)ncclGetLocalPointer(putWindow, kSrcOffset);

    KERNEL_ASSERT_EQ(*putSrcPtr, 0, "putSrcPtr should be 0 before LSA store");

    *putSrcPtr = kPutValue;
    volatile uint64_t* putDstPtr =
        (volatile uint64_t*)ncclGetLsaPointer(putWindow, kDstOffset, kDstRank);
    *putDstPtr = *putSrcPtr;
    __threadfence_system();
  }

  if (comm.rank == kDstRank) {
    volatile uint64_t* putDstPtr =
        (volatile uint64_t*)ncclGetLocalPointer(putWindow, kDstOffset);
    while (*putDstPtr != kPutValue) {
      continue;
    }
    KERNEL_ASSERT_EQ(*putDstPtr, kPutValue, "Ptr should be kPutValue after LSA store");
  }
#endif
}

}  // namespace

class AsymmetricSizes_test : public ncclDevApiCommon_test {
protected:
  std::vector<void*> signalBuffers;
  std::vector<ncclWindow_t> signalWindows;
  std::vector<void*> putBuffers;
  std::vector<ncclWindow_t> putWindows;

  void SetUp() override {
    ncclDevApiCommon_test::SetUp();

    std::vector<size_t> putBufferSizes(nVis, kSrcWindowSize);
    if (nVis > kDstRank) {
      putBufferSizes[kDstRank] = kDstWindowSize;
    }

    allocateAndRegisterWindows(nVis, comms, sizeof(uint64_t), signalBuffers, signalWindows);
    allocateAndRegisterWindows(nVis, comms, putBufferSizes, putBuffers, putWindows);
  }

  void TearDown() override {
    deregisterAndFreeWindows(nVis, comms, signalBuffers, signalWindows);
    deregisterAndFreeWindows(nVis, comms, putBuffers, putWindows);
    ncclDevApiCommon_test::TearDown();
  }

  void runGinPutAsymmetricSizesTest(bool useSignal) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    TESTCHECK(createDevComms(reqs));

    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      putAsymmetricSizesKernel<<<1, 1, 0, streams[i]>>>(devComms[i], putWindows[i],
                                                        useSignal, signalWindows[i], 0);
    }
    syncAllDevices();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  }

  void runLsaStoreAsymmetricSizesTest() {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    TESTCHECK(createDevComms(reqs));

    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      lsaStoreAsymmetricSizesKernel<<<1, 1, 0, streams[i]>>>(devComms[i], putWindows[i]);
    }
    syncAllDevices();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);
  }
};

TEST_F(AsymmetricSizes_test, gin_put_asymmetric_sizes_no_signal) {
  runGinPutAsymmetricSizesTest(false);
}

TEST_F(AsymmetricSizes_test, gin_put_asymmetric_sizes_signal) {
  runGinPutAsymmetricSizesTest(true);
}

TEST_F(AsymmetricSizes_test, lsa_store_asymmetric_sizes) {
  runLsaStoreAsymmetricSizesTest();
}
