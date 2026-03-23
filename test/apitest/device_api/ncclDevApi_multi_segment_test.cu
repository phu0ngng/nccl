/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*
 * API test: multi-segment allocations with 16 segments.
 * Runs LSA AllReduce using segmented buffers (16 segments,
 * alternating device / host NUMA)
 */

#include "nccl_device.h"
#include "ncclDevApiCommon_test.cuh"

#include <cstring>
#include <vector>

static const int numSegments = 16;
static const size_t segmentSize = 2 * 1024 * 1024;

static size_t getTotalSizeForConfig(const segment_descriptor_t* descriptors, int n) {
  size_t total = 0;
  for (int i = 0; i < n; i++) total += descriptors[i].segment_size;
  return total;
}

static void getConfig16Segments(segment_descriptor_t* out) {
  for (int i = 0; i < numSegments; i++) {
    out[i].location_type = (i % 2 == 0) ? SEGMENT_LOCATION_DEVICE : SEGMENT_LOCATION_HOST_NUMA;
    out[i].location_id = -1;
    out[i].segment_size = segmentSize;
  }
}

static void getConfig16SegmentsGpuOnly(segment_descriptor_t* out) {
  for (int i = 0; i < numSegments; i++) {
    out[i].location_type = SEGMENT_LOCATION_DEVICE;
    out[i].location_id = -1;
    out[i].segment_size = segmentSize;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Pure GIN AlltoAll kernel
////////////////////////////////////////////////////////////////////////////////

static const int ginCtaCount = 1;
static const int ginThreadsPerCta = 512;

template <typename T>
__global__ void PureGinAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset,
                                      ncclWindow_t recvwin, size_t recvoffset,
                                      size_t count, struct ncclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = 0;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  const size_t size = count * sizeof(T);
  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());
}

////////////////////////////////////////////////////////////////////////////////
// LSA AllReduce kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void lsaAllReduceKernel(ncclWindow_t sendwin, size_t sendoffset,
                                   ncclWindow_t recvwin, size_t recvoffset,
                                   size_t count, struct ncclDevComm devComm) {
#if __CUDA_ARCH__ >= 700
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm, ncclTeamLsa(devComm),
                                         devComm.lsaBarrier, blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);

  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    float v = 0;
    for (int peer = 0; peer < nRanks; peer++) {
      float* sendPtr = (float*)ncclGetLsaPointer(sendwin, sendoffset, peer);
      v += sendPtr[offset];
    }
    for (int peer = 0; peer < nRanks; peer++) {
      float* recvPtr = (float*)ncclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[offset] = v;
    }
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
#endif
}

static const int lsaCtaCount = 1;
static const int lsaThreadsPerCta = 256;
static const size_t allReduceCount = 8 * 1024 * 1024;

// Self-contained test class so that this does not affect other tests
class ncclDevApi_multi_segment_test : public ::testing::Test {
protected:
  int nVis = 0;
  ncclComm_t* comms = nullptr;
  cudaStream_t* streams = nullptr;
  std::vector<ncclDevComm> devComms;

  segment_descriptor_t descriptors[numSegments];

  size_t bufferSizeBytes = 0;
  std::vector<void*> sendPtrs;
  std::vector<void*> recvPtrs;
  std::vector<ncclWindow_t> sendWins;
  std::vector<ncclWindow_t> recvWins;

  void SetUp() override {
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
    comms = (ncclComm_t*)calloc(sizeof(ncclComm_t), nVis);
    ASSERT_NE(nullptr, comms);
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, nullptr));

    streams = (cudaStream_t*)calloc(nVis, sizeof(cudaStream_t));
    ASSERT_NE(nullptr, streams);
    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
    }
    cudaGetLastError();  // Clear any stale errors
  }

  void TearDown() override {
    syncAllDevices();
    destroyDevComms();

    if (comms) {
      for (int i = 0; i < nVis; i++) {
        if (comms[i]) ncclCommDestroy(comms[i]);
      }
      free(comms);
      comms = nullptr;
    }
    if (streams) {
      for (int i = 0; i < nVis; i++) {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
      }
      free(streams);
      streams = nullptr;
    }
  }

  void syncAllDevices() {
    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
    }
  }

  void destroyDevComms() {
    if (!devComms.empty()) {
      for (int i = 0; i < nVis; i++) {
        cudaSetDevice(i);
        if (comms && comms[i]) {
          ncclDevCommDestroy(comms[i], &devComms[i]);
        }
      }
      devComms.clear();
    }
  }

  TestResult_t createDevComms(const ncclDevCommRequirements& reqs) {
    if (devComms.size() != 0) return TestResult_t::testError;
    devComms.resize(nVis);

    for (int i = 0; i < nVis; i++) {
      ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
      ncclResult_t res = ncclCommQueryProperties(comms[i], &props);
      if (res != ncclSuccess) return TestResult_t::testError;
      if (!props.deviceApiSupport) return TestResult_t::testSkipped;
      bool ginRequested = reqs.ginForceEnable || reqs.ginConnectionType != NCCL_GIN_CONNECTION_NONE;
      if (ginRequested && props.ginType == NCCL_GIN_TYPE_NONE) return TestResult_t::testSkipped;
    }

    ncclResult_t res = ncclGroupStart();
    if (res != ncclSuccess) return TestResult_t::testError;
    for (int i = 0; i < nVis; i++) {
      cudaError_t cudaErr = cudaSetDevice(i);
      if (cudaErr != cudaSuccess) { ncclGroupEnd(); return TestResult_t::testError; }
      res = ncclDevCommCreate(comms[i], &reqs, &devComms[i]);
      if (res != ncclSuccess) { ncclGroupEnd(); return TestResult_t::testError; }
    }
    res = ncclGroupEnd();
    if (res != ncclSuccess) return TestResult_t::testError;

    cudaGetLastError();
    return TestResult_t::testSuccess;
  }

  void allocateAndRegisterSegmentedWindows() {
    getConfig16Segments(descriptors);
    bufferSizeBytes = getTotalSizeForConfig(descriptors, numSegments);

    sendPtrs.resize(nVis);
    recvPtrs.resize(nVis);
    sendWins.resize(nVis);
    recvWins.resize(nVis);

    ncclResult_t res = ncclGroupStart();
    ASSERT_EQ(ncclSuccess, res);

    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      allocateSegmentedMemory(&sendPtrs[i], descriptors, numSegments);
      allocateSegmentedMemory(&recvPtrs[i], descriptors, numSegments);

      ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], sendPtrs[i], bufferSizeBytes,
                                                     &sendWins[i], NCCL_WIN_COLL_SYMMETRIC));
      ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], recvPtrs[i], bufferSizeBytes,
                                                     &recvWins[i], NCCL_WIN_COLL_SYMMETRIC));
    }

    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  }

  void deregisterAndFreeSegmentedWindows() {
    for (int i = 0; i < nVis; i++) {
      if (sendWins[i] != nullptr) ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[i], sendWins[i]));
      if (recvWins[i] != nullptr) ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[i], recvWins[i]));
      if (sendPtrs[i] != nullptr) {
        deallocateSegmentedMemory(sendPtrs[i], descriptors, numSegments);
        sendPtrs[i] = nullptr;
      }
      if (recvPtrs[i] != nullptr) {
        deallocateSegmentedMemory(recvPtrs[i], descriptors, numSegments);
        recvPtrs[i] = nullptr;
      }
    }
  }

  void allocateAndRegisterGpuOnlySegmentedWindows() {
    getConfig16SegmentsGpuOnly(descriptors);
    bufferSizeBytes = getTotalSizeForConfig(descriptors, numSegments);

    sendPtrs.resize(nVis);
    recvPtrs.resize(nVis);
    sendWins.resize(nVis);
    recvWins.resize(nVis);

    ncclResult_t res = ncclGroupStart();
    ASSERT_EQ(ncclSuccess, res);

    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      allocateSegmentedMemory(&sendPtrs[i], descriptors, numSegments);
      allocateSegmentedMemory(&recvPtrs[i], descriptors, numSegments);

      ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], sendPtrs[i], bufferSizeBytes,
                                                     &sendWins[i], NCCL_WIN_COLL_SYMMETRIC));
      ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], recvPtrs[i], bufferSizeBytes,
                                                     &recvWins[i], NCCL_WIN_COLL_SYMMETRIC));
    }

    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  }

  void runGinAlltoAllAndVerify() {
    // Derive per-rank-pair count from buffer size so we span multiple segments
    // regardless of the number of ranks.
    const size_t alltoallCount = bufferSizeBytes / (nVis * sizeof(float));
    const size_t totalElements = alltoallCount * nVis;
    const size_t sizeBytes = totalElements * sizeof(float);
    ASSERT_LE(sizeBytes, bufferSizeBytes);

    std::vector<float> h_send(totalElements);
    std::vector<float> h_recv(totalElements);

    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));

      // Each rank sends unique values to each destination
      for (size_t j = 0; j < totalElements; j++) {
        int destRank = j / alltoallCount;
        int elemIdx = j % alltoallCount;
        h_send[j] = (float)(i * 1000 + destRank * 100 + elemIdx);
      }
      segmentedMemcpyToDevice(sendPtrs[i], h_send.data(), sizeBytes,
                              descriptors, numSegments);

      // Clear receive buffer
      ASSERT_EQ(cudaSuccess, cudaMemset(recvPtrs[i], 0, sizeBytes));

      PureGinAlltoAllKernel<float><<<ginCtaCount, ginThreadsPerCta, 0, streams[i]>>>(
          sendWins[i], 0, recvWins[i], 0, alltoallCount, devComms[i]);
    }
    syncAllDevices();

    // Verify: rank i should receive from rank src the chunk src sent to i
    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      segmentedMemcpyToHost(recvPtrs[i], h_recv.data(), sizeBytes,
                            descriptors, numSegments);
      for (int srcRank = 0; srcRank < nVis; srcRank++) {
        for (size_t e = 0; e < alltoallCount; e++) {
          size_t recvIdx = srcRank * alltoallCount + e;
          float expected = (float)(srcRank * 1000 + i * 100 + e);
          ASSERT_FLOAT_EQ(h_recv[recvIdx], expected)
              << "rank " << i << " from src " << srcRank << " index " << e;
        }
      }
    }
  }

  void runLsaAllReduceAndVerify() {
    const size_t sizeBytes = allReduceCount * sizeof(float);
    ASSERT_LE(sizeBytes, bufferSizeBytes);

    std::vector<float> h_send(allReduceCount);
    std::vector<float> h_recv(allReduceCount);

    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      for (size_t j = 0; j < allReduceCount; j++) h_send[j] = (float)i;
      segmentedMemcpyToDevice(sendPtrs[i], h_send.data(), sizeBytes,
                              descriptors, numSegments);

      lsaAllReduceKernel<<<lsaCtaCount, lsaThreadsPerCta, 0, streams[i]>>>(
          sendWins[i], 0, recvWins[i], 0, allReduceCount, devComms[i]);
    }
    syncAllDevices();

    float expected = (float)((nVis * (nVis - 1)) / 2);
    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      segmentedMemcpyToHost(recvPtrs[i], h_recv.data(), sizeBytes,
                            descriptors, numSegments);
      for (size_t j = 0; j < allReduceCount; j++) {
        ASSERT_FLOAT_EQ(h_recv[j], expected) << "rank " << i << " index " << j;
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Test cases: LSA with 16 segments interleaved between GPU and CPU
////////////////////////////////////////////////////////////////////////////////

TEST_F(ncclDevApi_multi_segment_test, LSA_allreduce_16_segments) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = lsaCtaCount;
  TESTCHECK(createDevComms(reqs));

  allocateAndRegisterSegmentedWindows();
  runLsaAllReduceAndVerify();
  deregisterAndFreeSegmentedWindows();
}

////////////////////////////////////////////////////////////////////////////////
// Test cases: GIN with 16 GPU-only segments
////////////////////////////////////////////////////////////////////////////////

TEST_F(ncclDevApi_multi_segment_test, GIN_alltoall_16_gpu_segments) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.worldGinBarrierCount = ginCtaCount;
  reqs.ginSignalCount = 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  TESTCHECK(createDevComms(reqs));

  allocateAndRegisterGpuOnlySegmentedWindows();
  runGinAlltoAllAndVerify();
  deregisterAndFreeSegmentedWindows();
}

////////////////////////////////////////////////////////////////////////////////
// Test cases: Test window register failure with interleaved segments + GIN
////////////////////////////////////////////////////////////////////////////////

TEST_F(ncclDevApi_multi_segment_test, sysmem_segments_register_returns_invalid_argument) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.worldGinBarrierCount = ginCtaCount;
  reqs.ginSignalCount = 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  TESTCHECK(createDevComms(reqs));

  getConfig16Segments(descriptors);
  bufferSizeBytes = getTotalSizeForConfig(descriptors, numSegments);

  sendPtrs.resize(nVis);
  sendWins.resize(nVis);

  ncclResult_t res = ncclGroupStart();
  ASSERT_EQ(ncclSuccess, res);

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    allocateSegmentedMemory(&sendPtrs[i], descriptors, numSegments);
    ASSERT_NE(nullptr, sendPtrs[i]);

    ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], sendPtrs[i], bufferSizeBytes,
                                                           &sendWins[i], NCCL_WIN_COLL_SYMMETRIC));
  }

  ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());

  for (int i = 0; i < nVis; i++) {
    deallocateSegmentedMemory(sendPtrs[i], descriptors, numSegments);
    sendPtrs[i] = nullptr;
  }
}
