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

////////////////////////////////////////////////////////////////////////////////
// LSA AllReduce kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void lsaAllReduceKernel(ncclWindow_t sendwin, size_t sendoffset,
                                   ncclWindow_t recvwin, size_t recvoffset,
                                   size_t count, struct ncclDevComm devComm) {
#if __CUDA_ARCH__ >= 700
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm, ncclTeamLsa(devComm),
                                         devComm.lsaBarrier, blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

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

class ncclDevApi_multi_segment_test : public ncclDevApiCommon_test {
protected:
  segment_descriptor_t descriptors[numSegments];

  size_t bufferSizeBytes = 0;
  std::vector<void*> sendPtrs;
  std::vector<void*> recvPtrs;
  std::vector<ncclWindow_t> sendWins;
  std::vector<ncclWindow_t> recvWins;

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
// Test cases: LSA with 16 segments
////////////////////////////////////////////////////////////////////////////////

TEST_F(ncclDevApi_multi_segment_test, LSA_allreduce_16_segments) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = lsaCtaCount;
  TESTCHECK(createDevComms(reqs));

  allocateAndRegisterSegmentedWindows();
  runLsaAllReduceAndVerify();
  deregisterAndFreeSegmentedWindows();
}
