#include "ncclCommon_test.cuh"
#include <vector>
#include <functional>

#define ASSERT_NCCL(call) \
  do { \
    ncclResult_t _ret = (call); \
    ASSERT_TRUE(_ret == ncclSuccess || _ret == ncclInProgress); \
  } while(0)

class ncclCommGrow_test : public ::testing::Test {
  protected:
  int nVis = 0;
  std::vector<ncclComm_t> allGrownComms;
  
  virtual void SetUp() {
    register_segv_handler();
    (void)setenv("NCCL_CHECK_POINTERS", "1", 0);
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
  }
  
  virtual void TearDown() {
    // Cleanup any grown comms
    for (auto comm : allGrownComms) {
      if (comm) ncclCommDestroy(comm);
    }
    allGrownComms.clear();
  }

  // Helper to wait for async communicator initialization/operations to complete
  void waitCommsReady(ncclComm_t *comms, int nranks) {
    int complete;
    ncclResult_t state;
    do {
      complete = 1;
      for (int i = 0; i < nranks; ++i) {
        if (comms[i]) {
          ASSERT_EQ(ncclSuccess, ncclCommGetAsyncError(comms[i], &state));
          if (state == ncclInProgress) {
            complete = 0;
            break;
          }
        }
      }
      usleep(10);
    } while(!complete);
    // Verify all completed successfully
    for (int i = 0; i < nranks; ++i) {
      if (comms[i]) {
        ASSERT_EQ(ncclSuccess, ncclCommGetAsyncError(comms[i], &state));
        ASSERT_EQ(ncclSuccess, state);
      }
    }
  }

  void initCommsWithConfig(std::vector<ncclComm_t>& comms, int nranks, bool blocking = true) {
    comms.resize(nranks, nullptr);
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = blocking ? 1 : 0;
    
    ncclUniqueId idInit;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&idInit));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nranks; ++i) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      ASSERT_EQ(ncclSuccess, 
                ncclCommInitRankConfig(&comms[i], nranks, idInit, i, &config));
    }
    ASSERT_NCCL(ncclGroupEnd());
    waitCommsReady(comms.data(), nranks);
  }

  // Helper to test AllReduce on communicators
  void testAllReduce(const std::vector<ncclComm_t>& comms) {
    int nranks = comms.size();
    std::vector<int*> sendbuf(nranks), recvbuf(nranks);
    std::vector<cudaStream_t> streams(nranks);
    
    // Allocate buffers and streams, initialize send data
    for (int i = 0; i < nranks; ++i) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      ASSERT_EQ(cudaSuccess, cudaMalloc(&sendbuf[i], sizeof(int)));
      ASSERT_EQ(cudaSuccess, cudaMalloc(&recvbuf[i], sizeof(int)));
      ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
      
      int val = i + 1;
      ASSERT_EQ(cudaSuccess, cudaMemcpy(sendbuf[i], &val, sizeof(int), cudaMemcpyHostToDevice));
    }
    
    // AllReduce
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nranks; ++i) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      ASSERT_NCCL(ncclAllReduce(sendbuf[i], recvbuf[i], 1, ncclInt32, ncclSum, comms[i], streams[i]));
    }
    ASSERT_NCCL(ncclGroupEnd());
    
    // Wait for AllReduce to complete
    waitCommsReady(const_cast<ncclComm_t*>(comms.data()), nranks);
    
    // Verify result
    int expected = (nranks * (nranks + 1)) / 2; // sum of 1..nranks
    for (int i = 0; i < nranks; ++i) {
      int result;
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
      ASSERT_EQ(cudaSuccess, cudaMemcpy(&result, recvbuf[i], sizeof(int), cudaMemcpyDeviceToHost));
      ASSERT_EQ(expected, result);
    }
    
    // Cleanup
    for (int i = 0; i < nranks; ++i) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      cudaFree(sendbuf[i]);
      cudaFree(recvbuf[i]);
      cudaStreamDestroy(streams[i]);
    }
  }

  void splitCommsWithConfig(
      std::vector<ncclComm_t>& comms,
      const std::function<int(int)>& colorFunc,
      const std::function<int(int)>& keyFunc,
      int expectedSize,
      std::vector<ncclComm_t>* allOut = nullptr) {
    int nranks = comms.size();
    
    std::vector<ncclComm_t> split(nranks, nullptr);
    
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nranks; ++i) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      ASSERT_NCCL(ncclCommSplit(comms[i], colorFunc(i), keyFunc(i), &split[i], NULL));
    }
    ASSERT_NCCL(ncclGroupEnd());
    waitCommsReady(comms.data(), nranks);
    
    // Verify split comm sizes
    for (int i = 0; i < nranks; ++i) {
      ASSERT_NE(nullptr, split[i]);
      int count;
      ASSERT_EQ(ncclSuccess, ncclCommCount(split[i], &count));
      ASSERT_EQ(expectedSize, count);
    }
    
    // Always destroy original comms (they're replaced by split ones)
    for (auto comm : comms) {
      ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
    }
    comms.clear();
    
    if (allOut) {
      // Caller takes ownership of split comms
      *allOut = std::move(split);
    } else {
      // Auto-cleanup: destroy all split comms
      for (int i = 0; i < nranks; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(split[i]));
      }
    }
  }

  void shrinkCommsWithConfig(
      std::vector<ncclComm_t>& comms, const std::vector<int>& excludeRanks,
      std::vector<ncclComm_t>* allOut = nullptr) {
    int oldSize = comms.size();
    int nExclude = excludeRanks.size();
    int newSize = oldSize - nExclude;
    
    std::vector<ncclComm_t> shrunk(newSize, nullptr);
    std::vector<char> excluded(oldSize, 0);
    for (int rank : excludeRanks) {
      ASSERT_GE(rank, 0);
      ASSERT_LT(rank, oldSize);
      excluded[rank] = 1;
    }
    
    std::vector<ncclComm_t> activeParents(newSize, nullptr);
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    int outIdx = 0;
    for (int i = 0; i < oldSize; ++i) {
      if (excluded[i]) continue;
      activeParents[outIdx] = comms[i];
      ASSERT_NCCL(ncclCommShrink(
          comms[i],
          const_cast<int*>(excludeRanks.data()),
          nExclude,
          &shrunk[outIdx], NULL, NCCL_SHRINK_DEFAULT));
      ++outIdx;
    }
    ASSERT_NCCL(ncclGroupEnd());
    waitCommsReady(activeParents.data(), newSize);
    
    // Verify shrunk comm sizes
    for (int i = 0; i < newSize; ++i) {
      ASSERT_NE(nullptr, shrunk[i]);
      int count;
      ASSERT_EQ(ncclSuccess, ncclCommCount(shrunk[i], &count));
      ASSERT_EQ(newSize, count);
    }
    
    // Always destroy original comms (they're replaced by shrunk ones)
    for (auto comm : comms) {
      ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
    }
    comms.clear();
    
    if (allOut) {
      // Caller takes ownership of shrunk comms
      *allOut = std::move(shrunk);
    } else {
      // Auto-cleanup: destroy all shrunk comms
      for (int i = 0; i < newSize; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(shrunk[i]));
      }
    }
  }

  void growCommsWithConfig(
      std::vector<ncclComm_t>& comms, int oldN, int total, 
      std::vector<ncclComm_t>* allOut = nullptr, bool blocking = true) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = blocking ? 1 : 0;
    
    ncclUniqueId growId;
    ASSERT_EQ(ncclSuccess, ncclCommGetUniqueId(comms[0], &growId));

    std::vector<ncclComm_t> all(total, nullptr);
    
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int idx = 0; idx < total; ++idx) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(idx));
      const bool isExisting = idx < oldN;
      // Rank 0 (existing) and all new ranks provide the growId
      const ncclUniqueId* uid = (idx == 0 || !isExisting) ? &growId : nullptr;
      ncclComm_t sourceComm = isExisting ? comms[idx] : nullptr;
      int rank = isExisting ? -1 : idx;
      
      ASSERT_NCCL(ncclCommGrow(sourceComm, total, uid, rank, 
                                &all[idx], &config));
    }
    
    ASSERT_NCCL(ncclGroupEnd());
    waitCommsReady(all.data(), total);
    
    // Verify all comms are created and have correct size
    for (int idx = 0; idx < total; ++idx) {
      ASSERT_NE(nullptr, all[idx]);
      int count;
      ASSERT_EQ(ncclSuccess, ncclCommCount(all[idx], &count));
      ASSERT_EQ(total, count);
    }
    
    // Always destroy old comms (they're replaced by grown ones)
    for (int idx = 0; idx < oldN; ++idx) {
      ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[idx]));
      comms[idx] = nullptr;
    }
    
    if (allOut) {
      // Caller takes ownership of grown comms
      *allOut = std::move(all);
    } else {
      // Auto-cleanup: destroy all grown comms
      for (int idx = 0; idx < total; ++idx) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(all[idx]));
      }
    }
  }
};

TEST_F(ncclCommGrow_test, basic_grow_single) {
  if (nVis < 3) return;

  std::vector<ncclComm_t> comms;
  initCommsWithConfig(comms, 2);
  growCommsWithConfig(comms, 2, 3);
}

TEST_F(ncclCommGrow_test, basic_grow_multi) {
  if (nVis < 4) return;
  int oldN = 2;

  std::vector<ncclComm_t> comms;
  initCommsWithConfig(comms, oldN, /*blocking=*/false);
  growCommsWithConfig(comms, oldN, nVis, nullptr, /*blocking=*/false);
}

// Multiple sequential grows (non-blocking mode)
TEST_F(ncclCommGrow_test, sequential_grows) {
  if (nVis < 4) return;
  
  // Start with 2 GPUs (non-blocking)
  std::vector<ncclComm_t> comms;
  initCommsWithConfig(comms, 2, /*blocking=*/false);
  
  // First grow: 2 -> 3 (non-blocking, take ownership for next grow)
  std::vector<ncclComm_t> comm1;
  growCommsWithConfig(comms, 2, 3, &comm1, /*blocking=*/false);
  
  // Second grow: 3 -> 4 (non-blocking, auto-cleaned by helper)
  growCommsWithConfig(comm1, 3, 4, nullptr, /*blocking=*/false);
}

// Test grow with collective operations
// Test grow + allreduce (non-blocking mode)
TEST_F(ncclCommGrow_test, allreduce_after_grow) {
  int oldN = nVis / 2;
  if (oldN < 2) return;
  
  std::vector<ncclComm_t> comms;
  initCommsWithConfig(comms, oldN, /*blocking=*/false);
  
  // Grow (non-blocking) and collect comms (helper verifies and destroys old comms)
  growCommsWithConfig(comms, oldN, nVis, &allGrownComms, /*blocking=*/false);
  
  // Test AllReduce on grown communicators
  testAllReduce(allGrownComms);
}

TEST_F(ncclCommGrow_test, invalid_operations) {
  if (nVis < 3) return;
  
  std::vector<ncclComm_t> comms;
  initCommsWithConfig(comms, 2);
  
  ncclUniqueId growId;
  ASSERT_EQ(ncclSuccess, ncclCommGetUniqueId(comms[0], &growId));
  
  // NULL newcomm should fail
  ASSERT_EQ(ncclInvalidArgument, ncclCommGrow(comms[0], 3, &growId, -1, NULL, NULL));
  
  // Invalid total ranks (total <= current) should fail
  ncclComm_t newcomm;
  ASSERT_EQ(ncclInvalidArgument, ncclCommGrow(comms[0], 1, &growId, -1, &newcomm, NULL));
  
  // Existing rank passing explicit rank instead of -1 should fail
  ASSERT_EQ(ncclInvalidArgument, ncclCommGrow(comms[0], 3, &growId, 0, &newcomm, NULL));
  
  // New rank passing NULL uniqueId should fail
  ASSERT_EQ(ncclInvalidArgument, ncclCommGrow(NULL, 3, NULL, 0, &newcomm, NULL));

  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
  }
}

// Nonblocking grow
TEST_F(ncclCommGrow_test, nonblocking_grow) {
  int oldN = nVis / 2;
  if (oldN < 2) return;
  
  // Init nonblocking and grow (auto-cleaned by helper)
  std::vector<ncclComm_t> comms;
  initCommsWithConfig(comms, oldN, /*blocking=*/false);
  growCommsWithConfig(comms, oldN, nVis, nullptr, /*blocking=*/false);
}

// Grow then split
TEST_F(ncclCommGrow_test, grow_then_split) {
  if (nVis < 4) return;
  
  int oldN = 2;
  int total = 4;
  
  // Init 2 GPUs and grow to 4 (helper verifies and destroys old comms)
  std::vector<ncclComm_t> comms;
  initCommsWithConfig(comms, oldN);
  
  std::vector<ncclComm_t> grownComms;
  growCommsWithConfig(comms, oldN, total, &grownComms);
  
  // Split grown comm into 2 groups (helper verifies and auto-destroys all comms)
  splitCommsWithConfig(
      grownComms,
      [](int i) { return i / 2; },    // color: two groups
      [](int i) { return i % 2; },    // key within each group
      2);
}

// Grow then shrink
TEST_F(ncclCommGrow_test, grow_then_shrink) {
  if (nVis < 3) return;
  
  // Init 2 GPUs and grow to 3 (helper verifies and destroys old comms)
  std::vector<ncclComm_t> comms;
  initCommsWithConfig(comms, 2);
  
  std::vector<ncclComm_t> grownComms;
  growCommsWithConfig(comms, 2, 3, &grownComms);
  
  // Shrink back to 2 (exclude rank 2, helper verifies and auto-destroys all comms)
  std::vector<int> excludeRanks = {2};
  shrinkCommsWithConfig(grownComms, excludeRanks);
}

