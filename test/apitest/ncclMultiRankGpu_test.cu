#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "ncclCommon_test.cuh"

class ncclMultiRankGpuTest : public ncclShelveEnvTest {
public:
  static void SetUpTestCase() {
    register_segv_handler();

    EXPECT_EQ(cudaSuccess, cudaGetDeviceCount(&nDev));
    nRanks = nDev * (nDev > 4 ? 2 : 3);
    bufCount = 4 * 1024;

    comms.resize(nRanks, nullptr);
    sendBuffs.resize(nRanks, nullptr);
    recvBuffs.resize(nRanks, nullptr);
    streams.resize(nRanks, nullptr);
    devByRank.resize(nRanks, -1);

    for (int i = 0; i < nRanks; i++) {
      devByRank[i] = (i * nDev) / nRanks;
    }
  }

  virtual void SetUp() override {
    ncclShelveEnvTest::SetUp();
    overrideEnvVariable("NCCL_CHECK_POINTERS", "1"); // API tests expect this behaviour (ncclCommInitAll)
    overrideEnvVariable("NCCL_MULTI_RANK_GPU_ENABLE", "1");

    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&oldDev));
  }

  virtual void TearDown() override {
    std::vector<bool> done(nRanks);
    int totalDone = 0;
    while (totalDone < nRanks) {
      // Make sure all the streams have finished.
      for (int i = 0; i < nRanks; i++) {
        if (done[i]) continue;
        EXPECT_EQ(cudaSuccess, cudaSetDevice(devByRank[i]));
        cudaError_t cudaErr = streams[i] ? cudaStreamQuery(streams[i]) : cudaSuccess;
        if (cudaErr != cudaErrorNotReady) {
          EXPECT_EQ(cudaSuccess, cudaErr)
                   << "Rank : " << i << " error: " << cudaGetErrorName(cudaErr)
                   << "(" << cudaGetErrorString(cudaErr) << ")" << std::endl;
          done[i] = true;
          totalDone++;
        }
      }
    }

    // Deallocate and clear resources on all ranks.
    for (int i = 0; i < nRanks; i++) {
      EXPECT_EQ(cudaSuccess, cudaSetDevice(devByRank[i]));
      if (sendBuffs[i]) {
        ASSERT_EQ(cudaSuccess, cudaFree(sendBuffs[i]));
        sendBuffs[i] = nullptr;
      }
      if (recvBuffs[i]) {
        ASSERT_EQ(cudaSuccess, cudaFree(recvBuffs[i]));
        recvBuffs[i] = nullptr;
      }
      if (streams[i]) {
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[i]));
        streams[i] = nullptr;
      }

      if (comms[i]) {
        ASSERT_EQ(ncclSuccess, ncclCommFinalize(comms[i]));
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
        comms[i] = nullptr;
      }
    }

    if (oldDev >= 0) cudaSetDevice(oldDev);
    oldDev = -1;
  }

  static void TearDownTestCase() {
    std::vector<int         >().swap(devByRank);
    std::vector<cudaStream_t>().swap(streams  );
    std::vector<void*       >().swap(recvBuffs);
    std::vector<void*       >().swap(sendBuffs);
    std::vector<ncclComm_t  >().swap(comms    );
  }
protected:
  static int nDev;
  static int nRanks;
  static size_t bufCount;
  static std::vector<ncclComm_t>   comms;
  static std::vector<void*>        sendBuffs;
  static std::vector<void*>        recvBuffs;
  static std::vector<cudaStream_t> streams;
  static std::vector<int>          devByRank;

  int oldDev = -1;
  ncclUniqueId commId;
};

int    ncclMultiRankGpuTest::nRanks   = 0;
int    ncclMultiRankGpuTest::nDev     = 0;
size_t ncclMultiRankGpuTest::bufCount = 0;
std::vector<ncclComm_t>   ncclMultiRankGpuTest::comms;
std::vector<void*>        ncclMultiRankGpuTest::sendBuffs;
std::vector<void*>        ncclMultiRankGpuTest::recvBuffs;
std::vector<cudaStream_t> ncclMultiRankGpuTest::streams;
std::vector<int>          ncclMultiRankGpuTest::devByRank;


TEST_F(ncclMultiRankGpuTest, initAll_Disallowed) {
  ParameterChanger changer("NCCL_MULTI_RANK_GPU_ENABLE", "0");
  ASSERT_EQ(ncclInvalidUsage, ncclCommInitAll(comms.data(), nRanks, devByRank.data()));
}

TEST_F(ncclMultiRankGpuTest, initAll) {
  ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms.data(), nRanks, devByRank.data()));
}

TEST_F(ncclMultiRankGpuTest, initRank) {
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nRanks; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(devByRank[i]));
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comms[i], nRanks, commId, i));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
}

TEST_F(ncclMultiRankGpuTest, initRankConfig) {
  ncclConfig_t gconfig = NCCL_CONFIG_INITIALIZER;
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nRanks; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(devByRank[i]));
    ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nRanks, commId, i, &gconfig));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
}

TEST_F(ncclMultiRankGpuTest, initRankScalable) {
  ncclConfig_t gconfig = NCCL_CONFIG_INITIALIZER;
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nRanks; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(devByRank[i]));
    ASSERT_EQ(ncclSuccess, ncclCommInitRankScalable(&comms[i], nRanks, i, 1, &commId, &gconfig));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
}

TEST_F(ncclMultiRankGpuTest, broadcast) {
  std::vector<int32_t> buf(bufCount);
  for (int i = 0; i < nRanks; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(devByRank[i])) << "rank " << i;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&sendBuffs[i], bufCount * sizeof(int32_t))) << "rank " << i;
    for (int j = 0; j < bufCount; ++j) {
      buf[j] = (i == 0) ? j : 0;
    }
    EXPECT_EQ(cudaSuccess, cudaMemcpy(sendBuffs[i], buf.data(), bufCount * sizeof(int32_t), cudaMemcpyHostToDevice)) << "rank " << i;
  }

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nRanks; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(devByRank[i])) << "rank " << i;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i])) << "rank " << i;
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comms[i], nRanks, commId, i)) << "rank " << i;
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nRanks; i++) {
    ASSERT_EQ(ncclSuccess, ncclBroadcast(sendBuffs[i], sendBuffs[i], bufCount, ncclInt32, 0, comms[i], streams[i])) << "rank " << i;
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  for (int i = 0; i < nRanks; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(devByRank[i])) << "rank " << i;
    EXPECT_EQ(cudaSuccess, cudaMemcpy(buf.data(), sendBuffs[i], bufCount * sizeof(int32_t), cudaMemcpyDeviceToHost)) << "rank " << i;
    for (int j = 0; j < bufCount; ++j) {
      ASSERT_EQ(j, buf[j]) << "Result of ncclBroadcast did not match for rank " << i;
    }
  }
}


TEST_F(ncclMultiRankGpuTest, allReduce) {
  std::vector<int32_t> buf(bufCount, 0);
  std::vector<int32_t> bufAccum(bufCount, 0);
  srand(6225);
  for (int i = 0; i < nRanks; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(devByRank[i])) << "rank " << i;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&sendBuffs[i], bufCount * sizeof(int32_t))) << "rank " << i;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&recvBuffs[i], bufCount * sizeof(int32_t))) << "rank " << i;
    for (int j = 0; j < bufCount; ++j) {
      buf[j] = rand() % 100;
      bufAccum[j] += buf[j];
    }
    EXPECT_EQ(cudaSuccess, cudaMemcpy(sendBuffs[i], buf.data(), bufCount * sizeof(int32_t), cudaMemcpyHostToDevice)) << "rank " << i;
  }

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nRanks; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(devByRank[i])) << "rank " << i;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i])) << "rank " << i;
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comms[i], nRanks, commId, i)) << "rank " << i;
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nRanks; i++) {
    ASSERT_EQ(ncclSuccess, ncclAllReduce(sendBuffs[i], recvBuffs[i], bufCount, ncclInt32, ncclSum, comms[i], streams[i])) << "rank " << i;
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  for (int i = 0; i < nRanks; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(devByRank[i])) << "rank " << i;
    EXPECT_EQ(cudaSuccess, cudaMemcpy(buf.data(), recvBuffs[i], bufCount * sizeof(int32_t), cudaMemcpyDeviceToHost)) << "rank " << i;
    for (int j = 0; j < bufCount; ++j) {
      ASSERT_EQ(bufAccum[j], buf[j]) << "Result of ncclAllReduce did not match for rank " << i;
    }
  }
}
