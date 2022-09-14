#include "ncclCommon_test.cuh"

class ncclCommInitRankConfig_test : public ::testing::Test {
  protected:
    ncclComm_t *comms;
    int ndev;
    ncclUniqueId commId;
    const int rank0 = 0;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    int expectMask;

    virtual void SetUp() {
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0);
        expectMask = (1 << ncclSuccess) | (1 << ncclInProgress);
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
        EXPECT_EQ(cudaSuccess, cudaGetDeviceCount(&ndev));
        EXPECT_NE(nullptr, comms = (ncclComm_t*) calloc(ndev, sizeof(ncclComm_t)));
        config.blocking = 0;
    }

    virtual void TearDown() {
        for (int i = 0; i < ndev; ++i) {
            if (comms[i]) {
                ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
            }
        }
        free(comms);
    }

    void waitCommsReady(ncclComm_t *comms, int nranks) {
        int complete;
        ncclResult_t state;
        do {
            complete = 1;
            for (int i = 0; i < nranks; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommGetAsyncError(comms[i], &state));
                if (state == ncclInProgress) {
                    complete = 0;
                    break;
                }
            }
            usleep(10);
        } while(!complete);
    }
};

TEST_F(ncclCommInitRankConfig_test, basic) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void) ncclCommInitRankConfig(&comms[i], ndev, commId, i, &config);
    }
    ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));
    waitCommsReady(comms, ndev);
}

TEST_F(ncclCommInitRankConfig_test, basic_null) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, commId, i, NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
}

TEST_F(ncclCommInitRankConfig_test, attr_null) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[0], 1, commId, rank0, NULL));
}

TEST_F(ncclCommInitRankConfig_test, with_config) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_NE(0, expectMask & (1 << ncclCommInitRankConfig(&comms[0], 1, commId, rank0, &config)));
    waitCommsReady(comms, 1);
}

TEST_F(ncclCommInitRankConfig_test, comm_null) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(NULL, ndev, commId, rank0, &config));
}

TEST_F(ncclCommInitRankConfig_test, ndev_zero) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comms[0], 0, commId, rank0, &config));
}

TEST_F(ncclCommInitRankConfig_test, dev_negative) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comms[0], -1, commId, rank0, &config));
}

TEST_F(ncclCommInitRankConfig_test, rank_outofboundary) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comms[0], 1, commId, ndev, &config));
}

TEST_F(ncclCommInitRankConfig_test, rank_negative) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comms[0], ndev, commId, -1, &config));
}
