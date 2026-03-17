#include "ncclCommon_test.cuh"
#include <pthread.h>
#include <vector>

class ncclCommRevoke_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = NULL;
    int nVis = 0;
    virtual void SetUp() {
        register_segv_handler();
        (void)setenv("NCCL_CHECK_POINTERS", "1", 0);
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
        comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    };
    virtual void TearDown() {
        if (NULL != comms) {
            for (int i = 0; i < nVis; ++i) {
                if (comms[i]) {
                    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
                    comms[i] = NULL;
                }
            }
            free(comms);
            comms = NULL;
        }
    };

    void AssertOpsReturnRevokedAfterRevoke(bool blocking) {
        if (nVis < 1) return;

        ncclUniqueId id;
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));

        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = blocking ? 1 : 0;

        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
            ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
        }
        {
            ncclResult_t groupEndRet = ncclGroupEnd();
            if (!blocking) {
                ASSERT_TRUE(groupEndRet == ncclSuccess || groupEndRet == ncclInProgress);
            } else {
                ASSERT_EQ(ncclSuccess, groupEndRet);
            }
        }

        if (!blocking) {
            waitCommsReady(comms, nVis);
        }

        // Revoke
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
            ncclResult_t r = ncclCommRevoke(comms[i], NCCL_REVOKE_DEFAULT);
            if (!blocking) {
                ASSERT_TRUE(r == ncclInProgress || r == ncclSuccess);
            } else {
                ASSERT_EQ(ncclSuccess, r);
            }
        }
        {
            ncclResult_t groupEndRet = ncclGroupEnd();
            if (!blocking) {
                ASSERT_TRUE(groupEndRet == ncclSuccess || groupEndRet == ncclInProgress);
            } else {
                ASSERT_EQ(ncclSuccess, groupEndRet);
            }
        }
        if (!blocking) {
            waitCommsReady(comms, nVis);
        }

        // Attempt an op and expect ncclInvalidUsage (revoked comm)
        std::vector<void*> send(nVis, nullptr), recv(nVis, nullptr);
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
            ASSERT_EQ(cudaSuccess, cudaMalloc(&send[i], sizeof(int)));
            ASSERT_EQ(cudaSuccess, cudaMalloc(&recv[i], sizeof(int)));
        }
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
            ncclResult_t opResult = ncclAllReduce(send[i], recv[i], 1, ncclInt32, ncclSum, comms[i], 0);
            ASSERT_EQ(ncclInvalidUsage, opResult);
        }
        ncclGroupEnd();
        for (int i = 0; i < nVis; ++i) {
            cudaFree(send[i]);
            cudaFree(recv[i]);
        }
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

    void CommonRevokeTest(bool blocking, ncclResult_t expected = ncclSuccess) {
        if (nVis < 1) return;

        ncclUniqueId id;
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = blocking ? 1 : 0;

        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
            ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
        }
        {
            ncclResult_t groupEndRet = ncclGroupEnd();
            if (!blocking) {
                ASSERT_TRUE(groupEndRet == ncclSuccess || groupEndRet == ncclInProgress);
            } else {
                ASSERT_EQ(ncclSuccess, groupEndRet);
            }
        }

        if (!blocking) {
            waitCommsReady(comms, nVis);
        }

        // Revoke
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
            ncclResult_t r = ncclCommRevoke(comms[i], NCCL_REVOKE_DEFAULT);
            if (blocking) {
                ASSERT_EQ(expected, r);
            }
        }
        {
            ncclResult_t groupEndRet = ncclGroupEnd();
            if (!blocking) {
                ASSERT_TRUE(groupEndRet == ncclSuccess || groupEndRet == ncclInProgress);
            } else {
                ASSERT_EQ(ncclSuccess, groupEndRet);
            }
        }

        // Ensure revoke completes before proceeding
        waitCommsReady(comms, nVis);
    }
};

TEST_F(ncclCommRevoke_test, blocking_default) {
    CommonRevokeTest(true);
}

TEST_F(ncclCommRevoke_test, nonblocking_default) {
    CommonRevokeTest(false);
}

TEST_F(ncclCommRevoke_test, null) {
    ASSERT_EQ(ncclSuccess, ncclCommRevoke(NULL, NCCL_REVOKE_DEFAULT));
}

TEST_F(ncclCommRevoke_test, reject_when_finalizing) {
    if (nVis < 1) return;

    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, 1, NULL));

    // Finalize sets finalizeCalled; revoke should then reject.
    ASSERT_EQ(ncclSuccess, ncclCommFinalize(comms[0]));
    ASSERT_EQ(ncclInvalidArgument, ncclCommRevoke(comms[0], NCCL_REVOKE_DEFAULT));
}

TEST_F(ncclCommRevoke_test, split_after_revoke_blocking) {
    if (nVis < 2) return;

    ncclUniqueId id;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 1;

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    // Revoke to quiesce communicator
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ncclResult_t stBefore;
        ASSERT_EQ(ncclSuccess, ncclCommGetAsyncError(comms[i], &stBefore));
        ASSERT_EQ(ncclSuccess, ncclCommRevoke(comms[i], NCCL_REVOKE_DEFAULT));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    // Ensure revoke completes before split
    waitCommsReady(comms, nVis);

    // Split into two groups (even/odd)
    std::vector<ncclComm_t> split(nVis, nullptr);
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        int color = i & 1;
        ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], color, i, &split[i], nullptr));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    // Cleanup split comms
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(split[i]));
    }
}

TEST_F(ncclCommRevoke_test, split_after_revoke_nonblocking) {
    if (nVis < 2) return;

    ncclUniqueId id;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    {
        ncclResult_t groupEndRet = ncclGroupEnd();
        ASSERT_TRUE(groupEndRet == ncclSuccess || groupEndRet == ncclInProgress);
    }
    waitCommsReady(comms, nVis);

    // Revoke asynchronously, then wait
    for (int i = 0; i < nVis; ++i) {
        (void)ncclCommRevoke(comms[i], NCCL_REVOKE_DEFAULT);
    }
    waitCommsReady(comms, nVis);

    // Split into two groups (even/odd)
    std::vector<ncclComm_t> split(nVis, nullptr);
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        int color = i & 1;
        ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], color, i, &split[i], nullptr));
    }
    {
        ncclResult_t groupEndRet = ncclGroupEnd();
        ASSERT_TRUE(groupEndRet == ncclSuccess || groupEndRet == ncclInProgress);
    }
    waitCommsReady(comms, nVis);

    // Cleanup split comms
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(split[i]));
    }
}


TEST_F(ncclCommRevoke_test, ops_return_revoked_after_blocking_revoke) {
    AssertOpsReturnRevokedAfterRevoke(true);
}

TEST_F(ncclCommRevoke_test, ops_return_revoked_after_nonblocking_revoke) {
    AssertOpsReturnRevokedAfterRevoke(false);
}
