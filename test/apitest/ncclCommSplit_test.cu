#include "ncclCommon_test.cuh"

class ncclCommSplit_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = NULL;
    ncclComm_t* comms2 = NULL;
    int nVis = 0;
    virtual void SetUp() {
        register_segv_handler();
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0); // API tests expect this behaviour (ncclCommSplit)
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
        comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
        comms2 = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    };
    virtual void TearDown() {
        if (NULL != comms) {
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
                comms[i] = NULL;
            }
            free(comms);
            comms = NULL;
        }
        if (NULL != comms2) {
            for (int i = 0; i < nVis; ++i) {
                if (comms2[i]) {
                    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms2[i]));
                    comms2[i] = NULL;
                }
            }
            free(comms);
            free(comms2);
            comms = NULL;
            comms2 = NULL;
        }
    };

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
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >=16)
TEST_F(ncclCommSplit_test, comm_dup) {
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i=0; i<nVis; i++) {
        ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], 0, i, &comms2[i], NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i=0; i<nVis; i++) ASSERT_NE((long)comms2[i], NULL);
}
TEST_F(ncclCommSplit_test, same_key) {
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i=0; i<nVis; i++) {
        ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], 0, 0, &comms2[i], NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i=0; i<nVis; i++) ASSERT_NE((long)comms2[i], NULL);
}
TEST_F(ncclCommSplit_test, half) {
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
    int split = nVis/2;
    if (split > 0) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i=0; i<nVis; i++) {
            ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], i/(split), i%split, &comms2[i], NULL));
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
        for (int i=0; i<nVis; i++) ASSERT_NE((long)comms2[i], NULL);
    }
}
TEST_F(ncclCommSplit_test, reverse) {
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i=0; i<nVis; i++) {
        ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], 0, nVis-1-i, &comms2[i], NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i=0; i<nVis; i++) ASSERT_NE((long)comms2[i], NULL);
}
TEST_F(ncclCommSplit_test, comm_partial) {
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i=0; i<nVis; i++) {
        ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], i<2 ? 0 : -1, i, &comms2[i], NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i=0; i<nVis; i++) {
        if (i<2) {
            ASSERT_NE((long)comms2[i], NULL);
        } else {
            ASSERT_EQ((long)comms2[i], NULL);
        }
    }
}

#define NUM_SLEEP_CASES 3
TEST_F(ncclCommSplit_test, abort) {
    ncclComm_t* localComms = NULL;
    ncclComm_t* childComms = NULL;
    ncclUniqueId id;
    int expectMask = (1 << ncclSuccess) | (1 << ncclInProgress);
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    int sleepTimes[NUM_SLEEP_CASES] = {10, 100, 1000}; /* sleep in us */

    config.blocking = 0;
    ASSERT_NE(nullptr, localComms = (ncclComm_t*)calloc(sizeof(ncclComm_t), nVis));
    ASSERT_NE(nullptr, childComms = (ncclComm_t*)calloc(sizeof(ncclComm_t), nVis));

    for (int s = 0; s < NUM_SLEEP_CASES + 1; s++) {
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
            (void) ncclCommInitRankConfig(&localComms[i], nVis, id, i, &config);
        }
        ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

        waitCommsReady(localComms, nVis);

        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(ncclSuccess, ncclCommSplit(localComms[i], 0, nVis - i, &childComms[i], NULL));
        }
        ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));
        
        if (s == NUM_SLEEP_CASES) {
            waitCommsReady(localComms, nVis);
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommDestroy(localComms[i]));
            }
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommDestroy(childComms[i]));
            }
        } else {
            usleep(sleepTimes[s]);
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommAbort(localComms[i]));
            }
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommAbort(childComms[i]));
            }
        }
    }
    
    free(localComms);
    free(childComms);
}

TEST_F(ncclCommSplit_test, abort_res_share_env) {
    ncclComm_t* localComms = NULL;
    ncclComm_t* childComms = NULL;
    ncclUniqueId id;
    int expectMask = (1 << ncclSuccess) | (1 << ncclInProgress);
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    int sleepTimes[NUM_SLEEP_CASES] = {10, 100, 1000}; /* sleep in us */

    config.blocking = 0;
    (void) setenv("NCCL_COMM_SPLIT_SHARE_RESOURCES", "1", 1);
    ASSERT_NE(nullptr, localComms = (ncclComm_t*)calloc(sizeof(ncclComm_t), nVis));
    ASSERT_NE(nullptr, childComms = (ncclComm_t*)calloc(sizeof(ncclComm_t), nVis));

    for (int s = 0; s < NUM_SLEEP_CASES + 1; s++) {
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
            (void) ncclCommInitRankConfig(&localComms[i], nVis, id, i, &config);
        }
        ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

        waitCommsReady(localComms, nVis);

        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(ncclSuccess, ncclCommSplit(localComms[i], 0, nVis - i, &childComms[i], NULL));
        }
        ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));
        
        if (s == NUM_SLEEP_CASES) {
            waitCommsReady(localComms, nVis);
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommDestroy(localComms[i]));
            }
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommDestroy(childComms[i]));
            }
        } else {
            usleep(sleepTimes[s]);
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommAbort(localComms[i]));
            }
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommAbort(childComms[i]));
            }
        }
    }
    
    free(localComms);
    free(childComms);
    (void) setenv("NCCL_COMM_SPLIT_SHARE_RESOURCES", "0", 1);
}

#endif
// EOF
