#include "ncclCommon_test.cuh"

class ncclConfig_test : public ::testing::Test {
  public:
    int nVis;
    int expectMask;
    void SetUp() {
        expectMask = (1 << ncclSuccess) | (1 << ncclInProgress);
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
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

TEST_F(ncclConfig_test, basic) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.blocking = 0;
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void) ncclCommInitRankConfig(&comms[i], nVis, id, i, &config);
    }
    ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

    waitCommsReady(comms, nVis);

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, blocking) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, config_null) {
    ncclUniqueId id;
    ncclComm_t* comms;

    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}
