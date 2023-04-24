#include "ncclCommon_test.cuh"

class ncclCommDestroy_test : public ::testing::Test {
  public:
    int nVis;
    int expectMask;
    void SetUp() {
        register_segv_handler();
        ncclCommon_destroysrComms();
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0);
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

TEST_F(ncclCommDestroy_test, basic) {
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

    SUCCEED();
}

TEST_F(ncclCommDestroy_test, null) {
    ASSERT_EQ(ncclSuccess, ncclCommDestroy(NULL));
    SUCCEED();
}

TEST_F(ncclCommDestroy_test, group_destroy_nonblocking) {
    ncclComm_t* comms = NULL;
    int nVis;
    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.blocking = 0;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
    ASSERT_NE(nullptr, comms = (ncclComm_t*)calloc(sizeof(ncclComm_t), nVis));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void) ncclCommInitRankConfig(&comms[i], nVis, id, i, &config);
    }
    ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

    waitCommsReady(comms, nVis);

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    free(comms);

    SUCCEED();
}

TEST_F(ncclCommDestroy_test, group_destroy_blocking) {
    ncclComm_t* comms = NULL;
    int nVis;
    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
    ASSERT_NE(nullptr, comms = (ncclComm_t*)calloc(sizeof(ncclComm_t), nVis));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void) ncclCommInitRankConfig(&comms[i], nVis, id, i, &config);
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    free(comms);

    SUCCEED();
}
