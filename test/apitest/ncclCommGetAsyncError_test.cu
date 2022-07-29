#include "ncclCommon_test.cuh"

class ncclCommGetAsyncError_test : public ::testing::Test {
  public:
    ncclComm_t localComm;
    int nVis;
    int iteration = 1;
    void SetUp() {
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0);
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

TEST_F(ncclCommGetAsyncError_test, basic) {
    ncclComm_t* comms = NULL;
    ncclResult_t state;
    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.blocking = 0;
    for (int loop = 0; loop < iteration; ++loop) {
        ASSERT_NE(nullptr, comms = (ncclComm_t*)calloc(sizeof(ncclComm_t), nVis));
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
            (void) ncclCommInitRankConfig(&comms[i], nVis, id, i, &config);
        }
        ASSERT_EQ(ncclInProgress, ncclGroupEnd());

        waitCommsReady(comms, nVis);
        
        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(ncclSuccess, ncclCommGetAsyncError(comms[i], &state));
            ASSERT_EQ(ncclSuccess, state);
        }

        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i)
            (void) ncclCommFinalize(comms[i]);
        ASSERT_EQ(ncclInProgress, ncclGroupEnd());

        waitCommsReady(comms, nVis);

        for (int i = 0; i < nVis; ++i) {
            ASSERT_EQ(ncclSuccess, ncclCommGetAsyncError(comms[i], &state));
            ASSERT_EQ(ncclSuccess, state);
            ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
        }

        free(comms);
    }
}

TEST_F(ncclCommGetAsyncError_test, null_comm) {
    ncclResult_t state;
    ASSERT_EQ(ncclInvalidArgument, ncclCommGetAsyncError(NULL, &state));
}

TEST_F(ncclCommGetAsyncError_test, null_state) {
    ncclComm_t *comms;
    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.blocking = 0;
    ASSERT_NE(nullptr, comms = (ncclComm_t*)calloc(sizeof(ncclComm_t), nVis));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void) ncclCommInitRankConfig(&comms[i], nVis, id, i, &config);
    }
    ASSERT_EQ(ncclInProgress, ncclGroupEnd());

    waitCommsReady(comms, nVis);
    
    ASSERT_EQ(ncclInvalidArgument, ncclCommGetAsyncError(comms[0], NULL));
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }
    free(comms);
}
