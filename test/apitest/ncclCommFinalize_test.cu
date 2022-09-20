#include "ncclCommon_test.cuh"

struct threadArgs {
    int rank;
    int nVis;
    int iteration;
    volatile int* retPtr;
    volatile ncclUniqueId* gidPtr;
    pthread_barrier_t* barrierPtr;
    volatile int* abortFlagPtr;
};

class ncclCommFinalize_test : public ::testing::Test {
  protected:
    int nVis;
    int iteration;
    int expectMask;
    virtual void SetUp() {
        ncclCommon_destroysrComms();
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0);
        iteration = 1;
        expectMask = (1 << ncclSuccess) | (1 << ncclInProgress);
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
    }
    virtual void TearDown() {}
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

static void oneGPUPerThreadExe(int rank, int nVis, int iteration, volatile ncclUniqueId* gidPtr, 
                               pthread_barrier_t* barrierPtr, volatile int* abortFlagPtr) {
    ncclUniqueId lid;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    int expectMask = (1 << ncclSuccess) | (1 << ncclInProgress);
    config.blocking = 0;
    ASSERT_EQ(cudaSuccess, cudaSetDevice(rank));

    for (int loop = 0; loop < iteration; ++loop) {
        ncclComm_t comm;
        if (rank == 0) {
            if (ncclGetUniqueId(&lid) != ncclSuccess) {
                *abortFlagPtr = 1;
                pthread_barrier_wait(barrierPtr);
                GTEST_FAIL();
            }
            memcpy((void*) gidPtr, (const void*) &lid, sizeof(ncclUniqueId));
            pthread_barrier_wait(barrierPtr);
        } else {
            pthread_barrier_wait(barrierPtr);
            ASSERT_EQ(*abortFlagPtr, 0);
            memcpy((void*) &lid, (const void*) gidPtr, sizeof(ncclUniqueId));
        }

        ASSERT_NE(0, expectMask & (1 << ncclCommInitRankConfig(&comm, nVis, lid, rank, &config)));
        waitCommsReady(&comm, 1);

        ASSERT_NE(0, expectMask & (1 << ncclCommFinalize(comm)));
        waitCommsReady(&comm, 1);

        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
    }
}

static void* oneGPUPerThreadTest(void* args) {
    struct threadArgs* thdArgs = (struct threadArgs*)args;

    oneGPUPerThreadExe(thdArgs->rank, thdArgs->nVis, thdArgs->iteration, thdArgs->gidPtr, thdArgs->barrierPtr, thdArgs->abortFlagPtr);
    if (::testing::Test::HasFatalFailure())
        *thdArgs->retPtr = 1;

    return NULL;
}

TEST_F(ncclCommFinalize_test, user_finalize_wait) {
    ncclComm_t* comms = NULL;
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
        ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));
        
        waitCommsReady(comms, nVis);

        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i)
            (void) ncclCommFinalize(comms[i]);
        ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

        waitCommsReady(comms, nVis);

        for (int i = 0; i < nVis; ++i)
            ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));

        free(comms);
    }
    
    SUCCEED();
}

TEST_F(ncclCommFinalize_test, user_finalize) {
    ncclComm_t* comms = NULL;
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
        ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

        waitCommsReady(comms, nVis);
        
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; ++i)
            (void) ncclCommFinalize(comms[i]);
        ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

        waitCommsReady(comms, nVis);

        for (int i = 0; i < nVis; ++i)
            ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));

        free(comms);
    }

    SUCCEED();
}

TEST_F(ncclCommFinalize_test, user_no_finalize) {
    ncclComm_t* comms = NULL;
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
        ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

        waitCommsReady(comms, nVis);

        for (int i = 0; i < nVis; ++i)
            ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));

        free(comms);
    }
    
    SUCCEED();
}

TEST_F(ncclCommFinalize_test, null_comm) {
    ASSERT_EQ(ncclSuccess, ncclCommFinalize(NULL));
    SUCCEED();
}

TEST_F(ncclCommFinalize_test, one_gpu_per_thread) {
    struct threadArgs* args;
    pthread_t* threads;
    volatile ncclUniqueId gid;
    volatile int ret, abortFlag;
    pthread_barrier_t barrier;
    
    ret = 0;
    abortFlag = 0;
    pthread_barrier_init(&barrier, NULL, nVis);
    threads = (pthread_t*) malloc(sizeof(pthread_t) * nVis);
    args = (struct threadArgs*) malloc(sizeof(struct threadArgs) * nVis);
    for (int i = 0; i < nVis; ++i) {
        args[i].rank = i;
        args[i].nVis = nVis;
        args[i].iteration = iteration;
        args[i].retPtr = &ret;
        args[i].gidPtr = &gid;
        args[i].barrierPtr = &barrier;
        args[i].abortFlagPtr = &abortFlag;
        ASSERT_EQ(0, pthread_create(&threads[i], NULL, oneGPUPerThreadTest, (void*) &args[i]));
    }

    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(0, pthread_join(threads[i], NULL));
    }

    ASSERT_EQ(0, ret);

    pthread_barrier_destroy(&barrier);
    free(args);
    free(threads);

    SUCCEED();
}

TEST_F(ncclCommFinalize_test, group_finalize) {
    ncclComm_t* comms = NULL;
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
    ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

    waitCommsReady(comms, nVis);

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i)
        (void) ncclCommFinalize(comms[i]);
    ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

    waitCommsReady(comms, nVis);

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));

    free(comms);
    SUCCEED();
}

TEST_F(ncclCommFinalize_test, double_finalize) {
    ncclComm_t* comms = NULL;
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
    ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));
    
    waitCommsReady(comms, nVis);

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i)
        (void) ncclCommFinalize(comms[i]);
    ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

    waitCommsReady(comms, nVis);

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclInvalidArgument, ncclCommFinalize(comms[i]));
    ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ncclCommAbort(comms[i]);

    free(comms);
    SUCCEED();
}
