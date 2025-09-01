#include "ncclCommon_test.cuh"
#include <unistd.h>

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
        register_segv_handler();
        // TODO: Fix this when we can change NCCL_PARM values, as it
        // only actually takes affect if this is the first test run.
        // And clean it up later.
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


#if 0
// This test cannot work currently. We need to be able to reset the
// values of a NCCL_PARAM. We might be able to do this soon with the env
// plugin. If so, then we should run several iterations with the
// NCCL_SOCKET_POLL_TIMEOUT zero and non-zero until we can establish a
// trend of lower cpu times on init.
// If a sleep is introduced before the second socketWait in the
// socketFinalizeConnect then the CPU benefit is much more obvious.
// Wall clock time is unchanged, but CPU time does show a benefit, as
// intended.
static void* CPUInitTimeThreadTest(void* args) {
    struct threadArgs* thdArgs = (struct threadArgs*)args;

    clock_t start = clock();
    oneGPUPerThreadExe(thdArgs->rank, thdArgs->nVis, thdArgs->iteration, thdArgs->gidPtr, thdArgs->barrierPtr, thdArgs->abortFlagPtr);
    clock_t end = clock();
    *thdArgs->retPtr = (int) ((end-start) * 1000LL / CLOCKS_PER_SEC);
    return NULL;
}

TEST_F(ncclCommFinalize_test, lessCpuWhenUsingPollTimeout) {
    if (nVis < 2) {
      return; // Cannot do the test on less than two GPUs.
    }

    struct threadArgs* args;
    pthread_t* threads;
    volatile ncclUniqueId gid;
    volatile int abortFlag;
    pthread_barrier_t barrier;

    int* ret = (int*) calloc(nVis, sizeof(int));
    abortFlag = 0;
    pthread_barrier_init(&barrier, NULL, nVis);
    threads = (pthread_t*) malloc(sizeof(pthread_t) * nVis);
    args = (struct threadArgs*) malloc(sizeof(struct threadArgs) * nVis);

    for (int i = 0; i < nVis; ++i) {
        args[i].rank = i;
        args[i].nVis = nVis;
        args[i].iteration = iteration;
        args[i].retPtr = &ret[i];
        args[i].gidPtr = &gid;
        args[i].barrierPtr = &barrier;
        args[i].abortFlagPtr = &abortFlag;
    }

    (void) setenv("NCCL_NET", "Socket", /*overwrite=*/1);
    (void) setenv("NCCL_SHM_DISABLE", "1", /*overwrite=*/1);
    (void) setenv("NCCL_P2P_DISABLE", "1", /*overwrite=*/1);
    (void) setenv("NCCL_NVLS_ENABLE", "0", /*overwrite=*/1);

    const int NUM_SCENARIOS = 4;
    // Note, the parameter can't be changed once it is queried.
    const char* scenario_values[NUM_SCENARIOS] = { "1000", "1000", "1000", "1000" };
    //const char* scenario_values[NUM_SCENARIOS] = { "0", "0", "0", "0" };
    int scenario_avg_times[NUM_SCENARIOS];
    float avg_avg_time = 0.0f;
    for (int scenario_index = 0; scenario_index < NUM_SCENARIOS; ++scenario_index) {
      (void) setenv("NCCL_SOCKET_POLL_TIMEOUT_MSEC", scenario_values[scenario_index], /*overwrite=*/1);
      // When supported, the NCCL_PARAM needs to be reset so that the new value can be picked up.
      // Maybe the env plugin can help with this.
      for (int i = 0; i < nVis; ++i) {
          *args[i].retPtr = 0;
          ASSERT_EQ(0, pthread_create(&threads[i], NULL, CPUInitTimeThreadTest, (void*) &args[i]));
      }

      scenario_avg_times[scenario_index] = 0;
      for (int i = 0; i < nVis; ++i) {
          ASSERT_EQ(0, pthread_join(threads[i], NULL));

          scenario_avg_times[scenario_index] += *args[i].retPtr;
          printf("Scenario %d with val %s, rank %d has cpu time %dms\n",
                 scenario_index, scenario_values[scenario_index], i, *args[i].retPtr);
      }
      scenario_avg_times[scenario_index] /= nVis;
      if (scenario_index > 0) avg_avg_time += scenario_avg_times[scenario_index];
    }

    (void) unsetenv("NCCL_SOCKET_POLL_TIMEOUT_MSEC");
    for (int i=0; i<NUM_SCENARIOS; i++) {
        printf("Scenario %d, value %s, average time=%dms\n",
               i, scenario_values[i], scenario_avg_times[i]);
    }
    printf("Non-warmup average time is %f ms\n", avg_avg_time / (NUM_SCENARIOS-1));

    pthread_barrier_destroy(&barrier);
    free(ret);
    free(args);
    free(threads);

    SUCCEED();
}
#endif
