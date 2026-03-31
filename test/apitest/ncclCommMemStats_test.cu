#include "ncclCommon_test.cuh"

/*
 * API tests for ncclCommMemStats:
 *   ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value)
 *
 * ncclCommMemStats is a local query (no barriers), so these tests use
 * single-threaded ncclCommInitAll with grouped collectives.
 */

class ncclCommMemStats_test : public ::testing::Test {
protected:
    ncclComm_t *comms;
    int nVis;
    void **sendbuffs;
    void **recvbuffs;
    cudaStream_t *streams;
    const size_t size = 4 << 20; // 4MB

    virtual void SetUp() {
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
        comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
        ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));

        sendbuffs = (void**)calloc(nVis, sizeof(void*));
        recvbuffs = (void**)calloc(nVis, sizeof(void*));
        streams = (cudaStream_t*)calloc(nVis, sizeof(cudaStream_t));
        for (int i = 0; i < nVis; i++) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
            ASSERT_EQ(ncclSuccess, ncclMemAlloc(&sendbuffs[i], size));
            ASSERT_EQ(ncclSuccess, ncclMemAlloc(&recvbuffs[i], size));
            ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        }

        // Run a collective to populate internal memory state
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nVis; i++) {
            ASSERT_EQ(ncclSuccess, ncclAllReduce(sendbuffs[i], recvbuffs[i],
                      size / sizeof(float), ncclFloat, ncclSum, comms[i], streams[i]));
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
        for (int i = 0; i < nVis; i++) {
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
        }
    }

    virtual void TearDown() {
        for (int i = 0; i < nVis; i++) {
            ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[i]));
            ASSERT_EQ(ncclSuccess, ncclMemFree(sendbuffs[i]));
            ASSERT_EQ(ncclSuccess, ncclMemFree(recvbuffs[i]));
            ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
        }
        free(sendbuffs);
        free(recvbuffs);
        free(streams);
        free(comms);
    }
};

/* Query all four stat types successfully on every comm */
TEST_F(ncclCommMemStats_test, basic_query_all) {
    for (int i = 0; i < nVis; i++) {
        uint64_t value;
        ASSERT_EQ(ncclSuccess, ncclCommMemStats(comms[i], ncclStatGpuMemTotal, &value));
        ASSERT_EQ(ncclSuccess, ncclCommMemStats(comms[i], ncclStatGpuMemSuspend, &value));
        ASSERT_EQ(ncclSuccess, ncclCommMemStats(comms[i], ncclStatGpuMemSuspended, &value));
        ASSERT_EQ(ncclSuccess, ncclCommMemStats(comms[i], ncclStatGpuMemPersist, &value));
    }
}

/* After init + collective, total tracked GPU memory should be > 0 */
TEST_F(ncclCommMemStats_test, nonzero_after_collective) {
    for (int i = 0; i < nVis; i++) {
        uint64_t total;
        ASSERT_EQ(ncclSuccess, ncclCommMemStats(comms[i], ncclStatGpuMemTotal, &total));
        EXPECT_GT(total, (uint64_t)0) << "rank " << i << " has zero total memory after collective";
    }
}

/* Total = persist + suspendable */
TEST_F(ncclCommMemStats_test, total_equals_persist_plus_suspend) {
    for (int i = 0; i < nVis; i++) {
        uint64_t total, persist, suspendable;
        ASSERT_EQ(ncclSuccess, ncclCommMemStats(comms[i], ncclStatGpuMemTotal, &total));
        ASSERT_EQ(ncclSuccess, ncclCommMemStats(comms[i], ncclStatGpuMemPersist, &persist));
        ASSERT_EQ(ncclSuccess, ncclCommMemStats(comms[i], ncclStatGpuMemSuspend, &suspendable));
        ASSERT_EQ(total, persist + suspendable)
            << "rank " << i << ": total=" << total
            << " persist=" << persist << " suspend=" << suspendable;
    }
}

/* Initially not suspended */
TEST_F(ncclCommMemStats_test, not_suspended_initially) {
    for (int i = 0; i < nVis; i++) {
        uint64_t suspended;
        ASSERT_EQ(ncclSuccess, ncclCommMemStats(comms[i], ncclStatGpuMemSuspended, &suspended));
        ASSERT_EQ((uint64_t)0, suspended) << "rank " << i << " reports suspended state";
    }
}

/* Null value pointer returns ncclInvalidArgument */
TEST_F(ncclCommMemStats_test, null_value_ptr) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommMemStats(comms[0], ncclStatGpuMemTotal, NULL));
}

/* Null comm returns ncclInvalidArgument */
TEST_F(ncclCommMemStats_test, null_comm) {
    uint64_t value;
    ASSERT_EQ(ncclInvalidArgument, ncclCommMemStats(NULL, ncclStatGpuMemTotal, &value));
}

/* Invalid stat enum value returns ncclInvalidArgument */
TEST_F(ncclCommMemStats_test, invalid_stat_type) {
    uint64_t value;
    ASSERT_EQ(ncclInvalidArgument,
              ncclCommMemStats(comms[0], (ncclCommMemStat_t)999, &value));
}
