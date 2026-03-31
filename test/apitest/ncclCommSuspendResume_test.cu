#include "ncclCommon_test.cuh"
#include <pthread.h>
#include <vector>

/*
 * API tests for Dynamic Memory Offload:
 *   ncclCommSuspend(ncclComm_t comm, int flags)
 *   ncclCommResume(ncclComm_t comm)
 *
 * Because ncclCommSuspend/Resume contain internal bootstrap barriers
 * that require all ranks to participate simultaneously, these tests
 * use one-thread-per-GPU with ncclCommInitRank.
 */

struct SuspendResumeThreadArgs;
typedef void (*SuspendResumeExeFn)(SuspendResumeThreadArgs*);

struct SuspendResumeThreadArgs {
    int rank;
    int nVis;
    volatile int* retPtr;
    volatile ncclUniqueId* gidPtr;
    pthread_barrier_t* barrier;
    volatile int* abortFlagPtr;
    SuspendResumeExeFn exeFn;
};

class ncclCommSuspendResume_test : public ::testing::Test {
protected:
    int nVis;
    virtual void SetUp() {
        register_segv_handler();
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
    }
    virtual void TearDown() {}
};

/* Helper: exchange ncclUniqueId across threads via pthread barrier */
static void exchangeId(SuspendResumeThreadArgs* args, ncclUniqueId* lid) {
    if (args->rank == 0) {
        if (ncclGetUniqueId(lid) != ncclSuccess) {
            *args->abortFlagPtr = 1;
            pthread_barrier_wait(args->barrier);
            GTEST_FAIL();
            return;
        }
        memcpy((void*)args->gidPtr, lid, sizeof(ncclUniqueId));
        pthread_barrier_wait(args->barrier);
    } else {
        pthread_barrier_wait(args->barrier);
        ASSERT_EQ(0, (int)*args->abortFlagPtr);
        memcpy(lid, (const void*)args->gidPtr, sizeof(ncclUniqueId));
    }
}

/* Generic thread entry: calls the exeFn stored in args */
static void* genericThread(void* arg) {
    SuspendResumeThreadArgs* a = (SuspendResumeThreadArgs*)arg;
    a->exeFn(a);
    if (::testing::Test::HasFatalFailure()) *a->retPtr = 1;
    return NULL;
}

/* Launch one-thread-per-GPU running exeFn, then join all */
static void runThreaded(int nVis, SuspendResumeExeFn exeFn) {
    pthread_t* thr = (pthread_t*)malloc(sizeof(pthread_t) * nVis);
    SuspendResumeThreadArgs* a = (SuspendResumeThreadArgs*)malloc(
        sizeof(SuspendResumeThreadArgs) * nVis);
    volatile ncclUniqueId gid;
    volatile int ret = 0, abort = 0;
    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, nVis);
    for (int i = 0; i < nVis; i++) {
        a[i].rank = i;
        a[i].nVis = nVis;
        a[i].retPtr = &ret;
        a[i].gidPtr = &gid;
        a[i].barrier = &bar;
        a[i].abortFlagPtr = &abort;
        a[i].exeFn = exeFn;
        ASSERT_EQ(0, pthread_create(&thr[i], NULL, genericThread, &a[i]));
    }
    for (int i = 0; i < nVis; i++) {
        ASSERT_EQ(0, pthread_join(thr[i], NULL));
    }
    ASSERT_EQ(0, ret);
    pthread_barrier_destroy(&bar);
    free(a);
    free(thr);
}

/* Helper struct to reduce boilerplate across test functions */
struct SuspendResumeCtx {
    ncclComm_t comm;
    void *sendbuff, *recvbuff;
    cudaStream_t stream;
    size_t size;

    void init(SuspendResumeThreadArgs* args, size_t sz) {
        size = sz;
        ncclUniqueId lid;

        ASSERT_EQ(cudaSuccess, cudaSetDevice(args->rank));
        exchangeId(args, &lid);

        ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, args->nVis, lid, args->rank));

        sendbuff = nullptr;
        recvbuff = nullptr;
        ASSERT_EQ(ncclSuccess, ncclMemAlloc(&sendbuff, size));
        ASSERT_EQ(ncclSuccess, ncclMemAlloc(&recvbuff, size));
        ASSERT_EQ(cudaSuccess, cudaMemset(sendbuff, 0, size));
        ASSERT_EQ(cudaSuccess, cudaMemset(recvbuff, 0, size));

        ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // Warmup collective to allocate internal buffers
        ASSERT_EQ(ncclSuccess, ncclAllReduce(sendbuff, recvbuff,
                  size / sizeof(float), ncclFloat, ncclSum, comm, stream));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    void teardown() {
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
        ASSERT_EQ(ncclSuccess, ncclMemFree(sendbuff));
        ASSERT_EQ(ncclSuccess, ncclMemFree(recvbuff));
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
    }
};


/* ================================================================
 * Test: basic — suspend/resume lifecycle with collectives
 * ================================================================ */
static void basicExe(SuspendResumeThreadArgs* args) {
    SuspendResumeCtx ctx;
    ctx.init(args, 1 << 20);

    // Suspend memory
    ASSERT_EQ(ncclSuccess, ncclCommSuspend(ctx.comm, NCCL_SUSPEND_MEM));

    // Resume memory
    ASSERT_EQ(ncclSuccess, ncclCommResume(ctx.comm));

    // Collective after resume should still work
    ASSERT_EQ(ncclSuccess, ncclAllReduce(ctx.sendbuff, ctx.recvbuff,
              ctx.size / sizeof(float), ncclFloat, ncclSum, ctx.comm, ctx.stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(ctx.stream));

    ctx.teardown();
}

TEST_F(ncclCommSuspendResume_test, basic) {
    runThreaded(nVis, basicExe);
}

/* ================================================================
 * Test: mem_stats — verify memory stats across suspend/resume
 * ================================================================ */
static void memStatsExe(SuspendResumeThreadArgs* args) {
    SuspendResumeCtx ctx;
    ctx.init(args, 1 << 20);

    // Query stats before suspend
    uint64_t totalBefore, suspendable, suspended, persist;
    ASSERT_EQ(ncclSuccess, ncclCommMemStats(ctx.comm, ncclStatGpuMemTotal, &totalBefore));
    ASSERT_EQ(ncclSuccess, ncclCommMemStats(ctx.comm, ncclStatGpuMemSuspend, &suspendable));
    ASSERT_EQ(ncclSuccess, ncclCommMemStats(ctx.comm, ncclStatGpuMemSuspended, &suspended));
    ASSERT_EQ(ncclSuccess, ncclCommMemStats(ctx.comm, ncclStatGpuMemPersist, &persist));

    // Should not be suspended yet
    ASSERT_EQ((uint64_t)0, suspended);
    // Total = persist + suspendable
    ASSERT_EQ(totalBefore, persist + suspendable);

    // Suspend
    ASSERT_EQ(ncclSuccess, ncclCommSuspend(ctx.comm, NCCL_SUSPEND_MEM));

    // Should now be suspended
    uint64_t suspendedAfter;
    ASSERT_EQ(ncclSuccess, ncclCommMemStats(ctx.comm, ncclStatGpuMemSuspended, &suspendedAfter));
    ASSERT_EQ((uint64_t)1, suspendedAfter);

    // Resume
    ASSERT_EQ(ncclSuccess, ncclCommResume(ctx.comm));

    // Should no longer be suspended; total should be restored
    uint64_t totalAfter, suspendedFinal;
    ASSERT_EQ(ncclSuccess, ncclCommMemStats(ctx.comm, ncclStatGpuMemTotal, &totalAfter));
    ASSERT_EQ(ncclSuccess, ncclCommMemStats(ctx.comm, ncclStatGpuMemSuspended, &suspendedFinal));
    ASSERT_EQ((uint64_t)0, suspendedFinal);
    ASSERT_EQ(totalBefore, totalAfter);

    ctx.teardown();
}

TEST_F(ncclCommSuspendResume_test, mem_stats) {
    runThreaded(nVis, memStatsExe);
}

/* ================================================================
 * Test: double_suspend_error — second suspend returns error
 * ================================================================ */
static void doubleSuspendExe(SuspendResumeThreadArgs* args) {
    SuspendResumeCtx ctx;
    ctx.init(args, 1 << 20);

    // First suspend succeeds
    ASSERT_EQ(ncclSuccess, ncclCommSuspend(ctx.comm, NCCL_SUSPEND_MEM));

    // Second suspend should fail — already suspended
    ASSERT_EQ(ncclInvalidUsage, ncclCommSuspend(ctx.comm, NCCL_SUSPEND_MEM));

    // Resume to allow clean teardown
    ASSERT_EQ(ncclSuccess, ncclCommResume(ctx.comm));

    ctx.teardown();
}

TEST_F(ncclCommSuspendResume_test, double_suspend_error) {
    runThreaded(nVis, doubleSuspendExe);
}

/* ================================================================
 * Test: resume_without_suspend_error
 * Resume when not suspended should return ncclInvalidUsage.
 * The error check happens before any barrier, so no deadlock.
 * ================================================================ */
static void resumeNoSuspendExe(SuspendResumeThreadArgs* args) {
    SuspendResumeCtx ctx;
    ctx.init(args, 1 << 20);

    // Resume without prior suspend should fail
    ASSERT_EQ(ncclInvalidUsage, ncclCommResume(ctx.comm));

    ctx.teardown();
}

TEST_F(ncclCommSuspendResume_test, resume_without_suspend_error) {
    runThreaded(nVis, resumeNoSuspendExe);
}

/* ================================================================
 * Test: destroy_while_suspended — destroy without resuming first
 * ================================================================ */
static void destroyWhileSuspendedExe(SuspendResumeThreadArgs* args) {
    SuspendResumeCtx ctx;
    ctx.init(args, 1 << 20);

    // Suspend
    ASSERT_EQ(ncclSuccess, ncclCommSuspend(ctx.comm, NCCL_SUSPEND_MEM));

    // Tear down without resuming — should succeed
    ctx.teardown();
}

TEST_F(ncclCommSuspendResume_test, destroy_while_suspended) {
    runThreaded(nVis, destroyWhileSuspendedExe);
}

/* ================================================================
 * Test: multiple_cycles — repeated suspend/resume with collectives
 * ================================================================ */
static void multipleCyclesExe(SuspendResumeThreadArgs* args) {
    SuspendResumeCtx ctx;
    ctx.init(args, 1 << 20);

    const int numCycles = 3;
    for (int c = 0; c < numCycles; c++) {
        // Run collective
        ASSERT_EQ(ncclSuccess, ncclAllReduce(ctx.sendbuff, ctx.recvbuff,
                  ctx.size / sizeof(float), ncclFloat, ncclSum, ctx.comm, ctx.stream));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(ctx.stream));

        // Suspend
        ASSERT_EQ(ncclSuccess, ncclCommSuspend(ctx.comm, NCCL_SUSPEND_MEM));

        uint64_t suspended;
        ASSERT_EQ(ncclSuccess, ncclCommMemStats(ctx.comm, ncclStatGpuMemSuspended, &suspended));
        ASSERT_EQ((uint64_t)1, suspended) << "cycle " << c;

        // Resume
        ASSERT_EQ(ncclSuccess, ncclCommResume(ctx.comm));

        ASSERT_EQ(ncclSuccess, ncclCommMemStats(ctx.comm, ncclStatGpuMemSuspended, &suspended));
        ASSERT_EQ((uint64_t)0, suspended) << "cycle " << c;
    }

    ctx.teardown();
}

TEST_F(ncclCommSuspendResume_test, multiple_cycles) {
    runThreaded(nVis, multipleCyclesExe);
}

/* ================================================================
 * Single-threaded error tests (no barrier reached for null comm)
 * ================================================================ */
TEST_F(ncclCommSuspendResume_test, null_comm_suspend) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommSuspend(NULL, NCCL_SUSPEND_MEM));
}

TEST_F(ncclCommSuspendResume_test, null_comm_resume) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommResume(NULL));
}
