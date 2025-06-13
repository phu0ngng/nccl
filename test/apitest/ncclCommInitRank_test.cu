#include "ncclCommon_test.cuh"

#include <map>
#include <regex.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

class ncclCommInitRank_test : public ::testing::Test {
  protected:
    ncclComm_t comm;
    int ndev = 1;
    ncclUniqueId commId;
    int rank = 0;
    virtual void SetUp() override {
        register_segv_handler();
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0); // API tests expect this behaviour (ncclCommInitRank)
        (void) setenv("NCCL_SET_THREAD_NAME", "1", 0); // Test that this doesn't break things
        comm = NCCL_COMM_NULL;
    }
    virtual void TearDown() override {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
    }
};


TEST_F(ncclCommInitRank_test, basic) {
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, ndev, commId, rank));
}
TEST_F(ncclCommInitRank_test, comm_null) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, ndev, commId, rank));
}
TEST_F(ncclCommInitRank_test, id_dup) {
    ncclUniqueId* id1 = (ncclUniqueId*)malloc(sizeof(ncclUniqueId));
    EXPECT_EQ(ncclSuccess, ncclGetUniqueId(id1));
    ncclUniqueId* id2 = (ncclUniqueId*)malloc(sizeof(ncclUniqueId));
    memcpy(id2, id1, sizeof(ncclUniqueId));
    memset(id1, 0, sizeof(ncclUniqueId));
    free(id1);
    EXPECT_EQ(ncclSuccess, ncclCommInitRank(&comm, 1, *id2, 0));
    free(id2);
}
TEST_F(ncclCommInitRank_test, ndev_zero) {
    ASSERT_EQ(ncclInvalidArgument,
              ncclCommInitRank(&comm, 0, commId, rank));
}
TEST_F(ncclCommInitRank_test, dev_negative) {
    ASSERT_EQ(ncclInvalidArgument,
              ncclCommInitRank(&comm, -1, commId, rank));
}
TEST_F(ncclCommInitRank_test, rank_outofboundary) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(&comm, ndev, commId, 1));
}
TEST_F(ncclCommInitRank_test, rank_negative) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(&comm, ndev, commId, -1));
}
TEST_F(ncclCommInitRank_test, DISABLED_dev_too_many) { // cause dead loop
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(&comm, 10, commId, rank));
}
TEST_F(ncclCommInitRank_test, magic) {
    uint64_t ncclMagic = 0x0280028002800280;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, ndev, commId, rank));
    ASSERT_EQ(ncclMagic, ((uint64_t*)comm)[0]);
}
TEST_F(ncclCommInitRank_test, socket_connection_crash) {
    ncclComm_t testComm = NULL;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
    memset(&commId, 0, sizeof(ncclUniqueId));  // Zero out the ID to make it invalid
    int testRank = 0;
    ASSERT_EQ(ncclInternalError, ncclCommInitRank(&testComm, 2, commId, testRank));
    ASSERT_EQ(NULL, testComm);
}

class ncclCommInitRankParseListTest : public ncclOutputTest {
  // Tests for ParseList, as accessed through the NCCL_PROTO and NCCL_ALGO environment variables.
  protected:
    int nDev=0;
    ncclUniqueId commId;
    virtual void SetUp() override {
        ncclOutputTest::SetUp();
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
        cudaGetDeviceCount(&nDev);
        if (nDev<2) {
            static bool once_only = false;
            if (!once_only) {
                printf("Warning: ncclCommInitRankParseListTest do not engage for less than 2 GPUs.\n");
                // See src/graph/tuning.cc circa line 180: "if (nRanks <= 1) return ncclSuccess;"
                once_only = true;
            }
            // GTEST_SKIP requires a more recent googletest version. For now, we'll just pass the test.
            // GTEST_SKIP() << "Testing ParseList requires more than one device.";
        }
    }
    void runTest(const char* envName, const char* envValue,
                 ncclResult_t initResult, ncclResult_t endResult, ncclResult_t destroyResult,
                 const char* logRegex) {
        if (nDev<2) return; // Skip the test... a warning has already been issued

        overrideEnvVariable("NCCL_COMM_ID","127.0.0.1:46001");
        overrideEnvVariable(envName, envValue);

        ncclUniqueId id;
        EXPECT_EQ(ncclSuccess, ncclGetUniqueId(&id));
        ncclComm_t comms[nDev];
        EXPECT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < nDev; i++) {
            ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
            EXPECT_EQ(initResult, ncclCommInitRank(comms+i, nDev, id, i));
        }
        EXPECT_EQ(endResult, ncclGroupEnd());
        for (int i = 0; i < nDev; i++) {
            EXPECT_EQ(destroyResult, ncclCommDestroy(comms[i]));
        }

        verifyResult(logRegex);
    }
};

// Verify correct behavior in both proto and algo
// Do not validate the default for the settings (algo/proto) that is not
// being tested, as it may be set outside the test.
TEST_F(ncclCommInitRankParseListTest, protoEmpty) {
    const char regex[] = "    Broadcast . *1 *2 *1 *. *. *. *. *. *. *. *..*"
                         "       Reduce . *1 *2 *1 *. *. *. *. *. *. *. *..*"
                         "    AllGather . *1 *2 *1 *. *. *. *. *. *. *. *..*"
                         "ReduceScatter . *1 *2 *1 *. *. *. *. *. *. *. *..*"
                         "    AllReduce . *1 *2 *1 *. *. *. *. *. *. *. *..*";
    runTest("NCCL_PROTO", "", ncclSuccess, ncclSuccess, ncclSuccess, regex);
}
TEST_F(ncclCommInitRankParseListTest, algoEmpty) {
    const char regex[] = "    Broadcast . *. *. *. *. *1 *1 *1 *1 *1 *1 *1.*"
                         "       Reduce . *. *. *. *. *1 *1 *1 *1 *1 *1 *1.*"
                         "    AllGather . *. *. *. *. *1 *1 *1 *1 *1 *1 *1.*"
                         "ReduceScatter . *. *. *. *. *1 *1 *1 *1 *1 *1 *1.*"
                         "    AllReduce . *. *. *. *. *1 *1 *1 *1 *1 *1 *1.*";
    runTest("NCCL_ALGO", "", ncclSuccess, ncclSuccess, ncclSuccess, regex);
}
TEST_F(ncclCommInitRankParseListTest, protoValid) {
    const char regex[] = "    Broadcast . *1 *0 *0 *. *. *. *. *. *. *. *..*"
                         "       Reduce . *1 *0 *0 *. *. *. *. *. *. *. *..*"
                         "    AllGather . *1 *0 *0 *. *. *. *. *. *. *. *..*"
                         "ReduceScatter . *1 *0 *0 *. *. *. *. *. *. *. *..*"
                         "    AllReduce . *1 *0 *0 *. *. *. *. *. *. *. *..*";
    runTest("NCCL_PROTO", "LL", ncclSuccess, ncclSuccess, ncclSuccess, regex);
}
TEST_F(ncclCommInitRankParseListTest, algoValid) {
    const char regex[] = "    Broadcast . *. *. *. *. *0 *1 *0 *0 *0 *0 *0.*"
                         "       Reduce . *. *. *. *. *0 *1 *0 *0 *0 *0 *0.*"
                         "    AllGather . *. *. *. *. *0 *1 *0 *0 *0 *0 *0.*"
                         "ReduceScatter . *. *. *. *. *0 *1 *0 *0 *0 *0 *0.*"
                         "    AllReduce . *. *. *. *. *0 *1 *0 *0 *0 *0 *0.*";
    runTest("NCCL_ALGO", "RING", ncclSuccess, ncclSuccess, ncclSuccess, regex);
}
TEST_F(ncclCommInitRankParseListTest, protoBad) {
    runTest("NCCL_PROTO", "bad_value", ncclSuccess, ncclInvalidUsage, ncclSuccess, "NCCL WARN Unrecognized element token");
}
TEST_F(ncclCommInitRankParseListTest, algoBad) {
    runTest("NCCL_ALGO", "bad_value", ncclSuccess, ncclInvalidUsage, ncclSuccess, "NCCL WARN Unrecognized element token");
}
// Now that basic behavior has been verified, just use NCCL_PROTO to test parsing of more specific behavior
TEST_F(ncclCommInitRankParseListTest, protoBadPrefix) {
    runTest("NCCL_PROTO", "bad_prefix:LL", ncclSuccess, ncclInvalidUsage, ncclSuccess, "NCCL WARN Unrecognized prefix token");
}
TEST_F(ncclCommInitRankParseListTest, protoBadPrefixedElement) {
    runTest("NCCL_PROTO", "reduce:LL,bad,Simple", ncclSuccess, ncclInvalidUsage, ncclSuccess, "NCCL WARN Unrecognized element token");
}
TEST_F(ncclCommInitRankParseListTest, notGlobal) {
    const char regex[] = "    Broadcast . *0 *1 *1 *. *. *. *. *. *. *. *..*"
                         "       Reduce . *0 *1 *1 *. *. *. *. *. *. *. *..*"
                         "    AllGather . *0 *1 *1 *. *. *. *. *. *. *. *..*"
                         "ReduceScatter . *0 *1 *1 *. *. *. *. *. *. *. *..*"
                         "    AllReduce . *0 *1 *1 *. *. *. *. *. *. *. *..*";
    runTest("NCCL_PROTO", "^LL", ncclSuccess, ncclSuccess, ncclSuccess, regex);
}
TEST_F(ncclCommInitRankParseListTest, funcOverride) {
    const char regex[] = "    Broadcast . *1 *0 *1 *. *. *. *. *. *. *. *..*"
                         "       Reduce . *1 *0 *1 *. *. *. *. *. *. *. *..*"
                         "    AllGather . *1 *0 *1 *. *. *. *. *. *. *. *..*"
                         "ReduceScatter . *1 *0 *1 *. *. *. *. *. *. *. *..*"
                         "    AllReduce . *0 *1 *0 *. *. *. *. *. *. *. *..*";
    runTest("NCCL_PROTO", "LL,Simple;allreduce:LL128", ncclSuccess, ncclSuccess, ncclSuccess, regex);
}
TEST_F(ncclCommInitRankParseListTest, globalNotWithFuncOverrides) {
    const char regex[] = "    Broadcast . *0 *0 *1 *. *. *. *. *. *. *. *..*"
                         "       Reduce . *0 *1 *0 *. *. *. *. *. *. *. *..*"
                         "    AllGather . *0 *1 *0 *. *. *. *. *. *. *. *..*"
                         "ReduceScatter . *0 *1 *0 *. *. *. *. *. *. *. *..*"
                         "    AllReduce . *0 *1 *0 *. *. *. *. *. *. *. *..*";
    runTest("NCCL_PROTO", "^LL,Simple;BROADCAST:Simple", ncclSuccess, ncclSuccess, ncclSuccess, regex);
}
TEST_F(ncclCommInitRankParseListTest, noGlobal) {
    const char regex[] = "    Broadcast . *1 *1 *0 *. *. *. *. *. *. *. *..*"
                         "       Reduce . *1 *0 *1 *. *. *. *. *. *. *. *..*"
                         "    AllGather . *1 *2 *1 *. *. *. *. *. *. *. *..*"
                         "ReduceScatter . *1 *2 *1 *. *. *. *. *. *. *. *..*"
                         "    AllReduce . *1 *2 *1 *. *. *. *. *. *. *. *..*";
    runTest("NCCL_PROTO", "broadcast:^Simple;reduce:Simple,LL", ncclSuccess, ncclSuccess, ncclSuccess, regex);
}
TEST_F(ncclCommInitRankParseListTest, funcOverrideGlobalNotFirst) {
    runTest("NCCL_PROTO", "allreduce:LL128;LL,Simple", ncclSuccess, ncclInvalidUsage, ncclSuccess,
            "All entries except the first must have a prefix");
}

