#include "ncclCommon_test.cuh"

class ncclCommInitRank_test : public ::testing::Test {
  protected:
    ncclComm_t comm;
    int ndev = 1;
    ncclUniqueId commId;
    int rank = 0;
    virtual void SetUp() {
        register_segv_handler();
        ncclCommon_destroysrComms();
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0); // API tests expect this behaviour (ncclCommInitRank)
        (void) setenv("NCCL_SET_THREAD_NAME", "1", 0); // Test that this doesn't break things
        comm = NCCL_COMM_NULL;
    };
    virtual void TearDown() {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
    };
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
