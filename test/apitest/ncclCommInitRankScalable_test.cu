#include "ncclCommon_test.cuh"

class ncclCommInitRankScalable_test : public ::testing::Test {
  protected:
    ncclComm_t comm;
    int ndev = 1;
    int nId = 1;
    ncclUniqueId commId;
    int rank = 0;
    virtual void SetUp() {
        register_segv_handler();
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0); // API tests expect this behaviour (ncclCommInitRank)
        (void) setenv("NCCL_SET_THREAD_NAME", "1", 0); // Test that this doesn't break things
        comm = NCCL_COMM_NULL;
    };
    virtual void TearDown() {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
    };
};
TEST_F(ncclCommInitRankScalable_test, basic) {
  ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
  ASSERT_EQ(ncclSuccess, ncclCommInitRankScalable(&comm, ndev,  rank,1, &commId, NULL));
}
TEST_F(ncclCommInitRankScalable_test, comm_null) {
  ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankScalable(NULL, ndev,rank, 1, &commId,  NULL));
}
TEST_F(ncclCommInitRankScalable_test, nId_zero) {
  ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
  ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankScalable(&comm, ndev, rank,0, &commId,  NULL));
}
TEST_F(ncclCommInitRankScalable_test, nId_too_many) {
  ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
  ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankScalable(&comm, ndev,rank, 874, &commId,  NULL));
}
