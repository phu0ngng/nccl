#include "ncclCommon_test.cuh"

class ncclCommShrink_test : public ::testing::Test {
protected:
  ncclComm_t* comms = NULL;
  ncclComm_t* comms2 = NULL;
  int nVis = 0;
  virtual void SetUp() {
    register_segv_handler();
    (void)setenv("NCCL_CHECK_POINTERS", "1", 0);
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
    comms2 = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
  };
  void CommonShrinkTest(ncclComm_t* comms, int* ranks, int rankSize, ncclComm_t* comms2, int flag, ncclResult_t expected = ncclSuccess, bool share = false) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; i++) {
      if (!ranks || std::find(ranks, ranks + rankSize, i) == ranks + rankSize) {
        ncclConfig_t shrinkConfig = NCCL_CONFIG_INITIALIZER;
        shrinkConfig.shrinkShare = share;
        ASSERT_EQ(expected, ncclCommShrink(comms ? comms[i] : NULL, ranks, rankSize, comms2 ? &comms2[i] : NULL, &shrinkConfig, flag));
      }
    }
    ASSERT_EQ(expected, ncclGroupEnd());
    if (expected == ncclSuccess) {
      int count1, count2;
      ASSERT_EQ(ncclSuccess, ncclCommCount(comms[0], &count1));
      ASSERT_EQ(ncclSuccess, ncclCommCount(comms2[0], &count2));
      ASSERT_EQ(count1 - count2, rankSize);
    }
    for (int i = 0; i < nVis; i++) {
      if (comms2 && comms2[i]) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms2[i]));
        comms2[i] = NULL;
      }
    }
  }
  virtual void TearDown() {
    for (int i = 0; i < nVis; ++i) {
      ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
      comms[i] = NULL;
    }
    free(comms);
    comms = NULL;
    free(comms2);
    comms2 = NULL;
  };
};
TEST_F(ncclCommShrink_test, basic) {
  int ranks[] = {1}; // at least 2 GPUs
  if (nVis >= 2) CommonShrinkTest(comms, ranks, 1, comms2, NCCL_SHRINK_DEFAULT);
}
TEST_F(ncclCommShrink_test, basic_error_mode) {
  int ranks[] = {1}; // at least 2 GPUs
  if (nVis >= 2) CommonShrinkTest(comms, ranks, 1, comms2, NCCL_SHRINK_ABORT);
}
TEST_F(ncclCommShrink_test, not_sorted) {
  int ranks[] = {3, 1}; // at least 4 GPUs
  if (nVis >= 4) CommonShrinkTest(comms, ranks, sizeof(ranks) / sizeof(ranks[0]), comms2, NCCL_SHRINK_DEFAULT);
}
TEST_F(ncclCommShrink_test, not_sorted_error_mode) {
  int ranks[] = {3, 1}; // at least 4 GPUs
  if (nVis >= 4) CommonShrinkTest(comms, ranks, sizeof(ranks) / sizeof(ranks[0]), comms2, NCCL_SHRINK_ABORT);
}
TEST_F(ncclCommShrink_test, rank_null) { CommonShrinkTest(comms, NULL, 0, comms2, NCCL_SHRINK_DEFAULT, ncclInvalidArgument); }
TEST_F(ncclCommShrink_test, rank_null_error_mode) { CommonShrinkTest(comms, NULL, 0, comms2, NCCL_SHRINK_ABORT, ncclInvalidArgument); }
TEST_F(ncclCommShrink_test, shrink_null) {
  int ranks[] = {1};
  if (nVis >= 2) CommonShrinkTest(comms, ranks, 1, NULL, NCCL_SHRINK_DEFAULT, ncclInvalidArgument);
}
TEST_F(ncclCommShrink_test, shrink_null_error_mode) {
  int ranks[] = {1};
  if (nVis >= 2) CommonShrinkTest(comms, ranks, 1, NULL, NCCL_SHRINK_ABORT, ncclInvalidArgument);
}
TEST_F(ncclCommShrink_test, shrink_share) {
  int ranks[] = {1};
  if (nVis >= 2) CommonShrinkTest(comms, ranks, 1, comms2, NCCL_SHRINK_DEFAULT, ncclSuccess, true);
}
