#include "ncclCommon_test.cuh"
class ncclCommCount_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = NULL;
    int nVis = 0, ndev = 0;
    int count = -1;
    virtual void SetUp() {
        comms = ncclCommon_getComms(&nVis);
        ndev = nVis;
    };
    virtual void TearDown() {
    };
};
TEST_F(ncclCommCount_test, basic) {
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommCount(comms[i], &count));
        ASSERT_EQ(ndev, count);
    }
};
// 1.
TEST_F(ncclCommCount_test, comm_null) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommCount(NULL, &count));
};
// 2.
TEST_F(ncclCommCount_test, count_null) {
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclInvalidArgument, ncclCommCount(comms[i], NULL));
    }
};
// 3.
TEST_F(ncclCommCount_test, comm_some) {
    ndev = nVis - (nVis > 1 ? 1 : 0);
    ncclComm_t comms[ndev];
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, ndev, NULL));
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommCount(comms[i], &count));
        ASSERT_EQ(ndev, count);
    }
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }
}
