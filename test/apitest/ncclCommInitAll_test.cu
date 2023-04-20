#include "ncclCommon_test.cuh"

class ncclCommInitAll_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = NULL;
    int nVis = 0;
    virtual void SetUp() {
        register_segv_handler();
        ncclCommon_destroysrComms();
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0); // API tests expect this behaviour (ncclCommInitAll)
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
        comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    };
    virtual void TearDown() {
        if (NULL != comms) {
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
                comms[i] = NULL;
            }
            free(comms);
            comms = NULL;
        }
    };
};
TEST_F(ncclCommInitAll_test, basic) {
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
};
TEST_F(ncclCommInitAll_test, devListRev) {
    int* devList = (int*)calloc(nVis, sizeof(int));
    for (int i=0; i<nVis; i++) {
      devList[i] = nVis-1-i;
    }
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
    free(devList);
};
// 1.
TEST_F(ncclCommInitAll_test, comms_null) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitAll(NULL, nVis, NULL));
};
// 2.
TEST_F(ncclCommInitAll_test, ndev_negative) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitAll(comms, -1, NULL));
};
#if 0
// This was only failing becuase NCCL was artifically limiting to 128 devices
// per thread. That has been replaced with dynamic allocation so the new failure
// mode is that NCCL hangs indefinitely since threads representing valid devices
// are awaiting bootstrap connections from threads which aborted due to failing
// cudaSetDevice().
TEST_F(ncclCommInitAll_test, ndev_toomany) {
    comms = (ncclComm_t*)calloc(256, sizeof(ncclComm_t));
    ASSERT_EQ(ncclInvalidUsage,
              ncclCommInitAll(comms, 256, NULL));
};
#endif
TEST_F(ncclCommInitAll_test, devList_null) {
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
};
TEST_F(ncclCommInitAll_test, devList_duplicate) {
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    int* devList = (int*)calloc(nVis, sizeof(int));
    ASSERT_EQ(nVis > 1 ? ncclInvalidUsage : ncclSuccess,
              ncclCommInitAll(comms, nVis, devList));
    free(devList);
};
TEST_F(ncclCommInitAll_test, devList_nonexist) {
    int* devList = (int*)calloc(nVis, sizeof(int));
    for (int i = 0; i < nVis; ++i) {
        devList[i] = 1000 + i;
    }
    ASSERT_EQ(ncclUnhandledCudaError, ncclCommInitAll(comms, nVis, devList));
    free(devList);
};
// EOF
