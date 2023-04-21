#include "ncclCommon_test.cuh"
class ncclCommCuDevice_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = NULL;
    int nVis = 0, device = -1;
    virtual void SetUp() {
        register_segv_handler();
        comms = ncclCommon_getComms(&nVis);
    };
    virtual void TearDown() {
    };
};
TEST_F(ncclCommCuDevice_test, basic) {
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommCuDevice(comms[i], &device));
        ASSERT_EQ(device, i);
    }
}
TEST_F(ncclCommCuDevice_test, null_comm) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommCuDevice(NULL, &device));
}
TEST_F(ncclCommCuDevice_test, null_device) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommCuDevice(comms[0], NULL));
}
// EOF
