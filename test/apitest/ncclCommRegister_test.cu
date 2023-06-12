#include "ncclCommon_test.cuh"

class ncclCommRegister_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = nullptr;
    int nVis = 0, rank = -1;
    virtual void SetUp() {
        register_segv_handler();
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0);
        comms = ncclCommon_getComms(&nVis);
    };
    virtual void TearDown() {
    };
};
TEST_F(ncclCommRegister_test, basic) {
    const int size = 1024;
    void* buff = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buff, size));
    ASSERT_NE(nullptr, buff);
    ASSERT_EQ(ncclSuccess, ncclCommRegister(comms[0], buff, size));
    ASSERT_EQ(ncclSuccess, ncclCommDeregister(comms[0], buff, size));
    ASSERT_EQ(cudaSuccess, cudaFree(buff));
}
TEST_F(ncclCommRegister_test, null_comm) {
    const int size = 1024;
    void* buff = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buff, size));
    ASSERT_NE(nullptr, buff);
    ASSERT_EQ(ncclInvalidArgument, ncclCommRegister(nullptr, buff, size));
    ASSERT_EQ(cudaSuccess, cudaFree(buff));
}
TEST_F(ncclCommRegister_test, null_buff) {
    ASSERT_EQ(ncclInvalidArgument, ncclCommRegister(comms[0], nullptr, 1024));
}
TEST_F(ncclCommRegister_test, host_buff) {
    const int size = 1024;
    void* buff = nullptr;
    ASSERT_NE(nullptr, buff = malloc(size));
    ASSERT_EQ(ncclInvalidArgument, ncclCommRegister(comms[0], buff, size));
    free(buff);
}
TEST_F(ncclCommRegister_test, many) {
    const int n = 1024;
    const int size = 1024;
    void** buffs = nullptr;
    ASSERT_NE(buffs = (void**)malloc(sizeof(void*)*n), nullptr);
    for (int i=0; i<n; i++) {
      ASSERT_EQ(cudaSuccess, cudaMalloc(buffs+i, size));
      ASSERT_NE(nullptr, buffs[i]);
    }
    for (int i=0; i<n; i++) ASSERT_EQ(ncclSuccess, ncclCommRegister(comms[0], buffs[i], size));
    for (int i=0; i<n; i++) ASSERT_EQ(ncclSuccess, ncclCommDeregister(comms[0], buffs[i], size));
    for (int i=0; i<n; i++) ASSERT_EQ(cudaSuccess, cudaFree(buffs[i]));
    free(buffs);
}
TEST_F(ncclCommRegister_test, nested) {
    const int size = 17*1024*1024;
    char* buff = nullptr;
    int end = 16*1024*1024; // Keep 1M margin
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buff, size));
    ASSERT_NE(nullptr, buff);
    ASSERT_EQ(ncclSuccess, ncclCommRegister(comms[0], buff, end)); // #1 Covers most of the buffer
    ASSERT_EQ(ncclSuccess, ncclCommRegister(comms[0], buff+4096, end-8192)); // #2 Within 4K on each sides
    ASSERT_EQ(ncclSuccess, ncclCommRegister(comms[0], buff, end)); // #3 Same as #1
    ASSERT_EQ(ncclSuccess, ncclCommRegister(comms[0], buff, end-1024*1024)); // #4 Same start
    ASSERT_EQ(ncclSuccess, ncclCommRegister(comms[0], buff+1024*1024, end-1024*1024)); // #5 Same end
    ASSERT_EQ(ncclSuccess, ncclCommRegister(comms[0], buff+end, size-end)); // #6 Disjoint
    ASSERT_EQ(ncclSuccess, ncclCommRegister(comms[0], buff, size)); // #7 Full buffer covering #1 and #6
    ASSERT_EQ(ncclSuccess, ncclCommRegister(comms[0], buff, end)); // #8 Same as #1 again but now within #7
    ASSERT_EQ(ncclSuccess, ncclCommDeregister(comms[0], buff, end));
    ASSERT_EQ(ncclSuccess, ncclCommDeregister(comms[0], buff+end, size-end));
    ASSERT_EQ(ncclSuccess, ncclCommDeregister(comms[0], buff, size));
    ASSERT_EQ(ncclSuccess, ncclCommDeregister(comms[0], buff+1024*1024, end-1024*1024));
    ASSERT_EQ(ncclSuccess, ncclCommDeregister(comms[0], buff, end-1024*1024));
    ASSERT_EQ(ncclSuccess, ncclCommDeregister(comms[0], buff, end));
    ASSERT_EQ(ncclSuccess, ncclCommDeregister(comms[0], buff+4096, end-8192));
    ASSERT_EQ(ncclSuccess, ncclCommDeregister(comms[0], buff, end));
    ASSERT_EQ(cudaSuccess, cudaFree(buff));
}
// EOF
