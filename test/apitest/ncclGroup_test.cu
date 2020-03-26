class ncclGroup_test : public ::testing::Test {
    protected:
        ncclComm_t comm;
        virtual void SetUp() {
            ASSERT_EQ(ncclSuccess, ncclCommInitAll(&comm, 1, NULL));
        };
        virtual void TearDown() {
            ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
        };
};
TEST_F(ncclGroup_test, basic) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
}
TEST_F(ncclGroup_test, no_start) {
    ASSERT_EQ(ncclInvalidUsage, ncclGroupEnd());
}
// NCCL doesn't test for that
TEST_F(ncclGroup_test, DISABLED_different_stream) {
    cudaStream_t stream;
    ncclComm_t comm;
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(&comm, 1, NULL));
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_EQ(ncclSuccess, ncclAllReduce(NULL, NULL, 0, ncclFloat, ncclSum, comm, NULL));
    ASSERT_EQ(ncclInvalidUsage, ncclAllReduce(NULL, NULL, 0, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(ncclInvalidUsage, ncclGroupEnd());
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
}
// EOF
