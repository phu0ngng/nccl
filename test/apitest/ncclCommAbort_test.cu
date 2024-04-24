TEST(ncclCommAbort, basic) {
    int ndev = 1;
    ncclComm_t* comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, ndev, NULL));
    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommAbort(comms[i]));
    free(comms);
    SUCCEED();
}
TEST(ncclCommAbort, null) {
    ASSERT_EQ(ncclSuccess, ncclCommAbort(NULL));
    SUCCEED();
}

TEST(ncclCommAbort, group_abort) {
    int ndev;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&ndev));
    ncclComm_t* comms0 = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ncclComm_t* comms1 = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms0, ndev, NULL));
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms1, ndev, NULL));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommAbort(comms0[i]));
        ASSERT_EQ(ncclSuccess, ncclCommAbort(comms1[i]));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    free(comms0);
    free(comms1);
    SUCCEED();
}
