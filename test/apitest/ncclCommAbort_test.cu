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
