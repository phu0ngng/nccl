TEST(ncclGetUniqueId, basic) {
    ncclUniqueId id;
    EXPECT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    // Free resources
    ncclComm_t comm;
    EXPECT_EQ(ncclSuccess, ncclCommInitRank(&comm, 1, id, 0));
    EXPECT_EQ(ncclSuccess, ncclCommDestroy(comm));
}
TEST(ncclGetUniqueId, null) {
    EXPECT_EQ(ncclInvalidArgument, ncclGetUniqueId(NULL));
}
TEST(ncclGetUniqueId, env_id_dual_api) {
    int nDev;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nDev));
    setenv("NCCL_COMM_ID","127.0.0.1:16000",1);
    ncclUniqueId id1, id2;
    EXPECT_EQ(ncclSuccess, ncclGetUniqueId(&id1));
    EXPECT_EQ(ncclSuccess, ncclGetUniqueIdFromEnv(&id2));
    ncclComm_t comm[nDev];
    EXPECT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nDev; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      EXPECT_EQ(ncclSuccess, ncclCommInitRank(comm+i, nDev, (i==0) ? id1:id2, i));
    }
    EXPECT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i = 0; i < nDev; i++) {
      EXPECT_EQ(ncclSuccess, ncclCommDestroy(comm[i]));
    }
    setenv("NCCL_COMM_ID","",1);
}
TEST(ncclGetUniqueId, env_id_single_api) {
    int nDev;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nDev));
    setenv("NCCL_COMM_ID","127.0.0.1:16001",1);
    ncclUniqueId id;
    EXPECT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ncclComm_t comm[nDev];
    EXPECT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nDev; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      EXPECT_EQ(ncclSuccess, ncclCommInitRank(comm+i, nDev, id, i));
    }
    EXPECT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i = 0; i < nDev; i++) {
      EXPECT_EQ(ncclSuccess, ncclCommDestroy(comm[i]));
    }
    setenv("NCCL_COMM_ID","",1);
}
