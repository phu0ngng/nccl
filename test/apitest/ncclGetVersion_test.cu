TEST(ncclGetVersion, basic) {
    int version;
    EXPECT_EQ(ncclSuccess, ncclGetVersion(&version));
    EXPECT_EQ(version, NCCL_VERSION_CODE);
}
TEST(ncclGetVersion, new_gt_old) {
    int version;
    EXPECT_EQ(ncclSuccess, ncclGetVersion(&version));
    EXPECT_GT(version, NCCL_VERSION(2,8,4));
}
TEST(ncclGetVersion, code_gt_old) {
    EXPECT_GT(NCCL_VERSION_CODE, NCCL_VERSION(2,8,4));
}
TEST(ncclGetVersion, code) {
    EXPECT_EQ(NCCL_VERSION_CODE, NCCL_VERSION(NCCL_MAJOR,NCCL_MINOR,NCCL_PATCH));
    EXPECT_GT(NCCL_VERSION(2,9,0), NCCL_VERSION(2,8,4));
    EXPECT_GT(NCCL_VERSION(2,10,0), NCCL_VERSION(2,9,99));
    EXPECT_GT(NCCL_VERSION(2,99,99), NCCL_VERSION(2,99,0));
    EXPECT_GT(NCCL_VERSION(3,0,0), NCCL_VERSION(2,99,99));
}
TEST(ncclGetVersion, macro) {
    int version;
    EXPECT_EQ(ncclSuccess, ncclGetVersion(&version));
    EXPECT_EQ(version, NCCL_VERSION(NCCL_MAJOR,NCCL_MINOR,NCCL_PATCH));
}
TEST(ncclGetVersion, macro_new_gt_old) {
    int version;
    EXPECT_EQ(ncclSuccess, ncclGetVersion(&version));
    EXPECT_GT(NCCL_VERSION(2,9,0), NCCL_VERSION(2,8,4));
}
TEST(ncclGetVersion, null) {
    EXPECT_EQ(ncclInvalidArgument, ncclGetVersion(NULL));
}
