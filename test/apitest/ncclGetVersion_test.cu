TEST(ncclGetVersion, basic) {
    int version;
    EXPECT_EQ(ncclSuccess, ncclGetVersion(&version));
    EXPECT_EQ(version, NCCL_VERSION_CODE);
}
TEST(ncclGetVersion, null) {
    EXPECT_EQ(ncclInvalidArgument, ncclGetVersion(NULL));
}
