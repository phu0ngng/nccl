TEST(ncclGetErrorString, basic) {
    EXPECT_STREQ("no error", ncclGetErrorString(ncclSuccess));
    EXPECT_STREQ("unhandled cuda error (run with NCCL_DEBUG=INFO for details)", ncclGetErrorString(ncclUnhandledCudaError));
    EXPECT_STREQ("unhandled system error (run with NCCL_DEBUG=INFO for details)", ncclGetErrorString(ncclSystemError));
    EXPECT_STREQ("internal error - please report this issue to the NCCL developers", ncclGetErrorString(ncclInternalError));
    EXPECT_STREQ("invalid argument (run with NCCL_DEBUG=WARN for details)", ncclGetErrorString(ncclInvalidArgument));
    EXPECT_STREQ("invalid usage (run with NCCL_DEBUG=WARN for details)", ncclGetErrorString(ncclInvalidUsage));
    EXPECT_STREQ("remote process exited or there was a network error", ncclGetErrorString(ncclRemoteError));
    EXPECT_STREQ("NCCL operation in progress", ncclGetErrorString(ncclInProgress));
    EXPECT_STREQ("unknown result code", ncclGetErrorString((ncclResult_t)-1));
};
