#include "ncclCommon_test.cuh"

#include <map>
#include <regex.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>


// Tests the logging mechanism more than init rank.
class ncclDebugLogTest: public ncclOutputTest {
  protected:
    ncclUniqueId commId;
    int rank = 0;
    virtual void SetUp() override {
        ncclOutputTest::SetUp();
        overrideEnvVariable("NCCL_DEBUG_SUBSYS", "ALL");
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
    }
};

#define TIMESTAMP_NS_REGEX \
  "\\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{9}\\]"
#define TIMESTAMP_MS_REGEX \
  "\\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{3}\\]"
#define TIMESTAMP_REGEX \
  "\\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\\]"
#define REST_REGEX "[^ :]*:[0-9]+:[0-9]+ \\[[0-9]+\\]"
/* WARN: ... [cuda] init.cc:line (func) NCCL WARN ... ; INFO suffix: NCCL INFO init.cc:line (func) */
#define INIT_CC_WARN_TAIL " init\\.cc:[0-9]+ [(][^)]+[)] NCCL WARN improper usage of ncclCommInitRank: .*"
#define INIT_CC_INFO_TAIL " NCCL INFO init\\.cc:[0-9]+ [(][^)]+[)]"
#define HOST_PID_TID_ONLY "[^ ]+:[0-9]+:[0-9]+"
#define REG_POSIX_ERE REG_EXTENDED

TEST_F(ncclDebugLogTest, timestampOnWarn) {
    overrideEnvVariable("NCCL_DEBUG", "INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "WARN");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_FORMAT", "[%F %T.%9f] ");
    ncclResetDebugInitInternal();
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, 0, commId, rank));    // bad nranks=0
    verifyResult(".*\n" TIMESTAMP_NS_REGEX " " REST_REGEX INIT_CC_WARN_TAIL
                 "\n" REST_REGEX INIT_CC_INFO_TAIL ".*",
                 true, REG_POSIX_ERE);
}

TEST_F(ncclDebugLogTest, timestampOnInfo) {
    overrideEnvVariable("NCCL_DEBUG", "INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "INFO");
    ncclResetDebugInitInternal();
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, 0, commId, rank));
    verifyResult(".*\n" REST_REGEX INIT_CC_WARN_TAIL
                 "\n" TIMESTAMP_REGEX " " REST_REGEX INIT_CC_INFO_TAIL ".*",
                 true, REG_POSIX_ERE);
}

TEST_F(ncclDebugLogTest, timestampOnNotInfo) {
    overrideEnvVariable("NCCL_DEBUG", "INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "^INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_FORMAT", "[%F %T.%3f] ");
    ncclResetDebugInitInternal();
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, 0, commId, rank));
    verifyResult(".*\n" TIMESTAMP_MS_REGEX " " REST_REGEX INIT_CC_WARN_TAIL
                 "\n" REST_REGEX INIT_CC_INFO_TAIL ".*",
                 true, REG_POSIX_ERE);
}

TEST_F(ncclDebugLogTest, timestampOnNotWarn) {
    overrideEnvVariable("NCCL_DEBUG", "INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "^WARN");
    ncclResetDebugInitInternal();
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, 0, commId, rank));
    verifyResult(".*\n" REST_REGEX INIT_CC_WARN_TAIL
                 "\n" TIMESTAMP_REGEX " " REST_REGEX INIT_CC_INFO_TAIL ".*",
                 true, REG_POSIX_ERE);
}

TEST_F(ncclDebugLogTest, timestampOnAll) {
    overrideEnvVariable("NCCL_DEBUG", "INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "ALL");
    ncclResetDebugInitInternal();
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, 0, commId, rank));
    verifyResult(".*\n" TIMESTAMP_REGEX " " REST_REGEX INIT_CC_WARN_TAIL
                 "\n" TIMESTAMP_REGEX " " REST_REGEX INIT_CC_INFO_TAIL ".*",
                 true, REG_POSIX_ERE);
}

TEST_F(ncclDebugLogTest, timestampOnTrace) {
    overrideEnvVariable("NCCL_DEBUG", "TRACE");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "TRACE");
    ncclResetDebugInitInternal();
    ncclUniqueId id;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    verifyResult(".*" TIMESTAMP_REGEX " "
                 HOST_PID_TID_ONLY " NCCL CALL ncclGetUniqueId[(].*[)]\n.*",
                 true, REG_POSIX_ERE);
}

TEST_F(ncclDebugLogTest, timestampNotOnTrace) {
    overrideEnvVariable("NCCL_DEBUG", "TRACE");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "^TRACE");
    ncclResetDebugInitInternal();
    ncclUniqueId id;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    verifyResult(".*" HOST_PID_TID_ONLY " NCCL CALL ncclGetUniqueId[(].*[)]\n.*",
                 true, REG_POSIX_ERE);
}

