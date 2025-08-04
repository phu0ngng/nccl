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

#define TIMESTAMP_NS_REGEX "\\[....-..-.. ..:..:..\\..........\\]"
#define TIMESTAMP_MS_REGEX "\\[....-..-.. ..:..:..\\....\\]"
#define TIMESTAMP_REGEX "\\[....-..-.. ..:..:..\\]"
#define REST_REGEX "[^ :]*:[0-9]*:[0-9]* \\[[0-9]*\\]"

TEST_F(ncclDebugLogTest, timestampOnWarn) {
    overrideEnvVariable("NCCL_DEBUG", "INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "WARN");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_FORMAT", "[%F %T.%9f] ");
    ncclResetDebugInitInternal();
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, 0, commId, rank));    // bad nranks=0
    verifyResult(".*\n" TIMESTAMP_NS_REGEX " " REST_REGEX " init.cc:[0-9]* NCCL WARN improper usage of ncclCommInitRank: .*"
                 "\n" REST_REGEX " NCCL INFO init.cc.*");
}

TEST_F(ncclDebugLogTest, timestampOnInfo) {
    overrideEnvVariable("NCCL_DEBUG", "INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "INFO");
    ncclResetDebugInitInternal();
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, 0, commId, rank));
    verifyResult(".*\n" REST_REGEX " init.cc:[0-9]* NCCL WARN improper usage of ncclCommInitRank: .*"
                 "\n" TIMESTAMP_REGEX " " REST_REGEX " NCCL INFO init.cc.*");
}

TEST_F(ncclDebugLogTest, timestampOnNotInfo) {
    overrideEnvVariable("NCCL_DEBUG", "INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "^INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_FORMAT", "[%F %T.%3f] ");
    ncclResetDebugInitInternal();
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, 0, commId, rank));
    verifyResult(".*\n" TIMESTAMP_MS_REGEX " " REST_REGEX " init.cc:[0-9]* NCCL WARN improper usage of ncclCommInitRank: .*"
                 "\n" REST_REGEX " NCCL INFO init.cc.*");
}

TEST_F(ncclDebugLogTest, timestampOnNotWarn) {
    overrideEnvVariable("NCCL_DEBUG", "INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "^WARN");
    ncclResetDebugInitInternal();
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, 0, commId, rank));
    verifyResult(".*\n" REST_REGEX " init.cc:[0-9]* NCCL WARN improper usage of ncclCommInitRank: .*"
                 "\n" TIMESTAMP_REGEX " " REST_REGEX " NCCL INFO init.cc.*");
}

TEST_F(ncclDebugLogTest, timestampOnAll) {
    overrideEnvVariable("NCCL_DEBUG", "INFO");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "ALL");
    ncclResetDebugInitInternal();
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRank(NULL, 0, commId, rank));
    verifyResult(".*\n" TIMESTAMP_REGEX " " REST_REGEX " init.cc:[0-9]* NCCL WARN improper usage of ncclCommInitRank: .*"
                 "\n" TIMESTAMP_REGEX " " REST_REGEX " NCCL INFO init.cc.*");
}

TEST_F(ncclDebugLogTest, timestampOnTrace) {
    overrideEnvVariable("NCCL_DEBUG", "TRACE");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "TRACE");
    ncclResetDebugInitInternal();
    ncclUniqueId id;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    verifyResult(".*" TIMESTAMP_REGEX " " "[^ :]*:[0-9]*:[0-9]*" " NCCL CALL ncclGetUniqueId\\(.*\\)\n.*");
}

TEST_F(ncclDebugLogTest, timestampNotOnTrace) {
    overrideEnvVariable("NCCL_DEBUG", "TRACE");
    overrideEnvVariable("NCCL_DEBUG_TIMESTAMP_LEVELS", "^TRACE");
    ncclResetDebugInitInternal();
    ncclUniqueId id;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    verifyResult(".*" "[^ :]*:[0-9]*:[0-9]*" " NCCL CALL ncclGetUniqueId\\(.*\\)\n.*");
}

