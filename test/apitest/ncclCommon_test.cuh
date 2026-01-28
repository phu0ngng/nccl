#include <stdint.h>
#include <gtest/gtest.h>
#include <nccl.h>
#pragma once
template <typename OP, typename DT>
void freePP(OP op, DT**& ptr, const int len) {
    if (ptr != NULL) {
        for (int i = 0; i < len; ++i) {
            EXPECT_EQ(cudaSuccess, cudaSetDevice(i));
            EXPECT_NO_FATAL_FAILURE(op(ptr[i]));
            ptr[i] = NULL;
        };
        free(ptr);
        ptr = NULL;
    };
};

// Persistent NCCL communicators for shared comm split
ncclComm_t* ncclCommon_getSplitShareComms();

// Persistent NCCL communicators and buffers
ncclComm_t* ncclCommon_getComms(int* nGpus);

// Persistent NCCL IB communicators and buffers
ncclComm_t* ncclCommon_getIBComms(int nGpus);

// Persistent NCCL Sockets communicators and buffers
ncclComm_t* ncclCommon_getSocketsComms(int nGpus);

// Persistent NCCL send/recv communicators
ncclComm_t* ncclCommon_getsrComms(int* nGpus);
void ncclCommon_destroysrComms();
void ncclCommon_destroySplitComms();
void ncclCommon_destroyComms();
void ncclCommon_destroyIBComms();
void ncclCommon_destroySocketComms();
void register_segv_handler();

void ncclCommon_getBuff(void*** sendbuffs, void*** recvbuffs, void*** sendbuffs_host, void*** recvbuffs_host, void*** sendbuffs_pinned, void*** recvbuffs_pinned, void*** sendbuffs_pinned_device, void*** recvbuffs_pinned_device, cudaStream_t** streams);

template <typename DT>
class ncclCommon_test : public ::testing::Test {
  public:
    static int N;
    static int nVis;
    static ncclComm_t* comms;
    static ncclComm_t* commsIB;
    static ncclComm_t* commsSockets;
    static DT **sendbuffs, **recvbuffs, //
        **sendbuffs_host, **recvbuffs_host, //
        **sendbuffs_pinned, **recvbuffs_pinned,
        **sendbuffs_pinned_device, **recvbuffs_pinned_device;
    static cudaStream_t* streams;
    static ncclDataType_t DataType();
    static void SetUpTestCase();
    static void TearDownTestCase();
    static const std::vector<ncclRedOp_t> RedOps;
  protected:
    int root = -1;
    void SetUp(){};
    void TearDown() {
        int done[nVis];
        int total = 0;
        for (int i = 0; i < this->nVis; ++i)
            done[i] = 0;
        while (total < nVis) {
            for (int i = 0; i < this->nVis; ++i) {
                EXPECT_EQ(cudaSuccess, cudaSetDevice(i));
                if (done[i])
                    continue;
                cudaError_t cudaErr = cudaStreamQuery(this->streams[i]);
                if (cudaErr != cudaErrorNotReady) {
                    EXPECT_EQ(cudaSuccess, cudaErr)
                        << "Rank : " << i << ". Error: " << cudaGetErrorName(cudaErr)
                        << "(" << cudaGetErrorString(cudaErr) << ")" << std::endl;
                    done[i] = 1;
                    total++;
                }
            }
        }
    };
};
template <typename DT>
const std::vector<ncclRedOp_t> ncclCommon_test<DT>::RedOps =
  {ncclSum, ncclProd, ncclMax, ncclMin, ncclAvg};
template <typename DT>
int ncclCommon_test<DT>::N = 4 * 1024 * 1024;
template <typename DT>
int ncclCommon_test<DT>::nVis = -1;
template <typename DT>
ncclComm_t* ncclCommon_test<DT>::comms = NULL;
template <typename DT>
ncclComm_t* ncclCommon_test<DT>::commsIB = NULL;
template <typename DT>
ncclComm_t* ncclCommon_test<DT>::commsSockets = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::sendbuffs = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::recvbuffs = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::sendbuffs_host = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::recvbuffs_host = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::sendbuffs_pinned = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::recvbuffs_pinned = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::sendbuffs_pinned_device = NULL;
template <typename DT>
DT** ncclCommon_test<DT>::recvbuffs_pinned_device = NULL;
template <typename DT>
cudaStream_t* ncclCommon_test<DT>::streams = NULL;
template <typename DT>
void ncclCommon_test<DT>::SetUpTestCase() {
    register_segv_handler();
    (void) setenv("NCCL_CHECK_POINTERS", "1", 0); // API tests expect this behaviour (ncclCommInitAll)
    comms = ncclCommon_getComms(&nVis);
    commsIB = ncclCommon_getIBComms(nVis);
    if (commsIB != NULL) {
        commsSockets = ncclCommon_getSocketsComms(nVis);
    }

    ncclCommon_getBuff(
        (void***)&sendbuffs,
        (void***)&recvbuffs,
        (void***)&sendbuffs_host,
        (void***)&recvbuffs_host,
        (void***)&sendbuffs_pinned,
        (void***)&recvbuffs_pinned,
        (void***)&sendbuffs_pinned_device,
        (void***)&recvbuffs_pinned_device,
        &streams);
};

static int commClean = -1;

template <typename DT>
void ncclCommon_test<DT>::TearDownTestCase() {
    if (commClean == -1) {
      char* str = getenv("NCCL_APITEST_COMM_CLEANUP");
      commClean = str ? atoi(str) : 0;
    }
    if (commClean) {
      ncclCommon_destroyComms();
      ncclCommon_destroyIBComms();
      ncclCommon_destroySocketComms();
      ncclCommon_destroySplitComms();
    }
    // Always clean those up, we don't use them everywhere
    ncclCommon_destroysrComms();
};
typedef ::testing::Types<char, int, half, float, double, long long,
                         unsigned long long>
    testDataTypes;
typedef ::testing::Types<char>
    testNoType;
// TYPED_TEST_CASE(ncclCommon_test, testDataTypes);

class ParameterChanger {
  // Changes the environment as directed during construction and
  // repairs it at destruction.
  public:
    ParameterChanger(const char* envVarName, const char* envVarValue)
        : name(envVarName), oldValue(nullptr) {
      const char* _oldValue = getenv(envVarName);
      oldValue = (_oldValue != nullptr) ? strdup(_oldValue) : nullptr;
      if (setenv(envVarName, envVarValue, 1)) {
        printf("FATAL: Could not set environment variable \"%s\""
               " to value \"%s\".\n",
               envVarName, envVarValue);
        exit(1);
      }
    }
    ~ParameterChanger() {
      if (oldValue != nullptr) {
        if (setenv(name.c_str(), oldValue, 1)) {
          printf("FATAL: Could not set environment variable \"%s\""
                 " to \"%s\" when restoring.\n", name.c_str(), oldValue);
          exit(1);
        }
        free(oldValue);
        oldValue = nullptr;
      } else {
        if (unsetenv(name.c_str())) {
          printf("FATAL: Could not unset environment variable \"%s\""
                 " when restoring.\n", name.c_str());
          exit(1);
        }
      }
    }
    // Note that this does not support move semantics, so be careful
    // putting it in a vector.
  private:
    const std::string name;
    char* oldValue; // char* instead of string so it can be null
};

class ncclShelveEnvTest : public ::testing::Test {
  // Allows testing NCCL when an environment variable needs to be temporarily changed.
  // Note: Parameters that use NCCL_PARAM are cached upon first access.
  private:
    std::map<std::string, std::string> oldValues;
  protected:
    virtual void SetUp() override {
        ::testing::Test::SetUp();
    }
    virtual void TearDown() override {
        for (auto iter : oldValues) {
            if (iter.second.empty()) {
                unsetenv(iter.first.c_str());
            } else {
                if (setenv(iter.first.c_str(), iter.second.c_str(), 1)) {
                    printf("FATAL: Could not set environment variable \"%s\" to value \"%s\".\n", iter.first.c_str(), iter.second.c_str());
                    exit(1);
                }
            }
        }
        ::testing::Test::TearDown();
    }
    void overrideEnvVariable(const char* envVarName, const char* newEnvVarValue) {
        const char* curValue = getenv(envVarName);
        if (oldValues.find(envVarName) == oldValues.end()) {
            oldValues[envVarName] = (curValue==NULL) ? "" : curValue;
        }
        if (newEnvVarValue) {
            if (setenv(envVarName, newEnvVarValue, 1)) {
                printf("FATAL: Could not set environment variable \"%s\" to value \"%s\".\n", envVarName, newEnvVarValue);
                exit(1);
            }
        } else {
            if (unsetenv(envVarName)) {
                printf("FATAL: Could not unset environment variable \"%s\".\n", envVarName);
                exit(1);
            }
        }
    }
};

extern "C"
void  ncclResetDebugInitInternal();

class ncclOutputTest : public ncclShelveEnvTest {
  // This class reroutes the output to a file so that the results can be verified against a regex.
  protected:
    char logFileName[PATH_MAX];

    virtual void SetUp() override {
        ncclShelveEnvTest::SetUp();
        snprintf(logFileName, sizeof(logFileName), "/tmp/test_log_%d.tmp", getpid());
        overrideEnvVariable("NCCL_DEBUG", "INFO");
        overrideEnvVariable("NCCL_DEBUG_SUBSYS", "ENV");
        overrideEnvVariable("NCCL_DEBUG_FILE", logFileName);
        ncclResetDebugInitInternal();
    }
    virtual void TearDown() override {
        remove(logFileName);
        ncclShelveEnvTest::TearDown();
        ncclResetDebugInitInternal();
    }
    void verifyResult(const char* expectedRegex, bool expectMatch=true, int regexCompFlags=0, int regexExecFlags=0) {
        // This reads the entire file and searches for expectedRegex.
        // Note that C++ Regexes are problematic. "\d" does not work (not even "\\d"). "[0-9]" will work.
        // "+" must be escaped: "[0-9]\\+".
        FILE* f = fopen(logFileName, "rt");
        ASSERT_NE(f, nullptr) << "Could not open log file used by test: " << logFileName;

        fseek(f, 0, SEEK_END);
        size_t len = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<char> buffer(len+1);
        size_t len_read = fread(buffer.data(), 1, len, f);
        buffer[len_read] = '\0';
        fclose(f);
        ASSERT_EQ(len_read, len) << "FATAL: read " << len_read << " bytes from file, but was trying to read " << len << ".";

        regex_t reg;
        int e = regcomp(&reg, expectedRegex, regexCompFlags);
        if (e) {
          regfree(&reg);
        }
        ASSERT_EQ(e, 0) << "regcomp returned " << e << "... this indicates the test itself is broken.";

        regmatch_t match;
        int c = regexec(&reg, buffer.data(), 1, &match, regexExecFlags);
        regfree(&reg);
        if (expectMatch) {
            EXPECT_EQ(c, 0) << "Failed to match \"" << expectedRegex << "\" to \"" << buffer.data() << "\"";
        } else {
            EXPECT_NE(c, 0) << "Failed to NOT match \"" << expectedRegex << "\" to \"" << buffer.data() << "\"";
        }
    }
};

// EOF
