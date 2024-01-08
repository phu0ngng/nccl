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
// EOF
