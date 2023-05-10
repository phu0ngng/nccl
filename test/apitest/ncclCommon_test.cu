#include "ncclCommon_test.cuh"
// these are Template specialization
#define GEN_DATATYPE(X, Y)                                                     \
    template <>                                                                \
    ncclDataType_t ncclCommon_test<X>::DataType() {                            \
        return (Y);                                                            \
    };
GEN_DATATYPE(char, ncclChar);
GEN_DATATYPE(half, ncclHalf);
GEN_DATATYPE(int, ncclInt);
GEN_DATATYPE(float, ncclFloat);
GEN_DATATYPE(double, ncclDouble);
GEN_DATATYPE(long long, ncclInt64);
GEN_DATATYPE(unsigned long long, ncclUint64);

int totalGpus = 0;
ncclComm_t* commsArray = NULL;
ncclComm_t* splitCommsArray = NULL;
ncclComm_t* commsIBArray = NULL;
ncclComm_t* commsSocketsArray = NULL;
ncclComm_t* srCommsArray = NULL;
bool srCommsInit = false;
bool initialized = false;

ncclComm_t* ncclCommon_getSplitShareComms() {
  if (splitCommsArray == NULL) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclUniqueId id;
    EXPECT_EQ(cudaSuccess, cudaGetDeviceCount(&totalGpus));
    EXPECT_NE(nullptr, splitCommsArray = (ncclComm_t*)calloc(sizeof(ncclComm_t), totalGpus));
    config.splitShare = 1;
    EXPECT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    EXPECT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < totalGpus; ++i) {
        EXPECT_EQ(cudaSuccess, cudaSetDevice(i));
        EXPECT_EQ(ncclSuccess, ncclCommInitRankConfig(&splitCommsArray[i], totalGpus, id, i, &config));
    }
    EXPECT_EQ(ncclSuccess, ncclGroupEnd());
  }
  return splitCommsArray;
}

ncclComm_t* ncclCommon_getComms(int* nGpus) {
  if (commsArray == NULL) {
    EXPECT_EQ(cudaSuccess, cudaGetDeviceCount(&totalGpus));
    EXPECT_NE(nullptr, commsArray = (ncclComm_t*)calloc(sizeof(ncclComm_t), totalGpus));
    EXPECT_EQ(ncclSuccess, ncclCommInitAll(commsArray, totalGpus, NULL));
  }
  *nGpus = totalGpus;
  return commsArray;
}

ncclComm_t* ncclCommon_getsrComms(int* nGpus) {
  if (srCommsArray == NULL && !srCommsInit) {
    srCommsInit = true;
    EXPECT_EQ(cudaSuccess, cudaGetDeviceCount(&totalGpus));
    EXPECT_NE(nullptr, srCommsArray = (ncclComm_t*)calloc(sizeof(ncclComm_t), totalGpus));
    EXPECT_EQ(ncclSuccess, ncclCommInitAll(srCommsArray, totalGpus, NULL));
  }
  *nGpus = totalGpus;
  return srCommsArray;
}

void ncclCommon_destroysrComms() {
  if (srCommsArray != NULL) {
    for (int i = 0; i < totalGpus; ++i) {
      EXPECT_EQ(ncclSuccess, ncclCommDestroy(srCommsArray[i]));  
    }
    free(srCommsArray);
    srCommsArray = NULL;
  }
  return;
}

ncclComm_t* ncclCommon_getIBComms(int nGpus) {
  if (commsIBArray == NULL && !initialized) {
    initialized = true;
    EXPECT_NE(nullptr, commsIBArray = (ncclComm_t*)calloc(sizeof(ncclComm_t), nGpus));
    (void) setenv("NCCL_NET", "IB", 1);
    if (ncclCommInitAll(commsIBArray, nGpus, NULL) != ncclSuccess) {
        std::cerr << "ncclGroupEnd() failed when trying to init IB communicators. Skipping multi_net tests." << std::endl;
        // This platform doesn't have IB network, so mark the test as skipped here
        free(commsIBArray);
        commsIBArray = NULL;
    }
  }

  (void) unsetenv("NCCL_NET");
  return commsIBArray;
}

ncclComm_t* ncclCommon_getSocketsComms(int nGpus) {
  if (commsSocketsArray == NULL) {
    EXPECT_NE(nullptr, commsSocketsArray = (ncclComm_t*)calloc(sizeof(ncclComm_t), nGpus));
    (void) setenv("NCCL_NET", "Socket", 1);
    EXPECT_EQ(ncclSuccess, ncclCommInitAll(commsSocketsArray, nGpus, NULL));
  }

  (void) unsetenv("NCCL_NET");
  return commsSocketsArray;
}

void** sbuffs = NULL;
void** rbuffs;
void** sbuffs_host;
void** rbuffs_host;
void** sbuffs_pinned;
void** rbuffs_pinned;
void** sbuffs_pinned_device;
void** rbuffs_pinned_device;
cudaStream_t* cuda_streams;

static int maxsize = 4 * 1024 * 1024 * sizeof(uint64_t);

void ncclCommon_getBuff(void*** sendbuffs, void*** recvbuffs, void*** sendbuffs_host, void*** recvbuffs_host, void*** sendbuffs_pinned, void*** recvbuffs_pinned, void*** sendbuffs_pinned_device, void*** recvbuffs_pinned_device, cudaStream_t** streams) {
  if (sbuffs == NULL) {
    cuda_streams = (cudaStream_t*)calloc(totalGpus, sizeof(cudaStream_t));
    sbuffs = (void**)calloc(totalGpus, sizeof(void*));
    rbuffs = (void**)calloc(totalGpus, sizeof(void*));
    sbuffs_host = (void**)calloc(totalGpus, sizeof(void*));
    rbuffs_host = (void**)calloc(totalGpus, sizeof(void*));
    sbuffs_pinned = (void**)calloc(totalGpus, sizeof(void*));
    rbuffs_pinned = (void**)calloc(totalGpus, sizeof(void*));
    sbuffs_pinned_device = (void**)calloc(totalGpus, sizeof(void*));
    rbuffs_pinned_device = (void**)calloc(totalGpus, sizeof(void*));
    for (int i = 0; i < totalGpus; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&sbuffs[i], maxsize));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&rbuffs[i], maxsize));
        ASSERT_EQ(cudaSuccess, cudaMemset(sbuffs[i], 0, maxsize));
        ASSERT_EQ(cudaSuccess, cudaMemset(rbuffs[i], 0, maxsize));
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&cuda_streams[i])) << i;
        sbuffs_host[i] = calloc(1, maxsize);
        rbuffs_host[i] = calloc(1, maxsize);
        sbuffs_pinned[i] = calloc(1, maxsize);
        ASSERT_EQ(cudaSuccess,
                  cudaHostRegister(sbuffs_pinned[i], maxsize,
                                   cudaHostRegisterDefault));
        ASSERT_EQ(cudaSuccess, cudaHostGetDevicePointer(&sbuffs_pinned_device[i], 
				   sbuffs_pinned[i], 0));
        rbuffs_pinned[i] = calloc(1, maxsize);
        ASSERT_EQ(cudaSuccess,
                  cudaHostRegister(rbuffs_pinned[i], maxsize,
                                   cudaHostRegisterDefault));
        ASSERT_EQ(cudaSuccess, cudaHostGetDevicePointer(&rbuffs_pinned_device[i], 
				   rbuffs_pinned[i], 0));
    }
  }
  *sendbuffs = sbuffs;
  *recvbuffs = rbuffs;
  *sendbuffs_host = sbuffs_host;
  *recvbuffs_host = rbuffs_host;
  *sendbuffs_pinned = sbuffs_pinned;
  *recvbuffs_pinned = rbuffs_pinned;
  *sendbuffs_pinned_device = sbuffs_pinned_device;
  *recvbuffs_pinned_device = rbuffs_pinned_device;
  *streams = cuda_streams;
}

#undef GEN_DATATYPE
