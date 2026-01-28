#include "ncclCommon_test.cuh"
#include <cstdint>
#include <execinfo.h>
#include <stdint.h>
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
bool initialized = false;
bool handleRegistered = false;
bool segvLogPrinted = false;

// Turn off caching for Multi-rank GPU. Additional NCCL_PARAMS can be turned off,
// but any general disabling should be discussed widely.
ParameterChanger gNoCaching("NCCL_NO_CACHE", "NCCL_MULTI_RANK_GPU_ENABLE");

static void destroyComms(ncclComm_t** array) {
  ncclComm_t* a = *array;
  if (a != NULL) {
    for (int i = 0; i < totalGpus; ++i) {
      EXPECT_EQ(ncclSuccess, ncclCommDestroy(a[i]));
    }
    free(a);
    *array = NULL;
  }
}

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

void ncclCommon_destroySplitComms() { destroyComms(&splitCommsArray); }

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void segfault_handler(int sig) {
  void *log[32];
  size_t size;

  pthread_mutex_lock(&mutex);
  if (segvLogPrinted == false) {
    size = backtrace(log, 32);
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(log, size, STDERR_FILENO);
    segvLogPrinted = true;
  }
  pthread_mutex_unlock(&mutex);

  exit(1);
}

void register_segv_handler() {
  if (handleRegistered == false) {
    /* register segfault handler */
    signal(SIGSEGV, segfault_handler);
    handleRegistered = true;
  }
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


void ncclCommon_destroyComms() { destroyComms(&commsArray); }

ncclComm_t* ncclCommon_getsrComms(int* nGpus) {
  if (srCommsArray == NULL) {
    EXPECT_EQ(cudaSuccess, cudaGetDeviceCount(&totalGpus));
    EXPECT_NE(nullptr, srCommsArray = (ncclComm_t*)calloc(sizeof(ncclComm_t), totalGpus));
    EXPECT_EQ(ncclSuccess, ncclCommInitAll(srCommsArray, totalGpus, NULL));
  }
  *nGpus = totalGpus;
  return srCommsArray;
}

void ncclCommon_destroysrComms() { destroyComms(&srCommsArray); }

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

void ncclCommon_destroyIBComms() { destroyComms(&commsIBArray); }

ncclComm_t* ncclCommon_getSocketsComms(int nGpus) {
  if (commsSocketsArray == NULL) {
    EXPECT_NE(nullptr, commsSocketsArray = (ncclComm_t*)calloc(sizeof(ncclComm_t), nGpus));
    (void) setenv("NCCL_NET", "Socket", 1);
    EXPECT_EQ(ncclSuccess, ncclCommInitAll(commsSocketsArray, nGpus, NULL));
  }

  (void) unsetenv("NCCL_NET");
  return commsSocketsArray;
}

void ncclCommon_destroySocketComms() { destroyComms(&commsSocketsArray); }

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
  static bool alloc_pinned_buf = true;
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
        if (alloc_pinned_buf && cudaHostRegister(sbuffs_pinned[i], maxsize, cudaHostRegisterDefault) != cudaSuccess || cudaHostGetDevicePointer(&sbuffs_pinned_device[i], sbuffs_pinned[i], 0) != cudaSuccess) {
          free(sbuffs_pinned[i]);
          sbuffs_pinned[i] = NULL;
          alloc_pinned_buf = false;
        }

        rbuffs_pinned[i] = calloc(1, maxsize);
        if (alloc_pinned_buf && cudaHostRegister(rbuffs_pinned[i], maxsize, cudaHostRegisterDefault) != cudaSuccess || cudaHostGetDevicePointer(&rbuffs_pinned_device[i], rbuffs_pinned[i], 0) != cudaSuccess) {
          free(rbuffs_pinned[i]);
          rbuffs_pinned[i] = NULL;
          alloc_pinned_buf = false;
        }
    }
  }
  *sendbuffs = sbuffs;
  *recvbuffs = rbuffs;
  *sendbuffs_host = sbuffs_host;
  *recvbuffs_host = rbuffs_host;
  if (alloc_pinned_buf) {
    *sendbuffs_pinned = sbuffs_pinned;
    *recvbuffs_pinned = rbuffs_pinned;
    *sendbuffs_pinned_device = sbuffs_pinned_device;
    *recvbuffs_pinned_device = rbuffs_pinned_device;
  } else {
    *sendbuffs_pinned = NULL;
    *recvbuffs_pinned = NULL;
    *sendbuffs_pinned_device = NULL;
    *recvbuffs_pinned_device = NULL;
    free(sbuffs_pinned);
    free(rbuffs_pinned);
    free(sbuffs_pinned_device);
    free(rbuffs_pinned_device);
    sbuffs_pinned = NULL;
    rbuffs_pinned = NULL;
    sbuffs_pinned_device = NULL;
    rbuffs_pinned_device = NULL;
  }

  *streams = cuda_streams;
}

#undef GEN_DATATYPE
