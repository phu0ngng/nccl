/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#include "nccl.h"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#endif
#include <stdio.h>
#include <cstdint>
#include <algorithm>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
#include <pthread.h>
#include "nccl1_compat.h"
#include "timer.h"
#include <cuda.h>

// For nccl.h < 2.13 since we define a weak fallback
extern "C" char const* ncclGetLastError(ncclComm_t comm);

#define HOST_RMA_IMPL 10

#define CUCHECK(cmd) do {                           \
  CUresult err = cmd;                               \
  if( err != CUDA_SUCCESS ) {                       \
    char hostname[1024];                            \
    const char *errStr;                             \
    cuGetErrorString(err, &errStr);                 \
    getHostName(hostname, 1024);                    \
    printf("%s: Test CU failure %s:%d '%s'\n",      \
         hostname,                                  \
        __FILE__,__LINE__,errStr);                  \
    return testCudaError;                           \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if( err != cudaSuccess ) {                        \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test CUDA failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    return testCudaError;                           \
  }                                                 \
} while(0)

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,12,10)
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d "           \
           "'%s / %s'\n",                           \
           hostname,__FILE__,__LINE__,              \
           ncclGetErrorString(res),                 \
           ncclGetLastError(NULL));                 \
    return testNcclError;                           \
  }                                                 \
} while(0)
#else
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    return testNcclError;                           \
  }                                                 \
} while(0)
#endif

typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testCudaError = 2,
  testNcclError = 3,
  testTimeout = 4,
  testNotImplemented = 5,
  testSkipped = 6,
  testParameterizationError = 7,
  testNumResults = 8
} testResult_t;

// Relay errors up and trace
#define TESTCHECK(cmd) do {                         \
  testResult_t r = cmd;                             \
  if (r == testSkipped) {                           \
    return testSkipped;                             \
  }                                                 \
  if (r!= testSuccess) {                            \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf(" .. %s pid %d: Test failure %s:%d\n",   \
         hostname, getpid(),                        \
        __FILE__,__LINE__);                         \
    return r;                                       \
  }                                                 \
} while(0)

struct testColl {
  const char name[20];
  void (*getCollByteCount)(
      size_t *sendcount, size_t *recvcount, size_t *paramcount,
      size_t *sendInplaceOffset, size_t *recvInplaceOffset,
      size_t count, size_t eltSize, int nranks);
  void (*initConfig)(ncclConfig_t* config);
  testResult_t (*initData)(struct threadArgs* args, ncclDataType_t type,
      ncclRedOp_t op, int root, int rep, int in_place);
  void (*getBw)(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks);
  testResult_t (*runColl)(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset,
      size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int implIndex);
};
extern struct testColl allReduceTest;
extern struct testColl allGatherTest;
extern struct testColl reduceScatterTest;
extern struct testColl broadcastTest;
extern struct testColl reduceTest;
extern struct testColl alltoAllTest;

struct testEngine {
  void (*getBuffSize)(size_t *sendcount, size_t *recvcount, size_t count, int nranks);
  testResult_t (*runTest)(struct threadArgs* args, int root, ncclDataType_t type,
      const char* typeName, ncclRedOp_t op, const char* opName);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
  testResult_t (*getDevCommRequirements)(int deviceImpl, ncclDevCommRequirements_t* reqs, ncclComm_t comm, const char** testSkipReason);
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  bool (*getDevCommRequirements)(int deviceImpl, ncclDevCommRequirements_t* reqs);
#endif
};

extern struct testEngine ncclTestEngine;

struct threadArgs {
  size_t** nbytes;
  size_t minbytes;
  size_t maxbytes;
  size_t stepbytes;
  size_t stepfactor;

  int totalProcs;
  int globalProc;
  int nProcs;
  int proc;
  int nThreads;
  int thread;
  int nGpus;
  int* gpus;
  int localRank;
  int commNum;
  void*** sendbuffs;
  size_t** sendBytes;
  size_t** sendInplaceOffset;
  void*** recvbuffs;
  size_t** recvInplaceOffset;
  int nIds;
  ncclUniqueId* ncclId;
  ncclComm_t** comms;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  ncclDevComm** devComms;
#endif
  cudaStream_t* streams;
  cudaEvent_t* events;
  float* ms;

  void*** expected;
  size_t** expectedBytes;
  int* errors;
  double* bw;
  int* bw_count;

  double meanTime;
  double meanAlgBw;
  double meanBusBw;
  int reportErrors;

  int compThreadStop;
  volatile int* compThreadCount;
  int compThreadCountLast;
  char* replayFile;

  struct testColl* collTest;
  int sleepId;
  void** hostbuffs;

  int64_t* initGpuMem;
  int64_t* bufferMemory;
  int64_t* devMemUsed;

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
  void*** sendRegHandles;
  void*** recvRegHandles;
#endif
};

typedef testResult_t (*threadFunc_t)(struct threadArgs* args);
struct testThread {
  pthread_t thread;
  threadFunc_t func;
  struct threadArgs args;
  testResult_t ret;
};

// Provided by common.cu
extern void Barrier(struct threadArgs* args);
extern testResult_t barrierRmaSignal(ncclComm_t comm, cudaStream_t stream);
extern int minCudaArch;  // Minimum CUDA architecture across all GPUs in the test
void setTestSkipReason(const char* reason);


extern testResult_t TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op,  const char* opName, int root);
extern testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op, const uint64_t seed, const int nranks);
extern testResult_t InitData(void* data, const size_t count, size_t offset, ncclDataType_t type, ncclRedOp_t op, const uint64_t seed, const int nranks, const int rank);
extern testResult_t AllocateBuffs(void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes, void **expected, size_t nbytes, size_t *allocBytes);

#include <unistd.h>

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i < maxlen; i++) {
    if (hostname[i] == '\0') {
      return;
    }
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

#include <stdint.h>

static uint64_t getHash(const char* string, size_t n) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (size_t c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t getHostHash(const char* hostname) {
  char hostHash[1024];

  // Fall back is the hostname if something fails
  (void) strncpy(hostHash, hostname, sizeof(hostHash));
  int offset = strlen(hostHash);

  FILE *file = fopen(HOSTID_FILE, "r");
  if (file != NULL) {
    char *p;
    if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
        free(p);
    }
  }
  fclose(file);

  // Make sure the string is terminated
  hostHash[sizeof(hostHash)-1]='\0';

  return getHash(hostHash, strlen(hostHash));
}

#define HAVE_BF16 0
#define HAVE_FP8 0

#if NCCL_MAJOR >= 2
  #if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
    #undef HAVE_BF16
    #define HAVE_BF16 1
    #if defined(__CUDA_FP8_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,24,0)
      #undef HAVE_FP8
      #define HAVE_FP8 1
    #endif
  #endif
#endif

// Check compile-time macros to determine which fp8 multimem architectures are available
// These macros are set by the build system (Makefile/CMake) based on NVCC_GENCODE
// Priority matches multimem__funcs.h: sm_100f > sm_100a > sm_101f > sm_101a > sm_120a > sm_121a
#if defined(NCCL_BUILD_SM100F) && NCCL_BUILD_SM100F == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM100F 1
#elif defined(NCCL_BUILD_SM100A) && NCCL_BUILD_SM100A == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM100A 1
#elif defined(NCCL_BUILD_SM101F) && NCCL_BUILD_SM101F == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM101F 1
#elif defined(NCCL_BUILD_SM101A) && NCCL_BUILD_SM101A == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM101A 1
#elif defined(NCCL_BUILD_SM120A) && NCCL_BUILD_SM120A == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM120A 1
#elif defined(NCCL_BUILD_SM121A) && NCCL_BUILD_SM121A == 1
  #define FP8_MULTIMEM_ARCH_AVAILABLE 1
  #define FP8_MULTIMEM_ARCH_SM121A 1
#else
  #define FP8_MULTIMEM_ARCH_AVAILABLE 0
#endif

static size_t wordSize(ncclDataType_t type) {
  switch(type) {
    case ncclChar:
#if NCCL_MAJOR >= 2
    //case ncclInt8:
    case ncclUint8:
#endif
#if HAVE_FP8
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
#endif
      return 1;
    case ncclHalf:
#if HAVE_BF16
    case ncclBfloat16:
#endif
    //case ncclFloat16:
      return 2;
    case ncclInt:
    case ncclFloat:
#if NCCL_MAJOR >= 2
    //case ncclInt32:
    case ncclUint32:
    //case ncclFloat32:
#endif
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclDouble:
    //case ncclFloat64:
      return 8;
    default: return 0;
  }
}

extern int test_ncclVersion; // init'd with ncclGetVersion()
extern int deviceCtaCount; // number of CTAs for device implementation
constexpr int test_opNumMax = (int)ncclNumOps + (NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0) ? 1 : 0);
extern int test_opnum;
extern int test_typenum;
extern ncclDataType_t test_types[ncclNumTypes];
extern const char *test_typenames[ncclNumTypes];
extern ncclRedOp_t test_ops[];
extern const char *test_opnames[];

static int ncclstringtotype(char *str) {
    for (int t=0; t<test_typenum; t++) {
      if (strcmp(str, test_typenames[t]) == 0) {
        return t;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid type %s, defaulting to %s .. \n", str, test_typenames[ncclFloat]);
    return ncclFloat;
}

static int ncclstringtoop (char *str) {
    for (int o=0; o<test_opnum; o++) {
      if (strcmp(str, test_opnames[o]) == 0) {
        return o;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid op %s, defaulting to %s .. \n", str, test_opnames[ncclSum]);
    return ncclSum;
}

extern int is_main_proc;
extern thread_local int is_main_thread;

/* If NCCL version is not smaller than 2.14.0, we support nonblocking
 * communicator where the state of a communicator can be ncclInProgress.
 * We define NCCLCHECK_COMM_WAIT and NCCLCHECK_COMM_WAITBATCH macro for
 * convenience to wait on inprogress communicators. */
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
static testResult_t waitCommState(ncclComm_t comm) {
  ncclResult_t state;
  do {
    NCCLCHECK(ncclCommGetAsyncError(comm, &state));
#ifdef MPI_SUPPORT
    int flag;
    extern pthread_mutex_t mpiLock;
    pthread_mutex_lock(&mpiLock);
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    pthread_mutex_unlock(&mpiLock);
#endif
  } while (state == ncclInProgress);
  if (state != ncclSuccess) return testNcclError;
  return testSuccess;
}

static testResult_t waitCommStateBatch(ncclComm_t * comms, int num) {
  ncclResult_t state;
  for (int idx = 0; idx < num; ++idx) {
    do {
      NCCLCHECK(ncclCommGetAsyncError(comms[idx], &state));
#ifdef MPI_SUPPORT
      int flag;
      extern pthread_mutex_t mpiLock;
      pthread_mutex_lock(&mpiLock);
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      pthread_mutex_unlock(&mpiLock);
#endif
    } while (state == ncclInProgress);
    if (state != ncclSuccess) return testNcclError;
  }
  return testSuccess;
}

#define NCCLCHECK_COMM_WAIT(cmd, comm) do {           \
  ncclResult_t res = cmd;                             \
  if (res == ncclInProgress) {                        \
    TESTCHECK(waitCommState(comm));                   \
  } else if (res != ncclSuccess) {                    \
    char hostname[1024];                              \
    getHostName(hostname, 1024);                      \
    printf("%s: Test NCCL failure %s:%d '%s'\n",      \
         hostname,                                    \
        __FILE__,__LINE__,ncclGetErrorString(res));   \
    return testNcclError;                             \
  }                                                   \
} while(0)

#define NCCLCHECK_COMM_WAITBATCH(cmd, comms, num) do {        \
  ncclResult_t res = cmd;                                     \
  if (res == ncclInProgress) {                                \
    TESTCHECK(waitCommStateBatch(comms, num));                \
  } else if (res != ncclSuccess) {                            \
    char hostname[1024];                                      \
    getHostName(hostname, 1024);                              \
    printf("%s: Test NCCL failure %s:%d '%s'\n",              \
         hostname,                                            \
        __FILE__,__LINE__,ncclGetErrorString(res));           \
    return testNcclError;                                     \
  }                                                           \
} while(0)

#else /* #if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0) */

#define NCCLCHECK_COMM_WAIT(cmd, comm) do {           \
  NCCLCHECK(cmd);                                     \
} while(0)

#define NCCLCHECK_COMM_WAITBATCH(cmd, comms, num) do {      \
  NCCLCHECK(cmd);                                           \
} while(0)
#endif

testResult_t faultToleranceTests(int nThreads, int nGpus, int ncclProc, int ncclProcs, int localRank, const char* ft_list);
testResult_t threadLaunch(struct testThread* thread);

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
template <typename F>
testResult_t testLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, bool isMultimem = false) {
  // Check if multimem is requested but architecture doesn't support it (requires sm_90 or higher)
  if (isMultimem && minCudaArch < 90) {
    setTestSkipReason("multimem not supported on this system (requires sm_90+)\n");
    return testSkipped;
  }
  // FP8 multimem kernel availability is checked at compile-time via FP8_MULTIMEM_ARCH_AVAILABLE
  // If kernels weren't compiled for the current architecture, SPECIALIZE_KERNEL_MULTIMEM will return nullptr

  if (kernel == nullptr) {
    if (isMultimem) {
      if (op != ncclSum) {
        setTestSkipReason("multimem kernels only support sum reduction\n");
      } else if (type == ncclInt8 || type == ncclUint8) {
        setTestSkipReason("multimem kernels do not support int8/uint8\n");
      }
#if !HAVE_BF16
      else if (type == ncclBfloat16) {
        setTestSkipReason("BF16 not supported in this build\n");
      }
#endif
#if !HAVE_FP8
      else if (type == ncclFloat8e4m3 || type == ncclFloat8e5m2) {
        setTestSkipReason("FP8 not supported in this build\n");
      }
#elif !FP8_MULTIMEM_ARCH_AVAILABLE
      else if (type == ncclFloat8e4m3 || type == ncclFloat8e5m2) {
        setTestSkipReason("FP8 multimem not built for this architecture\n");
      }
#endif
      else {
        setTestSkipReason("multimem kernel not available for this type\n");
      }
    } else {
      setTestSkipReason("kernel not available for this type/op\n");
    }
    return testSkipped;
  }
  ncclDevComm* devComm = (ncclDevComm*)comm;

  ncclWindow_t sendwin = (ncclWindow_t)sendbuff;
  ncclWindow_t recvwin = (ncclWindow_t)recvbuff;
  kernel<<<deviceCtaCount, 512, 0, stream>>>(sendwin, sendoffset, recvwin, recvoffset, count, root, *devComm);
  return testSuccess;
}

// Helper macros to conditionally reference kernels only when compile-time conditions are met
// This prevents template instantiation when conditions are false by avoiding any reference to the kernel template
// FP8: requires both HAVE_FP8 and FP8_MULTIMEM_ARCH_AVAILABLE
#if HAVE_FP8 && FP8_MULTIMEM_ARCH_AVAILABLE
#define _FP8_MULTIMEM_KERNEL(kernel, fp8_type) kernel<__nv_fp8_##fp8_type>
#else
#define _FP8_MULTIMEM_KERNEL(kernel, fp8_type) nullptr
#endif

#if HAVE_FP8
#define _FP8_KERNEL(kernel, fp8_type) kernel<__nv_fp8_##fp8_type>
#else
#define _FP8_KERNEL(kernel, fp8_type) nullptr
#endif

// BF16: requires HAVE_BF16
#if HAVE_BF16
#define _BF16_KERNEL(kernel) kernel<__nv_bfloat16>
#else
#define _BF16_KERNEL(kernel) nullptr
#endif


#define SPECIALIZE_KERNEL(kernel, type, op) \
  ( op != ncclSum ? nullptr : \
   type == ncclInt8 ? kernel<int8_t> : \
   type == ncclUint8 ? kernel<uint8_t> : \
   type == ncclInt32 ? kernel<int32_t> : \
   type == ncclUint32 ? kernel<uint32_t> : \
   type == ncclInt64 ? kernel<int64_t> : \
   type == ncclUint64 ? kernel<uint64_t> : \
   type == ncclFloat16 ? kernel<half> : \
   type == ncclBfloat16 ? _BF16_KERNEL(kernel) : \
   type == ncclFloat8e4m3 ? _FP8_KERNEL(kernel, e4m3) :       \
   type == ncclFloat8e5m2 ? _FP8_KERNEL(kernel, e5m2) : \
   type == ncclFloat32 ? kernel<float> : \
   type == ncclFloat64 ? kernel<double> : \
   nullptr \
  )

// Specialization macro for multimem kernels - excludes int8_t and uint8_t (not supported for multimem)
// Returns nullptr for int8_t/uint8_t to allow tests to skip instead of triggering static_assert
// Structure: runtime check (type) first, then helper macros handle compile-time checks
// Helper macros expand to nullptr when conditions aren't met, preventing template instantiation
#define SPECIALIZE_KERNEL_MULTIMEM(kernel, type, op) \
  ( op != ncclSum ? nullptr : \
   type == ncclInt8 ? nullptr : \
   type == ncclUint8 ? nullptr : \
   type == ncclInt32 ? kernel<int32_t> : \
   type == ncclUint32 ? kernel<uint32_t> : \
   type == ncclInt64 ? kernel<int64_t> : \
   type == ncclUint64 ? kernel<uint64_t> : \
   type == ncclFloat16 ? kernel<half> : \
   type == ncclBfloat16 ? _BF16_KERNEL(kernel) : \
   type == ncclFloat8e4m3 ? _FP8_MULTIMEM_KERNEL(kernel, e4m3) : \
   type == ncclFloat8e5m2 ? _FP8_MULTIMEM_KERNEL(kernel, e5m2) : \
   type == ncclFloat32 ? kernel<float> : \
   type == ncclFloat64 ? kernel<double> : \
   nullptr \
  )

// Float/double-only specialization for kernels that only support float and double (e.g. AllReduce kernels 1 and 3).
// Returns nullptr for other types so the test can skip instead of failing.
#define SPECIALIZE_KERNEL_FLOAT_DOUBLE(kernel, type, op) \
  ( op != ncclSum ? nullptr : \
   type == ncclFloat32 ? kernel<float> : \
   type == ncclFloat64 ? kernel<double> : \
   nullptr \
  )
#else
template <typename F>
testResult_t testLaunchDeviceKernel(F kernel, void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  return testSkipped;
}
#define SPECIALIZE_KERNEL(kernel, type, op) nullptr
#define SPECIALIZE_KERNEL_MULTIMEM(kernel, type, op) nullptr
#define SPECIALIZE_KERNEL_FLOAT_DOUBLE(kernel, type, op) nullptr
#endif

bool isFp8ValidForReductions(ncclDataType_t type);
#endif
