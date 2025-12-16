/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//#define DEBUG_PRINT 1

#include "common.h"
#include <pthread.h>
#include <cstdio>
#include <type_traits>
#include <getopt.h>
#include <signal.h>
#include <libgen.h>
#include <assert.h>
#include <string.h>
#include <ctype.h>
#include "cuda.h"
#include "util.h"

#include "../verifiable/verifiable.h"

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

int test_ncclVersion = 0; // init'd with ncclGetVersion()

// profiler start and stop
extern int (*ncclProfilerStart)(int64_t profilerMask, const char* profilerDump);
extern int (*ncclProfilerStop)(void);

#if NCCL_MAJOR >= 2
  ncclDataType_t test_types[ncclNumTypes] = {
    ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclHalf, ncclFloat, ncclDouble
  #if HAVE_BF16
    , ncclBfloat16
  #endif
  #if HAVE_FP8
    , ncclFloat8e4m3, ncclFloat8e5m2
  #endif
  };
  const char *test_typenames[ncclNumTypes] = {
    "int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double"
  #if HAVE_BF16
    , "bfloat16"
  #endif
  #if HAVE_FP8
    , "f8e4m3", "f8e5m2"
  #endif
  };
  int test_typenum = -1;

  const char *test_opnames[] = {"sum", "prod", "max", "min", "avg", "mulsum"};
  ncclRedOp_t test_ops[] = {ncclSum, ncclProd, ncclMax, ncclMin
  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
    , ncclAvg
  #endif
  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
    , ncclNumOps // stand in for ncclRedOpCreatePreMulSum() created on-demand
  #endif
  };
  int test_opnum = -1;
#else
  ncclDataType_t test_types[ncclNumTypes] = {ncclChar, ncclInt, ncclHalf, ncclFloat, ncclDouble, ncclInt64, ncclUint64};
  const char *test_typenames[ncclNumTypes] = {"char", "int", "half", "float", "double", "int64", "uint64"};
  int test_typenum = 7;
  const char *test_opnames[] = {"sum", "prod", "max", "min"};
  ncclRedOp_t test_ops[] = {ncclSum, ncclProd, ncclMax, ncclMin};
  int test_opnum = 4;
#endif

// For libnccl's < 2.13
extern "C" __attribute__((weak)) char const* ncclGetLastError(ncclComm_t comm) {
  return "";
}

int is_main_proc = 0;
thread_local int is_main_thread = 0;
/* mpi multi-thread lock for MPI call serialization */
pthread_mutex_t mpiLock = PTHREAD_MUTEX_INITIALIZER;

// Command line parameter defaults
int nThreads = 1;
int nGpus = 1;
size_t minBytes = 32*1024*1024;
size_t maxBytes = 32*1024*1024;
size_t stepBytes = 1*1024*1024;
size_t stepFactor = 1;
int datacheck = 1;
int warmup_iters = 1;
int iters = 20;
int agg_iters = 1;
static int run_cycles = 1;
static int ncclop = ncclSum;
static int nccltype = ncclFloat;
static int ncclroot = 0;
int parallel_init = 0;
int blocking_coll = 0;
static int streamnull = 0;
int side_comp = 0;
static int timeout = 60;
int cudaGraphLaunches = 0;
static int report_cputime = 0;
static int out_of_place = 1;
static int unalign = 0;
static int trafficClass;
static int profilerMask;
static const char* profilerDumpDefault = "perftest";
static char* profilerDump = (char *)profilerDumpDefault;
static int profilerIters = INT_MAX;
int tuning;
int memory_report = 0;
static int deviceImpl = 0;
static int hostRmaImpl = 0;

int deviceCtaCount = 16; // Default number of CTAs for device implementation

static const char* testSkipReason = NULL;

// Report average iteration time: (0=RANK0,1=AVG,2=MIN,3=MAX)
static int average = 1;
static int commblocking = NCCL_CONFIG_UNDEF_INT;
static int ft_test = 0;
static char* ft_list = NULL;
static size_t tbytes = SIZE_MAX;
static int split_share = NCCL_CONFIG_UNDEF_INT;
static int split_comm = 0;
static char* splitMaskEnv = NULL;
static int commNum = 1;
#define LOCAL_REGISTER 1
#define SYMMETRIC_REGISTER 2
#define SYMMETRIC_REGISTER_SEND 3
#define SYMMETRIC_REGISTER_RECV 4
static int local_register = 0;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
static int ctaPolicy = -1;
#endif
static int per_coll_perf = 0;
static int simulate = 0;
static int nIdsUser = NCCL_CONFIG_UNDEF_INT; // number of ncclUniqueIds created
static int minCudaArch = 1<<30;

static char* replay_file = NULL;

static FILE* dump_file = NULL;
static double dump_values[32]; // 8 to 16G

enum output_file_type_t {
  JSON_FILE_OUTPUT,
  UNSPECIFIED_FILE_OUTPUT
};

// Return pointer to extension in `path` if one is found. An extension
// is the last `.` in the `path`, if there is no `/` following the `.`
// and there are characters after `.`.
//
// Therefore: returns 0 if no meaningful extension was found, or returns offset
// into string where extension begins
static const char *getExtension(const char *path) {
  if (path == nullptr) return nullptr;
  int last_dot = -1;
  int last_slash = -1;

  int pos;
  for (pos = 0; path[pos] != '\0'; ++pos) {
    switch (path[pos]) {
    case '.':
      last_dot = pos;
      break;
    case '/':
      last_slash = pos;
      break;
    default:
      break;
    }
  }

  if (last_dot > last_slash && last_dot + 1 != pos) {
    return path + last_dot + 1;
  }

  return nullptr;
}

static output_file_type_t classifyOutputFile(const char *filename) {
  const char *extension = getExtension(filename);
  if (extension != nullptr && strcasecmp(extension, "json") == 0) {
    return JSON_FILE_OUTPUT;
  }

  return UNSPECIFIED_FILE_OUTPUT;
}

static void outputFileInit(output_file_type_t output_file_type,
                           const char *output_file, char argc, char **argv, char **envp) {
  switch (output_file_type) {
  case JSON_FILE_OUTPUT:
    jsonOutputInit(output_file, argc, argv, envp);
    break;
  case UNSPECIFIED_FILE_OUTPUT:
  default:
    break;
  }
}

static void outputFileFinalize(output_file_type_t output_file_type) {
  switch (output_file_type) {
  case JSON_FILE_OUTPUT:
    jsonOutputFinalize();
    break;
  case UNSPECIFIED_FILE_OUTPUT:
  default:
    break;
  }
}

// Side computation constants
#define COMP_SIZE (1 << 22)
#define NUM_BLOCKS 64

static double parsesize(const char *value) {
    long long int units;
    double size;
    char size_lit;

    int count = sscanf(value, "%lf %c", &size, &size_lit);

    switch (count) {
    case 2:
      switch (size_lit) {
      case 'G':
      case 'g':
        units = 1024*1024*1024;
        break;
      case 'M':
      case 'm':
        units = 1024*1024;
        break;
      case 'K':
      case 'k':
        units = 1024;
        break;
      default:
        return -1.0;
      };
      break;
    case 1:
      units = 1;
      break;
    default:
      return -1.0;
    }

    return size * units;
}

// return true if the rank has to host a root
// this function matches the behavior of the scalable API
static int rankHasRoot(int rank, int nRanks, int nRoots) {
  // roots are divided in two groups:
  // - the first (nRanks % nRoots) are associated to (nRanks / nRoots + 1) ranks
  // - the rest are associated to (nRanks / nRoots) ranks
  // similarly ranks are divided in two groups:
  // - the first where their associated root is tied to (nRanks / nRoots + 1) ranks
  // - the second where their associated root is tied to (nRanks / nRoots) ranks;
  // the limit between the two groups is given by (nRanks % nRoots) * (nRanks / nRoots + 1)
  int rmr = nRanks % nRoots;
  int rpr = nRanks / nRoots;
  int rlim = rmr * (rpr+1);
  if (rank < rlim) {
    // all the ranks below rlim are associated to roots that are tied to (rpr + 1) ranks
    return !(rank % (rpr + 1));
  } else {
    // all the ranks above rlim are associated to roots that are tied to (rpr) ranks
    return !((rank - rlim) % rpr);
  }
}

testResult_t CheckDelta(void* results, void* expected, size_t count, size_t offset, ncclDataType_t type, ncclRedOp_t op, uint64_t seed, int nranks, int64_t *wrongEltN) {
  CUDACHECK(ncclVerifiableVerify(results, expected, count, (int)type, (int)op, nranks, seed, offset, wrongEltN, cudaStreamDefault));
  CUDACHECK(cudaDeviceSynchronize());
  return testSuccess;
}

testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op, uint64_t seed, int nranks) {
  CUDACHECK(ncclVerifiablePrepareExpected(data, count, (int)type, (int)op, nranks, seed, offset, cudaStreamDefault));
  return testSuccess;
}

testResult_t InitData(void* data, const size_t count, size_t offset, ncclDataType_t type, ncclRedOp_t op, uint64_t seed, int nranks, int rank) {
  CUDACHECK(ncclVerifiablePrepareInput(data, count, (int)type, (int)op, nranks, rank, seed, offset, cudaStreamDefault));
  return testSuccess;
}

void Barrier(struct threadArgs *args) {
  thread_local int epoch = 0;
  static pthread_mutex_t lock[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
  static pthread_cond_t cond[2] = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
  static int counter[2] = {0, 0};

  pthread_mutex_lock(&lock[epoch]);
  if(++counter[epoch] == args->nThreads)
    pthread_cond_broadcast(&cond[epoch]);

  if(args->thread+1 == args->nThreads) {
    while(counter[epoch] != args->nThreads)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);
    #ifdef MPI_SUPPORT
      MPI_Barrier(MPI_COMM_WORLD);
    #endif
    counter[epoch] = 0;
    pthread_cond_broadcast(&cond[epoch]);
  }
  else {
    while(counter[epoch] != 0)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);
  }
  pthread_mutex_unlock(&lock[epoch]);
  epoch ^= 1;
}

testResult_t barrierRmaSignal(ncclComm_t comm, cudaStream_t stream) {
  int rank, nranks;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  NCCLCHECK(ncclCommCount(comm, &nranks));

  int ctx = 0;

  ncclWaitSignalDesc_t* waitDescs = (ncclWaitSignalDesc_t*)malloc(sizeof(ncclWaitSignalDesc_t) * (nranks - 1));
  if (waitDescs == NULL) {
    return testInternalError;
  }

  int descIdx = 0;
  for (int i = 0; i < nranks; i++) {
    if (i != rank) {
      waitDescs[descIdx].opCnt = 1;
      waitDescs[descIdx].peer = i;
      waitDescs[descIdx].sigIdx = 0;
      waitDescs[descIdx].ctx = ctx;
      descIdx++;
    }
  }

  NCCLCHECK(ncclGroupStart());

  for (int i = 0; i < nranks; i++) {
    if (i != rank) {
      NCCLCHECK(ncclSignal(i, 0, ctx, 0, comm, stream));
    }
  }

  NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);

  NCCLCHECK(ncclWaitSignal(nranks - 1, waitDescs, comm, stream));

  free(waitDescs);

  return testSuccess;
}

// Inter-thread/process barrier+allreduce. The quality of the return value
// for average=0 (which means broadcast from rank=0) is dubious. The returned
// value will actually be the result of process-local broadcast from the local thread=0.
template<typename T>
void Allreduce(struct threadArgs* args, T* value, int average) {
  thread_local int epoch = 0;
  static pthread_mutex_t lock[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
  static pthread_cond_t cond[2] = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
  static T accumulator[2];
  static int counter[2] = {0, 0};

  pthread_mutex_lock(&lock[epoch]);
  if(counter[epoch] == 0) {
    if(average != 0 || args->thread == 0) accumulator[epoch] = *value;
  } else {
    switch(average) {
    case /*r0*/ 0: if(args->thread == 0) accumulator[epoch] = *value; break;
    case /*avg*/1: accumulator[epoch] += *value; break;
    case /*min*/2: accumulator[epoch] = std::min<T>(accumulator[epoch], *value); break;
    case /*max*/3: accumulator[epoch] = std::max<T>(accumulator[epoch], *value); break;
    case /*sum*/4: accumulator[epoch] += *value; break;
    }
  }

  if(++counter[epoch] == args->nThreads)
    pthread_cond_broadcast(&cond[epoch]);

  if(args->thread+1 == args->nThreads) {
    while(counter[epoch] != args->nThreads)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);

    #ifdef MPI_SUPPORT
    if(average != 0) {
      static_assert(std::is_same<T, long long>::value || std::is_same<T, double>::value, "Allreduce<T> only for T in {long long, double}");
      MPI_Datatype ty = std::is_same<T, long long>::value ? MPI_LONG_LONG :
                        std::is_same<T, double>::value ? MPI_DOUBLE :
                        MPI_Datatype();
      MPI_Op op = average == 1 ? MPI_SUM :
                  average == 2 ? MPI_MIN :
                  average == 3 ? MPI_MAX :
                  average == 4 ? MPI_SUM : MPI_Op();
      MPI_Allreduce(MPI_IN_PLACE, (void*)&accumulator[epoch], 1, ty, op, MPI_COMM_WORLD);
    }
    #endif

    if(average == 1) accumulator[epoch] /= args->totalProcs*args->nThreads;
    counter[epoch] = 0;
    pthread_cond_broadcast(&cond[epoch]);
  }
  else {
    while(counter[epoch] != 0)
      pthread_cond_wait(&cond[epoch], &lock[epoch]);
  }
  pthread_mutex_unlock(&lock[epoch]);

  *value = accumulator[epoch];
  epoch ^= 1;
}

testResult_t CheckData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int64_t *wrongElts) {
  size_t count;
  void* data;
  int64_t *wrongPerGpu = nullptr;

  CUDACHECK(cudaHostAlloc((void**)&wrongPerGpu, args->nGpus * sizeof(int64_t), cudaHostAllocMapped));
  for (int id = 0; id < args->commNum; ++id) {
    for (int i = 0; i < args->nGpus; i++) {
      int rank, nranks;

      CUDACHECK(cudaSetDevice(args->gpus[i]));
      NCCLCHECK(ncclCommUserRank(args->comms[id][i], &rank));
      NCCLCHECK(ncclCommCount(args->comms[id][i], &nranks));
      data = in_place ? ((void*)((uintptr_t)args->recvbuffs[id][i] + args->recvInplaceOffset[id][i] * rank)) : args->recvbuffs[id][i];
      count = args->expectedBytes[id][i] / wordSize(type);
      TESTCHECK(CheckDelta(data, args->expected[id][i], count, 0, type, op, 0, nranks, wrongPerGpu + i));

#if DEBUG_PRINT
      if (args->reportErrors && wrongPerGpu[i] != 0) {
        printf("split_id=%d rank=%d #wrong=%d\n", id, rank, (int)wrongPerGpu[i]);
        char* expectedHost = (char*)malloc(args->expectedBytes[id][i]);
        char* dataHost = (char*)malloc(args->expectedBytes[id][i]);
        int eltsz = wordSize(type);
        cudaMemcpy(expectedHost, args->expected[id][i], args->expectedBytes[id][i], cudaMemcpyDeviceToHost);
        cudaMemcpy(dataHost, data, args->expectedBytes[id][i], cudaMemcpyDeviceToHost);

        for (int j = 0; j < args->expectedBytes[id][i] / eltsz; j++) {
          unsigned long long want, got;
          want = 0;
          memcpy(&want, expectedHost + j * eltsz, eltsz);
          got = 0;
          memcpy(&got, dataHost + j * eltsz, eltsz);
          if (want != got) {
            printf(" rank=%d elt[%d]: want=0x%llx got=0x%llx\n", rank, j, want, got);
          }
        }
        free(expectedHost);
        free(dataHost);
      }
#endif
    }
  }

  *wrongElts = 0;
  for (int i=0; i < args->nGpus; i++) *wrongElts += wrongPerGpu[i];
  cudaFreeHost(wrongPerGpu);

  if (args->reportErrors && *wrongElts) args->errors[0]++;
  return testSuccess;
}

testResult_t testStreamSynchronize(int ngpus, cudaStream_t* streams, ncclComm_t** comms, int commNum = 1) {
  cudaError_t cudaErr;
  int remaining = ngpus;
  int* done = (int*)malloc(sizeof(int)*ngpus);
  memset(done, 0, sizeof(int)*ngpus);
  timer tim;

  while (remaining) {
    int idle = 1;
#ifdef MPI_SUPPORT
    if (comms) {
      int flag;
      /* poke MPI progress for OpenMPI */
      pthread_mutex_lock(&mpiLock);
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      pthread_mutex_unlock(&mpiLock);
    }
#endif
    for (int id0 = 0; id0 < commNum; ++id0) {
      for (int i = 0; i < ngpus; i++) {
        if (done[i]) continue;

        cudaErr = cudaStreamQuery(streams[i]);
        if (cudaErr == cudaSuccess) {
          done[i] = 1;
          remaining--;
          idle = 0;
          continue;
        }

        if (cudaErr != cudaErrorNotReady) CUDACHECK(cudaErr);

  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,4,0)
        if (test_ncclVersion >= NCCL_VERSION(2, 4, 0) && comms) {
          ncclResult_t ncclAsyncErr;
          NCCLCHECK(ncclCommGetAsyncError(comms[id0][i], &ncclAsyncErr));
          if (ncclAsyncErr != ncclSuccess) {
            // An asynchronous error happened. Stop the operation and destroy
            // the communicator
            char hostname[1024];
            getHostName(hostname, 1024);
            printf(
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,12,10)
              "\n%s: Async error detected: %s / %s %s:%d\n",
#else
              "\n%s: Async error detected: %s %s:%d\n",
#endif
              hostname,
              ncclGetErrorString(ncclAsyncErr),
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,12,10)
              ncclGetLastError(NULL),
#endif
              __FILE__, __LINE__);

            for (int id1 = 0; id1 < commNum; ++id1)
              for (int i = 0; i < ngpus; i++)
                NCCLCHECK(ncclCommAbort(comms[id1][i]));
            // Abort the perf test
            return testNcclError;
          }

          double delta = tim.elapsed();
          if (delta > timeout && timeout > 0) {
            char hostname[1024];
            getHostName(hostname, 1024);
            printf("%s: Test timeout (%ds) %s:%d\n",
              hostname,
              timeout,
              __FILE__, __LINE__);

            for (int id1 = 0; id1 < commNum; ++id1)
              for (int i = 0; i < ngpus; i++)
                NCCLCHECK(ncclCommAbort(comms[id1][i]));
            free(done);
            return testTimeout;
          }
        }
  #endif
      }
   }
   // We might want to let other threads (including NCCL threads) use the CPU.
   if (idle) sched_yield();
  }
  free(done);
  return testSuccess;
}

testResult_t startColl(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t opIndex, int root, int in_place, int iter) {
  // Try to change offset for each iteration so that we avoid cache effects and catch race conditions in ptrExchange
  size_t count, totalnbytes, steps, shift;

  for (int id = 0; id < args->commNum; ++id) {
    if (args->nGpus > 1 || commblocking == 0) NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < args->nGpus; i++) {
#ifndef NCCL_MAJOR
      CUDACHECK(cudaSetDevice(args->gpus[i]));
#endif
      int rank;
      char* recvBuff;
      char* sendBuff;
      ncclRedOp_t op;

      count = args->nbytes[id][i] / wordSize(type);
      totalnbytes = max(args->sendBytes[id][i], args->expectedBytes[id][i]);
      steps = totalnbytes ? args->maxbytes / totalnbytes : 1;
      shift = totalnbytes * (iter % steps);
      recvBuff = ((char*)args->recvbuffs[id][i]) + shift;
      sendBuff = ((char*)args->sendbuffs[id][i]) + shift;
      NCCLCHECK(ncclCommUserRank(args->comms[id][i], &rank));

      if (opIndex < ncclNumOps) {
        op = opIndex;
      }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
      else {
        union {
          int8_t i8; uint8_t u8; int32_t i32; uint32_t u32; int64_t i64; uint64_t u64;
          half f16; float f32; double f64;
          #if HAVE_BF16
          __nv_bfloat16 bf16;
          #endif
          #if HAVE_FP8
          __nv_fp8_e4m3 f8e4m3; __nv_fp8_e5m2 f8e5m2;
          #endif
        };
        switch (type) {
          case ncclInt8: i8 = ncclVerifiablePremulScalar<int8_t>(rank); break;
          case ncclUint8: u8 = ncclVerifiablePremulScalar<uint8_t>(rank); break;
          case ncclInt32: i32 = ncclVerifiablePremulScalar<int32_t>(rank); break;
          case ncclUint32: u32 = ncclVerifiablePremulScalar<uint32_t>(rank); break;
          case ncclInt64: i64 = ncclVerifiablePremulScalar<int64_t>(rank); break;
          case ncclUint64: u64 = ncclVerifiablePremulScalar<uint64_t>(rank); break;
          case ncclFloat16: f16 = ncclVerifiablePremulScalar<half>(rank); break;
          case ncclFloat32: f32 = ncclVerifiablePremulScalar<float>(rank); break;
          case ncclFloat64: f64 = ncclVerifiablePremulScalar<double>(rank); break;
          #if HAVE_BF16
          case ncclBfloat16: bf16 = ncclVerifiablePremulScalar<__nv_bfloat16>(rank); break;
          #endif
          #if HAVE_FP8
          case ncclFloat8e4m3: f8e4m3 = ncclVerifiablePremulScalar<__nv_fp8_e4m3>(rank); break;
          case ncclFloat8e5m2: f8e5m2 = ncclVerifiablePremulScalar<__nv_fp8_e5m2>(rank); break;
          #endif
          default: break; // Just to silence clang
        }
        NCCLCHECK(ncclRedOpCreatePreMulSum(&op, &u64, type, ncclScalarHostImmediate, args->comms[id][i]));
      }
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
      if (hostRmaImpl) {
        void* sendwin = args->sendRegHandles[id][i];
        void* recvwin = args->recvRegHandles[id][i];
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        TESTCHECK(args->collTest->runColl(
              (void*)(in_place ? recvwin : sendwin), shift + (in_place ? args->sendInplaceOffset[id][i] * rank : 0),
              (void*)recvwin, shift + in_place ? args->recvInplaceOffset[id][i] * rank : 0,
              count, type, op, root, args->comms[id][i], args->streams[i], HOST_RMA_IMPL));
      } else
#endif
      if (deviceImpl == 0) {
        TESTCHECK(args->collTest->runColl(
              (void*)(in_place ? recvBuff : sendBuff), in_place ? args->sendInplaceOffset[id][i] * rank : 0,
              (void*)recvBuff, in_place ? args->recvInplaceOffset[id][i] * rank : 0,
              count, type, op, root, args->comms[id][i], args->streams[i], 0));
      } else {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
        void* sendwin = args->sendRegHandles[id][i];
        void* recvwin = args->recvRegHandles[id][i];
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        TESTCHECK(args->collTest->runColl(
              (void*)(in_place ? recvwin : sendwin), shift + in_place ? args->sendInplaceOffset[id][i] * rank : 0,
              (void*)recvwin, shift + in_place ? args->recvInplaceOffset[id][i] * rank : 0,
              count, type, op, root, (ncclComm_t)(args->devComms[id]+i), args->streams[i], deviceImpl));
#endif
      }

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
      if (opIndex >= ncclNumOps) {
        NCCLCHECK(ncclRedOpDestroy(op, args->comms[id][i]));
      }
#endif
    }
    if (args->nGpus > 1 || commblocking == 0) NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), args->comms[id], args->nGpus);
  }

  if (blocking_coll) {
    // Complete op before returning
    TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms, args->commNum));
  }
  if (blocking_coll) Barrier(args);
  return testSuccess;
}

testResult_t completeColl(struct threadArgs* args) {
  if (blocking_coll) return testSuccess;

  TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms, args->commNum));
  return testSuccess;
}

static testResult_t getIteration(size_t nbytes, int* itersPtr) {
  if (tbytes == SIZE_MAX) {
    *itersPtr = iters;
  } else {
    if (nbytes == 0)
      *itersPtr = iters;
    else
      *itersPtr = max(min((size_t)iters, tbytes / nbytes), 1UL);
  }
  return testSuccess;
}

testResult_t getElapsedTimes(struct threadArgs* args, int eventIters, int aggIters) {
  args->ms = (float*) malloc(sizeof(float)*args->nGpus*eventIters);
  // Get timings
  for (int i = 0; i < args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    for (int j = 0; j < eventIters; j++) {
      CUDACHECK(cudaEventElapsedTime(&args->ms[i*(eventIters) + j], args->events[i*(eventIters+1) + j], args->events[i*(eventIters+1) + j+1]));

      // This elapsed time is for iterations equal to agg_iters.
      // We need to divide by aggIters
      args->ms[i*(eventIters)+j] /= aggIters;
    }
  }
  return testSuccess;
}

testResult_t recordEvents(struct threadArgs* args, int iterations, int iteration) {
  for(int i = 0; i < args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    CUDACHECK(cudaEventRecord(args->events[i*(iterations+1) + iteration], args->streams[i]));
  }
  return testSuccess;
}

testResult_t initEvents(struct threadArgs* args, int eventIters) {
  // Creating (iterations + 1) events and then calculate the time between two events.
  args->events = (cudaEvent_t*) malloc(sizeof(cudaEvent_t)*args->nGpus*(eventIters + 1));
  for (int i = 0; i < args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    for (int j = 0; j < eventIters + 1; j++) {
      CUDACHECK(cudaEventCreate(&args->events[(i*(eventIters+1) + j)]));
    }
  }
  return testSuccess;
}

testResult_t destroyEvents(struct threadArgs* args, int eventIters) {
  for (int i = 0; i < args->nGpus; i++) {
    for (int j = 0; j < eventIters + 1; j++)
      CUDACHECK(cudaEventDestroy(args->events[i*(eventIters+1)+j]));
  }
  free(args->events);
  return testSuccess;
}

testResult_t BenchTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int actualIters, int record) {
  size_t count = args->nbytes[0][0] / wordSize(type);
  if (datacheck) {
    // Initialize sendbuffs, recvbuffs and expected
    TESTCHECK(args->collTest->initData(args, type, op, root, 99, in_place));
  }

  // Sync
#if 0
  TESTCHECK(startColl(args, type, op, root, in_place, 0));
#endif
  TESTCHECK(completeColl(args));

  Barrier(args);
  if (record) TESTCHECK(initEvents(args, actualIters));
  args->compThreadCountLast = *(args->compThreadCount);

#if CUDART_VERSION >= 11030
  cudaGraph_t graphs[args->nGpus];
  cudaGraphExec_t graphExec[args->nGpus];
  if (cudaGraphLaunches >= 1) {
    // Begin cuda graph capture
    for (int i=0; i<args->nGpus; i++) {
      // Thread local mdoe is needed for:
      // - Multi-thread mode: where graph capture and instantiation can happen concurrently across threads
      // - P2P pre-connect: when there is no warm-up, P2P pre-connect is done during graph capture.
      //   Since pre-connect calls cudaMalloc, we cannot use global capture mode
      CUDACHECK(cudaStreamBeginCapture(args->streams[i], cudaStreamCaptureModeThreadLocal));
    }
  }
#endif

  // Performance Benchmark
  timer tim;
  int profilerIter = 0;
  for (int iter = 0; iter < actualIters; iter++) {
    if (agg_iters>1) NCCLCHECK(ncclGroupStart());

    if (record) TESTCHECK(recordEvents(args, actualIters, iter));

    for (int aiter = 0; aiter < agg_iters; aiter++) {
      if (profilerMask && profilerIter < profilerIters) ncclProfilerStart(profilerMask, profilerDump);
      TESTCHECK(startColl(args, type, op, root, in_place, iter*agg_iters+aiter));
      if (profilerMask && profilerIter++ < profilerIters) ncclProfilerStop();
    }
    if (agg_iters>1) NCCLCHECK(ncclGroupEnd());
  }

  if (record) TESTCHECK(recordEvents(args, actualIters, actualIters));

#if CUDART_VERSION >= 11030
  if (cudaGraphLaunches >= 1) {
    // End cuda graph capture
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs+i));
    }
    // Instantiate cuda graph
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaGraphInstantiate(graphExec+i, graphs[i], NULL, NULL, 0));
    }
    // Resync CPU, restart timing, launch cuda graph
    Barrier(args);
    tim.reset();
    for (int l=0; l<cudaGraphLaunches; l++) {
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
      }
    }
  }
#endif

  double cputimeSec = tim.elapsed()/(actualIters*agg_iters);
  TESTCHECK(completeColl(args));

  if (record) {
    TESTCHECK(getElapsedTimes(args, actualIters, agg_iters));
    TESTCHECK(destroyEvents(args, actualIters));
  }

  int compThreadCount = (*(args->compThreadCount)) - args->compThreadCountLast;
  double deltaSec = tim.elapsed();
  deltaSec = deltaSec/(actualIters*agg_iters);
  if (cudaGraphLaunches >= 1) deltaSec = deltaSec/cudaGraphLaunches;
  Allreduce(args, &deltaSec, average);

#if CUDART_VERSION >= 11030
  if (cudaGraphLaunches >= 1) {
    //destroy cuda graph
    for (int i=0; i<args->nGpus; i++) {
      CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
      CUDACHECK(cudaGraphDestroy(graphs[i]));
    }
  }
#endif

  double algBw, busBw;
  args->collTest->getBw(count, wordSize(type), deltaSec, &algBw, &busBw, args->nProcs*args->nThreads*args->nGpus);

  Barrier(args);

  int64_t wrongElts = 0;
  static __thread int rep = 0;
  rep++;
  for (int c = 0; c < datacheck; c++) {
      // Initialize sendbuffs, recvbuffs and expected
      TESTCHECK(args->collTest->initData(args, type, op, root, rep, in_place));

      // Ensure GPU buffer initialization completes across all ranks before validation
      for (int i = 0; i < args->nGpus; i++) {
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        CUDACHECK(cudaStreamSynchronize(args->streams[i]));
      }
      Barrier(args);  // MPI barrier to ensure all ranks' GPUs are ready

#if CUDART_VERSION >= 11030
      if (cudaGraphLaunches >= 1) {
        // Begin cuda graph capture for data check
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaStreamBeginCapture(args->streams[i], args->nThreads > 1 ? cudaStreamCaptureModeThreadLocal : cudaStreamCaptureModeGlobal));
        }
      }
#endif

      //test validation in single itertion, should ideally be included into the multi-iteration run
      TESTCHECK(startColl(args, type, op, root, in_place, 0));

#if CUDART_VERSION >= 11030
      if (cudaGraphLaunches >= 1) {
        // End cuda graph capture
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaStreamEndCapture(args->streams[i], graphs+i));
        }
        // Instantiate cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphInstantiate(graphExec+i, graphs[i], NULL, NULL, 0));
        }
        // Launch cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphLaunch(graphExec[i], args->streams[i]));
        }
      }
#endif

      TESTCHECK(completeColl(args));

#if CUDART_VERSION >= 11030
      if (cudaGraphLaunches >= 1) {
        //destroy cuda graph
        for (int i=0; i<args->nGpus; i++) {
          CUDACHECK(cudaGraphExecDestroy(graphExec[i]));
          CUDACHECK(cudaGraphDestroy(graphs[i]));
        }
      }
#endif

      TESTCHECK(CheckData(args, type, op, root, in_place, &wrongElts));

      //aggregate delta from all threads and procs
      long long wrongElts1 = wrongElts;
      Allreduce(args, &wrongElts1, /*sum*/4);
      wrongElts = wrongElts1;
      if (wrongElts) break;
  }

  double timeUsec = (report_cputime ? cputimeSec : deltaSec)*1.0E6;

  float totalTime = 0.0;
  if (simulate) {
    for (int iter = 0; iter < actualIters; iter++) {

      NCCLCHECK(ncclGroupStart());

      for (int aiter = 0; aiter < agg_iters; aiter++)
        TESTCHECK(startColl(args, type, op, root, in_place, iter*agg_iters+aiter));

      ncclSimInfo_t simInfo = NCCL_SIM_INFO_INITIALIZER;
      NCCLCHECK(ncclGroupSimulateEnd(&simInfo));
      totalTime += simInfo.estimatedTime;
    }
    totalTime /= (actualIters*agg_iters);
  }
  double sideBw = ((double)compThreadCount)*COMP_SIZE*NUM_BLOCKS/(1000*timeUsec);
  writeBenchmarkLineBody(timeUsec, totalTime, algBw, busBw, sideBw, args->reportErrors, wrongElts, report_cputime, in_place==0, simulate);

  if (record) {
    args->meanTime = timeUsec;
    args->meanAlgBw = algBw;
    args->meanBusBw = busBw;
  }

  if (dump_file) {
    /* only dump first split and communicator */
    size_t nBytes = max(args->sendBytes[0][0], args->expectedBytes[0][0]);
    // Dump 8B to 16G to file.
    for (int p=0; p<32; p++) if (nBytes == (8ULL<<p)) {
      if (dump_values[p] == 0.0) {
        dump_values[p] = timeUsec;
      } else {
        dump_values[p] = std::min(timeUsec, dump_values[p]);
      }
    }
  }

  args->bw[0] += busBw;
  args->bw_count[0]++;
  return testSuccess;
}

void setupArgs(size_t size, ncclDataType_t type, struct threadArgs* args) {
  size_t count = size / wordSize(type);
  for (int id = 0; id < args->commNum; ++id) {
    for (int i = 0; i < args->nGpus; ++i) {
      int nranks;
      size_t sendCount, recvCount, paramCount, sendInplaceOffset, recvInplaceOffset;

      ncclCommCount(args->comms[id][i], &nranks);
      args->collTest->getCollByteCount(&sendCount, &recvCount, &paramCount, &sendInplaceOffset, &recvInplaceOffset, (size_t)count, wordSize(type), (size_t)nranks);
      args->nbytes[id][i] = paramCount * wordSize(type);
      args->sendBytes[id][i] = sendCount * wordSize(type);
      args->expectedBytes[id][i] = recvCount * wordSize(type);
      args->sendInplaceOffset[id][i] = sendInplaceOffset * wordSize(type);
      args->recvInplaceOffset[id][i] = recvInplaceOffset * wordSize(type);
    }
  }
}

testResult_t TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName, int root) {
  // Sync to avoid first-call timeout
  Barrier(args);

  // Add forced misalignment
  for (int id = 0; id < args->commNum; ++id) {
    for (int i = 0; i < args->nGpus; i++) {
      args->sendbuffs[id][i] = (char*)args->sendbuffs[id][i] + unalign * wordSize(type);
      args->recvbuffs[id][i] = (char*)args->recvbuffs[id][i] + unalign * wordSize(type);
    }
  }

  // Warm-up for all sizes
  for (size_t size = args->minbytes; size <= args->maxbytes; size = ((args->stepfactor > 1) ? size * args->stepfactor : size + args->stepbytes)) {
    setupArgs(size, type, args);
    for (int iter = 0; iter < warmup_iters; iter++) {
      TESTCHECK(startColl(args, type, op, root, 0, iter));
      TESTCHECK(startColl(args, type, op, root, 1, iter));
    }
    TESTCHECK(completeColl(args));
  }

  // Benchmark
  long repeat = run_cycles;
  do {
    for (size_t size = args->minbytes; size<=args->maxbytes; size = ((args->stepfactor > 1) ? size*args->stepfactor : size+args->stepbytes)) {
        setupArgs(size, type, args);
        int actualIters;
        TESTCHECK(getIteration(size, &actualIters));
        writeBenchmarkLinePreamble(max(args->sendBytes[0][0], args->expectedBytes[0][0]), args->nbytes[0][0] / wordSize(type), typeName, opName, root);
        if (args->replayFile != NULL || !out_of_place) {
          writeBenchMarkLineNullBody();  // only do in-place for trace replay
        } else {
          TESTCHECK(BenchTime(args, type, op, root, 0, actualIters, per_coll_perf));
        }
        TESTCHECK(BenchTime(args, type, op, root, 1, actualIters, 0));
        if (per_coll_perf) printPerCollPerf(args, type, op, root, actualIters, per_coll_perf);
        writeBenchmarkLineTerminator(actualIters, args->replayFile == NULL ? "" : args->collTest->name);
    }
  } while (--repeat);

  // Revert forced misalignment
  for (int id = 0; id < args->commNum; ++id) {
    for (int i = 0; i < args->nGpus; i++) {
      args->sendbuffs[id][i] = (char*)args->sendbuffs[id][i] - unalign * wordSize(type);
      args->recvbuffs[id][i] = (char*)args->recvbuffs[id][i] - unalign * wordSize(type);
    }
  }
  return testSuccess;
}

static void getGPUMemoryInfo(int64_t* ptotalGpuMem, int64_t* pfreeGpuMem) {
  size_t freeGpuMem, totalGpuMem = 0;
  cudaMemGetInfo(&freeGpuMem, &totalGpuMem);
  if (ptotalGpuMem != nullptr) *ptotalGpuMem = totalGpuMem;
  if (pfreeGpuMem != nullptr) *pfreeGpuMem = freeGpuMem;
}

testResult_t threadRunTests(struct threadArgs* args) {
  //  capture the free memory before
  int64_t* totalGpuFreeMem = (int64_t*)calloc(args->nGpus*2, sizeof(int64_t));
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    getGPUMemoryInfo(nullptr, &totalGpuFreeMem[g]);
  }

  // Set device to the first of our GPUs. If we don't do that, some operations
  // will be done on the current GPU (by default : 0) and if the GPUs are in
  // exclusive mode those operations will fail.
  CUDACHECK(cudaSetDevice(args->gpus[0]));
  TESTCHECK(ncclTestEngine.runTest(args, ncclroot, (ncclDataType_t)nccltype, test_typenames[nccltype], (ncclRedOp_t)ncclop, test_opnames[ncclop]));

  // Capture the memory used by the GPUs
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    getGPUMemoryInfo(nullptr, &totalGpuFreeMem[g + args->nGpus]);
    *args->devMemUsed = std::max(*args->devMemUsed, totalGpuFreeMem[g] - totalGpuFreeMem[g + args->nGpus]);
  }
  free(totalGpuFreeMem);
  return testSuccess;
}

testResult_t threadInit(struct threadArgs* args) {
  int nranks = args->totalProcs * args->nThreads * args->nGpus;
  ncclComm_t globalComms[args->nGpus];

  // Capture GPU memory before initializing the NCCL communicators
  int64_t* initFreeGpuMem = (int64_t*)calloc(args->nGpus*3, sizeof(int64_t));
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    getGPUMemoryInfo(nullptr, &initFreeGpuMem[g]);
  }

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = commblocking;
  config.splitShare = split_share;
  config.trafficClass = trafficClass;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
  if (ctaPolicy >= 0)
    config.CTAPolicy = ctaPolicy;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  config.nvlinkCentricSched = 1;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
  if (cudaGraphLaunches >= 1)
    config.graphUsageMode = 1;
  else
    config.graphUsageMode = 0;
#endif
#endif
#endif
#endif

  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<args->nGpus; i++) {
    int rank = args->globalProc*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    CUDACHECK(cudaSetDevice(args->gpus[i]));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,23,0)
    if (nIdsUser != NCCL_CONFIG_UNDEF_INT) {
      NCCLCHECK(ncclCommInitRankScalable(globalComms + i, nranks, rank, args->nIds, args->ncclId, &config));
    } else {
      NCCLCHECK(ncclCommInitRankConfig(globalComms + i, nranks, *args->ncclId, rank, &config));
    }
#else
    NCCLCHECK(ncclCommInitRankConfig(globalComms + i, nranks, *args->ncclId, rank, &config));
#endif
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, args->nGpus);
  /* split comm if required. */
  if (splitMaskEnv) {
    /* split based on split mask */
    uint64_t mask = strtoul(splitMaskEnv, NULL, 16);
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < args->nGpus; ++i) {
      int rank = args->globalProc * args->nThreads * args->nGpus + args->thread * args->nGpus + i;
      NCCLCHECK(ncclCommSplit(globalComms[i], rank & mask, rank, &args->comms[0][i], &config));
    }
    NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, args->nGpus);
  } else if (split_comm == 2) {
    /* create split comm with predefined split pattern. */
    for (int splitCase = 0; splitCase < args->commNum; ++splitCase) {
      switch (splitCase) {
        case 0: {
          /* duplicate communicator but in reversed rank */
          NCCLCHECK(ncclGroupStart());
          for (int i = 0; i < args->nGpus; ++i) {
            int myrank = args->globalProc * args->nThreads * args->nGpus + args->thread * args->nGpus + i;
            NCCLCHECK(ncclCommSplit(globalComms[i], 0, nranks - myrank, &args->comms[splitCase][i], &config));
          }
          NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, args->nGpus);
          break;
        }
        case 1: {
          /* split with odd and even rank */
          NCCLCHECK(ncclGroupStart());
          for (int i = 0; i < args->nGpus; ++i) {
            int myrank = args->globalProc * args->nThreads * args->nGpus + args->thread * args->nGpus + i;
            NCCLCHECK(ncclCommSplit(globalComms[i], myrank & 1, myrank, &args->comms[splitCase][i], &config));
          }
          NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, args->nGpus);
          break;
        }
        case 2: {
          /* 3:1 split */
          NCCLCHECK(ncclGroupStart());
          for (int i = 0; i < args->nGpus; ++i) {
            int myrank = args->globalProc * args->nThreads * args->nGpus + args->thread * args->nGpus + i;
            NCCLCHECK(ncclCommSplit(globalComms[i], 4 * (myrank + 1) <= 3 * nranks, myrank, &args->comms[splitCase][i], &config));
          }
          NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, args->nGpus);
          break;
        }
        case 3: {
          /* duplicate communicator but in reversed rank (again) */
          NCCLCHECK(ncclGroupStart());
          for (int i = 0; i < args->nGpus; ++i) {
            int myrank = args->globalProc * args->nThreads * args->nGpus + args->thread * args->nGpus + i;
            NCCLCHECK(ncclCommSplit(globalComms[i], 0, nranks - myrank, &args->comms[splitCase][i], &config));
          }
          NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, args->nGpus);
          break;
        }
              /* we can add more to split pattern */
        default:
          return testInternalError;
      }
    }
  } else if (split_comm == 1) {
    /* duplicate globalcomm. */
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < args->nGpus; i++) {
      NCCLCHECK(ncclCommSplit(globalComms[i], 0, 0, &args->comms[0][i], &config));
    }
    NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, args->nGpus);
  } else {
    /* no split required by users */
    for (int i = 0; i < args->nGpus; ++i) args->comms[0][i] = globalComms[i];
  }

  if (split_comm) {
    /* destroy global NCCL communicators */
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < args->nGpus; i++) {
      NCCLCHECK(ncclCommFinalize(globalComms[i]));
    }
    NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, args->nGpus);

    for (int i = 0; i < args->nGpus; i++)
      NCCLCHECK(ncclCommDestroy(globalComms[i]));
  }

  // Capture the memory used by the GPUs after initializing the NCCL communicators
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    getGPUMemoryInfo(nullptr, &initFreeGpuMem[g + args->nGpus]);
    *args->initGpuMem = std::max(*args->initGpuMem, initFreeGpuMem[g] - initFreeGpuMem[g + args->nGpus]);
  }
  /* allocate buffer for each split comm. */
  NCCLCHECK(ncclGroupStart());
  for (int id = 0; id < args->commNum; ++id) {
    for (int i = 0; i < args->nGpus; i++) {
      int nranks;
      size_t sendBytes, recvBytes;
      size_t allocBytes;
      NCCLCHECK(ncclCommCount(args->comms[id][i], &nranks));
      ncclTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes, (size_t)nranks);
      CUDACHECK(cudaSetDevice(args->gpus[i]));
      TESTCHECK(AllocateBuffs(args->sendbuffs[id] + i, sendBytes, args->recvbuffs[id] + i, recvBytes, args->expected[id] + i, (size_t)maxBytes, &allocBytes));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
      if (local_register == SYMMETRIC_REGISTER_SEND) {
        NCCLCHECK(ncclCommWindowRegister(args->comms[id][i], args->sendbuffs[id][i], allocBytes, (ncclWindow_t*)&args->sendRegHandles[id][i], NCCL_WIN_COLL_SYMMETRIC));
      } else if (local_register == SYMMETRIC_REGISTER_RECV) {
        NCCLCHECK(ncclCommWindowRegister(args->comms[id][i], args->recvbuffs[id][i], allocBytes, (ncclWindow_t*)&args->recvRegHandles[id][i], NCCL_WIN_COLL_SYMMETRIC));
      } else if (local_register == SYMMETRIC_REGISTER) {
        NCCLCHECK(ncclCommWindowRegister(args->comms[id][i], args->sendbuffs[id][i], allocBytes, (ncclWindow_t*)&args->sendRegHandles[id][i], NCCL_WIN_COLL_SYMMETRIC));
        NCCLCHECK(ncclCommWindowRegister(args->comms[id][i], args->recvbuffs[id][i], allocBytes, (ncclWindow_t*)&args->recvRegHandles[id][i], NCCL_WIN_COLL_SYMMETRIC));
      } else {
        if (local_register) NCCLCHECK(ncclCommRegister(args->comms[id][i], args->sendbuffs[id][i], allocBytes, &args->sendRegHandles[id][i]));
        if (local_register) NCCLCHECK(ncclCommRegister(args->comms[id][i], args->recvbuffs[id][i], allocBytes, &args->recvRegHandles[id][i]));
      }
#else
      if (local_register) NCCLCHECK(ncclCommRegister(args->comms[id][i], args->sendbuffs[id][i], allocBytes, &args->sendRegHandles[id][i]));
      if (local_register) NCCLCHECK(ncclCommRegister(args->comms[id][i], args->recvbuffs[id][i], allocBytes, &args->recvRegHandles[id][i]));
#endif
#endif
    }
  }
  NCCLCHECK(ncclGroupEnd());
  // Capture memory used by test buffers
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    getGPUMemoryInfo(nullptr, &initFreeGpuMem[g + args->nGpus*2]);
    args->bufferMemory[args->thread] = std::max(args->bufferMemory[args->thread], initFreeGpuMem[g + args->nGpus] - initFreeGpuMem[g + args->nGpus*2]);
  }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  /* Create device communicators based on test-specific requirements */
  if (deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (!ncclTestEngine.getDevCommRequirements) {
      fprintf(stderr, "Device implementation %d is not supported by this test\n", deviceImpl);
      return testNotImplemented;
    }
    ncclCommProperties commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCLCHECK(ncclCommQueryProperties(args->comms[0][0], &commProperties));
    if (!commProperties.deviceApiSupport) {
      testSkipReason = "Device API is not supported on this system\n";
      return testSkipped;
    }
    TESTCHECK(ncclTestEngine.getDevCommRequirements(deviceImpl, &reqs, &commProperties, &testSkipReason));
#else
    ncclDevCommRequirements reqs = {};
    if (!ncclTestEngine.getDevCommRequirements ||
        !ncclTestEngine.getDevCommRequirements(deviceImpl, &reqs)) {
      fprintf(stderr, "Device implementation %d is not supported by this test\n", deviceImpl);
      return testNotImplemented;
    }
#endif

    ncclResult_t result;
    NCCLCHECK(ncclGroupStart());
    for (int id = 0; id < args->commNum; ++id) {
      for (int i = 0; i < args->nGpus; i++) {
        result = ncclDevCommCreate(args->comms[id][i], &reqs, args->devComms[id]+i);
        if (result != ncclSuccess) {
          NCCLCHECK(ncclGroupEnd());
          return testNcclError;
        }
      }
    }
    result = ncclGroupEnd();
    if (result == ncclInProgress) {
      for (int id = 0; id < args->commNum; ++id) {
        TESTCHECK(waitCommStateBatch(args->comms[id], args->nGpus));
      }
    } else if (result != ncclSuccess) {
      return testNcclError;
    }
  }
  // Capture memory used by test buffers
  int64_t deviceCommMaxMem = 0;
  for (int g = 0; g < args->nGpus; ++g) {
    CUDACHECK(cudaSetDevice(args->gpus[g]));
    int64_t freeGpuMem;
    getGPUMemoryInfo(nullptr, &freeGpuMem);
    deviceCommMaxMem = std::max(deviceCommMaxMem, initFreeGpuMem[g + args->nGpus*2] - freeGpuMem);
  }
  *args->initGpuMem += deviceCommMaxMem;
#endif
  free(initFreeGpuMem);

  TESTCHECK(threadRunTests(args));

  return testSuccess;
}

__global__ void compute(void* _ptr, size_t _size) {
  uint64_t *ptr = (uint64_t*)(_ptr);
  uint64_t size = _size / sizeof(uint64_t);
  ptr += size*blockIdx.x;
  for (uint64_t offset=threadIdx.x; offset < size; offset += blockDim.x) {
     ptr[offset] <<= 1;
  }
}

__global__ void poll(void* _ptr, size_t _size) {
  uint64_t *ptr = (uint64_t*)(_ptr);
  uint64_t size = _size / sizeof(uint64_t);
  ptr += size*blockIdx.x;

  for (uint64_t offset=threadIdx.x; offset < size; offset += blockDim.x) {
    volatile uint64_t* poll_ptr = ptr + offset;
    uint64_t value = *poll_ptr;
    // This ensures the polling actually happens
    if (value != 0) {
    }
  }
}

testResult_t compThread(struct threadArgs* args) {
  void* ptrs[args->nGpus];
  int gpuids[args->nGpus];
  cudaStream_t streams[args->nGpus];
  int gpu0; {
    char* str = getenv("NCCL_TESTS_DEVICE");
    gpu0 = str ? atoi(str) : -1;
  }
  for (int i=0; i<args->nGpus; i++) {
    gpuids[i] = (gpu0 != -1 ? gpu0 : args->localRank*args->nThreads*args->nGpus) + args->thread*args->nGpus + i;
    CUDACHECK(cudaSetDevice(gpuids[i]));
    CUDACHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));
    if (side_comp == 1) CUDACHECK(cudaMalloc(ptrs+i, ((uint64_t)COMP_SIZE)*NUM_BLOCKS));
  }
  while (args->compThreadStop == 0) {
    if (side_comp == 1) {
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaSetDevice(gpuids[i]));
        compute<<<NUM_BLOCKS, 1024, 0, streams[i]>>>(ptrs[i], COMP_SIZE);
      }
      TESTCHECK(testStreamSynchronize(args->nGpus, streams, NULL));
      (*args->compThreadCount)++;
    } else if (side_comp == 2) {
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaSetDevice(gpuids[i]));
        CUDACHECK(cudaMalloc(ptrs+i, ((uint64_t)COMP_SIZE)/1024));
      }
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaSetDevice(gpuids[i]));
        compute<<<1, 1024, 0, streams[i]>>>(ptrs[i], COMP_SIZE/1024);
      }
      TESTCHECK(testStreamSynchronize(args->nGpus, streams, NULL));
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaFree(ptrs[i]));
      }
      fflush(stdout);
      fflush(stderr);
      pid_t pid = fork();
      if (pid == 0) {
        uint64_t* p = (uint64_t*)malloc(sizeof(uint64_t));
        p[0] = 0xfedcba9284353;
        usleep(40000);
        free(p);
        // Do not exit, as it would call the CUDA destructors which may break the parent.
        // Replace with another process that does nothing instead. That also simulates
        // The behavior of a popen() call.
        execl("/bin/true", "/bin/true", NULL);
      } else {
        usleep(40000);
      }
    } else if (side_comp == 3) {
      for (int i=0; i<args->nGpus; i++) {
        CUDACHECK(cudaSetDevice(gpuids[i]));
        // poll the RX buffer from the GPU
        poll<<<NUM_BLOCKS, 1024, 0, streams[i]>>>(args->recvbuffs[0][i], (args->nbytes[0][i])/NUM_BLOCKS);
      }
      TESTCHECK(testStreamSynchronize(args->nGpus, streams, NULL));
      (*args->compThreadCount)++;
    }
  }
  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaStreamDestroy(streams[i]));
    if (side_comp == 1) CUDACHECK(cudaFree(ptrs[i]));
  }
  return testSuccess;
}

void* threadLauncher(void* thread_) {
  struct testThread* thread = (struct testThread*)thread_;
  thread->ret = thread->func(&thread->args);
  return NULL;
}
testResult_t threadLaunch(struct testThread* thread) {
  pthread_create(&thread->thread, NULL, threadLauncher, thread);
  return testSuccess;
}

testResult_t AllocateBuffs(void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes, void **expected, size_t nbytes, size_t *allocBytes) {
    nbytes += 8*unalign; // pad with size of max datatype in case all datatypes selected
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
    NCCLCHECK(ncclMemAlloc(sendbuff, nbytes));
    NCCLCHECK(ncclMemAlloc(recvbuff, nbytes));
    if (datacheck) NCCLCHECK(ncclMemAlloc(expected, recvBytes));
#else
    CUDACHECK(cudaMalloc(sendbuff, nbytes));
    CUDACHECK(cudaMalloc(recvbuff, nbytes));
    if (datacheck) CUDACHECK(cudaMalloc(expected, recvBytes));
#endif
    CUDACHECK(cudaMemset(*sendbuff, 0, nbytes));
    CUDACHECK(cudaMemset(*recvbuff, 0, nbytes));
    if (datacheck) CUDACHECK(cudaMemset(*expected, 0, recvBytes));
    *allocBytes = nbytes;
    return testSuccess;
}

testResult_t run(); // Main function

int main(int argc, char* argv[], char **envp) {
  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,4,0)
    ncclGetVersion(&test_ncclVersion);
  #else
    test_ncclVersion = NCCL_VERSION_CODE;
  #endif
  //printf("# NCCL_VERSION_CODE=%d ncclGetVersion=%d\n", NCCL_VERSION_CODE, test_ncclVersion);
  #if NCCL_VERSION_CODE >= NCCL_VERSION(2,0,0)
    test_opnum = 4;
    test_typenum = 9;
    if (NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0) && test_ncclVersion >= NCCL_VERSION(2,10,0)) {
      test_opnum++; // ncclAvg
    }
    if (NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0) && test_ncclVersion >= NCCL_VERSION(2,11,0)) {
      test_opnum++; // PreMulSum
    }
    #if HAVE_BF16
      test_typenum++; // bfloat16
    #endif
    #if HAVE_FP8
      test_typenum += 2; // fp8 e4m3,e5m2
    #endif
  #endif

  // Parse args
  double parsed;
  int longindex;
  char *output_file = nullptr;

  static struct option longopts[] = {
    {"nthreads", required_argument, 0, 't'},
    {"ngpus", required_argument, 0, 'g'},
    {"minbytes", required_argument, 0, 'b'},
    {"maxbytes", required_argument, 0, 'e'},
    {"stepbytes", required_argument, 0, 'i'},
    {"stepfactor", required_argument, 0, 'f'},
    {"iters", required_argument, 0, 'n'},
    {"agg_iters", required_argument, 0, 'm'},
    {"warmup_iters", required_argument, 0, 'w'},
    {"run_cycles", required_argument, 0, 'N'},
    {"parallel_init", required_argument, 0, 'p'},
    {"check", required_argument, 0, 'c'},
    {"op", required_argument, 0, 'o'},
    {"datatype", required_argument, 0, 'd'},
    {"root", required_argument, 0, 'r'},
    {"blocking", required_argument, 0, 'z'},
    {"stream_null", required_argument, 0, 'y'},
    {"side_comp", required_argument, 0, 'k'},
    {"replay", required_argument, 0, 'l'},
    {"timeout", required_argument, 0, 'T'},
    {"cudagraph", required_argument, 0, 'G'},
    {"report_cputime", required_argument, 0, 'C'},
    {"out_of_place", required_argument, 0, 'O'},
    {"unalign", required_argument, 0, 'u'},
    {"output_file", required_argument, 0, 'J'},
    {"average", required_argument, 0, 'a'},
    {"commblocking", required_argument, 0, 'B'},
    {"ft_test", required_argument, 0, 'F'},
    {"ft_list", required_argument, 0, 'L'},
    {"tbytes", required_argument, 0, 's'},
    {"split_share", required_argument, 0, 'S'},
    {"split_comm", required_argument, 0, 'P'},
    {"local_register", required_argument, 0, 'R'},
    {"cta_policy", required_argument, 0, 'x'},
    {"per_coll_perf", required_argument, 0, 'A'},
    {"simulate", required_argument, 0, 'E'},
    {"init_ids", required_argument, 0, 'I'},
    {"traffic_class", required_argument, 0, 'q'},
    {"tuning", required_argument, 0, 'U'},
    {"device_implementation", required_argument, 0, 'D'},
    {"device_cta_count", required_argument, 0, 'V'},
    {"memory", required_argument, 0, 'M'},
    {"host_rma_implementation", no_argument, 0, 'H'},

    {"help", no_argument, 0, 'h'},
    {}
  };

  while(1) {
    int c;
    c = getopt_long(argc, argv, "t:g:b:e:i:f:n:m:w:N:c:p:o:d:r:I:z:y:k:h:l:T:G:C:O:u:a:B:F:L:s:S:P:R:A:E:J:q:U:x:D:V:M:H", longopts, &longindex);

    if (c == -1)
      break;

    switch(c) {
      case 't':
        nThreads = strtol(optarg, NULL, 0);
        break;
      case 'g':
        nGpus = strtol(optarg, NULL, 0);
        break;
      case 'b':
        parsed = parsesize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'minbytes'\n");
          return -1;
        }
        minBytes = (size_t)parsed;
        break;
      case 'e':
        parsed = parsesize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'maxbytes'\n");
          return -1;
        }
        maxBytes = (size_t)parsed;
        break;
      case 'i':
        parsed = parsesize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'stepBytes'\n");
          return -1;
        }
        stepBytes = (size_t)parsed;
        break;
      case 'f':
        stepFactor = strtol(optarg, NULL, 0);
        break;
      case 'n':
        iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'm':
#if NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 2)
        agg_iters = (int)strtol(optarg, NULL, 0);
#else
        fprintf(stderr, "Option -m not supported before NCCL 2.2. Ignoring\n");
#endif
        break;
      case 'w':
        warmup_iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'N':
        run_cycles = (int)strtol(optarg, NULL, 0);
        break;
      case 'c':
        datacheck = (int)strtol(optarg, NULL, 0);
        break;
      case 'p':
        parallel_init = (int)strtol(optarg, NULL, 0);
        break;
      case 'o':
        ncclop = ncclstringtoop(optarg);
        break;
      case 'd':
        nccltype = ncclstringtotype(optarg);
        break;
      case 'r':
        ncclroot = strtol(optarg, NULL, 0);
        break;
      case 'I':
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,23,0)
        nIdsUser = (int)strtol(optarg, NULL, 0);
        if (nIdsUser < 0) {
          printf("Option -I (--init_ids) invalid value: %d < 0. Ignoring\n", nIdsUser);
          nIdsUser = NCCL_CONFIG_UNDEF_INT;
        }
#else
        printf("Option -I (--init_ids) not supported before NCCL 2.23. Ignoring\n");
#endif
        break;
      case 'z':
        blocking_coll = strtol(optarg, NULL, 0);
        break;
      case 'y':
        streamnull = strtol(optarg, NULL, 0);
        break;
      case 'k':
        side_comp = strtol(optarg, NULL, 0);
        break;
      case 'l':
        replay_file = optarg;
        warmup_iters = 0;    // by default, no warm-up in case of trace replay
        break;
      case 'T':
        timeout = strtol(optarg, NULL, 0);
        break;
      case 'G':
#if (NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 9)) && CUDART_VERSION >= 11030
        cudaGraphLaunches = strtol(optarg, NULL, 0);
#else
        printf("Option -G (CUDA graph) not supported before NCCL 2.9 + CUDA 11.3. Ignoring\n");
#endif
        break;
      case 'C':
        report_cputime = strtol(optarg, NULL, 0);
        break;
      case 'O':
        out_of_place = strtol(optarg, NULL, 0);
        break;
      case 'u':
        unalign = (int)strtol(optarg, NULL, 0);
        break;
      case 'J':
        output_file = strdup(optarg);
        break;
      case 'a':
        average = (int)strtol(optarg, NULL, 0);
        break;
      case 'R':
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
        local_register = (int)strtol(optarg, NULL, 0);
        if (((local_register == SYMMETRIC_REGISTER) || (local_register == SYMMETRIC_REGISTER_SEND) || (local_register == SYMMETRIC_REGISTER_RECV)) &&
          test_ncclVersion < NCCL_VERSION(2, 27, 0)) {
          printf("Option -R 2/3/4 (symmetric) is not supported before NCCL 2.27. Defaulting to local registration\n");
          local_register = LOCAL_REGISTER;
        }
#else
        printf("Option -R (register) is not supported before NCCL 2.19. Ignoring\n");
#endif
        break;
      case 'B':
        commblocking = (int)strtol(optarg, NULL, 0);
        break;
      case 'F':
        ft_test = (int)strtol(optarg, NULL, 0);
        break;
      case 'L':
        ft_list = optarg;
        break;
      case 's':
        parsed = parsesize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'tbytes'\n");
          return -1;
        }
        tbytes = (size_t)parsed;
        break;
      case 'S':
        split_share = (int)strtol(optarg, NULL, 0);
        break;
      case 'P':
        split_comm = (int)strtol(optarg, NULL, 0);
        break;
      case 'A':
        per_coll_perf = (int)strtol(optarg, NULL, 0);
        break;
      case 'E':
        simulate = (int)strtol(optarg, NULL, 0);
        break;
      case 'M':
        memory_report = (int)strtol(optarg, NULL, 0);
        break;
      case 'q':
        trafficClass = (int)strtol(optarg, NULL, 0);
        break;
      case 'U':
        tuning = (int)strtol(optarg, NULL, 0);
        break;
      case 'x':
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
        ctaPolicy = (int)strtol(optarg, NULL, 0);
        if (ctaPolicy > 1 && test_ncclVersion < NCCL_VERSION(2,28,0)) {
          printf("Option -x (cta_policy) %d is not supported before NCCL 2.28. Ignoring\n", ctaPolicy);
          ctaPolicy = -1;
        }
#else
        printf("Option -x (cta_policy) is not supported before NCCL 2.27. Ignoring\n");
#endif
        break;
      case 'D':
        if (test_ncclVersion >= NCCL_VERSION(2,28,0)) {
          deviceImpl = (int)strtol(optarg, NULL, 0);
        } else {
          fprintf(stderr, "Option -D (device implementation) requires NCCL >= 2.28.0\n");
          return -1;
        }
        break;
      case 'V':
        if (test_ncclVersion >= NCCL_VERSION(2,28,0)) {
          deviceCtaCount = (int)strtol(optarg, NULL, 0);
          if (deviceCtaCount <= 0 || deviceCtaCount > 128) {
            fprintf(stderr, "device_cta_count (-V) must be positive and less than 128, got %d. "
                    "Using default value 16.\n", deviceCtaCount);
            deviceCtaCount = 16;
          }
        } else {
          fprintf(stderr, "Option -V (device CTA count) requires NCCL >= 2.28.0\n");
          return -1;
        }
        break;
      case 'H':
        if (test_ncclVersion >= NCCL_VERSION(2,29,0)) {
          hostRmaImpl = 1;
        } else {
          fprintf(stderr, "Option -H (host RMA implementation) requires NCCL >= 2.29.0\n");
          return -1;
        }
        break;
      case 'h':
      default:
        if (c != 'h') printf("invalid option '%c'\n", c);
        printf("USAGE: %s \n\t"
            "[-t,--nthreads <num threads>] \n\t"
            "[-g,--ngpus <gpus per thread>] \n\t"
            "[-b,--minbytes <min size in bytes>] \n\t"
            "[-e,--maxbytes <max size in bytes>] \n\t"
            "[-i,--stepbytes <increment size>] \n\t"
            "[-f,--stepfactor <increment factor>] \n\t"
            "[-n,--iters <iteration count>] \n\t"
            "[-m,--agg_iters <aggregated iteration count>] \n\t"
            "[-w,--warmup_iters <warmup iteration count>] \n\t"
            "[-N,--run_cycles <cycle count> run & print each cycle (default: 1; 0=infinite)] \n\t"
            "[-p,--parallel_init <0/1>] \n\t"
            "[-c,--check <check iteration count>] \n\t"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0)
            "[-o,--op <sum/prod/min/max/avg/mulsum/all>] \n\t"
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
            "[-o,--op <sum/prod/min/max/avg/all>] \n\t"
#else
            "[-o,--op <sum/prod/min/max/all>] \n\t"
#endif
            "[-d,--datatype <nccltype/all>] \n\t"
            "[-r,--root <root>] \n\t"
            "[-z,--blocking <0/1>] \n\t"
            "[-y,--stream_null <0/1>] \n\t"
            "[-k,--side_comp <0/1>] \n\t"
            "[-l,--replay <path to replay file>] \n\t"
            "[-T,--timeout <time in seconds>] \n\t"
            "[-G,--cudagraph <num graph launches>] \n\t"
            "[-C,--report_cputime <0/1>] \n\t"
            "[-O,--out_of_place <0/1>] \n\t"
            "[-u,--unalign <index of first element>] \n\t"
            "[-J,--output_file <file> write output to filepath, if accessible. Infer type from suffix (only json supported presently.)] \n\t"
            "[-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] \n\t"
            "[-R,--local_register <0/1/2/3/4> enable local (1) or symmetric (2/3/4) buffer registration on send buffers (3)/recv buffers (4)/all buffers (1/2) (default: disable 0)] \n\t"
            "[-x,--cta_policy <0/1/2> set CTA policy (NCCL_CTA_POLICY_DEFAULT (0), NCCL_CTA_POLICY_EFFICIENCY (1), NCCL_CTA_POLICY_ZERO (2)) (default: do not set)] \n\t"
            "[-B,--commblocking <0/1> enable blocking communicator (default: 1)] \n\t"
            "[-F,--ft_test <0/1> enable fault tolerance test (default: 0)] \n\t"
            "[-L,--ft_list <init/allreduce/alltoall/finalize/split/all> only enable specified fault tolerance test (default: all)] \n\t"
            "[-s,--tbytes total bytes allowed to transmit (default: unlimited); tbytes would limit #iterations] \n\t"
            "[-S,--split_share <0/1> enable shared resources during communicator split (default: 0)] \n\t"
            "[-P,--split_comm <0/1/2> enable communicator split (default: 0 disable; 1 dup global comm; 2 three split patterns)] \n\t"
            "[-A,--per_coll_perf <0/1/2> Report performance per-collective (default: 0 disable; 1 report per-collective performance and std deviation; 2: report only std deviation)] \n\t"
            "[-I,--init_ids <num ids> enable scalable API for ncclCommInitRank using <num ids> ncclUniqueIds (default: disabled; 0 is equivalent to 1 ncclUniqueId per 128 NCCL ranks; value must be >=0)] \n\t"
            "[-q,--traffic_class <tclass> set network traffic class] \n\t"
            "[-U,--tuning <0/1> report NCCL tuning info (default: 0)] \n\t"
            "[-D,--device_implementation <implementation number> enable device implementation (default: 0, use NCCL implementation; requires -R 2 if > 0)] \n\t"
            "[-V,--device_cta_count <number> set number of CTAs for device implementation (default: 16)] \n\t"
            "[-H,--host_rma_implementation enable Host RMA API implementations (requires -R 2)] \n\t"
            "[-M,--memory_report <0/1> enable memory usage report (default: 0)] \n\t"

            "[-h,--help]\n",
          basename(argv[0]));
        return 0;
    }
  }
  if (minBytes > maxBytes) {
    fprintf(stderr, "invalid sizes for 'minbytes' and 'maxbytes': %llu > %llu\n",
           (unsigned long long)minBytes,
           (unsigned long long)maxBytes);
    return -1;
  }
  if (hostRmaImpl && deviceImpl > 0) {
    fprintf(stderr, "Cannot use both -H (host RMA implementation) and -D (device implementation) at the same time\n");
    return -1;
  }
  if (deviceImpl > 0 && (local_register != SYMMETRIC_REGISTER)) {
    fprintf(stderr, "device implementation (-D > 0) requires enabling symmetric memory registration (-R 2)\n");
    return -1;
  }
  if (hostRmaImpl && (local_register != SYMMETRIC_REGISTER)) {
    fprintf(stderr, "host RMA implementation (-H) requires enabling symmetric memory registration (-R 2)\n");
    return -1;
  }

  // Let NCCL load the perftest profiler implementation
  if (tuning) {
    setenv("NCCL_PROFILER_PLUGIN", "STATIC_PLUGIN", 1);
  } else {
    if (ncclProfilerLoad() == 0) {
      const char* profilerMaskStr = getenv("NCCL_PERF_PROFILER_MASK");
      if (profilerMaskStr) {
        profilerMask = strtol(profilerMaskStr, nullptr, 0);
      }
      const char* profilerDumpStr = getenv("NCCL_PERF_PROFILER_DUMP");
      if (profilerDumpStr) {
        profilerDump = (char *)profilerDumpStr;
      }
      const char* profilerItersStr = getenv("NCCL_PERF_PROFILER_ITERS");
      if (profilerItersStr) {
        profilerIters = strtol(getenv("NCCL_PERF_PROFILER_ITERS"), nullptr, 0);
      }
      if (profilerMask != 0) {
        setenv("NCCL_PROFILER_PLUGIN", "example", 1);
      }
    }
  }
#ifdef MPI_SUPPORT
  int provide;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provide);
  assert(provide >= MPI_THREAD_SERIALIZED);
#endif

  const output_file_type_t output_file_type = classifyOutputFile(output_file);
  outputFileInit(output_file_type, output_file, argc, argv, envp);

  if(output_file) {
    free(output_file);
    output_file = nullptr;
  }

  testResult_t result = run();

  outputFileFinalize(output_file_type);

  ncclProfilerUnload();

  if (result == testSkipped) {
    if (is_main_proc) {
      printf("# TEST SKIPPED: %s\n", testSkipReason ? testSkipReason : "Unknown reason");
    }
  #ifdef MPI_SUPPORT
    MPI_Finalize();
  #endif
    return 0;
  }

  TESTCHECK(result);

  return 0;
}

// parse int for base 2/10/16, will ignore first whitespaces
static bool parseInt(char *s, int *num) {
  char *p = NULL;
  if (!s || !num)
    return false;
  while (*s && isspace(*s)) ++s;
  if (!*s) return false;

  if (strncasecmp(s, "0b", 2) == 0)
    *num = (int)strtoul(s + 2, &p, 2);
  else
    *num = (int)strtoul(s, &p, 0);

  if (p == s)
    return false;
  return true;
}

testResult_t run() {
  int totalProcs = 1, proc = 0, ncclProcs = 1, ncclProc = 0, color = 0;
  int localRank = 0;
  char hostname[1024];
  getHostName(hostname, 1024);

#ifdef MPI_SUPPORT
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  uint64_t hostHashs[totalProcs];
  hostHashs[proc] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
  for (int p=0; p<totalProcs; p++) {
    if (p == proc) break;
    if (hostHashs[p] == hostHashs[proc]) localRank++;
  }

  if (splitMaskEnv = getenv("NCCL_TESTS_SPLIT_MASK")) {
    color = proc & strtoul(splitMaskEnv, NULL, 16);
  } else if (splitMaskEnv = getenv("NCCL_TESTS_SPLIT")) {
    if (
      (strncasecmp(splitMaskEnv, "AND", strlen("AND")) == 0 && parseInt(splitMaskEnv + strlen("AND"), &color)) ||
      (strncasecmp(splitMaskEnv, "&", strlen("&")) == 0 && parseInt(splitMaskEnv + strlen("&"), &color))
    )
        color = proc & color;
    if (
      (strncasecmp(splitMaskEnv, "OR", strlen("OR")) == 0 && parseInt(splitMaskEnv + strlen("OR"), &color)) ||
      (strncasecmp(splitMaskEnv, "|", strlen("|")) == 0 && parseInt(splitMaskEnv + strlen("|"), &color))
    )
        color = proc | color;
    if (
      (strncasecmp(splitMaskEnv, "MOD", strlen("MOD")) == 0 && parseInt(splitMaskEnv + strlen("MOD"), &color)) ||
      (strncasecmp(splitMaskEnv, "%", strlen("%")) == 0 && parseInt(splitMaskEnv + strlen("%"), &color))
    )
        color = proc % color;
    if (
      (strncasecmp(splitMaskEnv, "DIV", strlen("DIV")) == 0 && parseInt(splitMaskEnv + strlen("DIV"), &color)) ||
      (strncasecmp(splitMaskEnv, "/", strlen("/")) == 0 && parseInt(splitMaskEnv + strlen("/"), &color))
    )
        color = proc / color;
  }

  MPI_Comm mpi_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, proc, &mpi_comm);
  MPI_Comm_size(mpi_comm, &ncclProcs);
  MPI_Comm_rank(mpi_comm, &ncclProc);
#endif
  is_main_thread = is_main_proc = (proc == 0) ? 1 : 0;

  jsonIdentifyWriter(is_main_thread);

  char* envstr = getenv("NCCL_TESTS_DUMP_FILE");
  if (envstr && is_main_proc) dump_file = fopen(envstr, "w");

  size_t maxMem = ~0;
  testResult_t report_result = writeDeviceReport(&maxMem, localRank, proc, totalProcs, color, hostname);
  if(report_result != testSuccess) {
    return report_result;
  }

  /* Now we support 4 split pattern when split_comm is enabled:
   * (1) keep all ranks in a group but in reversed order;
   * (2) split ranks into 2 groups based odd and even rank;
   * (3) split ranks into 1ppn on non-MNNVL platform.
   * (4) keep all ranks in a group but in reversed order (duplicate)
   * If NCCL_TESTS_SPLIT_MASK is set, we only split based on split mask. */
  if (splitMaskEnv == NULL && split_comm == 2) {
    commNum = 4;
    agg_iters = 1; /* we cannot aggregate coll on multiple split communicators. */
  } else if (split_comm == 1) {
    commNum = 1;
  }
  // We need sendbuff, recvbuff, expected (when datacheck enabled), plus 1G for the rest.
  size_t reserveMem =  std::min(DIVUP(maxMem, (16ULL << 30)) * (1ULL << 30), 4ULL << 30);
  size_t memMaxBytes = (maxMem - reserveMem * commNum - (1LL << 30)) / (datacheck ? 3 : 2) / commNum;
  assert(maxMem > reserveMem * commNum + (1LL << 30));
  if (maxBytes > memMaxBytes) {
    maxBytes = memMaxBytes;
    if (minBytes > maxBytes) minBytes = maxBytes;
    if (proc == 0) printf("#\n# Reducing maxBytes to %ld due to memory limitation\n", maxBytes);
  }

  int gpus[nGpus*nThreads];
  cudaStream_t streams[nGpus*nThreads];
  void* sendbuffs[commNum][nGpus*nThreads];
  void* recvbuffs[commNum][nGpus*nThreads];
  void* expected[commNum][nGpus*nThreads];
  memset(sendbuffs, 0, sizeof(sendbuffs));
  memset(recvbuffs, 0, sizeof(recvbuffs));
  memset(expected, 0, sizeof(expected));
  size_t sendBytes, recvBytes;

  /* only when communicators are nonblocking and ft test is enabled, we
   * perform fault tolerance tests. */
  if (ft_test && commblocking == 0) {
    TESTCHECK(faultToleranceTests(nThreads, nGpus, ncclProc, ncclProcs, localRank, ft_list));
  }

  envstr = getenv("NCCL_TESTS_DEVICE");
  int gpu0 = envstr ? atoi(envstr) : -1;
  minCudaArch = 1<<30;
  for (int i = 0; i < nGpus * nThreads; ++i) {
    gpus[i] = (gpu0 != -1 ? gpu0 : localRank * nThreads * nGpus) + i;
    CUDACHECK(cudaSetDevice(gpus[i]));
    if (streamnull) {
      streams[i] = NULL;
    } else {
      CUDACHECK(cudaStreamCreateWithFlags(streams + i, cudaStreamNonBlocking));
    }
    int archMajor, archMinor;
    CUDACHECK(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, gpus[i]));
    CUDACHECK(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, gpus[i]));
    minCudaArch = std::min(minCudaArch, 100*archMajor + 10*archMinor);
  }
#ifdef MPI_SUPPORT
  MPI_Allreduce(MPI_IN_PLACE, &minCudaArch, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
  char* ncclIdLocal;
  ncclUniqueId* ncclId;
  ncclComm_t globalComms[nThreads*nGpus];
  ncclComm_t comms[commNum][nThreads*nGpus];
  memset(globalComms, 0, sizeof(globalComms));
  memset(comms, 0, sizeof(comms));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  ncclDevComm devComms[commNum][nThreads*nGpus];
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
  void* sendRegHandles[commNum][nThreads*nGpus];
  void* recvRegHandles[commNum][nThreads*nGpus];
  memset(sendRegHandles, 0, sizeof(sendRegHandles));
  memset(recvRegHandles, 0, sizeof(recvRegHandles));
#endif
  int nranks = totalProcs * nThreads * nGpus;
  // make sure the number of Ids does not exceed the number of total ranks
  int nIds = 1;
  if (nIdsUser == 0) {
    nIds = (nranks > 128) ? (nranks / 128) : 1;
  } else if (nIdsUser != NCCL_CONFIG_UNDEF_INT && nIdsUser > 0) {
    nIds = (nIdsUser > nranks) ? nranks : nIdsUser;
  }
  // each proc can have multiple local ranks, so multiple local roots
  int nIdsLocal = 0;
  for (int i = 0; i < nGpus * nThreads; ++i) {
    nIdsLocal += rankHasRoot(proc * nThreads * nGpus + i, nranks, nIds);
  }
  ncclIdLocal = (char*)calloc(nIdsLocal, NCCL_UNIQUE_ID_BYTES);
  for (int i = 0; i < nIdsLocal; ++i) {
    NCCLCHECK(ncclGetUniqueId((ncclUniqueId*)(ncclIdLocal + i * NCCL_UNIQUE_ID_BYTES)));
  }
#ifdef MPI_SUPPORT
  ncclId = (ncclUniqueId*)calloc(nIds, NCCL_UNIQUE_ID_BYTES);
  int sendCount = nIdsLocal * NCCL_UNIQUE_ID_BYTES;
  int* recvCount = (int*)calloc(totalProcs, sizeof(int));
  int* recvDispl = (int*)calloc(totalProcs, sizeof(int));
  // all gather how many roots per proc
  MPI_Allgather(&sendCount, 1, MPI_INT, recvCount, 1, MPI_INT, MPI_COMM_WORLD);
  for (int i = 1; i < totalProcs; ++i) {
    recvDispl[i] = recvDispl[i - 1] + recvCount[i - 1];
  }
  MPI_Allgatherv(ncclIdLocal, sendCount, MPI_CHAR, ncclId, recvCount, recvDispl, MPI_CHAR, MPI_COMM_WORLD);
  free(recvDispl);
  free(recvCount);
  free(ncclIdLocal);           // can free the local list, the whole list is free'd at test completion
  MPI_Barrier(MPI_COMM_WORLD); // Ensure Bcast is complete for HCOLL
#else
  ncclId = (ncclUniqueId*)ncclIdLocal;
#endif
  int64_t initGpuMem[nThreads] = {0};
  int64_t bufferMemory[nThreads] = {0};
  if (!parallel_init) {
    // Capture the memory used by the GPUs before initializing the NCCL communicators
    int64_t* initFreeGpuMem = (int64_t*)calloc(nGpus*3, sizeof(int64_t));
    for (int g = 0; g < nGpus; ++g) {
      CUDACHECK(cudaSetDevice(gpus[g]));
      getGPUMemoryInfo(nullptr, &initFreeGpuMem[g]);
    }
    //if parallel init is not selected, use main thread to initialize NCCL
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = commblocking;
    config.splitShare = split_share;
    config.trafficClass = trafficClass;
    config.commName = "perftest";
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
    if (ctaPolicy >= 0)
      config.CTAPolicy = ctaPolicy;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    config.nvlinkCentricSched = 1;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
    if (cudaGraphLaunches >= 1)
      config.graphUsageMode = 1;
    else
      config.graphUsageMode = 0;
#endif
#endif
#endif
#endif
    NCCLCHECK(ncclGroupStart());
    for (int i=0; i<nGpus*nThreads; i++) {
      CUDACHECK(cudaSetDevice(gpus[i]));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 23, 0)
      if (nIdsUser != NCCL_CONFIG_UNDEF_INT) {
        NCCLCHECK(ncclCommInitRankScalable(globalComms + i, nranks, proc * nThreads * nGpus + i, nIds, ncclId, &config));
      } else {
        NCCLCHECK(ncclCommInitRankConfig(globalComms + i, nranks, *ncclId, proc * nThreads * nGpus + i, &config));
      }
#else
      NCCLCHECK(ncclCommInitRankConfig(globalComms + i, nranks, *ncclId, proc * nThreads * nGpus + i, &config));
#endif
    }
    NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, nGpus * nThreads);
    /* split comm if required. */
    if (splitMaskEnv) {
      /* split based on split mask */
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < nGpus * nThreads; ++i) {
        NCCLCHECK(ncclCommSplit(globalComms[i], color, proc, &comms[0][i], &config));
      }
      NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, nGpus * nThreads);
    } else if (split_comm == 2) {
      int nDevs;
      CUDACHECK(cudaGetDeviceCount(&nDevs));
      /* create split comm with predefined split pattern. */
      for (int splitCase = 0; splitCase < commNum; ++splitCase) {
        switch (splitCase) {
          case 0: {
            /* duplicate communicator but in reversed rank */
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nGpus * nThreads; ++i) {
              int myrank = proc * nThreads * nGpus + i;
              NCCLCHECK(ncclCommSplit(globalComms[i], 0, nranks - myrank, &comms[splitCase][i], &config));
            }
            NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, nGpus * nThreads);
            break;
          }
          case 1: {
            /* split with odd and even rank */
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nGpus * nThreads; ++i) {
              int myrank = proc * nThreads * nGpus + i;
              NCCLCHECK(ncclCommSplit(globalComms[i], myrank & 1, myrank, &comms[splitCase][i], &config));
            }
            NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, nGpus * nThreads);
            break;
          }
          case 2: {
            /* 1ppn split */
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nGpus * nThreads; ++i) {
              int myrank = proc * nThreads * nGpus + i;
              NCCLCHECK(ncclCommSplit(globalComms[i], myrank % nDevs, myrank, &comms[splitCase][i], &config));
            }
            NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, nGpus * nThreads);
            break;
          }
          case 3: {
            /* duplicate communicator but in reversed rank (again) */
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nGpus * nThreads; ++i) {
              int myrank = proc * nThreads * nGpus + i;
              NCCLCHECK(ncclCommSplit(globalComms[i], 0, nranks - myrank, &comms[splitCase][i], &config));
            }
            NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, nGpus * nThreads);
            break;
          }
                /* we can add more to split pattern */
          default:
            return testInternalError;
        }
      }
    } else if (split_comm == 1) {
      /* duplicate globalcomm. */
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < nGpus * nThreads; i++) {
        NCCLCHECK(ncclCommSplit(globalComms[i], 0, 0, &comms[0][i], &config));
      }
      NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, nGpus * nThreads);
    } else {
      /* no split required by users */
      for (int i = 0; i < nGpus * nThreads; ++i) comms[0][i] = globalComms[i];
    }

    if (split_comm || splitMaskEnv) {
      /* destroy global NCCL communicators */
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < nGpus * nThreads; i++) {
        NCCLCHECK(ncclCommFinalize(globalComms[i]));
      }
      NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, nGpus * nThreads);

      for (int i = 0; i < nGpus * nThreads; ++i)
        NCCLCHECK(ncclCommDestroy(globalComms[i]));
    }

    // Capture the memory used by the GPUs after initializing the NCCL communicators
    for (int g = 0; g < nGpus; ++g) {
      CUDACHECK(cudaSetDevice(gpus[g]));
      getGPUMemoryInfo(nullptr, &initFreeGpuMem[g + nGpus]);
    }
    for ( size_t t = 0; t < nThreads; ++t) {
      for (int g = 0; g < nGpus; ++g) {
        initGpuMem[t] = std::max(initGpuMem[t], initFreeGpuMem[g] - initFreeGpuMem[g + nGpus]);
      }
    }
    /* allocate buffer for each split comm. */
    NCCLCHECK(ncclGroupStart());
    for (int id = 0; id < commNum; ++id) {
      for (int i = 0; i < nGpus * nThreads; i++) {
        int nranks;
        size_t allocBytes;
        NCCLCHECK(ncclCommCount(comms[id][i], &nranks));
        ncclTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes, (size_t)nranks);
        CUDACHECK(cudaSetDevice(gpus[i]));
        TESTCHECK(AllocateBuffs(sendbuffs[id] + i, sendBytes, recvbuffs[id] + i, recvBytes, expected[id] + i, (size_t)maxBytes, &allocBytes));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
        if (local_register == SYMMETRIC_REGISTER_SEND) {
          NCCLCHECK(ncclCommWindowRegister(comms[id][i], sendbuffs[id][i], allocBytes, (ncclWindow_t*)&sendRegHandles[id][i], NCCL_WIN_COLL_SYMMETRIC));
        } else if (local_register == SYMMETRIC_REGISTER_RECV) {
          NCCLCHECK(ncclCommWindowRegister(comms[id][i], recvbuffs[id][i], allocBytes, (ncclWindow_t*)&recvRegHandles[id][i], NCCL_WIN_COLL_SYMMETRIC));
        } else if (local_register == SYMMETRIC_REGISTER) {
          NCCLCHECK(ncclCommWindowRegister(comms[id][i], sendbuffs[id][i], allocBytes, (ncclWindow_t*)&sendRegHandles[id][i], NCCL_WIN_COLL_SYMMETRIC));
          NCCLCHECK(ncclCommWindowRegister(comms[id][i], recvbuffs[id][i], allocBytes, (ncclWindow_t*)&recvRegHandles[id][i], NCCL_WIN_COLL_SYMMETRIC));
        } else {
          if (local_register == LOCAL_REGISTER) NCCLCHECK(ncclCommRegister(comms[id][i], sendbuffs[id][i], allocBytes, &sendRegHandles[id][i]));
          if (local_register == LOCAL_REGISTER) NCCLCHECK(ncclCommRegister(comms[id][i], recvbuffs[id][i], allocBytes, &recvRegHandles[id][i]));
        }
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
        if (local_register == LOCAL_REGISTER) NCCLCHECK(ncclCommRegister(comms[id][i], sendbuffs[id][i], allocBytes, &sendRegHandles[id][i]));
        if (local_register == LOCAL_REGISTER) NCCLCHECK(ncclCommRegister(comms[id][i], recvbuffs[id][i], allocBytes, &recvRegHandles[id][i]));
#endif
      }
    }
    NCCLCHECK(ncclGroupEnd());
    // Capture memory used by after allocating buffers
    for (int g = 0; g < nGpus; ++g) {
      CUDACHECK(cudaSetDevice(gpus[g]));
      getGPUMemoryInfo(nullptr, &initFreeGpuMem[g + nGpus*2]);
    }
    for ( size_t t = 0; t < nThreads; ++t) {
      for (int g = 0; g < nGpus; ++g) {
        bufferMemory[t] = std::max(bufferMemory[t], initFreeGpuMem[g + nGpus] - initFreeGpuMem[g + nGpus*2]);
      }
    }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    /* Create device communicators based on test-specific requirements */
    if (deviceImpl) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
      ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      if (!ncclTestEngine.getDevCommRequirements) {
        fprintf(stderr, "Device implementation %d is not supported by this test\n", deviceImpl);
        return testNotImplemented;
      }
      ncclCommProperties commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
      NCCLCHECK(ncclCommQueryProperties(comms[0][0], &commProperties));
      if (!commProperties.deviceApiSupport) {
        testSkipReason = "Device API is not supported on this system\n";
        return testSkipped;
      }
      TESTCHECK(ncclTestEngine.getDevCommRequirements(deviceImpl, &reqs, &commProperties, &testSkipReason));
#else
      ncclDevCommRequirements reqs = {};
      if (!ncclTestEngine.getDevCommRequirements ||
        !ncclTestEngine.getDevCommRequirements(deviceImpl, &reqs)) {
        fprintf(stderr, "Device implementation %d is not supported by this test\n", deviceImpl);
        return testNotImplemented;
      }
#endif

      ncclResult_t result;
      NCCLCHECK(ncclGroupStart());
      for (int id = 0; id < commNum; ++id) {
        for (int i = 0; i < nGpus * nThreads; i++) {
          result = ncclDevCommCreate(comms[id][i], &reqs, devComms[id]+i);
          if (result != ncclSuccess) {
            NCCLCHECK(ncclGroupEnd());
            return testNcclError;
          }
        }
      }
      result = ncclGroupEnd();
      if (result == ncclInProgress) {
        for (int id = 0; id < commNum; ++id) {
          TESTCHECK(waitCommStateBatch(comms[id], nGpus * nThreads));
        }
      } else if (result != ncclSuccess) {
        return testNcclError;
      }
    }
    int64_t deviceCommMaxMem = 0;
    for (int g = 0; g < nGpus; ++g) {
      CUDACHECK(cudaSetDevice(gpus[g]));
      int64_t freeGpuMem;
      getGPUMemoryInfo(nullptr, &freeGpuMem);
      deviceCommMaxMem = std::max(deviceCommMaxMem, initFreeGpuMem[g + nGpus*2] - freeGpuMem);
    }
    for ( size_t t = 0; t < nThreads; ++t) {
      initGpuMem[t] += deviceCommMaxMem;
    }
#endif
    free(initFreeGpuMem);
  }

  int errors[nThreads];
  double bw[nThreads];
  int64_t devMemUsed[nThreads];
  int bw_count[nThreads];
  for (int t=0; t<nThreads; t++) {
    bw[t] = 0.0;
    errors[t] = bw_count[t] = 0;
    devMemUsed[t] = std::numeric_limits<int64_t>::min();
  }

  writeResultHeader(report_cputime, simulate);

  struct testThread threads[nThreads];
  struct testThread compThreads[nThreads];
  memset(threads, 0, sizeof(struct testThread)*nThreads);

  if (side_comp && signal(SIGCHLD, SIG_IGN) == SIG_ERR) {
    printf("Failed to set up automatic cleanup of zombie processes\n");
    exit(EXIT_FAILURE);
  }
  int compThreadCounts[nThreads];
  memset(compThreadCounts, 0, sizeof(int)*nThreads);

  for (int t=nThreads-1; t>=0; t--) {
    threads[t].args.minbytes=minBytes;
    threads[t].args.maxbytes=maxBytes;
    threads[t].args.stepbytes=stepBytes;
    threads[t].args.stepfactor=stepFactor;
    threads[t].args.localRank = localRank;

    threads[t].args.totalProcs=totalProcs;
    threads[t].args.nProcs=ncclProcs;
    threads[t].args.proc=ncclProc;
    threads[t].args.globalProc=proc;
    threads[t].args.nThreads=nThreads;
    threads[t].args.thread=t;
    threads[t].args.nGpus=nGpus;
    threads[t].args.gpus=gpus+t*nGpus;

    threads[t].args.sendbuffs = (void***)malloc(sizeof(void**) * commNum);
    threads[t].args.recvbuffs = (void***)malloc(sizeof(void**) * commNum);
    threads[t].args.expected = (void***)malloc(sizeof(void**) * commNum);
    threads[t].args.comms = (ncclComm_t**)malloc(sizeof(ncclComm_t*) * commNum);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    threads[t].args.devComms = (ncclDevComm**)malloc(sizeof(ncclDevComm*) * commNum);
#endif
    threads[t].args.sendBytes = (size_t**)malloc(sizeof(size_t*) * commNum);
    threads[t].args.expectedBytes = (size_t**)malloc(sizeof(size_t*) * commNum);
    threads[t].args.sendInplaceOffset = (size_t**)malloc(sizeof(size_t*) * commNum);
    threads[t].args.recvInplaceOffset = (size_t**)malloc(sizeof(size_t*) * commNum);
    threads[t].args.nbytes = (size_t**)malloc(sizeof(size_t*) * commNum);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
    threads[t].args.sendRegHandles = (void***)malloc(sizeof(threads[t].args.sendRegHandles[0]) * commNum);
    threads[t].args.recvRegHandles = (void***)malloc(sizeof(threads[t].args.recvRegHandles[0]) * commNum);
#endif
    for (int id = 0; id < commNum; ++id) {
      threads[t].args.sendbuffs[id] = sendbuffs[id]+t*nGpus;
      threads[t].args.recvbuffs[id] = recvbuffs[id]+t*nGpus;
      threads[t].args.expected[id] = expected[id]+t*nGpus;
      threads[t].args.comms[id] = comms[id]+t*nGpus;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
      threads[t].args.devComms[id] = devComms[id]+t*nGpus;
#endif
      threads[t].args.sendBytes[id] = (size_t*)malloc(sizeof(size_t) * nGpus);
      threads[t].args.expectedBytes[id] = (size_t*)malloc(sizeof(size_t) * nGpus);
      threads[t].args.sendInplaceOffset[id] = (size_t*)malloc(sizeof(size_t) * nGpus);
      threads[t].args.recvInplaceOffset[id] = (size_t*)malloc(sizeof(size_t) * nGpus);
      threads[t].args.nbytes[id] = (size_t*)malloc(sizeof(size_t) * nGpus);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
      threads[t].args.sendRegHandles[id] = sendRegHandles[id]+t*nGpus;
      threads[t].args.recvRegHandles[id] = recvRegHandles[id]+t*nGpus;
#endif
    }

    threads[t].args.commNum = commNum;
    threads[t].args.nIds = nIds;
    threads[t].args.ncclId = ncclId;
    threads[t].args.streams=streams+t*nGpus;

    threads[t].args.errors=errors+t;
    threads[t].args.bw=bw+t;
    threads[t].args.bw_count=bw_count+t;
    threads[t].args.initGpuMem = initGpuMem + t;
    threads[t].args.bufferMemory = bufferMemory + t;
    threads[t].args.devMemUsed = devMemUsed + t;

    threads[t].args.reportErrors = datacheck;

    threads[t].args.replayFile = replay_file;

    threads[t].args.compThreadCount = compThreadCounts+t;

    if (side_comp) {
      memset(compThreads+t, 0, sizeof(struct testThread));
      memcpy(&compThreads[t].args, &threads[t].args, sizeof(struct threadArgs));
      compThreads[t].func = compThread;
      TESTCHECK(threadLaunch(compThreads+t));
    }
    threads[t].func = parallel_init ? threadInit : threadRunTests;
    if (t)
      TESTCHECK(threadLaunch(threads+t));
    else
      TESTCHECK(threads[t].func(&threads[t].args));
  }

  // Wait for other threads and accumulate stats and errors
  for (int t=nThreads-1; t>=0; t--) {
    if (t) pthread_join(threads[t].thread, NULL);
    if (threads[t].ret != testSkipped) {
      TESTCHECK(threads[t].ret);
    }
    if (t) {
      errors[0] += errors[t];
      bw[0] += bw[t];
      bw_count[0] += bw_count[t];
      devMemUsed[0] = std::max(devMemUsed[0], devMemUsed[t]);
      initGpuMem[0] = std::max(initGpuMem[0], initGpuMem[t]);
      bufferMemory[0] = std::max(bufferMemory[0], bufferMemory[t]);
    }
    if (side_comp) {
       compThreads[t].args.compThreadStop = 1;
       pthread_join(compThreads[t].thread, NULL);
       if (compThreads[t].ret != testSkipped) {
         TESTCHECK(compThreads[t].ret);
       }
    }
    for (int id = 0; id < commNum; ++id) {
      free(threads[t].args.sendBytes[id]);
      free(threads[t].args.expectedBytes[id]);
      free(threads[t].args.sendInplaceOffset[id]);
      free(threads[t].args.recvInplaceOffset[id]);
      free(threads[t].args.nbytes[id]);
    }
    free(threads[t].args.sendbuffs);
    free(threads[t].args.recvbuffs);
    free(threads[t].args.expected);
    free(threads[t].args.comms);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    free(threads[t].args.devComms);
#endif
    free(threads[t].args.sendBytes);
    free(threads[t].args.expectedBytes);
    free(threads[t].args.sendInplaceOffset);
    free(threads[t].args.recvInplaceOffset);
    free(threads[t].args.nbytes);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
    free(threads[t].args.sendRegHandles);
    free(threads[t].args.recvRegHandles);
#endif
  }
  // once all threads are done, free the ncclIds
  free(ncclId);

#ifdef MPI_SUPPORT
  MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &devMemUsed[0], 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &initGpuMem[0], 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &bufferMemory[0], 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
#endif

  // Free off CUDA allocated memory
  for (int id = 0; id < commNum; ++id) {
    for (int i=0; i<nGpus*nThreads; i++) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
      if (test_ncclVersion >= NCCL_VERSION(2,27,0) && !(local_register == LOCAL_REGISTER)) {
        if (local_register == SYMMETRIC_REGISTER_SEND) {
          NCCLCHECK(ncclCommWindowDeregister(comms[id][i], (ncclWindow_t)sendRegHandles[id][i]));
        } else if (local_register == SYMMETRIC_REGISTER_RECV) {
          NCCLCHECK(ncclCommWindowDeregister(comms[id][i], (ncclWindow_t)recvRegHandles[id][i]));
        } else if (local_register == SYMMETRIC_REGISTER) {
          NCCLCHECK(ncclCommWindowDeregister(comms[id][i], (ncclWindow_t)sendRegHandles[id][i]));
          NCCLCHECK(ncclCommWindowDeregister(comms[id][i], (ncclWindow_t)recvRegHandles[id][i]));
        }
      } else
#endif
      {
        if (local_register) NCCLCHECK(ncclCommDeregister(comms[id][i], sendRegHandles[id][i]));
        if (local_register) NCCLCHECK(ncclCommDeregister(comms[id][i], recvRegHandles[id][i]));
      }
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
      if (deviceImpl) {
        NCCLCHECK(ncclDevCommDestroy(comms[id][i], devComms[id]+i));
      }
#endif
      NCCLCHECK(ncclCommDestroy(comms[id][i]));
    }
  }

  for (int id = 0; id < commNum; ++id) {
    for (int i=0; i<nGpus*nThreads; i++) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
      if (sendbuffs[id][i]) NCCLCHECK(ncclMemFree(sendbuffs[id][i]));
      if (recvbuffs[id][i]) NCCLCHECK(ncclMemFree(recvbuffs[id][i]));
      if (datacheck) NCCLCHECK(ncclMemFree(expected[id][i]));
#else
      if (sendbuffs[id][i]) CUDACHECK(cudaFree(sendbuffs[id][i]));
      if (recvbuffs[id][i]) CUDACHECK(cudaFree(recvbuffs[id][i]));
      if (datacheck) NCCLCHECK(cudaFree(expected[id][i]));
#endif
    }
  }

  envstr = getenv("NCCL_TESTS_MIN_BW");
  const double check_avg_bw = envstr ? atof(envstr) : -1;
  bw[0] /= bw_count[0];

  writeResultFooter(errors, bw, check_avg_bw);
  if (memory_report) {
    memInfo_t memInfos[3];
    memInfos[0] = { initGpuMem[0], "Initialization" };
    memInfos[1] = { bufferMemory[0], "User-Allocated" };
    memInfos[2] = { devMemUsed[0], "Collective" };
    writeMemInfo(memInfos, 3);
  }
  finalizeFooter();

#ifdef MPI_SUPPORT
  MPI_Comm_free(&mpi_comm);
  MPI_Finalize();
#endif

  if (dump_file) {
    for (int p=0; p<32; p++) {
      fprintf(dump_file, "%.1f\n", dump_values[p]);
    }
    fclose(dump_file);
  }

  writeErrors();

  // 'cuda-memcheck --leak-check full' requires this
  cudaDeviceReset();

  if (errors[0] || bw[0] < check_avg_bw*(0.9))
    return testNumResults;
  else
    return testSuccess;
}
bool isFp8ValidForReductions(ncclDataType_t type) {
#if HAVE_FP8
  if ((type == ncclFloat8e4m3 || type == ncclFloat8e5m2) && minCudaArch < 900) {
    return false;
  }
#endif
  return true;
}
