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
#include "cuda.h"
#include <limits.h>
#include <assert.h>

#include "../verifiable/verifiable.h"

int test_ncclVersion = 0; // init'd with ncclGetVersion()

#if NCCL_MAJOR >= 2
  ncclDataType_t test_types[ncclNumTypes] = {
    ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclHalf, ncclFloat, ncclDouble
  #if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
    , ncclBfloat16
  #endif
  };
  const char *test_typenames[ncclNumTypes] = {
    "int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double"
  #if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
    , "bfloat16"
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
static int nThreads = 1;
static int nGpus = 1;
static size_t minBytes = 32*1024*1024;
static size_t maxBytes = 32*1024*1024;
static size_t stepBytes = 1*1024*1024;
static size_t stepFactor = 1;
static int datacheck = 1;
static int warmup_iters = 20;
static int iters = 20;
static int agg_iters = 1;
static int ncclop = ncclSum;
static int nccltype = ncclFloat;
static int ncclroot = 0;
static int parallel_init = 0;
static int blocking_coll = 0;
static int streamnull = 0;
static int side_comp = 0;
static int timeout = 60;
static int cudaGraphLaunches = 0;
static int report_cputime = 0;
static int out_of_place = 1;
static int unalign = 0;
// Report average iteration time: (0=RANK0,1=AVG,2=MIN,3=MAX)
static int average = 1;
static int commblocking = NCCL_CONFIG_UNDEF_INT;
static int ft_test = 0;
static char* ft_list = NULL;
static size_t tbytes = SIZE_MAX;
static int split_share = NCCL_CONFIG_UNDEF_INT;
static int split_comm = 0;
static int commNum = 1;
static int local_register = 0;
static int per_coll_perf = 0;

static char* replay_file = NULL;

static FILE* dump_file = NULL;
static double dump_values[30]; // 8 to 4G

// Side computation constants
#define COMP_SIZE (1 << 22)
#define NUM_BLOCKS 32

static double parsesize(const char *value) {
    long long int units;
    double size;
    char size_lit;

    int count = sscanf(value, "%lf %1s", &size, &size_lit);

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
    int flag;
    /* poke MPI progress for OpenMPI */
    pthread_mutex_lock(&mpiLock);
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    pthread_mutex_unlock(&mpiLock);
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
            for (int id1 = 0; id1 < commNum; ++id1)
              for (int i = 0; i < ngpus; i++)
                NCCLCHECK(ncclCommAbort(comms[id1][i]));
            // Abort the perf test
            NCCLCHECK(ncclAsyncErr);
          }
        }
        double delta = tim.elapsed();
        if (delta > timeout && timeout > 0) {
          for (int id1 = 0; id1 < commNum; ++id1)
            for (int i = 0; i < ngpus; i++)
              NCCLCHECK(ncclCommAbort(comms[id1][i]));
          char hostname[1024];
          getHostName(hostname, 1024);
          printf("%s: Test timeout (%ds) %s:%d\n",
            hostname,
            timeout,
            __FILE__, __LINE__);
          free(done);
          return testTimeout;
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
#if defined(__CUDA_BF16_TYPES_EXIST__)
          __nv_bfloat16 bf16;
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
#if defined(__CUDA_BF16_TYPES_EXIST__)
          case ncclBfloat16: bf16 = ncclVerifiablePremulScalar<__nv_bfloat16>(rank); break;
#endif
        }
        NCCLCHECK(ncclRedOpCreatePreMulSum(&op, &u64, type, ncclScalarHostImmediate, args->comms[id][i]));
      }
#endif

      TESTCHECK(args->collTest->runColl(
        (void*)(in_place ? recvBuff + args->sendInplaceOffset[id][i] * rank : sendBuff),
        (void*)(in_place ? recvBuff + args->recvInplaceOffset[id][i] * rank : recvBuff),
        count, type, op, root, args->comms[id][i], args->streams[i]));

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
  for (int iter = 0; iter < actualIters; iter++) {
    if (agg_iters>1) NCCLCHECK(ncclGroupStart());

    if (record) TESTCHECK(recordEvents(args, actualIters, iter));

    for (int aiter = 0; aiter < agg_iters; aiter++) {
      TESTCHECK(startColl(args, type, op, root, in_place, iter*agg_iters+aiter));
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
  if (datacheck) {
      // Initialize sendbuffs, recvbuffs and expected
      TESTCHECK(args->collTest->initData(args, type, op, root, rep, in_place));

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
  }

  double timeUsec = (report_cputime ? cputimeSec : deltaSec)*1.0E6;
  char timeStr[100];
  if (timeUsec >= 10000.0) {
    sprintf(timeStr, "%7.0f", timeUsec);
  } else if (timeUsec >= 100.0) {
    sprintf(timeStr, "%7.1f", timeUsec);
  } else {
    sprintf(timeStr, "%7.2f", timeUsec);
  }
  double sideBw = ((double)compThreadCount)*COMP_SIZE*NUM_BLOCKS/(1000*timeUsec);

  if (args->reportErrors) {
     if (side_comp == 1) {
       PRINT("  %7s  %6.2f  %6.2f  %5g %6.2f", timeStr, algBw, busBw, (double)wrongElts, sideBw);
     } else {
       PRINT("  %7s  %6.2f  %6.2f  %5g", timeStr, algBw, busBw, (double)wrongElts);
     }
  } else {
     if (side_comp == 1) {
       PRINT("  %7s  %6.2f  %6.2f    N/A %6.2f", timeStr, algBw, busBw, sideBw);
     } else {
       PRINT("  %7s  %6.2f  %6.2f    N/A", timeStr, algBw, busBw);
     }
  }

  if (record) {
    args->meanTime = timeUsec;
    args->meanAlgBw = algBw;
    args->meanBusBw = busBw;
  }

  if (dump_file) {
    /* only dump first split and communicator */
    size_t nBytes = max(args->sendBytes[0][0], args->expectedBytes[0][0]);
    // Dump 8B to 4G to file.
    for (int p=0; p<30; p++) if (nBytes == (8ULL<<p)) {
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
      args->collTest->getCollByteCount(&sendCount, &recvCount, &paramCount, &sendInplaceOffset, &recvInplaceOffset, (size_t)count, (size_t)nranks);
      args->nbytes[id][i] = paramCount * wordSize(type);
      args->sendBytes[id][i] = sendCount * wordSize(type);
      args->expectedBytes[id][i] = recvCount * wordSize(type);
      args->sendInplaceOffset[id][i] = sendInplaceOffset * wordSize(type);
      args->recvInplaceOffset[id][i] = recvInplaceOffset * wordSize(type);
    }
  }
}

void printPerCollPerf(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int actualIters, int per_coll_perf) {
  double varianceTime = 0, varianceAlgBw = 0, varianceBusBw = 0;
  size_t count = args->nbytes[0][0] / wordSize(type);
  double algBw, busBw;
  char timeStr[100];
  for (int i = 0; i < args->nGpus; i++) {
    for (int j = 0; j < actualIters; j++) {
      double timeSec = args->ms[i*(actualIters)+j] / 1.0E3;
      double timeUsec = timeSec*1.0E6;
      if (timeUsec >= 10000.0) {
        sprintf(timeStr, "%7.0f", timeUsec);
      } else if (timeUsec >= 100.0) {
        sprintf(timeStr, "%7.1f", timeUsec);
      } else {
        sprintf(timeStr, "%7.2f", timeUsec);
      }
      args->collTest->getBw(count, wordSize(type), timeSec, &algBw, &busBw, args->nProcs*args->nThreads*args->nGpus);
      varianceTime += pow((args->meanTime - timeUsec), 2);
      varianceAlgBw += pow((args->meanAlgBw - algBw), 2);
      varianceBusBw += pow((args->meanBusBw - busBw), 2);

      if (per_coll_perf == 1)
      {
        PRINT("\n%35sGpu%2d Coll%3d %4s %7s  %6.2f  %6.2f  %5s\n",
          " ", args->gpus[i], j, " ", timeStr, algBw, busBw, "N/A");
      }
    }
  }

  varianceTime /= actualIters;
  varianceAlgBw /= actualIters;
  varianceBusBw /= actualIters;

  PRINT("\n%24sCoefficient of variation %5s %1.4f  %1.4f  %1.4f\n", " ", " ", sqrt(varianceTime)/args->meanTime, sqrt(varianceAlgBw)/args->meanAlgBw, sqrt(varianceBusBw)/args->meanBusBw);
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

  // Warm-up for large size
  setupArgs(args->maxbytes, type, args);
  for (int iter = 0; iter < warmup_iters; iter++) {
    TESTCHECK(startColl(args, type, op, root, 0, iter));
  }
  TESTCHECK(completeColl(args));

  // Warm-up for small size
  setupArgs(args->minbytes, type, args);
  for (int iter = 0; iter < warmup_iters; iter++) {
    TESTCHECK(startColl(args, type, op, root, 0, iter));
  }
  TESTCHECK(completeColl(args));

  // Benchmark
  for (size_t size = args->minbytes; size<=args->maxbytes; size = ((args->stepfactor > 1) ? size*args->stepfactor : size+args->stepbytes)) {
      setupArgs(size, type, args);
      int actualIters;
      TESTCHECK(getIteration(args->nbytes[0][0], &actualIters));
      char rootName[100];
      sprintf(rootName, "%6i", root);
      PRINT("%12li  %12li  %8s  %6s  %6s", max(args->sendBytes[0][0], args->expectedBytes[0][0]), args->nbytes[0][0] / wordSize(type), typeName, opName, rootName);
      if (args->replayFile != NULL || !out_of_place) {
        PRINT("                                ");  // only do in-place for trace replay
      } else {
        TESTCHECK(BenchTime(args, type, op, root, 0, actualIters, per_coll_perf));
      }
      TESTCHECK(BenchTime(args, type, op, root, 1, actualIters, 0));
      if (per_coll_perf) printPerCollPerf(args, type, op, root, actualIters, per_coll_perf);
      PRINT("  %5d", actualIters);
      PRINT("    %s\n", args->replayFile == NULL ? "" : args->collTest->name);
  }

  // Revert forced misalignment
  for (int id = 0; id < args->commNum; ++id) {
    for (int i = 0; i < args->nGpus; i++) {
      args->sendbuffs[id][i] = (char*)args->sendbuffs[id][i] - unalign * wordSize(type);
      args->recvbuffs[id][i] = (char*)args->recvbuffs[id][i] - unalign * wordSize(type);
    }
  }
  return testSuccess;
}

testResult_t threadRunTests(struct threadArgs* args) {
  // Set device to the first of our GPUs. If we don't do that, some operations
  // will be done on the current GPU (by default : 0) and if the GPUs are in
  // exclusive mode those operations will fail.
  CUDACHECK(cudaSetDevice(args->gpus[0]));
  TESTCHECK(ncclTestEngine.runTest(args, ncclroot, (ncclDataType_t)nccltype, test_typenames[nccltype], (ncclRedOp_t)ncclop, test_opnames[ncclop]));
  return testSuccess;
}

testResult_t threadInit(struct threadArgs* args) {
  int nranks = args->totalProcs * args->nThreads * args->nGpus;
  char* splitMaskEnv = getenv("NCCL_TESTS_SPLIT_MASK");
  ncclComm_t globalComms[args->nGpus];

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = commblocking;
  config.splitShare = split_share;

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nGpus; ++i) {
    int rank = args->globalProc * args->nThreads * args->nGpus + args->thread * args->nGpus + i;
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    NCCLCHECK(ncclCommInitRankConfig(globalComms + i, nranks, args->ncclId, rank, &config));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, nGpus);
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
    for (int splitCase = 0; splitCase < commNum; ++splitCase) {
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
          for (int i = 0; i < nGpus; ++i) {
            int myrank = args->globalProc * args->nThreads * args->nGpus + args->thread * args->nGpus + i;
            NCCLCHECK(ncclCommSplit(globalComms[i], 4 * (myrank + 1) <= 3 * nranks, myrank, &args->comms[splitCase][i], &config));
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
    for (int i = 0; i < args->nGpus; ++i) {
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
    for (int i = 0; i < args->nGpus; ++i) {
      NCCLCHECK(ncclCommFinalize(globalComms[i]));
    }
    NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, args->nGpus);

    for (int i = 0; i < args->nGpus; ++i)
      NCCLCHECK(ncclCommDestroy(globalComms[i]));
  }

  TESTCHECK(threadRunTests(args));

  for (int id = 0; id < args->commNum; ++id) {
    for (int i = 0; i < args->nGpus; i++) {
      NCCLCHECK(ncclCommDestroy(args->comms[id][i]));
    }
  }

  return testSuccess;
}

__global__ void compute(void* _ptr, int _size) {
  uint64_t *ptr = (uint64_t*)(_ptr);
  uint64_t size = _size / sizeof(uint64_t);
  ptr += size*blockIdx.x;
  for (uint64_t offset=threadIdx.x; offset < size; offset += blockDim.x) {
     ptr[offset] <<= 1;
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
    NCCLCHECK(ncclMemAlloc(sendbuff, nbytes));
    NCCLCHECK(ncclMemAlloc(recvbuff, nbytes));
    if (datacheck) NCCLCHECK(ncclMemAlloc(expected, recvBytes));
    CUDACHECK(cudaMemset(*sendbuff, 0, nbytes));
    CUDACHECK(cudaMemset(*recvbuff, 0, nbytes));
    if (datacheck) CUDACHECK(cudaMemset(*expected, 0, recvBytes));
    *allocBytes = nbytes;
    return testSuccess;
}

testResult_t run(); // Main function

int main(int argc, char* argv[]) {
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
      #if defined(__CUDA_BF16_TYPES_EXIST__)
        test_typenum++; // bfloat16
      #endif
    }
    if (NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0) && test_ncclVersion >= NCCL_VERSION(2,11,0)) {
      test_opnum++; // PreMulSum
    }
  #endif

  // Parse args
  double parsed;
  int longindex;
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
    {"average", required_argument, 0, 'a'},
    {"commblocking", required_argument, 0, 'B'},
    {"ft_test", required_argument, 0, 'F'},
    {"ft_list", required_argument, 0, 'L'},
    {"tbytes", required_argument, 0, 's'},
    {"split_share", required_argument, 0, 'S'},
    {"split_comm", required_argument, 0, 'P'},
    {"local_register", required_argument, 0, 'R'},
    {"per_coll_perf", required_argument, 0, 'A'},
    {"help", no_argument, 0, 'h'},
    {}
  };

  while(1) {
    int c;
    c = getopt_long(argc, argv, "t:g:b:e:i:f:n:m:w:s:p:c:o:d:r:z:y:k:h:l:T:G:C:O:u:a:B:F:L:s:S:P:R:A:", longopts, &longindex);

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
        stepBytes = strtol(optarg, NULL, 0);
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
      case 'a':
        average = (int)strtol(optarg, NULL, 0);
        break;
      case 'B':
        commblocking = (int)strtol(optarg, NULL, 0);
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
      case 'R':
        local_register = (int)strtol(optarg, NULL, 0);
        break;
      case 'A':
        per_coll_perf = (int)strtol(optarg, NULL, 0);
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
            "[-p,--parallel_init <0/1>] \n\t"
            "[-c,--check <0/1>] \n\t"
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
            "[-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] \n\t"
            "[-B,--commblocking <0/1> enable blocking communicator (default: 1)] \n\t"
            "[-F,--ft_test <0/1> enable fault tolerance test (default: 0)] \n\t"
            "[-L,--ft_list <init/allreduce/alltoall/finalize/split/all> only enable specified fault tolerance test (default: all)] \n\t"
            "[-s,--tbytes total bytes allowed to transmit (default: unlimited); tbytes would limit #iterations] \n\t"
            "[-S,--split_share <0/1> enable shared resources during communicator split (default: 0)] \n\t"
            "[-P,--split_comm <0/1/2> enable communicator split (default: 0 disable; 1 dup global comm; 2 three split patterns)] \n\t"
            "[-R,--local_register <0/1> enable local buffer registration (default: 0 disable)] \n\t"
            "[-A,--per_coll_perf <0/1/2> Report performance per-collective (default: 0 disable; 1 report per-collective performance and std deviation; 2: report only std deviation)] \n\t"
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
#ifdef MPI_SUPPORT
  int provide;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provide);
  assert(provide >= MPI_THREAD_SERIALIZED);
#endif
  TESTCHECK(run());
  return 0;
}

testResult_t run() {
  int totalProcs = 1, proc = 0, ncclProcs = 1, ncclProc = 0, color = 0;
  int localRank = 0;
  char hostname[1024];
  getHostName(hostname, 1024);

char* splitMaskEnv = NULL;
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

  splitMaskEnv = getenv("NCCL_TESTS_SPLIT_MASK");
  uint64_t mask = splitMaskEnv ? strtoul(splitMaskEnv, NULL, 16) : 0;
  MPI_Comm mpi_comm;
  color = proc & mask;
  MPI_Comm_split(MPI_COMM_WORLD, color, proc, &mpi_comm);
  MPI_Comm_size(mpi_comm, &ncclProcs);
  MPI_Comm_rank(mpi_comm, &ncclProc);
#endif
  is_main_thread = is_main_proc = (proc == 0) ? 1 : 0;

  char* envstr = getenv("NCCL_TESTS_DUMP_FILE");
  if (envstr && is_main_proc) dump_file = fopen(envstr, "w");

  PRINT("# nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: %d agg iters: %d validation: %d graph: %d\n",
        nThreads, nGpus, minBytes, maxBytes,
        (stepFactor > 1)?stepFactor:stepBytes, (stepFactor > 1)?"factor":"bytes",
        warmup_iters, iters, agg_iters, datacheck, cudaGraphLaunches);
  if (blocking_coll) PRINT("# Blocking Enabled: wait for completion and barrier after each collective \n");
  if (parallel_init) PRINT("# Parallel Init Enabled: threads call into NcclInitRank concurrently \n");
  PRINT("#\n");

  PRINT("# Using devices\n");
#define MAX_LINE 2048
  char line[MAX_LINE];
  int len = 0;
  size_t maxMem = ~0;
  envstr = getenv("NCCL_TESTS_DEVICE");
  int gpu0 = envstr ? atoi(envstr) : -1;
  for (int i=0; i<nThreads*nGpus; i++) {
    int cudaDev = (gpu0 != -1 ? gpu0 : localRank*nThreads*nGpus) + i;
    int rank = proc*nThreads*nGpus+i;
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
    len += snprintf(line+len, MAX_LINE-len, "#  Rank %2d Group %2d Pid %6d on %10s device %2d [0x%02x] %s\n",
                    rank, color, getpid(), hostname, cudaDev, prop.pciBusID, prop.name);
    maxMem = std::min(maxMem, prop.totalGlobalMem);
  }

#if MPI_SUPPORT
  char *lines = (proc == 0) ? (char *)malloc(totalProcs*MAX_LINE) : NULL;
  // Gather all output in rank order to root (0)
  MPI_Gather(line, MAX_LINE, MPI_BYTE, lines, MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (proc == 0) {
    for (int p = 0; p < totalProcs; p++)
      PRINT("%s", lines+MAX_LINE*p);
    free(lines);
  }
  MPI_Allreduce(MPI_IN_PLACE, &maxMem, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
#else
  PRINT("%s", line);
#endif

  /* Now we support 3 split pattern when split_comm is enabled:
   * (1) keep all ranks in a group but in reversed order;
   * (2) split ranks into 2 groups based odd and even rank;
   * (3) split ranks into 2 groups with 3:1 ratio.
   * If NCCL_TESTS_SPLIT_MASK is set, we only split based on split mask. */
  if (splitMaskEnv == NULL && split_comm == 2) {
    commNum = 3;
    agg_iters = 1; /* we cannot aggregate coll on multiple split communicators. */
  } else if (split_comm == 1) {
    commNum = 1;
  }
  // We need sendbuff, recvbuff, expected (when datacheck enabled), plus 2G for the rest.
  size_t memMaxBytes = ((maxMem - (2LL<<30)) / (datacheck ? 3 : 2)) / commNum;
  if (maxBytes > memMaxBytes) {
    maxBytes = memMaxBytes;
    if (proc == 0) printf("#\n# Reducing maxBytes to %ld due to memory limitation\n", maxBytes);
  }

  int gpus[nGpus*nThreads];
  cudaStream_t streams[nGpus*nThreads];
  void* sendbuffs[commNum][nGpus*nThreads];
  void* recvbuffs[commNum][nGpus*nThreads];
  void* expected[commNum][nGpus*nThreads];
  size_t sendBytes, recvBytes;

  /* only when communicators are nonblocking and ft test is enabled, we
   * perform fault tolerance tests. */
  if (ft_test && commblocking == 0) {
    TESTCHECK(faultToleranceTests(nThreads, nGpus, ncclProc, ncclProcs, localRank, ft_list));
  }

  envstr = getenv("NCCL_TESTS_DEVICE");
  gpu0 = envstr ? atoi(envstr) : -1;
  for (int i = 0; i < nGpus * nThreads; ++i) {
    gpus[i] = (gpu0 != -1 ? gpu0 : localRank * nThreads * nGpus) + i;
    CUDACHECK(cudaSetDevice(gpus[i]));
    if (streamnull) {
      streams[i] = NULL;
    } else {
      CUDACHECK(cudaStreamCreateWithFlags(streams + i, cudaStreamNonBlocking));
    }
  }

  ncclUniqueId ncclId;

  //if parallel init is not selected, use main thread to initialize NCCL
  ncclComm_t* globalComms = NULL;
  ncclComm_t comms[commNum][nThreads*nGpus];
  void* sendRegHandles[commNum][nThreads*nGpus];
  void* recvRegHandles[commNum][nThreads*nGpus];
  int nranks = totalProcs * nThreads * nGpus;
  if (proc == 0) {
      NCCLCHECK(ncclGetUniqueId(&ncclId));
    }
#ifdef MPI_SUPPORT
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
  if (!parallel_init) {
    globalComms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nThreads * nGpus);
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = commblocking;
    config.splitShare = split_share;

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGpus * nThreads; ++i) {
      CUDACHECK(cudaSetDevice(gpus[i]));
      NCCLCHECK(ncclCommInitRankConfig(globalComms + i, nranks, ncclId, proc * nThreads * nGpus + i, &config));
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
            /* 3:1 split */
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nGpus * nThreads; ++i) {
              int myrank = proc * nThreads * nGpus + i;
              NCCLCHECK(ncclCommSplit(globalComms[i], 4 * (myrank + 1) <= 3 * nranks, myrank, &comms[splitCase][i], &config));
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
      for (int i = 0; i < nGpus * nThreads; ++i) {
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
      for (int i = 0; i < nGpus * nThreads; ++i) {
        NCCLCHECK(ncclCommFinalize(globalComms[i]));
      }
      NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), globalComms, nGpus * nThreads);

      for (int i = 0; i < nGpus * nThreads; ++i)
        NCCLCHECK(ncclCommDestroy(globalComms[i]));
    }
  }

  /* allocate buffer for each split comm. */
  for (int id = 0; id < commNum; ++id) {
    for (int i = 0; i < nGpus * nThreads; i++) {
      int nranks;
      size_t allocBytes;
      NCCLCHECK(ncclCommCount(comms[id][i], &nranks));
      ncclTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes, (size_t)nranks);
      CUDACHECK(cudaSetDevice(gpus[i]));
      TESTCHECK(AllocateBuffs(sendbuffs[id] + i, sendBytes, recvbuffs[id] + i, recvBytes, expected[id] + i, (size_t)maxBytes, &allocBytes));
      if (local_register) {
        NCCLCHECK(ncclCommRegister(comms[id][i], sendbuffs[id][i], allocBytes, &sendRegHandles[id][i]));
        NCCLCHECK(ncclCommRegister(comms[id][i], recvbuffs[id][i], allocBytes, &recvRegHandles[id][i]));
      }
    }
  }

  int errors[nThreads];
  double bw[nThreads];
  double* delta;
  CUDACHECK(cudaHostAlloc(&delta, sizeof(double)*nThreads*NUM_BLOCKS, cudaHostAllocPortable | cudaHostAllocMapped));
  int bw_count[nThreads];
  for (int t=0; t<nThreads; t++) {
    bw[t] = 0.0;
    errors[t] = bw_count[t] = 0;
  }

  const char* timeStr = report_cputime ? "cputime" : "time";
  PRINT("#\n");
  PRINT("# %10s  %12s  %8s  %6s  %6s           out-of-place                       in-place          \n", "", "", "", "", "");
  PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s %6s  %7s  %6s  %6s %6s %6s\n", "size", "count", "type", "redop", "root",
      timeStr, "algbw", "busbw", "#wrong", timeStr, "algbw", "busbw", "#wrong", "#iters");
  PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s  %5s\n", "(B)", "(elements)", "", "", "",
      "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "", "");

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
    threads[t].args.sendBytes = (size_t**)malloc(sizeof(size_t*) * commNum);
    threads[t].args.expectedBytes = (size_t**)malloc(sizeof(size_t*) * commNum);
    threads[t].args.sendInplaceOffset = (size_t**)malloc(sizeof(size_t*) * commNum);
    threads[t].args.recvInplaceOffset = (size_t**)malloc(sizeof(size_t*) * commNum);
    threads[t].args.nbytes = (size_t**)malloc(sizeof(size_t*) * commNum);
    for (int id = 0; id < commNum; ++id) {
      threads[t].args.sendbuffs[id] = sendbuffs[id]+t*nGpus;
      threads[t].args.recvbuffs[id] = recvbuffs[id]+t*nGpus;
      threads[t].args.expected[id] = expected[id]+t*nGpus;
      threads[t].args.comms[id] = comms[id]+t*nGpus;
      threads[t].args.sendBytes[id] = (size_t*)malloc(sizeof(size_t) * nGpus);
      threads[t].args.expectedBytes[id] = (size_t*)malloc(sizeof(size_t) * nGpus);
      threads[t].args.sendInplaceOffset[id] = (size_t*)malloc(sizeof(size_t) * nGpus);
      threads[t].args.recvInplaceOffset[id] = (size_t*)malloc(sizeof(size_t) * nGpus);
      threads[t].args.nbytes[id] = (size_t*)malloc(sizeof(size_t) * nGpus);
    }

    threads[t].args.commNum = commNum;
    threads[t].args.ncclId = ncclId;
    threads[t].args.streams=streams+t*nGpus;

    threads[t].args.errors=errors+t;
    threads[t].args.bw=bw+t;
    threads[t].args.bw_count=bw_count+t;

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
    TESTCHECK(threads[t].ret);
    if (t) {
      errors[0] += errors[t];
      bw[0] += bw[t];
      bw_count[0] += bw_count[t];
    }
    if (side_comp) {
       compThreads[t].args.compThreadStop = 1;
       pthread_join(compThreads[t].thread, NULL);
       TESTCHECK(compThreads[t].ret);
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
    free(threads[t].args.sendBytes);
    free(threads[t].args.expectedBytes);
    free(threads[t].args.sendInplaceOffset);
    free(threads[t].args.recvInplaceOffset);
    free(threads[t].args.nbytes);
  }

#ifdef MPI_SUPPORT
  MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Free off CUDA allocated memory
  for (int id = 0; id < commNum; ++id) {
    for (int i=0; i<nGpus*nThreads; i++) {
      if (local_register) {
        NCCLCHECK(ncclCommDeregister(comms[id][i], sendRegHandles[id][i]));
        NCCLCHECK(ncclCommDeregister(comms[id][i], recvRegHandles[id][i]));
      }
      if (sendbuffs[id][i]) NCCLCHECK(ncclMemFree(sendbuffs[id][i]));
      if (recvbuffs[id][i]) NCCLCHECK(ncclMemFree(recvbuffs[id][i]));
      if (datacheck) NCCLCHECK(ncclMemFree(expected[id][i]));
    }
  }

  if (!parallel_init) {
    for (int id = 0; id < commNum; ++id) {
      for(int i=0; i<nGpus*nThreads; ++i)
        NCCLCHECK(ncclCommDestroy(comms[id][i]));
    }
    free(globalComms);
  }

  CUDACHECK(cudaFreeHost(delta));

  envstr = getenv("NCCL_TESTS_MIN_BW");
  double check_avg_bw = envstr ? atof(envstr) : -1;
  bw[0] /= bw_count[0];

  PRINT("# Out of bounds values : %d %s\n", errors[0], errors[0] ? "FAILED" : "OK");
  PRINT("# Avg bus bandwidth    : %g %s\n", bw[0], check_avg_bw == -1 ? "" : (bw[0] < check_avg_bw*(0.9) ? "FAILED" : "OK"));
  PRINT("#\n");
#ifdef MPI_SUPPORT
  MPI_Finalize();
#endif

  if (dump_file) {
    for (int p=0; p<30; p++) {
      fprintf(dump_file, "%.1f\n", dump_values[p]);
    }
    fclose(dump_file);
  }

  PRINT("%s\n", ncclGetLastError(NULL));

  // 'cuda-memcheck --leak-check full' requires this
  cudaDeviceReset();

  if (errors[0] || bw[0] < check_avg_bw*(0.9))
    exit(EXIT_FAILURE);
  else
    exit(EXIT_SUCCESS);
}
