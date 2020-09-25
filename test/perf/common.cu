/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include <pthread.h>
#include <cstdio>
#include <getopt.h>
#include <signal.h>
#include <libgen.h>
#include "cuda.h"

#if NCCL_MAJOR >= 2
ncclDataType_t test_types[ncclNumTypes] = {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclHalf, ncclFloat, ncclDouble};
const char *test_typenames[ncclNumTypes] = {"int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double"};
#else
ncclDataType_t test_types[ncclNumTypes] = {ncclChar, ncclInt, ncclHalf, ncclFloat, ncclDouble, ncclInt64, ncclUint64};
const char *test_typenames[ncclNumTypes] = {"char", "int", "half", "float", "double", "int64", "uint64"};
#endif
ncclRedOp_t test_ops[ncclNumOps] = {ncclSum, ncclProd, ncclMax, ncclMin};
const char *test_opnames[ncclNumOps] = {"sum", "prod", "max", "min"};

thread_local int is_main_thread = 0;

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
static int cudaGraphMode = 0;

static char* replay_file = NULL;

static FILE* dump_file = NULL;
static double dump_values[30]; // 8 to 4G

// Side computation constants
#define COMP_SIZE (1 << 22)
#define NUM_BLOCKS 32

double parsesize(char *value) {
    long long int units;
    double size;

    if (strchr(value, 'G') != NULL) {
        units=1024*1024*1024;
    } else if (strchr(value, 'M') != NULL) {
        units=1024*1024;
    } else if (strchr(value, 'K') != NULL) {
        units=1024;
    } else {
        units=1;
    }

    size = atof(value)*units;
    return size;
}

double DeltaMaxValue(ncclDataType_t type) {
  switch(type) {
    case ncclHalf: return 1e-2;
    case ncclFloat: return 1e-5;
    case ncclDouble: return 1e-12;
    case ncclInt:
#if NCCL_MAJOR >= 2
    case ncclUint8:
    //case ncclInt32:
    case ncclUint32:
#endif
    case ncclInt64:
    case ncclUint64: return 1e-200;
  }
  return 1e-200;
}

template<typename T> __device__
double absDiff(T a, T b) {
  return fabs((double)(b - a));
}

template<> __device__
double absDiff<half>(half a, half b) {
  float x = __half2float(a);
  float y = __half2float(b);
  return fabs((double)(y-x));
}

template<typename T> __device__
float toFloat(T a) {
  return (float)a;
}
template<> __device__
float toFloat(half a) {
  return __half2float(a);
}

template<typename T, int BSIZE> __global__
void deltaKern(void* A_, void* B_, size_t count, double* max) {
  const T* A = (const T*)A_;
  const T* B = (const T*)B_;
  __shared__ double temp[BSIZE];
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  double locmax = 0.0;
  for(size_t i=tid; i<count; i+=blockDim.x*gridDim.x) {

    double delta = absDiff(A[i], B[i]);
    if( delta > locmax ) {
      locmax = delta;
#ifdef DEBUG_PRINT
      if (delta > .1) printf("Error at %d/%ld(%p) : %f != %f\n", i, count, B+i, toFloat(A[i]), toFloat(B[i]));
#endif
    }
  }

  tid = threadIdx.x;
  temp[tid] = locmax;
  for(int stride = BSIZE/2; stride > 1; stride>>=1) {
    __syncthreads();
    if( tid < stride )
      temp[tid] = temp[tid] > temp[tid+stride] ? temp[tid] : temp[tid+stride];
  }
  __syncthreads();
  if( threadIdx.x == 0)
    max[blockIdx.x] = temp[0] > temp[1] ? temp[0] : temp[1];
}

testResult_t CheckDelta(void* results, void* expected, size_t count, ncclDataType_t type, double* devmax) {
  switch (type) {
    case ncclHalf:
      deltaKern<half, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;
    case ncclFloat:
      deltaKern<float, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;
    case ncclDouble:
      deltaKern<double, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;

    case ncclChar:
#if NCCL_MAJOR >= 2
    case ncclUint8:
#endif
      deltaKern<uint8_t, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;
    case ncclInt:
#if NCCL_MAJOR >= 2
    case ncclUint32:
#endif
      deltaKern<uint32_t, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;
    case ncclInt64:
    case ncclUint64:
      deltaKern<uint64_t, 512><<<NUM_BLOCKS, 512>>>(results, expected, count, devmax); break;
  }
  CUDACHECK(cudaDeviceSynchronize());
  for (int i=1; i<NUM_BLOCKS; i++) devmax[0] = std::max(devmax[0], devmax[i]);
  return testSuccess;
}

// For integer values, we use values between 0 and 255
template<typename T>
__device__ T testValue(const size_t offset, const int rep, const int rank) {
  uint8_t v = (rep+rank+offset) % 256;
  return (T)v;
}

// For floating point datatype, we use values between 0 and 1 otherwise the
// Product operation will produce NaNs.
template<>
__device__ double testValue<double>(const size_t offset, const int rep, const int rank) {
  return 1.0/(1.0+(double)testValue<int>(offset, rep, rank));
}
template<>
__device__ float testValue<float>(const size_t offset, const int rep, const int rank) {
  return 1.0/(1.0+(float)testValue<int>(offset, rep, rank));
}
template<>
__device__ half testValue<half>(const size_t offset, const int rep, const int rank) {
  return __float2half(testValue<float>(offset, rep, rank));
}

// Operations
template<typename T>
__device__ T ncclOpSum(T a, T b) { return a+b; }
template<typename T>
__device__ T ncclOpProd(T a, T b) { return a*b; }
template<typename T>
__device__ T ncclOpMax(T a, T b) { return a>b ? a : b; }
template<typename T>
__device__ T ncclOpMin(T a, T b) { return a<b ? a : b; }

// Definitions for half
template<>
__device__ half ncclOpSum(half a, half b) { return __float2half(__half2float(a)+__half2float(b)); }
template<>
__device__ half ncclOpProd(half a, half b) { return __float2half(__half2float(a)*__half2float(b)); }
template<>
__device__ half ncclOpMax(half a, half b) { return __half2float(a)>__half2float(b) ? a : b; }
template<>
__device__ half ncclOpMin(half a, half b) { return __half2float(a)<__half2float(b) ? a : b; }

template<typename T, T (*Op)(T, T)>
__global__ void InitDataReduceKernel(T* data, const size_t N, const size_t offset, const int rep, const int nranks) {
  for (size_t o=blockIdx.x*blockDim.x+threadIdx.x; o<N; o+=gridDim.x*blockDim.x) {
    T val = testValue<T>(o+offset, rep, 0);
    for (int i=1; i<nranks; i++) {
      val = Op(val, testValue<T>(o+offset, rep, i));
    }
    data[o] = val;
  }
}

#define KERN(type, op) (void*)InitDataReduceKernel<type, op<type>>
#define OPS(type) KERN(type, ncclOpSum), KERN(type, ncclOpProd), KERN(type, ncclOpMax), KERN(type, ncclOpMin)

static void* const redInitDataKerns[ncclNumOps*ncclNumTypes] = {
  OPS(int8_t), OPS(uint8_t), OPS(int32_t), OPS(uint32_t), OPS(int64_t), OPS(uint64_t), OPS(half), OPS(float), OPS(double)
};

testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op, const int rep, const int nranks) {
  dim3 grid = { 32, 1, 1 };
  dim3 block = { 256, 1, 1 };
  void* args[5] = { (void*)&data, (void*)&count, (void*)&offset, (void*)&rep, (void*)&nranks };
  CUDACHECK(cudaLaunchKernel(redInitDataKerns[type*ncclNumOps+op], grid, block, args, 0, cudaStreamDefault));
  return testSuccess;
}

template<typename T>
__global__ void InitDataKernel(T* data, const size_t N, const int rep, const int rank) {
  for (size_t o=blockIdx.x*blockDim.x+threadIdx.x; o<N; o+=gridDim.x*blockDim.x)
    data[o] = testValue<T>(o, rep, rank);
}

static void* const initDataKerns[ncclNumTypes] = {
  (void*)InitDataKernel<  int8_t>,
  (void*)InitDataKernel< uint8_t>,
  (void*)InitDataKernel< int32_t>,
  (void*)InitDataKernel<uint32_t>,
  (void*)InitDataKernel< int64_t>,
  (void*)InitDataKernel<uint64_t>,
  (void*)InitDataKernel<    half>,
  (void*)InitDataKernel<   float>,
  (void*)InitDataKernel<  double>
};

template<typename T>
testResult_t InitDataType(void* dest, const size_t N, const int rep, const int rank) {
  T* ptr = (T*)dest;
  InitDataKernel<<<16, 512>>>(ptr, N, rep, rank);
  return testSuccess;
}

testResult_t InitData(void* data, const size_t count, ncclDataType_t type, const int rep, const int rank) {
  dim3 grid = { 32, 1, 1 };
  dim3 block = { 256, 1, 1 };
  void* args[4] = { (void*)&data, (void*)&count, (void*)&rep, (void*)&rank };
  CUDACHECK(cudaLaunchKernel(initDataKerns[type], grid, block, args, 0, cudaStreamDefault));
  return testSuccess;
}

void Barrier(struct threadArgs* args)
{
  while (args->barrier[args->barrier_idx] != args->thread) pthread_yield();

  args->barrier[args->barrier_idx] = args->thread + 1;

  if (args->thread+1 == args->nThreads) {
#ifdef MPI_SUPPORT
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    args->barrier[args->barrier_idx] = 0;
  } else {
    while (args->barrier[args->barrier_idx]) pthread_yield();
  }

  args->barrier_idx=!args->barrier_idx;
}

testResult_t CheckData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, double *delta) {
  size_t count = args->expectedBytes/wordSize(type);
  double maxDelta = 0.0;
  for (int i=0; i<args->nGpus; i++) {
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    void *data = in_place ? ((void *)((uintptr_t)args->recvbuffs[i] + args->recvInplaceOffset*rank)) : args->recvbuffs[i];
    TESTCHECK(CheckDelta(data , args->expected[i], count, type, args->delta));
    maxDelta = std::max(*(args->deltaHost), maxDelta);

#ifdef DEBUG_PRINT
    if (rank == 0) {
       int *expectedHost = (int *)malloc(args->expectedBytes);
       int *dataHost = (int *)malloc(args->expectedBytes);

       cudaMemcpy(expectedHost, args->expected[0], args->expectedBytes, cudaMemcpyDeviceToHost);
       printf("\n Expected: ");
       for(int j=0; j<args->expectedBytes/sizeof(int); j++) {
         printf("%d:%d ", j, expectedHost[j]);
       }
       printf("\n");

       cudaMemcpy(dataHost, data, args->expectedBytes, cudaMemcpyDeviceToHost);
       printf("\n Actual: ");
       for (int j=0; j<args->expectedBytes/sizeof(int); j++) {
         printf("%d:%d ", j, dataHost[j]);
       }
       printf("\n");
    }
#endif
  }
  double nranks = args->nProcs*args->nThreads*args->nGpus;
  if (args->reportErrors && maxDelta > DeltaMaxValue(type)*(nranks - 1)) args->errors[0]++;
  *delta = maxDelta;
  return testSuccess;
}

testResult_t testStreamSynchronize(int ngpus, cudaStream_t* streams, ncclComm_t* comms) {
  cudaError_t cudaErr;
  int remaining = ngpus;
  int* done = (int*)malloc(sizeof(int)*ngpus);
  memset(done, 0, sizeof(int)*ngpus);
  auto start = std::chrono::high_resolution_clock::now();

  while (remaining) {
   int idle = 1;
   for (int i=0; i<ngpus; i++) {
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
     int version;
     NCCLCHECK(ncclGetVersion(&version));
     if (version >= NCCL_VERSION(2,4,0) && comms) {
       ncclResult_t ncclAsyncErr;
       NCCLCHECK(ncclCommGetAsyncError(comms[i], &ncclAsyncErr));
       if (ncclAsyncErr != ncclSuccess) {
         // An asynchronous error happened. Stop the operation and destroy
         // the communicator
         for (int i=0; i<ngpus; i++)
           NCCLCHECK(ncclCommAbort(comms[i]));
         // Abort the perf test
         NCCLCHECK(ncclAsyncErr);
       }
     }
     auto delta = std::chrono::high_resolution_clock::now() - start;
     if (std::chrono::duration_cast<std::chrono::seconds>(delta).count() > timeout) {
       for (int i=0; i<ngpus; i++)
         NCCLCHECK(ncclCommAbort(comms[i]));
       char hostname[1024];
       getHostName(hostname, 1024);
       printf("%s: Test timeout (%ds) %s:%d\n",
           hostname,
           timeout,
           __FILE__,__LINE__);
       free(done);
       return testTimeout;
     }
#endif
   }

   // We might want to let other threads (including NCCL threads) use the CPU.
   if (idle) pthread_yield();
  }
  free(done);
  return testSuccess;
}

testResult_t startColl(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int iter) {
  size_t count = args->nbytes / wordSize(type);

  // Try to change offset for each iteration so that we avoid cache effects and catch race conditions in ptrExchange
  size_t totalnbytes = max(args->sendBytes, args->expectedBytes);
  size_t steps = totalnbytes ? args->maxbytes / totalnbytes : 1;
  size_t shift = totalnbytes * (iter % steps);

  if (args->nGpus > 1) NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < args->nGpus; i++) {
#ifndef NCCL_MAJOR
    CUDACHECK(cudaSetDevice(args->gpus[i]));
#endif
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    char* recvBuff = ((char*)args->recvbuffs[i]) + shift;
    char* sendBuff = ((char*)args->sendbuffs[i]) + shift;
    TESTCHECK(args->collTest->runColl(
          (void*)(in_place ? recvBuff + args->sendInplaceOffset*rank : sendBuff),
          (void*)(in_place ? recvBuff + args->recvInplaceOffset*rank : recvBuff),
        count, type, op, root, args->comms[i], args->streams[i]));
  }
  if (args->nGpus > 1) NCCLCHECK(ncclGroupEnd());

  if (blocking_coll) {
    // Complete op before returning
    TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
  }
  if (blocking_coll) Barrier(args);
  return testSuccess;
}

testResult_t completeColl(struct threadArgs* args) {
  if (blocking_coll) return testSuccess;

  TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms));
  return testSuccess;
}

testResult_t BenchTime(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place) {
  size_t count = args->nbytes / wordSize(type);
  if (datacheck) {
    // Initialize sendbuffs, recvbuffs and expected
    TESTCHECK(args->collTest->initData(args, type, op, root, 99, in_place));
  }

  // Sync
  TESTCHECK(startColl(args, type, op, root, in_place, 0));
  TESTCHECK(completeColl(args));

  Barrier(args);
  args->compThreadCountLast = *(args->compThreadCount);

  cudaGraph_t graph = NULL;
  cudaGraphExec_t graphExec;
  cudaStream_t s = args->streams[0];  //FIXME: currently only works for 1 GPU per thread mode
  if (cudaGraphMode == 1) {
    // Begin cuda graph capture
    CUDACHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  }

  // Performance Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iters; iter++) {
    if (agg_iters>1) NCCLCHECK(ncclGroupStart());
    for (int aiter = 0; aiter < agg_iters; aiter++) {
      TESTCHECK(startColl(args, type, op, root, in_place, iter*agg_iters+aiter));
    }
    if (agg_iters>1) NCCLCHECK(ncclGroupEnd());
  }

  if (cudaGraphMode == 1) {
    // End cuda graph capture
    CUDACHECK(cudaStreamEndCapture(s, &graph));

    // Instantiate cuda graph
    CUDACHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Resync CPU, restart timing, launch cuda graph
    Barrier(args);
    start = std::chrono::high_resolution_clock::now();
    CUDACHECK(cudaGraphLaunch(graphExec, s));
  }

  TESTCHECK(completeColl(args));

  int compThreadCount = (*(args->compThreadCount)) - args->compThreadCountLast;
  auto delta = std::chrono::high_resolution_clock::now() - start;
  double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
  deltaSec = deltaSec/(iters*agg_iters);

  if (cudaGraphMode == 1) {
    //destroy cuda graph
    CUDACHECK(cudaGraphExecDestroy(graphExec));
    CUDACHECK(cudaGraphDestroy(graph));
  }

  double algBw, busBw;
  args->collTest->getBw(count, wordSize(type), deltaSec, &algBw, &busBw, args->nProcs*args->nThreads*args->nGpus);

  Barrier(args);

  double maxDelta = 0;
  static __thread int rep = 0;
  rep++;
  if (datacheck) {
      // Initialize sendbuffs, recvbuffs and expected
      TESTCHECK(args->collTest->initData(args, type, op, root, rep, in_place));

      //test validation in single itertion, should ideally be included into the multi-iteration run
      TESTCHECK(startColl(args, type, op, root, in_place, 0));
      TESTCHECK(completeColl(args));

      TESTCHECK(CheckData(args, type, op, root, in_place, &maxDelta));

      //aggregate delta from all threads and procs
      Barrier(args);
      if (args->thread == 0) {
        for (int i=1; i<args->nThreads; i++) {
          maxDelta += args->deltaThreads[i];
        }
#ifdef MPI_SUPPORT
        MPI_Allreduce(MPI_IN_PLACE, &maxDelta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
      }
      Barrier(args);
  }

  double timeUsec = deltaSec*1.0E6;
  char timeStr[10];
  if (timeUsec >= 10000.0) {
    sprintf(timeStr, "%7.0f", timeUsec);
  } else if (timeUsec >= 100.0) {
    sprintf(timeStr, "%7.1f", timeUsec);
  } else {
    sprintf(timeStr, "%7.2f", timeUsec);
  }
  double sideBw = ((double)compThreadCount)*COMP_SIZE*NUM_BLOCKS/(1000*timeUsec);
  if (datacheck) {
     if (side_comp == 1) {
       PRINT("  %7s  %6.2f  %6.2f  %5.0le %6.2f", timeStr, algBw, busBw, maxDelta, sideBw);
     } else {
       PRINT("  %7s  %6.2f  %6.2f  %5.0le", timeStr, algBw, busBw, maxDelta);
     }
  } else {
     if (side_comp == 1) {
       PRINT("  %7s  %6.2f  %6.2f    N/A %6.2f", timeStr, algBw, busBw, sideBw);
     } else {
       PRINT("  %7s  %6.2f  %6.2f    N/A", timeStr, algBw, busBw);
     }
  }
  if (dump_file) {
    size_t nBytes = max(args->sendBytes, args->expectedBytes);
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
  int nranks = args->nProcs*args->nGpus*args->nThreads;
  size_t count, sendCount, recvCount, paramCount, sendInplaceOffset, recvInplaceOffset;

  count = size / wordSize(type);
  args->collTest->getCollByteCount(&sendCount, &recvCount, &paramCount, &sendInplaceOffset, &recvInplaceOffset, (size_t)count, (size_t)nranks);

  args->nbytes = paramCount * wordSize(type);
  args->sendBytes = sendCount * wordSize(type);
  args->expectedBytes = recvCount * wordSize(type);
  args->sendInplaceOffset = sendInplaceOffset * wordSize(type);
  args->recvInplaceOffset = recvInplaceOffset * wordSize(type);
}

testResult_t TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName, int root) {
  // Sync to avoid first-call timeout
  Barrier(args);

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
      char rootName[10];
      if (root == -1)
        sprintf(rootName, "%6s", "");
      else
        sprintf(rootName, "%6i", root);
      PRINT("%12li  %12li  %6s  %6s  %6s", max(args->sendBytes, args->expectedBytes), args->nbytes / wordSize(type), typeName, opName, rootName);
      if (args->replayFile != NULL) {
        PRINT("                                ");  // only do in-place for trace replay
      } else {
        TESTCHECK(BenchTime(args, type, op, root, 0));
      }
      TESTCHECK(BenchTime(args, type, op, root, 1));
      PRINT("    %s\n", args->replayFile == NULL ? "" : args->collTest->name);
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
  char hostname[1024];
  getHostName(hostname, 1024);
  int nranks =  args->nProcs*args->nThreads*args->nGpus;

  //set main thread again
  is_main_thread = (args->proc == 0 && args->thread == 0) ? 1 : 0;

  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<args->nGpus; i++) {
    int rank = args->proc*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    NCCLCHECK(ncclCommInitRank(args->comms+i, nranks, args->ncclId, rank));
  }
  NCCLCHECK(ncclGroupEnd());

  TESTCHECK(threadRunTests(args));

  for (int i=0; i<args->nGpus; i++) {
    NCCLCHECK(ncclCommDestroy(args->comms[i]));
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
  for (int i=0; i<args->nGpus; i++) {
    char* str = getenv("NCCL_TESTS_DEVICE");
    gpuids[i] = (str ? atoi(str) : args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus) + i;
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

testResult_t AllocateBuffs(void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes, void **expected, size_t nbytes, int nranks) {
    CUDACHECK(cudaMalloc(sendbuff, nbytes));
    CUDACHECK(cudaMalloc(recvbuff, nbytes));
    if (datacheck) CUDACHECK(cudaMalloc(expected, recvBytes));
    return testSuccess;
}

testResult_t run(); // Main function

int main(int argc, char* argv[]) {
  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

  // Parse args
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
    {"help", no_argument, 0, 'h'}
  };

  while(1) {
    int c;
    c = getopt_long(argc, argv, "t:g:b:e:i:f:n:m:w:s:p:c:o:d:r:z:y:k:h:l:T:G:", longopts, &longindex);

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
        minBytes = (size_t)parsesize(optarg);
        break;
      case 'e':
        maxBytes = (size_t)parsesize(optarg);
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
#if NCCL_MAJOR >= 2 && NCCL_MINOR >= 2
        agg_iters = (int)strtol(optarg, NULL, 0);
#else
        printf("Option -m not supported before NCCL 2.2. Ignoring\n");
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
        cudaGraphMode = strtol(optarg, NULL, 0);
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
            "[-o,--op <sum/prod/min/max/all>] \n\t"
            "[-d,--datatype <nccltype/all>] \n\t"
            "[-r,--root <root>] \n\t"
            "[-z,--blocking <0/1>] \n\t"
            "[-y,--stream_null <0/1>] \n\t"
            "[-k,--side_comp <0/1>] \n\t"
            "[-l,--replay <path to replay file>] \n\t"
            "[-T,--timeout <time in seconds>] \n\t"
            "[-G,--cudagraph <0/1>] \n\t"
	    "[-h,--help]\n",
            basename(argv[0]));
        return 0;
    }
  }
#ifdef MPI_SUPPORT
#ifdef MPI_COLLNET_SUPPORT
  int threadProvided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadProvided);
  printf("Thread provided = %d\n", threadProvided);
#else
  MPI_Init(&argc, &argv);
#endif
#endif
  TESTCHECK(run());
  return 0;
}

#ifdef MPI_COLLNET_SUPPORT
extern "C"
void ncclCollNetMpiHook(MPI_Comm comm);
#endif

testResult_t run() {
  int nProcs = 1, proc = 0;
  int localRank = 0;
  char hostname[1024];
  getHostName(hostname, 1024);

#ifdef MPI_SUPPORT
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  uint64_t hostHashs[nProcs];
  hostHashs[proc] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
  for (int p=0; p<nProcs; p++) {
    if (p == proc) break;
    if (hostHashs[p] == hostHashs[proc]) localRank++;
  }
#ifdef MPI_COLLNET_SUPPORT
  MPI_Comm master_comm;
  int color = localRank;
  MPI_Comm_split(MPI_COMM_WORLD, color, proc, &master_comm);
  ncclCollNetMpiHook(master_comm);
#endif
#endif
  is_main_thread = (proc == 0) ? 1 : 0;

  char* envstr = getenv("NCCL_TESTS_DUMP_FILE");
  if (envstr && is_main_thread) dump_file = fopen(envstr, "w");

  PRINT("# nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: %d agg iters: %d validation: %d \n",
        nThreads, nGpus, minBytes, maxBytes,
        (stepFactor > 1)?stepFactor:stepBytes, (stepFactor > 1)?"factor":"bytes",
        warmup_iters, iters, agg_iters, datacheck);
  if (blocking_coll) PRINT("# Blocking Enabled: wait for completion and barrier after each collective \n");
  if (parallel_init) PRINT("# Parallel Init Enabled: threads call into NcclInitRank concurrently \n");
  PRINT("#\n");

  PRINT("# Using devices\n");
#define MAX_LINE 2048
  char line[MAX_LINE];
  int len = 0;
  size_t maxMem = ~0;
  for (int i=0; i<nThreads*nGpus; i++) {
    envstr = getenv("NCCL_TESTS_DEVICE");
    int cudaDev = envstr ? atoi(envstr) : localRank*nThreads*nGpus+i;
    int rank = proc*nThreads*nGpus+i;
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
    len += snprintf(line+len, MAX_LINE-len, "#   Rank %2d Pid %6d on %10s device %2d [0x%02x] %s\n",
                    rank, getpid(), hostname, cudaDev, prop.pciBusID, prop.name);
    maxMem = std::min(maxMem, prop.totalGlobalMem);
  }

#if MPI_SUPPORT
  char *lines = (proc == 0) ? (char *)malloc(nProcs*MAX_LINE) : NULL;
  // Gather all output in rank order to root (0)
  MPI_Gather(line, MAX_LINE, MPI_BYTE, lines, MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (proc == 0) {
    for (int p = 0; p < nProcs; p++)
      PRINT("%s", lines+MAX_LINE*p);
    free(lines);
  }
  MPI_Allreduce(MPI_IN_PLACE, &maxMem, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
#else
  PRINT("%s", line);
#endif

  // We need sendbuff, recvbuff, expected (when datacheck enabled), plus 1G for the rest.
  size_t memMaxBytes = (maxMem - (1<<30)) / (datacheck ? 3 : 2);
  if (maxBytes > memMaxBytes) {
    maxBytes = memMaxBytes;
    if (proc == 0) printf("#\n# Reducing maxBytes to %ld due to memory limitation\n", maxBytes);
  }

  ncclUniqueId ncclId;
  if (proc == 0) {
    NCCLCHECK(ncclGetUniqueId(&ncclId));
  }
#ifdef MPI_SUPPORT
  MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
  int gpus[nGpus*nThreads];
  cudaStream_t streams[nGpus*nThreads];
  void* sendbuffs[nGpus*nThreads];
  void* recvbuffs[nGpus*nThreads];
  void* expected[nGpus*nThreads];
  size_t sendBytes, recvBytes;

  ncclTestEngine.getBuffSize(&sendBytes, &recvBytes, (size_t)maxBytes, (size_t)nProcs*nGpus*nThreads);

  for (int i=0; i<nGpus*nThreads; i++) {
    envstr = getenv("NCCL_TESTS_DEVICE");
    gpus[i] = envstr ? atoi(envstr) : localRank*nThreads*nGpus+i;
    CUDACHECK(cudaSetDevice(gpus[i]));
    TESTCHECK(AllocateBuffs(sendbuffs+i, sendBytes, recvbuffs+i, recvBytes, expected+i, (size_t)maxBytes, nProcs*nThreads*nGpus));
    if (streamnull)
      streams[i] = NULL;
    else
      CUDACHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));
  }

  //if parallel init is not selected, use main thread to initialize NCCL
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nThreads*nGpus);
  if (!parallel_init) {
     if (nProcs == 1) {
       NCCLCHECK(ncclCommInitAll(comms, nGpus*nThreads, gpus));
     } else {
       NCCLCHECK(ncclGroupStart());
       for (int i=0; i<nGpus*nThreads; i++) {
         CUDACHECK(cudaSetDevice(gpus[i]));
         NCCLCHECK(ncclCommInitRank(comms+i, nProcs*nThreads*nGpus, ncclId, proc*nThreads*nGpus+i));
       }
       NCCLCHECK(ncclGroupEnd());
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

  PRINT("#\n");
  PRINT("# %10s  %12s  %6s  %6s  %6s           out-of-place                       in-place          \n", "", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %6s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "size", "count", "type", "redop", "root",
      "time", "algbw", "busbw", "error", "time", "algbw", "busbw", "error");
  PRINT("# %10s  %12s  %6s  %6s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "", "", "",
      "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "");

  int* sync = (int*)calloc(2, sizeof(int));
  int* barrier = (int*)calloc(2, sizeof(int));

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

    threads[t].args.nProcs=nProcs;
    threads[t].args.proc=proc;
    threads[t].args.nThreads=nThreads;
    threads[t].args.thread=t;
    threads[t].args.nGpus=nGpus;
    threads[t].args.gpus=gpus+t*nGpus;
    threads[t].args.sendbuffs = sendbuffs+t*nGpus;
    threads[t].args.recvbuffs = recvbuffs+t*nGpus;
    threads[t].args.expected = expected+t*nGpus;
    threads[t].args.ncclId = ncclId;
    threads[t].args.comms=comms+t*nGpus;
    threads[t].args.streams=streams+t*nGpus;

    threads[t].args.barrier = (volatile int*)barrier;
    threads[t].args.barrier_idx = 0;
    threads[t].args.sync = (volatile int*)sync;
    threads[t].args.sync_idx = 0;
    threads[t].args.deltaThreads = delta;
    threads[t].args.deltaHost = (delta + t*NUM_BLOCKS);
    threads[t].args.delta = delta;
    threads[t].args.errors=errors+t;
    threads[t].args.bw=bw+t;
    threads[t].args.bw_count=bw_count+t;

    threads[t].args.reportErrors = 1;

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
  }

#ifdef MPI_SUPPORT
  MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (!parallel_init) {
    for(int i=0; i<nGpus*nThreads; ++i)
      NCCLCHECK(ncclCommDestroy(comms[i]));
    free(comms);
  }

  // Free off CUDA allocated memory
  for (int i=0; i<nGpus*nThreads; i++) {
    CUDACHECK(cudaFree(sendbuffs[i]));
    CUDACHECK(cudaFree(recvbuffs[i]));
    if (datacheck) CUDACHECK(cudaFree(expected[i]));
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

  // 'cuda-memcheck --leak-check full' requires this
  cudaDeviceReset();

  if (errors[0] || bw[0] < check_avg_bw*(0.9))
    exit(EXIT_FAILURE);
  else
    exit(EXIT_SUCCESS);
}
