
/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "stress.h"
#include <pthread.h>
#include <cstdio>
#include <getopt.h>
#include <signal.h>
#include <libgen.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "cuda.h"

thread_local int is_main_thread = 0;

// Command line parameter defaults
static int nThreads = 1;
static int nGpus = 1;
static size_t maxBytes = 0ULL;
static int datacheck = 1;
static FILE* inputfile = NULL;
static int timeout = 60;

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

#define NUM_BLOCKS 32
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
    void *data = in_place ? ((void *)((uintptr_t)args->recvBuffs[i] + args->recvInplaceOffset*rank)) : args->recvBuffs[i];
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
  if (maxDelta > DeltaMaxValue(type)*(nranks - 1)) args->errors[0]++;
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

testResult_t AllocateBuffs(void **sendbuff, void **recvbuff, void **expected, size_t nbytes, int nranks) {
  CUDACHECK(cudaMalloc(sendbuff, nbytes));
  CUDACHECK(cudaMalloc(recvbuff, nbytes));
  if (datacheck) CUDACHECK(cudaMalloc(expected, nbytes));
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
    {"help", no_argument, 0, 'h'}
  };

  while(1) {
    int c;
    c = getopt_long(argc, argv, "t:g:b:e:i:f:n:m:w:s:p:c:o:d:r:z:y:k:h:l:T:", longopts, &longindex);

    if (c == -1)
      break;

    switch(c) {
      case 't':
        nThreads = strtol(optarg, NULL, 0);
        break;
      case 'g':
        nGpus = strtol(optarg, NULL, 0);
        break;
      case 'c':
        datacheck = (int)strtol(optarg, NULL, 0);
        break;
      case 'T':
        timeout = strtol(optarg, NULL, 0);
        break;
      case 'h':
      default:
        if (c != 'h') printf("invalid option '%c'\n", c);
        printf("USAGE: %s \n\t"
            "[-t,--nthreads <num threads>] \n\t"
            "[-g,--ngpus <gpus per thread>] \n\t"
            "[-c,--check <0/1>] \n\t"
            "[-o,--op <sum/prod/min/max/all>] \n\t"
            "[-d,--datatype <nccltype/all>] \n\t"
            "[-r,--root <root>] \n\t"
            "[-T,--timeout <time in seconds>] \n\t"
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
  if (argc > optind) {
    inputfile = fopen(argv[optind], "r");
    if (inputfile == NULL) {
      printf("Could not open %s (%d : %s)\n", argv[optind], errno, strerror(errno));
      return 1;
    }
  } else {
    inputfile = stdin;
  }

  TESTCHECK(run());
  return 0;
}

static int getRank(struct threadArgs* targs, int g) {
  return (targs->proc*targs->nThreads + targs->thread)*targs->nGpus + g;
}
static int getNranks(struct threadArgs* targs) {
  return targs->nProcs*targs->nThreads*targs->nGpus;
}
static testResult_t allReduce(struct testCall* args, struct threadArgs* targs) {
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupStart());
  for (int i=0; i<targs->nGpus; i++) {
    int rank = getRank(targs, i);
    if (args->rank != -1 && args->rank != rank) continue;
    NCCLCHECK(ncclAllReduce(targs->sendBuffs[i], targs->recvBuffs[i], args->count, args->datatype, args->redop, targs->comms[i], targs->streams[i]));
    if (args->group) targs->sendBuffs[i] += args->count * wordSize(args->datatype);
    if (args->group) targs->recvBuffs[i] += args->count * wordSize(args->datatype);
  }
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupEnd());
  return testSuccess;
}
static testResult_t allGather(struct testCall* args, struct threadArgs* targs) {
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupStart());
  for (int i=0; i<targs->nGpus; i++) {
    int rank = getRank(targs, i);
    int nranks = getNranks(targs);
    if (args->rank != -1 && args->rank != rank) continue;
    NCCLCHECK(ncclAllGather(targs->sendBuffs[i], targs->recvBuffs[i], args->count, args->datatype, targs->comms[i], targs->streams[i]));
    if (args->group) targs->sendBuffs[i] += args->count * wordSize(args->datatype);
    if (args->group) targs->recvBuffs[i] += nranks * args->count * wordSize(args->datatype);
  }
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupEnd());
  return testSuccess;
}
static testResult_t reduceScatter(struct testCall* args, struct threadArgs* targs) {
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupStart());
  for (int i=0; i<targs->nGpus; i++) {
    int rank = getRank(targs, i);
    int nranks = getNranks(targs);
    if (args->rank != -1 && args->rank != rank) continue;
    NCCLCHECK(ncclReduceScatter(targs->sendBuffs[i], targs->recvBuffs[i], args->count, args->datatype, args->redop, targs->comms[i], targs->streams[i]));
    if (args->group) targs->sendBuffs[i] += nranks * args->count * wordSize(args->datatype);
    if (args->group) targs->recvBuffs[i] += args->count * wordSize(args->datatype);
  }
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupEnd());
  return testSuccess;
}
static testResult_t broadcast(struct testCall* args, struct threadArgs* targs) {
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupStart());
  for (int i=0; i<targs->nGpus; i++) {
    int rank = getRank(targs, i);
    if (args->rank != -1 && args->rank != rank) continue;
    NCCLCHECK(ncclBroadcast(targs->sendBuffs[i], targs->recvBuffs[i], args->count, args->datatype, args->root, targs->comms[i], targs->streams[i]));
    if (args->group && args->root == rank) targs->sendBuffs[i] += args->count * wordSize(args->datatype);
    if (args->group) targs->recvBuffs[i] += args->count * wordSize(args->datatype);
  }
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupEnd());
  return testSuccess;
}
static testResult_t reduce(struct testCall* args, struct threadArgs* targs) {
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupStart());
  for (int i=0; i<targs->nGpus; i++) {
    int rank = getRank(targs, i);
    if (args->rank != -1 && args->rank != rank) continue;
    NCCLCHECK(ncclReduce(targs->sendBuffs[i], targs->recvBuffs[i], args->count, args->datatype, args->redop, args->root, targs->comms[i], targs->streams[i]));
    if (args->group) targs->sendBuffs[i] += args->count * wordSize(args->datatype);
    if (args->group && args->root == rank) targs->recvBuffs[i] += args->count * wordSize(args->datatype);
  }
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupEnd());
  return testSuccess;
}
static testResult_t groupStart(struct testCall* args, struct threadArgs* targs) {
  NCCLCHECK(ncclGroupStart());
  return testSuccess;
}
static testResult_t groupEnd(struct testCall* args, struct threadArgs* targs) {
  NCCLCHECK(ncclGroupEnd());
  return testSuccess;
}
static testResult_t send(struct testCall* args, struct threadArgs* targs) {
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupStart());
  for (int i=0; i<targs->nGpus; i++) {
    int rank = getRank(targs, i);
    if (args->rank != -1 && args->rank != rank) continue;
    NCCLCHECK(ncclSend(targs->sendBuffs[i], args->count, args->datatype, args->root, targs->comms[i], targs->streams[i]));
    if (args->group) targs->sendBuffs[i] += args->count * wordSize(args->datatype);
  }
  NCCLCHECK(ncclGroupStart());
  return testSuccess;
}
static testResult_t recv(struct testCall* args, struct threadArgs* targs) {
  if (targs->nGpus > 1) NCCLCHECK(ncclGroupStart());
  for (int i=0; i<targs->nGpus; i++) {
    int rank = getRank(targs, i);
    if (args->rank != -1 && args->rank != rank) continue;
    NCCLCHECK(ncclRecv(targs->recvBuffs[i], args->count, args->datatype, args->root, targs->comms[i], targs->streams[i]));
    if (args->group) targs->recvBuffs[i] += args->count * wordSize(args->datatype);
  }
  NCCLCHECK(ncclGroupStart());
  return testSuccess;
}

testFunc_t testFuncArray[] = { allReduce, allGather, reduceScatter, broadcast, reduce, groupStart, groupEnd, send, recv };
const char *testFuncNames[] = { "ncclAllReduce", "ncclAllGather", "ncclReduceScatter", "ncclBroadcast", "ncclReduce", "ncclGroupStart", "ncclGroupEnd", "ncclSend", "ncclRecv" };
const char *testTypeNames[ncclNumTypes] = {"ncclInt8", "ncclUint8", "ncclInt32", "ncclUint32", "ncclInt64", "ncclUint64", "ncclHalf", "ncclFloat", "ncclDouble"};
const char *testOpNames[ncclNumOps] = {"ncclSum", "ncclProd", "ncclMax", "ncclMin"};

testResult_t threadRunTests(struct threadArgs* targs) {
  int nranks = getNranks(targs);
  targs->errors = 0;
  for (struct testCall* c = targs->calls; c < targs->calls+targs->nCalls; c++) {
    if (c->group == 0) {
      for (int i=0; i<targs->nGpus; i++) {
        targs->sendBuffs[i] = (char*)targs->sendBuffsBase[i];
        targs->recvBuffs[i] = (char*)targs->recvBuffsBase[i];
      }
    }
    if (c->rank >= nranks || c->root >= nranks) continue;
    TESTCHECK(testFuncArray[c->func](c, targs));
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

testResult_t testLoadCalls(FILE* input, struct testCall** callsPtr, int* nCallsPtr) {
  struct testCall* calls = NULL;
  int nCallsSize = 0;
  int nCalls = 0;
  char funcStr[128];
  char dtypeStr[128];
  char redopStr[128];
  size_t bytes = 0ULL;
  int group = 0;
  int read = 0;
  char line[1024];

  while (read != EOF) {
    if (nCalls == nCallsSize) {
      calls = (struct testCall*)realloc(calls, (nCallsSize+128)*sizeof(struct testCall));
      nCallsSize += 128;
    }
    struct testCall* call = calls+nCalls;

    // Read next line
    int offset = 0;
    while (offset < 1023) {
      read = fgetc(input);
      if (read == '\n' || read == EOF) break;
      line[offset++] = read;
    }
    line[offset] = '\0';

    // Parse line
    if (line[0] == '#') continue; // Comments
    call->group = group;
    int fields = sscanf(line, "%s %d %ld %d %s %s", funcStr, &call->rank, &call->count, &call->root, dtypeStr, redopStr);
    if (fields <= 0) continue;

    TESTCHECK(ncclStringToFunc(funcStr, &call->func, &call->name));
    if (fields >= 5) TESTCHECK(ncclStringToType(dtypeStr, &call->datatype));
    if (fields >= 6) TESTCHECK(ncclStringToOp(redopStr, &call->redop));
    if (strcmp(funcStr, "ncclGroupStart") == 0) group++;
    else if (strcmp(funcStr, "ncclGroupEnd") == 0) group--;
    else {
      bytes += call->count * wordSize(call->datatype);
      maxBytes = std::max(bytes, maxBytes);
      if (group == 0) bytes = 0;
    }
    nCalls++;
  }
  *callsPtr = calls;
  *nCallsPtr = nCalls;
  return testSuccess;
}

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
#endif
  is_main_thread = (proc == 0) ? 1 : 0;

  PRINT("# nThread %d nGpus %d validation: %d \n",
        nThreads, nGpus, datacheck);
  PRINT("#\n");

  PRINT("# Using devices\n");
#define MAX_LINE 2048
  char line[MAX_LINE];
  int len = 0;
  for (int i=0; i<nThreads*nGpus; i++) {
    char* envstr = getenv("NCCL_TESTS_DEVICE");
    int cudaDev = envstr ? atoi(envstr) : localRank*nThreads*nGpus+i;
    int rank = proc*nThreads*nGpus+i;
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
    len += snprintf(line+len, MAX_LINE-len, "#   Rank %2d Pid %6d on %10s device %2d [0x%02x] %s\n",
                    rank, getpid(), hostname, cudaDev, prop.pciBusID, prop.name);
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
#else
  PRINT("%s", line);
#endif

  ncclUniqueId ncclId;
  if (proc == 0) {
    NCCLCHECK(ncclGetUniqueId(&ncclId));
  }
#ifdef MPI_SUPPORT
  MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

  struct testCall* calls;
  int nCalls;
  TESTCHECK(testLoadCalls(inputfile, &calls, &nCalls));
  PRINT("Successfully loaded %d operations\n", nCalls);

  int gpus[nGpus*nThreads];
  cudaStream_t streams[nGpus*nThreads];
  void* sendBuffsBase[nGpus*nThreads];
  void* recvBuffsBase[nGpus*nThreads];
  char* sendBuffs[nGpus*nThreads];
  char* recvBuffs[nGpus*nThreads];
  void* expected[nGpus*nThreads];

  for (int i=0; i<nGpus*nThreads; i++) {
    char* envstr = getenv("NCCL_TESTS_DEVICE");
    gpus[i] = envstr ? atoi(envstr) : localRank*nThreads*nGpus+i;
    CUDACHECK(cudaSetDevice(gpus[i]));
    TESTCHECK(AllocateBuffs(sendBuffsBase+i, recvBuffsBase+i, expected+i, (size_t)maxBytes, nProcs*nThreads*nGpus));
    CUDACHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));
  }

  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nThreads*nGpus);
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

  int errors[nThreads];
  memset(errors, 0, sizeof(int)*nThreads);
  double* delta;
  CUDACHECK(cudaHostAlloc(&delta, sizeof(double)*nThreads*NUM_BLOCKS, cudaHostAllocPortable | cudaHostAllocMapped));

  int* sync = (int*)calloc(2, sizeof(int));
  int* barrier = (int*)calloc(2, sizeof(int));

  struct testThread threads[nThreads];
  memset(threads, 0, sizeof(struct testThread)*nThreads);

  for (int t=nThreads-1; t>=0; t--) {
    threads[t].args.localRank = localRank;
    threads[t].args.nProcs=nProcs;
    threads[t].args.proc=proc;
    threads[t].args.nThreads=nThreads;
    threads[t].args.thread=t;
    threads[t].args.nGpus=nGpus;
    threads[t].args.gpus=gpus+t*nGpus;
    threads[t].args.sendBuffsBase = sendBuffsBase+t*nGpus;
    threads[t].args.recvBuffsBase = recvBuffsBase+t*nGpus;
    threads[t].args.sendBuffs = sendBuffs+t*nGpus;
    threads[t].args.recvBuffs = recvBuffs+t*nGpus;
    threads[t].args.expected = expected+t*nGpus;
    threads[t].args.comms=comms+t*nGpus;
    threads[t].args.streams=streams+t*nGpus;

    threads[t].args.barrier = (volatile int*)barrier;
    threads[t].args.barrier_idx = 0;
    threads[t].args.sync = (volatile int*)sync;
    threads[t].args.sync_idx = 0;
    threads[t].args.deltaThreads = delta;
    threads[t].args.deltaHost = (delta + t*NUM_BLOCKS);
    threads[t].args.delta = delta;
    threads[t].args.calls = calls;
    threads[t].args.nCalls = nCalls;
    threads[t].args.errors=errors+t;
    threads[t].func = threadRunTests;
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
    }
  }

#ifdef MPI_SUPPORT
  MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  for(int i=0; i<nGpus*nThreads; ++i)
    NCCLCHECK(ncclCommDestroy(comms[i]));
  free(comms);

  // Free off CUDA allocated memory
  for (int i=0; i<nGpus*nThreads; i++) {
    CUDACHECK(cudaFree(sendBuffsBase[i]));
    CUDACHECK(cudaFree(recvBuffsBase[i]));
    if (datacheck) CUDACHECK(cudaFree(expected[i]));
  }
  CUDACHECK(cudaFreeHost(delta));

#ifdef MPI_SUPPORT
  MPI_Finalize();
#endif

  // 'cuda-memcheck --leak-check full' requires this
  cudaDeviceReset();
  if (errors[0])
    return testDataError;
  else
    return testSuccess;
}
