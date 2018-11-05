/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include <pthread.h>
#include <cstdio>
#include <getopt.h>
#include <signal.h>
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

static int datacheck = 1;
static int warmup_iters = 20;
static int iters = 20;
static int agg_iters = 1;
static int ncclop = ncclSum;
static int nccltype = ncclFloat;
static int ncclroot = 0;
static int swap_args = 0;
static int parallel_init = 0;
static int blocking_coll = 0;
static int streamnull = 0;
static int side_comp = 0;
static char* replay_file = NULL;
static void **expected = NULL;
static void **expectedHost = NULL;

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
  int tid = threadIdx.x;
  double locmax = 0.0;
  for(int i=tid; i<count; i+=blockDim.x) {

    double delta = absDiff(A[i], B[i]);
    if( delta > locmax ) {
      locmax = delta;
#ifdef DEBUG_PRINT
      if (delta > .1) printf("Error at %d/%d : %f != %f\n", i, count, toFloat(A[i]), toFloat(B[i]));
#endif
    }
  }

  temp[tid] = locmax;
  for(int stride = BSIZE/2; stride > 1; stride>>=1) {
    __syncthreads();
    if( tid < stride )
      temp[tid] = temp[tid] > temp[tid+stride] ? temp[tid] : temp[tid+stride];
  }
  __syncthreads();
  if( threadIdx.x == 0)
    *max = temp[0] > temp[1] ? temp[0] : temp[1];
}


void CheckDelta(void* expected, void* results, size_t count, ncclDataType_t type, double* devmax) {
  switch (type) {
    case ncclHalf:
      deltaKern<half, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclFloat:
      deltaKern<float, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclDouble:
      deltaKern<double, 512><<<1, 512>>>(results, expected, count, devmax); break;

    case ncclChar:
#if NCCL_MAJOR >= 2
    case ncclUint8:
#endif
      deltaKern<uint8_t, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclInt:
#if NCCL_MAJOR >= 2
    case ncclUint32:
#endif
      deltaKern<uint32_t, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclInt64:
    case ncclUint64:
      deltaKern<uint64_t, 512><<<1, 512>>>(results, expected, count, devmax); break;
  }
}

#define CURAND_CHK(cmd)                                                         \
    do {                                                                        \
      curandStatus_t error = (cmd);                                             \
      if (error != CURAND_STATUS_SUCCESS) {                                     \
        printf("CuRAND error %i at %s:%i\n", error, __FILE__ , __LINE__);       \
        exit(EXIT_FAILURE);                                                     \
      }                                                                         \
    } while (false)


template<typename T>
void GenerateRandom(curandGenerator_t generator, T * const dest,
    const size_t N);

template<>
void GenerateRandom<int8_t>(curandGenerator_t generator, int8_t * const dest,
    const size_t N) {
  size_t align = (4 - (((size_t)dest) & 3)) % 4;
  CURAND_CHK(curandGenerate(generator, (unsigned int*)(dest+align),
      N * sizeof(int8_t) / sizeof(int)));
  CUDACHECK(cudaMemcpy(dest, dest+4, align, cudaMemcpyDeviceToDevice));
}
template<>
void GenerateRandom<uint8_t>(curandGenerator_t generator, uint8_t * const dest,
    const size_t N) {
  size_t align = (4 - (((size_t)dest) & 3)) % 4;
  CURAND_CHK(curandGenerate(generator, (unsigned int*)(dest+align),
      N * sizeof(uint8_t) / sizeof(int)));
  CUDACHECK(cudaMemcpy(dest, dest+4, align, cudaMemcpyDeviceToDevice));
}

template<>
void GenerateRandom<int32_t>(curandGenerator_t generator, int32_t * const dest,
    const size_t N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int*)dest, N));
}

template<>
void GenerateRandom<uint32_t>(curandGenerator_t generator, uint32_t * const dest,
    const size_t N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int*)dest, N));
}

template<>
void GenerateRandom<float>(curandGenerator_t generator, float * const dest,
    const size_t N) {
  CURAND_CHK(curandGenerateUniform(generator, dest, N));
}

template<>
void GenerateRandom<double>(curandGenerator_t generator, double * const dest,
    const size_t N) {
  CURAND_CHK(curandGenerateUniformDouble(generator, dest, N));
}

template<>
void GenerateRandom<uint64_t>(curandGenerator_t generator, uint64_t * const dest,
    const size_t N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int *)dest, N*2));
}

template<>
void GenerateRandom<int64_t>(curandGenerator_t generator, int64_t * const dest,
    const size_t N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int *)dest, N*2));
}

template<typename T>
void RandomizeType(void* dest, const size_t N, const int randomSeed) {
  T* ptr = (T*)dest;
  curandGenerator_t gen;
  CURAND_CHK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
  CURAND_CHK(curandSetPseudoRandomGeneratorSeed(gen, randomSeed));
  GenerateRandom<T>(gen, ptr, N);
  CURAND_CHK(curandDestroyGenerator(gen));
  CUDACHECK(cudaDeviceSynchronize());
}

__global__ void halve(const float * src, half* dest, size_t N) {
  for(int tid = threadIdx.x + blockIdx.x*blockDim.x;
      tid < N; tid += blockDim.x * gridDim.x)
    dest[tid] = __float2half(src[tid]);
}

void RandomizeHalf(void* dest, const size_t N, const int randomSeed) {
  half* ptr = (half*)dest;
  curandGenerator_t gen;
  CURAND_CHK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
  CURAND_CHK(curandSetPseudoRandomGeneratorSeed(gen, randomSeed));

  float* temp;
  CUDACHECK(cudaMalloc(&temp, N*sizeof(float)));
  GenerateRandom<float>(gen, temp, N);
  halve<<<128, 512>>>(temp, ptr, N);
  CURAND_CHK(curandDestroyGenerator(gen));
  CUDACHECK(cudaFree(temp));
  CUDACHECK(cudaDeviceSynchronize());
}

void Randomize(void* ptr, const size_t count, ncclDataType_t type, const int seed) {
  switch (type) {
    case ncclChar:   RandomizeType<int8_t>  (ptr, count, seed); break;
#if NCCL_MAJOR >= 2
    case ncclUint8:  RandomizeType<uint8_t> (ptr, count, seed); break;
#endif
    case ncclInt:    RandomizeType<int32_t> (ptr, count, seed); break;
#if NCCL_MAJOR >= 2
    case ncclUint32: RandomizeType<uint32_t>(ptr, count, seed); break;
#endif
    case ncclInt64:  RandomizeType<int64_t> (ptr, count, seed); break;
    case ncclUint64: RandomizeType<uint64_t>(ptr, count, seed); break;
    case ncclHalf:   RandomizeHalf          (ptr, count, seed); break;
    case ncclFloat:  RandomizeType<float>   (ptr, count, seed); break;
    case ncclDouble: RandomizeType<double>  (ptr, count, seed); break;
  }
}

template<typename T, int OP> __global__ static
void accumKern(T* acum, const T* contrib, size_t N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    T c = contrib[i];
    T a = acum[i];
    if(OP == ncclSum) {
      acum[i] = a+c;
    } else if(OP == ncclProd) {
      acum[i] = a*c;
    } else if(OP == ncclMax) {
      acum[i] = (a > c) ? a : c;
    } else if(OP == ncclMin) {
      acum[i] = (a < c) ? a : c;
    }
  }
}

template<> __global__
void accumKern<half, ncclSum>(half* acum, const half* contrib, size_t N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( a + c );
  }
}

template<> __global__
void accumKern<half, ncclProd>(half* acum, const half* contrib, size_t N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( a * c );
  }
}

template<> __global__
void accumKern<half, ncclMax>(half* acum, const half* contrib, size_t N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( (a>c) ? a : c );
  }
}

template<> __global__
void accumKern<half, ncclMin>(half* acum, const half* contrib, size_t N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( (a<c) ? a : c );
  }
}

template<typename T>
void accVecType(void* out, void* in, size_t n, ncclRedOp_t op) {
  switch(op) {
    case ncclSum:  accumKern<T, ncclSum> <<<256,256>>>((T*)out, (T*)in, n); break;
    case ncclProd: accumKern<T, ncclProd><<<256,256>>>((T*)out, (T*)in, n); break;
    case ncclMax:  accumKern<T, ncclMax> <<<256,256>>>((T*)out, (T*)in, n); break;
    case ncclMin:  accumKern<T, ncclMin> <<<256,256>>>((T*)out, (T*)in, n); break;
    default:
      printf("Unknown reduction operation.\n");
      exit(EXIT_FAILURE);
  }
}

void Accumulate(void* out, void* in, size_t n, ncclDataType_t type, ncclRedOp_t op) {
  switch (type) {
    case ncclChar:   accVecType<int8_t>   (out, in, n, op); break;
#if NCCL_MAJOR >= 2
    case ncclUint8:  accVecType<uint8_t>  (out, in, n, op); break;
#endif
    case ncclInt:  accVecType<int32_t>  (out, in, n, op); break;
#if NCCL_MAJOR >= 2
    case ncclUint32: accVecType<uint32_t> (out, in, n, op); break;
#endif
    case ncclInt64:  accVecType<int64_t>  (out, in, n, op); break;
    case ncclUint64: accVecType<uint64_t> (out, in, n, op); break;
    case ncclHalf:   accVecType<half>     (out, in, n, op); break;
    case ncclFloat:  accVecType<float>    (out, in, n, op); break;
    case ncclDouble: accVecType<double>   (out, in, n, op); break;
    default:
      printf("Unknown reduction type.\n");
      exit(EXIT_FAILURE);
  }
}

void Barrier(struct threadArgs_t* args)
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

void RandomizeAccumulate(void* data, void* accum, size_t count, ncclDataType_t type, ncclRedOp_t op, int seed, int rank) {
  Randomize(data, count, type, seed);
  if (rank == 0) {
    CUDACHECK(cudaMemcpy(accum, data, count*wordSize(type), cudaMemcpyDeviceToHost));
  } else {
    Accumulate(accum, data, count, type, op);
  }
}

double CheckData(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place) {
  size_t count = args->expectedBytes/wordSize(type);
  double maxDelta = 0.0;
  for (int i=0; i<args->nGpus; i++) {
    int device;
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void *data = in_place ? ((void *)((uintptr_t)args->recvbuffs[i] + args->recvInplaceOffset*rank)) : args->recvbuffs[i];
    CheckDelta(data , args->expected[i], count, type, args->delta);
    cudaDeviceSynchronize();
    maxDelta = std::max(*(args->deltaHost), maxDelta);

#ifdef DEBUG_PRINT
    if (rank == 0) { 
       int *temp = (int *)malloc(args->expectedBytes);

       printf("\n Expected: ");
       for(int j=0; j<args->expectedBytes/sizeof(int); j++) { 
       	printf("%d:%d ", j, *((int *)args->expectedHost[0] + j));
       }
       printf("\n");

       cudaMemcpy(temp, data, args->expectedBytes, cudaMemcpyDeviceToHost);
       printf("\n Actual: ");
       for (int j=0; j<args->expectedBytes/sizeof(int); j++) { 
       	printf("%d:%d ", j, *((int *)temp + j));
       }
       printf("\n");
       free(temp);
    }
#endif
  }
  double nranks = args->nProcs*args->nThreads*args->nGpus;
  if (maxDelta > DeltaMaxValue(type)*(nranks - 1)) args->errors[0]++;
  return maxDelta;
}

void InitSendRecv(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t count = args->sendBytes / wordSize(type);
  static int rep = 1;
  for (int i=0; i<args->nGpus; i++) {
    int device;
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    // Always zero recvbuff
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    // Then init send or recv buff depending on in_place
    void* data = in_place ? (void *)((uintptr_t)args->recvbuffs[i] + args->sendInplaceOffset*rank) : args->sendbuffs[i];
    int seed = rank+count+rep+in_place;
    Randomize(data, count, type, seed);

#ifdef DEBUG_PRINT
    if (rank == 2) { 
       int *temp = (int *)malloc(args->sendBytes);
       cudaMemcpy(temp, data, args->sendBytes, cudaMemcpyDeviceToHost);
       printf("\n Send Data at rank %d:", rank);
       for (int i=0; i<args->sendBytes/sizeof(int); i++) { 
       	printf("%d:%d ", i, *((int *)temp + i));
       }
       printf("\n");
       free(temp);
    }
#endif

    cudaDeviceSynchronize();
  }
  rep++;
}

cudaError_t cudaStreamSyncYield(cudaStream_t stream) {
  while (1) {
    cudaError_t err = cudaStreamQuery(stream);
    if (err != cudaErrorNotReady) return err;
    pthread_yield();
  }
}

void startColl(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int thread_offset) {
  size_t count = args->nbytes / wordSize(type);

  if (swap_args) {
      args = (struct threadArgs_t*)args->proc_args + (args->thread + thread_offset)%args->nThreads;
      if (args->nGpus == 1) {
        int cudaDev;
        NCCLCHECK(ncclCommCuDevice(args->comms[0], &cudaDev));
        CUDACHECK(cudaSetDevice(cudaDev));
      }
  }

  if (args->nGpus == 1) {
    int rank = args->proc*args->nThreads + args->thread;
    args->coll->RunColl((void*)(in_place ? ((void *)((uintptr_t)args->recvbuffs[0] + args->sendInplaceOffset*rank)) : args->sendbuffs[0]),
        (void*)(in_place ? (void*)((uintptr_t)args->recvbuffs[0] + args->recvInplaceOffset*rank) : args->recvbuffs[0]),
        count, type, op, root, args->comms[0], args->streams[0]);
  } else {
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < args->nGpus; i++) {
#ifndef NCCL_MAJOR
      int cudaDev;
      NCCLCHECK(ncclCommCuDevice(args->comms[i], &cudaDev));
      CUDACHECK(cudaSetDevice(cudaDev));
#endif
      int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
      args->coll->RunColl((void*)(in_place ? ((void *)((uintptr_t)args->recvbuffs[i] + args->sendInplaceOffset*rank)) : args->sendbuffs[i]),
          (void*)(in_place ? (void*)((uintptr_t)args->recvbuffs[i] + args->recvInplaceOffset*rank) : args->recvbuffs[i]),
          count, type, op, root, args->comms[i], args->streams[i]);
    }
    NCCLCHECK(ncclGroupEnd());
  }

  if (swap_args || blocking_coll) {
    //if args have been swapped, complete op before returning
    for (int i = 0; i < args->nGpus; ++i) {
      CUDACHECK(cudaStreamSyncYield(args->streams[i]));
    }
  }
  if (blocking_coll) Barrier(args);
}

void completeColl(struct threadArgs_t* args) {
  //it swap_args was enabled, op would have been completed immediately
  if (swap_args || blocking_coll) return;

  for (int i = 0; i < args->nGpus; ++i) {
    CUDACHECK(cudaStreamSyncYield(args->streams[i]));
  }
}

void BenchTime(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place) {
  if (iters == 0) return;
  size_t count = args->nbytes / wordSize(type);

  // Sync
  //startColl(args, type, op, root, in_place, 0);
  completeColl(args);

  Barrier(args);

  // Performance Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iters; iter++) {
    if (agg_iters>1) NCCLCHECK(ncclGroupStart());
    for (int iter = 0; iter < agg_iters; iter++) {
      startColl(args, type, op, root, in_place, iter);
    }
    if (agg_iters>1) NCCLCHECK(ncclGroupEnd());
  }
  completeColl(args);

  auto delta = std::chrono::high_resolution_clock::now() - start;
  double timeSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
  timeSec = timeSec/(iters*agg_iters);

  double algBw, busBw;
  args->coll->GetBw(count, wordSize(type), timeSec, &algBw, &busBw, args->nProcs*args->nThreads*args->nGpus);

  Barrier(args);

  double maxDelta = 0;
  if (datacheck) { 
      InitSendRecv(args, type, op, root, in_place, args->thread == 0 ? 1 : 0);
      args->coll->InitRecvResult(args, type, op, root, in_place, args->thread == 0 ? 1 : 0);
      cudaDeviceSynchronize();

      //test validation in single itertion, should ideally be included into the multi-iteration run
      startColl(args, type, op, root, in_place, 0);
      completeColl(args);

      maxDelta = CheckData(args, type, op, root, in_place);

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

  double timeUsec = timeSec*1.0E6;
  char timeStr[10];
  if (timeUsec > 10000.0) {
    sprintf(timeStr, "%7.0f", timeUsec);
  } else if (timeUsec > 100.0) {
    sprintf(timeStr, "%7.1f", timeUsec);
  } else {
    sprintf(timeStr, "%7.2f", timeUsec);
  }
  if (datacheck) { 
     PRINT("  %7s  %6.2f  %6.2f  %5.0le", timeStr, algBw, busBw, maxDelta);
  } else {
     PRINT("  %7s  %6.2f  %6.2f  %5s", timeStr, algBw, busBw, "N/A");
  }

  args->bw[0] += busBw;
  args->bw_count[0]++;
}

void setupArgs(size_t size, ncclDataType_t type, struct threadArgs_t* args) {
  int nranks = args->nProcs*args->nGpus*args->nThreads;
  size_t count, sendCount, recvCount, paramCount, sendInplaceOffset, recvInplaceOffset, procSharedCount;
  int sameExpected;
  
  count = size / wordSize(type);
  args->coll->getCollByteCount(&sendCount, &recvCount, &paramCount, &sendInplaceOffset, &recvInplaceOffset, &procSharedCount, &sameExpected, (size_t)count, (size_t)nranks);

  args->nbytes = paramCount * wordSize(type);
  args->sendBytes = sendCount * wordSize(type);
  args->expectedBytes = recvCount * wordSize(type);
  args->sendInplaceOffset = sendInplaceOffset * wordSize(type);
  args->recvInplaceOffset = recvInplaceOffset * wordSize(type);

  for (int i = 0; i < args->nGpus; i++) {
    if (sameExpected) {
      args->expectedHost[i] = expectedHost[0];
      args->expected[i] = expected[0];
    } else {
      args->expectedHost[i] = expectedHost[args->thread*args->nGpus + i];
      args->expected[i] = expected[args->thread*args->nGpus + i];
    }
  }
}

void TimeTest(struct threadArgs_t* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName, int root) {
  // Warm-up for large size
  setupArgs(args->maxbytes, type, args);
  for (int iter = 0; iter < warmup_iters; iter++) {
    startColl(args, type, op, root, 0, iter);
  }
  completeColl(args);

  // Warm-up for small size
  setupArgs(args->minbytes, type, args);
  for (int iter = 0; iter < warmup_iters; iter++) {
    startColl(args, type, op, root, 0, iter);
  }
  completeColl(args);

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
        BenchTime(args, type, op, root, 0);
      }
      BenchTime(args, type, op, root, 1);
      PRINT("    %s\n", args->replayFile == NULL ? "" : args->coll->name);
  }
}

void* threadRunTests(void* args) {
  struct threadArgs_t* targs = (struct threadArgs_t*)args;
  // Set device to the first of our GPUs. If we don't do that, some operations
  // will be done on the current GPU (by default : 0) and if the GPUs are in
  // exclusive mode those operations will fail.
  int gpuid = targs->localRank*targs->nThreads*targs->nGpus + targs->thread*targs->nGpus;
  CUDACHECK(cudaSetDevice(gpuid));

  RunTest(targs, ncclroot, (ncclDataType_t)nccltype, test_typenames[nccltype], (ncclRedOp_t)ncclop, test_opnames[ncclop]);

  return NULL;
}

void* threadInit(void* args) {
  struct threadArgs_t* targs = (struct threadArgs_t*)args;
  char hostname[1024];
  getHostName(hostname, 1024);
  int nranks =  targs->nProcs*targs->nThreads*targs->nGpus;

  //set main thread again
  is_main_thread = (targs->proc == 0 && targs->thread == 0) ? 1 : 0;

  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<targs->nGpus; i++) {
    int rank = targs->proc*targs->nThreads*targs->nGpus + targs->thread*targs->nGpus + i;
    int gpuid = targs->localRank*targs->nThreads*targs->nGpus + targs->thread*targs->nGpus + i;
    CUDACHECK(cudaSetDevice(gpuid));
    NCCLCHECK(ncclCommInitRank(targs->comms+i, nranks, targs->ncclId, rank));
  }
  NCCLCHECK(ncclGroupEnd());

  threadRunTests(args);

  for (int i=0; i<targs->nGpus; i++) {
    NCCLCHECK(ncclCommDestroy(targs->comms[i]));
  }
  return NULL;
}


__global__ void compute(void* _ptr, int _size) {
  uint64_t *ptr = (uint64_t*)_ptr;
  int size = _size / sizeof(uint64_t);
  for (int offset=threadIdx.x; offset < size; offset += blockDim.x) {
     ptr[offset] <<= 1;
  }
}
#define COMP_SIZE (1 << 20)
void* compThread(void* args) {
  struct threadArgs_t* targs = (struct threadArgs_t*)args;
  void* ptrs[targs->nGpus];
  int gpuids[targs->nGpus];
  cudaStream_t streams[targs->nGpus];
  for (int i=0; i<targs->nGpus; i++) {
    gpuids[i] = targs->localRank*targs->nThreads*targs->nGpus + targs->thread*targs->nGpus + i;
    CUDACHECK(cudaSetDevice(gpuids[i]));
    CUDACHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));
  }
  while (targs->compThreadStop == 0) {
    for (int i=0; i<targs->nGpus; i++) {
      CUDACHECK(cudaSetDevice(gpuids[i]));
      CUDACHECK(cudaMalloc(ptrs+i, COMP_SIZE));
    }
    for (int i=0; i<targs->nGpus; i++) {
      CUDACHECK(cudaSetDevice(gpuids[i]));
      compute<<<1, 256, 0, streams[i]>>>(ptrs[i], COMP_SIZE);
    }
    for (int i=0; i<targs->nGpus; i++) {
      CUDACHECK(cudaStreamSyncYield(streams[i]));
    }
    for (int i=0; i<targs->nGpus; i++) {
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
  for (int i=0; i<targs->nGpus; i++) {
    CUDACHECK(cudaStreamDestroy(streams[i]));
  }
  return NULL;
}

void AllocateBuffs(void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes, void **expected, void **expectedHost, size_t nbytes, int nranks, int sameExpected) {
    static int is_first = 1;
    static void *cached_ptr = NULL;
    static void *cached_hostptr = NULL;

    CUDACHECK(cudaMalloc(sendbuff, sendBytes));
    //work around for inline reduce scatter where recv count is smaller that send count
    CUDACHECK(cudaMalloc(recvbuff, (sendBytes > recvBytes) ? sendBytes : recvBytes));

    if (is_first || !sameExpected) {
        CUDACHECK(cudaHostAlloc(expectedHost, recvBytes, cudaHostAllocPortable | cudaHostAllocMapped));
        *expected = *expectedHost;
        cached_ptr = *expected;
        cached_hostptr = *expectedHost;
        is_first = 0;
    } else {
        *expected = cached_ptr;
        *expectedHost = cached_hostptr;
    }
}
 
int main(int argc, char* argv[]) {
 int nThreads = 1, nGpus = 1;
 size_t minBytes = 32*1024*1024, maxBytes = 32*1024*1024, stepBytes = 1*1024*1024, stepFactor = 1;
 int longindex;
 int nProcs = 1, proc = 0;
 int localRank = 0;
 char hostname[1024];
 getHostName(hostname, 1024);
 
 static struct option longopts[] = {
    {"nthreads", required_argument, 0, 't'}, 
    {"ngpus", required_argument, 0, 'g'}, 
    {"minbytes", required_argument, 0, 'b'}, 
    {"maxbytes", required_argument, 0, 'e'}, 
    {"stepbytes", required_argument, 0, 'i'},
    {"stepfactor", required_argument, 0, 'f'},
    {"iters", required_argument, 0, 'n'},
    {"agg-iters", required_argument, 0, 'm'},
    {"warmup_iters", required_argument, 0, 'w'},
    {"swap_comms", required_argument, 0, 's'},
    {"parallel_init", required_argument, 0, 'p'},
    {"check", required_argument, 0, 'c'},
    {"op", required_argument, 0, 'o'},
    {"datatype", required_argument, 0, 'd'},
    {"root", required_argument, 0, 'r'},
    {"blocking", required_argument, 0, 'z'},
    {"stream_null", required_argument, 0, 'y'},
    {"side_comp", required_argument, 0, 'k'},
    {"replay", required_argument, 0, 'l'},
    {"help", no_argument, 0, 'h'}
 };

 while(1) {
      int c;
      c = getopt_long(argc, argv, "t:g:b:e:i:f:n:m:w:s:p:c:o:d:r:z:y:k:h:l:", longopts, &longindex);

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
	 case 's':
	     swap_args = (int)strtol(optarg, NULL, 0);
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
         case 'h':
	         printf("USAGE: ./test \n\t" 
	 	 "[-t,--nthreads <num threads>] \n\t"
		 "[-g,--ngpus <gpus per thread>] \n\t"
		 "[-b,--minbytes <min size in bytes>] \n\t"
		 "[-e,--maxbytes <max size in bytes>] \n\t"
	         "[-i,--stepbytes <increment size>] \n\t"
		 "[-f,--stepfactor <increment factor>] \n\t"
		 "[-n,--iters <iteration count>] \n\t"
		 "[-m,--agg-iters <aggregated iteration count>] \n\t"
		 "[-w,--warmup_iters <warmup iteration count>] \n\t"
		 "[-s,--swap_args <0/1>] \n\t"
		 "[-p,--parallel_init <0/1>] \n\t"
		 "[-c,--check <0/1>] \n\t"
		 "[-o,--op <sum/prod/min/max/all>] \n\t"
		 "[-d,--datatype <nccltype/all>] \n\t"
		 "[-r,--root <root>] \n\t"
		 "[-z,--blocking <0/1>] \n\t"
		 "[-y,--stream_null <0/1>] \n\t"
		 "[-k,--side_comp <0/1>] \n\t"
		 "[-l,--replay <path to replay file>] \n\t"
		 "[-h,--help]\n");
	         return 0;
	 default: 
	         printf("invalid option \n");
	         printf("USAGE: ./test \n\t" 
	 	 "[-t,--nthreads <num threads>] \n\t"
		 "[-g,--ngpus <gpus per thread>] \n\t"
		 "[-b,--minbytes <min size in bytes>] \n\t"
		 "[-e,--maxbytes <max size in bytes>] \n\t"
	         "[-i,--stepbytes <increment size>] \n\t"
		 "[-f,--stepfactor <increment factor>] \n\t"
		 "[-n,--iters <iteration count>] \n\t"
		 "[-m,--agg-iters <aggregated iteration count>] \n\t"
		 "[-w,--warmup_iters <warmup iteration count>] \n\t"
		 "[-s,--swap_args <0/1>] \n\t"
		 "[-p,--parallel_init <0/1>] \n\t"
		 "[-c,--check <0/1>] \n\t"
		 "[-o,--op <sum/prod/min/max/all>] \n\t"
		 "[-d,--datatype <nccltype/all>] \n\t"
		 "[-r,--root <root>] \n\t"
		 "[-z,--blocking <0/1>] \n\t"
		 "[-y,--stream_null <0/1>] \n\t"
		 "[-k,--side_comp <0/1>] \n\t"
		 "[-l,--replay <path to replay file>] \n\t"
		 "[-h,--help]\n");
	         return 0;
      }
  }

  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

#ifdef MPI_SUPPORT
  MPI_Init(&argc, &argv);
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

  PRINT("# nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: %d validation: %d \n", nThreads, nGpus, minBytes, maxBytes,
      (stepFactor > 1)?stepFactor:stepBytes, (stepFactor > 1)?"factor":"bytes", warmup_iters, iters, datacheck);
  if (swap_args) PRINT("# Swap Comms Enabled: swapping communicators among threads for each iteration \n");
  if (blocking_coll) PRINT("# Blocking Enabled: wait for completion and barrier after each collective \n");
  if (parallel_init) PRINT("# Parallel Init Enabled: threads call into NcclInitRank concurrently \n");
  PRINT("#\n");

  PRINT("# Using devices\n");
#define MAX_LINE 2048
  char line[MAX_LINE];
  int len = 0;
  for (int i=0; i<nThreads*nGpus; i++) {
    int cudaDev = localRank*nThreads*nGpus+i;
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
  cudaStream_t streams[nGpus*nThreads];
  void* sendbuffs[nGpus*nThreads];
  void* recvbuffs[nGpus*nThreads];
  expected = (void**)malloc(sizeof(void*)*nGpus*nThreads);
  expectedHost = (void**)malloc(sizeof(void*)*nGpus*nThreads);
  void *procSharedHost, *procShared;
  size_t sendBytes, recvBytes, procSharedBytes;
  int sameExpected;

  GetBuffSize(&sendBytes, &recvBytes, &procSharedBytes, &sameExpected, (size_t)maxBytes, (size_t)nProcs*nGpus*nThreads);

  for (int i=0; i<nGpus*nThreads; i++) {
    CUDACHECK(cudaSetDevice(localRank*nThreads*nGpus+i));
    AllocateBuffs(sendbuffs+i, sendBytes, recvbuffs+i, recvBytes, expected+i, expectedHost+i, (size_t)maxBytes, nProcs*nThreads*nGpus, sameExpected);
    if (streamnull)
      streams[i] = NULL;
    else
      CUDACHECK(cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking));
  }

  if (procSharedBytes > 0) { 
      CUDACHECK(cudaHostAlloc(&procSharedHost, procSharedBytes, cudaHostAllocPortable | cudaHostAllocMapped));
      procShared = procSharedHost;
  }

  //if parallel init is not selected, use main thread to initialize NCCL
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nThreads*nGpus);
  if (!parallel_init) {
     if (nProcs == 1) {
       int gpuArray[nGpus*nThreads];
       for (int i=0; i<nGpus*nThreads; i++) gpuArray[i] = i;
       NCCLCHECK(ncclCommInitAll(comms, nGpus*nThreads, gpuArray));
     } else {
       NCCLCHECK(ncclGroupStart());
       for (int i=0; i<nGpus*nThreads; i++) {
         CUDACHECK(cudaSetDevice(localRank*nThreads*nGpus+i));
         NCCLCHECK(ncclCommInitRank(comms+i, nProcs*nThreads*nGpus, ncclId, proc*nThreads*nGpus+i)); 
       }
       NCCLCHECK(ncclGroupEnd());
     }
  }

  int errors[nThreads];
  double bw[nThreads];
  double* delta;
  CUDACHECK(cudaHostAlloc(&delta, sizeof(double)*nThreads, cudaHostAllocPortable | cudaHostAllocMapped));
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

  pthread_t threads[nThreads];
  pthread_t compThreads[nThreads];
  struct threadArgs_t args[nThreads];
  memset(args, 0, sizeof(struct threadArgs_t)*nThreads);

  if (side_comp && signal(SIGCHLD, SIG_IGN) == SIG_ERR) {
    printf("Failed to set up automatic cleanup of zombie processes\n");
    exit(EXIT_FAILURE);
  }

  for (int t=nThreads-1; t>=0; t--) {
    args[t].proc_args = (void *)args;
    args[t].minbytes=minBytes;
    args[t].maxbytes=maxBytes;
    args[t].stepbytes=stepBytes;
    args[t].stepfactor=stepFactor;
    args[t].localRank = localRank;

    args[t].nProcs=nProcs;
    args[t].proc=proc;
    args[t].nThreads=nThreads;
    args[t].thread=t;
    args[t].nGpus=nGpus;
    args[t].sendbuffs = sendbuffs+t*nGpus;
    args[t].recvbuffs = recvbuffs+t*nGpus;
    args[t].ncclId = ncclId;
    args[t].comms=comms+t*nGpus;
    args[t].streams=streams+t*nGpus;

    args[t].expectedHost = (void**)malloc(nGpus*sizeof(void*));
    args[t].expected = (void**)malloc(nGpus*sizeof(void*));
    args[t].procSharedHost = procSharedHost; 
    args[t].procShared = procShared; 
    args[t].barrier = (volatile int*)barrier;
    args[t].barrier_idx = 0;
    args[t].sync = (volatile int*)sync;
    args[t].sync_idx = 0;
    args[t].deltaThreads = delta;
    args[t].deltaHost = (delta + t);
    args[t].delta = delta;
    args[t].errors=errors+t;
    args[t].bw=bw+t;
    args[t].bw_count=bw_count+t;

    args[t].replayFile = replay_file;

    if (side_comp) {
      pthread_create(compThreads+t, NULL, compThread, args+t);
    }
    if (!parallel_init) { 
       if (t) 
         pthread_create(threads+t, NULL, threadRunTests, args+t);
       else
         threadRunTests(args);
    } else {
        if (t || (parallel_init && (proc == 0))) 
         pthread_create(threads+t, NULL, threadInit, args+t);
       else  
         threadInit(args);
    }
  }

  // Wait for other threads
  for (int t=nThreads-1; t>=0; t--) {
    if (t || (parallel_init && (proc == 0))) pthread_join(threads[t], NULL);
    errors[0] += errors[t];
    bw[0] += bw[t];
    bw_count[0] += bw_count[t];
    if (side_comp) {
       args[t].compThreadStop = 1;
       pthread_join(compThreads[t], NULL);
    }
  }

#ifdef MPI_SUPPORT
    MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (!parallel_init) {
    for(int i=0; i<nGpus*nThreads; ++i)
      ncclCommDestroy(comms[i]);
    free(comms);
  }

  char* str = getenv("NCCL_TESTS_MIN_BW");
  double check_avg_bw = str ? atof(str) : -1;
  bw[0] /= bw_count[0];

  PRINT("# Out of bounds values : %d %s\n", errors[0], errors[0] ? "FAILED" : "OK");
  PRINT("# Avg bus bandwidth    : %g %s\n", bw[0], check_avg_bw == -1 ? "" : (bw[0] < check_avg_bw*(0.9) ? "FAILED" : "OK"));
  PRINT("#\n");
#ifdef MPI_SUPPORT
  MPI_Finalize();
#endif
  if (errors[0] || bw[0] < check_avg_bw*(0.9))
    exit(EXIT_FAILURE);
  else 
    exit(EXIT_SUCCESS);
}
