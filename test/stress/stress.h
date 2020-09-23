/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#include "nccl.h"
#include <stdio.h>
#include <algorithm>
#include <curand.h>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
#include <pthread.h>

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

typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testCudaError = 2,
  testNcclError = 3,
  testTimeout = 4,
  testDataError = 5,
  testNumResults = 6
} testResult_t;

// Relay errors up and trace
#define TESTCHECK(cmd) do {                         \
  testResult_t r = cmd;                             \
  if (r!= testSuccess) {                            \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf(" .. %s: Test failure %s:%d %d\n",       \
         hostname,                                  \
        __FILE__,__LINE__, r);                      \
    return r;                                       \
  }                                                 \
} while(0)

struct testCall {
  const char* name;
  int func;
  int rank;
  size_t count;
  int root;
  ncclDataType_t datatype;
  ncclRedOp_t redop;
  int group;
};

struct threadArgs {
  int nProcs;
  int proc;
  int nThreads;
  int thread;
  int nGpus;
  int* gpus;
  int localRank;
  void** sendBuffsBase;
  size_t sendBytes;
  size_t sendInplaceOffset;
  void** recvBuffsBase;
  size_t recvInplaceOffset;
  ncclComm_t* comms;
  cudaStream_t* streams;
  void** expected;
  size_t expectedBytes;
  volatile int* sync;
  int sync_idx;
  volatile int* barrier;
  int barrier_idx;
  int syncRank;
  int syncNranks;
  double* deltaThreads;
  double* deltaHost;
  double* delta;
  struct testCall* calls;
  int nCalls;
  char** sendBuffs;
  char** recvBuffs;
  int* errors;
};

typedef testResult_t (*threadFunc_t)(struct threadArgs* args);
struct testThread {
  pthread_t thread;
  threadFunc_t func;
  struct threadArgs args;
  testResult_t ret;
};

#include <chrono>
#include <unistd.h>

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

#include <stdint.h>

static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static size_t wordSize(ncclDataType_t type) {
  switch(type) {
    case ncclChar:
#if NCCL_MAJOR >= 2
    //case ncclInt8:
    case ncclUint8:
#endif
      return 1;
    case ncclHalf:
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

const int testNumFuncs = 9;
typedef testResult_t (*testFunc_t)(struct testCall* args, struct threadArgs* targs);
extern const char* testFuncNames[testNumFuncs];
extern const char *testTypeNames[ncclNumTypes];
extern const char *testOpNames[ncclNumOps];

static testResult_t ncclStringToFunc(char *str, int* func, const char** name) {
  for (int f=0; f<testNumFuncs; f++) {
    if (strcmp(str, testFuncNames[f]) == 0) {
      *func = f;
      *name = testFuncNames[f];
      return testSuccess;
    }
  }
  printf("Unknown function %s\n", str);
  return testInternalError;
}

static testResult_t ncclStringToType(char *str, ncclDataType_t* type) {
  for (int t=0; t<ncclNumTypes; t++) {
    if (strcmp(str, testTypeNames[t]) == 0) {
      *type = (ncclDataType_t)t;
      return testSuccess;
    }
  }
  printf("Unknown type %s\n", str);
  return testInternalError;
}

static testResult_t ncclStringToOp (char *str, ncclRedOp_t* redop) {
  for (int o=0; o<ncclNumOps; o++) {
    if (strcmp(str, testOpNames[o]) == 0) {
      *redop = (ncclRedOp_t)o;
      return testSuccess;
    }
  }
  printf("Unknown operation %s\n", str);
  return testInternalError;
}

extern thread_local int is_main_thread;
#define PRINT if (is_main_thread) printf

#endif
