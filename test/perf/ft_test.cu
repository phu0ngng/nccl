#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include <assert.h>
#include "common.h"

enum {
  FT_TEST_INIT = 0,
  FT_TEST_ALLREDUCE = 1,
  FT_TEST_ALLTOALL = 2,
  FT_TEST_FINALIZE = 3,
  FT_TEST_SPLIT = 4,
  FT_TEST_SHRINK = 5,
  FT_TEST_ABORT = 6,
  FT_TEST_NUM = 7,
};

#define PRINT if (is_main_thread) printf

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
#define NUM_SLEEP_CASES 5
int sleepTimes[NUM_SLEEP_CASES] = { 10, 100, 10000, 1000000, 2000000}; /* sleep in us */
size_t size = 32 * 1024 * 1024;

static testResult_t checkCommsState(ncclComm_t* comms, int nGpus, ncclResult_t stateStart, ncclResult_t stateExpect) {
  ncclResult_t state;
  testResult_t ret = testSuccess;
  int complete;
  do {
    complete = 1;
    for (int j = 0; j < nGpus; ++j) {
      NCCLCHECK(ncclCommGetAsyncError(comms[j], &state));
      if (state == stateStart) {
        complete = 0;
        break;
      }
      usleep(10);
    }
#ifdef MPI_SUPPORT
    int flag;
    extern pthread_mutex_t mpiLock;
    pthread_mutex_lock(&mpiLock);
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    pthread_mutex_unlock(&mpiLock);
#endif
  } while (!complete);

  for (int j = 0; j < nGpus; ++j) {
    NCCLCHECK(ncclCommGetAsyncError(comms[j], &state));
    if (state != stateExpect) {
      PRINT("Distributed FT:\tCheck comms[%d] state, query state %s != expected state %s  [FAIL]\n", j, ncclGetErrorString(state), ncclGetErrorString(stateExpect));
      ret = testNcclError;
    }
  }

  return ret;
}

static testResult_t initBufferStream(void** sendbuffs, void** recvbuffs, void** hostbuffs, cudaStream_t* streams, int sDev, int nGpus, size_t size) {
  for (int i = 0; i < nGpus; ++i) {
    int dev = sDev + i;
    CUDACHECK(cudaSetDevice(dev));
    hostbuffs[i] = malloc(size);
    assert(hostbuffs[i] != NULL);
    CUDACHECK(cudaMalloc((void**)&sendbuffs[i], size));
    CUDACHECK(cudaMalloc((void**)&recvbuffs[i], size));
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }
  return testSuccess;
}

static void finalizeBufferStream(void** sendbuffs, void** recvbuffs, void** hostbuffs, cudaStream_t* streams, int nGpus) {
  for (int i = 0; i < nGpus; ++i) {
    cudaFree(sendbuffs[i]);
    cudaFree(recvbuffs[i]);
    free(hostbuffs[i]);
    cudaStreamDestroy(streams[i]);
  }
  return;
}

static testResult_t distributeFTInitTest(struct threadArgs* args) {
  void** sendbuffs = args->sendbuffs[0];
  void** recvbuffs = args->recvbuffs[0];
  cudaStream_t* streams = args->streams;
  char** hostbuffs = (char**)args->hostbuffs;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;
  ncclComm_t* comms = args->comms[0];
  int nGpus = args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * args->nGpus;
  int sDev = args->localRank * args->nThreads * args->nGpus + args->thread * args->nGpus;
  int sleepId = args->sleepId;

  config.blocking = 0;
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, *args->ncclId, rank, &config));
  }
  ret = ncclGroupEnd();
  assert(ret == ncclSuccess || ret == ncclInProgress);

  if (sleepId < NUM_SLEEP_CASES) {
    usleep(sleepTimes[sleepId]);
    for (int j = 0; j < nGpus; ++j) ncclCommAbort(comms[j]);
    goto exit;
  } else {
    TESTCHECK(checkCommsState(comms, nGpus, ncclInProgress, ncclSuccess));
  }

  //communicating using NCCL
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j)
    NCCLCHECK(ncclAllReduce((const void*)sendbuffs[j], (void*)recvbuffs[j], size, ncclInt8, ncclProd, comms[j], streams[j]));
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  //completing NCCL operation by synchronizing on the CUDA stream
  for (int j = 0; j < nGpus; ++j)
    CUDACHECK(cudaStreamSynchronize(streams[j]));

  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    CUDACHECK(cudaSetDevice(dev));
    CUDACHECK(cudaMemcpy(hostbuffs[j], recvbuffs[j], size, cudaMemcpyDeviceToHost));
    for (int k = 0; k < size; ++k) {
      if ((int)hostbuffs[j][k] != 1) {
        printf("Proc %d, Thread %d - \tncclAllReduce wrong result %d at device %d buffer location [%d] != expected %d\n", args->proc, args->thread, hostbuffs[j][k], dev, k, 1);
      }
    }
  }

  //finalizing NCCL
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    NCCLCHECK(ncclCommFinalize(comms[j]));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  for (int j = 0; j < nGpus; ++j)
    NCCLCHECK(ncclCommDestroy(comms[j]));

exit:
  return testSuccess;
}

static testResult_t distributeFTCommSplitTest(struct threadArgs* args) {
  void** sendbuffs = args->sendbuffs[0];
  void** recvbuffs = args->recvbuffs[0];
  cudaStream_t* streams = args->streams;
  char** hostbuffs = (char**)args->hostbuffs;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;
  ncclComm_t* comms = args->comms[0];
  int nGpus = args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * args->nGpus;
  int sDev = args->localRank * args->nThreads * args->nGpus + args->thread * args->nGpus;
  int sleepId = args->sleepId;
  ncclComm_t splitComms[nGpus];

  config.blocking = 0;
  config.splitShare = 1;
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, *args->ncclId, rank, &config));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    NCCLCHECK(ncclCommSplit(comms[j], rank & 1, rank, &splitComms[j], &config));
  }
  ret = ncclGroupEnd();
  assert(ret == ncclSuccess || ret == ncclInProgress);

  if (sleepId < NUM_SLEEP_CASES) {
    usleep(sleepTimes[sleepId]);
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nGpus; ++j) {
      ncclCommAbort(comms[j]);
      ncclCommAbort(splitComms[j]);
    }
    NCCLCHECK(ncclGroupEnd());
    goto exit;
  } else {
    TESTCHECK(checkCommsState(comms, nGpus, ncclInProgress, ncclSuccess));
  }

  //communicating using NCCL
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j)
    NCCLCHECK(ncclAllReduce((const void*)sendbuffs[j], (void*)recvbuffs[j], size, ncclInt8, ncclProd, splitComms[j], streams[j]));
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), splitComms, nGpus);

  //completing NCCL operation by synchronizing on the CUDA stream
  for (int j = 0; j < nGpus; ++j)
    CUDACHECK(cudaStreamSynchronize(streams[j]));

  for (int j = 0; j < nGpus; ++j) {
    CUDACHECK(cudaMemcpy(hostbuffs[j], recvbuffs[j], size, cudaMemcpyDeviceToHost));
    for (int k = 0; k < size; ++k) {
      if ((int)hostbuffs[j][k] != 1) {
        printf("Proc %d, Thread %d, Gpu %d - \tncclAllReduce wrong result %d buffer location [%d] != expected %d\n", args->proc, args->thread, j, hostbuffs[j][k], k, 1);
      }
    }
  }

  //finalizing NCCL
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    NCCLCHECK(ncclCommFinalize(comms[j]));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    NCCLCHECK(ncclCommFinalize(splitComms[j]));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), splitComms, nGpus);

  for (int j = 0; j < nGpus; ++j) {
    NCCLCHECK(ncclCommDestroy(comms[j]));
    NCCLCHECK(ncclCommDestroy(splitComms[j]));
  }

exit:
  return testSuccess;
}

static testResult_t distributeFTAllreduceTest(struct threadArgs* args) {
  void** sendbuffs = args->sendbuffs[0];
  void** recvbuffs = args->recvbuffs[0];
  cudaStream_t* streams = args->streams;
  char** hostbuffs = (char**)args->hostbuffs;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;
  ncclComm_t* comms = args->comms[0];
  int nGpus = args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * args->nGpus;
  int sDev = args->localRank * args->nThreads * args->nGpus + args->thread * args->nGpus;
  int sleepId = args->sleepId;

  config.blocking = 0;
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, *args->ncclId, rank, &config));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  //communicating using NCCL
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j)
    NCCLCHECK(ncclAllReduce((const void*)sendbuffs[j], (void*)recvbuffs[j], size, ncclInt8, ncclProd, comms[j], streams[j]));
  ret = ncclGroupEnd();
  assert(ret == ncclSuccess || ret == ncclInProgress);

  if (sleepId < NUM_SLEEP_CASES) {
    usleep(sleepTimes[sleepId]);
    for (int j = 0; j < nGpus; ++j) ncclCommAbort(comms[j]);
    goto exit;
  } else {
    TESTCHECK(checkCommsState(comms, nGpus, ncclInProgress, ncclSuccess));
  }

  //completing NCCL operation by synchronizing on the CUDA stream
  for (int j = 0; j < nGpus; ++j)
    CUDACHECK(cudaStreamSynchronize(streams[j]));

  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    CUDACHECK(cudaSetDevice(dev));
    CUDACHECK(cudaMemcpy(hostbuffs[j], recvbuffs[j], size, cudaMemcpyDeviceToHost));
    for (int k = 0; k < size; ++k) {
      if ((int)hostbuffs[j][k] != 1) {
        printf("Proc %d, Thread %d - \tncclAllReduce wrong result %d at device %d buffer location [%d] != expected %d\n", args->proc, args->thread, hostbuffs[j][k], dev, k, 1);
      }
    }
  }

  //finalizing NCCL
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    NCCLCHECK(ncclCommFinalize(comms[j]));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  for (int j = 0; j < nGpus; ++j)
    NCCLCHECK(ncclCommDestroy(comms[j]));

exit:
  return testSuccess;
}

static testResult_t distributeFTAlltoAllTest(struct threadArgs* args) {
  void** sendbuffs = args->sendbuffs[0];
  void** recvbuffs = args->recvbuffs[0];
  cudaStream_t* streams = args->streams;
  char** hostbuffs = (char**)args->hostbuffs;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;
  ncclComm_t* comms = args->comms[0];
  int nGpus = args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * args->nGpus;
  int sDev = args->localRank * args->nThreads * args->nGpus + args->thread * args->nGpus;
  int sleepId = args->sleepId;
  size_t count;

  config.blocking = 0;
  //initializing NCCL

  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, *args->ncclId, rank, &config));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);
#ifdef MPI_SUPPORT
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  count = size / totalGpus;
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    for (int k = 0; k < totalGpus; ++k) {
      NCCLCHECK(ncclSend(((char*)sendbuffs[j]) + k * count, count, ncclChar, k, comms[j], streams[j]));
      NCCLCHECK(ncclRecv(((char*)recvbuffs[j]) + k * count, count, ncclChar, k, comms[j], streams[j]));
    }
  }
  ret = ncclGroupEnd();
  assert(ret == ncclSuccess || ret == ncclInProgress);

  if (sleepId < NUM_SLEEP_CASES) {
    usleep(sleepTimes[sleepId]);
    for (int j = 0; j < nGpus; ++j) ncclCommAbort(comms[j]);
    goto exit;
  } else {
    TESTCHECK(checkCommsState(comms, nGpus, ncclInProgress, ncclSuccess));
  }

  //completing NCCL operation by synchronizing on the CUDA stream
  for (int j = 0; j < nGpus; ++j)
    CUDACHECK(cudaStreamSynchronize(streams[j]));

  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    CUDACHECK(cudaSetDevice(dev));
    CUDACHECK(cudaMemcpy(hostbuffs[j], recvbuffs[j], size, cudaMemcpyDeviceToHost));
    for (int k = 0; k < count * nGpus; ++k) {
      if ((int)hostbuffs[j][k] != 1) {
        printf("Proc %d, Thread %d - \tsendrecv wrong result %d at device %d buffer location [%d] != expected %d\n", args->proc, args->thread, hostbuffs[j][k], dev, k, 1);
      }
    }
  }

  //finalizing NCCL
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    NCCLCHECK(ncclCommFinalize(comms[j]));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  for (int j = 0; j < nGpus; ++j)
    NCCLCHECK(ncclCommDestroy(comms[j]));

exit:
  return testSuccess;
}

static testResult_t distributeFTFinalizeTest(struct threadArgs* args) {
  void** sendbuffs = args->sendbuffs[0];
  void** recvbuffs = args->recvbuffs[0];
  cudaStream_t* streams = args->streams;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;
  ncclComm_t* comms = args->comms[0];
  int nGpus = args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * args->nGpus;
  int sDev = args->localRank * args->nThreads * args->nGpus + args->thread * args->nGpus;
  int sleepId = args->sleepId;

  config.blocking = 0;
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, *args->ncclId, rank, &config));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  //communicating using NCCL
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j)
    NCCLCHECK(ncclAllReduce((const void*)sendbuffs[j], (void*)recvbuffs[j], size, ncclInt8, ncclProd, comms[j], streams[j]));
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  //finalizing NCCL
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    NCCLCHECK(ncclCommFinalize(comms[j]));
  }
  ret = ncclGroupEnd();
  assert(ret == ncclSuccess || ret == ncclInProgress);

  if (sleepId < NUM_SLEEP_CASES) {
    usleep(sleepTimes[sleepId]);
    for (int j = 0; j < nGpus; ++j) ncclCommAbort(comms[j]);
    goto exit;
  } else {
    TESTCHECK(checkCommsState(comms, nGpus, ncclInProgress, ncclSuccess));
    for (int j = 0; j < nGpus; ++j)
      NCCLCHECK(ncclCommDestroy(comms[j]));
  }

exit:
  return testSuccess;
}

static testResult_t distributeFTShrinkTest(struct threadArgs* args) {
  ncclComm_t* comms = args->comms[0];
  testResult_t ret = testSuccess;
  int nGpus = args->nGpus;

  int totalGpus = args->nProcs * args->nThreads * nGpus;
  int sDev = args->localRank * args->nThreads * nGpus + args->thread * nGpus;
  void** sendbuffs = args->sendbuffs[0];
  void** recvbuffs = args->recvbuffs[0];
  cudaStream_t* streams = args->streams;
  int badIdx = 1;

  // not enought gpus to run the test, return
  if (totalGpus <= 1) return testSuccess;

  // First time: Initialize all communicators
  ncclConfig_t initConfig = NCCL_CONFIG_INITIALIZER;
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, *args->ncclId, rank, &initConfig));
  }
  NCCLCHECK(ncclGroupEnd());

  // Allocate arrays for split communicators
  ncclComm_t* splitComms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGpus);
  ncclComm_t* shrinkComms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGpus);

  // Step 1: Shrink existing communicators for all ranks except the badIdx one
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    ncclConfig_t shrinkConfig = NCCL_CONFIG_INITIALIZER;
    NCCLCHECK(ncclCommShrink(comms[j], &badIdx, 1, &shrinkComms[j], &shrinkConfig, NCCL_SHRINK_ABORT));
  }
  NCCLCHECK(ncclGroupEnd());
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) {
      ncclCommAbort(comms[j]);
      comms[j] = NULL;
    } else {
      NCCLCHECK(ncclCommDestroy(comms[j]));
      comms[j] = shrinkComms[j];
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // Step 2: Create split communicators to emulate PyTorch usage
  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    NCCLCHECK(ncclCommSplit(comms[j], 1, rank, &splitComms[j], &splitConfig));
  }
  NCCLCHECK(ncclGroupEnd());

  // Step 3: Do allreduce using split comms
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    int h_val = rank + 1;
    CUDACHECK(cudaMemcpy(sendbuffs[j], &h_val, sizeof(int), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    NCCLCHECK(ncclAllReduce(sendbuffs[j], recvbuffs[j], 1, ncclInt32, ncclSum, splitComms[j], streams[j]));
  }
  NCCLCHECK(ncclGroupEnd());

  // Step 4: Synchronize streams and verify results
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    CUDACHECK(cudaStreamSynchronize(streams[j]));
  }

  // Calculate expected result
  int expected = totalGpus * (totalGpus + 1) / 2 - (badIdx + 1);

  // Verify results
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    int h_result = 0;
    CUDACHECK(cudaMemcpy(&h_result, recvbuffs[j], sizeof(int), cudaMemcpyDeviceToHost));
    if (h_result != expected) {
      printf("Distributed FT: Allreduce result on device %d is %d, expected %d\n",
             j, h_result, expected);
      ret = testNcclError;
    }
  }

  // Step 5: Cleanup split comms
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    NCCLCHECK(ncclCommDestroy(splitComms[j]));
  }
  NCCLCHECK(ncclGroupEnd());

  // Final cleanup of base comms
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    NCCLCHECK(ncclCommDestroy(comms[j]));
  }
  NCCLCHECK(ncclGroupEnd());

  free(splitComms);
  return ret;
}

testResult_t commAbortHangTest(struct threadArgs* args) {
#if CUDART_VERSION >= 12020
  int driverVersion = 0;
  CUDACHECK(cudaDriverGetVersion(&driverVersion));
  if (driverVersion < 12060) return testSuccess;
  int nGpus = args->nGpus;
  int sDev = args->localRank * args->nThreads * args->nGpus + args->thread * args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * args->nGpus;
  ncclComm_t* comms = args->comms[0];
  void** sendbuffs = args->sendbuffs[0];
  void** recvbuffs = args->recvbuffs[0];
  cudaStream_t* streams = args->streams;
  size_t count;
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommInitRank(&comms[j], totalGpus, *args->ncclId, rank));
  }
  NCCLCHECK(ncclGroupEnd());
  count = size / totalGpus;
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    for (int k = 0; k < totalGpus; ++k) {
      NCCLCHECK(ncclSend(((char*)sendbuffs[j]) + k * count, count, ncclChar, k, comms[j], streams[j]));
      NCCLCHECK(ncclRecv(((char*)recvbuffs[j]) + k * count, count, ncclChar, k, comms[j], streams[j]));
    }
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  if (args->proc == 0) {
    for (int i = 0; i < nGpus; i++) NCCLCHECK(ncclCommAbort(comms[i]));
#ifdef MPI_SUPPORT
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  else {
#ifdef MPI_SUPPORT
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    for (int i = 0; i < nGpus; i++) NCCLCHECK(ncclCommAbort(comms[i]));
  }
#endif /* CUDART_VERSION >= 12020 */
  return testSuccess;
}
testResult_t faultToleranceTests(int nThreads, int nGpus, int ncclProc, int ncclProcs, int localRank, const char* ft_list) {
  struct testThread* threads;

  ncclUniqueId ncclId;
  void** sendbuffs;
  void** recvbuffs;
  void** hostbuffs;
  cudaStream_t* streams;
  ncclComm_t* comms;
  int localnGpus = nThreads * nGpus;
  int sDev = localRank * localnGpus;
  int totalGpus = ncclProcs * nThreads;
  int testEnable[FT_TEST_NUM];
  const char* testName[FT_TEST_NUM] = {"init", "allreduce", "alltoall", "finalize", "split", "shrink", "abort"};
  threadFunc_t testFuncs[FT_TEST_NUM] = {distributeFTInitTest, distributeFTAllreduceTest, distributeFTAlltoAllTest, distributeFTFinalizeTest, distributeFTCommSplitTest, distributeFTShrinkTest, commAbortHangTest};

  if (ft_list != NULL) {
    /* users only set a subset of ft tests. */
    char* tmp_list;
    char* token;
    int len = strlen(ft_list);

    tmp_list = (char*)malloc(len + 1);
    memcpy(tmp_list, ft_list, len + 1);
    memset(testEnable, 0, sizeof(int) * FT_TEST_NUM);
    token = strtok(tmp_list, ",/:|");
    while (token != NULL) {
      if (strcmp(token, "init") == 0) {
        testEnable[FT_TEST_INIT] = 1;
      } else if (strcmp(token, "allreduce") == 0) {
        testEnable[FT_TEST_ALLREDUCE] = 1;
      } else if (strcmp(token, "alltoall") == 0) {
        testEnable[FT_TEST_ALLTOALL] = 1;
      } else if (strcmp(token, "finalize") == 0) {
        testEnable[FT_TEST_FINALIZE] = 1;
      } else if (strcmp(token, "split") == 0) {
        testEnable[FT_TEST_SPLIT] = 1;
      } else if (strcmp(token, "shrink") == 0) {
        testEnable[FT_TEST_SHRINK] = 1;
      } else if (strcmp(token, "abort") == 0) {
        testEnable[FT_TEST_ABORT] = 1;
      } else if (strcmp(token, "all") == 0) {
        for (int i = 0; i < FT_TEST_NUM; ++i) testEnable[i] = 1;
      } else {
        printf("Incorrect fault tolerance test: %s\n", token);
        return testInternalError;
      }
      token = strtok(NULL, ",/:|");
    }
    free(tmp_list);
  } else {
    for (int i = 0; i < FT_TEST_NUM; ++i) testEnable[i] = 1;
  }

  is_main_thread = (ncclProc == 0) ? 1 : 0;
  comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * localnGpus);
  sendbuffs = (void**)malloc(sizeof(void*) * localnGpus);
  recvbuffs = (void**)malloc(sizeof(void*) * localnGpus);
  hostbuffs = (void**)malloc(sizeof(void*) * localnGpus);
  streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * localnGpus);
  threads = (struct testThread*)malloc(nThreads * sizeof(struct testThread));
  TESTCHECK(initBufferStream(sendbuffs, recvbuffs, hostbuffs, streams, sDev, localnGpus, size));

  for (int t = nThreads - 1; t >= 0; t--) {
    threads[t].args.nThreads = nThreads;
    threads[t].args.nGpus = nGpus;
    threads[t].args.proc = ncclProc;
    threads[t].args.nProcs = ncclProcs;
    threads[t].args.localRank = localRank;
    threads[t].args.thread = t;
    threads[t].args.sendbuffs = (void***)malloc(sizeof(void**));
    threads[t].args.recvbuffs = (void***)malloc(sizeof(void**));
    threads[t].args.comms = (ncclComm_t**)malloc(sizeof(ncclComm_t*));
    threads[t].args.sendbuffs[0] = sendbuffs + t * nGpus;
    threads[t].args.recvbuffs[0] = recvbuffs + t * nGpus;
    threads[t].args.hostbuffs = hostbuffs + t * nGpus;
    threads[t].args.comms[0] = comms + t * nGpus;
    threads[t].args.streams = streams + t * nGpus;
  }

  for (int ft_id = 0; ft_id < FT_TEST_NUM; ++ft_id) {
    if (testEnable[ft_id]) {
      for (int i = 0; i < localnGpus; ++i) {
        int dev = sDev + i;
        CUDACHECK(cudaSetDevice(dev));
        memset(hostbuffs[i], 0, size);
        CUDACHECK(cudaMemset(sendbuffs[i], 1, size));
        CUDACHECK(cudaMemset(recvbuffs[i], 0, size));
      }
      PRINT("\t================ Test fault tolerance for NCCL %s ================\n", testName[ft_id]);
      for (int i = 0; i <= NUM_SLEEP_CASES; ++i) {
        if (ncclProc == 0) {
          NCCLCHECK(ncclGetUniqueId(&ncclId));
        }
#ifdef MPI_SUPPORT
        MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
        for (int t = nThreads - 1; t >= 0; t--) {
          threads[t].args.nIds = 1;
          threads[t].args.ncclId = &ncclId;
          threads[t].args.sleepId = i;
          threads[t].func = testFuncs[ft_id];
          if (t)
            TESTCHECK(threadLaunch(threads + t));
          else
            TESTCHECK(threads[t].func(&threads[t].args));
        }

        for (int t = nThreads - 1; t > 0; t--) {
          pthread_join(threads[t].thread, NULL);
          TESTCHECK(threads[t].ret);
        }
#ifdef MPI_SUPPORT
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        if (i < NUM_SLEEP_CASES) {
          PRINT("Distributed FT:\tSleep %dus, abort %d communicators in %s\t[SUCCESS]\n", sleepTimes[i], totalGpus, testName[ft_id]);
        }
      }
      PRINT("Test fault tolerance for NCCL %s\t[SUCCESS]\n\n", testName[ft_id]);
    }
  }

  for (int t = nThreads - 1; t >= 0; t--) {
    free(threads[t].args.sendbuffs);
    free(threads[t].args.recvbuffs);
    free(threads[t].args.comms);
  }
  finalizeBufferStream(sendbuffs, recvbuffs, hostbuffs, streams, localnGpus);
  free(sendbuffs);
  free(recvbuffs);
  free(hostbuffs);
  free(streams);
  free(threads);
  return testSuccess;
}

#else /* NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0) */

testResult_t faultToleranceTests(int nThreads, int nGpus, int ncclProc, int ncclProcs, int localRank, const char* ft_list) {
  return testSuccess;
}

#endif
#undef PRINT
