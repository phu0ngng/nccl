#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include <assert.h>
#include <vector>
#include <mutex>
#include "common.h"

enum {
  FT_TEST_INIT = 0,
  FT_TEST_ALLREDUCE = 1,
  FT_TEST_ALLTOALL = 2,
  FT_TEST_FINALIZE = 3,
  FT_TEST_SPLIT = 4,
  FT_TEST_SHRINK = 5,
  FT_TEST_ABORT = 6,
  FT_TEST_REVOKE = 7,
  FT_TEST_REVOKE_SHRINK = 8,
  FT_TEST_REVOKE_SPLIT = 9,
  FT_TEST_GROW = 10,
  FT_TEST_REVOKE_SHRINK_GROW = 11,
  FT_TEST_NUM = 12,
};

#define PRINT if (is_main_thread) printf

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
#define NUM_SLEEP_CASES 5
int sleepTimes[NUM_SLEEP_CASES] = { 10, 100, 10000, 1000000, 2000000}; /* sleep in us */
size_t size = 32 * 1024 * 1024;

// Revoke test can take a long time, so only run first and last sleep cases
static inline int ft_should_run_revoke_sleep(int sleepId) {
  return (sleepId == 0 || sleepId == NUM_SLEEP_CASES);
}

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
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
#ifdef MPI_SUPPORT
    int flag;
    extern std::mutex mpiLock;
    {
      std::lock_guard<std::mutex> lock(mpiLock);
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    }
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
  {
    ncclResult_t ret = ncclGroupEnd();
    assert(ret == ncclSuccess || ret == ncclInProgress);
  }

  if (sleepId < NUM_SLEEP_CASES) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleepTimes[sleepId]));
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
  ncclComm_t* comms = args->comms[0];
  int nGpus = args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * args->nGpus;
  int sDev = args->localRank * args->nThreads * args->nGpus + args->thread * args->nGpus;
  int sleepId = args->sleepId;
  std::vector<ncclComm_t> splitComms(nGpus);

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
  {
    ncclResult_t ret = ncclGroupEnd();
    assert(ret == ncclSuccess || ret == ncclInProgress);
  }

  if (sleepId < NUM_SLEEP_CASES) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleepTimes[sleepId]));
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
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), splitComms.data(), nGpus);

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
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), splitComms.data(), nGpus);

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
  {
    ncclResult_t ret = ncclGroupEnd();
    assert(ret == ncclSuccess || ret == ncclInProgress);
  }

  if (sleepId < NUM_SLEEP_CASES) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleepTimes[sleepId]));
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
  {
    ncclResult_t ret = ncclGroupEnd();
    assert(ret == ncclSuccess || ret == ncclInProgress);
  }

  if (sleepId < NUM_SLEEP_CASES) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleepTimes[sleepId]));
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
  {
    ncclResult_t ret = ncclGroupEnd();
    assert(ret == ncclSuccess || ret == ncclInProgress);
  }

  if (sleepId < NUM_SLEEP_CASES) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleepTimes[sleepId]));
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

static testResult_t distributeFTRevokeTest(struct threadArgs* args) {
  void** sendbuffs = args->sendbuffs[0];
  void** recvbuffs = args->recvbuffs[0];
  cudaStream_t* streams = args->streams;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclComm_t* comms = args->comms[0];
  int nGpus = args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * args->nGpus;
  int sDev = args->localRank * args->nThreads * args->nGpus + args->thread * args->nGpus;
  int sleepId = args->sleepId;

  if (!ft_should_run_revoke_sleep(sleepId)) return testSuccess;

  config.blocking = 0;
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, *args->ncclId, rank, &config));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  // Launch a collective to have ongoing work
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j)
    NCCLCHECK(ncclAllReduce((const void*)sendbuffs[j], (void*)recvbuffs[j], size, ncclInt8, ncclProd, comms[j], streams[j]));
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);
  if (sleepId < NUM_SLEEP_CASES) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleepTimes[sleepId]));
  }
  // Revoke staged across nodes
#ifdef MPI_SUPPORT
  if (args->proc == 0) {
    for (int j = 0; j < nGpus; ++j) {
      ncclResult_t rv = ncclCommRevoke(comms[j], NCCL_REVOKE_DEFAULT);
      assert(rv == ncclSuccess || rv == ncclInProgress);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int j = 0; j < nGpus; ++j) {
      ncclResult_t rv = ncclCommRevoke(comms[j], NCCL_REVOKE_DEFAULT);
      assert(rv == ncclSuccess || rv == ncclInProgress);
    }
  }
#else
  for (int j = 0; j < nGpus; ++j) {
    ncclResult_t rv = ncclCommRevoke(comms[j], NCCL_REVOKE_DEFAULT);
    assert(rv == ncclSuccess || rv == ncclInProgress);
  }
#endif
  TESTCHECK(checkCommsState(comms, nGpus, ncclInProgress, ncclSuccess));
  // Destroy after revoke completes
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    NCCLCHECK(ncclCommDestroy(comms[j]));
  }
  NCCLCHECK(ncclGroupEnd());
  return testSuccess;
}

// Demonstrate: revoke cancels ongoing work and we can then shrink the communicator set
static testResult_t distributeFTRevokeThenShrinkTest(struct threadArgs* args) {
  void** sendbuffs = args->sendbuffs[0];
  void** recvbuffs = args->recvbuffs[0];
  cudaStream_t* streams = args->streams;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclComm_t* comms = args->comms[0];
  int nGpus = args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * args->nGpus;
  int sDev = args->localRank * args->nThreads * args->nGpus + args->thread * args->nGpus;
  int sleepId = args->sleepId;
  int badIdx = 1;

  if (!ft_should_run_revoke_sleep(sleepId)) return testSuccess;
  if (totalGpus <= 1) return testSuccess;

  config.blocking = 0;
  // Init base comms
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, *args->ncclId, rank, &config));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  // Launch some collective work
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j)
    NCCLCHECK(ncclAllReduce((const void*)sendbuffs[j], (void*)recvbuffs[j], size, ncclInt8, ncclProd, comms[j], streams[j]));
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  if (sleepId < NUM_SLEEP_CASES) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleepTimes[sleepId]));
  }

  // Revoke staged across nodes
#ifdef MPI_SUPPORT
  if (args->proc == 0) {
    for (int j = 0; j < nGpus; ++j) {
      ncclResult_t rv = ncclCommRevoke(comms[j], NCCL_REVOKE_DEFAULT);
      assert(rv == ncclSuccess || rv == ncclInProgress);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int j = 0; j < nGpus; ++j) {
      ncclResult_t rv = ncclCommRevoke(comms[j], NCCL_REVOKE_DEFAULT);
      assert(rv == ncclSuccess || rv == ncclInProgress);
    }
  }
#else
  for (int j = 0; j < nGpus; ++j) {
    ncclResult_t rv = ncclCommRevoke(comms[j], NCCL_REVOKE_DEFAULT);
    assert(rv == ncclSuccess || rv == ncclInProgress);
  }
#endif
  // Wait revoke completion across ranks, then Shrink should succeed
  TESTCHECK(checkCommsState(comms, nGpus, ncclInProgress, ncclSuccess));

  // Now shrink out one rank (badIdx), then cleanly destroy others
  ncclComm_t* shrinkComms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGpus);
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    ncclConfig_t shrinkConfig = NCCL_CONFIG_INITIALIZER;
    shrinkConfig.blocking = 0;
    NCCLCHECK(ncclCommShrink(comms[j], &badIdx, 1, &shrinkComms[j], &shrinkConfig, NCCL_SHRINK_DEFAULT));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) {
      NCCLCHECK(ncclCommDestroy(comms[j]));
      comms[j] = NULL;
    } else {
      NCCLCHECK(ncclCommDestroy(comms[j]));
      comms[j] = shrinkComms[j];
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // Optionally run a quick collective on the shrunk comms to ensure they are usable
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    NCCLCHECK(ncclAllReduce((const void*)sendbuffs[j], (void*)recvbuffs[j], 1024, ncclInt8, ncclSum, comms[j], streams[j]));
  }
  // Build a filtered list of valid communicators for the batch wait (exclude badIdx)
  int numValidComms = 0;
  std::vector<ncclComm_t> validComms(nGpus);
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    validComms[numValidComms++] = comms[j];
  }
  // Ensure the grouped allreduce completes successfully before synchronizing streams
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), validComms.data(), numValidComms);
  for (int j = 0; j < nGpus; ++j) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    CUDACHECK(cudaStreamSynchronize(streams[j]));
  }

  // Cleanup
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;
    if (comms[j]) { NCCLCHECK(ncclCommDestroy(comms[j])); }
  }
  NCCLCHECK(ncclGroupEnd());
  free(shrinkComms);
  return testSuccess;
}

// Demonstrate: revoke cancels ongoing work and then we can split the communicator
static testResult_t distributeFTRevokeThenSplitTest(struct threadArgs* args) {
  void** sendbuffs = args->sendbuffs[0];
  void** recvbuffs = args->recvbuffs[0];
  cudaStream_t* streams = args->streams;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclComm_t* comms = args->comms[0];
  int nGpus = args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * args->nGpus;
  int sDev = args->localRank * args->nThreads * args->nGpus + args->thread * args->nGpus;
  int sleepId = args->sleepId;

  if (!ft_should_run_revoke_sleep(sleepId)) return testSuccess;

  config.blocking = 0;
  // Init base comms
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, *args->ncclId, rank, &config));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  // Launch some collective work
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    NCCLCHECK(ncclAllReduce((const void*)sendbuffs[j], (void*)recvbuffs[j], 1024, ncclInt8, ncclSum, comms[j], streams[j]));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  if (sleepId < NUM_SLEEP_CASES) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleepTimes[sleepId]));
  }

  // Revoke staged across nodes
#ifdef MPI_SUPPORT
  if (args->proc == 0) {
    for (int j = 0; j < nGpus; ++j) {
      ncclResult_t rv = ncclCommRevoke(comms[j], NCCL_REVOKE_DEFAULT);
      assert(rv == ncclSuccess || rv == ncclInProgress);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int j = 0; j < nGpus; ++j) {
      ncclResult_t rv = ncclCommRevoke(comms[j], NCCL_REVOKE_DEFAULT);
      assert(rv == ncclSuccess || rv == ncclInProgress);
    }
  }
#else
  for (int j = 0; j < nGpus; ++j) {
    ncclResult_t rv = ncclCommRevoke(comms[j], NCCL_REVOKE_DEFAULT);
    assert(rv == ncclSuccess || rv == ncclInProgress);
  }
#endif
  TESTCHECK(checkCommsState(comms, nGpus, ncclInProgress, ncclSuccess));

  // Split into two groups (even/odd ranks)
  ncclComm_t* splitComms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGpus);
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    int color = rank & 1;
    NCCLCHECK(ncclCommSplit(comms[j], color, rank, &splitComms[j], NULL));
  }
  /* For non-blocking communicators, group end may return ncclInProgress.
   * Wait on the parent comms to complete the group job (which produces splitComms).
   */
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  // Quick collective on split comms (only ranks where splitComms != NULL)
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    if (splitComms[j]) {
      NCCLCHECK(ncclAllReduce((const void*)sendbuffs[j], (void*)recvbuffs[j], 1, ncclInt32, ncclSum, splitComms[j], streams[j]));
    }
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), splitComms, nGpus);
  for (int j = 0; j < nGpus; ++j) CUDACHECK(cudaStreamSynchronize(streams[j]));

  // Cleanup
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    if (splitComms[j]) { NCCLCHECK(ncclCommDestroy(splitComms[j])); }
    NCCLCHECK(ncclCommDestroy(comms[j]));
  }
  NCCLCHECK(ncclGroupEnd());
  free(splitComms);
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

static testResult_t distributeFTGrowTest(struct threadArgs* args) {
  ncclComm_t* comms = args->comms[0];
  testResult_t ret = testSuccess;
  int nGpus = args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * nGpus;
  int sDev = args->localRank * args->nThreads * nGpus + args->thread * nGpus;
  void** sendbuffs = args->sendbuffs[0];
  void** recvbuffs = args->recvbuffs[0];
  cudaStream_t* streams = args->streams;

  // Declare variables used in goto cleanup paths at the top
  ncclResult_t groupEndResult = ncclSuccess;
  int waitCount = 0;
  ncclResult_t asyncErr = ncclSuccess;

  // Need at least 2 GPUs for grow
  if (totalGpus < 2) return testSuccess;

  // Start with half the GPUs
  int oldN = totalGpus / 2;
  if (oldN == 0) oldN = 1;

  // Check if buffers exist (for MPSG mode: new ranks might not have buffers)
  bool hasBuffers = (sendbuffs && sendbuffs[0] && recvbuffs && recvbuffs[0] && streams && streams[0]);
  int myRank = args->proc * args->nThreads * nGpus + args->thread * nGpus;

  // Allocate grown comms early (before any goto)
  ncclComm_t* grownComms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGpus);
  if (grownComms == NULL) return testNcclError;
  memset(grownComms, 0, sizeof(ncclComm_t) * nGpus);

  // Initialize first half of ranks
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank < oldN) {
      CUDACHECK(cudaSetDevice(dev));
      NCCLCHECK(ncclCommInitRank(&comms[j], oldN, *args->ncclId, rank));
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // Get grow UniqueId from rank 0
  ncclUniqueId growId;
  memset(&growId, 0, sizeof(ncclUniqueId));

  if (myRank == 0) {
    // Only rank 0 (which is in existing ranks) calls this
    if (comms[0] == NULL) {
      ret = testNcclError;
      goto cleanup;
    }
    NCCLCHECK(ncclCommGetUniqueId(comms[0], &growId));
  }

#ifdef MPI_SUPPORT
  extern std::mutex mpiLock;
  {
    std::lock_guard<std::mutex> lock(mpiLock);
    MPI_Bcast(&growId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  }
#endif

  // Grow: existing and new ranks
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    CUDACHECK(cudaSetDevice(dev));

    if (rank < oldN) {
      // Existing rank
      const ncclUniqueId* uid = (rank == 0) ? &growId : NULL;
      NCCLCHECK(ncclCommGrow(comms[j], totalGpus, uid, -1, &grownComms[j], NULL));
    } else {
      // New rank
      NCCLCHECK(ncclCommGrow(NULL, totalGpus, &growId, rank, &grownComms[j], NULL));
    }
  }

  groupEndResult = ncclGroupEnd();

  if (groupEndResult != ncclSuccess && groupEndResult != ncclInProgress) {
    ret = testNcclError;
    goto cleanup;
  }

  // Wait for async completion
  waitCount = 0;
  do {
    if (grownComms[0] == NULL) {
      ret = testNcclError;
      goto cleanup;
    }

    ncclResult_t checkRes = ncclCommGetAsyncError(grownComms[0], &asyncErr);
    if (checkRes != ncclSuccess) {
      ret = testNcclError;
      goto cleanup;
    }

    if (asyncErr == ncclInProgress) {
      waitCount++;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  } while (asyncErr == ncclInProgress && waitCount < 100000);

  if (asyncErr != ncclSuccess) {
    ret = testNcclError;
    goto cleanup;
  }

  // Verify grown comms were created
  for (int j = 0; j < nGpus; j++) {
    if (grownComms[j] == NULL) {
      ret = testNcclError;
      goto cleanup;
    }
  }

  // Run AllReduce on grown comm (only if buffers exist)

  if (hasBuffers) {
    for (int j = 0; j < nGpus; j++) {
      int dev = sDev + j;
      int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
      CUDACHECK(cudaSetDevice(dev));
      int h_val = rank + 1;
      CUDACHECK(cudaMemcpy(sendbuffs[j], &h_val, sizeof(int), cudaMemcpyHostToDevice));
    }

    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nGpus; j++) {
      if (grownComms[j]) {
        NCCLCHECK(ncclAllReduce(sendbuffs[j], recvbuffs[j], 1, ncclInt32, ncclSum, grownComms[j], streams[j]));
      }
    }
    NCCLCHECK(ncclGroupEnd());

    for (int j = 0; j < nGpus; j++) {
      if (grownComms[j]) {
        CUDACHECK(cudaStreamSynchronize(streams[j]));
      }
    }

    // Verify AllReduce result
    int expected = totalGpus * (totalGpus + 1) / 2;
    for (int j = 0; j < nGpus; j++) {
      int h_result = 0;
      CUDACHECK(cudaMemcpy(&h_result, recvbuffs[j], sizeof(int), cudaMemcpyDeviceToHost));
      if (h_result != expected) {
        printf("Distributed FT Grow: AllReduce result is %d, expected %d\n", h_result, expected);
        ret = testNcclError;
      }
    }
  } else {
    // New ranks without buffers - grow succeeded, that's enough
    PRINT("Distributed FT Grow: New rank joined successfully (no AllReduce verification)\n");
  }

cleanup:
  // Cleanup
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank < oldN && comms[j]) {
      NCCLCHECK(ncclCommDestroy(comms[j]));
      comms[j] = NULL;
    }
    if (grownComms[j]) {
      NCCLCHECK(ncclCommDestroy(grownComms[j]));
    }
  }
  NCCLCHECK(ncclGroupEnd());

  free(grownComms);
  return ret;
}

// Enhanced FT test: shrink -> grow (fault recovery cycle)
// Flow: 16 ranks → shrink to 15 ranks (removing rank 1) → grow from 15 to 16
static testResult_t distributeFTShrinkGrowTest(struct threadArgs* args) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclComm_t* comms = args->comms[0];
  int nGpus = args->nGpus;
  int totalGpus = args->nProcs * args->nThreads * args->nGpus;
  int sDev = args->localRank * args->nThreads * args->nGpus + args->thread * args->nGpus;
  int sleepId = args->sleepId;
  int badIdx = 1;  // Rank 1 will be removed

  if (!ft_should_run_revoke_sleep(sleepId)) return testSuccess;
  if (totalGpus < 2) return testSuccess;  // Need at least 2 ranks for this test

  config.blocking = 0;

  // Step 1: Init all communicators
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; ++j) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, *args->ncclId, rank, &config));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

  if (sleepId < NUM_SLEEP_CASES) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleepTimes[sleepId]));
  }

  // Step 2: Shrink to remove the bad rank
  ncclComm_t* shrinkComms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGpus);
  memset(shrinkComms, 0, sizeof(ncclComm_t) * nGpus);

  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank == badIdx) continue;  // Bad rank doesn't participate

    int dev = sDev + j;
    CUDACHECK(cudaSetDevice(dev));
    NCCLCHECK(ncclCommShrink(comms[j], &badIdx, 1, &shrinkComms[j], &config, NCCL_SHRINK_ABORT));
  }

  // Build filtered list of original comms for batch wait (exclude badIdx)
  int numValidComms = 0;
  std::vector<ncclComm_t> validComms(nGpus);
  for (int j = 0; j < nGpus; j++) {
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    if (rank != badIdx) {
      validComms[numValidComms++] = comms[j];
    }
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), validComms.data(), numValidComms);

  // Destroy old comms (abort the bad one)
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

  // Step 4: Grow back to original size (using the bad rank's slot with a new rank)
  // Get grow UniqueId from rank 0 of shrunken comm
  ncclUniqueId growId;
  testResult_t ret = testSuccess;
  if (args->proc == 0 && args->thread == 0) {
    NCCLCHECK(ncclCommGetUniqueId(comms[0], &growId));
  }
#ifdef MPI_SUPPORT
  extern std::mutex mpiLock;
  {
    std::lock_guard<std::mutex> lock(mpiLock);
    MPI_Bcast(&growId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  }
#endif

  // Allocate grown comms
  ncclComm_t* grownComms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nGpus);
  memset(grownComms, 0, sizeof(ncclComm_t) * nGpus);

  // Grow: healthy ranks grow, bad rank joins as new
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    int dev = sDev + j;
    int rank = args->proc * args->nThreads * nGpus + args->thread * nGpus + j;
    CUDACHECK(cudaSetDevice(dev));

    if (rank == badIdx) {
      // Bad rank joins as new rank
      NCCLCHECK(ncclCommGrow(NULL, totalGpus, &growId, totalGpus - 1, &grownComms[j], &config));
    } else {
      // Healthy ranks grow from shrunken comm
      const ncclUniqueId* uid = (rank == 0) ? &growId : NULL;
      NCCLCHECK(ncclCommGrow(comms[j], totalGpus, uid, -1, &grownComms[j], &config));
    }
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), grownComms, nGpus);

  // Cleanup
  NCCLCHECK(ncclGroupStart());
  for (int j = 0; j < nGpus; j++) {
    // Destroy shrunken comm if it exists (badIdx has NULL)
    if (shrinkComms[j]) {
      NCCLCHECK(ncclCommDestroy(shrinkComms[j]));
    }
    // Destroy grown comm (all ranks have these)
    NCCLCHECK(ncclCommDestroy(grownComms[j]));
  }
  NCCLCHECK(ncclGroupEnd());

  free(shrinkComms);
  free(grownComms);
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
  const char* testName[FT_TEST_NUM] = {"init", "allreduce", "alltoall", "finalize", "split", "shrink", "abort", "revoke", "revoke_shrink", "revoke_split", "grow", "shrink_grow"};
  threadFunc_t testFuncs[FT_TEST_NUM] = {distributeFTInitTest, distributeFTAllreduceTest, distributeFTAlltoAllTest, distributeFTFinalizeTest, distributeFTCommSplitTest, distributeFTShrinkTest, commAbortHangTest, distributeFTRevokeTest, distributeFTRevokeThenShrinkTest, distributeFTRevokeThenSplitTest, distributeFTGrowTest, distributeFTShrinkGrowTest};

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
      } else if (strcmp(token, "revoke") == 0) {
        testEnable[FT_TEST_REVOKE] = 1;
      } else if (strcmp(token, "revoke_shrink") == 0) {
        testEnable[FT_TEST_REVOKE_SHRINK] = 1;
      } else if (strcmp(token, "revoke_split") == 0) {
        testEnable[FT_TEST_REVOKE_SPLIT] = 1;
      } else if (strcmp(token, "grow") == 0) {
        testEnable[FT_TEST_GROW] = 1;
      } else if (strcmp(token, "shrink_grow") == 0) {
        testEnable[FT_TEST_REVOKE_SHRINK_GROW] = 1;
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
          if (threads[t].thread.joinable()) threads[t].thread.join();
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
