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
  FT_TEST_NUM = 4,
};

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,14,0)
#define NUM_SLEEP_CASES 8
int sleepTimes[NUM_SLEEP_CASES] = { 100, 1000, 10000, 100000, 1000000, 2000000, 4000000, 8000000 }; /* sleep in us */

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
  void** sendbuffs = args->sendbuffs;
  void** recvbuffs = args->recvbuffs;
  cudaStream_t* streams = args->streams;
  char** hostbuffs = (char**)args->hostbuffs;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;
  ncclComm_t* comms = args->comms;
  size_t size = args->nbytes;
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
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, args->ncclId, rank, &config));
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

static testResult_t distributeFTAllreduceTest(struct threadArgs* args) {
  void** sendbuffs = args->sendbuffs;
  void** recvbuffs = args->recvbuffs;
  cudaStream_t* streams = args->streams;
  char** hostbuffs = (char**)args->hostbuffs;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;
  ncclComm_t* comms = args->comms;
  size_t size = args->nbytes;
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
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, args->ncclId, rank, &config));
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
  void** sendbuffs = args->sendbuffs;
  void** recvbuffs = args->recvbuffs;
  cudaStream_t* streams = args->streams;
  char** hostbuffs = (char**)args->hostbuffs;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;
  ncclComm_t* comms = args->comms;
  size_t size = args->nbytes;
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
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, args->ncclId, rank, &config));
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), comms, nGpus);

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
  void** sendbuffs = args->sendbuffs;
  void** recvbuffs = args->recvbuffs;
  cudaStream_t* streams = args->streams;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;
  ncclComm_t* comms = args->comms;
  size_t size = args->nbytes;
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
    NCCLCHECK(ncclCommInitRankConfig(&comms[j], totalGpus, args->ncclId, rank, &config));
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

testResult_t faultToleranceTests(int nThreads, int nGpus, int ncclProc, int ncclProcs, int localRank, const char* ft_list) {
  struct testThread* threads;
  size_t size = 32 * 1024 * 1024;
  ncclUniqueId ncclId;
  void** sendbuffs;
  void** recvbuffs;
  void** hostbuffs;
  cudaStream_t* streams;
  ncclComm_t* comms;
  int localnGpus = nThreads * nGpus;
  int sDev = localRank * localnGpus;
  int totalGpus = ncclProcs * nThreads;
  int test_list[FT_TEST_NUM];

  if (ft_list != NULL) {
    /* users only set a subset of ft tests. */
    char* tmp_list;
    char* token;
    int len = strlen(ft_list);

    printf("ft_list len %d, %s\n", len, ft_list);
    tmp_list = (char*)malloc(len + 1);
    memcpy(tmp_list, ft_list, len + 1);
    memset(test_list, 0, sizeof(int) * FT_TEST_NUM);
    token = strtok(tmp_list, ",/:|");
    while (token != NULL) {
      printf("token %s\n", token);
      if (strcmp(token, "init") == 0) {
        test_list[FT_TEST_INIT] = 1;
      } else if (strcmp(token, "allreduce") == 0) {
        test_list[FT_TEST_ALLREDUCE] = 1;
      } else if (strcmp(token, "alltoall") == 0) {
        test_list[FT_TEST_ALLTOALL] = 1;
      } else if (strcmp(token, "finalize") == 0) {
        test_list[FT_TEST_FINALIZE] = 1;
      } else if (strcmp(token, "all") == 0) {
        for (int i = 0; i < FT_TEST_NUM; ++i) test_list[i] = 1;
      } else {
        printf("Incorrect fault tolerance test: %s\n", token);
        return testInternalError;
      }
      token = strtok(NULL, ",/:|");
    }
    free(tmp_list);
  } else {
    for (int i = 0; i < FT_TEST_NUM; ++i) test_list[i] = 1;
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
    threads[t].args.nbytes = size;
    threads[t].args.sendbuffs = sendbuffs + t * nGpus;
    threads[t].args.recvbuffs = recvbuffs + t * nGpus;
    threads[t].args.hostbuffs = hostbuffs + t * nGpus;
    threads[t].args.comms = comms + t * nGpus;
    threads[t].args.streams = streams + t * nGpus;
  }

  if (test_list[FT_TEST_INIT]) {
    for (int i = 0; i < localnGpus; ++i) {
      int dev = sDev + i;
      CUDACHECK(cudaSetDevice(dev));
      memset(hostbuffs[i], 0, size);
      CUDACHECK(cudaMemset(sendbuffs[i], 1, size));
      CUDACHECK(cudaMemset(recvbuffs[i], 0, size));
    }
    PRINT("\t================ Test fault tolerance for NCCL init ================\n");
    for (int i = 0; i <= NUM_SLEEP_CASES; ++i) {
      if (ncclProc == 0) {
        NCCLCHECK(ncclGetUniqueId(&ncclId));
      }
#ifdef MPI_SUPPORT
      MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
      for (int t = nThreads - 1; t >= 0; t--) {
        threads[t].args.ncclId = ncclId;
        threads[t].args.sleepId = i;
        threads[t].func = distributeFTInitTest;
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
        PRINT("Distributed FT:\tSleep %dus, abort %d communicators at ncclCommInitRankConfig\t[SUCCESS]\n", sleepTimes[i], totalGpus);
      } else {
        PRINT("Distributed FT:\tInitialize %d communicators\t[SUCCESS]\n", totalGpus);
      }
    }
    PRINT("Test fault tolerance for NCCL init\t[SUCCESS]\n\n");
  }
  
  if (test_list[FT_TEST_ALLREDUCE]) {
    for (int i = 0; i < localnGpus; ++i) {
      int dev = sDev + i;
      CUDACHECK(cudaSetDevice(dev));
      memset(hostbuffs[i], 0, size);
      CUDACHECK(cudaMemset(sendbuffs[i], 1, size));
      CUDACHECK(cudaMemset(recvbuffs[i], 0, size));
    }
    PRINT("\t================ Test fault tolerance for NCCL allreduce ================\n");
    for (int i = 0; i <= NUM_SLEEP_CASES; ++i) {
      if (ncclProc == 0) {
        NCCLCHECK(ncclGetUniqueId(&ncclId));
      }
#ifdef MPI_SUPPORT
      MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
      for (int t = nThreads - 1; t >= 0; t--) {
        threads[t].args.ncclId = ncclId;
        threads[t].args.sleepId = i;
        threads[t].func = distributeFTAllreduceTest;
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
        PRINT("Distributed FT:\tSleep %dus, abort %d communicators at ncclAllReduce\t[SUCCESS]\n", sleepTimes[i], totalGpus);
      } else {
        PRINT("Distributed FT:\t %d number of ncclAllReduce issue\t[SUCCESS]\n", totalGpus);
      }
    }
    PRINT("Test fault tolerance for NCCL allreduce\t[SUCCESS]\n\n");
  }
  
  if (test_list[FT_TEST_ALLTOALL]) {
    for (int i = 0; i < localnGpus; ++i) {
      int dev = sDev + i;
      CUDACHECK(cudaSetDevice(dev));
      memset(hostbuffs[i], 0, size);
      CUDACHECK(cudaMemset(sendbuffs[i], 1, size));
      CUDACHECK(cudaMemset(recvbuffs[i], 0, size));
    }
    PRINT("\t================ Test fault tolerance for NCCL alltoall ================\n");
    for (int i = 0; i <= NUM_SLEEP_CASES; ++i) {
      if (ncclProc == 0) {
        NCCLCHECK(ncclGetUniqueId(&ncclId));
      }
#ifdef MPI_SUPPORT
      MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
      for (int t = nThreads - 1; t >= 0; t--) {
        threads[t].args.ncclId = ncclId;
        threads[t].args.sleepId = i;
        threads[t].func = distributeFTAlltoAllTest;
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
        PRINT("Distributed FT:\tSleep %dus, abort %d communicators at NCCL alltoall\t[SUCCESS]\n", sleepTimes[i], totalGpus);
      } else {
        PRINT("Distributed FT:\t %d number of NCCL alltoall issue\t[SUCCESS]\n", totalGpus);
      }
    }
    PRINT("Test fault tolerance for NCCL alltoall\t[SUCCESS]\n\n");
  }

  if (test_list[FT_TEST_FINALIZE]) {
    for (int i = 0; i < localnGpus; ++i) {
      int dev = sDev + i;
      CUDACHECK(cudaSetDevice(dev));
      memset(hostbuffs[i], 0, size);
      CUDACHECK(cudaMemset(sendbuffs[i], 1, size));
      CUDACHECK(cudaMemset(recvbuffs[i], 0, size));
    }
    PRINT("\t================ Test fault tolerance for NCCL finalize ================\n");
    for (int i = 0; i <= NUM_SLEEP_CASES; ++i) {
      if (ncclProc == 0) {
        NCCLCHECK(ncclGetUniqueId(&ncclId));
      }
#ifdef MPI_SUPPORT
      MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
      for (int t = nThreads - 1; t >= 0; t--) {
        threads[t].args.ncclId = ncclId;
        threads[t].args.sleepId = i;
        threads[t].func = distributeFTFinalizeTest;
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
        PRINT("Distributed FT:\tSleep %dus, abort %d communicators at ncclCommFinalize\t[SUCCESS]\n", sleepTimes[i], totalGpus);
      } else {
        PRINT("Distributed FT:\tDestroy %d communicators\t[SUCCESS]\n", totalGpus);
      }
    }
    PRINT("Test fault tolerance for NCCL finalize\t[SUCCESS]\n\n");
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