#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include <assert.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    char hostname[1024];                            \
    gethostname(hostname, 1024);                    \
    printf("%s: Cuda failure %s:%d '%s'\n",         \
         hostname,                                  \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess && r != ncclInProgress) {     \
    char hostname[1024];                            \
    gethostname(hostname, 1024);                    \
    printf("%s: NCCL failure %s:%d '%s'\n",         \
         hostname,                                  \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NUM_SLEEP_CASES 6
int sleepTimes[NUM_SLEEP_CASES] = {100, 1000, 10000, 100000, 1000000, 4000000}; /* sleep in us */

static int checkCommsState(ncclComm_t* comms, int nVis, ncclResult_t stateStart, ncclResult_t stateExpect) {
  ncclResult_t state;
  int errors = 0, complete;
  do {
    complete = 1;
    for (int j = 0; j < nVis; ++j) {
      NCCLCHECK(ncclCommGetAsyncError(comms[j], &state));
      if (state == stateStart) {
        complete = 0;
        break;
      }
      usleep(10);
    }
  } while (!complete);

  for (int j = 0; j < nVis; ++j) {
    NCCLCHECK(ncclCommGetAsyncError(comms[j], &state));
    if (state != stateExpect) {
      errors++;
      printf("FT-NCCL:\tCheck comms[%d] state, query state %s != expected state %s  [FAIL]\n", j, ncclGetErrorString(state), ncclGetErrorString(stateExpect));
    }
  }

  return errors;
}

static void initBufferStream(void **sendbuffDptr, void **recvbuffDptr, char **bufHostPtr, cudaStream_t* sa, int nVis, int size) {
  for (int i = 0; i < nVis; ++i) {
    CUDACHECK(cudaSetDevice(i));
    bufHostPtr[i] = (char*) malloc(size);
    assert(bufHostPtr[i] != NULL);
    memset(bufHostPtr[i], 0, size);
    NCCLCHECK(ncclMemAlloc((void**)&sendbuffDptr[i], size));
    NCCLCHECK(ncclMemAlloc((void**)&recvbuffDptr[i], size));
    CUDACHECK(cudaMemset(sendbuffDptr[i], 1, size));
    CUDACHECK(cudaMemset(recvbuffDptr[i], 0, size));
    CUDACHECK(cudaStreamCreate(&sa[i]));
  }
  
  CUDACHECK(cudaStreamSynchronize(0));
  return;
}

static void finalizeBufferStream(void **sendbuffDptr, void **recvbuffDptr, char **bufHostPtr, cudaStream_t* sa, int nVis) {
  for (int i = 0; i < nVis; ++i) {
    CUDACHECK(cudaSetDevice(i));
    ncclMemFree(sendbuffDptr[i]);
    ncclMemFree(recvbuffDptr[i]);
    free(bufHostPtr[i]);
    cudaStreamDestroy(sa[i]);
  }
  return;
}

int faultToleranceInitTest(ncclComm_t* comms, int nVis, int size) {
  int errors = 0;
  void** sendbuffDptr;
  void** recvbuffDptr;
  char** bufHostPtr;
  cudaStream_t* sa;
  ncclUniqueId id;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;

  config.blocking = 0;
  sendbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  recvbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  bufHostPtr = (char**) malloc(sizeof(char*) * nVis);
  sa = (cudaStream_t*) malloc(sizeof(cudaStream_t) * nVis);

  initBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis, size);

  //initializing NCCL
  for (int i = 0; i < NUM_SLEEP_CASES; ++i) {
    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaSetDevice(j));
      NCCLCHECK(ncclCommInitRankConfig(&comms[j], nVis, id, j, &config));
    }
    ret = ncclGroupEnd();

    usleep(sleepTimes[i]);
    if (i != NUM_SLEEP_CASES - 1) {
      for (int j = 0; j < nVis; ++j) ncclCommAbort(comms[j]);
      printf("FT-NCCL:\tSleep %dus, group ret %s, abort %d communicators at ncclCommInitAll\t[SUCCESS]\n", sleepTimes[i], ncclGetErrorString(ret), nVis);
      continue;
    } else {
      errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
      if (errors) goto exit;
      printf("FT-NCCL:\tSleep %dus, group ret %s, initialize %d communicators\t[SUCCESS]\n", sleepTimes[i], ncclGetErrorString(ret), nVis);
    }

    //communicating using NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclAllReduce((const void*)sendbuffDptr[j], (void*)recvbuffDptr[j], size, ncclInt8, ncclSum, comms[j], sa[j]));
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    //completing NCCL operation by synchronizing on the CUDA stream
    for (int j = 0; j < nVis; ++j)
      CUDACHECK(cudaStreamSynchronize(sa[j]));

    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaSetDevice(j));
      CUDACHECK(cudaMemcpy(bufHostPtr[j], recvbuffDptr[j], size, cudaMemcpyDeviceToHost));
      for (int k = 0; k < size; ++k) {
        if ((int) bufHostPtr[j][k] != nVis) {
          errors++;
          printf("FT-NCCL:\tncclAllReduce wrong result %d at %d buffer location [%d] != expected %d\n", bufHostPtr[j][k], j, k, nVis);
        }
      }
    }
    
    //finalizing NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclCommFinalize(comms[j]));
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclCommDestroy(comms[j]));
  }

exit:
  if (!errors) 
    printf("Test fault tolerance for NCCL init\t[SUCCESS]\n\n");
  else
    printf("Test fault tolerance for NCCL init, errors %d\t[FAIL]\n\n", errors);
  finalizeBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis);
  free(bufHostPtr);
  free(sendbuffDptr);
  free(recvbuffDptr);
  free(sa);
  return errors;
}

int faultToleranceAllreduceTest(ncclComm_t* comms, int nVis, int size) {
  int errors = 0;
  void** sendbuffDptr;
  void** recvbuffDptr;
  char** bufHostPtr;
  cudaStream_t* sa;
  ncclUniqueId id;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;

  config.blocking = 0;
  sendbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  recvbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  bufHostPtr = (char**) malloc(sizeof(char*) * nVis);
  sa = (cudaStream_t*) malloc(sizeof(cudaStream_t) * nVis);

  initBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis, size);

  //initializing NCCL
  for (int i = 0; i < NUM_SLEEP_CASES; ++i) {
    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaSetDevice(j));
      memset(bufHostPtr[j], 0, size);
      CUDACHECK(cudaMemset(sendbuffDptr[j], 1, size));
      CUDACHECK(cudaMemset(recvbuffDptr[j], 0, size));
    }
    CUDACHECK(cudaStreamSynchronize(0));
    
    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaSetDevice(j));
      NCCLCHECK(ncclCommInitRankConfig(&comms[j], nVis, id, j, &config));
    }
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    //communicating using NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclAllReduce((const void*)sendbuffDptr[j], (void*)recvbuffDptr[j], size, ncclInt8, ncclSum, comms[j], sa[j]));
    /* try one more time */
    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclAllReduce((const void*)sendbuffDptr[j], (void*)recvbuffDptr[j], size, ncclInt8, ncclSum, comms[j], sa[j]));
    ret = ncclGroupEnd();

    usleep(sleepTimes[i]);
    if (i != NUM_SLEEP_CASES - 1) {
      for (int j = 0; j < nVis; ++j) ncclCommAbort(comms[j]);
      printf("FT-NCCL:\tSleep %dus, group ret %s, abort %d communicators at ncclAllReduce, \t[SUCCESS]\n", sleepTimes[i], ncclGetErrorString(ret), nVis);
      continue;
    } else {
      errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
      if (errors) goto exit;
      printf("FT-NCCL:\tSleep %dus, group ret %s, ncclAllReduce issue\t[SUCCESS]\n", sleepTimes[i], ncclGetErrorString(ret));
    }

    //completing NCCL operation by synchronizing on the CUDA stream
    for (int j = 0; j < nVis; ++j)
      CUDACHECK(cudaStreamSynchronize(sa[j]));

    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaSetDevice(j));
      CUDACHECK(cudaMemcpy(bufHostPtr[j], recvbuffDptr[j], size, cudaMemcpyDeviceToHost));
      for (int k = 0; k < size; ++k) {
        if ((int) bufHostPtr[j][k] != nVis) {
          errors++;
          printf("FT-NCCL:\tncclAllReduce wrong result %d at %d buffer location [%d] != expected %d\n", bufHostPtr[j][k], j, k, nVis);
        }
      }
    }
    
    //finalizing NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclCommFinalize(comms[j]));
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclCommDestroy(comms[j]));
  }

exit:
  if (!errors) 
    printf("Test fault tolerance for NCCL allreduce\t[SUCCESS]\n\n");
  else
    printf("Test fault tolerance for NCCL allreduce, errors %d\t[FAIL]\n\n", errors);
  finalizeBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis);
  free(bufHostPtr);
  free(sendbuffDptr);
  free(recvbuffDptr);
  free(sa);
  return errors;
}

int faultToleranceAlltoAllTest(ncclComm_t* comms, int nVis, int size) {
  int errors = 0;
  void** sendbuffDptr;
  void** recvbuffDptr;
  char** bufHostPtr;
  cudaStream_t* sa;
  ncclUniqueId id;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclResult_t ret;
  size_t count;

  config.blocking = 0;
  sendbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  recvbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  bufHostPtr = (char**) malloc(sizeof(char*) * nVis);
  sa = (cudaStream_t*) malloc(sizeof(cudaStream_t) * nVis);

  initBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis, size);

  //initializing NCCL
  for (int i = 0; i < NUM_SLEEP_CASES; ++i) {
    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaSetDevice(j));
      NCCLCHECK(ncclCommInitRankConfig(&comms[j], nVis, id, j, &config));
    }
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    count = size / nVis;
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j) {
      for (int k = 0; k < nVis; ++k) {
        NCCLCHECK(ncclSend(((char*)sendbuffDptr[j]) + k * count, count, ncclChar, k, comms[j], sa[j]));
        NCCLCHECK(ncclRecv(((char*)recvbuffDptr[j]) + k * count, count, ncclChar, k, comms[j], sa[j]));
      }
    }
    ret = ncclGroupEnd();

    usleep(sleepTimes[i]);
    if (i != NUM_SLEEP_CASES - 1) {
      for (int j = 0; j < nVis; ++j) ncclCommAbort(comms[j]);
      printf("FT-NCCL:\tSleep %dus, group ret %s, abort %d communicators at NCCL alltoall, \t[SUCCESS]\n", sleepTimes[i], ncclGetErrorString(ret), nVis);
      continue;
    } else {
      errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
      if (errors) goto exit;
      printf("FT-NCCL:\tSleep %dus, group ret %s, NCCL alltoall issue\t[SUCCESS]\n", sleepTimes[i], ncclGetErrorString(ret));
    }

    //completing NCCL operation by synchronizing on the CUDA stream
    for (int j = 0; j < nVis; ++j)
      CUDACHECK(cudaStreamSynchronize(sa[j]));

    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaSetDevice(j));
      CUDACHECK(cudaMemcpy(bufHostPtr[j], recvbuffDptr[j], size, cudaMemcpyDeviceToHost));
      for (int k = 0; k < count * nVis; ++k) {
        if ((int) bufHostPtr[j][k] != 1) {
          errors++;
          printf("FT-NCCL:\tsendrecv wrong result %d at %d buffer location [%d] != expected %d\n", bufHostPtr[j][k], j, k, 1);
        }
      }
    }
    
    //finalizing NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclCommFinalize(comms[j]));
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclCommDestroy(comms[j]));
  }

exit:
  if (!errors) 
    printf("Test fault tolerance for NCCL alltoall\t[SUCCESS]\n\n");
  else
    printf("Test fault tolerance for NCCL alltoall, errors %d\t[FAIL]\n\n", errors);
  finalizeBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis);
  free(bufHostPtr);
  free(sendbuffDptr);
  free(recvbuffDptr);
  free(sa);
  return errors;
}

int faultToleranceFinalizeTest(ncclComm_t* comms, int nVis, int size) {
  int errors = 0;
  void** sendbuffDptr;
  void** recvbuffDptr;
  char** bufHostPtr;
  cudaStream_t* sa;
  ncclUniqueId id;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  config.blocking = 0;
  sendbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  recvbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  bufHostPtr = (char**) malloc(sizeof(char*) * nVis);
  sa = (cudaStream_t*) malloc(sizeof(cudaStream_t) * nVis);

  initBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis, size);
  
  //initializing NCCL
  for (int i = 0; i < NUM_SLEEP_CASES; ++i) {
    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaSetDevice(j));
      memset(bufHostPtr[j], 0, size);
      CUDACHECK(cudaMemset(sendbuffDptr[j], 1, size));
      CUDACHECK(cudaMemset(recvbuffDptr[j], 0, size));
    }
    CUDACHECK(cudaStreamSynchronize(0));

    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j) {
        CUDACHECK(cudaSetDevice(j));
        NCCLCHECK(ncclCommInitRankConfig(&comms[j], nVis, id, j, &config));
    }
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    //communicating using NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclAllReduce((const void*)sendbuffDptr[j], (void*)recvbuffDptr[j], size, ncclInt8, ncclSum, comms[j], sa[j]));
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    //completing NCCL operation by synchronizing on the CUDA stream
    for (int j = 0; j < nVis; ++j)
      CUDACHECK(cudaStreamSynchronize(sa[j]));

    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaMemcpy(bufHostPtr[j], recvbuffDptr[j], size, cudaMemcpyDeviceToHost));
      for (int k = 0; k < size; ++k) {
        if ((int)bufHostPtr[j][k] != nVis) {
          errors++;
          printf("FT-NCCL:\tncclAllReduce wrong result %d at %d buffer location [%d] != expected %d\n", bufHostPtr[j][k], j, k, nVis);
        }
      }
    }

    //finalizing NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclCommFinalize(comms[j]));
    NCCLCHECK(ncclGroupEnd());
    usleep(sleepTimes[i]);
    
    if (i != NUM_SLEEP_CASES - 1) {
      for (int j = 0; j < nVis; ++j) ncclCommAbort(comms[j]);
      printf("FT-NCCL:\tSleep %dus, abort %d communicators at ncclCommFinalize\t[SUCCESS]\n", sleepTimes[i], nVis);
      continue;
    } else {
      errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
      if (errors) goto exit;
      for (int j = 0; j < nVis; ++j)
        NCCLCHECK(ncclCommDestroy(comms[j]));
      printf("FT-NCCL:\tSleep %dus, destroy %d communicators\t[SUCCESS]\n", sleepTimes[i], nVis);
    }
  }

exit:
  if (!errors) 
    printf("Test fault tolerance for NCCL finalize\t[SUCCESS]\n\n");
  else
    printf("Test fault tolerance for NCCL finalize, errors %d\t[FAIL]\n\n", errors);
  finalizeBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis);
  free(bufHostPtr);
  free(sendbuffDptr);
  free(recvbuffDptr);
  free(sa);
  return errors;
}

int main(int argc, char* argv[])
{
  int size = 32 * 1024 * 1024;
  int nVis, errors = 0;
  ncclComm_t* comms;
  
  CUDACHECK(cudaGetDeviceCount(&nVis));
  comms = (ncclComm_t*)calloc(sizeof(ncclComm_t), nVis);

  printf("\t================ Test fault tolerance for NCCL init ================\n");
  errors += faultToleranceInitTest(comms, nVis, size);
  printf("\t================ Test fault tolerance for NCCL allreduce ================\n");
  errors += faultToleranceAllreduceTest(comms, nVis, size);
  printf("\t================ Test fault tolerance for NCCL alltoall ================\n");
  errors += faultToleranceAlltoAllTest(comms, nVis, size);
  printf("\t================ Test fault tolerance for NCCL finalize ================\n");
  errors += faultToleranceFinalizeTest(comms, nVis, size);
  
  free(comms);
  printf("[Summary] Single node NCCL fault tolerance test completes, %d errors.\n\n", errors);
  return errors;
}