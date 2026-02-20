#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include <assert.h>
#include <pthread.h>
#include <getopt.h>
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

static size_t wordSize(ncclDataType_t type) {
  switch(type) {
    case ncclChar:
#if NCCL_MAJOR >= 2
    //case ncclInt8:
    case ncclUint8:
#endif
#if HAVE_FP8
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
#endif
      return 1;
    case ncclHalf:
#if HAVE_BF16
    case ncclBfloat16:
#endif
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


#define NUM_SLEEP_CASES 6
int sleepTimes[NUM_SLEEP_CASES] = {0, 100, 1000, 10000, 100000, 1000000}; /* sleep in us */

// Buffer registration options
#define DISABLE_REGISTER 0
#define LOCAL_REGISTER 1
#define SYMMETRIC_REGISTER 2
#define SYMMETRIC_REGISTER_SEND 3
#define SYMMETRIC_REGISTER_RECV 4
static int registerMode = 0;
static ncclDataType_t dataType = ncclInt8;

static ncclDataType_t parseDataType(const char* str) {
  if (strcmp(str, "int8") == 0 || strcmp(str, "char") == 0) return ncclInt8;
  if (strcmp(str, "uint8") == 0) return ncclUint8;
  if (strcmp(str, "int32") == 0 || strcmp(str, "int") == 0) return ncclInt32;
  if (strcmp(str, "uint32") == 0) return ncclUint32;
  if (strcmp(str, "int64") == 0) return ncclInt64;
  if (strcmp(str, "uint64") == 0) return ncclUint64;
  if (strcmp(str, "half") == 0 || strcmp(str, "float16") == 0) return ncclHalf;
  if (strcmp(str, "float") == 0 || strcmp(str, "float32") == 0) return ncclFloat;
  if (strcmp(str, "double") == 0 || strcmp(str, "float64") == 0) return ncclDouble;
#if HAVE_BF16
  if (strcmp(str, "bfloat16") == 0 || strcmp(str, "bf16") == 0) return ncclBfloat16;
#endif
  return ncclNumTypes; // invalid
}

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
    CUDACHECK(cudaStreamCreate(&sa[i]));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaMemsetAsync(sendbuffDptr[i], 1, size, sa[i]));
    CUDACHECK(cudaMemsetAsync(recvbuffDptr[i], 0, size, sa[i]));
  }

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

static void registerBuffers(ncclComm_t* comms, void **sendbuffDptr, void **recvbuffDptr,
                            void **sendRegHandles, void **recvRegHandles, int nVis, int size) {
  if (!registerMode) return;

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nVis; ++i) {
    CUDACHECK(cudaSetDevice(i));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
    if (registerMode == SYMMETRIC_REGISTER_SEND) {
      NCCLCHECK(ncclCommWindowRegister(comms[i], sendbuffDptr[i], size, (ncclWindow_t*)&sendRegHandles[i], NCCL_WIN_COLL_SYMMETRIC));
    } else if (registerMode == SYMMETRIC_REGISTER_RECV) {
      NCCLCHECK(ncclCommWindowRegister(comms[i], recvbuffDptr[i], size, (ncclWindow_t*)&recvRegHandles[i], NCCL_WIN_COLL_SYMMETRIC));
    } else if (registerMode == SYMMETRIC_REGISTER) {
      NCCLCHECK(ncclCommWindowRegister(comms[i], sendbuffDptr[i], size, (ncclWindow_t*)&sendRegHandles[i], NCCL_WIN_COLL_SYMMETRIC));
      NCCLCHECK(ncclCommWindowRegister(comms[i], recvbuffDptr[i], size, (ncclWindow_t*)&recvRegHandles[i], NCCL_WIN_COLL_SYMMETRIC));
    } else {
      if (registerMode == LOCAL_REGISTER) NCCLCHECK(ncclCommRegister(comms[i], sendbuffDptr[i], size, &sendRegHandles[i]));
      if (registerMode == LOCAL_REGISTER) NCCLCHECK(ncclCommRegister(comms[i], recvbuffDptr[i], size, &recvRegHandles[i]));
    }
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
    if (registerMode == LOCAL_REGISTER) NCCLCHECK(ncclCommRegister(comms[i], sendbuffDptr[i], size, &sendRegHandles[i]));
    if (registerMode == LOCAL_REGISTER) NCCLCHECK(ncclCommRegister(comms[i], recvbuffDptr[i], size, &recvRegHandles[i]));
#endif
  }
  NCCLCHECK(ncclGroupEnd());
}

static void deregisterBuffers(ncclComm_t* comms, void **sendRegHandles, void **recvRegHandles, int nVis) {
  if (!registerMode) return;

  NCCLCHECK(ncclGroupStart());

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
  for (int i = 0; i < nVis; ++i) {
    CUDACHECK(cudaSetDevice(i));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,27,0)
    if (registerMode == SYMMETRIC_REGISTER_SEND) {
      NCCLCHECK(ncclCommWindowDeregister(comms[i], (ncclWindow_t)sendRegHandles[i]));
    } else if (registerMode == SYMMETRIC_REGISTER_RECV) {
      NCCLCHECK(ncclCommWindowDeregister(comms[i], (ncclWindow_t)recvRegHandles[i]));
    } else if (registerMode == SYMMETRIC_REGISTER) {
      NCCLCHECK(ncclCommWindowDeregister(comms[i], (ncclWindow_t)sendRegHandles[i]));
      NCCLCHECK(ncclCommWindowDeregister(comms[i], (ncclWindow_t)recvRegHandles[i]));
    } else
#endif
    {
      if (registerMode) NCCLCHECK(ncclCommDeregister(comms[i], sendRegHandles[i]));
      if (registerMode) NCCLCHECK(ncclCommDeregister(comms[i], recvRegHandles[i]));
    }
  }
#endif
  NCCLCHECK(ncclGroupEnd());

}

int faultToleranceInitTest(ncclComm_t* comms, int nVis, int size) {
  int errors = 0;
  int nGpusParticipate = 0;
  void** sendbuffDptr;
  void** recvbuffDptr;
  char** bufHostPtr;
  cudaStream_t* sa;
  ncclUniqueId id;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  void** sendRegHandles = NULL;
  void** recvRegHandles = NULL;

  config.blocking = 0;
  sendbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  recvbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  bufHostPtr = (char**) malloc(sizeof(char*) * nVis);
  sa = (cudaStream_t*) malloc(sizeof(cudaStream_t) * nVis);
  if (registerMode) {
    sendRegHandles = (void**) calloc(nVis, sizeof(void*));
    recvRegHandles = (void**) calloc(nVis, sizeof(void*));
  }
  int count = size / wordSize(dataType);

  initBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis, size);

  //initializing NCCL
  for (int i = 0; i < NUM_SLEEP_CASES; ++i) {
    nGpusParticipate = (i == NUM_SLEEP_CASES-1) ? (nVis) : (nVis-1);
    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nGpusParticipate; ++j) {
      CUDACHECK(cudaSetDevice(j));
      NCCLCHECK(ncclCommInitRankConfig(&comms[j], nVis, id, j, &config));
    }
    NCCLCHECK(ncclGroupEnd());

    usleep(sleepTimes[i]);
    if (nVis != nGpusParticipate) {
      for (int j = 0; j < nGpusParticipate; ++j) NCCLCHECK(ncclCommAbort(comms[j]));
      printf("FT-NCCL:\tSleep %8dus, abort %d/%d ranks during ncclCommInitRankConfig\t[SUCCESS]\n", sleepTimes[i], nGpusParticipate, nVis);
      continue;
    } else {
      errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
      if (errors) goto exit;
      printf("FT-NCCL:\tSleep %8dus, initialize %d ranks\t\t[SUCCESS]\n", sleepTimes[i], nVis);
    }

    // Register buffers after comms are created
    registerBuffers(comms, sendbuffDptr, recvbuffDptr, sendRegHandles, recvRegHandles, nVis, size);
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    //communicating using NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclAllReduce((const void*)sendbuffDptr[j], (void*)recvbuffDptr[j], count, dataType, ncclSum, comms[j], sa[j]));
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    //completing NCCL operation by synchronizing on the CUDA stream
    for (int j = 0; j < nVis; ++j)
      CUDACHECK(cudaStreamSynchronize(sa[j]));

    //finalizing NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j) {
      deregisterBuffers(&comms[j], &sendRegHandles[j], &recvRegHandles[j], 1);
      NCCLCHECK(ncclCommFinalize(comms[j]));
    }
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
  free(sendRegHandles);
  free(recvRegHandles);
  return errors;
}

int faultToleranceAllreduceTest(ncclComm_t* comms, int nVis, int size) {
  int errors = 0;
  int nGpusParticipate = 0;
  void** sendbuffDptr;
  void** recvbuffDptr;
  char** bufHostPtr;
  cudaStream_t* sa;
  ncclUniqueId id;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  void** sendRegHandles = NULL;
  void** recvRegHandles = NULL;

  config.blocking = 0;
  sendbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  recvbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  bufHostPtr = (char**) malloc(sizeof(char*) * nVis);
  sa = (cudaStream_t*) malloc(sizeof(cudaStream_t) * nVis);
  if (registerMode) {
    sendRegHandles = (void**) calloc(nVis, sizeof(void*));
    recvRegHandles = (void**) calloc(nVis, sizeof(void*));
  }
  int count = size / wordSize(dataType);

  initBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis, size);

  //initializing NCCL
  for (int i = 0; i < NUM_SLEEP_CASES; ++i) {
    nGpusParticipate = (i == NUM_SLEEP_CASES-1) ? (nVis) : (nVis-1);
    for (int j = 0; j < nVis; ++j) {
      memset(bufHostPtr[j], 0, size);
      CUDACHECK(cudaMemsetAsync(sendbuffDptr[j], 1, size, sa[j]));
      CUDACHECK(cudaMemsetAsync(recvbuffDptr[j], 0, size, sa[j]));
    }

    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaSetDevice(j));
      NCCLCHECK(ncclCommInitRankConfig(&comms[j], nVis, id, j, &config));
    }
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    // Register buffers after comms are created
    registerBuffers(comms, sendbuffDptr, recvbuffDptr, sendRegHandles, recvRegHandles, nVis, size);
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    //communicating using NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nGpusParticipate; ++j)
      NCCLCHECK(ncclAllReduce((const void*)sendbuffDptr[j], (void*)recvbuffDptr[j], count, dataType, ncclSum, comms[j], sa[j]));
    NCCLCHECK(ncclGroupEnd());

    usleep(sleepTimes[i]);
    if (nVis != nGpusParticipate) {
      for (int j = 0; j < nVis; ++j) ncclCommAbort(comms[j]);
      printf("FT-NCCL:\tSleep %8dus, abort %d/%d ranks during ncclAllReduce\t[SUCCESS]\n", sleepTimes[i], nGpusParticipate, nVis);
      continue;
    } else {
      errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
      if (errors) goto exit;
      printf("FT-NCCL:\tSleep %8dus, ncclAllReduce issue\t\t[SUCCESS]\n", sleepTimes[i]);
    }

    //completing NCCL operation by synchronizing on the CUDA stream
    for (int j = 0; j < nVis; ++j)
      CUDACHECK(cudaStreamSynchronize(sa[j]));

    // Deregister buffers before finalizing
    deregisterBuffers(comms, sendRegHandles, recvRegHandles, nVis);

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
  free(sendRegHandles);
  free(recvRegHandles);
  return errors;
}

int faultToleranceAlltoAllTest(ncclComm_t* comms, int nVis, int size) {
  int errors = 0;
  int nGpusParticipate = 0;
  void** sendbuffDptr;
  void** recvbuffDptr;
  char** bufHostPtr;
  cudaStream_t* sa;
  ncclUniqueId id;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  size_t count;
  void** sendRegHandles = NULL;
  void** recvRegHandles = NULL;

  config.blocking = 0;
  sendbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  recvbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  bufHostPtr = (char**) malloc(sizeof(char*) * nVis);
  sa = (cudaStream_t*) malloc(sizeof(cudaStream_t) * nVis);
  if (registerMode) {
    sendRegHandles = (void**) calloc(nVis, sizeof(void*));
    recvRegHandles = (void**) calloc(nVis, sizeof(void*));
  }

  initBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis, size);

  //initializing NCCL
  for (int i = 0; i < NUM_SLEEP_CASES; ++i) {
    nGpusParticipate = (i == NUM_SLEEP_CASES-1) ? (nVis) : (nVis-1);
    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaSetDevice(j));
      NCCLCHECK(ncclCommInitRankConfig(&comms[j], nVis, id, j, &config));
    }
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    // Register buffers after comms are created
    registerBuffers(comms, sendbuffDptr, recvbuffDptr, sendRegHandles, recvRegHandles, nVis, size);
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    count = size / nVis;
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j) {
      for (int k = 0; k < nVis; ++k) {
        if (k < nGpusParticipate) {
          NCCLCHECK(ncclSend(((char*)sendbuffDptr[j]) + k * count, count, ncclChar, k, comms[j], sa[j]));
        }
        NCCLCHECK(ncclRecv(((char*)recvbuffDptr[j]) + k * count, count, ncclChar, k, comms[j], sa[j]));
      }
    }
    NCCLCHECK(ncclGroupEnd());

    usleep(sleepTimes[i]);
    if (nVis != nGpusParticipate) {
      for (int j = 0; j < nVis; ++j) NCCLCHECK(ncclCommAbort(comms[j]));
      printf("FT-NCCL:\tSleep %8dus, abort %d/%d ranks during NCCL alltoall\t[SUCCESS]\n", sleepTimes[i], nGpusParticipate, nVis);
      continue;
    } else {
      errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
      if (errors) goto exit;
      printf("FT-NCCL:\tSleep %8dus, NCCL alltoall issue\t\t[SUCCESS]\n", sleepTimes[i]);
    }

    //completing NCCL operation by synchronizing on the CUDA stream
    for (int j = 0; j < nVis; ++j)
      CUDACHECK(cudaStreamSynchronize(sa[j]));

    // Deregister buffers before finalizing
    deregisterBuffers(comms, sendRegHandles, recvRegHandles, nVis);

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
  free(sendRegHandles);
  free(recvRegHandles);
  return errors;
}

int faultToleranceFinalizeTest(ncclComm_t* comms, int nVis, int size) {
  int errors = 0;
  int nGpusParticipate = 0;
  void** sendbuffDptr;
  void** recvbuffDptr;
  char** bufHostPtr;
  cudaStream_t* sa;
  ncclUniqueId id;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  void** sendRegHandles = NULL;
  void** recvRegHandles = NULL;

  config.blocking = 0;
  sendbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  recvbuffDptr = (void**) malloc(sizeof(void*) * nVis);
  bufHostPtr = (char**) malloc(sizeof(char*) * nVis);
  sa = (cudaStream_t*) malloc(sizeof(cudaStream_t) * nVis);
  if (registerMode) {
    sendRegHandles = (void**) calloc(nVis, sizeof(void*));
    recvRegHandles = (void**) calloc(nVis, sizeof(void*));
  }
  int count = size / wordSize(dataType);

  initBufferStream(sendbuffDptr, recvbuffDptr, bufHostPtr, sa, nVis, size);

  //initializing NCCL
  for (int i = 0; i < NUM_SLEEP_CASES; ++i) {
    nGpusParticipate = (i == NUM_SLEEP_CASES-1) ? (nVis) : (nVis-1);
    for (int j = 0; j < nVis; ++j) {
      memset(bufHostPtr[j], 0, size);
      CUDACHECK(cudaMemsetAsync(sendbuffDptr[j], 1, size, sa[j]));
      CUDACHECK(cudaMemsetAsync(recvbuffDptr[j], 0, size, sa[j]));
    }

    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j) {
      CUDACHECK(cudaSetDevice(j));
      NCCLCHECK(ncclCommInitRankConfig(&comms[j], nVis, id, j, &config));
    }
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    // Register buffers after comms are created
    registerBuffers(comms, sendbuffDptr, recvbuffDptr, sendRegHandles, recvRegHandles, nVis, size);
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    //communicating using NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nVis; ++j)
      NCCLCHECK(ncclAllReduce((const void*)sendbuffDptr[j], (void*)recvbuffDptr[j], count, dataType, ncclSum, comms[j], sa[j]));
    NCCLCHECK(ncclGroupEnd());
    errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
    if (errors) goto exit;

    //completing NCCL operation by synchronizing on the CUDA stream
    for (int j = 0; j < nVis; ++j)
      CUDACHECK(cudaStreamSynchronize(sa[j]));
  
    /* participating ranks follow orderly shutdown */
    deregisterBuffers(comms, sendRegHandles, recvRegHandles, nVis);

    //finalizing NCCL
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < nGpusParticipate; ++j) NCCLCHECK(ncclCommFinalize(comms[j]));
    NCCLCHECK(ncclGroupEnd());
    usleep(sleepTimes[i]);

    if (nVis != nGpusParticipate) {
      /* participating ranks are stuck in finalize, all must abort. */
      for (int j = 0; j < nVis; ++j) NCCLCHECK(ncclCommAbort(comms[j]));
      printf("FT-NCCL:\tSleep %8dus, abort %d/%d ranks during ncclCommFinalize\t\t[SUCCESS]\n", sleepTimes[i], nGpusParticipate, nVis);
      continue;
    } else {
      /* all ranks finalized, now destroy */
      errors += checkCommsState(comms, nVis, ncclInProgress, ncclSuccess);
      if (errors) goto exit;
      for (int j = 0; j < nVis; ++j)
        NCCLCHECK(ncclCommDestroy(comms[j]));
      printf("FT-NCCL:\tSleep %8dus, destroy %d communicators\t\t\t[SUCCESS]\n", sleepTimes[i], nVis);
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
  free(sendRegHandles);
  free(recvRegHandles);
  return errors;
}

struct threadArgs {
  int rank;
  ncclComm_t comm;
  pthread_barrier_t* barrier;
  int* errors;
};

static void* threadFunc(void* args_) {
  struct threadArgs* args = (struct threadArgs*)args_;
  ncclComm_t comm = args->comm;
  pthread_barrier_t* barrier = args->barrier;
  int* errors = args->errors;
  int errorLocal = 0;

  CUDACHECK(cudaSetDevice(args->rank));
  pthread_barrier_wait(barrier);
  errorLocal += checkCommsState(&comm, 1, ncclInProgress, ncclSuccess);

  if (!errorLocal) {
    NCCLCHECK(ncclCommDestroy(comm));
  } else {
    ncclCommAbort(comm);
  }
  __atomic_fetch_add(errors, errorLocal, __ATOMIC_RELAXED);
  return NULL;
}

int faultToleranceMultiThreadTest(ncclComm_t* comms, int nVis, int size) {
  int errors = 0;
  ncclUniqueId id;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  pthread_barrier_t barrier;
  pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * nVis);
  struct threadArgs* args = (struct threadArgs*)malloc(sizeof(struct threadArgs) * nVis);

  pthread_barrier_init(&barrier, NULL, nVis);

  NCCLCHECK(ncclGetUniqueId(&id));
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    CUDACHECK(cudaSetDevice(i));
    NCCLCHECK(ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
  }
  NCCLCHECK(ncclGroupEnd());
  for (int i = 0; i < nVis; i++) {
    args[i].rank = i;
    args[i].comm = comms[i];
    args[i].barrier = &barrier;
    args[i].errors = &errors;
    pthread_create(&threads[i], NULL, threadFunc, (void*)&args[i]);
  }
  for (int i = 0; i < nVis; i++) {
    pthread_join(threads[i], NULL);
  }

  if (!errors)
    printf("Test fault tolerance for NCCL multi-thread\t[SUCCESS]\n\n");
  else
    printf("Test fault tolerance for NCCL multi-thread, errors %d\t[FAIL]\n\n", errors);
  pthread_barrier_destroy(&barrier);
  free(threads);
  free(args);
  return errors;
}

int main(int argc, char* argv[])
{
  int size = 32 * 1024 * 1024;
  int nVis, errors = 0;
  ncclComm_t* comms;
  setlinebuf(stdout);

  // Parse command line arguments
  static struct option long_options[] = {
    {"register", required_argument, 0, 'R'},
    {"buffer-size", required_argument, 0, 'b'},
    {"datatype", required_argument, 0, 'd'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
  };

  int c;
  while ((c = getopt_long(argc, argv, "R:b:d:h", long_options, NULL)) != -1) {
    switch (c) {
      case 'd':
        dataType = parseDataType(optarg);
        if (dataType == ncclNumTypes) {
          printf("Invalid datatype '%s'. Using default int8\n", optarg);
          dataType = ncclInt8;
        }
        break;
      case 'b':
        size = (int)strtol(optarg, NULL, 0);
        if (size <= 0) {
          printf("Invalid buffer size %d. Using default 32MB\n", size);
          size = 32 * 1024 * 1024;
        }
        break;
      case 'R':
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,19,0)
        registerMode = (int)strtol(optarg, NULL, 0);
        if (((registerMode == SYMMETRIC_REGISTER) || (registerMode == SYMMETRIC_REGISTER_SEND) || (registerMode == SYMMETRIC_REGISTER_RECV)) &&
          NCCL_VERSION_CODE < NCCL_VERSION(2, 27, 0)) {
          printf("Option -R 2/3/4 (symmetric) is not supported before NCCL 2.27. Defaulting to local registration\n");
          registerMode = LOCAL_REGISTER;
        }
#else
        printf("Option -R (register) is not supported before NCCL 2.19. Ignoring\n");
#endif
        if (registerMode < DISABLE_REGISTER || registerMode > SYMMETRIC_REGISTER_RECV) {
          printf("Invalid register option %d. Defaulting to disable registration\n", registerMode);
          registerMode = DISABLE_REGISTER;
        }
        break;
      case 'h':
        printf("Usage: %s [options]\n"
               "Options:\n"
               "  -b, --buffer-size <size>          Buffer size in bytes (default: 33554432 = 32MB)\n"
               "  -d, --datatype <type>             Data type for allreduce operations (default: int8)\n"
               "                                    Supported: int8, char, uint8, int32, int, uint32,\n"
               "                                               int64, uint64, half, float16, float,\n"
               "                                               float32, double, float64, bfloat16, bf16\n"
               "  -R, --register <0/1/2/3/4>        Enable buffer registration:\n"
               "                                    0 = disable (default)\n"
               "                                    1 = local registration\n"
               "                                    2 = symmetric registration (all buffers)\n"
               "                                    3 = symmetric registration (send buffers)\n"
               "                                    4 = symmetric registration (recv buffers)\n"
               "  -h, --help                        Show this help message\n",
               argv[0]);
        return 0;
      default:
        break;
    }
  }

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
  printf("\t================ Test fault tolerance for NCCL multi-thread ==============\n");
  errors += faultToleranceMultiThreadTest(comms, nVis, size);

  free(comms);
  printf("[Summary] Single node NCCL fault tolerance test completes, %d errors.\n\n", errors);
  return errors;
}
