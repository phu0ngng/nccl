#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <sched.h>
#include <pthread.h>

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

int nRanks = -1;
constexpr int nComms = 32;
ncclUniqueId uid[nComms];
bool uidValid[nComms];

void* rankMain(void *arg) {
  int rank = reinterpret_cast<intptr_t>(arg);
  cudaStream_t stream[nComms];
  ncclComm_t comm[nComms];
  char* buffer;

  CUDACHECK(cudaSetDevice(rank));
  CUDACHECK(cudaMalloc(&buffer, 1<<20));
  CUDACHECK(cudaMemset(buffer, rank, 1<<20));

  for (int c=0; c < nComms; c++) {
    CUDACHECK(cudaStreamCreateWithFlags(&stream[c], cudaStreamNonBlocking));
    if (rank == 0) {
      NCCLCHECK(ncclGetUniqueId(&uid[c]));
      __atomic_store_n(&uidValid[c], true, __ATOMIC_RELEASE);
    } else {
      while (!__atomic_load_n(&uidValid[c], __ATOMIC_ACQUIRE)) sched_yield();
    }
    if (rank == 0) printf("Comm init %d\n", c);
    NCCLCHECK(ncclCommInitRank(&comm[c], nRanks, uid[c], rank));
  }
  if (rank == 0) printf("Comms initialized\n");


  cudaGraph_t graph[nComms];
  cudaGraphExec_t gexec[nComms];
  for (int c=0; c < nComms; c++) {
    if (rank == 0) printf("Creating graph %d\n", c);
    CUDACHECK(cudaStreamBeginCapture(stream[c], cudaStreamCaptureModeRelaxed));
    NCCLCHECK(ncclBroadcast(buffer + c, buffer + c, 1, ncclUint8, 0, comm[c], stream[c]));
    CUDACHECK(cudaStreamEndCapture(stream[c], &graph[c]));
    CUDACHECK(cudaGraphInstantiate(&gexec[c], graph[c], nullptr, nullptr, 0));
  }

  for (int i=0; i < 10; i++) {
    if (rank == 0) printf("Launching round %d\n", i);
    for (int c=0; c < nComms; c++) {
      if (i%2) {
        NCCLCHECK(ncclBroadcast(buffer + c, buffer + c, 1, ncclUint8, i%nRanks, comm[c], stream[c]));
      } else {
        CUDACHECK(cudaGraphLaunch(gexec[c], stream[c]));
      }
    }
  }

  for (int c=0; c < nComms; c++) {
    if (rank == 0) printf("Syncing %d\n", c);
    CUDACHECK(cudaStreamSynchronize(stream[c]));
    CUDACHECK(cudaGraphExecDestroy(gexec[c]));
    CUDACHECK(cudaGraphDestroy(graph[c]));
    NCCLCHECK(ncclCommDestroy(comm[c]));
    CUDACHECK(cudaStreamDestroy(stream[c]));
  }
  cudaFree(buffer);

  return nullptr;
}

int main(int argn, char** argv) {
  setlinebuf(stdout);
  CUDACHECK(cudaGetDeviceCount(&nRanks));

  printf("Overlap test: nranks=%d\n", nRanks);

  pthread_t threads[64];
  for (int r=0; r < nRanks; r++) {
    pthread_create(&threads[r], nullptr, rankMain, reinterpret_cast<void*>(intptr_t(r)));
  }
  for (int r=0; r < nRanks; r++) {
    pthread_join(threads[r], nullptr);
  }

  printf("SUCCESS\n");
  return 0;
}
