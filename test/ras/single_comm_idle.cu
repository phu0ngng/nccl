//
// This test creates a single NCCL communicator.
//
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <unistd.h>

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__, cudaGetErrorString(e));  \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__, ncclGetErrorString(r));  \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

static void launchRasClient(const char* out) {
  char cmd[1024];
  int ret;
  snprintf(cmd, sizeof(cmd), "%s >%s", getenv("NCCLRAS"), out);
  ret = system(cmd);
  if (ret)
    printf("system returned %d\n", ret);
}

int main(int argc, char* argv[])
{
  int mpiRank, nMpiRanks, localRank = 0;

  // Initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nMpiRanks));

  // Calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nMpiRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[mpiRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nMpiRanks; p++) {
     if (p == mpiRank) break;
     if (hostHashs[p] == hostHashs[mpiRank]) localRank++;
  }

  CUDACHECK(cudaSetDevice(localRank));

  ncclUniqueId id;
  ncclComm_t comm;

  int myRank = mpiRank, nRanks = nMpiRanks, root = 0;

  // Get NCCL unique ID at rank 0 and broadcast it to all the others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, root, MPI_COMM_WORLD));

  // Initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  sleep(5);
  if (myRank == 0) launchRasClient("single_comm_idle.after_init.out");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Finalizing NCCL
  ncclCommDestroy(comm);

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (myRank == 0) launchRasClient("single_comm_idle.after_destroy.out");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Finalizing MPI
  MPICHECK(MPI_Finalize());

  return 0;
}
