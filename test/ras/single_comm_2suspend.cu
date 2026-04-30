//
// This test creates a single NCCL communicator.  After that, the last two ranks
// suspend themselves (thus terminating RAS keepalives).
//
// Must be run on at least 3 ranks total.  4+ ranks will exercise the creation
// of fallbacks to fallbacks.
//
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <chrono>
#include <thread>
#include <vector>

#ifdef NCCL_OS_WINDOWS
#error "Suspend tests require POSIX signals (kill/SIGSTOP) and cannot build on Windows"
#endif
#include <signal.h>
#include <unistd.h>

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include "os.h"

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

static void getHostName(char* hostname, int maxlen) {
  ncclTestGetHostname(hostname, maxlen);
  for (int i=0; i< maxlen && hostname[i] != '\0'; i++) {
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

  assert(nMpiRanks >= 3);

  // Calculating localRank based on hostname which is used in selecting a GPU
  std::vector<uint64_t> hostHashs(nMpiRanks);
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[mpiRank] = ncclTestGetHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs.data(), sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
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

  std::this_thread::sleep_for(std::chrono::seconds(5));

  if (myRank >= nRanks - 2) {
    kill(getpid(), SIGSTOP);
  }

  std::this_thread::sleep_for(std::chrono::seconds(2));

  // Should block until at least 5 seconds after suspend and indicate leg
  // timeouts and incomplete information.
  if (mpiRank == 0) launchRasClient("single_comm_2suspend.02s_after_suspend.out");

  std::this_thread::sleep_for(std::chrono::seconds(10));

  // Should indicate incomplete information.
  if (mpiRank == 0) launchRasClient("single_comm_2suspend.12s_after_suspend.out");

  std::this_thread::sleep_for(std::chrono::seconds(58));

  // Should indicate a dead process.
  if (mpiRank == 0) launchRasClient("single_comm_2suspend.70s_after_suspend.out");

  std::this_thread::sleep_for(std::chrono::seconds(10));

  // Finalizing NCCL
  ncclCommDestroy(comm);

  // Finalizing MPI
  //MPICHECK(MPI_Finalize());

  return 0;
}
