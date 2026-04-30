//
// This test creates two NCCL communicators on non-overlapping halves of
// ranks, and then it creates a third communicator consisting of rank 0 of
// each of the existing two communicators.
//
// The test can be used to verify that RAS correctly joins two RAS networks
// into one and propagates the network updates across all the peers.
//
// Must be run on at least 2 ranks total.  4 ranks+ will exercise all the
// scenarios when it comes to the peers update ( because some ranks will
// not be the members of the third communicator).  6 ranks+ will exercise
// all the scenarios when it comes to the RAS network reconfiguration
// (because some connections will not be needed anymore after the third
// communicator is created).
//
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <chrono>
#include <thread>
#include <vector>

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

  assert(nMpiRanks >= 2);

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

  // Communicators 1 and 2 span half of the ranks each and are disjoint.
  // Communicator 3 is just two ranks that "join" the disjoint communicators 1 and 2.
  ncclUniqueId id1, id2, id3;
  ncclComm_t comm1, comm2, comm3;

  int myRank1, nRanks1, root1;
  int myRank2, nRanks2, root2;
  int myRank3, nRanks3, root3;

  nRanks1 = nMpiRanks/2;
  nRanks2 = nMpiRanks - nRanks1;
  nRanks3 = 2;

  myRank1 = (mpiRank < nRanks1 ? mpiRank : -1);
  myRank2 = (mpiRank >= nRanks1 ? mpiRank-nRanks1 : -1);
  if (myRank1 == 0)
    myRank3 = 0;
  else if (myRank2 == 0)
    myRank3 = 1;
  else
    myRank3 = -1;

  root1 = 0;
  root2 = nRanks1;
  root3 = 0;

  // Get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank1 == 0) ncclGetUniqueId(&id1);
  if (myRank2 == 0) ncclGetUniqueId(&id2);
  if (myRank3 == 0) ncclGetUniqueId(&id3);
  MPICHECK(MPI_Bcast((void *)&id1, sizeof(id1), MPI_BYTE, root1, MPI_COMM_WORLD));
  MPICHECK(MPI_Bcast((void *)&id2, sizeof(id2), MPI_BYTE, root2, MPI_COMM_WORLD));
  MPICHECK(MPI_Bcast((void *)&id3, sizeof(id3), MPI_BYTE, root3, MPI_COMM_WORLD));

  // Initializing NCCL
  if (myRank1 >= 0)
    NCCLCHECK(ncclCommInitRank(&comm1, nRanks1, id1, myRank1));
  std::this_thread::sleep_for(std::chrono::seconds(5));
  if (mpiRank == 0) launchRasClient("triple_comm_idle.after_comm1.out");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (myRank2 >= 0)
    NCCLCHECK(ncclCommInitRank(&comm2, nRanks2, id2, myRank2));
  std::this_thread::sleep_for(std::chrono::seconds(5));
  if (mpiRank == 0) launchRasClient("triple_comm_idle.after_comm2.out");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (myRank3 >= 0)
    NCCLCHECK(ncclCommInitRank(&comm3, nRanks3, id3, myRank3));
  std::this_thread::sleep_for(std::chrono::seconds(5));
  if (mpiRank == 0) launchRasClient("triple_comm_idle.after_comm3.out");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Finalizing NCCL
  if (myRank1 >= 0)
    ncclCommDestroy(comm1);
  if (myRank2 >= 0)
    ncclCommDestroy(comm2);
  if (myRank3 >= 0)
    ncclCommDestroy(comm3);

  // Finalizing MPI
  MPICHECK(MPI_Finalize());

  return 0;
}
