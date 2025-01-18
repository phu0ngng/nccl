//
// This test creates three NCCL communicators and performs a series of p2p and
// collective operations to verify that the tracking of collective operation
// counts works as expected.
//
// One of the ranks will initially be delayed to some collective operations,
// which should result in a MISMATCH report from RAS.  It should then catch
// up with the ramining ranks.
//
// Should be run on at least 4 ranks total.
//
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

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
  int size = 1024*1024;

  // Initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nMpiRanks));

  assert(nMpiRanks >= 2);

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

  float* sendbuff = nullptr;
  float* recvbuff = nullptr;
  CUDACHECK(cudaMalloc((void**)&sendbuff, size * sizeof(*sendbuff)));
  CUDACHECK(cudaMalloc((void**)&recvbuff, size * sizeof(*recvbuff)));
  CUDACHECK(cudaMemset(sendbuff, 1, size * sizeof(*sendbuff)));
  CUDACHECK(cudaMemset(recvbuff, 0, size * sizeof(*recvbuff)));

  // Communicator 1 spans all the ranks.
  // Communicators 2 and 3 are disjoint split-share halves of communicator 1.
  ncclUniqueId id1;
  ncclComm_t comm1, comm2 = nullptr, comm3 = nullptr;
  cudaStream_t s1, s2;

  int myRank1, nRanks1, root1;
  int myRank2, nRanks2;
  int myRank3;

  nRanks1 = nMpiRanks;
  nRanks2 = nRanks1 / 2;

  myRank1 = mpiRank;
  myRank2 = (myRank1 < nRanks2 ? myRank1 : -1);
  myRank3 = (myRank1 < nRanks2 ? -1 : myRank1 - nRanks2);

  root1 = 0;

  // Get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank1 == 0) ncclGetUniqueId(&id1);
  MPICHECK(MPI_Bcast((void *)&id1, sizeof(id1), MPI_BYTE, root1, MPI_COMM_WORLD));

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.splitShare = 1;

  // Initializing NCCL
  // Creating comm1
  if (myRank1 >= 0)
    NCCLCHECK(ncclCommInitRankConfig(&comm1, nRanks1, id1, myRank1, &config));
  sleep(5);
  if (mpiRank == 0) launchRasClient("triple_comm_latecoll.after_comm1.out");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Splitting comm1 into comm2 and comm3
  NCCLCHECK(ncclCommSplit(comm1, (myRank2 != -1), myRank1, (myRank2 != -1 ? &comm2 : &comm3), nullptr));
  sleep(5);
  if (mpiRank == 0) launchRasClient("triple_comm_latecoll.after_commsplit.out");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  CUDACHECK(cudaStreamCreate(&s1));
  CUDACHECK(cudaStreamCreate(&s2));

  // Communicating over comm1 using collectives
  if (mpiRank == 0)
    printf("starting the first 10 allreduce calls and a broadcast\n");
  for (int i = 0; i < 10; i++)
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm1, s1));
  NCCLCHECK(ncclBroadcast(sendbuff, recvbuff, size, ncclFloat, 0, comm1, s1));
  CUDACHECK(cudaStreamSynchronize(s1));
  if (mpiRank == 0)
    printf("finished the first 10 allreduce calls and a broadcast\n");

  // Communicating over comm1 using p2p's
  if (mpiRank == 0)
    printf("starting the 10 p2p calls\n");
  for (int i = 0; i < 10; i++) {
    if (myRank1 == 0)
      NCCLCHECK(ncclSend(sendbuff, size, ncclFloat, nRanks1-1, comm1, s1));
    if (myRank1 == nRanks1-1)
      NCCLCHECK(ncclRecv(recvbuff, size, ncclFloat, 0, comm1, s1));
  }
  CUDACHECK(cudaStreamSynchronize(s1));
  if (mpiRank == 0)
    printf("finished the 10 p2p calls\n");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Communicating over comm2 using collectives
  if (comm2 != nullptr) {
    if (mpiRank == 0)
      printf("starting the 5 comm2 allreduce calls and a broadcast\n");
    for (int i = 0; i < 5; i++)
      NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm2, s2));
    NCCLCHECK(ncclBroadcast(sendbuff, recvbuff, size, ncclFloat, 0, comm2, s2));
    CUDACHECK(cudaStreamSynchronize(s2));
    if (mpiRank == 0)
      printf("finished the 5 comm2 allreduce calls and a broadcast\n");
  }
  sleep(5);
  if (mpiRank == 0) launchRasClient("triple_comm_latecoll.after_colls_indiv.out");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Communicating over comm1 using aggregated collectives
  if (mpiRank == 0)
    printf("starting the aggregated 10 allreduce calls and a broadcast\n");
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < 10; i++)
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm1, s1));
  NCCLCHECK(ncclBroadcast(sendbuff, recvbuff, size, ncclFloat, 0, comm1, s1));
  NCCLCHECK(ncclGroupEnd());
  CUDACHECK(cudaStreamSynchronize(s1));
  sleep(5);
  if (mpiRank == 0) {
    printf("finished the aggregated 10 allreduce calls and a broadcast\n");
    launchRasClient("triple_comm_latecoll.after_colls_agg.out");
  }

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Note: RAS doesn't currently update counters for graph-captured collectives.  But we still test to ensure
  // that the counters on different ranks don't get out of sync.
  if (mpiRank == 0)
    printf("starting the graph capturing of 10 allreduce calls and a broadcast\n");
  // Communicating over comm1 using collectives with graph capture
  cudaGraph_t graph;
  CUDACHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < 10; i++)
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm1, s1));
  NCCLCHECK(ncclBroadcast(sendbuff, recvbuff, size, ncclFloat, 0, comm1, s1));
  CUDACHECK(cudaStreamEndCapture(s1, &graph));
  CUDACHECK(cudaStreamSynchronize(s1));
  if (mpiRank == 0)
    printf("finished the graph capturing of 10 allreduce calls and a broadcast\n");
  cudaGraphExec_t instance;
  CUDACHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
  if (mpiRank == 0)
    printf("starting the first graph launch\n");
  CUDACHECK(cudaGraphLaunch(instance, s1));
  if (mpiRank == 0)
    printf("starting the second graph launch\n");
  CUDACHECK(cudaGraphLaunch(instance, s1));
  CUDACHECK(cudaStreamSynchronize(s1));
  if (mpiRank == 0)
    printf("finished the two graph launches\n");
  CUDACHECK(cudaGraphExecDestroy(instance));
  CUDACHECK(cudaGraphDestroy(graph));
  sleep(5);
  if (mpiRank == 0) launchRasClient("triple_comm_latecoll.after_colls_graph.out");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Delaying the last process.
  if (myRank1 == nRanks1-1)
    sleep(10);

  // Communicating over comm1 using collectives
  for (int i = 0; i < 10; i++)
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm1, s1));
  NCCLCHECK(ncclBroadcast(sendbuff, recvbuff, size, ncclFloat, 0, comm1, s1));
  sleep(5);
  if (mpiRank == 0) launchRasClient("triple_comm_latecoll.after_delay.out");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  CUDACHECK(cudaStreamSynchronize(s1));
  if (mpiRank == 0) launchRasClient("triple_comm_latecoll.after_sync.out");

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  CUDACHECK(cudaStreamDestroy(s1));
  CUDACHECK(cudaStreamDestroy(s2));

  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));

  // Finalizing NCCL
  if (myRank3 >= 0)
    ncclCommDestroy(comm3);
  if (myRank2 >= 0)
    ncclCommDestroy(comm2);
  ncclCommDestroy(comm1);

  // Finalizing MPI
  MPICHECK(MPI_Finalize());

  return 0;
}
