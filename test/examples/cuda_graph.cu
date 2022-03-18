/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
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

#define GRAPH_LAUNCH_ITERATIONS 2

int main(int argc, char* argv[]) {
  int size = 32*1024*1024;

  int myRank, nRanks, localRank = 0;

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  ncclUniqueId id;
  ncclComm_t comm;
  int32_t *sendbuff, *recvbuff;
  cudaStream_t s, uncapStream;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMallocHost(&sendbuff, size * sizeof(int32_t)));
  CUDACHECK(cudaMallocHost(&recvbuff, size * sizeof(int32_t)));
  CUDACHECK(cudaMemset(sendbuff, 1, size * sizeof(int32_t)));
  CUDACHECK(cudaStreamCreate(&s));
  CUDACHECK(cudaStreamCreate(&uncapStream));

  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclInt32, ncclSum, comm, uncapStream));
  CUDACHECK(cudaStreamSynchronize(uncapStream));

  //create cuda graph
  cudaGraph_t graph;
  CUDACHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));

  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclInt32, ncclSum, comm, s));

  CUDACHECK(cudaStreamEndCapture(s, &graph));

  //check graph nodes
  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  CUDACHECK(cudaGraphGetNodes(graph, nodes, &numNodes));
  printf("Num of nodes in the graph created using stream capture API = %zu\n", numNodes);

  //instantiate cuda graph
  cudaGraphExec_t graphExec;
  CUDACHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  //launch cuda graph
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    CUDACHECK(cudaGraphLaunch(graphExec, s));
  }

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclInt32, ncclSum, comm, uncapStream));
  CUDACHECK(cudaStreamSynchronize(uncapStream));

  //check result
  printf("[MPI Rank %d] recvbuff[0] %d \n", myRank, recvbuff[0]);
  int32_t initVal = 0;
  memset(&initVal, 1, sizeof(initVal));
  int pass = (recvbuff[0] == nRanks*initVal) ? 1 : 0;

  //free device buffers
  CUDACHECK(cudaFreeHost(sendbuff));
  CUDACHECK(cudaFreeHost(recvbuff));

  //destroy cuda graph
  CUDACHECK(cudaGraphExecDestroy(graphExec));
  CUDACHECK(cudaGraphDestroy(graph));

  //finalizing NCCL
  ncclCommDestroy(comm);

  //finalizing MPI
  MPICHECK(MPI_Finalize());

  //Needed for cuda-memcheck --leak-check full
  cudaDeviceReset();

  printf("[MPI Rank %d] %s \n", myRank, pass ? "Success" : "Fail");
  return 0;
}
