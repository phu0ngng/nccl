/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/* This test simulates a hang in the transmission due to e.g. mismatch in
 * collective parameters across launches. The client waits some amount of
 * time for the operations to complete and aborts the operations (by calling
 * ncclCommAbort) after the timeout is reached. It then checks that the
 * streams actually get freed up by NCCL kernels.
 *
 * This test differs from the abort test in test/single/ in that it uses
 * two processes (using fork()). Parent will be the root of the broadcast
 * and child will be receiving the broadcast. Parent will issue a broadcast
 * of half of the bytes the child is expecting.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "nccl.h"
#include "test_utilities.h"

int errors = 0;
double avg_bw = 0.0;
int avg_count = 0;
bool is_reduction = false;
void usage() {
  printf("**************************************************************************\n");
  printf("* Tests nccl hang recovery (multiprocess version).                       *\n");
  printf("* Usage: ./abort_gpu_per_process <0|1|2>                                 *\n");
  printf("*     The optional argument toggles between different test modes:        *\n");
  printf("*     - 0: LL disabled                                                   *\n");
  printf("*     - 1: LL forced                                                     *\n");
  printf("*     - 2: a big non-LL collective followed by a small LL collective     *\n");
  printf("*     - 3: a small LL collective followed by a big non-LL collective     *\n");
  printf("**************************************************************************\n");
}

int main(int argc, char* argv[]) {
  usage();
  unsigned long firstN, secondN;
  firstN = secondN = 1024 * 1024 * 1024;
  int mode = 0;
  if (argc > 1) {
    mode = atoi(argv[1]);
    if (!(mode >=0 && mode <= 3)) mode = 0;
  }
  if (mode == 0) {
    printf("NOTE: NOT using LL\n");
    setenv("NCCL_LL_THRESHOLD", "0", 1);
  } else if (mode == 1) {
    printf("NOTE: Using LL\n");
    char thresh_str[100];
    snprintf(thresh_str, 100, "%lu", firstN+1);
    setenv("NCCL_LL_THRESHOLD", thresh_str, 1);
  } else if (mode == 2) {
    printf("NOTE: non-LL followed by LL\n");
    secondN = 10;
  } else {
    printf("NOTE: LL followed by non-LL\n");
    firstN = 10;
  }
  bool parent, child;
  int fd[2];
  if (pipe(fd) == -1) {
    perror("opening pipe failed");
    exit(EXIT_FAILURE);
  }
  pid_t pid = fork();
  if (pid == -1) {
    perror("fork failed");
    exit(EXIT_FAILURE); 
  }
  {
    int nVis = 0;
    CUDACHECK(cudaGetDeviceCount(&nVis));
    if (nVis < 2) {
      printf("Error: Need at least two GPUs to run the test.\n");
      exit(EXIT_FAILURE);
    }
  }

  if (pid > 0) {
    parent = true;
  } else {
    parent = false;
  }
  child = !parent;

  if (parent) {
    close(fd[0]); // close the reading end
  } else {
    close(fd[1]); // close the writing end
  }

  int myrank;
  ncclUniqueId ncclId;
  if (parent) {
    ssize_t nbytes;
    NCCLCHECK(ncclGetUniqueId(&ncclId));
    nbytes = write(fd[1], &ncclId, sizeof(ncclId));
    if (nbytes == -1) {
      perror("Writing to pipe failed.");
      exit(EXIT_FAILURE);
    }
    if (nbytes != sizeof(ncclId)) {
      perror("Wrote wrong number of bytes.");
      exit(EXIT_FAILURE);
    }
    close(fd[1]);
    myrank = 0;
  } else {
    ssize_t nbytes;
    nbytes = read(fd[0], &ncclId, sizeof(ncclId));
    if (nbytes == -1) {
      perror("Reading from pipe failed.");
      exit(EXIT_FAILURE);
    }
    if (nbytes != sizeof(ncclId)) {
      perror("Read wrong number of bytes.");
      exit(EXIT_FAILURE);
    }
    close(fd[0]);
    myrank = 1;
  }
  CUDACHECK(cudaSetDevice(myrank));

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, 2, ncclId, myrank)); 

  {
    int cudaDev;
    int rank;
    cudaDeviceProp prop;
    NCCLCHECK(ncclCommCuDevice(comm, &cudaDev));
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
    printf("#   Rank %2d uses device %2d [0x%02x] %s\n", rank, cudaDev,
        prop.pciBusID, prop.name);
  }

  uint8_t* buff1, *buff2;
  cudaStream_t s;
  CUDACHECK(cudaMalloc(&buff1, firstN * sizeof(uint8_t)));
  CUDACHECK(cudaMalloc(&buff2, secondN * sizeof(uint8_t)));
  CUDACHECK(cudaStreamCreate(&s));
  CUDACHECK(cudaMemset(buff1, parent ? 0 : 1, firstN * sizeof(uint8_t)));
  CUDACHECK(cudaMemset(buff2, parent ? 0 : 1, secondN * sizeof(uint8_t)));
  CUDACHECK(cudaDeviceSynchronize());

  // get approximate timing for the broadcast operation
  auto start_warmup = std::chrono::high_resolution_clock::now();
  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclBcast((void*)buff1, firstN, ncclUint8, 0, comm, s));
  NCCLCHECK(ncclBcast((void*)buff2, secondN, ncclUint8, 0, comm, s));
  NCCLCHECK(ncclGroupEnd());
  CUDACHECK(cudaStreamSynchronize(s));
  auto stop_warmup = std::chrono::high_resolution_clock::now();
  double elapsedSec_warmup =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          stop_warmup - start_warmup).count();
  double timeout = elapsedSec_warmup * 3;
  if (child) {
    printf("Test run took %.3g seconds. Timeout will be set to %.3g seconds.\n"
        "Starting an operation that will cause NCCL kernels to hang.\n",
        elapsedSec_warmup, timeout);
  }

  auto start = std::chrono::high_resolution_clock::now();
  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclBcast((void*)buff1, parent ? firstN/2 : firstN, ncclUint8, 0, comm, s));
  NCCLCHECK(ncclBcast((void*)buff2, secondN, ncclUint8, 0, comm, s));
  NCCLCHECK(ncclGroupEnd());
  while (std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::high_resolution_clock::now() - start).count() < timeout); /* spin */
  if (parent) {
    printf("[parent] About to destroy the communicator.\n");
  } else {
    printf("[child] About to destroy the communicator.\n");
  }
  ncclResult_t nccl_res = ncclCommAbort(comm);
  if (parent) {
    printf("[parent] Aborting communicator returned %s.\n", ncclGetErrorString(nccl_res));
  } else {
    printf("[child] Aborting communicator returned %s.\n", ncclGetErrorString(nccl_res));
  }
  fflush(stdout);
  CUDACHECK(cudaStreamQuery(s));
  if (parent) wait(NULL);
  return EXIT_SUCCESS;
}

