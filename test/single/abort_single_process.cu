/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/* This test simulates a hang in the transmission due to e.g. mismatch in
 * collective parameters across launches. The client waits some amount of
 * time for the operations to complete and aborts the operations (by calling
 * ncclCommDestroy) after the timeout is reached. It then checks that the
 * streams actually get freed up by NCCL kernels.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nccl.h"
#include "test_utilities.h"

int errors = 0;
double avg_bw = 0.0;
int avg_count = 0;
bool is_reduction = false;

template<typename T>
int RunTest(T** buff, const size_t N, const ncclDataType_t type, const int root,
    ncclComm_t* const comms, const int *dList) {
  // initialize data
  int nDev = 0;
  NCCLCHECK(ncclCommCount(comms[0], &nDev));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  T* buffer = (T*)malloc(N * sizeof(T));
  T* result = (T*)malloc(N * sizeof(T));
  memset(result, 0, N * sizeof(T));

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(dList[i]));
    CUDACHECK(cudaStreamCreate(s+i));

    if (i == root) {
      Randomize(buff[root], N, root);
      CUDACHECK(cudaMemcpy(result, buff[root], N * sizeof(T),
          cudaMemcpyDeviceToHost));
    } else {
      CUDACHECK(cudaMemset(buff[i], 0, N * sizeof(T)));
    }

    CUDACHECK(cudaDeviceSynchronize());
  }

  // get approximate timing for the broadcast operation
  auto start_warmup = std::chrono::high_resolution_clock::now();
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclBcast((void*)buff[i], N, type, root, comms[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(dList[i]));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }
  auto stop_warmup = std::chrono::high_resolution_clock::now();
  double elapsedSec_warmup =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          stop_warmup - start_warmup).count();
  double timeout = elapsedSec_warmup * 10;
  printf("Test run took %g seconds. Timeout will be set to %g seconds.\n"
      "Starting an operation that will cause NCCL kernels to hang.\n",
      elapsedSec_warmup, timeout);

//  for (int n = 1; n <= N; n = n << 1)
  auto start = std::chrono::high_resolution_clock::now();
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    size_t n = N;
    if (i == root) n = N/2;
    NCCLCHECK(ncclBcast((void*)buff[i], n, type, root, comms[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());
  while (std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::high_resolution_clock::now() - start).count() < timeout); /* spin */

  bool anyNotReady = false;
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(dList[i]));
    cudaError_t err = cudaStreamQuery(s[i]);
    if (err == cudaErrorNotReady) {
      anyNotReady = true;
    }
  }
  if (anyNotReady == false) {
    printf("Test failed. At least one of the streams should be in the \"not ready\" state.\n");
    return EXIT_FAILURE;
  }

  // HANGS on cudaFree() unless we reverse the order of destruction
  //for (int i = 0; i < nDev; ++i) {
  for (int i = nDev-1; i >= 0; --i) {
    printf("Destroying communicator %i\n", i);
    ncclResult_t nccl_res = ncclCommDestroy(comms[i]);
    printf("Destroying communicator %d returned %s.\n", i, ncclGetErrorString(nccl_res));
  }

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(dList[i]));
    cudaError_t err = cudaStreamQuery(s[i]);
    printf("Stream %d status is %s\n", i, cudaGetErrorString(err));
  }

  for(int i=0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(dList[i]));
    CUDACHECK(cudaStreamDestroy(s[i]));
  }
  free(s);
  free(buffer);
  free(result);
  return EXIT_SUCCESS;
}

void usage() {
  printf("Tests nccl hang recovery.\n");
}

int main(int argc, char* argv[]) {
  {
    int nVis = 0;
    CUDACHECK(cudaGetDeviceCount(&nVis));
    if (nVis < 2) {
      printf("Error: Need at least two GPUs to run the test.\n");
      usage();
      exit(EXIT_FAILURE);
    }
  }

  const unsigned long N = 1024 * 1024 * 1024;

  int nDev = 2;
  int devices[nDev];
  devices[0] = 0;
  devices[1] = 1;
  ncclComm_t comms[nDev];
  NCCLCHECK(ncclCommInitAll(comms, nDev, devices));

  printf("# Using devices\n");
  for (int g = 0; g < nDev; ++g) {
    int cudaDev;
    int rank;
    cudaDeviceProp prop;
    NCCLCHECK(ncclCommCuDevice(comms[g], &cudaDev));
    NCCLCHECK(ncclCommUserRank(comms[g], &rank));
    CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
    printf("#   Rank %2d uses device %2d [0x%02x] %s\n", rank, cudaDev,
        prop.pciBusID, prop.name);
  }
  printf("\n");

  uint8_t* buff[nDev];

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(devices[i]));
    CUDACHECK(cudaMalloc(buff + i, N * sizeof(uint8_t)));
  }

  exit(RunTest<uint8_t>(buff, N, ncclUint8, 0, comms, devices));

  if (false)
    exit(EXIT_FAILURE);
  else 
    exit(EXIT_SUCCESS);
}

