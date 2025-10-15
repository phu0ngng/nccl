#include <stdio.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdlib.h>

int get_device() {
  int current_device;
  cudaError_t cuda_res = cudaGetDevice(&current_device);
  assert(cuda_res == cudaSuccess);
  return current_device;
}

int main() {
  int deviceCount;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);
  assert(error == cudaSuccess);

  if (deviceCount < 2) {
    printf("Skip this test on a single GPU platform [WAIVE]\n");
    return 0;
  }
  ncclComm_t* comms = (ncclComm_t*)malloc(deviceCount * sizeof(ncclComm_t));
  assert(comms != NULL);
  ncclResult_t res;
  int prev_dev, after_dev;

  res = ncclCommInitAll(comms, deviceCount, NULL);
  assert(res == ncclSuccess);
  prev_dev = get_device();

  for (int i = 0; i < deviceCount; ++i) {
    res = ncclCommDestroy(comms[i]);
    assert(res == ncclSuccess);
  }

  after_dev = get_device();
  if (prev_dev != after_dev) {
    printf("Check device ID after ncclCommDestroy [FAIL]\n");
    exit(1);
  } else {
    printf("Check device ID after ncclCommDestroy [PASS]\n");
  }
  free(comms);
  return 0;
}
