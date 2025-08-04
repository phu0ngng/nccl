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
  int num_devices = 2;
  int devs[2] = { 0, 1 };
  ncclComm_t comms[2];
  ncclResult_t res;
  int prev_dev, after_dev;

  res = ncclCommInitAll(comms, num_devices, devs);
  assert(res == ncclSuccess);
  prev_dev = get_device();

  for (int i = 0; i < num_devices; ++i) {
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
  return 0;
}
