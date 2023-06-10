#include <cuda_runtime.h>
#include <iostream>

#include "verifiable.h"

int main(int arg_n, char **args) {
  std::cerr<<"You are hoping to see no output beyond this line."<<std::endl;
  cudaSetDevice(0);
  ncclVerifiableLaunchSelfTest();
  cudaDeviceSynchronize();
  return 0;
}
