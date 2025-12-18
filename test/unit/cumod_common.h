#ifndef CUMOD_COMMON_H
#define CUMOD_COMMON_H

#include <cuda.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#define MAX_PATH_LENGTH 1024

#define CU_CHECK(call) do {                         \
    CUresult err = call;                            \
    if (err != CUDA_SUCCESS) {                      \
        const char* errStr;                         \
        cuGetErrorString(err, &errStr);             \
        printf("CUDA error at %s:%d  '%s'\n",       \
            __FILE__, __LINE__, errStr);            \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

// IR utility functions for loading and managing CUDA modules
static void initCumodule(CUmodule* module, const char* cubinName) {
  char exePath[MAX_PATH_LENGTH];
  ssize_t count = readlink("/proc/self/exe", exePath, MAX_PATH_LENGTH);
  if (count == -1 || count >= MAX_PATH_LENGTH) {
    fprintf(stderr, "%s:%d readlink returned error: %d\n", __FILE__, __LINE__, errno);
    exit(EXIT_FAILURE);
  }
  exePath[count] = '\0';

  char cubinPath[MAX_PATH_LENGTH];
  char* exeDir = dirname(exePath);
  if (snprintf(cubinPath, MAX_PATH_LENGTH, "%s/%s", exeDir, cubinName) >= MAX_PATH_LENGTH) {
    fprintf(stderr, "%s:%d snprintf could not fit path name into %d bytes: %s/%s\n",
            __FILE__, __LINE__, MAX_PATH_LENGTH, exeDir, cubinName);
    exit(EXIT_FAILURE);
  }
  printf("CUBIN Selected: %s\n", cubinPath);
  CU_CHECK(cuModuleLoad(module, cubinPath));
}

static void finiCumodule(CUmodule* module) {
  if (*module != NULL) {
    CU_CHECK(cuModuleUnload(*module));
    *module = NULL;
  }
}

static void initTestCaseKernel(CUmodule module, CUfunction* kernel, const char* kernelName) {
  CU_CHECK(cuModuleGetFunction(kernel, module, kernelName));
}

#endif // CUMOD_COMMON_H