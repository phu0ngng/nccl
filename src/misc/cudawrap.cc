/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "debug.h"
#include "cudawrap.h"

#include <dlfcn.h>

#define DECLARE_CUDA_PFN(symbol) PFN_##symbol pfn_##symbol

/* CUDA Driver functions loaded with cuGetProcAddress for versioning */
DECLARE_CUDA_PFN(cuCtxCreate);
DECLARE_CUDA_PFN(cuCtxDestroy);
DECLARE_CUDA_PFN(cuCtxSetCurrent);
DECLARE_CUDA_PFN(cuDeviceGet);
DECLARE_CUDA_PFN(cuDeviceGetAttribute);
DECLARE_CUDA_PFN(cuGetErrorString);
DECLARE_CUDA_PFN(cuGetErrorName);
DECLARE_CUDA_PFN(cuMemAddressFree);
DECLARE_CUDA_PFN(cuMemAddressReserve);
DECLARE_CUDA_PFN(cuMemCreate);
DECLARE_CUDA_PFN(cuMemRelease);
DECLARE_CUDA_PFN(cuMemGetAddressRange);
DECLARE_CUDA_PFN(cuMemExportToShareableHandle);
DECLARE_CUDA_PFN(cuMemImportFromShareableHandle);
DECLARE_CUDA_PFN(cuMemSetAccess);
DECLARE_CUDA_PFN(cuMemMap);
DECLARE_CUDA_PFN(cuMemUnmap);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN(cuMemGetHandleForAddressRange); // DMA-BUF support
#endif

/* CUDA Driver functions loaded with dlsym() */
DECLARE_CUDA_PFN(cuInit);
DECLARE_CUDA_PFN(cuDriverGetVersion);
DECLARE_CUDA_PFN(cuGetProcAddress);

static enum { cudaUninitialized, cudaInitializing, cudaInitialized, cudaError } cudaState = cudaUninitialized;

#define CUDA_DRIVER_MIN_VERSION 11040

static void *cudaLib;
static int cudaDriverVersion;

/*
  Load the CUDA symbols
 */
static int cudaPfnFuncLoader(void) {
  CUresult res;

#define LOAD_SYM(symbol, ignore) do {                                   \
    res = pfn_cuGetProcAddress(#symbol, (void **) (&pfn_##symbol), cudaDriverVersion, 0); \
    if (res != 0) {                                                     \
      if (!ignore) {                                                    \
        WARN("Retrieve %s version %d failed with %d", #symbol, cudaDriverVersion, res); \
        return ncclSystemError; }                                       \
    } } while(0)

  LOAD_SYM(cuGetErrorString, 0);
  LOAD_SYM(cuGetErrorName, 0);
  LOAD_SYM(cuDeviceGet, 0);
  LOAD_SYM(cuDeviceGetAttribute, 0);

  LOAD_SYM(cuCtxCreate, 1);
  LOAD_SYM(cuCtxDestroy, 1);
  LOAD_SYM(cuCtxSetCurrent, 1);

  LOAD_SYM(cuMemCreate, 1);
  LOAD_SYM(cuMemRelease, 1);
  LOAD_SYM(cuMemAddressFree, 1);
  LOAD_SYM(cuMemAddressReserve, 1);
  LOAD_SYM(cuMemGetAddressRange, 1);
  LOAD_SYM(cuMemExportToShareableHandle, 1);
  LOAD_SYM(cuMemImportFromShareableHandle, 1);
  LOAD_SYM(cuMemSetAccess, 1);
  LOAD_SYM(cuMemMap, 1);
  LOAD_SYM(cuMemUnmap, 1);
#if CUDA_VERSION >= 11070
  LOAD_SYM(cuMemGetHandleForAddressRange, 1); // DMA-BUF support
#endif
  return ncclSuccess;
}

ncclResult_t cudaLibraryInit(void) {
  CUresult res;

  if (cudaState == cudaInitialized)
    return ncclSuccess;
  if (cudaState == cudaError)
    return ncclSystemError;

  if (__sync_bool_compare_and_swap(&cudaState, cudaUninitialized, cudaInitializing) == false) {
    // Another thread raced in front of us. Wait for it to be done.
    while (cudaState == cudaInitializing) pthread_yield();
    return (cudaState == cudaInitialized) ? ncclSuccess : ncclSystemError;
  }

  /*
   * Load CUDA driver library
   */
  char path[1024];
  char *ncclCudaPath = getenv("NCCL_CUDA_PATH");
  if (ncclCudaPath == NULL)
    snprintf(path, 1024, "%s", "libcuda.so");
  else
    snprintf(path, 1024, "%s%s", ncclCudaPath, "libcuda.so");

  cudaLib = dlopen(path, RTLD_LAZY);
  if (cudaLib == NULL) {
    WARN("Failed to find CUDA library in %s (NCCL_CUDA_PATH=%s)", ncclCudaPath, ncclCudaPath);
    goto error;
  }

  /*
   * Load initial CUDA functions
   */

  pfn_cuInit = (PFN_cuInit) dlsym(cudaLib, "cuInit");
  if (pfn_cuInit == NULL) {
    WARN("Failed to load CUDA missing symbol cuInit");
    goto error;
  }

  pfn_cuDriverGetVersion = (PFN_cuDriverGetVersion) dlsym(cudaLib, "cuDriverGetVersion");
  if (pfn_cuDriverGetVersion == NULL) {
    WARN("Failed to load CUDA missing symbol cuDriverGetVersion");
    goto error;
  }

  pfn_cuGetProcAddress = (PFN_cuGetProcAddress) dlsym(cudaLib, "cuGetProcAddress");
  if (pfn_cuGetProcAddress == NULL) {
    WARN("Failed to load CUDA missing symbol cuGetProcAddress");
    goto error;
  }

  /*
   * Required to initialize the CUDA Driver.
   * Multiple calls of cuInit() will return immediately
   * without making any relevant change
   */
  pfn_cuInit(0);

  res = pfn_cuDriverGetVersion(&cudaDriverVersion);
  if (res != 0) {
    WARN("cuDriverGetVersion failed with %d", res);
    goto error;
  }

  if (cudaDriverVersion < CUDA_DRIVER_MIN_VERSION) {
    WARN("CUDA Driver version found is %d. Minimum requirement is %d",
         cudaDriverVersion, CUDA_DRIVER_MIN_VERSION);
    goto error;
  }

  if (cudaPfnFuncLoader()) {
    WARN("CUDA some PFN functions not found in the library");
    goto error;
  }

  cudaState = cudaInitialized;
  return ncclSuccess;

error:
  cudaState = cudaError;
  return ncclSystemError;
}


