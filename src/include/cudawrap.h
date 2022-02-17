/*************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CUDAWRAP_H_
#define NCCL_CUDAWRAP_H_

#include <cuda.h>
#include <cudaTypedefs.h>

#define CUPFN(cmd) pfn_##cmd

// Map cu API call to pfn_cu API call
#define CUWRAP(cmd) ((void *)pfn_##cmd != NULL ? (pfn_##cmd) : CUDA_SUCCESS)

// Check CUDA PFN driver calls
#define CUCHECK(cmd) do {				      \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure '%s'", errStr);		      \
      return ncclUnhandledCudaError;			      \
    }							      \
} while(false)

#define CUCHECKGOTO(cmd, res, label) do {		      \
    CUresult err = pfn_##cmd;				      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure '%s'", errStr);		      \
      res = ncclUnhandledCudaError;			      \
      goto label;					      \
    }							      \
} while(false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(cmd) do {						\
    CUresult err = pfn_##cmd;						\
    if( err != CUDA_SUCCESS ) {						\
      const char *errStr;						\
      (void) pfn_cuGetErrorString(err, &errStr);			\
      INFO(NCCL_ALL,"%s:%d Cuda failure '%s'", __FILE__, __LINE__, errStr);	\
    }									\
} while(false)

#define CUCHECKTHREAD(cmd, args) do {					\
    CUresult err = pfn_##cmd;						\
    if (err != CUDA_SUCCESS) {						\
      INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, err); \
      args->ret = ncclUnhandledCudaError;				\
      return args;							\
    }									\
} while(0)

#define DECLARE_CUDA_PFN_EXTERN(symbol) extern PFN_##symbol pfn_##symbol

/* CUDA Driver functions loaded with cuGetProcAddress for versioning */
DECLARE_CUDA_PFN_EXTERN(cuGetErrorString);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorName);
DECLARE_CUDA_PFN_EXTERN(cuMemAddressFree);
DECLARE_CUDA_PFN_EXTERN(cuMemAddressReserve);
DECLARE_CUDA_PFN_EXTERN(cuMemCreate);
DECLARE_CUDA_PFN_EXTERN(cuMemRelease);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAddressRange);
DECLARE_CUDA_PFN_EXTERN(cuMemExportToShareableHandle);
DECLARE_CUDA_PFN_EXTERN(cuMemImportFromShareableHandle);
DECLARE_CUDA_PFN_EXTERN(cuMemSetAccess);
DECLARE_CUDA_PFN_EXTERN(cuMemMap);
DECLARE_CUDA_PFN_EXTERN(cuMemUnmap);

/* CUDA Driver functions loaded with dlsym() */
DECLARE_CUDA_PFN_EXTERN(cuInit);
DECLARE_CUDA_PFN_EXTERN(cuDriverGetVersion);
DECLARE_CUDA_PFN_EXTERN(cuGetProcAddress);


ncclResult_t cudaLibraryInit(void);

#endif
