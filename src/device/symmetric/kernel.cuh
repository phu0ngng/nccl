#ifndef NCCL_DEVICE_SYMMETRIC_KERNEL_H_
#define NCCL_DEVICE_SYMMETRIC_KERNEL_H_

#include "dev_kernels.h"

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclDevkRun_AllReduce_AGxLL_R(struct ncclDevkDevWorkArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclDevkRun_AllReduce_AGxLLMC_R(struct ncclDevkDevWorkArgs const* args);

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclDevkRun_AllReduce_RSxLD_AGxST(struct ncclDevkDevWorkArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclDevkRun_AllReduce_RSxLDMC_AGxSTMC(struct ncclDevkDevWorkArgs const* args);

__device__ __forceinline__ void ncclDevkRun_AllGather_LL(struct ncclDevkDevWorkArgs const* args);
__device__ __forceinline__ void ncclDevkRun_AllGather_LLMC(struct ncclDevkDevWorkArgs const* args);
__device__ __forceinline__ void ncclDevkRun_AllGather_ST(struct ncclDevkDevWorkArgs const* args);
__device__ __forceinline__ void ncclDevkRun_AllGather_STMC(struct ncclDevkDevWorkArgs const* args);

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclDevkRun_ReduceScatter_LL(struct ncclDevkDevWorkArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclDevkRun_ReduceScatter_LD(struct ncclDevkDevWorkArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclDevkRun_ReduceScatter_LDMC(struct ncclDevkDevWorkArgs const* args);
#endif
