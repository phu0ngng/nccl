#ifndef NCCL_DEVICE_SYMMETRIC_KERNEL_H_
#define NCCL_DEVICE_SYMMETRIC_KERNEL_H_

#include "sym_kernels.h"

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R(struct ncclSymkDevArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLLMC_R(struct ncclSymkDevArgs const* args);

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLD_AGxST(struct ncclSymkDevArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC(struct ncclSymkDevArgs const* args);

__device__ __forceinline__ void ncclSymkRun_AllGather_LL(struct ncclSymkDevArgs const* args);
__device__ __forceinline__ void ncclSymkRun_AllGather_LLMC(struct ncclSymkDevArgs const* args);
__device__ __forceinline__ void ncclSymkRun_AllGather_ST(struct ncclSymkDevArgs const* args);
__device__ __forceinline__ void ncclSymkRun_AllGather_STMC(struct ncclSymkDevArgs const* args);

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LL(struct ncclSymkDevArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LD(struct ncclSymkDevArgs const* args);
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LDMC(struct ncclSymkDevArgs const* args);
#endif
