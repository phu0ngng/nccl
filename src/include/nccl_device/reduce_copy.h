/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#ifndef _NCCL_DEVICE_REDUCE_COPY_H_
#define _NCCL_DEVICE_REDUCE_COPY_H_

#include "utility.h"
#include "core.h"
#include "ptr.h"
#include "impl/reduce_copy__types.h"

// Forward declarations only - implementations in impl/reduce_copy__funcs.h

// Forward declarations for public API functions
// Implementations are in impl/reduce_copy__funcs.h

// SERIES 1.x - Generic ReduceCopy with RedOp (LSA sources only)
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename RedOp, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceLsaCopy(
    Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
    RedOp redOp, IntCount count);

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename RedOp, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceMultimemCopy(
    Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
    RedOp redOp, IntCount count);

// SERIES 2.x - Sum-Specific ReduceCopy (lambda-based foundation)
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumLsaCopy(
    Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, IntCount count);

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(
    Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, IntCount count);

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(
    Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, IntCount count);

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumMultimemCopy(
    Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, IntCount count);

// SERIES 3.x - ReduceSum (N->1)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(
    Coop coop, SrcLambda srcLambda, int nSrc, T* dstPtr, IntCount count);

template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(
    Coop coop, ncclSymPtr<T> src, T* dstPtr, IntCount count, ncclTeam team);

template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(
    Coop coop, ncclSymPtr<T> src, T* dstPtr, IntCount count, ncclDevComm_t devComm);

template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(
    Coop coop, ncclWindow_t window, size_t offset, T* dstPtr, IntCount count, ncclDevComm_t devComm);

template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(
    Coop coop, ncclSymPtr<T> src, T* dstPtr, IntCount count, ncclMultimemHandle multimemHandle);

// 3.3b] Multimem ReduceSum (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(
    Coop coop, T* mcSrcPtr, T* dstPtr, IntCount count);

// 3.4] Local ReduceSum (lambda-based)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(
    Coop coop, SrcLambda srcLambda, int nSrc, T* dstPtr, IntCount count);

// 3.5] Local ReduceSum (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(
    Coop coop, int nSrc, T* basePtr, size_t displ, T* dstPtr, IntCount count);

// SERIES 4.x - Copy/Broadcast (1->N)

// 4.1] LSA Copy (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(
    Coop coop, T* srcPtr, DstLambda dstLambda, int nDst, IntCount count);

// 4.2a] LSA Copy (with ncclSymPtr + team)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(
    Coop coop, T* srcPtr, ncclSymPtr<T> dst, IntCount count, ncclTeam team);

// 4.2b] LSA Copy (with ncclSymPtr + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(
    Coop coop, T* srcPtr, ncclSymPtr<T> dst, IntCount count, ncclDevComm_t devComm);

// 4.2c] LSA Copy (with window + offset)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(
    Coop coop, T* srcPtr, ncclWindow_t window, size_t offset, IntCount count, ncclDevComm_t devComm);

// 4.3a] Multimem Copy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemCopy(
    Coop coop, T* srcPtr, ncclSymPtr<T> dst, IntCount count, ncclMultimemHandle multimemHandle);

// 4.3b] Multimem Copy (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemCopy(
    Coop coop, T* srcPtr, T* mcDstPtr, IntCount count);

// 4.4] Local Copy (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalCopy(
    Coop coop, T* srcPtr, DstLambda dstLambda, int nDst, IntCount count);

// 4.5] Local Copy (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalCopy(
    Coop coop, T* srcPtr, int nDst, T* basePtr, size_t displ, IntCount count);

// SERIES 5.x - ReduceSumCopy (N->M)

// 5.1a] LSA ReduceSumCopy (same team)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(
    Coop coop, ncclSymPtr<T> src, ncclSymPtr<T> dst, IntCount count, ncclTeam team);

// 5.1b] LSA ReduceSumCopy (with devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(
    Coop coop, ncclSymPtr<T> src, ncclSymPtr<T> dst, IntCount count, ncclDevComm_t devComm);

// 5.1c] LSA ReduceSumCopy (with windows)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(
    Coop coop, ncclWindow_t srcWindow, size_t srcOffset, ncclWindow_t dstWindow,
    size_t dstOffset, IntCount count, ncclDevComm_t devComm);

// 5.1d] LSA ReduceSumCopy (different teams)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(
    Coop coop, ncclSymPtr<T> src, ncclTeam srcTeam, ncclSymPtr<T> dst, ncclTeam dstTeam, IntCount count);

// 5.2a] Multimem ReduceSumCopy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(
    Coop coop, ncclSymPtr<T> src, ncclMultimemHandle srcHandle,
    ncclSymPtr<T> dst, ncclMultimemHandle dstHandle, IntCount count);

// 5.2b] Multimem ReduceSumCopy (with raw pointers)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(
    Coop coop, T* mcSrcPtr, T* mcDstPtr, IntCount count);

// 5.3a] LSA -> Multimem ReduceSumCopy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(
    Coop coop, ncclSymPtr<T> src, ncclTeam srcTeam,
    ncclSymPtr<T> dst, ncclMultimemHandle dstHandle, IntCount count);

// 5.3b] LSA -> Multimem ReduceSumCopy (with raw dst pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(
    Coop coop, ncclSymPtr<T> src, ncclTeam srcTeam, T* mcDstPtr, IntCount count);

// 5.3c] Multimem -> LSA ReduceSumCopy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(
    Coop coop, ncclSymPtr<T> src, ncclMultimemHandle srcHandle,
    ncclSymPtr<T> dst, ncclTeam dstTeam, IntCount count);

// 5.3d] Multimem -> LSA ReduceSumCopy (with raw src pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(
    Coop coop, T* mcSrcPtr, ncclSymPtr<T> dst, ncclTeam dstTeam, IntCount count);

// 5.4] Local ReduceSumCopy (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL=8*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSumCopy(
    Coop coop, int nSrc, T* srcBasePtr, size_t srcDispl,
    int nDst, T* dstBasePtr, size_t dstDispl, IntCount count);

#endif // _NCCL_DEVICE_REDUCE_COPY_H_
