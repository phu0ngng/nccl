/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "devcomm.h"
#include "op128.h"
#include <cstdio>
#include <cstdint>

#include <cuda_runtime.h>

// Define min for ssize_t
inline __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

inline __device__ int loadInt(int* ptr) {
  int v;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
      : "=r"(v) : "l"(ptr));
  return v;
}

template<typename RedFn, typename T, int Unroll, int BytePerPack,
         int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes>
__device__ __forceinline__ void ReduceCopyPacks(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void **srcPtrs, int nDsts, void **dstPtrs,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead
  ) {
  // A hunk is the amount of contiguous data a warp consumes per loop iteration
  // assuming all threads partake.
  constexpr int BytePerHunk = Unroll*WARP_SIZE*BytePerPack;
  int nWarps = nThreads/WARP_SIZE;
  int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  // This thread's initial position.
  IntBytes threadBytesBehind = nBytesBehind + (warp*BytePerHunk + lane*BytePerPack);
  IntBytes threadBytesAhead = nBytesAhead - (warp*BytePerHunk + lane*BytePerPack);
  // Number of hunks to be consumed over all warps.
  int nHunksAhead = nBytesAhead/BytePerHunk;
  // Advance collective position.
  nBytesBehind += nHunksAhead*BytePerHunk;
  nBytesAhead -= nHunksAhead*BytePerHunk;
  if (Unroll == 1 && BytePerPack <= nBytesAhead) {
    // Only Unroll=1 can do partial hunks (where not all threads partake).
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead%BytePerPack);
    nBytesAhead = nBytesAhead%BytePerPack;
  }
  nHunksAhead -= warp;

  RedFn redFn(redArg);
  uintptr_t minSrcs[MinSrcs + !MinSrcs];
  uintptr_t minDsts[MinDsts + !MinDsts];
  #pragma unroll
  for (int s=0; s < MinSrcs; s++)
    minSrcs[s] = cvta_to_global(srcPtrs[s]) + threadBytesBehind;
  #pragma unroll
  for (int d=0; d < MinDsts; d++)
    minDsts[d] = cvta_to_global(dstPtrs[d]) + threadBytesBehind;

  while (Unroll == 1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
    BytePack<BytePerPack> acc[Unroll];

    { RedFn preFn(0 < PreOpSrcs ? preOpArgs[0] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        acc[u] = ld_global<BytePerPack>(minSrcs[0]);
        minSrcs[0] += WARP_SIZE*BytePerPack;
        if (0 < PreOpSrcs) acc[u] = applyPreOp(preFn, acc[u]);
      }
    }

    #pragma unroll (MinSrcs-1 + !(MinSrcs-1))
    for (int s=1; s < MinSrcs; s++) {
      BytePack<BytePerPack> tmp[Unroll];
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        tmp[u] = ld_global<BytePerPack>(minSrcs[s]);
        minSrcs[s] += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    for (int s=MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++) {
      uintptr_t src = cvta_to_global(srcPtrs[s]) + threadBytesBehind;
      BytePack<BytePerPack> tmp[Unroll];
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        tmp[u] = ld_global<BytePerPack>(src);
        src += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    if (postOp) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++)
        acc[u] = applyPostOp(redFn, acc[u]);
    }

    #pragma unroll (MinDsts + !MinDsts)
    for (int d=0; d < MinDsts; d++) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        st_global<BytePerPack>(minDsts[d], acc[u]);
        minDsts[d] += WARP_SIZE*BytePerPack;
      }
    }
    for (int d=MinDsts; (MinDsts < MaxDsts) && (d < MaxDsts) && (d < nDsts); d++) {
      uintptr_t dst = cvta_to_global(dstPtrs[d]) + threadBytesBehind;
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        st_global<BytePerPack>(dst, acc[u]);
        dst += WARP_SIZE*BytePerPack;
      }
    }

    nWarps = nThreads/WARP_SIZE;
    #pragma unroll
    for (int s=0; s < MinSrcs; s++) minSrcs[s] += (nWarps-1)*BytePerHunk;
    #pragma unroll
    for (int d=0; d < MinDsts; d++) minDsts[d] += (nWarps-1)*BytePerHunk;
    threadBytesBehind += nWarps*BytePerHunk;
    threadBytesAhead -= nWarps*BytePerHunk;
    nHunksAhead -= nWarps;
  }

  nWarps = nThreads/WARP_SIZE;
  warp = thread/WARP_SIZE;
  lane = thread%WARP_SIZE;
  // The last loop iteration could have been partial, i.e. not taken by all
  // threads. The threads that weren't included need an extra subtraction to
  // make the value warp uniform.
  if (nHunksAhead > 0) nHunksAhead -= nWarps;
  // Rotate warps so the warp which got the least work here will be warp 0.
  // This effectively assigns: warp = (warp-nHunks+nWarps)%nWarps;
  warp = -nHunksAhead;
  thread = warp*WARP_SIZE + lane;
}

template<int Unroll, typename RedFn, typename T,
         int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes>
__device__ __forceinline__ void ReduceOrCopyMulti(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void **srcPtrs, int nDsts, void **dstPtrs,
    IntBytes nElts
  ) {
  //int nWarps = nThreads/WARP_SIZE;
  //int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  // Check that all is 16B aligned. If not don't use 16B load/stores.
  int allAligned = 1;
  if (lane < nSrcs) allAligned &= 0 == cvta_to_global(srcPtrs[lane])%16;
  if (lane < nDsts) allAligned &= 0 == cvta_to_global(dstPtrs[lane])%16;
  allAligned = __all_sync(~0u, allAligned);

  IntBytes nBytesBehind = 0;
  IntBytes nBytesAhead = nElts*sizeof(T);
  if (allAligned) {
    ReduceCopyPacks<RedFn, T, Unroll, /*BytePerPack=*/16,
      MinSrcs, MaxSrcs, MinDsts, MaxDsts, PreOpSrcs>
      (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
       nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
    if (nBytesAhead == 0) return;

    ReduceCopyPacks<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/16,
      MinSrcs, MaxSrcs, MinDsts, MaxDsts, PreOpSrcs>
      (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
       nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
    if (nBytesAhead == 0) return;
  }

  ReduceCopyPacks<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
    MinSrcs, MaxSrcs, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
  if (nBytesAhead == 0) return;

  ReduceCopyPacks<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/sizeof(T),
    MinSrcs, MaxSrcs, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrs, nDsts, dstPtrs, /*&*/nBytesBehind, /*&*/nBytesAhead);
}

#endif // COMMON_KERNEL_H_
