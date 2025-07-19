#include "sym_kernels.h"
#include "nccl_sym.h"
#include "kernel.cuh"
#include "primitives.cuh"

template<int BytePerPack, int UnrollPacks, int UnrollPeers, typename T, typename Red>
static __device__ __forceinline__ void allreduceDeep(
    ncclSymkKernelStuff const& stuff, int tn, int t,
    bool waitNeeded, ncclSymMemBarrierSession<ncclCoopCta>& bar,
    Red red, ncclSymPtr<char> input, ncclSymPtr<char> output, int32_t nIters
  ) {
  using Pack = BytePack<BytePerPack>;
  using Acc = typename Red::EltType;
  using AccPack = BytePack<BytePerPack*sizeof(Acc)/sizeof(T)>;

  ncclSymTeam world = ncclSymTeamWorld(stuff.comm);
  int wn = tn/WARP_SIZE;
  int w = t/WARP_SIZE;
  int lane = t%WARP_SIZE;
  int const& rank = stuff.comm.rank;
  int const& nRanks = stuff.comm.nRanks;

  ncclSymPtr<Pack> inpPacks = (ncclSymPtr<Pack>)input + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  Pack acc0[UnrollPacks];

  nIters -= w;
  if (0 < nIters) {
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      acc0[u] = inpPacks.peerPtr(world, rank)[u*WARP_SIZE];
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

  if (0 < nIters) {
    while (true) {
      AccPack acc1[UnrollPacks];
      int r = rank;
      if (++r == nRanks) r = 0;
      { Pack tmp1[UnrollPacks];
        #pragma unroll
        for (int u=0; u < UnrollPacks; u++) {
          tmp1[u] = inpPacks.peerPtr(world, r)[u*WARP_SIZE];
        }
        #pragma unroll
        for (int u=0; u < UnrollPacks; u++) {
          acc1[u] = applyReduce(red, applyCast<T, Acc>(acc0[u]), applyCast<T, Acc>(tmp1[u]));
        }
      }

      if (++r == nRanks) r = 0;

      int dr = 2;
      #pragma unroll 2
      for (int partial=0; partial <= 1; partial++) {
        #pragma unroll 1
        for (int i = 0;
             partial ? i < 1 : (dr + UnrollPeers <= nRanks);
             partial ? i++ : (dr += UnrollPeers)) {
          if (partial && dr == nRanks) break;

          Pack tmp1[UnrollPeers][UnrollPacks];
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && ur!=0 && dr+ur == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              tmp1[ur][u] = inpPacks.peerPtr(world, r)[u*WARP_SIZE];
            }
            if (++r == nRanks) r = 0;
          }
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && ur!=0 && dr+ur == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              acc1[u] = applyReduce(red, acc1[u], applyCast<T, Acc>(tmp1[ur][u]));
            }
          }
        }
      }

      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) acc0[u] = applyCast<Acc, T>(acc1[u]);

      dr = 0;
      r = rank;
      #pragma unroll 2
      for (int partial=0; partial <= 1; partial++) {
        #pragma unroll 1
        for (int i = 0;
             partial ? i < 1 : (dr + UnrollPeers <= nRanks);
             partial ? i++ : (dr += UnrollPeers)) {
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && dr == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              outPacks.peerPtr(world, r)[u*WARP_SIZE] = acc0[u];
            }
            if (++r == nRanks) r = 0;
          }
        }
      }
      
      inpPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      outPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;

      // Load data for next iteration.
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        acc0[u] = inpPacks.peerPtr(world, rank)[u*WARP_SIZE];
      }
    }
  }
}

template<int UnrollPeers, typename Red, typename T>
static __device__ __forceinline__ void allreduceEnds(
    ncclSymkKernelStuff const& stuff, int tn, int t, Red red,
    ncclSymPtr<T> input, ncclSymPtr<T> output,
    size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  using Acc = typename Red::EltType;

  ncclSymTeam world = ncclSymTeamWorld(stuff.comm);
  int const& rank = stuff.comm.rank;
  int const& nRanks = stuff.comm.nRanks;

  ncclSymPtr<BytePack<sizeof(T)>> inpPacks = (ncclSymPtr<BytePack<sizeof(T)>>)input;
  ncclSymPtr<BytePack<sizeof(T)>> outPacks = (ncclSymPtr<BytePack<sizeof(T)>>)output;

  #pragma unroll 1
  for (size_t i = t; i < nPreElts+nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts-nSufElts-nPreElts+i;
    BytePack<sizeof(T)> acc0 = inpPacks.peerPtr(world, rank)[elt];
    BytePack<sizeof(Acc)> acc1;
    BytePack<sizeof(T)> tmp[UnrollPeers];
    int dr = 1;
    int r = rank+1;
    if (nRanks == r) r = 0;
    bool first = true;

    #pragma unroll 2
    for (int partial=0; partial <= 1; partial++) {
      #pragma unroll 1
      for (int j = 0;
           partial ? j < 1 : (dr + UnrollPeers <= nRanks);
           partial ? j++ : (dr += UnrollPeers)) {
        if (partial && dr == nRanks) break;

        #pragma unroll
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && u!=0 && dr+u == nRanks) break;
          tmp[u] = inpPacks.peerPtr(world, r)[elt];
          r += 1;
          if (r == nRanks) r = 0;
        }
        if (first) {
          first = false;
          acc1 = applyCast<T, Acc>(acc0);
        }
        #pragma unroll
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && u!=0 && dr+u == nRanks) break;
          acc1 = applyReduce(red, acc1, applyCast<T, Acc>(tmp[u]));
        }
      }
    }

    acc0 = applyCast<Acc, T>(acc1);
    dr = 0;
    r = rank;
    #pragma unroll 2
    for (int partial=0; partial <= 1; partial++) {
      #pragma unroll 1
      for (int j=0;
           partial ? j < 1 : (dr + UnrollPeers <= nRanks);
           partial ? j++ : (dr += UnrollPeers)) {
        #pragma unroll
        for (int u=0; u < UnrollPeers-partial; u++) {
          if (partial && dr+u == nRanks) break;
          outPacks.peerPtr(world, r)[elt] = acc0;
          r += 1;
          if (r == nRanks) r = 0;
        }
      }
    }
  }
}

template<typename Red, typename T>
static __device__ void allreduce(
    ncclSymkKernelStuff const& stuff, int tn, int t,
    bool waitNeeded, ncclSymMemBarrierSession<ncclCoopCta>& bar,
    Red red, ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {
  int nRanks = stuff.comm.nRanks;
  int nBlocks = stuff.nBlocks;
  size_t nBytes = nElts*sizeof(T);

  uint32_t nPreBytes = (16u - input.offset)%16u;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  constexpr int MinWarpPerBlock = 4;

  if ((input.offset - output.offset)%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 2;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nRanks*nBlocks, stuff.nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      allreduceDeep<BytePerPack, UnrollPacks, UnrollPeers, T>(
        stuff, tn, t, waitNeeded, bar, red,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && (input.offset - output.offset)%4 == 0)) {
    constexpr int BytePerPack = 4, UnrollPacks = 4, UnrollPeers = 4;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nRanks*nBlocks, stuff.nRanks_nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      allreduceDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers, T>(
        stuff, tn, t, waitNeeded, bar, red,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes-cursor)/sizeof(T);
  allreduceEnds<UnrollPeers>(stuff, tn, t, red, input, output, nElts, nPreBytes/sizeof(T), nSufElts);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLD_AGxST(ncclSymkDevArgs const* args) {
  ncclSymkKernelStuff stuff{args};
  ncclSymMemBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), args->comm, ncclSymTeamTagNear(), blockIdx.x
  };

  int rank = args->comm.rank;
  int nRanks = args->comm.nRanks;
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(args->redOpArg);

  // Threads numbered globally such that we round robin warps by rank then block.
  int gt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                     rank, nRanks,
                     blockIdx.x, gridDim.x,
                     threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
  int gtn = nRanks*gridDim.x*blockDim.x;

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);

  allreduce(
    stuff, gtn, gt, /*waitNeeded=*/true, bar, red,
    ncclSymPtr<T>(args->inputWin, args->inputOff),
    ncclSymPtr<T>(args->outputWin, args->outputOff),
    args->nElts
  );

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template<typename Red, typename T>
static __device__ void allreduceMultimem(
    int tn, int t, Red red, T* input, T* output, size_t nElts
  ) {
  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input);
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output);
  size_t nBytes = nElts*sizeof(T);

  constexpr int BytePerPack = LoadMultimem_BigPackSize<Red>::BigPackSize;
  uint32_t nPreBytes = (BytePerPack - inputUptr)%BytePerPack;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t nSufBytes;

  if (alignof(T) == BytePerPack || (inputUptr-outputUptr)%BytePerPack == 0) {
    constexpr int UnrollPacks = 16*8/BytePerPack;
    constexpr int BytePerChunk = UnrollPacks*WARP_SIZE*BytePerPack;
    uintptr_t cursor = nPreBytes;
    int nChunks = (nBytes-cursor)/BytePerChunk;
    uintptr_t cursorAfter = cursor + uintptr_t(nChunks)*BytePerChunk;
    nSufBytes = nBytes - cursorAfter;
    cursor += (t/WARP_SIZE)*UnrollPacks*WARP_SIZE*BytePerPack;
    cursor += (t%WARP_SIZE)*BytePerPack;
    int nIters = nChunks - t/WARP_SIZE;
    #pragma unroll 1
    while (0 < nIters) {
      BytePack<BytePerPack> tmp[UnrollPacks];
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        tmp[u] = applyLoadMultimem<Red, BytePerPack>(red, inputUptr + cursor + u*WARP_SIZE*BytePerPack);
      }
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        multimem_st_global(outputUptr + cursor + u*WARP_SIZE*BytePerPack, tmp[u]);
      }
      cursor += tn*UnrollPacks*BytePerPack;
      nIters -= tn/WARP_SIZE;
    }
  } else {
    nPreBytes = 0;
    nSufBytes = nBytes;
  }

  // Get the prefix+suffix element one at a time.
  #pragma unroll 4
  for (uintptr_t i = t*sizeof(T); i < nPreBytes + nSufBytes; i += tn*sizeof(T)) {
    uintptr_t cursor = i < nPreBytes ? i : nBytes-nSufBytes+(i-nPreBytes);
    BytePack<sizeof(T)> val = applyLoadMultimem<Red, sizeof(T)>(red, inputUptr + cursor);
    multimem_st_global(outputUptr + cursor, val);
    cursor += tn*sizeof(T);
  }
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC(ncclSymkDevArgs const* args) {
  ncclSymMemBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), args->comm, ncclSymTeamTagNear(), blockIdx.x, /*multimem=*/true
  };
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/true>::Type> red(args->redOpArg);

  // Threads numbered globally such that we round robin warps by rank then block.
  int gt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                     args->comm.rank, args->comm.nRanks,
                     blockIdx.x, gridDim.x,
                     threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
  int gtn = args->comm.nRanks*gridDim.x*blockDim.x;

  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
  allreduceMultimem(
    gtn, gt, red,
    ncclSymPtr<T>(args->inputWin, args->inputOff).multimemPtr(args->comm.nearMultimem),
    ncclSymPtr<T>(args->outputWin, args->outputOff).multimemPtr(args->comm.nearMultimem),
    args->nElts
  );
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R_impl(ncclSymkDevArgs const* args, bool multimem) {
  ncclSymkKernelStuff stuff(args);
  ncclSymLLA2ASession<ncclCoopCta> lla2a(
    ncclCoopCta(), args->comm, ncclSymTeamTagNear(),
    blockIdx.x, ncclSymkMaxThreads, multimem
  );
  
  int rank = args->comm.rank;
  int nRanks = args->comm.nRanks;
  using Acc = typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type;
  Red<Acc> red(args->redOpArg);

  using Pack = BytePack<8>;
  using AccPack = BytePack<8*sizeof(Acc)/sizeof(T)>;
  constexpr int EltPerPack = 8/sizeof(T);
  int nElts = args->nElts;
  int nPacks = divUp(nElts, EltPerPack);

  bool packAligned = 8 <= alignof(T) || (
      args->nElts*sizeof(T) | (uint32_t)args->inputOff | (uint32_t)args->outputOff
    )%8 == 0;

  uint32_t nPackPerBlock, nPackModBlock;
  idivmodFast32(&nPackPerBlock, &nPackModBlock, nPacks, stuff.nBlocks, stuff.nBlocks_rcp32);
  int begin = blockIdx.x*nPackPerBlock + minval<int>(blockIdx.x, nPackModBlock);
  int end = begin + nPackPerBlock + (blockIdx.x < nPackModBlock ? 1 : 0);

  nPacks = end - begin;
  nElts -= begin*EltPerPack;
  nElts = min(nElts, nPacks*EltPerPack);
  T* input = (T*)ncclSymGetLocalPointer(args->inputWin, args->inputOff) + begin*EltPerPack;
  T* output = (T*)ncclSymGetLocalPointer(args->outputWin, args->outputOff) + begin*EltPerPack;

  ncclCoopCta cta;
  int t = threadIdx.x;
  int tn = ncclSymkMaxThreads;

  if (__builtin_expect(packAligned, true)) {
    #pragma unroll 1
    while (0 < nPacks) {
      if (t < nPacks) {
        int nIterPacks = min(nPacks, tn);
        Pack inp = loadPack<Pack>((Pack*)input, t, nPacks);
        lla2a.bcast(/*slot=*/nIterPacks*rank + t, inp);
        AccPack out = lla2a.template recvReduce</*Unroll=*/8, Pack>(
          /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
          /*eltToAcc=*/[&] __device__ (Pack x)->AccPack {
            return applyCast<T, Acc>(x);
          },
          /*reduce=*/[&] __device__ (AccPack a, AccPack b)->AccPack {
            return applyReduce(red, a, b);
          }
        );
        storePack((Pack*)output, t, nPacks, applyCast<Acc, T>(out));
      }
      lla2a.endEpoch(cta);

      input += tn*EltPerPack;
      output += tn*EltPerPack;
      nPacks -= tn;
    }
  } else {
    #pragma unroll 1
    while (0 < nElts) {
      if (t*EltPerPack < nElts) {
        int nIterPacks = min(nPacks, tn);
        Pack inp = loadPack<Pack>(input, t*EltPerPack, nElts);
        lla2a.bcast(/*slot=*/nIterPacks*rank + t, inp);
        AccPack out = lla2a.template recvReduce</*Unroll=*/8, Pack>(
          /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
          /*eltToAcc=*/[&] __device__ (Pack x)->AccPack {
            return applyCast<T, Acc>(x);
          },
          /*reduce=*/[&] __device__ (AccPack a, AccPack b)->AccPack {
            return applyReduce(red, a, b);
          }
        );
        storePack(output, t*EltPerPack, nElts, applyCast<Acc, T>(out));
      }
      lla2a.endEpoch(cta);

      input += tn*EltPerPack;
      output += tn*EltPerPack;
      nElts -= tn*EltPerPack;
      nPacks -= tn;
    }
  }
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R(ncclSymkDevArgs const* args) {
  ncclSymkRun_AllReduce_AGxLL_R_impl<Red, T>(args, /*multimem=*/false);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLLMC_R(ncclSymkDevArgs const* args) {
  ncclSymkRun_AllReduce_AGxLL_R_impl<Red, T>(args, /*multimem=*/true);
}
