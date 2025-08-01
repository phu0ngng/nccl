#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"

template<int BytePerPack, int UnrollPacks, int UnrollPeers>
static __device__ void bcastDeep(
    ncclSymkKernelStuff const& stuff, int tn, int t,
    bool waitNeeded, ncclSymMemBarrierSession<ncclCoopCta>& bar,
    ncclSymPtr<char> input, ncclSymPtr<char> output, bool inPlace, int nIters
  ) {
  using Pack = BytePack<BytePerPack>;
  int wn = tn/WARP_SIZE;
  int w = t/WARP_SIZE;
  int lane = t%WARP_SIZE;
  int const& rank = stuff.comm.rank;
  int const& nRanks = stuff.comm.nRanks;

  Pack* inpPacks = (Pack*)input.localPtr() + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  Pack tmp[UnrollPacks];

  nIters -= w;
  if (0 < nIters) {
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      tmp[u] = inpPacks[u*WARP_SIZE];
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

  if (0 < nIters) {
    while (true) {
      int dr = inPlace ? 1 : 0;
      int r = rank + dr;
      if (r == nRanks) r = 0;
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
              outPacks.nearPtr(r)[u*WARP_SIZE] = tmp[u];
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
        tmp[u] = inpPacks[u*WARP_SIZE];
      }
    }
  }
}

template<int UnrollPeers, typename T>
static __device__ void bcastEnds(
    ncclSymkKernelStuff const& stuff, int tn, int t,
    ncclSymPtr<T> input, ncclSymPtr<T> output, bool inPlace, size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  int const& rank = stuff.comm.rank;
  int const& nRanks = stuff.comm.nRanks;
  BytePack<sizeof(T)>* inpPacks = (BytePack<sizeof(T)>*)input.localPtr();
  ncclSymPtr<BytePack<sizeof(T)>> outPacks = (ncclSymPtr<BytePack<sizeof(T)>>)output;
  #pragma unroll 1
  for (size_t i = t; i < nPreElts+nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts-nPreElts-nSufElts+i;
    BytePack<sizeof(T)> tmp = inpPacks[elt];
    int dr = inPlace ? 1 : 0;
    int r = rank + dr;
    if (r == nRanks) r = 0;
    #pragma unroll 1
    for (; dr + UnrollPeers <= nRanks; dr += UnrollPeers) {
      #pragma unroll UnrollPeers
      for (int u=0; u < UnrollPeers; u++) {
        outPacks.nearPtr(r)[elt] = tmp;
        if (++r == nRanks) r = 0;
      }
    }
    #pragma unroll UnrollPeers
    for (int u=0; u < UnrollPeers; u++) {
      if (dr+u == nRanks) break;
      outPacks.nearPtr(r)[elt] = tmp;
      if (++r == nRanks) r = 0;
    }
  }
}

template<typename T>
static __device__ void bcast(
    ncclSymkKernelStuff const& stuff, int tn, int t,
    bool waitNeeded, ncclSymMemBarrierSession<ncclCoopCta>& bar,
    ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {
  bool inPlace = (input == output);
  size_t nBytes = nElts*sizeof(T);

  uint32_t nPreBytes = (16 - input.offset)%16;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  constexpr int MinWarpPerBlock = 4;

  if ((input.offset - output.offset)%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 2;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, stuff.nBlocks, stuff.nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      bcastDeep<BytePerPack, UnrollPacks, UnrollPeers>(
        stuff, tn, t, waitNeeded, bar,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        inPlace, chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && (input.offset - output.offset)%4 == 0)) {
    constexpr int BytePerPack = 4, UnrollPacks = 4, UnrollPeers = 4;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, stuff.nBlocks, stuff.nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      bcastDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers>(
        stuff, tn, t, waitNeeded, bar,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        inPlace, chunks*MinWarpPerBlock
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes-cursor)/sizeof(T);
  bcastEnds<UnrollPeers>(stuff, tn, t, input, output, inPlace, nElts, nPreBytes/sizeof(T), nSufElts);
}

__device__ __forceinline__ void ncclSymkRun_AllGather_ST(ncclSymkDevArgs const* args) {
  ncclSymkKernelStuff stuff{args};
  ncclSymMemBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), args->comm, ncclTeamTagNear(), blockIdx.x
  };
  int const& rank = args->comm.rank;

  // Threads numbered over rank.
  int bt = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                     blockIdx.x, gridDim.x,
                     threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
  int btn = gridDim.x*blockDim.x;

  bar.arrive(ncclCoopCta(), cuda::memory_order_relaxed);

  bcast(stuff, btn, bt, /*waitNeeded=*/true, bar,
    ncclSymPtr<char>(args->inputWin, args->inputOff),
    ncclSymPtr<char>(args->outputWin, args->outputOff) + rank*args->nElts,
    args->nElts);

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template<typename T>
static __device__ void bcastMultimem(
    ncclSymkKernelStuff& stuff, int tn, int t, ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {
  size_t nBytes = nElts*sizeof(T);
  uintptr_t inputUptr = reinterpret_cast<uintptr_t>(input.localPtr());
  uintptr_t outputUptr = reinterpret_cast<uintptr_t>(output.multimemPtr(stuff.comm.nearMultimem));
  uint32_t nPreBytes = (16 - input.offset)%16;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t nSufBytes;

  if ((inputUptr-outputUptr)%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 8;
    constexpr int BytePerChunk = UnrollPacks*WARP_SIZE*BytePerPack;
    uintptr_t cursor = nPreBytes;
    uint32_t nChunks = (nBytes-cursor)/BytePerChunk;
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
        tmp[u] = *reinterpret_cast<BytePack<BytePerPack>*>(inputUptr + cursor + u*WARP_SIZE*BytePerPack);
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
    BytePack<sizeof(T)> val = *reinterpret_cast<BytePack<sizeof(T)>*>(inputUptr + cursor);
    multimem_st_global(outputUptr + cursor, val);
    cursor += tn*sizeof(T);
  }
}

__device__ __forceinline__ void ncclSymkRun_AllGather_STMC(ncclSymkDevArgs const* args) {
  ncclSymkKernelStuff stuff{args};
  ncclSymMemBarrierSession<ncclCoopCta> bar(
    ncclCoopCta(), args->comm, ncclTeamTagNear(), blockIdx.x, /*multimem=*/true
  );
  int const& rank = args->comm.rank;

  ncclSymPtr<char> input(args->inputWin, args->inputOff);
  ncclSymPtr<char> output(args->outputWin, args->outputOff);
  size_t bytes = args->nElts;
  // Round robin memory to blocks.
  int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                    blockIdx.x, gridDim.x,
                    threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
  int tn = gridDim.x*blockDim.x;

  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  bcastMultimem(stuff, tn, t, input, output + rank*bytes, bytes);

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

template<typename EltType>
static __device__ void allgather_LL_body(
    ncclSymkKernelStuff& stuff, ncclSymLLA2ASession<ncclCoopCta>& lla2a,
    EltType* input, EltType* output, int nElts, int nPacks, int nStrideElts
  ) {
  using Pack = BytePack<8>;
  constexpr int EltPerPack = 8/sizeof(EltType);
  int rank = stuff.comm.rank;
  int nRanks = stuff.comm.nRanks;
  int t = threadIdx.x;
  constexpr int tn = ncclSymkMaxThreads;

  #pragma unroll 1
  while (0 < nElts) {
    int nIterPacks = min(nPacks, tn);
    if (t < nIterPacks) {
      Pack x = loadPack<Pack>(input, t*EltPerPack, nElts);
      lla2a.bcast(/*slot=*/nIterPacks*rank + t, x);
    }

    int tn_div_nPacks = tn/nIterPacks;
    int tn_mod_nPacks = tn%nIterPacks;
    int peer = t/nIterPacks;
    int pack = t%nIterPacks;
    #if 1
      // NOTE: Unrolling speedup on eos nranks=8 size=64K: 5.7us vs 6.7us
      constexpr int Unroll = 4;
      #pragma unroll 1
      for (int i = t; i < (nRanks*nIterPacks & -(Unroll*tn)); i += Unroll*tn) {
        Pack got[Unroll];
        lla2a.template recvUnrolled<Unroll, Unroll>(i, Unroll, tn, /*&*/got);
        #pragma unroll
        for (int u=0; u < Unroll; u++) {
          storePack<Pack>(output + peer*nStrideElts, pack*EltPerPack, nElts, got[u]);
          peer += tn_div_nPacks;
          pack += tn_mod_nPacks;
          if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
        }
      }

      int i = (nRanks*nIterPacks & -(Unroll*tn)) + t;
      int n = (nRanks*nIterPacks)/tn % Unroll;
      if (i + n*tn < nRanks*nIterPacks) n += 1;
      if (n != 0) {
        Pack got[Unroll];
        lla2a.template recvUnrolled<1, Unroll>(i, n, tn, /*&*/got);
        #pragma unroll
        for (int u=0; u < Unroll; u++) {
          if (u != 0 && u == n) break;
          storePack(output + peer*nStrideElts, pack*EltPerPack, nElts, got[u]);
          peer += tn_div_nPacks;
          pack += tn_mod_nPacks;
          if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
        }
      }
    #else
      // The non-unrolled but "obviously correct" implementation for reference.
      #pragma unroll 1
      for (int i = t; i < nRanks*nIterPacks; i += tn) {
        Pack got = lla2a.template recv<Pack>(i);
        storePack(output + peer*nStrideElts, pack*EltPerPack, nElts, got);
        peer += tn_div_nPacks;
        pack += tn_mod_nPacks;
        if (nIterPacks <= pack) { peer += 1; pack -= nIterPacks; }
      }
    #endif

    lla2a.endEpoch(ncclCoopCta());

    input += tn*EltPerPack;
    output += tn*EltPerPack;
    nElts -= tn*EltPerPack;
    nPacks -= tn;
  }
}

static __device__ void ncclSymkRun_AllGather_LL_impl(ncclSymkDevArgs const* args, bool multimem) {
  ncclSymkKernelStuff stuff(args);
  ncclSymLLA2ASession<ncclCoopCta> lla2a(
    ncclCoopCta(), stuff.comm, ncclTeamTagNear(), blockIdx.x, /*maxElts=*/ncclSymkMaxThreads, multimem
  );

  using Pack = BytePack<8>;
  constexpr int BytePerPack = 8;
  int nElts = args->nElts;
  int nPacks = divUp(nElts, BytePerPack);

  uint32_t nPackPerBlock, nPackModBlock;
  idivmodFast32(&nPackPerBlock, &nPackModBlock, nPacks, stuff.nBlocks, stuff.nBlocks_rcp32);
  int blockPackBegin = blockIdx.x*nPackPerBlock + minval<int>(blockIdx.x, nPackModBlock);
  int blockPackEnd = blockPackBegin + nPackPerBlock + (blockIdx.x < nPackModBlock ? 1 : 0);
  int nBlockPacks = blockPackEnd - blockPackBegin;
  int nBlockElts = nElts - blockPackBegin*BytePerPack;
  nBlockElts = min(nBlockElts, nBlockPacks*BytePerPack);
  char* blockInput = ncclSymPtr<char>(args->inputWin, args->inputOff).localPtr() + blockPackBegin*BytePerPack;
  char* blockOutput = ncclSymPtr<char>(args->outputWin, args->outputOff).localPtr() + blockPackBegin*BytePerPack;

  uint32_t lowBits = args->nElts;
  lowBits |= (uint32_t)args->inputOff;
  lowBits |= (uint32_t)args->outputOff;
  if (__builtin_expect(lowBits%8 == 0, true)) {
    // NOTE: Specializing for 8-byte alignment in one case help at size=65K: 8.9us vs 5.6us
    allgather_LL_body(stuff, lla2a, (BytePack<8>*)blockInput, (BytePack<8>*)blockOutput, nBlockElts/8, nBlockPacks, nElts/8);
  } else {
    allgather_LL_body(stuff, lla2a, blockInput, blockOutput, nBlockElts, nBlockPacks, nElts);
  }
}

__device__ __forceinline__ void ncclSymkRun_AllGather_LL(ncclSymkDevArgs const* args) {
  ncclSymkRun_AllGather_LL_impl(args, /*multimem=*/false);
}

__device__ __forceinline__ void ncclSymkRun_AllGather_LLMC(ncclSymkDevArgs const* args) {
  ncclSymkRun_AllGather_LL_impl(args, /*multimem=*/true);
}
