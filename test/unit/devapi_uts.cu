#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "bitops.h"
#include "common.h"

#include <vector>
#include <cassert>

constexpr bool Prints = false;
constexpr int MaxGlobalBlocks = 1024;
constexpr int BlockPerRank = 4;

struct Hash {
  uint64_t a, b;
  __host__ __device__ Hash mix(uint64_t c) {
    uint64_t a = this->a, b = this->b;
    a += b ^ c ^ 0x12345678;
    a ^= a>>32;
    a *= 0x9e3779b97f4a7c15;
    b ^= a;
    b ^= b>>32;
    b *= 0x9002c923e4c9ef35;

    a -= b;
    a ^= a>>32;
    a *= 0x2166b8d4b10b4329;
    b ^= a;
    b ^= b>>32;
    b *= 0xc9be6464ff473853;
    return {a, b};
  }
  __host__ __device__ Hash& operator+=(Hash x) {
    a += x.a;
    b += x.b;
    return *this;
  }
};

__host__ __device__ int getOwner(int nBlocks, Hash h) {
  return unsigned(h.b) % unsigned(nBlocks);
}

#if 1 // Random distribution of nodes

__host__ __device__ int getRootCount(int block) {
  return 100;
}
__host__ __device__ Hash getRoot(int block, int index) {
  Hash h;
  // h.a = index*0x1f1659ab6daba57b + block*0xb694070de1846119; // Compiler bug? Causes runHost to add roots forever until it crashes!
  h.a = index*0xb694070de1846119 + block*0xe16044f66407281d; // the "fix"
  h.a ^= h.a>>32;
  h.b = block; // getOwner(h) == block
  return h;
}

__host__ __device__ int getKidCount(int gen, int block, Hash h) {
  // First 10 generations are ~1.5X the size of predecessor.
  // Then next 10 are ~0.5X of predecessor.
  // Then nothing.
  return h.a & (gen < 10 ? 3 : gen < 20 ? 1 : 0);
}

__host__ __device__ Hash getKidHash(int gen, int block, Hash par, int kid) {
  Hash ret = par;
  ret = ret.mix(uint64_t(gen)<<32 | kid);
  return ret;
}

#else // Predictable distribution of nodes (for debug)

__host__ __device__ int getRootCount(int block) {
  return 1;
}
__host__ __device__ Hash getRoot(int block, int index) {
  Hash h;
  h.a = 0xbeef;
  h.b = block; // getOwner(h) == block
  return h;
}

__host__ __device__ int getKidCount(int gen, int block, Hash h) {
  // Double node counts for 10 rounds then nothing
  return (gen < 10 ? 2 : 0);
}

__host__ __device__ Hash getKidHash(int gen, int block, Hash par, int kid) {
  // Nodes are just rotated each generation.
  Hash ret = par;
  ret.b += 1;
  return ret;
}
#endif

void runHost(int rank, int nBlocks, Hash* outSums, int* outMaxPileSize, int* outMaxInboxSize) {
  int* pileSize = new int[nBlocks];
  int* inboxSize = new int[nBlocks*nBlocks];

  // initialize batch
  std::vector<Hash> batch;
  for (int b=0; b < nBlocks; b++) {
    outSums[b] = {0, 0};
    int rn = getRootCount(b);
    for (int i=0; i < rn; i++) {
      batch.push_back(getRoot(b, i));
    }
  }
  *outMaxPileSize = 0;
  *outMaxInboxSize = 0;
  
  int gen = 0;
  do {
    for (int b=0; b < nBlocks; b++) pileSize[b] = 0;
    for (int bb=0; bb < nBlocks*nBlocks; bb++) inboxSize[bb] = 0;
    std::vector<Hash> next;
    for (Hash h: batch) {
      int b = getOwner(nBlocks, h);
      outSums[b].a += h.a;
      outSums[b].b += 1;
      int nKids = getKidCount(gen, b, h);
      pileSize[b] += 1; // for this node
      pileSize[b] += nKids; // for its kids
      *outMaxPileSize = std::max(*outMaxPileSize, pileSize[b]);
      for (int k=0; k < nKids; k++) {
        Hash kh = getKidHash(gen, b, h, k);
        int kb = getOwner(nBlocks, kh);
        inboxSize[b*nBlocks + kb] += 1;
        *outMaxInboxSize = std::max(*outMaxInboxSize, inboxSize[b*nBlocks + kb]);
        next.push_back(kh);
      }
    }
    if (Prints && rank==0) {
      for (int b=0; b < nBlocks; b++) {
        int recvd = 0;
        for (int b1=0; b1 < nBlocks; b1++) recvd += inboxSize[b1*nBlocks + b];
        printf("HOST gen=%d b=%d recvd=%d\n", gen, b, recvd);
      }
    }
    batch = std::move(next);
    gen += 1;
  } while (!batch.empty());

  *outMaxPileSize = 1 << log2Up(*outMaxPileSize);
  delete[] pileSize;
  delete[] inboxSize;
}

struct Args {
  ncclDevComm comm;
  ncclSymPtr<Hash> inboxBase;
  int inboxSize;
  ncclSymPtr<Hash> pileBase;
  int pileSize;
  uint32_t barrierSignal;
  uint32_t inboxSignal;
  Hash* outSums;
};

struct Barrier {
  ncclDevComm const& comm;
  ncclGin const& net;
  unsigned signalBase;
  unsigned phase;
  __device__ Barrier(ncclDevComm const& comm, ncclGin const& net, int signalBase):
    comm(comm), net(net), signalBase(signalBase) {
    phase = 0;
  }
  __device__ bool anySync(bool val) {
#if __CUDA_ARCH__ >= 700
    int t = threadIdx.x;
    int tn = blockDim.x;
    int bn = BlockPerRank*comm.nRanks;
    bool ret;
    // Do barrier random number of times just for stress.
    int iters = 2 + (phase*0xdeadbeef >> 27);
    #pragma unroll 1
    for (int iter=0; iter < iters; iter++) {
      __syncthreads();
      if (t == 0) net.resetSignal(signalBase + 2*blockIdx.x + (phase%2 ^ 1));
      __syncthreads();
      // contribute to barrier
      #pragma unroll 1
      for (int b=t; b < bn; b += tn) {
        unsigned sig = signalBase + 2*(b%BlockPerRank) + phase%2;
        net.signal(ncclTeamWorld(comm), b/BlockPerRank, ncclGin_SignalAdd{sig, uint64_t(1)<<32 | (val ? 1 : 0)});
      }
      // wait for barrier
      unsigned sig = signalBase + 2*blockIdx.x + phase%2;
      if (t == 0) net.waitSignal(ncclCoopThread(), sig, uint64_t(bn)<<32);
      __syncthreads();
      uint64_t got = net.readSignal(sig, 64, cuda::memory_order_relaxed);
      ret = uint32_t(got) != 0;
      phase += 1;
    }
    return ret;
#else
    return false;
#endif
  }
};

__global__ void runDevice(Args args) {
#if __CUDA_ARCH__ >= 700
  int t = threadIdx.x;
  int tn = blockDim.x;
  ncclDevComm& comm = args.comm;
  ncclTeam world = ncclTeamWorld(comm);
  int bme = comm.rank*BlockPerRank + blockIdx.x;
  int bn = comm.nRanks*BlockPerRank;

  ncclGin net(comm, 0);
  Barrier bar(args.comm, net, args.barrierSignal);

  ncclSymPtr<Hash> pile = args.pileBase + blockIdx.x*args.pileSize;
  uint32_t pileHead = 0, pileTail = 0;

  { // initialize pile
    #pragma unroll 1
    for (int i=t; i < getRootCount(bme); i += tn) {
      pile.localPtr()[pileTail + i] = getRoot(bme, i);
    }
    pileTail += getRootCount(bme);
  }
  
  int gen = 0;
  #pragma unroll 1
  while (1) {
    __shared__ uint32_t pileCounter;
    __shared__ int msgCounter[MaxGlobalBlocks];

    { // spawn next generation
      __syncthreads();
      if (t==0) pileCounter = 0;
      if (Prints && t==0 && pileTail-pileHead != 0) printf("DEV [%d] gen=%d pile=%d\n", bme, gen, pileTail-pileHead);
      __syncthreads();
      #pragma unroll 1
      for (uint32_t i=t; i < pileTail-pileHead; i += tn) {
        int p = (pileHead + i) & (args.pileSize-1);
        Hash h = pile.localPtr()[p];
        asm volatile(
          "red.relaxed.gpu.add.u64 [%0],%1;"
          "red.relaxed.gpu.add.u64 [%2],1;" ::
          "l"(&args.outSums[blockIdx.x].a), "l"(h.a),
          "l"(&args.outSums[blockIdx.x].b) /*, "l"(h.b)*/
        );
        int kn = getKidCount(gen, bme, h);
        #pragma unroll 1
        for (int k=0; k < kn; k++) {
          int j = atomicAdd_block(&pileCounter, 1);
          p = (pileTail + j) & (args.pileSize-1);
          pile.localPtr()[p] = getKidHash(gen, bme, h, k);
        }
      }
      __syncthreads();
      pileHead = pileTail;
      pileTail += pileCounter;
      assert(pileCounter <= args.pileSize);
      __syncthreads();
    }

    bar.anySync(false); // ensure inboxes are quiesced

    // scatter pile to inboxes
    __syncthreads();
    #pragma unroll 1
    for (int b=t; b < bn; b += tn) msgCounter[b] = 0;
    __syncthreads();

    bool pileIsEmpty = pileHead == pileTail;
    #pragma unroll 1
    for (uint32_t i=t; i < pileTail-pileHead; i += tn) {
      int p = (pileHead + i) & (args.pileSize-1);
      Hash h = pile.localPtr()[p];
      int b = getOwner(bn, h);
      int r = b/BlockPerRank;
      int m = atomicAdd_block(&msgCounter[b], 1);
      assert(m < args.inboxSize);
      net.put(
        world, r, args.inboxBase + ((b%BlockPerRank)*bn + bme)*args.inboxSize + m, pile + p, 1,
        ncclGin_SignalInc{args.inboxSignal + (b%BlockPerRank)*bn + bme}
      );
    }
    __syncthreads();
    if (Prints) {
      #pragma unroll 1
      for (int b=t; b < bn; b += tn) {
        if (msgCounter[b] != 0) printf("DEV gen=%d %d->%d sent=%d\n", gen, bme, b, msgCounter[b]);
      }
    }

    // Dense alltoall barrier ensures all prior puts have landed.
    //net.flush(ncclCoopCta());
    bool quit = !bar.anySync(!pileIsEmpty);
    if (quit) break;

    // gather inbox into pile
    pileHead = pileTail;
    __syncthreads();
    if (t==0) pileCounter = 0;
    __syncthreads();
    #pragma unroll 1
    for (int b=t/32; b < bn; b += tn/32) {
      int sig = args.inboxSignal + blockIdx.x*bn + b;
      uint64_t got = net.readSignal(sig);
      uint64_t* shadow = net.getSignalShadowPtr(sig);
      int nMsgs = got - *shadow;
      if (Prints && nMsgs != 0 && t%32 == 0) printf("DEV gen=%d %d->%d recvd=%d\n", gen, b, bme, nMsgs);
      __syncwarp();
      if (t%32 == 0) *shadow = got;
      __syncwarp();
      #pragma unroll 1
      for (int m = t%32; m < nMsgs; m += 32) {
        Hash h = args.inboxBase.localPtr()[(blockIdx.x*bn + b)*args.inboxSize + m];
        assert(getOwner(bn, h) == bme);
        int p = atomicAdd_block(&pileCounter, 1);
        p = (pileHead + p) & (args.pileSize-1);
        pile.localPtr()[p] = h;
      }
    }
    __syncthreads();
    pileTail = pileHead + pileCounter;
    __syncthreads();
    gen += 1;
  }
#endif
}

int main(int argc, char** argv) {
  int rank, nRanks;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  if (MaxGlobalBlocks < BlockPerRank*nRanks) {
    if (rank == 0) fprintf(stderr, "ERROR: Maximum rank limit exceeded.\n");
    return 1;
  }

  uint64_t* hosts = new uint64_t[nRanks];
  { char hostname[1024] = {};
    getHostName(hostname, 1024);
    hosts[rank] = getHostHash(hostname);
  }
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hosts, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  int dev = 0;
  for (int r=0; r < rank; r++) {
    if (hosts[r] == hosts[rank]) dev++;
  }
  delete[] hosts;

  int nGlobalBlocks = nRanks*BlockPerRank;
  Hash* sums = new Hash[nGlobalBlocks];
  int maxPileSize, maxInboxSize;
  runHost(rank, nGlobalBlocks, sums, &maxPileSize, &maxInboxSize);
  if (Prints && rank==0) printf("Max pile size=%d\nMax inbox size=%d\n", maxPileSize, maxInboxSize);

  ncclUniqueId id;
  ncclComm_t comm;
  if (rank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  CUDACHECK(cudaSetDevice(dev));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;                  
  config.blocking = 1;                                               
  NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, rank, &config));

  void* buf;
  size_t winSize = BlockPerRank*maxPileSize; // piles
  winSize += BlockPerRank*nGlobalBlocks*maxInboxSize; // inboxes
  winSize += BlockPerRank; // validation checksums
  winSize *= sizeof(Hash);
  ncclMemAlloc(&buf, winSize);
  CUDACHECK(cudaMemset(buf, 0, winSize));

  ncclWindow_t win; 
  NCCLCHECK(ncclCommWindowRegister(comm, buf, winSize, &win, 0));

  Args args;
  { ncclDevCommRequirements reqs = {};
    reqs.ginSignalCount = 2*BlockPerRank; // barrier signals
    reqs.ginSignalCount += BlockPerRank*nGlobalBlocks; // inbox signals
    reqs.ginForceEnable = true;
    NCCLCHECK(ncclDevCommCreate(comm, &reqs, &args.comm));
  }
  args.pileBase = ncclSymPtr<Hash>(win, 0);
  args.pileSize = maxPileSize;
  args.inboxBase = args.pileBase + BlockPerRank*maxPileSize;
  args.inboxSize = maxInboxSize;
  args.barrierSignal = 0;
  args.inboxSignal = 2*BlockPerRank;
  args.outSums = (Hash*)buf + BlockPerRank*maxPileSize + BlockPerRank*nGlobalBlocks*maxInboxSize;

  // run kernel
  printf("[MPI Rank %d] Starting kernel\n", rank);
  runDevice<<<BlockPerRank, 512, 0, stream>>>(args);
  CUDACHECK(cudaStreamSynchronize(stream));
  printf("[MPI Rank %d] Completed kernel\n", rank);

  bool validated = true;
  { // validate
    Hash got[BlockPerRank];
    CUDACHECK(cudaMemcpy(got, args.outSums, BlockPerRank*sizeof(Hash), cudaMemcpyDeviceToHost));
    for (int cta=0; cta < BlockPerRank; cta++) {
      int b = rank*BlockPerRank + cta;
      if (sums[b].a != got[cta].a || sums[b].b != got[cta].b) {
        validated = false;
        fprintf(stderr, "[MPI Rank %d] ERROR: Validation failed cta=%d gotHash=%lx wantHash=%lx gotCount=%ld wantCount=%ld\n", rank, cta, (unsigned long)got[cta].a, (unsigned long)sums[b].a, (unsigned long)got[cta].b, (unsigned long)sums[b].b);
      }
    }
  }

  // cleanup
  delete[] sums;
  NCCLCHECK(ncclCommWindowDeregister(comm, win));
  NCCLCHECK(ncclMemFree(buf));
  NCCLCHECK(ncclDevCommDestroy(comm, &args.comm));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  MPICHECK(MPI_Finalize());

  if (validated) printf("[MPI Rank %d] Success\n", rank);
  return validated ? 0 : 1;
}
