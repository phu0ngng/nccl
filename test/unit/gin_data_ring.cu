//#define NCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG 1

#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "gin/gin_host.h"
#include "nccl_device/gin/gin_device_api.h"

#include "comm.h" // Needed to extract GIN handle from communicator.
#undef NCCLCHECK  // Undefine CHECK macros which comm.h brings, to
#undef CUDACHECK  // avoid conflict with the ones in common.h below.

#include "common.h"
#include <cassert>

//constexpr int BlockPerRank = 4;
constexpr int BlockPerRank = 16;

//constexpr int BufElts = 10;
//constexpr int BufElts = 129;
constexpr int BufElts = 1<<20;

__global__ void runDevice(ncclGinCtx_M<-1u> ctx, ncclGinWindow_t win, int* buf) {
#if __CUDA_ARCH__ >= 700
  int nRanks = ctx.nRanks;
  int rank = ctx.rank;
  int t = threadIdx.x;
  int tn = blockDim.x;
  int b = rank*gridDim.x + blockIdx.x;
  int bn = nRanks*gridDim.x;
  int up_rank = (b+bn-1)%bn / gridDim.x;
  int up_block = (b+bn-1)%bn % gridDim.x;
  int down_rank = (b+1)%bn / gridDim.x;
  int down_block = (b+1)%bn % gridDim.x;
  auto acq = cuda::memory_order_acquire;

  const int sendOff = 0;
  const int recvOff = gridDim.x*BufElts;
  unsigned sigData0 = 0;
  unsigned sigFree0 = gridDim.x;
  int accum = 0;
  for (uint32_t round=0; round < 10; round++) {
    if (t==0 && b == int(round%bn)) printf("Round %d\n", round);
    // Consume recv buf (rbuf) and populate send buf (sbuf)
    for (int i=t; i < BufElts; i += tn) {
      int got = (buf + recvOff + blockIdx.x*BufElts)[i];
      if (got != (int)round) {
        printf("[%d:%d] round=%d got=%d want=%d\n", rank, blockIdx.x, round, got, round);
      }
      assert(got == (int)round);
      (buf + sendOff + blockIdx.x*BufElts)[i] = 1 + got; // Thus data received should equal its round index
    }

    __syncthreads();
    if (t==0) {
      // Send signal upstream to indicate free space
      ncclGinSignalDescriptor signal;
      signal.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
      signal.indexedSignal.signalId = sigFree0 + up_block;
      ncclGinCall<ncclGinApi_Put>(ctx, ncclCoopThread(), up_rank,
        /*hasData=*/false, nullptr, 0, nullptr, 0, 0,
        signal, ncclGinSignalInc, 0,
        /*hasCounter=*/false, 0,
        /*hasDescriptor=*/false, nullptr,
        cuda::thread_scope_thread, cuda::thread_scope_thread);
      // Wait for downstream to give us free space
      uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, sigFree0 + blockIdx.x);
      cuda::atomic_ref<uint64_t> ref{*ptr};
      while (ref.load(acq) < 1+round) continue;
    }
    __syncthreads();

    // Send data downstream in chunks.
    int nChunks = min(BufElts, 1 + (round*0xdeadbeefu >> (32-10)));
    int chunkElts = BufElts/nChunks;
    nChunks = (BufElts + chunkElts-1)/chunkElts;
    #pragma unroll 1
    for (int i=t; i < nChunks; i += tn) {
      ncclGinSignalDescriptor signal;
      signal.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
      signal.indexedSignal.signalId = sigData0 + down_block;
      ncclGinCall<ncclGinApi_Put>(ctx, ncclCoopThread(), down_rank, /*hasData=*/true,
        win, (recvOff + down_block*BufElts + i*chunkElts)*sizeof(int),
        win, (sendOff + blockIdx.x*BufElts + i*chunkElts)*sizeof(int),
        min(chunkElts, BufElts - i*chunkElts)*sizeof(int),
        signal, ncclGinSignalInc, 0,
        /*hasCounter=*/false, 0,
        /*hasDescriptor=*/false, nullptr,
        cuda::thread_scope_thread, cuda::thread_scope_device);
    }
    __syncthreads();
    if (t==0) {
      uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, sigData0 + blockIdx.x);
      cuda::atomic_ref<uint64_t> ref{*ptr};
      while (ref.load(acq) < accum + nChunks) continue;
    }
    __syncthreads();
    // Wait for outgoing is complete.
    ncclGinCall<ncclGinApi_Flush>(ctx, ncclCoopCta(), acq);
    __syncthreads();
    accum += nChunks;
  }
#endif
}

int main(int argc, char** argv) {
  int rank, nRanks;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  uint64_t* hosts = new uint64_t[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hosts[rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hosts, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  int dev = 0;
  for (int r=0; r < rank; r++) {
    if (hosts[r] == hosts[rank]) dev++;
  }
  delete[] hosts;

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
  NCCLCHECK(ncclGinConnectOnce(comm));
  uint32_t sigs;
  NCCLCHECK(ncclGinAllocSignalsCounters(comm, 2*BlockPerRank, &sigs, 0, nullptr));
  assert(sigs == 0);

  // Allocate and register symmetric memory
  void *buf;
  size_t bufSize = 2*BlockPerRank*BufElts*sizeof(int);
  NCCLCHECK(ncclMemAlloc((void**)&buf, bufSize));
  CUDACHECK(cudaMemset(buf, 0, bufSize));
  CUDACHECK(cudaDeviceSynchronize());

  // Get window handles
  void* hostWins[NCCL_GIN_MAX_CONTEXTS];
  ncclGinWindow_t devWins[NCCL_GIN_MAX_CONTEXTS];
  NCCLCHECK(ncclGinRegister(comm, buf, bufSize, hostWins, devWins, /*winFlags=*/0));
  // Get GIN resources
  ncclGinCtx_M<-1u> gctx;
  gctx.backend = comm->sharedRes->ginState.ginDevHandles[0]->netDeviceType;
  gctx.handle = comm->sharedRes->ginState.ginDevHandles[0]->handle;
  gctx.rank = rank;
  gctx.nRanks = nRanks;

  // run kernel
  printf("[MPI Rank %d] Starting kernel\n", rank);
  runDevice<<<BlockPerRank, 512, 0, stream>>>(gctx, devWins[0], (int*)buf);
  CUDACHECK(cudaStreamSynchronize(stream));
  printf("[MPI Rank %d] Completed kernel\n", rank);

  // cleanup
  NCCLCHECK(ncclGinDeregister(comm, hostWins));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  MPICHECK(MPI_Finalize());
  return 0;
}
