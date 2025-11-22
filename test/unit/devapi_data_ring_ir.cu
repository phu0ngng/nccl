#include "nccl_device_wrapper.h"

constexpr int BufElts = 1<<20;
constexpr bool Prints = false;

extern "C" __global__ void runDevice(ncclDevComm comm, ncclDevResourceHandle hbuf) {
#if __CUDA_ARCH__ >= 700
  int t = threadIdx.x;
  int tn = blockDim.x;
  ncclTeam world = ncclTeamWorld(comm);
  int b = world.rank*gridDim.x + blockIdx.x;
  int bn = world.nRanks*gridDim.x;
  int up_rank = (b+bn-1)%bn / gridDim.x;
  int up_block = (b+bn-1)%bn % gridDim.x;
  int down_rank = (b+1)%bn / gridDim.x;
  int down_block = (b+1)%bn % gridDim.x;

  // Get local pointer for direct memory access
  int* sbuf_local = (int*)ncclGetResourceBufferLocalPointer(comm, hbuf);
  int* rbuf_local = sbuf_local + gridDim.x*BufElts;

  // Get window and offset for GIN operations
  ncclWindow_t bufWindow = comm.resourceWindow;
  size_t bufBaseOffset = ncclGetResourceBufferOffset(hbuf);

  unsigned sigData0 = 0;
  unsigned sigFree0 = gridDim.x;

  // Initialize CoopAny (CTA) in raw storage
  alignas(ncclCoopAny) unsigned char coop_storage[sizeof(ncclCoopAny)];
  ncclCoopAny* coop = reinterpret_cast<ncclCoopAny*>(coop_storage);
  ncclCoopAnyInitCta(coop);

  // Initialize GIN via v1 API
  alignas(ncclGin_C) unsigned char gin_storage[sizeof(ncclGin_C)];
  ncclGin_C* net = reinterpret_cast<ncclGin_C*>(gin_storage);
  ncclGin_C_init(net, NCCL_GIN_BACKEND_MASK_ALL, comm, 0);

  // Allocate storage ONCE outside loop to reduce register pressure
  alignas(ncclCoopAny) unsigned char thread_coop_storage[sizeof(ncclCoopAny)];
  ncclCoopAny* thread_coop = reinterpret_cast<ncclCoopAny*>(thread_coop_storage);
  ncclCoopAnyInitThread(thread_coop);

  int accum = 0;
  for (uint32_t round=0; round < 10; round++) {
    if (t==0 && b == round%bn) printf("Round %d\n", round);
    // Consume recv buf (rbuf) and populate send buf (sbuf)
    for (int i=t; i < BufElts; i += tn) {
      int got = rbuf_local[blockIdx.x*BufElts + i];
      if (Prints) {
        if (got != (int)round) {
          printf("[%d:%d]: ERROR round=%d rbuf[%d]=%d but expected %d\n", world.rank, blockIdx.x, round, i, got, round);
        }
      } else {
        assert(got == (int)round);
      }
      sbuf_local[blockIdx.x*BufElts + i] = 1 + got;
    }
    __syncthreads();

    if (t==0) {
      ncclGinSignal(net, world, up_rank,
                    /*isSignal=*/true, /*signalId=*/sigFree0 + up_block,
                    /*signalOp=*/ncclGinSignalInc, /*signalOpArg=*/1,
                    *thread_coop,
                    /*isDescriptor=*/false, /*descriptor=*/nullptr,
                    cuda::thread_scope_thread, cuda::thread_scope_device);
    }

    // Wait for downstream to give us free space
    if (Prints) if (t==0) printf("[%d:%d] round=%d Send free signal\n", world.rank, blockIdx.x, round);
    ncclGinWaitSignal(net, *coop, sigFree0 + blockIdx.x, round+1, 64, cuda::memory_order_acquire);
    if (Prints) if (t==0) printf("[%d:%d] round=%d Got free signal\n", world.rank, blockIdx.x, round);

    // Send data downstream in chunks
    int nChunks = min(BufElts, 1 + (round*0xdeadbeefu >> (32-10)));
    int chunkElts = BufElts/nChunks;
    nChunks = (BufElts + chunkElts-1)/chunkElts;
    if (Prints) if (t==0) printf("[%d:%d] round=%d Sending %d chunks\n", world.rank, blockIdx.x, round, nChunks);

    #pragma unroll 1
    for (int i=t; i < nChunks; i += tn) {
      ncclWindow_t dstWnd = bufWindow;
      size_t dstOffset = bufBaseOffset + (gridDim.x*BufElts + down_block*BufElts + i*chunkElts) * sizeof(int);
      ncclWindow_t srcWnd = bufWindow;
      size_t srcOffset = bufBaseOffset + (blockIdx.x*BufElts + i*chunkElts) * sizeof(int);
      size_t bytes = min(chunkElts, BufElts - i*chunkElts) * sizeof(int);

      ncclGinPut(net, world, down_rank, dstWnd, dstOffset, srcWnd, srcOffset, bytes,
                 /*isSignal=*/true, /*signalId=*/sigData0 + down_block,
                 /*signalOp=*/ncclGinSignalInc, /*signalOpArg=*/1,
                 /*isCounter=*/false, /*counterId=*/0,
                 *thread_coop,
                 /*isDescriptor=*/false, /*descriptor=*/nullptr,
                 cuda::thread_scope_thread, cuda::thread_scope_device);
    }

    if (Prints) if (t==0) printf("[%d:%d] round=%d wait chunks\n", world.rank, blockIdx.x, round);
    // Wait for all chunks from upstream
    ncclGinWaitSignal(net, *coop, sigData0 + blockIdx.x, accum + nChunks, 64, cuda::memory_order_acquire);
    // Wait for all outgoing to settle
    ncclGinFlush(net, *coop, cuda::memory_order_acquire);
    if (Prints) if (t==0) printf("[%d:%d] round=%d Got chunks\n", world.rank, blockIdx.x, round);

    accum += nChunks;
  }
#endif
}

