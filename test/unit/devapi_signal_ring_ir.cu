#include "nccl_device_wrapper.h"

constexpr bool Prints = true;

extern "C" __global__ void runDevice(ncclDevComm comm) {
#if __CUDA_ARCH__ >= 700
  int t = threadIdx.x;
  ncclTeam world = ncclTeamWorld(comm);

  // Initialize CoopAny (CTA) in raw storage
  alignas(ncclCoopAny) unsigned char coop_storage[sizeof(ncclCoopAny)];
  ncclCoopAny* coop = reinterpret_cast<ncclCoopAny*>(coop_storage);
  ncclCoopAnyInitCta(coop);

  // Initialize GIN via C-ABI v1
  alignas(ncclGin_C) unsigned char gin_storage[sizeof(ncclGin_C)];
  ncclGin_C* net = reinterpret_cast<ncclGin_C*>(gin_storage);
  ncclGin_C_init(net, NCCL_GIN_BACKEND_MASK_ALL, comm, 0);

  // Initialize thread-level coop for signal
  alignas(ncclCoopAny) unsigned char thread_coop_storage[sizeof(ncclCoopAny)];
  ncclCoopAny* thread_coop = reinterpret_cast<ncclCoopAny*>(thread_coop_storage);
  ncclCoopAnyInitThread(thread_coop);

  for (int round=0; round < 10; round++) {
    if (t==0) {
      ncclGinSignal(net, world, (world.rank + 1) % world.nRanks,
                    /*isSignal=*/true, /*signalId=*/0, /*signalOp=*/ncclGinSignalAdd, /*signalOpArg=*/1,
                    *thread_coop,
                    /*isDescriptor=*/false, /*descriptor=*/nullptr,
                    cuda::thread_scope_thread, cuda::thread_scope_device);
    }

    ncclGinWaitSignal(net, *coop, /*signal=*/0, /*least=*/round+1, /*bits=*/64, cuda::memory_order_acquire);

    if (world.rank == round % world.nRanks && t==0) {
      if (Prints) printf("[Rank %d] Round %d\n", world.rank, round);
    }
  }
#endif
}

