#include "nccl_device_wrapper.h"

extern "C" __global__ void runDevice(ncclDevComm comm) {
#if __CUDA_ARCH__ >= 700
  int t = threadIdx.x;

  // Compute world team on device
  ncclTeam world = ncclTeamWorld(comm);

  // Initialize CoopAny (CTA) in raw storage
  alignas(ncclCoopAny) unsigned char coop_storage[sizeof(ncclCoopAny)];
  ncclCoopAny* coop = reinterpret_cast<ncclCoopAny*>(coop_storage);
  ncclCoopAnyInitCta(coop);

  // Initialize GIN network via v1 API
  alignas(ncclGin_C) unsigned char net_storage[sizeof(ncclGin_C)];
  ncclGin_C* net = reinterpret_cast<ncclGin_C*>(net_storage);
  ncclGin_C_init(net, NCCL_GIN_BACKEND_MASK_ALL, comm, 0);

  // Initialize Barrier session (LSA + GIN) via C-ABI
  alignas(ncclBarrierSession_C) unsigned char sess_storage[sizeof(ncclBarrierSession_C)];
  ncclBarrierSession_C* session = reinterpret_cast<ncclBarrierSession_C*>(sess_storage);

  ncclLsaBarrierHandle innerHandle = comm.lsaBarrier;
  ncclGinBarrierHandle outerHandle = comm.railGinBarrier;
  ncclMultimemHandle mmHandle{}; // unused when multimem=false

  ncclBarrierSessionInit(
      session,
      *coop,
      world,  // innerTeam
      world,  // outerTeam
      *net,
      innerHandle,
      outerHandle,
      blockIdx.x,
      /*multimem=*/false,
      mmHandle);

  // Barrier rounds
  for (int round = 0; round < 10; ++round) {
    ncclBarrierSessionSync(session, *coop, cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
    if (t == 0 && blockIdx.x == 0 && world.rank == (round % world.nRanks)) {
      printf("Round %d\n", round);
    }
  }
#endif
}

