#include "nccl_device_wrapper.h"

extern "C" __global__ void runDevice(ncclDevComm comm) {
#if __CUDA_ARCH__ >= 700
    // Compute LSA team on device
    ncclTeam team = ncclTeamLsa(comm);

    // Initialize CoopAny (CTA) in raw storage
    alignas(ncclCoopAny) unsigned char coop_storage[sizeof(ncclCoopAny)];
    ncclCoopAny* coop = reinterpret_cast<ncclCoopAny*>(coop_storage);
    ncclCoopAnyInitCta(coop);

    // Initialize LSA barrier session Any via C-ABI
    alignas(ncclLsaBarrierSession_C) unsigned char sess_storage[sizeof(ncclLsaBarrierSession_C)];
    ncclLsaBarrierSession_C* session = reinterpret_cast<ncclLsaBarrierSession_C*>(sess_storage);

    ncclLsaBarrierHandle handle = comm.lsaBarrier;
    ncclMultimemHandle mmHandle{}; // unused when multimem=false

    ncclLsaBarrierSessionInit(
        session,
        *coop,
        comm,
        team,
        handle,
        blockIdx.x,
        /*multimem=*/false,
        mmHandle);

    // Barrier rounds
    for (int round = 0; round < 10; ++round) {
        ncclLsaBarrierSessionSync(session, *coop, cuda::memory_order_relaxed);
        if (threadIdx.x == 0 && blockIdx.x == 0 && team.rank == (round % team.nRanks)) {
            printf("Round %d\n", round);
        }
    }
#endif
}
