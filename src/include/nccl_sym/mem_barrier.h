#ifndef _NCCL_SYM_MEM_BARRIER_H_
#define _NCCL_SYM_MEM_BARRIER_H_
#include "impl/core__types.h"
#include <cuda/atomic>

struct ncclSymMemBarrierHandle;

__host__ ncclResult_t ncclSymMemBarrierCreateRequirement(ncclSymTeam, int nBarriers, ncclSymMemBarrierHandle* outHandle, ncclSymResourceRequirements* outReq);

#if __CUDACC__
template<typename Coop>
struct ncclSymMemBarrierSession_internal;

template<typename Coop>
struct ncclSymMemBarrierSession: ncclSymMemBarrierSession_internal<Coop> {
  NCCL_DEVICE_INLINE ncclSymMemBarrierSession(Coop, ncclSymComm const&, ncclSymTeam, ncclSymMemBarrierHandle, uint32_t index, bool multimem=false, ncclSymMultimemHandle mmHandle={});

  NCCL_DEVICE_INLINE ncclSymMemBarrierSession(Coop, ncclSymComm const&, ncclSymTeamTagNear, uint32_t index, bool multimem=false);

  NCCL_DEVICE_INLINE ~ncclSymMemBarrierSession();

  ncclSymMemBarrierSession(ncclSymMemBarrierSession const&) = delete; // Sessions are not copyable

  NCCL_DEVICE_INLINE void arrive(Coop, cuda::memory_order);
  NCCL_DEVICE_INLINE void wait(Coop, cuda::memory_order);
  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order);
};
#endif

#endif // _NCCL_SYM_MEM_BARRIER_H_
