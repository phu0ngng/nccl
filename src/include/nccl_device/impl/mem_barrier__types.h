#ifndef _NCCL_DEVICE_MEM_BARRIER__TYPES_H_
#define _NCCL_DEVICE_MEM_BARRIER__TYPES_H_
#include "../mem_barrier.h"
#include "core__types.h"

struct ncclSymMemBarrierHandle {
  ncclSymResourceBufferHandle bufHandle;
  int nBarriers;
};

#if __CUDACC__
template<typename Coop>
struct ncclSymMemBarrierSession_internal {
  Coop coop;
  ncclSymComm const& comm;
  ncclTeam team;
  ncclSymMemBarrierHandle handle;
  int index;
  bool multimem;
  ncclMultimemHandle mmHandle;
  uint32_t epoch;

  NCCL_DEVICE_INLINE uint32_t* mcInbox(bool multimem) {
    uint32_t* state;
    if (multimem) { // multicast
      state = (uint32_t*)ncclSymGetResourceBufferMultimemPointer(comm, handle.bufHandle, mmHandle);
    } else { // unicast
      state = (uint32_t*)ncclSymGetResourceBufferLocalPointer(comm, handle.bufHandle);
    }
    return state + 2*handle.nBarriers + index;
  }

  NCCL_DEVICE_INLINE uint32_t* ucInbox(int owner, int peer) {
    uint32_t* state = (uint32_t*)ncclSymGetResourceBufferPeerPointer(comm, handle.bufHandle, team, owner);
    return state + 3*handle.nBarriers + index*team.nRanks + peer;
  }
};
#endif

#endif // _NCCL_DEVICE_MEM_BARRIER__TYPES_H_
