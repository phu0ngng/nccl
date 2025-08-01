#ifndef _NCCL_DEVICE_LL_A2A__TYPES_H_
#define _NCCL_DEVICE_LL_A2A__TYPES_H_
#include "../ll_a2a.h"
#include "core__types.h"

struct ncclSymLLA2AHandle {
  ncclSymResourceBufferHandle bufHandle;
  uint32_t nSlots;
};

#if __CUDACC__
template<typename Coop>
struct ncclSymLLA2ASession_internal {
  Coop coop;
  ncclSymComm const& comm;
  ncclTeam team;
  ncclSymLLA2AHandle handle;
  int block;
  int pitch;
  bool multimem;
  ncclMultimemHandle mmHandle;
  uint32_t epoch;
  uint32_t slotsOffset;

  NCCL_DEVICE_INLINE uint32_t calcSlotOffset() const {
    return block*(1 + 2*handle.nSlots) + 1 + (epoch & 1)*handle.nSlots;
  }
};
#endif

#endif // _NCCL_DEVICE_LL_A2A__TYPES_H_
