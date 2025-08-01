#ifndef _NCCL_DEVICE_LL_A2A_H_
#define _NCCL_DEVICE_LL_A2A_H_
#include "impl/core__types.h"

struct ncclSymLLA2AHandle;

__host__ int ncclSymLLA2ACalcSlots(int maxElts, int maxEltSize);

__host__ ncclResult_t ncclSymLLA2ACreateRequirement(int nBlocks, int nSlots, ncclSymLLA2AHandle* outHandle, ncclSymResourceRequirements* outReq);

#if __CUDACC__
template<typename Coop>
struct ncclSymLLA2ASession_internal;

template<typename Coop>
struct ncclSymLLA2ASession: ncclSymLLA2ASession_internal<Coop> {
  NCCL_DEVICE_INLINE ncclSymLLA2ASession(Coop, ncclSymComm const&, ncclSymTeam, ncclSymLLA2AHandle, uint32_t block, int maxElts, bool multimem=false, ncclSymMultimemHandle mmHandle={});

  NCCL_DEVICE_INLINE ncclSymLLA2ASession(Coop, ncclSymComm const&, ncclSymTeamTagNear, uint32_t block, int maxElts, bool multimem=false);

  NCCL_DEVICE_INLINE ~ncclSymLLA2ASession();

  ncclSymLLA2ASession(ncclSymLLA2ASession const&) = delete; // Sessions are not copyable
  
  template<typename T>
  NCCL_DEVICE_INLINE void send(int peer, int slot, T data);

  template<typename T>
  NCCL_DEVICE_INLINE void bcast(int slot, T data);

  template<typename T>
  NCCL_DEVICE_INLINE T recv(int slot);

  template<int MinEltCount, int MaxEltCount, typename T>
  NCCL_DEVICE_INLINE void recvUnrolled(int eltStart, int eltCount, int eltStride, T(&vals)[MaxEltCount]);

  template<int Unroll, typename Elt, typename EltToAcc, typename Reduce>
  NCCL_DEVICE_INLINE auto recvReduce(int eltStart, int eltCount, int eltStride, EltToAcc eltToAcc, Reduce red)
    -> decltype(eltToAcc(nccl::utility::declval<Elt>())) ;
  
  // End an alltoall region. For every peer in team you must have done both of the
  // following each of which can be accomplished using any thread in coop:
  //  1. Targeted that peer with at least one send().
  //  2. Received from a slot targeted by that peer.
  NCCL_DEVICE_INLINE void endEpoch(Coop);
};
#endif

#endif // _NCCL_DEVICE_LL_A2A_H_
