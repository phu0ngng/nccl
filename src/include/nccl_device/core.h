#ifndef _NCCL_DEVICE_CORE_H_
#define _NCCL_DEVICE_CORE_H_
#include <nccl.h>
#include "coop.h"
#include "utility.h"

struct ncclSymComm;
struct ncclSymTeam;
// typedef struct ncclWindow_vidmem* ncclWindow_t; // in nccl.h
struct ncclSymMultimemHandle;

typedef uint32_t ncclSymResourceBufferHandle;

struct ncclSymMemBarrierHandle;
struct ncclSymLLA2AHandle;

struct ncclSymTeam {
  int nRanks, rank, stride;
};

template<typename T> struct ncclSymPtr;

struct ncclSymTeamTagWorld {};
struct ncclSymTeamTagNear {};
struct ncclSymTeamTagRail {};

struct ncclSymCommRequirements {
  struct ncclSymResourceRequirements* resourceRequirementsList;
  struct ncclSymTeamRequirements* teamRequirementsList;

  bool nearMultimem; // Enable multimem on near team

  int nearMemBarrierCount;
  ncclSymMemBarrierHandle* outNearMemBarrierHandle; // If non-null, target assigned during ncclSymCommCreate.

  int nearLLA2ABlockCount, nearLLA2ASlotCount;
  ncclSymLLA2AHandle* outNearLLA2AHandle; // If non-null, target assigned during ncclSymCommCreate.
};
struct ncclSymResourceRequirements {
  struct ncclSymResourceRequirements* next;
  size_t bufferSize, bufferAlign;
  ncclSymResourceBufferHandle* outBufferHandle; // If non-null, target assigned during ncclSymCommCreate.
};
struct ncclSymTeamRequirements {
  struct ncclSymTeamRequirements* next;
  struct ncclSymTeam team;
  bool multimem;
  ncclSymMultimemHandle* outMultimemHandle; // If non-null, target assigned during ncclSymCommCreate.
};

__host__ ncclResult_t ncclSymCommCreate(ncclComm_t, ncclSymCommRequirements const*, ncclSymComm* outDevComm);
__host__ ncclResult_t ncclSymCommDestroy(ncclComm_t, ncclSymComm const* devComm);

////////////////////////////////////////////////////////////////////////////////
// Team API:

NCCL_HOST_DEVICE_INLINE ncclSymTeam ncclSymTeamWorld(ncclSymComm const&);
__host__ ncclSymTeam ncclSymTeamWorld(ncclComm_t);

NCCL_HOST_DEVICE_INLINE ncclSymTeam ncclSymTeamNear(ncclSymComm const&);
__host__ ncclSymTeam ncclSymTeamNear(ncclComm_t);

NCCL_HOST_DEVICE_INLINE bool ncclSymTeamRankIsMember(ncclSymTeam a, ncclSymTeam b, int bPeer);
NCCL_HOST_DEVICE_INLINE int ncclSymTeamRankToTeam(ncclSymTeam a, ncclSymTeam b, int bPeer);

NCCL_HOST_DEVICE_INLINE int ncclSymTeamRankToWorld(ncclSymComm const&, ncclSymTeam, int rank);
__host__ int ncclSymTeamRankToWorld(ncclComm_t, ncclSymTeam, int rank);

NCCL_HOST_DEVICE_INLINE int ncclSymTeamRankToNear(ncclSymComm const&, ncclSymTeam, int rank);
__host__ int ncclSymTeamRankToNear(ncclComm_t, ncclSymTeam, int rank);

NCCL_HOST_DEVICE_INLINE ncclSymTeam ncclSymTeamInnerFactor(ncclSymTeam parent, int innerSize);
NCCL_HOST_DEVICE_INLINE ncclSymTeam ncclSymTeamOuterFactor(ncclSymTeam parent, int innerSize);

// Interpret each team as a set of ranks. This function assumes that `subset`
// is a subset of `parent`. Thus the number of ranks in the set difference of
// `parent` minus `subset` is `super.nRanks - subset.nRanks`. Given `index` this
// function returns the index'th element of `parent` minus `subset`.
NCCL_HOST_DEVICE_INLINE int ncclSymTeamRankInDifference(ncclSymTeam parent, ncclSymTeam subset, int index);

// Equivalent to ncclSymTeamOuterFactor of near team.
NCCL_HOST_DEVICE_INLINE ncclSymTeam ncclSymTeamRail(ncclSymComm const&);
__host__ ncclSymTeam ncclSymTeamRail(ncclComm_t);

// Get offset of resource buffer within `comm.resourceWindow`.
NCCL_HOST_DEVICE_INLINE size_t ncclSymGetResourceBufferOffset(ncclSymResourceBufferHandle);

#if __CUDACC__
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclSymGetResourceBuffer(ncclSymComm const&, ncclSymResourceBufferHandle);
#endif

////////////////////////////////////////////////////////////////////////////////
// Window API:

#if __CUDACC__
template<typename Coop>
NCCL_DEVICE_INLINE ncclWindow_t ncclSymFindWindow(Coop, ncclSymComm const&, void const *ptr);

NCCL_DEVICE_INLINE void* ncclSymGetLocalPointer(ncclWindow_t w, size_t offset);
NCCL_DEVICE_INLINE void* ncclSymGetNearPointer(ncclWindow_t w, size_t offset, int nearPeer);
NCCL_DEVICE_INLINE void* ncclSymGetPeerPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_DEVICE_INLINE void* ncclSymGetPeerPointer(ncclWindow_t w, size_t offset, ncclSymTeam tm, int peer);
NCCL_DEVICE_INLINE void* ncclSymGetMultimemPointer(ncclWindow_t w, size_t offset, ncclSymMultimemHandle mmHandle);
NCCL_DEVICE_INLINE void* ncclSymGetNearMultimemPointer(ncclWindow_t w, size_t offset, ncclSymComm const&);
#endif

#if __CUDACC__
// Convenience for combining ncclSymGet***Pointer() with resource handle.
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferLocalPointer(ncclSymComm const&, ncclSymResourceBufferHandle);
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferNearPointer(ncclSymComm const&, ncclSymResourceBufferHandle, int nearPeer);
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferPeerPointer(ncclSymComm const&, ncclSymResourceBufferHandle, ncclSymTeam, int peer);
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferMultimemPointer(ncclSymComm const&, ncclSymResourceBufferHandle, ncclSymMultimemHandle);
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferNearMultimemPointer(ncclSymComm const&, ncclSymResourceBufferHandle);
#endif

#endif // _NCCL_DEVICE_CORE_H_
