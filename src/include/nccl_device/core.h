#ifndef _NCCL_DEVICE_CORE_H_
#define _NCCL_DEVICE_CORE_H_
#include <nccl.h>
#include "coop.h"
#include "utility.h"

struct ncclSymComm;
struct ncclTeam;
// typedef struct ncclWindow_vidmem* ncclWindow_t; // in nccl.h
struct ncclMultimemHandle;

typedef uint32_t ncclSymResourceBufferHandle;

struct ncclLsaBarrierHandle;
struct ncclSymLLA2AHandle;

struct ncclTeam {
  int nRanks, rank, stride;
};

template<typename T> struct ncclSymPtr;

struct ncclTeamTagWorld {};
struct ncclTeamTagLsa {};
struct ncclTeamTagRail {};

struct ncclSymCommRequirements {
  struct ncclSymResourceRequirements* resourceRequirementsList;
  struct ncclTeamRequirements* teamRequirementsList;

  bool multimem; // Enable multimem on lsa team

  int lsaBarrierCount;
  ncclLsaBarrierHandle* outLsaBarrierHandle; // If non-null, target assigned during ncclSymCommCreate.

  int lsaLLA2ABlockCount, lsaLLA2ASlotCount;
  ncclSymLLA2AHandle* outLsaLLA2AHandle; // If non-null, target assigned during ncclSymCommCreate.
};
struct ncclSymResourceRequirements {
  struct ncclSymResourceRequirements* next;
  size_t bufferSize, bufferAlign;
  ncclSymResourceBufferHandle* outBufferHandle; // If non-null, target assigned during ncclSymCommCreate.
};
struct ncclTeamRequirements {
  struct ncclTeamRequirements* next;
  struct ncclTeam team;
  bool multimem;
  ncclMultimemHandle* outMultimemHandle; // If non-null, target assigned during ncclSymCommCreate.
};

__host__ ncclResult_t ncclSymCommCreate(ncclComm_t, ncclSymCommRequirements const*, ncclSymComm* outDevComm);
__host__ ncclResult_t ncclSymCommDestroy(ncclComm_t, ncclSymComm const* devComm);

////////////////////////////////////////////////////////////////////////////////
// Team API:

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamWorld(ncclSymComm const&);
__host__ ncclTeam ncclTeamWorld(ncclComm_t);

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamLsa(ncclSymComm const&);
__host__ ncclTeam ncclTeamLsa(ncclComm_t);

NCCL_HOST_DEVICE_INLINE bool ncclTeamRankIsMember(ncclTeam a, ncclTeam b, int bPeer);
NCCL_HOST_DEVICE_INLINE int ncclTeamRankToTeam(ncclTeam a, ncclTeam b, int bPeer);

NCCL_HOST_DEVICE_INLINE int ncclTeamRankToWorld(ncclSymComm const&, ncclTeam, int rank);
__host__ int ncclTeamRankToWorld(ncclComm_t, ncclTeam, int rank);

NCCL_HOST_DEVICE_INLINE int ncclTeamRankToLsa(ncclSymComm const&, ncclTeam, int rank);
__host__ int ncclTeamRankToLsa(ncclComm_t, ncclTeam, int rank);

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamInnerFactor(ncclTeam parent, int innerSize);
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamOuterFactor(ncclTeam parent, int innerSize);

// Interpret each team as a set of ranks. This function assumes that `subset`
// is a subset of `parent`. Thus the number of ranks in the set difference of
// `parent` minus `subset` is `super.nRanks - subset.nRanks`. Given `index` this
// function returns the index'th element of `parent` minus `subset`.
NCCL_HOST_DEVICE_INLINE int ncclTeamRankInDifference(ncclTeam parent, ncclTeam subset, int index);

// Equivalent to ncclTeamOuterFactor of lsa team.
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamRail(ncclSymComm const&);
__host__ ncclTeam ncclTeamRail(ncclComm_t);

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
NCCL_DEVICE_INLINE void* ncclSymGetLsaPointer(ncclWindow_t w, size_t offset, int lsaPeer);
NCCL_DEVICE_INLINE void* ncclSymGetPeerPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_DEVICE_INLINE void* ncclSymGetPeerPointer(ncclWindow_t w, size_t offset, ncclTeam tm, int peer);
NCCL_DEVICE_INLINE void* ncclSymGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle mmHandle);
NCCL_DEVICE_INLINE void* ncclSymGetMultimemPointer(ncclWindow_t w, size_t offset, ncclSymComm const&);
#endif

#if __CUDACC__
// Convenience for combining ncclSymGet***Pointer() with resource handle.
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferLocalPointer(ncclSymComm const&, ncclSymResourceBufferHandle);
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferLsaPointer(ncclSymComm const&, ncclSymResourceBufferHandle, int lsaPeer);
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferPeerPointer(ncclSymComm const&, ncclSymResourceBufferHandle, ncclTeam, int peer);
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferMultimemPointer(ncclSymComm const&, ncclSymResourceBufferHandle, ncclMultimemHandle);
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferMultimemPointer(ncclSymComm const&, ncclSymResourceBufferHandle);
#endif

#endif // _NCCL_DEVICE_CORE_H_
