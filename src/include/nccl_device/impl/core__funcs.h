#ifndef _NCCL_DEVICE_CORE__FUNCS_H_
#define _NCCL_DEVICE_CORE__FUNCS_H_
#include "core__types.h"
#include "comm__types.h"
#include "ptr__types.h"

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamWorld(ncclDevComm const &comm) {
  ncclTeam ans;
  ans.nRanks = comm.nRanks;
  ans.rank = comm.rank;
  ans.stride = 1;
  return ans;
}

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamLsa(ncclDevComm const &comm) {
  ncclTeam ans;
  ans.nRanks = comm.lsaSize;
  ans.rank = comm.lsaRank;
  ans.stride = 1;
  return ans;
}

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamRail(ncclDevComm const& comm) {
  ncclTeam ans;
  ans.nRanks = nccl::utility::idivFast32(comm.nRanks, comm.lsaSize, comm.lsaSize_rcp32);
  ans.rank = nccl::utility::idivFast32(comm.rank, comm.lsaSize, comm.lsaSize_rcp32);
  ans.stride = comm.lsaSize;
  return ans;
}

NCCL_HOST_DEVICE_INLINE bool ncclTeamRankIsMember(ncclTeam a, ncclTeam b, int brank) {
  int wrank = (brank - b.rank)*b.stride;
  uint32_t adelta = wrank/a.stride;
  uint32_t amod = wrank%a.stride;
  int arank = a.rank + adelta;
  return 0 <= arank && arank < a.nRanks && amod == 0;
}

NCCL_HOST_DEVICE_INLINE int ncclTeamRankToTeam(ncclTeam a, ncclTeam b, int brank) {
  int wrank = (brank - b.rank)*b.stride;
  uint32_t adelta = wrank/a.stride;
  //uint32_t amod = wrank%a.stride;
  int arank = a.rank + adelta;
  return arank;
}

NCCL_HOST_DEVICE_INLINE int ncclTeamRankToWorld(ncclDevComm const& comm, ncclTeam tm, int rank) {
  return comm.rank + (rank - tm.rank)*tm.stride;
}

NCCL_HOST_DEVICE_INLINE int ncclTeamRankToLsa(ncclDevComm const& comm, ncclTeam tm, int rank) {
  return comm.lsaRank + (rank - tm.rank)*tm.stride;
}

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamInnerFactor(ncclTeam parent, int innerSize) {
  ncclTeam ans;
  ans.nRanks = innerSize;
  ans.rank = parent.rank%innerSize;
  ans.stride = parent.stride;
  return ans;
}

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamOuterFactor(ncclTeam parent, int innerSize) {
  ncclTeam ans;
  ans.nRanks = parent.nRanks/innerSize;
  ans.rank = parent.rank/innerSize;
  ans.stride = parent.stride*innerSize;
  return ans;
}

NCCL_HOST_DEVICE_INLINE int ncclTeamRankInDifference(ncclTeam parent, ncclTeam subset, int index) {
  int stride = subset.stride/parent.stride;
  int below = parent.rank - subset.rank*stride;
  if (stride < 0) {
    stride = -stride;
    below -= (subset.nRanks-1)*stride;
  }
  if (index < below) {
    return index;
  } else if (index-below < (subset.nRanks-1)*(stride-1)) {
    return below + 1 + ((index-below)/(stride-1))*stride + (index-below)%(stride-1);
  } else {
    return below + 1 + (subset.nRanks-1)*stride + (index - below - (subset.nRanks-1)*(stride-1));
  }
}

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclSymGetLocalPointer(ncclWindow_t w, size_t offset) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);
  int lsaRank = nccl::utility::loadConst(&w->lsaRank);
  return (void*)(nccl::utility::add4G(base, lsaRank*stride4G) + offset);
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclSymGetLsaPointer(ncclWindow_t w, size_t offset, int lsaPeer) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);
  return (void*)(nccl::utility::add4G(base, lsaPeer*stride4G) + offset);
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclSymGetPeerPointer(ncclWindow_t w, size_t offset, int peer) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);
  int worldRank = nccl::utility::loadConst(&w->worldRank);
  int lsaRank = nccl::utility::loadConst(&w->lsaRank);
  int lsaPeer = lsaRank + (peer - worldRank);
  return (void*)(nccl::utility::add4G(base, lsaPeer*stride4G) + offset);
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclSymGetPeerPointer(ncclWindow_t w, size_t offset, ncclTeam tm, int peer) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);
  int lsaRank = nccl::utility::loadConst(&w->lsaRank);
  int lsaPeer = lsaRank + (peer - tm.rank)*tm.stride;
  return (void*)(nccl::utility::add4G(base, lsaPeer*stride4G) + offset);
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclSymGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle mm) {
  void* ptr = mm.mcBasePtr;
  ptr = reinterpret_cast<char(*)[4096]>(ptr) + nccl::utility::loadConst(&w->mcOffset4K);
  return (void*)((char*)ptr + offset);
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclSymGetMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const& comm) {
  return ncclSymGetMultimemPointer(w, offset, comm.multimem);
}
#endif

#if __CUDACC__
template<typename Coop>
NCCL_DEVICE_INLINE ncclWindow_t ncclSymFindWindow(Coop coop, ncclDevComm const& comm, void const *ptr) {
  using nccl::utility::loadConst;
  auto coalesced = ncclCoopCoalesced(coop);
  ncclDevCommWindowTable* t = comm.windowTable;
  while (true) {
    bool found = false;
    #pragma unroll 1
    for (int i=coalesced.thread_rank(); i < 32; i += coalesced.size()) {
      uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
      ncclDevCommWindowTable::Entry e = loadConst(&t->entries[i]);
      if ((e.base != 0) && (e.size != 0) && (e.window != 0)) {
        if (uptr - uintptr_t(e.base) < uintptr_t(e.size)) {
          found = true;
          break;
        }
      }
    }
    uint32_t mask = __ballot_sync(coalesced.laneMask(), found);
    if (mask != 0) {
      int index = __popc((mask-1) & coalesced.laneMask());
      return loadConst(&t->entries[index].window);
    }
    t = loadConst(&t->next);
  }
}
#endif

#if 0
#if __CUDACC__
template<typename Coop>
NCCL_DEVICE_INLINE ncclMultimemHandle ncclSymFindMultimem(Coop coop, ncclDevComm const &comm, ncclTeam tm) {
  using nccl::utility::loadConst;
  auto coalesced = ncclCoopCoalesced(coop);
  ncclDevComm::TeamTable* e = comm.teamTable;
  while (true) {
    #pragma unroll 1
    for (int i=coalesced.thread_rank(); i < 32; i += coalesced.size()) {
      ncclTeam tm1 = loadConst(&e->team[i]);
      if (tm1.rank == tm.rank && tm1.nRanks == tm.nRanks && tm1.stride == tm.stride) {
        found = true;
        break;
      }
    }
    uint32_t mask = __ballot_sync(coalesced.laneMask(), found);
    if (mask != 0) {
      int index = __popc((mask-1) & coalesced.laneMask());
      ncclMultimemHandle mm;
      mm.mcBaseAddr = loadConst(&e->mcBaseAddr[index]);
      return mm;
    }
    e = loadConst(&e->next);
  }
}
#endif
#endif

NCCL_HOST_DEVICE_INLINE size_t ncclSymGetResourceBufferOffset(ncclSymResourceBufferHandle h) {
  return size_t(h)*128;
}

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferLocalPointer(ncclDevComm const& comm, ncclSymResourceBufferHandle h) {
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = nccl::utility::add4G(lsaFlatBase, comm.lsaRank*stride4G);
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferLsaPointer(ncclDevComm const& comm, ncclSymResourceBufferHandle h, int lsaPeer) {
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = nccl::utility::add4G(lsaFlatBase, lsaPeer*stride4G);
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferPeerPointer(ncclDevComm const& comm, ncclSymResourceBufferHandle h, ncclTeam team, int peer) {
  int lsaPeer = comm.lsaRank + (peer - team.rank)*team.stride;
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = nccl::utility::add4G(lsaFlatBase, lsaPeer*stride4G);
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferMultimemPointer(ncclDevComm const& comm, ncclSymResourceBufferHandle h, ncclMultimemHandle mm) {
  void* ptr = mm.mcBasePtr;
  ptr = reinterpret_cast<char(*)[4096]>(ptr) + comm.resourceWindow_inlined.mcOffset4K;
  ptr = reinterpret_cast<char(*)[128]>(ptr) + h;
  return ptr;
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclSymGetResourceBufferMultimemPointer(ncclDevComm const& comm, ncclSymResourceBufferHandle h) {
  return ncclSymGetResourceBufferMultimemPointer(comm, h, comm.multimem);
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclSymGetResourceBuffer(ncclDevComm const& comm, ncclSymResourceBufferHandle h) {
  return ncclSymPtr<char>(comm.resourceWindow, size_t(h)*128);
}
#endif

#endif
