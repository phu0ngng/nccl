#ifndef NCCL_SYM_RUNTIME_H_
#define NCCL_SYM_RUNTIME_H_
#include "nccl.h"
#include "nccl_sym.h"
#include "nccl_common.h"
#include "allocator.h"
#include "bitops.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////////////////
// ncclSymr[_]: runtime implements for symmetric API.

struct ncclSymrMemory;
struct ncclSymrWindow {
  struct ncclSymrMemory* memory;
  void* userPtr;
  size_t size;
  size_t bigOffset; // Offset in big VA space.
  int winFlags;
  void* localRegHandle;
  struct ncclWindow_vidmem* vidmem;
};
struct ncclSymrWindowSorted;
struct ncclSymrTeam;

struct ncclSymrRegTask {
  struct ncclSymrRegTask *next;
  void* userPtr;
  size_t userSize;
  int winFlags;
  ncclWindow_t* outWinDev;
};

struct ncclSymrState {
  // Like localRank/localRanks except "near" ranks must be consecutive in the world
  // and all near subsets have the same number of ranks. If any condition is
  // false then the near team is just the singleton of self.
  int nearSelf;
  int nearSize;
  int* nearRankList;

  size_t granularity; // cuMemGetAllocationGranularity
  struct ncclSymrMemory* memHead;
  struct ncclSymrWindowSorted* winSorted;
  int winSortedCapacity, winSortedCount;
  struct ncclSymrTeam* teamHead;
  size_t bigSize; // size of our big logical space (128GB?)
  struct ncclSpace bigSpace; // allocates our big VA space.
  void* nearFlatBase; // base ptr for all near ranks big VA's concatenated together: size = nearRanks*bigSize
  struct ncclShadowPool shadows;
  struct ncclSymCommWindowTable* windowTable;

  struct ncclIntruQueue<struct ncclSymrRegTask, &ncclSymrRegTask::next> regTaskQueue;
};

// We assume ncclComm has a `ncclSymrState symState` member.
ncclResult_t ncclSymrInitOnce(struct ncclComm* comm);
ncclResult_t ncclSymrFinalize(struct ncclComm* comm);

// If found *outWinHost will be populated and *outWinId >= 0, otherwise *outWinId == -1
ncclResult_t ncclSymrFindWindow(struct ncclComm* comm, void const* userPtr, struct ncclSymrWindow** outWin);

ncclResult_t ncclSymrWindowRegisterInGroup(
  struct ncclComm* comm, void* ptr, size_t size, int winFlags, ncclWindow_t* outWinDev
);

#endif
