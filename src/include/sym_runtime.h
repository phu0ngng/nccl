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
  struct ncclReg* localRegHandle;
  struct ncclWindow_vidmem* vidmem;
};
struct ncclSymrWindowSorted;
struct ncclSymrTeam;

struct ncclSymrRegTask {
  struct ncclSymrRegTask *next;
  CUmemGenericAllocationHandle memHandle;
  size_t memSize;
  size_t memOffset;
  void* userPtr;
  size_t userSize;
  int winFlags;
  struct ncclWindow_vidmem* winDev;
  struct ncclReg* localRegHandle;
  cudaStream_t stream;
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
ncclResult_t ncclSymrInit(struct ncclComm* comm);
ncclResult_t ncclSymrFinalize(struct ncclComm* comm);

// If found *outWinHost will be populated and *outWinId >= 0, otherwise *outWinId == -1
ncclResult_t ncclSymrFindWindow(struct ncclComm* comm, void const* userPtr, struct ncclSymrWindow** outWin);

ncclResult_t ncclSymrRegisterInternal(
    struct ncclComm* comm,
    CUmemGenericAllocationHandle memHandle, size_t memSize, size_t memOffset,
    void* userPtr, size_t userSize, int winFlags,
    struct ncclWindow_vidmem** outWinDev, struct ncclReg* localRegHandle, cudaStream_t stream
  );

#endif
