#ifndef _NCCL_SYM_COMM__TYPES_H_
#define _NCCL_SYM_COMM__TYPES_H_
#include "../comm.h"
#include "core__types.h"
#include "mem_barrier__types.h"
#include "ll_a2a__types.h"

struct ncclSymCommWindowTable {
  struct Entry {
    uintptr_t base, size;
    ncclWindow_t window;
  } entries[32];
  struct ncclSymCommWindowTable* next;
};

struct ncclSymComm {
  int rank, nRanks;
  uint32_t nRanks_rcp32;
  int nearRank, nearSize;
  uint32_t nearSize_rcp32;

  struct ncclSymCommWindowTable* windowTable;

  ncclWindow_t resourceWindow;
  ncclWindow_vidmem resourceWindow_inlined;

  ncclSymMultimemHandle nearMultimem;
  ncclSymMemBarrierHandle nearMemBarrier;
  ncclSymLLA2AHandle nearLLA2A;
};

#endif // _NCCL_SYM_COMM__TYPES_H_
