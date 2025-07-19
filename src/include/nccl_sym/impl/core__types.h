#ifndef _NCCL_SYM_CORE__TYPES_H_
#define _NCCL_SYM_CORE__TYPES_H_
#include "../core.h"

// nccl.h has: typedef ncclWindow_vidmem* ncclWindow_t;
struct ncclWindow_vidmem {
  void* winHost;
  //ncclGinWindow_t ginWin;
  char* nearFlatBase; // pointer to first byte for rank 0 of near team
  int nearRank;
  int worldRank;
  uint32_t stride4G;
  uint32_t mcOffset4K;
};

struct ncclSymMultimemHandle {
  void* mcBasePtr;
};

#endif
