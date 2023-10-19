/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "device.h"
#include "collectives.h"
#include "core.h"
#include "utils.h"
#include "strongstream.h"
#define NCCL_MAX_LOCAL_RANKS 64

typedef enum : uint8_t {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollnetChain,
  ncclPatternCollnetDirect,
  ncclPatternNvls,
  ncclPatternNvlsTree,
  ncclPatternSend,
  ncclPatternRecv
} ncclPattern_t;

// Used to pass NCCL call information from API to enqueue module.
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root; // peer for p2p operations
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
};

inline int ncclFuncTrafficPerElement(ncclFunc_t func, int nRanks) {
  switch (func) {
  case ncclFuncAllReduce: return 2;
  case ncclFuncAllGather: return nRanks;
  case ncclFuncReduceScatter: return nRanks;
  default: return 1;
  }
}

inline size_t ncclFuncSendCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncReduceScatter ? nRanks*count : count;
}
inline size_t ncclFuncRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather ? nRanks*count : count;
}
inline size_t ncclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather || func == ncclFuncReduceScatter ? nRanks*count : count;
}

#endif
