#ifndef NCCL_CE_COLL_H_
#define NCCL_CE_COLL_H_

#include "nccl.h"
#include "nccl_common.h"
#include "bitops.h"

// Memory operations per rank for different synchronization protocols
#define NCCL_CE_SYNC_OPS_PER_RANK_MC 2  
#define NCCL_CE_SYNC_OPS_PER_RANK_UC 3  

struct ncclCeColl {
  uint8_t* baseUCSymReadyPtr;
  uint8_t* baseUCSymComplPtr;
  uint32_t ceSeqNum;
  cudaStream_t ceLocalCopyStream;
  cudaEvent_t ceLocalCopyEvent;
};

struct alignas(16) ncclCeCollArgs {  
  ncclFunc_t func;
  int rootRank;
  size_t nElts;
  size_t eltSize;
  uint8_t* sendBuff;
  uint8_t* recvBuff;
};

bool ncclCeImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty);

ncclResult_t ncclCeInit(struct ncclComm* comm);

ncclResult_t ncclMemOpSync(struct ncclComm* comm, bool isComplete, cudaStream_t stream);

ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan);

ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

#endif /* NCCL_CE_COLL_H_ */