#include "core.h"
#include "nccl_device/impl/ll_a2a__funcs.h"

NCCL_API_CXX(int, ncclSymLLA2ACalcSlots, int maxElts, int maxEltSize);
int ncclSymLLA2ACalcSlots(int maxElts, int maxEltSize) {
  return maxElts*divUp(maxEltSize, 8);
}

NCCL_API_CXX(ncclResult_t, ncclSymLLA2ACreateRequirement, int nBlocks, int nSlots, ncclSymLLA2AHandle* outHandle, ncclSymResourceRequirements* outReq);
ncclResult_t ncclSymLLA2ACreateRequirement(
    int nBlocks, int nSlots, ncclSymLLA2AHandle* outHandle,
    ncclSymResourceRequirements* outReq
  ) {
  outHandle->nSlots = nSlots;
  memset(outReq, 0, sizeof(*outReq));
  outReq->bufferSize = nBlocks*(1 + 2*nSlots)*16;
  outReq->bufferAlign = 16;
  outReq->outBufferHandle = &outHandle->bufHandle;
  return ncclSuccess;
}
