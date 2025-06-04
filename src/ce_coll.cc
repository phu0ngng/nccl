#include "comm.h"
#include "register_inline.h"
#include <cuda.h>
#include "cudawrap.h"
#include "ce_coll.h"
#include "alloc.h"

ncclResult_t ncclCeInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  uint8_t* ceDevBase;
  size_t ceDevBaseSize = alignUp(comm->nRanks*sizeof(uint32_t), 16) * 2;
  NCCLCHECKGOTO(ncclCommSymmetricAllocInternal(comm, ceDevBaseSize, 16 /*alignment*/, (void**)&ceDevBase), ret, fail);
  comm->ceColl.baseUCSymReadyPtr = ceDevBase;
  comm->ceColl.baseUCSymComplPtr = ceDevBase + alignUp(comm->nRanks*sizeof(uint32_t), 16);
  comm->ceColl.ceSeqNum = 0;
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&comm->ceColl.ceLocalCopyStream, cudaStreamNonBlocking), ret, fail);
  CUDACHECKGOTO(cudaEventCreate(&comm->ceColl.ceLocalCopyEvent), ret, fail);
  INFO(NCCL_INIT, "Init CE, rank %d baseUCSymReadyPtr %p, baseUCSymComplPtr %p, seq num %d", comm->rank, comm->ceColl.baseUCSymReadyPtr, comm->ceColl.baseUCSymComplPtr, comm->ceColl.ceSeqNum);

exit:
  return ret;
fail:
  goto exit;
}

bool ncclCeImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  int driverVersion;
  if (ncclCudaDriverVersion(&driverVersion) != ncclSuccess) return false;

  // CE is supported in CUDA 12.5 and later
  if (driverVersion >= 12050) {
    switch (coll) {
    case ncclFuncAllGather:
      return true;
    default:
      return false;
    }
  }
  return false;
}

ncclResult_t ncclMemOpSync(struct ncclComm* comm, bool isComplete, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Check if we are in a CUDA graph
  bool capturing = ncclCudaGraphValid((&comm->planner)->capturingGraph);

  // Get pointers to the ready and complete synchronization arrays
  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;
  
  // Determine sequence number for synchronization
  uint32_t currentSeq = ++ comm->ceColl.ceSeqNum;
  size_t batchSize = capturing ? comm->nRanks*2 : comm->nRanks;
  size_t opIdx = 0;  

  // Prepare batch memory operations for synchronization
  CUstreamBatchMemOpParams* batchParams = nullptr;
  NCCLCHECKGOTO(ncclCalloc(&batchParams, batchSize), ret, fail);

  if(comm->nvlsSupport) {
    // Write our own ready/complete flag to multi-cast address
    batchParams[opIdx] = {};
    batchParams[opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    batchParams[opIdx].writeValue.address = (CUdeviceptr)(isComplete ? peerMCSymPtr(comm, comm->rank, &completePtrs[comm->rank]) : peerMCSymPtr(comm, comm->rank, &readyPtrs[comm->rank]));
    batchParams[opIdx].writeValue.value = currentSeq;
    batchParams[opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    opIdx++;

    // Add local wait operations for each other rank
    for (int i = 0; i < comm->nRanks; i++) {
      if (i != comm->rank) {
        batchParams[opIdx] = {};
        batchParams[opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
        batchParams[opIdx].waitValue.address = (CUdeviceptr)(isComplete ? peerUCSymPtr(comm, comm->rank, &completePtrs[i]) : peerUCSymPtr(comm, comm->rank, &readyPtrs[i]));
        batchParams[opIdx].waitValue.value = currentSeq;
        batchParams[opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
        opIdx++;
      }
    }
  } else {
    // Write our own ready/complete flag
    batchParams[opIdx] = {};
    batchParams[opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    batchParams[opIdx].writeValue.address = (CUdeviceptr)(isComplete ? peerUCSymPtr(comm, comm->rank, &completePtrs[comm->rank]) : peerUCSymPtr(comm, comm->rank, &readyPtrs[comm->rank]));
    batchParams[opIdx].writeValue.value = currentSeq;
    batchParams[opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    opIdx++;

    // Add remote wait operations for each other rank
    for (int i = 0; i < comm->nRanks; i++) {
      if (i != comm->rank) {
        batchParams[opIdx] = {};
        batchParams[opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
        batchParams[opIdx].waitValue.address = (CUdeviceptr)(isComplete ? peerUCSymPtr(comm, i, &completePtrs[i]) : peerUCSymPtr(comm, i, &readyPtrs[i]));
        batchParams[opIdx].waitValue.value = currentSeq;
        batchParams[opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
        opIdx++;
      }
    }
  }

  // For CUDA graph capture, add reset operation
  if (capturing) {
    for (int i = 0; i < comm->nRanks; i++) {
      batchParams[opIdx] = {};
      batchParams[opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
      batchParams[opIdx].writeValue.address = (CUdeviceptr)(isComplete ? peerUCSymPtr(comm, comm->rank, &completePtrs[i]) : peerUCSymPtr(comm, comm->rank, &readyPtrs[i]));
      batchParams[opIdx].writeValue.value = 0;
      batchParams[opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      opIdx++;
    }
  }
  
  // Execute all memory operations in a single batch
  CUCHECKGOTO(cuStreamBatchMemOp(stream, opIdx, batchParams, 0), ret, fail);

exit:
  if (batchParams) free(batchParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  
  // Calculate the size of each rank's data chunk
  const size_t bytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff + comm->rank * bytes;
  bool capturing;
#if CUDA_VERSION >= 12080 && CUDART_VERSION >= 12080 
  cudaMemcpyAttributes attrs = {};
  void**  srcs     = nullptr;
  void**  dsts     = nullptr;
  size_t* sizes    = nullptr;
  size_t* attrIdxs = nullptr;
  NCCLCHECKGOTO(ncclCalloc(&srcs,     comm->nRanks-1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&dsts,     comm->nRanks-1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&sizes,    comm->nRanks-1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&attrIdxs, comm->nRanks-1), ret, fail);
  for (int i = 0; i < comm->nRanks-1; ++i) sizes[i] = bytes;
#endif
  // Check if we are in a CUDA graph capture 
  capturing = ncclCudaGraphValid((&comm->planner)->capturingGraph);
  
  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, false, stream), ret, fail);

  //--------------Graph capture--------------
  // cudaMemcpyAsync is not supported during CUDA graph capture
  if (capturing) {

    // Copy own data to receive buffer if operation is out-of-place
    if (myRecvBuff != mySendBuff) {
      CUDACHECKGOTO(cudaMemcpyAsync(
        (void*)myRecvBuff,
        (void*)mySendBuff,
        bytes,
        cudaMemcpyDeviceToDevice,
        stream), ret, fail);
    }

    // Copy data from other ranks to receive buffer
    for (int offset = 1; offset < comm->nRanks; offset++) {
      int targetRank = (comm->rank + offset) % comm->nRanks;
      void* peerRecvBuff = peerUCSymPtr(comm, targetRank, myRecvBuff);
      CUDACHECKGOTO(cudaMemcpyAsync(
        (void*)peerRecvBuff,
        (void*)mySendBuff,
        bytes,
        cudaMemcpyDeviceToDevice,
        stream), ret, fail);
    }
  }
  //--------------No graph capture--------------
  else {
    // Use a separate stream for the local copy
    if (myRecvBuff != mySendBuff) {
        CUDACHECKGOTO(cudaMemcpyAsync(
        (void*)myRecvBuff,
        (void*)mySendBuff,
        bytes,
        cudaMemcpyDeviceToDevice,
        comm->ceColl.ceLocalCopyStream), ret, fail);
    }

#if CUDA_VERSION >= 12080 && CUDART_VERSION >= 12080 
    // For CUDA 12.8+, use batch memory copy for better performance
    for (int offset = 1; offset < comm->nRanks; offset++) {
      int targetRank = (comm->rank + offset) % comm->nRanks;
      srcs[offset-1] = (void*)mySendBuff;
      dsts[offset-1] = (void*)peerUCSymPtr(comm, targetRank, myRecvBuff);
    }

    // Configure copy attributes for performance
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.flags = cudaMemcpyFlagPreferOverlapWithCompute;

    // Perform all transfers in a single batch operation
    CUDACHECKGOTO(cudaMemcpyBatchAsync(
      dsts,               /* dst array */
      srcs,               /* src array */
      sizes,              /* size array */
      (size_t)comm->nRanks-1,
      &attrs,
      attrIdxs,
      1,  // Using one set of attributes
      nullptr,
      stream), ret, fail);

#else 
    // For older CUDA versions, fall back to individual transfers
    for (int offset = 1; offset < comm->nRanks; offset++) {
      int targetRank = (comm->rank + offset) % comm->nRanks;

      void* peerRecvBuff = peerUCSymPtr(comm, targetRank, myRecvBuff);
      CUDACHECKGOTO(cudaMemcpyAsync(
        (void*)peerRecvBuff,
        (void*)mySendBuff,
        bytes,
        cudaMemcpyDeviceToDevice,
        stream), ret, fail);
    }
#endif
    
    // Ensure the local copy stream is complete
    if (myRecvBuff != mySendBuff) {
      CUDACHECKGOTO(cudaEventRecord(comm->ceColl.ceLocalCopyEvent, comm->ceColl.ceLocalCopyStream), ret, fail);
      CUDACHECKGOTO(cudaStreamWaitEvent(stream, comm->ceColl.ceLocalCopyEvent, 0), ret, fail);
    }
  }

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, true, stream), ret, fail);
  
exit:
#if CUDA_VERSION >= 12080 && CUDART_VERSION >= 12080 
  if (srcs)     free(srcs);
  if (dsts)     free(dsts);
  if (sizes)    free(sizes);
  if (attrIdxs) free(attrIdxs);
#endif
  return ret;
fail:
  goto exit;
}


ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t stream = comm->planner.streams->stream;
  struct ncclCeCollArgs* args = plan->ceCollArgs;

  switch (args->func) {
    case ncclFuncAllGather:
      NCCLCHECKGOTO(ncclCeAllGather(comm, args, stream), ret, fail);
      break;
    default:
      ret = ncclInvalidUsage;
  }

exit:
  return ret;
fail:
  goto exit;
}
