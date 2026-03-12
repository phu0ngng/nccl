/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * AllGather Performance Test Implementation
 *
 * This file implements multiple AllGather kernel variants optimized for different
 * use cases within CUDA P2P connectivity.
 *
 * IMPORTANT: LSA kernels require CUDA P2P connectivity since they require Load-Store Accessible (LSA) memory.
 *            Multimem kernels require multimem support for atomic gather operations.
 *
 * Performance Note: The CTA-level kernels (2 and 4) achieve better performance than the thread-level
 *                   kernels (1 and 3) due to CTA-level cooperation (ncclCoopCta) which enables warp-level
 *                   memory coalescing. Thread-level cooperation (ncclCoopThread) in kernels 1 and 3 does
 *                   not enable this optimization.
 *
 * Kernel Selection Strategy:
 * - deviceImpl = 0: NCCL's built-in AllGather implementation (fallback)
 * - deviceImpl = 1: allGatherLsaKernel - LSA with peer loops (float/double only; same style as AllReduce kernel 1).
 * - deviceImpl = 2: allGatherLsaCtaKernel - LSA with CTA-level cooperation (ncclCoopCta) for warp-level memory coalescing.
 * - deviceImpl = 3: allGatherMultimemKernel - Multimem with multimem_ops.h (float/double only; same style as AllReduce kernel 3).
 * - deviceImpl = 4: allGatherMultimemCtaKernel - Multimem with CTA-level cooperation (ncclCoopCta) for warp-level memory coalescing.
 * - deviceImpl = 10 (HOST_RMA_IMPL): AllGatherRmaPut - Host-side RMA implementation using ncclPut APIs.
 */

#include "cuda_runtime.h"
#include "common.h"
#include <algorithm>
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#endif
#include "multimem_ops.h"

void AllGatherGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t base = (count/nranks) & -(16/eltSize);
  *sendcount = base;
  *recvcount = base*nranks;
  *sendInplaceOffset = base;
  *recvInplaceOffset = 0;
  *paramcount = base;
}

testResult_t AllGatherInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount;
  int nranks, rank;
  void* data;
  for (int id = 0; id < args->commNum; ++id) {
    for (int i = 0; i < args->nGpus; i++) {
      CUDACHECK(cudaSetDevice(args->gpus[i]));
      sendcount = args->sendBytes[id][i] / wordSize(type);
      NCCLCHECK(ncclCommUserRank(args->comms[id][i], &rank));
      NCCLCHECK(ncclCommCount(args->comms[id][i], &nranks));
      CUDACHECK(cudaMemset(args->recvbuffs[id][i], 0, args->expectedBytes[id][i]));
      data = in_place ? ((char*)args->recvbuffs[id][i]) + rank * args->sendBytes[id][i] : args->sendbuffs[id][i];
      TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33 * rep + rank, 1, 0));
      for (int j = 0; j < nranks; j++) {
        TESTCHECK(InitData((char*)args->expected[id][i] + args->sendBytes[id][i] * j, sendcount, 0, type, ncclSum, 33 * rep + j, 1, 0));
      }
      CUDACHECK(cudaDeviceSynchronize());
    }
  }

  return testSuccess;
}

void AllGatherGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1))/((double)nranks);
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// set devComm reqs for allgather device kernels
testResult_t AllGatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm, const char** testSkipReason) {
  if (!reqs || !comm) return testInternalError;

  ncclCommProperties_t commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
  if (ncclCommQueryProperties(comm, &commProperties) != ncclSuccess) {
    return testNcclError;
  }

  if (deviceImpl > 0 && commProperties.nRanks != ncclTeamLsa(comm).nRanks) {
    *testSkipReason = "DeviceImplementation >= 1 requires CUDA P2P connectivity "
                      "across all ranks. Not all ranks of this communicator "
                      "have P2P connectivity.\n";
    return testSkipped;
  }

  switch(deviceImpl) {
    case 0: // NCCL's built-in implementation
      return testSuccess;
    case 1: // allGatherLsaKernel
    case 2: // allGatherLsaCtaKernel
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    case 3: // allGatherMultimemKernel
    case 4: // allGatherMultimemCtaKernel
      if (!commProperties.multimemSupport) {
        *testSkipReason = "multimem not supported";
        return testSkipped;
      }
      reqs->lsaMultimem = true;
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    default:
      return testNotImplemented;
  }
}
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
bool AllGatherGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
    case 1: // allGatherLsaKernel
    case 2: // allGatherLsaCtaKernel
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
    case 3: // allGatherMultimemKernel
    case 4: // allGatherMultimemCtaKernel
      reqs->lsaMultimem = true;
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
    default:
      return false;
  }
}
#endif

/*
 * Kernel 1: allGatherLsaKernel - LSA-based AllGather with peer loops (float/double only)
 *
 * Purpose: Simple AllGather using direct LSA peer access. Each rank copies its send chunk
 * to the (rank*count) offset in every peer's receive buffer.
 *
 * Solution: Grid-stride loop over count; for each element, read from our send buffer and
 * write to all peers' recv buffers at rank*count + offset. Specialized for float/double only.
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - requires all ranks in same CUDA P2P connectivity.
 */
template <typename T>
__global__ void allGatherLsaKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = devComm.rank, nRanks = devComm.nRanks;

  const int globalTid = threadIdx.x + blockDim.x * blockIdx.x;
  const int globalNthreads = blockDim.x * gridDim.x;

  T* mySendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    T v = mySendPtr[offset];
    for (int peer = 0; peer < nRanks; peer++) {
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[rank * count + offset] = v;
    }
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

/*
 * Kernel 2: allGatherLsaCtaKernel - CTA-level LSA-based AllGather
 *
 * Purpose: High-performance AllGather implementation using CTA-level cooperation (ncclCoopCta)
 * for warp-level memory coalescing and maximum memory bandwidth.
 *
 * Solution: This rank's chunk is divided among all local blocks. Each block uses
 * the CTA-level ncclLsaCopy helper which enables warp-level memory coalescing.
 *
 * Key Optimizations:
 * - LSA barriers for faster synchronization than global barriers
 * - Direct peer access within CUDA P2P connectivity for optimal bandwidth
 * - CTA-level cooperation (ncclCoopCta) enables warp-level memory coalescing
 * - Warp-coordinated access patterns for optimal memory bandwidth
 * - Chunking allows all local blocks to cooperate on this rank's chunk
 *
 * Performance Advantage over Kernel 1: CTA-level cooperation allows warp-level
 * memory coalescing, significantly improving bandwidth compared to thread-level cooperation.
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - Same as basic LSA kernel. Requires
 * CUDA P2P connectivity due to LSA memory access patterns.
 *
 * Use Case: Large messages where maximum memory bandwidth is critical.
 */
template <typename T>
__global__ void allGatherLsaCtaKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  // Create cooperative group once and reuse
  ncclCoopCta ctaCoop;
  const ncclTeam team = ncclTeamLsa(devComm);

  ncclLsaBarrierSession<ncclCoopCta> bar { ctaCoop, devComm, team, devComm.lsaBarrier, blockIdx.x };
  bar.sync(ctaCoop, cuda::memory_order_relaxed);

  const int rank = devComm.rank;

  // Calculate work distribution: each block gets a fixed chunk of work
  // Simple division: divide count evenly among blocks
  const size_t eltsPerBlock = count / gridDim.x;
  const size_t remainder = count % gridDim.x;

  // Calculate this block's starting offset and chunk size
  // Distribute remainder one elt at a time to first blocks
  const size_t blockStart = blockIdx.x * eltsPerBlock + ((size_t)blockIdx.x < remainder ? (size_t)blockIdx.x : remainder);
  const size_t blockCount = eltsPerBlock + (blockIdx.x < remainder ? 1 : 0);

  // Process this block's assigned chunk
  if (blockCount > 0) {
    // Get pointer to this rank's send buffer (source)
    T* mySendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* srcPtr = mySendPtr + blockStart;

    // Calculate destination offset for this block's portion
    // Points to (rank * count + blockStart) in each peer's receive buffer
    size_t dstOffset = recvoffset + (rank * count + blockStart) * sizeof(T);

    // Use the CTA-level copy helper with default UNROLL factor
    // Uses the window API that takes windows and offsets directly
    ncclLsaCopy<T, ncclCoopCta, size_t>(ctaCoop, srcPtr, recvwin, dstOffset, blockCount, team);
  }

  bar.sync(ctaCoop, cuda::memory_order_release);
}

/*
 * Kernel 3: allGatherMultimemKernel - Multimem-based AllGather with multimem_ops.h (float/double only)
 *
 * Purpose: AllGather using multimem store to broadcast this rank's send chunk to all peers'
 * receive buffers at offset rank*count.
 *
 * Solution: Grid-stride loop; for each element load from local send and multimemStore to
 * recv_ptr + rank*count + offset (broadcasts to all peers). Same style as AllReduce kernel 3.
 *
 * Hardware Requirements: Hopper+ with multimem support.
 */
template <typename T>
__global__ void allGatherMultimemKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, true };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = devComm.rank;

  // Read from single local send buffer; write to all peers via multimem (broadcast)
  T* send_ptr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  T* recv_ptr = reinterpret_cast<T*>(ncclGetLsaMultimemPointer(recvwin, recvoffset, devComm));

  const int globalTid = threadIdx.x + blockDim.x * blockIdx.x;
  const int globalNthreads = blockDim.x * gridDim.x;

  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    multimemStore(recv_ptr + rank * count + offset, send_ptr[offset]);
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

/*
 * Kernel 4: allGatherMultimemCtaKernel - CTA-level Multimem-based AllGather
 *
 * Purpose: High-performance AllGather implementation using CTA-level cooperation (ncclCoopCta)
 * with multimem operations for warp-level memory coalescing and maximum memory bandwidth.
 *
 * Solution: This rank's chunk is divided among all local blocks. Each block uses
 * the CTA-level ncclMultimemCopy helper which enables warp-level memory coalescing.
 *
 * Key Optimizations:
 * - Multimem atomic operations for gather across ranks
 * - CTA-level cooperation (ncclCoopCta) enables warp-level memory coalescing
 * - Warp-coordinated access patterns for optimal memory bandwidth
 * - Chunking allows all local blocks to cooperate on this rank's chunk
 *
 * Performance Advantage over Kernel 3: CTA-level cooperation allows warp-level
 * memory coalescing, significantly improving bandwidth compared to thread-level cooperation.
 *
 * Use Case: Large messages where maximum memory bandwidth is critical with multimem.
 */
template <typename T>
__global__ void allGatherMultimemCtaKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  // Block divisibility requirement: align to 16 bytes when possible
  constexpr int BLOCK_DIVISIBILITY = (16 % sizeof(T) == 0) ? (16 / sizeof(T)) : 1;

  // Get multimem handle from devComm
  ncclMultimemHandle multimemHandle = devComm.lsaMultimem;

  // Create cooperative group once and reuse
  ncclCoopCta ctaCoop;

  // Create barrier session with multimem flag
  ncclLsaBarrierSession<ncclCoopCta> bar { ctaCoop, devComm, ncclTeamTagLsa(), blockIdx.x, true };
  bar.sync(ctaCoop, cuda::memory_order_relaxed);

  const int rank = devComm.rank;

  // Calculate work distribution: each block gets a fixed chunk of work
  // Ensure eltsPerBlock is divisible by BLOCK_DIVISIBILITY
  const size_t totalDivisibleElts = (count / BLOCK_DIVISIBILITY) * BLOCK_DIVISIBILITY;
  const size_t eltsPerBlock = (totalDivisibleElts / gridDim.x / BLOCK_DIVISIBILITY) * BLOCK_DIVISIBILITY;
  const size_t remainder = count - eltsPerBlock * gridDim.x;

  // Calculate this block's starting offset and chunk size
  // All blocks except the last get eltsPerBlock elts (divisible by BLOCK_DIVISIBILITY)
  // The last block gets eltsPerBlock + remainder
  const size_t blockStart = blockIdx.x * eltsPerBlock;
  const size_t blockCount = (blockIdx.x == gridDim.x - 1) ? eltsPerBlock + remainder : eltsPerBlock;

  // Process this block's assigned chunk
  if (blockCount > 0) {
    // Get pointer to this rank's send buffer (source)
    T* mySendPtr = (T*)ncclGetLocalPointer(sendwin, sendoffset);
    T* srcPtr = mySendPtr + blockStart;

    // Calculate destination offset for this block's portion
    // Points to (rank * count + blockStart) in each peer's receive buffer
    size_t dstOffset = recvoffset + (rank * count + blockStart) * sizeof(T);

    // Use the CTA-level multimem copy helper with default UNROLL factor
    // Uses the window API that takes windows and offsets directly
    ncclMultimemCopy<T, ncclCoopCta, size_t>(ctaCoop, srcPtr, recvwin, dstOffset, blockCount, multimemHandle);
  }

  bar.sync(ctaCoop, cuda::memory_order_release);
}

/*
 * AllGather implementation using RMA host put APIs
 */
testResult_t AllGatherRmaPut(void* sendWindow, size_t sendoffset, void* recvWindow, size_t recvoffset,
                             size_t count, ncclDataType_t type, ncclComm_t comm, cudaStream_t stream) {
  int rank, nranks;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  NCCLCHECK(ncclCommCount(comm, &nranks));

  ncclWindow_t sendWin = (ncclWindow_t)sendWindow;
  ncclWindow_t recvWin = (ncclWindow_t)recvWindow;

  void* sendPtr = NULL;
  void* recvPtr = NULL;
  NCCLCHECK(ncclWinGetUserPtr(comm, sendWin, &sendPtr));
  NCCLCHECK(ncclWinGetUserPtr(comm, recvWin, &recvPtr));

  // Calculate elt size and total bytes to transfer
  size_t eltSize = wordSize(type);
  size_t bytes = count * eltSize;

  // Use RMA context 0
  int ctx = 0;

  // Check if it is in-place
  bool isInPlace = ((char*)sendPtr + sendoffset == (char*)recvPtr + recvoffset + rank * bytes);

  // Calculate where this rank's data should go in each peer's receive buffer
  // In allgather, rank i's data goes to offset: recvoffset + rank * bytes
  size_t peerWinOffset = recvoffset + rank * bytes;

  // Build descriptors array for signal waiting
  ncclWaitSignalDesc_t* waitDescs = (ncclWaitSignalDesc_t*)malloc(sizeof(ncclWaitSignalDesc_t) * nranks);
  if (waitDescs == NULL) {
    return testInternalError;
  }

  int descIdx = 0;
  for (int i = 0; i < nranks; i++) {
    // Skip waiting for signal from ourselves if in-place
    if (isInPlace && i == rank) {
      continue;
    }
    waitDescs[descIdx].opCnt = 1;  // Expect 1 signal from each peer
    waitDescs[descIdx].peer = i;
    waitDescs[descIdx].sigIdx = 0;
    waitDescs[descIdx].ctx = ctx;
    descIdx++;
  }

  NCCLCHECK(ncclGroupStart());

  // Send each chunk to its destination peer
  for (int peer = 0; peer < nranks; peer++) {
    int targetRank = (rank + peer) % nranks;
    // Skip sending to self if in-place
    if (isInPlace && targetRank == rank) {
      continue;
    }
    NCCLCHECK(ncclPutSignal((char*)sendPtr + sendoffset, count, type, targetRank,
                      recvWin, peerWinOffset, 0, ctx, 0, comm, stream));
  }

  NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);

  // Wait for signals from all peers to ensure all data has been written
  NCCLCHECK(ncclWaitSignal(descIdx, waitDescs, comm, stream));

  // Free allocated memory
  free(waitDescs);

  return testSuccess;
}

testResult_t AllGatherRunColl(void* sendbuff,  size_t sendoffset,void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {

  char* sptr = (char*)sendbuff + sendoffset;
  char* rptr = (char*)recvbuff + recvoffset;

  switch (deviceImpl) {
  case 0:
    // NCCL built-in AllGather
    NCCLCHECK_COMM_WAIT(ncclAllGather(sptr, rptr, count, type, comm, stream), comm);
    return testSuccess;
  case HOST_RMA_IMPL:
    // RMA host put implementation
    TESTCHECK(AllGatherRmaPut(sendbuff, sendoffset, recvbuff, recvoffset, count, type, comm, stream));
    return testSuccess;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  case 1:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_FLOAT_DOUBLE(allGatherLsaKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
  case 2:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(allGatherLsaCtaKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
  case 3:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_FLOAT_DOUBLE(allGatherMultimemKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, true));
    return testSuccess;
  case 4:
    // AllGather doesn't use ld_reduce, so use regular specialization (not multimem-specific)
    // This will trigger compile-time error if multimem is incorrectly used (except fp4, which is LSA-only)
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(allGatherMultimemCtaKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
#endif
  }

  return testNotImplemented;
}

struct testColl allGatherTest = {
  "AllGather",
  AllGatherGetCollByteCount,
  /*initConfig=*/NULL,
  AllGatherInitData,
  AllGatherGetBw,
  AllGatherRunColl
};

void AllGatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllGatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AllGatherRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &allGatherTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

struct testEngine allGatherEngine = {
  .getBuffSize = AllGatherGetBuffSize,
  .runTest = AllGatherRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = AllGatherGetDevCommRequirements
#endif
};

#pragma weak ncclTestEngine=allGatherEngine
