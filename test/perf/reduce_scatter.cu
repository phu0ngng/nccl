/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * ReduceScatter Performance Test Implementation
 *
 * This file implements multiple ReduceScatter kernel variants optimized for different
 * use cases within CUDA P2P connectivity.
 *
 * IMPORTANT: LSA kernels require CUDA P2P connectivity since they require Load-Store Accessible (LSA) memory.
 *            Multimem kernels require multimem support for atomic reduction operations.
 *
 * Kernel Selection Strategy:
 * - deviceImpl = 0: NCCL's built-in ReduceScatter implementation (fallback)
 * - deviceImpl = 1: reduceScatterLsaThreadKernel - LSA with thread-level cooperation (ncclCoopThread).
 * - deviceImpl = 2: reduceScatterLsaCtaKernel - LSA with CTA-level cooperation (ncclCoopCta) for warp-level memory coalescing.
 * - deviceImpl = 3: reduceScatterMultimemThreadKernel - Multimem with thread-level cooperation (ncclCoopThread).
 * - deviceImpl = 4: reduceScatterMultimemCtaKernel - Multimem with CTA-level cooperation (ncclCoopCta) for warp-level memory coalescing.
 */

#include "cuda_runtime.h"
#include "common.h"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
// ReduceCopy API (including multimem and vector utilities) now included via nccl_device.h
#endif

void ReduceScatterGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  size_t base = (count/nranks) & -(16/eltSize);
  *sendcount = base*nranks;
  *recvcount = base;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = base;
  *paramcount = base;
}

testResult_t ReduceScatterInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount;
  size_t recvcount;
  int nranks, rank;
  void* data;

  for (int id = 0; id < args->commNum; ++id) {
    for (int i = 0; i < args->nGpus; i++) {
      CUDACHECK(cudaSetDevice(args->gpus[i]));
      sendcount = args->sendBytes[id][i] / wordSize(type);
      recvcount = args->expectedBytes[id][i] / wordSize(type);
      NCCLCHECK(ncclCommUserRank(args->comms[id][i], &rank));
      NCCLCHECK(ncclCommCount(args->comms[id][i], &nranks));
      CUDACHECK(cudaMemset(args->recvbuffs[id][i], 0, args->expectedBytes[id][i]));
      data = in_place ? args->recvbuffs[id][i] : args->sendbuffs[id][i];
      TESTCHECK(InitData(data, sendcount, 0, type, op, rep, nranks, rank));
      CUDACHECK(cudaMemcpy(args->expected[id][i], args->recvbuffs[id][i], args->expectedBytes[id][i], cudaMemcpyDefault));
      TESTCHECK(InitDataReduce(args->expected[id][i], recvcount, rank * recvcount, type, op, rep, nranks));
      CUDACHECK(cudaDeviceSynchronize());
    }
  }

  return testSuccess;
}

void ReduceScatterGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1))/((double)nranks);
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// set devComm reqs for reduce scatter device kernels
testResult_t ReduceScatterGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm, const char** testSkipReason) {
  if (!reqs || !comm) return testInternalError;

  ncclCommProperties_t commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
  if (ncclCommQueryProperties(comm, &commProperties) != ncclSuccess) {
    return testNcclError;
  }

  switch(deviceImpl) {
    case 0: // NCCL's built-in implementation
      return testSuccess;
    case 1: // reduceScatterLsaThreadKernel
    case 2: // reduceScatterLsaCtaKernel
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    case 3: // reduceScatterMultimemThreadKernel
    case 4: // reduceScatterMultimemCtaKernel
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
bool ReduceScatterGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;
  memset(reqs, 0, sizeof(*reqs));

  switch(deviceImpl) {
    case 1: // reduceScatterLsaThreadKernel
    case 2: // reduceScatterLsaCtaKernel
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
    case 3: // reduceScatterMultimemThreadKernel
    case 4: // reduceScatterMultimemCtaKernel
      reqs->lsaMultimem = true;
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
    default:
      return false;
  }
}
#endif

/*
 * Kernel 1: reduceScatterLsaThreadKernel - Thread-level LSA-based ReduceScatter
 *
 * Purpose: Provides a simple ReduceScatter implementation with thread-level granularity.
 * Each thread independently reduces its portion of this rank's chunk.
 *
 * Solution: This rank's chunk is divided among all threads (across all blocks). Each
 * thread independently calls the generic ncclLsaReduceSum helper on its portion.
 *
 * Key Optimizations:
 * - LSA barriers for faster synchronization than global barriers
 * - Direct peer access within CUDA P2P connectivity for optimal bandwidth
 * - Thread-level granularity (no thread cooperation overhead)
 * - All threads on this rank cooperate on this rank's chunk
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - This kernel requires all participating
 * ranks to be within the same CUDA P2P connectivity.
 *
 * Use Case: Small to medium messages, or when thread-level granularity is preferred.
 */
template <typename T>
__global__ void reduceScatterLsaThreadKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  // Create cooperative group once and reuse
  ncclCoopCta ctaCoop;
  const ncclTeam team = ncclTeamLsa(devComm);

  ncclLsaBarrierSession<ncclCoopCta> bar { ctaCoop, devComm, team, devComm.lsaBarrier, blockIdx.x };
  bar.sync(ctaCoop, cuda::memory_order_relaxed);

  const int rank = devComm.rank;

  // Calculate this rank's chunk boundaries in the input buffers
  const size_t chunkSize = count;  // count is already per-rank
  const size_t myChunkOffset = rank * chunkSize;

  // Split this rank's chunk across all threads (in all blocks)
  const int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  const int nThreads = blockDim.x * gridDim.x;

  // Simple work distribution: divide chunkSize evenly among threads
  const size_t eltsPerThread = (chunkSize + nThreads - 1) / nThreads;
  const size_t threadStart = threadId * eltsPerThread;
  // CRITICAL: Check if threadStart >= chunkSize to prevent unsigned underflow
  // If threadStart >= chunkSize, this thread has no work (threadCount = 0)
  const size_t threadCount = (threadStart < chunkSize) ? min(eltsPerThread, chunkSize - threadStart) : 0;

  if (threadCount > 0) {
    // Create a thread-level cooperative group (coop size = 1)
    ncclCoopThread threadCoop;

    // Get pointer to this thread's portion of the output buffer
    T* recvPtr = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    T* dstPtr = recvPtr + threadStart;

    // Calculate source offset for this thread's portion
    // Points to (myChunkOffset + threadStart) in each peer's send buffer
    size_t srcOffset = sendoffset + (myChunkOffset + threadStart) * sizeof(T);

    // Use the generic reduceSum helper (thread-level, no cooperation)
    // Uses the window API that takes windows and offsets directly
    ncclLsaReduceSum<T, ncclCoopThread, size_t>(threadCoop, sendwin, srcOffset, dstPtr, threadCount, team);
  }

  bar.sync(ctaCoop, cuda::memory_order_release);
}

/*
 * Kernel 2: reduceScatterLsaCtaKernel - CTA-level LSA-based ReduceScatter
 *
 * Purpose: High-performance ReduceScatter implementation using CTA-level cooperation
 * with warp-level memory coalescing for maximum memory bandwidth.
 *
 * Solution: This rank's chunk is divided among all local blocks. Each block uses
 * the CTA-level ncclLsaReduceSum helper which splits threads into warps for optimal
 * memory coalescing patterns.
 *
 * Key Optimizations:
 * - LSA barriers for faster synchronization than global barriers
 * - Direct peer access within CUDA P2P connectivity for optimal bandwidth
 * - CTA-level cooperation with automatic warp splitting for memory coalescing
 * - Warp-coordinated memory access patterns for optimal bandwidth
 * - Chunking allows all local blocks to cooperate on this rank's chunk
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - Same as basic LSA kernel. Requires
 * CUDA P2P connectivity due to LSA memory access patterns.
 *
 * Use Case: Large messages where maximum memory bandwidth is critical.
 */
template <typename T>
__global__ void reduceScatterLsaCtaKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  // Create cooperative group once and reuse
  ncclCoopCta ctaCoop;
  const ncclTeam team = ncclTeamLsa(devComm);

  ncclLsaBarrierSession<ncclCoopCta> bar { ctaCoop, devComm, team, devComm.lsaBarrier, blockIdx.x };
  bar.sync(ctaCoop, cuda::memory_order_relaxed);

  const int rank = devComm.rank;

  // Calculate this rank's chunk boundaries in the input buffers
  const size_t chunkSize = count;  // count is already per-rank
  const size_t myChunkOffset = rank * chunkSize;

  // Calculate work distribution: each block gets a fixed chunk of work
  // Simple division: divide chunkSize evenly among blocks
  const size_t eltsPerBlock = chunkSize / gridDim.x;
  const size_t remainder = chunkSize % gridDim.x;

  // Calculate this block's starting offset and chunk size
  // Distribute remainder one elt at a time to first blocks
  const size_t blockStart = blockIdx.x * eltsPerBlock + min((size_t)blockIdx.x, remainder);
  const size_t blockCount = eltsPerBlock + (blockIdx.x < remainder ? 1 : 0);

  // Process this block's assigned chunk
  if (blockCount > 0) {
    // Get pointer to this block's portion of the output buffer
    T* recvPtr = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    T* dstPtr = recvPtr + blockStart;

    // Calculate source offset for this block's portion
    // Points to (myChunkOffset + blockStart) in each peer's send buffer
    size_t srcOffset = sendoffset + (myChunkOffset + blockStart) * sizeof(T);

    // Use the CTA-level reduceSum helper with default UNROLL factor
    // Uses the window API that takes windows and offsets directly
    ncclLsaReduceSum<T, ncclCoopCta, size_t>(ctaCoop, sendwin, srcOffset, dstPtr, blockCount, team);
  }

  bar.sync(ctaCoop, cuda::memory_order_release);
}

/*
 * Kernel 3: reduceScatterMultimemThreadKernel - Thread-level Multimem-based ReduceScatter
 *
 * Purpose: Provides a ReduceScatter implementation using multimem operations with thread-level granularity.
 * Each thread independently reduces its portion of this rank's chunk using multimem loads.
 *
 * Solution: This rank's chunk is divided among all threads (across all blocks). Each
 * thread independently calls the generic ncclMultimemReduceSum helper on its portion.
 *
 * Key Optimizations:
 * - Multimem atomic operations for reduction across ranks
 * - Thread-level granularity (no thread cooperation overhead)
 * - All threads on this rank cooperate on this rank's chunk
 *
 * Use Case: Small to medium messages, or when thread-level granularity is preferred with multimem.
 */
template <typename T>
__global__ void reduceScatterMultimemThreadKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  // Thread divisibility requirement: align to 16 bytes when possible
  constexpr int THREAD_DIVISIBILITY = (16 % sizeof(T) == 0) ? (16 / sizeof(T)) : 1;

  // Get multimem handle from devComm
  ncclMultimemHandle multimemHandle = devComm.lsaMultimem;

  // Create barrier session with multimem flag
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, true };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = devComm.rank;

  // Calculate this rank's chunk boundaries in the input buffers
  const size_t chunkSize = count;  // count is already per-rank
  const size_t myChunkOffset = rank * chunkSize;

  // Split this rank's chunk across all threads (in all blocks)
  const int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  const int nThreads = blockDim.x * gridDim.x;

  // Calculate work distribution: each thread gets a chunk divisible by THREAD_DIVISIBILITY
  const size_t totalDivisibleElts = (chunkSize / THREAD_DIVISIBILITY) * THREAD_DIVISIBILITY;
  const size_t eltsPerThread = (totalDivisibleElts / nThreads / THREAD_DIVISIBILITY) * THREAD_DIVISIBILITY;
  const size_t remainder = chunkSize - eltsPerThread * nThreads;

  // Calculate this thread's starting offset and chunk size
  // All threads except the last get eltsPerThread elts (divisible by THREAD_DIVISIBILITY)
  // The last thread gets eltsPerThread + remainder
  const size_t threadStart = threadId * eltsPerThread;
  const size_t threadCount = (threadId == nThreads - 1) ? eltsPerThread + remainder : eltsPerThread;

  if (threadCount > 0) {
    // Create a thread-level cooperative group (coop size = 1)
    ncclCoopThread threadCoop;

    // Calculate source offset for this thread's portion
    // Points to (myChunkOffset + threadStart) in each peer's send buffer
    size_t srcOffset = sendoffset + (myChunkOffset + threadStart) * sizeof(T);

    // Get pointer to this thread's portion of the output buffer
    T* recvPtr = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    T* dstPtr = recvPtr + threadStart;

    // Use the generic multimem reduceSum helper (thread-level, no cooperation)
    // Uses the window API that takes windows and offsets directly
    ncclMultimemReduceSum<T, ncclCoopThread, size_t>(threadCoop, sendwin, srcOffset, dstPtr, threadCount, multimemHandle);
  }

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

/*
 * Kernel 4: reduceScatterMultimemCtaKernel - CTA-level Multimem-based ReduceScatter
 *
 * Purpose: High-performance ReduceScatter implementation using CTA-level cooperation
 * with warp-level memory coalescing and multimem operations for maximum memory bandwidth.
 *
 * Solution: This rank's chunk is divided among all local blocks. Each block uses
 * the CTA-level ncclMultimemReduceSum helper which splits threads into warps for
 * optimal memory coalescing patterns.
 *
 * Key Optimizations:
 * - Multimem atomic operations for reduction across ranks
 * - CTA-level cooperation with automatic warp splitting for memory coalescing
 * - Warp-coordinated memory access patterns for optimal bandwidth
 * - Chunking allows all local blocks to cooperate on this rank's chunk
 *
 * Use Case: Large messages where maximum memory bandwidth is critical with multimem.
 */
template <typename T>
__global__ void reduceScatterMultimemCtaKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
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

  // Calculate this rank's chunk boundaries in the input buffers
  const size_t chunkSize = count;  // count is already per-rank
  const size_t myChunkOffset = rank * chunkSize;

  // Calculate work distribution: each block gets a fixed chunk of work
  // Ensure eltsPerBlock is divisible by BLOCK_DIVISIBILITY
  const size_t totalDivisibleElts = (chunkSize / BLOCK_DIVISIBILITY) * BLOCK_DIVISIBILITY;
  const size_t eltsPerBlock = (totalDivisibleElts / gridDim.x / BLOCK_DIVISIBILITY) * BLOCK_DIVISIBILITY;
  const size_t remainder = chunkSize - eltsPerBlock * gridDim.x;

  // Calculate this block's starting offset and chunk size
  // All blocks except the last get eltsPerBlock elts (divisible by BLOCK_DIVISIBILITY)
  // The last block gets eltsPerBlock + remainder
  const size_t blockStart = blockIdx.x * eltsPerBlock;
  const size_t blockCount = (blockIdx.x == gridDim.x - 1) ? eltsPerBlock + remainder : eltsPerBlock;

  // Process this block's assigned chunk
  if (blockCount > 0) {
    // Calculate source offset for this block's portion
    // Points to (myChunkOffset + blockStart) in each peer's send buffer
    size_t srcOffset = sendoffset + (myChunkOffset + blockStart) * sizeof(T);

    // Get pointer to this block's portion of the output buffer
    T* recvPtr = (T*)ncclGetLocalPointer(recvwin, recvoffset);
    T* dstPtr = recvPtr + blockStart;

    // Use the CTA-level multimem reduceSum helper with default UNROLL factor
    // Uses the window API that takes windows and offsets directly
    ncclMultimemReduceSum<T, ncclCoopCta, size_t>(ctaCoop, sendwin, srcOffset, dstPtr, blockCount, multimemHandle);
  }

  bar.sync(ctaCoop, cuda::memory_order_release);
}

testResult_t ReduceScatterRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {
  char* sptr = (char*)sendbuff + sendoffset;
  char* rptr = (char*)recvbuff + recvoffset;

  switch (deviceImpl) {
  case 0:
    NCCLCHECK_COMM_WAIT(ncclReduceScatter(sptr, rptr, count, type, op, comm, stream), comm);
    return testSuccess;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  case 1:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(reduceScatterLsaThreadKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
  case 2:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(reduceScatterLsaCtaKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
  case 3:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_MULTIMEM(reduceScatterMultimemThreadKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, true));
    return testSuccess;
  case 4:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_MULTIMEM(reduceScatterMultimemCtaKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, true));
    return testSuccess;
#endif
  }

  return testNotImplemented;
}

struct testColl reduceScatterTest = {
  "ReduceScatter",
  ReduceScatterGetCollByteCount,
  ReduceScatterInitData,
  ReduceScatterGetBw,
  ReduceScatterRunColl
};

void ReduceScatterGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ReduceScatterGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t ReduceScatterRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &reduceScatterTest;
  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if ((int)op != -1) {
    run_ops = &op;
    run_opnames = &opName;
    op_count = 1;
  } else {
    op_count = test_opnum;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  for (int i=0; i<type_count; i++) {
    if (!isFp8ValidForReductions(run_types[i])) {
	    if ((int)type != -1) {
            printf("SKIP: FP8 reduction operations require sm90+ hardware\n");
        }
      continue;
    }
    for (int j=0; j<op_count; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1));
    }
  }
  return testSuccess;
}

struct testEngine reduceScatterEngine = {
  .getBuffSize = ReduceScatterGetBuffSize,
  .runTest = ReduceScatterRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  .getDevCommRequirements = ReduceScatterGetDevCommRequirements
#endif
};

#pragma weak ncclTestEngine=reduceScatterEngine
