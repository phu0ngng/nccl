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
 * - deviceImpl = 1: reduceScatterLsaKernel - LSA with peer loops (float/double only; same style as AllReduce kernel 1).
 * - deviceImpl = 2: reduceScatterLsaCtaKernel - LSA with CTA-level cooperation (ncclCoopCta) for warp-level memory coalescing.
 * - deviceImpl = 3: reduceScatterMultimemKernel - Multimem with multimem_ops.h (float/double only; same style as AllReduce kernel 3).
 * - deviceImpl = 4: reduceScatterMultimemCtaKernel - Multimem with CTA-level cooperation (ncclCoopCta) for warp-level memory coalescing.
 */

#include "cuda_runtime.h"
#include "common.h"
#include <algorithm>
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#endif
#include "multimem_ops.h"

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

  if (deviceImpl > 0 && commProperties.nRanks != ncclTeamLsa(comm).nRanks) {
    *testSkipReason = "DeviceImplementation >= 1 requires CUDA P2P connectivity "
                      "across all ranks. Not all ranks of this communicator "
                      "have P2P connectivity.\n";
    return testSkipped;
  }

  switch(deviceImpl) {
    case 0: // NCCL's built-in implementation
      return testSuccess;
    case 1: // reduceScatterLsaKernel
    case 2: // reduceScatterLsaCtaKernel
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    case 3: // reduceScatterMultimemKernel
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
    case 1: // reduceScatterLsaKernel
    case 2: // reduceScatterLsaCtaKernel
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
    case 3: // reduceScatterMultimemKernel
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
 * Kernel 1: reduceScatterLsaKernel - LSA-based ReduceScatter with peer loops (float/double only)
 *
 * Purpose: Simple ReduceScatter using direct LSA peer access. Each rank reduces all peers'
 * send chunks for its segment and writes to local recv.
 *
 * Solution: Grid-stride loop over this rank's chunk (count elements); for each element,
 * sum over all peers' send at myChunkOffset+offset and write to recv[offset]. Same style as AllReduce kernel 1.
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - requires all ranks in same CUDA P2P connectivity.
 */
template <typename T>
__global__ void reduceScatterLsaKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const size_t myChunkOffset = rank * count;

  const int globalTid = threadIdx.x + blockDim.x * blockIdx.x;
  const int globalNthreads = blockDim.x * gridDim.x;

  T* recvPtr = (T*)ncclGetLocalPointer(recvwin, recvoffset);

  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    T v = T{0};
    for (int peer = 0; peer < nRanks; peer++) {
      T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, peer);
      v += sendPtr[myChunkOffset + offset];
    }
    recvPtr[offset] = v;
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
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
  bar.sync(ctaCoop, cuda::memory_order_acquire);

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
  const size_t blockStart = blockIdx.x * eltsPerBlock + ((size_t)blockIdx.x < remainder ? (size_t)blockIdx.x : remainder);
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
 * Kernel 3: reduceScatterMultimemKernel - Multimem-based ReduceScatter with multimem_ops.h (float/double only)
 *
 * Purpose: ReduceScatter using multimem load-reduce and store. For each element in this rank's
 * chunk, reduce from all peers and write to local recv. Same style as AllReduce kernel 3.
 *
 * Hardware Requirements: Hopper+ with multimem support.
 */
template <typename T>
__global__ void reduceScatterMultimemKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, true };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  const int rank = devComm.rank;

  // Read from many (multimem) via multimemLoadSum; write to single local recv buffer
  T* send_ptr = reinterpret_cast<T*>(ncclGetLsaMultimemPointer(sendwin, sendoffset, devComm));
  T* recv_ptr = (T*)ncclGetLocalPointer(recvwin, recvoffset);

  const size_t myChunkOffset = rank * count;

  const int globalTid = threadIdx.x + blockDim.x * blockIdx.x;
  const int globalNthreads = blockDim.x * gridDim.x;

  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    T v = multimemLoadSum<T, T>(send_ptr + myChunkOffset + offset);
    recv_ptr[offset] = v;
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
  bar.sync(ctaCoop, cuda::memory_order_acquire);

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
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_FLOAT_DOUBLE(reduceScatterLsaKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
  case 2:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(reduceScatterLsaCtaKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
  case 3:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_FLOAT_DOUBLE(reduceScatterMultimemKernel, type, op),
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
  /*initConfig=*/NULL,
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

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ ReduceScatterGetBuffSize,
  /* .runTest = */ ReduceScatterRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  /* .getDevCommRequirements = */ ReduceScatterGetDevCommRequirements
#endif
};
