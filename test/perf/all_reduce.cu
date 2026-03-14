/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * AllReduce Performance Test Implementation
 *
 * This file implements multiple AllReduce kernel variants optimized for different
 * use cases within CUDA P2P connectivity.
 * These kernels are designed to highlight the device API functionality. As well as how to optimize for best performance.
 *
 * IMPORTANT: All custom kernels require CUDA P2P connectivity since they require Load-Store Accessible (LSA) memory.
 *
 * Kernel Selection Strategy:
 * - deviceImpl = 0: NCCL's built-in AllReduce implementation (fallback)
 * - deviceImpl = 1: allReduceLsaKernel - Basic LSA with peer loops (v2.29.1 style; ncclGetLsaPointer).
 * - deviceImpl = 2: allReduceLsaReduceCopyKernel - LSA with CTA-level cooperation using ReduceCopy API (ncclLsaReduceSumCopy).
 * - deviceImpl = 3: allReduceMultimemKernel - Basic Multimem with thread-level loop (v2.29.1 style; multimem_ops.h).
 * - deviceImpl = 4: allReduceMultimemReduceCopyKernel - Multimem with CTA-level ReduceCopy API (ncclMultimemReduceSumCopy).
 */

#include "cuda_runtime.h"
#include "common.h"
#include <algorithm>
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#endif
#include "multimem_ops.h"

void AllReduceGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t AllReduceInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
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
      TESTCHECK(InitDataReduce(args->expected[id][i], recvcount, 0, type, op, rep, nranks));
      CUDACHECK(cudaDeviceSynchronize());
    }
  }

  return testSuccess;
}

void AllReduceGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(2*(nranks - 1)))/((double)nranks);
  *busBw = baseBw * factor;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// set devComm reqs for allreduce device kernels
testResult_t AllReduceGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm, const char** testSkipReason) {
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
    case 1: // allReduceLsaKernel
    case 2: // allReduceLsaReduceCopyKernel
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    case 3: // allReduceMultimemKernel
    case 4: // allReduceMultimemReduceCopyKernel
      if (!commProperties.multimemSupport) {
        *testSkipReason = "This test requires multimem support, but multimem support is not enabled for this communicator.\n";
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
bool AllReduceGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;

  switch(deviceImpl) {
    case 1: // allReduceLsaKernel
    case 2: // allReduceLsaReduceCopyKernel
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
    case 3: // allReduceMultimemKernel
    case 4: // allReduceMultimemReduceCopyKernel
      reqs->lsaMultimem = true;
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
    default:
      return false;
  }
}
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
/*
 * Kernel 1: allReduceLsaKernel - Basic LSA-based AllReduce
 *
 * Purpose: Provides a simple, deterministic AllReduce implementation for small to
 * medium message sizes within CUDA P2P connectivity.
 *
 * Solution: Implements AllReduce using direct peer-to-peer memory access through
 * LSA windows. Each rank reads from all other ranks, performs reduction, and
 * writes the result back to all ranks using cooperative thread arrays.
 *
 * Key Optimizations:
 * - LSA barriers for faster synchronization than global barriers
 * - Global grid stride loop to distribute work across all ranks
 * - Direct peer access within CUDA P2P connectivity for optimal bandwidth
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - This kernel requires all participating
 * ranks to be within the same CUDA P2P connectivity.
 *
 * Use Case: Small to medium messages (< 1MB) where simplicity and determinism
 * are more important than maximum bandwidth.
 */
template <typename T>
__global__ void allReduceLsaKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = devComm.rank, nRanks = devComm.nRanks;

  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    T v = T{0};
    for (int peer=0; peer<nRanks; peer++) {
      T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, peer);
      v += sendPtr[offset];
    }
    for (int peer=0; peer<nRanks; peer++) {
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[offset] = v;
    }
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

/*
 * Kernel 2: allReduceLsaReduceCopyKernel - LSA-based AllReduce using ReduceCopy API
 *
 * Purpose: Simplified AllReduce implementation using the ReduceCopy convenience API,
 * which handles vectorization, loop unrolling, and alignment automatically for
 * performance on large messages within CUDA P2P connectivity.
 *
 * Solution: Uses ncclLsaReduceSumCopy convenience function which internally handles
 * vectorized loads/stores, loop unrolling, and graceful alignment handling. The kernel
 * focuses on proper work distribution across blocks while the API handles optimization.
 *
 * Key Optimizations (handled by ReduceCopy API):
 * - Vector load/store instructions for improved memory bandwidth (128-bit operations)
 * - Loop unrolling to reduce loop overhead and improve instruction-level parallelism
 * - Warp-coalesced memory access patterns for optimal memory controller utilization
 * - Graceful handling of misaligned data with scalar fallback
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - Requires CUDA P2P connectivity
 * due to LSA memory access patterns.
 *
 * Use Case: Large messages where maximum memory bandwidth is critical.
 * The ReduceCopy API automatically optimizes for data alignment.
 */
template <typename T>
__global__ void allReduceLsaReduceCopyKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  // Create CTA-level cooperative group
  ncclCoopCta coop;

  // Create barrier session
  ncclLsaBarrierSession<ncclCoopCta> bar { coop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(coop, cuda::memory_order_relaxed);

  // Calculate work distribution: each block gets a fixed chunk of work
  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const int globalBlockIdx = rank + blockIdx.x * nRanks;
  const int globalNumBlocks = gridDim.x * nRanks;

  // Simple work distribution: divide count evenly among blocks
  const size_t eltsPerBlock = count / globalNumBlocks;
  const size_t remainder = count % globalNumBlocks;

  // Calculate this block's starting offset and chunk size
  // Distribute remainder one elt at a time to first blocks
  const size_t blockStart = globalBlockIdx * eltsPerBlock + ((size_t)globalBlockIdx < remainder ? (size_t)globalBlockIdx : remainder);
  const size_t chunkSize = eltsPerBlock + (globalBlockIdx < remainder ? 1 : 0);

  // Process this block's assigned chunk
  if (chunkSize > 0) {
    // Uses default UNROLL factor for optimal performance / trade-off with register pressure.
    ncclLsaReduceSumCopy<T, ncclCoopCta, size_t>(
      coop,
      sendwin, sendoffset + blockStart * sizeof(T),
      recvwin, recvoffset + blockStart * sizeof(T),
      chunkSize,
      devComm
    );
  }

  bar.sync(coop, cuda::memory_order_release);
}


/*
 * Kernel 3: allReduceMultimemKernel - Multi-memory Hardware-Accelerated AllReduce (v2.29.1 style)
 *
 * Purpose: High-performance AllReduce implementation using multi-memory primitives
 * that leverage hardware acceleration for memory operations, significantly reducing
 * SM utilization while maintaining high bandwidth within CUDA P2P connectivity.
 *
 * Solution: Replaces the O(Nrank) peer loop approach with hardware-accelerated
 * multi-memory operations. The kernel initiates CUDA P2P reductions directly through
 * hardware, eliminating the need for explicit peer-to-peer communication loops.
 *
 * Key Optimizations:
 * - Multi-memory primitives for hardware-accelerated operations
 * - Eliminates O(Nrank) scaling by using hardware reduction capabilities
 * - Hardware-assisted memory synchronization and reduction
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - Requires CUDA P2P connectivity and
 * multi-memory support. Hardware acceleration is only available within the
 * same CUDA P2P connectivity where multi-memory operations can be performed.
 *
 * Use Case: Large CUDA P2P connectivity where scaling to more ranks is desired.
 *
 * Hardware Requirements: Hopper+ architecture with multi-memory support enabled.
 */
template <typename T>
__global__ void allReduceMultimemKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, true };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = devComm.rank, nRanks = devComm.nRanks;

  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  T* send_ptr = reinterpret_cast<T*>(ncclGetLsaMultimemPointer(sendwin, sendoffset, devComm));
  T* recv_ptr = reinterpret_cast<T*>(ncclGetLsaMultimemPointer(recvwin, recvoffset, devComm));
  for (size_t offset=globalTid; offset < count; offset += globalNthreads) {
    T v = multimemLoadSum<T,T>(send_ptr + offset);
    multimemStore<T,T>(recv_ptr + offset, v);
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

/*
 * Kernel 4: allReduceMultimemReduceCopyKernel - Multimem-based AllReduce using ReduceCopy API
 *
 * Purpose: High-performance AllReduce implementation combining multi-memory hardware
 * acceleration with the ReduceCopy convenience API for automatic optimization of
 * vectorization, loop unrolling, and alignment handling.
 *
 * Solution: Uses ncclMultimemReduceSumCopy convenience function which combines
 * the hardware-accelerated multimem operations (like kernels 3-4) with the automatic
 * optimization features of the ReduceCopy API (like kernel 2). This provides both
 * hardware acceleration and simplified implementation.
 *
 * Key Optimizations (handled by ReduceCopy API + Multimem):
 * - Multi-memory primitives for hardware-accelerated operations
 * - Eliminates O(Nrank) scaling by using hardware reduction capabilities
 * - Vector load/store instructions for maximum memory bandwidth (128-bit operations)
 * - Aggressive loop unrolling for improved instruction-level parallelism
 * - Warp-coalesced memory access patterns for optimal memory controller utilization
 * - Graceful handling of misaligned data with scalar fallback
 * - Hardware-assisted memory synchronization and reduction
 *
 * CUDA P2P Connectivity Requirement: CRITICAL - Requires CUDA P2P connectivity and
 * multi-memory support. Hardware acceleration is only available within the same
 * CUDA P2P connectivity where multi-memory operations can be performed.
 *
 * Hardware Requirements: Hopper+ architecture with multi-memory support enabled.
 *
 * Use Case: Large messages where both hardware acceleration and maximum memory
 * bandwidth are critical. Combines the best of kernels 3 (multimem) and 2
 * (ReduceCopy API) for optimal performance with cleaner code.
 */
template <typename T>
__global__ void allReduceMultimemReduceCopyKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  // Block divisibility requirement: align to 16 bytes when possible
  constexpr int BLOCK_DIVISIBILITY = (16 % sizeof(T) == 0) ? (16 / sizeof(T)) : 1;

  // Get multimem handle from devComm
  ncclMultimemHandle multimemHandle = devComm.lsaMultimem;

  // Create CTA-level cooperative group
  ncclCoopCta coop;

  // Create barrier session with multimem flag
  ncclLsaBarrierSession<ncclCoopCta> bar { coop, devComm, ncclTeamTagLsa(), blockIdx.x, true };
  bar.sync(coop, cuda::memory_order_relaxed);

  // Calculate work distribution: each block gets a fixed chunk of work (same as kernel 2)
  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const int globalBlockIdx = rank + blockIdx.x * nRanks;
  const int globalNumBlocks = gridDim.x * nRanks;

  // Calculate how much work each block should do
  // Ensure eltsPerBlock is divisible by BLOCK_DIVISIBILITY
  const size_t totalDivisibleElts = (count / BLOCK_DIVISIBILITY) * BLOCK_DIVISIBILITY;
  const size_t eltsPerBlock = (totalDivisibleElts / globalNumBlocks / BLOCK_DIVISIBILITY) * BLOCK_DIVISIBILITY;
  const size_t remainder = count - eltsPerBlock * globalNumBlocks;

  // Calculate this block's starting offset and chunk size
  // All blocks except the last get eltsPerBlock elts (divisible by BLOCK_DIVISIBILITY)
  // The last block gets eltsPerBlock + remainder
  const size_t blockStart = globalBlockIdx * eltsPerBlock;
  const size_t chunkSize = (globalBlockIdx == globalNumBlocks - 1) ? eltsPerBlock + remainder : eltsPerBlock;

  // Process this block's assigned chunk (no loop needed, same as kernel 2)
  if (chunkSize > 0) {
    // Calculate offsets for this block's chunk
    size_t srcOffset = sendoffset + blockStart * sizeof(T);
    size_t dstOffset = recvoffset + blockStart * sizeof(T);

    // Call multimem ReduceCopy API to process this chunk
    // Uses the window API that takes windows and offsets directly
    ncclMultimemReduceSumCopy<T, ncclCoopCta, size_t>(
      coop,
      sendwin, srcOffset, multimemHandle,
      recvwin, dstOffset, multimemHandle,
      chunkSize
    );
  }

  bar.sync(coop, cuda::memory_order_release);
}

#endif

testResult_t AllReduceRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {

  char* sptr = (char*)sendbuff + sendoffset;
  char* rptr = (char*)recvbuff + recvoffset;

  switch (deviceImpl) {
  case 0:
    NCCLCHECK_COMM_WAIT(ncclAllReduce(sptr, rptr, count, type, op, comm, stream), comm);
    return testSuccess;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  case 1:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_FLOAT_DOUBLE(allReduceLsaKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
  case 2:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(allReduceLsaReduceCopyKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    return testSuccess;
  case 3:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_FLOAT_DOUBLE(allReduceMultimemKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, true));
    return testSuccess;
  case 4:
    TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL_MULTIMEM(allReduceMultimemReduceCopyKernel, type, op),
               sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream, true));
    return testSuccess;
#endif
  }

  return testNotImplemented;
}

struct testColl allReduceTest = {
  "AllReduce",
  AllReduceGetCollByteCount,
  /*initConfig=*/NULL,
  AllReduceInitData,
  AllReduceGetBw,
  AllReduceRunColl
};

void AllReduceGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllReduceGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AllReduceRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &allReduceTest;
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
    op_count = 1;
    run_ops = &op;
    run_opnames = &opName;
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
  /* .getBuffSize = */ AllReduceGetBuffSize,
  /* .runTest = */ AllReduceRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  /* .getDevCommRequirements = */ AllReduceGetDevCommRequirements
#endif
};
