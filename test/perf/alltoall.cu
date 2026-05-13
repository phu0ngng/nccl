/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
// ReduceCopy API (including vector utilities) now included via nccl_device.h
#endif

void AlltoAllGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *paramcount = (count/nranks) & ~(16/eltSize - 1);
  *sendcount = nranks*(*paramcount);
  *recvcount = *sendcount;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
}

testResult_t AlltoAllInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
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
      data = in_place ? args->recvbuffs[id][i] : args->sendbuffs[id][i];
      TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33 * rep + rank, 1, 0));
      for (int j = 0; j < nranks; j++) {
        size_t partcount = sendcount / nranks;
        TESTCHECK(InitData((char*)args->expected[id][i] + j * partcount * wordSize(type), partcount, rank * partcount, type, ncclSum, 33 * rep + j, 1, 0));
      }
      CUDACHECK(cudaDeviceSynchronize());
    }
  }

  // We don't support in-place alltoall
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

/*
 * AlltoAll implementation using RMA host put APIs
 */
testResult_t AlltoAllRmaPut(void* sendWindow, size_t sendoffset, void* recvWindow, size_t recvoffset,
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

  // Calculate element size and bytes per chunk
  size_t eltSize = wordSize(type);
  size_t chunkBytes = count * eltSize;

  // Use RMA context 0
  int ctx = 0;

  // Allocate array for wait signal descriptors
  ncclWaitSignalDesc_t* waitDescs = (ncclWaitSignalDesc_t*)malloc(sizeof(ncclWaitSignalDesc_t) * nranks);
  if (waitDescs == NULL) {
    return testInternalError;
  }

  for (int i = 0; i < nranks; i++) {
    waitDescs[i].opCnt = 1;
    waitDescs[i].peer = i;
    waitDescs[i].sigIdx = 0;
    waitDescs[i].ctx = ctx;
  }

  NCCLCHECK(ncclGroupStart());

  // Send each chunk to its destination peer
  for (int peer = 0; peer < nranks; peer++) {
    int targetRank = (rank + peer) % nranks;
    void* srcPtr = (char*)sendPtr + sendoffset + targetRank * chunkBytes;
    size_t dstOffset = recvoffset + rank * chunkBytes;

    NCCLCHECK(ncclPutSignal(srcPtr, count, type, targetRank,
                      recvWin, dstOffset, 0, ctx, 0, comm, stream));
  }

  NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);

  // Wait for signals from all peers to ensure all data has been written
  NCCLCHECK(ncclWaitSignal(nranks, waitDescs, comm, stream));

  // Free allocated memory
  free(waitDescs);

  return testSuccess;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,29,0)
// set devComm reqs for alltoall device kernels
testResult_t AlltoAllGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs, ncclComm_t comm, const char** testSkipReason) {
  if (!reqs || !comm) return testInternalError;

  ncclCommProperties_t commProperties = NCCL_COMM_PROPERTIES_INITIALIZER;
  if (ncclCommQueryProperties(comm, &commProperties) != ncclSuccess) {
    return testNcclError;
  }

  switch(deviceImpl) {
    case 1: // NvlAlltoAllKernel
    case 2: // NvlAlltoAllKernelOptimized
      if (commProperties.nRanks != ncclTeamLsa(comm).nRanks) {
        *testSkipReason =
            "DeviceImplementation 1 and 2 requires CUDA P2P connectivity across all ranks. Not all "
            "ranks of this communicator have P2P connectivity.\n";
        return testSkipped;
      }
      reqs->lsaBarrierCount = deviceCtaCount;
      return testSuccess;
    case 5: // GinAlltoAllKernelMultiContext
      reqs->ginContextCount = deviceCtaCount;
      // fall through
    case 3: // GinAlltoAllKernel
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
      reqs->worldGinBarrierCount = deviceCtaCount;
#endif
      // fall through
    case 4: // HybridAlltoAllKernel (LSA+GIN)
      if (commProperties.ginType == NCCL_GIN_TYPE_NONE) {
        *testSkipReason = "This test requires GIN support, but GIN support is not enabled for this communicator.\n";
        return testSkipped;
      }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
      if (deviceImpl == 4)
#endif
        reqs->barrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 3)
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      reqs->ginForceEnable = true;
#endif
      return testSuccess;
    case 6: // HierAlltoAllKernel (LSA+GIN)
      if (commProperties.railedGinType == NCCL_GIN_TYPE_NONE) {
        *testSkipReason = "This test requires Rail GIN support, but it is not enabled for this communicator.\n";
        return testSkipped;
      }
      reqs->ginContextCount = deviceCtaCount;
      reqs->barrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount*2*ncclTeamRail(comm).nRanks;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 3)
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
#else
      reqs->ginForceEnable = true;
#endif
      return testSuccess;
    default:
      return testNotImplemented;
  }
}
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
bool AlltoAllGetDevCommRequirements(int deviceImpl, ncclDevCommRequirements* reqs) {
  if (!reqs) return false;

  switch(deviceImpl) {
    case 1: // NvlAlltoAllKernel
    case 2: // NvlAlltoAllKernelOptimized
      reqs->lsaBarrierCount = deviceCtaCount;
      return true;
    case 3: // GinAlltoAllKernel
    case 4: // HybridAlltoAllKernel (LSA+GIN)
      reqs->barrierCount = deviceCtaCount;
      reqs->ginSignalCount = deviceCtaCount;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 3)
      reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
      reqs->ginForceEnable = true;
#endif
      return true;
    default:
      return false;
  }
}
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
// shared scalar AlltoAll implementation used by both kernels
template <typename T>
__device__ void AlltoAllScalarImpl(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int rank, int nRanks, int tid, int nthreads) {
  T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, rank);

  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int peer = 0; peer < nRanks; peer++) {
      T value = sendPtr[peer * count + offset];
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[rank * count + offset] = value;
    }
  }
}

// Device implementation #1 - simple NVL kernel
template <typename T>
__global__ void NvlAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// Device implementation #2 - optimized NVL kernel using vectorization and unrolling
template <typename T>
__global__ void NvlAlltoAllKernelOptimized(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  using TN = uint4; // Alltoall is type insensitive, so using generic uint4 for data transport
  constexpr int VECTOR_FACTOR = sizeof(TN) / sizeof(T);
  constexpr int UNROLL_FACTOR = 128/sizeof(TN);
  constexpr int PEER_UNROLL = 2;

  int rank = devComm.rank, nRanks = devComm.nRanks;
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int nthreads = blockDim.x * gridDim.x;

  T* sendPtr = (T*)ncclGetLsaPointer(sendwin, sendoffset, rank);

  // alignment check: can we use vectorized operations?
  bool canVectorize = (sizeof(TN) > sizeof(T)) &&  // Only if vectorization helps
                      (reinterpret_cast<uintptr_t>(sendPtr) % sizeof(TN) == 0) &&  // Base aligned
                      ((count * sizeof(T)) % sizeof(TN) == 0);  // Stride compatible

  if (canVectorize) {
    size_t vector_count = count / VECTOR_FACTOR;
    int elements_per_iteration = nthreads * UNROLL_FACTOR;

    // process aligned vectorized elements without bounds checks
    size_t aligned_vector_count = (vector_count / elements_per_iteration) * elements_per_iteration;
    for (size_t base_offset = tid; base_offset < aligned_vector_count; base_offset += elements_per_iteration) {
      // unroll a limited number of peers at a time
      for (int peerBase = 0; peerBase < nRanks; peerBase += PEER_UNROLL) {
        int peersInGroup = min(PEER_UNROLL, nRanks - peerBase);

        #pragma unroll
        for (int p = 0; p < peersInGroup; p++) {
          int peer = peerBase + p;
          TN* sendVecPtr = (TN*)(sendPtr + peer * count);
          TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
          TN values[UNROLL_FACTOR];

          // split load/store into separate loops for better overlap and ILP
          #pragma unroll
          for (int i = 0; i < UNROLL_FACTOR; i++) {
            size_t offset = base_offset + i * nthreads;
            values[i] = sendVecPtr[offset];
          }
          #pragma unroll
          for (int i = 0; i < UNROLL_FACTOR; i++) {
            size_t offset = base_offset + i * nthreads;
            recvVecPtr[offset] = values[i];
          }
        }
      }
    }

    // handle remaining vectorized elements that didn't fit in aligned chunks
    for (size_t base_offset = aligned_vector_count + tid; base_offset < vector_count; base_offset += nthreads) {
      for (int peer = 0; peer < nRanks; peer++) {
        TN* sendVecPtr = (TN*)(sendPtr + peer * count);
        TN* recvVecPtr = (TN*)((T*)ncclGetLsaPointer(recvwin, recvoffset, peer) + rank * count);
        recvVecPtr[base_offset] = sendVecPtr[base_offset];
      }
    }

    // handle any remaining elements not divisible by vectorization factor
    size_t scalar_start = vector_count * VECTOR_FACTOR;
    for (size_t offset = scalar_start + tid; offset < count; offset += nthreads) {
      for (int peer = 0; peer < nRanks; peer++) {
        T value = sendPtr[peer * count + offset];
        T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, peer);
        recvPtr[rank * count + offset] = value;
      }
    }
  } else {
    // simple scalar fallback for unaligned data (identical to simple kernel)
    AlltoAllScalarImpl<T>(sendwin, sendoffset, recvwin, recvoffset, count, rank, nRanks, tid, nthreads);
  }

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

#if !defined(NCCL_OS_WINDOWS)
template <typename T>
__global__ void GinAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = blockIdx.x;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x };
#else
  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
#endif
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  /* send to all peers via GIN */
  const size_t size = count * sizeof(T);
  for (int r=tid; r<devComm.nRanks; r+=nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + r * size,
        size, ncclGin_SignalInc{signalIndex});
  }

  /* only the CTA whose thread range covers devComm.rank waits for its signal */
  int receivingCta = (devComm.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta)
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 30, 0)
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
#endif
}

template <typename T>
__global__ void HybridAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = blockIdx.x;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);
  const int startLsa = world.rank - lsa.rank;
  const int lsaSize  = lsa.nRanks;

  /* handle remote peers (i.e., non-LSA) using GIN */
  const size_t size = count * sizeof(T);
  for (int r = tid; r < world.nRanks; r += nthreads) {
    if (r < startLsa || r >= startLsa + lsaSize) {
      gin.put(world, r,
          recvwin, recvoffset + world.rank * size,
          sendwin, sendoffset + r * size,
          size, ncclGin_SignalInc{signalIndex});
    }
  }

  /* handle local peers with LSA */
  T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);
  for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int lp = 0; lp < lsa.nRanks; lp++) {
      int wr = startLsa + lp;
      T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);
      recvPtr[world.rank * count + offset] = sendLocal[wr * count + offset];
    }
  }

  int numRemotePeers = world.nRanks - lsa.nRanks;
  int receivingCta = (world.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta)
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + numRemotePeers);
  gin.flush(ncclCoopCta());

  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
}

template <typename T>
__global__ void GinAlltoAllKernelMultiContext(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  /* determine number of contexts to use, based on CTA count and max available contexts */
  int numContexts = min(gridDim.x, devComm.ginContextCount);
  /* divide CTAs across contexts; each CTA uses its own global signal */
  int ginContext = (blockIdx.x * numContexts) / gridDim.x;
  unsigned int signalIndex = blockIdx.x;
  ncclGin gin { devComm, ginContext };
  uint64_t signalValue = gin.readSignal(signalIndex);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x };
#else
  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
#endif
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);
  int myPeerStart = (blockIdx.x * devComm.nRanks) / gridDim.x;
  int myPeerEnd = ((blockIdx.x + 1) * devComm.nRanks) / gridDim.x;
  /* each CTA sends to 1+ assigned peers; threads within CTA parallelize the work */
  /* all ranks' CTA K increment signal K on each peer they send to */
  const size_t size = count * sizeof(T);
  for (int peer = myPeerStart + threadIdx.x; peer < myPeerEnd; peer += blockDim.x) {
    gin.put(ncclTeamWorld(devComm), peer,
        recvwin, recvoffset + devComm.rank * size,
        sendwin, sendoffset + peer * size,
        size, ncclGin_SignalInc{blockIdx.x});
  }
  /* only the CTA assigned to handle this rank receives data from all peers */
  int receivingCta = ((devComm.rank + 1) * gridDim.x - 1) / devComm.nRanks;
  if (blockIdx.x == receivingCta)
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 30, 0)
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
#endif
}

// Hierarchical rail-only alltoall
// Each rank uses its receive buffer as intermediate buffer to receive from all other ranks
// on the same rail, then pipelines data to other ranks on the same LSA domain, finishing by
// its own data (which doesn't need to be copied).
// Latency is fairly high with large LSA domains because we need N consecutive rounds. To
// improve latency, we would need to use some scratch space to allow for all data to be sent
// at once.
template <typename T>
__global__ void HierAlltoAllKernel(ncclWindow_t sendWin, size_t sendoffset, ncclWindow_t recvWin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
#if __CUDA_ARCH__ >= 700 // required for ncclLocalCopy
  int ginContext = blockIdx.x % devComm.ginContextCount;
  ncclGin gin { devComm, ginContext };
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);

  ncclTeam ginTeam = ncclTeamRail(devComm);
  ncclTeam lsaTeam = ncclTeamLsa(devComm);
  unsigned int signalSendBase = blockIdx.x*ginTeam.nRanks;
  unsigned int signalRecvBase = (gridDim.x + blockIdx.x)*ginTeam.nRanks;

  // Divide work on the different CTAs
  size_t size = count * sizeof(T);
  ssize_t blockSize = (size+gridDim.x-1) / gridDim.x;
  const size_t offset = blockSize * blockIdx.x;
  if (blockSize + offset > size) blockSize = size - offset;
  if (blockSize <= 0) return;
  const size_t winRecvOffset = recvoffset + offset;
  const size_t winSendOffset = sendoffset + offset;

  if (tid < 32) {
    // First warp drives sends
    for (int li = lsaTeam.nRanks-1; li >= 0; li--) {
      int lsaRank = (lsaTeam.rank + li) % lsaTeam.nRanks;
      for (int gi = 1+tid; gi < ginTeam.nRanks; gi+=32) {
        int gSendPeer = (ginTeam.rank + gi) % ginTeam.nRanks;
        // Wait for credit
        gin.waitSignalMeetShadow(ncclCoopThread(), signalRecvBase+gSendPeer);
        // Write to peer
        int sendPeer = gSendPeer * lsaTeam.nRanks + lsaRank;
        size_t sendOffset = sendPeer * size;
        size_t recvOffset = devComm.rank * size;
        gin.put(ginTeam, gSendPeer,
            recvWin, recvOffset+winRecvOffset,
            sendWin, sendOffset+winSendOffset,
            blockSize, ncclGin_SignalInc{signalSendBase+ginTeam.rank});
        // Block next send until we receive ack
        if (li) gin.increaseSignalShadow(signalRecvBase+gSendPeer, 1);
      }
    }
  } else {
    // Other workers receive, copy through LSA, and return credits.
    ncclCoopWarpSpan workers(1, nthreads/32 - 1, 1);
    for (int li = lsaTeam.nRanks-1; li >= 0; li--) {
      int lsaRank = (lsaTeam.rank + li) % lsaTeam.nRanks;

      // First handle the local LSA copy (our rank -> local ranks, including to ourself)
      // This should cover the time to receive the first chunk.
      int nodeFirstRank = ginTeam.rank * lsaTeam.nRanks;
      size_t sendOffset = (nodeFirstRank + lsaRank) * size;
      char* source = (char*)ncclGetLocalPointer(sendWin, sendOffset+winSendOffset);
      size_t recvOffset = (nodeFirstRank + lsaTeam.rank) * size;
      char* dstPtr = (char*)ncclGetLsaPointer(recvWin, recvOffset+winRecvOffset, lsaRank);
      ncclLocalCopy(workers, source, 1, dstPtr, 0, blockSize);

      // Then receive from the network
      for (int gi = 1; gi < ginTeam.nRanks; gi++) {
        int gRecvPeer = (ginTeam.rank - gi + ginTeam.nRanks) % ginTeam.nRanks;
        // Recv from peer
        if (workers.thread_rank() == 0) {
          gin.increaseSignalShadow(signalSendBase+gRecvPeer, 1);
          gin.waitSignalMeetShadow(ncclCoopThread(), signalSendBase+gRecvPeer);
        }
        workers.sync();
        // On the last round, lsaRank should be ourself. No need to copy, nor return credits.
        if (li) {
          // Copy chunk through LSA
          size_t recvOffset = (gRecvPeer * lsaTeam.nRanks + lsaTeam.rank) * size;
          char* source = (char*)ncclGetLocalPointer(recvWin, recvOffset+winRecvOffset);
          char* dstPtr = (char*)ncclGetLsaPointer(recvWin, recvOffset+winRecvOffset, lsaRank);
          ncclLocalCopy(workers, source, 1, dstPtr, 0, blockSize);
          // Ack recv (return credit)
          if (workers.thread_rank() == 1) {
            gin.signal(ginTeam, gRecvPeer, ncclGin_SignalInc{signalRecvBase+ginTeam.rank});
          }
        }
      }
    }
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
#endif // ARCH >= 700
}
#endif // !defined(NCCL_OS_WINDOWS)
#endif

testResult_t AlltoAllRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int implementation) {
  if (implementation == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    NCCLCHECK_COMM_WAIT(ncclAlltoAll(sptr, rptr, count, type, comm, stream), comm);
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,7,0)
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = count * wordSize(type);
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(ncclSend(sptr+r*rankOffset, count, type, r, comm, stream));
      NCCLCHECK(ncclRecv(rptr+r*rankOffset, count, type, r, comm, stream));
    }
    NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);
#else
    printf("NCCL 2.7 or later is needed for alltoall. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#endif
  } else {
    switch(implementation) {
      case 1:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(NvlAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
      case 2:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(NvlAlltoAllKernelOptimized, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
#if !defined(NCCL_OS_WINDOWS)
      case 3:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(GinAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
      case 4:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(HybridAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
      case 5:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(GinAlltoAllKernelMultiContext, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
      case 6:
        TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(HierAlltoAllKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
        return testSuccess;
#endif // !defined(NCCL_OS_WINDOWS)

      case HOST_RMA_IMPL:
        // RMA host put implementation
        TESTCHECK(AlltoAllRmaPut(sendbuff, sendoffset, recvbuff, recvoffset, count, type, comm, stream));
        return testSuccess;
      default:
        return testNotImplemented;
    }
  }
  return testSuccess;
}

struct testColl alltoAllTest = {
  "AlltoAll",
  AlltoAllGetCollByteCount,
  /*initConfig=*/NULL,
  AlltoAllInitData,
  AlltoAllGetBw,
  AlltoAllRunColl
};

void AlltoAllGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AlltoAllRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &alltoAllTest;
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

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ AlltoAllGetBuffSize,
  /* .runTest = */ AlltoAllRunTest,
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
  /* .getDevCommRequirements = */ AlltoAllGetDevCommRequirements
#endif
};
