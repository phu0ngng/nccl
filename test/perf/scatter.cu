/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void ScatterGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *recvcount = (count/nranks) & -(16/eltSize);
  *sendcount = (*recvcount)*nranks;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = *recvcount;
  *paramcount = *recvcount;
}

testResult_t ScatterInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount;
  size_t recvcount;
  int rank;
  void* data;

  for (int id = 0; id < args->commNum; ++id) {
    for (int i = 0; i < args->nGpus; i++) {
      CUDACHECK(cudaSetDevice(args->gpus[i]));
      sendcount = args->sendBytes[id][i] / wordSize(type);
      recvcount = args->expectedBytes[id][i] / wordSize(type);
      NCCLCHECK(ncclCommUserRank(args->comms[id][i], &rank));
      CUDACHECK(cudaMemset(args->recvbuffs[id][i], 0, args->expectedBytes[id][i]));
      data = in_place ? args->recvbuffs[id][i] : args->sendbuffs[id][i];
      if (rank == root) TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, rep, 1, 0));
      TESTCHECK(InitData(args->expected[id][i], recvcount, rank * recvcount, type, ncclSum, rep, 1, 0));
      CUDACHECK(cudaDeviceSynchronize());
    }
  }

  return testSuccess;
}

void ScatterGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

/*
 * Scatter implementation using RMA host put APIs
 */
testResult_t ScatterRmaPut(void* sendWindow, size_t sendoffset, void* recvWindow, size_t recvoffset,
                           size_t count, ncclDataType_t type, int root, ncclComm_t comm, cudaStream_t stream) {
  int rank, nranks;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  NCCLCHECK(ncclCommCount(comm, &nranks));

  ncclWindow_t sendWin = (ncclWindow_t)sendWindow;
  ncclWindow_t recvWin = (ncclWindow_t)recvWindow;

  void* sendBasePtr = NULL;
  void* recvBasePtr = NULL;
  NCCLCHECK(ncclWinGetUserPtr(comm, sendWin, &sendBasePtr));
  NCCLCHECK(ncclWinGetUserPtr(comm, recvWin, &recvBasePtr));
  void* sendPtr = (char*)sendBasePtr + sendoffset;
  void* recvPtr = (char*)recvBasePtr + recvoffset;

  size_t eltSize = wordSize(type);
  size_t chunkBytes = count * eltSize;
  int ctx = 0;

  NCCLCHECK(ncclGroupStart());

  // Check if this is an in-place scatter operation
  bool isInPlace = (recvPtr == (void*)((char*)sendPtr + rank * chunkBytes));

  if (rank == root) {
    // Root puts different chunks to each rank
    for (int peer = 0; peer < nranks; peer++) {
      if (peer == rank && isInPlace) {
        continue;
      }

      size_t srcOffset = peer * chunkBytes;
      size_t dstOffset = isInPlace ? (recvoffset + (peer - rank) * chunkBytes) : recvoffset;
      NCCLCHECK(ncclPutSignal((char*)sendPtr + srcOffset, count, type, peer,
                        recvWin, dstOffset, NCCL_SIGNAL, ctx, comm, stream));
    }
  }

  // All ranks wait for signal from root (except root itself if in-place)
  if (rank != root || !isInPlace) {
    int nsignals = 1;
    NCCLCHECK(ncclWaitSignal(1, &root, &nsignals, NCCL_SIGNAL, ctx, comm, stream));
  }

  NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);

  return testSuccess;
}

testResult_t ScatterRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int implementation) {
  if (implementation == 0) {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    size_t rankOffset = count * wordSize(type);
    if (count == 0) return testSuccess;

    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    NCCLCHECK_COMM_WAIT(ncclScatter(sptr, rptr, count, type, root, comm, stream), comm);
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,7,0)
    NCCLCHECK(ncclGroupStart());
    if (rank == root) {
      for (int r=0; r<nRanks; r++) {
        NCCLCHECK(ncclSend(sptr + r * rankOffset, count, type, r, comm, stream));
      }
    }
    NCCLCHECK(ncclRecv(rptr, count, type, root, comm, stream));
    NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);
#else
    printf("NCCL 2.7 or later is needed for scatter. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#endif
  } else if (implementation == HOST_RMA_IMPL) {
    // RMA host put implementation
    TESTCHECK(ScatterRmaPut(sendbuff, sendoffset, recvbuff, recvoffset, count, type, root, comm, stream));
  } else {
    return testNotImplemented;
  }
  return testSuccess;
}

struct testColl scatterTest = {
  "Scatter",
  ScatterGetCollByteCount,
  ScatterInitData,
  ScatterGetBw,
  ScatterRunColl
};

void ScatterGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ScatterGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t ScatterRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &scatterTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;
  int begin_root, end_root;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if (root != -1) {
    begin_root = end_root = root;
  } else {
    begin_root = 0;
    end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) {
    for (int j=begin_root; j<=end_root; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", j));
    }
  }
  return testSuccess;
}

struct testEngine scatterEngine = {
  .getBuffSize = ScatterGetBuffSize,
  .runTest = ScatterRunTest
};

#pragma weak ncclTestEngine=scatterEngine
