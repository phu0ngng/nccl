/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void GatherGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = (count/nranks) & ~(16/eltSize - 1);
  *recvcount = (*sendcount)*nranks;
  *sendInplaceOffset = *sendcount;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t GatherInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
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
      TESTCHECK(InitData(data, sendcount, rank * sendcount, type, ncclSum, rep, 1, 0));
      CUDACHECK(cudaMemcpy(args->expected[id][i], args->recvbuffs[id][i], args->expectedBytes[id][i], cudaMemcpyDefault));
      if (rank == root) {
        TESTCHECK(InitData(args->expected[id][i], nranks * sendcount, 0, type, ncclSum, rep, 1, 0));
      }
      CUDACHECK(cudaDeviceSynchronize());
    }
  }

  return testSuccess;
}

void GatherGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

/*
 * Gather implementation using RMA host put APIs
 */
testResult_t GatherRmaPut(void* sendWindow, size_t sendoffset, void* recvWindow, size_t recvoffset,
                          size_t count, ncclDataType_t type, int root, ncclComm_t comm, cudaStream_t stream) {
  int rank, nranks;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  NCCLCHECK(ncclCommCount(comm, &nranks));

  ncclWindow_t sendWin = (ncclWindow_t)sendWindow;
  ncclWindow_t recvWin = (ncclWindow_t)recvWindow;

  void* sendPtr = NULL;
  void* recvPtr = NULL;
  NCCLCHECK(ncclWinGetUserPtr(comm, sendWin, &sendPtr));
  NCCLCHECK(ncclWinGetUserPtr(comm, recvWin, &recvPtr));

  size_t eltSize = wordSize(type);
  size_t chunkBytes = count * eltSize;
  int ctx = 0;

  NCCLCHECK(ncclGroupStart());

  // Each rank puts its data to root at offset [myRank*chunkBytes]
  size_t dstOffset = recvoffset + rank * chunkBytes;
  NCCLCHECK(ncclPutSignal((char*)sendPtr + sendoffset, count, type, root,
                    recvWin, dstOffset, 0, ctx, 0, comm, stream));

  NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);

  if (rank == root) {
    // Root waits for signals from all ranks
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

    NCCLCHECK(ncclWaitSignal(nranks, waitDescs, comm, stream));

    free(waitDescs);
  }

  return testSuccess;
}

testResult_t GatherRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int implementation) {
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
    NCCLCHECK_COMM_WAIT(ncclGather(sptr, rptr, count, type, root, comm, stream), comm);
#elif NCCL_VERSION_CODE >= NCCL_VERSION(2,7,0)
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(sptr, count, type, root, comm, stream));
    if (rank == root) {
      for (int r=0; r<nRanks; r++) {
        NCCLCHECK(ncclRecv(rptr + r * rankOffset, count, type, r, comm, stream));
      }
    }
    NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);
#else
    printf("NCCL 2.7 or later is needed for gather. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#endif
  } else if (implementation == HOST_RMA_IMPL) {
    // RMA host put implementation
    TESTCHECK(GatherRmaPut(sendbuff, sendoffset, recvbuff, recvoffset, count, type, root, comm, stream));
  } else {
    return testNotImplemented;
  }
  return testSuccess;
}

struct testColl gatherTest = {
  "Gather",
  GatherGetCollByteCount,
  /*initConfig=*/NULL,
  GatherInitData,
  GatherGetBw,
  GatherRunColl
};

void GatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  GatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t GatherRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &gatherTest;
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

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ GatherGetBuffSize,
  /* .runTest = */ GatherRunTest
};
