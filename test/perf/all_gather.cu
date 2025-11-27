/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

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

  // Calculate element size and total bytes to transfer
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

  // Wait for signals from all peers to ensure all data has been written
  NCCLCHECK(ncclWaitSignal(descIdx, waitDescs, comm, stream));

  NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);

  // Free allocated memory
  free(waitDescs);

  return testSuccess;
}

testResult_t AllGatherRunColl(void* sendbuff,  size_t sendoffset,void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int implementation) {

  char* sptr = (char*)sendbuff + sendoffset;
  char* rptr = (char*)recvbuff + recvoffset;

  switch (implementation) {
  case 0:
    // NCCL built-in AllGather
    NCCLCHECK_COMM_WAIT(ncclAllGather(sptr, rptr, count, type, comm, stream), comm);
    return testSuccess;
  case HOST_RMA_IMPL:
    // RMA host put implementation
    TESTCHECK(AllGatherRmaPut(sendbuff, sendoffset, recvbuff, recvoffset, count, type, comm, stream));
    return testSuccess;
  default:
    return testNotImplemented;
  }
}

struct testColl allGatherTest = {
  "AllGather",
  AllGatherGetCollByteCount,
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
  .runTest = AllGatherRunTest
};

#pragma weak ncclTestEngine=allGatherEngine
