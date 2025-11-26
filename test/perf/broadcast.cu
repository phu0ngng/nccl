/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void BroadcastGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t BroadcastInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
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
      TESTCHECK(InitData(args->expected[id][i], recvcount, 0, type, ncclSum, rep, 1, 0));
      CUDACHECK(cudaDeviceSynchronize());
    }
  }

  return testSuccess;
}

void BroadcastGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

/*
 * Broadcast implementation using RMA host put APIs
 */
testResult_t BroadcastRmaPut(void* sendWindow, size_t sendoffset, void* recvWindow, size_t recvoffset,
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

  int ctx = 0;

  NCCLCHECK(ncclGroupStart());

  if (rank == root) {
    // Root puts data to all ranks (including itself)
    for (int peer = 0; peer < nranks; peer++) {
      NCCLCHECK(ncclPutSignal((char*)sendPtr + sendoffset, count, type, peer,
                        recvWin, recvoffset, 0, ctx, comm, stream));
    }
  }

  // All ranks wait for signal from root
  int nsignals = 1;
  NCCLCHECK(ncclWaitSignal(1, &root, &nsignals, 0, ctx, comm, stream));

  NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);

  return testSuccess;
}

testResult_t BroadcastRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int implementation) {
  if (implementation == 0) {
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));

    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_MAJOR >= 2 && NCCL_MINOR >= 2
    NCCLCHECK_COMM_WAIT(ncclBroadcast(sptr, rptr, count, type, root, comm, stream), comm);
#else
    if (rank == root) {
      NCCLCHECK_COMM_WAIT(ncclBcast(sptr, count, type, root, comm, stream), comm);
    } else {
      NCCLCHECK_COMM_WAIT(ncclBcast(rptr, count, type, root, comm, stream), comm);
    }
#endif
  } else if (implementation == HOST_RMA_IMPL) {
    // RMA host put implementation
    TESTCHECK(BroadcastRmaPut(sendbuff, sendoffset, recvbuff, recvoffset, count, type, root, comm, stream));
  } else {
    return testNotImplemented;
  }
  return testSuccess;
}

struct testColl broadcastTest = {
  "Broadcast",
  BroadcastGetCollByteCount,
  BroadcastInitData,
  BroadcastGetBw,
  BroadcastRunColl
};

void BroadcastGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  BroadcastGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t BroadcastRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &broadcastTest;
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

struct testEngine broadcastEngine = {
  .getBuffSize = BroadcastGetBuffSize,
  .runTest = BroadcastRunTest
};

#pragma weak ncclTestEngine=broadcastEngine
