/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void SendRecvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t SendRecvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
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
      TESTCHECK(InitData(data, sendcount, rank * sendcount, type, ncclSum, rep, 1, 0));
      int peer = (rank - 1 + nranks) % nranks;
      TESTCHECK(InitData(args->expected[id][i], recvcount, peer * recvcount, type, ncclSum, rep, 1, 0));
      CUDACHECK(cudaDeviceSynchronize());
    }
  }

  // We don't support in-place sendrecv
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void SendRecvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

/*
 * SendRecv implementation using RMA host put APIs
 */
testResult_t SendRecvRmaPut(void* sendWindow, size_t sendoffset, void* recvWindow, size_t recvoffset,
                            size_t count, ncclDataType_t type, ncclComm_t comm, cudaStream_t stream) {
  int rank, nranks;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  NCCLCHECK(ncclCommCount(comm, &nranks));

  // Calculate peers in ring topology
  int recvPeer = (rank - 1 + nranks) % nranks;
  int sendPeer = (rank + 1) % nranks;

  ncclWindow_t sendWin = (ncclWindow_t)sendWindow;
  ncclWindow_t recvWin = (ncclWindow_t)recvWindow;

  void* sendPtr = NULL;
  void* recvPtr = NULL;
  NCCLCHECK(ncclWinGetUserPtr(comm, sendWin, &sendPtr));
  NCCLCHECK(ncclWinGetUserPtr(comm, recvWin, &recvPtr));

  int ctx = 0;

  // Put my data to next peer's receive buffer
  NCCLCHECK(ncclPutSignal((char*)sendPtr + sendoffset, count, type, sendPeer,
                    recvWin, recvoffset, 0, ctx, 0, comm, stream));

  // Wait for signal from previous peer
  ncclWaitSignalDesc_t waitDesc = {1, recvPeer, 0, ctx};
  NCCLCHECK(ncclWaitSignal(1, &waitDesc, comm, stream));

  return testSuccess;
}

testResult_t SendRecvRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int implementation) {
  if (implementation == 0) {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    int recvPeer = (rank-1+nRanks) % nRanks;
    int sendPeer = (rank+1) % nRanks;

    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(sptr, count, type, sendPeer, comm, stream));
    NCCLCHECK(ncclRecv(rptr, count, type, recvPeer, comm, stream));
    NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);
  } else if (implementation == HOST_RMA_IMPL) {
    // RMA host put implementation
    TESTCHECK(SendRecvRmaPut(sendbuff, sendoffset, recvbuff, recvoffset, count, type, comm, stream));
  } else {
    return testNotImplemented;
  }
  return testSuccess;
}

void collInitConfig(ncclConfig_t* config) {
  // The communication pattern communicates with 2 peers (rank + 1 and rank - 1).
  // Setting the value to 1 is possible because of the way NCCL schedules the send and recv on both peers.
  // Users assuming no knowledge of internal NCCL implementation should use a value of 2.
  config->maxP2pPeers = 1;
}

struct testColl sendRecvTest = {
  "SendRecv",
  SendRecvGetCollByteCount,
  collInitConfig,
  SendRecvInitData,
  SendRecvGetBw,
  SendRecvRunColl
};

void SendRecvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  SendRecvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t SendRecvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &sendRecvTest;
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
    for (int j=0; j<op_count; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1));
    }
  }
  return testSuccess;
}

NCCL_WEAK struct testEngine ncclTestEngine = {
  /* .getBuffSize = */ SendRecvGetBuffSize,
  /* .runTest = */ SendRecvRunTest
};
