/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

//InitRecvResult is not ready yet for that, so the test will report FAILED if checks are enabled.
//#define TRIANGULAR

void ReduceScattervGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
    *recvcount = (count/nranks) & -(16/eltSize);
    *sendcount = (*recvcount)*nranks;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = count/nranks;
    *paramcount = *recvcount;
}

testResult_t ReduceScattervInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
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

void ReduceScattervGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * (nranks - 1)) / 1.0E9 / sec;
#ifdef TRIANGULAR
  const double halfSize = (((double)nranks*nranks+1)/2) / (nranks*nranks);
  baseBw *= halfSize;
#endif

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

testResult_t ncclReduceScatterv(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  int nranks, rank;
  NCCLCHECK(ncclCommCount(comm, &nranks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nranks; i++) {
#ifdef TRIANGULAR
    size_t rankCount = (count / nranks) * (i+1);
#else
    size_t rankCount = count;
#endif
    void* sendbuffOffset = ((char*)sendbuff)+i*count*wordSize(type);
    NCCLCHECK(ncclReduce(sendbuffOffset, recvbuff, rankCount, type, op, i, comm, stream));
  }

  NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);
  return testSuccess;
}

testResult_t ReduceScattervRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    TESTCHECK(ncclReduceScatterv(sptr, rptr, count, type, op, comm, stream));
  } else {
    return testNotImplemented;
  }
  return testSuccess;
}

struct testColl reduceScattervTest = {
  "ReduceScatterv",
  ReduceScattervGetCollByteCount,
  ReduceScattervInitData,
  ReduceScattervGetBw,
  ReduceScattervRunColl
};

void ReduceScattervGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ReduceScattervGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t ReduceScattervRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &reduceScattervTest;
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
    for (int j=0; j<op_count; j++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1));
    }
  }
  return testSuccess;
}

struct testEngine reduceScattervEngine = {
  .getBuffSize = ReduceScattervGetBuffSize,
  .runTest = ReduceScattervRunTest
};

#pragma weak ncclTestEngine=reduceScattervEngine
