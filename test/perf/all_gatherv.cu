/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

//InitRecvResult is not ready yet for that, so the test will report FAILED if checks are enabled.
//#define TRIANGULAR

void AllGathervGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
    *sendcount = (count/nranks) & -(16/eltSize);
    *recvcount = (*sendcount)*nranks;
    *sendInplaceOffset = count/nranks;
    *recvInplaceOffset = 0;
    *paramcount = *sendcount;
}

testResult_t AllGathervInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
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
        TESTCHECK(InitData(((char*)args->expected[id][i]) + args->sendBytes[id][i] * j, sendcount, 0, type, ncclSum, 33 * rep + j, 1, 0));
      }
      CUDACHECK(cudaDeviceSynchronize());
    }
  }

  return testSuccess;
}

void AllGathervGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * (nranks - 1)) / 1.0E9 / sec;
#ifdef TRIANGULAR
  const double halfSize = (((double)nranks*nranks+1)/2) / (nranks*nranks);
  baseBw *= halfSize;
#endif

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

testResult_t ncclAllGatherv(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclComm_t comm, cudaStream_t stream) {
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
    void* recvbuffOffset = ((char*)recvbuff)+i*count*wordSize(type);

#if NCCL_MAJOR >= 2 && NCCL_MINOR >= 2
    NCCLCHECK(ncclBroadcast(sendbuff, recvbuffOffset, rankCount, type, i, comm, stream));
#else
    if (i == rank) {
      if (sendbuff != recvbuffOffset) CUDACHECK(cudaMemcpyAsync(recvbuffOffset, sendbuff, rankCount*wordSize(type), cudaMemcpyDeviceToDevice, stream));
      NCCLCHECK(ncclBcast(sendbuff, rankCount, type, i, comm, stream));
    } else {
      NCCLCHECK(ncclBcast(recvbuffOffset, rankCount, type, i, comm, stream));
    }
#endif
  }
  NCCLCHECK_COMM_WAIT(ncclGroupEnd(), comm);
  return testSuccess;
}

testResult_t AllGathervRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    TESTCHECK(ncclAllGatherv(sptr, rptr, count, type, comm, stream));
  } else {
    return testNotImplemented;
  }
  return testSuccess;
}

struct testColl allGathervTest = {
  "AllGather",
  AllGathervGetCollByteCount,
  AllGathervInitData,
  AllGathervGetBw,
  AllGathervRunColl
};

void AllGathervGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllGathervGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AllGathervRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &allGathervTest;
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
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", 0));
  }
  return testSuccess;
}

struct testEngine allGathervEngine = {
  AllGathervGetBuffSize,
  AllGathervRunTest
};

#pragma weak ncclTestEngine=allGathervEngine
