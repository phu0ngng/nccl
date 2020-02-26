/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void AlltoAllGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = (count/nranks)*nranks;
  *recvcount = (count/nranks)*nranks;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t AlltoAllInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    char* str = getenv("NCCL_TESTS_DEVICE");
    int gpuid = str ? atoi(str) : args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    CUDACHECK(cudaSetDevice(gpuid));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    for (int j=0; j<nranks; j++) {
      TESTCHECK(InitData(((char*)data)+args->sendBytes/nranks*j, sendcount/nranks, type, rep, rank));
      TESTCHECK(InitData(((char*)args->expected[i])+args->sendBytes/nranks*j, sendcount/nranks, type, rep, in_place?rank:j));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  return testSuccess;
}

void AlltoAllGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks))/((double)(nranks-1));
  *busBw = baseBw * factor;
}

testResult_t AlltoAllRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  if(sendbuff==recvbuff) return testSuccess;
  int nRanks,typebytes=wordSize(type);
  NCCLCHECK(ncclCommCount(comm,&nRanks));
  size_t chunk = count / nRanks;
  if(chunk==0) return testSuccess;
  NCCLCHECK(ncclGroupStart());
  for(int nnn=0;nnn<nRanks;nnn++) {
    NCCLCHECK(ncclSend((const void*)((char*)sendbuff+chunk*nnn*typebytes), chunk, type,nnn,comm, stream));
    NCCLCHECK(ncclRecv((void*)((char*)recvbuff+typebytes*chunk*nnn), chunk, type, nnn,comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  return testSuccess;
}

struct testColl alltoAllTest = {
  "AlltoAll",
  AlltoAllGetCollByteCount,
  AlltoAllInitData,
  AlltoAllGetBw,
  AlltoAllRunColl
};

void AlltoAllGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
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
    type_count = ncclNumTypes;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "", -1));
  }
  return testSuccess;
}

struct testEngine alltoAllEngine = {
  AlltoAllGetBuffSize,
  AlltoAllRunTest
};

#pragma weak ncclTestEngine=alltoAllEngine
