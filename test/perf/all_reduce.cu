/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
#include "nccl_device.h"
#endif

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

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
// Device implementation #1
template <typename T>
__global__ void allReduceDeterministicKernel(ncclWindow_t sendwin, size_t sendoffset, ncclWindow_t recvwin, size_t recvoffset, size_t count, int root, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = devComm.rank, nRanks = devComm.nRanks;

  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    T v = 0;
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
#endif

testResult_t AllReduceRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
    NCCLCHECK_COMM_WAIT(ncclAllReduce(sptr, rptr, count, type, op, comm, stream), comm);
  } else {
    if (deviceImpl == 1) {
      TESTCHECK(testLaunchDeviceKernel(SPECIALIZE_KERNEL(allReduceDeterministicKernel, type, op), sendbuff, sendoffset, recvbuff, recvoffset, count, type, op, root, comm, stream));
    } else {
      return testNotImplemented;
    }
  }
  return testSuccess;
}

struct testColl allReduceTest = {
  "AllReduce",
  AllReduceGetCollByteCount,
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

struct testEngine allReduceEngine = {
  AllReduceGetBuffSize,
  AllReduceRunTest
};

#pragma weak ncclTestEngine=allReduceEngine
