/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

//InitRecvResult is not ready yet for that, so the test will report FAILED if checks are enabled.
//#define TRIANGULAR

void getCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t *procSharedCount, int *sameExpected, size_t count, int nranks) {
    *sendcount = (count/nranks)*nranks;
    *recvcount = count/nranks;
    *sameExpected = 0;
    *procSharedCount = *sendcount;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = count/nranks;
    *paramcount = *recvcount;
}

void InitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t recvbytes = args->expectedBytes;
  size_t recvcount = args->expectedBytes / wordSize(type);
  size_t sendbytes = args->sendBytes;
  size_t sendcount = args->sendBytes / wordSize(type);

  while (args->sync[args->sync_idx] != args->thread) pthread_yield();

  for (int i=0; i<args->nGpus; i++) {
    int device;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];

    if (is_first && i == 0) {
      CUDACHECK(cudaMemcpy(args->procSharedHost, data, sendbytes, cudaMemcpyDeviceToHost));
    } else {
      Accumulate(args->procShared, data, sendcount, type, op);
    }

    CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[args->sync_idx] = args->thread + 1;

  if (args->thread+1 == args->nThreads) {
#ifdef MPI_SUPPORT
    if (sendbytes > 0) {
      // Last thread does the MPI reduction
      void* remote, *remoteHost = malloc(sendbytes);
      void* myInitialData = malloc(sendbytes);
      memcpy(myInitialData, args->procSharedHost, sendbytes);
      CUDACHECK(cudaHostRegister(remoteHost, sendbytes, 0));
      CUDACHECK(cudaHostGetDevicePointer(&remote, remoteHost, 0));

      for (int i=0; i<args->nProcs; i++) {
        if (i == args->proc) {
          MPI_Bcast(myInitialData, sendbytes, MPI_BYTE, i, MPI_COMM_WORLD);
          free(myInitialData);
        } else {
          MPI_Bcast(remoteHost, sendbytes, MPI_BYTE, i, MPI_COMM_WORLD);
          Accumulate(args->procShared, remote, sendcount, type, op);
          cudaDeviceSynchronize();
        }
      }
      CUDACHECK(cudaHostUnregister(remoteHost));
      free(remoteHost);
    }
#endif
    args->sync[args->sync_idx] = 0;
  } else {
    while (args->sync[args->sync_idx]) pthread_yield();
  }

  for (int i=0; i<args->nGpus; i++) {
      int offset = ((args->proc*args->nThreads + args->thread)*args->nGpus + i)*recvbytes;
      memcpy(args->expectedHost[i], (void *)((uintptr_t)args->procSharedHost + offset), recvbytes);
  }

  args->sync_idx = !args->sync_idx;
}

void GetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * (nranks - 1)) / 1.0E9 / sec;
#ifdef TRIANGULAR
  const double halfSize = (((double)nranks*nranks+1)/2) / (nranks*nranks);
  baseBw *= halfSize;
#endif

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

ncclResult_t ncclReduceScatterv(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
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
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}

void RunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclReduceScatterv(sendbuff, recvbuff, count, type, op, comm, stream));
}

void RunTest(struct threadArgs_t* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;

  if ((int)type != -1) { 
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = ncclNumTypes;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if ((int)op != -1) { 
    run_ops = &op;
    run_opnames = &opName;
    op_count = 1;
  } else { 
    op_count = sizeof(test_ops)/sizeof(test_ops[0]);
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  for (int i=0; i<type_count; i++) { 
      for (int j=0; j<op_count; j++) { 
          TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], -1);
      }
  }   
}
