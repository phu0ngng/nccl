/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#include "nccl_coll.h"

//InitRecvResult is not ready yet for that, so the test will report FAILED if checks are enabled.
//#define TRIANGULAR

void AllGathervGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t *procSharedCount, int *sameExpected, size_t count, int nranks) {
    *sendcount = count/nranks;
    *recvcount = (count/nranks)*nranks;
    *sameExpected = 1;
    *procSharedCount = 0;
    *sendInplaceOffset = count/nranks;
    *recvInplaceOffset = 0;
    *paramcount = *sendcount;
}

void AllGathervInitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t nBytes = args->nbytes;
  size_t count = nBytes / wordSize(type);
  int proc = args->proc;
  int nThreads = args->nThreads;
  int t = args->thread;
  int nGpus = args->nGpus;

  while (args->sync[args->sync_idx] != t) pthread_yield();

  for (int i=0; i<nGpus; i++) {
    int device;
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));

    void* data = in_place ? (void *)((uintptr_t)args->recvbuffs[i] + args->sendInplaceOffset*rank) : args->sendbuffs[i];

    CUDACHECK(cudaMemcpy((void *)((uintptr_t)args->expectedHost[0] + ((proc*nThreads + t)*nGpus + i)*nBytes), 
                data, 
                nBytes, cudaMemcpyDeviceToHost));

    CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[args->sync_idx] = t + 1;

  if (t+1 == nThreads) {
#ifdef MPI_SUPPORT
    // Last thread does the MPI allgather
    MPI_Allgather(MPI_IN_PLACE, nBytes*nThreads*nGpus, MPI_BYTE, 
        args->expectedHost[0], 
        nBytes*nThreads*nGpus, MPI_BYTE, MPI_COMM_WORLD);
#endif
    args->sync[args->sync_idx] = 0;
  } else {
    while (args->sync[args->sync_idx]) pthread_yield();
  }

  args->sync_idx=!args->sync_idx;
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

ncclResult_t ncclAllGatherv(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclComm_t comm, cudaStream_t stream) {
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
    if (i == rank) {
      if (sendbuff != recvbuffOffset) CUDACHECK(cudaMemcpyAsync(recvbuffOffset, sendbuff, rankCount*wordSize(type), cudaMemcpyDeviceToDevice, stream));
      NCCLCHECK(ncclBcast(sendbuff, rankCount, type, i, comm, stream));
    } else {
      NCCLCHECK(ncclBcast(recvbuffOffset, rankCount, type, i, comm, stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}

void AllGathervRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclAllGatherv(sendbuff, recvbuff, count, type, comm, stream));
}

struct ncclColl_t allGatherv = {
  "AllGatherv",
  AllGathervGetCollByteCount,
  AllGathervInitRecvResult,
  AllGathervGetBw,
  AllGathervRunColl,
};
