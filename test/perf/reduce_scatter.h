/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#include "nccl_coll.h"

void ReduceScatterGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t *procSharedCount, int *sameExpected, size_t count, int nranks) {
  *sendcount = (count/nranks)*nranks;
  *recvcount = count/nranks;
  *sameExpected = 0;
  *procSharedCount = *sendcount;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = count/nranks;
  *paramcount = *recvcount;
}

void ReduceScatterInitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
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
      static void* remote = NULL;
      if (remote == NULL)
        CUDACHECK(cudaHostAlloc(&remote, args->maxbytes, cudaHostAllocPortable | cudaHostAllocMapped));
      void* myInitialData = malloc(sendbytes);
      memcpy(myInitialData, args->procSharedHost, sendbytes);
      for (int i=0; i<args->nProcs; i++) {
        if (i == args->proc) {
          MPI_Bcast(myInitialData, sendbytes, MPI_BYTE, i, MPI_COMM_WORLD);
          free(myInitialData);
        } else {
          MPI_Bcast(remote, sendbytes, MPI_BYTE, i, MPI_COMM_WORLD);
          Accumulate(args->procShared, remote, sendcount, type, op);
          cudaDeviceSynchronize();
        }
      }
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

void ReduceScatterGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * (nranks - 1)) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

void ReduceScatterRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclReduceScatter(sendbuff, recvbuff, count, type, op, comm, stream));
}

struct ncclColl_t reduceScatter = {
  "ReduceScatter",
  ReduceScatterGetCollByteCount,
  ReduceScatterInitRecvResult,
  ReduceScatterGetBw,
  ReduceScatterRunColl,
};
