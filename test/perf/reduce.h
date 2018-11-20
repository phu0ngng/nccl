/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <assert.h>
#include "cuda_runtime.h"
#include "common.h"
#include "nccl_coll.h"

void ReduceGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t *procSharedCount, int *sameExpected, size_t count, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sameExpected = 0;
  *procSharedCount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

void ReduceInitRecvResult(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t count = args->expectedBytes / wordSize(type);
  int root_gpu = root%args->nGpus;

  assert(args->expectedBytes == args->nbytes);

  while (args->sync[args->sync_idx] != args->thread) pthread_yield();

  for (int i=0; i<args->nGpus; i++) {
    int device;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];

    if (is_first && i == 0) {
      CUDACHECK(cudaMemcpy(args->procSharedHost, data, count*wordSize(type), cudaMemcpyDeviceToHost));
    } else {
      Accumulate(args->procShared, data, count, type, op);
    }

    CUDACHECK(cudaDeviceSynchronize());
  }

  args->sync[args->sync_idx] = args->thread + 1;

  if (args->thread+1 == args->nThreads) {
#ifdef MPI_SUPPORT
    int root_proc = root/(args->nThreads*args->nGpus);
    if (args->expectedBytes) {
      static void* remote = NULL;
      if (remote == NULL)
        CUDACHECK(cudaHostAlloc(&remote, args->maxbytes, cudaHostAllocPortable | cudaHostAllocMapped));
      // Last thread does the MPI reduction
      if (root_proc == args->proc) { 
        for (int i=0; i<args->nProcs; i++) {
          if (i == args->proc) continue;
          MPI_Recv(remote, args->expectedBytes, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          Accumulate(args->procShared, remote, count, type, op);
          CUDACHECK(cudaDeviceSynchronize());
        }
      } else {
        MPI_Send(args->procSharedHost, args->expectedBytes, MPI_BYTE, root_proc, 0, MPI_COMM_WORLD);
      }
    }
#endif
    args->sync[args->sync_idx] = 0;
  } else {
    while (args->sync[args->sync_idx]) pthread_yield();
  }

  //if root fill expected bytes with reduced data
  // else if in_place, leave fill it with original data, else set to zero
  for (int i=0; i<args->nGpus; i++) {
      int rank = (args->proc*args->nThreads + args->thread)*args->nGpus + i;
      if (rank == root) { 
          memcpy(args->expectedHost[root_gpu], args->procSharedHost, args->expectedBytes); 
      } else { 
         if (in_place == 1) {
              CUDACHECK(cudaMemcpy(args->expectedHost[i], args->recvbuffs[i], args->expectedBytes, cudaMemcpyDeviceToHost));
          } else {
              memset(args->expectedHost[i], 0, args->expectedBytes); 
          }
      } 
  }

  args->sync_idx = !args->sync_idx;
}

void ReduceGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;
  *algBw = baseBw;
  *busBw = baseBw;
}

void ReduceRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclReduce(sendbuff, recvbuff, count, type, op, root, comm, stream));
}

struct ncclColl_t reduce = {
  "Reduce",
  ReduceGetCollByteCount,
  ReduceInitRecvResult,
  ReduceGetBw,
  ReduceRunColl,
};
