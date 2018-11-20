/*************************************************************************
 *  Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * 
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION, nor the names of their
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ************************************************************************/

#include "mpi.h"
#include "nccl_coll_net.h"

/************************************************************************
 * This is an example using the NCCL network API to use MPI for
 * inter-node communication.
 *
 * This file should be included as part of the application code.
 ************************************************************************/

/* Functions to be used by the application */

// ncclCollNetMpiHook : make NCCL use MPI as inter-node communication system.
// This function should be called after MPI_Init and before any NCCL call
// (in particular ncclCommGetUniqueId and ncclCommInitRank).
// If MPI is used concurrently with NCCL, it is recommended to create a 
// dedicated communicator for NCCL (usually a dup of MPI_COMM_WORLD).
void ncclCollNetMpiHook(MPI_Comm comm);

// ncclCollNetMpiLock/ncclCollNetMpiUnlock : protect MPI calls if MPI is not thread-safe.
// NCCL being an asynchronous communication library, MPI may be called from
// threads. If the MPI implementation is not THREAD_MULTIPLE, it is critical
// to guard other MPI calls in the application using those two functions.
void ncclCollNetMpiLock();
void ncclCollNetMpiUnlock();

/* NCCL MPI Plugin */

// Functions prototypes
int ncclCollNetMpiDevices(int* ndev, int** scores);
int ncclCollNetMpiPtrSupport(int dev, int* supportedTypes);
int ncclCollNetMpiListen(int dev, void* handle, void** listenComm);
int ncclCollNetMpiConnect(int dev, void* handle, void** sendComm);
int ncclCollNetMpiAccept(void *listenComm, void** recvComm);
int ncclCollNetMpiIsend(void* sendComm, void* data, int size, int type, void** request);
int ncclCollNetMpiIrecv(void* recvComm, void* data, int size, int type, void** request);
int ncclCollNetMpiFlush(void* recvComm, void* data, int size);
int ncclCollNetMpiTest(void* request, int* done, int* size);
int ncclCollNetMpiClose(void* comm);

// MPI Net Module
ncclCollNet_t ncclCollNetMpi = {
  "MPI",
  ncclCollNetMpiDevices,
  ncclCollNetMpiPtrSupport,
  ncclCollNetMpiListen,
  ncclCollNetMpiConnect,
  ncclCollNetMpiAccept,
  ncclCollNetMpiIsend,
  ncclCollNetMpiIrecv,
  ncclCollNetMpiFlush,
  ncclCollNetMpiTest,
  ncclCollNetMpiClose,
  ncclCollNetMpiClose,
  ncclCollNetMpiClose
};

static MPI_Comm ncclCollNetMpiComm;

void ncclCollNetMpiHook(MPI_Comm comm) {
  ncclCollNetMpiComm = comm;
  collNet = &ncclCollNetMpi;
}

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

pthread_mutex_t ncclCollNetMpiGlobalLock = PTHREAD_MUTEX_INITIALIZER;
static int ncclCollNetMpiLockMode = -1;

void ncclCollNetMpiLock() {
  if (ncclCollNetMpiLockMode == 1) pthread_mutex_lock(&ncclCollNetMpiGlobalLock);
}
void ncclCollNetMpiUnlock() {
  if (ncclCollNetMpiLockMode == 1) pthread_mutex_unlock(&ncclCollNetMpiGlobalLock);
}

static void ncclCollNetMpiGetLockMode() {
  int provided;
  MPI_Query_thread(&provided);
  // MPI implementations may have thread safety bugs; provide a way
  // to force locking.
  char* str = getenv("NCCL_MPI_FORCE_LOCK");
  if (provided < MPI_THREAD_MULTIPLE || (str && atoi(str) == 1)) {
    ncclCollNetMpiLockMode = 1;
  } else {
    ncclCollNetMpiLockMode = 0;
  }
}

#define MPI_PROTECT(retvar, cmd) do { \
  if (ncclCollNetMpiLockMode == -1) ncclCollNetMpiGetLockMode(); \
  ncclCollNetMpiLock(); \
  retvar = cmd; \
  ncclCollNetMpiUnlock(); \
} while (0)

/* Dynamic request pool management */
static int numRequests = 0;
MPI_Request* ncclCollNetMpiRequests = NULL;
int* ncclCollNetMpiRequestUsed = NULL;
pthread_mutex_t ncclCollNetMpiRequestsLock = PTHREAD_MUTEX_INITIALIZER;

MPI_Request* ncclCollNetMpiGetRequest() {
  pthread_mutex_lock(&ncclCollNetMpiRequestsLock);
  for (int i=0; i<numRequests; i++) {
    if (ncclCollNetMpiRequestUsed[i] == 0) {
      ncclCollNetMpiRequestUsed[i] = 1; 
      pthread_mutex_unlock(&ncclCollNetMpiRequestsLock);
      return ncclCollNetMpiRequests + i;
    }
  }
  // No free request found, grow the pool
  int newNumRequests = numRequests + 32;
  MPI_Request* newRequests = (MPI_Request*)malloc(newNumRequests*sizeof(MPI_Request));
  int* newUsed = (int*)malloc(newNumRequests*sizeof(int));
  for (int i=0; i<numRequests; i++) {
    newRequests[i] = ncclCollNetMpiRequests[i];
    newUsed[i] = ncclCollNetMpiRequestUsed[i];
  } 
  for (int i=numRequests; i<newNumRequests; i++)
    newUsed[i] = 0;
  free(ncclCollNetMpiRequests);
  ncclCollNetMpiRequests = newRequests;
  free(ncclCollNetMpiRequestUsed);
  ncclCollNetMpiRequestUsed = newUsed;
  numRequests = newNumRequests;
  pthread_mutex_unlock(&ncclCollNetMpiRequestsLock);
  return ncclCollNetMpiGetRequest();
}

void ncclCollNetMpiFreeRequest(MPI_Request* request) {
  pthread_mutex_lock(&ncclCollNetMpiRequestsLock);
  ncclCollNetMpiRequestUsed[request-ncclCollNetMpiRequests] = 0;
  pthread_mutex_unlock(&ncclCollNetMpiRequestsLock);
}

/* Opaque structures */

// We generate a tag for each handle, but our rank doesn't change.
struct ncclCollNetMpiHandle {
  int rank;
  int nranks;
  int tag;
  int commId;
};

struct ncclCollNetMpiListenComm {
  /* no rank, we listen to ANY_SOURCE */
  int tag;
  int rank;
  int nranks;
};

struct ncclCollNetMpiRecvComm {
  int remRank;
  int tag;
  int nranks;
};

struct ncclCollNetMpiSendComm {
  int remRank;
  int tag;
  int nranks;
};

// Generate a "unique" tag
static int mpiTag = 1;
static void getTag(int *tag) {
  int val = mpiTag;
  while (__sync_val_compare_and_swap(&mpiTag, val, val+1) != val) {
   val++;
  }
  *tag = val;
}

static int getCudaSupport() {
  static int cudaSupport = -1;
  if (cudaSupport == -1) {
    char* str = getenv("NCCL_MPI_CUDA_SUPPORT");
    cudaSupport = str ? atoi(str) : 0;
  }
  return cudaSupport;
}

int ncclCollNetMpiDevices(int* ndev, int** scores) {
  *ndev = 1;
  int* sc = (int*)malloc(sizeof(int));
  sc[0] = NCCL_MAX_SCORE;
  *scores = sc;
  return 0;
}

int ncclCollNetMpiPtrSupport(int dev, int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  if (getCudaSupport()) *supportedTypes |= NCCL_PTR_CUDA;
  return 0;
}

int ncclCollNetMpiListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclCollNetMpiListenComm* comm = (struct ncclCollNetMpiListenComm*)malloc(sizeof(struct ncclCollNetMpiListenComm));
  struct ncclCollNetMpiHandle* handle = (struct ncclCollNetMpiHandle*) opaqueHandle;
  assert(sizeof(struct ncclCollNetMpiHandle) < NCCL_COLL_NET_HANDLE_MAXSIZE);
  int tag;
  getTag(&tag);
  comm->tag = handle->tag = tag;
  handle->commId = 0xdeadbeef;
  int ret;
  MPI_PROTECT(ret, MPI_Comm_rank(ncclCollNetMpiComm, &handle->rank));
  MPI_PROTECT(ret, MPI_Comm_size(ncclCollNetMpiComm, &handle->nranks));
  comm->rank = handle->rank;
  comm->nranks = handle->nranks;
  printf("Create listen tag %d\n", tag);
  *listenComm = comm;
  return ret;
}

// rank of root
static int root = 0;

int ncclCollNetMpiConnect(int dev, void* opaqueHandle, void** sendComm) {
  struct ncclCollNetMpiSendComm* comm = (struct ncclCollNetMpiSendComm*)malloc(sizeof(struct ncclCollNetMpiSendComm));
  struct ncclCollNetMpiHandle* handle = (struct ncclCollNetMpiHandle*) opaqueHandle;
  int err;
  int myTmpTag;
  getTag(&myTmpTag);
  MPI_Request request;
  printf("Connect to rank %d tag %d\n", handle->rank, handle->tag);
  MPI_PROTECT(err, MPI_Isend(&myTmpTag, sizeof(myTmpTag), MPI_BYTE, handle->rank, handle->tag, ncclCollNetMpiComm, &request));
  int done = 0;
  while (done == 0) MPI_PROTECT(err, MPI_Test(&request, &done, MPI_STATUSES_IGNORE));
  comm->remRank = root = handle->rank;
  *sendComm = comm;
  return err;
}

int ncclCollNetMpiAccept(void *listenComm, void** recvComm) {
  struct ncclCollNetMpiListenComm* lComm = (struct ncclCollNetMpiListenComm*)listenComm;
  struct ncclCollNetMpiRecvComm* rComm = (struct ncclCollNetMpiRecvComm*)malloc(sizeof(struct ncclCollNetMpiRecvComm));
  int remTmpTag;
  MPI_Status status;
  int err = 0;
  MPI_Request request;
  int c = 0;
  if (lComm->rank == root) {
    printf("Waiting for any rank tag %d\n", lComm->tag);
    while (c < lComm->nranks) {
      MPI_PROTECT(err, MPI_Irecv(&remTmpTag, sizeof(remTmpTag), MPI_BYTE, MPI_ANY_SOURCE, lComm->tag, ncclCollNetMpiComm, &request));
      int done = 0;
      while (done == 0) MPI_PROTECT(err, MPI_Test(&request, &done, &status));
      int remRank = status.MPI_SOURCE;
      printf("Got connection from %d next tag %d\n", remRank, remTmpTag);
      c++;
    }
  }

  rComm->remRank = root;
  rComm->nranks = lComm->nranks;
  *recvComm = rComm;
  return err;
}

#define CHECK_PTR(type) do {          \
  if (type == NCCL_PTR_CUDA) {        \
    if (getCudaSupport() == 0)        \
      return 1;                       \
  } else if (type != NCCL_PTR_HOST) { \
    return 1;                         \
  }                                   \
} while(0)

int ncclCollNetMpiIsend(void* sendComm, void* data, int size, int type, void** request) {
  //printf("ncclCollNetMpiIsend\n");
  int ret;
  //CHECK_PTR(type);
  struct ncclCollNetMpiSendComm* comm = (struct ncclCollNetMpiSendComm*)sendComm;
  MPI_Request* mpiRequest = ncclCollNetMpiGetRequest();
  *request = mpiRequest;
  //printf("Send : %p %d %d %d %p\n", data, size, comm->rank, comm->tag, mpiRequest);
  MPI_PROTECT(ret, MPI_Isend(data, size, MPI_BYTE, comm->remRank, comm->tag, ncclCollNetMpiComm, mpiRequest));
  return ret;
}

int ncclCollNetMpiIrecv(void* recvComm, void* data, int size, int type, void** request) {
  //printf("ncclCollNetMpiIrecv\n");
  int ret;
  //CHECK_PTR(type);
  struct ncclCollNetMpiRecvComm* comm = (struct ncclCollNetMpiRecvComm*)recvComm;
  MPI_Request* mpiRequest = ncclCollNetMpiGetRequest();
  *request = mpiRequest;
  //printf("Recv : %p %d %p %p\n", data, size, comm, mpiRequest);
  //MPI_PROTECT(ret, MPI_Irecv(data, size, MPI_BYTE, 1/*MPI_ANY_SOURCE*/, 1/*comm->tag*/, ncclCollNetMpiComm, mpiRequest));
  MPI_PROTECT(ret, MPI_Irecv(data, size, MPI_BYTE, comm->remRank, comm->tag, ncclCollNetMpiComm, mpiRequest));
  return ret;
}

int ncclCollNetMpiFlush(void* recvComm, void* data, int size) {
  // not implemented
  return -1;
}

int ncclCollNetMpiTest(void* request, int* done, int* size) {
  //printf("ncclCollNetMpiTest\n");
  MPI_Request* mpiRequest = (MPI_Request*)request;
  MPI_Status status;
  int err;
  MPI_PROTECT(err, MPI_Test(mpiRequest, done, &status));
  if (err == 0 && *done == 1) {
    if (size) MPI_PROTECT(err, MPI_Get_count(&status, MPI_BYTE, size));
    ncclCollNetMpiFreeRequest(request);
  }
  return err;
}

// No need to close connections in MPI
int ncclCollNetMpiClose(void* comm) {
  if (comm) {
    free(comm);
  }
  return 0;
}
