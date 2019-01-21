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
#include "nccl_net.h"

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
int ncclCollNetMpiInit(ncclDebugLogger_t logFunction);
int ncclCollNetMpiDevices(int* ndev);
int ncclCollNetMpiPciPath(int dev, char** path);
int ncclCollNetMpiPtrSupport(int dev, int* supportedTypes);
int ncclCollNetMpiListen(int dev, void* handle, void** listenComm);
int ncclCollNetMpiConnect(void* handles[], int nranks, void* listenComm, void** collComm);
int ncclCollNetMpiReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported);
int ncclCollNetMpiIallreduce(void* collComm, void* sendData, void* recvData, int count, ncclDataType_t dataType, ncclRedOp_t redOp, int type, void** request);
int ncclCollNetMpiFlush(void* recvComm, void* data, int size);
int ncclCollNetMpiTest(void* request, int* done, int* size);
int ncclCollNetMpiClose(void* comm);

// MPI Net Module
ncclCollNet_t NCCL_COLLNET_PLUGIN_SYMBOL = {
  "MPI",
  ncclCollNetMpiInit,
  ncclCollNetMpiDevices,
  ncclCollNetMpiPciPath,
  ncclCollNetMpiPtrSupport,
  ncclCollNetMpiListen,
  ncclCollNetMpiConnect,
  ncclCollNetMpiReduceSupport,
  ncclCollNetMpiIallreduce,
  ncclCollNetMpiFlush,
  ncclCollNetMpiTest,
  ncclCollNetMpiClose,
  ncclCollNetMpiClose
};

static MPI_Comm ncclCollNetMpiComm;

void ncclCollNetMpiHook(MPI_Comm comm) {
  ncclCollNetMpiComm = comm;
  collNet = &NCCL_COLLNET_PLUGIN_SYMBOL;
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
#define OFFSET_FIFO_SIZE (1<<10)
size_t offsetFifo[OFFSET_FIFO_SIZE];
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
  int root;
  int commId;
};

struct ncclCollNetMpiListenComm {
  /* no rank, we listen to ANY_SOURCE */
  int rank;
  int nranks;
};

struct ncclCollNetMpiRecvComm {
  int root;
  int rank;
  int nranks;
};

struct ncclCollNetMpiSendComm {
  int root;
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

static __inline__ int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return -1;
  }
}

static int getCudaSupport() {
  static int cudaSupport = -1;
  if (cudaSupport == -1) {
    char* str = getenv("NCCL_MPI_CUDA_SUPPORT");
    cudaSupport = str ? atoi(str) : 0;
  }
  return cudaSupport;
}

int ncclCollNetMpiInit(ncclDebugLogger_t logFunction) {
  printf("ncclCollNetMpiInit not implemented\n");
  return 0;
}

int ncclCollNetMpiDevices(int* ndev) {
  *ndev = 1;
  return 0;
}

int ncclCollNetMpiPciPath(int dev, char** path) {
  printf("ncclCollNetMpiPciPath not implemented\n");
  return 0;
}

int ncclCollNetMpiPtrSupport(int dev, int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  if (getCudaSupport()) *supportedTypes |= NCCL_PTR_CUDA;
  return 0;
}

int ncclCollNetMpiReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported) {
  *supported = 1;
  return 0;
}

int ncclCollNetMpiListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclCollNetMpiListenComm* comm = (struct ncclCollNetMpiListenComm*)malloc(sizeof(struct ncclCollNetMpiListenComm));
  struct ncclCollNetMpiHandle* handle = (struct ncclCollNetMpiHandle*) opaqueHandle;
  assert(sizeof(struct ncclCollNetMpiHandle) < NCCL_NET_HANDLE_MAXSIZE);
  //int tag;
  //getTag(&tag);
  //comm->tag = handle->tag = tag;
  int ret;
  MPI_PROTECT(ret, MPI_Comm_rank(ncclCollNetMpiComm, &handle->rank));
  MPI_PROTECT(ret, MPI_Comm_size(ncclCollNetMpiComm, &handle->nranks));
  comm->rank = handle->rank;
  comm->nranks = handle->nranks;
  handle->root = handle->rank; //assume I am the root
  handle->commId = 0xdeadbeef + handle->rank;
  printf("Create listen rank %d\n", handle->rank);
  *listenComm = comm;
  return ret;
}

// rank of root
static int root = 0;

int ncclCollNetMpiConnect(void* opaqueHandles[], int nranks, void* listenComm, void** collComm) {
  struct ncclCollNetMpiSendComm* comm = (struct ncclCollNetMpiSendComm*)malloc(sizeof(struct ncclCollNetMpiSendComm));
  struct ncclCollNetMpiHandle* handle = (struct ncclCollNetMpiHandle*)(opaqueHandles[0]); // take 0 as root
  int err;
  //int myTmpTag;
  //getTag(&myTmpTag);
  MPI_Request request;
  printf("Connect to root %d commId %x\n", handle->root, handle->commId);
  MPI_PROTECT(err, MPI_Isend(&handle->commId, sizeof(handle->commId), MPI_BYTE, handle->root, 0, ncclCollNetMpiComm, &request));
  int done = 0;
  while (done == 0) MPI_PROTECT(err, MPI_Test(&request, &done, MPI_STATUSES_IGNORE));
  comm->root = root = handle->root;
  comm->nranks = handle->nranks;
  // init offset fifo
  offsetFifo[0] = 0;
  for (int i = 1; i < OFFSET_FIFO_SIZE; i++) {
    offsetFifo[i] = -1;
  }
  *collComm = comm;
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

int ncclCollNetMpiIallreduce(void* collComm, void* sendData, void* recvData, int count, ncclDataType_t dataType, ncclRedOp_t redOp, int type, void** request) {
  //printf("ncclCollNetMpiIallreduce\n");
  int ret;
  //CHECK_PTR(type);
  struct ncclCollNetMpiSendComm* comm = (struct ncclCollNetMpiSendComm*)collComm;
  MPI_Request* mpiRequest = ncclCollNetMpiGetRequest();
  *request = mpiRequest;
  MPI_PROTECT(ret, MPI_Iallreduce(sendData, recvData, count*ncclTypeSize(dataType), MPI_BYTE, MPI_SUM/*TODO*/, ncclCollNetMpiComm, mpiRequest));
  return ret;
}

int ncclCollNetMpiIsend(void* sendComm, void* data, void* dst, int size, int type, void** request) {
  //printf("ncclCollNetMpiIsend\n");
  int ret;
  //CHECK_PTR(type);
  struct ncclCollNetMpiSendComm* comm = (struct ncclCollNetMpiSendComm*)sendComm;
  MPI_Request* mpiRequest = ncclCollNetMpiGetRequest();
  *request = mpiRequest;
  //printf("Send : %p %d %d %d %p\n", data, size, comm->rank, comm->tag, mpiRequest);
  MPI_PROTECT(ret, MPI_Ireduce(data, dst, size, MPI_BYTE, MPI_SUM/*TODO*/, comm->root, ncclCollNetMpiComm, mpiRequest));
  return ret;
}

#define BLOCK_RECV

int ncclCollNetMpiIrecv(void* recvComm, void* data, int size, int type, void** request) {
  //printf("ncclCollNetMpiIrecv\n");
  int ret = 0;
  //CHECK_PTR(type);
  struct ncclCollNetMpiRecvComm* comm = (struct ncclCollNetMpiRecvComm*)recvComm;
#if defined(BLOCK_RECV)
  MPI_PROTECT(ret, MPI_Bcast(data, size, MPI_BYTE, comm->root, ncclCollNetMpiComm));
  *request = 0xdeadbeef;
#else
  MPI_Request* mpiRequest = ncclCollNetMpiGetRequest();
  *request = mpiRequest;
  MPI_PROTECT(ret, MPI_Ibcast(data, size, MPI_BYTE, comm->root, ncclCollNetMpiComm, mpiRequest));
#endif
  //printf("MPI bcast : %p %d %p %p\n", data, size, comm, *request);
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
