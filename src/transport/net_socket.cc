/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "param.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <poll.h>
#include <limits.h>

/* Init functions */
static char ncclNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];
static union socketAddress ncclNetIfAddrs[MAX_IFS];
static int ncclNetIfs = -1;
pthread_mutex_t ncclSocketLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t ncclSocketInit(ncclDebugLogger_t logFunction) {
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclSocketLock);
    if (ncclNetIfs == -1) {
      ncclNetIfs = findInterfaces(ncclNetIfNames, ncclNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (ncclNetIfs <= 0) {
        WARN("NET/Socket : no interface found");
        return ncclInternalError;
      } else {
        char line[1024];
        char addrline[1024];
        line[0] = '\0';
        for (int i=0; i<ncclNetIfs; i++) {
          snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%s", i, ncclNetIfNames+i*MAX_IF_NAME_SIZE,
              socketToString(&ncclNetIfAddrs[i].sa, addrline));
        }
        line[1023] = '\0';
        INFO(NCCL_INIT|NCCL_NET,"NET/Socket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&ncclSocketLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketPtrSupport(int dev, int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  return ncclSuccess;
}

ncclResult_t ncclSocketDevices(int* ndev) {
  *ndev = ncclNetIfs;
  return ncclSuccess;
}

ncclResult_t ncclSocketPciPath(int dev, char** path) {
  char devicepath[PATH_MAX];
  snprintf(devicepath, PATH_MAX, "/sys/class/net/%s/device", ncclNetIfNames+dev*MAX_IF_NAME_SIZE);
  *path = realpath(devicepath, NULL);
  if (*path == NULL) {
    INFO(NCCL_NET|NCCL_INIT, "Could not find real path of %s", devicepath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t GetSocketAddr(int dev, union socketAddress* addr) {
  if (dev >= ncclNetIfs) return ncclInternalError;
  memcpy(addr, ncclNetIfAddrs+dev, sizeof(*addr));
  return ncclSuccess;
}

/* Communication functions */

#define MAX_SOCKETS 16
#define MAX_THREADS 16
NCCL_PARAM(SocketNsocks, "NSOCKETS", 1);
NCCL_PARAM(SocketNthreads, "SOCKET_NTHREADS", 1);

struct ncclSocketHandle {
  union socketAddress connectAddr;
  uint16_t port[MAX_SOCKETS];
};

struct ncclSocketRequest {
  int op;
  void* data;
  int size;
  int fd;
  int ctrlFd;
  int offset;
  int used;
  ncclResult_t result;
};

#define MAX_REQUESTS 128

struct ncclSocketReqs {
  int next;
  struct ncclSocketRequest* requests;
};

enum threadState {start, stop};

struct ncclSocketThreadArgs {
  struct ncclSocketComm* comm;
  int threadId;
};

struct ncclSocketComm {
  int ctrlFd;
  int fd[MAX_SOCKETS];
  int nSocks;
  int nThreads;
  int nextFd;
  struct ncclSocketReqs reqs;
  pthread_t proxyThread[MAX_THREADS];
  struct ncclSocketThreadArgs args[MAX_THREADS];
  enum threadState state;
};

void* persistentSocketThread(void *args_) {
  struct ncclSocketThreadArgs* args = (struct ncclSocketThreadArgs*)args_;
  struct ncclSocketComm* comm = args->comm;
  volatile enum threadState* state = &comm->state;
  while (1) {
    int idle = 1;
    for (int i=args->threadId; i<MAX_REQUESTS; i+=comm->nThreads) {
      struct ncclSocketRequest* r = (struct ncclSocketRequest*) comm->reqs.requests+i;
      if (r != NULL && r->used == 1 && r->offset >= 0 && r->offset < r->size) {
        r->result = socketProgress(r->op, r->fd, r->data, r->size, &r->offset);
        if (r->result != ncclSuccess) {
          WARN("NET/Socket : socket progress error");
          return NULL;
        }
        idle = 0;
      }
    }
    if (*state == stop) return NULL;
    if (idle) sched_yield();
  }
}

ncclResult_t ncclSocketNewComm(struct ncclSocketComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->ctrlFd = -1;
  for (int i=0; i < MAX_SOCKETS; i++) {
    (*comm)->fd[i] = -1;
  }
  int nSocks = ncclParamSocketNsocks();
  int nThreads = ncclParamSocketNthreads();
  if (nSocks > MAX_SOCKETS) {
    WARN("NET/Socket : The number of sockets set is greater than the maximum allowed, setting to the maximum (%d)", MAX_SOCKETS);
    nSocks = MAX_SOCKETS;
  }
  if (nThreads > MAX_THREADS) {
    WARN("NET/Socket : The number of threads set is greater than the maximum allowed, setting to the maximum (%d)", MAX_THREADS);
    nThreads = MAX_THREADS;
  }
  (*comm)->nSocks = nSocks;
  (*comm)->nThreads = nThreads;
  (*comm)->nextFd = 0;
  return ncclSuccess;
}

ncclResult_t ncclSocketListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclSocketHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclSocketHandle size too large");
  struct ncclSocketComm* comm;
  NCCLCHECK(ncclSocketNewComm(&comm));
  if (dev < 0) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  NCCLCHECK(GetSocketAddr(dev, &handle->connectAddr));
  union socketAddress copy = handle->connectAddr;
  NCCLCHECK(createListenSocket(&comm->ctrlFd, &handle->connectAddr));
  for (int i=0; i<comm->nSocks; i++) {
    union socketAddress addr = copy;
    NCCLCHECK(createListenSocket(comm->fd+i, &addr));
    handle->port[i] = socketToPort(&addr.sa);
  }
  *listenComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(int dev, void* opaqueHandle, void** sendComm) {
  struct ncclSocketComm* comm;
  NCCLCHECK(ncclSocketNewComm(&comm));
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  NCCLCHECK(connectAddress(&comm->ctrlFd, &handle->connectAddr));
  for (int i=0; i<comm->nSocks; i++) {
    union socketAddress addr = handle->connectAddr;
    setSocketPort(&addr.sa, handle->port[i]);
    NCCLCHECK(connectAddress(comm->fd+i, &addr));
  }
  *sendComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclSocketAccept(void* listenComm, void** recvComm) {
  struct ncclSocketComm* lComm = (struct ncclSocketComm*)listenComm;
  struct ncclSocketComm* rComm;
  NCCLCHECK(ncclSocketNewComm(&rComm));
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(lComm->ctrlFd, (struct sockaddr*)&sockaddr, &socklen), "accept", rComm->ctrlFd);
  for (int i=0; i<rComm->nSocks; i++) {
    struct sockaddr_in sockaddr;
    socklen_t socklen = sizeof(struct sockaddr_in);
    SYSCHECKVAL(accept(lComm->fd[i], (struct sockaddr*)&sockaddr, &socklen), "accept", rComm->fd[i]);
  }
  *recvComm = rComm;
  return ncclSuccess;
}

ncclResult_t ncclSocketGetRequest(struct ncclSocketComm* comm, int op, void* data, int size, struct ncclSocketRequest** req) {
  struct ncclSocketReqs* reqs = &comm->reqs;
  if (reqs->requests == NULL) {
    NCCLCHECK(ncclCalloc(&reqs->requests, MAX_REQUESTS));
    reqs->next = 0;
    comm->state = start;
    for (int i=0; i<comm->nThreads; i++) {
      comm->args[i].comm = comm;
      comm->args[i].threadId = i;
      pthread_create(comm->proxyThread+i, NULL, persistentSocketThread, comm->args+i);
    }
  }
  struct ncclSocketRequest* r = reqs->requests+reqs->next;
  if (r->used == 0) {
    r->op = op;
    r->data = data;
    r->size = size;
    r->fd = comm->fd[comm->nextFd];
    r->ctrlFd = comm->ctrlFd;
    r->offset = -1;
    r->used = 1;
    r->result = ncclSuccess;
    comm->nextFd = (comm->nextFd + 1) % comm->nSocks;
    reqs->next = (reqs->next+1)%MAX_REQUESTS;
    *req = r;
    return ncclSuccess;
  }
  WARN("Socket : unable to allocate requests");
  return ncclInternalError;
}

ncclResult_t ncclSocketTest(void* request, int* done, int* size) {
  *done = 0;
  struct ncclSocketRequest *r = (struct ncclSocketRequest*)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return ncclInternalError;
  }
  if (r->result != ncclSuccess) return r->result;
  if (r->offset == -1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    NCCLCHECK(socketProgress(r->op, r->ctrlFd, &data, sizeof(int), &offset));

    if (offset == 0) return ncclSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int)) NCCLCHECK(socketWait(r->op, r->ctrlFd, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == NCCL_SOCKET_RECV && data > r->size) {
      WARN("NET/Socket : message truncated : receiving %d bytes instead of %d", data, r->size);
      return ncclInternalError;
    }
    r->size = data;
    r->offset = 0;
  }
  if (r->offset == r->size) {
    if (size) *size = r->size;
    *done = 1;
    r->used = 0;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclSocketDeregMr(void* comm, void* mhandle) { return ncclSuccess; }

ncclResult_t ncclSocketIsend(void* sendComm, void* data, int size, void* mhandle, void** request) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)sendComm;
  NCCLCHECK(ncclSocketGetRequest(comm, NCCL_SOCKET_SEND, data, size, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)recvComm;
  NCCLCHECK(ncclSocketGetRequest(comm, NCCL_SOCKET_RECV, data, size, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketFlush(void* recvComm, void* data, int size, void* mhandle) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclSocketClose(void* opaqueComm) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)opaqueComm;
  if (comm) {
    comm->state = stop;
    for (int i=0; i<comm->nThreads; i++) {
      if (comm->proxyThread[i]) {
        pthread_join(comm->proxyThread[i], NULL);
      }
    }
    free(comm->reqs.requests);
    if (comm->ctrlFd != -1) close(comm->ctrlFd);
    for (int i=0; i<comm->nSocks; i++) {
      if (comm->fd[i] != -1) close(comm->fd[i]);
    }
    free(comm);
  }
  return ncclSuccess;
}

ncclNet_t ncclNetSocket = {
  "Socket",
  ncclSocketInit,
  ncclSocketDevices,
  ncclSocketPciPath,
  ncclSocketPtrSupport,
  ncclSocketListen,
  ncclSocketConnect,
  ncclSocketAccept,
  ncclSocketRegMr,
  ncclSocketDeregMr,
  ncclSocketIsend,
  ncclSocketIrecv,
  ncclSocketFlush,
  ncclSocketTest,
  ncclSocketClose,
  ncclSocketClose,
  ncclSocketClose
};
