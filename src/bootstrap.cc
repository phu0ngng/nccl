/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include <unistd.h>
#include <sys/types.h>
#include "proxy.h"

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE+1];
static union socketAddress bootstrapNetIfAddr;
static int bootstrapNetInitDone = 0;
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetInitDone == 0) {
      char* env = getenv("NCCL_COMM_ID");
      if (env) {
        union socketAddress remoteAddr;
        if (GetSocketAddrFromString(&remoteAddr, env) != ncclSuccess) {
          WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          return ncclInvalidArgument;
        }
        if (findInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          return ncclSystemError;
        }
      } else {
        int nIfs = findInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1);
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return ncclInternalError;
        }
      }
      char line[SOCKET_NAME_MAXLEN+MAX_IF_NAME_SIZE+2];
      sprintf(line, " %s:", bootstrapNetIfName);
      socketToString(&bootstrapNetIfAddr, line+strlen(line));
      INFO(NCCL_INIT, "Bootstrap : Using%s", line);
      bootstrapNetInitDone = 1;
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

static ncclResult_t bootstrapNetAccept(int listenFd, int* recvFd, union socketAddress *addr) {
  struct sockaddr *saddr = &addr->sa;
  socklen_t socklen = sizeof(union socketAddress);
  SYSCHECKVAL(accept(listenFd, saddr, &socklen), "accept", *recvFd);
  return ncclSuccess;
}

// Additional sync functions
static ncclResult_t bootstrapNetSend(int fd, union socketAddress *addr, void* data, int size) {
  NCCLCHECK(socketSend(fd, addr, &size, sizeof(int)));
  NCCLCHECK(socketSend(fd, addr, data, size));
  return ncclSuccess;
}
static ncclResult_t bootstrapNetRecv(int fd, union socketAddress *addr, void* data, int size) {
  int recvSize;
  NCCLCHECK(socketRecv(fd, addr, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  NCCLCHECK(socketRecv(fd, addr, data, std::min(recvSize, size)));
  return ncclSuccess;
}

struct extInfo {
  int rank;
  int nranks;
  union socketAddress extAddressListenRoot;
  union socketAddress extAddressListen;
};

#include <sys/resource.h>

static ncclResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return ncclSuccess;
}

static void *bootstrapRoot(void* args) {
  int listenFd = (uint64_t)args;
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;
  struct extInfo info;
  union socketAddress *rankAddresses = NULL;
  union socketAddress *rankAddressesRoot = NULL; // for initial rank <-> root information exchange
  union socketAddress *zero = NULL;
  NCCLCHECKGOTO(ncclCalloc(&zero, 1), res, out);
  setFilesLimit();

  TRACE(NCCL_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  do {
    int tmpFd;
    union socketAddress addr;
    NCCLCHECKGOTO(bootstrapNetAccept(listenFd, &tmpFd, &addr), res, out);
    NCCLCHECKGOTO(bootstrapNetRecv(tmpFd, &addr, &info, sizeof(info)), res, out);
    close(tmpFd);

    if (c == 0) {
      nranks = info.nranks;
      NCCLCHECKGOTO(ncclCalloc(&rankAddresses, nranks), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nranks), res, out);
    }

    if (nranks != info.nranks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    if (memcmp(zero, &rankAddressesRoot[info.rank], sizeof(union socketAddress)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    // Save the connection handle for that rank
    memcpy(rankAddressesRoot+info.rank, &info.extAddressListenRoot, sizeof(union socketAddress));
    memcpy(rankAddresses+info.rank, &info.extAddressListen, sizeof(union socketAddress));

    ++c;
    TRACE(NCCL_INIT, "Received connect from rank %d total %d/%d",  info.rank, c, nranks);
  } while (c < nranks);
  TRACE(NCCL_INIT, "COLLECTED ALL %d HANDLES", nranks);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r=0; r<nranks; ++r) {
    int next = (r+1) % nranks;
    int tmpSendFd;
    NCCLCHECKGOTO(connectAddress(&tmpSendFd, rankAddressesRoot+r), res, out);
    NCCLCHECKGOTO(bootstrapNetSend(tmpSendFd, rankAddressesRoot+r, rankAddresses+next, sizeof(union socketAddress)), res, out);
    close(tmpSendFd);
  }
  TRACE(NCCL_INIT, "SENT OUT ALL %d HANDLES", nranks);

out:
  close(listenFd);
  if (rankAddresses) free(rankAddresses);
  if (rankAddressesRoot) free(rankAddressesRoot);
  if (zero) free(zero);

  TRACE(NCCL_INIT, "DONE");
  return NULL;
}

ncclResult_t bootstrapCreateRoot(ncclUniqueId* id, bool idFromEnv) {
  union socketAddress* connectAddr = (union socketAddress*) id;
  int listenFd;
  NCCLCHECK(createListenSocket(&listenFd, connectAddr));
  pthread_t thread;
  pthread_create(&thread, NULL, bootstrapRoot, (void*)(uint64_t)listenFd);
  return ncclSuccess;
}

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* id) {
  static_assert(sizeof(union socketAddress) < sizeof(ncclUniqueId), "NetId does not fit inside ncclUniqueId");
  memset(id, 0, sizeof(ncclUniqueId));
  union socketAddress* connectAddr = (union socketAddress*) id;

  char* env = getenv("NCCL_COMM_ID");
  if (env) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (GetSocketAddrFromString(connectAddr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
  } else {
    memcpy(id, &bootstrapNetIfAddr, sizeof(union socketAddress));
    NCCLCHECK(bootstrapCreateRoot(id, false));
  }

  return ncclSuccess;
}

struct unexConn {
  int peer;
  int tag;
  int fd;
  union socketAddress addr;
  struct unexConn* next;
};

struct extState {
  int extListenFd;
  int extRingRecvFd;
  int extRingSendFd;
  union socketAddress extRingRecvAddr, extRingSendAddr;
  union socketAddress* peerCommAddresses;
  union socketAddress* peerAllocAddresses;
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;

  // Intermediate memory allocation service
  struct ncclProxyServiceArgs* proxyArgs;
  pthread_t proxyThread;
};

ncclResult_t bootstrapProxyConnect(void* commState, int transport, int send, int rank, int* fd) {
  struct extState* state = (struct extState*)commState;
  NCCLCHECK(connectAddress(fd, state->peerAllocAddresses+rank));
  NCCLCHECK(socketSend(*fd, &transport, sizeof(int)));
  NCCLCHECK(socketSend(*fd, &send, sizeof(int)));
  return ncclSuccess;
}

ncclResult_t bootstrapFree(int fd) {
  SYSCHECK(close(fd), "close");
  return ncclSuccess;
}

ncclResult_t bootstrapInit(ncclUniqueId * id, struct ncclComm* comm) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  struct extState* state;
  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  comm->bootstrap = state;

  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);

  struct extInfo info = { 0 };
  info.rank = rank;
  info.nranks = nranks;
  int tmpSendFd, tmpRecvFd;

  int extListenFdRoot;
  memcpy(&info.extAddressListen,     &bootstrapNetIfAddr, sizeof(union socketAddress));
  memcpy(&info.extAddressListenRoot, &bootstrapNetIfAddr, sizeof(union socketAddress));
  NCCLCHECK(createListenSocket(&state->extListenFd, &info.extAddressListen));
  NCCLCHECK(createListenSocket(&extListenFdRoot, &info.extAddressListenRoot));

  // stagger connection times to avoid an overload of the root
  if (nranks > 128) {
    long msec = rank;
    struct timespec tv;
    tv.tv_sec = msec / 1000;
    tv.tv_nsec = 1000000 * (msec % 1000);
    TRACE(NCCL_INIT, "rank %d delaying connection to root by %ld msec", rank, msec);
    (void) nanosleep(&tv, NULL);
  }

  // send info on my listening socket to root
  union socketAddress* rootAddr = (union socketAddress*)id;
  NCCLCHECK(connectAddress(&tmpSendFd, rootAddr));
  NCCLCHECK(bootstrapNetSend(tmpSendFd, rootAddr,  &info, sizeof(info)));
  close(tmpSendFd);

  // get info on my "next" rank in the bootstrap ring from root
  union socketAddress addr;
  NCCLCHECK(bootstrapNetAccept(extListenFdRoot, &tmpRecvFd, &addr));
  NCCLCHECK(bootstrapNetRecv(tmpRecvFd, &addr, &state->extRingSendAddr, sizeof(state->extRingSendAddr)));
  close(tmpRecvFd);
  close(extListenFdRoot);

  NCCLCHECK(connectAddress(&state->extRingSendFd, &state->extRingSendAddr));
  // Accept the connect request from the previous rank in the AllGather ring
  NCCLCHECK(bootstrapNetAccept(state->extListenFd, &state->extRingRecvFd, &state->extRingRecvAddr));

  // AllGather all listen handlers
  NCCLCHECK(ncclCalloc(&state->peerCommAddresses, nranks));
  memcpy(state->peerCommAddresses+rank, &info.extAddressListen, sizeof(union socketAddress));
  NCCLCHECK(bootstrapAllGather(state, state->peerCommAddresses, sizeof(union socketAddress)));

  // Create the memory allocation service
  NCCLCHECK(ncclCalloc(&state->peerAllocAddresses, nranks));
  memcpy(state->peerAllocAddresses+rank, &bootstrapNetIfAddr, sizeof(union socketAddress));
  NCCLCHECK(ncclCalloc(&state->proxyArgs, 1));
  state->proxyArgs->comm = comm;
  CUDACHECK(cudaGetDevice(&state->proxyArgs->cudaDev));
  NCCLCHECK(createListenSocket(&state->proxyArgs->listenFd, state->peerAllocAddresses+rank));
  pthread_create(&state->proxyThread, NULL, ncclProxyService, state->proxyArgs);
  NCCLCHECK(bootstrapAllGather(state, state->peerAllocAddresses, sizeof(union socketAddress)));

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  struct extState* state = (struct extState*)commState;
  char* data = (char*)allData;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(NCCL_INIT, "rank %d nranks %d size %d", rank, nranks, size);

  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from left
   * and send previous step's data from (rank-i) to right
   */
  for (int i=0; i<nranks-1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;

    // Send slice to the right
    NCCLCHECK(bootstrapNetSend(state->extRingSendFd, &state->extRingSendAddr, data+sslice*size, size));
    // Recv slice from the left
    NCCLCHECK(bootstrapNetRecv(state->extRingRecvFd, &state->extRingRecvAddr, data+rslice*size, size));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  struct extState* state = (struct extState*)commState;
  int tmpSendFd;
  union socketAddress *addr = state->peerCommAddresses+peer;
  NCCLCHECK(connectAddress(&tmpSendFd, addr));
  NCCLCHECK(bootstrapNetSend(tmpSendFd, addr, &state->rank, sizeof(int)));
  NCCLCHECK(bootstrapNetSend(tmpSendFd, addr, &tag, sizeof(int)));
  NCCLCHECK(bootstrapNetSend(tmpSendFd, addr, data, size));
  close(tmpSendFd);
  return ncclSuccess;
}

ncclResult_t unexpectedEnqueue(struct extState* state, int peer, int tag, int fd, union socketAddress *addr) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  unex->fd = fd;
  unex->addr = *addr;

  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = unex;
  return ncclSuccess;
}

int unexpectedDequeue(struct extState* state, int peer, int tag, union socketAddress *addr) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  while (elem) {
    if (elem->peer == peer && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      int fd = elem->fd;
      *addr = elem->addr;
      free(elem);
      return fd;
    }
    prev = elem;
    elem = elem->next;
  }
  return -1;
}

// We can't know who we'll receive from, so we need to receive everything at once
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  struct extState* state = (struct extState*)commState;

  int tmpRecvFd;
  union socketAddress addr;

  // Search unexpected connections first
  if ((tmpRecvFd = unexpectedDequeue(state, peer, tag, &addr)) != -1) {
    NCCLCHECK(bootstrapNetRecv(tmpRecvFd, &addr, ((char*)data), size));
    close(tmpRecvFd);
    return ncclSuccess;
  }

  // Then look for new connections
  while (1) {
    union socketAddress addr;
    NCCLCHECK(bootstrapNetAccept(state->extListenFd, &tmpRecvFd, &addr));
    int newPeer, newTag;
    NCCLCHECK(bootstrapNetRecv(tmpRecvFd, &addr, &newPeer, sizeof(int)));
    NCCLCHECK(bootstrapNetRecv(tmpRecvFd, &addr, &newTag, sizeof(int)));
    if (newPeer == peer && newTag == tag) {
      NCCLCHECK(bootstrapNetRecv(tmpRecvFd, &addr, ((char*)data), size));
      close(tmpRecvFd);
      return ncclSuccess;
    }
    // Unexpected connection. Save for later.
    NCCLCHECK(unexpectedEnqueue(state, newPeer, newTag, tmpRecvFd, &addr));
  }
}

ncclResult_t bootstrapClose(void* commState) {
  struct extState* state = (struct extState*)commState;
  if (state->unexpectedConnections != NULL) {
    WARN("Unexpected connections are not empty");
    return ncclInternalError;
  }
  close(state->extListenFd);
  close(state->extRingSendFd);
  close(state->extRingRecvFd);

  state->proxyArgs->stop = 1;

  // Join the proxyThread so we catch resource leaks as being hung here
  // pthread_join(state->proxyThread, nullptr);

  free(state->peerCommAddresses);
  free(state->peerAllocAddresses);
  free(state);

  return ncclSuccess;
}

ncclResult_t bootstrapAbort(void* commState) {
  struct extState* state = (struct extState*)commState;
  if (commState == NULL) return ncclSuccess;
  if (state->extListenFd) close(state->extListenFd);
  if (state->extRingSendFd) close(state->extRingSendFd);
  if (state->extRingRecvFd) close(state->extRingRecvFd);
  if (state->proxyArgs) state->proxyArgs->stop = 2;
  free(state->peerCommAddresses);
  free(state->peerAllocAddresses);
  free(state);
  return ncclSuccess;
}
