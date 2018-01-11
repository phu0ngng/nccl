/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#define NETCHECKJUMP(call, out) do { \
  int res = call; \
  if (res != 0) { \
    /* Print the back trace*/ \
    INFO("%s:%d -> %d [bthread]", __FILE__, __LINE__, res); \
    goto out; \
  } \
} while (0);

struct extId {
  ncclNetHandle_t extHandle;
  void* extListenComm;
  uint64_t hostHash;
  pid_t pid;
  int fd;
  pthread_t boostrapThread;
};

struct bootstrapOp {
  int op;
  int size;
};

struct extInfo {
  int rank;
  int nranks;
  ncclNetHandle_t extHandle;
};

enum { 
  BOOTSTRAP_ALLGATHER = 1,
  BOOTSTRAP_RINGEXCHANGE,
};

static void *bootstrapRoot(void* commId) {
  struct extInfo info;
  struct extId* id = (struct extId*)commId;
  struct bootstrapOp bop;
  void **extSendComm = NULL;
  void **extRecvComm = NULL;
  int size, alloc_size = 0; 
  char* data = NULL;
  bool idFromEnv = id->pid < 0;

  /* Receive addresses from all ranks */
  int nranks = 0, c = 0;
  do {
      void* tmpRecvComm;
      NETCHECKJUMP(ncclNetSocket.accept(id->extListenComm, &tmpRecvComm), out);
      NETCHECKJUMP(ncclNetSocket.recv(tmpRecvComm, &info, sizeof(info)), out);
      if (!c) { 
          extSendComm = (void**)calloc(info.nranks, sizeof(void*));
          extRecvComm = (void**)calloc(info.nranks, sizeof(void*));
          nranks = info.nranks;
      }

      if (nranks != info.nranks) { 
	  WARN("Bootstrap Root : mismatch in rank count from procs %d : %d\n", nranks, info.nranks);
	  goto out;
      }

      extRecvComm[info.rank] = tmpRecvComm;
      NETCHECKJUMP(ncclNetSocket.connect(idFromEnv ? -1 : 0, info.extHandle, extSendComm+info.rank), out);
      c++;
  } while (c < nranks);

  do {
      NETCHECKJUMP(ncclNetSocket.recv(extRecvComm[0], &bop, sizeof(struct bootstrapOp)), out);
      if (bop.size == -1) { 
          break;
      } else { 
	  size = bop.size;
	  if (size*nranks*2 > alloc_size) { 
	      if (data) free(data);
	      data = (char *)malloc(size*nranks*2);
	      alloc_size = size*nranks*2;
	  }
      }

      if (bop.op == BOOTSTRAP_ALLGATHER) {  
          for (int r=0; r<nranks; r++) {
              NETCHECKJUMP(ncclNetSocket.recv(extRecvComm[r], data+size*r, size), out);
          }

          for (int r=0; r<nranks; r++) {
              NETCHECKJUMP(ncclNetSocket.send(extSendComm[r], data, size*nranks), out);
          }
      } else if (bop.op == BOOTSTRAP_RINGEXCHANGE) {
	  // Receive from all and build total table
          for (int r=0; r<nranks; r++) {
              NETCHECKJUMP(ncclNetSocket.recv(extRecvComm[r], data+r*2*size, 2*size), out);
          }
        
          // Get prev/next request from everyone and answer.
          for (int r=0; r<nranks; r++) {
              int offset;
              NETCHECKJUMP(ncclNetSocket.recv(extRecvComm[r], &offset, sizeof(int)), out);
              NETCHECKJUMP(ncclNetSocket.send(extSendComm[r], data+offset, size), out);
              NETCHECKJUMP(ncclNetSocket.recv(extRecvComm[r], &offset, sizeof(int)), out);
              NETCHECKJUMP(ncclNetSocket.send(extSendComm[r], data+offset, size), out);
          }
      } else {
	  WARN("Bootstrap Root : invalid op type received %d\n", bop.op);
	  break;
      }
  } while (1);

out:
  ncclNetSocket.closeListen(id->extListenComm);
  for (int r=0; r<nranks; r++) {
      if (extSendComm[r]) ncclNetSocket.closeSend(extSendComm[r]);
      if (extRecvComm[r]) ncclNetSocket.closeRecv(extRecvComm[r]);
  }
  free(commId);
  if (extSendComm) free(extSendComm);
  if (extRecvComm) free(extRecvComm);
  return NULL;
}

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out) {
  static_assert(sizeof(extId) < sizeof(ncclUniqueId), "NetId does not fit inside ncclUniqueId");
  extId* id = (extId*)out;

  char hostname[1024];
  getHostName(hostname, 1024);
  char* env = getenv("NCCL_COMM_ID");
  if (env) {
    if (ncclSocketCreateHandle(&id->extHandle, env) != 0) {
      WARN("Invalid NCCL_COMM_ID, please use format: NCCL_COMM_ID=<ipv4>:<port> or NCCL_COMM_ID=[<ipv6>]:<port>");
      return ncclInvalidArgument;
    }
    id->pid = -1;
  } else {
    id->pid = getpid();
  }

  // if handle is preset, just listen on that handle, no need to specify an interface
  NETCHECK(ncclNetSocket.listen(env ? dontCareIf : defaultIf, &id->extHandle, &id->extListenComm));

  id->hostHash = getHostHash(hostname);

  ncclUniqueId* threadIdCopy = (ncclUniqueId*)malloc(sizeof(ncclUniqueId));
  memcpy(threadIdCopy, id, sizeof(ncclUniqueId));
  pthread_create(&id->boostrapThread, NULL, bootstrapRoot, (void *)threadIdCopy);
  return ncclSuccess;
}

ncclResult_t bootstrapGetUniqueIdFromEnv(ncclUniqueId* out) {
  static_assert(sizeof(extId) < sizeof(ncclUniqueId), "NetId does not fit inside ncclUniqueId");
  extId* id = (extId*)out;

  // use negative pid to indicate that comm id is from env (pid_t is int)
  // if ncclGetUniqueID is called, getpid() would return pid >= 1
  id->pid = -1;

  char* env = getenv("NCCL_COMM_ID");
  if (env && strlen(env) > 1) {
    if (ncclSocketCreateHandle(&id->extHandle, env) == 0) {
      return ncclSuccess;
    }
  }
  WARN("Invalid NCCL_COMM_ID or none specified, please use format: NCCL_COMM_ID=<ipv4>:<port> or NCCL_COMM_ID=[<ipv6>]:<port>");
  return ncclInvalidArgument;
}

struct extState {
  void* extRecvComm;
  void* extSendComm;
  int rank;
  int nranks;
};

ncclResult_t bootstrapInit(ncclUniqueId* commId, int rank, int nranks, void** commState) {
  struct extId* id = (struct extId*)commId;
  bool idFromEnv = id->pid < 0;
  struct extState* state = (struct extState*)malloc(sizeof(struct extState));
  state->rank = rank;
  state->nranks = nranks;
  *commState = state;

  struct extInfo info;
  info.rank = rank;
  info.nranks = nranks;
  void* tmpListenComm;
  // Pass the remote address to listen via info
  if (idFromEnv) {
    memcpy(&info.extHandle, &id->extHandle, sizeof(ncclNetHandle_t));
  }
  // listen will return the local address via info (specify interface type 'findSubnetIf')
  int dev = idFromEnv ? findSubnetIf : defaultIf;
  NETCHECK(ncclNetSocket.listen(dev, &info.extHandle, &tmpListenComm));
  NETCHECK(ncclNetSocket.connect(dev, id->extHandle, &state->extSendComm));
  NETCHECK(ncclNetSocket.send(state->extSendComm, &info, sizeof(info)));
  NETCHECK(ncclNetSocket.accept(tmpListenComm, &state->extRecvComm));
  NETCHECK(ncclNetSocket.closeListen(tmpListenComm));

  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  struct extState* state = (struct extState*)commState;
  char* data = (char*)allData;
  struct bootstrapOp bop;

  bop.op = BOOTSTRAP_ALLGATHER;
  bop.size = size;

  if (!state->rank) { 
      NETCHECK(ncclNetSocket.send(state->extSendComm, &bop, sizeof(struct bootstrapOp)));
  } 

  NETCHECK(ncclNetSocket.send(state->extSendComm, data+state->rank*size, size));
  NETCHECK(ncclNetSocket.recv(state->extRecvComm, data, size*state->nranks));

  return ncclSuccess;
}

ncclResult_t bootstrapRingExchange(void* commState, void* prevNextData, int prev, int next, int size) {
  struct extState* state = (struct extState*)commState;
  char* mydata = (char*)prevNextData;
  int prev_offset = prev*2*size+size, next_offset = next*2*size;

  struct bootstrapOp bop;
  bop.op = BOOTSTRAP_RINGEXCHANGE;
  bop.size = size;

  if (!state->rank) {
      NETCHECK(ncclNetSocket.send(state->extSendComm, &bop, sizeof(struct bootstrapOp))); 
  }

  // Send data to root
  NETCHECK(ncclNetSocket.send(state->extSendComm, mydata, 2*size));

  // Receive prev and next data
  NETCHECK(ncclNetSocket.send(state->extSendComm, &prev_offset, sizeof(int)));
  NETCHECK(ncclNetSocket.recv(state->extRecvComm, mydata, size));
  NETCHECK(ncclNetSocket.send(state->extSendComm, &next_offset, sizeof(int)));
  NETCHECK(ncclNetSocket.recv(state->extRecvComm, mydata+size, size));


  return ncclSuccess;
}

ncclResult_t bootstrapClose(void* commState) {
  struct extState* state = (struct extState*)commState;
  struct bootstrapOp bop;
  bop.size = -1;

  if (!state->rank) { 
      NETCHECK(ncclNetSocket.send(state->extSendComm, &bop, sizeof(struct bootstrapOp)));
  } 

  NETCHECK(ncclNetSocket.closeSend(state->extSendComm));
  NETCHECK(ncclNetSocket.closeRecv(state->extRecvComm));

  free(state);

  return ncclSuccess;
}
