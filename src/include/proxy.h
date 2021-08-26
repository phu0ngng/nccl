/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PROXY_H_
#define NCCL_PROXY_H_

#include "socket.h"
#include <pthread.h>

enum ncclProxyOpState { ncclProxyOpNone, ncclProxyOpReady, ncclProxyOpProgress };

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclComm*, struct ncclProxyArgs*);

#define NCCL_PROXY_MAX_SUBS MAXCHANNELS
static_assert(NCCL_MAX_WORK_ELEMENTS <= MAXCHANNELS, "Not enough sub space for max work elements");

struct ncclProxySubArgs {
  int channelId;
  struct ncclProxyConnection* connection;
  int nsteps;
  ssize_t sendbytes;
  ssize_t recvbytes;
  int sendChunkSize;
  int recvChunkSize;
  int delta;

  // Internal state
  uint64_t base;
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;
  uint64_t transmitted;
  uint64_t done;
  uint64_t end;
  void* requests[NCCL_STEPS];
};

struct ncclProxyArgs {
  proxyProgressFunc_t progress;
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];
  int nsubs;
  int done;
  int sliceSteps;
  int chunkSteps;
  int chunkSize;
  uint64_t opCount;
  int protocol;
  ncclDataType_t dtype;
  ncclRedOp_t redOp;
  ncclPattern_t pattern;
  int root;
  int state;
  char* sharedBuff[NCCL_STEPS];
  int sharedSize[NCCL_STEPS];

  int idle;

  // Element linking
  pthread_mutex_t mutex;
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
  struct ncclProxyArgs** proxyAppendPtr;
};
#define NCCL_MAX_NETDEVS 128

struct ncclProxySharedP2p {
  int refcount;
  int size;
  char* cudaBuff;
  char* hostBuff;
  cudaIpcMemHandle_t ipc;
  struct ncclProxyArgs* proxyAppend[MAXCHANNELS]; // Separate send and recv
  void** transportResources[NCCL_MAX_NETDEVS];
};

struct ncclProxySharedCollNet {
  int size;
  char* cudaBuff;
  char* hostBuff;
  struct ncclProxyArgs* proxyAppend[2*NCCL_MAX_NETDEVS];
  void* resources;
};

struct ncclProxyPeer {
  struct ncclProxySharedP2p send;
  struct ncclProxySharedP2p recv;
};

struct ncclProxyPool;
struct ncclProxyProgressState {
  pthread_t thread;
  pthread_cond_t cond;
  pthread_mutex_t opsMutex;
  pthread_mutex_t poolMutex;
  bool stop;
  struct ncclProxyPeer** localPeers;
  struct ncclProxySharedCollNet collNet;
  struct ncclProxyArgs* ops;           // Running operations, used by proxy thread
  struct ncclProxyArgs* postedOps;     // Posted operations, shared between proxy and main thread, locked with opsMutex
  struct ncclProxyArgs* postedOpsEnd;
  struct ncclProxyArgs* pool;          // Free operations for main thread
  struct ncclProxyArgs* poolFreed;     // Freed operations by the progress thread
  struct ncclProxyArgs* poolReturned;  // Shared between main and progress thread, lock with poolMutex

  struct ncclProxyPool* pools;
};

struct ncclProxyState {
  // Service thread
  pthread_t thread;
  struct ncclSocket* listenSock;
  int stop;
  union ncclSocketAddress* peerAddresses;
  struct ncclSocket* peerSocks;

  // Progress thread
  struct ncclProxyProgressState progressState;
};

struct ncclProxyConnection {
  int send, transport, shared;
  struct ncclSocket* sock;
  struct ncclTransportComm* tcomm;
  struct ncclProxyArgs *proxyAppend;
  struct ncclProxyArgs **proxyAppendPtr;
  void* transportResources;
};

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t ncclProxySaveColl(struct ncclComm* comm, struct ncclProxyArgs* args, int nranks);
ncclResult_t ncclProxyComputeP2p(struct ncclInfo* info, struct ncclProxyArgs* args);
ncclResult_t ncclProxySaveP2p(struct ncclComm* comm, struct ncclProxyArgs* args);
ncclResult_t ncclProxyStart(struct ncclComm* comm);
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses);
ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int rank, struct ncclProxyConnector* proxyConn);
enum ncclProxyMsgType {
  ncclProxyMsgInit = 1,
  ncclProxyMsgSetup = 2,
  ncclProxyMsgConnect = 3,
  ncclProxyMsgAppend = 4,
  ncclProxyMsgStart = 5,
  ncclProxyMsgClose = 6,
  ncclProxyMsgAbort = 7,
  ncclProxyMsgStop = 8
};
ncclResult_t ncclProxyCall(struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize);
ncclResult_t ncclProxyDestroy(struct ncclComm* comm);

ncclResult_t ncclProxySharedBuffersInitCollNet(struct ncclComm* comm, int cuda, int* size, char** ptr);
ncclResult_t ncclProxySharedBuffersGetCollNet(struct ncclComm* comm, int type, int slot, int index, int* offset);
ncclResult_t ncclProxySharedBuffersDestroyCollNet(struct ncclComm* comm);
#endif
