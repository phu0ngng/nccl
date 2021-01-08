/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PROXY_H_
#define NCCL_PROXY_H_

#include <pthread.h>

enum ncclProxyOpState { ncclProxyOpNone, ncclProxyOpReady, ncclProxyOpProgress };

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyArgs*);


#define NCCL_PROXY_MAX_SUBS NCCL_MAX_WORK_ELEMENTS

struct ncclProxySubArgs {
  struct ncclChannel* channel;
  struct ncclConnector* connector;
  int nsteps;
  size_t sendbytes;
  size_t recvbytes;

  // Internal state
  uint64_t posted;
  uint64_t received; // Only used by recv proxy to wait for flush.
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
  uint64_t opCount;
  int protocol;
  ncclDataType_t dtype;
  ncclRedOp_t redOp;
  int state;

  int idle;

  // Element linking
  pthread_mutex_t mutex;
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
  struct ncclProxyArgs** proxyAppendPtr;
};

struct ncclProxySharedBuffers {
  int nslots;
  int slotSize;
  char* cudaBuff[2*MAXCHANNELS];
  int* cudaUsed[2*MAXCHANNELS];
  char* hostBuff[2*MAXCHANNELS];
  int* hostUsed[2*MAXCHANNELS];
  struct ncclProxyArgs* proxyAppend[2*MAXCHANNELS]; // Separate send and recv
};

struct ncclProxyPool;
struct ncclProxyState {
  pthread_cond_t cond;
  pthread_mutex_t opsMutex;
  pthread_mutex_t poolMutex;
  bool stop;
  struct ncclProxySharedBuffers* sharedBuffs;
  struct ncclProxyArgs* ops;
  struct ncclProxyArgs* nextOps;
  struct ncclProxyArgs* nextOpsEnd;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
};

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t ncclProxySaveColl(struct ncclProxyArgs* args, int pattern, int root, int nranks);
ncclResult_t ncclProxySaveP2p(struct ncclInfo* info, struct ncclChannel* channel);
ncclResult_t ncclProxyStart(struct ncclComm* comm);
ncclResult_t ncclProxyCreate(struct ncclComm* comm);
ncclResult_t ncclProxyDestroy(struct ncclComm* comm);

ncclResult_t ncclProxySharedBuffersInit(struct ncclComm* comm, int cuda, int* size, char** ptr);
ncclResult_t ncclProxySharedBuffersAlloc(struct ncclComm* comm, int cuda, int type, int channel, int size, char** ptr);
ncclResult_t ncclProxySharedBuffersFree(struct ncclComm* comm, int cuda, int type, int channel, int size, char* ptr);
ncclResult_t ncclProxySharedBuffersDestroy(struct ncclComm* comm);

#include <unistd.h>

// Spin wait until func evaluates to true
template<typename FUNC>
inline void transportProxyWait(const FUNC& func) {
  while (!func()) {
    sched_yield();
  }
}

#endif
