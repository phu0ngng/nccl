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

struct ncclProxySharedBuffers {
  int size;
  char* cudaBuff;
  char* hostBuff;
  struct ncclProxyArgs* proxyAppend[2*MAXCHANNELS]; // Separate send and recv
  // Collnet sharing is technically per device, but for now MAXDEVICES == MAXCHANNELS.
  struct ncclProxyArgs* proxyAppendCollNet[2*MAXCHANNELS];
  void* collNetResources;
};

struct ncclProxyPool;
struct ncclProxyState {
  pthread_cond_t cond;
  pthread_mutex_t opsMutex;
  pthread_mutex_t poolMutex;
  struct ncclProxyArgs* freeList;
  struct ncclProxyArgs* freeListEnd;
  int freeListCount;
  bool stop;
  struct ncclProxySharedBuffers sharedBuffs;
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
ncclResult_t ncclProxySharedBuffersGetP2p(struct ncclComm* comm, int cuda, int type, int channel, int slot, int index, char** ptr);
ncclResult_t ncclProxySharedBuffersGetCollNet(struct ncclComm* comm, int cuda, int type, int slot, int index, char** ptr);
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
