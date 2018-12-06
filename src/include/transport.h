/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

#include "nccl.h"
#include <stdint.h>
#include "nvmlwrap.h"

#define NTRANSPORTS 3

extern struct ncclTransport ncclTransports[];

// Forward declarations
struct ncclRing;
struct ncclConnector;
struct ncclComm;

struct ncclPeerInfo {
  int rank;
  int cudaDev;
  int nvmlDev;
  uint64_t hostHash;
  uint64_t pidHash;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
};

// Used to hold the transport connection values
typedef int64_t ncclTvalue_t;

#define CONNECT_SIZE 128
struct ncclConnect {
  char data[CONNECT_SIZE];
};

struct ncclProxyArgs {
  struct ncclChannel* channel;
  struct ncclConnector* connector;
  int sliceSteps;
  int chunkSteps;
  int nsteps;
  uint64_t opCount;
  int llMode;
  ncclDataType_t dtype;
  ncclRedOp_t redOp;
  int active;   // add component before this line -- it is left out during initialization
};

struct ncclTransportComm {
  ncclResult_t (*setup)(struct ncclPeerInfo*, struct ncclPeerInfo*, struct ncclConnect*, struct ncclConnector*, int buffSize, int channelId);
  ncclResult_t (*connect)(struct ncclConnect*, struct ncclConnector*);
  ncclResult_t (*free)(void*);
  ncclResult_t (*proxy)(struct ncclProxyArgs*);
};

struct ncclTransport {
  const char name[4];
  ncclResult_t (*canConnect)(ncclTvalue_t*, struct ncclPeerInfo*, struct ncclPeerInfo*);
  ncclResult_t (*getRings)(int, int*, int*, ncclTvalue_t*, int*, int*, int*, int, int*);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

struct ncclCollTransportComm {
  ncclResult_t (*setup)(struct ncclPeerInfo*, struct ncclConnect*, struct ncclConnector*, struct ncclConnector*, int buffSize, int channelId);
  ncclResult_t (*connect)(struct ncclConnect*, int nranks, struct ncclConnector*, struct ncclConnector*);
  ncclResult_t (*free)(void*, void*);
  ncclResult_t (*sendProxy)(struct ncclProxyArgs*);
  ncclResult_t (*recvProxy)(struct ncclProxyArgs*);//TODO: merge
};

struct ncclCollTransport {
  const char name[4];
  ncclResult_t (*canConnect)(ncclTvalue_t*, struct ncclPeerInfo*, struct ncclPeerInfo*);
  struct ncclCollTransportComm allreduce;
};

#include <pthread.h>

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

#define TRANSPORT_PROXY_FIFO_SIZE NCCL_MAX_OPS

struct transportProxyInfo {
  pthread_t thread;
  threadFunc_t func;
  volatile int proxyReady;
  struct ncclProxyArgs argsFifo[TRANSPORT_PROXY_FIFO_SIZE];
  volatile uint64_t argsFifoHead;
  volatile uint64_t argsFifoTail;
  pthread_cond_t cond;
  pthread_mutex_t mutex;
  struct ncclComm *comm;
};

ncclResult_t transportCreateProxy(struct ncclConnector* connector, threadFunc_t proxyFunc);
ncclResult_t transportDestroyProxy(struct ncclConnector* connector);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t transportSaveProxies(struct ncclProxyArgs* args, int pattern, int root, int nranks);
ncclResult_t transportStartProxies(struct ncclComm* comm);

#include <unistd.h>

// Spin wait until func evaluates to true
template<typename FUNC>
inline void transportProxyWait(const FUNC& func) {
  while (!func()) {
    sched_yield();
  }
}

inline void transportProxyIdle(int idle) {
  sched_yield();
}

#endif
