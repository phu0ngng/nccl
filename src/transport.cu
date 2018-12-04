/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport,
  netTransport,
};

static void FifoPullArgs(struct transportProxyInfo* info, struct ncclProxyArgs *args) {
  struct ncclProxyArgs *fifoArgs = info->argsFifo + (info->argsFifoHead % TRANSPORT_PROXY_FIFO_SIZE);
  pthread_mutex_lock(&info->mutex);
  while (fifoArgs->active == 0)
    pthread_cond_wait(&info->cond, &info->mutex);
  __sync_synchronize();
  memcpy(args, fifoArgs, sizeof(struct ncclProxyArgs));
  __sync_synchronize();
  fifoArgs->active = 0;
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
  info->argsFifoHead++;
}

static struct ncclProxyArgs* FifoGetNextArgs(struct transportProxyInfo* info) {
  if (info == NULL) return NULL;
  struct ncclProxyArgs* fifoArgs = info->argsFifo + (info->argsFifoTail % TRANSPORT_PROXY_FIFO_SIZE);
  pthread_mutex_lock(&info->mutex);
  while (fifoArgs->active == 1)
    pthread_cond_wait(&info->cond, &info->mutex);
  pthread_mutex_unlock(&info->mutex);
  info->argsFifoTail++;
  return fifoArgs;
}

static void FifoPushArgs(struct transportProxyInfo* info) {
  if (info == NULL) return;

  struct ncclProxyArgs* fifoArgs = info->argsFifo + ((info->argsFifoTail-1) % TRANSPORT_PROXY_FIFO_SIZE);
  if (fifoArgs->active == 0) return;

  pthread_mutex_lock(&info->mutex);
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

static void WaitProxyReady(struct transportProxyInfo* info) {
  pthread_mutex_lock(&info->mutex);
  while (info->proxyReady == 0)
    pthread_cond_wait(&info->cond, &info->mutex);
  pthread_mutex_unlock(&info->mutex);
}

static void SetProxyReady(struct transportProxyInfo* info) {
  pthread_mutex_lock(&info->mutex);
  info->proxyReady = 1;
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

static void StopProxy(struct transportProxyInfo* info) {
  struct ncclProxyArgs* fifoArgs = FifoGetNextArgs(info);
  fifoArgs->active = -1;
  FifoPushArgs(info);
}

#define RECV 0
#define SEND 1

static bool NeedProxy(int type, int pattern, int root, struct ncclRing* ring, int nranks) {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  // Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = pattern == ncclPatternPipelineFrom ?
      /*                            no recv /  no send    if root = */
      /* bcast  */ (type == RECV ?   myrank : nextrank ):
      /* reduce */ (type == RECV ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

enum { proxyRecv=0, proxySend=1 };

template <int type>
static void SaveProxy(int peer, struct ncclProxyArgs* args) {
  if (peer < 0) return;
  struct ncclPeer* peerComm = args->channel->peers+peer;
  struct ncclConnector* connector = type == proxyRecv ? &peerComm->recv : &peerComm->send;
  struct transportProxyInfo* info = connector->proxyInfo;
  if (info == NULL) return;
//  TRACE(NET, "Saving %s proxy to peer %d\n", type == proxyRecv ? "recv" : "send", peer);
  args->connector = connector;
  struct ncclProxyArgs* fifoArgs = FifoGetNextArgs(info);
  memcpy(fifoArgs, args, sizeof(struct ncclProxyArgs));
  __sync_synchronize();
  fifoArgs->active = 1;
}

ncclResult_t transportSaveProxies(struct ncclProxyArgs* args, int pattern, int root, int nranks) {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice || pattern == ncclPatternPipelineFrom || pattern == ncclPatternPipelineTo) {
    struct ncclRing* ring = &args->channel->ring;
    if (NeedProxy(RECV, pattern, root, ring, nranks)) SaveProxy<proxyRecv>(ring->prev, args);
    if (NeedProxy(SEND, pattern, root, ring, nranks)) SaveProxy<proxySend>(ring->next, args);
  }
  if (pattern == ncclPatternTreeUp || pattern == ncclPatternTreeUpDown) {
    // Tree up
    struct ncclTree* tree = &args->channel->tree;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) SaveProxy<proxyRecv>(tree->down[i], args);
    SaveProxy<proxySend>(tree->up, args);
  }
  if (pattern == ncclPatternTreeDown || pattern == ncclPatternTreeUpDown) {
    // Tree down
    struct ncclTree* tree = &args->channel->tree;
    for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) SaveProxy<proxySend>(tree->down[i], args);
    SaveProxy<proxyRecv>(tree->up, args);
  }
  return ncclSuccess;
}

ncclResult_t transportStartProxies(ncclComm* comm) {
  for (int r=0; r<comm->nChannels; r++) {
    struct ncclRing* ring = &comm->channels[r].ring;
    FifoPushArgs(comm->channels[r].peers[ring->prev].recv.proxyInfo);
    FifoPushArgs(comm->channels[r].peers[ring->next].send.proxyInfo);

    struct ncclTree* tree = &comm->channels[r].tree;
    for (int i=0; tree->down[i] >= 0; i++) FifoPushArgs(comm->channels[r].peers[tree->down[i]].recv.proxyInfo);
    if (tree->up >= 0) FifoPushArgs(comm->channels[r].peers[tree->up].recv.proxyInfo);
    for (int i=0; tree->down[i] >= 0; i++) FifoPushArgs(comm->channels[r].peers[tree->down[i]].send.proxyInfo);
    if (tree->up >= 0) FifoPushArgs(comm->channels[r].peers[tree->up].send.proxyInfo);
  }
  pthread_yield(); // Let other threads run
  return ncclSuccess;
}

void* persistentThread(void *opaqueInfo) {
  struct transportProxyInfo* info = (struct transportProxyInfo*)opaqueInfo;
  // Signal the main thread the context is created and it can proceed.
  SetProxyReady(info);
  INFO(INIT, "Proxy ready");
  while (1) {
    struct ncclProxyArgs args;
    FifoPullArgs(info, &args);
    if (args.active == -1) {
      // Main thread asked to stop
      return NULL;
    }
    ncclResult_t res = info->func(&args);
    if (res != ncclSuccess) {
      info->comm->fatalError = res;
      WARN("%s:%d -> %d [Proxy thread error]", __FILE__, __LINE__, res);
    }
  }
}

ncclResult_t transportCreateProxy(struct ncclConnector* connector) {
  threadFunc_t proxyfunc = (threadFunc_t) connector->transportComm->proxy;
  if (proxyfunc) {
    struct transportProxyInfo* info;
    NCCLCHECK(ncclCalloc(&info, 1));
    connector->proxyInfo = info;
    info->cond = PTHREAD_COND_INITIALIZER;
    info->mutex = PTHREAD_MUTEX_INITIALIZER;
    info->func = proxyfunc;
    info->argsFifoHead = info->argsFifoTail = 0;
    info->proxyReady = 0;
    info->comm = connector->comm;
    pthread_create(&connector->proxyInfo->thread, NULL, persistentThread, info);
    // Wait for thread to initialize its CUDA context.
    WaitProxyReady(info);
  }
  return ncclSuccess;
}

ncclResult_t transportDestroyProxy(struct ncclConnector* connector) {
  if (connector->proxyInfo) {
    StopProxy(connector->proxyInfo);
    pthread_join(connector->proxyInfo->thread, NULL);
    free(connector->proxyInfo);
    connector->proxyInfo = NULL;
  }
  return ncclSuccess;
}
