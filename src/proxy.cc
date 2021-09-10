/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "collectives.h"
#include "socket.h"

enum { proxyRecv=0, proxySend=1 };

static bool NeedProxy(int type, int pattern, int root, struct ncclRing* ring, int nranks) {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  // Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = pattern == ncclPatternPipelineFrom ?
      /*                            no recv /  no send    if root = */
      /* bcast  */ (type == proxyRecv ?   myrank : nextrank ):
      /* reduce */ (type == proxyRecv ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

#define PROXYARGS_ALLOCATE_SIZE 128
struct ncclProxyPool {
  struct ncclProxyPool *next;
  struct ncclProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];
};

static ncclResult_t allocateArgs(struct ncclComm* comm, struct ncclProxyArgs** argsptr) {
  struct ncclProxyProgressState* state = &comm->proxyState.progressState;
  struct ncclProxyArgs* elem;
  if (state->pool == NULL) {
    // Check whether there are freed elements
    if (state->poolReturned) {
      pthread_mutex_lock(&state->poolMutex);
      state->pool = state->poolReturned;
      state->poolReturned = NULL;
      pthread_mutex_unlock(&state->poolMutex);
    } else {
      // Allocate a new pool of elements. Make sure we allocate the memory close
      // to the network thread
      struct ncclProxyPool* newPool;
      cpu_set_t affinitySave;
      if (CPU_COUNT(&comm->cpuAffinity)) {
        sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
        sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
      }
      NCCLCHECK(ncclCalloc(&newPool, 1));
      if (CPU_COUNT(&comm->cpuAffinity)) {
        sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
      }

      struct ncclProxyArgs* newElems = newPool->elems;
      // Chain newly allocated elements
      for (int i=0; i<PROXYARGS_ALLOCATE_SIZE; i++) {
        if (i+1 < PROXYARGS_ALLOCATE_SIZE) newElems[i].next = newElems+i+1;
      }
      // Add them all to the pool list
      state->pool = newElems;
      // Save the pool memory block for later resource release
      newPool->next = state->pools;
      state->pools = newPool;
    }
  }
  elem = state->pool;
  state->pool = state->pool->next;
  elem->next = elem->nextPeer = NULL;
  *argsptr = elem;
  return ncclSuccess;
}

//#define DEBUG_PROXY 1
#ifdef DEBUG_PROXY
#define DEBUG_PROXY_PRINT printf
#else
#define DEBUG_PROXY_PRINT(...)
#endif

#define OP_INDEX(op) ((op) ? (op)-state->pools->elems : -1)
#define OP_SEEN 0x100000
ncclResult_t dumpProxyState(struct ncclProxyProgressState* state) {
#ifdef DEBUG_PROXY
  struct ncclProxyArgs* op = state->ops;
  while (op) {
    if (op->idle & OP_SEEN) {
      WARN("Active list loop at element %ld", OP_INDEX(op));
    }
    op->idle |= OP_SEEN;
    printf("[%ld(%ld/%d)]", OP_INDEX(op), op->opCount, op->nsubs);
    if (op->nextPeer) {
      printf("(%ld)", OP_INDEX(op->nextPeer));
      struct ncclProxyArgs* n = op->nextPeer;
      n->idle |= OP_SEEN;
      while (n->nextPeer) {
        n = n->nextPeer;
        n->idle |= OP_SEEN;
      }
    }
    printf("->");
    op = op->next;
  }
  printf("[X]\n");

  struct ncclProxyArgs* free = state->pool;
  while (free) {
    if (free->idle & OP_SEEN) {
      WARN("Free list loop at element %ld", OP_INDEX(free));
    }
    free->idle |= OP_SEEN;
    free = free->next;
  }

  struct ncclProxyPool* p = state->pools;
  int i = 0;
  while (p) {
    for (int e=0; e<PROXYARGS_ALLOCATE_SIZE; e++) {
      if ((p->elems[e].idle & OP_SEEN) == 0) {
        WARN("Element %d of pool %d has been lost", e, i);
        struct ncclProxyArgs* free = state->pool;
        printf("Free list ");
        while (free) {
          printf("--> %ld ", OP_INDEX(free));
          free = free->next;
        }
        printf("\n");
        return ncclInternalError;
      }
      p->elems[e].idle -= OP_SEEN;
    }
    p = p->next;
    i++;
  }
#endif
  return ncclSuccess;
}

static ncclResult_t ProxyAppend(struct ncclProxyProgressState* state, struct ncclProxyArgs* args) {
  struct ncclProxyArgs* proxyAppend = *args->proxyAppendPtr;
  int shared = args->subs[0].connection->shared;
  if (proxyAppend) {
    if (shared && proxyAppend->opCount == args->opCount) {
      if ((proxyAppend->sliceSteps != args->sliceSteps) ||
          (proxyAppend->chunkSteps != args->chunkSteps) ||
          (proxyAppend->protocol != args->protocol) ||
          (proxyAppend->dtype != args->dtype) ||
          (proxyAppend->redOp != args->redOp)) {
        WARN("Proxy append mismatch");
        return ncclInternalError;
      }
      if (proxyAppend->nsubs >= NCCL_PROXY_MAX_SUBS) {
        WARN("Proxy append out of bound");
        return ncclInternalError;
      }
      memcpy(proxyAppend->subs+proxyAppend->nsubs, args->subs, sizeof(struct ncclProxySubArgs));
      proxyAppend->nsubs++;
      args->next = proxyAppend->next;
      // Free args as we merged them
      args->next = state->poolFreed;
      state->poolFreed = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld/%5ld) as group with %5ld\n", OP_INDEX(args), shared, proxyAppend->opCount, args->opCount, OP_INDEX(proxyAppend));
    } else {
      proxyAppend->nextPeer = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld/%5ld) as nextPeer of %5ld\n", OP_INDEX(args), shared, proxyAppend->opCount, args->opCount, OP_INDEX(proxyAppend));
      *(args->proxyAppendPtr) = args;
    }
  } else {
    // Nothing running for that peer. Add to the list
    if (state->ops == NULL) {
      // Create the list
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as first element\n", OP_INDEX(args), shared, args->opCount);
      state->ops = args;
    } else {
      // Append element at the end of the list
      struct ncclProxyArgs* last = state->ops;
      while (last->next) last = last->next;
      last->next = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as last element\n", OP_INDEX(args),shared, args->opCount);
    }
    *(args->proxyAppendPtr) = args;
  }
  return ncclSuccess;
}

static ncclResult_t SaveProxy(struct ncclChannel* channel, int type, int peer, struct ncclProxyArgs* args, int connIndex) {
  if (peer < 0) return ncclSuccess;

  struct ncclPeer* peerComm = channel->peers+peer;
  struct ncclConnector* connector = type == proxyRecv ? peerComm->recv+connIndex : peerComm->send+connIndex;
  if (connector->transportComm == NULL) {
    WARN("Rank %d has no transport for %s peer %d on channel %d", connector->comm->rank,
        type == proxyRecv ? "recv" : "send", peer, channel->id);
    return ncclInternalError;
  }
  if (connector->transportComm->proxyProgress == NULL) return ncclSuccess;

  args->state = ncclProxyOpReady;

  NCCLCHECK(ncclProxyCall(&connector->proxyConn, ncclProxyMsgAppend, args, sizeof(struct ncclProxyArgs), NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclProxySaveColl(struct ncclComm* comm, struct ncclProxyArgs* args, int nranks) {
  struct ncclChannel* channel = comm->channels+args->subs[0].channelId;
  int pattern = args->pattern;
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice || pattern == ncclPatternPipelineFrom || pattern == ncclPatternPipelineTo) {
    struct ncclRing* ring = &channel->ring;
    if (NeedProxy(proxyRecv, pattern, args->root, ring, nranks)) NCCLCHECK(SaveProxy(channel, proxyRecv, ring->prev, args, 0));
    if (NeedProxy(proxySend, pattern, args->root, ring, nranks)) NCCLCHECK(SaveProxy(channel, proxySend, ring->next, args, 0));
  }
  if (pattern == ncclPatternTreeUp || pattern == ncclPatternTreeUpDown) {
    // Tree up
    struct ncclTree* tree = &channel->tree;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(channel, proxyRecv, tree->down[i], args, 0));
    NCCLCHECK(SaveProxy(channel, proxySend, tree->up, args, 0));
  }
  if (pattern == ncclPatternTreeDown || pattern == ncclPatternTreeUpDown) {
    // Tree down
    struct ncclTree* tree = &channel->tree;
    for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(channel, proxySend, tree->down[i], args, 0));
    NCCLCHECK(SaveProxy(channel, proxyRecv, tree->up, args, 0));
  }
  if (pattern == ncclPatternCollTreeUpDown) {
    // CollTree up
    NCCLCHECK(SaveProxy(channel, proxySend, channel->collTree.out, args, 1));  // For CollTree up, we are using push
    // CollTree down
    NCCLCHECK(SaveProxy(channel, proxyRecv, channel->collTree.out, args, 0));
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyComputeP2p(struct ncclInfo* info, struct ncclProxyArgs* args) {
  memset(args, 0, sizeof(struct ncclProxyArgs));
  int channelId = info->channelId;
  args->nsubs = 1;
  struct ncclProxySubArgs* sub = args->subs;

  struct ncclChannel* channel = info->comm->channels+channelId;
  sub->channelId = channelId;
  args->sliceSteps = 1;
  args->chunkSteps = 1;
  args->protocol = NCCL_PROTO_SIMPLE;
  args->dtype = info->datatype;
  sub->delta = info->delta;
  sub->recvbytes = info->recvbytes;
  sub->sendbytes = info->sendbytes;

  int stepSize = info->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/SENDRECV_SLICEFACTOR;
  info->recvChunkSize = stepSize;
  info->sendChunkSize = stepSize;

  if (info->delta > 0 && info->recvbytes >= 0) {
    int peerrecv = (info->comm->nRanks+info->comm->rank-info->delta)%info->comm->nRanks;
    if (channel->peers[peerrecv].recv[0].transportComm && channel->peers[peerrecv].recv[0].transportComm->proxyProgress) {
      // Tune chunk size for the network
      if (info->recvbytes < stepSize) info->recvChunkSize /= 4;
      else if (info->recvbytes < 8*stepSize) info->recvChunkSize /= 2;
    }
    sub->recvChunkSize = info->recvChunkSize;
  }
  if (info->delta > 0 && info->sendbytes >= 0) {
    int peersend = (info->comm->rank+info->delta)%info->comm->nRanks;
    if (channel->peers[peersend].send[0].transportComm && channel->peers[peersend].send[0].transportComm->proxyProgress) {
      // Tune chunk size for the network
      if (info->sendbytes < stepSize) info->sendChunkSize /= 4;
      else if (info->sendbytes < 8*stepSize) info->sendChunkSize /= 2;
    }
    sub->sendChunkSize = info->sendChunkSize;
  }
  return ncclSuccess;
}

ncclResult_t ncclProxySaveP2p(struct ncclComm* comm, struct ncclProxyArgs* args) {
  struct ncclProxySubArgs* sub = args->subs;
  struct ncclChannel* channel = comm->channels+sub->channelId;
  args->opCount = channel->workFifoTail-1;
  const ssize_t recvbytesOrig = sub->recvbytes;
  const ssize_t sendbytesOrig = sub->sendbytes;
  if (sub->delta > 0 && recvbytesOrig >= ssize_t(0)) {
    int peerrecv = (comm->nRanks+comm->rank-sub->delta)%comm->nRanks;
    sub->recvbytes = recvbytesOrig;
    sub->sendbytes = 0;
    sub->nsteps = DIVUP(sub->recvbytes, sub->recvChunkSize);
    if (sub->nsteps == 0) sub->nsteps = 1;
    NCCLCHECK(SaveProxy(channel, proxyRecv, peerrecv, args, 0));
  }
  if (sub->delta > 0 && sendbytesOrig >= ssize_t(0)) {
    int peersend = (comm->rank+sub->delta)%comm->nRanks;
    sub->sendbytes = sendbytesOrig;
    sub->recvbytes = 0;
    sub->nsteps = DIVUP(sub->sendbytes, sub->sendChunkSize);
    if (sub->nsteps == 0) sub->nsteps = 1;
    NCCLCHECK(SaveProxy(channel, proxySend, peersend, args, 0));
  }
  // Reset proxy args for potentially multiple cuda graph launches
  // It is safe as long as SaveProxy copies contents of args to op
  sub->recvbytes = recvbytesOrig;
  sub->sendbytes = sendbytesOrig;
  return ncclSuccess;
}

static ncclResult_t removeOp(struct ncclProxyProgressState* state, struct ncclProxyArgs** opPtr, struct ncclProxyArgs** prevOpPtr) {
  struct ncclProxyArgs* freeOp = *opPtr;
  DEBUG_PROXY_PRINT("Remove %ld -> %ld -> %ld\n", OP_INDEX(*prevOpPtr), OP_INDEX(freeOp), OP_INDEX(freeOp->next));
  struct ncclProxyArgs* next = freeOp->next;
  *opPtr = next;
  if (freeOp->nextPeer) {
    // replace op by nextPeer
    struct ncclProxyArgs* nextPeer = freeOp->nextPeer;
    if (*prevOpPtr) {
      (*prevOpPtr)->next = nextPeer;
    } else {
      state->ops = nextPeer;
    }
    nextPeer->next = next;
    *(prevOpPtr) = nextPeer;
  } else {
    *(freeOp->proxyAppendPtr) = NULL;
    if (*prevOpPtr) {
      (*prevOpPtr)->next = next;
    } else {
      state->ops = next;
    }
  }
  freeOp->next = state->poolFreed;
  state->poolFreed = freeOp;
  DEBUG_PROXY_PRINT("Removed %5ld (%5ld)                                               : ", OP_INDEX(freeOp), OP_INDEX(*freeOp->proxyAppendPtr));
  NCCLCHECK(dumpProxyState(state));
  return ncclSuccess;
}

static ncclResult_t progressOps(struct ncclComm* comm, struct ncclProxyProgressState* state, struct ncclProxyArgs** opsPtr, int* idle) {
  struct ncclProxyArgs* prevOp = NULL;
  struct ncclProxyArgs* op = *opsPtr;
  while (op) {
    if (op->state == ncclProxyOpNone) return ncclInternalError;
    NCCLCHECK(op->progress(comm, op));
    *idle &= op->idle;
    if (op->state == ncclProxyOpNone) {
      NCCLCHECK(removeOp(state, &op, &prevOp));
    } else {
      prevOp = op;
      op = op->next;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclProxyGetPostedOps(struct ncclProxyProgressState* state) {
  pthread_mutex_lock(&state->opsMutex);
  // Sort operations as we append them : collectives and
  // receives first, then sends.

  struct ncclProxyArgs* next, *prev = NULL, *op = state->postedOps;
  while (op) {
    next = op->next;
    if (op->subs[0].sendbytes) {
      if (prev) prev->next = next;
      else state->postedOps = next;
      op->next = NULL;
      NCCLCHECK(ProxyAppend(state, op));
    } else prev = op;
    op = next;
  }
  op = state->postedOps;
  while (op) {
    next = op->next;
    op->next = NULL;
    NCCLCHECK(ProxyAppend(state, op));
    op = next;
  }
  state->postedOps = op;
  if (op == NULL) state->postedOpsEnd = NULL;
  pthread_mutex_unlock(&state->opsMutex);
  return ncclSuccess;
}

ncclResult_t ncclProxyAppendPosted(struct ncclProxyProgressState* state) {
  // Return any freed element first
  if (state->poolFreed) {
    struct ncclProxyArgs* end = state->poolFreed;
    while (end->next) end = end->next;
    pthread_mutex_lock(&state->poolMutex);
    end->next = state->poolReturned;
    state->poolReturned = state->poolFreed;
    pthread_mutex_unlock(&state->poolMutex);
    state->poolFreed = NULL;
  }

  // Then wait until we have new work to do
  pthread_mutex_lock(&state->opsMutex);
  while (state->postedOps == NULL) {
    if (state->stop) return ncclSuccess;
    pthread_cond_wait(&state->cond, &state->opsMutex);
  }
  pthread_mutex_unlock(&state->opsMutex);

  NCCLCHECK(ncclProxyGetPostedOps(state));

  if (state->poolFreed) {
    struct ncclProxyArgs* end = state->poolFreed;
    while (end->next) end = end->next;
    pthread_mutex_lock(&state->poolMutex);
    end->next = state->poolReturned;
    state->poolReturned = state->poolFreed;
    pthread_mutex_unlock(&state->poolMutex);
    state->poolFreed = NULL;
  }

  return ncclSuccess;
}


void* ncclProxyProgress(void *comm_) {
  struct ncclComm* comm = (struct ncclComm*)comm_;
  struct ncclProxyProgressState* state = &comm->proxyState.progressState;
  char threadName[16];
  sprintf(threadName, "NCCLproxy %5d", comm->rank);
  nvtxNameOsThreadA(syscall(SYS_gettid), threadName);

  struct ncclProxyArgs** opsPtr = &state->ops;
  while (1) {
    if (*comm->abortFlag) {
      return NULL;
    }

    while (*opsPtr == NULL) {
      if (state->stop) {
        // No more commands to process and proxy has been requested to stop
        return NULL;
      }
      ncclResult_t ret = ncclProxyAppendPosted(state);
      if (ret != ncclSuccess) {
        comm->fatalError = ret;
        INFO(NCCL_ALL,"%s:%d -> %d [Proxy Thread]", __FILE__, __LINE__, ret);
        return NULL;
      }
    }
    int idle = 1;
    ncclResult_t ret = progressOps(comm, state, opsPtr, &idle);
    if (ret != ncclSuccess) {
      comm->fatalError = ret;
      INFO(NCCL_ALL,"%s:%d -> %d [Proxy Thread]", __FILE__, __LINE__, ret);
      return NULL;
    }
    if (idle) {
      if (state->postedOps) {
        ncclProxyGetPostedOps(state);
      }
      sched_yield(); // No request progressed. Let others run.
    }
  }
}

ncclResult_t ncclProxyStart(struct ncclComm* comm) {
  for (int r=0; r<comm->localRanks; r++) {
    if (comm->proxyState.peerSocks && comm->proxyState.peerSocks[r].fd) {
      int msg = ncclProxyMsgStart;
      NCCLCHECK(ncclSocketSend(comm->proxyState.peerSocks+r, &msg, sizeof(int)));
    }
  }
  comm->opCount++;
  return ncclSuccess;
}

ncclResult_t ncclProxySharedBuffersInitCollNet(struct ncclComm* comm, int cuda, int* size, char** ptr) {
  struct ncclProxySharedCollNet* state = &comm->proxyState.progressState.collNet;
  if (state->size == 0) {
    state->size = 2*comm->nChannels*comm->buffSizes[NCCL_PROTO_SIMPLE];
  }

  *size = state->size;

  if (cuda && state->cudaBuff == NULL) {
    NCCLCHECK(ncclCudaCalloc(&state->cudaBuff, *size));
  } else if (state->hostBuff == NULL) {
    NCCLCHECK(ncclCudaHostCalloc(&state->hostBuff, *size));
  }
  *ptr = cuda ? state->cudaBuff : state->hostBuff;
  return ncclSuccess;
}

ncclResult_t ncclProxySharedBuffersGetCollNet(struct ncclComm* comm, int type, int slot, int channel, int* offset) {
  // Use different pools for different channels and also separate send/recv.
  int slotSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  int globalSlot = (type*NCCL_STEPS+slot)*comm->nChannels+channel;
  *offset = slotSize * globalSlot;
  return ncclSuccess;
}

ncclResult_t ncclProxySharedBuffersDestroyCollNet(struct ncclComm* comm) {
  struct ncclProxySharedCollNet* state = &comm->proxyState.progressState.collNet;
  if (state->size == 0) return ncclSuccess;
  CUDACHECK(cudaFree(state->cudaBuff));
  NCCLCHECK(ncclCudaHostFree(state->hostBuff));
  return ncclSuccess;
}

ncclResult_t ncclProxyProgressCreate(struct ncclComm* comm) {
  struct ncclProxyProgressState* state = &comm->proxyState.progressState;
  if (!state->thread) {
    state->cond = PTHREAD_COND_INITIALIZER;
    state->opsMutex = PTHREAD_MUTEX_INITIALIZER;
    state->poolMutex = PTHREAD_MUTEX_INITIALIZER;
    state->ops = NULL;
    pthread_create(&state->thread, NULL, ncclProxyProgress, comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyProgressDestroy(struct ncclComm* comm) {
  struct ncclProxyProgressState* state = &comm->proxyState.progressState;

  // Request the proxy to stop and then wake it
  pthread_mutex_lock(&state->opsMutex);
  state->stop = true;
  pthread_cond_signal(&state->cond);
  pthread_mutex_unlock(&state->opsMutex);
  if (state->thread) pthread_join(state->thread, NULL);

  // Free off any memory allocated for the proxy arg pools
  pthread_mutex_lock(&state->poolMutex);
  while (state->pools != NULL) {
    struct ncclProxyPool *next = state->pools->next;
    free(state->pools);
    state->pools = next;
  }
  pthread_mutex_unlock(&state->poolMutex);

  NCCLCHECK(ncclProxySharedBuffersDestroyCollNet(comm));

  return ncclSuccess;
}

struct ncclProxyAsyncOp {
  int type;
  struct ncclProxyConnection* connection;
  int reqSize, respSize;
  char *reqBuff, *respBuff;
  struct ncclProxyAsyncOp* next;
};

struct ncclProxyLocalPeer {
  struct ncclSocket sock;
  struct ncclProxyAsyncOp asyncOps;
  struct ncclProxyArgs* nextOps;
  struct ncclProxyArgs* nextOpsEnd;
};

#define NCCL_PROXY_CONN_POOL_SIZE_POW2 7
#define NCCL_PROXY_CONN_POOL_SIZE (1<<(NCCL_PROXY_CONN_POOL_SIZE_POW2))
#define NCCL_PROXY_CONN_POOL_MASK ((NCCL_PROXY_CONN_POOL_SIZE)-1)
struct ncclProxyConnectionPool {
  struct ncclProxyConnection** pools;
  int banks;
  int offset;
  struct ncclProxyAsyncOp* ops;
};

static ncclResult_t ncclProxyNewConnection(struct ncclProxyConnectionPool* pool, int* id) {
  if (pool->offset == NCCL_PROXY_CONN_POOL_SIZE) {
    NCCLCHECK(ncclRealloc(&pool->pools, pool->banks, pool->banks+1));
    NCCLCHECK(ncclCalloc(pool->pools+pool->banks, NCCL_PROXY_CONN_POOL_SIZE));
    pool->banks++;
    pool->offset = 0;
  }
  *id = ((pool->banks-1) << NCCL_PROXY_CONN_POOL_SIZE_POW2) + pool->offset;
  pool->offset++;
  return ncclSuccess;
}

static ncclResult_t ncclProxyGetConnection(struct ncclProxyConnectionPool* pool, int id, struct ncclProxyConnection** conn) {
  int bank = id>>NCCL_PROXY_CONN_POOL_SIZE_POW2;
  int offset = id&NCCL_PROXY_CONN_POOL_MASK;
  if ((pool->pools == NULL) || (bank > pool->banks) || (pool->pools[bank] == NULL)) return ncclInternalError;
  *conn = pool->pools[bank]+offset;
  return ncclSuccess;
}

static ncclResult_t proxyFree(struct ncclProxyConnection* connection, struct ncclComm* comm) {
  if (connection->send) {
    NCCLCHECK(ncclTransports[connection->transport].send.proxyFree(connection, comm));
  } else {
    NCCLCHECK(ncclTransports[connection->transport].recv.proxyFree(connection, comm));
  }
  return ncclSuccess;
}

static ncclResult_t ncclProxyFreeConnections(struct ncclProxyConnectionPool* pool, struct ncclComm* comm) {
  for (int b=0; b<pool->banks; b++) {
    int max = b == pool->banks-1 ? pool->offset : NCCL_PROXY_CONN_POOL_SIZE;
    for (int i=0; i<max; i++) {
      NCCLCHECK(proxyFree(pool->pools[b]+i, comm));
    }
    free(pool->pools[b]);
  }
  free(pool->pools);
  return ncclSuccess;
}

#define MAX_LOCAL_PEERS 128
#include "transport.h"

ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int rank, struct ncclProxyConnector* proxyConn) {
  // Keep one connection per mlocal rank
  proxyConn->id = -1;
  if (comm->proxyState.peerSocks == NULL) {
    NCCLCHECK(ncclCalloc(&comm->proxyState.peerSocks, comm->localRanks));
    for (int r=0; r<comm->localRanks; r++) comm->proxyState.peerSocks[r].abortFlag = comm->abortFlag;
  }
  NCCLCHECK(ncclTopoGetLocalRank(comm->topo, rank, &proxyConn->localRank));
  struct ncclSocket* sock = comm->proxyState.peerSocks+proxyConn->localRank;
  if (sock->fd == 0) {
    memcpy(&sock->addr, comm->proxyState.peerAddresses+rank, sizeof(union ncclSocketAddress));
    NCCLCHECK(ncclSocketConnect(sock));
  }
  int type = ncclProxyMsgInit;
  NCCLCHECK(ncclSocketSend(sock, &type, sizeof(int)));
  NCCLCHECK(ncclSocketSend(sock, &transport, sizeof(int)));
  NCCLCHECK(ncclSocketSend(sock, &send, sizeof(int)));
  NCCLCHECK(ncclSocketRecv(sock, &proxyConn->id, sizeof(int)));
  INFO(NCCL_NET, "Connection to proxy localRank %d -> id %d", proxyConn->localRank, proxyConn->id);
  proxyConn->comm = comm;
  return ncclSuccess;
}

ncclResult_t ncclProxyCall(struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize) {
  if (proxyConn->comm->proxyState.peerSocks == NULL) return ncclInternalError;
  struct ncclSocket* sock = proxyConn->comm->proxyState.peerSocks+proxyConn->localRank;
  if (sock->fd == 0) return ncclInternalError;

  NCCLCHECK(ncclSocketSend(sock, &type, sizeof(int)));
  NCCLCHECK(ncclSocketSend(sock, &proxyConn->id, sizeof(int)));
  NCCLCHECK(ncclSocketSend(sock, &reqSize, sizeof(int)));
  NCCLCHECK(ncclSocketSend(sock, &respSize, sizeof(int)));
  if (reqSize) NCCLCHECK(ncclSocketSend(sock, reqBuff, reqSize));
  //INFO(NCCL_NET, "Proxy Call connection %d, type %d, req %d, resp %d", proxyConn->id, type, reqSize, respSize);
  if (respSize) NCCLCHECK(ncclSocketRecv(sock, respBuff, respSize));
  return ncclSuccess;
}

static ncclResult_t ncclProxyAppendAsyncOp(struct ncclProxyConnectionPool* connectionPool, struct ncclProxyAsyncOp* asyncOp) {
  asyncOp->next = connectionPool->ops;
  connectionPool->ops = asyncOp;
  return ncclSuccess;
}

static ncclResult_t proxyConnInit(struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool) {
  struct ncclSocket* sock = &peer->sock;
  char buf[SOCKET_NAME_MAXLEN+1];
  buf[SOCKET_NAME_MAXLEN] = '\0';
  int id;
  struct ncclProxyConnection* connection;
  NCCLCHECK(ncclProxyNewConnection(connectionPool, &id));
  NCCLCHECK(ncclProxyGetConnection(connectionPool, id, &connection));
  connection->sock = sock;
  NCCLCHECK(ncclSocketRecv(sock, &connection->transport, sizeof(int)));
  NCCLCHECK(ncclSocketRecv(sock, &connection->send, sizeof(int)));
  NCCLCHECK(ncclSocketSend(sock, &id, sizeof(int)));
  connection->tcomm = connection->send ? &ncclTransports[connection->transport].send : &ncclTransports[connection->transport].recv;
  buf[SOCKET_NAME_MAXLEN] = '\0';
  INFO(NCCL_NET, "New proxy %s connection %d from %s, transport %d", connection->send ? "send":"recv", id, ncclSocketToString(&sock->addr, buf), connection->transport);
  return ncclSuccess;
}

static ncclResult_t proxyProgressAsync(struct ncclProxyAsyncOp* op, struct ncclComm* comm) {
  int done = 1;
  if (op->type == ncclProxyMsgSetup) {
    NCCLCHECK(op->connection->tcomm->proxySetup(op->connection, comm, op->reqBuff, op->reqSize, op->respBuff, op->respSize, &done));
  } else if (op->type == ncclProxyMsgConnect) {
    NCCLCHECK(op->connection->tcomm->proxyConnect(op->connection, comm, op->reqBuff, op->reqSize, op->respBuff, op->respSize, &done));
  } else return ncclInternalError;
  if (done) {
    if (op->respSize) NCCLCHECK(ncclSocketSend(op->connection->sock, op->respBuff, op->respSize));
    if (op->reqBuff) free(op->reqBuff);
    if (op->respBuff) free(op->respBuff);
    op->reqBuff = NULL;
    op->respBuff = NULL;
    op->type = 0;
  }
  return ncclSuccess;
}

static ncclResult_t proxyConnSetupConnect(int type, struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool, struct ncclComm* comm) {
  struct ncclSocket* sock = &peer->sock;
  struct ncclProxyAsyncOp* asyncOp = &peer->asyncOps;
  int id;
  asyncOp->type = type;
  NCCLCHECK(ncclSocketRecv(sock, &id, sizeof(int)));
  NCCLCHECK(ncclProxyGetConnection(connectionPool, id, &asyncOp->connection));

  NCCLCHECK(ncclSocketRecv(sock, &asyncOp->reqSize, sizeof(int)));
  NCCLCHECK(ncclSocketRecv(sock, &asyncOp->respSize, sizeof(int)));
  if (asyncOp->reqSize) {
    NCCLCHECK(ncclCalloc(&asyncOp->reqBuff, asyncOp->reqSize));
    NCCLCHECK(ncclSocketRecv(sock, asyncOp->reqBuff, asyncOp->reqSize));
  }
  if (asyncOp->respSize) NCCLCHECK(ncclCalloc(&asyncOp->respBuff, asyncOp->respSize));
  NCCLCHECK(proxyProgressAsync(asyncOp, comm));
  return ncclSuccess;
}
static ncclResult_t proxyConnAppend(struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool, struct ncclComm* comm) {
  struct ncclSocket* sock = &peer->sock;
  int id;
  int reqSize = 0, respSize = 0;
  struct ncclProxyConnection* connection;
  struct ncclProxyArgs* op;
  NCCLCHECK(allocateArgs(comm, &op));
  NCCLCHECK(ncclSocketRecv(sock, &id, sizeof(int)));
  NCCLCHECK(ncclProxyGetConnection(connectionPool, id, &connection));
  NCCLCHECK(ncclSocketRecv(sock, &reqSize, sizeof(int)));
  NCCLCHECK(ncclSocketRecv(sock, &respSize, sizeof(int)));
  if (reqSize != sizeof(struct ncclProxyArgs)) return ncclInternalError;
  if (respSize != 0) return ncclInternalError;
  NCCLCHECK(ncclSocketRecv(sock, op, sizeof(struct ncclProxyArgs)));
  op->progress = connection->tcomm->proxyProgress;
  op->proxyAppendPtr = connection->proxyAppendPtr;
  op->subs[0].connection = connection;
  if (peer->nextOps == NULL) peer->nextOps = op;
  else peer->nextOpsEnd->next = op;
  peer->nextOpsEnd = op;
  return ncclSuccess;
}
static ncclResult_t proxyConnStart(struct ncclProxyLocalPeer* peer, struct ncclComm* comm) {
  struct ncclProxyProgressState* progressState = &comm->proxyState.progressState;
  NCCLCHECK(ncclProxyProgressCreate(comm));
  if (peer->nextOps == NULL) return ncclSuccess;
  pthread_mutex_lock(&progressState->opsMutex);
  if (progressState->postedOps) progressState->postedOpsEnd->next = peer->nextOps;
  else progressState->postedOps = peer->nextOps;
  progressState->postedOpsEnd = peer->nextOpsEnd;
  peer->nextOps = peer->nextOpsEnd = NULL;
  pthread_cond_signal(&progressState->cond);
  pthread_mutex_unlock(&progressState->opsMutex);
  return ncclSuccess;
}

#include <poll.h>

void* ncclProxyService(void* _args) {
  struct ncclComm* comm =  (struct ncclComm *) _args;
  if (cudaSetDevice(comm->cudaDev) != cudaSuccess) {
    WARN("[Proxy Service] Failed to set CUDA device %d", comm->cudaDev);
  }
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);

  // Prepare poll descriptor
  struct ncclProxyConnectionPool connectionPool;
  connectionPool.pools = NULL;
  connectionPool.banks = 0;
  connectionPool.offset = NCCL_PROXY_CONN_POOL_SIZE;

  struct pollfd pollfds[MAX_LOCAL_PEERS+1];
  struct ncclProxyLocalPeer peers[MAX_LOCAL_PEERS];
  for (int s=0; s<MAX_LOCAL_PEERS; s++) {
    peers[s].sock.fd = pollfds[s].fd = -1;
    peers[s].sock.abortFlag = NULL;
    peers[s].sock.asyncFlag = 0;
    pollfds[s].events = POLLHUP|POLLIN;
    peers[s].asyncOps.type = 0;
  }
  pollfds[MAX_LOCAL_PEERS].fd = comm->proxyState.listenSock->fd;
  pollfds[MAX_LOCAL_PEERS].events = POLLIN;

  int maxnpeers = 0;
  int npeers = 0;
  int stop = 0;
  while (stop == 0 || (stop == 1 && npeers > 0)) {
    if (int error = poll(pollfds, MAX_LOCAL_PEERS+1, 100/*ms*/) < 0) {
      WARN("[Proxy Service] Poll failed with error %d", error);
      return NULL;
    }
    if (pollfds[MAX_LOCAL_PEERS].revents) {
      int s = 0;
      while (s < MAX_LOCAL_PEERS && peers[s].sock.fd != -1) s++;
      if (s == MAX_LOCAL_PEERS) {
        WARN("[Proxy service] Too many connections (%d max)", MAX_LOCAL_PEERS);
        return NULL;
      }
      if (maxnpeers < s+1) maxnpeers = s+1;
      struct ncclSocket* sock = &peers[s].sock;
      if (ncclSocketAccept(sock, comm->proxyState.listenSock) != ncclSuccess) {
        WARN("[Service thread] Accept failed %s", strerror(errno));
      } else {
        pollfds[s].fd = sock->fd;
        npeers++;
      }
    }
    for (int s=0; s<maxnpeers; s++) {
      struct ncclSocket* sock = &peers[s].sock;
      struct ncclProxyAsyncOp* op = &peers[s].asyncOps;
      if (op->type != 0) {
        if (proxyProgressAsync(op, comm) != ncclSuccess) {
          WARN("[Proxy Service] Call to Setup/Connect failed");
          close(sock->fd);
          sock->fd = pollfds[s].fd = -1;
          npeers--;
          op->type = 0;
        }
      } else if (pollfds[s].revents & POLLIN) {
        int type;
        if (ncclSocketRecv(sock, &type, sizeof(int)) != ncclSuccess) {
          WARN("[Service thread] Recv failed");
          close(sock->fd);
          sock->fd = pollfds[s].fd = -1;
        } else {
          ncclResult_t res = ncclSuccess;
          if (type == ncclProxyMsgAbort) {
            stop = 2;
          } else if (type == ncclProxyMsgStop) {
            stop = 1;
          } else if (type == ncclProxyMsgClose) {
            close(sock->fd);
            sock->fd = pollfds[s].fd = -1;
            npeers--;
          } else if (type == ncclProxyMsgInit) {
            res = proxyConnInit(peers+s, &connectionPool);
          } else if (type == ncclProxyMsgSetup || type == ncclProxyMsgConnect) {
            res = proxyConnSetupConnect(type, peers+s, &connectionPool, comm);
          } else if (type == ncclProxyMsgAppend) {
            res = proxyConnAppend(peers+s, &connectionPool, comm);
          } else if (type == ncclProxyMsgStart) {
            res = proxyConnStart(peers+s, comm);
          }
          if (res != ncclSuccess) {
            WARN("[Proxy Service] Failed to process message of type %d", type);
            close(sock->fd);
            sock->fd = pollfds[s].fd = -1;
            npeers--;
          }
        }
      } else if (pollfds[s].revents & POLLHUP) {
        close(sock->fd);
        sock->fd = pollfds[s].fd = -1;
        npeers--;
      }
    }
  }
  for (int s=0; s<maxnpeers; s++) {
    if (peers[s].sock.fd != -1) close(peers[s].sock.fd);
  }
  ncclProxyFreeConnections(&connectionPool, comm);
  close(comm->proxyState.listenSock->fd);
  free(comm->proxyState.listenSock);
  if (ncclProxyProgressDestroy(comm) != ncclSuccess) {
    WARN("[Proxy Service] proxyDestroy failed");
  }
  return NULL;
}

ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses) {
  comm->proxyState.listenSock = sock;
  comm->proxyState.peerAddresses = peerAddresses;
  pthread_create(&comm->proxyState.thread, NULL, ncclProxyService, comm);
  return ncclSuccess;
}

ncclResult_t ncclProxyDestroy(struct ncclComm* comm) {
  struct ncclProxyState* state = &comm->proxyState;
  if (state->peerAddresses) {
    struct ncclSocket sock;
    sock.abortFlag = NULL;
    sock.asyncFlag = 0;
    memcpy(&sock.addr, comm->proxyState.peerAddresses+comm->rank, sizeof(union ncclSocketAddress));
    NCCLCHECK(ncclSocketConnect(&sock));
    int type = comm->abortFlag ? ncclProxyMsgAbort : ncclProxyMsgStop;
    NCCLCHECK(ncclSocketSend(&sock, &type, sizeof(int)));
    close(sock.fd);
    free(state->peerAddresses);
  }
  if (state->peerSocks) {
    for (int i=0; i<comm->localRanks; i++) {
      if (state->peerSocks[i].fd) {
        int type = ncclProxyMsgClose;
        NCCLCHECK(ncclSocketSend(state->peerSocks+i, &type, sizeof(int)));
        close(state->peerSocks[i].fd);
      }
    }
    free(state->peerSocks);
  }
  void* ret;
  pthread_join(state->thread, &ret);
  return ncclSuccess;
}
