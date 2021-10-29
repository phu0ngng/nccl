/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "collectives.h"
#include "socket.h"
#include "shm.h"

#include <sys/time.h>
#include <x86intrin.h>
double gettime() {
  static double freq = -1;
  if (freq == -1) {
    printf("Calibrating clock, please wait ...");
    fflush(stdout);
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t timeCycles = __rdtsc();
    double time = - tv.tv_sec*1E6 - tv.tv_usec;
    uint64_t total = 0ULL;
    for (int i=0; i<1000; i++) { total += __rdtsc(); if (i%100 == 0) printf("."); }
    gettimeofday(&tv, NULL);
    timeCycles = __rdtsc() - timeCycles;
    time += tv.tv_sec*1E6 + tv.tv_usec;
    freq = timeCycles/time;
    printf("Time %g, rdtsc delta %ld, freq %g cycles/usec\n", time, timeCycles, freq);
  }
  return __rdtsc()/freq;
}
double proxyStartTime = 0;
uint64_t proxyStartCount = 0;
double proxyAppendTime = 0;
uint64_t proxyAppendCount = 0;

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

void ncclInterProcessLock(volatile int* lock) {
  while (!__sync_bool_compare_and_swap(lock, 0, 1)) sched_yield();
}

void ncclInterProcessUnlock(volatile int* lock) {
  __sync_synchronize();
  *lock = 0;
}

ncclResult_t ncclLocalOpAppend(struct ncclProxyConnector* proxyConn, struct ncclProxyOp* proxyOp) {
  struct ncclProxyOpsPool** pools = proxyConn->comm->proxyState.opsPools;
  if (pools == NULL) return ncclInternalError;
  // Allocate pool if needed
  if (pools[proxyConn->localRank] == NULL) {
    char poolPath[] = "/dev/shm/nccl-XXXXXX";
    NCCLCHECK(ncclProxyCall(proxyConn, ncclProxyMsgOpsAlloc, NULL, 0, poolPath+sizeof("/dev/shm/nccl-")-1, sizeof("XXXXXX")-1));
    NCCLCHECK(ncclShmOpen(poolPath, sizeof(struct ncclProxyOpsPool), (void**)pools+proxyConn->localRank, NULL, 0));
    NCCLCHECK(ncclShmUnlink(poolPath));
  }
  struct ncclProxyOpsPool* pool = pools[proxyConn->localRank];

  ncclInterProcessLock(&pool->lock);
  if (pool->freeOps == -1) {
    ncclInterProcessUnlock(&pool->lock);
    WARN("Error: local op alloc failed.");
    return ncclInternalError;
  }
  int opIndex = pool->freeOps;
  struct ncclProxyOp* op = pool->ops+opIndex;
  pool->freeOps = op->next;
  memcpy(op, proxyOp, sizeof(struct ncclProxyOp));
  op->next = -1;
  op->connection = proxyConn->connection;
  if (pool->nextOps == -1) {
    pool->nextOps = pool->nextOpsEnd = opIndex;
  } else {
    pool->ops[pool->nextOpsEnd].next = opIndex;
    pool->nextOpsEnd = opIndex;
  }
  ncclInterProcessUnlock(&pool->lock);
  return ncclSuccess;
}

static ncclResult_t SaveProxy(struct ncclChannel* channel, int type, int peer, struct ncclProxyOp* op, int connIndex) {
  if (peer < 0) return ncclSuccess;

  struct ncclPeer* peerComm = channel->peers+peer;
  struct ncclConnector* connector = type == proxyRecv ? peerComm->recv+connIndex : peerComm->send+connIndex;
  if (connector->transportComm == NULL) {
    WARN("Rank %d has no transport for %s peer %d on channel %d", connector->comm->rank,
        type == proxyRecv ? "recv" : "send", peer, channel->id);
    return ncclInternalError;
  }
  if (connector->transportComm->proxyProgress == NULL) return ncclSuccess;

  proxyAppendCount++;
  proxyAppendTime -= gettime();
  NCCLCHECK(ncclLocalOpAppend(&connector->proxyConn, op));
  proxyAppendTime += gettime();
  return ncclSuccess;
}

ncclResult_t ncclProxySaveColl(struct ncclComm* comm, struct ncclProxyOp* op, int nranks) {
  struct ncclChannel* channel = comm->channels+op->channelId;
  int pattern = op->pattern;
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice || pattern == ncclPatternPipelineFrom || pattern == ncclPatternPipelineTo) {
    struct ncclRing* ring = &channel->ring;
    if (NeedProxy(proxyRecv, pattern, op->root, ring, nranks)) NCCLCHECK(SaveProxy(channel, proxyRecv, ring->prev, op, 0));
    if (NeedProxy(proxySend, pattern, op->root, ring, nranks)) NCCLCHECK(SaveProxy(channel, proxySend, ring->next, op, 0));
  }
  if (pattern == ncclPatternTreeUp || pattern == ncclPatternTreeUpDown) {
    // Tree up
    struct ncclTree* tree = &channel->tree;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(channel, proxyRecv, tree->down[i], op, 0));
    NCCLCHECK(SaveProxy(channel, proxySend, tree->up, op, 0));
  }
  if (pattern == ncclPatternTreeDown || pattern == ncclPatternTreeUpDown) {
    // Tree down
    struct ncclTree* tree = &channel->tree;
    for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(channel, proxySend, tree->down[i], op, 0));
    NCCLCHECK(SaveProxy(channel, proxyRecv, tree->up, op, 0));
  }
  if (pattern == ncclPatternCollTreeUpDown) {
    // CollTree up
    NCCLCHECK(SaveProxy(channel, proxySend, channel->collTree.out, op, 1));  // For CollTree up, we are using push
    // CollTree down
    NCCLCHECK(SaveProxy(channel, proxyRecv, channel->collTree.out, op, 0));
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyComputeP2p(struct ncclInfo* info, struct ncclProxyOp* op) {
  memset(op, 0, sizeof(struct ncclProxyOp));
  int channelId = info->channelId;
  struct ncclChannel* channel = info->comm->channels+channelId;
  op->channelId = channelId;
  op->sliceSteps = 1;
  op->chunkSteps = 1;
  op->protocol = NCCL_PROTO_SIMPLE;
  op->dtype = info->datatype;

  int stepSize = info->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/SENDRECV_SLICEFACTOR;
  info->chunkSize = stepSize;
  op->root = info->root;
  op->nbytes = info->count;
  struct ncclPeer* peer = channel->peers + op->root;

  if (info->coll == ncclFuncSend) {
    op->pattern = ncclPatternSend;
    if (op->root != info->comm->rank && peer->send[0].transportComm && peer->send[0].transportComm->proxyProgress) {
      // Tune chunk size for the network
      if (info->count < stepSize) info->chunkSize /= 4;
      else if (info->count < 8*stepSize) info->chunkSize /= 2;
    }
  } else if (info->coll == ncclFuncRecv) {
    op->pattern = ncclPatternRecv;
    if (op->root != info->comm->rank && peer->recv[0].transportComm && peer->recv[0].transportComm->proxyProgress) {
      // Tune chunk size for the network
      if (info->count < stepSize) info->chunkSize /= 4;
      else if (info->count < 8*stepSize) info->chunkSize /= 2;
    }
  } else {
    WARN("P2p operation is neither send or recv");
    return ncclInternalError;
  }
  op->chunkSize = info->chunkSize;
  return ncclSuccess;
}

ncclResult_t ncclProxySaveP2p(struct ncclComm* comm, struct ncclProxyOp* op) {
  struct ncclChannel* channel = comm->channels+op->channelId;
  op->opCount = channel->workFifoTail-1;
  if (op->root == comm->rank) return ncclSuccess;
  if (op->pattern == ncclPatternRecv) {
    op->nsteps = DIVUP(op->nbytes, op->chunkSize);
    if (op->nsteps == 0) op->nsteps = 1;
    NCCLCHECK(SaveProxy(channel, proxyRecv, op->root, op, 0));
  } else if (op->pattern == ncclPatternSend) {
    op->nsteps = DIVUP(op->nbytes, op->chunkSize);
    if (op->nsteps == 0) op->nsteps = 1;
    NCCLCHECK(SaveProxy(channel, proxySend, op->root, op, 0));
  }
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
  // sends first, then receives

  struct ncclProxyArgs* next, *prev = NULL, *op = state->postedOps;
  while (op) {
    next = op->next;
    if (op->pattern == ncclPatternSend) {
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
  char threadName[NCCL_THREAD_NAMELEN];
  snprintf(threadName, NCCL_THREAD_NAMELEN, "NCCL Progress%2d", comm->cudaDev);
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
      proxyStartCount++;
      proxyStartTime -= gettime();
      int msg = ncclProxyMsgStart;
      NCCLCHECK(ncclSocketSend(comm->proxyState.peerSocks+r, &msg, sizeof(int)));
      proxyStartTime += gettime();
    }
  }
  comm->opCount++;
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
    ncclSetThreadName(state->thread, "NCCL Progress%2d", comm->cudaDev);
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
  struct ncclProxyOpsPool* pool;
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
  proxyConn->connection = NULL;
  proxyConn->rank = rank;
  if (comm->proxyState.peerSocks == NULL) {
    NCCLCHECK(ncclCalloc(&comm->proxyState.peerSocks, comm->localRanks));
    NCCLCHECK(ncclCalloc(&comm->proxyState.opsPools, comm->localRanks));
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
  NCCLCHECK(ncclSocketRecv(sock, &proxyConn->connection, sizeof(void*)));
  INFO(NCCL_NET, "Connection to proxy localRank %d -> connection %p", proxyConn->localRank, proxyConn->connection);
  proxyConn->comm = comm;
  return ncclSuccess;
}

ncclResult_t ncclProxyCall(struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize) {
  if (proxyConn->comm->proxyState.peerSocks == NULL) return ncclInternalError;
  struct ncclSocket* sock = proxyConn->comm->proxyState.peerSocks+proxyConn->localRank;
  if (sock->fd == 0) return ncclInternalError;

  NCCLCHECK(ncclSocketSend(sock, &type, sizeof(int)));
  NCCLCHECK(ncclSocketSend(sock, &proxyConn->connection, sizeof(void*)));
  NCCLCHECK(ncclSocketSend(sock, &reqSize, sizeof(int)));
  NCCLCHECK(ncclSocketSend(sock, &respSize, sizeof(int)));
  if (reqSize) NCCLCHECK(ncclSocketSend(sock, reqBuff, reqSize));
  //INFO(NCCL_NET, "Proxy Call connection %p, type %d, req %d, resp %d", proxyConn->connection, type, reqSize, respSize);
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
  NCCLCHECK(ncclSocketSend(sock, &connection, sizeof(void*)));
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
  asyncOp->type = type;
  NCCLCHECK(ncclSocketRecv(sock, &asyncOp->connection, sizeof(void*)));

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
static ncclResult_t proxyOpsAlloc(struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool, struct ncclComm* comm) {
  struct ncclSocket* sock = &peer->sock;
  struct ncclProxyConnection* connection;
  NCCLCHECK(ncclSocketRecv(sock, &connection, sizeof(void*)));

  int reqSize, respSize;
  NCCLCHECK(ncclSocketRecv(sock, &reqSize, sizeof(int)));
  NCCLCHECK(ncclSocketRecv(sock, &respSize, sizeof(int)));
  if (reqSize) return ncclInternalError;

  int size = sizeof(struct ncclProxyOpsPool);
  struct ncclProxyOpsPool* pool = NULL;

  char shmPath[sizeof("/dev/shm/nccl-XXXXXX")];
  shmPath[0] = '\0';
  NCCLCHECK(ncclShmOpen(shmPath, size, (void**)&pool, NULL, 1));

  // Init pool
  pool->nextOps = -1;
  pool->freeOps = 0;
  for (int i=0; i<MAXCHANNELS*NCCL_MAX_OPS-1; i++) pool->ops[i].next = i+1;
  pool->ops[MAXCHANNELS*NCCL_MAX_OPS-1].next = -1;
  pool->lock = 0;
  peer->pool = pool;

  if (respSize != sizeof("XXXXXX")-1) return ncclInternalError;
  NCCLCHECK(ncclSocketSend(sock, shmPath+sizeof("/dev/shm/nccl-")-1, sizeof("XXXXXX")-1));
  return ncclSuccess;
}

static ncclResult_t ncclProxyOpToArgs(struct ncclProxyOp* op, struct ncclProxyArgs* args) {
  memset(args, 0, sizeof(struct ncclProxyArgs));
  struct ncclProxySubArgs* sub = args->subs;
  sub->channelId = op->channelId;
  sub->connection = op->connection;
  sub->nsteps = op->nsteps;
  sub->nbytes = op->nbytes;
  sub->peer = op->root;
  args->nsubs = 1;
  args->done = 0;
  args->sliceSteps = op->sliceSteps;
  args->chunkSteps = op->chunkSteps;
  args->chunkSize = op->chunkSize;
  args->opCount = op->opCount;
  args->protocol = op->protocol;
  args->dtype = op->dtype;
  args->redOp = op->redOp;
  args->pattern = op->pattern;
  args->state = ncclProxyOpReady;
  args->progress = op->connection->tcomm->proxyProgress;
  args->proxyAppendPtr = op->connection->proxyAppendPtr;
  return ncclSuccess;
}

static ncclResult_t proxyConnStart(struct ncclProxyLocalPeer* peer, struct ncclComm* comm) {
  struct ncclProxyProgressState* progressState = &comm->proxyState.progressState;
  NCCLCHECK(ncclProxyProgressCreate(comm));
  if (peer->pool == NULL) return ncclSuccess;

  // Extract peer ops list
  ncclInterProcessLock(&peer->pool->lock);
  if (peer->pool->nextOps == -1) {
    ncclInterProcessUnlock(&peer->pool->lock);
    return ncclSuccess;
  }
#if 0
  // Check for lost elements
  int index = peer->pool->nextOps;
  do {
    struct ncclProxyOp* op = peer->pool->ops+index;
    op->protocol |= 0x100;
    index = op->next;
  } while (index != -1);
  index = peer->pool->freeOps;
  do {
    struct ncclProxyOp* op = peer->pool->ops+index;
    op->protocol |= 0x100;
    index = op->next;
  } while (index != -1);
  for (index=0; index<MAXCHANNELS*NCCL_MAX_OPS; index++) {
    if ((peer->pool->ops[index].protocol & 0x100) == 0) {
      WARN("Elem %d lost\n", index);
      return ncclInternalError;
    }
    peer->pool->ops[index].protocol &= 0xff;
  }
#endif
  int peerOpStart = peer->pool->nextOps, peerOpEnd = peer->pool->nextOpsEnd;
  peer->pool->nextOps = peer->pool->nextOpsEnd = -1;
  ncclInterProcessUnlock(&peer->pool->lock);

  // Copy peer ops list to local ops pool.
  struct ncclProxyArgs* nextArgs = NULL, *nextArgsEnd = NULL;
  struct ncclProxyOp* peerOp;
  for (int opIndex = peerOpStart; opIndex != -1; opIndex = peerOp->next) {
    peerOp = peer->pool->ops+opIndex;
    if (peerOp->connection == NULL) { WARN("Peer op %d has NULL connection", opIndex); return ncclInternalError; }
    struct ncclProxyArgs* args;
    NCCLCHECK(allocateArgs(comm, &args));
    NCCLCHECK(ncclProxyOpToArgs(peerOp, args));
    if (nextArgs == NULL) {
      nextArgs = nextArgsEnd = args;
    } else {
      nextArgsEnd->next = args;
      nextArgsEnd = args;
    }
  }

  // Post local ops to progress thread
  pthread_mutex_lock(&progressState->opsMutex);
  if (progressState->postedOps) progressState->postedOpsEnd->next = nextArgs;
  else progressState->postedOps = nextArgs;
  progressState->postedOpsEnd = nextArgsEnd;
  pthread_cond_signal(&progressState->cond);
  pthread_mutex_unlock(&progressState->opsMutex);

  // Return peer ops list to peer pool
  ncclInterProcessLock(&peer->pool->lock);
  peer->pool->ops[peerOpEnd].next = peer->pool->freeOps;
  peer->pool->freeOps = peerOpStart;
  ncclInterProcessUnlock(&peer->pool->lock);
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
          if (peers[s].pool) {
            if (ncclShmClose(peers[s].pool, NULL, sizeof(struct ncclProxyOpsPool)) != ncclSuccess) {
              WARN("[Service thread] shm close Failed");
            }
          }
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
          } else if (type == ncclProxyMsgOpsAlloc) {
            res = proxyOpsAlloc(peers+s, &connectionPool, comm);
          } else if (type == ncclProxyMsgStart) {
            res = proxyConnStart(peers+s, comm);
          }
          if (res != ncclSuccess) {
            WARN("[Proxy Service] Failed to process message of type %d, retcode %d", type, res);
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
  printf("ProxyAppend Count %ld, %g us/call, ProxyStart Count %ld, %g us/call\n", proxyAppendCount, proxyAppendTime/proxyAppendCount, proxyStartCount, proxyStartTime/proxyStartCount);
  return NULL;
}

ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses) {
  comm->proxyState.listenSock = sock;
  comm->proxyState.peerAddresses = peerAddresses;
  pthread_create(&comm->proxyState.thread, NULL, ncclProxyService, comm);
  ncclSetThreadName(comm->proxyState.thread, "NCCL Service %2d", comm->cudaDev);
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
        if (state->opsPools[i]) {
          NCCLCHECK(ncclShmClose(state->opsPools[i], NULL, sizeof(struct ncclProxyOpsPool)));
        }
        int type = ncclProxyMsgClose;
        NCCLCHECK(ncclSocketSend(state->peerSocks+i, &type, sizeof(int)));
        close(state->peerSocks[i].fd);
      }
    }
    free(state->peerSocks);
    free(state->opsPools);
  }
  void* ret;
  pthread_join(state->thread, &ret);
  return ncclSuccess;
}
