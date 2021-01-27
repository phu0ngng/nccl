/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "graph.h"
#include "collectives.h"
#include <assert.h>

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
  struct ncclProxyState* state = &comm->proxyState;
  struct ncclProxyArgs* elem;
  pthread_mutex_lock(&state->poolMutex);
  if (state->pool == NULL) {
    // Allocate a new pool of elements
    struct ncclProxyPool* newPool;
    NCCLCHECK(ncclCalloc(&newPool, 1));
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
  elem = state->pool;
  state->pool = state->pool->next;
  pthread_mutex_unlock(&state->poolMutex);
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
ncclResult_t dumpProxyState(struct ncclProxyState* state) {
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

static ncclResult_t ProxyAppend(struct ncclProxyState* state, struct ncclProxyArgs* args) {
  struct ncclProxyArgs* proxyAppend = *args->proxyAppendPtr;
  int shared = args->subs[0].connector->conn.shared;
  if (proxyAppend) {
    if (shared && proxyAppend->opCount == args->opCount) {
      assert(proxyAppend->sliceSteps == args->sliceSteps);
      assert(proxyAppend->chunkSteps == args->chunkSteps);
      assert(proxyAppend->protocol == args->protocol);
      assert(proxyAppend->dtype == args->dtype);
      assert(proxyAppend->redOp == args->redOp);
      memcpy(proxyAppend->subs+proxyAppend->nsubs, args->subs, sizeof(struct ncclProxySubArgs));
      proxyAppend->nsubs++;
      args->next = proxyAppend->next;
      // Free args as we merged them
      args->next = state->pool;
      state->pool = args;
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

static ncclResult_t SaveProxy(int type, int peer, struct ncclProxyArgs* args) {
  if (peer < 0) return ncclSuccess;

  struct ncclChannel* channel = args->subs[0].channel;
  struct ncclPeer* peerComm = channel->peers+peer;
  struct ncclConnector* connector = type == proxyRecv ? &peerComm->recv : &peerComm->send;
  if (connector->transportComm == NULL) {
    WARN("[%d] Error no transport for %s peer %d on channel %d", connector->comm->rank,
        type == proxyRecv ? "recv" : "send", peer, channel->id);
    return ncclInternalError;
  }
  if (connector->transportComm->proxy == NULL) return ncclSuccess;

  struct ncclProxyState* state = &connector->comm->proxyState;
  struct ncclProxyArgs* op;
  NCCLCHECK(allocateArgs(connector->comm, &op));
  memcpy(op, args, sizeof(struct ncclProxyArgs));
  op->subs[0].connector = connector;
  op->progress = connector->transportComm->proxy;
  op->state = ncclProxyOpReady;
  op->proxyAppendPtr = connector->proxyAppendPtr;

  if (state->nextOps == NULL) state->nextOps = op;
  else state->nextOpsEnd->next = op;
  state->nextOpsEnd = op;
  return ncclSuccess;
}

ncclResult_t ncclProxySaveColl(struct ncclProxyArgs* args, int pattern, int root, int nranks) {
  struct ncclChannel* channel = args->subs[0].channel;
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice || pattern == ncclPatternPipelineFrom || pattern == ncclPatternPipelineTo) {
    struct ncclRing* ring = &channel->ring;
    if (NeedProxy(proxyRecv, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy(proxyRecv, ring->prev, args));
    if (NeedProxy(proxySend, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy(proxySend, ring->next, args));
  }
  if (pattern == ncclPatternTreeUp || pattern == ncclPatternTreeUpDown) {
    // Tree up
    struct ncclTree* tree = &channel->tree;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(proxyRecv, tree->down[i], args));
    NCCLCHECK(SaveProxy(proxySend, tree->up, args));
  }
  if (pattern == ncclPatternTreeDown || pattern == ncclPatternTreeUpDown) {
    // Tree down
    struct ncclTree* tree = &channel->tree;
    for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(proxySend, tree->down[i], args));
    NCCLCHECK(SaveProxy(proxyRecv, tree->up, args));
  }
  if (pattern == ncclPatternCollTreeUp) {
    // CollTree up
    NCCLCHECK(SaveProxy(proxySend, channel->collTree.out, args));
  }
  if (pattern == ncclPatternCollTreeDown) {
    // CollTree down
    NCCLCHECK(SaveProxy(proxyRecv, channel->collTree.out, args));
  }
  return ncclSuccess;
}

ncclResult_t ncclProxySaveP2p(struct ncclInfo* info, struct ncclChannel* channel) {
  struct ncclProxyArgs args;
  memset(&args, 0, sizeof(struct ncclProxyArgs));
  args.nsubs = 1;
  struct ncclProxySubArgs* sub = args.subs;
  sub->channel = channel;
  args.sliceSteps = 1;
  args.chunkSteps = 1;
  args.protocol = NCCL_PROTO_SIMPLE;
  args.opCount = channel->workFifoTail-1;
  args.dtype = info->datatype;
  if (info->delta > 0 && info->recvbytes >= 0) {
    int peerrecv = (info->comm->nRanks+info->comm->rank-info->delta)%info->comm->nRanks;
    sub->nsteps = DIVUP(info->recvbytes, info->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/SENDRECV_SLICEFACTOR);
    if (sub->nsteps == 0) sub->nsteps = 1;
    sub->recvbytes = info->recvbytes;
    sub->sendbytes = 0;
    NCCLCHECK(SaveProxy(proxyRecv, peerrecv, &args));
  }
  if (info->delta > 0 && info->sendbytes >= 0) {
    int peersend = (info->comm->rank+info->delta)%info->comm->nRanks;
    sub->nsteps = DIVUP(info->sendbytes, info->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/SENDRECV_SLICEFACTOR);
    if (sub->nsteps == 0) sub->nsteps = 1;
    sub->sendbytes = info->sendbytes;
    sub->recvbytes = 0;
    NCCLCHECK(SaveProxy(proxySend, peersend, &args));
  }
  return ncclSuccess;
}

static ncclResult_t removeOp(struct ncclProxyState* state, struct ncclProxyArgs** opPtr, struct ncclProxyArgs** prevOpPtr) {
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
  pthread_mutex_lock(&state->poolMutex);
  freeOp->next = state->pool;
  state->pool = freeOp;
  pthread_mutex_unlock(&state->poolMutex);
  DEBUG_PROXY_PRINT("Removed %5ld (%5ld)                                               : ", OP_INDEX(freeOp), OP_INDEX(*freeOp->proxyAppendPtr));
  NCCLCHECK(dumpProxyState(state));
  return ncclSuccess;
}

static ncclResult_t progressOps(struct ncclProxyState* state, struct ncclProxyArgs** opsPtr, int* idle, struct ncclComm* comm) {
  struct ncclProxyArgs* prevOp = NULL;
  struct ncclProxyArgs* op = *opsPtr;
  while (op) {
    if (op->state == ncclProxyOpNone) return ncclInternalError;
    // opCount >= lastOpCount are part of an ongoing GroupStart/GroupEnd that hasn't started
    // yet and might be cancelled before they even start. Hold on on those.
    if (op->opCount < comm->lastOpCount) {
      NCCLCHECK(op->progress(op));
      *idle &= op->idle;
    }
    if (op->state == ncclProxyOpNone) {
      NCCLCHECK(removeOp(state, &op, &prevOp));
    } else {
      prevOp = op;
      op = op->next;
    }
  }
  return ncclSuccess;
}

void* persistentThread(void *comm_) {
  struct ncclComm* comm = (struct ncclComm*)comm_;
  struct ncclProxyState* state = &comm->proxyState;
  char threadName[16];
  sprintf(threadName, "NCCLproxy %5d", comm->rank);
  nvtxNameOsThreadA(syscall(SYS_gettid), threadName);

  pthread_mutex_lock(&state->opsMutex);
  struct ncclProxyArgs** opsPtr = &state->ops;
  while (1) {
    if (*comm->abortFlag) {
      pthread_mutex_unlock(&state->opsMutex);
      return NULL;
    }

    while (*opsPtr == NULL) {
      if (state->stop) {
        // No more commands to process and proxy has been requested to stop
        pthread_mutex_unlock(&state->opsMutex);
        return NULL;
      }
      pthread_cond_wait(&state->cond, &state->opsMutex);
    }
    int idle = 1;
    ncclResult_t ret = progressOps(state, opsPtr, &idle, comm);
    if (ret != ncclSuccess) {
      comm->fatalError = ret;
      INFO(NCCL_ALL,"%s:%d -> %d [Proxy Thread]", __FILE__, __LINE__, ret);
      pthread_mutex_unlock(&state->opsMutex);
      return NULL;
    }
    if (idle) {
      pthread_mutex_unlock(&state->opsMutex);
      sched_yield(); // No request progressed. Let others run.
      pthread_mutex_lock(&state->opsMutex);
    }
  }
}

ncclResult_t ncclProxyStart(struct ncclComm* comm) {
  struct ncclProxyState* state = &comm->proxyState;
  pthread_mutex_lock(&state->opsMutex);

  // Sort operations as we append them : collectives and
  // receives first, then sends.
  ncclProxyArgs* next, *prev = NULL, *op = state->nextOps;
  while (op) {
    next = op->next;
    if (op->subs[0].sendbytes) {
      if (prev) prev->next = next;
      else state->nextOps = next;
      op->next = NULL;
      NCCLCHECK(ProxyAppend(state, op));
    } else prev = op;
    op = next;
  }
  op = state->nextOps;
  while (op) {
    next = op->next;
    op->next = NULL;
    NCCLCHECK(ProxyAppend(state, op));
    op = next;
  }
  state->nextOps = state->nextOpsEnd = NULL;
  NCCLCHECK(dumpProxyState(state));

  if (state->ops != NULL)
    pthread_cond_signal(&state->cond);
  pthread_mutex_unlock(&state->opsMutex);
  return ncclSuccess;
}

ncclResult_t ncclProxySharedBuffersInit(struct ncclComm* comm, int cuda, int* size, char** ptr) {
  struct ncclProxySharedBuffers* state = &comm->proxyState.sharedBuffs;
  if (state->size == 0) {
    int p2pnChannels = 1;
    while (p2pnChannels < comm->nChannels) p2pnChannels *= 2;
    int p2pSize = 2*p2pnChannels*NCCL_MAX_WORK_ELEMENTS*comm->buffSizes[NCCL_PROTO_SIMPLE]/SENDRECV_SLICEFACTOR;
    int collNetSize = 2*comm->nChannels*comm->buffSizes[NCCL_PROTO_SIMPLE];
    state->size = std::max(p2pSize, collNetSize);
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

ncclResult_t ncclProxySharedBuffersGetP2p(struct ncclComm* comm, int cuda, int type, int channel, int slot, int index, char** ptr) {
  struct ncclProxySharedBuffers* state = &comm->proxyState.sharedBuffs;
  // Use different pools for different channels and also separate send/recv.
  char* buff = cuda ? state->cudaBuff : state->hostBuff;
  int slotSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/(NCCL_STEPS*SENDRECV_SLICEFACTOR);
  int globalSlot = (((type*comm->p2pnChannels+channel)*NCCL_STEPS)+slot)*NCCL_MAX_WORK_ELEMENTS+index;
  *ptr = buff + slotSize * globalSlot;
  return ncclSuccess;
}
ncclResult_t ncclProxySharedBuffersGetCollNet(struct ncclComm* comm, int cuda, int type, int channel, int slot, int index, char** ptr) {
  struct ncclProxySharedBuffers* state = &comm->proxyState.sharedBuffs;
  // Use different pools for different channels and also separate send/recv.
  char* buff = cuda ? state->cudaBuff : state->hostBuff;
  int slotSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  int globalSlot = ((type*NCCL_STEPS+slot)*comm->nChannels)+channel;
  *ptr = buff + slotSize * globalSlot;
  return ncclSuccess;
}

ncclResult_t ncclProxySharedBuffersDestroy(struct ncclComm* comm) {
  struct ncclProxySharedBuffers* state = &comm->proxyState.sharedBuffs;
  CUDACHECK(cudaFree(state->cudaBuff));
  NCCLCHECK(ncclCudaHostFree(state->hostBuff));
  return ncclSuccess;
}

ncclResult_t ncclProxyCreate(struct ncclComm* comm) {
  if (!comm->proxyThread) {
    comm->proxyState.cond = PTHREAD_COND_INITIALIZER;
    comm->proxyState.opsMutex = PTHREAD_MUTEX_INITIALIZER;
    comm->proxyState.poolMutex = PTHREAD_MUTEX_INITIALIZER;
    comm->proxyState.ops = NULL;
    pthread_create(&comm->proxyThread, NULL, persistentThread, comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyDestroy(struct ncclComm* comm) {
  struct ncclProxyState* state = &comm->proxyState;

  // Request the proxy to stop and then wake it
  pthread_mutex_lock(&state->opsMutex);
  state->stop = true;
  pthread_cond_signal(&state->cond);
  pthread_mutex_unlock(&state->opsMutex);
  if (comm->proxyThread) pthread_join(comm->proxyThread, NULL);

  // Free off any memory allocated for the proxy arg pools
  pthread_mutex_lock(&state->poolMutex);
  struct ncclProxyState* proxyState = &comm->proxyState;
  while (proxyState->pools != NULL) {
    struct ncclProxyPool *next = proxyState->pools->next;
    free(proxyState->pools);
    proxyState->pools = next;
  }
  pthread_mutex_unlock(&state->poolMutex);

  NCCLCHECK(ncclProxySharedBuffersDestroy(comm));

  return ncclSuccess;
}
