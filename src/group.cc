/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "group.h"
#include "debug.h"
#include "enqueue.h"
#include "transport.h"

#define MAX_ASYNC_OPS 128
thread_local pthread_t ncclGroupThreads[MAX_ASYNC_OPS];
thread_local int ncclGroupIndex = 0;
thread_local int ncclGroupMode = 0;
thread_local ncclResult_t ncclGroupError = ncclSuccess;

bool ncclAsyncMode() {
  return ncclGroupMode > 0;
}

ncclResult_t ncclAsyncErrCheck(ncclResult_t ret) {
  if (ncclGroupError == ncclSuccess || ret != ncclSuccess) ncclGroupError = ret;
  return ret;
}

struct ncclInitArgs {
  ncclInitFunc_t func;
  int cudaDev;
  ncclComm_t* newcomm;
  int ndev;
  ncclUniqueId commId;
  int myrank;
};
struct ncclCollArgs {
  ncclComm_t comm;
  int *send, *recv;
  int *nsend, *nrecv;
};

enum ncclAsyncFuncType {
  ASYNC_FUNC_INVALID = 0,
  ASYNC_FUNC_INIT = 1,
  ASYNC_FUNC_COLL = 2,
};
struct ncclAsyncArgs {
  ncclResult_t ret;
  enum ncclAsyncFuncType funcType;
  union {
    ncclCollArgs coll;
    ncclInitArgs init;
  };
};

thread_local struct ncclAsyncArgs ncclGroupArgs[MAX_ASYNC_OPS];

#define NCCLCHECKTHREAD(a) do { \
  if ((args->ret = (a)) != ncclSuccess) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
    return args; \
  } \
} while(0)

#define CUDACHECKTHREAD(a) do { \
  if ((a) != cudaSuccess) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
    args->ret = ncclUnhandledCudaError; \
    return args; \
  } \
} while(0)

void* ncclAsyncThreadMain(void* args_) {
  struct ncclAsyncArgs* args = (struct ncclAsyncArgs*)args_;
  NCCLCHECKTHREAD(args->init.func(args->init.newcomm, args->init.ndev, args->init.commId, args->init.myrank, args->init.cudaDev));
  return args;
}

ncclResult_t ncclAsyncInit(ncclInitFunc_t func, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, int cudaDev) {
  if (ncclGroupIndex >= MAX_ASYNC_OPS) {
    WARN("Too many async operations in progress, max is %d", MAX_ASYNC_OPS);
    return ncclAsyncErrCheck(ncclInvalidUsage);
  }
  int index = ncclGroupIndex++;
  struct ncclAsyncArgs* args = ncclGroupArgs+index;
  args->funcType = ASYNC_FUNC_INIT;
  args->init.func = func;
  args->init.cudaDev = cudaDev;
  args->init.newcomm = newcomm;
  args->init.ndev = ndev;
  memcpy(&args->init.commId, &commId, sizeof(commId));
  args->init.myrank = myrank;
  return ncclSuccess;
}

ncclResult_t ncclAsyncColl(ncclComm_t comm) {
  struct ncclAsyncArgs* args = ncclGroupArgs;
  for (int i=0; i<ncclGroupIndex; i++) {
    if (args->coll.comm == comm) return ncclSuccess;
    args++;
  }
  if (ncclGroupIndex >= MAX_ASYNC_OPS) {
    WARN("Too many async operations in progress, max is %d", MAX_ASYNC_OPS);
    return ncclAsyncErrCheck(ncclInvalidUsage);
  }
  ncclGroupIndex++;
  args->funcType = ASYNC_FUNC_COLL;
  args->coll.comm = comm;
  args->coll.recv=NULL; args->coll.send=NULL;
  args->coll.nsend=0; args->coll.nrecv=0;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGroupStart);
ncclResult_t ncclGroupStart() {
  if (ncclGroupMode == 0) {
    memset(ncclGroupArgs, 0, sizeof(struct ncclAsyncArgs)*MAX_ASYNC_OPS);
  }
  ncclGroupMode++;
  return ncclSuccess;
}

//check if peers need to be connected on one of the channels
static int isPeerConnected(struct ncclChannel* channel, int peer, int send) {
  return send ? channel->peers[peer].send.connected : channel->peers[peer].recv.connected;
}

static ncclResult_t scheduleSendRecv(struct ncclComm* comm, int delta, int channelId, size_t recvbytes, void* recvbuff, size_t sendbytes, const void* sendbuff) {
  struct ncclInfo info = { ncclCollSendRecv, "SendRecv",
    sendbuff, recvbuff, std::max<size_t>(sendbytes,recvbytes), ncclInt8, ncclSum, -1, comm, comm->userStream, /* Args */
    SENDRECV_CHUNKSTEPS, SENDRECV_SLICESTEPS };
  info.delta=delta;
  info.channelId=channelId;
  info.sendbytes=sendbytes;
  info.recvbytes=recvbytes;
  if (delta == 0) info.nBytes=sendbytes;
  NCCLCHECK(ncclSaveKernel(&info));
  return ncclSuccess;
}

void* ncclAsyncThreadPreconnect(void* args_) {
  struct ncclAsyncArgs* args = (struct ncclAsyncArgs*)args_;
  CUDACHECKTHREAD(cudaSetDevice(args->coll.comm->cudaDev));
  for (int c=0; c<args->coll.comm->nChannels; c++) {
    struct ncclChannel* channel = args->coll.comm->channels+c;
    NCCLCHECKTHREAD(ncclTransportP2pSetup(args->coll.comm, NULL, channel, args->coll.nrecv[c], args->coll.recv+c*args->coll.comm->nRanks, args->coll.nsend[c], args->coll.send+c*args->coll.comm->nRanks));
  }
  return args;
}

NCCL_API(ncclResult_t, ncclGroupEnd);
ncclResult_t ncclGroupEnd() {
  ncclGroupMode--;
  if (ncclGroupMode > 0) return ncclSuccess;
  int savedDev;
  CUDACHECK(cudaGetDevice(&savedDev));
  int activeThreads = 0;
  int doneArray[MAX_ASYNC_OPS];
  for (int i=0; i<ncclGroupIndex; i++) doneArray[i] = 1;
  ncclResult_t ret = ncclGroupError;
  if (ret != ncclSuccess) goto group_cleanup;

  /* Launch async ncclCommInitRank */
  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_INIT) {
      pthread_create(ncclGroupThreads+i, NULL, ncclAsyncThreadMain, args);
      activeThreads++;
      doneArray[i] = 0;
    }
  }
  /* For init, since we use threads, we just wait for threads to complete */
  while (activeThreads) {
    for (int i=0; i<ncclGroupIndex; i++) {
      struct ncclAsyncArgs* args = ncclGroupArgs+i;
      if (args->funcType == ASYNC_FUNC_INIT && doneArray[i] == 0) {
        int err = pthread_tryjoin_np(ncclGroupThreads[i], NULL);
        if (err == EBUSY) continue;
        if (err != 0) ret = ncclSystemError;
        if (args->ret != ncclSuccess) ret = args->ret;
        doneArray[i] = 1;
        activeThreads--;
      }
    }
  }

  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_COLL) {
      struct ncclP2Plist* p2plist = &args->coll.comm->p2plist;
      if (p2plist->count != 0) {
        struct ncclComm* comm = args->coll.comm;
        int rank = comm->rank;
        int nRanks = comm->nRanks;
        int ops = 0;
        int connect = 0;
        if (args->coll.nrecv == NULL)
          NCCLCHECK(ncclCalloc(&args->coll.nrecv, comm->nChannels));
        if (args->coll.nsend == NULL)
          NCCLCHECK(ncclCalloc(&args->coll.nsend, comm->nChannels));
        for (int delta=1; delta<nRanks; delta++) {
          uint32_t from = (rank+nRanks-delta)%nRanks;
          uint32_t to = (rank+delta)%nRanks;
	  int channelStart = (delta-1) % (comm->nChannels);
          if (p2plist->peerlist[from].recvbytes >= 0 || p2plist->peerlist[to].recvbytes >= 0) ops++;

          if (p2plist->peerlist[from].recvbytes >= 0) {
            int nChannels;
            NCCLCHECK(ncclTopoGetNchannels(comm->topo, rank, from, &nChannels));
            for (int c=channelStart; c<channelStart+nChannels; c++) {
              int channelId = c % comm->nChannels;
              if (isPeerConnected(comm->channels+channelId, from, 0) == 1) continue;
              if (args->coll.recv == NULL)
                NCCLCHECK(ncclCalloc(&args->coll.recv, comm->nChannels*nRanks));
              args->coll.recv[channelId*nRanks+args->coll.nrecv[channelId]++] = from;
            }
            connect = 1;
          }
          if (p2plist->peerlist[to].recvbytes >= 0) {
            int nChannels;
            NCCLCHECK(ncclTopoGetNchannels(comm->topo, rank, to, &nChannels));
            for (int c=channelStart; c<channelStart+nChannels; c++) {
              int channelId = c % comm->nChannels;
              if (isPeerConnected(comm->channels+channelId, to, 1) == 1) continue;
              if(args->coll.send == NULL)
                NCCLCHECK(ncclCalloc(&args->coll.send, comm->nChannels*nRanks));
              args->coll.send[channelId*nRanks+args->coll.nsend[channelId]++] = to;
            }
            connect = 1;
          }
        }
        if (connect) {
          pthread_create(ncclGroupThreads+i, NULL, ncclAsyncThreadPreconnect, args);
        } else {
          free(args->coll.nsend); args->coll.nsend = NULL;
          free(args->coll.nrecv); args->coll.nrecv = NULL;
        }
        // Keep the number of operations for next step
        p2plist->count = ops;
      }
    }
  }

  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_COLL && (args->coll.nrecv || args->coll.nsend)) {
      int err = pthread_join(ncclGroupThreads[i], NULL);
      if (err != 0) {
        WARN("Error waiting for pthread_join : %s\n", strerror(errno));
        return ncclSystemError;
      }
      NCCLCHECK(args->ret);
      if (args->coll.send != NULL) { free(args->coll.send); args->coll.send = NULL; }
      if (args->coll.recv != NULL) { free(args->coll.recv); args->coll.recv = NULL; }
      free(args->coll.nsend); args->coll.nsend = NULL;
      free(args->coll.nrecv); args->coll.nrecv = NULL;
    }
  }

  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_COLL) {
      struct ncclComm* comm = args->coll.comm;
      int rank = comm->rank;
      int nRanks = comm->nRanks;
      struct ncclP2Plist* p2plist = &args->coll.comm->p2plist;
      if (p2plist->peerlist[rank].recvbytes > 0 || p2plist->peerlist[rank].sendbytes > 0) {
        if (p2plist->peerlist[rank].sendbytes != p2plist->peerlist[rank].recvbytes) {
          WARN("Error : send/recv to/from self : size mismatch.");
          return ncclInvalidUsage;
        }
        if (p2plist->peerlist[rank].sendbuff != p2plist->peerlist[rank].recvbuff) {
          CUDACHECK(cudaMemcpyAsync(p2plist->peerlist[rank].recvbuff, p2plist->peerlist[rank].sendbuff,
                p2plist->peerlist[rank].sendbytes, cudaMemcpyDeviceToDevice, args->coll.comm->userStream));
        }
      }
      if (p2plist->count) {
        for (int delta=1; delta<nRanks; delta++) {
          uint32_t from = (rank+nRanks-delta)%nRanks;
          uint32_t to = (rank+delta)%nRanks;

          // Compute how much to split operations
          int nChannelsFrom, nChannelsTo;
          NCCLCHECK(ncclTopoGetNchannels(comm->topo, rank, from, &nChannelsFrom));
          NCCLCHECK(ncclTopoGetNchannels(comm->topo, rank, to, &nChannelsTo));
          int channelsPerOp = std::max(nChannelsFrom, nChannelsTo);
          // Natural step size matching buffer steps.
          size_t stepSize = args->coll.comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
          // Split each operation on nChannels max.
          size_t chunkSize = DIVUP(p2plist->maxBytes, channelsPerOp);
          chunkSize = std::max((size_t)1, DIVUP(chunkSize, stepSize)) * stepSize;

          size_t offset = 0;
          int remaining = 1;
          int channelId = (delta-1) % comm->nChannels;
          while (remaining) {
            remaining = 0;
            size_t recvbytes = p2plist->peerlist[from].recvbytes-offset;
            size_t sendbytes = p2plist->peerlist[to].sendbytes-offset;
            if (recvbytes > chunkSize) { remaining = 1; recvbytes = chunkSize; } else p2plist->peerlist[from].recvbytes = -1;
            if (sendbytes > chunkSize) { remaining = 1; sendbytes = chunkSize; } else p2plist->peerlist[to].sendbytes = -1;
            if (sendbytes >= 0 || recvbytes >= 0) {
              NCCLCHECK(scheduleSendRecv(args->coll.comm, delta, channelId,
                    recvbytes, ((char*)(p2plist->peerlist[from].recvbuff)) + offset,
                    sendbytes, ((const char*)(p2plist->peerlist[to].sendbuff)) + offset));
            }
            offset += chunkSize;
            channelId = (channelId+1) % comm->nChannels;
          }
        }
        p2plist->count = 0;
        p2plist->maxBytes = 0;
      }
    }
  }

  /* Collectives are done in three steps :
   * 1. Barrier Check In. Only the last call may call cudaLaunchKernel[cooperative]
   * 2. Barrier Wait. No CUDA call is permitted
   * 3. Enqueue Events. CUDA event wait/enqueue.
   * This is needed because step 2 cannot call any CUDA primitive, otherwise if
   * cudaFree happens between 1 and 3, it could block that CUDA call and
   * prevent some ranks from launching their network threads, which would
   * prevent the NCCL call from completing, blocking the cudaFree call.
   */
  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_COLL) {
      if (args->coll.comm->userStream == NULL)
        CUDACHECKGOTO(cudaSetDevice(args->coll.comm->cudaDev), ret, end);
      NCCLCHECKGOTO(ncclBarrierEnqueue(args->coll.comm), ret, end);
    }
  }
  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_COLL) {
      CUDACHECKGOTO(cudaSetDevice(args->coll.comm->cudaDev), ret, end);
      NCCLCHECKGOTO(ncclBarrierEnqueueWait(args->coll.comm), ret, end);
    }
  }
  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_COLL) {
      if (args->coll.comm->userStream == NULL)
        CUDACHECKGOTO(cudaSetDevice(args->coll.comm->cudaDev), ret, end);
      NCCLCHECKGOTO(ncclEnqueueEvents(args->coll.comm), ret, end);
    }
  }

  goto end;
group_cleanup:
  if (ret != ncclSuccess) {
    // At least one call in the group failed. Since we want to make that group
    // an atomic operation, we need to cancel all operations.
    for (int i=0; i<ncclGroupIndex; i++) {
      struct ncclAsyncArgs* args = ncclGroupArgs+i;
      if (args->funcType == ASYNC_FUNC_INIT) {
        if (args->init.newcomm) NCCLCHECK(ncclCommDestroy(*args->init.newcomm));
        *args->init.newcomm = NULL;
      } else {
        struct ncclComm* comm = args->coll.comm;
        for (int c=0; c<comm->nChannels; c++) {
          struct ncclChannel* channel = comm->channels+c;
          for (int i=0; i<channel->collCount; i++) {
            channel->collectives[(channel->collStart + i)%NCCL_MAX_OPS].active = 0;
          }
          channel->collFifoTail = channel->collStart;
          channel->collCount = 0;
        }
        /* Cancel all proxy ops : mark them as ncclProxyOpNone and they should be freed later on */
        struct ncclProxyState* state = &comm->proxyState;
        struct ncclProxyArgs *op, *start;
        pthread_mutex_lock(&state->mutex);
        op = start = state->ops;
        while (op) {
          if (op->opCount >= comm->lastOpCount) op->state = ncclProxyOpNone;
          struct ncclProxyArgs* peerOp = op->nextPeer;
          while (peerOp) {
            if (peerOp->opCount >= comm->lastOpCount) peerOp->state = ncclProxyOpNone;
            peerOp = peerOp->nextPeer;
          }
          op = op->next;
          if (op == start) break;
        }
        comm->opCount = comm->lastOpCount;
        pthread_cond_signal(&state->cond);
        pthread_mutex_unlock(&state->mutex);

        comm->myParams->gridDim.x = comm->myParams->blockDim.x = 0;
        comm->userStreamSet = false;
      }
    }
  }
end:
  ncclGroupError = ncclSuccess;
  ncclGroupIndex = 0;
  CUDACHECK(cudaSetDevice(savedDev)); // do other clean-ups first before calling cudaSetDevice, because this call can fail too
  return ret;
}
