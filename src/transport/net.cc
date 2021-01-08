/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "net.h"
#include "graph.h"
#include "collectives.h"

struct netConnectInfo {
  ncclNetHandle_t netHandle;
};

#define LOC_HOSTMEM 0
#define LOC_DEVMEM  1
#define LOC_COUNT   2

struct netSendResources {
  void* netSendComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  int netDev;
  int useGdr;
  int shared;
  char* buffers[LOC_COUNT];
  int buffSizes[LOC_COUNT];
  void* mhandles[LOC_COUNT];
  void** mhandlesProto[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
};

struct netRecvResources {
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;
  int netDev;
  int useGdr;
  int shared;
  char* buffers[LOC_COUNT];
  int buffSizes[LOC_COUNT];
  void* mhandles[LOC_COUNT];
  void** mhandlesProto[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
};

/* Determine if two peers can communicate with NET */
ncclResult_t netCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  return ncclSuccess;
}

NCCL_PARAM(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t netSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId) {
  struct netSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  send->conn.shared = resources->shared = ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : graph ? 0 : 1;

  NCCLCHECK(ncclTopoGetNetDev(comm->topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, resources->netDev, 1, &resources->useGdr));

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));
  NCCLCHECK(ncclCudaHostCalloc(&resources->recvMem, 1));

  send->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;
  send->conn.tail = &resources->recvMem->tail;
  send->conn.sizesFifo = resources->recvMem->sizesFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  send->conn.ptrsFifo = resources->shared ? resources->recvMem->ptrsFifo : NULL;
  send->conn.head = &resources->sendMem->head;
  resources->sendMem->head = resources->shared ? -NCCL_STEPS : 0; // Don't give any credit yet when sharing buffers
  for (int i=0; i<NCCL_STEPS; i++) send->conn.sizesFifo[i] = -1;

  if (resources->shared == 0) {
    int protoLoc[NCCL_NUM_PROTOCOLS];
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      protoLoc[p] = p != NCCL_PROTO_LL && resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
    }
    int buffSizes[NCCL_NUM_PROTOCOLS];
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      buffSizes[p] = send->comm->buffSizes[p];
      resources->buffSizes[protoLoc[p]] += buffSizes[p];
    }

    if (resources->buffSizes[LOC_DEVMEM]) {
      NCCLCHECK(ncclCudaCalloc(resources->buffers+LOC_DEVMEM, resources->buffSizes[LOC_DEVMEM]));
    }
    if (resources->buffSizes[LOC_HOSTMEM]) {
      NCCLCHECK(ncclCudaHostCalloc(resources->buffers+LOC_HOSTMEM, resources->buffSizes[LOC_HOSTMEM]));
    }

    int offsets[LOC_COUNT];
    offsets[LOC_HOSTMEM] = offsets[LOC_DEVMEM] = 0;
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      resources->mhandlesProto[p] = resources->mhandles+protoLoc[p];
      send->conn.buffs[p] = resources->buffers[protoLoc[p]] + offsets[protoLoc[p]];
      offsets[protoLoc[p]] += buffSizes[p];
    }
  }

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d%s%s", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "", resources->shared ? "/Shared" : "");
  return ncclSuccess;
}

ncclResult_t netRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId) {
  struct netRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  recv->conn.shared = resources->shared = ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : graph ? 0 : 1;

  NCCLCHECK(ncclTopoGetNetDev(comm->topo, myInfo->rank, graph, channelId, &resources->netDev));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, resources->netDev, 0, &resources->useGdr));

  NCCLCHECK(ncclCudaHostCalloc(&resources->sendMem, 1));
  NCCLCHECK(ncclCudaHostCalloc(&resources->recvMem, 1));

  recv->conn.direct |= resources->useGdr ? NCCL_DIRECT_NIC : 0;
  recv->conn.tail = &resources->recvMem->tail;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  recv->conn.ptrsFifo = resources->shared ? resources->recvMem->ptrsFifo : NULL;
  recv->conn.head = &resources->sendMem->head;

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree not for p2p
    int protoLoc[NCCL_NUM_PROTOCOLS];
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      protoLoc[p] = resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
    }

    int buffSizes[NCCL_NUM_PROTOCOLS];
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      buffSizes[p] = recv->comm->buffSizes[p];
      resources->buffSizes[protoLoc[p]] += buffSizes[p];
    }

    if (resources->buffSizes[LOC_DEVMEM]) {
      NCCLCHECK(ncclCudaCalloc(resources->buffers+LOC_DEVMEM, resources->buffSizes[LOC_DEVMEM]));
    }
    if (resources->buffSizes[LOC_HOSTMEM]) {
      NCCLCHECK(ncclCudaHostCalloc(resources->buffers+LOC_HOSTMEM, resources->buffSizes[LOC_HOSTMEM]));
    }

    int offsets[LOC_COUNT];
    offsets[LOC_HOSTMEM] = offsets[LOC_DEVMEM] = 0;
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      resources->mhandlesProto[p] = resources->mhandles+protoLoc[p];
      recv->conn.buffs[p] = resources->buffers[protoLoc[p]] + offsets[protoLoc[p]];
      offsets[protoLoc[p]] += buffSizes[p];
    }
  }

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d : %d[%lx] -> %d[%lx] [receive] via NET/%s/%d%s%s", channelId, peerInfo->rank, peerInfo->busId, myInfo->rank, myInfo->busId, ncclNetName(), resources->netDev,
      resources->useGdr ? "/GDRDMA" : "", resources->shared ? "/Shared" : "");
  struct netConnectInfo* info = (struct netConnectInfo*) connectInfo;
  NCCLCHECK(ncclNetListen(resources->netDev, &info->netHandle, &resources->netListenComm));

  return ncclSuccess;
}

ncclResult_t netSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct netSendResources* resources = (struct netSendResources*)send->transportResources;
  struct netConnectInfo* info = (struct netConnectInfo*)connectInfo;

  // Connect to remote peer
  NCCLCHECK(ncclNetConnect(resources->netDev, info->netHandle, &resources->netSendComm));

  if (resources->shared) {
    // Get shared buffers
    int loc = resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
    NCCLCHECK(ncclProxySharedBuffersInit(send->comm, resources->useGdr, resources->buffSizes+loc, resources->buffers+loc));
    resources->mhandlesProto[NCCL_PROTO_SIMPLE] = resources->mhandles+loc;
  }

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->buffers[LOC_DEVMEM], resources->buffSizes[LOC_DEVMEM], NCCL_PTR_CUDA, &resources->mhandles[LOC_DEVMEM]));
  }
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->buffers[LOC_HOSTMEM], resources->buffSizes[LOC_HOSTMEM], NCCL_PTR_HOST, &resources->mhandles[LOC_HOSTMEM]));
  }
  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t netRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct netRecvResources* resources = (struct netRecvResources*)recv->transportResources;

  // Finish connection establishment from remote peer
  NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
  NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

  if (resources->shared) {
    // Get shared buffers
    int loc = resources->useGdr ? LOC_DEVMEM : LOC_HOSTMEM;
    NCCLCHECK(ncclProxySharedBuffersInit(recv->comm, resources->useGdr, resources->buffSizes+loc, resources->buffers+loc));
    resources->mhandlesProto[NCCL_PROTO_SIMPLE] = resources->mhandles+loc;
  }

  if (resources->buffSizes[LOC_DEVMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netRecvComm, resources->buffers[LOC_DEVMEM], resources->buffSizes[LOC_DEVMEM], NCCL_PTR_CUDA, &resources->mhandles[LOC_DEVMEM]));
  }
  if (resources->buffSizes[LOC_HOSTMEM]) {
    NCCLCHECK(ncclNetRegMr(resources->netRecvComm, resources->buffers[LOC_HOSTMEM], resources->buffSizes[LOC_HOSTMEM], NCCL_PTR_HOST, &resources->mhandles[LOC_HOSTMEM]));
  }
  return ncclSuccess;
}

ncclResult_t netSendFree(void* transportResources) {
  struct netSendResources* resources = (struct netSendResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  for (int l=0; l<LOC_COUNT; l++) {
    if (resources->buffers[l])
      NCCLCHECK(ncclNetDeregMr(resources->netSendComm, resources->mhandles[l]));
  }
  if (resources->shared == 0) {
    NCCLCHECK(ncclCudaHostFree(resources->buffers[LOC_HOSTMEM]));
    CUDACHECK(cudaFree(resources->buffers[LOC_DEVMEM]));
  }
  NCCLCHECK(ncclNetCloseSend(resources->netSendComm));
  free(resources);
  return ncclSuccess;
}

ncclResult_t netRecvFree(void* transportResources) {
  struct netRecvResources* resources = (struct netRecvResources*)transportResources;
  NCCLCHECK(ncclCudaHostFree(resources->sendMem));
  NCCLCHECK(ncclCudaHostFree(resources->recvMem));
  for (int l=0; l<LOC_COUNT; l++) {
    if (resources->buffers[l])
      NCCLCHECK(ncclNetDeregMr(resources->netRecvComm, resources->mhandles[l]));
  }
  if (resources->shared == 0) {
    NCCLCHECK(ncclCudaHostFree(resources->buffers[LOC_HOSTMEM]));
    CUDACHECK(cudaFree(resources->buffers[LOC_DEVMEM]));
  }
  NCCLCHECK(ncclNetCloseRecv(resources->netRecvComm));
  free(resources);
  return ncclSuccess;
}

static_assert(NCCL_STEPS <= NCCL_NET_MAX_REQUESTS, "Not enough net requests to cover for steps");

#define STEP_PRINTF(...)
//#define STEP_PRINTF printf

ncclResult_t netSendProxy(struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct netSendResources* resources = (struct netSendResources*) (sub->connector->transportResources);
      // Round to next multiple of sliceSteps
      resources->step = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = resources->step;
      sub->end = sub->done + sub->nsteps;
      STEP_PRINTF("[%d/%d/%ld/%d] Send starts at %ld, ends at %ld\n", sub->connector->comm->rank, sub->channel->id, args->opCount, s, sub->done, sub->end);
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct netSendResources* resources = (struct netSendResources*) (sub->connector->transportResources);
      void* mhandle = *(resources->mhandlesProto[p]);
      int stepSize = sub->connector->comm->buffSizes[p] / NCCL_STEPS;
      char* localBuff = sub->connector->conn.buffs[p];
      int buffSize = stepSize*args->sliceSteps;
      if (resources->shared) buffSize /= SENDRECV_SLICEFACTOR;
      if (sub->sendbytes < buffSize) buffSize = sub->sendbytes;
      // Post buffers to the GPU
      if (sub->posted < sub->end && sub->posted < sub->done + NCCL_STEPS) {
        if (resources->shared) {
          char* ptr;
          NCCLCHECK(ncclProxySharedBuffersAlloc(sub->connector->comm, resources->useGdr, 0, sub->channel->id, buffSize, &ptr));
          if (ptr == NULL) return ncclInternalError;
          resources->recvMem->ptrsFifo[sub->posted%NCCL_STEPS] = ptr;
          __sync_synchronize();
          volatile uint64_t* sendHead = &resources->sendMem->head;
          sub->posted += args->sliceSteps;
          *sendHead = sub->posted - NCCL_STEPS;
        } else sub->posted += args->sliceSteps;
        STEP_PRINTF("[%d/%d/%ld/%d] %ld Send posted -> %ld\n", sub->connector->comm->rank, sub->channel->id, args->opCount, s, sub->posted, sub->posted + args->sliceSteps);
        args->idle = 0;
        continue;
      }
      // Check whether we received data from the GPU and send it to the network
      int buffSlot = sub->transmitted%NCCL_STEPS;
      if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
        volatile int* sizesFifo = resources->recvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        if (sizesFifo[buffSlot] != -1 && (*recvTail > sub->transmitted || p == NCCL_PROTO_LL)) {
          // We have something to receive, let's check if it's completely ready.
          int size = sizesFifo[buffSlot];
          char* buff = resources->shared ? (char*)resources->recvMem->ptrsFifo[buffSlot] : localBuff+buffSlot*stepSize;
          int ready = 1;
          if (p == NCCL_PROTO_LL128) {
            int ready = resources->useGdr;
            if (!ready) {
              // When data is in sysmem, we need to wait until all flags are correct since the GPU only
              // called threadfence()
              uint64_t flag = sub->transmitted + 1;
              int nFifoLines = DIVUP(sizesFifo[buffSlot], sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
              volatile uint64_t* lines = (volatile uint64_t*)buff;
              ready = 1;
              for (int i=0; i<nFifoLines; i++) {
                if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
              }
            }
          } else if (p == NCCL_PROTO_LL) {
            uint32_t flag = NCCL_LL_FLAG(sub->transmitted + 1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)buff;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
            }
          }
          if (ready) {
            // Data is ready, try to send.
            NCCLCHECK(ncclNetIsend(resources->netSendComm, buff, size, mhandle, sub->requests+buffSlot));
            if (sub->requests[buffSlot] != NULL) {
              TRACE(NCCL_NET, "sendProxy [%d/%d] Isend (LL) posted, req %p", sub->transmitted, buffSlot, sub->requests[buffSlot]);
              sizesFifo[buffSlot] = -1;
              // Make sure size is reset to zero before we update the head.
              __sync_synchronize();
              STEP_PRINTF("[%d/%d/%ld/%d] %ld Send transmitted [%d], transmitted -> %ld\n", sub->connector->comm->rank, sub->channel->id, args->opCount, s, sub->transmitted, buffSlot, sub->transmitted + args->sliceSteps);
              sub->transmitted += args->sliceSteps;
              args->idle = 0;
              continue;
            }
          }
        }
      }
      // Check whether the network has completed some send operations.
      if (sub->done < sub->transmitted) {
        int done;
        int buffSlot = sub->done%NCCL_STEPS;
        NCCLCHECK(ncclNetTest(sub->requests[buffSlot], &done, NULL));
        if (done) {
          TRACE(NCCL_NET, "sendProxy [%d/%d] request %p done, size %d", sub->done, buffSlot, sub->requests[buffSlot]);
          if (resources->shared) {
            char* ptr = (char*)resources->recvMem->ptrsFifo[sub->done%NCCL_STEPS];
            NCCLCHECK(ncclProxySharedBuffersFree(sub->connector->comm, resources->useGdr, 0, sub->channel->id, buffSize, ptr));
          }
          STEP_PRINTF("[%d/%d/%ld/%d] %ld Send done [%d], done -> %ld\n", sub->connector->comm->rank, sub->channel->id, args->opCount, s, sub->done, buffSlot, sub->done + args->sliceSteps);
          sub->done += args->sliceSteps;

          if (resources->shared == 0) {
            resources->sendMem->head = sub->done;
          }
          args->idle = 0;
          if (sub->done == sub->end) {
            STEP_PRINTF("[%d/%d/%ld/%d] Send ends at %ld\n", sub->connector->comm->rank, sub->channel->id, args->opCount, s, resources->step);
            resources->step = sub->end;
            args->done++;
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

ncclResult_t netRecvProxy(struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct netRecvResources* resources = (struct netRecvResources*) (sub->connector->transportResources);
      // Round to next multiple of sliceSteps
      resources->step = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->received = sub->transmitted = sub->done = resources->step;
      sub->end = sub->done + sub->nsteps;
      STEP_PRINTF("[%d/%d/%ld/%d] Recv starts at %ld, ends at %ld\n", sub->connector->comm->rank, sub->channel->id, args->opCount, s, sub->done, sub->end);
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      if (sub->done == sub->end) continue;
      struct netRecvResources* resources = (struct netRecvResources*) (sub->connector->transportResources);
      void* mhandle = *(resources->mhandlesProto[p]);
      int stepSize = sub->connector->comm->buffSizes[p] / NCCL_STEPS;
      char* localBuff = sub->connector->conn.buffs[p];
      int buffSize = stepSize*args->sliceSteps;
      if (resources->shared) buffSize /= SENDRECV_SLICEFACTOR;
      if (sub->recvbytes < buffSize) buffSize = sub->recvbytes;
      if ((sub->posted < sub->done + NCCL_STEPS) && (sub->posted < sub->end)) {
        int buffSlot = sub->posted%NCCL_STEPS;
        char* ptr;
        if (resources->shared) {
          NCCLCHECK(ncclProxySharedBuffersAlloc(sub->connector->comm, resources->useGdr, 1, sub->channel->id, buffSize, &ptr));
          if (ptr == NULL) return ncclInternalError;
          volatile void** ptrsFifo = (volatile void**)resources->recvMem->ptrsFifo;
          ptrsFifo[buffSlot] = ptr;
        } else {
          ptr = localBuff+buffSlot*stepSize;
        }
        NCCLCHECK(ncclNetIrecv(resources->netRecvComm, ptr, buffSize, mhandle, sub->requests+buffSlot));
        if (sub->requests[buffSlot] != NULL) {
          TRACE(NCCL_NET, "recvProxy [%d/%d] posted recv request %p", sub->posted, buffSlot, sub->requests[buffSlot]);
          STEP_PRINTF("[%d/%d/%ld/%d] %ld Recv posted [%d], posted -> %ld\n", sub->connector->comm->rank, sub->channel->id, args->opCount, s, sub->posted, buffSlot, sub->posted + args->sliceSteps);
          sub->posted += args->sliceSteps;
          args->idle = 0;
          continue;
        } else if (resources->shared) {
          NCCLCHECK(ncclProxySharedBuffersFree(sub->connector->comm, resources->useGdr, 1, sub->channel->id, buffSize, ptr));
        }
      }
      if (sub->posted > sub->received) {
        int buffSlot = sub->received%NCCL_STEPS;
        int done, size;
        NCCLCHECK(ncclNetTest(sub->requests[buffSlot], &done, &size));
        if (done) {
          STEP_PRINTF("[%d/%d/%ld/%d] %ld Recv received [%d], received -> %ld\n", sub->connector->comm->rank, sub->channel->id, args->opCount, s, sub->received, buffSlot, sub->received + args->sliceSteps);
          sub->received += args->sliceSteps;
          if (size > 0 && p == NCCL_PROTO_SIMPLE && resources->useGdr) {
            // Don't pass data to the GPU yet, flush first.
            volatile void** ptrsFifo = (volatile void**)resources->recvMem->ptrsFifo;
            char* ptr = resources->shared ? (char*)(ptrsFifo[buffSlot]) : localBuff+buffSlot*stepSize;
            NCCLCHECK(ncclNetIflush(resources->netRecvComm, ptr, size, mhandle, sub->requests+buffSlot));
          } else {
            sub->requests[buffSlot] = NULL;
          }
          args->idle = 0;
          continue;
        }
      }
      if (sub->received > sub->transmitted) {
        // Progress flush operations
        int buffSlot = sub->transmitted%NCCL_STEPS;
        int done = 1;
        if (sub->requests[buffSlot]) NCCLCHECK(ncclNetTest(sub->requests[buffSlot], &done, NULL));
        if (done) {
          STEP_PRINTF("[%d/%d/%ld/%d] %ld Recv transmitted [%d], transmitted -> %ld\n", sub->connector->comm->rank, sub->channel->id, args->opCount, s, sub->transmitted, buffSlot, sub->transmitted + args->sliceSteps);
          sub->transmitted += args->sliceSteps;
          __sync_synchronize();
          resources->recvMem->tail = sub->transmitted;
          args->idle = 0;
          continue;
        }
      }
      if (sub->transmitted > sub->done) {
        volatile uint64_t* sendHead = &resources->sendMem->head;
        uint64_t done = *sendHead;
        while (done > sub->done &&
            // LL and LL128 can acknowledge 0-bytes send before they even happen. Don't go past what we transmitted.
            sub->transmitted > sub->done) {
          if (resources->shared) {
            char* ptr = (char*)resources->recvMem->ptrsFifo[sub->done%NCCL_STEPS];
            NCCLCHECK(ncclProxySharedBuffersFree(sub->connector->comm, resources->useGdr, 1, sub->channel->id, buffSize, ptr));
          }
          STEP_PRINTF("[%d/%d/%ld/%d] %ld Recv done -> %ld\n", sub->connector->comm->rank, sub->channel->id, args->opCount, s, sub->done, sub->done + args->sliceSteps);
          sub->done += args->sliceSteps;
          args->idle = 0;
          if (sub->done == sub->end) {
            resources->step = sub->end;
            STEP_PRINTF("[%d/%d/%ld/%d] Recv ends at %ld\n", sub->connector->comm->rank, sub->channel->id, args->opCount, s, resources->step);
            args->done++;
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  netCanConnect,
  { netSendSetup, netSendConnect, netSendFree, netSendProxy },
  { netRecvSetup, netRecvConnect, netRecvFree, netRecvProxy }
};
