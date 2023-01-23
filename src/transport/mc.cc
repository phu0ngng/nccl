/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "proxy.h"

#define USE_POSIX_FD 1

#if USE_POSIX_FD
#define MC_CU_MEM_HANDLE_TYPE CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
#else
#define MC_CU_MEM_HANDLE_TYPE CU_MEM_HANDLE_TYPE_NONE
#endif

/* Determine if two peers can communicate through mc */
ncclResult_t mcCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // This transport cannot be used for p2p
  *ret = 0;
  return ncclSuccess;
}

ncclResult_t mcSendFree(struct ncclConnector* send) {
  return ncclSuccess;
}

ncclResult_t mcRecvFree(struct ncclConnector* recv) {
  return ncclSuccess;
}

struct ncclTransport mcTransport = {
  "MC",
  mcCanConnect,
  { NULL, NULL, mcSendFree, NULL, NULL, NULL, NULL, NULL },
  { NULL, NULL, mcRecvFree, NULL, NULL, NULL, NULL, NULL }
};

#define CU_INIT_UUID_STATIC
#include <cuda_etbl/multicast.h>
#undef CU_INIT_UUID_STATIC

#define pfn_cuMemDeviceSupportsMulticast etblMulticast->DeviceSupportsMulticast
#define pfn_cuMemMulticastCreate etblMulticast->MulticastCreate
#define pfn_cuMemMulticastBindMem etblMulticast->MulticastBindMem

static const CUetblMulticast *etblMulticast = NULL;

pthread_mutex_t mcInitLock = PTHREAD_MUTEX_INITIALIZER;

static ncclResult_t ncclMcInitEtbl(struct ncclComm* comm) {
  comm->mcSupport = 0;

  pthread_mutex_lock(&mcInitLock);
  if (etblMulticast == NULL) {
    if (ncclCudaLibraryInit() != ncclSuccess) return ncclSuccess;
    if (pfn_cuGetExportTable((const void **)&etblMulticast, &CU_ETID_Multicast) != CUDA_SUCCESS) {
      pthread_mutex_unlock(&mcInitLock);
      return ncclSuccess;
    }
  }
  pthread_mutex_unlock(&mcInitLock);

#if USE_POSIX_FD
  {
    // Check for WAR for 3818216
    char *env;
    if ((env = getenv("CUDA_e0371668")) == NULL || atoi(env) != 1) {
      WARN("Need to 'export CUDA_e0371668=1' in the environment for MC support");
      return ncclSuccess;
    }
  }
#endif

  if (etblMulticast == NULL ||
      pfn_cuMemMulticastCreate == NULL ||
      pfn_cuMemMulticastBindMem == NULL ||
      pfn_cuMemDeviceSupportsMulticast == NULL)
    return ncclSuccess;

  int dev;
  CUCHECK(cuCtxGetDevice(&dev));
  CUCHECK(cuMemDeviceSupportsMulticast(&comm->mcSupport, dev));
  INFO(NCCL_INIT, "MC ETBL functions loaded, MC support %savailable", comm->mcSupport ? "" : "not ");
  return ncclSuccess;
}

#define MC_HANDLE_SIZE 64

typedef CUmemGenericAllocationHandle mcHandle_t; //TODO

struct mcResources {
  CUmemAllocationProp properties;
  CUmemAccessDesc accessDesc;
  size_t size;
  size_t granularity;
  size_t sizeWAR; // For 3418538 WAR
  CUmemGenericAllocationHandle mcHandle; // Multicast handle for MC buffer
  char* mcBuff; // Multicast MC buffer address
  CUmemGenericAllocationHandle ucHandle; // Unicast Handle for MC buffer
  char* ucBuff; // Unicast MC buffer address
};


ncclResult_t mcGetProperties(struct ncclComm *comm, struct mcResources* resources, size_t size) {
  int dev;
  CUCHECK(cuCtxGetDevice(&dev));

  CUmemAllocationProp* prop = &resources->properties;
  memset(prop, 0, sizeof(*prop));
  prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop->location.id = dev;
  prop->requestedHandleTypes = MC_CU_MEM_HANDLE_TYPE;

  CUCHECK(cuMemGetAllocationGranularity(&resources->granularity, prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

  ALIGN_SIZE(size, resources->granularity);
  resources->size = size;
  ALIGN_SIZE(size, 512ULL*1024*1024); // Align up until 3418538 fixed
  resources->sizeWAR = size;

  memset(&resources->accessDesc, 0, sizeof(resources->accessDesc));
  resources->accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  resources->accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  resources->accessDesc.location.id = dev;

  return ncclSuccess;
}

ncclResult_t mcGroupCreate(struct ncclComm *comm, struct mcResources* resources, int rank, unsigned int nranks, char* shareableHandle) {
  size_t size = resources->sizeWAR;

  // Create MC group
  multicastObjectProp prop = { 0 };
  prop.size = size;
  prop.numDevices = nranks;
  prop.requestedHandleTypes = MC_CU_MEM_HANDLE_TYPE;
  prop.flags = 0;

  INFO(NCCL_MC, "MC Creating group nranks %d size %zi on rank %d", nranks, size, rank);
  CUCHECK(cuMemMulticastCreate(&resources->mcHandle, &prop));

  if (MC_CU_MEM_HANDLE_TYPE != CU_MEM_HANDLE_TYPE_NONE) {
    // Get a handle to pass to other ranks
    CUCHECK(cuMemExportToShareableHandle(shareableHandle, resources->mcHandle, MC_CU_MEM_HANDLE_TYPE, 0));
  }
  else {
    memcpy(shareableHandle, &resources->mcHandle, sizeof(resources->mcHandle));
  }

  INFO(NCCL_MC, "MC Created group %llx nranks %d size %zi on rank %d", resources->mcHandle, nranks, size, rank);

  return ncclSuccess;
}

ncclResult_t mcGroupDestroy(mcHandle_t handle) {
  // TODO: Destroy an MC group
  return ncclSuccess;
}

ncclResult_t mcGroupConnect(struct ncclComm *comm, struct mcResources* resources, int rank, char* shareableHandle) {
  CUmemAllocationHandleType type = MC_CU_MEM_HANDLE_TYPE;

  INFO(NCCL_MC, "Importing MC shareableHandle %p from rank %d", shareableHandle, rank);

  // Import and map the remote memory descriptor to the local GPU
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // cuMem UDS support
    int fd = *(int *)shareableHandle;
    TRACE(NCCL_MC, "MC rank %d Importing shareable handle from rank %d fd %d", comm->localRank, rank, fd);
    // cuMem API support
    struct ncclProxyConnector proxyConn;
    NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 1, rank, &proxyConn));
    INFO(NCCL_MC, "MC rank %d request conversion of fd %d from rank %d", comm->localRank, fd, rank);
    NCCLCHECK(ncclProxyCall(&proxyConn, ncclProxyMsgConvertFd, shareableHandle, sizeof(int), &fd, sizeof(int)));
    INFO(NCCL_MC, "MC rank %d received converted fd %d from rank %d", comm->localRank, fd, rank);
    CUCHECK(cuMemImportFromShareableHandle(&resources->mcHandle, (void *)(uintptr_t)fd, type));
  } else {
    if (MC_CU_MEM_HANDLE_TYPE != CU_MEM_HANDLE_TYPE_NONE) {
      CUCHECK(cuMemImportFromShareableHandle(&resources->mcHandle, (void *)shareableHandle, type));
    } else {
      memcpy(&resources->mcHandle, shareableHandle, sizeof(resources->mcHandle));
    }
  }
  return ncclSuccess;
}

ncclResult_t mcGroupDisconnect(mcHandle_t handle) {
  // TODO: Disconnect to an MC group created by another rank
  return ncclSuccess;
}

ncclResult_t mcGroupBindMem(struct ncclComm *comm, struct mcResources* resources) {
  size_t size = resources->size;
  CUdeviceptr ptr = 0;

  // Map a VA for UC memory
  CUCHECK(cuMemAddressReserve(&ptr, size, resources->granularity, 0U, 0));

  // Alloc local physical mem for this MC group
  CUCHECK(cuMemCreate(&resources->ucHandle, size, &resources->properties, 0));
  CUCHECK(cuMemMap(ptr, size, 0, resources->ucHandle, 0));
  CUCHECK(cuMemSetAccess(ptr, size, &resources->accessDesc, 1));
  CUDACHECK(cudaMemset((void*)ptr, 0, size));
  resources->ucBuff = (char*)ptr;
  INFO(NCCL_MC, "MC Mapped UC at %p size %zi", resources->ucBuff, size);

  // Bind physical memory to the MC group
  INFO(NCCL_MC, "MC Bind mem %p UC handle %llx MC handle %llx size %zi", (void*)ptr, resources->ucHandle, resources->mcHandle, size);
  CUCHECK(cuMemMulticastBindMem(resources->mcHandle, 0, resources->ucHandle, 0, size, 0));

  return ncclSuccess;
}

ncclResult_t mcGroupMapMem(struct ncclComm *comm, struct mcResources* resources) {
  size_t size = resources->sizeWAR;
  CUdeviceptr ptr = 0;

  // Create a VA for the MC
  CUCHECK(cuMemAddressReserve(&ptr, size, resources->granularity, 0U, 0));
  // Map the VA locally
  CUCHECK(cuMemMap(ptr, size, 0, resources->mcHandle, 0));
  resources->mcBuff = (char*)ptr;
  INFO(NCCL_MC, "MC Mapped MC at %p size %zi", resources->mcBuff, size);

  // Having completed the BindMem we can now call SetAccess
  // NB: It will block until all ranks have bound to the Group
  INFO(NCCL_MC, "MC SetAccess MC %p size %zi", resources->mcBuff, size);
  CUCHECK(cuMemSetAccess((CUdeviceptr)resources->mcBuff, size, &resources->accessDesc, 1));
  INFO(NCCL_MC, "MC SetAccess MC %p size %zi - DONE", resources->mcBuff, size);

  return ncclSuccess;
}

ncclResult_t mcGroupUnbindMem(mcHandle_t handle, char* mem) {
  // TODO: Free `mem` and unbind it from the mem handle
  return ncclSuccess;
}

#include "bootstrap.h"

#define MC_MEM_ALIGN_SIZE (1 << 21)

ncclResult_t ncclMcSetup(struct ncclComm* comm) {
  int nHeads = comm->channels[0].mc.nHeads;
  int headRank = comm->channels[0].mc.headRank;

  NCCLCHECK(ncclMcInitEtbl(comm));
  if (comm->mcSupport == 0 || comm->localRanks <= 1 || nHeads == 0) return ncclSuccess;

  ncclResult_t res = ncclSuccess;
  struct mcResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  comm->mcResources = resources;

  size_t buffSize = comm->buffSizes[NCCL_PROTO_SIMPLE];
  size_t memSize = MC_MEM_ALIGN_SIZE;
  size_t mcPerRankSize = comm->nChannels*2*(buffSize+memSize);
  size_t mcTotalSize = mcPerRankSize*nHeads;

  INFO(NCCL_INIT|NCCL_MC, "MC comm %p headRank %d nHeads %d buffSize %zi memSize %zi mcPerRankSize %zi mcTotalSize %zi",
       comm, headRank, nHeads, buffSize, memSize, mcPerRankSize, mcTotalSize);

  char* mcShareableHandle = NULL;
  NCCLCHECKGOTO(ncclCalloc(&mcShareableHandle, MC_HANDLE_SIZE), res, cleanup);
  NCCLCHECKGOTO(mcGetProperties(comm, resources, mcTotalSize), res, cleanup);
  if (comm->localRank == 0) {
    NCCLCHECKGOTO(mcGroupCreate(comm, resources, comm->localRank, comm->localRanks, mcShareableHandle), res, cleanup);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, mcShareableHandle, MC_HANDLE_SIZE), res, cleanup);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, mcShareableHandle, MC_HANDLE_SIZE), res, cleanup);
    NCCLCHECKGOTO(mcGroupConnect(comm, resources, comm->localRankToRank[0], mcShareableHandle), res, cleanup);
  }

  NCCLCHECKGOTO(mcGroupBindMem(comm, resources), res, cleanup);
  // Local intra-node barrier to ensure everyone has bound their memory to the group
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), res, cleanup);
  NCCLCHECKGOTO(mcGroupMapMem(comm, resources), res, cleanup);

#if 0
  {
    uint64_t dummy[1024];

    for (int i=0; i < sizeof(dummy)/sizeof(dummy[0]); i++) {
      dummy[i] = 0xdeadbabefeedface ^ i;
      dummy[i] ^= (headRank << 28);
    }

    printf("MC: headRank %d Writing %zi bytes to %p\n", headRank, sizeof(dummy), resources->mcBuff);
    cudaMemcpy(resources->mcBuff, dummy, sizeof(dummy), cudaMemcpyHostToDevice);

    CUDACHECK(cudaDeviceSynchronize());
    NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), res, cleanup);

    for (int h = 0; h < nHeads; h++) {
      char *buf = resources->ucBuff + h*mcPerRankSize;
      cudaMemcpy(&dummy[0], buf, sizeof(dummy), cudaMemcpyDeviceToHost);
      printf("MC: headRank %d.%d UC data %p %lx %lx %lx %lx\n", headRank, h, buf, dummy[0], dummy[1], dummy[2], dummy[3]);
      printf("MC: headRank %d.%d UC data %p %lx %lx %lx %lx\n", headRank, h, buf, dummy[1020], dummy[1021], dummy[1022], dummy[1023]);
    }
  }
#endif

  for (int h=0; h<nHeads; h++) {
    int mcPeer = comm->nRanks+1+h;
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;

      char* mem = NULL;
      struct ncclChannelPeer* peer = channel->peers+mcPeer;

      // Reduce UC -> MC
      mem = resources->ucBuff + (h*2*comm->nChannels+c)*(buffSize+memSize);
      peer->send[0].transportComm = &mcTransport.send;
      peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->send[0].conn.head = (uint64_t*)(mem+buffSize);
      peer->send[0].conn.tail = (uint64_t*)(mem+buffSize+memSize/2);
      mem = resources->mcBuff + (h*2*comm->nChannels+c)*(buffSize+memSize);
      peer->recv[1].transportComm = &mcTransport.recv;
      peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->recv[1].conn.head = (uint64_t*)(mem+buffSize);
      peer->recv[1].conn.tail = (uint64_t*)(mem+buffSize+memSize/2);
      peer->recv[1].conn.flags |= NCCL_MC_MIN_POLL;

      // Broadcast MC -> UC
      mem = resources->ucBuff + ((h*2+1)*comm->nChannels+c)*(buffSize+memSize);
      peer->recv[0].transportComm = &mcTransport.recv;
      peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->recv[0].conn.head = (uint64_t*)(mem+buffSize);
      peer->recv[0].conn.tail = (uint64_t*)(mem+buffSize+memSize/2);
      mem = resources->mcBuff + ((h*2+1)*comm->nChannels+c)*(buffSize+memSize);
      peer->send[1].transportComm = &mcTransport.send;
      peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->send[1].conn.head = (uint64_t*)(mem+buffSize);
      peer->send[1].conn.tail = (uint64_t*)(mem+buffSize+memSize/2);
      peer->send[1].conn.flags |= NCCL_MC_MIN_POLL;

      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[mcPeer].send[0], &peer->send[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[mcPeer].recv[0], &peer->recv[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[mcPeer].send[1], &peer->send[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[mcPeer].recv[1], &peer->recv[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);

      /*INFO(NCCL_INIT|NCCL_MC, "Peer %d Channel %d MC buff %p/%p UC Buff %p/%p",
          mcPeer, c,
          resources->mcBuff + (h*2*comm->nChannels+c)*(buffSize+memSize),
          resources->mcBuff + ((h*2+1)*comm->nChannels+c)*(buffSize+memSize),
          resources->ucBuff + (h*2*comm->nChannels+c)*(buffSize+memSize),
          resources->ucBuff + ((h*2+1)*comm->nChannels+c)*(buffSize+memSize));*/
    }
  }

cleanup:
  free(mcShareableHandle);
  return res;
}

ncclResult_t ncclMcFree(struct ncclComm* comm) {
  struct mcResources* resources = (struct mcResources*)comm->mcResources;
  if (resources == NULL) return ncclSuccess;
  NCCLCHECK(mcGroupUnbindMem(resources->mcHandle, resources->ucBuff));
  NCCLCHECK(mcGroupDisconnect(resources->mcHandle));
  if (comm->localRank == 0) NCCLCHECK(mcGroupDestroy(resources->mcHandle));
  free(resources);
  comm->mcResources = NULL;
  return ncclSuccess;
}
