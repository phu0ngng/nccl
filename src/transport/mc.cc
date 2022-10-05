/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "proxy.h"

//#define MC_CU_MEM_HANDLE_TYPE CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
#define MC_CU_MEM_HANDLE_TYPE CU_MEM_HANDLE_TYPE_NONE

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

static ncclResult_t ncclMcInitEtbl(struct ncclComm* comm) {
  comm->mcSupport = 0;
  if (ncclCudaLibraryInit() != ncclSuccess) return ncclSuccess;
  if (pfn_cuGetExportTable((const void **)&etblMulticast, &CU_ETID_Multicast) != CUDA_SUCCESS)
    return ncclSuccess;

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

  resources->granularity = 512ULL*1024ULL*1024ULL; // HACK until 3418538 fixed

  ALIGN_SIZE(size, resources->granularity);
  resources->size = size;

  memset(&resources->accessDesc, 0, sizeof(resources->accessDesc));
  resources->accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  resources->accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  resources->accessDesc.location.id = dev;

  return ncclSuccess;
}

ncclResult_t mcGroupCreate(struct ncclComm *comm, struct mcResources* resources, int rank, unsigned int nranks, char* shareableHandle) {
  size_t size = resources->size;

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
    fd = *(int *)shareableHandle;
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

ncclResult_t mcGroupBindMem(struct ncclComm *comm, struct mcResources* resources, int rank, size_t mcPerRankSize) {
  size_t size = resources->size;
  CUdeviceptr ptr = 0;

  INFO(NCCL_MC, "MC BindMem comm %p rank %d mcPerRankSize %zi", comm, rank, mcPerRankSize);

  // Map a VA for UC memory
  CUCHECK(cuMemAddressReserve(&ptr, size, resources->granularity, 0U, 0));

  // Alloc local physical mem for this MC group
  CUCHECK(cuMemCreate(&resources->ucHandle, size, &resources->properties, 0));
  CUCHECK(cuMemMap(ptr, size, 0, resources->ucHandle, 0));
  CUCHECK(cuMemSetAccess(ptr, size, &resources->accessDesc, 1));
  CUDACHECK(cudaMemset((void*)ptr, 0, size));
  resources->ucBuff = (char*)ptr;
  INFO(NCCL_MC, "MC Mapped UC at %p rank %d", resources->ucBuff, rank);

  // Bind physical memory to the MC group
  INFO(NCCL_MC, "MC Binding local mem %p handle %llx size %zi to MC handle %llx for rank %d", (void*)ptr, resources->ucHandle, size, resources->mcHandle, rank);
  CUCHECK(cuMemMulticastBindMem(resources->mcHandle, 0, resources->ucHandle, 0, size, 0));

  // Create a VA for the MC
  CUCHECK(cuMemAddressReserve(&ptr, size, resources->granularity, 0U, 0));
  // Map the VA locally
  CUCHECK(cuMemMap(ptr, size, 0, resources->mcHandle, 0));
  resources->mcBuff = (char*)ptr;
  INFO(NCCL_MC, "MC Mapped MC at %p rank %d", resources->mcBuff, rank);
  return ncclSuccess;
}

ncclResult_t mcGroupAccessMem(struct ncclComm *comm, struct mcResources* resources) {
  // Having completed the BindMem we can now call SetAccess
  // NB: It will block until all ranks have bound to the Group
  INFO(NCCL_MC, "MC SetAccess MC %p size %zi", resources->mcBuff, resources->size);
  CUCHECK(cuMemSetAccess((CUdeviceptr)resources->mcBuff, resources->size, &resources->accessDesc, 1));
  INFO(NCCL_MC, "MC SetAccess MC %p size %zi - DONE", resources->mcBuff, resources->size);

  return ncclSuccess;
}

ncclResult_t mcGroupUnbindMem(mcHandle_t handle, char* mem) {
  // TODO: Free `mem` and unbind it from the mem handle
  return ncclSuccess;
}

#include "bootstrap.h"

NCCL_PARAM(McBuffSize, "MC_BUFFSIZE", (1UL<<22));
#define MC_MEM_ALIGN_SIZE (1 << 21)

ncclResult_t ncclMcSetup(struct ncclComm* comm) {
  NCCLCHECK(ncclMcInitEtbl(comm));
  if (comm->mcSupport == 0 || comm->localRanks <= 1) return ncclSuccess;

  int rank = comm->localRank, nranks = comm->localRanks;
  ncclResult_t res = ncclSuccess;
  struct mcResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  comm->mcResources = resources;

  size_t buffSize = comm->mcBuffSize = ncclParamMcBuffSize();
  size_t memSize = 2*sizeof(uint64_t);
  ALIGN_SIZE(buffSize, MC_MEM_ALIGN_SIZE);
  ALIGN_SIZE(memSize, MC_MEM_ALIGN_SIZE);
  size_t mcPerRankSize = comm->nChannels*2*(buffSize+memSize);
  size_t mcTotalSize = mcPerRankSize*nranks;

  INFO(NCCL_INIT|NCCL_MC, "MC comm %p rank %d nranks %d buffSize %zi memSize %zi mcPerRankSize %zi mcTotalSize %zi",
       comm, rank, nranks, buffSize, memSize, mcPerRankSize, mcTotalSize);

  char* mcShareableHandle = NULL;
  NCCLCHECKGOTO(ncclCalloc(&mcShareableHandle, MC_HANDLE_SIZE), res, cleanup);
  NCCLCHECKGOTO(mcGetProperties(comm, resources, mcTotalSize), res, cleanup);
  if (rank == 0) {
    NCCLCHECKGOTO(mcGroupCreate(comm, resources, rank, nranks, mcShareableHandle), res, cleanup);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, rank, nranks, 0, mcShareableHandle, MC_HANDLE_SIZE), res, cleanup);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, rank, nranks, 0, mcShareableHandle, MC_HANDLE_SIZE), res, cleanup);
    NCCLCHECKGOTO(mcGroupConnect(comm, resources, 0, mcShareableHandle), res, cleanup);
  }

  NCCLCHECKGOTO(mcGroupBindMem(comm, resources, rank, mcPerRankSize), res, cleanup);
  // Local intra-node barrier to ensure everyone has bound their memory to the group
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), res, cleanup);
  NCCLCHECKGOTO(mcGroupAccessMem(comm, resources), res, cleanup);

#if 0
  {
    uint64_t dummy[1024];

    for (int i=0; i < sizeof(dummy)/sizeof(dummy[0]); i++) {
      dummy[i] = 0xdeadbabefeedface ^ i;
      dummy[i] ^= (rank << 28);
    }

    printf("MC: rank %d Writing %zi bytes to %p\n", rank, sizeof(dummy), resources->mcBuff);
    cudaMemcpy(resources->mcBuff, dummy, sizeof(dummy), cudaMemcpyHostToDevice);

    CUDACHECK(cudaDeviceSynchronize());
    NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), res, cleanup);

    for (int r = 0; r < nranks; r++) {
      char *buf = resources->ucBuff + r*mcPerRankSize;
      cudaMemcpy(&dummy[0], buf, sizeof(dummy), cudaMemcpyDeviceToHost);
      printf("MC: rank %d.%d UC data %p %lx %lx %lx %lx\n", rank, r, buf, dummy[0], dummy[1], dummy[2], dummy[3]);
      printf("MC: rank %d.%d UC data %p %lx %lx %lx %lx\n", rank, r, buf, dummy[1020], dummy[1021], dummy[1022], dummy[1023]);
    }
  }
#endif

  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->mc.nHeads = nranks;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) channel->mc.down[i] = channel->mc.up[i] = -1;
    channel->mc.down[0] = comm->nRanks+1+comm->localRank;
    channel->mc.out = -1;       // Network not yet implemented.
    channel->mc.headRank = -1;  // Network not yet implemented.
    channel->mc.shift = 0; // We don't need to shuffle communication, we're not doing an alltoall
    channel->mc.depth = 0;
  }

  for (int r=0; r<nranks; r++) {
    int mcPeer = comm->nRanks+1+r;
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      channel->mc.up[r] = mcPeer;

      char* mem = NULL;
      struct ncclChannelPeer* peer = channel->peers+mcPeer;

      // Reduce UC -> MC
      mem = resources->ucBuff + (r*2*comm->nChannels+c)*(buffSize+memSize);
      peer->send[0].transportComm = &mcTransport.send;
      peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->send[0].conn.head = (uint64_t*)(mem+buffSize);
      peer->send[0].conn.tail = (uint64_t*)(mem+buffSize+sizeof(uint64_t));
      mem = resources->mcBuff + (r*2*comm->nChannels+c)*(buffSize+memSize);
      peer->recv[1].transportComm = &mcTransport.recv;
      peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->recv[1].conn.head = (uint64_t*)(mem+buffSize);
      peer->recv[1].conn.tail = (uint64_t*)(mem+buffSize+sizeof(uint64_t));
      peer->recv[1].conn.flags |= NCCL_MC_MIN_POLL;

      // Broadcast MC -> UC
      mem = resources->ucBuff + ((r*2+1)*comm->nChannels+c)*(buffSize+memSize);
      peer->recv[0].transportComm = &mcTransport.recv;
      peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->recv[0].conn.head = (uint64_t*)(mem+buffSize);
      peer->recv[0].conn.tail = (uint64_t*)(mem+buffSize+sizeof(uint64_t));
      mem = resources->mcBuff + ((r*2+1)*comm->nChannels+c)*(buffSize+memSize);
      peer->send[1].transportComm = &mcTransport.send;
      peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->send[1].conn.head = (uint64_t*)(mem+buffSize);
      peer->send[1].conn.tail = (uint64_t*)(mem+buffSize+sizeof(uint64_t));
      peer->send[1].conn.flags |= NCCL_MC_MIN_POLL;

      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[mcPeer].send[0], &peer->send[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[mcPeer].recv[0], &peer->recv[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[mcPeer].send[1], &peer->send[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[mcPeer].recv[1], &peer->recv[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
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
