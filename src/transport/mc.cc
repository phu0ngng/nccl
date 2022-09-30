/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"

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
  CUmemGenericAllocationHandle* mcHandles; // Multicast handles of each head rank
  char* mcBuff; // Multicast region mapped
  CUmemGenericAllocationHandle* buffHandles; // Handles for my buffers for each MC
  char** buffs; // Buffers for each MC
};


ncclResult_t mcGetProperties(struct mcResources* resources, size_t size) {
  int dev;
  CUCHECK(cuCtxGetDevice(&dev));

  CUmemAllocationProp* prop = &resources->properties;
  prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop->location.id = dev;
  prop->requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  CUCHECK(cuMemGetAllocationGranularity(&resources->granularity, prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  ALIGN_SIZE(size, resources->granularity);
  resources->size = size;

  resources->accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  resources->accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  resources->accessDesc.location.id = dev;

  return ncclSuccess;
}

ncclResult_t mcGroupCreate(struct mcResources* resources, int rank, unsigned int nranks, char* shareableHandle) {
  size_t size = resources->size;
  // Create MC group
  multicastObjectProp prop = { .size = resources->size, .numDevices = nranks };
  CUCHECK(cuMemMulticastCreate(resources->mcHandles+rank, &prop));

  // Get a handle to pass to other ranks
  CUCHECK(cuMemExportToShareableHandle(shareableHandle, resources->mcHandles[rank], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

  // Map the MC locally
  CUdeviceptr ptr;
  CUCHECK(cuMemAddressReserve(&ptr, size, resources->granularity, 0, 0));
  CUCHECK(cuMemMap(ptr, size, 0, resources->mcHandles[rank], 0));
  CUCHECK(cuMemSetAccess(ptr, size, &resources->accessDesc, 1));
  resources->mcBuff = (char*)ptr;
  return ncclSuccess;
}

ncclResult_t mcGroupDestroy(mcHandle_t handle) {
  // TODO: Destroy an MC group
  return ncclSuccess;
}

ncclResult_t mcGroupConnect(struct mcResources* resources, int rank, char* shareableHandle) {
  CUCHECK(cuMemImportFromShareableHandle(resources->mcHandles+rank,
        (void *)shareableHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  return ncclSuccess;
}

ncclResult_t mcGroupDisconnect(mcHandle_t handle) {
  // TODO: Disconnect to an MC group created by another rank
  return ncclSuccess;
}

ncclResult_t mcGroupBindMem(struct mcResources* resources, int rank) {
  // Alloc mem
  size_t size = resources->size;
  CUdeviceptr ptr;
  CUCHECK(cuMemCreate(resources->buffHandles+rank, size, &resources->properties, 0));
  CUCHECK(cuMemAddressReserve(&ptr, size, resources->granularity, 0, 0));
  CUCHECK(cuMemMap(ptr, size, 0, resources->buffHandles[rank], 0));
  CUCHECK(cuMemSetAccess(ptr, size, &resources->accessDesc, 1));
  resources->buffs[rank] = (char*)ptr;

  // Bind to MC
  CUCHECK(cuMemMulticastBindMem(resources->mcHandles[rank], 0, resources->buffHandles[rank], 0, resources->size, 0));
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
  if (comm->mcSupport == 0) return ncclSuccess;

  int rank = comm->localRank, nranks = comm->localRanks;
  ncclResult_t res = ncclSuccess;
  struct mcResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  comm->mcResources = resources;

  int buffSize = ncclParamMcBuffSize();
  int memSize = 2*sizeof(uint64_t);
  ALIGN_SIZE(buffSize, MC_MEM_ALIGN_SIZE);
  ALIGN_SIZE(memSize, MC_MEM_ALIGN_SIZE);
  int mcTotalSize = comm->nChannels*(buffSize+memSize);

  char* mcShareableHandles = NULL;
  NCCLCHECKGOTO(ncclCalloc(&mcShareableHandles, nranks*MC_HANDLE_SIZE), res, cleanup);
  NCCLCHECKGOTO(ncclCalloc(&resources->mcHandles, nranks), res, cleanup);
  NCCLCHECKGOTO(ncclCalloc(&resources->buffHandles, nranks), res, cleanup);
  NCCLCHECKGOTO(ncclCalloc(&resources->buffs, nranks), res, cleanup);
  NCCLCHECKGOTO(mcGetProperties(resources, mcTotalSize), res, cleanup);
  NCCLCHECKGOTO(mcGroupCreate(resources, rank, nranks, mcShareableHandles+rank), res, cleanup);
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, rank, nranks, mcShareableHandles, MC_HANDLE_SIZE), res, cleanup);

  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->mc.nHeads = nranks;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) channel->mc.down[i] = channel->mc.up[i] = -1;
    channel->mc.down[0] = comm->nRanks+1+nranks;
    channel->mc.out = -1;       // Network not yet implemented.
    channel->mc.headRank = -1;  // Network not yet implemented.
    channel->mc.shift = 0; // We don't need to shuffle communication, we're not doing an alltoall
    channel->mc.depth = 0;
  }

  for (int r=0; r<nranks; r++) {
    if (r != rank) NCCLCHECKGOTO(mcGroupConnect(resources, r, mcShareableHandles+r*MC_HANDLE_SIZE), res, cleanup);
    NCCLCHECKGOTO(mcGroupBindMem(resources, r), res, cleanup);
    int mcPeer = comm->nRanks+1+r;
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      channel->mc.up[r] = mcPeer;

      char* mem = resources->buffs[r] + c*(buffSize+memSize);
      struct ncclChannelPeer* peer = channel->peers+mcPeer;

      peer->send[0].transportComm = &mcTransport.send;
      peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->send[0].conn.head = (uint64_t*)(mem+buffSize);
      peer->send[0].conn.tail = (uint64_t*)(mem+buffSize+sizeof(uint64_t));

      peer->recv[0].transportComm = &mcTransport.recv;
      peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->recv[0].conn.head = (uint64_t*)(mem+buffSize);
      peer->recv[0].conn.tail = (uint64_t*)(mem+buffSize+sizeof(uint64_t));
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[mcPeer].send[0], &peer->send[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[mcPeer].recv[0], &peer->recv[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
    }
  }
cleanup:
  free(mcShareableHandles);
  return res;
}

ncclResult_t ncclMcFree(struct ncclComm* comm) {
  struct mcResources* resources = (struct mcResources*)comm->mcResources;
  if (resources == NULL) return ncclSuccess;
  for (int r=0; r<comm->localRanks; r++) {
    if (resources->buffs[r]) NCCLCHECK(mcGroupUnbindMem(resources->mcHandles[r], resources->buffs[r]));
    NCCLCHECK(mcGroupDisconnect(resources->mcHandles[r]));
  }
  NCCLCHECK(mcGroupDestroy(resources->mcHandles[comm->localRank]));
  free(resources->mcHandles);
  free(resources->buffHandles);
  free(resources->buffs);
  free(resources);
  comm->mcResources = NULL;
  return ncclSuccess;
}
