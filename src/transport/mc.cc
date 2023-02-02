/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "proxy.h"

#if CUDART_VERSION >= 12010

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

#define MC_HANDLE_SIZE 64

struct mcResources {
  CUmemAllocationProp properties;
  CUmemAccessDesc accessDesc;
  int dev;
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
  CUmulticastObjectProp mcProp = { 0 };
  mcProp.size = size;
  mcProp.numDevices = nranks;
  mcProp.handleTypes = MC_CU_MEM_HANDLE_TYPE;
  mcProp.flags = 0;

  INFO(NCCL_MC, "MC Creating group nranks %d size %zi on rank %d", nranks, size, rank);
  CUCHECK(cuMulticastCreate(&resources->mcHandle, &mcProp));

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

ncclResult_t mcGroupAddDevice(struct ncclComm *comm, struct mcResources* resources, int dev) {
  INFO(NCCL_MC, "MC group %llx adding dev %d", resources->mcHandle, dev);
  CUCHECK(cuMulticastAddDevice(resources->mcHandle, dev));
  resources->dev = dev;
  return ncclSuccess;
}

ncclResult_t mcGroupUnbind(struct ncclComm *comm, struct mcResources* resources) {
  int dev = resources->dev;
  size_t size = resources->sizeWAR;
  INFO(NCCL_MC, "MC Unbind MC handle %llx size %zi dev %d", resources->mcHandle, size, dev);

  // Unbind physical memory from group for the given device
  CUCHECK(cuMulticastUnbind(resources->mcHandle, dev, 0/*mcOffset*/, size));

  return ncclSuccess;
}

ncclResult_t mcGroupConnect(struct ncclComm *comm, struct mcResources* resources, int rank, char* shareableHandle) {
  CUmemAllocationHandleType type = MC_CU_MEM_HANDLE_TYPE;

  INFO(NCCL_MC, "MC importing shareableHandle %p from rank %d", shareableHandle, rank);

  // Import and map the remote memory descriptor to the local GPU
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // cuMem UDS support
    int fd = *(int *)shareableHandle;
    TRACE(NCCL_MC, "MC rank %d Importing shareable handle from rank %d fd %d", comm->localRank, rank, fd);
    // cuMem API support
    struct ncclProxyConnector proxyConn;
    NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 1, rank, &proxyConn));
    INFO(NCCL_MC, "MC rank %d request conversion of fd %d from rank %d", comm->localRank, fd, rank);
    NCCLCHECK(ncclProxyCallBlocking(&proxyConn, ncclProxyMsgConvertFd, shareableHandle, sizeof(int), &fd, sizeof(int)));
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
  // NB: It will block until all ranks have been added to the Group
  INFO(NCCL_MC, "MC Bind mem %p UC handle 0x%llx MC handle 0x%llx size %zi", (void*)ptr, resources->ucHandle, resources->mcHandle, size);
  CUCHECK(cuMulticastBindMem(resources->mcHandle, 0/*mcOffset*/, resources->ucHandle, 0/*memOffset*/, size, 0/*flags*/));

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

  return ncclSuccess;
}

ncclResult_t mcGroupUnmapMem(struct ncclComm *comm, struct mcResources* resources) {
  size_t size;
  CUdeviceptr ptr;
  INFO(NCCL_MC, "MC Unmap mem UC handle 0x%llx(%p) MC handle 0x%llx(%p)",
       resources->ucHandle, resources->ucBuff, resources->mcHandle, resources->mcBuff);

  // Release the UC memory and mapping
  ptr = (CUdeviceptr)resources->ucBuff;
  size = resources->size;
  CUCHECK(cuMemUnmap(ptr, size));
  CUCHECK(cuMemAddressFree(ptr, size));
  CUCHECK(cuMemRelease(resources->ucHandle));

  // Release the MC memory and mapping
  ptr = (CUdeviceptr)resources->mcBuff;
  size = resources->sizeWAR;
  CUCHECK(cuMemUnmap(ptr, size));
  CUCHECK(cuMemAddressFree(ptr, size));
  CUCHECK(cuMemRelease(resources->mcHandle));

  return ncclSuccess;
}

#include "bootstrap.h"
#include "channel.h"

#define MC_MEM_ALIGN_SIZE (1 << 21)

NCCL_PARAM(McChannels, "MC_NCHANNELS", 16);

NCCL_PARAM(McEnable, "MC_ENABLE", 1);

ncclResult_t ncclMcSetup(struct ncclComm* comm) {
  if (!ncclParamMcEnable() || comm->localRanks <= 1 || comm->nNodes>1) return ncclSuccess;
  int dev, driverVersion;
  CUCHECK(cuCtxGetDevice(&dev));
  CUDACHECK(cudaDriverGetVersion(&driverVersion));
  comm->mcSupport = 0;
  // MC Multicast support requires CUDA12.1 UMD + KMD
  if (pfn_cuMulticastCreate != NULL && driverVersion >= 12010) {
    CUCHECK(cuDeviceGetAttribute(&comm->mcSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
  }
  INFO(NCCL_INIT, "MC multicast support is %savailable on dev %d", comm->mcSupport ? "" : "not ", dev);
  if (comm->mcSupport == 0) return ncclSuccess;

  int nChannels = comm->mcChannels = ncclParamMcChannels();
  int rank = comm->localRank, nranks = comm->localRanks;

  for (int c=0; c<nChannels; c++) {
    NCCLCHECK(initChannel(comm, c));
  }
  ncclResult_t res = ncclSuccess;
  struct mcResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  comm->mcResources = resources;

  size_t buffSize = comm->buffSizes[NCCL_PROTO_SIMPLE];
  size_t memSize = MC_MEM_ALIGN_SIZE;
  size_t mcPerRankSize = nChannels*2*(buffSize+memSize);
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

  NCCLCHECKGOTO(mcGroupAddDevice(comm, resources, dev), res, cleanup);
  NCCLCHECKGOTO(mcGroupBindMem(comm, resources), res, cleanup);
  // Local intra-node barrier to ensure everyone has bound their memory to the group
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), res, cleanup);
  NCCLCHECKGOTO(mcGroupMapMem(comm, resources), res, cleanup);

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->mc.nHeads = nranks;
    for (int i=0; i<NCCL_MAX_MC_ARITY; i++) channel->mc.up[i] = -1;
    channel->mc.down = comm->nRanks+1+comm->localRank;
    channel->mc.out = -1;       // Network not yet implemented.
    channel->mc.headRank = comm->localRank;  // Network not yet implemented.
  }

  for (int r=0; r<nranks; r++) {
    int mcPeer = comm->nRanks+1+r;
    for (int c=0; c<nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      channel->mc.up[r] = mcPeer;

      char* mem = NULL;
      struct ncclChannelPeer* peer = channel->peers+mcPeer;

      // Reduce UC -> MC
      mem = resources->ucBuff + (r*2*nChannels+c)*(buffSize+memSize);
      peer->send[0].transportComm = &mcTransport.send;
      peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->send[0].conn.head = (uint64_t*)(mem+buffSize);
      peer->send[0].conn.tail = (uint64_t*)(mem+buffSize+memSize/2);
      mem = resources->mcBuff + (r*2*nChannels+c)*(buffSize+memSize);
      peer->recv[1].transportComm = &mcTransport.recv;
      peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->recv[1].conn.head = (uint64_t*)(mem+buffSize);
      peer->recv[1].conn.tail = (uint64_t*)(mem+buffSize+memSize/2);
      peer->recv[1].conn.flags |= NCCL_MC_MIN_POLL;

      // Broadcast MC -> UC
      mem = resources->ucBuff + ((r*2+1)*nChannels+c)*(buffSize+memSize);
      peer->recv[0].transportComm = &mcTransport.recv;
      peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->recv[0].conn.head = (uint64_t*)(mem+buffSize);
      peer->recv[0].conn.tail = (uint64_t*)(mem+buffSize+memSize/2);
      mem = resources->mcBuff + ((r*2+1)*nChannels+c)*(buffSize+memSize);
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
          resources->mcBuff + (r*2*nChannels+c)*(buffSize+memSize),
          resources->mcBuff + ((r*2+1)*nChannels+c)*(buffSize+memSize),
          resources->ucBuff + (r*2*nChannels+c)*(buffSize+memSize),
          resources->ucBuff + ((r*2+1)*nChannels+c)*(buffSize+memSize));*/
    }
  }

  free(mcShareableHandle);
  return res;

cleanup:
  comm->mcSupport = 0;
  free(mcShareableHandle);
  return res;
}

ncclResult_t ncclMcFree(struct ncclComm* comm) {
  struct mcResources* resources = (struct mcResources*)comm->mcResources;
  if (resources == NULL) return ncclSuccess;
  NCCLCHECK(mcGroupUnbind(comm, resources));
  NCCLCHECK(mcGroupUnmapMem(comm, resources));
  free(resources);
  comm->mcResources = NULL;
  return ncclSuccess;
}

#else

/*
 * Pre CUDA 12.1 stubs
 */

ncclResult_t ncclMcSetup(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclMcFree(struct ncclComm* comm) {
  return ncclSuccess;
}

#endif /* CUDA_VERSION >= 12010 */
