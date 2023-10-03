/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "nccl.h"
#include "comm.h"
#include "net.h"

struct ncclReg {
  uintptr_t addr;
  size_t pages;
  int refs;
  int nComms;
  void** sComms;
  void** rComms;
  void** handles;
};

ncclResult_t ncclNetDeregister(struct ncclComm* comm, struct ncclReg* reg) {
  ncclDebugNoWarn = NCCL_NET;
  for (int d=0; d<reg->nComms; d++) {
    if (reg->handles[d] != NULL) NCCLCHECK(comm->ncclNet->deregMr(reg->sComms[d], reg->handles[d]));
    if (reg->sComms[d]) NCCLCHECK(comm->ncclNet->closeSend(reg->sComms[d]));
    if (reg->rComms[d]) NCCLCHECK(comm->ncclNet->closeRecv(reg->rComms[d]));
  }
  reg->nComms = 0;
  free(reg->sComms);
  free(reg->rComms);
  free(reg->handles);
  reg->sComms = reg->rComms = reg->handles = NULL;
  ncclDebugNoWarn = 0;
  return ncclSuccess;
}

ncclResult_t ncclNetRegister(struct ncclComm* comm, void* addr, size_t size, struct ncclReg* reg) {
  int netDevs;
  NCCLCHECK(comm->ncclNet->devices(&netDevs));

  int localNetDevCount = 0;
  int* localNetDevs;
  void *lComm = NULL;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECK(ncclCalloc(&localNetDevs, comm->p2pnChannels));

  // Find local devices for p2p operations
  for (int c=0; c<comm->p2pnChannels; c++) {
    int dev;
    if (ncclTopoGetLocalNet(comm->topo, comm->rank, c, &dev) != ncclSuccess) goto end; // No local net
    ncclNetProperties_t props;
    NCCLCHECKGOTO(comm->ncclNet->getProperties(dev, &props), ret, end);
    if (props.regIsGlobal == 0) { // We need to be sure all NICs support global registration.
      localNetDevCount = 0;
      break;
    }
    localNetDevs[localNetDevCount++] = dev;
  }

  NCCLCHECKGOTO(ncclCalloc(&reg->sComms, localNetDevCount), ret, end);
  NCCLCHECKGOTO(ncclCalloc(&reg->rComms, localNetDevCount), ret, end);
  NCCLCHECKGOTO(ncclCalloc(&reg->handles, localNetDevCount), ret, end);
  reg->nComms = localNetDevCount;

  ncclDebugNoWarn = NCCL_NET;
  for (int d=0; d<localNetDevCount; d++) {
    int dev = localNetDevs[d];
    reg->handles[d] = reg->sComms[d] = reg->rComms[d] = NULL;

    ncclNetHandle_t netHandle;
    NCCLCHECKGOTO(comm->ncclNet->listen(dev, &netHandle, &lComm), ret, end);

    bool connected;
    connected = false;
    while (!connected) {
      if (*comm->abortFlag) {
        goto end;
      }

      if (reg->sComms[d] == NULL)
        NCCLCHECKGOTO(comm->ncclNet->connect(dev, &netHandle, reg->sComms+d, NULL), ret, end);

      if (reg->rComms[d] == NULL)
        NCCLCHECKGOTO(comm->ncclNet->accept(lComm, reg->rComms+d, NULL), ret, end);

      connected = (reg->rComms[d] != NULL) && (reg->sComms[d] != NULL);
    }
    NCCLCHECK(comm->ncclNet->closeListen(lComm));
    lComm = NULL;

    if (comm->ncclNet->regMr(reg->sComms[d], addr, size, NCCL_PTR_CUDA, reg->handles+d) != ncclSuccess) {
      reg->handles[d] = NULL;
    }
  }
end:
  ncclDebugNoWarn = 0;
  if (ret != ncclSuccess) NCCLCHECK(ncclNetDeregister(comm, reg));
  free(localNetDevs);
  return ret;
}

ncclResult_t ncclRegFind(struct ncclComm* comm, void* data, size_t size, int* found) {
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);
  *found = 0;

  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;

  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot]->addr) return ncclSuccess;
    if ((addr >= cache->slots[slot]->addr) &&
        ((addr-cache->slots[slot]->addr)/pageSize+pages) <= cache->slots[slot]->pages) {
      *found = cache->slots[slot]->nComms ? 1 : 0;
      return ncclSuccess;
    }
  }
}
NCCL_PARAM(RegDisable, "REGISTER_DISABLE", 0);

ncclResult_t ncclRegister(struct ncclComm* comm, void* data, size_t size, void** handle) {
  if (ncclParamRegDisable()) return ncclSuccess;
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);

  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;
  for (int slot=0; /*true*/; slot++) {
    if ((slot == cache->population) || (addr < cache->slots[slot]->addr)) {
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclReg*));
      NCCLCHECK(ncclCalloc(cache->slots+slot, 1));
      struct ncclReg* regSlot = cache->slots[slot];
      regSlot->addr = addr;
      regSlot->pages = pages;
      regSlot->refs = 1;
      NCCLCHECK(ncclNetRegister(comm, (void*)addr, pages*pageSize, regSlot));
      cache->population += 1;
      *handle = regSlot;
      return ncclSuccess;
    } else if ((addr >= cache->slots[slot]->addr) &&
        ((addr-cache->slots[slot]->addr)/pageSize+pages) <= cache->slots[slot]->pages) {
      cache->slots[slot]->refs++;
      *handle = cache->slots[slot];
      return ncclSuccess;
    }
  }
}

ncclResult_t ncclRegCleanup(struct ncclComm* comm) {
  struct ncclRegCache* cache = &comm->regCache;
  for (int i=0; i<cache->population; i++) {
    INFO(NCCL_INIT, "Cleanup buffer %p pages %lx", (void*)cache->slots[i]->addr, cache->slots[i]->pages);
    NCCLCHECK(ncclNetDeregister(comm, cache->slots[i]));
    free(cache->slots[i]);
  }
  free(cache->slots);
  return ncclSuccess;
}

NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 1);
NCCL_PARAM(LocalRegisterNet, "LOCAL_REGISTER_NET", 1);
NCCL_PARAM(LocalRegisterNvls, "LOCAL_REGISTER_NVLS", 1);

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;

  if (ncclParamLocalRegister()) {
    if (comm == NCCL_COMM_NULL || buff == NULL || handle == NULL || size == 0) {
      WARN("Invalid arguments comm %p, buff %p, size %ld, handle %p", comm, buff, size, handle);
      ret = ncclInvalidArgument;
    } else {
      if (ncclParamLocalRegisterNet()) {
        NCCLCHECK(ncclRegister(comm, buff, size, handle));
      }
#if CUDART_VERSION >= 12010
      if (comm->nvlsSupport && ncclParamLocalRegisterNvls()) {
        size_t granularity;
        CUmulticastObjectProp prop = comm->nvlsResources->properties;

        prop.size = size;
        CUCHECK(cuMulticastGetGranularity(&granularity, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

        if ((uintptr_t)buff % comm->nvlsResources->ucGran == 0 && size % granularity == 0) {
          /* we can direct register what user provide */
          struct ncclRegRequest* req;
          NCCLCHECK(ncclCalloc(&req, 1));
          req->buff = (uintptr_t)buff;
          req->size = size;
          ncclIntruQueueEnqueue(&comm->regRequestQueue, req);
          *handle = (void*)req;
        } else {
          void* base;
          size_t baseSize;
          /* Since we don't provide actually allocated buffer size for users by ncclMemAlloc,
           * therefore, we need to get the full range of the buffer by cuMemGetAddressRange to
           * register buffers. */
          CUCHECK(cuMemGetAddressRange((CUdeviceptr*)&base, &baseSize, (CUdeviceptr)buff));
          if ((uintptr_t)base % comm->nvlsResources->ucGran == 0 && baseSize % granularity == 0) {
            struct ncclRegRequest* req;
            NCCLCHECK(ncclCalloc(&req, 1));
            req->buff = (uintptr_t)base;
            req->size = baseSize;
            ncclIntruQueueEnqueue(&comm->regRequestQueue, req);
            *handle = (void*)req;
          } else {
            WARN("register fails, buffer %p (aligned %s, granularity %ld) and size %ld (aligned %s, granularity %ld) for registration", buff, (uintptr_t)buff % comm->nvlsResources->ucGran == 0 ? "TRUE" : "FALSE", comm->nvlsResources->ucGran, size, size % granularity == 0 ? "TRUE" : "FALSE", granularity);
            ret = ncclInvalidArgument;
          }
        }
      }
#endif
    }
  }

  return ret;
}

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  ncclResult_t ret = ncclSuccess;

  struct ncclRegRequest* dreq = (struct ncclRegRequest*)handle;
  if (ncclParamLocalRegister()) {
    if (comm == NCCL_COMM_NULL || handle == NULL) {
      WARN("Invalid arguments comm %p, handle %p", comm, handle);
      ret = ncclInvalidArgument;
    } else {
      if (ncclParamLocalRegisterNet()) {
        struct ncclReg* reg = (struct ncclReg*)handle;
        struct ncclRegCache* cache = &comm->regCache;
        int slot;
        for (slot=0; slot<cache->population && cache->slots[slot] != reg; slot++);
        if (slot == cache->population) {
          WARN("Deregister: Could not find handle");
          return ncclInvalidUsage;
        }
        if (--reg->refs) return ncclSuccess;
        NCCLCHECK(ncclNetDeregister(comm, reg));
        free(reg);
        memmove(cache->slots+slot, cache->slots+slot+1, (cache->population-slot-1)*sizeof(struct ncclReg*));
        cache->population -= 1;
        return ncclSuccess;
      }
#if CUDART_VERSION >= 12010
      if (comm->nvlsSupport && ncclParamLocalRegisterNvls()) {
        struct ncclRegRecord* rec;

        /* first release register record */
        rec = ncclIntruQueueHead(&comm->regRecordQueue);

        while (rec) {
          if (rec->buff == dreq->buff && rec->size == dreq->size) {
            NCCLCHECK(ncclNvlsDeregBuffer(&rec->mcHandle, rec->regAddr, rec->dev, rec->regSize));
            ncclIntruQueueDelete(&comm->regRecordQueue, rec);
            free(rec->addrs);
            free(rec);
            break;
          }
          rec = rec->next;
        }

        /* then free register request */
        if (ncclIntruQueueDelete(&comm->regRequestQueue, dreq) == false) {
          WARN("Invalid handle %p", handle);
          ret = ncclInvalidArgument;
        }
      }
#endif
    }
  }

  return ret;
}

NCCL_API(ncclResult_t, ncclMemAlloc, void **ptr, size_t size);
ncclResult_t  ncclMemAlloc(void **ptr, size_t size) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;

#if CUDART_VERSION >= 12010
  size_t memGran = 0;
  size_t mcGran = 0;
  CUdevice currentDev;
  CUmemAllocationProp memprop = {};
  CUmulticastObjectProp mcprop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int flag = 0;
  int dcnt;
  int mcSupport = 0;

  if (ptr == NULL || size == 0) goto fallback;

  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  if (CUPFN(cuMulticastCreate) != NULL)
    CUCHECK(cuDeviceGetAttribute(&mcSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, currentDev));

  if (mcSupport) {
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memprop.requestedHandleTypes = NVLS_CU_MEM_HANDLE_TYPE;
    memprop.location.id = currentDev;
    // Query device to see if RDMA support is available
    CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, currentDev));
    if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1;
    CUCHECK(cuMemGetAllocationGranularity(&memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    /* mc property */
    CUDACHECK(cudaGetDeviceCount(&dcnt));
    mcprop.size = size;
    /* device cnt is a dummy value right now, it might affect mc granularity in the future. */
    mcprop.numDevices = dcnt;
    mcprop.handleTypes = NVLS_CU_MEM_HANDLE_TYPE;
    mcprop.flags = 0;
    CUCHECK(cuMulticastGetGranularity(&mcGran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

    /* only size needs to be aligned to mcGran */
    ALIGN_SIZE(size, mcGran);
    /* Allocate the physical memory on the device */
    CUCHECK(cuMemCreate(&handle, size, &memprop, 0));
    /* Reserve a virtual address range */
    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, memGran, 0, 0));
    /* Map the virtual address range to the physical allocation */
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
    /* Now allow RW access to the newly mapped memory */
    for (int i = 0; i < dcnt; ++i) {
      int p2p = 0;
      if (i == cudaDev || ((cudaDeviceCanAccessPeer(&p2p, cudaDev, i) == cudaSuccess) && p2p)) {
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = i;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
      }
    }
    goto exit;
  }

fallback:
#endif
  CUDACHECKGOTO(cudaMalloc(ptr, size), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclMemFree, void *ptr);
ncclResult_t  ncclMemFree(void *ptr) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;
  int saveDevice;

  CUDACHECK(cudaGetDevice(&saveDevice));
#if CUDART_VERSION >= 12010
  CUdevice ptrDev = 0;
  int mcSupport = 0;

  if (ptr == NULL) goto fallback;

  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  CUCHECKGOTO(cuPointerGetAttribute((void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr), ret, fail);
  if (CUPFN(cuMulticastCreate) != NULL)
    CUCHECKGOTO(cuDeviceGetAttribute(&mcSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, ptrDev), ret, fail);

  CUDACHECKGOTO(cudaSetDevice((int)ptrDev), ret, fail);
  if (mcSupport) {
    NCCLCHECKGOTO(ncclCuMemFree(ptr), ret, fail);
    goto exit;
  }

fallback:
#endif
  CUDACHECKGOTO(cudaFree(ptr), ret, fail);

exit:
  cudaSetDevice(saveDevice);
  return ret;
fail:
  goto exit;
}
