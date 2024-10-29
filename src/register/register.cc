/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "nccl.h"
#include "comm.h"
#include "net.h"
#include "register.h"
#include "transport.h"

ncclResult_t ncclRegFind(struct ncclComm* comm, const void* data, size_t size, struct ncclReg** reg) {
  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t pageSize = cache->pageSize;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;

  *reg = NULL;
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot]->addr) return ncclSuccess;
    if ((addr >= cache->slots[slot]->addr) &&
        ((addr-cache->slots[slot]->addr)/pageSize+pages) <= cache->slots[slot]->pages) {
      *reg = cache->slots[slot];
      return ncclSuccess;
    }
  }
}
NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 1);

ncclResult_t ncclRegister(struct ncclComm* comm, void* data, size_t size, void** handle) {
  if (!ncclParamLocalRegister()) {
    *handle = NULL;
    return ncclSuccess;
  }
  INFO(NCCL_REG, "register comm %p buffer %p size %zi", comm, data, size);
  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t pageSize = cache->pageSize;
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

static ncclResult_t regCleanup(struct ncclComm* comm, struct ncclReg* reg) {
  if (reg->state & NET_REG_COMPLETE) {
    struct ncclRegNetHandles* netHandle = reg->netHandleHead;
    struct ncclRegNetHandles* netHandlePrev;
    while(netHandle) {
      NCCLCHECK(ncclNetDeregBuffer(comm, netHandle->proxyConn, netHandle->handle));
      netHandlePrev = netHandle;
      netHandle = netHandle->next;
      free(netHandlePrev);
    }
  }
  if (reg->state & NVLS_REG_COMPLETE) {
    NCCLCHECK(ncclNvlsDeregBuffer(comm, &reg->mcHandle, reg->regAddr, reg->dev, reg->regSize));
    reg->regAddr = (CUdeviceptr)NULL;
  }
  if (reg->state & COLLNET_REG_COMPLETE) {
    NCCLCHECK(ncclCollnetDeregBuffer(comm, reg->collnetProxyconn, reg->collnetHandle));
  }
  if (reg->state & IPC_REG_COMPLETE) {
    for (int i = 0; i < NCCL_MAX_LOCAL_RANKS; ++i)
      if (reg->ipcInfos[i])
        NCCLCHECK(ncclIpcDeregBuffer(comm, reg->ipcInfos[i]));
    if (reg->regIpcAddrs.hostPeerRmtAddrs) free(reg->regIpcAddrs.hostPeerRmtAddrs);
    if (reg->regIpcAddrs.devPeerRmtAddrs) NCCLCHECK(ncclCudaFree(reg->regIpcAddrs.devPeerRmtAddrs));
  }
  return ncclSuccess;
}

ncclResult_t ncclRegCleanup(struct ncclComm* comm) {
  struct ncclRegCache* cache = &comm->regCache;
  for (int i = 0; i < cache->population; i++) {
    struct ncclReg* reg = cache->slots[i];
    INFO(NCCL_INIT, "Cleanup buffer %p pages %lx", (void*)reg->addr, reg->pages);
    NCCLCHECK(regCleanup(comm, reg));
    free(reg);
  }
  free(cache->slots);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  NCCLCHECK(CommCheck(comm, "ncclCommRegister", "comm"));
  if (comm->checkPointers) NCCLCHECK(CudaPtrCheck(buff, comm, "buff", "ncclCommRegister"));
  NCCLCHECK(ncclRegister(comm, buff, size, handle));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  NCCLCHECK(CommCheck(comm, "ncclCommRegister", "comm"));
  struct ncclReg* reg = (struct ncclReg*)handle;
  struct ncclRegCache* cache = &comm->regCache;
  int slot;
  int saveDev;
  if (handle == NULL) goto exit;
  CUDACHECK(cudaGetDevice(&saveDev));
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  for (slot=0; slot<cache->population && cache->slots[slot] != reg; slot++);
  if (slot == cache->population) {
    WARN("Deregister: Could not find handle");
    return ncclInvalidUsage;
  }
  if (--reg->refs) return ncclSuccess;
  NCCLCHECK(regCleanup(comm, reg));
  free(reg);
  memmove(cache->slots+slot, cache->slots+slot+1, (cache->population-slot-1)*sizeof(struct ncclReg*));
  cache->population -= 1;
  CUDACHECK(cudaSetDevice(saveDev));
exit:
  return ncclSuccess;
}
