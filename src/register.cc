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

// ncclResult_t ncclNetRegister(struct ncclComm* comm, void* addr, size_t size, struct ncclReg* reg) {
//   int netDevs;
//   NCCLCHECK(comm->ncclNet->devices(&netDevs));

//   int localNetDevCount = 0;
//   int* localNetDevs;
//   void *lComm = NULL;
//   ncclResult_t ret = ncclSuccess;

//   NCCLCHECK(ncclCalloc(&localNetDevs, comm->p2pnChannels));

//   // Find local devices for p2p operations
//   for (int c=0; c<comm->p2pnChannels; c++) {
//     int dev;
//     if (ncclTopoGetLocalNet(comm->topo, comm->rank, c, &dev) != ncclSuccess) goto end; // No local net
//     ncclNetProperties_t props;
//     NCCLCHECKGOTO(comm->ncclNet->getProperties(dev, &props), ret, end);
//     if (props.regIsGlobal == 0) { // We need to be sure all NICs support global registration.
//       localNetDevCount = 0;
//       break;
//     }
//     localNetDevs[localNetDevCount++] = dev;
//   }

//   NCCLCHECKGOTO(ncclCalloc(&reg->sComms, localNetDevCount), ret, end);
//   NCCLCHECKGOTO(ncclCalloc(&reg->rComms, localNetDevCount), ret, end);
//   NCCLCHECKGOTO(ncclCalloc(&reg->handles, localNetDevCount), ret, end);
//   reg->nComms = localNetDevCount;

//   ncclDebugNoWarn = NCCL_NET;
//   for (int d=0; d<localNetDevCount; d++) {
//     int dev = localNetDevs[d];
//     reg->handles[d] = reg->sComms[d] = reg->rComms[d] = NULL;

//     ncclNetHandle_t netHandle;
//     NCCLCHECKGOTO(comm->ncclNet->listen(dev, &netHandle, &lComm), ret, end);

//     bool connected;
//     connected = false;
//     while (!connected) {
//       if (*comm->abortFlag) {
//         goto end;
//       }

//       if (reg->sComms[d] == NULL)
//         NCCLCHECKGOTO(comm->ncclNet->connect(dev, &netHandle, reg->sComms+d), ret, end);

//       if (reg->rComms[d] == NULL)
//         NCCLCHECKGOTO(comm->ncclNet->accept(lComm, reg->rComms+d), ret, end);

//       connected = (reg->rComms[d] != NULL) && (reg->sComms[d] != NULL);
//     }
//     NCCLCHECK(comm->ncclNet->closeListen(lComm));
//     lComm = NULL;

//     if (comm->ncclNet->regMr(reg->sComms[d], addr, size, NCCL_PTR_CUDA, reg->handles+d) != ncclSuccess) {
//       reg->handles[d] = NULL;
//     }
//   }
// end:
//   ncclDebugNoWarn = 0;
//   if (ret != ncclSuccess) NCCLCHECK(ncclNetDeregister(comm, reg));
//   free(localNetDevs);
//   return ret;
// }

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
      // NCCLCHECK(ncclNetRegister(comm, (void*)addr, pages*pageSize, regSlot));
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
    // NCCLCHECK(ncclNetDeregister(comm, cache->slots[i]));
    free(cache->slots[i]);
  }
  free(cache->slots);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  NCCLCHECK(PtrCheck(comm, "ncclCommRegister", "comm"));
  if (comm->checkPointers) NCCLCHECK(CudaPtrCheck(buff, comm, "buff", "ncclCommRegister"));
  NCCLCHECK(ncclRegister(comm, buff, size, handle));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  NCCLCHECK(PtrCheck(comm, "ncclCommRegister", "comm"));
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
