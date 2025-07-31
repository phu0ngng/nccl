#include "sym_runtime.h"
#include "comm.h"
#include "device.h"
#include "transport.h"
#include "group.h"

NCCL_PARAM(WinStride, "WIN_STRIDE", -1);

// Complete types from src/include/sym_runtime.h
struct ncclSymrMemory {
  int refCount;
  struct ncclSymrMemory* next;
  CUmemGenericAllocationHandle memHandle;
  size_t size;
  size_t bigOffset; // offset in big VA space
};

struct ncclSymrWindowSorted {
  uintptr_t userAddr;
  size_t size;
  struct ncclSymrWindow* win;
};

struct ncclSymrTeam {
  struct ncclSymrTeam* next;
  struct ncclSymTeam team;
  CUmemGenericAllocationHandle mcHandle;
  void* mcBasePtr;
  int worldRankList[];
};

////////////////////////////////////////////////////////////////////////////////
// Helpers at the bottom:

// Find least index such that `arg < sorted[i].key` (least upper bound)
template<typename Obj, typename Key>
static int listFindSortedLub(Key Obj::*key, Obj* sorted, int count, Key arg);

template<typename Obj>
static void listInsert(Obj** list, int* capacity, int* count, int index, Obj val);

template<typename Obj>
static void listRemove(Obj* list, int* count, int index);

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclSymrInitOnce(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSymrState* symr = &comm->symrState;
  if (symr->bigSize != 0) return ncclSuccess;

  bool nearIsLocal = true;
  for (int i=0; i < comm->localRanks; i++) {
    nearIsLocal &= comm->localRankToRank[i] == comm->localRankToRank[0] + i;
  }
  symr->nearSelf = nearIsLocal ? comm->localRank : 0;
  symr->nearSize = nearIsLocal ? comm->localRanks : 1;
  symr->nearRankList = (int*)malloc(symr->nearSize*sizeof(int));
  for (int i=0; i < symr->nearSize; i++) {
    symr->nearRankList[i] = comm->rank + (i - symr->nearSelf);
  }

  CUmemAllocationProp memProp = {};
  memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  memProp.requestedHandleTypes = ncclCuMemHandleType;
  memProp.location.id = comm->cudaDev;
  CUCHECKGOTO(cuMemGetAllocationGranularity(&symr->granularity, &memProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail_nearRankList);

  symr->bigSize = ncclParamWinStride();
  if (-symr->bigSize <= 1) {
    symr->bigSize = 1;
    for (int r=0; r < comm->nRanks; ++r) {
      symr->bigSize = std::max<size_t>(symr->bigSize, comm->peerInfo[r].totalGlobalMem);
    }
  }
  symr->bigSize = alignUp(symr->bigSize, size_t(1)<<32);
  INFO(NCCL_INIT, "Symmetric VA size=%ldGB", (long)symr->bigSize>>30);
  
  ncclSpaceConstruct(&symr->bigSpace);
  ncclShadowPoolConstruct(&symr->shadows);
  return ncclSuccess;

fail_nearRankList:
  free(symr->nearRankList);
  return ret;
}

static void symTeamDestroyAll(struct ncclComm* comm); // Further down

ncclResult_t ncclSymrFinalize(struct ncclComm* comm) {
  struct ncclSymrState* symr = &comm->symrState;
  if (symr->bigSize == 0) return ncclSuccess;

  while (!ncclIntruQueueEmpty(&symr->regTaskQueue)) {
    struct ncclSymrRegTask* task = ncclIntruQueueDequeue(&symr->regTaskQueue);
    free(task);
  }
  
  symTeamDestroyAll(comm);
  { // delete windowTable
    cudaStream_t stream;
    if (cudaSuccess == cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)) {
      struct ncclSymCommWindowTable* tableDev = symr->windowTable;
      while (tableDev != nullptr) {
        struct ncclSymCommWindowTable* tableHost;
        if (ncclSuccess != ncclShadowPoolToHost(&symr->shadows, tableDev, &tableHost)) break;
        struct ncclSymCommWindowTable* next = tableHost->next;
        ncclShadowPoolFree(&symr->shadows, tableDev, stream);
        tableDev = next;
      }
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }
  CUdeviceptr flatAddr = reinterpret_cast<CUdeviceptr>(symr->nearFlatBase);
  CUCHECKIGNORE(cuMemUnmap(flatAddr, symr->nearSize*symr->bigSize));
  CUCHECKIGNORE(cuMemAddressFree(flatAddr, symr->nearSize*symr->bigSize));
  ncclShadowPoolDestruct(&symr->shadows);
  ncclSpaceDestruct(&symr->bigSpace);
  free(symr->nearRankList);
  free(symr->winSorted);
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////

static ncclResult_t symMemoryMapNearTeam(
    struct ncclComm* comm, CUmemGenericAllocationHandle memHandle, size_t size, size_t bigOffset
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSymrState* symr = &comm->symrState;
  CUmemAccessDesc accessDesc = {};
  union Message {
    CUmemGenericAllocationHandle memHandle;
    CUmemFabricHandle fabricHandle;
  };

  Message* messages = (Message*)calloc(symr->nearSize, sizeof(Message));
  if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    messages[symr->nearSelf].memHandle = memHandle;
  } else {
    CUCHECKGOTO(cuMemExportToShareableHandle(&messages[symr->nearSelf].fabricHandle, memHandle, ncclCuMemHandleType, 0), ret, fail);
  }

  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, symr->nearRankList, symr->nearSelf, symr->nearSize, messages, sizeof(Message)), ret, fail);

  if (symr->nearFlatBase == nullptr) { // Create on first need.
    CUdeviceptr addr;
    CUCHECKGOTO(cuMemAddressReserve(&addr, symr->nearSize*symr->bigSize, NCCL_MAX_PAGE_SIZE, 0, 0), ret, fail);
    symr->nearFlatBase = reinterpret_cast<void*>(addr);
  }
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = comm->cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  for (int r = 0; r < symr->nearSize; r++) {
    CUmemGenericAllocationHandle impHandle;
    if (r == symr->nearSelf) {
      impHandle = memHandle;
    } else {
      if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        int fd = -1;
        NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, symr->nearRankList[r], &messages[r], &fd), ret, fail);
        CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle, reinterpret_cast<void*>((uintptr_t)fd), ncclCuMemHandleType), ret, fail);
        SYSCHECKGOTO(close(fd), "close", ret, fail);
      } else {
        CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle, (void*)&messages[r].fabricHandle, ncclCuMemHandleType), ret, fail);
      }
    }
    CUdeviceptr addr = reinterpret_cast<uintptr_t>((char*)symr->nearFlatBase + r*symr->bigSize + bigOffset);
    CUCHECKGOTO(cuMemMap(addr, size, 0, impHandle, 0), ret, fail);
    CUCHECKGOTO(cuMemSetAccess(addr, size, &accessDesc, 1), ret, fail);
    if (r != symr->nearSelf) {
      CUCHECKGOTO(cuMemRelease(impHandle), ret, fail);
    }
  }
  // Ensure everyone has imported my mem handle.
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, symr->nearRankList, symr->nearSelf, symr->nearSize, 0xbeef), ret, fail);
leave:
  free(messages);
  return ret;
fail:
  goto leave;
}

static ncclResult_t symBindTeamMemory(
    struct ncclComm* comm, struct ncclSymrTeam* tm, struct ncclSymrMemory* mem
  ) {
  if (comm->nvlsSupport && tm->mcBasePtr != nullptr) {
  #if CUDART_VERSION >= 12010
    INFO(NCCL_NVLS, "Binding multicast memory at big=%lx to team {%d x %d}", mem->bigOffset, tm->team.nRanks, tm->team.stride);
    CUCHECK(cuMulticastBindMem(tm->mcHandle, mem->bigOffset, mem->memHandle, 0, mem->size, 0));
  #endif
  }
  return ncclSuccess;
}

static ncclResult_t symUnbindTeamMemory(
    struct ncclComm* comm, struct ncclSymrTeam* tm, struct ncclSymrMemory* mem
  ) {
  if (comm->nvlsSupport && tm->mcBasePtr != nullptr) {
  #if CUDART_VERSION >= 12010
    CUCHECK(cuMulticastUnbind(tm->mcHandle, comm->cudaDev, mem->bigOffset, mem->size));
  #endif
  }
  return ncclSuccess;
}

// Caller must barrier the team afterward.
static ncclResult_t symTeamObtain(
    struct ncclComm* comm, struct ncclSymTeam team, bool multimem,
    struct ncclSymrTeam** outTeam
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSymrState* symr = &comm->symrState;
  struct ncclSymrTeam* t = symr->teamHead;
  bool teamIsNew = false;
  while (true) {
    if (t == nullptr) {
      teamIsNew = true;
      t = (struct ncclSymrTeam*)malloc(sizeof(struct ncclSymrTeam) + team.nRanks*sizeof(int));
      t->team = team;
      t->mcHandle = 0x0;
      t->mcBasePtr = nullptr;
      for (int i=0; i < team.nRanks; i++) {
        t->worldRankList[i] = comm->rank + (i - team.rank)*team.stride;
      }
      break;
    } else if (t->team.rank == team.rank && t->team.nRanks == team.nRanks && t->team.stride == team.stride) {
      if (!multimem || t->mcBasePtr != nullptr) {
        // Matching team is sufficient
        if (outTeam) *outTeam = t;
        return ncclSuccess;
      }
      break; // Need to enable multimem
    }
  }

  if (multimem) {
    if (!comm->nvlsSupport) {
      WARN("Multicast support requested for team but none available on system.");
      ret = ncclInvalidArgument;
      goto fail;
    } else {
    #if CUDART_VERSION >= 12010
      CUmemGenericAllocationHandle mcHandle = 0;
      CUdeviceptr mcAddr = 0;
      CUmulticastObjectProp mcProp = {};
      char shareableHandle[NVLS_HANDLE_SIZE];

      mcProp.numDevices = team.nRanks;
      mcProp.handleTypes = ncclCuMemHandleType;
      mcProp.flags = 0;
      mcProp.size = symr->bigSize;
      if (team.rank == 0) {
        NCCLCHECKGOTO(ncclNvlsGroupCreate(comm, &mcProp, team.rank, team.nRanks, &mcHandle, shareableHandle), ret, fail);
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, team.rank, team.nRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail_mcHandle);
      } else {
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, team.rank, team.nRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
        NCCLCHECKGOTO(ncclNvlsGroupConnect(comm, shareableHandle, t->worldRankList[0], &mcHandle), ret, fail);
      }

      CUCHECKGOTO(cuMulticastAddDevice(mcHandle, comm->cudaDev), ret, fail_mcHandle);
      CUCHECKGOTO(cuMemAddressReserve(&mcAddr, symr->bigSize, NCCL_MAX_PAGE_SIZE, 0, 0), ret, fail_mcHandle);
      CUCHECKGOTO(cuMemMap(mcAddr, symr->bigSize, 0, mcHandle, 0), ret, fail_mcHandle_mcAddr);
      { CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = comm->cudaDev;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECKGOTO(cuMemSetAccess(mcAddr, symr->bigSize, &accessDesc, 1), ret, fail_mcHandle_mcAddr_unmap);
      }
      t->mcHandle = mcHandle;
      t->mcBasePtr = reinterpret_cast<void*>(mcAddr);

      // Bind new team with all existing memories.
      for (struct ncclSymrMemory* mem = symr->memHead; mem != nullptr; mem = mem->next) {
        NCCLCHECKGOTO(symBindTeamMemory(comm, t, mem), ret, fail_mcHandle_mcAddr_unmap_mems);
      }

      if (false) { // Error labels:
      fail_mcHandle_mcAddr_unmap_mems:
        for (struct ncclSymrMemory* mem = symr->memHead; mem != nullptr; mem = mem->next) {
          symUnbindTeamMemory(comm, t, mem);
        }
      fail_mcHandle_mcAddr_unmap:
        CUCHECKIGNORE(cuMemUnmap(mcAddr, symr->bigSize));
        goto fail_mcHandle_mcAddr; // silence unused label warning
      fail_mcHandle_mcAddr:
        CUCHECKIGNORE(cuMemAddressFree(mcAddr, symr->bigSize));
        goto fail_mcHandle; // silence unused label warning
      fail_mcHandle:
        CUCHECKIGNORE(cuMemRelease(mcHandle));
        goto fail; // silence unused label warning
      }
    #else
      goto fail; // silence unused label warning
    #endif
    }
  }

  if (teamIsNew) {
     // Add to list
    t->next = symr->teamHead;
    symr->teamHead = t;
  }
  if (outTeam) *outTeam = t;
  return ret;

fail:
  if (teamIsNew) free(t);
  return ret;
}

static void symTeamDestroyAll(struct ncclComm* comm) {
  struct ncclSymrState* symr = &comm->symrState;
  while (symr->teamHead != nullptr) {
    struct ncclSymrTeam* t = symr->teamHead;
    symr->teamHead = t->next;
    if (t->mcBasePtr != nullptr) {
      for (struct ncclSymrMemory* m = symr->memHead; m != nullptr; m = m->next) {
        symUnbindTeamMemory(comm, t, m);
      }
      CUdeviceptr mcAddr = reinterpret_cast<CUdeviceptr>(t->mcBasePtr);
      CUCHECKIGNORE(cuMemUnmap(mcAddr, symr->bigSize));
      CUCHECKIGNORE(cuMemAddressFree(mcAddr, symr->bigSize));
      CUCHECKIGNORE(cuMemRelease(t->mcHandle));
    }
    free(t);
  }
}

// On success we take caller's reference on memHandle.
// Due to multicast binds for each pre-exiting team, this function requires
// caller do a world barrier before returning to user.
static ncclResult_t symMemoryObtain(
    struct ncclComm* comm, CUmemGenericAllocationHandle memHandle, size_t size,
    struct ncclSymrMemory** outMem
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSymrState* symr = &comm->symrState;
  int64_t bigOffset = 0;

  struct ncclSymrMemory* mem = symr->memHead;
  while (mem != nullptr) {
    if (mem->memHandle == memHandle) {
      CUCHECKIGNORE(cuMemRelease(memHandle));
      goto leave;
    }
    mem = mem->next;
  }
  // New memory.
  mem = (struct ncclSymrMemory*)malloc(sizeof(struct ncclSymrMemory));
  mem->refCount = 0;
  mem->memHandle = memHandle;
  mem->size = size;
 
  // Grab offset in the big space.
  NCCLCHECKGOTO(ncclSpaceAlloc(&symr->bigSpace, symr->bigSize, size, symr->granularity, &bigOffset), ret, fail_mem);
  mem->bigOffset = bigOffset;

  // Map unicast addresses into flat VA space for near team.
  NCCLCHECKGOTO(symMemoryMapNearTeam(comm, memHandle, size, bigOffset), ret, fail_mem_space);

  // Bind new memory with each existing team.
  for (struct ncclSymrTeam* t = symr->teamHead; t != nullptr; t = t->next) {
    NCCLCHECKGOTO(symBindTeamMemory(comm, t, mem), ret, fail_mem_space_teams);
  }
  // Add to list of mems.
  mem->next = symr->memHead;
  symr->memHead = mem;

leave:
  mem->refCount += 1;
  *outMem = mem;
  return ret;

fail_mem_space_teams:
  for (struct ncclSymrTeam* t = symr->teamHead; t != nullptr; t = t->next) {
    symUnbindTeamMemory(comm, t, mem);
  }
fail_mem_space:
  ncclSpaceFree(&symr->bigSpace, bigOffset, size);
fail_mem:
  free(mem);
//fail:
  return ret;
}

static void symMemoryDropRef(
    struct ncclComm* comm, struct ncclSymrMemory* mem
  ) {
  if (mem != nullptr && 0 == --mem->refCount) {
    struct ncclSymrState* symr = &comm->symrState;
    for (struct ncclSymrTeam* t = symr->teamHead; t != nullptr; t = t->next) {
      symUnbindTeamMemory(comm, t, mem);
    }
    ncclSpaceFree(&symr->bigSpace, mem->bigOffset, mem->size);
    CUCHECKIGNORE(cuMemRelease(mem->memHandle));

    struct ncclSymrMemory** ptr = &symr->memHead;
    while (*ptr != mem) ptr = &(*ptr)->next;
    *ptr = mem->next; // Remove from list.

    free(mem);
  }
}

static ncclResult_t symWindowTableInitOnce(struct ncclComm* comm, cudaStream_t stream) {
  struct ncclSymrState* symr = &comm->symrState;
  struct ncclSymCommWindowTable* tableDev = symr->windowTable;
  if (tableDev == nullptr) { // Create on first need.
    NCCLCHECK(ncclShadowPoolAlloc<ncclSymCommWindowTable>(&symr->shadows, &tableDev, nullptr, stream));
    symr->windowTable = tableDev;
  }
  return ncclSuccess;
}

// On success we take callers reference on `mem`.
static ncclResult_t symWindowCreate(
    struct ncclComm* comm, struct ncclSymrMemory* mem,
    size_t memOffset, void* userPtr, size_t userSize, int winFlags, void* localReg,
    struct ncclWindow_vidmem** outWinDev, struct ncclSymrWindow** outWin,
    cudaStream_t stream
  ) {
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr);
  struct ncclSymrState* symr = &comm->symrState;
  struct ncclSymrWindow* win;

  win = (struct ncclSymrWindow*)malloc(sizeof(struct ncclSymrWindow));
  memset(win, 0, sizeof(*win));
  win->memory = mem;
  win->size = userSize;
  win->bigOffset = mem->bigOffset + memOffset;
  win->winFlags = winFlags;
  win->localRegHandle = localReg;
  if (userPtr == nullptr) {
    // Null means caller has no VA and will use the near team flat VA address.
    win->userPtr = (char*)symr->nearFlatBase + (symr->nearSelf*symr->bigSize) + mem->bigOffset;
  } else {
    win->userPtr = userPtr;
  }

  struct ncclWindow_vidmem* winDev;
  struct ncclWindow_vidmem* winDevHost;
  NCCLCHECK(ncclShadowPoolAlloc(&symr->shadows, &winDev, &winDevHost, stream));
  win->vidmem = winDev;
  winDevHost->nearFlatBase = (char*)symr->nearFlatBase + win->bigOffset;
  winDevHost->mcOffset4K = win->bigOffset>>12;
  winDevHost->stride4G = symr->bigSize>>32;
  winDevHost->nearRank = symr->nearSelf;
  winDevHost->worldRank = comm->rank;
  winDevHost->winHost = (void*)win;
  CUDACHECK(cudaMemcpyAsync(winDev, winDevHost, sizeof(struct ncclWindow_vidmem), cudaMemcpyHostToDevice, stream));

  NCCLCHECK(symWindowTableInitOnce(comm, stream)); // ensure symr->windowTable exists
  struct ncclSymCommWindowTable* tableDev = symr->windowTable;
  struct ncclSymCommWindowTable* tableHost;
  NCCLCHECK(ncclShadowPoolToHost(&symr->shadows, tableDev, &tableHost));
  while (true) {
    int i = 0;
    while (i < 32 && tableHost->entries[i].window != nullptr) i += 1;
    if (i < 32) {
      tableHost->entries[i].base = userAddr;
      tableHost->entries[i].size = userAddr + userSize;
      tableHost->entries[i].window = winDev;
      CUDACHECK(cudaMemcpyAsync(&tableDev->entries[i], &tableHost->entries[i], sizeof(tableHost->entries[i]), cudaMemcpyHostToDevice, stream));
      break;
    }
    if (tableHost->next == nullptr) {
      NCCLCHECK(ncclShadowPoolAlloc<ncclSymCommWindowTable>(&symr->shadows, &tableHost->next, nullptr, stream));
      CUDACHECK(cudaMemcpyAsync(&tableDev->next, &tableHost->next, sizeof(tableHost->next), cudaMemcpyHostToDevice, stream));
    }
    tableDev = tableHost->next;
    NCCLCHECK(ncclShadowPoolToHost(&symr->shadows, tableHost->next, &tableHost));
  }

  { // insert into winSorted[]
    int i = listFindSortedLub(&ncclSymrWindowSorted::userAddr, symr->winSorted, symr->winSortedCount, userAddr);
    struct ncclSymrWindowSorted winSort;
    winSort.userAddr = userAddr;
    winSort.size = userSize;
    winSort.win = win;
    listInsert(&symr->winSorted, &symr->winSortedCapacity, &symr->winSortedCount, i, winSort);
  }

  if (outWinDev) *outWinDev = winDev;
  if (outWin) *outWin = win;
  return ncclSuccess;
}

static ncclResult_t symWindowDestroy(struct ncclComm* comm, struct ncclWindow_vidmem* winDev, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSymrState* symr = &comm->symrState;
  struct ncclWindow_vidmem* winDevHost;
  struct ncclSymrWindow* winHost;

  NCCLCHECKGOTO(ncclShadowPoolToHost(&symr->shadows, winDev, &winDevHost), ret, fail);
  winHost = (struct ncclSymrWindow*)winDevHost->winHost;

  symMemoryDropRef(comm, winHost->memory);

  { struct ncclSymCommWindowTable* tableDev = symr->windowTable;
    struct ncclSymCommWindowTable* tableHost;
    NCCLCHECKGOTO(ncclShadowPoolToHost(&symr->shadows, tableDev, &tableHost), ret, remove_winSorted);
    while (true) {
      int i = 0;
      while (i < 32 && tableHost->entries[i].window != winDev) i += 1;
      if (i < 32) {
        memset(&tableHost->entries[i], 0, sizeof(tableHost->entries[i]));
        CUDACHECKGOTO(cudaMemsetAsync(&tableDev->entries[i], 0, sizeof(tableDev->entries[i]), stream), ret, remove_winSorted);
        break;
      }
      if (tableHost->next == nullptr) break; // Error didn't find window in table
      tableDev = tableHost->next;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&symr->shadows, tableHost->next, &tableHost), ret, remove_winSorted);
    }
  }
  NCCLCHECKGOTO(ncclShadowPoolFree(&symr->shadows, winDev, stream), ret, remove_winSorted);

  NCCLCHECKGOTO(ncclCommDeregister(comm, winHost->localRegHandle), ret, remove_winSorted);

remove_winSorted:
  { int i = listFindSortedLub(&ncclSymrWindowSorted::userAddr, symr->winSorted, symr->winSortedCount, reinterpret_cast<uintptr_t>(winHost->userPtr));
    i -= 1; // least upper bound is just after ours.
    listRemove(symr->winSorted, &symr->winSortedCount, i);
  }
  free(winHost);
fail:
  return ret;
}

ncclResult_t ncclSymrWindowRegisterInGroup(
    struct ncclComm* comm,
    void* userPtr, size_t userSize, int winFlags, ncclWindow_t* outWinDev
  ) {
  ncclResult_t ret = ncclSuccess;
  CUdeviceptr memAddr = 0;
  size_t memSize = 0;
  CUmemGenericAllocationHandle memHandle = 0x0;
  size_t memOffset;
  struct ncclSymrMemory* mem = nullptr;
  cudaStream_t stream = nullptr;
  void* localRegHandle = nullptr;

  NCCLCHECKGOTO(ncclCommRegister(comm, userPtr, userSize, &localRegHandle), ret, fail);

  if (!comm->symmetricSupport) {
    // We just return the local registration handle directly in this case, as there's no reason to allocate the
    // ncclWindow_vidmem structure on the device, etc.
    *outWinDev = reinterpret_cast<struct ncclWindow_vidmem*>(localRegHandle);
    return ncclSuccess;
  }
  if (winFlags & NCCL_WIN_COLL_SYMMETRIC) {
    // Defer symmetric kernel init until at least one window with that flag exists.
    NCCLCHECKGOTO(ncclSymkInitOnce(comm), ret, fail);
  }

  // Get underlying cumem handle:
  CUCHECKGOTO(cuMemGetAddressRange(&memAddr, &memSize, reinterpret_cast<CUdeviceptr>(userPtr)), ret, fail_locReg);
  memOffset = reinterpret_cast<CUdeviceptr>(userPtr) - memAddr;
  if (memOffset%NCCL_WIN_REQUIRED_ALIGNMENT != 0) {
    WARN("Window address must be suitably aligned.");
    ret = ncclInvalidArgument;
    goto fail;
  }

  CUCHECKGOTO(cuMemRetainAllocationHandle(&memHandle, reinterpret_cast<void*>(memAddr)), ret, fail_locReg);

  // Trade cumem handle for ncclSymStateMemmory*
  NCCLCHECKGOTO(symMemoryObtain(comm, memHandle, memSize, &mem), ret, fail_locReg_memHandle);
  memHandle = 0x0; // symMemoryObtain took our reference

  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);

  NCCLCHECKGOTO(symWindowCreate(
      comm, mem, memOffset, userPtr, userSize, winFlags, localRegHandle, outWinDev, nullptr, stream
    ), ret, fail_locReg_memHandle_mem_stream);
  mem = nullptr; // symWindowCreate took our reference
  
  CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail_locReg_memHandle_mem_stream_win);

  // symWindowCreate needs barrier.
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef), ret, fail_locReg_memHandle_mem_stream_win);

  cudaStreamDestroy(stream);
  return ret;

fail_locReg_memHandle_mem_stream_win:
  symWindowDestroy(comm, *outWinDev, stream);
  *outWinDev = nullptr;
  cudaStreamSynchronize(stream);
fail_locReg_memHandle_mem_stream:
  cudaStreamDestroy(stream);
  symMemoryDropRef(comm, mem);
fail_locReg_memHandle:
  if (memHandle != 0x0) { CUCHECKIGNORE(cuMemRelease(memHandle)); }
fail_locReg:
  ncclCommDeregister(comm, localRegHandle);
fail:
  *outWinDev = nullptr;
  return ret;
}

////////////////////////////////////////////////////////////////////////////////

NCCL_API(ncclResult_t, ncclCommWindowRegister, ncclComm_t comm, void* ptr, size_t size, ncclWindow_t* win, int winFlags);
ncclResult_t ncclCommWindowRegister(
    struct ncclComm* comm, void* userPtr, size_t userSize,
    struct ncclWindow_vidmem** outWinDev, int winFlags
  ) {
  ncclResult_t ret = ncclSuccess;
  int saveDev;
  struct ncclSymrRegTask* task;

  if (userPtr == nullptr || userSize == 0 || !ncclParamLocalRegister() || !ncclCuMemEnable()) goto exit;

  CUDACHECK(cudaGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  NCCLCHECKGOTO(ncclSymrInitOnce(comm), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  task->userPtr = userPtr;
  task->userSize = userSize;
  task->winFlags = winFlags;
  task->outWinDev = outWinDev;
  ncclIntruQueueEnqueue(&comm->symrState.regTaskQueue, task);
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  cudaSetDevice(saveDev);
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommWindowDeregister, ncclComm_t comm, ncclWindow_t win);
ncclResult_t ncclCommWindowDeregister(struct ncclComm* comm, struct ncclWindow_vidmem* winDev) {
  ncclResult_t ret = ncclSuccess;
  int saveDev;
  cudaStream_t stream;

  if (winDev == nullptr) goto exit;

  if (!comm->symmetricSupport) {
    NCCLCHECKGOTO(ncclCommDeregister(comm, winDev), ret, fail);
    goto exit;
  }
  CUDACHECKGOTO(cudaGetDevice(&saveDev), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail_dev);
  NCCLCHECKGOTO(symWindowDestroy(comm, winDev, stream), ret, fail_dev_stream);
fail_dev_stream:
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
fail_dev:
  cudaSetDevice(saveDev);
fail:
exit:
  return ret;
}

ncclResult_t ncclSymrFindWindow(
    struct ncclComm* comm, void const* userPtr, struct ncclSymrWindow** outWin
  ) {
  struct ncclSymrState* symr = &comm->symrState;
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr);
  int i = listFindSortedLub(&ncclSymrWindowSorted::userAddr, symr->winSorted, symr->winSortedCount, userAddr);
  if (0 < i && (userAddr - symr->winSorted[i-1].userAddr < symr->winSorted[i-1].size)) {
    *outWin = symr->winSorted[i-1].win;
  } else {
    *outWin = nullptr;
  }
  return ncclSuccess;
}

NCCL_API_CXX(ncclResult_t, ncclSymCommCreate, ncclComm_t comm, struct ncclSymCommRequirements const* reqs, struct ncclSymComm* outSymComm);
ncclResult_t ncclSymCommCreate(
    struct ncclComm* comm, struct ncclSymCommRequirements const* reqs,
    struct ncclSymComm* outSymComm
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSymrState* symr = &comm->symrState;

  memset(outSymComm, 0, sizeof(*outSymComm));
  outSymComm->rank = comm->rank;
  outSymComm->nRanks = comm->nRanks;
  outSymComm->nRanks_rcp32 = idivRcp32(comm->nRanks);
  outSymComm->nearRank = symr->nearSelf;
  outSymComm->nearSize = symr->nearSize;
  outSymComm->nearSize_rcp32 = idivRcp32(symr->nearSize);

  struct ncclSymTeam world = ncclSymTeamWorld(comm);
  struct ncclSymTeam near = ncclSymTeamInnerFactor(world, symr->nearSize);
  struct ncclSymrTeam* tmNear;
  cudaStream_t stream;
  size_t bufSizeTotal;
  struct ncclSymResourceRequirements* resReqsHead;
  struct ncclSymResourceRequirements nearMemBarReq;
  struct ncclSymResourceRequirements nearLLA2AReq;
  CUmemGenericAllocationHandle memHandle;
  struct ncclSymrMemory* mem;
  struct ncclSymrWindow* win;
  struct ncclWindow_vidmem* winHost;

  NCCLCHECKGOTO(ncclSymrInitOnce(comm), ret, fail);

  NCCLCHECKGOTO(symTeamObtain(comm, near, /*multicast=*/reqs->nearMultimem, &tmNear), ret, fail);
  outSymComm->nearMultimem.mcBasePtr = tmNear->mcBasePtr;

  { struct ncclSymTeamRequirements* tr = reqs->teamRequirementsList;
    while (tr != nullptr) {
      if (tr->multimem) {
        struct ncclSymrTeam* tm;
        NCCLCHECKGOTO(symTeamObtain(comm, tr->team, tr->multimem, &tm), ret, fail);
        if (tr->outMultimemHandle != nullptr) tr->outMultimemHandle->mcBasePtr = tm->mcBasePtr;
      }
      tr = tr->next;
    }
  }

  resReqsHead = reqs->resourceRequirementsList;

  ncclSymMemBarrierCreateRequirement(near, reqs->nearMemBarrierCount, &outSymComm->nearMemBarrier, &nearMemBarReq);
  nearMemBarReq.next = resReqsHead;
  resReqsHead = &nearMemBarReq;

  ncclSymLLA2ACreateRequirement(reqs->nearLLA2ABlockCount, reqs->nearLLA2ASlotCount, &outSymComm->nearLLA2A, &nearLLA2AReq);
  nearLLA2AReq.next = resReqsHead;
  resReqsHead = &nearLLA2AReq;

  { struct ncclSymResourceRequirements* rr = resReqsHead;
    bufSizeTotal = 0;
    while (rr != nullptr) {
      bufSizeTotal = alignUp(bufSizeTotal, std::max<size_t>(128, rr->bufferAlign));
      bufSizeTotal += rr->bufferSize;
      rr = rr->next;
    }
    bufSizeTotal = alignUp(bufSizeTotal, symr->granularity);
  }

  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);

  NCCLCHECKGOTO(symWindowTableInitOnce(comm, stream), ret, fail_stream); // ensure symr->windowTable exists
  outSymComm->windowTable = symr->windowTable;

  if (bufSizeTotal == 0) {
    outSymComm->resourceWindow = nullptr;
    outSymComm->resourceWindow_inlined = {};
  } else {
    CUmemAllocationProp memProp = {};
    memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memProp.requestedHandleTypes = ncclCuMemHandleType;
    memProp.location.id = comm->cudaDev;
    CUCHECKGOTO(cuMemCreate(&memHandle, bufSizeTotal, &memProp, 0), ret, fail);

    NCCLCHECKGOTO(symMemoryObtain(comm, memHandle, bufSizeTotal, &mem), ret, fail);
    memHandle = 0x0; // Reference given to symMemoryObtain

    NCCLCHECKGOTO(symWindowCreate( // Requires world barrier afterward.
      comm, mem, /*memOffset=*/0, nullptr, bufSizeTotal, /*winFlags=*/0,
      /*localReg=*/nullptr, &outSymComm->resourceWindow, &win,
      stream), ret, fail);
    mem = nullptr; // Reference given to symWindowCreate
    NCCLCHECKGOTO(ncclShadowPoolToHost(&symr->shadows, win->vidmem, &winHost), ret, fail_stream_mem_win);
    outSymComm->resourceWindow_inlined = *winHost;

    struct ncclSymResourceRequirements* rr = resReqsHead;
    bufSizeTotal = 0; // Sum this again to assign positions to the constituent buffers.
    while (rr != nullptr) {
      bufSizeTotal = alignUp(bufSizeTotal, std::max<size_t>(128, rr->bufferAlign));
      if (rr->outBufferHandle != nullptr) *rr->outBufferHandle = bufSizeTotal/128;
      bufSizeTotal += rr->bufferSize;
      rr = rr->next;
    }
    bufSizeTotal = alignUp(bufSizeTotal, symr->granularity);

    CUDACHECKGOTO(cudaMemsetAsync(win->userPtr, 0, bufSizeTotal, stream), ret, fail);
  }

  CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail);
  
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef), ret, fail_stream_mem_win);
  CUDACHECKGOTO(cudaStreamDestroy(stream), ret, fail);
  return ret;

fail_stream_mem_win:
  symWindowDestroy(comm, win->vidmem, stream);
  cudaStreamSynchronize(stream);
  symMemoryDropRef(comm, mem);
fail_stream:
  cudaStreamDestroy(stream);
fail:
  return ret;
}

NCCL_API_CXX(ncclResult_t, ncclSymCommDestroy, ncclComm_t comm, struct ncclSymComm const* symComm);
ncclResult_t ncclSymCommDestroy(
    struct ncclComm* comm, struct ncclSymComm const* symComm
  ) {
  //struct ncclSymrState* symr = &comm->symrState;
  if (symComm->resourceWindow != nullptr) {
    NCCLCHECK(ncclCommWindowDeregister(comm, symComm->resourceWindow));
  }
  return ncclSuccess;
}


// Get the corresponding pointer in another near rank's symmetric memory window
ncclResult_t ncclSymrGetNearRankPtr(struct ncclComm* comm, struct ncclSymrWindow* winHost, size_t offset, int nearRank, void** outPtr) {
  if (winHost == nullptr || outPtr == nullptr) {
    return ncclInvalidArgument;
  }

  struct ncclSymrState* symr = &comm->symrState;
  
  // Validate nearRank is within bounds
  if (nearRank < 0 || nearRank >= symr->nearSize) {
    return ncclInvalidArgument;
  }

  // Validate offset is within bounds
  if (offset < 0 || offset >= winHost->size) {
    return ncclInvalidArgument;
  }

  // Calculate the address with offset for the specified near rank 
  *outPtr = (void*)((uintptr_t)symr->nearFlatBase + nearRank * symr->bigSize + winHost->bigOffset + offset);
  return ncclSuccess;
}

// Get the multicast address for a given team
ncclResult_t ncclSymrGetNearTeamPtrMC(struct ncclComm* comm, struct ncclSymrWindow* winHost, size_t offset, struct ncclSymTeam nearTeam, void** outPtr){
  if (winHost == nullptr || outPtr == nullptr) {
    return ncclInvalidArgument;
  }

  if (!comm->nvlsSupport) {
    return ncclInvalidUsage;
  }

  bool multimem = true;
  struct ncclSymrTeam* tm;
  NCCLCHECK(symTeamObtain(comm, nearTeam, multimem, &tm));
    
  // Return the base multicast address for this team with offset
  *outPtr = (void*)((uintptr_t)tm->mcBasePtr + winHost->bigOffset + offset);
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////

// Find the least index strictly greater than arg.
template<typename Obj, typename Key>
static int listFindSortedLub(Key Obj::*key, Obj* sorted, int count, Key arg) {
  int lo = 0, hi = count;
  while (lo + 16 < hi) {
    int i = (lo + hi)/2;
    if (sorted[i].*key <= arg) lo = i+1;
    else hi = i;
  }
  int i = lo;
  while (i < hi && sorted[i].*key <= arg) i++;
  return i;
}

template<typename Obj>
static void listInsert(Obj** list, int* capacity, int* count, int index, Obj val) {
  if (*capacity < *count + 1) {
    *capacity *= 2;
    if (*capacity == 0) *capacity = 16;
    *list = (Obj*)realloc(*list, (*capacity)*sizeof(Obj));
  }
  for (int j = *count; j != index; j--) {
    (*list)[j] = (*list)[j-1];
  }
  (*list)[index] = val;
  *count += 1;
}

template<typename Obj>
static void listRemove(Obj* list, int* count, int index) {
  for (int i = index; i+1 < *count; i++) {
    list[i] = list[i+1];
  }
  *count -= 1;
}

