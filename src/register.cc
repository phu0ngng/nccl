/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "nccl.h"
#include "comm.h"
#include "net.h"

struct ncclNetCommReg {
  void* rComm;
  void* sComm;
  void* handle;
};

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  *handle = NULL;
  int netDevs;
  NCCLCHECK(comm->ncclNet->devices(&netDevs));

  struct ncclNetCommReg* commReg;
  NCCLCHECK(ncclCalloc(&commReg, netDevs));

  // Find local devices for p2p operations
  for (int c=0; c<comm->p2pnChannels; c++) {
    int dev;
    NCCLCHECK(ncclTopoGetLocalNet(comm->topo, comm->rank, c, &dev));
    commReg[dev].handle = &commReg[dev].handle; // Mark as non-NULL
  }

  ncclDebugNoWarn = NCCL_NET;
  void *lComm = NULL;
  for (int dev=0; dev<netDevs; dev++) {
    if (commReg[dev].handle == NULL) continue;
    commReg[dev].handle = NULL;
    ncclNetHandle_t netHandle;
    ncclResult_t ret;
    NCCLCHECKGOTO(comm->ncclNet->listen(dev, &netHandle, &lComm), ret, cleanup);

    bool connected;
    connected = false;
    while (!connected) {
      // If we're aborting now, skip to cleanup
      if (*comm->abortFlag) {
        goto cleanup;
      }

      if (commReg[dev].sComm == NULL)
        NCCLCHECKGOTO(comm->ncclNet->connect(dev, &netHandle, &commReg[dev].sComm), ret, cleanup);

      if (commReg[dev].rComm == NULL)
        NCCLCHECKGOTO(comm->ncclNet->accept(lComm, &commReg[dev].rComm), ret, cleanup);

      connected = (commReg[dev].rComm != NULL) && (commReg[dev].sComm != NULL);
    }
    NCCLCHECK(comm->ncclNet->closeListen(lComm));
    lComm = NULL;

    comm->ncclNet->regMr(commReg[dev].sComm, buff, size, NCCL_PTR_CUDA, &commReg[dev].handle);
  }
  ncclDebugNoWarn = 0;
  *handle = commReg;
  return ncclSuccess;
cleanup:
  if (lComm) NCCLCHECK(comm->ncclNet->closeListen(lComm));
  for (int dev=0; dev<netDevs; dev++) {
    if (commReg[dev].rComm != NULL)
      NCCLCHECK(comm->ncclNet->closeRecv(commReg[dev].rComm));
    if (commReg[dev].sComm != NULL)
      NCCLCHECK(comm->ncclNet->closeSend(commReg[dev].sComm));
  }
  free(commReg);
  ncclDebugNoWarn = 0;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUnregister, const ncclComm_t comm, void* handle);
ncclResult_t ncclCommUnregister(const ncclComm_t comm, void* handle) {
  int netDevs;
  NCCLCHECK(comm->ncclNet->devices(&netDevs));

  struct ncclNetCommReg* commReg = (struct ncclNetCommReg*)handle;
  ncclDebugNoWarn = NCCL_NET;
  for (int dev=0; dev<netDevs; dev++) {
    if (commReg[dev].handle != NULL) comm->ncclNet->deregMr(commReg[dev].sComm, commReg[dev].handle);
  }
  ncclDebugNoWarn = 0;
  return ncclSuccess;
}
