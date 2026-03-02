/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* IPC via Unix domain sockets and fd passing is not supported on Windows.
 * Provide stubs so the API compiles; callers get ncclInternalError when using IPC. */

#include "ipcsocket.h"
#include "utils.h"
#include "os.h"
#include <stdlib.h>
#include <string.h>

ncclResult_t ncclIpcSocketInit(ncclIpcSocket *handle, int rank, uint64_t hash, volatile uint32_t* abortFlag) {
  (void)rank;
  (void)hash;
  (void)abortFlag;
  if (handle == NULL) return ncclInternalError;
  handle->fd = -1;
  handle->socketName[0] = '\0';
  handle->abortFlag = nullptr;
  return ncclInternalError;  /* IPC sockets not supported on Windows */
}

ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket* handle, int* fd) {
  if (handle == NULL) {
    WARN("ncclIpcSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (fd) *fd = (int)handle->fd;
  return ncclSuccess;
}

ncclResult_t ncclIpcSocketClose(ncclIpcSocket *handle) {
  if (handle == NULL) return ncclInternalError;
  if (handle->fd < 0) return ncclSuccess;
  handle->fd = -1;
  handle->socketName[0] = '\0';
  return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, int *recvFd) {
  (void)handle;
  (void)hdr;
  (void)hdrLen;
  (void)recvFd;
  return ncclInternalError;  /* not supported on Windows */
}

ncclResult_t ncclIpcSocketRecvFd(ncclIpcSocket *handle, int *fd) {
  return ncclIpcSocketRecvMsg(handle, NULL, 0, fd);
}

ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, const int sendFd, int rank, uint64_t hash) {
  (void)handle;
  (void)hdr;
  (void)hdrLen;
  (void)sendFd;
  (void)rank;
  (void)hash;
  return ncclInternalError;  /* not supported on Windows */
}

ncclResult_t ncclIpcSocketSendFd(ncclIpcSocket *handle, const int sendFd, int rank, uint64_t hash) {
  return ncclIpcSocketSendMsg(handle, NULL, 0, sendFd, rank, hash);
}
