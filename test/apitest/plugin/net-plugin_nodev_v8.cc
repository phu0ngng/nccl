/*************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"

#define __hidden __attribute__((visibility("hidden")))

__hidden ncclResult_t pluginInit(ncclDebugLogger_t logFunction) { return ncclSuccess; }
__hidden ncclResult_t pluginDevices(int* ndev) { *ndev = 0; return ncclSuccess; }
__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_v8_t* props) { return ncclSuccess; }
__hidden ncclResult_t pluginListen(int dev, void* handle, void** listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginConnect(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) { return ncclInternalError; }
__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMr(void* collComm, void* data, size_t size, int type, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMrDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginDeregMr(void* collComm, void* mhandle) { return ncclInternalError;}
__hidden ncclResult_t pluginIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginTest(void* request, int* done, int* size) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseSend(void* sendComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseRecv(void* recvComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseListen(void* listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecvConsumed(void* recvComm, int n, void* request) { return ncclInternalError; }

extern "C" const ncclNet_v8_t ncclNetPlugin_v8 = {
  .name = "ncclNetPlugin_v8",
  .init = pluginInit,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties,
  .listen = pluginListen,
  .connect = pluginConnect,
  .accept = pluginAccept,
  .regMr = pluginRegMr,
  .regMrDmaBuf = pluginRegMrDmaBuf,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend,
  .irecv = pluginIrecv,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
  .irecvConsumed = pluginIrecvConsumed,
};
