/*************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <net.h>

#define __hidden __attribute__((visibility("hidden")))

struct pluginContext {
  int data;
};

__hidden ncclResult_t pluginInit(void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) { 
  struct pluginContext* context = (struct pluginContext*)malloc(sizeof(*context));
  *ctx = context;
  return ncclSuccess; 
}
__hidden ncclResult_t pluginDevices(int* ndev) { *ndev = 0; return ncclSuccess; }
__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_v11_t* props) { return ncclSuccess; }
__hidden ncclResult_t pluginListen(void* ctx, int dev, void* handle, void** listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginConnect(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) { return ncclInternalError; }
__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMr(void* collComm, void* data, size_t size, int type, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMrDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginDeregMr(void* collComm, void* mhandle) { return ncclInternalError;}
__hidden ncclResult_t pluginIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginTest(void* request, int* done, int* size) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseSend(void* sendComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseRecv(void* recvComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseListen(void* listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecvConsumed(void* recvComm, int n, void* request) { return ncclInternalError; }
__hidden ncclResult_t pluginGetDeviceMr(void* comm, void* mhandle, void** dptr_mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginMakeVDevice(int* d, ncclNetVDeviceProps_v11_t* props) { return ncclInternalError; }
__hidden ncclResult_t pluginFinalize(void* ctx) {
  free(ctx);
  return ncclSuccess;
}
__hidden ncclResult_t pluginSetNetAttr(void* ctx, ncclNetAttr_v11_t* netAttr) { return ncclInternalError; }

extern "C" __attribute__((visibility("default"))) const ncclNet_v11_t ncclNetPlugin_v11 = {
  .name = "ncclNetPlugin_v11",
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
  .getDeviceMr = pluginGetDeviceMr,
  .irecvConsumed = pluginIrecvConsumed,
  .makeVDevice   = pluginMakeVDevice,
  .finalize = pluginFinalize,
  .setNetAttr = pluginSetNetAttr,
};
