/*************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"
#include "net_device.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define __hidden __attribute__((visibility("hidden")))

#define NCCL_PLUGIN_MAX_RECVS 1
#define NCCL_MAX_NET_SIZE_BYTES (1*1024*1024*1024*1024L) //1TB

struct pluginListenComm {
  int dev;
};

struct pluginSendComm {
  int data;
};

struct pluginRecvComm {
  int data;
};

struct pluginRequest {
  int sizes[NCCL_PLUGIN_MAX_RECVS];
  int tags[NCCL_PLUGIN_MAX_RECVS];
  int ntags;
};

struct pluginMemHandle {
  int data;
};

struct pluginContext {
  int data;
};

__hidden void** netContext;

__hidden ncclResult_t pluginInit(void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  netContext = ctx;
  return ncclSuccess;
}
__hidden ncclResult_t pluginDevices(int* ndev) { *ndev = 1; return ncclSuccess; }
__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_v11_t* props) {
  props->name = (char *)"ncclNetPlugin_v11";
  props->pciPath = NULL;
  props->guid = 0;
  props->ptrSupport = NCCL_PTR_HOST;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = 100000;
  props->port = 0;
  props->latency = 0;
  props->maxComms = 1024*1024;
  props->maxRecvs = NCCL_PLUGIN_MAX_RECVS;
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxMultiRequestSize = NCCL_PLUGIN_MAX_RECVS;
  return ncclSuccess;
}
__hidden ncclResult_t pluginListen(void* ctx, int dev, void* handle, void** listenComm) {
  struct pluginListenComm* comm = (struct pluginListenComm*)malloc(sizeof(*comm));
  comm->dev = dev;
  *listenComm = comm;
  return ncclSuccess;
}
__hidden ncclResult_t pluginConnect(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  struct pluginSendComm* comm = (struct pluginSendComm*)malloc(sizeof(*comm));
  *sendComm = comm;
  return ncclSuccess;
}
__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) {
  struct pluginRecvComm* comm = (struct pluginRecvComm*)malloc(sizeof(*comm));
  *recvComm = comm;
  return ncclSuccess;
}
__hidden ncclResult_t pluginRegMr(void* collComm, void* data, size_t size, int type, void** mhandle) {
  struct pluginMemHandle* m = (struct pluginMemHandle*)malloc(sizeof(*m));
  *mhandle = m;
  return ncclSuccess;
}
__hidden ncclResult_t pluginRegMrDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  return pluginRegMr(collComm, data, size, type, mhandle);
}
__hidden ncclResult_t pluginDeregMr(void* collComm, void* mhandle) {
  free(mhandle);
  return ncclSuccess;
}
__hidden ncclResult_t pluginIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  struct pluginRequest* r = (struct pluginRequest*)malloc(sizeof(*r));
  r->sizes[0] = size;
  r->tags[0] = tag;
  r->ntags = 1;
  *request = r;
  return ncclSuccess;
}
__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  struct pluginRequest* r = (struct pluginRequest*)malloc(sizeof(*r));
  r->ntags = n;
  memcpy(r->sizes, sizes, sizeof(int)*n);
  memcpy(r->tags, tags, sizeof(int)*n);
  *request = r;
  return ncclSuccess;
}
__hidden ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct pluginRequest* r = (struct pluginRequest*)malloc(sizeof(*r));
  r->ntags = n;
  memcpy(r->sizes, sizes, sizeof(int)*n);
  *request = r;
  return ncclSuccess;
}
__hidden ncclResult_t pluginTest(void* request, int* done, int* size) {
  return ncclSuccess;
}
__hidden ncclResult_t pluginCloseSend(void* sendComm) {
  free(sendComm);
  return ncclSuccess;
}
__hidden ncclResult_t pluginCloseRecv(void* recvComm) {
  free(recvComm);
  return ncclSuccess;
}
__hidden ncclResult_t pluginCloseListen(void* listenComm) {
  free(listenComm);
  return ncclSuccess;
}
__hidden ncclResult_t pluginIrecvConsumed(void* recvComm, int n, void* request) {
  return ncclSuccess;
}
__hidden ncclResult_t pluginGetDeviceMr(void* comm, void* mhandle, void** dptr_mhandle) {
  return ncclSuccess;
}
__hidden ncclResult_t pluginMakeVDevice(int* d, ncclNetVDeviceProps_v11_t* props) {
  return ncclSuccess;
}
__hidden ncclResult_t pluginFinalize(void* ctx) {
  free(ctx);
  return ncclSuccess;
}
__hidden ncclResult_t pluginSetNetAttr(void* ctx, ncclNetAttr_v11_t* netAttr) {
  return ncclSuccess;
}

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

__hidden ncclNetDeviceType ncclGinDeviceType = NCCL_NET_DEVICE_HOST;

__hidden ncclResult_t ginPluginInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  if (ctx != netContext) ncclGinDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginGetProperties(int dev, ncclNetProperties_v11_t* props) {
  props->name = (char *)"ncclGinPlugin_v11";
  props->pciPath = NULL;
  props->guid = 0;
  props->ptrSupport = NCCL_PTR_HOST;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = 100000;
  props->port = 0;
  props->latency = 0;
  props->maxComms = 1024*1024;
  props->maxRecvs = NCCL_PLUGIN_MAX_RECVS;
  props->netDeviceType = ncclGinDeviceType;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxMultiRequestSize = NCCL_PLUGIN_MAX_RECVS;
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginListen(void* ctx, int dev, void* handle, void** listenComm) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginCreateContext(void* collComm, int nSignals, int nCounters, void** ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle, void** ginHandle) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginDeregMrSym(void* collComm, void* mhandle) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginDestroyContext(void* ginContext) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginCloseColl(void* collComm) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginCloseListen(void* listenComm) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginIput(void* collComm, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginIputSignal(void* collComm, uint64_t srcOff, void*srcMhandle, size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff, void* signalMhandle, uint64_t signalValue, uint32_t signalOp, void**request) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginTest(void* collComm, void* request, int* done) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginProgress(void* collComm) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginQueryLastError(void* ginCtx, bool* hasError) {
  return ncclSuccess;
}

__hidden ncclResult_t ginPluginFinalize(void* ctx) {
  return ncclSuccess;
}

extern "C" __attribute__((visibility("default"))) const ncclGin_v11_t ncclGinPlugin_v11 = {
  .name = "ncclGinPlugin_v11",
  .init = ginPluginInit,
  .devices = pluginDevices,
  .getProperties = ginPluginGetProperties,
  .listen = ginPluginListen,
  .connect = ginPluginConnect,
  .createContext = ginPluginCreateContext,
  .regMrSym = ginPluginRegMrSym,
  .regMrSymDmaBuf = ginPluginRegMrSymDmaBuf,
  .deregMrSym = ginPluginDeregMrSym,
  .destroyContext = ginPluginDestroyContext,
  .closeColl = ginPluginCloseColl,
  .closeListen = ginPluginCloseListen,
  .iput = ginPluginIput,
  .iputSignal = ginPluginIputSignal,
  .test = ginPluginTest,
  .ginProgress = ginPluginProgress,
  .queryLastError = ginPluginQueryLastError,
  .finalize = ginPluginFinalize,
};
