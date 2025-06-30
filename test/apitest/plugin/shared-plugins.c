/*************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "net.h"
#include "tuner.h"
#include "profiler.h"
#include "net_device.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define __hidden __attribute__((visibility("hidden")))

#define NCCL_PLUGIN_MAX_RECVS 1
#define NCCL_MAX_NET_SIZE_BYTES (1*1024*1024*1024*1024L) //1TB

#define MAX_CONTEXT_COUNT 8
#define MAX_DEVICE_COUNT 2

static struct pluginContext {
  uint64_t commId;
  int devices;
} context[MAX_CONTEXT_COUNT] = {
    { 0UL, MAX_DEVICE_COUNT },
    { 0UL, MAX_DEVICE_COUNT },
    { 0UL, MAX_DEVICE_COUNT },
    { 0UL, MAX_DEVICE_COUNT },
    { 0UL, MAX_DEVICE_COUNT },
    { 0UL, MAX_DEVICE_COUNT },
    { 0UL, MAX_DEVICE_COUNT },
    { 0UL, MAX_DEVICE_COUNT },
};

__hidden int netContextCounter;
__hidden int tunerContextCounter;
__hidden int profilerContextCounter;

struct netPluginListenComm {
  int dev;
  struct pluginContext* context;
};

struct netPluginSendComm {
  int data;
  struct pluginContext* context;
};

struct netPluginRecvComm {
  int data;
  struct pluginContext* context;
};

struct netPluginRequest {
  int sizes[NCCL_PLUGIN_MAX_RECVS];
  int tags[NCCL_PLUGIN_MAX_RECVS];
  int ntags;
};

struct netPluginMemHandle {
  int data;
};

__hidden ncclResult_t netPluginInit(void** ctx, uint64_t commId, ncclDebugLogger_t logfn, ncclProfilerCallback_t profFunction) {
  int counter = __atomic_fetch_add(&netContextCounter, 1, __ATOMIC_RELAXED);
  if (counter == MAX_CONTEXT_COUNT) return ncclInternalError;
  context[counter].commId = commId;
  //fprintf(stdout, "commId: %lu\n", commId);
  //fprintf(stdout, "context[%d]: %lu\n", counter, context[counter].commId);
  *ctx = &context[counter];
  return ncclSuccess;
}

__hidden ncclResult_t netPluginDevices(int* ndev) { *ndev = context[0].devices; return ncclSuccess; }
__hidden ncclResult_t netPluginGetProperties(int dev, ncclNetProperties_t* props) {
  props->name = "ncclNetPlugin_v11";
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
  props->vProps.devs[0] = MAX_DEVICE_COUNT;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  return ncclSuccess;
}
__hidden ncclResult_t netPluginListen(void* ctx, int dev, void* handle, void** listenComm) {
  if (((struct pluginContext *)ctx)->commId == 0UL) return ncclInternalError;
  struct netPluginListenComm* comm = (struct netPluginListenComm*)malloc(sizeof(*comm));
  comm->dev = dev;
  comm->context = ctx;
  *listenComm = comm;
  return ncclSuccess;
}
__hidden ncclResult_t netPluginConnect(void* ctx, int dev, ncclNetCommConfig_t* config, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  if (((struct pluginContext *)ctx)->commId == 0UL) return ncclInternalError;
  struct netPluginSendComm* comm = (struct netPluginSendComm*)malloc(sizeof(*comm));
  comm->context = ctx;
  *sendComm = comm;
  return ncclSuccess;
}
__hidden ncclResult_t netPluginAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) {
  struct netPluginListenComm* listen = (struct netPluginListenComm*)listenComm;
  if (listen->context->commId == 0UL) return ncclInternalError;
  struct netPluginRecvComm* comm = (struct netPluginRecvComm*)malloc(sizeof(*comm));
  comm->context = listen->context;
  *recvComm = comm;
  return ncclSuccess;
}
__hidden ncclResult_t netPluginRegMr(void* collComm, void* data, size_t size, int type, void** mhandle) {
  struct netPluginMemHandle* m = (struct netPluginMemHandle*)malloc(sizeof(*m));
  *mhandle = m;
  return ncclSuccess;
}
__hidden ncclResult_t netPluginRegMrDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  return netPluginRegMr(collComm, data, size, type, mhandle);
}
__hidden ncclResult_t netPluginDeregMr(void* collComm, void* mhandle) {
  free(mhandle);
  return ncclSuccess;
}
__hidden ncclResult_t netPluginIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  if (((struct netPluginSendComm *)sendComm)->context->commId == 0UL) return ncclInternalError;
  struct netPluginRequest* r = (struct netPluginRequest*)malloc(sizeof(*r));
  r->sizes[0] = size;
  r->tags[0] = tag;
  r->ntags = 1;
  *request = r;
  return ncclSuccess;
}
__hidden ncclResult_t netPluginIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  if (((struct netPluginRecvComm *)recvComm)->context->commId == 0UL) return ncclInternalError;
  struct netPluginRequest* r = (struct netPluginRequest*)malloc(sizeof(*r));
  r->ntags = n;
  memcpy(r->sizes, sizes, sizeof(int)*n);
  memcpy(r->tags, tags, sizeof(int)*n);
  *request = r;
  return ncclSuccess;
}
__hidden ncclResult_t netPluginIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  if (((struct netPluginRecvComm *)recvComm)->context->commId == 0UL) return ncclInternalError;
  struct netPluginRequest* r = (struct netPluginRequest*)malloc(sizeof(*r));
  r->ntags = n;
  memcpy(r->sizes, sizes, sizeof(int)*n);
  *request = r;
  return ncclSuccess;
}
__hidden ncclResult_t netPluginTest(void* request, int* done, int* size) {
  if (done) *done = 1;
  free(request);
  return ncclSuccess;
}
__hidden ncclResult_t netPluginCloseSend(void* sendComm) {
  if (((struct netPluginSendComm *)sendComm)->context->commId == 0UL) return ncclInternalError;
  free(sendComm);
  return ncclSuccess;
}
__hidden ncclResult_t netPluginCloseRecv(void* recvComm) {
  if (((struct netPluginRecvComm *)recvComm)->context->commId == 0UL) return ncclInternalError;
  free(recvComm);
  return ncclSuccess;
}
__hidden ncclResult_t netPluginCloseListen(void* listenComm) {
  if (((struct netPluginListenComm *)listenComm)->context->commId == 0UL) return ncclInternalError;
  free(listenComm);
  return ncclSuccess;
}
__hidden ncclResult_t netPluginIrecvConsumed(void* recvComm, int n, void* request) {
  if (((struct netPluginRecvComm *)recvComm)->context->commId == 0UL) return ncclInternalError;
  return ncclSuccess;
}
__hidden ncclResult_t netPluginGetDeviceMr(void* comm, void* mhandle, void** dptr_mhandle) {
  return ncclSuccess;
}
__hidden ncclResult_t netPluginMakeVDevice(void* ctx, int* d, ncclNetVDeviceProps_t* props) {
  if (((struct pluginContext *)ctx)->commId == 0UL) return ncclInternalError;
  return ncclSuccess;
}

__hidden ncclResult_t netPluginFinalize(void *ctx) {
  if (((struct pluginContext *)ctx)->commId == 0UL) return ncclInternalError;
  ((struct pluginContext *)ctx)->commId = 0UL;
  __atomic_store_n(&netContextCounter, 0, __ATOMIC_RELAXED);
  return ncclSuccess;
}

const ncclNet_t ncclNetPlugin_v11 = {
  .name = "ncclNetPlugin_v11",
  .init = netPluginInit,
  .devices = netPluginDevices,
  .getProperties = netPluginGetProperties,
  .listen = netPluginListen,
  .connect = netPluginConnect,
  .accept = netPluginAccept,
  .regMr = netPluginRegMr,
  .regMrDmaBuf = netPluginRegMrDmaBuf,
  .deregMr = netPluginDeregMr,
  .isend = netPluginIsend,
  .irecv = netPluginIrecv,
  .iflush = netPluginIflush,
  .test = netPluginTest,
  .closeSend = netPluginCloseSend,
  .closeRecv = netPluginCloseRecv,
  .closeListen = netPluginCloseListen,
  .getDeviceMr = netPluginGetDeviceMr,
  .irecvConsumed = netPluginIrecvConsumed,
  .makeVDevice   = netPluginMakeVDevice,
  .finalize = netPluginFinalize,
};

__hidden ncclResult_t tunerPluginInit(void** ctx, uint64_t commId, size_t nranks, size_t nnodes, ncclDebugLogger_t logFunction) {
  int counter = __atomic_fetch_add(&tunerContextCounter, 1, __ATOMIC_RELAXED);
  __atomic_fetch_sub(&context[counter].devices, 1, __ATOMIC_RELAXED);
  //fprintf(stdout, "commId: %lu\n", commId);
  //fprintf(stdout, "context[%d]: %lu\n", counter, context[counter].commId);
  assert(context[counter].commId == commId);
  return ncclSuccess;
}

__hidden ncclResult_t tunerPluginGetCollInfo(void* ctx, ncclFunc_t collType, size_t nbytes, int numpipeops, float** costtable, int numalgo, int numproto, int regbuff, int* nchannels) { return ncclSuccess; }

__hidden ncclResult_t tunerPluginFinalize(void* ctx) {
  __atomic_store_n(&tunerContextCounter, 0, __ATOMIC_RELAXED);
  return ncclSuccess;
}

const ncclTuner_t ncclTunerPlugin_v5 = {
  .name = "ncclTunerPlugin_v5",
  .init = tunerPluginInit,
  .getCollInfo = tunerPluginGetCollInfo,
  .finalize = tunerPluginFinalize,
};

__hidden ncclResult_t profilerPluginInit(void** ctx, uint64_t commId, int* eActivationMask, const char* commName, int nnodes, int nranks, int rank, ncclDebugLogger_t logfn) {
  int counter = __atomic_fetch_add(&profilerContextCounter, 1, __ATOMIC_RELAXED);
  __atomic_fetch_sub(&context[counter].devices, 1, __ATOMIC_RELAXED);
  //fprintf(stdout, "commId: %lu\n", commId);
  //fprintf(stdout, "context[%d]: %lu\n", counter, context[counter].commId);
  assert(context[counter].commId == commId);
  return ncclSuccess;
}

__hidden ncclResult_t profilerPluginStartEvent(void* ctx, void** eHandle, ncclProfilerEventDescr_t* eDescr) { return ncclSuccess; }

__hidden ncclResult_t profilerPluginRecordEventState(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs) { return ncclSuccess; }

__hidden ncclResult_t profilerPluginStopEvent(void* eHandle) { return ncclSuccess; }

__hidden ncclResult_t profilerPluginFinalize(void* context) { return ncclSuccess; }

const ncclProfiler_t ncclProfiler_v5 = {
  .name = "ncclProfilerPlugin_v5",
  .init = profilerPluginInit,
  .startEvent = profilerPluginStartEvent,
  .stopEvent = profilerPluginStopEvent,
  .recordEventState = profilerPluginRecordEventState,
  .finalize = profilerPluginFinalize,
};
