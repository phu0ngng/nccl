/*************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"
#include <stdio.h>
#include <string.h>

#define __hidden __attribute__((visibility("hidden")))
#define NCCL_PLUGIN_MAX_RECVS 1

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

__hidden ncclResult_t pluginInit(ncclDebugLogger_t logFunction) { return ncclSuccess; }
__hidden ncclResult_t pluginDevices(int* ndev) { *ndev = 1; return ncclSuccess; }
__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_v6_t* props) {
  props->name = (char *)"ncclNetPlugin_v6";
  props->pciPath = NULL;
  props->guid = 0;
  props->ptrSupport = NCCL_PTR_HOST;
  props->speed = 100000;
  props->port = 0;
  props->latency = 0;
  props->maxComms = 1024*1024;
  props->maxRecvs = NCCL_PLUGIN_MAX_RECVS;
  return ncclSuccess;
}
__hidden ncclResult_t pluginListen(int dev, void* handle, void** listenComm) {
  struct pluginListenComm* comm = (struct pluginListenComm*)malloc(sizeof(*comm));
  comm->dev = dev;
  *listenComm = comm;
  return ncclSuccess;
}
__hidden ncclResult_t pluginConnect(int dev, void* handle, void** sendComm) {
  struct pluginSendComm* comm = (struct pluginSendComm*)malloc(sizeof(*comm));
  *sendComm = comm;
  return ncclSuccess;
}
__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm) {
  struct pluginRecvComm* comm = (struct pluginRecvComm*)malloc(sizeof(*comm));
  *recvComm = comm;
  return ncclSuccess;
}
__hidden ncclResult_t pluginRegMr(void* collComm, void* data, int size, int type, void** mhandle) {
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
__hidden ncclResult_t pluginIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  struct pluginRequest* r = (struct pluginRequest*)malloc(sizeof(*r));
  r->sizes[0] = size;
  r->tags[0] = tag;
  r->ntags = 1;
  *request = r;
  return ncclSuccess;
}
__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
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

extern "C" const ncclNet_v6_t ncclNetPlugin_v6 = {
  .name = "ncclNetPlugin_v6",
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
};
