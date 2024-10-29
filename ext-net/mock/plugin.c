/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"
#include <string.h>
#include <stdio.h>
#include <pthread.h>

pthread_mutex_t mockLock = PTHREAD_MUTEX_INITIALIZER;

#define __hidden __attribute__ ((visibility("hidden")))
#define NCCL_PLUGIN_MAX_RECVS 8
int max_requests = NCCL_NET_MAX_REQUESTS;
int nPhysDevs = 0;
int nVirtualDevs = 0;
#define MAX_MOCK_DEVS  32
#define MAX_MOCK_VDEVS MAX_MOCK_DEVS*8
ncclNetVDeviceProps_t mockVDevProps[MAX_MOCK_VDEVS];
ncclNetProperties_t   mockProps[MAX_MOCK_DEVS];

struct mockListenComm {
  int dev;
};

struct mockSendComm {
  int data;
};

struct mockRecvComm {
  int data;
};

struct mockRequest {
  int sizes[NCCL_PLUGIN_MAX_RECVS];
  int tags[NCCL_PLUGIN_MAX_RECVS];
  int ntags;
};

struct mockMrHandle {
  int data;
};

struct mockHandle {
  // union ncclSocketAddress connectAddr; // Filled by the target
  uint64_t magic; // random number to help debugging
  // struct ncclIbCommStage stage; // Used by the other side when connecting
};

__hidden ncclResult_t pluginMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  if (nVirtualDevs < MAX_MOCK_VDEVS) {
    if (props->ndevs > NCCL_NET_MAX_DEVS_PER_NIC) return ncclInvalidArgument;
    int deviceIndex = nVirtualDevs;
    memcpy(mockVDevProps + deviceIndex, props, sizeof(ncclNetVDeviceProps_t));
    nVirtualDevs++;
    return ncclSuccess;
  } else {
    return ncclInvalidUsage;
  }
}

ncclResult_t pluginAddDevice(ncclNetProperties_t* props) {
  if (nPhysDevs < MAX_MOCK_DEVS) {
    int deviceIndex = nPhysDevs;
    ncclNetProperties_t* dst = mockProps + deviceIndex;
    memcpy(dst, props, sizeof(ncclNetProperties_t));
    nPhysDevs++;
    ncclNetVDeviceProps_t vProps = {};
    vProps.ndevs = 1;
    vProps.devs[0] = deviceIndex;
    return pluginMakeVDevice(&deviceIndex, &vProps);
  } else {
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

void mallocAndStrCpy(char** dst, const char* src) {
  *dst = (char*) malloc(strlen(src)*sizeof(char));
  strcpy(*dst, src);
}

__hidden ncclResult_t pluginInit(ncclDebugLogger_t logFunction) {
  pthread_mutex_lock(&mockLock);
  for (int i = 0; i < nPhysDevs; i++) {
    ncclNetProperties_t* m = mockProps + i;
    free(m->name);
    free(m->pciPath);
  }

  memset(mockProps,     0, sizeof(mockProps));
  memset(mockVDevProps, 0, sizeof(mockVDevProps));
  nPhysDevs    = 0;
  nVirtualDevs = 0;

  // Add test devices for now
  ncclNetProperties_t props0 = {};
  mallocAndStrCpy(&props0.name, "mock_0");
  mallocAndStrCpy(&props0.pciPath, "/sys/devices/pci0000:00/0000:00:02.0/0000:02:00.0/0000:03:08.0/0000:05:00.0");
  props0.guid             = 0;
  props0.ptrSupport       = 0;
  props0.regIsGlobal      = 1;
  props0.forceFlush       = 0;
  props0.speed            = 100000;
  props0.port             = 1;
  props0.maxComms         = 1024;
  props0.maxRecvs         = NCCL_PLUGIN_MAX_RECVS;
  props0.netDeviceType    = NCCL_NET_DEVICE_HOST;
  props0.netDeviceVersion = 0;
  props0.maxP2pBytes      = NCCL_MAX_NET_SIZE_BYTES;
  props0.maxCollBytes     = NCCL_MAX_NET_SIZE_BYTES;
  pluginAddDevice(&props0);

  // Dev 1
  ncclNetProperties_t props1 = {};
  mallocAndStrCpy(&props1.name, "mock_1");
  mallocAndStrCpy(&props1.pciPath, "/sys/devices/pci0000:00/0000:00:03.0/0000:09:00.0/0000:0a:0c.0/0000:0d:00.0");
  props1.guid = 1;
  props1.ptrSupport = 0;
  props1.regIsGlobal = 1;
  props1.forceFlush  = 0;
  props1.speed       = 100000;
  props1.port       = 1;
  props1.maxComms       = 1024;
  props1.maxRecvs       =   NCCL_PLUGIN_MAX_RECVS;
  props1.netDeviceType    = NCCL_NET_DEVICE_HOST;
  props1.netDeviceVersion = 0;
  props1.maxP2pBytes      = NCCL_MAX_NET_SIZE_BYTES;
  props1.maxCollBytes     = NCCL_MAX_NET_SIZE_BYTES;
  pluginAddDevice(&props1);
  pthread_mutex_unlock(&mockLock);

  return ncclSuccess;
}

__hidden ncclResult_t pluginDevices(int* ndev) {
  *ndev = nVirtualDevs;
  return ncclSuccess;
}

__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_t* props) {
  if (dev < nVirtualDevs) {
    int pDevIndex = mockVDevProps[dev].devs[0];
    memcpy(props, mockProps + pDevIndex, sizeof(ncclNetProperties_t));
    props->vProps = mockVDevProps[dev];
    return ncclSuccess;
  } else {
    return ncclInvalidUsage;
  }
}

__hidden ncclResult_t pluginListen(int dev, void* /*handle*/, void** listenComm) {
  if (dev < nVirtualDevs) {
    mockListenComm* lComm = (mockListenComm*) malloc(sizeof(mockListenComm));
    lComm->dev = dev;
    *listenComm = lComm;
    return ncclSuccess;
  } else {
    return ncclInvalidUsage;
  }
}

__hidden ncclResult_t pluginConnect(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  if (dev < nVirtualDevs) {
    mockSendComm* sComm = (mockSendComm*) malloc(sizeof(mockSendComm));
    *sendComm = sComm;
    return ncclSuccess;
  } else {
    return ncclInvalidUsage;
  }
}

__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  mockListenComm* lComm = (mockListenComm*) listenComm;
  if (lComm->dev < nVirtualDevs) {
    mockRecvComm* rComm = (mockRecvComm*) malloc(sizeof(mockRecvComm));
    *recvComm = rComm;
    return ncclSuccess;
  } else {
    return ncclInvalidUsage;
  }
}

__hidden ncclResult_t pluginIrecvConsumed(void* recvComm, int n, void* request) {
  return ncclSuccess;
}

__hidden ncclResult_t pluginCloseSend(void* sendComm) { 
  if (sendComm) free(sendComm);
  return ncclSuccess;
}

__hidden ncclResult_t pluginCloseRecv(void* recvComm) { 
  if (recvComm) free(recvComm);
  return ncclSuccess;
}

__hidden ncclResult_t pluginCloseListen(void* listenComm) { 
  if (listenComm) free(listenComm);
  return ncclSuccess;
}

__hidden ncclResult_t pluginIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
  mockRequest* r = (mockRequest*) malloc(sizeof(mockRequest));
  r->sizes[0] = size;
  r->tags[0]  = tag;
  r->ntags    = 1;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
  mockRequest* r = (mockRequest*) malloc(sizeof(mockRequest));
  r->ntags = n;
  memcpy(r->sizes, sizes, sizeof(int)*n);
  memcpy(r->tags, tags, sizeof(int)*n);
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (last == -1) return ncclSuccess;

  mockRequest* r = (mockRequest*) malloc(sizeof(mockRequest));
  r->ntags = n;
  memcpy(r->sizes, sizes, sizeof(int)*n);
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t pluginTest(void* request, int* done, int* sizes) {
  *done = 1;
  if (request == NULL) return ncclSuccess;
  mockRequest* r = (mockRequest*) request;
  if (sizes) memcpy(sizes, r->sizes, sizeof(int)*r->ntags);
  free(request);
  return ncclSuccess;
}

__hidden ncclResult_t pluginRegMr(void* collComm, void* data, size_t size, int type, void** mhandle) {
  *mhandle = (mockMrHandle*) malloc(sizeof(mockMrHandle));
  return ncclSuccess;
}

__hidden ncclResult_t pluginRegMrDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  return pluginRegMr(collComm, data, size, type, mhandle);
}

__hidden ncclResult_t pluginDeregMr(void* collComm, void* mhandle) {
  if (mhandle) free(mhandle);
  return ncclSuccess;
}

__hidden ncclResult_t pluginCollConnect(void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  *collComm = (mockSendComm*) malloc(sizeof(mockSendComm));
  return ncclSuccess;
}

__hidden ncclResult_t pluginReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported) {
  *supported = 1;
  return ncclSuccess;
}

__hidden ncclResult_t pluginIAllReduce(void* collComm, void* sendData, void* recvData, size_t count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
  mockRequest* r = (mockRequest*) malloc(sizeof(mockRequest));
  r->ntags = 1;
  r->sizes[0] = count;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t pluginIAllGather(void* collComm, void* sendData, int nRecvParts, ncclNetSGE_v9_t* recvParts,
                             size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                             void* sendMhandle, void** request) {
  mockRequest* r = (mockRequest*) malloc(sizeof(mockRequest));
  r->ntags = 1;
  r->sizes[0] = bytesPerRank;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t pluginIReduceScatter(void* collComm, int nSendParts, ncclNetSGE_v9_t* sendParts, void* recvData,
                                 size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                                 ncclDataType_t dataType, ncclRedOp_t redOp,
                                 void* recvMhandle, void** request) {
  mockRequest* r = (mockRequest*) malloc(sizeof(mockRequest));
  r->ntags = 1;
  r->sizes[0] = bytesPerRank;
  *request = r;
  return ncclSuccess;
}


__hidden ncclResult_t pluginCollNetIflush(void* collComm, void* data, int size, void* mhandle, void** request) {
  mockRequest* r = (mockRequest*) malloc(sizeof(mockRequest));
  r->ntags = 1;
  r->sizes[0] = size;
  *request = r;
  return ncclSuccess;
}

__hidden ncclResult_t pluginCloseColl(void* collComm) {
  if (collComm) free(collComm);
  collComm = NULL;
  return ncclSuccess;
}

#define NET_PLUGIN_NAME "MockPlugin"

ncclNet_v9_t ncclNetPlugin_v9 = {
  .name = NET_PLUGIN_NAME,
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
  .getDeviceMr = NULL,
  .irecvConsumed = pluginIrecvConsumed,
  .makeVDevice   = pluginMakeVDevice,
};

#define COLLNET_PLUGIN_NAME "CollNetMockPlugin"

ncclCollNet_v9_t ncclCollNetPlugin_v9 = {
  .name = COLLNET_PLUGIN_NAME,
  .init = pluginInit,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties,
  .listen = pluginListen,
  .connect = pluginCollConnect,
  .reduceSupport = pluginReduceSupport,
  .regMr = pluginRegMr,
  .regMrDmaBuf = pluginRegMrDmaBuf,
  .deregMr = pluginDeregMr,
  .iallreduce = pluginIAllReduce,
  .iallgather = pluginIAllGather,
  .ireducescatter = pluginIReduceScatter,
  .iflush = pluginCollNetIflush,
  .test = pluginTest,
  .closeColl = pluginCloseColl,
  .closeListen = pluginCloseListen,
  .makeVDevice   = pluginMakeVDevice
};
