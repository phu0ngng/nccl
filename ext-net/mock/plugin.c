/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"
#include <string.h>

#define __hidden __attribute__ ((visibility("hidden")))
#define NCCL_PLUGIN_MAX_RECVS 1
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
  int data;
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
    memcpy(mockProps + deviceIndex, props, sizeof(ncclNetProperties_t));
    nPhysDevs++;
    ncclNetVDeviceProps_t vProps = {};
    vProps.ndevs = 1;
    vProps.devs[0] = deviceIndex;
    return pluginMakeVDevice(&deviceIndex, &vProps);
  } else {
    return ncclInvalidUsage;
  }
}

__hidden ncclResult_t pluginInit(ncclDebugLogger_t logFunction) {
  memset(mockProps,     0, sizeof(mockProps));
  memset(mockVDevProps, 0, sizeof(mockVDevProps));
  nPhysDevs    = 0;
  nVirtualDevs = 0;
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
  *request = (mockRequest*) malloc(sizeof(mockRequest));
  return ncclSuccess;
}

__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
  *request = (mockRequest*) malloc(sizeof(mockRequest));
  return ncclSuccess;
}

__hidden ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  *request = (mockRequest*) malloc(sizeof(mockRequest));
  return ncclSuccess;
}

__hidden ncclResult_t pluginTest(void* request, int* done, int* size) {
  free(request);
  *done = 1;
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
  *request = (mockRequest*) malloc(sizeof(mockRequest));
  return ncclSuccess;
}

__hidden ncclResult_t pluginIAllGather(void* collComm, void* sendData, int nRecvParts, ncclNetSGE_v9_t* recvParts,
                             size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                             void* sendMhandle, void** request) {
  *request = (mockRequest*) malloc(sizeof(mockRequest));
  return ncclSuccess;
}

__hidden ncclResult_t pluginIReduceScatter(void* collComm, int nSendParts, ncclNetSGE_v9_t* sendParts, void* recvData,
                                 size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                                 ncclDataType_t dataType, ncclRedOp_t redOp,
                                 void* recvMhandle, void** request) {
  *request = (mockRequest*) malloc(sizeof(mockRequest));
  return ncclSuccess;
}


__hidden ncclResult_t pluginCollNetIflush(void* collComm, void* data, int size, void* mhandle, void** request) {
  *request = (mockRequest*) malloc(sizeof(mockRequest));
  return ncclSuccess;
}

__hidden ncclResult_t pluginCloseColl(void* collComm) {
  if (collComm) free(collComm);
  return ncclSuccess;
}

#define NET_PLUGIN_NAME "MockPlugin"

const ncclNet_v9_t ncclNetPlugin_v9 = {
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

const ncclCollNet_v9_t ncclCollNetPlugin_v9 = {
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
  .closeColl = NULL,
  .closeListen = pluginCloseListen,
  .makeVDevice   = pluginMakeVDevice
};
