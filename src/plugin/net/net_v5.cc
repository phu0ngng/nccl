/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_net.h"
#include "net_device.h"
#include "proxy.h"

#define MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.
#define MAX_COLLNET_SIZE (512*1024*1024L) //Set for initial collent plugins when size was not dynamically queried

static ncclNet_t ncclNet;
static ncclCollNet_t ncclCollNet;
static ncclNet_v5_t* ncclNet_v5;
static ncclCollNet_v5_t* ncclCollNet_v5;

static ncclResult_t ncclNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v5_t p5;
  ncclResult_t ans = ncclNet_v5->getProperties(dev, &p5);
  if (ans != ncclSuccess) return ans;
  props->name = p5.name;
  props->pciPath = p5.pciPath;
  props->guid = p5.guid;
  props->ptrSupport = p5.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p5.speed;
  props->port = p5.port;
  props->maxComms = p5.maxComms;
  props->maxRecvs = p5.maxRecvs;
  props->latency = p5.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclNet_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclNet_v5->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclNet_connect(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  return ncclNet_v5->connect(dev, handle, sendComm);
}

static ncclResult_t ncclNet_accept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  return ncclNet_v5->accept(listenComm, recvComm);
}

static ncclResult_t ncclNet_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
  int sizeInt;
  if (size > MAX_NET_SIZE) return ncclInternalError;
  sizeInt = (int)size;
  ncclResult_t ans = ncclNet_v5->isend(sendComm, data, sizeInt, tag, mhandle, request);
  return ans;
}

static ncclResult_t ncclNet_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
  int sizesInt[NCCL_PROXY_MAX_SUBS];
  //reset to NULL if optional receive completion is set
  if (*request == (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION) *request = NULL;
  for (int i=0; i<n; i++) {
    if (sizes[i] > MAX_NET_SIZE) return ncclInternalError;
    sizesInt[i] = (int) sizes[i];
  }
  ncclResult_t ans = ncclNet_v5->irecv(recvComm, n, data, sizesInt, tags, mhandles, request);
  return ans;
}

static ncclResult_t ncclCollNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v5_t p5;
  ncclResult_t ans = ncclCollNet_v5->getProperties(dev, &p5);
  if (ans != ncclSuccess) return ans;
  props->name = p5.name;
  props->pciPath = p5.pciPath;
  props->guid = p5.guid;
  props->ptrSupport = p5.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p5.speed;
  props->port = p5.port;
  props->maxComms = p5.maxComms;
  props->maxRecvs = p5.maxRecvs;
  props->latency = p5.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclCollNet_v5->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclCollNet_iallreduce(void* collComm, void* sendData, void* recvData, size_t count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
  int countInt;
  if (count > MAX_NET_SIZE) return ncclInternalError;
  countInt = (int)count;
  ncclResult_t ans = ncclCollNet_v5->iallreduce(collComm, sendData, recvData, countInt, dataType, redOp,
                 sendMhandle, recvMhandle, request);
  return ans;
}

ncclNet_t* getNcclNet_v5(void* lib) {
  ncclNet_v5 = (ncclNet_v5_t*)dlsym(lib, "ncclNetPlugin_v5");
  if (ncclNet_v5) {
    ncclNet.name = ncclNet_v5->name;
    ncclNet.init = ncclNet_v5->init;
    ncclNet.devices = ncclNet_v5->devices;
    ncclNet.getProperties = ncclNet_getProperties;
    ncclNet.listen = ncclNet_v5->listen;
    ncclNet.connect = ncclNet_connect;
    ncclNet.accept =  ncclNet_accept;
    ncclNet.regMr = ncclNet_regMr;
    ncclNet.regMrDmaBuf = NULL;
    ncclNet.deregMr = ncclNet_v5->deregMr;
    ncclNet.isend = ncclNet_isend;
    ncclNet.irecv = ncclNet_irecv;
    ncclNet.iflush = ncclNet_v5->iflush;
    ncclNet.test = ncclNet_v5->test;
    ncclNet.closeSend = ncclNet_v5->closeSend;
    ncclNet.closeRecv = ncclNet_v5->closeRecv;
    ncclNet.closeListen = ncclNet_v5->closeListen;
    ncclNet.getDeviceMr = NULL;
    ncclNet.irecvConsumed = NULL;
    ncclNet.makeVDevice = NULL;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v5)", ncclNet_v5->name);
    return &ncclNet;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin symbol (>= v5). ncclNetPlugin symbols v4 and lower are not supported.");
  return NULL;
}

ncclCollNet_t* getNcclCollNet_v5(void* lib) {
  ncclCollNet_v5 = (ncclCollNet_v5_t*)dlsym(lib, "ncclCollNetPlugin_v5");
  if (ncclCollNet_v5) {
    ncclCollNet.name = ncclCollNet_v5->name;
    ncclCollNet.init = ncclCollNet_v5->init;
    ncclCollNet.devices = ncclCollNet_v5->devices;
    ncclCollNet.getProperties = ncclCollNet_getProperties;
    ncclCollNet.listen = ncclCollNet_v5->listen;
    ncclCollNet.connect = ncclCollNet_v5->connect;
    ncclCollNet.reduceSupport = ncclCollNet_v5->reduceSupport;
    ncclCollNet.regMr = ncclCollNet_regMr;
    ncclCollNet.regMrDmaBuf = NULL;
    ncclCollNet.deregMr = ncclCollNet_v5->deregMr;
    ncclCollNet.iallreduce = ncclCollNet_iallreduce;
    ncclCollNet.iallgather = nullptr;
    ncclCollNet.ireducescatter = nullptr;
    ncclCollNet.iflush = ncclCollNet_v5->iflush;
    ncclCollNet.test = ncclCollNet_v5->test;
    ncclCollNet.closeColl = ncclCollNet_v5->closeColl;
    ncclCollNet.closeListen = ncclCollNet_v5->closeListen;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v5)", ncclCollNet_v5->name);
    return &ncclCollNet;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin symbol (>= v5). ncclCollNetPlugin symbols v4 and lower are not supported.");
  return NULL;
}
