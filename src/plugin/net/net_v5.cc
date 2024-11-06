/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_net.h"
#include "net_v5.h"
#include "net_device.h"
#include "proxy.h"

#define MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.
#define MAX_COLLNET_SIZE (512*1024*1024L) //Set for initial collent plugins when size was not dynamically queried

static ncclNet_v9_t ncclNet_v5_as_v9;
static ncclCollNet_v9_t ncclCollNet_v5_as_v9;
static ncclNet_v5_t* ncclNet_v5;
static ncclCollNet_v5_t* ncclCollNet_v5;

static ncclResult_t ncclNet_v5_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
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

static ncclResult_t ncclNet_v5_as_v9_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclNet_v5->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclNet_v5_as_v9_connect(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  return ncclNet_v5->connect(dev, handle, sendComm);
}

static ncclResult_t ncclNet_v5_as_v9_accept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  return ncclNet_v5->accept(listenComm, recvComm);
}

static ncclResult_t ncclNet_v5_as_v9_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
  int sizeInt;
  if (size > MAX_NET_SIZE) return ncclInternalError;
  sizeInt = (int)size;
  ncclResult_t ans = ncclNet_v5->isend(sendComm, data, sizeInt, tag, mhandle, request);
  return ans;
}

static ncclResult_t ncclNet_v5_as_v9_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
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

static ncclResult_t ncclCollNet_v5_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
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

static ncclResult_t ncclCollNet_v5_as_v9_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclCollNet_v5->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclCollNet_v5_as_v9_iallreduce(void* collComm, void* sendData, void* recvData, size_t count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
  int countInt;
  if (count > MAX_NET_SIZE) return ncclInternalError;
  countInt = (int)count;
  ncclResult_t ans = ncclCollNet_v5->iallreduce(collComm, sendData, recvData, countInt, dataType, redOp,
                 sendMhandle, recvMhandle, request);
  return ans;
}

ncclNet_v9_t* getNcclNet_v5_as_v9(void* lib) {
  ncclNet_v5 = (ncclNet_v5_t*)dlsym(lib, "ncclNetPlugin_v5");
  if (ncclNet_v5) {
    ncclNet_v5_as_v9.name = ncclNet_v5->name;
    ncclNet_v5_as_v9.init = ncclNet_v5->init;
    ncclNet_v5_as_v9.devices = ncclNet_v5->devices;
    ncclNet_v5_as_v9.getProperties = ncclNet_v5_as_v9_getProperties;
    ncclNet_v5_as_v9.listen = ncclNet_v5->listen;
    ncclNet_v5_as_v9.connect = ncclNet_v5_as_v9_connect;
    ncclNet_v5_as_v9.accept =  ncclNet_v5_as_v9_accept;
    ncclNet_v5_as_v9.regMr = ncclNet_v5_as_v9_regMr;
    ncclNet_v5_as_v9.regMrDmaBuf = NULL;
    ncclNet_v5_as_v9.deregMr = ncclNet_v5->deregMr;
    ncclNet_v5_as_v9.isend = ncclNet_v5_as_v9_isend;
    ncclNet_v5_as_v9.irecv = ncclNet_v5_as_v9_irecv;
    ncclNet_v5_as_v9.iflush = ncclNet_v5->iflush;
    ncclNet_v5_as_v9.test = ncclNet_v5->test;
    ncclNet_v5_as_v9.closeSend = ncclNet_v5->closeSend;
    ncclNet_v5_as_v9.closeRecv = ncclNet_v5->closeRecv;
    ncclNet_v5_as_v9.closeListen = ncclNet_v5->closeListen;
    ncclNet_v5_as_v9.getDeviceMr = NULL;
    ncclNet_v5_as_v9.irecvConsumed = NULL;
    ncclNet_v5_as_v9.makeVDevice = NULL;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v5)", ncclNet_v5->name);
    return &ncclNet_v5_as_v9;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin symbol (>= v5). ncclNetPlugin symbols v4 and lower are not supported.");
  return NULL;
}

ncclCollNet_v9_t* getNcclCollNet_v5_as_v9(void* lib) {
  ncclCollNet_v5 = (ncclCollNet_v5_t*)dlsym(lib, "ncclCollNetPlugin_v5");
  if (ncclCollNet_v5) {
    ncclCollNet_v5_as_v9.name = ncclCollNet_v5->name;
    ncclCollNet_v5_as_v9.init = ncclCollNet_v5->init;
    ncclCollNet_v5_as_v9.devices = ncclCollNet_v5->devices;
    ncclCollNet_v5_as_v9.getProperties = ncclCollNet_v5_as_v9_getProperties;
    ncclCollNet_v5_as_v9.listen = ncclCollNet_v5->listen;
    ncclCollNet_v5_as_v9.connect = ncclCollNet_v5->connect;
    ncclCollNet_v5_as_v9.reduceSupport = ncclCollNet_v5->reduceSupport;
    ncclCollNet_v5_as_v9.regMr = ncclCollNet_v5_as_v9_regMr;
    ncclCollNet_v5_as_v9.regMrDmaBuf = NULL;
    ncclCollNet_v5_as_v9.deregMr = ncclCollNet_v5->deregMr;
    ncclCollNet_v5_as_v9.iallreduce = ncclCollNet_v5_as_v9_iallreduce;
    ncclCollNet_v5_as_v9.iallgather = nullptr;
    ncclCollNet_v5_as_v9.ireducescatter = nullptr;
    ncclCollNet_v5_as_v9.iflush = ncclCollNet_v5->iflush;
    ncclCollNet_v5_as_v9.test = ncclCollNet_v5->test;
    ncclCollNet_v5_as_v9.closeColl = ncclCollNet_v5->closeColl;
    ncclCollNet_v5_as_v9.closeListen = ncclCollNet_v5->closeListen;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v5)", ncclCollNet_v5->name);
    return &ncclCollNet_v5_as_v9;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin symbol (>= v5). ncclCollNetPlugin symbols v4 and lower are not supported.");
  return NULL;
}
