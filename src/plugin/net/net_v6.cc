/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_net.h"
#include "net_v6.h"
#include "net_device.h"
#include "proxy.h"

#define MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.
#define MAX_COLLNET_SIZE (512*1024*1024L) //Set for initial collent plugins when size was not dynamically queried

static ncclNet_v9_t ncclNet_v6_as_v9;
static ncclCollNet_v9_t ncclCollNet_v6_as_v9;
static ncclNet_v6_t* ncclNet_v6;
static ncclCollNet_v6_t* ncclCollNet_v6;

static ncclResult_t ncclNet_v6_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclNet_v6->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v6_as_v9_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclNet_v6->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclNet_v6_as_v9_connect(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  return ncclNet_v6->connect(dev, handle, sendComm);
}

static ncclResult_t ncclNet_v6_as_v9_accept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  return ncclNet_v6->accept(listenComm, recvComm);
}

static ncclResult_t ncclNet_v6_as_v9_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
  int sizeInt;
  if (size > MAX_NET_SIZE) return ncclInternalError;
  sizeInt = (int)size;
  ncclResult_t ans = ncclNet_v6->isend(sendComm, data, sizeInt, tag, mhandle, request);
  return ans;
}

static ncclResult_t ncclNet_v6_as_v9_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
  int sizesInt[NCCL_PROXY_MAX_SUBS];
  //reset to NULL if optional receive completion is set
  if (*request == (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION) *request = NULL;
  for (int i=0; i<n; i++) {
    if (sizes[i] > MAX_NET_SIZE) return ncclInternalError;
    sizesInt[i] = (int) sizes[i];
  }
  ncclResult_t ans = ncclNet_v6->irecv(recvComm, n, data, sizesInt, tags, mhandles, request);
  return ans;
}

static ncclResult_t ncclCollNet_v6_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclCollNet_v6->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v6_as_v9_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclCollNet_v6->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclCollNet_v6_as_v9_iallreduce(void* collComm, void* sendData, void* recvData, size_t count,
     ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
  int countInt;
  if (count > MAX_NET_SIZE) return ncclInternalError;
  countInt = (int)count;
  ncclResult_t ans = ncclCollNet_v6->iallreduce(collComm, sendData, recvData, countInt, dataType, redOp,
                 sendMhandle, recvMhandle, request);
  return ans;
}

ncclNet_v9_t* getNcclNet_v6_as_v9(void* lib) {
  ncclNet_v6 = (ncclNet_v6_t*)dlsym(lib, "ncclNetPlugin_v6");
  if (ncclNet_v6) {
    ncclNet_v6_as_v9.name = ncclNet_v6->name;
    ncclNet_v6_as_v9.init = ncclNet_v6->init;
    ncclNet_v6_as_v9.devices = ncclNet_v6->devices;
    ncclNet_v6_as_v9.getProperties = ncclNet_v6_as_v9_getProperties;
    ncclNet_v6_as_v9.listen = ncclNet_v6->listen;
    ncclNet_v6_as_v9.connect = ncclNet_v6_as_v9_connect;
    ncclNet_v6_as_v9.accept =  ncclNet_v6_as_v9_accept;
    ncclNet_v6_as_v9.regMr = ncclNet_v6_as_v9_regMr;
    ncclNet_v6_as_v9.regMrDmaBuf = ncclNet_v6->regMrDmaBuf;
    ncclNet_v6_as_v9.deregMr = ncclNet_v6->deregMr;
    ncclNet_v6_as_v9.isend = ncclNet_v6_as_v9_isend;
    ncclNet_v6_as_v9.irecv = ncclNet_v6_as_v9_irecv;
    ncclNet_v6_as_v9.iflush = ncclNet_v6->iflush;
    ncclNet_v6_as_v9.test = ncclNet_v6->test;
    ncclNet_v6_as_v9.closeSend = ncclNet_v6->closeSend;
    ncclNet_v6_as_v9.closeRecv = ncclNet_v6->closeRecv;
    ncclNet_v6_as_v9.closeListen = ncclNet_v6->closeListen;
    ncclNet_v6_as_v9.getDeviceMr = NULL;
    ncclNet_v6_as_v9.irecvConsumed = NULL;
    ncclNet_v6_as_v9.makeVDevice  = NULL;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v6)", ncclNet_v6->name);
    return &ncclNet_v6_as_v9;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v6 symbol.");
  return NULL;
}

ncclCollNet_v9_t* getNcclCollNet_v6_as_v9(void* lib) {
  ncclCollNet_v6 = (ncclCollNet_v6_t*)dlsym(lib, "ncclCollNetPlugin_v6");
  if (ncclCollNet_v6) {
    ncclCollNet_v6_as_v9.name = ncclCollNet_v6->name;
    ncclCollNet_v6_as_v9.init = ncclCollNet_v6->init;
    ncclCollNet_v6_as_v9.devices = ncclCollNet_v6->devices;
    ncclCollNet_v6_as_v9.getProperties = ncclCollNet_v6_as_v9_getProperties;
    ncclCollNet_v6_as_v9.listen = ncclCollNet_v6->listen;
    ncclCollNet_v6_as_v9.connect = ncclCollNet_v6->connect;
    ncclCollNet_v6_as_v9.reduceSupport = ncclCollNet_v6->reduceSupport;
    ncclCollNet_v6_as_v9.regMr = ncclCollNet_v6_as_v9_regMr;
    ncclCollNet_v6_as_v9.regMrDmaBuf = ncclCollNet_v6->regMrDmaBuf;
    ncclCollNet_v6_as_v9.deregMr = ncclCollNet_v6->deregMr;
    ncclCollNet_v6_as_v9.iallreduce = ncclCollNet_v6_as_v9_iallreduce;
    ncclCollNet_v6_as_v9.iallgather = nullptr;
    ncclCollNet_v6_as_v9.ireducescatter = nullptr;
    ncclCollNet_v6_as_v9.iflush = ncclCollNet_v6->iflush;
    ncclCollNet_v6_as_v9.test = ncclCollNet_v6->test;
    ncclCollNet_v6_as_v9.closeColl = ncclCollNet_v6->closeColl;
    ncclCollNet_v6_as_v9.closeListen = ncclCollNet_v6->closeListen;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v6)", ncclCollNet_v6->name);
    return &ncclCollNet_v6_as_v9;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.");
  return NULL;
}
