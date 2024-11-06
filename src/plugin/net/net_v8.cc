/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_net.h"
#include "net_v8.h"
#include "net_device.h"
#include "proxy.h"

#define MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.
#define MAX_COLLNET_SIZE (512*1024*1024L) //Set for initial collent plugins when size was not dynamically queried

static ncclNet_v9_t ncclNet_v8_as_v9;
static ncclCollNet_v9_t ncclCollNet_v8_as_v9;
static ncclNet_v8_t* ncclNet_v8;
static ncclCollNet_v8_t* ncclCollNet_v8;

static ncclResult_t ncclNet_v8_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v8_t p8;
  ncclResult_t ans = ncclNet_v8->getProperties(dev, &p8);
  if (ans != ncclSuccess) return ans;
  props->name = p8.name;
  props->pciPath = p8.pciPath;
  props->guid = p8.guid;
  props->ptrSupport = p8.ptrSupport;
  props->regIsGlobal = p8.regIsGlobal;
  props->forceFlush = 0;
  props->speed = p8.speed;
  props->port = p8.port;
  props->maxComms = p8.maxComms;
  props->maxRecvs = p8.maxRecvs;
  props->latency = p8.latency;
  props->netDeviceType = p8.netDeviceType;
  props->netDeviceVersion = p8.netDeviceVersion;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v8_as_v9_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
  int sizeInt;
  if (size > MAX_NET_SIZE) return ncclInternalError;
  sizeInt = (int)size;
  ncclResult_t ans = ncclNet_v8->isend(sendComm, data, sizeInt, tag, mhandle, request);
  return ans;
}

static ncclResult_t ncclNet_v8_as_v9_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
  int sizesInt[NCCL_PROXY_MAX_SUBS];
  //reset to NULL if optional receive completion is set
  if (*request == (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION) *request = NULL;
  for (int i=0; i<n; i++) {
    if (sizes[i] > MAX_NET_SIZE) return ncclInternalError;
    sizesInt[i] = (int) sizes[i];
  }
  ncclResult_t ans = ncclNet_v8->irecv(recvComm, n, data, sizesInt, tags, mhandles, request);
  return ans;
}

static ncclResult_t ncclCollNet_v8_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v8_t p8;
  ncclResult_t ans = ncclCollNet_v8->getProperties(dev, &p8);
  if (ans != ncclSuccess) return ans;
  props->name = p8.name;
  props->pciPath = p8.pciPath;
  props->guid = p8.guid;
  props->ptrSupport = p8.ptrSupport;
  props->regIsGlobal = p8.regIsGlobal;
  props->forceFlush = 0;
  props->speed = p8.speed;
  props->port = p8.port;
  props->maxComms = p8.maxComms;
  props->maxRecvs = p8.maxRecvs;
  props->latency = p8.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v8_as_v9_iallreduce(void* collComm, void* sendData, void* recvData, size_t count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
  int countInt;
  if (count > MAX_NET_SIZE) return ncclInternalError;
  countInt = (int)count;
  ncclResult_t ans = ncclCollNet_v8->iallreduce(collComm, sendData, recvData, countInt, dataType, redOp,
                 sendMhandle, recvMhandle, request);
  return ans;
}

static ncclResult_t ncclCollNet_v8_as_v9_iallgather (void* collComm, void* sendData, int nRecvParts, ncclNetSGE_v9_t* recvParts,
                           size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                           void* sendMhandle, void** request) {
  ncclNetSGE_v8_t recvPartsInt;
  if (nRecvParts > 1) return ncclInternalError;
  if (recvParts->size > MAX_COLLNET_SIZE) return ncclInternalError;
  recvPartsInt.mhandle = recvParts->mhandle;
  recvPartsInt.address = recvParts->address;
  recvPartsInt.size = (int)recvParts->size;
  ncclResult_t ans = ncclCollNet_v8->iallgather(collComm, sendData, nRecvParts, &recvPartsInt,
                  bytesPerRank, windowOffset, windowBytes,
                  sendMhandle, request);
  return ans;
}

static ncclResult_t ncclCollNet_v8_as_v9_ireducescatter(void* collComm, int nSendParts, ncclNetSGE_v9_t* sendParts, void* recvData,
                               size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                               ncclDataType_t dataType, ncclRedOp_t redOp,
                               void* recvMhandle, void** request) {
  ncclNetSGE_v8_t sendPartsInt;
  if (nSendParts > 1) return ncclInternalError;
  if (sendParts->size > MAX_COLLNET_SIZE) return ncclInternalError;
  sendPartsInt.mhandle = sendParts->mhandle;
  sendPartsInt.address = sendParts->address;
  sendPartsInt.size = (int)sendParts->size;
  ncclResult_t ans = ncclCollNet_v8->ireducescatter(collComm, nSendParts, &sendPartsInt,
                  recvData, bytesPerRank, windowOffset, windowBytes,
                  dataType, redOp,
                  recvMhandle, request);
  return ans;
}

ncclNet_v9_t* getNcclNet_v8_as_v9(void* lib) {
  ncclNet_v8 = (ncclNet_v8_t*)dlsym(lib, "ncclNetPlugin_v8");
  if (ncclNet_v8) {
    ncclNet_v8_as_v9.name = ncclNet_v8->name;
    ncclNet_v8_as_v9.init = ncclNet_v8->init;
    ncclNet_v8_as_v9.devices = ncclNet_v8->devices;
    ncclNet_v8_as_v9.getProperties = ncclNet_v8_as_v9_getProperties;
    ncclNet_v8_as_v9.listen = ncclNet_v8->listen;
    ncclNet_v8_as_v9.connect = ncclNet_v8->connect;
    ncclNet_v8_as_v9.accept =  ncclNet_v8->accept;
    ncclNet_v8_as_v9.regMr = ncclNet_v8->regMr;
    ncclNet_v8_as_v9.regMrDmaBuf = ncclNet_v8->regMrDmaBuf;
    ncclNet_v8_as_v9.deregMr = ncclNet_v8->deregMr;
    ncclNet_v8_as_v9.isend = ncclNet_v8_as_v9_isend;
    ncclNet_v8_as_v9.irecv = ncclNet_v8_as_v9_irecv;
    ncclNet_v8_as_v9.iflush = ncclNet_v8->iflush;
    ncclNet_v8_as_v9.test = ncclNet_v8->test;
    ncclNet_v8_as_v9.closeSend = ncclNet_v8->closeSend;
    ncclNet_v8_as_v9.closeRecv = ncclNet_v8->closeRecv;
    ncclNet_v8_as_v9.closeListen = ncclNet_v8->closeListen;
    ncclNet_v8_as_v9.getDeviceMr = ncclNet_v8->getDeviceMr;
    ncclNet_v8_as_v9.irecvConsumed = ncclNet_v8->irecvConsumed;
    ncclNet_v8_as_v9.makeVDevice   = NULL;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v8)", ncclNet_v8->name);
    return &ncclNet_v8_as_v9;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v8 symbol.");
  return NULL;
}

ncclCollNet_v9_t* getNcclCollNet_v8_as_v9(void* lib) {
  ncclCollNet_v8 = (ncclCollNet_v8_t*)dlsym(lib, "ncclCollNetPlugin_v8");
  if (ncclCollNet_v8) {
    ncclCollNet_v8_as_v9.name = ncclCollNet_v8->name;
    ncclCollNet_v8_as_v9.init = ncclCollNet_v8->init;
    ncclCollNet_v8_as_v9.devices = ncclCollNet_v8->devices;
    ncclCollNet_v8_as_v9.getProperties = ncclCollNet_v8_as_v9_getProperties;
    ncclCollNet_v8_as_v9.listen = ncclCollNet_v8->listen;
    ncclCollNet_v8_as_v9.connect = ncclCollNet_v8->connect;
    ncclCollNet_v8_as_v9.reduceSupport = ncclCollNet_v8->reduceSupport;
    ncclCollNet_v8_as_v9.regMr = ncclCollNet_v8->regMr;
    ncclCollNet_v8_as_v9.regMrDmaBuf = ncclCollNet_v8->regMrDmaBuf;
    ncclCollNet_v8_as_v9.deregMr = ncclCollNet_v8->deregMr;
    ncclCollNet_v8_as_v9.iallreduce = ncclCollNet_v8_as_v9_iallreduce;
    ncclCollNet_v8_as_v9.iallgather = ncclCollNet_v8_as_v9_iallgather;
    ncclCollNet_v8_as_v9.ireducescatter = ncclCollNet_v8_as_v9_ireducescatter;
    ncclCollNet_v8_as_v9.iflush = ncclCollNet_v8->iflush;
    ncclCollNet_v8_as_v9.test = ncclCollNet_v8->test;
    ncclCollNet_v8_as_v9.closeColl = ncclCollNet_v8->closeColl;
    ncclCollNet_v8_as_v9.closeListen = ncclCollNet_v8->closeListen;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v8)", ncclCollNet_v8->name);
    return &ncclCollNet_v8_as_v9;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.");
  return NULL;
}
