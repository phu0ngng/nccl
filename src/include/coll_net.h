/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef COLL_NET_H_
#define COLL_NET_H_

#include "nccl.h"
#include "nccl_net.h"

extern ncclCollNet_t* collNet;
typedef char collNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

// Translation to external API
static const char* collNetName() { return collNet->name; }
static ncclResult_t collNetDevices(int* ndev) { NCCLCHECK(collNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t collNetPciPath(int dev, char** path) { NCCLCHECK(collNet->pciPath(dev, path)); return ncclSuccess; }
static ncclResult_t collNetPtrSupport(int dev, int* supportedTypes) { NCCLCHECK(collNet->ptrSupport(dev, supportedTypes)); return ncclSuccess; }
static ncclResult_t collNetListen(int dev, void* handle, void** listenComm) { NCCLCHECK(collNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t collNetConnect(void* handles[], int nranks, void* listenComm, void** collComm) { NCCLCHECK(collNet->connect(handles, nranks, listenComm, collComm)); return ncclSuccess; }
static ncclResult_t collNetReduceSupport(ncclDataType_t dtype, ncclRedOp_t redOp, int* supported) { NCCLCHECK(collNet->reduceSupport(dtype, redOp, supported)); return ncclSuccess; }
static ncclResult_t collNetIallreduce(void* collComm, void* sendData, void* recvData, int size, ncclDataType_t dtype, ncclRedOp_t redOp, int type, void** request) {
  NCCLCHECK(collNet->iallreduce(collComm, sendData, recvData, size, dtype, redOp, type, request)); return ncclSuccess; }
static ncclResult_t collNetFlush(void* collComm, void* data, int size) { NCCLCHECK(collNet->flush(collComm, data, size)); return ncclSuccess; }
static ncclResult_t collNetTest(void* request, int* done, int* size) { NCCLCHECK(collNet->test(request, done, size)); return ncclSuccess; }
static ncclResult_t collNetCloseColl(void* collComm) { NCCLCHECK(collNet->closeColl(collComm)); return ncclSuccess; }
static ncclResult_t collNetCloseListen(void* listenComm) { NCCLCHECK(collNet->closeListen(listenComm)); return ncclSuccess; }

#endif
