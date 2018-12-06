/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef COLL_NET_H_
#define COLL_NET_H_

#include "nccl.h"
#include "nccl_coll_net.h"

typedef char collNetHandle_t[NCCL_COLL_NET_HANDLE_MAXSIZE];

// Translation to external API
static const char* collNetName() { return collNet->name; }
static ncclResult_t collNetDevices(int* ndev) { NCCLCHECK(collNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t collNetPciPath(int dev, char** path) { NCCLCHECK(collNet->pciPath(dev, path)); return ncclSuccess; }
static ncclResult_t collNetPtrSupport(int dev, int* supportedTypes) { NCCLCHECK(collNet->ptrSupport(dev, supportedTypes)); return ncclSuccess; }
static ncclResult_t collNetListen(int dev, void* handle, void** listenComm) { NCCLCHECK(collNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t collNetConnect(int dev, void* handles[], int nranks, void* listenComm, void** collComm) { NCCLCHECK(collNet->connect(dev, handles, nranks, listenComm, collComm)); return ncclSuccess; }
static ncclResult_t collNetReduceSupport(int* support) { NCCLCHECK(collNet->reduceSupport(support)); return ncclSuccess; }
static ncclResult_t collNetIsend(void* sendComm, void* src, void* dst, int size, int type, void** request) { NCCLCHECK(collNet->isend(sendComm, src, dst, size, type, request)); return ncclSuccess; }
static ncclResult_t collNetIrecv(void* recvComm, void* data, int size, int type, void** request) { NCCLCHECK(collNet->irecv(recvComm, data, size, type, request)); return ncclSuccess; }
static ncclResult_t collNetFlush(void* recvComm, void* data, int size) { NCCLCHECK(collNet->flush(recvComm, data, size)); return ncclSuccess; }
static ncclResult_t collNetTest(void* request, int* done, int* size) { NCCLCHECK(collNet->test(request, done, size)); return ncclSuccess; }
static ncclResult_t collNetCloseColl(void* collComm) { NCCLCHECK(collNet->closeColl(collComm)); return ncclSuccess; }
static ncclResult_t collNetCloseListen(void* listenComm) { NCCLCHECK(collNet->closeListen(listenComm)); return ncclSuccess; }

#endif
