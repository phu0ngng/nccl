/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_COMMON_H_
#define NET_COMMON_H_

#include "core.h"

#define NET_MAX_IFS 16
#define NET_MAX_GPUS 32

struct netInfoFuncs {
  const char*  (*name)();
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*pciPath)(int dev, char** path);
  ncclResult_t (*ptrSupport)(int dev, int* supportedTypes);
};

// Cache GPU-NIC distances to avoid re-computing them
#define NET_TVALUE_UNKNOWN 0ULL

// We encode 3 bits of distance per interface into a ncclTvalue_t (64-bit)
#define NET_BITS_PER_IF 3
#define NET_BITS_PER_IF_MASK ((1<<NET_BITS_PER_IF)-1)
static_assert(sizeof(ncclTvalue_t)*8 >= NET_MAX_IFS*NET_BITS_PER_IF, "NET_MAX_IFS*NET_BITS_PER_IF must fit in a ncclTvalue_t");

ncclTvalue_t getTvalue(short* distances, int ndev);

int getScore(ncclTvalue_t tvalue, int dev);

ncclResult_t netDistance(int cudaDev, int dev, short* distance, netInfoFuncs* netInfo);

ncclResult_t netDevices(int* ndev, short** distances, netInfoFuncs* netInfo);

int getDev(int ringId, ncclTvalue_t tvalue, int ndev);

ncclResult_t netGetGdrSupport(int dev, int read, int* useGdr, netInfoFuncs* netInfo);

#endif
