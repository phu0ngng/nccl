/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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

#define NET_SCORES_UNSET 0ULL

// We encode 3 bits of distance per interface into a 64-bit uint
#define NET_BITS_PER_IF 3
#define NET_BITS_PER_IF_MASK ((1<<NET_BITS_PER_IF)-1)
static_assert(sizeof(uint64_t)*8 >= NET_MAX_IFS*NET_BITS_PER_IF, "NET_MAX_IFS*NET_BITS_PER_IF must fit in 64 bits");

uint64_t getScores(short* distances, int ndev);

int getScore(uint64_t scores, int dev);

ncclResult_t netDistance(int cudaDev, int dev, short* distance, netInfoFuncs* netInfo);

ncclResult_t netDevices(int* ndev, short** distances, netInfoFuncs* netInfo);

int getDev(int cudaDev, int ringId, uint64_t* netScores, int* netNDev, netInfoFuncs* netInfo);

ncclResult_t netGetGdrSupport(int dev, int read, int* useGdr, netInfoFuncs* netInfo);

#endif
