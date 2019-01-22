/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "nccl.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>

ncclResult_t getCudaPath(int cudaDev, char** path);

enum ncclPathDist {
  PATH_PIX = 0,
  PATH_PXB = 1,
  PATH_PHB = 2,
  PATH_SOC = 3,
  PATH_ARRAY_SIZE = 4
};

extern const char* pathDists[PATH_ARRAY_SIZE];

int pciDistance(char* path1, char* path2);

#endif
