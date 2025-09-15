/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_OS_H_
#define NCCL_OS_H_

#include "nccl.h"

#include <cstdint>

uint64_t ncclOsGetpid();

ncclResult_t ncclOsSetCpuStackSize();

#endif
