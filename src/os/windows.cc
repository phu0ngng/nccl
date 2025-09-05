/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "os/os.h"

uint64_t ncclOsGetpid() {
  return (uint64_t)GetCurrentProcessId();
}

ncclResult_t ncclOsSetCpuStackSize() {
  // Not implemented on Windows
  return ncclSuccess;
}
