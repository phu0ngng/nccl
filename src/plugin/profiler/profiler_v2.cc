/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"

static ncclProfiler_t ncclProfiler;
static ncclProfiler_v2_t* ncclProfiler_v2;

ncclProfiler_t* getNcclProfiler_v2(void* lib) {
  ncclProfiler_v2 = (ncclProfiler_v2_t*)dlsym(lib, "ncclProfiler_v2");
  if (ncclProfiler_v2) {
    INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: loaded %s", ncclProfiler_v2->name);
    return &ncclProfiler;
  }
  INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: failed to find ncclProfiler_v2");
  return NULL;
}
