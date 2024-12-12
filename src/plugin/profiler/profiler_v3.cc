/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"

static ncclProfiler_t ncclProfiler;
static ncclProfiler_v3_t* ncclProfiler_v3;

static ncclResult_t ncclProfiler_startEvent(void* context, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
  return ncclProfiler_v3->startEvent(context, eHandle, (ncclProfilerEventDescr_v3_t*)eDescr);
}

static ncclResult_t ncclProfiler_recordEventState(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs) {
  return ncclProfiler_v3->recordEventState(eHandle, (ncclProfilerEventState_v3_t)eState, (ncclProfilerEventStateArgs_v3_t*)eStateArgs);
}

static ncclResult_t ncclProfiler_init(void** context, int* eActivationMask) {
  NCCLCHECK(ncclProfiler_v3->init(context, eActivationMask));
  ncclProfiler.startEvent = ncclProfiler_startEvent;
  ncclProfiler.stopEvent = ncclProfiler_v3->stopEvent;
  ncclProfiler.recordEventState = ncclProfiler_recordEventState;
  ncclProfiler.finalize = ncclProfiler_v3->finalize;
  return ncclSuccess;
}

ncclProfiler_t* getNcclProfiler_v3(void* lib) {
  ncclProfiler_v3 = (ncclProfiler_v3_t*)dlsym(lib, "ncclProfiler_v3");
  if (ncclProfiler_v3) {
    ncclProfiler.name = ncclProfiler_v3->name;
    ncclProfiler.init = ncclProfiler_init;
    INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: loaded %s", ncclProfiler_v3->name);
    return &ncclProfiler;
  }
  INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: failed to find ncclProfiler_v3");
  return NULL;
}
