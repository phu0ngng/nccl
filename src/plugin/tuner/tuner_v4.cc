/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <dlfcn.h>
#include "debug.h"
#include "nccl_tuner.h"
#include "checks.h"

static ncclTuner_v4_t* ncclTuner_v4;
static ncclTuner_t ncclTuner;

static ncclResult_t ncclTuner_finalize(void* ctx) {
  return ncclTuner_v4->destroy(ctx);
}

static ncclResult_t ncclTuner_init(void** context, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclTuner_v4->init(nRanks, nNodes, logfn, context));
  ncclTuner.getCollInfo = ncclTuner_v4->getCollInfo;
  ncclTuner.finalize = ncclTuner_finalize;
  return ncclSuccess;
}

ncclTuner_t* getNcclTuner_v4(void* lib) {
  ncclTuner_v4 = (ncclTuner_v4_t*)dlsym(lib, "ncclTunerPlugin_v4");
  if (ncclTuner_v4) {
    ncclTuner.name = ncclTuner_v4->name;
    ncclTuner.init = ncclTuner_init;

    INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Using tuner plugin %s", ncclTuner_v4->name);
    return &ncclTuner;
  }
  INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.");
  return NULL;
}
