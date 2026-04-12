/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_REDUCE_COPY_TEST_COMMS_H
#define NCCL_REDUCE_COPY_TEST_COMMS_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "checks.h"
#include "nccl.h"
#include "nccl_device.h"

struct ActiveComms {
  int n = 0;
  ncclComm_t* comms = nullptr;
  bool owns = false;
};

static ActiveComms createActiveComms(int maxGpus, ncclComm_t* baseComms, int baseNVis) {
  ActiveComms active;
  if (maxGpus <= 0 || maxGpus >= baseNVis) {
    active.n = baseNVis;
    active.comms = baseComms;
    active.owns = false;
    return active;
  }

  struct SplitCacheEntry {
    int size = 0;
    std::vector<ncclComm_t> comms;
  };
  static std::vector<SplitCacheEntry> splitCache;

  for (auto& entry : splitCache) {
    if (entry.size == maxGpus) {
      active.n = maxGpus;
      active.comms = entry.comms.data();
      active.owns = false;
      return active;
    }
  }

  SplitCacheEntry entry;
  entry.size = maxGpus;
  entry.comms.assign(maxGpus, nullptr);

  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  splitConfig.splitShare = 1;

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < baseNVis; ++i) {
    CUDACHECK(cudaSetDevice(i));
    const int color = (i < maxGpus) ? 0 : NCCL_SPLIT_NOCOLOR;
    if (i < maxGpus) {
      NCCLCHECK(ncclCommSplit(baseComms[i], color, i, &entry.comms[i], &splitConfig));
    } else {
      ncclComm_t splitComm = nullptr;
      NCCLCHECK(ncclCommSplit(baseComms[i], color, i, &splitComm, &splitConfig));
    }
  }
  NCCLCHECK_COMM_WAITBATCH(ncclGroupEnd(), baseComms, baseNVis);

  bool splitOk = true;
  for (int i = 0; i < maxGpus; ++i) {
    if (entry.comms[i] == nullptr) {
      splitOk = false;
      break;
    }
  }
  if (!splitOk) {
    for (int i = 0; i < maxGpus; ++i) {
      if (entry.comms[i]) {
        NCCLCHECK(ncclCommDestroy(entry.comms[i]));
        entry.comms[i] = nullptr;
      }
    }
    fprintf(stderr, "ncclCommSplit returned NULL comm (maxGpus=%d, baseNVis=%d)\n",
        maxGpus, baseNVis);
    exit(1);
  }

  splitCache.push_back(std::move(entry));
  active.n = maxGpus;
  active.comms = splitCache.back().comms.data();
  active.owns = false;
  return active;
}

static void destroyActiveComms(ActiveComms* active) {
  if (!active || !active->owns || !active->comms) return;
  for (int i = 0; i < active->n; ++i) {
    NCCLCHECK(ncclCommDestroy(active->comms[i]));
  }
  free(active->comms);
  active->comms = nullptr;
  active->n = 0;
  active->owns = false;
}

#endif // NCCL_REDUCE_COPY_TEST_COMMS_H
