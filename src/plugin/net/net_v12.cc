/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_net.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclNet_v12_t* ncclNet_v12;
static ncclCollNet_v12_t* ncclCollNet_v12;
static ncclGin_v12_t* ncclGin_v12;

ncclNet_t* getNcclNet_v12(void* lib) {
  ncclNet_v12 = (ncclNet_v12_t*)dlsym(lib, "ncclNetPlugin_v12");
  if (ncclNet_v12) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v12)", ncclNet_v12->name);
    return ncclNet_v12;
  }
  return nullptr;
}

ncclCollNet_t* getNcclCollNet_v12(void* lib) {
  ncclCollNet_v12 = (ncclCollNet_v12_t*)dlsym(lib, "ncclCollNetPlugin_v12");
  if (ncclCollNet_v12) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v12)", ncclCollNet_v12->name);
    return ncclCollNet_v12;
  }
  return nullptr;
}

ncclGin_t* getNcclGin_v12(void* lib) {
  ncclGin_v12 = (ncclGin_v12_t*)dlsym(lib, "ncclGinPlugin_v12");
  if (ncclGin_v12) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded gin plugin %s (v12)", ncclGin_v12->name);
    return ncclGin_v12;
  }
  return nullptr;
}
