/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_net.h"
#include "net_device.h"
#include "proxy.h"

static ncclNet_v9_t* ncclNet_v9;
static ncclCollNet_v9_t* ncclCollNet_v9;

ncclNet_t* getNcclNet_v9(void* lib) {
  ncclNet_v9 = (ncclNet_v9_t*)dlsym(lib, "ncclNetPlugin_v9");
  if (ncclNet_v9) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v9)", ncclNet_v9->name);
    return ncclNet_v9;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v9 symbol.");
  return NULL;
}

ncclCollNet_t* getNcclCollNet_v9(void* lib) {
  ncclCollNet_v9 = (ncclCollNet_v9_t*)dlsym(lib, "ncclCollNetPlugin_v9");
  if (ncclCollNet_v9) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v9)", ncclCollNet_v9->name);
    return ncclCollNet_v9;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.");
  return NULL;
}
