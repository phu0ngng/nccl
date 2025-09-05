/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "os/os.h"

#include "checks.h"
#include "utils.h"

#include <cstdint>
#include <unistd.h>
#include <pthread.h>
#include <sys/resource.h>

// Process Management
uint64_t ncclOsGetpid() {
  return (uint64_t)getpid();
}

// The default Linux stack size (8MB) is safe.
#define SAFE_STACK_SIZE (8192*1024)

ncclResult_t ncclOsSetCpuStackSize() {
  // Query the stack size used for newly launched threads.
  pthread_attr_t attr;
  size_t stackSize;
  PTHREADCHECK(pthread_attr_init(&attr), "pthread_attr_init");
  PTHREADCHECK(pthread_attr_getstacksize(&attr, &stackSize), "pthread_attr_getstacksize");

  if (stackSize < SAFE_STACK_SIZE) {
    // GNU libc normally uses RLIMIT_STACK as the default pthread stack size, unless it's set to "unlimited" --
    // in that case a fallback value of 2MB (!) is used.

    // Query the actual resource limit so that we can distinguish between the settings of 2MB and unlimited.
    struct rlimit stackLimit;
    char buf[30];
    SYSCHECK(getrlimit(RLIMIT_STACK, &stackLimit), "getrlimit");
    if (stackLimit.rlim_cur == RLIM_INFINITY)
      strcpy(buf, "unlimited");
    else
      snprintf(buf, sizeof(buf), "%ldKB", stackLimit.rlim_cur / 1024);
    INFO(NCCL_INIT | NCCL_ENV, "Stack size limit (%s) is unsafe; will use %dKB for newly launched threads",
         buf, SAFE_STACK_SIZE / 1024);

    // Change the default pthread stack size (via a nonportable API as the feature is not available in std::thread)
    PTHREADCHECK(pthread_attr_setstacksize(&attr, SAFE_STACK_SIZE), "pthread_attr_setstacksize");
    PTHREADCHECK(pthread_setattr_default_np(&attr), "pthread_setattr_default_np");
  }

  PTHREADCHECK(pthread_attr_destroy(&attr), "pthread_attr_destroy");
  return ncclSuccess;
}
