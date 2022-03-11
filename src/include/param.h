/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <atomic>
#include <stdint.h>

const char* userHomeDir();
void setEnvFile(const char* fileName);
void initEnv();

void ncclLoadParam(char const* env, int64_t deftVal, int64_t bogus, std::atomic<int64_t>* cache);

#define NCCL_PARAM(name, env, deftVal) \
  int64_t ncclParam##name() { \
    constexpr int64_t bogus = ~int64_t(~uint64_t(0)>>1); /* most negative int64_t */ \
    static_assert(deftVal != bogus, "default value cannot be the bogus value."); \
    static std::atomic<int64_t> cache{bogus}; \
    if (__builtin_expect(cache.load(std::memory_order_relaxed) == bogus, false)) { \
      ncclLoadParam("NCCL_" env, deftVal, bogus, &cache); \
    } \
    return cache.load(std::memory_order_relaxed); \
  }

#endif
