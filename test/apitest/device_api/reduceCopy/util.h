/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_UTIL_H_
#define _REDUCE_COPY_TEST_UTIL_H_

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>

// Helper function to check if verbose output is enabled (general test logging).
// Controlled by NCCL_TEST_VERBOSE=1 (or "true"). Use with VERBOSE_PRINTF().
inline bool isVerboseOutputEnabled() {
  const char* envVerbose = std::getenv("NCCL_TEST_VERBOSE");
  if (envVerbose && (strcmp(envVerbose, "1") == 0 || strcasecmp(envVerbose, "true") == 0)) {
    return true;
  }
  return false;
}

// Helper function to check if the matrix listener should print the detailed list of missing tests.
// Controlled by NCCL_TEST_REDUCE_COPY_LIST_MISSING=1 (or "true"). Works in combination with
// NCCL_TEST_REDUCE_COPY_TYPES (which types are tested); the missing list reflects the same type selection.
inline bool isMissingTestsListEnabled() {
  const char* env = std::getenv("NCCL_TEST_REDUCE_COPY_LIST_MISSING");
  if (env && (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0)) {
    return true;
  }
  return false;
}

// Test data RNG seed. NCCL_TEST_REDUCE_COPY_SEED overrides the default (hex or decimal).
inline uint64_t getReduceCopyTestSeed() {
  constexpr uint64_t kDefaultSeed = 0x1234567890ABCDEFULL;
  const char* env = std::getenv("NCCL_TEST_REDUCE_COPY_SEED");
  if (!env || !env[0]) return kDefaultSeed;
  char* end = nullptr;
  uint64_t v = static_cast<uint64_t>(strtoull(env, &end, 0));
  if (end && *end == '\0') return v;
  return kDefaultSeed;
}

// Macro for verbose printf (only prints if NCCL_TEST_VERBOSE=1)
#define VERBOSE_PRINTF(...) do { if (isVerboseOutputEnabled()) { printf(__VA_ARGS__); fflush(stdout); } } while(0)

#endif // _REDUCE_COPY_TEST_UTIL_H_

