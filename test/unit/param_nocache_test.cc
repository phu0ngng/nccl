/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Integration tests for NCCL_NO_CACHE support in the NcclParam framework.
//
// NCCL_NO_CACHE is itself a Cached param — once loaded, it's immutable for
// the process lifetime. All tests in this binary share a single NCCL_NO_CACHE
// value set before any test runs via a Google Test Environment.

#include <gtest/gtest.h>
#include "param/param.h"

#include <atomic>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// ============================================================================
// Global Environment: set NCCL_NO_CACHE once before all tests
// ============================================================================

class NoCacheEnvSetup : public ::testing::Environment {
public:
  void SetUp() override {
    // Target specific params for no-cache. This is immutable for the
    // process lifetime because ncclParamNoCacheSet is Cached.
    setenv("NCCL_NO_CACHE", "TEST_NOCACHE_TARGET,TEST_NOCACHE_SPECIFIC,TEST_NOCACHE_THREAD", 1);
  }
  void TearDown() override {
    unsetenv("NCCL_NO_CACHE");
  }
};

// Register global environment before main()
static ::testing::Environment* const nocache_env =
::testing::AddGlobalTestEnvironment(new NoCacheEnvSetup);

// ============================================================================
// Test Parameters
// ============================================================================

// Cached param in the no-cache set — should reload on every access
DEFINE_NCCL_PARAM(testNoCacheTarget, int32_t, TEST_NOCACHE_TARGET, 1,
                  NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT, "");

// Another Cached param in the no-cache set
DEFINE_NCCL_PARAM(testNoCacheSpecific, int32_t, TEST_NOCACHE_SPECIFIC, 2,
                  NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT, "");

// Cached param NOT in the no-cache set — should cache normally
DEFINE_NCCL_PARAM(testNoCacheOther, int32_t, TEST_NOCACHE_OTHER, 2,
                  NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT, "");

// Non-cached (Default) param — always reloads, NCCL_NO_CACHE irrelevant
DEFINE_NCCL_PARAM(testNoCacheDefault, int32_t, TEST_NOCACHE_DEFAULT, 3,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");

// Cached param NOT in the no-cache set — for testing empty/absent key
DEFINE_NCCL_PARAM(testNoCacheAbsent, int32_t, TEST_NOCACHE_ABSENT, 5,
                  NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT, "");

// ============================================================================
// Test Fixture
// ============================================================================

class NcclParamNoCacheTest : public ::testing::Test {
protected:
  void SetEnv(const char* key, const char* value) {
    setenv(key, value, 1);
    trackedKeys.insert(key);
  }

  void TearDown() override {
    for (const auto& key : trackedKeys) {
      unsetenv(key.c_str());
    }
    trackedKeys.clear();
  }

private:
  std::set<std::string> trackedKeys;
};

// ============================================================================
// Test: Cached param in no-cache set reloads on every access
// ============================================================================

TEST_F(NcclParamNoCacheTest, CachedParamInNoCacheSetReloads) {
  SetEnv("TEST_NOCACHE_TARGET", "10");

  // First access
  EXPECT_EQ(testNoCacheTarget(), 10);

  // Change the env — a Cached param would normally return 10, but
  // since it's in the no-cache set, it should reload.
  SetEnv("TEST_NOCACHE_TARGET", "20");
  EXPECT_EQ(testNoCacheTarget(), 20);

  SetEnv("TEST_NOCACHE_TARGET", "30");
  EXPECT_EQ(testNoCacheTarget(), 30);
}

// ============================================================================
// Test: Multiple targeted params in no-cache set all reload
// ============================================================================

TEST_F(NcclParamNoCacheTest, MultipleTargetedParamsReload) {
  SetEnv("TEST_NOCACHE_SPECIFIC", "100");
  EXPECT_EQ(testNoCacheSpecific(), 100);

  SetEnv("TEST_NOCACHE_SPECIFIC", "200");
  EXPECT_EQ(testNoCacheSpecific(), 200);
}

// ============================================================================
// Test: Cached param NOT in no-cache set still caches normally
// ============================================================================

TEST_F(NcclParamNoCacheTest, CachedParamNotInSetCachesNormally) {
  SetEnv("TEST_NOCACHE_OTHER", "100");

  // First access caches the value
  EXPECT_EQ(testNoCacheOther(), 100);

  // Change env — should still return cached value
  SetEnv("TEST_NOCACHE_OTHER", "200");
  EXPECT_EQ(testNoCacheOther(), 100);
}

// ============================================================================
// Test: Non-cached (Default) params are unaffected by NCCL_NO_CACHE
// ============================================================================

TEST_F(NcclParamNoCacheTest, DefaultParamsUnaffected) {
  SetEnv("TEST_NOCACHE_DEFAULT", "77");

  // Default params always reload (they don't cache), so this should work
  EXPECT_EQ(testNoCacheDefault(), 77);

  SetEnv("TEST_NOCACHE_DEFAULT", "88");
  EXPECT_EQ(testNoCacheDefault(), 88);
}

// ============================================================================
// Test: Cached param not listed in NCCL_NO_CACHE caches normally
// ============================================================================

TEST_F(NcclParamNoCacheTest, AbsentKeyStillCaches) {
  SetEnv("TEST_NOCACHE_ABSENT", "42");

  // First access caches the value
  EXPECT_EQ(testNoCacheAbsent(), 42);

  // Change env — should still return cached value
  SetEnv("TEST_NOCACHE_ABSENT", "99");
  EXPECT_EQ(testNoCacheAbsent(), 42);
}
