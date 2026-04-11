/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Thread-safety unit tests for the NcclParam framework (param.h).
//
// Each test uses its own DEFINE_NCCL_PARAM to avoid cross-test interference
// (especially important for Cached params which are one-shot).
// All env var setup happens in the main thread BEFORE spawning threads
// because setenv() is not thread-safe on POSIX.

#include <gtest/gtest.h>
#include "param/param.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// SpinBarrier (C++14 compatible — no std::barrier)
// ============================================================================

class SpinBarrier {
public:
  explicit SpinBarrier(int count) : count(count), waiting(0), generation(0) {}

  void wait() {
    int gen = generation.load(std::memory_order_acquire);
    if (waiting.fetch_add(1, std::memory_order_acq_rel) + 1 == count) {
      // Last thread to arrive: reset counter and advance generation.
      waiting.store(0, std::memory_order_relaxed);
      generation.fetch_add(1, std::memory_order_release);
    } else {
      // Spin until the generation advances.
      while (generation.load(std::memory_order_acquire) == gen) {
        // spin
      }
    }
  }

private:
  const int count;
  std::atomic<int> waiting;
  std::atomic<int> generation;
};

// ============================================================================
// Constants
// ============================================================================

static constexpr int kThreadCount = 8;
static constexpr int kIterations  = 1000;

// ============================================================================
// Test Fixture
// ============================================================================

class NcclParamThreadTest : public ::testing::Test {
protected:
  void SetEnv(const char* key, const char* value) {
    setenv(key, value, 1);
    tracked_keys_.insert(key);
  }

  void TearDown() override {
    for (const auto& key : tracked_keys_) {
      unsetenv(key.c_str());
    }
    tracked_keys_.clear();
  }

private:
  std::set<std::string> tracked_keys_;
};

// ============================================================================
// DEFINE_NCCL_PARAM declarations (one per test to avoid interference)
// ============================================================================

DEFINE_NCCL_PARAM(testThreadCachedRead, int32_t, TEST_THREAD_CACHED_READ, 0,
                  NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testThreadNonCachedRead, int32_t, TEST_THREAD_NONCACHED_READ, 0,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testThreadCachedFastPath, int32_t, TEST_THREAD_CACHED_FAST, 0,
                  NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testThreadRegistryA, int32_t, TEST_THREAD_REG_A, 0,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testThreadRegistryB, int32_t, TEST_THREAD_REG_B, 0,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testThreadString, const char*, TEST_THREAD_STRING, nullptr,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testThreadToString, int32_t, TEST_THREAD_TOSTR, 0,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testThreadDump, int32_t, TEST_THREAD_DUMP, 0,
                  NCCL_PARAM_FLAG_PUBLISHED, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testThreadCachedRace, int32_t, TEST_THREAD_CACHED_RACE, 0,
                  NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT, "");

// ============================================================================
// Test 1: ConcurrentCachedReads
// ============================================================================

TEST_F(NcclParamThreadTest, ConcurrentCachedReads) {
  SetEnv("TEST_THREAD_CACHED_READ", "42");
  // Force first load in main thread.
  int32_t warmup = testThreadCachedRead();
  ASSERT_EQ(warmup, 42);

  SpinBarrier barrier(kThreadCount);
  std::vector<std::thread> threads;
  std::atomic<bool> ok{true};

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&]() {
                           barrier.wait();
                           for (int i = 0; i < kIterations; ++i) {
                             int32_t v = testThreadCachedRead();
                             if (v != 42) ok.store(false, std::memory_order_relaxed);
                           }
                         });
  }
  for (auto& th : threads) th.join();
  EXPECT_TRUE(ok.load());
}

// ============================================================================
// Test 2: ConcurrentNonCachedReads
// ============================================================================

TEST_F(NcclParamThreadTest, ConcurrentNonCachedReads) {
  SetEnv("TEST_THREAD_NONCACHED_READ", "10");

  SpinBarrier barrier(kThreadCount);
  std::vector<std::thread> threads;
  std::atomic<bool> ok{true};

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&]() {
                           barrier.wait();
                           for (int i = 0; i < kIterations; ++i) {
                             int32_t v = testThreadNonCachedRead();
                             if (v != 10) ok.store(false, std::memory_order_relaxed);
                           }
                         });
  }
  for (auto& th : threads) th.join();
  EXPECT_TRUE(ok.load());
}

// ============================================================================
// Test 3: CachedFastPath_BypassesMutex
// ============================================================================

TEST_F(NcclParamThreadTest, CachedFastPath_BypassesMutex) {
  SetEnv("TEST_THREAD_CACHED_FAST", "99");
  // Force first load.
  int32_t warmup = testThreadCachedFastPath();
  ASSERT_EQ(warmup, 99);

  static constexpr int kFastIterations = 100000;
  SpinBarrier barrier(kThreadCount);
  std::vector<std::thread> threads;
  std::atomic<bool> ok{true};

  auto start = std::chrono::steady_clock::now();

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&]() {
                           barrier.wait();
                           for (int i = 0; i < kFastIterations; ++i) {
                             int32_t v = testThreadCachedFastPath();
                             if (v != 99) ok.store(false, std::memory_order_relaxed);
                           }
                         });
  }
  for (auto& th : threads) th.join();

  auto elapsed = std::chrono::steady_clock::now() - start;
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

  EXPECT_TRUE(ok.load());
  // Lock-free path should complete well under 2 seconds.
  EXPECT_LT(ms, 2000) << "Cached fast-path took " << ms << "ms — expected lock-free performance";
}

// ============================================================================
// Test 4: ConcurrentRegistryFind
// ============================================================================

TEST_F(NcclParamThreadTest, ConcurrentRegistryFind) {
  SpinBarrier barrier(kThreadCount);
  std::vector<std::thread> threads;
  std::atomic<bool> ok{true};

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&]() {
                           barrier.wait();
                           for (int i = 0; i < kIterations; ++i) {
                             const char* key = (i % 2 == 0) ? "TEST_THREAD_REG_A" : "TEST_THREAD_REG_B";
                             auto* entry = ncclParamRegistry::find(key);
                             if (entry == nullptr) ok.store(false, std::memory_order_relaxed);
                           }
                         });
  }
  for (auto& th : threads) th.join();
  EXPECT_TRUE(ok.load());
}

// ============================================================================
// Test 5: ConcurrentStringReads
// ============================================================================

TEST_F(NcclParamThreadTest, ConcurrentStringReads) {
  SetEnv("TEST_THREAD_STRING", "/path/to/config");

  SpinBarrier barrier(kThreadCount);
  std::vector<std::thread> threads;
  std::atomic<bool> ok{true};

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&]() {
                           barrier.wait();
                           for (int i = 0; i < kIterations; ++i) {
                             const char* v = testThreadString();
                             if (v == nullptr) {
                               ok.store(false, std::memory_order_relaxed);
                               continue;
                             }
                             std::string s(v);
                             if (s != "/path/to/config") {
                               ok.store(false, std::memory_order_relaxed);
                             }
                           }
                         });
  }
  for (auto& th : threads) th.join();
  EXPECT_TRUE(ok.load());
}

// ============================================================================
// Test 6: ConcurrentToString
// ============================================================================

TEST_F(NcclParamThreadTest, ConcurrentToString) {
  SetEnv("TEST_THREAD_TOSTR", "77");

  SpinBarrier barrier(kThreadCount);
  std::vector<std::thread> threads;
  std::atomic<bool> ok{true};

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&]() {
                           barrier.wait();
                           for (int i = 0; i < kIterations; ++i) {
                             std::string s = testThreadToString.toString();
                             if (s != "77") ok.store(false, std::memory_order_relaxed);
                           }
                         });
  }
  for (auto& th : threads) th.join();
  EXPECT_TRUE(ok.load());
}

// ============================================================================
// Test 7: ConcurrentDump
// ============================================================================

TEST_F(NcclParamThreadTest, ConcurrentDump) {
  SpinBarrier barrier(kThreadCount);
  std::vector<std::thread> threads;
  std::atomic<bool> ok{true};

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&]() {
                           barrier.wait();
                           for (int i = 0; i < kIterations; ++i) {
                             std::string d = testThreadDump.dump();
                             if (d.find("TEST_THREAD_DUMP") == std::string::npos) {
                               ok.store(false, std::memory_order_relaxed);
                             }
                           }
                         });
  }
  for (auto& th : threads) th.join();
  EXPECT_TRUE(ok.load());
}

// ============================================================================
// Test 8: CachedFirstLoad_AllThreadsSeeCorrectValue
// ============================================================================

TEST_F(NcclParamThreadTest, CachedFirstLoad_AllThreadsSeeCorrectValue) {
  SetEnv("TEST_THREAD_CACHED_RACE", "777");
  // Do NOT load before spawning threads — all threads race to first-load.

  SpinBarrier barrier(kThreadCount);
  std::vector<std::thread> threads;
  std::atomic<bool> ok{true};

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&]() {
                           barrier.wait();
                           int32_t v = testThreadCachedRace();
                           if (v != 777) ok.store(false, std::memory_order_relaxed);
                         });
  }
  for (auto& th : threads) th.join();
  EXPECT_TRUE(ok.load());
}

// ============================================================================
// Test 9: Concurrent reads on no-cache param
// ============================================================================

DEFINE_NCCL_PARAM(testNoCacheThread, int32_t, TEST_NOCACHE_THREAD, 0,
                  NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT, "");

TEST_F(NcclParamThreadTest, ThreadSafety_ConcurrentReadsOnNoCacheParam) {
  SetEnv("TEST_NOCACHE_THREAD", "42");

  SpinBarrier barrier(kThreadCount);
  std::vector<std::thread> threads;
  std::atomic<bool> ok{true};

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&]() {
                           barrier.wait();
                           for (int i = 0; i < kIterations; ++i) {
                             int32_t v = testNoCacheThread();
                             // Value should always be 42 (env doesn't change during test)
                             if (v != 42) ok.store(false, std::memory_order_relaxed);
                           }
                         });
  }
  for (auto& th : threads) th.join();
  EXPECT_TRUE(ok.load());
}
