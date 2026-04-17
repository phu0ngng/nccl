/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_MATRIX_H_
#define _REDUCE_COPY_TEST_MATRIX_H_

#include "config.h"
#include "api_function_traits.h"
#include "support.h"
#include "test_type_traits.h"
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <mutex>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cuda_fp16.h>
#if defined(__CUDA_BF16_TYPES_EXIST__)
#include <cuda_bf16.h>
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif

// Test matrix to track which test combinations have been executed
class TestMatrix {
public:
  // Key for test matrix: Function × Cooperation Level × Data Type × UNROLL × Count
  // According to test plan: 39 functions × 3 coop levels × 3 UNROLL values × data types × count variations
  struct TestKey {
    ApiFunctionId funcId;
    CooperationLevel coopLevel;
    std::string typeName;  // Use string for type name since we can't use type_info in constexpr context
    int unroll;            // UNROLL template parameter value
    size_t count;          // Data size (number of elements) - at least 2 values:
                           //   one that divides evenly, one that doesn't
    int maxTestGpus;       // 0 = all visible, otherwise explicit GPU count
    bool enableLambdaOffsets;

    bool operator==(const TestKey& other) const {
      return funcId == other.funcId &&
           coopLevel == other.coopLevel &&
           typeName == other.typeName &&
           unroll == other.unroll &&
           count == other.count &&
           maxTestGpus == other.maxTestGpus &&
           enableLambdaOffsets == other.enableLambdaOffsets;
    }
  };

  struct TestKeyHash {
    std::size_t operator()(const TestKey& key) const {
      std::size_t h1 = std::hash<int>{}(static_cast<int>(key.funcId));
      std::size_t h2 = std::hash<int>{}(static_cast<int>(key.coopLevel));
      std::size_t h3 = std::hash<std::string>{}(key.typeName);
      std::size_t h4 = std::hash<int>{}(key.unroll);
      std::size_t h5 = std::hash<size_t>{}(key.count);
      std::size_t h6 = std::hash<int>{}(key.maxTestGpus);
      std::size_t h7 = std::hash<bool>{}(key.enableLambdaOffsets);
      return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6);
    }
  };

private:
  std::unordered_map<TestKey, bool, TestKeyHash> matrix_;
  mutable std::mutex mutex_;
  int visibleGpus_ = -1;

  TestMatrix() = default;
  TestMatrix(const TestMatrix&) = delete;
  TestMatrix& operator=(const TestMatrix&) = delete;

public:
  static TestMatrix& getInstance() {
    static TestMatrix instance;
    return instance;
  }

  void setVisibleGpus(int nVis) {
    std::lock_guard<std::mutex> lock(mutex_);
    visibleGpus_ = nVis;
  }

  int getVisibleGpus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return visibleGpus_;
  }

  // Mark a test combination as executed
  template<typename T>
  void markTested(ApiFunctionId funcId, CooperationLevel coopLevel, int unroll, size_t count,
          int maxTestGpus, bool enableLambdaOffsets) {
    std::lock_guard<std::mutex> lock(mutex_);
    TestKey key{funcId, coopLevel, getTypeName<T>(), unroll, count, maxTestGpus, enableLambdaOffsets};
    matrix_[key] = true;
  }

  // Check if a test combination has been executed
  template<typename T>
  bool isTested(ApiFunctionId funcId, CooperationLevel coopLevel, int unroll, size_t count,
          int maxTestGpus, bool enableLambdaOffsets) const {
    std::lock_guard<std::mutex> lock(mutex_);
    TestKey key{funcId, coopLevel, getTypeName<T>(), unroll, count, maxTestGpus, enableLambdaOffsets};
    return matrix_.find(key) != matrix_.end() && matrix_.at(key);
  }

  // Get type name as string — generated from NCCL_REDUCE_COPY_TYPE_LIST in test_type_traits.h.
  template<typename T>
  static std::string getTypeName() {
#define X(cpptype, tag, guard, mmSrc, mul, copyOnly) \
    if NCCL_IF_CONSTEXPR (std::is_same<T, cpptype>::value) return #cpptype;
    NCCL_REDUCE_COPY_TYPE_LIST
#undef X
    return "unknown";
  }

  struct TypeInfo {
    std::string name;
    std::string tag;
    int defaultUnroll;       // 8*16/sizeof(T) = 128/sizeof(T)
    bool multimemSourceSupported;  // delegates to TestSupportChecker::isMultimemTypeSupported<T>()
    bool mulSupported;             // delegates to TypeSupportMatrix::isMulSupported<T>()
    bool copyOnlySupported;        // from test_type_traits.h copyOnly flag
  };

  static std::string normalizeTypeToken(const std::string& token) {
    std::string out;
    out.reserve(token.size());
    for (char c : token) {
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '_') continue;
      out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
  }

  // getAllTypes() — generated from NCCL_REDUCE_COPY_TYPE_LIST in test_type_traits.h.
  // multimemSourceSupported uses TestSupportChecker (compile+runtime) so the expected
  // matrix accounts for arch availability (e.g. FP8 multimem requires sm_90+).
  // mulSupported uses TypeSupportMatrix (compile-time) since it depends only on type.
  template<typename T>
  static TypeInfo makeTypeInfo(const std::string& name, const std::string& tag, bool copyOnly) {
    return {name, tag, getDefaultUnroll<T>(),
        TestSupportChecker::isMultimemTypeSupported<T>(),
        TypeSupportMatrix::isMulSupported<T>(),
        copyOnly};
  }

  static std::vector<TypeInfo> getAllTypes() {
#define X(cpptype, tag, guard, mmSrc, mul, copyOnly) makeTypeInfo<cpptype>(#cpptype, #tag, copyOnly),
    return {NCCL_REDUCE_COPY_TYPE_LIST};
#undef X
  }

  static std::unordered_set<std::string> getSelectedTypeNames(std::vector<std::string>* unknown = nullptr) {
    const char* raw = std::getenv("NCCL_TEST_REDUCE_COPY_TYPES");
    if (!raw || std::string(raw).empty()) {
      return {};
    }
    std::unordered_map<std::string, std::string> tokenToName;
    for (const auto& typeInfo : getAllTypes()) {
      tokenToName[normalizeTypeToken(typeInfo.name)] = typeInfo.name;
      tokenToName[normalizeTypeToken(typeInfo.tag)] = typeInfo.name;
    }
    std::unordered_set<std::string> selected;
    std::string input(raw);
    std::stringstream ss(input);
    std::string token;
    while (std::getline(ss, token, ',')) {
      std::string norm = normalizeTypeToken(token);
      if (norm.empty()) continue;
      if (norm == "all") {
        return {};
      }
      auto it = tokenToName.find(norm);
      if (it != tokenToName.end()) {
        selected.insert(it->second);
      } else if (unknown) {
        unknown->push_back(token);
      }
    }
    return selected;
  }

  static std::vector<std::string> getAllTypeNames() {
    std::vector<std::string> names;
    for (const auto& t : getAllTypes()) names.push_back(t.name);
    return names;
  }

  static std::vector<std::string> getNotSelectedTypeNames(std::vector<std::string>* unknown = nullptr) {
    std::unordered_set<std::string> selected = getSelectedTypeNames(unknown);
    std::vector<std::string> notSelected;
    if (selected.empty()) return notSelected;
    for (const auto& t : getAllTypes()) {
      if (!selected.count(t.name)) {
        notSelected.push_back(t.name);
      }
    }
    return notSelected;
  }

  // Get all expected test combinations
  // getExpectedTests() — enumerate every test key that SHOULD be executed.
  //
  // Expected combination rules (must stay in sync with generate_tests.py):
  //
  //   Axes:
  //     functions   — all ApiFunctionId values from api_function_traits.h (39 total)
  //     coopLevels  — Thread, Warp, Cta  (3)
  //     types       — all entries from test_type_traits.h NCCL_REDUCE_COPY_TYPE_LIST_ALL
  //     unrolls     — 1, defaultUnroll  (2, type-dependent)
  //     types (AllGather only) — restricted to copyOnly=true types (1B/2B/4B/8B representatives)
  //     counts      — NCCL_REDUCE_COPY_COUNTS_LIST from config.h
  //     gpuModes    — local functions: {1}; inter-rank functions: {0=all-visible, 3}
  //                   (clipped by visibleGpus when < 2 or < 3)
  //     offsetModes — lambda functions: {false, true}; others: {false}
  //
  //   Exclusions (same in C++ and Python):
  //     multimem-source function + type where typeInfo.multimemSourceSupported == false
  //       → skip (no multimem load/store specialization for this type).
  //     mul-variant function + type where typeInfo.mulSupported == false
  //       → skip (NCCL type-promotion conflict; OpMul accepts only exact element type).
  //
  //   Python alignment:
  //     generate_tests.py reads NCCL_REDUCE_COPY_TYPE_LIST_ALL for TYPES, derives
  //     MULTIMEM_EXCLUDED_TYPES and MUL_EXCLUDED_TYPES from the same source, and applies
  //     the same exclusions when emitting INSTANTIATE_TEST_CASE_P blocks.
  //     Any change to exclusion logic here must also be reflected in render_suite_file().
#if !defined(__CUDA_ARCH__)
  static std::vector<TestKey> getExpectedTests() {
    std::vector<TestKey> expected;

    // Get all API functions from central traits (single source of truth)
    std::vector<ApiFunctionId> functions = getAllApiFunctionIds();

    // Get all cooperation levels (3 levels)
    std::vector<CooperationLevel> coopLevels = {
      CooperationLevel::Thread,
      CooperationLevel::Warp,
      CooperationLevel::Cta
    };

    // Get all data types that should be tested
    // Based on test plan section 5: Standard types + Special floating-point types
    // Each type entry includes: type name and its default UNROLL value
    std::vector<TypeInfo> types = getAllTypes();
    std::unordered_set<std::string> selected = getSelectedTypeNames();

    // Get UNROLL values: 1, default, and 2x default (from test plan line 1677)
    // Note: Default and 2x default are type-dependent, so we compute them per type

    // Get count values: at least 2 - one that divides evenly across ranks, one that doesn't
    // For typical 4-rank setup:
    // - Evenly dividing: 32768 (divisible by 4)
    // - Not evenly dividing: 997 (prime, not divisible by 4)
    // These test large transfers and remainder handling edge cases
    std::vector<size_t> counts = {NCCL_REDUCE_COPY_COUNTS_LIST};

    // Generate all combinations: functions × 3 coop levels × types × 3 UNROLL values × 2 counts
    // × run modes (gpus, offsets)
    // Exclude multimem tests for types that don't support multimem
    // Exclude multiplication variant for types that don't support multiplication
    int visibleGpus = TestMatrix::getInstance().getVisibleGpus();
    for (auto funcId : functions) {
      bool isMultimemFunc = TestSupportChecker::isMultimemVariant(funcId);
      bool hasMultimemSource = TestSupportChecker::hasMultimemSource(funcId);
      bool hasMultimemDestination = TestSupportChecker::hasMultimemDestination(funcId);
      bool isMulFunc = apiTraitsIsMulVariant(funcId);
      bool isLocalFunc = TestSupportChecker::isLocalVariant(funcId);
      bool isLambdaFunc = apiTraitsIsLambda(funcId);

      for (auto coopLevel : coopLevels) {
        for (const auto& typeInfo : types) {
          if (!selected.empty() && !selected.count(typeInfo.name)) {
            continue;
          }
          // Skip if this function uses multimem sources but the type doesn't support them
          if (hasMultimemSource && !typeInfo.multimemSourceSupported) {
            continue;
          }
          // Skip if this is a multiplication function but the type doesn't support it
          if (isMulFunc && !typeInfo.mulSupported) {
            continue;
          }
          // Apply testing-group type restrictions (see api_function_traits.h X-macro comment):
          //   Group 2 (thin wrappers): restrict to copyOnly types (explicit-copy type set).
          //   Group 3 (handle/window translation): float only.
          //   Group 1 (owns alignment): all types — no restriction.
          {
            int grp = apiTraitsTestingGroup(funcId);
            if (grp == 2 && !typeInfo.copyOnlySupported) continue;
            if (grp == 3 && typeInfo.name != "float") continue;
          }

          // For each type, test with UNROLL=1 and default
          int unroll1 = 1;
          int unrollDefault = typeInfo.defaultUnroll;

          // For each UNROLL value, test with both count values
          for (size_t count : counts) {
            std::vector<int> gpuModes = isLocalFunc ? std::vector<int>{1} : std::vector<int>{0, 3};
            if (!isLocalFunc && visibleGpus >= 0 && visibleGpus < 2) {
              // Inter-rank tests are impossible without multiple GPUs.
              gpuModes.clear();
            } else if (!isLocalFunc && visibleGpus >= 0 && visibleGpus < 3) {
              // Drop explicit 3-GPU mode when fewer than 3 GPUs are visible.
              gpuModes.erase(std::remove(gpuModes.begin(), gpuModes.end(), 3), gpuModes.end());
            }
            if (isLocalFunc && visibleGpus == 0) {
              gpuModes.clear();
            }
            std::vector<bool> offsetModes = isLambdaFunc ? std::vector<bool>{false, true}
                                   : std::vector<bool>{false};
            for (int maxGpus : gpuModes) {
              for (bool offsets : offsetModes) {
                expected.push_back({funcId, coopLevel, typeInfo.name, unroll1, count, maxGpus, offsets});
                expected.push_back({funcId, coopLevel, typeInfo.name, unrollDefault, count, maxGpus, offsets});
              }
            }
          }
        }
      }
    }

    return expected;
  }

  // Verify all expected tests have been executed
  std::vector<TestKey> getMissingTests() const {
    // Build expected list before locking. getExpectedTests() queries visibleGpus_.
    std::vector<TestKey> expected = getExpectedTests();
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<TestKey> missing;

    for (const auto& key : expected) {
      auto it = matrix_.find(key);
      if (it == matrix_.end() || !it->second) {
        missing.push_back(key);
      }
    }

    return missing;
  }

  // Get summary statistics
  struct Stats {
    size_t totalExpected;
    size_t totalExecuted;
    size_t totalMissing;
  };

  struct TypeCoverage {
    std::string typeName;
    size_t expected;
    size_t executed;
    size_t missing;
    double coverage;
  };

  std::vector<TypeCoverage> getTypeCoverage() const {
    std::vector<TestKey> expected = getExpectedTests();
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<std::string, size_t> expectedCounts;
    std::unordered_map<std::string, size_t> executedCounts;

    for (const auto& key : expected) {
      expectedCounts[key.typeName]++;
      auto it = matrix_.find(key);
      if (it != matrix_.end() && it->second) {
        executedCounts[key.typeName]++;
      }
    }

    std::vector<TypeCoverage> result;
    std::unordered_set<std::string> selected = getSelectedTypeNames();
    for (const auto& typeInfo : getAllTypes()) {
      if (!selected.empty() && !selected.count(typeInfo.name)) {
        continue;
      }
      size_t expectedCount = expectedCounts[typeInfo.name];
      size_t executedCount = executedCounts[typeInfo.name];
      size_t missingCount = expectedCount - executedCount;
      double coverage = expectedCount > 0 ? (100.0 * executedCount / expectedCount) : 0.0;
      result.push_back({typeInfo.name, expectedCount, executedCount, missingCount, coverage});
    }
    return result;
  }

  Stats getStats() const {
    // Build expected list before locking. getExpectedTests() queries visibleGpus_.
    std::vector<TestKey> expected = getExpectedTests();
    std::lock_guard<std::mutex> lock(mutex_);
    size_t executed = 0;

    for (const auto& key : expected) {
      auto it = matrix_.find(key);
      if (it != matrix_.end() && it->second) {
        executed++;
      }
    }

    return {expected.size(), executed, expected.size() - executed};
  }

  // Get API function name as string (shared helper) — from central traits.
  static std::string getApiFunctionName(ApiFunctionId funcId) {
    return getApiFunctionNameFromTraits(funcId);
  }

  // Get cooperation level name as string (shared helper)
  static std::string getCooperationLevelName(CooperationLevel coopLevel) {
    switch (coopLevel) {
      case CooperationLevel::Thread: return "Thread";
      case CooperationLevel::Warp: return "Warp";
      case CooperationLevel::Cta: return "Cta";
      default: return "Unknown_" + std::to_string(static_cast<int>(coopLevel));
    }
  }

  // Get human-readable string for a test key
  static std::string keyToString(const TestKey& key) {
    std::ostringstream oss;
    oss << getApiFunctionName(key.funcId) << " × "
      << getCooperationLevelName(key.coopLevel) << " × "
      << key.typeName << " × UNROLL=" << key.unroll << " × count=" << key.count
      << " × gpus=" << key.maxTestGpus
      << " × offsets=" << (key.enableLambdaOffsets ? "true" : "false");
    return oss.str();
  }

#endif // !__CUDA_ARCH__

  // Reset matrix (useful for testing the matrix itself)
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    matrix_.clear();
  }
};

// Helper macro to mark a test as executed
#define MARK_TEST_EXECUTED(T, funcId, coopLevel, unroll, count, maxTestGpus, enableLambdaOffsets) \
  TestMatrix::getInstance().markTested<T>(funcId, coopLevel, unroll, count, maxTestGpus, enableLambdaOffsets)

#endif // _REDUCE_COPY_TEST_MATRIX_H_

