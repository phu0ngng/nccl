/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Unit tests for the ncclParam framework (param.h + c_api.h)
//
// All parameters use TEST_* prefixed keys to avoid collision with
// production NCCL_* parameters that may be registered by nccl_static.

#include <gtest/gtest.h>
#include "param/param.h"
#include <cstdlib>
#include <string>
#include <set>

// ============================================================================
// Test Enums and Option Sets
// ============================================================================

enum TestEnumOption : int32_t {
  TEST_ENUM_NONE = 0,
  TEST_ENUM_WARN = 1,
  TEST_ENUM_INFO = 2,
  TEST_ENUM_TRACE = 3
};

enum class TestBitmaskOption : uint32_t {
  INIT  = 1u << 0,
  COLL  = 1u << 1,
  P2P   = 1u << 2,
  GRAPH = 1u << 3,
  ALL   = 0x0F
};

// Option sets must be defined BEFORE any DEFINE_NCCL_PARAM that uses them.
static auto testEnumOptionSet = makeOptions(
  makeOption<int32_t>("NONE",  TEST_ENUM_NONE),
  makeOption<int32_t>("WARN",  TEST_ENUM_WARN),
  makeOption<int32_t>("INFO",  TEST_ENUM_INFO),
  makeOption<int32_t>("TRACE", TEST_ENUM_TRACE)
);

// Option set with per-option descriptions (for Change 1 tests)
static auto testDescEnumOptionSet = makeOptions(
  makeOption<int32_t>("OFF",   0, "Disable feature"),
  makeOption<int32_t>("ON",    1, "Enable feature"),
  makeOption<int32_t>("AUTO",  2)
);

static auto testBitmaskOptionSet = makeOptions(
  makeOption<TestBitmaskOption>("INIT",  TestBitmaskOption::INIT),
  makeOption<TestBitmaskOption>("COLL",  TestBitmaskOption::COLL),
  makeOption<TestBitmaskOption>("P2P",   TestBitmaskOption::P2P),
  makeOption<TestBitmaskOption>("GRAPH", TestBitmaskOption::GRAPH),
  makeOption<TestBitmaskOption>("ALL",   TestBitmaskOption::ALL)
);

// ============================================================================
// Parameter Definitions
// ============================================================================

// --- Group 1-2: Basic types ---
DEFINE_NCCL_PARAM(testDefineUseParam, int32_t, TEST_DEFINE_USE, 42,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testIntDefault, int32_t, TEST_INT_DEFAULT, 100,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testIntFromEnv, int32_t, TEST_INT_FROM_ENV, 100,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testBoolDefault, bool, TEST_BOOL_DEFAULT, false,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testBoolFromEnv, bool, TEST_BOOL_FROM_ENV, false,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testEnumOptions, int32_t, TEST_ENUM_OPTIONS, TEST_ENUM_NONE,
                  NCCL_PARAM_FLAG_NONE, ncclParamOneOf(testEnumOptionSet), "");
DEFINE_NCCL_PARAM(testBitmaskOptions, uint32_t, TEST_BITMASK_OPTIONS, 0u,
                  NCCL_PARAM_FLAG_NONE, ncclParamBitsetOf<TestBitmaskOption>(testBitmaskOptionSet), "");
DEFINE_NCCL_PARAM(testIntInvalidEnv, int32_t, TEST_INT_INVALID_ENV, 77,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testEnumInvalidEnv, int32_t, TEST_ENUM_INVALID_ENV, TEST_ENUM_WARN,
                  NCCL_PARAM_FLAG_NONE, ncclParamOneOf(testEnumOptionSet), "");

// --- Group 3: Env reload ---
DEFINE_NCCL_PARAM(testReloadable, int32_t, TEST_RELOADABLE, 10,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");

// --- Group 4: Cached (separate keys per scenario) ---
DEFINE_NCCL_PARAM(testMaxRetriesFromEnv, int32_t, TEST_MAX_RETRIES_FROM_ENV, 3,
                  NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT, "");

// --- Group 5: Bounded ---
DEFINE_NCCL_PARAM(testBoundedValid, int32_t, TEST_BOUNDED_VALID, 4,
                  NCCL_PARAM_FLAG_NONE, ncclParamBounded(1, 64), "");
DEFINE_NCCL_PARAM(testBoundedInvalid, int32_t, TEST_BOUNDED_INVALID, 4,
                  NCCL_PARAM_FLAG_NONE, ncclParamBounded(1, 64), "");
DEFINE_NCCL_PARAM(testLowerBoundOnly, int32_t, TEST_LOWER_BOUND_ONLY, 10,
                  NCCL_PARAM_FLAG_NONE, ncclParamBounded(0), "");

// --- Group 6: String ---
DEFINE_NCCL_PARAM(testStringDefault, const char*, TEST_STRING_DEFAULT, nullptr,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testStringCustom, const char*, TEST_STRING_CUSTOM, nullptr,
                  NCCL_PARAM_FLAG_PUBLISHED, NCCL_PARAM_DEFAULT, "");

// --- Group 7: Dump ---
DEFINE_NCCL_PARAM(testPublishedParam, int32_t, TEST_PUBLISHED_PARAM, 99,
                  NCCL_PARAM_FLAG_PUBLISHED, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testPrivateParam, int32_t, TEST_PRIVATE_PARAM, 88,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testDeprecatedParam, int32_t, TEST_DEPRECATED_PARAM, 55,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_DEPRECATED, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testPublishedReadOnly, int32_t, TEST_PUBLISHED_READONLY, 66,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testPublishedEnumOptions, int32_t, TEST_PUBLISHED_ENUM_OPTIONS, 0,
                  NCCL_PARAM_FLAG_PUBLISHED, ncclParamOneOf(testEnumOptionSet), "");
DEFINE_NCCL_PARAM(testPublishedBitmaskOptions, uint32_t, TEST_PUBLISHED_BITMASK_OPTIONS, 0u,
                  NCCL_PARAM_FLAG_PUBLISHED, ncclParamBitsetOf<TestBitmaskOption>(testBitmaskOptionSet), "");

// --- Group 8: C API ---
DEFINE_NCCL_PARAM(testCApiInt, int32_t, TEST_CAPI_INT, 42,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testCApiLong, int64_t, TEST_CAPI_LONG, 1000,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testCApiString, const char*, TEST_CAPI_STRING, nullptr,
                  NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT, "");
// --- Group 9: Option descriptions ---
DEFINE_NCCL_PARAM(testDescEnumParam, int32_t, TEST_DESC_ENUM, 0,
                  NCCL_PARAM_FLAG_PUBLISHED, ncclParamOneOf(testDescEnumOptionSet), "");

// --- Group 10: Source tracking ---
DEFINE_NCCL_PARAM(testSourceDefault, int32_t, TEST_SOURCE_DEFAULT, 42,
                  NCCL_PARAM_FLAG_PUBLISHED, NCCL_PARAM_DEFAULT, "");
DEFINE_NCCL_PARAM(testSourceEnvPlugin, int32_t, TEST_SOURCE_ENVPLUGIN, 42,
                  NCCL_PARAM_FLAG_PUBLISHED, NCCL_PARAM_DEFAULT, "");

// ============================================================================
// Test Fixture
// ============================================================================

class NcclParamTest : public ::testing::Test {
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
// DefineUse
// ============================================================================

static int32_t getDefineUseValue() {
  USE_NCCL_PARAM(testDefineUseParam, int32_t);
  return testDefineUseParam();
}

TEST_F(NcclParamTest, DefineUse_FunctionScopeAccess) {
  EXPECT_EQ(getDefineUseValue(), 42);
}

// ============================================================================
// EnvValues
// ============================================================================

TEST_F(NcclParamTest, EnvValues_IntDefault) {
  EXPECT_EQ(testIntDefault(), 100);
}

TEST_F(NcclParamTest, EnvValues_IntFromEnv) {
  SetEnv("TEST_INT_FROM_ENV", "200");
  EXPECT_EQ(testIntFromEnv(), 200);
}

TEST_F(NcclParamTest, EnvValues_BoolDefault) {
  EXPECT_EQ(testBoolDefault(), false);
}

TEST_F(NcclParamTest, EnvValues_BoolFromEnvVariants) {
  // True variants
  for (const char* val : {"1", "T", "TRUE"}) {
    SetEnv("TEST_BOOL_FROM_ENV", val);
    EXPECT_TRUE(testBoolFromEnv()) << "Expected true for: " << val;
  }
  // False variants
  for (const char* val : {"0", "F", "FALSE"}) {
    SetEnv("TEST_BOOL_FROM_ENV", val);
    EXPECT_FALSE(testBoolFromEnv()) << "Expected false for: " << val;
  }
}

TEST_F(NcclParamTest, EnvValues_EnumOptionsFromEnv) {
  SetEnv("TEST_ENUM_OPTIONS", "INFO");
  EXPECT_EQ(testEnumOptions(), TEST_ENUM_INFO);
}

TEST_F(NcclParamTest, EnvValues_BitmaskOptions) {
  // Single flag
  SetEnv("TEST_BITMASK_OPTIONS", "INIT");
  EXPECT_EQ(testBitmaskOptions(), 0x01u);

  // Multiple flags
  SetEnv("TEST_BITMASK_OPTIONS", "INIT,P2P");
  EXPECT_EQ(testBitmaskOptions(), 0x01u | 0x04u);

  // Composite flag
  SetEnv("TEST_BITMASK_OPTIONS", "ALL");
  EXPECT_EQ(testBitmaskOptions(), 0x0Fu);
}

TEST_F(NcclParamTest, EnvValues_IntInvalidFallback) {
  SetEnv("TEST_INT_INVALID_ENV", "not_a_number");
  EXPECT_EQ(testIntInvalidEnv(), 77);
}

TEST_F(NcclParamTest, EnvValues_EnumInvalidFallback) {
  SetEnv("TEST_ENUM_INVALID_ENV", "INVALID");
  EXPECT_EQ(testEnumInvalidEnv(), TEST_ENUM_WARN);
}

// ============================================================================
// EnvReload
// ============================================================================

TEST_F(NcclParamTest, EnvReload_NonReadOnlyReloads) {
  EXPECT_EQ(testReloadable(), 10);

  SetEnv("TEST_RELOADABLE", "20");
  EXPECT_EQ(testReloadable(), 20);

  SetEnv("TEST_RELOADABLE", "30");
  EXPECT_EQ(testReloadable(), 30);
}

// ============================================================================
// Cached
// ============================================================================

TEST_F(NcclParamTest, Cached_LoadsFromEnv) {
  SetEnv("TEST_MAX_RETRIES_FROM_ENV", "10");
  // First access loads from env and caches
  EXPECT_EQ(testMaxRetriesFromEnv(), 10);
  // Subsequent access returns cached value
  unsetenv("TEST_MAX_RETRIES_FROM_ENV");
  EXPECT_EQ(testMaxRetriesFromEnv(), 10);
}

// ============================================================================
// Bounded
// ============================================================================

TEST_F(NcclParamTest, Bounded_ValidRange) {
  // Lower bound
  SetEnv("TEST_BOUNDED_VALID", "1");
  EXPECT_EQ(testBoundedValid(), 1);

  // Upper bound
  SetEnv("TEST_BOUNDED_VALID", "64");
  EXPECT_EQ(testBoundedValid(), 64);

  // Middle of range
  SetEnv("TEST_BOUNDED_VALID", "16");
  EXPECT_EQ(testBoundedValid(), 16);
}

TEST_F(NcclParamTest, Bounded_OutOfRange) {
  // Below range falls back to default
  SetEnv("TEST_BOUNDED_INVALID", "0");
  EXPECT_EQ(testBoundedInvalid(), 4);

  // Above range falls back to default
  SetEnv("TEST_BOUNDED_INVALID", "100");
  EXPECT_EQ(testBoundedInvalid(), 4);
}

TEST_F(NcclParamTest, Bounded_LowerBoundOnly) {
  SetEnv("TEST_LOWER_BOUND_ONLY", "0");
  EXPECT_EQ(testLowerBoundOnly(), 0);

  SetEnv("TEST_LOWER_BOUND_ONLY", "1000000");
  EXPECT_EQ(testLowerBoundOnly(), 1000000);

  SetEnv("TEST_LOWER_BOUND_ONLY", "-1");
  EXPECT_EQ(testLowerBoundOnly(), 10);
}

// ============================================================================
// String
// ============================================================================

TEST_F(NcclParamTest, String_DefaultNullptr) {
  const char* val = testStringDefault();
  EXPECT_EQ(val, nullptr);
}

TEST_F(NcclParamTest, String_CustomValue) {
  SetEnv("TEST_STRING_CUSTOM", "/etc/test.conf");
  const char* val = testStringCustom();
  ASSERT_NE(val, nullptr);
  EXPECT_STREQ(val, "/etc/test.conf");
}

// ============================================================================
// Dump
// ============================================================================

TEST_F(NcclParamTest, Dump_PublishedHasContent) {
  std::string d = testPublishedParam.dump();
  EXPECT_FALSE(d.empty());
  EXPECT_NE(d.find("TEST_PUBLISHED_PARAM"), std::string::npos);
  EXPECT_NE(d.find("int32_t"), std::string::npos);
  EXPECT_NE(d.find("default="), std::string::npos);
}

TEST_F(NcclParamTest, Dump_PrivateHasContent) {
  std::string d = testPrivateParam.dump();
  EXPECT_FALSE(d.empty());
  EXPECT_NE(d.find("TEST_PRIVATE_PARAM"), std::string::npos);
}

TEST_F(NcclParamTest, Dump_DeprecatedShowsFlag) {
  std::string d = testDeprecatedParam.dump();
  EXPECT_NE(d.find("Deprecated"), std::string::npos);
}

TEST_F(NcclParamTest, Dump_ReadOnlyShowsFlag) {
  std::string d = testPublishedReadOnly.dump();
  EXPECT_NE(d.find("Cached"), std::string::npos);
}

TEST_F(NcclParamTest, Dump_EnumOptionsShowsAcceptedValues) {
  std::string d = testPublishedEnumOptions.dump();
  EXPECT_NE(d.find("One of:"), std::string::npos);
  EXPECT_NE(d.find("NONE"), std::string::npos);
  EXPECT_NE(d.find("WARN"), std::string::npos);
  EXPECT_NE(d.find("INFO"), std::string::npos);
  EXPECT_NE(d.find("TRACE"), std::string::npos);
}

TEST_F(NcclParamTest, Dump_BitmaskOptionsShowsAcceptedValues) {
  std::string d = testPublishedBitmaskOptions.dump();
  EXPECT_NE(d.find("Comma-separated list of:"), std::string::npos);
  EXPECT_NE(d.find("INIT"), std::string::npos);
  EXPECT_NE(d.find("P2P"), std::string::npos);
}

TEST_F(NcclParamTest, Dump_AlwaysReturnsContent) {
  // Both private and published params always return content from dump()
  std::string d1 = testPrivateParam.dump();
  EXPECT_FALSE(d1.empty());
  EXPECT_NE(d1.find("TEST_PRIVATE_PARAM"), std::string::npos);

  std::string d2 = testPublishedParam.dump();
  EXPECT_FALSE(d2.empty());
  EXPECT_NE(d2.find("TEST_PUBLISHED_PARAM"), std::string::npos);
}

// ============================================================================
// DumpAll Parameter
// ============================================================================

USE_NCCL_PARAM(ncclParamDumpAllFlag, bool);

TEST_F(NcclParamTest, DumpAllFlag_DefaultsFalse) {
  EXPECT_FALSE(ncclParamDumpAllFlag());
}

TEST_F(NcclParamTest, DumpAllFlag_SetFromEnv) {
  SetEnv("NCCL_PARAM_DUMP_ALL", "TRUE");
  EXPECT_TRUE(ncclParamDumpAllFlag());
}

// ============================================================================
// C API
// ============================================================================

TEST_F(NcclParamTest, CApi_BindSuccess) {
  ncclParamHandle_t h = nullptr;
  EXPECT_EQ(ncclParamBind(&h, "TEST_CAPI_INT"), ncclSuccess);
  EXPECT_NE(h, nullptr);
}

TEST_F(NcclParamTest, CApi_NotFoundErrors) {
  // Bind
  ncclParamHandle_t h = nullptr;
  EXPECT_EQ(ncclParamBind(&h, "TEST_NONEXISTENT_KEY_XYZ"), ncclInvalidArgument);

  // GetParameter
  const char* val = nullptr;
  int len = 0;
  EXPECT_EQ(ncclParamGetParameter("TEST_NONEXISTENT_KEY_XYZ", &val, &len), ncclInvalidArgument);
}

TEST_F(NcclParamTest, CApi_TypedGetI32) {
  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_CAPI_INT"), ncclSuccess);

  int32_t val = 0;
  EXPECT_EQ(ncclParamGetI32(h, &val), ncclSuccess);
  EXPECT_EQ(val, 42);
}

TEST_F(NcclParamTest, CApi_TypeMismatch) {
  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_CAPI_INT"), ncclSuccess);

  int64_t val = 0;
  EXPECT_EQ(ncclParamGetI64(h, &val), ncclInvalidArgument);
}

TEST_F(NcclParamTest, CApi_NullArgErrors) {
  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_CAPI_INT"), ncclSuccess);

  // Bind null args
  ncclParamHandle_t h2 = nullptr;
  EXPECT_EQ(ncclParamBind(nullptr, "TEST_CAPI_INT"), ncclInvalidArgument);
  EXPECT_EQ(ncclParamBind(&h2, nullptr), ncclInvalidArgument);

  // Get null args
  EXPECT_EQ(ncclParamGetI32(h, nullptr), ncclInvalidArgument);
  int32_t val = 0;
  EXPECT_EQ(ncclParamGetI32(nullptr, &val), ncclInvalidArgument);

  // GetParameter null args
  const char* str = nullptr;
  int len = 0;
  EXPECT_EQ(ncclParamGetParameter(nullptr, &str, &len), ncclInvalidArgument);
  EXPECT_EQ(ncclParamGetParameter("TEST_CAPI_INT", nullptr, &len), ncclInvalidArgument);
  EXPECT_EQ(ncclParamGetParameter("TEST_CAPI_INT", &str, nullptr), ncclInvalidArgument);

  // GetAllParameterKeys null args
  const char** keys = nullptr;
  int count = 0;
  EXPECT_EQ(ncclParamGetAllParameterKeys(nullptr, &count), ncclInvalidArgument);
  EXPECT_EQ(ncclParamGetAllParameterKeys(&keys, nullptr), ncclInvalidArgument);
}

TEST_F(NcclParamTest, CApi_GetStr) {
  SetEnv("TEST_CAPI_STRING", "/tmp/getstr.log");

  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_CAPI_STRING"), ncclSuccess);

  const char* str = nullptr;
  EXPECT_EQ(ncclParamGetStr(h, &str), ncclSuccess);
  ASSERT_NE(str, nullptr);
  EXPECT_STREQ(str, "/tmp/getstr.log");

  // Type mismatch: GetStr on a non-string param
  ncclParamHandle_t hInt = nullptr;
  ASSERT_EQ(ncclParamBind(&hInt, "TEST_CAPI_INT"), ncclSuccess);
  EXPECT_EQ(ncclParamGetStr(hInt, &str), ncclInvalidArgument);
}

TEST_F(NcclParamTest, CApi_GetParameter) {
  const char* val = nullptr;
  int len = 0;
  EXPECT_EQ(ncclParamGetParameter("TEST_CAPI_INT", &val, &len), ncclSuccess);
  ASSERT_NE(val, nullptr);
  EXPECT_GT(len, 0);
  EXPECT_EQ(std::string(val, len), "42");
}

TEST_F(NcclParamTest, CApi_GetAllParameterKeys) {
  // Enable showAll so non-published TEST_* keys are included
  SetEnv("NCCL_PARAM_DUMP_ALL", "TRUE");

  const char** keys = nullptr;
  int count = 0;
  EXPECT_EQ(ncclParamGetAllParameterKeys(&keys, &count), ncclSuccess);
  EXPECT_GT(count, 0);

  std::set<std::string> key_set;
  for (int i = 0; i < count; ++i) {
    ASSERT_NE(keys[i], nullptr);
    key_set.insert(keys[i]);
  }

  // Our TEST_* keys should be present
  EXPECT_TRUE(key_set.count("TEST_CAPI_INT") > 0);
  EXPECT_TRUE(key_set.count("TEST_INT_DEFAULT") > 0);
  EXPECT_TRUE(key_set.count("TEST_ENUM_OPTIONS") > 0);
}

TEST_F(NcclParamTest, CApi_StringParam) {
  SetEnv("TEST_CAPI_STRING", "/tmp/test.log");

  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_CAPI_STRING"), ncclSuccess);

  // GetStr returns the string value
  const char* str = nullptr;
  EXPECT_EQ(ncclParamGetStr(h, &str), ncclSuccess);
  ASSERT_NE(str, nullptr);
  EXPECT_STREQ(str, "/tmp/test.log");
}

// ============================================================================
// ncclParamGet (raw binary copy)
// ============================================================================

TEST_F(NcclParamTest, CApi_GetRawInt32) {
  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_CAPI_INT"), ncclSuccess);

  int32_t val = 0;
  int len = 0;
  EXPECT_EQ(ncclParamGet(h, &val, static_cast<int>(sizeof(val)), &len), ncclSuccess);
  EXPECT_EQ(len, static_cast<int>(sizeof(int32_t)));
  EXPECT_EQ(val, 42);
}

TEST_F(NcclParamTest, CApi_GetRawInt64) {
  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_CAPI_LONG"), ncclSuccess);

  int64_t val = 0;
  int len = 0;
  EXPECT_EQ(ncclParamGet(h, &val, static_cast<int>(sizeof(val)), &len), ncclSuccess);
  EXPECT_EQ(len, static_cast<int>(sizeof(int64_t)));
  EXPECT_EQ(val, 1000);
}

TEST_F(NcclParamTest, CApi_GetRawString) {
  SetEnv("TEST_CAPI_STRING", "/tmp/test.log");

  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_CAPI_STRING"), ncclSuccess);

  char buf[256] = {};
  int len = 0;
  EXPECT_EQ(ncclParamGet(h, buf, static_cast<int>(sizeof(buf)), &len), ncclSuccess);
  EXPECT_EQ(len, static_cast<int>(strlen("/tmp/test.log") + 1));
  EXPECT_STREQ(buf, "/tmp/test.log");
}

// const char* param with null default: raw copy yields a single '\0' byte
TEST_F(NcclParamTest, CApi_GetRawStringNullDefault) {
  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_STRING_DEFAULT"), ncclSuccess);

  char buf[16];
  buf[0] = '\x7F';  // Sentinel — verifies first byte is overwritten
  int len = 0;
  EXPECT_EQ(ncclParamGet(h, buf, static_cast<int>(sizeof(buf)), &len), ncclSuccess);
  EXPECT_EQ(len, 1);
  EXPECT_EQ(buf[0], '\0');
}

TEST_F(NcclParamTest, CApi_GetRawBufferTooSmall) {
  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_CAPI_INT"), ncclSuccess);

  int8_t small = 0;
  int len = 99;  // Should be reset to 0 on error
  EXPECT_EQ(ncclParamGet(h, &small, static_cast<int>(sizeof(small)), &len), ncclInvalidArgument);
  EXPECT_EQ(len, 0);
}

TEST_F(NcclParamTest, CApi_GetRawStringBufferTooSmall) {
  SetEnv("TEST_CAPI_STRING", "hello");

  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_CAPI_STRING"), ncclSuccess);

  char tiny[2] = {};
  int len = 0;
  // "hello" + null terminator = 6 bytes; buffer holds only 2
  EXPECT_EQ(ncclParamGet(h, tiny, static_cast<int>(sizeof(tiny)), &len), ncclInvalidArgument);
  EXPECT_EQ(len, 0);
}

TEST_F(NcclParamTest, CApi_GetRawNullArgs) {
  ncclParamHandle_t h = nullptr;
  ASSERT_EQ(ncclParamBind(&h, "TEST_CAPI_INT"), ncclSuccess);

  int32_t val = 0;
  int len = 0;
  EXPECT_EQ(ncclParamGet(nullptr, &val, static_cast<int>(sizeof(val)), &len), ncclInvalidArgument);
  EXPECT_EQ(ncclParamGet(h, nullptr, static_cast<int>(sizeof(val)), &len), ncclInvalidArgument);
  EXPECT_EQ(ncclParamGet(h, &val, static_cast<int>(sizeof(val)), nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclParamGet(h, &val, 0, &len), ncclInvalidArgument);
}

// ============================================================================
// Option Descriptions
// ============================================================================

TEST_F(NcclParamTest, OptionDesc_ShowsDescriptionInDump) {
  std::string d = testDescEnumParam.dump();
  // Options with desc should show "name - desc" on their own line
  EXPECT_NE(d.find("OFF - Disable feature"), std::string::npos);
  EXPECT_NE(d.find("ON - Enable feature"), std::string::npos);
  // Option without desc should have no description suffix
  EXPECT_NE(d.find("AUTO"), std::string::npos);
  EXPECT_EQ(d.find("AUTO -"), std::string::npos);
}

// ============================================================================
// Source Tracking
// ============================================================================

TEST_F(NcclParamTest, Source_DefaultWhenNoEnv) {
  // No env var set — should be Default
  (void)testSourceDefault();
  std::string d = testSourceDefault.dump();
  EXPECT_NE(d.find("set_by=Default"), std::string::npos);
}

TEST_F(NcclParamTest, Source_EnvVarWhenSetFromEnvPlugin) {
  SetEnv("TEST_SOURCE_ENVPLUGIN", "99");
  (void)testSourceEnvPlugin();
  std::string d = testSourceEnvPlugin.dump();
  EXPECT_NE(d.find("set_by=EnvPlugin"), std::string::npos);
}
