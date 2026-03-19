/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Unit tests for parser functions (parsers.h) in isolation — no NcclParam instances needed.

#include <gtest/gtest.h>
#include "param/parsers.h"
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

// ============================================================================
// DefaultParser<int32_t>
// ============================================================================

TEST(ParserTest, DefaultInt32_ResolveValid) {
  int32_t r;
  ASSERT_EQ(NcclDefaultParser<int32_t>::resolve("42", r), ncclSuccess);
  EXPECT_EQ(r, 42);

  ASSERT_EQ(NcclDefaultParser<int32_t>::resolve("-1", r), ncclSuccess);
  EXPECT_EQ(r, -1);

  ASSERT_EQ(NcclDefaultParser<int32_t>::resolve("0", r), ncclSuccess);
  EXPECT_EQ(r, 0);

  // Leading/trailing whitespace is trimmed
  ASSERT_EQ(NcclDefaultParser<int32_t>::resolve("  100  ", r), ncclSuccess);
  EXPECT_EQ(r, 100);
}

TEST(ParserTest, DefaultInt32_ResolveInvalid) {
  int32_t r;
  EXPECT_NE(NcclDefaultParser<int32_t>::resolve("abc", r), ncclSuccess);
  EXPECT_NE(NcclDefaultParser<int32_t>::resolve("", r), ncclSuccess);
  EXPECT_NE(NcclDefaultParser<int32_t>::resolve(nullptr, r), ncclSuccess);
  EXPECT_NE(NcclDefaultParser<int32_t>::resolve("12x", r), ncclSuccess);
  // Leading '+' rejected
  EXPECT_NE(NcclDefaultParser<int32_t>::resolve("+5", r), ncclSuccess);
}

TEST(ParserTest, DefaultInt32_ToString) {
  EXPECT_EQ(NcclDefaultParser<int32_t>::toString(42), "42");
  EXPECT_EQ(NcclDefaultParser<int32_t>::toString(-1), "-1");
}

TEST(ParserTest, DefaultInt32_ValidateAlwaysTrue) {
  EXPECT_EQ(NcclDefaultParser<int32_t>::validate(0), ncclSuccess);
  EXPECT_EQ(NcclDefaultParser<int32_t>::validate(-999), ncclSuccess);
  EXPECT_EQ(NcclDefaultParser<int32_t>::validate(999), ncclSuccess);
}

TEST(ParserTest, DefaultInt32_Desc) {
  auto d = NcclDefaultParser<int32_t>::desc();
  EXPECT_FALSE(d.empty());
}

// ============================================================================
// DefaultParser<bool>
// ============================================================================

TEST(ParserTest, DefaultBool_ResolveTrue) {
  bool r;
  for (const char* s : {"1", "T", "TRUE", "true", "True"}) {
    ASSERT_EQ(NcclDefaultParser<bool>::resolve(s, r), ncclSuccess) << "Expected true for: " << s;
    EXPECT_TRUE(r) << "Expected true for: " << s;
  }
}

TEST(ParserTest, DefaultBool_ResolveFalse) {
  bool r;
  for (const char* s : {"0", "F", "FALSE", "false", "False"}) {
    ASSERT_EQ(NcclDefaultParser<bool>::resolve(s, r), ncclSuccess) << "Expected false for: " << s;
    EXPECT_FALSE(r) << "Expected false for: " << s;
  }
}

TEST(ParserTest, DefaultBool_ResolveInvalid) {
  bool r;
  EXPECT_NE(NcclDefaultParser<bool>::resolve("yes", r), ncclSuccess);
  EXPECT_NE(NcclDefaultParser<bool>::resolve("2", r), ncclSuccess);
  EXPECT_NE(NcclDefaultParser<bool>::resolve("", r), ncclSuccess);
}

TEST(ParserTest, DefaultBool_ToString) {
  EXPECT_EQ(NcclDefaultParser<bool>::toString(true), "TRUE");
  EXPECT_EQ(NcclDefaultParser<bool>::toString(false), "FALSE");
}

TEST(ParserTest, DefaultBool_ValidateAlwaysTrue) {
  EXPECT_EQ(NcclDefaultParser<bool>::validate(true), ncclSuccess);
  EXPECT_EQ(NcclDefaultParser<bool>::validate(false), ncclSuccess);
}

TEST(ParserTest, DefaultBool_Desc) {
  auto d = NcclDefaultParser<bool>::desc();
  EXPECT_FALSE(d.empty());
}

// ============================================================================
// DefaultParser<const char*>
// ============================================================================

TEST(ParserTest, DefaultCStr_Resolve) {
  const char* r;
  ASSERT_EQ(NcclDefaultParser<const char*>::resolve("hello", r), ncclSuccess);
  EXPECT_STREQ(r, "hello");

  ASSERT_EQ(NcclDefaultParser<const char*>::resolve(nullptr, r), ncclSuccess);
  EXPECT_EQ(r, nullptr);
}

TEST(ParserTest, DefaultCStr_ToString) {
  EXPECT_EQ(NcclDefaultParser<const char*>::toString("hello"), "hello");
  EXPECT_EQ(NcclDefaultParser<const char*>::toString(nullptr), "");
}

TEST(ParserTest, DefaultCStr_ValidateAlwaysTrue) {
  const char* val = "anything";
  EXPECT_EQ(NcclDefaultParser<const char*>::validate(val), ncclSuccess);
  val = nullptr;
  EXPECT_EQ(NcclDefaultParser<const char*>::validate(val), ncclSuccess);
}

TEST(ParserTest, DefaultCStr_Desc) {
  auto d = NcclDefaultParser<const char*>::desc();
  EXPECT_FALSE(d.empty());
}

// ============================================================================
// DefaultParser unsupported type
// ============================================================================

struct UnsupportedType {};

TEST(ParserTest, DefaultUnsupported_ResolveReturnsFalse) {
  UnsupportedType r;
  EXPECT_NE(NcclDefaultParser<UnsupportedType>::resolve("anything", r), ncclSuccess);
}

TEST(ParserTest, DefaultUnsupported_ToStringReturnsUnsupported) {
  UnsupportedType v;
  EXPECT_EQ(NcclDefaultParser<UnsupportedType>::toString(v), "<unsupported>");
}

// ============================================================================
// NcclParamBounded (both bounds)
// ============================================================================

TEST(ParserTest, Bounded_ValidateInRange) {
  auto parser = NcclParamBounded<int32_t>(1, 64);
  EXPECT_EQ(parser.validate(1), ncclSuccess);
  EXPECT_EQ(parser.validate(32), ncclSuccess);
  EXPECT_EQ(parser.validate(64), ncclSuccess);
}

TEST(ParserTest, Bounded_ValidateOutOfRange) {
  auto parser = NcclParamBounded<int32_t>(1, 64);
  EXPECT_NE(parser.validate(0), ncclSuccess);
  EXPECT_NE(parser.validate(65), ncclSuccess);
  EXPECT_NE(parser.validate(-1), ncclSuccess);
}

TEST(ParserTest, Bounded_DescFormat) {
  auto parser = NcclParamBounded<int32_t>(1, 64);
  auto d = parser.desc();
  std::string ds(d.data(), d.size());
  EXPECT_NE(ds.find("1"), std::string::npos);
  EXPECT_NE(ds.find("64"), std::string::npos);
}

// ============================================================================
// NcclParamBounded (lower only)
// ============================================================================

TEST(ParserTest, BoundedLower_ValidateAtBound) {
  auto parser = NcclParamBounded<int32_t>(0);
  EXPECT_EQ(parser.validate(0), ncclSuccess);
}

TEST(ParserTest, BoundedLower_ValidateAboveBound) {
  auto parser = NcclParamBounded<int32_t>(0);
  EXPECT_EQ(parser.validate(1000000), ncclSuccess);
}

TEST(ParserTest, BoundedLower_ValidateBelowBound) {
  auto parser = NcclParamBounded<int32_t>(0);
  EXPECT_NE(parser.validate(-1), ncclSuccess);
}

TEST(ParserTest, BoundedLower_DescFormat) {
  auto parser = NcclParamBounded<int32_t>(0);
  auto d = parser.desc();
  std::string ds(d.data(), d.size());
  EXPECT_NE(ds.find(">="), std::string::npos);
  EXPECT_NE(ds.find("0"), std::string::npos);
}

// ============================================================================
// NcclParamOneOf
// ============================================================================

namespace {

enum TestEnum : int32_t {
  ENUM_NONE  = 0,
  ENUM_WARN  = 1,
  ENUM_INFO  = 2,
  ENUM_TRACE = 3
};

auto enumOptions = makeOptions(
  makeOption<int32_t>("NONE",  ENUM_NONE),
  makeOption<int32_t>("WARN",  ENUM_WARN),
  makeOption<int32_t>("INFO",  ENUM_INFO),
  makeOption<int32_t>("TRACE", ENUM_TRACE)
);

} // namespace

TEST(ParserTest, OneOf_ResolveValidToken) {
  auto parser = NcclParamOneOf(enumOptions);
  int32_t r;
  ASSERT_EQ(parser.resolve("INFO", r), ncclSuccess);
  EXPECT_EQ(r, ENUM_INFO);
}

TEST(ParserTest, OneOf_ResolveCaseInsensitive) {
  auto parser = NcclParamOneOf(enumOptions);
  int32_t r;
  ASSERT_EQ(parser.resolve("info", r), ncclSuccess);
  EXPECT_EQ(r, ENUM_INFO);

  ASSERT_EQ(parser.resolve("Warn", r), ncclSuccess);
  EXPECT_EQ(r, ENUM_WARN);
}

TEST(ParserTest, OneOf_ResolveInvalidToken) {
  auto parser = NcclParamOneOf(enumOptions);
  int32_t r;
  EXPECT_NE(parser.resolve("BOGUS", r), ncclSuccess);
  EXPECT_NE(parser.resolve("", r), ncclSuccess);
  EXPECT_NE(parser.resolve(nullptr, r), ncclSuccess);
}

TEST(ParserTest, OneOf_ToString) {
  auto parser = NcclParamOneOf(enumOptions);
  EXPECT_EQ(parser.toString(ENUM_INFO), "INFO");
  EXPECT_EQ(parser.toString(ENUM_NONE), "NONE");
}

TEST(ParserTest, OneOf_Desc) {
  auto parser = NcclParamOneOf(enumOptions);
  auto d = parser.desc();
  std::string ds(d.data(), d.size());
  EXPECT_NE(ds.find("One of:"), std::string::npos);
  EXPECT_NE(ds.find("NONE"), std::string::npos);
  EXPECT_NE(ds.find("TRACE"), std::string::npos);
}

// ============================================================================
// NcclParamBitsetOf
// ============================================================================

namespace {

enum class TestBit : uint32_t {
  INIT  = 1u << 0,
  COLL  = 1u << 1,
  P2P   = 1u << 2,
  GRAPH = 1u << 3,
  ALL   = 0x0F
};

auto bitOptions = makeOptions(
  makeOption<TestBit>("INIT",  TestBit::INIT),
  makeOption<TestBit>("COLL",  TestBit::COLL),
  makeOption<TestBit>("P2P",   TestBit::P2P),
  makeOption<TestBit>("GRAPH", TestBit::GRAPH),
  makeOption<TestBit>("ALL",   TestBit::ALL)
);

} // namespace

TEST(ParserTest, BitsetOf_ResolveSingle) {
  auto parser = NcclParamBitsetOf<TestBit>(bitOptions);
  uint32_t r;
  ASSERT_EQ(parser.resolve("INIT", r), ncclSuccess);
  EXPECT_EQ(r, 0x01u);
}

TEST(ParserTest, BitsetOf_ResolveMultiple) {
  auto parser = NcclParamBitsetOf<TestBit>(bitOptions);
  uint32_t r;
  ASSERT_EQ(parser.resolve("INIT,P2P", r), ncclSuccess);
  EXPECT_EQ(r, 0x01u | 0x04u);
}

TEST(ParserTest, BitsetOf_ResolveComposite) {
  auto parser = NcclParamBitsetOf<TestBit>(bitOptions);
  uint32_t r;
  ASSERT_EQ(parser.resolve("ALL", r), ncclSuccess);
  EXPECT_EQ(r, 0x0Fu);
}

TEST(ParserTest, BitsetOf_ResolveInvalid) {
  auto parser = NcclParamBitsetOf<TestBit>(bitOptions);
  uint32_t r;
  EXPECT_NE(parser.resolve("BOGUS", r), ncclSuccess);
  EXPECT_NE(parser.resolve(nullptr, r), ncclSuccess);
}

TEST(ParserTest, BitsetOf_ToStringDecompose) {
  auto parser = NcclParamBitsetOf<TestBit>(bitOptions);
  // Composite value exact match
  EXPECT_EQ(parser.toString(0x0F), "ALL");
  // Single bit
  EXPECT_EQ(parser.toString(0x01), "INIT");
  // Multi-bit decomposition
  std::string s = parser.toString(0x01u | 0x04u);
  EXPECT_NE(s.find("INIT"), std::string::npos);
  EXPECT_NE(s.find("P2P"), std::string::npos);
}

TEST(ParserTest, BitsetOf_Desc) {
  auto parser = NcclParamBitsetOf<TestBit>(bitOptions);
  auto d = parser.desc();
  std::string ds(d.data(), d.size());
  EXPECT_NE(ds.find("Comma-separated"), std::string::npos);
  EXPECT_NE(ds.find("INIT"), std::string::npos);
}

// ============================================================================
// NcclParamList (unordered_set)
// ============================================================================

using NcclStringSet = std::unordered_set<std::string>;

TEST(ParserTest, List_ResolveCommaDelimited) {
  auto parser = NcclParamListOf<NcclStringSet>(',');
  NcclStringSet r;
  ASSERT_EQ(parser.resolve("FOO,BAR,BAZ", r), ncclSuccess);
  EXPECT_EQ(r.size(), 3u);
  EXPECT_TRUE(r.count("FOO") > 0);
  EXPECT_TRUE(r.count("BAR") > 0);
  EXPECT_TRUE(r.count("BAZ") > 0);
}

TEST(ParserTest, List_ResolveWithWhitespace) {
  auto parser = NcclParamListOf<NcclStringSet>(',');
  NcclStringSet r;
  ASSERT_EQ(parser.resolve("  FOO , BAR , BAZ  ", r), ncclSuccess);
  EXPECT_EQ(r.size(), 3u);
  EXPECT_TRUE(r.count("FOO") > 0);
  EXPECT_TRUE(r.count("BAR") > 0);
  EXPECT_TRUE(r.count("BAZ") > 0);
}

TEST(ParserTest, List_ResolveSingleElement) {
  auto parser = NcclParamListOf<NcclStringSet>(',');
  NcclStringSet r;
  ASSERT_EQ(parser.resolve("ONLY", r), ncclSuccess);
  EXPECT_EQ(r.size(), 1u);
  EXPECT_TRUE(r.count("ONLY") > 0);
}

TEST(ParserTest, List_ResolveEmpty) {
  auto parser = NcclParamListOf<NcclStringSet>(',');
  NcclStringSet r;
  EXPECT_NE(parser.resolve("", r), ncclSuccess);
}

TEST(ParserTest, List_ResolveNull) {
  auto parser = NcclParamListOf<NcclStringSet>(',');
  NcclStringSet r;
  EXPECT_NE(parser.resolve(nullptr, r), ncclSuccess);
}

TEST(ParserTest, List_ResolveSkipsEmptyTokens) {
  auto parser = NcclParamListOf<NcclStringSet>(',');
  NcclStringSet r;
  ASSERT_EQ(parser.resolve("FOO,,BAR,  ,BAZ", r), ncclSuccess);
  EXPECT_EQ(r.size(), 3u);
  EXPECT_TRUE(r.count("FOO") > 0);
  EXPECT_TRUE(r.count("BAR") > 0);
  EXPECT_TRUE(r.count("BAZ") > 0);
}

TEST(ParserTest, List_ResolveCustomDelimiter) {
  auto parser = NcclParamListOf<NcclStringSet>(':');
  NcclStringSet r;
  ASSERT_EQ(parser.resolve("A:B:C", r), ncclSuccess);
  EXPECT_EQ(r.size(), 3u);
  EXPECT_TRUE(r.count("A") > 0);
  EXPECT_TRUE(r.count("B") > 0);
  EXPECT_TRUE(r.count("C") > 0);
}

TEST(ParserTest, List_ToString) {
  auto parser = NcclParamListOf<NcclStringSet>(',');
  NcclStringSet set;
  set.insert("A");
  std::string s = parser.toString(set);
  EXPECT_EQ(s, "A");
}

TEST(ParserTest, List_ToStringEmpty) {
  auto parser = NcclParamListOf<NcclStringSet>(',');
  NcclStringSet set;
  std::string s = parser.toString(set);
  EXPECT_EQ(s, "");
}

TEST(ParserTest, List_Desc) {
  auto parser = NcclParamListOf<NcclStringSet>(',');
  auto d = parser.desc();
  std::string ds(d.data(), d.size());
  EXPECT_NE(ds.find("Delimiter-separated"), std::string::npos);
  EXPECT_NE(ds.find(","), std::string::npos);
}

// ============================================================================
// NcclParamList (vector)
// ============================================================================

TEST(ParserTest, ListVector_ResolveCommaDelimited) {
  auto parser = NcclParamListOf<std::vector<std::string>>(',');
  std::vector<std::string> r;
  ASSERT_EQ(parser.resolve("X,Y,Z", r), ncclSuccess);
  EXPECT_EQ(r.size(), 3u);
  EXPECT_EQ(r[0], "X");
  EXPECT_EQ(r[1], "Y");
  EXPECT_EQ(r[2], "Z");
}

TEST(ParserTest, ListVector_ToString) {
  auto parser = NcclParamListOf<std::vector<std::string>>(',');
  std::vector<std::string> v = {"A", "B", "C"};
  std::string s = parser.toString(v);
  EXPECT_EQ(s, "A,B,C");
}
