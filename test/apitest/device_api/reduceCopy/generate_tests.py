#!/usr/bin/env python3
# *************************************************************************
#  * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  *************************************************************************

import argparse
import os
from pathlib import Path
import re
import sys

# License header to emit at the top of generated .cu files (same as other reduceCopy sources).
LICENSE_HEADER_CU = """/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
"""

# Single sources of truth: C++ headers parsed at script run time.
_TRAITS_HEADER = Path(__file__).resolve().parent / "api_function_traits.h"
_TYPE_TRAITS_HEADER = Path(__file__).resolve().parent / "test_type_traits.h"
# RST docs — used only by --validate-docs to cross-check for drift.
_RST_PATH = Path(__file__).resolve().parents[4] / "docs/userguide/source/api/device_reducecopy.rst"

# api_function_traits.h X-macro field indices (0 = name).
# X(name, lambda, local, multimemSrc, multimemDest, usesCustomRedOp, isMulVariant, category, nSrcDefault, nDstDefault, testingGroup)
_LAMBDA_IDX = 1
_LOCAL_IDX = 2
_MULTIMEM_SRC_IDX = 3
_MUL_VARIANT_IDX = 6
_CATEGORY_IDX = 7
_TESTING_GROUP_IDX = 10

# test_type_traits.h X-macro field indices (0 = cpptype).
# X(cpptype, tag, guard, multimemSrc, mulSupported, copyOnly)
_TYPE_TAG_IDX = 1
_TYPE_GUARD_IDX = 2
_TYPE_MULTIMEM_SRC_IDX = 3
_TYPE_MUL_SUPPORTED_IDX = 4
_TYPE_COPY_ONLY_IDX = 5
_TYPE_MIN_FIELDS = _TYPE_COPY_ONLY_IDX + 1


def _extract_macro_body(text, macro_name):
    """
    Return the body (continuation lines) of a #define macro as a single string.
    The define line may end with '\\' or may itself contain the first entry.
    Collects all lines that end with '\\' plus the final non-continuation line.
    """
    start = re.search(
        r"#define\s+" + re.escape(macro_name) + r"[^\S\n]*\\?\s*$",
        text, re.MULTILINE
    )
    if not start:
        return ""
    body_parts = []
    for line in text[start.end():].split("\n"):
        stripped = line.rstrip()
        if not stripped:
            # Empty line between define and first entry (common after trailing \).
            # Keep collecting — the body hasn't started yet.
            if body_parts:
                break  # empty line after body started → end of macro
            continue
        if stripped.endswith("\\"):
            body_parts.append(stripped[:-1])
        else:
            body_parts.append(stripped)
            break  # last continuation line (no trailing backslash)
    return "\n".join(body_parts)


def _load_api_functions_from_traits():
    """
    Parse api_function_traits.h NCCL_REDUCE_COPY_API_FUNC_LIST to get the ordered
    function list and per-function flags. Single source of truth: C++ header; Python
    stays in sync by reading it at script run time.
    """
    text = _TRAITS_HEADER.read_text(encoding="utf-8")

    # Validate expected count so silent regex failures are caught immediately.
    count_match = re.search(r"#define\s+NCCL_REDUCE_COPY_API_FUNC_COUNT\s+(\d+)", text)
    if not count_match:
        raise SystemExit("api_function_traits.h: NCCL_REDUCE_COPY_API_FUNC_COUNT not found")
    expected_count = int(count_match.group(1))

    # Parse only the NCCL_REDUCE_COPY_API_FUNC_LIST macro body.
    body = _extract_macro_body(text, "NCCL_REDUCE_COPY_API_FUNC_LIST")
    if not body.strip():
        raise SystemExit(
            "api_function_traits.h: NCCL_REDUCE_COPY_API_FUNC_LIST not found or empty"
        )

    pattern = re.compile(r"^\s*X\(\s*(.+?)\s*\)\s*$", re.MULTILINE)
    ordered_names = []
    traits_by_name = {}
    for m in pattern.finditer(body):
        inner = m.group(1)
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) <= _TESTING_GROUP_IDX:
            raise SystemExit(
                f"api_function_traits.h: X-macro entry has too few fields: {inner[:60]}..."
            )
        name = parts[0]
        ordered_names.append(name)
        traits_by_name[name] = {
            "lambda":        parts[_LAMBDA_IDX].lower() == "true",
            "local":         parts[_LOCAL_IDX].lower() == "true",
            "multimem_src":  parts[_MULTIMEM_SRC_IDX].lower() == "true",
            "mul_variant":   parts[_MUL_VARIANT_IDX].lower() == "true",
            "category":      parts[_CATEGORY_IDX],
            "testing_group": int(parts[_TESTING_GROUP_IDX]),
        }
    if not ordered_names:
        raise SystemExit(
            "api_function_traits.h: no X(...) entries found in NCCL_REDUCE_COPY_API_FUNC_LIST"
        )
    if len(ordered_names) != expected_count:
        raise SystemExit(
            f"api_function_traits.h: parsed {len(ordered_names)} functions but "
            f"NCCL_REDUCE_COPY_API_FUNC_COUNT={expected_count}; "
            "regex may have missed entries or the count is out of sync"
        )
    return ordered_names, traits_by_name


def _load_types_from_traits():
    """
    Parse test_type_traits.h NCCL_REDUCE_COPY_TYPE_LIST_ALL to get the ordered type
    list and per-type support flags. Single source of truth: C++ header; Python stays
    in sync by reading it at script run time.

    Returns:
        types: list of (cpptype, tag, guard_or_None) — same format as the old TYPES list.
        type_traits: dict tag → {multimem_src, mul_supported}
    """
    text = _TYPE_TRAITS_HEADER.read_text(encoding="utf-8")

    body = _extract_macro_body(text, "NCCL_REDUCE_COPY_TYPE_LIST_ALL")
    if not body.strip():
        raise SystemExit(
            "test_type_traits.h: NCCL_REDUCE_COPY_TYPE_LIST_ALL not found or empty"
        )

    pattern = re.compile(r"^\s*X\(\s*(.+?)\s*\)\s*$", re.MULTILINE)
    types = []
    type_traits = {}
    for m in pattern.finditer(body):
        inner = m.group(1)
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) < _TYPE_MIN_FIELDS:
            raise SystemExit(
                f"test_type_traits.h: X-macro entry has too few fields: {inner[:60]}..."
            )
        cpptype = parts[0]
        tag = parts[_TYPE_TAG_IDX]
        guard_raw = parts[_TYPE_GUARD_IDX]
        guard = None if guard_raw == "NONE" else guard_raw
        multimem_src = parts[_TYPE_MULTIMEM_SRC_IDX].lower() == "true"
        mul_supported = parts[_TYPE_MUL_SUPPORTED_IDX].lower() == "true"
        copy_only = parts[_TYPE_COPY_ONLY_IDX].lower() == "true"
        types.append((cpptype, tag, guard))
        type_traits[tag] = {
            "multimem_src": multimem_src,
            "mul_supported": mul_supported,
            "copy_only": copy_only,
        }
    if not types:
        raise SystemExit(
            "test_type_traits.h: no X(...) entries found in NCCL_REDUCE_COPY_TYPE_LIST_ALL"
        )
    return types, type_traits


_ORDERED_FUNCTIONS, _API_TRAITS = _load_api_functions_from_traits()
_TYPES_RAW, _TYPE_TRAITS = _load_types_from_traits()

# Ordered list of all API function names (same order as C++ ApiFunctionId enum).
FUNCTIONS = _ORDERED_FUNCTIONS

# Derived function sets from api_function_traits.h (no duplicate lists).
LAMBDA_FUNCTIONS        = {n for n, t in _API_TRAITS.items() if t["lambda"]}
LOCAL_FUNCTIONS         = {n for n, t in _API_TRAITS.items() if t["local"]}
MULTIMEM_SRC_FUNCTIONS  = {n for n, t in _API_TRAITS.items() if t["multimem_src"]}
MUL_FUNCTIONS           = {n for n, t in _API_TRAITS.items() if t["mul_variant"]}

# Testing-group sets (mirrors test_matrix.h getExpectedTests logic):
#   Group 2 — copyOnly types only (thin wrappers that delegate alignment to Group 1)
#   Group 3 — float only (pure handle/window translation, no type-specific logic)
#   Group 1 — all types (own alignment computation — sizeof(T) affects pack selection)
GROUP2_FUNCTIONS        = {n for n, t in _API_TRAITS.items() if t["testing_group"] == 2}
GROUP3_FUNCTIONS        = {n for n, t in _API_TRAITS.items() if t["testing_group"] == 3}

# Ordered type list from test_type_traits.h — replaces the old hardcoded TYPES list.
# Format: (cpptype, tag, guard_or_None)  — unchanged from previous TYPES convention.
TYPES = _TYPES_RAW

# Derived type sets: tags of types that do NOT support the given feature.
# Used to skip INSTANTIATE_TEST_CASE_P blocks for unsupported function × type combinations,
# keeping the generated suite in sync with C++ TestMatrix::getExpectedTests().
MULTIMEM_EXCLUDED_TYPES  = {tag for tag, t in _TYPE_TRAITS.items() if not t["multimem_src"]}
MUL_EXCLUDED_TYPES       = {tag for tag, t in _TYPE_TRAITS.items() if not t["mul_supported"]}
COPY_ONLY_EXCLUDED_TYPES = {tag for tag, t in _TYPE_TRAITS.items() if not t["copy_only"]}

# Derived function sets from category field.
ALLGATHER_FUNCTIONS = {n for n, t in _API_TRAITS.items() if t["category"] == "AllGather"}
NM_FUNCTIONS = {n for n, t in _API_TRAITS.items() if t["category"] == "ReduceSumCopyNtoM"} | {"LsaReduceSumLsaCopy"}

# NM split scenarios: (nSrc, nDstStart, label).
# srcRanks = {0..nSrc-1}, dstRanks = {nDstStart..nVis-1}.
# Only functions that support independent src/dst teams are tested in NM mode:
#   LsaReduceSumCopy_DifferentTeams and LsaReduceSumLsaCopy (lambda).
NM_SCENARIOS = [
    (2, 2, "Split2N"),   # no overlap: src={0,1}, dst={2..nVis-1}
    (3, 2, "Split3N"),   # overlap at rank 2: src={0,1,2}, dst={2..nVis-1}
]
NM_ELIGIBLE_FUNCTIONS = {"LsaReduceSumCopy_DifferentTeams", "LsaReduceSumLsaCopy"}

# Types for which NM split scenarios are generated.  Extend this set to enable NM
# tests for additional types without any other changes.
NM_ENABLED_TYPES = {"Float"}


COOP_LEVELS = [
    ("Thread", "CooperationLevel::Thread"),
    ("Warp", "CooperationLevel::Warp"),
    ("Cta", "CooperationLevel::Cta"),
]

UNROLL_KINDS = [
    ("One", "UnrollKind::One", "1"),
    ("Default", "UnrollKind::Default", "getDefaultUnroll<{type}>()"),
]

def load_counts_from_config(config_text):
    match = re.search(r"NCCL_REDUCE_COPY_COUNTS_LIST\s+([0-9,\s]+)", config_text)
    if not match:
        raise SystemExit("NCCL_REDUCE_COPY_COUNTS_LIST not found in config.h")
    counts = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
    if not counts:
        raise SystemExit("NCCL_REDUCE_COPY_COUNTS_LIST is empty")
    return counts


_CONFIG_TEXT = (Path(__file__).resolve().parent / "config.h").read_text(encoding="utf-8")
COUNTS = load_counts_from_config(_CONFIG_TEXT)


def _normalize_type_token(token):
    return token.replace(" ", "").strip().lower()


def get_selected_types():
    """
    Filter TYPES by NCCL_TEST_REDUCE_COPY_TYPES (comma-separated).
    Default: none (empty list) to allow targeted build/test runs.
    Use "all" to select all supported types.
    """
    raw = os.environ.get("NCCL_TEST_REDUCE_COPY_TYPES", "")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return []

    normalized = {_normalize_type_token(t) for t in tokens}
    if "all" in normalized:
        return list(TYPES)
    by_token = {}
    for type_name, type_tag, guard in TYPES:
        by_token[_normalize_type_token(type_tag)] = (type_name, type_tag, guard)
        by_token[_normalize_type_token(type_name)] = (type_name, type_tag, guard)

    selected = []
    unknown = []
    for token in normalized:
        if token in by_token:
            selected.append(by_token[token])
        else:
            unknown.append(token)

    if unknown:
        known = sorted({_normalize_type_token(t[1]) for t in TYPES})
        raise SystemExit(
            "Unknown NCCL_TEST_REDUCE_COPY_TYPES entries: "
            + ", ".join(sorted(unknown))
            + ". Known types: "
            + ", ".join(known)
        )
    return selected


def render_guard_prologue(guard):
    lines = []
    if guard:
        lines.append(f"#if defined({guard})")
    return lines


def render_suite_file(type_name, type_tag, guard):
    suite_name = f"ReduceCopy_{type_tag}"
    lines = [LICENSE_HEADER_CU.strip(), "", "// Generated by generate_tests.py. Do not edit."]
    lines.extend(render_guard_prologue(guard))
    lines.append('#include "test_reduce_copy.h"')
    lines.append(f"using {suite_name} = ReduceCopyWindowTestImpl<{type_name}>;")
    lines.append("")
    lines.append(f"TEST_P({suite_name}, Run) {{")
    lines.append(f"    const std::string typeName = TestMatrix::getTypeName<{type_name}>();")
    lines.append('    VERBOSE_PRINTF("[TEST] ===== Starting test Run (%s) =====\\n", typeName.c_str());')
    lines.append("")
    lines.append("    TestParams params = GetParam();")
    lines.append('    VERBOSE_PRINTF("[TEST] Test params: funcId=%d, count=%zu, coopLevel=%d\\n",')
    lines.append("           static_cast<int>(params.funcId), params.count, static_cast<int>(params.coopLevel));")
    lines.append("")
    lines.append("    runFullTest(params);")
    lines.append('    VERBOSE_PRINTF("[TEST] ===== Test Run (%s) completed =====\\n", typeName.c_str());')
    lines.append("}")
    lines.append("")
    for func_name in FUNCTIONS:
        # Skip multimem-source function × type combinations where the type has no multimem
        # specialization — mirrors TestMatrix::getExpectedTests() filtering in C++.
        if func_name in MULTIMEM_SRC_FUNCTIONS and type_tag in MULTIMEM_EXCLUDED_TYPES:
            continue
        # Skip mul-variant function × type combinations where the type can't use mul —
        # mirrors TestMatrix::getExpectedTests() isMulFunc && !typeInfo.mulSupported filtering.
        if func_name in MUL_FUNCTIONS and type_tag in MUL_EXCLUDED_TYPES:
            continue
        # Apply testing-group type restrictions — mirrors TestMatrix::getExpectedTests():
        #   Group 2 (thin wrappers): restrict to copyOnly types (explicit-copy type set).
        #   Group 3 (handle/window translation): float only.
        #   Group 1 (owns alignment): all types — no restriction.
        if func_name in GROUP2_FUNCTIONS and type_tag in COPY_ONLY_EXCLUDED_TYPES:
            continue
        if func_name in GROUP3_FUNCTIONS and type_tag != "Float":
            continue
        for coop_tag, coop_expr in COOP_LEVELS:
            prefix = f"{func_name}_{coop_tag}"
            lines.append("INSTANTIATE_TEST_CASE_P(")
            lines.append(f"    {prefix},")
            lines.append(f"    {suite_name},")
            param_entries = []
            for count in COUNTS:
                for _, unroll_kind_expr, _ in UNROLL_KINDS:
                    max_gpus_list = (1,) if func_name in LOCAL_FUNCTIONS else (0, 3)
                    for max_gpus in max_gpus_list:
                        if func_name in LAMBDA_FUNCTIONS:
                            offsets_list = ["false", "true"]
                        else:
                            offsets_list = ["false"]
                        for offsets in offsets_list:
                            param_entries.append(
                                f"makeWindowTestParamsWithRunMode<{type_name}>("
                                f"ApiFunctionId::{func_name}, {coop_expr}, {count}, {unroll_kind_expr}, "
                                f"{max_gpus}, {offsets})"
                            )
            lines.append(
                "    ::testing::ValuesIn(std::vector<TestParams>{"
                + ", ".join(param_entries)
                + "}),"
            )
            lines.append("    TestParamsNameGenerator()")
            lines.append(");")

    # NM split scenarios: emit one INSTANTIATE block per (function, coop, scenario).
    # Only eligible functions (DifferentTeams + LsaReduceSumLsaCopy) in NM mode,
    # and only for types listed in NM_ENABLED_TYPES.
    if type_tag in NM_ENABLED_TYPES:
        for func_name in FUNCTIONS:
            if func_name not in NM_ELIGIBLE_FUNCTIONS:
                continue
            # Apply the same multimem/mul/allgather filters as above.
            if func_name in MULTIMEM_SRC_FUNCTIONS and type_tag in MULTIMEM_EXCLUDED_TYPES:
                continue
            if func_name in MUL_FUNCTIONS and type_tag in MUL_EXCLUDED_TYPES:
                continue
            for coop_tag, coop_expr in COOP_LEVELS:
                for nSrc, nDstStart, scenario_label in NM_SCENARIOS:
                    prefix = f"{func_name}_{coop_tag}_NM_{scenario_label}"
                    lines.append("INSTANTIATE_TEST_CASE_P(")
                    lines.append(f"    {prefix},")
                    lines.append(f"    {suite_name},")
                    param_entries = []
                    for count in COUNTS:
                        for _, unroll_kind_expr, _ in UNROLL_KINDS:
                            param_entries.append(
                                f"makeNMTestParams<{type_name}>("
                                f"ApiFunctionId::{func_name}, {coop_expr}, {count}, {unroll_kind_expr}, "
                                f"{nSrc}, {nDstStart})"
                            )
                    lines.append(
                        "    ::testing::ValuesIn(std::vector<TestParams>{"
                        + ", ".join(param_entries)
                        + "}),"
                    )
                    lines.append("    TestParamsNameGenerator()")
                    lines.append(");")

    if guard:
        lines.append("#endif")
    lines.append("")
    return "\n".join(lines)


def render_kernel_instantiations_file(type_name, type_tag, guard, unroll_expr, coop_expr, op_kind):
    lines = [LICENSE_HEADER_CU.strip(), "", "// Generated by generate_tests.py. Do not edit."]
    lines.extend(render_guard_prologue(guard))
    lines.append('#include "kernels.cuh"')
    lines.append("")
    if op_kind == "allreduce":
        lines.append(
            "template void launchAllReduceKernel<"
            f"{type_name}, {unroll_expr}, {coop_expr}>("
            "ApiFunctionId functionID, "
            "ncclWindow_t sendwin, size_t sendoffset, "
            "ncclWindow_t recvwin, size_t recvoffset, "
            "int nSrc, int nDst, int nDstStart, "
            "size_t count, "
            "ncclDevComm devComm, "
            "int gridSize, int blockSize, "
            "cudaStream_t stream);"
        )
    elif op_kind == "sum":
        lines.append(
            "template void launchReduceSumKernel<"
            f"{type_name}, {unroll_expr}, {coop_expr}>("
            "ApiFunctionId functionID, "
            "ncclWindow_t sendwin, size_t sendoffset, "
            "ncclWindow_t recvwin, size_t recvoffset, "
            "int nSrc, int nDst, "
            "size_t count, "
            "ncclDevComm devComm, "
            "int gridSize, int blockSize, "
            "cudaStream_t stream);"
        )
    elif op_kind == "gather":
        lines.append(
            "template void launchAllGatherKernel<"
            f"{type_name}, {unroll_expr}, {coop_expr}>("
            "ApiFunctionId functionID, "
            "ncclWindow_t sendwin, size_t sendoffset, "
            "ncclWindow_t recvwin, size_t recvoffset, "
            "int nSrc, int nDst, "
            "size_t count, "
            "ncclDevComm devComm, "
            "int gridSize, int blockSize, "
            "cudaStream_t stream);"
        )
    else:
        raise ValueError(f"Unknown op_kind: {op_kind}")
    if guard:
        lines.append("#endif")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Docs / header cross-check (--validate-docs)
# ---------------------------------------------------------------------------

# Maps each RST ``.. _label:`` anchor (nccl* anchors only) to the ApiFunctionId name(s) it
# documents.  One RST overload may share a single test ID (window-devComm and window-team
# both exercise LsaReduceSumCopy_Windows), and one RST label may cover two IDs
# (ncclLsaReduceLsaCopy documents both the Generic and Mul variants).
_RST_LABEL_TO_API_IDS = {
    # Series 3.x — ReduceSum (N→1)
    "ncclLsaReduceSum-window-devComm":          ["LsaReduceSum_Window_DevComm"],
    "ncclLsaReduceSum-window-team":             ["LsaReduceSum_Window_Team"],
    "ncclLsaReduceSum-symptr-devComm":          ["LsaReduceSum_SymPtr_DevComm"],
    "ncclLsaReduceSum-symptr-team":             ["LsaReduceSum_SymPtr_Team"],
    "ncclLsaReduceSum-lambda":                  ["LsaReduceSum_Lambda"],
    "ncclLocalReduceSum-strided":               ["LocalReduceSum_Strided"],
    "ncclLocalReduceSum-lambda":                ["LocalReduceSum_Lambda"],
    "ncclMultimemReduceSum-symptr":             ["MultimemReduceSum_SymPtr"],
    "ncclMultimemReduceSum-raw":                ["MultimemReduceSum_RawPtr"],
    "ncclMultimemReduceSum-window":             ["MultimemReduceSum_Window"],
    # Series 4.x — Copy (1→N)
    "ncclLsaCopy-window-devComm":               ["LsaCopy_Window_DevComm"],
    "ncclLsaCopy-window-team":                  ["LsaCopy_Window_Team"],
    "ncclLsaCopy-symptr-devComm":               ["LsaCopy_SymPtr_DevComm"],
    "ncclLsaCopy-symptr-team":                  ["LsaCopy_SymPtr_Team"],
    "ncclLsaCopy-lambda":                       ["LsaCopy_Lambda"],
    "ncclLocalCopy-strided":                    ["LocalCopy_Strided"],
    "ncclLocalCopy-lambda":                     ["LocalCopy_Lambda"],
    "ncclMultimemCopy-symptr":                  ["MultimemCopy_SymPtr"],
    "ncclMultimemCopy-raw":                     ["MultimemCopy_RawPtr"],
    "ncclMultimemCopy-window":                  ["MultimemCopy_Window"],
    # Series 5.x — ReduceSumCopy (N→M) non-lambda
    "ncclLsaReduceSumCopy-window-devComm":      ["LsaReduceSumCopy_Windows"],  # both window overloads → one test ID
    "ncclLsaReduceSumCopy-window-team":         ["LsaReduceSumCopy_Windows"],
    "ncclLsaReduceSumCopy-symptr-devComm":      ["LsaReduceSumCopy_DevComm"],
    "ncclLsaReduceSumCopy-symptr-team":         ["LsaReduceSumCopy_SameTeam"],
    "ncclLsaReduceSumCopy-different-teams":     ["LsaReduceSumCopy_DifferentTeams"],
    "ncclLocalReduceSumCopy-strided":           ["LocalReduceSumCopy_Strided"],
    "ncclMultimemReduceSumCopy-window":         ["MultimemReduceSumCopy_Windows"],
    "ncclMultimemReduceSumCopy-symptr":         ["MultimemReduceSumCopy_SymPtr"],
    "ncclMultimemReduceSumCopy-raw":            ["MultimemReduceSumCopy_RawPtr"],
    "ncclLsaReduceSumMultimemCopy-symptr":      ["LsaReduceSumMultimemCopy_SymPtr"],
    "ncclLsaReduceSumMultimemCopy-raw":         ["LsaReduceSumMultimemCopy_RawPtr"],
    "ncclMultimemReduceSumLsaCopy-symptr":      ["MultimemReduceSumLsaCopy_SymPtr"],
    "ncclMultimemReduceSumLsaCopy-raw":         ["MultimemReduceSumLsaCopy_RawPtr"],
    # Series 2.x — ReduceSumCopy lambda forms
    "ncclLsaReduceSumLsaCopy-lambda":           ["LsaReduceSumLsaCopy"],
    "ncclLsaReduceSumMultimemCopy-lambda":      ["LsaReduceSumMultimemCopy"],
    "ncclMultimemReduceSumLsaCopy-lambda":      ["MultimemReduceSumLsaCopy"],
    "ncclMultimemReduceSumMultimemCopy-lambda": ["MultimemReduceSumMultimemCopy"],
    # Series 1.x — Generic RedOp (one RST label covers both Generic and Mul variants)
    "ncclLsaReduceLsaCopy":                     ["LsaReduceLsaCopy_Generic", "LsaReduceLsaCopy_Generic_Mul"],
    "ncclLsaReduceMultimemCopy":                ["LsaReduceMultimemCopy_Generic"],
}

# Maps each test type tag (from test_type_traits.h) to the type name used in the RST T section.
# int8/int32/int64 each cover both signed and unsigned C++ test types; there is no int16 test
# type (see _RST_TYPES_WITHOUT_TEST_COVERAGE).
_TEST_TAG_TO_RST_TYPE = {
    "Int":       "int32",
    "Uint":      "int32",
    "Int8":      "int8",
    "Uint8":     "int8",
    "LongLong":  "int64",
    "ULongLong": "int64",
    "Float":     "float",
    "Double":    "double",
    "Half":      "half",
    "Bf16":      "__nv_bfloat16",
    "Fp8E4M3":   "__nv_fp8_e4m3",
    "Fp8E5M2":   "__nv_fp8_e5m2",
}

# RST T-section types that are documented but have no corresponding C++ test type.
_RST_TYPES_WITHOUT_TEST_COVERAGE = {"int16"}


def _validate_docs():
    """
    Cross-check docs/userguide/source/api/device_reducecopy.rst against
    api_function_traits.h (via _RST_LABEL_TO_API_IDS) and test_type_traits.h
    (via _TEST_TAG_TO_RST_TYPE).  Exits 1 on any mismatch, 0 on clean.
    """
    if not _RST_PATH.exists():
        raise SystemExit(f"--validate-docs: RST not found: {_RST_PATH}")

    text = _RST_PATH.read_text(encoding="utf-8")
    errors = []
    known_api_ids = set(_ORDERED_FUNCTIONS)

    # ---- Function validation ----
    # All ``.. _nccl*:`` anchors present in the RST.
    rst_labels = set(re.findall(r'\.\. _(nccl[A-Za-z][^:]+):', text))

    # Every mapping key must exist as a label in the RST.
    for label in _RST_LABEL_TO_API_IDS:
        if label not in rst_labels:
            errors.append(f"[func] mapping key '{label}' not found as a label in the RST")

    # Every mapping value must be a known ApiFunctionId.
    for label, api_ids in _RST_LABEL_TO_API_IDS.items():
        for api_id in api_ids:
            if api_id not in known_api_ids:
                errors.append(
                    f"[func] mapping '{label}' → '{api_id}': "
                    f"'{api_id}' not in api_function_traits.h"
                )

    # Every RST nccl* anchor must have a mapping entry.
    for label in sorted(rst_labels):
        if label not in _RST_LABEL_TO_API_IDS:
            errors.append(
                f"[func] RST label '{label}' has no entry in _RST_LABEL_TO_API_IDS "
                f"(new overload added to docs without updating the mapping?)"
            )

    # Every ApiFunctionId must appear in at least one mapping value.
    mapped_ids = {aid for aids in _RST_LABEL_TO_API_IDS.values() for aid in aids}
    for func in _ORDERED_FUNCTIONS:
        if func not in mapped_ids:
            errors.append(
                f"[func] '{func}' (api_function_traits.h) has no RST label "
                f"(undocumented function?)"
            )

    # ---- Type validation ----
    # Parse the **T** template-parameter section (up to **Coop**).
    t_section = re.search(r'\*\*T\*\*.*?(?=\*\*Coop\*\*)', text, re.DOTALL)
    if not t_section:
        errors.append("[type] could not locate the **T** template parameter section in RST")
    else:
        rst_types = set(re.findall(r'``([^`]+)``', t_section.group(0)))

        # Every test tag must map to a type present in the RST T section.
        for tag in sorted(_TYPE_TRAITS):
            if tag not in _TEST_TAG_TO_RST_TYPE:
                errors.append(
                    f"[type] test tag '{tag}' missing from _TEST_TAG_TO_RST_TYPE"
                )
                continue
            rst_type = _TEST_TAG_TO_RST_TYPE[tag]
            if rst_type not in rst_types:
                errors.append(
                    f"[type] test tag '{tag}' maps to '{rst_type}' "
                    f"but '{rst_type}' not found in RST T section"
                )

        # Every type in the RST T section must be accounted for.
        expected_rst_types = set(_TEST_TAG_TO_RST_TYPE.values()) | _RST_TYPES_WITHOUT_TEST_COVERAGE
        for rst_type in sorted(rst_types):
            if rst_type not in expected_rst_types:
                errors.append(
                    f"[type] RST T section has '{rst_type}' which is not in "
                    f"_TEST_TAG_TO_RST_TYPE or _RST_TYPES_WITHOUT_TEST_COVERAGE "
                    f"(new type added to docs?)"
                )

    if errors:
        print("FAIL: docs/C++ header drift detected:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"OK: {_RST_PATH.name} matches api_function_traits.h "
        f"({len(_ORDERED_FUNCTIONS)} functions) and test_type_traits.h "
        f"({len(_TYPE_TRAITS)} type tags)."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=False)
    parser.add_argument("--list-files", action="store_true")
    parser.add_argument(
        "--list-types", action="store_true",
        help="Print space-separated GTest type tags (e.g. 'Int Float Bf16 ...') and exit. "
             "run_reducecopy_types.sh uses this to derive its TYPES list from the same "
             "source of truth (test_type_traits.h via this script)."
    )
    parser.add_argument(
        "--validate-docs", action="store_true",
        help="Cross-check docs/userguide/source/api/device_reducecopy.rst against "
             "api_function_traits.h and test_type_traits.h for drift. "
             "Exits 0 if in sync, 1 on any mismatch."
    )
    args = parser.parse_args()

    if args.validate_docs:
        _validate_docs()
        sys.exit(0)

    if args.list_types:
        # Emit all tags, one line, space-separated.  Guarded types are always included here;
        # at test time the suite file will be compiled only when the guard is satisfied.
        print(" ".join(tag for _, tag, _ in TYPES))
        sys.exit(0)

    if not args.output_dir:
        parser.error("--output-dir is required unless --list-types or --validate-docs is used")

    output_dir = Path(args.output_dir)
    selected_types = get_selected_types()
    if args.list_files:
        suite_files = [
            str(output_dir / f"suite_{type_tag}.cu") for _, type_tag, _ in selected_types
        ]
        instantiation_files = []
        for _, type_tag, _ in selected_types:
            for unroll_tag, _, _ in UNROLL_KINDS:
                for coop_tag, _ in COOP_LEVELS:
                    instantiation_files.append(
                        str(output_dir / f"inst_sum_{type_tag}_{unroll_tag}_{coop_tag}.cu")
                    )
                    instantiation_files.append(
                        str(output_dir / f"inst_allreduce_{type_tag}_{unroll_tag}_{coop_tag}.cu")
                    )
                    if type_tag not in COPY_ONLY_EXCLUDED_TYPES:
                        instantiation_files.append(
                            str(output_dir / f"inst_gather_{type_tag}_{unroll_tag}_{coop_tag}.cu")
                        )
        print(" ".join(suite_files + instantiation_files))
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for type_name, type_tag, guard in selected_types:
        suite_filename = f"suite_{type_tag}.cu"
        suite_path = output_dir / suite_filename
        suite_path.write_text(render_suite_file(type_name, type_tag, guard), encoding="utf-8")

    for type_name, type_tag, guard in selected_types:
        for unroll_tag, _, unroll_expr in UNROLL_KINDS:
            unroll_expr = unroll_expr.format(type=type_name)
            for coop_tag, coop_expr in COOP_LEVELS:
                sum_filename = f"inst_sum_{type_tag}_{unroll_tag}_{coop_tag}.cu"
                sum_path = output_dir / sum_filename
                sum_content = render_kernel_instantiations_file(
                    type_name, type_tag, guard, unroll_expr, coop_expr, "sum"
                )
                sum_path.write_text(sum_content, encoding="utf-8")

                reduce_filename = f"inst_allreduce_{type_tag}_{unroll_tag}_{coop_tag}.cu"
                reduce_path = output_dir / reduce_filename
                reduce_content = render_kernel_instantiations_file(
                    type_name, type_tag, guard, unroll_expr, coop_expr, "allreduce"
                )
                reduce_path.write_text(reduce_content, encoding="utf-8")

                if type_tag not in COPY_ONLY_EXCLUDED_TYPES:
                    gather_filename = f"inst_gather_{type_tag}_{unroll_tag}_{coop_tag}.cu"
                    gather_path = output_dir / gather_filename
                    gather_content = render_kernel_instantiations_file(
                        type_name, type_tag, guard, unroll_expr, coop_expr, "gather"
                    )
                    gather_path.write_text(gather_content, encoding="utf-8")


if __name__ == "__main__":
    main()
