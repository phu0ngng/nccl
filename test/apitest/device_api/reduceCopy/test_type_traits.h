/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_TYPE_TRAITS_H_
#define _REDUCE_COPY_TEST_TYPE_TRAITS_H_

// Single source of truth for reduceCopy test types, their compile guards, and support flags.
//
// Two macro forms are provided:
//   NCCL_REDUCE_COPY_TYPE_LIST_ALL — all types unconditionally, including guarded ones.
//                                    Python parses this form at generator run time.
//   NCCL_REDUCE_COPY_TYPE_LIST     — C++ active list; satellite macros conditionally
//                                    exclude types whose guard macro is not defined.
// These two macros encode the same information. Keep them in sync when adding/removing types.
//
// X(cpptype, tag, guard, multimemSrc, mulSupported, copyOnly)
//   cpptype     : C++ type expression (spaces allowed; no commas).
//   tag         : GTest name suffix ("Int", "Float", "Bf16", ...).
//   guard       : Preprocessor guard macro name, or NONE for always-available types.
//   multimemSrc : true  ↔ type has multimem load/store specializations (multimem source tests).
//   mulSupported: true  ↔ type works with the Mul variant (no NCCL type-promotion conflict).
//   copyOnly    : true  ↔ type is included in the test set for Copy-only (AllGather) functions.
//                 Selects representative types per byte width (1B/2B/4B/8B) so each element
//                 size is covered without duplicating every type variant.
//                 Included: int8_t (1B), fp8_e4m3 (1B), fp8_e5m2 (1B),
//                           bf16 (2B), float (4B), double (8B).
//                 Excluded: int, uint, uint8_t, half, long long, unsigned long long.
//
// Support flag rationale:
//   multimemSrc = false for int8_t / uint8_t (no multimem specialization in NCCL device lib)
//                 and for long long / unsigned long long (specialization is int64_t / uint64_t;
//                 aliasing is not guaranteed, so multimem tests are excluded).
//   mulSupported = false for FP8 types: NCCL internally promotes FP8 to half for vectorization,
//                 but OpMul only accepts the exact element type, causing a type mismatch.

// ---------------------------------------------------------------------------
// NCCL_REDUCE_COPY_TYPE_LIST_ALL  (Python-facing, unconditional)
// ---------------------------------------------------------------------------
#define NCCL_REDUCE_COPY_TYPE_LIST_ALL \
  X(int,                Int,      NONE,                      true,  true,  false) \
  X(unsigned int,       Uint,     NONE,                      true,  true,  false) \
  X(int8_t,             Int8,     NONE,                      false, true,  true ) \
  X(uint8_t,            Uint8,    NONE,                      false, true,  false) \
  X(long long,          LongLong, NONE,                      false, true,  false) \
  X(unsigned long long, ULongLong,NONE,                      false, true,  false) \
  X(float,              Float,    NONE,                      true,  true,  true ) \
  X(double,             Double,   NONE,                      true,  true,  true ) \
  X(half,               Half,     NONE,                      true,  true,  false) \
  X(__nv_bfloat16,      Bf16,     __CUDA_BF16_TYPES_EXIST__, true,  true,  true ) \
  X(__nv_fp8_e4m3,      Fp8E4M3,  __CUDA_FP8_TYPES_EXIST__,  true,  false, true ) \
  X(__nv_fp8_e5m2,      Fp8E5M2,  __CUDA_FP8_TYPES_EXIST__,  true,  false, true )

// ---------------------------------------------------------------------------
// Satellite macros for the C++ active list (conditional entries).
// Each mirrors the guarded entries in NCCL_REDUCE_COPY_TYPE_LIST_ALL above.
// ---------------------------------------------------------------------------
#if defined(__CUDA_BF16_TYPES_EXIST__)
#define NCCL_REDUCE_COPY_TYPE_LIST_BF16 \
  X(__nv_bfloat16, Bf16, __CUDA_BF16_TYPES_EXIST__, true, true, true)
#else
#define NCCL_REDUCE_COPY_TYPE_LIST_BF16
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
#define NCCL_REDUCE_COPY_TYPE_LIST_FP8 \
  X(__nv_fp8_e4m3, Fp8E4M3, __CUDA_FP8_TYPES_EXIST__, true, false, true) \
  X(__nv_fp8_e5m2, Fp8E5M2, __CUDA_FP8_TYPES_EXIST__, true, false, true)
#else
#define NCCL_REDUCE_COPY_TYPE_LIST_FP8
#endif

// Comma-prefixed helpers for use in testing::Types<> (avoids trailing-comma issues).
#if defined(__CUDA_BF16_TYPES_EXIST__)
#define NCCL_REDUCE_COPY_TYPE_COMMA_BF16 , __nv_bfloat16
#else
#define NCCL_REDUCE_COPY_TYPE_COMMA_BF16
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
#define NCCL_REDUCE_COPY_TYPE_COMMA_FP8 , __nv_fp8_e4m3, __nv_fp8_e5m2
#else
#define NCCL_REDUCE_COPY_TYPE_COMMA_FP8
#endif

// ---------------------------------------------------------------------------
// NCCL_REDUCE_COPY_TYPE_LIST  (C++-facing, conditionally compiled)
// Python should parse NCCL_REDUCE_COPY_TYPE_LIST_ALL instead.
// ---------------------------------------------------------------------------
#define NCCL_REDUCE_COPY_TYPE_LIST \
  X(int,                Int,      NONE, true,  true,  false) \
  X(unsigned int,       Uint,     NONE, true,  true,  false) \
  X(int8_t,             Int8,     NONE, false, true,  true ) \
  X(uint8_t,            Uint8,    NONE, false, true,  false) \
  X(long long,          LongLong, NONE, false, true,  false) \
  X(unsigned long long, ULongLong,NONE, false, true,  false) \
  X(float,              Float,    NONE, true,  true,  true ) \
  X(double,             Double,   NONE, true,  true,  true ) \
  X(half,               Half,     NONE, true,  true,  false) \
  NCCL_REDUCE_COPY_TYPE_LIST_BF16 \
  NCCL_REDUCE_COPY_TYPE_LIST_FP8

#endif // _REDUCE_COPY_TEST_TYPE_TRAITS_H_
