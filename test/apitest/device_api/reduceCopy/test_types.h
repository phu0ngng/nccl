/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_REDUCE_COPY_TEST_TYPES_H
#define NCCL_REDUCE_COPY_TEST_TYPES_H

#include "common.cuh"
#include "test_type_traits.h"
#include <cuda_fp16.h>
#if defined(__CUDA_BF16_TYPES_EXIST__)
#include <cuda_bf16.h>
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif

// Typed test suite derived from NCCL_REDUCE_COPY_TYPE_LIST_ALL in test_type_traits.h.
// The always-available types are listed first; conditional types are appended via comma-prefixed
// satellite macros (NCCL_REDUCE_COPY_TYPE_COMMA_*) to avoid trailing-comma issues.
typedef ::testing::Types<
  int, unsigned int, int8_t, uint8_t, long long, unsigned long long,
  float, double, half
  NCCL_REDUCE_COPY_TYPE_COMMA_BF16
  NCCL_REDUCE_COPY_TYPE_COMMA_FP8
> ReduceCopyTestTypes;

TYPED_TEST_CASE(ReduceCopyTestBase, ReduceCopyTestTypes);

#endif // NCCL_REDUCE_COPY_TEST_TYPES_H
