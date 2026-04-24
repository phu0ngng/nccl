/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Internal test-only helpers — NOT part of the installed public API.
// Include this header only from unit test sources (C++ only), never from library code.
#pragma once
#include "nccl_ep.h"
#include <stdint.h>

// Returns the device pointer to the sparse-to-dense map for a HT handle.
// S2D layout: int32_t[max_tokens_per_rank][n_ranks_per_node * experts_per_rank]
// GPU-major:    s2d[token][dest * epr + 0] = recv slot; entries 1..epr-1 = -1.
// Expert-major: s2d[token][dest * epr + k] = expert-major slot for local expert k, or -1.
const int32_t* ncclEpHandle_test_getSparseToDenseMap(ncclEpHandle_t handle);

// Number of rows in the S2D: config.max_tokens_per_rank
int ncclEpHandle_test_getMaxTokensPerRank(ncclEpHandle_t handle);

// Number of ranks per NVLink node (== nRanks for single-node)
int ncclEpHandle_test_getNRanksPerNode(ncclEpHandle_t handle);

// Number of local experts per rank (num_experts / nRanks)
int ncclEpHandle_test_getExpertsPerRank(ncclEpHandle_t handle);
