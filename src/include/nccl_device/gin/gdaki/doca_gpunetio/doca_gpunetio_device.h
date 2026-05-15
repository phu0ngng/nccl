/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Shim: redirects gin_gdaki.h's `#include "doca_gpunetio/doca_gpunetio_device.h"`
// to the real header under src/transport/. NCCL core's `src.build` normally
// stages this path under build/include/, but contrib/nccl_ep builds skip that
// step, so we resolve in-tree via this shim.
#pragma once
#include "../../../../../transport/net_ib/gdaki/doca-gpunetio/include/doca_gpunetio_device.h"
