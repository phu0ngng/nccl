/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "reduce_scatter.h"
#include "collectives.h"

#define UNROLL 8

IMPL_COLL2(ncclReduceScatter, min, FuncMin);
