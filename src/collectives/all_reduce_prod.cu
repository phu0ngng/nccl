/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "all_reduce.h"
#include "collectives.h"
#include <stdint.h>

#define UNROLL 8

IMPL_COLL2(ncclAllReduce, prod, FuncProd);
